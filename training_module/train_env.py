# training_module/train_env.py
from __future__ import annotations

import json
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

# BaseEnv: rutas/paths y convenios
from base_env.context import BaseContext

# Opcional: usar RiskManager avanzado si está disponible
try:
    from base_env.risk_manager_advanced import RiskManager  # noqa
    _HAS_RISK = True
except Exception:
    _HAS_RISK = False

# Simulador de ejecución (latencia/slippage/fees) si deseas integrarlo más adelante
try:
    from core.oms.execution_sim import ExecutionSim
    _HAS_EXEC_SIM = True
except Exception:
    _HAS_EXEC_SIM = False


@dataclass
class EnvConfig:
    symbol: str = "BTCUSDT"
    exec_tf: str = "5m"
    window_size: int = 64
    fee_rate: float = 0.0004   # taker por defecto
    slippage_bp: float = 1.0   # 1 basis point (0.01%)
    initial_equity: float = 10_000.0
    reward_dd_penalty: float = 0.0  # penalización por drawdown (0..1)
    normalize_obs: bool = True


class TradingEnv:
    """
    Entorno simple estilo Gym (reset/step) para entrenamiento PPO.
    Lee el dataset entrenable generado por build-dataset:
      data/warehouse/features/symbol=<symbol>/timeframe=<tf>/train_dataset.parquet

    Señales/columnas esperadas:
      - timestamp (ms)
      - OHLCV exec_tf (renombrados a open,high,low,close,volume)
      - features técnicos (prefijo ta_*)
      - labels: y (int {-1,0,1}), tte (opcional)
      - contexto MTF (prefijos dir_* y conf_*) opcional

    Acciones discretas:
      0 = HOLD, 1 = ENTER_LONG (si no en posición), 2 = EXIT (si en posición)

    Obs: ventana causal (W, F) de columnas numéricas excluyendo ['timestamp','y','tte'].
    Reward: ΔPnL neto paso a paso (con fees y slippage básicos) - opcional penalización por DD.
    """

    ACTION_HOLD = 0
    ACTION_ENTER = 1
    ACTION_EXIT = 2

    def __init__(self, cfg: EnvConfig):
        self.cfg = cfg
        self.ctx = BaseContext()
        self.ds_path = (
            Path(self.ctx.features_dir)
            / f"symbol={cfg.symbol}"
            / f"timeframe={cfg.exec_tf}"
            / "train_dataset.parquet"
        )
        if not self.ds_path.exists():
            raise FileNotFoundError(
                f"No se encuentra el dataset entrenable en: {self.ds_path}\n"
                "Genera primero con: python app.py build-dataset --symbol {symbol} --exec-tf {tf}"
            )
        self._df = pd.read_parquet(self.ds_path)
        # columnas de observación = numéricas salvo timestamp,y,tte
        drop_cols = {"timestamp", "y", "tte"}
        self.feature_cols: List[str] = [
            c for c in self._df.columns
            if c not in drop_cols and pd.api.types.is_numeric_dtype(self._df[c])
        ]
        if len(self.feature_cols) == 0:
            raise ValueError("No hay columnas de features numéricas para la observación.")

        # buffers / estado
        self._idx: int = 0
        self._window: deque = deque(maxlen=self.cfg.window_size)
        self._position_qty: float = 0.0
        self._entry_price: float = 0.0
        self._equity: float = float(self.cfg.initial_equity)
        self._equity_peak: float = float(self.cfg.initial_equity)
        self._last_close: float = 0.0
        self._done: bool = False

    # ------------- API estilo Gym -------------
    def reset(self, seed: int | None = None) -> Tuple[np.ndarray, Dict]:
        # buscamos un índice válido donde quepa la ventana completa
        self._idx = max(self.cfg.window_size, 2)
        self._position_qty = 0.0
        self._entry_price = 0.0
        self._equity = float(self.cfg.initial_equity)
        self._equity_peak = float(self.cfg.initial_equity)
        self._done = False
        self._window.clear()

        obs = self._emit_obs_for_index(self._idx)
        info = self._info()
        return obs, info

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        if self._done:
            return self._empty_obs(), 0.0, True, False, self._info()

        # Precios actuales (cierra la barra actual como referencia)
        row = self._df.iloc[self._idx]
        close = float(row["close"])
        self._last_close = close

        # --- Slippage básico (1bp) + fee taker ---
        def apply_trade_fill(px: float) -> float:
            slip = px * (self.cfg.slippage_bp / 10000.0)
            return px + slip

        reward = 0.0
        fee_cost = 0.0

        # Acción → posición
        if action == self.ACTION_ENTER and self._position_qty <= 1e-12:
            # sizing: usar RiskManager si existe, sino riesgo fijo del 0.6% capital y stop = 1*ATR aprox
            qty = self._compute_position_size(self._idx)
            if qty > 0:
                fill_px = apply_trade_fill(close)
                notional = qty * fill_px
                fee_cost = notional * self.cfg.fee_rate
                self._position_qty = qty
                self._entry_price = fill_px
                self._equity -= fee_cost  # fee al abrir

        elif action == self.ACTION_EXIT and self._position_qty > 1e-12:
            fill_px = apply_trade_fill(close)
            notional = self._position_qty * fill_px
            pnl = (fill_px - self._entry_price) * self._position_qty
            fee_cost = notional * self.cfg.fee_rate
            reward = pnl - fee_cost  # reward al cerrar
            self._equity += reward  # equity se actualiza con PnL neto
            # cerrar
            self._position_qty = 0.0
            self._entry_price = 0.0

        # HOLD no realiza cambios (pnl no realizado no entra al reward, solo realized)
        # Si quisieras reward mark-to-market, añade variación de upnl aquí.

        # penalización por drawdown (opcional)
        self._equity_peak = max(self._equity_peak, self._equity)
        dd = 0.0 if self._equity_peak <= 0 else 1.0 - (self._equity / self._equity_peak + 1e-12)
        if self.cfg.reward_dd_penalty > 0:
            reward -= float(self.cfg.reward_dd_penalty) * float(dd)

        # avanzar
        self._idx += 1
        if self._idx >= len(self._df):
            self._done = True

        obs = self._emit_obs_for_index(self._idx)
        info = self._info()
        terminated = self._done
        truncated = False
        return obs, float(reward), terminated, truncated, info

    # ------------- Helpers internos -------------
    def _compute_position_size(self, i: int) -> float:
        """
        Tamaño basado en %equity y una distancia SL aproximada por ATR si existe.
        Fallback seguro si no hay ATR: usar un notional fijo de ~1% del equity.
        """
        price = float(self._df.iloc[i]["close"])
        atr = float(self._df.iloc[i].get("ta_atr", 0.0))
        # Por defecto riesgo 0.6% del equity
        risk_pct = 0.006
        risk_usdt = self._equity * risk_pct

        if atr and atr > 0:
            qty = max(risk_usdt / atr, 0.0)
        else:
            # notional ~ 1% equity
            qty = max((self._equity * 0.01) / max(price, 1e-8), 0.0)

        # límites mínimos prácticos
        min_notional = 10.0
        qty = max(qty, min_notional / max(price, 1e-8))
        return float(qty)

    def _emit_obs_for_index(self, i: int) -> np.ndarray:
        # asegura que haya ventana completa
        left = i - self.cfg.window_size
        if left < 0:
            left = 0
        df_win = self._df.iloc[left:i][self.feature_cols]

        if len(df_win) < self.cfg.window_size:
            # pad con la primera fila válida
            if len(df_win) == 0:
                base = np.zeros((self.cfg.window_size, len(self.feature_cols)), dtype=np.float32)
                return base
            first = df_win.iloc[0:1].values
            pad_rows = [first for _ in range(self.cfg.window_size - len(df_win))]
            arr = np.vstack([*pad_rows, df_win.values]).astype(np.float32)
        else:
            arr = df_win.values.astype(np.float32)

        if self.cfg.normalize_obs:
            arr = self._zscore_last_dim(arr)

        return arr

    @staticmethod
    def _zscore_last_dim(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
        mu = x.mean(axis=0, keepdims=True)
        sd = x.std(axis=0, keepdims=True)
        return (x - mu) / (sd + eps)

    def _empty_obs(self) -> np.ndarray:
        return np.zeros((self.cfg.window_size, len(self.feature_cols)), dtype=np.float32)

    def _info(self) -> Dict:
        pos = float(self._position_qty)
        upnl = 0.0
        if pos > 1e-12 and self._last_close > 0 and self._entry_price > 0:
            upnl = (self._last_close - self._entry_price) * pos
        dd = 0.0 if self._equity_peak <= 0 else 1.0 - (self._equity / self._equity_peak + 1e-12)
        return {
            "equity": float(self._equity),
            "equity_peak": float(self._equity_peak),
            "drawdown": float(dd),
            "position_qty": pos,
            "entry_price": float(self._entry_price),
        }
