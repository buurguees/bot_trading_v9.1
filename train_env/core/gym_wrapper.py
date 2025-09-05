# train_env/gym_wrapper.py
from __future__ import annotations

import gymnasium as gym
import numpy as np
import random
import json
from pathlib import Path
from typing import Dict, Any, Tuple, List, Optional, Union

from base_env.base_env import BaseTradingEnv
from ..utilities.reward_shaper import RewardShaper
from ..utilities.strategy_curriculum import StrategyCurriculum
from ..utilities.strategy_logger import StrategyLogger


class TradingGymWrapper(gym.Env):
    """
    Gym wrapper para BaseTradingEnv con:
      - Override de acción (policy / close / block / force_long / force_short)
      - Shaping de reward
      - Deduplicación de aperturas por (bar_time, side)
      - Curriculum learning (evita estrategias malas y sugiere ajustes)
      - Soporte opcional de leverage MultiDiscrete (Futures)

    SPOT:
      action_space = Discrete(5)  # 0=policy, 1=close, 2=block, 3=force_long, 4=force_short

    FUTURES:
      action_space = MultiDiscrete([5, Nlevers])
        - a[0]: acción trading (como arriba)
        - a[1]: índice de leverage (map a [min, max, step] definido en YAML)
    """

    metadata = {"render_modes": []}

    # Map de acciones por claridad externa/debug
    ACTION_MEANINGS = {
        0: "policy",
        1: "close",
        2: "block",
        3: "force_long",
        4: "force_short",
    }

    def __init__(
        self,
        base_env: BaseTradingEnv,
        reward_yaml: str,
        tfs: List[str],
        leverage_spec: Optional[Dict[str, Union[str, float]]] = None,
        strategy_log_path: Optional[str] = None,
        verbosity: int = 0,
    ):
        super().__init__()
        if not tfs:
            raise ValueError("Se requiere al menos un timeframe en 'tfs'.")

        self.env = base_env
        self.tfs = tfs
        self.shaper = RewardShaper(reward_yaml)
        self.strategy_log_path = strategy_log_path
        self.verbosity = verbosity

        # --- Deduplicación de decisiones por (bar_time, side_sign) ---
        self._decision_cache: set[Tuple[int, int]] = set()
        self._last_bar_time: Optional[int] = None

        # --- Curriculum learning ---
        self.curriculum: Optional[StrategyCurriculum] = None
        if strategy_log_path:
            strategies_file = strategy_log_path.replace(
                "_provisional.jsonl", "_strategies.json"
            )
            try:
                self.curriculum = StrategyCurriculum(
                    strategies_file, verbose=self.verbosity >= 1
                )
                if self.verbosity >= 1:
                    print(f"[CURRICULUM] Integrado con {strategies_file}")
            except FileNotFoundError:
                if self.verbosity >= 1:
                    print(f"[CURRICULUM] No existe: {strategies_file}")
            except json.JSONDecodeError as e:
                if self.verbosity >= 1:
                    print(f"[CURRICULUM] JSON inválido: {e}")
            except Exception as e:
                if self.verbosity >= 1:
                    print(f"[CURRICULUM] Falló la carga: {e}")

        # --- Especificación de leverage (Futures) ---
        self._lev_spec: Optional[Tuple[float, float, float, int]] = None
        if leverage_spec:
            mn = float(leverage_spec["min"])
            mx = float(leverage_spec["max"])
            st = float(leverage_spec.get("step", 1.0))
            n_levels = int(round((mx - mn) / st)) + 1
            if n_levels <= 0:
                raise ValueError("Especificación de leverage inválida.")
            self._lev_spec = (mn, mx, st, n_levels)
            self.action_space = gym.spaces.MultiDiscrete([5, n_levels])
        else:
            self.action_space = gym.spaces.Discrete(5)

        # --- Observation space (vector plano) ---
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(self._obs_dim(),), dtype=np.float32
        )

        # --- Strategy logger ---
        self.strategy_log = StrategyLogger(
            strategy_log_path or "models/tmp/tmp_provisional.jsonl", segment_id=0
        )

    # -------- Helpers de observación --------

    def _obs_dim(self) -> int:
        """Dimensión del vector de observación aplanado."""
        per_tf = 7  # close, ema20, ema50, rsi14, atr14, macd_hist, bb_p
        pos = 4     # side, qty, entry_price, unrealized_pnl
        ana = 2     # confidence, side_hint
        return len(self.tfs) * per_tf + pos + ana

    def _flatten_obs(self, obs: Dict[str, Any]) -> np.ndarray:
        """
        Aplana la observación dict -> np.ndarray (float32).
        Robustez: maneja claves faltantes con defaults sensatos.
        """
        vec: List[float] = []
        tfs_block = obs.get("tfs", {}) or {}
        feats_block = obs.get("features", {}) or {}

        for tf in self.tfs:
            bar = tfs_block.get(tf, {}) or {}
            feats = feats_block.get(tf, {}) or {}
            vec.extend(
                [
                    float(bar.get("close", 0.0) or 0.0),
                    float(feats.get("ema20", 0.0) or 0.0),
                    float(feats.get("ema50", 0.0) or 0.0),
                    float(feats.get("rsi14", 50.0) or 50.0),
                    float(feats.get("atr14", 0.0) or 0.0),
                    float(feats.get("macd_hist", 0.0) or 0.0),
                    float(feats.get("bb_p", 0.5) or 0.5),
                ]
            )

        pos = obs.get("position", {}) or {}
        vec.extend(
            [
                float(pos.get("side", 0) or 0),
                float(pos.get("qty", 0.0) or 0.0),
                float(pos.get("entry_price", 0.0) or 0.0),
                float(pos.get("unrealized_pnl", 0.0) or 0.0),
            ]
        )

        ana = obs.get("analysis", {}) or {}
        vec.extend(
            [
                float(ana.get("confidence", 0.0) or 0.0),
                float(ana.get("side_hint", 0) or 0),
            ]
        )

        return np.asarray(vec, dtype=np.float32)

    # -------- Deduplicación --------

    def _new_bar_reset(self, bar_time: Optional[int]) -> None:
        """Limpia la cache de decisiones si cambia la barra."""
        if bar_time is not None and self._last_bar_time != bar_time:
            self._decision_cache.clear()
            self._last_bar_time = bar_time
            if self.verbosity >= 2:
                print(f"[DEDUP] Nueva barra {bar_time} → cache reseteada")

    def _dup_allows(self, bar_time: Optional[int], side: int) -> bool:
        """
        True si permitimos abrir (no duplicado en misma barra y side).
        side: 1=long, -1=short, 0=neutral.
        """
        if side == 0 or bar_time is None:
            return True
        key = (int(bar_time), int(np.sign(side)))
        if key in self._decision_cache:
            if self.verbosity >= 2:
                print(f"🔍 Duplicado detectado {key} → BLOQUEADO")
            return False
        self._decision_cache.add(key)
        return True

    def _clear_decision_cache_on_close(
        self, events: List[Dict[str, Any]], info: Dict[str, Any]
    ) -> None:
        """Limpia cache si se cierra posición (por evento o flag)."""
        if any(e.get("kind") == "CLOSE" for e in events) or info.get(
            "just_closed", False
        ):
            self._decision_cache.clear()
            if self.verbosity >= 2:
                print("🧹 Cache de decisiones limpiada (CLOSE)")

    # -------- Fallback SL/TP por ATR --------

    def _get_atr_fallback_distance(self, current_price: float) -> Tuple[float, float]:
        """
        Distancias SL/TP por ATR.
        Retorna (sl_distance, tp_distance) en unidades de precio.
        """
        try:
            obs = self.env.get_observation()
            features = obs.get("features", {}) or {}
            atr_value: Optional[float] = None
            for tf in self.tfs:
                tf_features = features.get(tf, {}) or {}
                atr = tf_features.get("atr14")
                if atr is not None and float(atr) > 0:
                    atr_value = float(atr)
                    break
            if atr_value is None or atr_value <= 0:
                atr_value = current_price * 0.01  # fallback 1%
            sl_distance = atr_value * 2.0  # SL 2*ATR
            tp_distance = atr_value * 3.0  # TP 3*ATR (R:R=1.5)
            return sl_distance, tp_distance
        except Exception as e:
            if self.verbosity >= 1:
                print(f"⚠️ ATR fallback error: {e}. Usando 2%/3%.")
            return current_price * 0.02, current_price * 0.03

    # -------- API gym --------

    def reset(
        self, *, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Reset del episodio."""
        # Semillas reproducibles
        if seed is not None:
            try:
                self.env.seed(seed)  # si el BaseTradingEnv lo expone
            except Exception:
                pass
            np.random.seed(seed)
            random.seed(seed)
            super().reset(seed=seed)

        obs = self.env.reset()

        # Inicializar milestones de reward avanzado
        try:
            self.shaper.advanced_rewards.initialize_run(
                initial_balance=self.env._init_cash, target_balance=self.env._target_cash
            )
        except Exception:
            # Si no existe advanced_rewards, no bloqueamos el reset
            pass

        # Reset de deduplicación
        self._decision_cache.clear()
        self._last_bar_time = None

        return self._flatten_obs(obs), {}

    def _lev_from_idx(self, idx: int) -> float:
        """Idx de leverage → valor real."""
        if self._lev_spec is None:
            raise ValueError("Leverage no configurado.")
        mn, _, st, n_levels = self._lev_spec
        idx = max(0, min(int(idx), n_levels - 1))
        return float(mn + idx * st)

    def step(
        self, action: Union[int, np.ndarray]
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """Un paso del entorno con shaping y deduplicación."""

        leverage: Optional[float] = None
        lev_idx: Optional[int] = None

        if self._lev_spec is not None:
            if not isinstance(action, (np.ndarray, list, tuple)) or len(action) != 2:
                raise ValueError("Se esperaba acción MultiDiscrete de longitud 2.")
            trade_action = int(action[0])
            lev_idx = int(action[1])
            leverage = self._lev_from_idx(lev_idx)
        else:
            trade_action = int(action)

        # Curriculum: 5% de las veces intentar modificar (evitar malas estrategias)
        if self.curriculum and random.random() < 0.05:
            bad_strategies: List[Dict[str, Any]] = []
            try:
                if self.strategy_log_path:
                    bad_strat_file = self.strategy_log_path.replace(
                        "_provisional.jsonl", "_bad_strategies.json"
                    )
                    if Path(bad_strat_file).exists():
                        with open(bad_strat_file, "r") as f:
                            bad_strategies = json.load(f)
            except Exception as e:
                if self.verbosity >= 1:
                    print(f"[CURRICULUM] No se cargaron malas estrategias: {e}")

            suggestion = self.curriculum.suggest_action_modification(
                trade_action, {}, bad_strategies
            )
            if suggestion is not None:
                if self.verbosity >= 2:
                    print(f"[CURRICULUM] Acción {trade_action} → {suggestion}")
                trade_action = suggestion

        # Obtener obs actual para timestamp/price
        current_obs = {}
        try:
            current_obs = self.env.get_observation()
        except Exception:
            pass

        main_tf = self.tfs[0]
        current_bar = (current_obs.get("tfs", {}) or {}).get(main_tf, {}) or {}
        current_bar_time: Optional[int] = current_bar.get("ts")
        current_price: float = float(current_bar.get("close", 0.0) or 0.0)

        # Determinar side según acción
        current_side = 0
        if trade_action == 3:
            current_side = 1
        elif trade_action == 4:
            current_side = -1

        # Reset dedup por nueva barra y comprobar duplicado
        self._new_bar_reset(current_bar_time)
        can_open = self._dup_allows(current_bar_time, current_side)

        # Fallback dinamico SL/TP
        if trade_action not in [1, 2] and hasattr(self.env, "set_sl_tp_fallback"):
            if current_price > 0:
                sl_distance, tp_distance = self._get_atr_fallback_distance(
                    current_price
                )
                try:
                    self.env.set_sl_tp_fallback(sl_distance, tp_distance)
                except Exception as e:
                    if self.verbosity >= 1:
                        print(f"⚠️ set_sl_tp_fallback() falló: {e}")

        # Inyectar override en BaseTradingEnv
        try:
            self.env.set_action_override(
                trade_action, leverage_override=leverage, leverage_index=lev_idx
            )
        except TypeError:
            # Compatibilidad si el método no acepta leverage_index
            self.env.set_action_override(trade_action, leverage_override=leverage)

        # Avanzar el BaseTradingEnv
        payload = {"decision": None, "allow_open": can_open}
        result = self.env.step(payload)

        # Normalizar API Gymnasium
        if len(result) == 5:
            obs, base_r, terminated, truncated, info = result
        else:
            obs, base_r, done, info = result
            terminated, truncated = bool(done), False

        # Log de eventos de estrategia
        events = info.get("events", []) or []
        if events:
            try:
                self.strategy_log.append_many(events)
            except Exception as e:
                if self.verbosity >= 1:
                    print(f"⚠️ StrategyLogger.append_many falló: {e}")

        # Limpiar cache en cierre
        self._clear_decision_cache_on_close(events, info)

        # Shaping de reward
        balance_milestones = info.get("balance_milestones", 0)
        empty_run = (
            getattr(self.env, "_empty_runs_count", 0) > 0 and not events and not terminated
        )

        shaped, parts = self.shaper.compute(
            obs,
            base_r,
            events,
            empty_run,
            balance_milestones,
            initial_balance=getattr(self.env, "_init_cash", 0.0),
            target_balance=getattr(self.env, "_target_cash", 0.0),
        )

        # Devolver obs plana + info extendida
        flat_obs = self._flatten_obs(obs)
        info_out = {"r_parts": parts, "can_open": can_open, **info}
        if self._lev_spec is not None:
            info_out["leverage_used"] = leverage
            info_out["leverage_index"] = lev_idx
        info_out["trade_action_name"] = self.ACTION_MEANINGS.get(trade_action, "unknown")

        return flat_obs, float(shaped), bool(terminated), bool(truncated), info_out

    # -------- Passthrough útiles --------

    def needs_learning_rate_reset(self) -> bool:
        """Expose flag para que el vectorizado reinicie LR cuando toque."""
        try:
            return bool(self.env.needs_learning_rate_reset())
        except Exception:
            return False

    def reset_learning_rate_flag(self) -> None:
        """Resetea el flag de LR en el entorno base."""
        try:
            self.env.reset_learning_rate_flag()
        except Exception:
            pass

    def close(self) -> None:
        try:
            self.env.close()
        except Exception:
            pass
