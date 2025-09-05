# base_env/policy/gating.py
# Descripción: Política mínima operativa:
# - Abre si hay confluencia y NO hay posición (dedup)
# - SL/TP por ATR (exec TF → fallback base TF)
# - Cierra TODO por giro con confluencia o por pérdida de confluencia persistente

from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Optional
from ..config.models import HierarchicalConfig, RiskConfig
from ..tfs.calendar import tf_to_ms
from .rules import confluence_ok, side_from_hint, dedup_block, sl_tp_from_atr


@dataclass
class Decision:
    should_open: bool = False
    side: int = 0
    price_hint: float = 0.0
    sl: Optional[float] = None
    tp: Optional[float] = None
    ttl_bars: int = 0
    trailing: bool = False

    should_close_partial: bool = False
    should_close_all: bool = False
    close_qty: float = 0.0


class PolicyEngine:
    def __init__(self, cfg: HierarchicalConfig, risk_cfg: RiskConfig, exec_tf: Optional[str] = None, base_tf: str = "1m") -> None:
        self.cfg = cfg
        self.risk_cfg = risk_cfg
        self._last_open_ts: Optional[int] = None
        self._conf_loss_count: int = 0  # contador de pérdida de confluencia
        self.exec_tf = exec_tf or (cfg.execute_tfs[0] if cfg.execute_tfs else base_tf)
        self.base_tf = base_tf
        self.base_tf_ms = tf_to_ms(base_tf)
        self._debug_mode = False  # ← Debug deshabilitado

    def decide(self, obs: dict[str, Any]) -> Decision:
        ts_now = int(obs["ts"])
        analysis = obs.get("analysis", {})
        features = obs.get("features", {})
        tfs = obs.get("tfs", {})
        position = obs.get("position", {}) or {}

        price_exec = float(tfs.get(self.exec_tf, {}).get("close", 0.0))
        side_hint = side_from_hint(analysis)
        conf_ok = confluence_ok(analysis, self.cfg.min_confidence)
        
        # ← DEBUG: Loggear valores de análisis (solo en modo debug)
        if hasattr(self, '_debug_mode') and self._debug_mode:
            print(f"🔍 POLICY DEBUG: conf_ok={conf_ok}, side_hint={side_hint}, analysis={analysis}")
            print(f"🔍 POSITION DEBUG: position={position}")

        pos_side = int(position.get("side", 0))
        has_pos = pos_side != 0

        # ---------- CIERRES ----------
        if has_pos:
            # a) giro de señal con confluencia
            if conf_ok and ((pos_side > 0 and side_hint < 0) or (pos_side < 0 and side_hint > 0)):
                self._conf_loss_count = 0
                return Decision(should_close_all=True, price_hint=price_exec)

            # b) pérdida de confluencia persistente (2 barras seguidas sin confluencia)
            if not conf_ok:
                self._conf_loss_count += 1
                if self._conf_loss_count >= 2:
                    self._conf_loss_count = 0
                    return Decision(should_close_all=True, price_hint=price_exec)
            else:
                self._conf_loss_count = 0

            # no cerrar por policy → nada que abrir si ya hay pos
            return Decision(should_open=False, price_hint=price_exec)

        # ---------- APERTURA ----------
        # Verificar confluencia y señal válida
        if not conf_ok or side_hint == 0:
            return Decision(should_open=False, side=0, price_hint=price_exec)

        # Verificar deduplicación
        if dedup_block(ts_now, self._last_open_ts, window_bars=self.cfg.dedup_open_window_bars, base_tf_ms=self.base_tf_ms):
            return Decision(should_open=False, side=0, price_hint=price_exec)

        # Obtener ATR del TF de ejecución
        atr_val = float(features.get(self.exec_tf, {}).get("atr14", 0.0) or 0.0)
        if hasattr(self, '_debug_mode') and self._debug_mode:
            print(f"🔍 ATR DEBUG: atr_val={atr_val}, features={features.get(self.exec_tf, {})}")

        # Multiplicadores desde risk.yaml
        k_sl = float(self.risk_cfg.common.default_levels.sl_atr_mult)
        k_tp = float(self.risk_cfg.common.default_levels.tp_r_multiple)

        sl, tp = sl_tp_from_atr(price_exec, atr_val, side_hint, k_sl=k_sl, k_tp=k_tp)
        if hasattr(self, '_debug_mode') and self._debug_mode:
            print(f"🔍 SL/TP DEBUG: sl={sl}, tp={tp}, price_exec={price_exec}, side_hint={side_hint}, k_sl={k_sl}, k_tp={k_tp}")
        
        # ← FALLBACK: Si ATR no está disponible, usar porcentajes fijos
        if sl is None or tp is None:
            if hasattr(self, '_debug_mode') and self._debug_mode:
                print(f"🔧 ATR FALLBACK: Usando porcentajes fijos")
            sl_pct = 0.02  # 2% SL
            tp_pct = 0.03  # 3% TP
            if side_hint > 0:  # long
                sl = price_exec * (1 - sl_pct)
                tp = price_exec * (1 + tp_pct)
            else:  # short
                sl = price_exec * (1 + sl_pct)
                tp = price_exec * (1 - tp_pct)
            if hasattr(self, '_debug_mode') and self._debug_mode:
                print(f"🔧 FALLBACK RESULT: sl={sl}, tp={tp}")
        
        ttl_bars = int(self.risk_cfg.common.default_levels.ttl_bars_default)

        # Si sl o tp son None, no abrir
        if sl is None or tp is None or ttl_bars <= 0:
            if hasattr(self, '_debug_mode') and self._debug_mode:
                print(f"🚫 POLICY BLOCKED: sl={sl}, tp={tp}, ttl_bars={ttl_bars}")
            return Decision(should_open=False, side=0, price_hint=price_exec)

        # Al abrir: actualizar timestamp de última apertura
        self._last_open_ts = ts_now
        self._conf_loss_count = 0

        if hasattr(self, '_debug_mode') and self._debug_mode:
            print(f"🚀 POLICY DECISION: should_open=True, side={side_hint}, sl={sl}, tp={tp}, ttl={ttl_bars}")
        return Decision(
            should_open=True,
            side=side_hint,
            price_hint=price_exec,
            sl=sl,
            tp=tp,
            ttl_bars=ttl_bars,
            trailing=True,
        )
