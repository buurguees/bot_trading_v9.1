# base_env/policy/gating.py
# Descripción: Política mínima operativa:
# - Abre si hay confluencia y NO hay posición (dedup)
# - SL/TP por ATR (exec TF → fallback base TF)
# - Cierra TODO por giro con confluencia o por pérdida de confluencia persistente

from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Optional
from ..config.models import HierarchicalConfig
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
    def __init__(self, cfg: HierarchicalConfig, exec_tf: Optional[str] = None, base_tf: str = "1m") -> None:
        self.cfg = cfg
        self._last_open_ts: Optional[int] = None
        self._conf_loss_count: int = 0  # contador de pérdida de confluencia
        self.exec_tf = exec_tf or (cfg.execute_tfs[0] if cfg.execute_tfs else base_tf)
        self.base_tf = base_tf
        self.base_tf_ms = tf_to_ms(base_tf)

    def decide(self, obs: dict[str, Any]) -> Decision:
        ts_now = int(obs["ts"])
        analysis = obs.get("analysis", {})
        features = obs.get("features", {})
        tfs = obs.get("tfs", {})
        position = obs.get("position", {}) or {}

        price_exec = float(tfs.get(self.exec_tf, {}).get("close", 0.0))
        side_hint = side_from_hint(analysis)
        conf_ok = confluence_ok(analysis, self.cfg.min_confidence)

        has_pos = int(position.get("side", 0)) != 0
        pos_side = int(position.get("side", 0))

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
        if not conf_ok or side_hint == 0:
            return Decision(should_open=False, side=0, price_hint=price_exec)

        if dedup_block(ts_now, self._last_open_ts, self.cfg.dedup_open_window_bars, self.base_tf_ms):
            return Decision(should_open=False, side=0, price_hint=price_exec)

        # SL/TP por ATR: exec → fallback base
        atr_exec = features.get(self.exec_tf, {}).get("atr14")
        if atr_exec is None:
            atr_exec = features.get(self.base_tf, {}).get("atr14")

        sl, tp = sl_tp_from_atr(price_exec, atr_exec, side_hint, k_sl=1.5, k_tp=2.0)
        ttl_bars = 200

        self._last_open_ts = ts_now
        self._conf_loss_count = 0

        return Decision(
            should_open=True,
            side=side_hint,
            price_hint=price_exec,
            sl=sl,
            tp=tp,
            ttl_bars=ttl_bars,
            trailing=True,
        )
