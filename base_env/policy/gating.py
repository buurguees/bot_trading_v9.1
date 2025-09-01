# base_env/policy/gating.py
# Descripción: Política mínima: abre siguiendo side_hint si hay confluencia >= umbral
# y no está bloqueado por deduplicación. SL/TP por ATR del TF de ejecución.

from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Optional
from ..config.models import HierarchicalConfig
from ..tfs.calendar import tf_to_ms
from .rules import confluence_ok, side_from_hint, dedup_block, sl_tp_from_atr


@dataclass
class Decision:
    should_open: bool = False
    side: int = 0               # -1, 0, +1
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
        self.exec_tf = exec_tf or (cfg.execute_tfs[0] if cfg.execute_tfs else base_tf)
        self.base_tf_ms = tf_to_ms(base_tf)

    def decide(self, obs: dict[str, Any]) -> Decision:
        ts_now = int(obs["ts"])
        analysis = obs.get("analysis", {})
        features = obs.get("features", {})
        tfs = obs.get("tfs", {})

        # 1) confluencia
        if not confluence_ok(analysis, self.cfg.min_confidence):
            return Decision(should_open=False, side=0, price_hint=float(tfs.get(self.exec_tf, {}).get("close", 0.0)))

        # 2) lado propuesto por análisis
        side = side_from_hint(analysis)
        if side == 0:
            return Decision(should_open=False, side=0, price_hint=float(tfs.get(self.exec_tf, {}).get("close", 0.0)))

        # 3) deduplicación
        if dedup_block(ts_now, self._last_open_ts, self.cfg.dedup_open_window_bars, self.base_tf_ms):
            return Decision(should_open=False, side=0, price_hint=float(tfs.get(self.exec_tf, {}).get("close", 0.0)))

        # 4) SL/TP por ATR del TF de ejecución
        atr_exec = features.get(self.exec_tf, {}).get("atr14")
        price = float(tfs.get(self.exec_tf, {}).get("close", 0.0))
        sl, tp = sl_tp_from_atr(price, atr_exec, side, k_sl=1.5, k_tp=2.0)

        # 5) TTL básico (p. ej. 200 barras)
        ttl_bars = 200

        # registrar dedup
        self._last_open_ts = ts_now

        return Decision(
            should_open=True,
            side=side,
            price_hint=price,
            sl=sl,
            tp=tp,
            ttl_bars=ttl_bars,
            trailing=False,
            should_close_partial=False,
            should_close_all=False,
            close_qty=0.0,
        )
