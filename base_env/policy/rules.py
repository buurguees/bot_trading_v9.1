# base_env/policy/rules.py
# Descripción: Reglas de decisión: confluencia mínima, deduplicación, SL/TP por ATR.

from __future__ import annotations
from typing import Any, Dict, Optional


def confluence_ok(analysis: Dict[str, Any], min_confidence: float) -> bool:
    conf = float(analysis.get("confidence", 0.0))
    return conf >= float(min_confidence)


def side_from_hint(analysis: Dict[str, Any]) -> int:
    return int(analysis.get("side_hint", 0))


def dedup_block(ts_now: int, last_open_ts: Optional[int], window_bars: int, base_tf_ms: int) -> bool:
    if last_open_ts is None:
        return False
    # Si el tiempo transcurrido en ms es menor a ventana * tamaño_barra, bloquea
    return (ts_now - last_open_ts) < (window_bars * base_tf_ms)


def sl_tp_from_atr(entry: float, atr_val: Optional[float], side: int, k_sl: float = 1.5, k_tp: float = 2.0) -> tuple[Optional[float], Optional[float]]:
    if atr_val is None or atr_val <= 0 or entry <= 0 or side == 0:
        return None, None
    if side > 0:  # long
        sl = entry - k_sl * atr_val
        tp = entry + k_tp * atr_val
    else:         # short
        sl = entry + k_sl * atr_val
        tp = entry - k_tp * atr_val
    return sl, tp
