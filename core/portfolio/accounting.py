from __future__ import annotations
from typing import Dict

def unrealized_pnl(positions: Dict[str, float], avg_price: Dict[str, float], marks: Dict[str, float]) -> float:
    upnl = 0.0
    for sym, qty in positions.items():
        if qty == 0:
            continue
        upnl += (marks.get(sym, avg_price.get(sym, 0.0)) - avg_price.get(sym, 0.0)) * qty
    return upnl
