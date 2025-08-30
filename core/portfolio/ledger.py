from __future__ import annotations
from dataclasses import dataclass
from typing import List, Dict
from ..oms.order import Fill, Side

@dataclass
class LedgerLine:
    ts: int
    event: str
    data: dict

class Ledger:
    def __init__(self):
        self.lines: List[LedgerLine] = []
        self.positions: Dict[str, float] = {}
        self.avg_price: Dict[str, float] = {}
        self.realized_pnl: Dict[str, float] = {}

    def log(self, ts: int, event: str, **data):
        self.lines.append(LedgerLine(ts=ts, event=event, data=data))

    def apply_fill(self, f: Fill):
        if f.qty <= 0:
            return
        pos = self.positions.get(f.symbol, 0.0)
        avg = self.avg_price.get(f.symbol, 0.0)
        if f.side == Side.BUY:
            new_qty = pos + f.qty
            new_avg = ((pos * avg) + (f.qty * f.price)) / new_qty if new_qty != 0 else 0.0
            self.positions[f.symbol] = new_qty
            self.avg_price[f.symbol] = new_avg
        else:
            # Vende: calcula PnL realizado contra avg
            realized = (f.price - avg) * f.qty
            self.realized_pnl[f.symbol] = self.realized_pnl.get(f.symbol, 0.0) + realized - f.fee
            self.positions[f.symbol] = pos - f.qty
            if self.positions[f.symbol] <= 1e-12:
                self.positions[f.symbol] = 0.0
                self.avg_price[f.symbol] = 0.0
        self.log(f.ts, "fill", fill=f.__dict__)
