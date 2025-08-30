from __future__ import annotations
import math, random, time
from typing import Tuple
from .order import Order, Fill, Side

class ExecutionSim:
    """Simulador determinista con latencia y slippage sencillos."""
    def __init__(self, maker_fee=0.0002, taker_fee=0.0004, latency_ms=50, slippage_bp=1.0, seed=1337):
        self.maker_fee = maker_fee
        self.taker_fee = taker_fee
        self.latency_ms = latency_ms
        self.slippage_bp = slippage_bp
        random.seed(seed)

    def _slip(self, price: float, side: Side) -> float:
        s = price * (self.slippage_bp / 10000.0)
        return price + (s if side == Side.BUY else -s)

    def execute(self, order: Order, best_bid: float, best_ask: float, ts_ms: int) -> Fill:
        time.sleep(self.latency_ms / 1000.0)
        if order.type.name == "MARKET":
            px = best_ask if order.side == Side.BUY else best_bid
            px = self._slip(px, order.side)
            fee = self.taker_fee * (order.qty * px)
        else:
            # Limit muy simple: cruza si el precio toca el lado
            if order.side == Side.BUY and order.price >= best_ask:
                px = min(order.price, best_ask)
                fee = self.maker_fee * (order.qty * px)
            elif order.side == Side.SELL and order.price <= best_bid:
                px = max(order.price, best_bid)
                fee = self.maker_fee * (order.qty * px)
            else:
                # No fill (para simpleza rellenamos como no ejecutado)
                px = None
                fee = 0.0
        if px is None:
            # Sin fill: retornamos fill qty=0 (no error)
            return Fill(order_id=order.id, symbol=order.symbol, side=order.side, qty=0.0, price=0.0, fee=0.0, ts=ts_ms)
        return Fill(order_id=order.id, symbol=order.symbol, side=order.side, qty=order.qty, price=float(px), fee=float(fee), ts=ts_ms)
