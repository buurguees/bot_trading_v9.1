from __future__ import annotations
import uuid
from .order import Order, Side, OrderType
from .execution_sim import ExecutionSim

class SimRouter:
    def __init__(self, exec_sim: ExecutionSim):
        self.exec = exec_sim

    def place_market(self, symbol: str, side: Side, qty: float, best_bid: float, best_ask: float, ts_ms: int):
        order = Order(id=str(uuid.uuid4()), symbol=symbol, side=side, qty=qty, type=OrderType.MARKET)
        return self.exec.execute(order, best_bid, best_ask, ts_ms)
