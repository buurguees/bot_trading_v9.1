from __future__ import annotations
from dataclasses import dataclass
from enum import Enum, auto
from typing import Optional

class Side(str, Enum):
    BUY = "BUY"
    SELL = "SELL"

class OrderType(str, Enum):
    MARKET = "MARKET"
    LIMIT = "LIMIT"

class TimeInForce(str, Enum):
    GTC = "GTC"
    IOC = "IOC"
    FOK = "FOK"

@dataclass
class Order:
    id: str
    symbol: str
    side: Side
    qty: float
    price: Optional[float] = None
    type: OrderType = OrderType.MARKET
    tif: TimeInForce = TimeInForce.GTC
    reduce_only: bool = False

@dataclass
class Fill:
    order_id: str
    symbol: str
    side: Side
    qty: float
    price: float
    fee: float = 0.0
    ts: int = 0

@dataclass
class Position:
    symbol: str
    qty: float = 0.0
    avg_price: float = 0.0
