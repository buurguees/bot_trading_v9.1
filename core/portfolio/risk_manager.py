from __future__ import annotations
from dataclasses import dataclass

@dataclass
class RiskContext:
    equity_usdt: float
    atr: float
    price: float
    min_notional: float
    risk_per_trade_pct: float

def position_size(ctx: RiskContext) -> float:
    """Sizing simple basado en ATR y riesgo porcentual del equity."""
    risk_usdt = ctx.equity_usdt * ctx.risk_per_trade_pct
    # stop ≈ 1*ATR → qty = risk / ATR
    qty = max(risk_usdt / max(ctx.atr, 1e-8), ctx.min_notional/ max(ctx.price, 1e-8))
    return qty
