# base_env/accounting/ledger.py
# Descripción: Contabilidad: balances (spot/futuros), fees, PnL R/UR, MFE/MAE, drawdown.

from __future__ import annotations
from dataclasses import dataclass, field, asdict
from typing import Optional, Dict, Literal
from .fees import taker_fee


@dataclass
class PositionState:
    side: int = 0
    qty: float = 0.0
    entry_price: float = 0.0
    sl: Optional[float] = None
    tp: Optional[float] = None
    trail: Optional[float] = None
    ttl_bars: int = 0
    mfe: float = 0.0
    mae: float = 0.0
    unrealized_pnl: float = 0.0
    realized_pnl: float = 0.0

    def reset(self) -> None:
        self.side = 0
        self.qty = 0.0
        self.entry_price = 0.0
        self.sl = None
        self.tp = None
        self.trail = None
        self.ttl_bars = 0
        self.mfe = 0.0
        self.mae = 0.0
        self.unrealized_pnl = 0.0
        self.realized_pnl = 0.0

    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class PortfolioState:
    market: Literal["spot", "futures"] = "spot"
    equity_quote: float = 0.0      # spot
    equity_base: float = 0.0       # spot
    balance: float = 0.0           # futures (USDT o moneda de margen)
    drawdown_day_pct: float = 0.0

    def reset(self) -> None:
        # Inicialización de demo: equity 10,000
        if self.market == "spot":
            self.equity_quote = 10_000.0
            self.equity_base = 0.0
        else:
            self.balance = 10_000.0
        self.drawdown_day_pct = 0.0

    def to_dict(self) -> Dict:
        return asdict(self)


class Accounting:
    def __init__(self, fees_cfg, market: str) -> None:
        self.fees_cfg = fees_cfg
        self.market = market

    def apply_open(self, fill: Dict, portfolio: PortfolioState, pos: PositionState, cfg) -> None:
        price = float(fill["price"])
        qty = float(fill["qty"])
        notional = price * qty

        # fees spot - maneja tanto objetos como diccionarios
        if hasattr(self.fees_cfg, 'spot'):
            bps = float(self.fees_cfg.spot["taker_fee_bps"]) if portfolio.market == "spot" else float(self.fees_cfg.futures["taker_fee_bps"])
        else:
            bps = float(self.fees_cfg["spot"]["taker_fee_bps"]) if portfolio.market == "spot" else float(self.fees_cfg["futures"]["taker_fee_bps"])
        fee = taker_fee(notional, bps)

        # actualizar posición (abre o apila en demo simple)
        pos.side = int(fill.get("side", 0))
        pos.qty = qty
        pos.entry_price = price
        pos.sl = fill.get("sl", pos.sl)
        pos.tp = fill.get("tp", pos.tp)

        # actualizar cartera spot
        if portfolio.market == "spot":
            portfolio.equity_quote -= (notional + fee)
            portfolio.equity_base += qty

    def apply_close(self, fill: Dict, portfolio: PortfolioState, pos: PositionState, cfg) -> float:
        price = float(fill["price"])
        qty = float(fill.get("qty", pos.qty))
        notional = price * qty

        # fees - maneja tanto objetos como diccionarios
        if hasattr(self.fees_cfg, 'spot'):
            bps = float(self.fees_cfg.spot["taker_fee_bps"]) if portfolio.market == "spot" else float(self.fees_cfg.futures["taker_fee_bps"])
        else:
            bps = float(self.fees_cfg["spot"]["taker_fee_bps"]) if portfolio.market == "spot" else float(self.fees_cfg["futures"]["taker_fee_bps"])
        fee = taker_fee(notional, bps)

        # PnL realizado
        side = pos.side
        entry = pos.entry_price
        pnl = (price - entry) * qty * (1 if side > 0 else -1)

        # actualizar cartera spot
        if portfolio.market == "spot":
            portfolio.equity_quote += (notional - fee)
            portfolio.equity_base -= qty

        pos.realized_pnl += pnl
        # Cerrar posición (simple: todo)
        pos.side = 0
        pos.qty = 0.0
        pos.entry_price = 0.0
        pos.sl = None
        pos.tp = None
        pos.unrealized_pnl = 0.0
        pos.mfe = 0.0
        pos.mae = 0.0

        return pnl

    def update_unrealized(self, broker, pos: PositionState, portfolio: PortfolioState) -> None:
        if pos.side == 0 or pos.qty <= 0:
            pos.unrealized_pnl = 0.0
            return
        last = broker.get_price() or pos.entry_price
        side = pos.side
        qty = pos.qty
        pos.unrealized_pnl = (last - pos.entry_price) * qty * (1 if side > 0 else -1)
        # MFE/MAE
        pos.mfe = max(pos.mfe, pos.unrealized_pnl)
        pos.mae = min(pos.mae, pos.unrealized_pnl)

    def is_end_of_data(self, broker) -> bool:
        # Demo: siempre False; el caller decide cuándo parar por rango o cursor
        return False
