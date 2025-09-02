# base_env/accounting/ledger.py
# Descripción: Contabilidad Spot/Futuros con distinción Balance(cash) vs Equity (cash + mark-to-market).
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
    # ← NUEVO: tracking temporal de la posición
    open_ts: Optional[int] = None      # timestamp de apertura
    bars_held: int = 0                 # barras que estuvo realmente abierta
    mfe: float = 0.0
    mae: float = 0.0
    unrealized_pnl: float = 0.0
    realized_pnl: float = 0.0

    def reset(self) -> None:
        self.side = 0; self.qty = 0.0; self.entry_price = 0.0
        self.sl = None; self.tp = None; self.trail = None
        self.ttl_bars = 0; self.mfe = 0.0; self.mae = 0.0
        self.unrealized_pnl = 0.0; self.realized_pnl = 0.0
        # ← NUEVO: reset de tracking temporal
        self.open_ts = None; self.bars_held = 0

    def to_dict(self) -> Dict: return asdict(self)

@dataclass
class PortfolioState:
    market: Literal["spot", "futures"] = "spot"
    # Balance (cash realizado en USDT)
    cash_quote: float = 0.0
    # Inventario (base asset)
    equity_base: float = 0.0
    # Equity (valor total en USDT; se recalcula en update_unrealized)
    equity_quote: float = 0.0
    # Objetivo (solo informativo/log)
    target_quote: float = 0.0
    drawdown_day_pct: float = 0.0

    def reset(self, initial_cash: float = 10000.0, target_cash: float = 1_000_000.0) -> None:
        if self.market == "spot":
            self.cash_quote = float(initial_cash)
            self.equity_base = 0.0
            self.equity_quote = self.cash_quote  # sin posición
        else:
            # TODO: futuros: cash = balance margin
            self.cash_quote = float(initial_cash)
            self.equity_quote = self.cash_quote
        self.target_quote = float(target_cash)
        self.drawdown_day_pct = 0.0

    def to_dict(self) -> Dict: return asdict(self)

class Accounting:
    def __init__(self, fees_cfg: Dict, market: str) -> None:
        # fees_cfg puede ser pydantic u objeto/dict
        if hasattr(fees_cfg, "model_dump"):
            fees_cfg = fees_cfg.model_dump()
        self.fees_cfg = fees_cfg
        self.market = market

    def _taker_bps(self, portfolio: PortfolioState) -> float:
        if portfolio.market == "spot":
            return float(self.fees_cfg["spot"]["taker_fee_bps"])
        return float(self.fees_cfg["futures"]["taker_fee_bps"])

    def apply_open(self, fill: Dict, portfolio: PortfolioState, pos: PositionState, cfg) -> None:
        price = float(fill["price"]); qty = float(fill["qty"])
        notional = price * qty
        fee = taker_fee(notional, self._taker_bps(portfolio))
        pos.side = int(fill.get("side", 0)); pos.qty = qty; pos.entry_price = price
        pos.sl = fill.get("sl", pos.sl); pos.tp = fill.get("tp", pos.tp)
        if portfolio.market == "spot":
            portfolio.cash_quote -= (notional + fee)
            portfolio.equity_base += qty
        # equity_quote se recalcula en update_unrealized

    def apply_close(self, fill: Dict, portfolio: PortfolioState, pos: PositionState, cfg) -> float:
        price = float(fill["price"]); qty = float(fill.get("qty", pos.qty))
        notional = price * qty
        fee = taker_fee(notional, self._taker_bps(portfolio))
        side = pos.side; entry = pos.entry_price
        realized = (price - entry) * qty * (1 if side > 0 else -1)

        if portfolio.market == "spot":
            portfolio.cash_quote += (notional - fee)
            portfolio.equity_base -= qty

        pos.realized_pnl += realized
        # cerrar posición (simple)
        pos.side = 0; pos.qty = 0.0; pos.entry_price = 0.0
        pos.sl = None; pos.tp = None; pos.unrealized_pnl = 0.0
        pos.mfe = 0.0; pos.mae = 0.0
        return realized

    def update_unrealized(self, broker, pos: PositionState, portfolio: PortfolioState) -> None:
        # mark-to-market
        last = float(broker.get_price() or pos.entry_price or 0.0)
        if pos.side == 0 or pos.qty <= 0:
            pos.unrealized_pnl = 0.0
        else:
            pos.unrealized_pnl = (last - pos.entry_price) * pos.qty * (1 if pos.side > 0 else -1)
            pos.mfe = max(pos.mfe, pos.unrealized_pnl)
            pos.mae = min(pos.mae, pos.unrealized_pnl)
        # equity = cash + valor inventario
        portfolio.equity_quote = float(portfolio.cash_quote + portfolio.equity_base * last)

    def is_end_of_data(self, broker) -> bool:
        return False
