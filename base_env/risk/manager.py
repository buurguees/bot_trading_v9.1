# base_env/risk/manager.py
# Descripción: Aplica sizing spot (y deja hooks para futuros), respeta exposición y breakers.

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional
from ..config.models import RiskConfig, SymbolMeta
from .rules import size_spot


@dataclass
class SizedDecision:
    should_open: bool
    side: int
    qty: float
    price_hint: float
    sl: Optional[float]
    tp: Optional[float]
    should_close_partial: bool
    should_close_all: bool
    close_qty: float


class RiskManager:
    def __init__(self, cfg: RiskConfig, symbol_meta: SymbolMeta) -> None:
        self.cfg = cfg
        self.symbol_meta = symbol_meta

    def apply(self, portfolio, position, decision, obs) -> SizedDecision:
        # Si no se pretende abrir/cerrar, pasa tal cual
        if not decision.should_open and not decision.should_close_all and not decision.should_close_partial:
            return SizedDecision(False, 0, 0.0, decision.price_hint, decision.sl, decision.tp, False, False, 0.0)

        if decision.should_open and decision.side != 0:
            if portfolio.market == "spot":
                entry = float(decision.price_hint)
                sl = decision.sl
                qty = size_spot(
                    equity_quote=portfolio.equity_quote,
                    risk_pct_per_trade=self.cfg.spot.risk_pct_per_trade,
                    entry=entry,
                    sl=sl,
                    min_notional=float(self.symbol_meta.filters.get("minNotional", 5.0)),
                    lot_step=float(self.symbol_meta.filters.get("lotStep", 0.0001)),
                )
                # Si no alcanza minNotional o qty=0, cancela apertura
                if qty <= 0:
                    return SizedDecision(False, 0, 0.0, decision.price_hint, decision.sl, decision.tp, False, False, 0.0)
                return SizedDecision(True, decision.side, qty, decision.price_hint, decision.sl, decision.tp, False, False, 0.0)

            else:
                # TODO: Futuros (apalancamiento ≤ 3x). Por ahora, no abrir en demo.
                return SizedDecision(False, 0, 0.0, decision.price_hint, decision.sl, decision.tp, False, False, 0.0)

        # Cierres (demo: no generamos cierres parciales automáticos aquí)
        return SizedDecision(False, 0, 0.0, decision.price_hint, decision.sl, decision.tp, decision.should_close_partial, decision.should_close_all, decision.close_qty)

    def maintenance(self, portfolio, position, broker, events_bus) -> None:
        # TODO: breakers, trailing, TTL, mantenimiento (futuros)
        pass
