# base_env/risk/manager.py
# Descripción: Sizing spot y mantenimiento de posición (SL/TP/TTL/Trailing ATR).
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
    # ← NUEVO: información de leverage para futuros
    leverage_used: Optional[float] = None
    notional_effective: Optional[float] = None
    notional_max: Optional[float] = None


class RiskManager:
    def __init__(self, cfg: RiskConfig, symbol_meta: SymbolMeta) -> None:
        self.cfg = cfg
        self.symbol_meta = symbol_meta
        # Parámetros por defecto (puedes movedlos a RiskConfig si quieres)
        self.trail_mult_atr: float = 1.0  # 1 x ATR
        self.ttl_enabled: bool = True
        self.trail_enabled: bool = True

    def apply(self, portfolio, position, decision, obs) -> SizedDecision:
        # ----- CIERRES explícitos de la policy -----
        if decision.should_close_all:
            q = float(position.qty)
            if q <= 0:
                return SizedDecision(False, 0, 0.0, decision.price_hint, decision.sl, decision.tp, False, False, 0.0)
            return SizedDecision(False, 0, 0.0, decision.price_hint, decision.sl, decision.tp, False, True, q)

        if decision.should_close_partial:
            q = float(decision.close_qty)
            if q <= 0 or q > float(position.qty):
                q = max(0.0, float(position.qty) * 0.5)
            return SizedDecision(False, 0, 0.0, decision.price_hint, decision.sl, decision.tp, True, False, q)

        # ----- APERTURA -----
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
                if qty <= 0:
                    return SizedDecision(False, 0, 0.0, decision.price_hint, decision.sl, decision.tp, False, False, 0.0)
                return SizedDecision(True, decision.side, qty, decision.price_hint, decision.sl, decision.tp, False, False, 0.0)

            else:
                # Futuros con apalancamiento dinámico
                # Obtener leverage desde la configuración del entorno
                leverage = getattr(portfolio, 'leverage', 1.0)
                if leverage <= 1.0:
                    leverage = 1.0  # Fallback a sin apalancamiento
                
                return self.size_futures(
                    portfolio=portfolio,
                    decision=decision,
                    leverage=leverage,
                    account_equity=portfolio.equity_quote
                )

        # Default
        return SizedDecision(False, 0, 0.0, decision.price_hint, decision.sl, decision.tp, False, False, 0.0)

    def size_futures(self, portfolio, decision, leverage: float, account_equity: float):
        """Sizing en futuros: limita notional por equity*leverage y riesgo por SL."""
        from dataclasses import dataclass
        @dataclass
        class _Sized:
            should_open: bool = True
            side: int = 0
            qty: float = 0.0
            price_hint: float = 0.0
            sl: float | None = None
            tp: float | None = None
            should_close_all: bool = False
            should_close_partial: bool = False
            close_qty: float = 0.0
            # ← NUEVO: campos específicos para futuros
            leverage_used: float | None = None
            notional_effective: float | None = None
            notional_max: float | None = None
        if decision is None or not decision.should_open:
            return _Sized(should_open=False)

        price = float(decision.price_hint)
        risk_pct = float(getattr(self.cfg, "risk_per_trade_pct", 0.01))
        notional_cap = float(account_equity) * float(leverage)
        risk_cash = float(account_equity) * risk_pct

        stop_dist = abs(price - float(decision.sl)) if decision.sl else price * 0.01
        qty = risk_cash / max(stop_dist, 1e-9)
        # no exceder notional máximo
        if qty * price > notional_cap:
            qty = notional_cap / price

        return _Sized(
            side=decision.side, 
            qty=max(qty, 0.0), 
            price_hint=price, 
            sl=decision.sl, 
            tp=decision.tp,
            leverage_used=leverage,
            notional_effective=qty * price,
            notional_max=notional_cap
        )

    def check_bankruptcy(self, portfolio, initial_balance: float, events_bus, ts_now: int) -> bool:
        """
        Verifica si el portfolio ha entrado en quiebra.
        Quiebra = equity <= initial_balance * threshold_pct
        """
        threshold_pct = float(getattr(self.cfg.common, "bankruptcy", {}).get("threshold_pct", 20.0)) / 100.0
        bankruptcy_threshold = initial_balance * threshold_pct
        
        if portfolio.equity_quote <= bankruptcy_threshold:
            # Emitir evento de quiebra
            events_bus.emit("BANKRUPTCY", ts=ts_now, 
                           initial_balance=initial_balance,
                           current_equity=portfolio.equity_quote,
                           threshold_pct=threshold_pct * 100.0,
                           drawdown_pct=((portfolio.equity_quote - initial_balance) / initial_balance) * 100.0)
            return True
        return False

    def maintenance(self, portfolio, position, broker, events_bus, obs, exec_tf: str, ts_now: int):
        """
        Verifica SL/TP/TTL/Trailing.
        Si detecta condición de cierre TOTAL, devuelve SizedDecision de cierre todo.
        Si sólo actualiza SL trailing, no devuelve nada (efecto en 'position.sl').
        """
        if position.side == 0 or position.qty <= 0:
            return None

        close_price = float(broker.get_price() or position.entry_price)
        # SL/TP HIT
        if position.sl is not None:
            if position.side > 0 and close_price <= float(position.sl):
                events_bus.emit("SL_HIT", ts=ts_now, price=close_price, sl=position.sl)
                return SizedDecision(False, 0, 0.0, close_price, position.sl, position.tp, False, True, float(position.qty))
            if position.side < 0 and close_price >= float(position.sl):
                events_bus.emit("SL_HIT", ts=ts_now, price=close_price, sl=position.sl)
                return SizedDecision(False, 0, 0.0, close_price, position.sl, position.tp, False, True, float(position.qty))

        if position.tp is not None:
            if position.side > 0 and close_price >= float(position.tp):
                events_bus.emit("TP_HIT", ts=ts_now, price=close_price, tp=position.tp)
                return SizedDecision(False, 0, 0.0, close_price, position.sl, position.tp, False, True, float(position.qty))
            if position.side < 0 and close_price <= float(position.tp):
                events_bus.emit("TP_HIT", ts=ts_now, price=close_price, tp=position.tp)
                return SizedDecision(False, 0, 0.0, close_price, position.sl, position.tp, False, True, float(position.qty))

        # TTL
        if self.ttl_enabled and position.ttl_bars > 0:
            position.ttl_bars -= 1
            if position.ttl_bars <= 0:
                events_bus.emit("TTL_CLOSE", ts=ts_now)
                return SizedDecision(False, 0, 0.0, close_price, position.sl, position.tp, False, True, float(position.qty))

        # Trailing ATR simple
        if self.trail_enabled:
            atr_exec = obs.get("features", {}).get(exec_tf, {}).get("atr14")
            if atr_exec is not None and atr_exec > 0:
                if position.side > 0:
                    new_trail = close_price - self.trail_mult_atr * float(atr_exec)
                    if position.sl is None:
                        position.sl = new_trail
                        events_bus.emit("TRAIL_SET", ts=ts_now, sl=position.sl)
                    else:
                        if new_trail > float(position.sl):
                            position.sl = new_trail
                            events_bus.emit("TRAIL_MOVE", ts=ts_now, sl=position.sl)
                else:  # short
                    new_trail = close_price + self.trail_mult_atr * float(atr_exec)
                    if position.sl is None:
                        position.sl = new_trail
                        events_bus.emit("TRAIL_SET", ts=ts_now, sl=position.sl)
                    else:
                        if new_trail < float(position.sl):
                            position.sl = new_trail
                            events_bus.emit("TRAIL_MOVE", ts=ts_now, sl=position.sl)

        # Nada que cerrar ahora
        return None
