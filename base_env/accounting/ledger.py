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
    # Inventario (base asset) - solo para spot
    equity_base: float = 0.0
    # Equity (valor total en USDT; se recalcula en update_unrealized)
    equity_quote: float = 0.0
    # Objetivo (solo informativo/log)
    target_quote: float = 0.0
    drawdown_day_pct: float = 0.0
    # ← NUEVO: Para futuros - margen usado
    used_margin: float = 0.0

    def reset(self, initial_cash: float = 10000.0, target_cash: float = 1_000_000.0) -> None:
        if self.market == "spot":
            self.cash_quote = float(initial_cash)
            self.equity_base = 0.0
            self.equity_quote = self.cash_quote  # sin posición
        else:
            # ← NUEVO: Futuros - cash = balance disponible, sin margen usado
            self.cash_quote = float(initial_cash)
            self.used_margin = 0.0
            self.equity_quote = self.cash_quote  # sin posición
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
        # ← CRÍTICO: Asignar timestamp de apertura
        pos.open_ts = fill.get("ts", None)
        pos.bars_held = 0  # Reset contador de barras
        
        if portfolio.market == "spot":
            # ← NUEVO: Spot con soporte para shorts (margen)
            if pos.side > 0:  # Long
                # Spot Long: descuenta notional completo + fee
                portfolio.cash_quote -= (notional + fee)
                portfolio.equity_base += qty
            else:  # Short
                # Spot Short: usa margen (como futures) + fee
                margin_required = notional * 0.5  # 50% margen para shorts en spot
                portfolio.cash_quote -= (margin_required + fee)
                portfolio.used_margin += margin_required
                portfolio.equity_base -= qty  # Deuda en base asset
        else:
            # ← NUEVO: Futuros: solo descuenta margen + fee
            # Calcular margen requerido (notional / leverage)
            leverage = getattr(cfg, 'leverage', 3.0)  # ← CORREGIDO: 3.0 por defecto para futuros
            margin_required = notional / leverage
            portfolio.cash_quote -= (margin_required + fee)
            portfolio.used_margin += margin_required
        # equity_quote se recalcula en update_unrealized

    def apply_close(self, fill: Dict, portfolio: PortfolioState, pos: PositionState, cfg) -> float:
        price = float(fill["price"]); qty = float(fill.get("qty", pos.qty))
        notional = price * qty
        fee = taker_fee(notional, self._taker_bps(portfolio))
        side = pos.side; entry = pos.entry_price
        realized = (price - entry) * qty * (1 if side > 0 else -1)

        if portfolio.market == "spot":
            # ← NUEVO: Spot con soporte para shorts
            if side > 0:  # Cerrar Long
                # Spot Long: recibe notional completo - fee
                portfolio.cash_quote += (notional - fee)
                portfolio.equity_base -= qty
            else:  # Cerrar Short
                # Spot Short: libera margen + recibe PnL - fee
                margin_liberated = (entry * qty) * 0.5  # 50% margen
                portfolio.cash_quote += (realized + margin_liberated - fee)
                portfolio.used_margin -= margin_liberated
                portfolio.equity_base += qty  # Reduce deuda en base asset
        else:
            # ← NUEVO: Futuros: libera margen + recibe PnL - fee
            # Calcular margen liberado
            leverage = getattr(cfg, 'leverage', 3.0)  # ← CORREGIDO: 3.0 por defecto para futuros
            margin_liberated = (entry * qty) / leverage
            portfolio.cash_quote += (realized + margin_liberated - fee)
            portfolio.used_margin -= margin_liberated

        pos.realized_pnl += realized
        # cerrar posición (simple)
        pos.side = 0; pos.qty = 0.0; pos.entry_price = 0.0
        pos.sl = None; pos.tp = None; pos.unrealized_pnl = 0.0
        pos.mfe = 0.0; pos.mae = 0.0
        pos.open_ts = None; pos.bars_held = 0
        
        # ← CORREGIDO: NO forzar equity = balance, se recalcula en update_unrealized
        # portfolio.equity_quote se actualizará en update_unrealized()
        
        # ← NUEVO: Validar que se liberó todo el margen
        self._validate_portfolio_consistency(pos, portfolio)
        
        return realized

    def update_unrealized(self, broker, pos: PositionState, portfolio: PortfolioState) -> None:
        """
        DEPRECATED: Este método ya no se usa en el nuevo flujo.
        El cálculo de unrealized PnL se hace directamente en base_env.step()
        para evitar doble cálculo y drift.
        """
        # Solo mantener para compatibilidad, pero no hacer nada
        pass

    def _validate_portfolio_consistency(self, pos: PositionState, portfolio: PortfolioState) -> None:
        """Valida la consistencia del portfolio y corrige inconsistencias."""
        # 1. Sin posición: equity debe igualar cash (guard-rail estricto)
        if pos.side == 0 or pos.qty == 0.0:
            drift = abs(portfolio.equity_quote - portfolio.cash_quote)
            if drift > 1e-6:
                # ← NUEVO: Guard-rail estricto - corregir inmediatamente
                print(f"🔧 CORRIGIENDO DRIFT: Sin posición, equity={portfolio.equity_quote:.8f} → cash={portfolio.cash_quote:.8f} (drift: {drift:.8f})")
                portfolio.equity_quote = float(portfolio.cash_quote)
                # ← NUEVO: Resetear used_margin si no hay posición
                portfolio.used_margin = 0.0
        
        # 2. Validar que no hay valores NaN o Inf
        if not (portfolio.equity_quote == portfolio.equity_quote) or portfolio.equity_quote in (float('inf'), float('-inf')):
            print(f"⚠️ EQUITY INVÁLIDO: {portfolio.equity_quote}, corrigiendo a cash")
            portfolio.equity_quote = float(portfolio.cash_quote)
        
        if not (portfolio.cash_quote == portfolio.cash_quote) or portfolio.cash_quote in (float('inf'), float('-inf')):
            print(f"⚠️ CASH INVÁLIDO: {portfolio.cash_quote}, corrigiendo a 0")
            portfolio.cash_quote = 0.0
        
        # 3. Validar used_margin no negativo
        if portfolio.used_margin < 0:
            print(f"⚠️ USED_MARGIN NEGATIVO: {portfolio.used_margin}, corrigiendo a 0")
            portfolio.used_margin = 0.0
        
        # 4. Validar que used_margin es 0 sin posición
        if (pos.side == 0 or pos.qty == 0.0) and portfolio.used_margin > 1e-6:
            # Solo mostrar warning si el margen es significativo (> 0.01 USDT)
            if portfolio.used_margin > 0.01:
                print(f"⚠️ USED_MARGIN > 0 SIN POSICIÓN: {portfolio.used_margin}, liberando")
            portfolio.used_margin = 0.0
        
        # 5. Validar equity_base en spot
        if portfolio.market == "spot":
            if not (portfolio.equity_base == portfolio.equity_base) or portfolio.equity_base in (float('inf'), float('-inf')):
                print(f"⚠️ EQUITY_BASE INVÁLIDO: {portfolio.equity_base}, corrigiendo a 0")
                portfolio.equity_base = 0.0

    def is_end_of_data(self, broker) -> bool:
        return False
