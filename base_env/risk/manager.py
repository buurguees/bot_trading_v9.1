# base_env/risk/manager.py
# Descripción: Sizing spot y mantenimiento de posición (SL/TP/TTL/Trailing ATR).
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Tuple
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

    def apply(self, portfolio, position, decision, obs, events_bus=None, ts_now=None) -> SizedDecision:
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
                # Prohibir shorts si el símbolo no lo permite
                allow_shorts = bool(getattr(self.symbol_meta, "allow_shorts", True))
                if decision.side < 0 and not allow_shorts:
                    if events_bus and ts_now:
                        events_bus.emit("SHORTS_DISABLED", ts=ts_now, symbol=self.symbol_meta.symbol)
                    return SizedDecision(False, 0, 0.0, decision.price_hint, decision.sl, decision.tp, False, False, 0.0)
                # Aplicar SL/TP por defecto si faltan
                if sl is None:
                    sl, _ = self._get_default_sl_tp(entry, decision.side, obs)
                
                qty = size_spot(
                    equity_quote=portfolio.equity_quote,
                    risk_pct_per_trade=self.cfg.spot.risk_pct_per_trade,
                    entry=entry,
                    sl=sl,
                    min_notional=float(self.symbol_meta.filters.get("minNotional", 5.0)),
                    lot_step=float(self.symbol_meta.filters.get("lotStep", 0.0001)),
                    events_bus=events_bus,
                    ts_now=ts_now,
                )
                
                # Si el sizing falló pero train_force_min_notional está habilitado, intentar con tamaño mínimo
                if qty <= 0 and self.cfg.common.train_force_min_notional:
                    min_notional = float(self.symbol_meta.filters.get("minNotional", 5.0))
                    qty = min_notional / entry
                    print(f"🔧 FORZANDO MIN_NOTIONAL: qty ajustado a {qty:.6f} para cumplir {min_notional} USDT")
                if qty <= 0:
                    # ← NUEVO: Emitir evento RISK_BLOCKED cuando se bloquea por riesgo
                    if events_bus and ts_now:
                        events_bus.emit("RISK_BLOCKED", ts=ts_now, 
                                       reason="spot_sizing_failed", 
                                       equity=portfolio.equity_quote,
                                       entry=decision.price_hint,
                                       sl=decision.sl,
                                       risk_pct=self.cfg.spot.risk_pct_per_trade)
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
                    account_equity=portfolio.equity_quote,
                    obs=obs,
                    events_bus=events_bus,
                    ts_now=ts_now
                )

        # Default
        return SizedDecision(False, 0, 0.0, decision.price_hint, decision.sl, decision.tp, False, False, 0.0)

    def size_futures(self, portfolio, decision, leverage: float, account_equity: float, obs=None, events_bus=None, ts_now=None):
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
        print(f"🔍 RISK MANAGER INPUT: decision.should_open={decision.should_open if decision else None}, decision.side={decision.side if decision else None}, decision.sl={decision.sl if decision else None}, decision.tp={decision.tp if decision else None}")
        if decision is None or not decision.should_open:
            print(f"🚫 RISK MANAGER BLOCKED: decision is None or should_open=False")
            return _Sized(should_open=False)

        price = float(decision.price_hint)
        # Usar correctamente el porcentaje desde la config de futuros (expresado en %)
        try:
            risk_pct_percent = float(self.cfg.futures.risk_pct_per_trade)
        except Exception:
            # Fallback seguro: 1% si no está disponible
            risk_pct_percent = 1.0
        # Convertir a fracción
        risk_frac = max(0.0, risk_pct_percent / 100.0)
        notional_cap = max(0.0, float(account_equity)) * max(1.0, float(leverage))
        risk_cash = max(0.0, float(account_equity)) * risk_frac

        # ← NUEVO: Validar que SL/TP sean válidos antes de proceder
        min_sl_pct = self.cfg.common.default_levels.min_sl_pct
        tp_r_multiple = getattr(self.cfg.common.default_levels, 'tp_r_multiple', 2.0)
        
        # Aplicar SL/TP por defecto si faltan (sin bloquear)
        if decision.sl is None or decision.sl <= 0:
            print(f"🔧 APLICANDO SL POR DEFECTO: SL era {decision.sl}")
            min_sl_dist_absolute = price * (min_sl_pct / 100.0)
            if decision.side > 0:  # Long
                decision.sl = price - min_sl_dist_absolute
            else:  # Short
                decision.sl = price + min_sl_dist_absolute
            print(f"🔧 SL APLICADO: {decision.sl:.4f}")
        
        # Aplicar TP por defecto si falta
        if decision.tp is None or decision.tp <= 0:
            print(f"🔧 APLICANDO TP POR DEFECTO: TP era {decision.tp}")
            sl_distance = abs(price - float(decision.sl))
            if decision.side > 0:  # Long
                decision.tp = price + (sl_distance * tp_r_multiple)
            else:  # Short
                decision.tp = price - (sl_distance * tp_r_multiple)
            print(f"🔧 TP APLICADO: {decision.tp:.4f}")
        
        # Calcular distancia de SL para sizing
        sl_distance = abs(price - float(decision.sl))
        stop_dist = sl_distance
        
        qty = risk_cash / max(stop_dist, 1e-9)
        # no exceder notional máximo
        if qty * price > notional_cap:
            qty = notional_cap / price

        # ← NUEVO: Aplicar lotStep y tickSize del símbolo (usar estructura correcta)
        lot_step = float(self.symbol_meta.filters.get("lotStep", 0.0001))
        tick_size = float(self.symbol_meta.filters.get("tickSize", 0.1))
        min_notional = float(self.symbol_meta.filters.get("minNotional", 5.0))
        
        # Redondear qty a lotStep (round down para ser conservador)
        qty = int(qty / lot_step) * lot_step
        
        # Redondear precio a tickSize
        price = round(price / tick_size) * tick_size
        
        # Recalcular notional con precios redondeados
        notional = qty * price
        
        # ← NUEVO: Forzar minNotional si está habilitado
        train_force_min_notional = getattr(self.cfg.common, 'train_force_min_notional', False)
        if train_force_min_notional and notional < min_notional and qty > 0:
            # Escalar qty para cumplir minNotional (round up)
            qty_min = min_notional / price
            qty_min = int(qty_min / lot_step) * lot_step + lot_step  # Round up
            notional_min = qty_min * price
            
            # Verificar que no exceda el límite de notional
            if notional_min <= notional_cap:
                qty = qty_min
                notional = notional_min
                if events_bus and ts_now:
                    events_bus.emit("FORZANDO_MIN_NOTIONAL", ts=ts_now,
                                   original_qty=qty, final_qty=qty, notional=notional,
                                   min_notional=min_notional)
            else:
                # Si excede el límite, usar el máximo posible
                qty = notional_cap / price
                qty = int(qty / lot_step) * lot_step  # Round down
                notional = qty * price
                
                # Si aún es muy pequeño después del cap, bloquear
                if notional < min_notional:
                    if events_bus and ts_now:
                        events_bus.emit("MIN_NOTIONAL_BLOCKED", ts=ts_now,
                                       reason="notional_cap_too_low",
                                       qty=qty, notional=notional, min_notional=min_notional,
                                       lot_step=lot_step, tick_size=tick_size,
                                       equity=account_equity, leverage=leverage)
                    print(f"🚫 SIZING_BLOCKED: reason=MIN_NOTIONAL_BLOCKED, qty={qty:.6f}, notional={notional:.2f}, "
                          f"minNotional={min_notional:.2f}, lotStep={lot_step:.6f}, tickSize={tick_size:.4f}, "
                          f"equity={account_equity:.2f}, leverage={leverage:.1f}")
                    return _Sized(should_open=False)

        # ← NUEVO: Si aún es 0, emitir evento RISK_BLOCKED con detalles
        if qty <= 0:
            if events_bus and ts_now:
                events_bus.emit("RISK_BLOCKED", ts=ts_now, 
                               reason="futures_sizing_failed", 
                               equity=account_equity,
                               leverage=leverage,
                               entry=price,
                               sl=decision.sl,
                               risk_pct=risk_pct_percent,
                               notional_cap=notional_cap,
                               lot_step=lot_step,
                               tick_size=tick_size,
                               min_notional=min_notional)
            print(f"🚫 SIZING_BLOCKED: reason=futures_sizing_failed, qty={qty:.6f}, notional={notional:.2f}, "
                  f"minNotional={min_notional:.2f}, lotStep={lot_step:.6f}, tickSize={tick_size:.4f}, "
                  f"equity={account_equity:.2f}, leverage={leverage:.1f}")
            return _Sized(should_open=False)

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
        # ← NUEVO: Verificar si la quiebra está habilitada
        bankruptcy_enabled = self.cfg.common.bankruptcy.enabled
        if not bankruptcy_enabled:
            return False  # ← NUEVO: No verificar quiebra si está desactivada
            
        threshold_pct = float(self.cfg.common.bankruptcy.threshold_pct) / 100.0
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

    def _get_default_sl_tp(self, price: float, side: int, obs: dict) -> Tuple[Optional[float], Optional[float]]:
        """Calcula SL/TP por defecto usando configuración YAML."""
        default_cfg = self.cfg.common.default_levels
        
        if default_cfg.use_atr:
            # Buscar ATR en las observaciones
            atr = None
            for tf_data in obs.get("features", {}).values():
                if "atr14" in tf_data:
                    atr = tf_data["atr14"]
                    break
            
            if atr is not None and atr > 0:
                # Usar ATR con múltiplos configurados
                sl_dist = atr * default_cfg.sl_atr_mult
                tp_dist = sl_dist * default_cfg.tp_r_multiple
                
                if side > 0:  # Long
                    sl = price - sl_dist
                    tp = price + tp_dist
                else:  # Short
                    sl = price + sl_dist
                    tp = price - tp_dist
                
                return sl, tp
        
        # Fallback a porcentajes
        sl_pct = default_cfg.min_sl_pct / 100.0
        tp_pct = sl_pct * default_cfg.tp_r_multiple
        
        if side > 0:  # Long
            sl = price * (1 - sl_pct)
            tp = price * (1 + tp_pct)
        else:  # Short
            sl = price * (1 + sl_pct)
            tp = price * (1 - tp_pct)
        
        return sl, tp
