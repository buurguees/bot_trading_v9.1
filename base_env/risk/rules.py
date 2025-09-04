# base_env/risk/rules.py
# Descripción: Fórmulas de sizing y validaciones para spot.

from __future__ import annotations
from typing import Tuple
import math


def round_lot(qty: float, lot_step: float) -> float:
    if lot_step <= 0:
        return qty
    steps = math.floor(qty / lot_step)
    return steps * lot_step


def size_spot(
    equity_quote: float,
    risk_pct_per_trade: float,
    entry: float,
    sl: float | None,
    min_notional: float,
    lot_step: float,
    events_bus=None,
    ts_now=None,
) -> float:
    """Devuelve qty en base asset usando riesgo fijo / distancia a SL, con límite de capital."""
    if entry <= 0:
        return 0.0
    
    # ← NUEVO: Bloqueo por equity - si el equity es muy bajo, usar tamaño mínimo
    if equity_quote <= 0:
        if events_bus and ts_now:
            events_bus.emit("LOW_EQUITY", ts=ts_now, 
                           reason="equity_zero_or_negative", 
                           equity=equity_quote)
        return 0.0
    
    # ← NUEVO: Si el equity es muy bajo, usar tamaño mínimo para permitir trades
    if equity_quote < 100:  # Menos de 100 USDT
        min_qty = min_notional / entry
        qty = round_lot(min_qty, lot_step)
        return max(0.0, qty)
    
    # ← NUEVO: Si no hay SL, usar tamaño mínimo basado en 1% del equity
    if sl is None or sl <= 0:
        # Fallback: usar 1% del equity como tamaño mínimo
        min_qty = (equity_quote * 0.01) / entry
        qty = round_lot(min_qty, lot_step)
        if entry * qty < min_notional:
            # Si aún es muy pequeño, usar el mínimo notional
            qty = round_lot(min_notional / entry, lot_step)
        return max(0.0, qty)
    
    # Calcular qty basado en riesgo
    risk_amount = (risk_pct_per_trade / 100.0) * equity_quote  # pct a monto
    dist = abs(entry - sl)
    
    # ← NUEVO: Si el SL está muy cerca, usar tamaño mínimo en lugar de 0
    if dist <= 0 or dist < entry * 0.001:  # SL muy cerca (menos de 0.1%)
        if events_bus and ts_now:
            events_bus.emit("NO_SL_DISTANCE", ts=ts_now, 
                           reason="sl_too_close", 
                           entry=entry,
                           sl=sl,
                           distance=dist)
        # Fallback: usar 1% del equity como tamaño mínimo
        min_qty = (equity_quote * 0.01) / entry
        qty = round_lot(min_qty, lot_step)
        if entry * qty < min_notional:
            # Si aún es muy pequeño, usar el mínimo notional
            qty = round_lot(min_notional / entry, lot_step)
        return max(0.0, qty)
    
    qty_by_risk = risk_amount / dist
    
    # ← NUEVO: Límite superior basado en capital disponible
    # No podemos comprar más BTC del que podemos pagar
    max_qty_by_capital = equity_quote / entry
    
    # Usar el menor de los dos límites
    qty = min(qty_by_risk, max_qty_by_capital)
    
    # Redondeo y validaciones
    qty = round_lot(qty, lot_step)
    if entry * qty < min_notional:
        # ← NUEVO: Si es muy pequeño, usar el mínimo notional en lugar de 0
        original_qty = qty
        qty = round_lot(min_notional / entry, lot_step)
        if events_bus and ts_now:
            events_bus.emit("MIN_NOTIONAL_BLOCKED", ts=ts_now, 
                           reason="forced_min_notional", 
                           original_qty=original_qty,
                           final_qty=qty,
                           original_notional=entry * original_qty,
                           min_notional=min_notional)
    
    return max(0.0, qty)
