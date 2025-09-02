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
) -> float:
    """Devuelve qty en base asset usando riesgo fijo / distancia a SL, con límite de capital."""
    if entry <= 0 or sl is None or sl <= 0:
        return 0.0
    
    # ← NUEVO: Bloqueo por equity - si el equity es 0 o negativo, no se pueden abrir posiciones
    # Esto puede causar que el agente se quede sin opciones si el equity se agota
    if equity_quote <= 0:
        return 0.0
    
    # Calcular qty basado en riesgo
    risk_amount = (risk_pct_per_trade / 100.0) * equity_quote  # pct a monto
    dist = abs(entry - sl)
    if dist <= 0:
        return 0.0
    qty_by_risk = risk_amount / dist
    
    # ← NUEVO: Límite superior basado en capital disponible
    # No podemos comprar más BTC del que podemos pagar
    max_qty_by_capital = equity_quote / entry
    
    # Usar el menor de los dos límites
    qty = min(qty_by_risk, max_qty_by_capital)
    
    # Redondeo y validaciones
    qty = round_lot(qty, lot_step)
    if entry * qty < min_notional:
        return 0.0
    
    return max(0.0, qty)
