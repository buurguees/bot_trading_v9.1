# base_env/features/indicators.py
# Descripción: Indicadores técnicos básicos (EMA, RSI, ATR, MACD, Bollinger).
# Implementación simple y auto-contenida para usar en ventanas pequeñas.
# Ubicación: base_env/features/indicators.py
#
# NOTA: Pensado para calcular sobre una ventana ya recortada. Si llega una lista
# corta, devuelve lo que pueda sin explotar; valida tamaños mínimo.

from __future__ import annotations
from typing import List, Dict
import math


def _ema_series(values: List[float], period: int) -> List[float]:
    if period <= 0 or len(values) == 0:
        return []
    k = 2 / (period + 1)
    out: List[float] = []
    ema = None
    for v in values:
        if ema is None:
            ema = v
        else:
            ema = v * k + ema * (1 - k)
        out.append(float(ema))
    return out


def ema(values: List[float], period: int) -> float | None:
    if len(values) == 0:
        return None
    series = _ema_series(values, period)
    return series[-1] if series else None


def rsi(closes: List[float], period: int = 14) -> float | None:
    n = len(closes)
    if n < period + 1:
        return None
    gains = []
    losses = []
    for i in range(1, n):
        delta = closes[i] - closes[i - 1]
        gains.append(max(delta, 0.0))
        losses.append(max(-delta, 0.0))
    # medias móviles simples para primera semilla
    avg_gain = sum(gains[:period]) / period
    avg_loss = sum(losses[:period]) / period
    # suavizado Wilder
    for i in range(period, n - 1):
        avg_gain = (avg_gain * (period - 1) + gains[i]) / period
        avg_loss = (avg_loss * (period - 1) + losses[i]) / period
    if avg_loss == 0:
        return 100.0
    rs = avg_gain / avg_loss
    return 100.0 - (100.0 / (1.0 + rs))


def atr(highs: List[float], lows: List[float], closes: List[float], period: int = 14) -> float | None:
    n = len(closes)
    if n < period + 1 or len(highs) != n or len(lows) != n:
        return None
    trs: List[float] = []
    for i in range(1, n):
        hi = highs[i]
        lo = lows[i]
        prev_close = closes[i - 1]
        tr = max(hi - lo, abs(hi - prev_close), abs(lo - prev_close))
        trs.append(tr)
    # media móvil simple de TR para simplificar
    if len(trs) < period:
        return None
    return sum(trs[-period:]) / period


def macd(closes: List[float], fast: int = 12, slow: int = 26, signal: int = 9) -> Dict[str, float] | None:
    if len(closes) < slow + signal:
        return None
    ema_fast = _ema_series(closes, fast)
    ema_slow = _ema_series(closes, slow)
    # alinear por la cola
    macd_line = [a - b for a, b in zip(ema_fast[-len(ema_slow):], ema_slow)]
    signal_line = _ema_series(macd_line, signal)
    if not signal_line:
        return None
    hist = macd_line[-1] - signal_line[-1]
    return {"macd": macd_line[-1], "signal": signal_line[-1], "hist": hist}


def bollinger(closes: List[float], period: int = 20, dev: float = 2.0) -> Dict[str, float] | None:
    if len(closes) < period:
        return None
    window = closes[-period:]
    m = sum(window) / period
    variance = sum((x - m) ** 2 for x in window) / period
    std = math.sqrt(variance)
    upper = m + dev * std
    lower = m - dev * std
    width = (upper - lower) / m if m != 0 else 0.0
    pctB = (closes[-1] - lower) / (upper - lower) if (upper - lower) != 0 else 0.0
    return {"middle": m, "upper": upper, "lower": lower, "width": width, "pctB": pctB}
