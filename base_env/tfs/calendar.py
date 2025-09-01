# base_env/tfs/calendar.py
# Descripción: Utilidades de calendario y TF: conversión TF→ms, alineación de ts a TF, cierre anterior, y verificación de cierre.
# Ubicación: base_env/tfs/calendar.py
#
# Convenciones:
# - Timestamps en milisegundos UTC (int)
# - TF soportados: 1m, 5m, 15m, 1h, 4h, 1d

from __future__ import annotations
from typing import Literal

TF = Literal["1m", "5m", "15m", "1h", "4h", "1d"]


def tf_to_ms(tf: TF) -> int:
    """Devuelve el tamaño de barra en milisegundos para el TF dado."""
    if tf == "1m":
        return 60_000
    if tf == "5m":
        return 5 * 60_000
    if tf == "15m":
        return 15 * 60_000
    if tf == "1h":
        return 60 * 60_000
    if tf == "4h":
        return 4 * 60 * 60_000
    if tf == "1d":
        return 24 * 60 * 60_000
    raise ValueError(f"TF no soportado: {tf}")


def floor_ts_to_tf(ts_ms: int, tf: TF) -> int:
    """
    Redondea un timestamp hacia abajo al inicio de la barra del TF.
    Ejemplo: ts de 10:03:20 en 5m → 10:00:00
    """
    size = tf_to_ms(tf)
    return (ts_ms // size) * size


def prev_closed_bar_end(ts_ms: int, tf: TF) -> int:
    """
    Devuelve el timestamp de CIERRE de la ÚLTIMA barra completamente cerrada ANTES (o igual) del ts dado.
    Por convención: cierre = inicio + size.
    - Si ts ya cae EN el límite de cierre, devuelve ese cierre.
    - Si no, devuelve el cierre de la barra previa.
    """
    size = tf_to_ms(tf)
    # Cierre de la barra actual (si ts exacto al cierre) o la siguiente frontera
    bar_start = floor_ts_to_tf(ts_ms, tf)
    bar_close = bar_start + size
    # Si ya estamos EXACTAMENTE en el cierre, ese es el último cierre
    if ts_ms == bar_close:
        return bar_close
    # Si aún no hemos llegado al cierre actual, el último cierre fue el anterior
    # (bar_start es el inicio "abierto"; el cierre anterior fue bar_start)
    return bar_start


def is_close_boundary(ts_ms: int, tf: TF) -> bool:
    """Indica si el timestamp coincide exactamente con un cierre de barra de ese TF."""
    size = tf_to_ms(tf)
    return (ts_ms % size) == 0
