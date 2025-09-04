# base_env/utils/timestamp_utils.py
"""
Utilidades para manejo de timestamps y conversión a formatos legibles.
"""

from __future__ import annotations
from datetime import datetime, timezone
from typing import Optional, Union


def timestamp_to_utc_string(timestamp: Union[int, float, None]) -> Optional[str]:
    """
    Convierte un timestamp (en ms o s) a string UTC legible.
    
    Args:
        timestamp: Timestamp en milisegundos o segundos (Unix timestamp)
        
    Returns:
        String en formato UTC o None si timestamp es None/inválido
    """
    if timestamp is None:
        return None
    
    try:
        # Convertir a segundos si está en milisegundos
        if timestamp > 1e10:  # Probablemente en milisegundos
            timestamp = timestamp / 1000.0
        
        # Crear datetime object en UTC
        dt = datetime.fromtimestamp(timestamp, tz=timezone.utc)
        
        # Formato: YYYY-MM-DD HH:MM:SS UTC
        return dt.strftime("%Y-%m-%d %H:%M:%S UTC")
        
    except (ValueError, OSError, OverflowError):
        return None


def timestamp_to_utc_iso(timestamp: Union[int, float, None]) -> Optional[str]:
    """
    Convierte un timestamp a formato ISO 8601 UTC.
    
    Args:
        timestamp: Timestamp en milisegundos o segundos
        
    Returns:
        String en formato ISO 8601 UTC o None si timestamp es None/inválido
    """
    if timestamp is None:
        return None
    
    try:
        # Convertir a segundos si está en milisegundos
        if timestamp > 1e10:  # Probablemente en milisegundos
            timestamp = timestamp / 1000.0
        
        # Crear datetime object en UTC
        dt = datetime.fromtimestamp(timestamp, tz=timezone.utc)
        
        # Formato ISO 8601: YYYY-MM-DDTHH:MM:SS.fffZ
        return dt.isoformat().replace('+00:00', 'Z')
        
    except (ValueError, OSError, OverflowError):
        return None


def add_utc_timestamps(data: dict) -> dict:
    """
    Añade campos de timestamp UTC legibles a un diccionario.
    
    Args:
        data: Diccionario que puede contener campos de timestamp
        
    Returns:
        Diccionario con campos UTC añadidos
    """
    result = data.copy()
    
    # Campos de timestamp comunes a convertir
    timestamp_fields = [
        'ts', 'timestamp', 'ts_start', 'ts_end', 
        'open_ts', 'close_ts', 'created_at', 'updated_at',
        'first_trade_ts', 'last_trade_ts'
    ]
    
    for field in timestamp_fields:
        if field in result and result[field] is not None:
            # Añadir campo UTC correspondiente
            utc_field = f"{field}_utc"
            result[utc_field] = timestamp_to_utc_string(result[field])
            
            # También añadir versión ISO si es un timestamp principal
            if field in ['ts', 'timestamp', 'ts_start', 'ts_end', 'first_trade_ts', 'last_trade_ts']:
                iso_field = f"{field}_iso"
                result[iso_field] = timestamp_to_utc_iso(result[field])
    
    return result


def get_current_utc_timestamp() -> tuple[int, str]:
    """
    Obtiene el timestamp actual en ms y su representación UTC.
    
    Returns:
        Tupla (timestamp_ms, utc_string)
    """
    now = datetime.now(timezone.utc)
    timestamp_ms = int(now.timestamp() * 1000)
    utc_string = now.strftime("%Y-%m-%d %H:%M:%S UTC")
    
    return timestamp_ms, utc_string
