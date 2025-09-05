"""
Utilidades para migración opcional a Decimal end-to-end.
Solo se activa si se detectan errores monetarios relevantes.
"""

from decimal import Decimal, ROUND_HALF_UP
from typing import Union, Optional, Dict, Any
import json

# Tipo para compatibilidad
Number = Union[float, Decimal]

class DecimalEncoder(json.JSONEncoder):
    """JSON encoder personalizado para Decimal"""
    def default(self, obj: Any) -> Any:
        if isinstance(obj, Decimal):
            return float(obj)
        return super().default(obj)

def to_decimal(value: Number, precision: int = 8) -> Decimal:
    """Convierte Number a Decimal con precisión específica"""
    if isinstance(value, Decimal):
        return value.quantize(Decimal('0.' + '0' * precision), rounding=ROUND_HALF_UP)
    return Decimal(str(value)).quantize(Decimal('0.' + '0' * precision), rounding=ROUND_HALF_UP)

def to_float(value: Number) -> float:
    """Convierte Number a float de forma segura"""
    if isinstance(value, Decimal):
        return float(value)
    return float(value) if value is not None else 0.0

def safe_decimal_operation(func, *args, **kwargs) -> Number:
    """Wrapper para operaciones que pueden fallar con Decimal"""
    try:
        return func(*args, **kwargs)
    except (ArithmeticError, ValueError) as e:
        print(f"[DECIMAL-ERROR] Operación falló: {e}, convirtiendo a float")
        # Convertir args a float y reintentar
        float_args = [to_float(arg) if isinstance(arg, (Decimal, float)) else arg for arg in args]
        float_kwargs = {k: to_float(v) if isinstance(v, (Decimal, float)) else v for k, v in kwargs.items()}
        return func(*float_args, **float_kwargs)

def decimal_round(value: Number, precision: int = 8) -> Decimal:
    """Redondea Decimal con precisión específica"""
    if isinstance(value, Decimal):
        return value.quantize(Decimal('0.' + '0' * precision), rounding=ROUND_HALF_UP)
    return to_decimal(value, precision)

def decimal_compare(a: Number, b: Number, tolerance: float = 1e-8) -> bool:
    """Compara dos números con tolerancia para Decimal"""
    if isinstance(a, Decimal) or isinstance(b, Decimal):
        a_dec = to_decimal(a)
        b_dec = to_decimal(b)
        return abs(a_dec - b_dec) < Decimal(str(tolerance))
    return abs(float(a) - float(b)) < tolerance

# Configuración para activar/desactivar Decimal
DECIMAL_MODE = False  # Cambiar a True solo si se detectan errores monetarios

def get_number_type() -> type:
    """Retorna el tipo de número a usar según configuración"""
    return Decimal if DECIMAL_MODE else float

def create_number(value: Union[str, int, float, Decimal]) -> Number:
    """Crea un número del tipo configurado"""
    if DECIMAL_MODE:
        return to_decimal(value)
    return float(value)
