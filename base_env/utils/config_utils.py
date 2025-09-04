# bot_trading_v9_1/core/utils/config_utils.py
from dataclasses import asdict, is_dataclass
from typing import Any

def to_mapping(obj: Any) -> dict:
    """
    Devuelve un dict a partir de obj si es dataclass o Pydantic; si ya es dict, lo retorna tal cual.
    No falla si es None.
    """
    if obj is None:
        return {}
    if isinstance(obj, dict):
        return obj
    # Pydantic v2
    if hasattr(obj, "model_dump") and callable(obj.model_dump):
        return obj.model_dump()
    # Pydantic v1
    if hasattr(obj, "dict") and callable(obj.dict):
        return obj.dict()
    # Dataclass
    if is_dataclass(obj):
        return asdict(obj)
    # Fallback: leer atributos públicos
    return {k: getattr(obj, k) for k in dir(obj) if not k.startswith("_") and not callable(getattr(obj, k))}

def cfg_get(cfg: Any, key: str, default=None):
    """
    Accede de forma segura a una clave/atributo de cfg sea dict, dataclass o pydantic.
    """
    if cfg is None:
        return default
    if isinstance(cfg, dict):
        return cfg.get(key, default)
    if hasattr(cfg, key):
        return getattr(cfg, key, default)
    # último recurso: convertir
    return to_mapping(cfg).get(key, default)
