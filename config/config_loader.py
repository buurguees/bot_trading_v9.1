# config/config_loader.py
"""
Loader de configuración para modo autonomía total.

Ensambla EnvConfig desde YAMLs de config/ y detecta duplicados críticos.
Proporciona una fuente única de verdad para toda la configuración del entorno.
"""

from __future__ import annotations
from pathlib import Path
from typing import Dict, Any, List, Optional, Union
import yaml
import logging

logger = logging.getLogger(__name__)

def _read_yaml(p: Path) -> Dict[str, Any]:
    """Lee un archivo YAML de forma segura."""
    if not p.exists():
        return {}
    try:
        with open(p, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    except Exception as e:
        logger.warning(f"Error leyendo {p}: {e}")
        return {}

def load_env_config(config_dir: Union[str, Path]) -> 'SimpleNamespaceDict':
    """
    Ensambla EnvConfig desde YAMLs de config/.
    
    Args:
        config_dir: Directorio de configuración
        
    Returns:
        Configuración unificada del entorno
    """
    config_dir = Path(config_dir)
    if not config_dir.exists():
        raise FileNotFoundError(f"Directorio de configuración no existe: {config_dir}")
    
    # Cargar todos los YAMLs
    settings = _read_yaml(config_dir / "settings.yaml")
    symbols = _read_yaml(config_dir / "symbols.yaml")
    pipeline = _read_yaml(config_dir / "pipeline.yaml")
    hierarchical = _read_yaml(config_dir / "hierarchical.yaml")
    risk = _read_yaml(config_dir / "risk.yaml")
    train = _read_yaml(config_dir / "train.yaml")
    oms = _read_yaml(config_dir / "oms.yaml")
    
    # Cargar rewards (prioridad: rewards_optimized.yaml > rewards.yaml)
    rewards = _read_yaml(config_dir / "rewards_optimized.yaml")
    if not rewards:
        rewards = _read_yaml(config_dir / "rewards.yaml")
    
    # === Selección de símbolo activo ===
    symbol_meta = _extract_symbol_meta(symbols)
    
    # === TFS desde train.yaml (prioridad) o settings/hard default ===
    tfs = _extract_timeframes(train, settings)
    base_tf = _extract_base_timeframe(train, tfs)
    
    # === Configuración de riesgo ===
    risk_cfg = _extract_risk_config(risk)
    
    # === Configuración de fees ===
    fees_cfg = _extract_fees_config(settings, train)
    
    # === Configuración de entrenamiento ===
    train_cfg = _extract_train_config(train, settings)
    
    # === Construir configuración unificada ===
    cfg = {
        "symbol_meta": symbol_meta,
        "market": "futures" if "future" in symbol_meta["mode"] else "spot",
        "tfs": tfs,
        "base_tf": base_tf,
        "pipeline": pipeline,
        "hierarchical": hierarchical,
        "risk": risk_cfg,
        "fees": fees_cfg,
        "rewards": rewards,
        "leverage": symbol_meta.get("default_leverage", 3),
        "verbosity": train_cfg.get("verbosity", "low"),
        "startup_cooldown_steps": train_cfg.get("startup_cooldown_steps", 0),
        "initial_balance": train_cfg.get("initial_balance", 1000.0),
        "target_balance": train_cfg.get("target_balance", 1_000_000.0),
        "milestones_verbose": train_cfg.get("milestones_verbose", False),
        "seed": train_cfg.get("seed", 42),
        "log_dir": train_cfg.get("log_dir", "logs"),
        "models_dir": settings.get("models_dir", "models"),
        "results_dir": settings.get("results_dir", "results"),
        "timezone": settings.get("timezone", "UTC"),
        "exchange": settings.get("exchange", "binance"),
        "oms_config": oms
    }
    
    return SimpleNamespaceDict(cfg)

def _extract_symbol_meta(symbols: Dict[str, Any]) -> Dict[str, Any]:
    """Extrae metadatos del símbolo activo."""
    syms = symbols.get("symbols", [])
    if not syms:
        return {
            "symbol": "BTCUSDT",
            "mode": "train_futures",
            "filters": {},
            "leverage": {"min": 1, "max": 20, "step": 1},
            "allow_shorts": True,
            "default_leverage": 3
        }
    
    # Buscar símbolo con mode activo
    sym_entry = None
    for s in syms:
        if s.get("mode"):
            sym_entry = s
            break
    
    if sym_entry is None:
        sym_entry = syms[0]
    
    return {
        "symbol": sym_entry.get("symbol", "BTCUSDT"),
        "mode": sym_entry.get("mode", "train_futures"),
        "filters": sym_entry.get("filters", {}),
        "leverage": sym_entry.get("leverage", {"min": 1, "max": 20, "step": 1}),
        "allow_shorts": sym_entry.get("allow_shorts", True),
        "default_leverage": sym_entry.get("default_leverage", 3)
    }

def _extract_timeframes(train: Dict[str, Any], settings: Dict[str, Any]) -> List[str]:
    """Extrae timeframes de configuración."""
    # Prioridad: train.yaml > settings.yaml > default
    tfs = (train.get("data") or {}).get("tfs")
    if not tfs:
        tfs = settings.get("tfs")
    if not tfs:
        tfs = ["1m", "5m", "15m", "1h"]
    
    return tfs

def _extract_base_timeframe(train: Dict[str, Any], tfs: List[str]) -> str:
    """Extrae timeframe base."""
    base_tf = (train.get("data") or {}).get("base_tf")
    if not base_tf:
        base_tf = tfs[0]
    
    return base_tf

def _extract_risk_config(risk: Dict[str, Any]) -> Dict[str, Any]:
    """Extrae configuración de riesgo."""
    if not risk:
        return {"common": {}, "spot": {}, "futures": {}}
    
    # Si hay perfiles activos, usar el activo
    if "active_profile" in risk:
        active = risk["active_profile"]
        risk_section = risk.get(active, {})
        return {
            "common": risk_section.get("common", {}),
            "spot": risk_section.get("spot", {}),
            "futures": risk_section.get("futures", {})
        }
    
    # Estructura simple
    return {
        "common": risk.get("common", {}),
        "spot": risk.get("spot", {}),
        "futures": risk.get("futures", {})
    }

def _extract_fees_config(settings: Dict[str, Any], train: Dict[str, Any]) -> Dict[str, float]:
    """Extrae configuración de fees."""
    # Prioridad: train.yaml > settings.yaml > default
    fees = (train.get("env") or {}).get("fees")
    if not fees:
        fees = settings.get("fees")
    if not fees:
        fees = {"maker": 0.0002, "taker": 0.0004}
    
    return fees

def _extract_train_config(train: Dict[str, Any], settings: Dict[str, Any]) -> Dict[str, Any]:
    """Extrae configuración de entrenamiento."""
    train_cfg = train.get("env", {})
    
    # Valores por defecto desde settings si no están en train
    defaults = {
        "verbosity": settings.get("verbosity", "low"),
        "startup_cooldown_steps": 0,
        "initial_balance": 1000.0,
        "target_balance": 1_000_000.0,
        "milestones_verbose": False,
        "seed": train.get("seed", settings.get("seed", 42)),
        "log_dir": train.get("log_dir", "logs")
    }
    
    # Merge con defaults
    for key, default_value in defaults.items():
        if key not in train_cfg:
            train_cfg[key] = default_value
    
    return train_cfg

def validate_config_consistency(config_dir: Union[str, Path]) -> List[str]:
    """
    Valida consistencia de configuración y detecta duplicados críticos.
    
    Args:
        config_dir: Directorio de configuración
        
    Returns:
        Lista de problemas detectados
    """
    config_dir = Path(config_dir)
    problems: List[str] = []
    
    # Cargar archivos para validación
    settings = _read_yaml(config_dir / "settings.yaml")
    train = _read_yaml(config_dir / "train.yaml")
    oms = _read_yaml(config_dir / "oms.yaml")
    
    # Verificar existencia de archivos críticos
    critical_files = ["settings.yaml", "symbols.yaml", "pipeline.yaml", "hierarchical.yaml", "risk.yaml"]
    for file in critical_files:
        if not (config_dir / file).exists():
            problems.append(f"Archivo crítico faltante: {file}")
    
    # Verificar duplicados de semilla
    if "seed" in settings and "seed" in train:
        problems.append("Semilla duplicada en settings.yaml y train.yaml. Deja SOLO en train.yaml.")
    
    if oms and oms.get("seed") is not None:
        problems.append("Semilla en oms.yaml: mejor centralizar en train.yaml y eliminar de oms.yaml.")
    
    # Verificar duplicados de rutas
    if "log_dir" in train and "results_dir" in settings:
        problems.append("train.log_dir y settings.results_dir podrían solaparse. Define qué usa el launcher.")
    
    # Verificar rewards duplicados
    rewards_yaml = (config_dir / "rewards.yaml").exists()
    rewards_optimized_yaml = (config_dir / "rewards_optimized.yaml").exists()
    
    if rewards_yaml and rewards_optimized_yaml:
        problems.append("Existen rewards.yaml y rewards_optimized.yaml. Elige UNO y referencia solo ese en train.yaml.")
    
    # Verificar risk comentado
    if (config_dir / "risk_commented.yaml").exists():
        problems.append("Existe risk_commented.yaml (documentación). Asegúrate de NO cargarlo en runtime.")
    
    # Verificar configuración de OMS
    if oms:
        oms_duplicates = []
        if "seed" in oms:
            oms_duplicates.append("seed")
        if "log_dir" in oms:
            oms_duplicates.append("log_dir")
        if "data" in oms:
            oms_duplicates.append("data")
        
        if oms_duplicates:
            problems.append(f"oms.yaml contiene {', '.join(oms_duplicates)}. Mueve a train.yaml y deja oms.yaml solo para configuración del adapter.")
    
    return problems

def get_config_summary(config_dir: Union[str, Path]) -> Dict[str, Any]:
    """
    Obtiene resumen de configuración cargada.
    
    Args:
        config_dir: Directorio de configuración
        
    Returns:
        Resumen de configuración
    """
    try:
        cfg = load_env_config(config_dir)
        problems = validate_config_consistency(config_dir)
        
        return {
            "symbol": cfg.symbol_meta.symbol,
            "mode": cfg.symbol_meta.mode,
            "market": cfg.market,
            "tfs": cfg.tfs,
            "base_tf": cfg.base_tf,
            "leverage": cfg.leverage,
            "initial_balance": cfg.initial_balance,
            "target_balance": cfg.target_balance,
            "seed": cfg.seed,
            "problems_count": len(problems),
            "problems": problems
        }
    except Exception as e:
        return {"error": str(e)}

class SimpleNamespaceDict:
    """
    Fallback para tratar la configuración como objeto con notación de punto.
    
    Si ya tienes modelos Pydantic/dataclasses, reemplaza esto con EnvConfig(**cfg).
    """
    
    def __init__(self, d: Dict[str, Any]):
        for k, v in d.items():
            if isinstance(v, dict):
                v = SimpleNamespaceDict(v)
            self.__dict__[k] = v
    
    def __getattr__(self, item):
        return self.__dict__.get(item)
    
    def __repr__(self):
        return f"SimpleNamespaceDict({self.__dict__})"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convierte a diccionario."""
        result = {}
        for k, v in self.__dict__.items():
            if isinstance(v, SimpleNamespaceDict):
                result[k] = v.to_dict()
            else:
                result[k] = v
        return result
    
    def get(self, key: str, default: Any = None) -> Any:
        """Obtiene valor con default."""
        return self.__dict__.get(key, default)
    
    def hasattr(self, key: str) -> bool:
        """Verifica si tiene atributo."""
        return key in self.__dict__

# Función de conveniencia para uso directo
def load_config(config_dir: Union[str, Path]) -> SimpleNamespaceDict:
    """Función de conveniencia para cargar configuración."""
    return load_env_config(config_dir)

def validate_config(config_dir: Union[str, Path]) -> List[str]:
    """Función de conveniencia para validar configuración."""
    return validate_config_consistency(config_dir)
