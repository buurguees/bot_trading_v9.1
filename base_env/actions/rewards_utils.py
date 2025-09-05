# base_env/actions/rewards_utils.py
"""
Utilidades comunes para el sistema de rewards optimizado.

Este archivo ahora importa las utilidades centralizadas desde config_utils.py
para evitar duplicación de código y mantener consistencia.
"""

from __future__ import annotations
import logging
from typing import Dict, Any, Optional, Union, List, Callable
from functools import wraps
import numpy as np

# Importar utilidades centralizadas
from ..config.config_utils import (
    get_config_cache, get_profiler, get_batch_processor,
    get_vectorized_calculator, get_reward_validator,
    profile_reward_calculation, load_config_with_cache,
    get_config_value, validate_config_consistency,
    merge_configs, save_config_to_yaml, load_config_from_yaml,
    get_all_utils
)

logger = logging.getLogger(__name__)

# Re-exportar las utilidades centralizadas para compatibilidad
ConfigCache = get_config_cache().__class__
PerformanceProfiler = get_profiler().__class__
BatchProcessor = get_batch_processor().__class__
VectorizedCalculator = get_vectorized_calculator().__class__
RewardValidator = get_reward_validator().__class__

# Funciones de conveniencia que mantienen la API existente
def get_config_cache() -> ConfigCache:
    """Obtiene el cache global de configuraciones."""
    from ..config.config_utils import get_config_cache as _get_config_cache
    return _get_config_cache()

def get_profiler() -> PerformanceProfiler:
    """Obtiene el profiler global."""
    from ..config.config_utils import get_profiler as _get_profiler
    return _get_profiler()

def get_batch_processor() -> BatchProcessor:
    """Obtiene el procesador de lotes global."""
    from ..config.config_utils import get_batch_processor as _get_batch_processor
    return _get_batch_processor()

def get_vectorized_calculator() -> VectorizedCalculator:
    """Obtiene la calculadora vectorizada global."""
    from ..config.config_utils import get_vectorized_calculator as _get_vectorized_calculator
    return _get_vectorized_calculator()

def get_reward_validator() -> RewardValidator:
    """Obtiene el validador de rewards global."""
    from ..config.config_utils import get_reward_validator as _get_reward_validator
    return _get_reward_validator()

def profile_reward_calculation(operation_name: str):
    """
    Decorator para perfilar cálculos de rewards.
    
    Args:
        operation_name: Nombre de la operación a perfilar
    """
    from ..config.config_utils import profile_reward_calculation as _profile_reward_calculation
    return _profile_reward_calculation(operation_name)

def load_config_with_cache(config: Dict[str, Any], module_name: str) -> Dict[str, Any]:
    """
    Carga configuración con cache inteligente.
    
    Args:
        config: Configuración a cargar
        module_name: Nombre del módulo
        
    Returns:
        Configuración cargada y validada
    """
    from ..config.config_utils import load_config_with_cache as _load_config_with_cache
    return _load_config_with_cache(config, module_name)

def get_config_value(config: Dict[str, Any], key: str, default: Any = None, 
                    module_name: str = "unknown") -> Any:
    """
    Obtiene valor de configuración con validación.
    
    Args:
        config: Configuración del módulo
        key: Clave a obtener
        default: Valor por defecto
        module_name: Nombre del módulo para logging
        
    Returns:
        Valor de configuración o default
    """
    from ..config.config_utils import get_config_value as _get_config_value
    return _get_config_value(config, key, default, module_name)

def validate_config_consistency(config: Dict[str, Any], 
                              required_keys: List[str] = None) -> List[str]:
    """
    Valida consistencia de configuración.
    
    Args:
        config: Configuración a validar
        required_keys: Claves requeridas (opcional)
        
    Returns:
        Lista de problemas encontrados
    """
    from ..config.config_utils import validate_config_consistency as _validate_config_consistency
    return _validate_config_consistency(config, required_keys)

def merge_configs(base_config: Dict[str, Any], 
                 override_config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Fusiona configuraciones con override inteligente.
    
    Args:
        base_config: Configuración base
        override_config: Configuración de override
        
    Returns:
        Configuración fusionada
    """
    from ..config.config_utils import merge_configs as _merge_configs
    return _merge_configs(base_config, override_config)

def save_config_to_yaml(config: Dict[str, Any], file_path: Union[str, Path]) -> None:
    """
    Guarda configuración en archivo YAML.
    
    Args:
        config: Configuración a guardar
        file_path: Ruta del archivo
    """
    from ..config.config_utils import save_config_to_yaml as _save_config_to_yaml
    return _save_config_to_yaml(config, file_path)

def load_config_from_yaml(file_path: Union[str, Path]) -> Dict[str, Any]:
    """
    Carga configuración desde archivo YAML.
    
    Args:
        file_path: Ruta del archivo
        
    Returns:
        Configuración cargada
    """
    from ..config.config_utils import load_config_from_yaml as _load_config_from_yaml
    return _load_config_from_yaml(file_path)

def get_all_utils() -> Dict[str, Any]:
    """Obtiene todas las utilidades de configuración."""
    from ..config.config_utils import get_all_utils as _get_all_utils
    return _get_all_utils()

# Funciones específicas de rewards que no están en config_utils
def calculate_reward_statistics(rewards: List[float]) -> Dict[str, float]:
    """
    Calcula estadísticas de una lista de rewards.
    
    Args:
        rewards: Lista de rewards
        
    Returns:
        Diccionario con estadísticas
    """
    if not rewards:
        return {
            "mean": 0.0,
            "std": 0.0,
            "min": 0.0,
            "max": 0.0,
            "median": 0.0,
            "q25": 0.0,
            "q75": 0.0
        }
    
    rewards_array = np.array(rewards)
    return {
        "mean": float(np.mean(rewards_array)),
        "std": float(np.std(rewards_array)),
        "min": float(np.min(rewards_array)),
        "max": float(np.max(rewards_array)),
        "median": float(np.median(rewards_array)),
        "q25": float(np.percentile(rewards_array, 25)),
        "q75": float(np.percentile(rewards_array, 75))
    }

def normalize_rewards(rewards: List[float], method: str = "z_score") -> List[float]:
    """
    Normaliza una lista de rewards.
    
    Args:
        rewards: Lista de rewards a normalizar
        method: Método de normalización ("z_score", "min_max", "robust")
        
    Returns:
        Lista de rewards normalizados
    """
    if not rewards:
        return []
    
    rewards_array = np.array(rewards)
    
    if method == "z_score":
        mean = np.mean(rewards_array)
        std = np.std(rewards_array)
        if std == 0:
            return [0.0] * len(rewards)
        return ((rewards_array - mean) / std).tolist()
    
    elif method == "min_max":
        min_val = np.min(rewards_array)
        max_val = np.max(rewards_array)
        if max_val == min_val:
            return [0.0] * len(rewards)
        return ((rewards_array - min_val) / (max_val - min_val)).tolist()
    
    elif method == "robust":
        median = np.median(rewards_array)
        mad = np.median(np.abs(rewards_array - median))
        if mad == 0:
            return [0.0] * len(rewards)
        return ((rewards_array - median) / mad).tolist()
    
    else:
        return rewards

def clip_rewards(rewards: List[float], clip_range: tuple = (-10.0, 10.0)) -> List[float]:
    """
    Aplica clipping a una lista de rewards.
    
    Args:
        rewards: Lista de rewards a clipar
        clip_range: Tupla (min, max) para clipping
        
    Returns:
        Lista de rewards clipados
    """
    if not rewards:
        return []
    
    min_val, max_val = clip_range
    return [max(min_val, min(max_val, reward)) for reward in rewards]

def calculate_reward_components(reward: float, components: Dict[str, float]) -> Dict[str, float]:
    """
    Calcula componentes de reward con validación.
    
    Args:
        reward: Reward total
        components: Diccionario de componentes
        
    Returns:
        Diccionario de componentes validados
    """
    validated_components = {}
    total_component_sum = 0.0
    
    for name, value in components.items():
        if isinstance(value, (int, float)) and not np.isnan(value) and np.isfinite(value):
            validated_components[name] = float(value)
            total_component_sum += float(value)
        else:
            validated_components[name] = 0.0
    
    # Verificar que la suma de componentes sea razonable
    if abs(total_component_sum - reward) > 0.01:
        logger.warning(f"Suma de componentes ({total_component_sum}) no coincide con reward total ({reward})")
    
    return validated_components

def format_reward_summary(rewards: List[float], components: Dict[str, List[float]]) -> str:
    """
    Formatea un resumen de rewards para logging.
    
    Args:
        rewards: Lista de rewards totales
        components: Diccionario de componentes por nombre
        
    Returns:
        String formateado con resumen
    """
    if not rewards:
        return "No rewards to summarize"
    
    stats = calculate_reward_statistics(rewards)
    
    summary = f"Rewards Summary:\n"
    summary += f"  Total: {len(rewards)} rewards\n"
    summary += f"  Mean: {stats['mean']:.4f}\n"
    summary += f"  Std: {stats['std']:.4f}\n"
    summary += f"  Range: [{stats['min']:.4f}, {stats['max']:.4f}]\n"
    
    if components:
        summary += f"  Components:\n"
        for name, comp_rewards in components.items():
            if comp_rewards:
                comp_stats = calculate_reward_statistics(comp_rewards)
                summary += f"    {name}: mean={comp_stats['mean']:.4f}, std={comp_stats['std']:.4f}\n"
    
    return summary

def validate_reward_inputs(reward: float, components: Dict[str, float], 
                          module_name: str = "unknown") -> bool:
    """
    Valida inputs de reward para detectar problemas.
    
    Args:
        reward: Reward total
        components: Componentes del reward
        module_name: Nombre del módulo para logging
        
    Returns:
        True si los inputs son válidos
    """
    # Validar reward principal
    if not isinstance(reward, (int, float)) or np.isnan(reward) or not np.isfinite(reward):
        logger.error(f"{module_name}: Reward inválido: {reward}")
        return False
    
    # Validar componentes
    for name, value in components.items():
        if not isinstance(value, (int, float)) or np.isnan(value) or not np.isfinite(value):
            logger.error(f"{module_name}: Componente '{name}' inválido: {value}")
            return False
    
    # Validar rangos razonables
    if abs(reward) > 1000:
        logger.warning(f"{module_name}: Reward muy grande: {reward}")
    
    return True

def create_reward_report(rewards: List[float], components: Dict[str, List[float]], 
                        module_name: str = "unknown") -> Dict[str, Any]:
    """
    Crea un reporte completo de rewards.
    
    Args:
        rewards: Lista de rewards totales
        components: Componentes por nombre
        module_name: Nombre del módulo
        
    Returns:
        Diccionario con reporte completo
    """
    report = {
        "module_name": module_name,
        "total_rewards": len(rewards),
        "reward_statistics": calculate_reward_statistics(rewards),
        "components": {}
    }
    
    for name, comp_rewards in components.items():
        if comp_rewards:
            report["components"][name] = {
                "count": len(comp_rewards),
                "statistics": calculate_reward_statistics(comp_rewards)
            }
    
    return report
