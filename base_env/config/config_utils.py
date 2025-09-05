# base_env/config/config_utils.py
"""
Utilidades centralizadas de configuración para el sistema de rewards.

Consolida todas las funciones de configuración duplicadas en un solo lugar
para evitar conflictos y mantener consistencia.
"""

from __future__ import annotations
import time
import logging
import hashlib
from typing import Dict, Any, Optional, Union, List
from pathlib import Path
import yaml

logger = logging.getLogger(__name__)

class ConfigCache:
    """
    Cache inteligente para configuraciones con TTL y validación.
    
    Evita recargar configuraciones idénticas y proporciona
    invalidación automática basada en tiempo.
    """
    
    def __init__(self, default_ttl: int = 300):
        """
        Inicializa el cache de configuración.
        
        Args:
            default_ttl: TTL por defecto en segundos (5 minutos)
        """
        self.cache = {}
        self.default_ttl = default_ttl
        self.hits = 0
        self.misses = 0
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Obtiene valor del cache con validación de TTL.
        
        Args:
            key: Clave del cache
            default: Valor por defecto si no existe o expiró
            
        Returns:
            Valor cacheado o default
        """
        if key not in self.cache:
            self.misses += 1
            return default
        
        entry = self.cache[key]
        if time.time() > entry['expires_at']:
            del self.cache[key]
            self.misses += 1
            return default
        
        self.hits += 1
        return entry['value']
    
    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        """
        Establece valor en el cache con TTL.
        
        Args:
            key: Clave del cache
            value: Valor a cachear
            ttl: TTL en segundos (opcional)
        """
        ttl = ttl or self.default_ttl
        self.cache[key] = {
            'value': value,
            'expires_at': time.time() + ttl,
            'created_at': time.time()
        }
    
    def invalidate(self, key: str) -> None:
        """Invalida una entrada específica del cache."""
        self.cache.pop(key, None)
    
    def clear(self) -> None:
        """Limpia todo el cache."""
        self.cache.clear()
        self.hits = 0
        self.misses = 0
    
    def get_stats(self) -> Dict[str, Any]:
        """Obtiene estadísticas del cache."""
        total_requests = self.hits + self.misses
        hit_rate = self.hits / max(total_requests, 1) * 100
        
        return {
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": hit_rate,
            "entries": len(self.cache),
            "default_ttl": self.default_ttl
        }

class PerformanceProfiler:
    """
    Profiler de rendimiento para operaciones de configuración.
    
    Rastrea tiempo de ejecución y memoria de operaciones
    críticas para optimización.
    """
    
    def __init__(self):
        """Inicializa el profiler."""
        self.operations = {}
        self.total_time = 0.0
        self.operation_count = 0
    
    def start_operation(self, operation_name: str) -> None:
        """Inicia el cronómetro para una operación."""
        self.operations[operation_name] = {
            'start_time': time.time(),
            'end_time': None,
            'duration': 0.0,
            'call_count': 0
        }
    
    def end_operation(self, operation_name: str) -> float:
        """
        Termina el cronómetro para una operación.
        
        Args:
            operation_name: Nombre de la operación
            
        Returns:
            Duración en segundos
        """
        if operation_name not in self.operations:
            return 0.0
        
        end_time = time.time()
        start_time = self.operations[operation_name]['start_time']
        duration = end_time - start_time
        
        self.operations[operation_name]['end_time'] = end_time
        self.operations[operation_name]['duration'] += duration
        self.operations[operation_name]['call_count'] += 1
        
        self.total_time += duration
        self.operation_count += 1
        
        return duration
    
    def get_stats(self, operation_name: str) -> Dict[str, Any]:
        """Obtiene estadísticas de una operación específica."""
        if operation_name not in self.operations:
            return {}
        
        op = self.operations[operation_name]
        avg_duration = op['duration'] / max(op['call_count'], 1)
        
        return {
            "total_duration": op['duration'],
            "call_count": op['call_count'],
            "avg_duration": avg_duration,
            "last_duration": op['duration'] - (op.get('prev_duration', 0))
        }
    
    def get_all_stats(self) -> Dict[str, Any]:
        """Obtiene estadísticas de todas las operaciones."""
        return {
            "operations": {name: self.get_stats(name) for name in self.operations.keys()},
            "total_time": self.total_time,
            "operation_count": self.operation_count
        }

class BatchProcessor:
    """
    Procesador de lotes para operaciones de configuración.
    
    Determina cuándo usar procesamiento por lotes vs individual
    basándose en el tamaño y complejidad de las operaciones.
    """
    
    def __init__(self, batch_threshold: int = 10):
        """
        Inicializa el procesador de lotes.
        
        Args:
            batch_threshold: Umbral mínimo para usar procesamiento por lotes
        """
        self.batch_threshold = batch_threshold
        self.batch_operations = 0
        self.individual_operations = 0
    
    def should_use_batch(self, batch_size: int) -> bool:
        """
        Determina si usar procesamiento por lotes.
        
        Args:
            batch_size: Tamaño del lote
            
        Returns:
            True si debe usar procesamiento por lotes
        """
        if batch_size >= self.batch_threshold:
            self.batch_operations += 1
            return True
        else:
            self.individual_operations += 1
            return False
    
    def get_stats(self) -> Dict[str, Any]:
        """Obtiene estadísticas del procesador de lotes."""
        total_operations = self.batch_operations + self.individual_operations
        batch_rate = self.batch_operations / max(total_operations, 1) * 100
        
        return {
            "batch_operations": self.batch_operations,
            "individual_operations": self.individual_operations,
            "batch_rate": batch_rate,
            "batch_threshold": self.batch_threshold
        }

class VectorizedCalculator:
    """
    Calculadora vectorizada para operaciones de configuración.
    
    Proporciona operaciones optimizadas para procesamiento
    de múltiples configuraciones simultáneamente.
    """
    
    def __init__(self):
        """Inicializa la calculadora vectorizada."""
        self.operations_performed = 0
        self.total_elements_processed = 0
    
    def vectorized_operation(self, data: List[Any], operation: str) -> List[Any]:
        """
        Realiza operación vectorizada sobre una lista de datos.
        
        Args:
            data: Lista de datos a procesar
            operation: Tipo de operación a realizar
            
        Returns:
            Lista de resultados
        """
        self.operations_performed += 1
        self.total_elements_processed += len(data)
        
        # Implementar operaciones vectorizadas específicas
        if operation == "normalize":
            return [self._normalize_item(item) for item in data]
        elif operation == "validate":
            return [self._validate_item(item) for item in data]
        else:
            return data
    
    def _normalize_item(self, item: Any) -> Any:
        """Normaliza un elemento individual."""
        if isinstance(item, (int, float)):
            return float(item)
        elif isinstance(item, str):
            return item.strip().lower()
        else:
            return item
    
    def _validate_item(self, item: Any) -> bool:
        """Valida un elemento individual."""
        if item is None:
            return False
        if isinstance(item, str) and not item.strip():
            return False
        return True
    
    def get_stats(self) -> Dict[str, Any]:
        """Obtiene estadísticas de la calculadora vectorizada."""
        avg_elements = self.total_elements_processed / max(self.operations_performed, 1)
        
        return {
            "operations_performed": self.operations_performed,
            "total_elements_processed": self.total_elements_processed,
            "avg_elements_per_operation": avg_elements
        }

class RewardValidator:
    """
    Validador de rewards con reglas de negocio específicas.
    
    Valida configuraciones de rewards contra reglas de negocio
    y proporciona sugerencias de corrección.
    """
    
    def __init__(self):
        """Inicializa el validador de rewards."""
        self.validation_rules = self._load_validation_rules()
        self.validation_errors = []
        self.validation_warnings = []
    
    def _load_validation_rules(self) -> Dict[str, Any]:
        """Carga reglas de validación para rewards."""
        return {
            "weight_range": (0.0, 1.0),
            "cap_range": (0.0, 10.0),
            "required_fields": ["enabled", "weight"],
            "forbidden_combinations": [
                ("enabled", False, "weight", 0.0),
                ("weight", 0.0, "enabled", True)
            ]
        }
    
    def validate_reward_config(self, config: Dict[str, Any], module_name: str) -> bool:
        """
        Valida configuración de un módulo de reward.
        
        Args:
            config: Configuración del módulo
            module_name: Nombre del módulo
            
        Returns:
            True si la configuración es válida
        """
        self.validation_errors.clear()
        self.validation_warnings.clear()
        
        # Validar campos requeridos
        for field in self.validation_rules["required_fields"]:
            if field not in config:
                self.validation_errors.append(f"{module_name}: Campo requerido '{field}' faltante")
        
        # Validar rangos de valores
        if "weight" in config:
            weight = config["weight"]
            min_weight, max_weight = self.validation_rules["weight_range"]
            if not (min_weight <= weight <= max_weight):
                self.validation_errors.append(
                    f"{module_name}: Weight {weight} fuera de rango [{min_weight}, {max_weight}]"
                )
        
        if "per_trade_cap" in config:
            cap = config["per_trade_cap"]
            min_cap, max_cap = self.validation_rules["cap_range"]
            if not (min_cap <= cap <= max_cap):
                self.validation_warnings.append(
                    f"{module_name}: Cap {cap} fuera de rango recomendado [{min_cap}, {max_cap}]"
                )
        
        # Validar combinaciones prohibidas
        for combo in self.validation_rules["forbidden_combinations"]:
            field1, value1, field2, value2 = combo
            if (config.get(field1) == value1 and config.get(field2) == value2):
                self.validation_errors.append(
                    f"{module_name}: Combinación inválida {field1}={value1} y {field2}={value2}"
                )
        
        return len(self.validation_errors) == 0
    
    def get_validation_report(self) -> Dict[str, Any]:
        """Obtiene reporte de validación."""
        return {
            "errors": self.validation_errors.copy(),
            "warnings": self.validation_warnings.copy(),
            "is_valid": len(self.validation_errors) == 0
        }

# Instancias globales
_global_config_cache = ConfigCache()
_global_profiler = PerformanceProfiler()
_global_batch_processor = BatchProcessor()
_global_vectorized_calc = VectorizedCalculator()
_global_reward_validator = RewardValidator()

# Funciones de conveniencia
def get_config_cache() -> ConfigCache:
    """Obtiene el cache global de configuraciones."""
    return _global_config_cache

def get_profiler() -> PerformanceProfiler:
    """Obtiene el profiler global."""
    return _global_profiler

def get_batch_processor() -> BatchProcessor:
    """Obtiene el procesador de lotes global."""
    return _global_batch_processor

def get_vectorized_calculator() -> VectorizedCalculator:
    """Obtiene la calculadora vectorizada global."""
    return _global_vectorized_calc

def get_reward_validator() -> RewardValidator:
    """Obtiene el validador de rewards global."""
    return _global_reward_validator

def profile_reward_calculation(operation_name: str):
    """
    Decorator para perfilar cálculos de rewards.
    
    Args:
        operation_name: Nombre de la operación a perfilar
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            profiler = get_profiler()
            profiler.start_operation(operation_name)
            try:
                result = func(*args, **kwargs)
                return result
            finally:
                profiler.end_operation(operation_name)
        return wrapper
    return decorator

def load_config_with_cache(config: Dict[str, Any], module_name: str) -> Dict[str, Any]:
    """
    Carga configuración con cache inteligente.
    
    Args:
        config: Configuración a cargar
        module_name: Nombre del módulo
        
    Returns:
        Configuración cargada y validada
    """
    cache = get_config_cache()
    validator = get_reward_validator()
    
    # Generar clave de cache basada en contenido
    config_str = str(sorted(config.items()))
    cache_key = f"{module_name}_{hashlib.md5(config_str.encode()).hexdigest()[:8]}"
    
    # Intentar obtener del cache
    cached_config = cache.get(cache_key)
    if cached_config is not None:
        return cached_config
    
    # Validar configuración
    if not validator.validate_reward_config(config, module_name):
        logger.warning(f"Configuración inválida para {module_name}: {validator.get_validation_report()}")
    
    # Cachear configuración válida
    cache.set(cache_key, config, ttl=300)  # 5 minutos
    
    return config

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
    value = config.get(key, default)
    
    # Logging para debugging
    if value != default:
        logger.debug(f"{module_name}: {key} = {value}")
    
    return value

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
    problems = []
    
    if required_keys:
        for key in required_keys:
            if key not in config:
                problems.append(f"Clave requerida '{key}' faltante")
    
    # Validar tipos de datos comunes
    for key, value in config.items():
        if key.endswith('_weight') and not isinstance(value, (int, float)):
            problems.append(f"'{key}' debe ser numérico, encontrado: {type(value)}")
        
        if key.endswith('_enabled') and not isinstance(value, bool):
            problems.append(f"'{key}' debe ser booleano, encontrado: {type(value)}")
    
    return problems

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
    merged = base_config.copy()
    
    for key, value in override_config.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = merge_configs(merged[key], value)
        else:
            merged[key] = value
    
    return merged

def save_config_to_yaml(config: Dict[str, Any], file_path: Union[str, Path]) -> None:
    """
    Guarda configuración en archivo YAML.
    
    Args:
        config: Configuración a guardar
        file_path: Ruta del archivo
    """
    file_path = Path(file_path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(file_path, 'w', encoding='utf-8') as f:
        yaml.dump(config, f, default_flow_style=False, indent=2)

def load_config_from_yaml(file_path: Union[str, Path]) -> Dict[str, Any]:
    """
    Carga configuración desde archivo YAML.
    
    Args:
        file_path: Ruta del archivo
        
    Returns:
        Configuración cargada
    """
    file_path = Path(file_path)
    
    if not file_path.exists():
        logger.warning(f"Archivo de configuración no encontrado: {file_path}")
        return {}
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f) or {}
    except Exception as e:
        logger.error(f"Error cargando configuración desde {file_path}: {e}")
        return {}

# Función de conveniencia para obtener todas las utilidades
def get_all_utils() -> Dict[str, Any]:
    """Obtiene todas las utilidades de configuración."""
    return {
        "cache": get_config_cache(),
        "profiler": get_profiler(),
        "batch_processor": get_batch_processor(),
        "vectorized_calc": get_vectorized_calculator(),
        "validator": get_reward_validator()
    }
