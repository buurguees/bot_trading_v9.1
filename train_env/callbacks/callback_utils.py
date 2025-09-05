# train_env/callbacks/callback_utils.py
"""
Utilidades comunes para callbacks de entrenamiento.
Funciones compartidas para logging, validación, caching y manejo de archivos.
"""

from __future__ import annotations
import json
import logging
import threading
import time
from pathlib import Path
from typing import Dict, Any, List, Optional, Union, Callable
from dataclasses import dataclass
from datetime import datetime
import hashlib

logger = logging.getLogger(__name__)

@dataclass
class CallbackConfig:
    """Configuración común para callbacks."""
    verbose: int = 0
    log_level: int = logging.INFO
    enable_caching: bool = True
    cache_ttl: float = 300.0  # 5 minutos
    retry_attempts: int = 3
    retry_delay: float = 1.0

class FileCache:
    """Cache simple para archivos con TTL."""
    
    def __init__(self, ttl: float = 300.0):
        self.ttl = ttl
        self._cache: Dict[str, Dict[str, Any]] = {}
        self._lock = threading.RLock()
    
    def get(self, file_path: Path) -> Optional[Any]:
        """Obtiene datos del cache si son válidos."""
        with self._lock:
            key = str(file_path.absolute())
            if key in self._cache:
                entry = self._cache[key]
                if time.time() - entry['timestamp'] < self.ttl:
                    return entry['data']
                else:
                    del self._cache[key]
            return None
    
    def set(self, file_path: Path, data: Any) -> None:
        """Almacena datos en el cache."""
        with self._lock:
            key = str(file_path.absolute())
            self._cache[key] = {
                'data': data,
                'timestamp': time.time()
            }
    
    def invalidate(self, file_path: Path) -> None:
        """Invalida entrada del cache."""
        with self._lock:
            key = str(file_path.absolute())
            self._cache.pop(key, None)
    
    def clear(self) -> None:
        """Limpia todo el cache."""
        with self._lock:
            self._cache.clear()

class CallbackLogger:
    """Logger especializado para callbacks."""
    
    def __init__(self, name: str, verbose: int = 0, log_level: int = logging.INFO):
        self.logger = logging.getLogger(f"callback.{name}")
        self.verbose = verbose
        self.log_level = log_level
    
    def info(self, message: str, **kwargs) -> None:
        """Log info con contexto de callback."""
        if self.verbose > 0:
            self.logger.log(self.log_level, f"[{self.logger.name.upper()}] {message}", **kwargs)
    
    def debug(self, message: str, **kwargs) -> None:
        """Log debug con contexto de callback."""
        if self.verbose > 1:
            self.logger.debug(f"[{self.logger.name.upper()}] {message}", **kwargs)
    
    def warning(self, message: str, **kwargs) -> None:
        """Log warning con contexto de callback."""
        self.logger.warning(f"[{self.logger.name.upper()}] {message}", **kwargs)
    
    def error(self, message: str, **kwargs) -> None:
        """Log error con contexto de callback."""
        self.logger.error(f"[{self.logger.name.upper()}] {message}", **kwargs)

def safe_json_load(file_path: Path, default: Any = None, 
                  retry_attempts: int = 3, retry_delay: float = 1.0) -> Any:
    """
    Carga JSON de forma segura con reintentos.
    
    Args:
        file_path: Ruta al archivo JSON.
        default: Valor por defecto si falla la carga.
        retry_attempts: Número de reintentos.
        retry_delay: Delay entre reintentos.
        
    Returns:
        Datos cargados o valor por defecto.
    """
    for attempt in range(retry_attempts):
        try:
            if not file_path.exists():
                logger.warning(f"Archivo no encontrado: {file_path}")
                return default
            
            if file_path.stat().st_size == 0:
                logger.warning(f"Archivo vacío: {file_path}")
                return default
            
            with file_path.open('r', encoding='utf-8') as f:
                data = json.load(f)
                return data
                
        except json.JSONDecodeError as e:
            logger.error(f"Error JSON en {file_path} (intento {attempt + 1}): {e}")
            if attempt < retry_attempts - 1:
                time.sleep(retry_delay)
            else:
                return default
        except (OSError, IOError) as e:
            logger.error(f"Error I/O en {file_path} (intento {attempt + 1}): {e}")
            if attempt < retry_attempts - 1:
                time.sleep(retry_delay)
            else:
                return default
        except Exception as e:
            logger.error(f"Error inesperado en {file_path} (intento {attempt + 1}): {e}")
            return default
    
    return default

def safe_json_save(data: Any, file_path: Path, 
                  retry_attempts: int = 3, retry_delay: float = 1.0) -> bool:
    """
    Guarda JSON de forma segura con reintentos.
    
    Args:
        data: Datos a guardar.
        file_path: Ruta de destino.
        retry_attempts: Número de reintentos.
        retry_delay: Delay entre reintentos.
        
    Returns:
        True si se guardó exitosamente.
    """
    # Crear directorio padre si no existe
    file_path.parent.mkdir(parents=True, exist_ok=True)
    
    for attempt in range(retry_attempts):
        try:
            # Escribir a archivo temporal primero
            temp_path = file_path.with_suffix('.tmp')
            with temp_path.open('w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            
            # Mover archivo temporal a destino
            temp_path.replace(file_path)
            return True
            
        except (OSError, IOError) as e:
            logger.error(f"Error I/O guardando {file_path} (intento {attempt + 1}): {e}")
            if attempt < retry_attempts - 1:
                time.sleep(retry_delay)
        except Exception as e:
            logger.error(f"Error inesperado guardando {file_path} (intento {attempt + 1}): {e}")
            return False
    
    return False

def validate_callback_params(**kwargs) -> None:
    """
    Valida parámetros comunes de callbacks.
    
    Args:
        **kwargs: Parámetros a validar.
        
    Raises:
        ValueError: Si algún parámetro es inválido.
    """
    if 'every_steps' in kwargs and kwargs['every_steps'] <= 0:
        raise ValueError("every_steps debe ser positivo")
    
    if 'save_every_steps' in kwargs and kwargs['save_every_steps'] <= 0:
        raise ValueError("save_every_steps debe ser positivo")
    
    if 'top_k' in kwargs and kwargs['top_k'] <= 0:
        raise ValueError("top_k debe ser positivo")
    
    if 'verbose' in kwargs and kwargs['verbose'] < 0:
        raise ValueError("verbose debe ser no negativo")

def calculate_file_hash(file_path: Path) -> Optional[str]:
    """
    Calcula hash MD5 de un archivo para detectar cambios.
    
    Args:
        file_path: Ruta al archivo.
        
    Returns:
        Hash MD5 o None si hay error.
    """
    try:
        if not file_path.exists():
            return None
        
        hash_md5 = hashlib.md5()
        with file_path.open("rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()
    except Exception as e:
        logger.error(f"Error calculando hash de {file_path}: {e}")
        return None

def format_timesteps(timesteps: int) -> str:
    """Formatea timesteps de forma legible."""
    if timesteps >= 1_000_000:
        return f"{timesteps / 1_000_000:.1f}M"
    elif timesteps >= 1_000:
        return f"{timesteps / 1_000:.1f}K"
    else:
        return str(timesteps)

def format_duration(seconds: float) -> str:
    """Formatea duración en formato legible."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        return f"{seconds / 60:.1f}m"
    else:
        return f"{seconds / 3600:.1f}h"

class CallbackMetrics:
    """Métricas para callbacks."""
    
    def __init__(self):
        self.start_time = time.time()
        self.last_update = self.start_time
        self.operations_count = 0
        self.errors_count = 0
        self._lock = threading.RLock()
    
    def record_operation(self) -> None:
        """Registra una operación exitosa."""
        with self._lock:
            self.operations_count += 1
            self.last_update = time.time()
    
    def record_error(self) -> None:
        """Registra un error."""
        with self._lock:
            self.errors_count += 1
    
    def get_stats(self) -> Dict[str, Any]:
        """Obtiene estadísticas del callback."""
        with self._lock:
            current_time = time.time()
            return {
                "duration": current_time - self.start_time,
                "operations": self.operations_count,
                "errors": self.errors_count,
                "last_update": current_time - self.last_update,
                "error_rate": self.errors_count / max(self.operations_count, 1)
            }

def create_callback_logger(name: str, verbose: int = 0) -> CallbackLogger:
    """Crea un logger para callback."""
    return CallbackLogger(name, verbose)

def create_file_cache(ttl: float = 300.0) -> FileCache:
    """Crea un cache de archivos."""
    return FileCache(ttl)

def create_callback_metrics() -> CallbackMetrics:
    """Crea métricas para callback."""
    return CallbackMetrics()
