# train_env/worker_manager.py
"""
Worker Manager Inteligente para optimización de recursos en entrenamiento PPO.
Calcula el número óptimo de workers basado en recursos del sistema.

Mejoras implementadas:
- Robustez: Excepciones específicas, monitoreo dinámico, validación de disco/GPU
- Optimización: Profiling con tracemalloc, ajuste por n_steps, cálculos basados en datos reales
- Lógica SB3: Integración con parámetros PPO, soporte para entornos heterogéneos
- Monitoreo: Validación dinámica durante entrenamiento, estimación de memoria precisa
"""

from __future__ import annotations
import psutil
import os
import warnings
import time
import traceback
from typing import Tuple, Dict, Any, Optional, Union, List
from dataclasses import dataclass
import logging
import tracemalloc
try:
    from retrying import retry
except ImportError:
    # Fallback simple si retrying no está disponible
    def retry(stop_max_attempt_number=3, wait_fixed=1000):
        def decorator(func):
            return func
        return decorator

# Importación condicional de torch para evitar errores si no está disponible
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None

logger = logging.getLogger(__name__)

@dataclass
class SystemResources:
    """Información de recursos del sistema con métricas avanzadas."""
    cpu_count_physical: int
    cpu_count_logical: int
    memory_total_gb: float
    memory_available_gb: float
    memory_usage_percent: float
    disk_free_gb: float
    gpu_available: bool = False
    gpu_memory_free_mb: Optional[float] = None
    gpu_memory_total_mb: Optional[float] = None
    gpu_count: int = 0
    cpu_frequency_mhz: Optional[float] = None
    load_average: Optional[Tuple[float, float, float]] = None
    timestamp: float = 0.0

@dataclass
class WorkerRecommendation:
    """Recomendación de configuración de workers con métricas detalladas."""
    optimal_workers: int
    max_safe_workers: int
    recommended_batch_size: int
    memory_per_worker_mb: int
    warning_message: Optional[str] = None
    performance_estimate: str = "unknown"
    estimated_memory_usage_gb: float = 0.0
    estimated_training_time_hours: Optional[float] = None
    resource_utilization: Dict[str, float] = None
    gpu_recommendation: Optional[str] = None

@dataclass
class HeterogeneousWorkerConfig:
    """Configuración para entornos heterogéneos con diferentes tipos de workers."""
    data_workers: int = 1  # Workers para datos históricos pesados
    training_workers: int = 1  # Workers para entrenamiento
    eval_workers: int = 1  # Workers para evaluación
    memory_per_data_worker_mb: int = 2048
    memory_per_training_worker_mb: int = 1024
    memory_per_eval_worker_mb: int = 512

class WorkerManager:
    """Gestor inteligente de workers para entrenamiento PPO con monitoreo dinámico."""
    
    def __init__(self, safety_margin: float = 0.2, log_dir: Optional[str] = None, 
                 enable_monitoring: bool = True, monitoring_interval: float = 30.0):
        """
        Inicializa el WorkerManager con monitoreo dinámico.
        
        Args:
            safety_margin: Margen de seguridad para memoria (0.0-1.0).
            log_dir: Directorio para logs/checkpoints (para validar disco).
            enable_monitoring: Si habilitar monitoreo dinámico durante entrenamiento.
            monitoring_interval: Intervalo de monitoreo en segundos.
        """
        self.safety_margin = max(0.0, min(1.0, safety_margin))
        self.log_dir = log_dir or os.getcwd()
        self.enable_monitoring = enable_monitoring
        self.monitoring_interval = monitoring_interval
        self._system_resources = self._analyze_system_resources()
        self._last_memory_snapshot = None
        self._last_monitoring_time = 0.0
        self._memory_calibration_data = []
        self._performance_history = []
        
        # Inicializar tracemalloc si está habilitado
        if self.enable_monitoring and not tracemalloc.is_tracing():
            tracemalloc.start()
            logger.info("🔍 Monitoreo de memoria habilitado")
    
    @retry(stop_max_attempt_number=3, wait_fixed=1000)
    def _analyze_system_resources(self) -> SystemResources:
        """Analiza los recursos disponibles del sistema con métricas avanzadas."""
        try:
            # Información básica del sistema
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage(self.log_dir)
            cpu_freq = psutil.cpu_freq()
            load_avg = psutil.getloadavg() if hasattr(psutil, 'getloadavg') else None
            
            # Información de GPU
            gpu_available = False
            gpu_memory_free_mb = None
            gpu_memory_total_mb = None
            gpu_count = 0
            
            if TORCH_AVAILABLE and torch.cuda.is_available():
                try:
                    gpu_available = True
                    gpu_count = torch.cuda.device_count()
                    gpu_props = torch.cuda.get_device_properties(0)
                    gpu_memory_total_mb = gpu_props.total_memory / (1024**2)
                    # Estimar memoria libre (aproximación)
                    gpu_memory_free_mb = gpu_memory_total_mb * 0.8
                    logger.debug(f"GPU detectada: {gpu_props.name} ({gpu_memory_total_mb:.0f}MB total)")
                except Exception as gpu_error:
                    logger.warning(f"Error detectando GPU: {gpu_error}")
            
            return SystemResources(
                cpu_count_physical=psutil.cpu_count(logical=False) or 1,
                cpu_count_logical=psutil.cpu_count(logical=True) or 1,
                memory_total_gb=memory.total / (1024**3),
                memory_available_gb=memory.available / (1024**3),
                memory_usage_percent=memory.percent,
                disk_free_gb=disk.free / (1024**3),
                gpu_available=gpu_available,
                gpu_memory_free_mb=gpu_memory_free_mb,
                gpu_memory_total_mb=gpu_memory_total_mb,
                gpu_count=gpu_count,
                cpu_frequency_mhz=cpu_freq.current if cpu_freq else None,
                load_average=load_avg,
                timestamp=time.time()
            )
            
        except psutil.Error as e:
            logger.error(f"Error de psutil analizando recursos: {e}", exc_info=True)
            return self._get_fallback_resources()
        except FileNotFoundError as e:
            logger.error(f"Archivo no encontrado: {e}", exc_info=True)
            return self._get_fallback_resources()
        except Exception as e:
            logger.error(f"Error inesperado analizando recursos: {e}", exc_info=True)
            return self._get_fallback_resources()
    
    def _get_fallback_resources(self) -> SystemResources:
        """Recursos de fallback cuando no se puede analizar el sistema."""
        logger.warning("Usando recursos de fallback - análisis del sistema falló")
        return SystemResources(
            cpu_count_physical=2,
            cpu_count_logical=4,
            memory_total_gb=8.0,
            memory_available_gb=4.0,
            memory_usage_percent=50.0,
            disk_free_gb=10.0,
            gpu_available=False,
            gpu_memory_free_mb=None,
            gpu_memory_total_mb=None,
            gpu_count=0,
            cpu_frequency_mhz=None,
            load_average=None,
            timestamp=time.time()
        )
    
    def update_resources(self) -> SystemResources:
        """Actualiza dinámicamente los recursos del sistema."""
        self._system_resources = self._analyze_system_resources()
        return self._system_resources
    
    def monitor_during_training(self, current_workers: int, current_batch_size: int) -> Dict[str, Any]:
        """
        Monitoreo dinámico durante el entrenamiento.
        
        Args:
            current_workers: Número actual de workers.
            current_batch_size: Tamaño actual del batch.
            
        Returns:
            Dict con recomendaciones y advertencias.
        """
        if not self.enable_monitoring:
            return {"status": "monitoring_disabled"}
        
        current_time = time.time()
        if current_time - self._last_monitoring_time < self.monitoring_interval:
            return {"status": "too_soon", "next_check_in": self.monitoring_interval - (current_time - self._last_monitoring_time)}
        
        self._last_monitoring_time = current_time
        resources = self.update_resources()
        
        # Verificar si los recursos han cambiado significativamente
        warnings = []
        recommendations = []
        
        # Verificar memoria
        if resources.memory_usage_percent > 85:
            warnings.append(f"Uso de memoria crítico: {resources.memory_usage_percent:.1f}%")
            if current_workers > 1:
                recommendations.append("Considerar reducir workers")
        
        # Verificar CPU load
        if resources.load_average and resources.load_average[0] > resources.cpu_count_physical * 0.8:
            warnings.append(f"CPU sobrecargado: load {resources.load_average[0]:.2f}")
            recommendations.append("Considerar reducir batch_size")
        
        # Verificar disco
        if resources.disk_free_gb < 2:
            warnings.append(f"Espacio en disco crítico: {resources.disk_free_gb:.1f}GB")
            recommendations.append("Limpiar archivos temporales")
        
        # Calibrar memoria si es necesario
        if len(self._memory_calibration_data) < 5:
            self._calibrate_memory_usage(current_workers, current_batch_size)
        
        return {
            "status": "monitored",
            "warnings": warnings,
            "recommendations": recommendations,
            "resources": {
                "memory_usage_percent": resources.memory_usage_percent,
                "cpu_load": resources.load_average[0] if resources.load_average else None,
                "disk_free_gb": resources.disk_free_gb
            }
        }
    
    def _calibrate_memory_usage(self, workers: int, batch_size: int) -> None:
        """Calibra el uso de memoria con datos reales."""
        try:
            if not tracemalloc.is_tracing():
                return
            
            snapshot = tracemalloc.take_snapshot()
            if self._last_memory_snapshot:
                stats = snapshot.compare_to(self._last_memory_snapshot, 'lineno')
                peak_memory_mb = sum(stat.size for stat in stats[:10]) / (1024**2)
                
                # Almacenar datos de calibración
                self._memory_calibration_data.append({
                    'workers': workers,
                    'batch_size': batch_size,
                    'peak_memory_mb': peak_memory_mb,
                    'timestamp': time.time()
                })
                
                # Mantener solo los últimos 10 puntos
                if len(self._memory_calibration_data) > 10:
                    self._memory_calibration_data.pop(0)
                
                logger.debug(f"Calibración de memoria: {peak_memory_mb:.1f}MB para {workers} workers, batch {batch_size}")
            
            self._last_memory_snapshot = snapshot
            
        except Exception as e:
            logger.warning(f"Error en calibración de memoria: {e}")
    
    def get_calibrated_memory_estimate(self, workers: int, batch_size: int) -> int:
        """Obtiene estimación de memoria calibrada basada en datos reales."""
        if not self._memory_calibration_data:
            return workers * batch_size * 0.001  # Estimación básica
        
        # Calcular promedio de uso de memoria por worker
        total_memory = sum(data['peak_memory_mb'] for data in self._memory_calibration_data)
        total_workers = sum(data['workers'] for data in self._memory_calibration_data)
        
        if total_workers > 0:
            avg_memory_per_worker = total_memory / total_workers
            return int(avg_memory_per_worker * workers * 1.2)  # 20% margen
        
        return workers * batch_size * 0.001
    
    def get_optimal_workers(self, 
                          target_workers: Optional[int] = None,
                          memory_per_worker_mb: int = 1024,
                          min_workers: int = 1,
                          max_workers: Optional[int] = None,
                          n_steps: Optional[int] = None,
                          ppo_config: Optional[Dict[str, Any]] = None) -> WorkerRecommendation:
        """
        Calcula la configuración óptima de workers con integración SB3 avanzada.
        
        Args:
            target_workers: Número deseado de workers.
            memory_per_worker_mb: Memoria estimada por worker en MB.
            min_workers: Mínimo número de workers.
            max_workers: Máximo número de workers.
            n_steps: Pasos por rollout de PPO (para ajuste fino).
            ppo_config: Configuración PPO para ajustes específicos.
            
        Returns:
            WorkerRecommendation con configuración óptima.
        """
        resources = self.update_resources()
        
        # Ajuste con profiling y calibración
        if self._memory_calibration_data:
            memory_per_worker_mb = self.get_calibrated_memory_estimate(
                target_workers or 4, 
                ppo_config.get('batch_size', 64) if ppo_config else 64
            )
        else:
            memory_per_worker_mb = self._profile_memory_usage(memory_per_worker_mb)
        
        # Cálculos básicos
        cpu_optimal = max(min_workers, resources.cpu_count_physical - 1)
        memory_per_worker_gb = memory_per_worker_mb / 1024
        available_memory_gb = resources.memory_available_gb * (1 - self.safety_margin)
        memory_optimal = max(min_workers, int(available_memory_gb / memory_per_worker_gb))
        logical_optimal = max(min_workers, resources.cpu_count_logical // 2)
        
        # Ajuste por GPU
        gpu_optimal = None
        if resources.gpu_available and resources.gpu_memory_free_mb:
            gpu_optimal = max(min_workers, int(resources.gpu_memory_free_mb / memory_per_worker_mb))
            optimal_workers = min(cpu_optimal, memory_optimal, logical_optimal, gpu_optimal)
        else:
            optimal_workers = min(cpu_optimal, memory_optimal, logical_optimal)
        
        # Aplicar límites
        if max_workers is not None:
            optimal_workers = min(optimal_workers, max_workers)
        if target_workers is not None:
            optimal_workers = min(optimal_workers, target_workers)
        
        # Calcular batch size óptimo con integración SB3
        recommended_batch_size = self._calculate_optimal_batch_size(
            optimal_workers, resources, n_steps, ppo_config
        )
        
        # Generar advertencias y recomendaciones
        warning_message = self._generate_warnings(
            optimal_workers, target_workers, resources, memory_per_worker_mb
        )
        performance_estimate = self._estimate_performance(optimal_workers, resources)
        
        # Calcular métricas adicionales
        estimated_memory_usage_gb = (optimal_workers * memory_per_worker_mb) / 1024
        estimated_training_time = self._estimate_training_time(optimal_workers, resources, n_steps)
        
        # Calcular utilización de recursos
        resource_utilization = {
            "cpu_utilization": (optimal_workers / resources.cpu_count_physical) * 100,
            "memory_utilization": (estimated_memory_usage_gb / resources.memory_total_gb) * 100,
            "gpu_utilization": (optimal_workers / gpu_optimal * 100) if gpu_optimal else 0
        }
        
        # Recomendación de GPU
        gpu_recommendation = self._get_gpu_recommendation(resources, optimal_workers)
        
        return WorkerRecommendation(
            optimal_workers=optimal_workers,
            max_safe_workers=min(cpu_optimal, memory_optimal),
            recommended_batch_size=recommended_batch_size,
            memory_per_worker_mb=memory_per_worker_mb,
            warning_message=warning_message,
            performance_estimate=performance_estimate,
            estimated_memory_usage_gb=estimated_memory_usage_gb,
            estimated_training_time_hours=estimated_training_time,
            resource_utilization=resource_utilization,
            gpu_recommendation=gpu_recommendation
        )
    
    def _profile_memory_usage(self, estimated_memory_mb: int) -> int:
        """Profiling de memoria para ajustar estimación."""
        if not tracemalloc.is_tracing():
            tracemalloc.start()
        snapshot = tracemalloc.take_snapshot()
        if self._last_memory_snapshot:
            stats = snapshot.compare_to(self._last_memory_snapshot, 'lineno')
            peak_memory_mb = sum(stat.size for stat in stats[:10]) / (1024**2)
            estimated_memory_mb = max(estimated_memory_mb, int(peak_memory_mb * 1.2))  # 20% margen
        self._last_memory_snapshot = snapshot
        return estimated_memory_mb
    
    def _calculate_optimal_batch_size(self, 
                                    workers: int, 
                                    resources: SystemResources,
                                    n_steps: Optional[int] = None,
                                    ppo_config: Optional[Dict[str, Any]] = None) -> int:
        """Calcula el batch size óptimo con integración SB3."""
        # Batch size base según memoria
        base_batch_per_worker = 64
        if resources.memory_total_gb < 8:
            base_batch_per_worker = 32
        elif resources.memory_total_gb < 16:
            base_batch_per_worker = 48
        elif resources.memory_total_gb >= 32:
            base_batch_per_worker = 128
        
        # Ajuste por n_steps de PPO
        if n_steps:
            base_batch_per_worker = min(base_batch_per_worker, n_steps // workers)
        
        # Ajuste por configuración PPO específica
        if ppo_config:
            if 'batch_size' in ppo_config:
                base_batch_per_worker = min(base_batch_per_worker, ppo_config['batch_size'])
            if 'n_epochs' in ppo_config and ppo_config['n_epochs'] > 10:
                base_batch_per_worker = int(base_batch_per_worker * 0.8)  # Reducir si muchos epochs
        
        # Multiplicador por número de workers
        multiplier = 1.0 if workers <= 2 else 0.8 if workers <= 4 else 0.6
        
        # Ajuste por GPU
        if resources.gpu_available:
            multiplier *= 1.2  # GPU permite batch sizes más grandes
        
        return max(16, int(base_batch_per_worker * multiplier))
    
    def _estimate_training_time(self, workers: int, resources: SystemResources, n_steps: Optional[int]) -> Optional[float]:
        """Estima el tiempo de entrenamiento en horas."""
        if not n_steps:
            return None
        
        # Estimación basada en workers y recursos
        base_time_per_step = 0.001  # 1ms por step base
        if resources.gpu_available:
            base_time_per_step *= 0.5  # GPU es más rápido
        
        # Ajuste por número de workers (más workers = más overhead)
        worker_factor = 1.0 if workers <= 2 else 1.1 if workers <= 4 else 1.3
        
        # Ajuste por memoria (más memoria = más rápido)
        memory_factor = 1.0
        if resources.memory_total_gb >= 32:
            memory_factor = 0.8
        elif resources.memory_total_gb < 8:
            memory_factor = 1.5
        
        estimated_seconds = n_steps * base_time_per_step * worker_factor * memory_factor
        return estimated_seconds / 3600  # Convertir a horas
    
    def _get_gpu_recommendation(self, resources: SystemResources, workers: int) -> Optional[str]:
        """Genera recomendaciones específicas para GPU."""
        if not resources.gpu_available:
            return "GPU no disponible - usar CPU"
        
        if resources.gpu_memory_total_mb:
            if resources.gpu_memory_total_mb < 4000:  # < 4GB
                return "GPU con poca memoria - reducir batch_size"
            elif resources.gpu_memory_total_mb > 8000:  # > 8GB
                return "GPU potente - se puede aumentar batch_size"
            else:
                return "GPU estándar - configuración equilibrada"
        
        return "GPU disponible - configuración automática"
    
    def _generate_warnings(self, 
                          optimal_workers: int, 
                          target_workers: Optional[int],
                          resources: SystemResources,
                          memory_per_worker_mb: int) -> Optional[str]:
        warnings_list = []
        if resources.memory_usage_percent > 80:
            warnings_list.append(f"Uso de memoria alto ({resources.memory_usage_percent:.1f}%)")
        if target_workers and optimal_workers < target_workers:
            warnings_list.append(f"Reduciendo workers de {target_workers} a {optimal_workers}")
        estimated_memory_usage = (optimal_workers * memory_per_worker_mb) / 1024
        if estimated_memory_usage > resources.memory_available_gb * 0.8:
            warnings_list.append(f"Uso memoria estimado ({estimated_memory_usage:.1f}GB) alto")
        if optimal_workers >= resources.cpu_count_physical:
            warnings_list.append("Usando todos los cores físicos")
        # Nueva: advertencia de disco
        if resources.disk_free_gb < 5:
            warnings_list.append(f"Espacio libre bajo ({resources.disk_free_gb:.1f}GB)")
        return "; ".join(warnings_list) if warnings_list else None
    
    def _estimate_performance(self, workers: int, resources: SystemResources) -> str:
        """Estima el nivel de performance esperado."""
        score = workers * (1 if resources.gpu_available else 0.5)
        if score <= 2:
            return "bajo"
        elif score <= 4:
            return "medio"
        elif score <= 8:
            return "alto"
        else:
            return "muy alto"
    
    def get_system_info(self) -> Dict[str, Any]:
        """Retorna información detallada del sistema."""
        resources = self.update_resources()
        return {
            "cpu_physical": resources.cpu_count_physical,
            "cpu_logical": resources.cpu_count_logical,
            "memory_total_gb": round(resources.memory_total_gb, 2),
            "memory_available_gb": round(resources.memory_available_gb, 2),
            "memory_usage_percent": round(resources.memory_usage_percent, 1),
            "disk_free_gb": round(resources.disk_free_gb, 2),
            "gpu_available": resources.gpu_available,
            "gpu_memory_free_mb": resources.gpu_memory_free_mb,
            "safety_margin": self.safety_margin
        }
    
    def validate_configuration(self, 
                             n_envs: int, 
                             batch_size: int,
                             n_steps: int) -> Tuple[bool, Optional[str]]:
        """
        Valida si la configuración es segura.
        
        Args:
            n_envs: Número de entornos.
            batch_size: Tamaño del batch.
            n_steps: Pasos por rollout.
            
        Returns:
            Tuple[is_valid, warning_message]
        """
        resources = self.update_resources()
        estimated_memory_mb = (n_envs * batch_size * n_steps * 0.001)  # Ajustar con profiling
        estimated_memory_mb = self._profile_memory_usage(estimated_memory_mb)
        estimated_memory_gb = estimated_memory_mb / 1024
        if estimated_memory_gb > resources.memory_available_gb * 0.9:
            return False, f"Memoria estimada ({estimated_memory_gb:.1f}GB) excede límite"
        if n_envs > resources.cpu_count_physical * 2:
            return False, f"Demasiados workers ({n_envs}) para {resources.cpu_count_physical} cores"
        if resources.disk_free_gb < 5:
            return False, f"Espacio libre bajo ({resources.disk_free_gb:.1f}GB)"
        return True, None
    
    def get_heterogeneous_worker_config(self, 
                                      data_intensive: bool = False,
                                      eval_required: bool = True) -> HeterogeneousWorkerConfig:
        """
        Genera configuración para entornos heterogéneos.
        
        Args:
            data_intensive: Si el entrenamiento es intensivo en datos.
            eval_required: Si se requiere entorno de evaluación.
            
        Returns:
            HeterogeneousWorkerConfig con distribución de workers.
        """
        resources = self.update_resources()
        
        # Calcular workers base
        total_workers = min(resources.cpu_count_physical, 8)  # Máximo 8 workers
        
        if data_intensive:
            # Más workers para datos, menos para entrenamiento
            data_workers = max(1, total_workers // 2)
            training_workers = max(1, total_workers - data_workers - (1 if eval_required else 0))
            eval_workers = 1 if eval_required else 0
        else:
            # Más workers para entrenamiento
            training_workers = max(1, total_workers - (1 if eval_required else 0))
            data_workers = 1
            eval_workers = 1 if eval_required else 0
        
        # Ajustar memoria por tipo de worker
        memory_per_data_worker = 2048 if data_intensive else 1024
        memory_per_training_worker = 1024
        memory_per_eval_worker = 512
        
        return HeterogeneousWorkerConfig(
            data_workers=data_workers,
            training_workers=training_workers,
            eval_workers=eval_workers,
            memory_per_data_worker_mb=memory_per_data_worker,
            memory_per_training_worker_mb=memory_per_training_worker,
            memory_per_eval_worker_mb=memory_per_eval_worker
        )
    
    def get_ppo_optimized_config(self, 
                                total_timesteps: int,
                                observation_space_size: int,
                                action_space_size: int) -> Dict[str, Any]:
        """
        Genera configuración PPO optimizada basada en recursos del sistema.
        
        Args:
            total_timesteps: Número total de timesteps de entrenamiento.
            observation_space_size: Tamaño del espacio de observación.
            action_space_size: Tamaño del espacio de acción.
            
        Returns:
            Dict con configuración PPO optimizada.
        """
        resources = self.update_resources()
        recommendation = self.get_optimal_workers()
        
        # Calcular n_steps basado en recursos
        n_steps = min(2048, max(512, recommendation.recommended_batch_size * 2))
        
        # Ajustar batch_size
        batch_size = recommendation.recommended_batch_size
        
        # Calcular n_epochs basado en complejidad
        complexity_factor = (observation_space_size + action_space_size) / 1000
        n_epochs = max(4, min(20, int(10 * complexity_factor)))
        
        # Learning rate basado en recursos
        if resources.gpu_available:
            learning_rate = 3e-4
        else:
            learning_rate = 1e-4
        
        # Ajustar por memoria
        if resources.memory_total_gb < 8:
            batch_size = max(16, batch_size // 2)
            n_steps = max(256, n_steps // 2)
        
        return {
            "n_steps": n_steps,
            "batch_size": batch_size,
            "n_epochs": n_epochs,
            "learning_rate": learning_rate,
            "n_envs": recommendation.optimal_workers,
            "device": "cuda" if resources.gpu_available else "cpu",
            "verbose": 1
        }

def get_optimal_worker_config(target_workers: Optional[int] = None,
                            memory_per_worker_mb: int = 1024,
                            n_steps: Optional[int] = None,
                            ppo_config: Optional[Dict[str, Any]] = None) -> WorkerRecommendation:
    """Función de utilidad para obtener configuración óptima de workers."""
    manager = WorkerManager()
    return manager.get_optimal_workers(
        target_workers=target_workers,
        memory_per_worker_mb=memory_per_worker_mb,
        n_steps=n_steps,
        ppo_config=ppo_config
    )

def print_worker_recommendation(recommendation: WorkerRecommendation, detailed: bool = True):
    """Imprime recomendación de workers con formato mejorado."""
    print("🔧 WORKER MANAGER - Configuración Recomendada")
    print("=" * 60)
    print(f"Workers óptimos: {recommendation.optimal_workers}")
    print(f"Máximo workers seguros: {recommendation.max_safe_workers}")
    print(f"Batch size recomendado: {recommendation.recommended_batch_size}")
    print(f"Memoria por worker: {recommendation.memory_per_worker_mb}MB")
    print(f"Performance estimada: {recommendation.performance_estimate}")
    
    if detailed and recommendation.estimated_memory_usage_gb:
        print(f"Uso de memoria estimado: {recommendation.estimated_memory_usage_gb:.2f}GB")
    
    if detailed and recommendation.estimated_training_time_hours:
        print(f"Tiempo de entrenamiento estimado: {recommendation.estimated_training_time_hours:.2f} horas")
    
    if detailed and recommendation.resource_utilization:
        util = recommendation.resource_utilization
        print(f"Utilización de recursos:")
        print(f"  - CPU: {util.get('cpu_utilization', 0):.1f}%")
        print(f"  - Memoria: {util.get('memory_utilization', 0):.1f}%")
        if util.get('gpu_utilization', 0) > 0:
            print(f"  - GPU: {util['gpu_utilization']:.1f}%")
    
    if recommendation.gpu_recommendation:
        print(f"Recomendación GPU: {recommendation.gpu_recommendation}")
    
    if recommendation.warning_message:
        print(f"⚠️  Advertencias: {recommendation.warning_message}")
    
    print("=" * 60)

def create_worker_manager_with_logging(log_dir: str = "logs", 
                                     safety_margin: float = 0.2,
                                     enable_monitoring: bool = True) -> WorkerManager:
    """Crea un WorkerManager con logging configurado."""
    # Configurar logging estructurado
    import structlog
    try:
        structlog.configure(
            processors=[
                structlog.stdlib.filter_by_level,
                structlog.stdlib.add_logger_name,
                structlog.stdlib.add_log_level,
                structlog.stdlib.PositionalArgumentsFormatter(),
                structlog.processors.TimeStamper(fmt="iso"),
                structlog.processors.StackInfoRenderer(),
                structlog.processors.format_exc_info,
                structlog.processors.UnicodeDecoder(),
                structlog.processors.JSONRenderer()
            ],
            context_class=dict,
            logger_factory=structlog.stdlib.LoggerFactory(),
            wrapper_class=structlog.stdlib.BoundLogger,
            cache_logger_on_first_use=True,
        )
        logger.info("Logging estructurado configurado")
    except ImportError:
        logger.warning("structlog no disponible, usando logging estándar")
    
    return WorkerManager(
        safety_margin=safety_margin,
        log_dir=log_dir,
        enable_monitoring=enable_monitoring
    )