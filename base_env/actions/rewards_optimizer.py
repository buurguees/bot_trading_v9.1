# base_env/actions/rewards_optimizer.py
"""
Optimizador común para el sistema de rewards/penalties.

Mejoras implementadas:
- Procesamiento por lotes para entornos vectorizados
- Clipping dinámico y normalización global
- Caching inteligente de configuraciones
- Profiling de rendimiento
- Decay schedules globales
- Integración con curriculum learning
"""

from __future__ import annotations
import time
import logging
import numpy as np
from typing import Dict, Any, List, Optional, Callable, Union
from dataclasses import dataclass, field
from collections import defaultdict, deque
import threading
from pathlib import Path
import yaml
import json

logger = logging.getLogger(__name__)

@dataclass
class RewardStats:
    """Estadísticas de rendimiento de rewards."""
    total_calls: int = 0
    total_time: float = 0.0
    avg_time_per_call: float = 0.0
    min_reward: float = float('inf')
    max_reward: float = float('-inf')
    reward_std: float = 0.0
    reward_history: deque = field(default_factory=lambda: deque(maxlen=1000))
    
    def update(self, reward: float, call_time: float) -> None:
        """Actualiza estadísticas con nueva llamada."""
        self.total_calls += 1
        self.total_time += call_time
        self.avg_time_per_call = self.total_time / self.total_calls
        self.min_reward = min(self.min_reward, reward)
        self.max_reward = max(self.max_reward, reward)
        self.reward_history.append(reward)
        
        if len(self.reward_history) > 1:
            self.reward_std = np.std(list(self.reward_history))

@dataclass
class GlobalRewardConfig:
    """Configuración global para el sistema de rewards."""
    # Clipping dinámico
    initial_clip_range: tuple[float, float] = (-10.0, 10.0)
    adaptive_clipping: bool = True
    clip_std_multiplier: float = 3.0
    
    # Normalización
    enable_normalization: bool = True
    normalization_window: int = 1000
    target_reward_std: float = 1.0
    
    # Decay schedules
    global_decay_start: int = 10_000_000  # 10M steps
    global_decay_end: int = 50_000_000    # 50M steps
    exploration_decay_rate: float = 0.1
    
    # Profiling
    enable_profiling: bool = False
    profile_interval: int = 100_000  # Cada 100K steps
    
    # Curriculum learning
    enable_curriculum: bool = True
    curriculum_stages: List[Dict[str, Any]] = field(default_factory=list)
    
    # Batch processing
    enable_batch_processing: bool = True
    batch_size_threshold: int = 4  # Mínimo para usar batch processing

class RewardOptimizer:
    """Optimizador principal para el sistema de rewards."""
    
    def __init__(self, config: Optional[GlobalRewardConfig] = None):
        self.config = config or GlobalRewardConfig()
        self.stats: Dict[str, RewardStats] = defaultdict(RewardStats)
        self.reward_history: deque = deque(maxlen=self.config.normalization_window)
        self.current_stage: int = 0
        self.total_steps: int = 0
        self._lock = threading.RLock()
        
        # Cache para configuraciones
        self._config_cache: Dict[str, Any] = {}
        self._cache_timestamps: Dict[str, float] = {}
        
        logger.info("RewardOptimizer inicializado")
    
    def update_total_steps(self, steps: int) -> None:
        """Actualiza el contador total de steps."""
        with self._lock:
            self.total_steps = steps
            self._update_curriculum_stage()
    
    def _update_curriculum_stage(self) -> None:
        """Actualiza la etapa del curriculum learning."""
        if not self.config.enable_curriculum or not self.config.curriculum_stages:
            return
        
        for i, stage in enumerate(self.config.curriculum_stages):
            if self.total_steps >= stage.get('start_step', 0):
                self.current_stage = i
    
    def get_curriculum_config(self, module_name: str) -> Dict[str, Any]:
        """Obtiene configuración del curriculum para un módulo."""
        if not self.config.enable_curriculum or not self.config.curriculum_stages:
            return {}
        
        if self.current_stage >= len(self.config.curriculum_stages):
            return {}
        
        stage_config = self.config.curriculum_stages[self.current_stage]
        return stage_config.get('modules', {}).get(module_name, {})
    
    def apply_global_decay(self, reward: float, module_name: str) -> float:
        """Aplica decay global basado en el progreso del entrenamiento."""
        if self.total_steps < self.config.global_decay_start:
            return reward
        
        if self.total_steps > self.config.global_decay_end:
            return reward * 0.1  # Decay completo
        
        # Decay lineal entre start y end
        progress = (self.total_steps - self.config.global_decay_start) / (
            self.config.global_decay_end - self.config.global_decay_start
        )
        decay_factor = 1.0 - progress * self.config.exploration_decay_rate
        
        # Aplicar decay específico por módulo
        if module_name == 'exploration_bonus':
            return reward * decay_factor
        elif module_name == 'inactivity_penalty':
            return reward * (1.0 + progress * 0.5)  # Aumentar penalty con el tiempo
        
        return reward
    
    def apply_dynamic_clipping(self, reward: float) -> float:
        """Aplica clipping dinámico basado en estadísticas históricas."""
        if not self.config.adaptive_clipping:
            return np.clip(reward, *self.config.initial_clip_range)
        
        with self._lock:
            if len(self.reward_history) < 10:
                return np.clip(reward, *self.config.initial_clip_range)
            
            # Calcular límites basados en estadísticas
            reward_array = np.array(list(self.reward_history))
            mean_reward = np.mean(reward_array)
            std_reward = np.std(reward_array)
            
            # Límites adaptativos
            lower_bound = mean_reward - self.config.clip_std_multiplier * std_reward
            upper_bound = mean_reward + self.config.clip_std_multiplier * std_reward
            
            # Mantener límites razonables
            lower_bound = max(lower_bound, self.config.initial_clip_range[0])
            upper_bound = min(upper_bound, self.config.initial_clip_range[1])
            
            return np.clip(reward, lower_bound, upper_bound)
    
    def apply_normalization(self, reward: float) -> float:
        """Aplica normalización basada en ventana deslizante."""
        if not self.config.enable_normalization:
            return reward
        
        with self._lock:
            self.reward_history.append(reward)
            
            if len(self.reward_history) < 10:
                return reward
            
            # Normalización Z-score
            reward_array = np.array(list(self.reward_history))
            mean_reward = np.mean(reward_array)
            std_reward = np.std(reward_array)
            
            if std_reward < 1e-8:
                return reward
            
            # Normalizar y escalar al target std
            normalized = (reward - mean_reward) / std_reward
            return normalized * self.config.target_reward_std
    
    def process_reward(self, reward: float, module_name: str, 
                      call_time: float = 0.0) -> float:
        """Procesa un reward aplicando todas las optimizaciones."""
        start_time = time.time()
        
        try:
            # Aplicar curriculum learning
            curriculum_config = self.get_curriculum_config(module_name)
            if curriculum_config.get('enabled', True) is False:
                return 0.0
            
            # Aplicar decay global
            reward = self.apply_global_decay(reward, module_name)
            
            # Aplicar normalización
            reward = self.apply_normalization(reward)
            
            # Aplicar clipping dinámico
            reward = self.apply_dynamic_clipping(reward)
            
            # Actualizar estadísticas
            self.stats[module_name].update(reward, call_time)
            
            return reward
            
        except Exception as e:
            logger.error(f"Error procesando reward en {module_name}: {e}")
            return 0.0
        finally:
            # Profiling si está habilitado
            if self.config.enable_profiling and self.total_steps % self.config.profile_interval == 0:
                self._log_profiling_stats()
    
    def batch_process_rewards(self, rewards: List[float], module_name: str) -> List[float]:
        """Procesa múltiples rewards en lote para entornos vectorizados."""
        if not self.config.enable_batch_processing or len(rewards) < self.config.batch_size_threshold:
            return [self.process_reward(r, module_name) for r in rewards]
        
        try:
            # Convertir a numpy para operaciones vectorizadas
            reward_array = np.array(rewards)
            
            # Aplicar decay global (vectorizado)
            if self.total_steps >= self.config.global_decay_start:
                progress = min(1.0, (self.total_steps - self.config.global_decay_start) / 
                             (self.config.global_decay_end - self.config.global_decay_start))
                decay_factor = 1.0 - progress * self.config.exploration_decay_rate
                if module_name == 'exploration_bonus':
                    reward_array *= decay_factor
            
            # Aplicar normalización (vectorizada)
            if self.config.enable_normalization and len(self.reward_history) >= 10:
                with self._lock:
                    reward_array = np.array(list(self.reward_history))
                    mean_reward = np.mean(reward_array)
                    std_reward = np.std(reward_array)
                    if std_reward > 1e-8:
                        normalized = (reward_array - mean_reward) / std_reward
                        reward_array = normalized * self.config.target_reward_std
            
            # Aplicar clipping (vectorizado)
            reward_array = np.clip(reward_array, *self.config.initial_clip_range)
            
            # Actualizar estadísticas
            for reward in rewards:
                self.stats[module_name].update(reward, 0.0)
            
            return reward_array.tolist()
            
        except Exception as e:
            logger.error(f"Error en batch processing para {module_name}: {e}")
            return [self.process_reward(r, module_name) for r in rewards]
    
    def _log_profiling_stats(self) -> None:
        """Registra estadísticas de profiling."""
        logger.info("=== REWARD OPTIMIZER PROFILING ===")
        for module_name, stats in self.stats.items():
            if stats.total_calls > 0:
                logger.info(f"{module_name}: {stats.total_calls} calls, "
                          f"avg {stats.avg_time_per_call*1000:.2f}ms, "
                          f"reward range [{stats.min_reward:.3f}, {stats.max_reward:.3f}]")
    
    def get_module_stats(self, module_name: str) -> Optional[RewardStats]:
        """Obtiene estadísticas de un módulo específico."""
        return self.stats.get(module_name)
    
    def get_global_stats(self) -> Dict[str, Any]:
        """Obtiene estadísticas globales del sistema."""
        with self._lock:
            total_calls = sum(stats.total_calls for stats in self.stats.values())
            total_time = sum(stats.total_time for stats in self.stats.values())
            
            return {
                'total_calls': total_calls,
                'total_time': total_time,
                'avg_time_per_call': total_time / max(total_calls, 1),
                'total_steps': self.total_steps,
                'current_stage': self.current_stage,
                'modules_count': len(self.stats),
                'reward_history_size': len(self.reward_history)
            }
    
    def reset_episode(self) -> None:
        """Resetea el estado del optimizador para un nuevo episodio."""
        with self._lock:
            # Mantener estadísticas globales pero limpiar historial
            self.reward_history.clear()
            
            # Resetear estadísticas de módulos si es necesario
            for stats in self.stats.values():
                stats.reward_history.clear()
    
    def load_config_from_file(self, config_path: str) -> None:
        """Carga configuración desde archivo YAML."""
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config_data = yaml.safe_load(f)
            
            # Actualizar configuración
            if 'global_reward_config' in config_data:
                global_config = config_data['global_reward_config']
                for key, value in global_config.items():
                    if hasattr(self.config, key):
                        setattr(self.config, key, value)
            
            logger.info(f"Configuración cargada desde {config_path}")
            
        except Exception as e:
            logger.error(f"Error cargando configuración desde {config_path}: {e}")
    
    def save_stats_to_file(self, output_path: str) -> None:
        """Guarda estadísticas a archivo JSON."""
        try:
            stats_data = {
                'global_stats': self.get_global_stats(),
                'module_stats': {
                    name: {
                        'total_calls': stats.total_calls,
                        'total_time': stats.total_time,
                        'avg_time_per_call': stats.avg_time_per_call,
                        'min_reward': stats.min_reward,
                        'max_reward': stats.max_reward,
                        'reward_std': stats.reward_std
                    }
                    for name, stats in self.stats.items()
                }
            }
            
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(stats_data, f, indent=2)
            
            logger.info(f"Estadísticas guardadas en {output_path}")
            
        except Exception as e:
            logger.error(f"Error guardando estadísticas en {output_path}: {e}")

# Instancia global del optimizador
_global_optimizer: Optional[RewardOptimizer] = None

def get_global_optimizer() -> RewardOptimizer:
    """Obtiene la instancia global del optimizador."""
    global _global_optimizer
    if _global_optimizer is None:
        _global_optimizer = RewardOptimizer()
    return _global_optimizer

def set_global_optimizer(optimizer: RewardOptimizer) -> None:
    """Establece la instancia global del optimizador."""
    global _global_optimizer
    _global_optimizer = optimizer

def process_reward_optimized(reward: float, module_name: str, 
                           call_time: float = 0.0) -> float:
    """Función de conveniencia para procesar rewards."""
    return get_global_optimizer().process_reward(reward, module_name, call_time)

def batch_process_rewards_optimized(rewards: List[float], module_name: str) -> List[float]:
    """Función de conveniencia para procesar rewards en lote."""
    return get_global_optimizer().batch_process_rewards(rewards, module_name)
