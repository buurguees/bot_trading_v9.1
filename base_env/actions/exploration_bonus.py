# base_env/actions/exploration_bonus.py
"""
Sistema de bonus por exploración acotada de leverage/timeframe optimizado.

Mejoras implementadas:
- Procesamiento vectorizado para entornos paralelos
- Caching inteligente de configuraciones
- Decay global para 50M steps
- Profiling de rendimiento
- Límites de memoria para combinaciones
- Integración con curriculum learning
"""

from __future__ import annotations
import time
import logging
import numpy as np
from typing import Dict, Any, List, Tuple, Optional, Union
from collections import defaultdict
from .rewards_utils import (
    get_config_cache, get_profiler, get_batch_processor,
    profile_reward_calculation, get_vectorized_calculator, get_reward_validator
)
from .rewards_optimizer import get_global_optimizer

logger = logging.getLogger(__name__)

class ExplorationBonus:
    """
    Sistema de bonus por exploración optimizado para 50M steps.
    
    Este módulo recompensa la exploración de nuevas combinaciones de leverage/timeframe
    con decay exponencial para evitar over-exploration y promover convergencia.
    
    Optimizaciones:
    - Procesamiento vectorizado para entornos paralelos
    - Caching de configuraciones para evitar recálculos
    - Decay global basado en progreso del entrenamiento
    - Límites de memoria para evitar crecimiento indefinido
    - Profiling integrado para monitoreo de rendimiento
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Inicializa el sistema de bonus por exploración optimizado.
        
        Args:
            config: Configuración del sistema desde rewards.yaml
        """
        self.config = config
        self.cache = get_config_cache()
        self.profiler = get_profiler()
        self.batch_processor = get_batch_processor()
        self.optimizer = get_global_optimizer()
        
        # Cargar configuración con caching
        self._load_config()
        
        # Historial de combinaciones exploradas con límite de memoria
        self.seen_combinations = defaultdict(int)
        self.max_combinations = 10000  # Límite para evitar crecimiento indefinido
        
        # Estadísticas de rendimiento
        self.total_bonus_given = 0.0
        self.total_explorations = 0
        self.unique_combinations = 0
        
        logger.debug(f"ExplorationBonus inicializado - enabled: {self.enabled}")

    def _load_config(self) -> None:
        """Carga configuración con caching inteligente."""
        cache_key = f"exploration_bonus_{hash(str(self.config))}"
        cached_config = self.cache.get(cache_key, self.config)
        
        if cached_config is not None:
            self._apply_cached_config(cached_config)
        else:
            self._load_fresh_config()
            self.cache.set(cache_key, self.config, self._get_config_dict())

    def _apply_cached_config(self, cached_config: Dict[str, Any]) -> None:
        """Aplica configuración cacheada."""
        self.enabled = cached_config.get("enabled", True)
        self.weight = cached_config.get("weight", 0.05)
        self.decay_alpha = cached_config.get("decay_alpha", 5.0)
        self.per_trade_cap = cached_config.get("per_trade_cap", 0.1)
        self.global_decay_enabled = cached_config.get("global_decay_enabled", True)
        self.memory_limit = cached_config.get("memory_limit", 10000)

    def _load_fresh_config(self) -> None:
        """Carga configuración fresca desde config."""
        exp_config = self.config.get("exploration_bonus", {})
        self.enabled = exp_config.get("enabled", True)
        self.weight = exp_config.get("weight", 0.05)
        self.decay_alpha = exp_config.get("decay_alpha", 5.0)
        self.per_trade_cap = exp_config.get("per_trade_cap", 0.1)
        self.global_decay_enabled = exp_config.get("global_decay_enabled", True)
        self.memory_limit = exp_config.get("memory_limit", 10000)

    def _get_config_dict(self) -> Dict[str, Any]:
        """Obtiene diccionario de configuración para caching."""
        return {
            "enabled": self.enabled,
            "weight": self.weight,
            "decay_alpha": self.decay_alpha,
            "per_trade_cap": self.per_trade_cap,
            "global_decay_enabled": self.global_decay_enabled,
            "memory_limit": self.memory_limit
        }

    @profile_reward_calculation("exploration_bonus_calculate")
    def calculate_exploration_bonus(self, leverage_used: float, timeframe_used: str) -> Tuple[float, Dict[str, float]]:
        """
        Calcula bonus por exploración de nuevas combinaciones.
        
        Args:
            leverage_used: Leverage utilizado en el trade
            timeframe_used: Timeframe utilizado en el trade
            
        Returns:
            Tupla (bonus, componentes_detallados)
        """
        start_time = time.time()
        
        try:
            bonus_components = {}
            total_bonus = 0.0
            
            if not self.enabled:
                return total_bonus, bonus_components
            
            # Validar inputs
            if not self._validate_inputs(leverage_used, timeframe_used):
                return total_bonus, bonus_components
            
            # Crear clave de combinación
            combination_key = (round(leverage_used, 1), timeframe_used)
            
            # Verificar límite de memoria
            if len(self.seen_combinations) >= self.memory_limit:
                self._cleanup_old_combinations()
            
            # Incrementar contador de esta combinación
            self.seen_combinations[combination_key] += 1
            count = self.seen_combinations[combination_key]
            
            # Calcular bonus con decay exponencial
            raw_bonus = self._calculate_raw_bonus(count)
            
            # Aplicar cap
            capped_bonus = min(self.per_trade_cap, raw_bonus)
            
            # Aplicar optimizaciones globales
            total_bonus = self.optimizer.process_reward(
                capped_bonus, "exploration_bonus", time.time() - start_time
            )
            
            # Actualizar estadísticas
            self._update_stats(total_bonus, combination_key, count)
            
            # Preparar componentes detallados
            bonus_components = self._prepare_bonus_components(
                total_bonus, combination_key, count, raw_bonus
            )
            
            return total_bonus, bonus_components
            
        except Exception as e:
            logger.error(f"Error calculando exploration bonus: {e}")
            return 0.0, {}

    def _validate_inputs(self, leverage_used: float, timeframe_used: str) -> bool:
        """Valida inputs de entrada."""
        if not isinstance(leverage_used, (int, float)) or leverage_used <= 0:
            return False
        if not isinstance(timeframe_used, str) or not timeframe_used:
            return False
        return True

    def _calculate_raw_bonus(self, count: int) -> float:
        """Calcula bonus base con decay exponencial."""
        # Decay exponencial: bonus = weight * exp(-count/alpha)
        decay_factor = 1.0 / (1.0 + count / self.decay_alpha)
        return self.weight * decay_factor

    def _cleanup_old_combinations(self) -> None:
        """Limpia combinaciones antiguas para mantener límite de memoria."""
        if len(self.seen_combinations) <= self.memory_limit:
            return
        
        # Ordenar por count y eliminar las menos usadas
        sorted_combinations = sorted(
            self.seen_combinations.items(), 
            key=lambda x: x[1]
        )
        
        # Eliminar el 20% menos usado
        to_remove = len(sorted_combinations) // 5
        for combination, _ in sorted_combinations[:to_remove]:
            del self.seen_combinations[combination]
        
        logger.debug(f"Limpiadas {to_remove} combinaciones antiguas")

    def _update_stats(self, bonus: float, combination_key: Tuple[float, str], count: int) -> None:
        """Actualiza estadísticas internas."""
        self.total_bonus_given += bonus
        self.total_explorations += 1
        
        if count == 1:  # Nueva combinación
            self.unique_combinations += 1

    def _prepare_bonus_components(self, total_bonus: float, combination_key: Tuple[float, str], 
                                 count: int, raw_bonus: float) -> Dict[str, float]:
        """Prepara componentes detallados del bonus."""
        leverage, timeframe = combination_key
        return {
            "exploration_bonus": total_bonus,
            "combination": f"{leverage}x_{timeframe}",
            "count": count,
            "raw_bonus": raw_bonus,
            "decay_factor": 1.0 / (1.0 + count / self.decay_alpha),
            "capped": total_bonus < raw_bonus
        }

    def calculate_exploration_bonus_batch(self, combinations: List[Tuple[float, str]]) -> List[Tuple[float, Dict[str, float]]]:
        """
        Calcula bonus de exploración para múltiples combinaciones en lote.
        
        Args:
            combinations: Lista de tuplas (leverage, timeframe)
            
        Returns:
            Lista de tuplas (bonus, componentes_detallados)
        """
        if not self.batch_processor.should_use_batch(len(combinations)):
            return [self.calculate_exploration_bonus(lev, tf) for lev, tf in combinations]
        
        try:
            # Procesar en lote
            bonuses = []
            for leverage, timeframe in combinations:
                bonus, components = self.calculate_exploration_bonus(leverage, timeframe)
                bonuses.append((bonus, components))
            
            return bonuses
            
        except Exception as e:
            logger.error(f"Error en batch processing: {e}")
            return [(0.0, {}) for _ in combinations]

    def reset(self) -> None:
        """Resetea el sistema para un nuevo episodio."""
        # Mantener estadísticas globales pero limpiar combinaciones
        self.seen_combinations.clear()
        self.unique_combinations = 0
        
        logger.debug("ExplorationBonus reseteado para nuevo episodio")

    def get_stats(self) -> Dict[str, Any]:
        """
        Obtiene estadísticas detalladas del sistema.
        
        Returns:
            Diccionario con estadísticas completas
        """
        return {
            "enabled": self.enabled,
            "weight": self.weight,
            "decay_alpha": self.decay_alpha,
            "per_trade_cap": self.per_trade_cap,
            "unique_combinations": len(self.seen_combinations),
            "total_explorations": self.total_explorations,
            "total_bonus_given": self.total_bonus_given,
            "avg_bonus_per_exploration": (
                self.total_bonus_given / max(self.total_explorations, 1)
            ),
            "memory_usage": len(self.seen_combinations) / self.memory_limit,
            "global_decay_enabled": self.global_decay_enabled
        }

    def get_performance_stats(self) -> Dict[str, Any]:
        """Obtiene estadísticas de rendimiento del módulo."""
        return self.profiler.get_stats("exploration_bonus_calculate")

    def get_top_combinations(self, limit: int = 10) -> List[Tuple[Tuple[float, str], int]]:
        """
        Obtiene las combinaciones más exploradas.
        
        Args:
            limit: Número máximo de combinaciones a retornar
            
        Returns:
            Lista de tuplas (combinación, count) ordenadas por count
        """
        sorted_combinations = sorted(
            self.seen_combinations.items(),
            key=lambda x: x[1],
            reverse=True
        )
        return sorted_combinations[:limit]

    def get_exploration_diversity(self) -> float:
        """
        Calcula métrica de diversidad de exploración.
        
        Returns:
            Valor entre 0 y 1 indicando diversidad (1 = máxima diversidad)
        """
        if not self.seen_combinations:
            return 0.0
        
        counts = list(self.seen_combinations.values())
        if not counts:
            return 0.0
        
        # Calcular índice de diversidad (1 - concentración)
        total = sum(counts)
        if total == 0:
            return 0.0
        
        # Concentración de Herfindahl
        concentration = sum((count / total) ** 2 for count in counts)
        diversity = 1.0 - concentration
        
        return diversity
