# train_env/callbacks/strategy_consultant.py
"""
Callback para consultar estrategias existentes y mejorar el aprendizaje.

Mejoras implementadas:
- Logging estructurado en lugar de print
- Type hints completos
- Caching inteligente con TTL
- Validación robusta de datos
- Métricas de rendimiento
- Filtrado avanzado de estrategias
- Curriculum learning mejorado
"""

from __future__ import annotations
import json
import random
import time
from pathlib import Path
from typing import Dict, Any, List, Optional, Callable
from stable_baselines3.common.callbacks import BaseCallback
from .callback_utils import (
    CallbackLogger, validate_callback_params, safe_json_load,
    create_callback_logger, create_callback_metrics, format_timesteps,
    FileCache, create_file_cache
)

class StrategyConsultant(BaseCallback):
    """
    Callback que consulta estrategias existentes para mejorar el aprendizaje.
    
    Este callback implementa "curriculum learning" basado en estrategias exitosas
    previamente guardadas. Carga estrategias de forma eficiente con caching
    y proporciona métodos para acceder a las mejores estrategias.
    
    Args:
        strategies_file: Ruta al archivo JSON con estrategias.
        consult_every_steps: Intervalo de steps para recargar estrategias.
        verbose: Nivel de verbosidad (0: silent, 1: info, 2: debug).
        cache_ttl: TTL del cache en segundos (default: 300).
        max_strategies: Máximo número de estrategias a cargar.
        min_performance: Rendimiento mínimo para filtrar estrategias.
        strategy_filter: Función personalizada para filtrar estrategias.
        
    Example:
        >>> consultant = StrategyConsultant(
        ...     strategies_file="models/BTCUSDT/BTCUSDT_strategies.json",
        ...     consult_every_steps=50000,
        ...     verbose=1
        ... )
        >>> best_strategies = consultant.get_best_strategies(limit=5)
    """
    
    def __init__(self, 
                 strategies_file: str, 
                 consult_every_steps: int = 100000,
                 verbose: int = 0,
                 cache_ttl: float = 300.0,
                 max_strategies: int = 1000,
                 min_performance: float = 0.0,
                 strategy_filter: Optional[Callable[[Dict[str, Any]], bool]] = None):
        super().__init__(verbose)
        
        # Validar parámetros
        validate_callback_params(
            every_steps=consult_every_steps,
            verbose=verbose
        )
        
        self.strategies_file = Path(strategies_file)
        self.consult_every_steps = int(consult_every_steps)
        self.max_strategies = max(1, int(max_strategies))
        self.min_performance = float(min_performance)
        self.strategy_filter = strategy_filter
        self.last_consultation = 0
        
        # Configurar logging y métricas
        self.logger = create_callback_logger("StrategyConsultant", verbose)
        self.metrics = create_callback_metrics()
        
        # Cache para estrategias
        self.cache = create_file_cache(cache_ttl)
        self.strategies: List[Dict[str, Any]] = []
        self.strategies_metadata: Dict[str, Any] = {}
        
        self.logger.info(f"StrategyConsultant inicializado - consultando cada {format_timesteps(self.consult_every_steps)} steps")
    
    def _on_training_start(self) -> None:
        """Cargar estrategias al inicio del entrenamiento."""
        self.logger.info("Iniciando carga de estrategias")
        self._load_strategies()
        self.logger.info(f"Cargadas {len(self.strategies)} estrategias existentes")
    
    def _on_step(self) -> bool:
        """Consultar estrategias periódicamente."""
        if self.num_timesteps - self.last_consultation >= self.consult_every_steps:
            self._load_strategies()
            self.last_consultation = self.num_timesteps
            self.logger.info(f"Estrategias actualizadas: {len(self.strategies)} disponibles")
        return True
    
    def _load_strategies(self) -> None:
        """Carga las mejores estrategias del archivo JSON con caching."""
        try:
            # Verificar cache primero
            cached_data = self.cache.get(self.strategies_file)
            if cached_data is not None:
                self.strategies = cached_data.get('strategies', [])
                self.strategies_metadata = cached_data.get('metadata', {})
                self.logger.debug("Estrategias cargadas desde cache")
                return
            
            # Cargar desde archivo
            raw_data = safe_json_load(self.strategies_file, default=[])
            
            if not isinstance(raw_data, list):
                self.logger.warning("Archivo de estrategias no contiene una lista válida")
                self.strategies = []
                return
            
            # Filtrar y procesar estrategias
            self.strategies = self._filter_strategies(raw_data)
            
            # Crear metadata
            self.strategies_metadata = {
                'total_loaded': len(raw_data),
                'filtered_count': len(self.strategies),
                'load_time': time.time(),
                'file_size': self.strategies_file.stat().st_size if self.strategies_file.exists() else 0
            }
            
            # Guardar en cache
            cache_data = {
                'strategies': self.strategies,
                'metadata': self.strategies_metadata
            }
            self.cache.set(self.strategies_file, cache_data)
            
            self.metrics.record_operation()
            self.logger.debug(f"Estrategias cargadas: {len(self.strategies)}/{len(raw_data)}")
            
        except Exception as e:
            self.metrics.record_error()
            self.logger.error(f"Error cargando estrategias: {e}")
            self.strategies = []
    
    def _filter_strategies(self, raw_strategies: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Filtra estrategias según criterios de calidad."""
        filtered = []
        
        for strategy in raw_strategies:
            try:
                # Verificar estructura básica
                if not isinstance(strategy, dict):
                    continue
                
                # Aplicar filtro de rendimiento
                performance = strategy.get('performance', 0.0)
                if performance < self.min_performance:
                    continue
                
                # Aplicar filtro personalizado
                if self.strategy_filter and not self.strategy_filter(strategy):
                    continue
                
                # Verificar campos requeridos
                if not self._is_valid_strategy(strategy):
                    continue
                
                filtered.append(strategy)
                
                # Limitar número de estrategias
                if len(filtered) >= self.max_strategies:
                    break
                    
            except Exception as e:
                self.logger.debug(f"Error procesando estrategia: {e}")
                continue
        
        # Ordenar por rendimiento descendente
        filtered.sort(key=lambda x: x.get('performance', 0.0), reverse=True)
        
        return filtered
    
    def _is_valid_strategy(self, strategy: Dict[str, Any]) -> bool:
        """Valida si una estrategia tiene la estructura requerida."""
        required_fields = ['performance', 'actions', 'rewards']
        return all(field in strategy for field in required_fields)
    
    def get_best_strategies(self, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Devuelve las mejores estrategias para consulta.
        
        Args:
            limit: Número máximo de estrategias a retornar.
            
        Returns:
            Lista de las mejores estrategias ordenadas por rendimiento.
        """
        limit = max(1, min(limit, len(self.strategies)))
        return self.strategies[:limit]
    
    def get_random_strategy(self) -> Dict[str, Any]:
        """
        Devuelve una estrategia aleatoria de las mejores.
        
        Returns:
            Estrategia aleatoria o diccionario vacío si no hay estrategias.
        """
        if not self.strategies:
            return {}
        
        # Tomar de las mejores estrategias (top 20% o máximo 20)
        top_count = max(1, min(20, len(self.strategies) // 5))
        return random.choice(self.strategies[:top_count])
    
    def get_strategies_by_performance(self, min_perf: float, max_perf: float = float('inf')) -> List[Dict[str, Any]]:
        """
        Devuelve estrategias dentro de un rango de rendimiento.
        
        Args:
            min_perf: Rendimiento mínimo.
            max_perf: Rendimiento máximo.
            
        Returns:
            Lista de estrategias en el rango especificado.
        """
        return [s for s in self.strategies 
                if min_perf <= s.get('performance', 0.0) <= max_perf]
    
    def get_strategy_count(self) -> int:
        """Retorna el número total de estrategias cargadas."""
        return len(self.strategies)
    
    def get_strategies_metadata(self) -> Dict[str, Any]:
        """Retorna metadata sobre las estrategias cargadas."""
        return self.strategies_metadata.copy()
    
    def clear_cache(self) -> None:
        """Limpia el cache de estrategias."""
        self.cache.invalidate(self.strategies_file)
        self.logger.info("Cache de estrategias limpiado")
    
    def _on_training_end(self) -> None:
        """Finalización al terminar entrenamiento."""
        stats = self.metrics.get_stats()
        self.logger.info(f"Estadísticas finales - Estrategias: {len(self.strategies)}, "
                        f"Operaciones: {stats['operations']}, Errores: {stats['errors']}")
    
    def get_performance_stats(self) -> Dict[str, float]:
        """Retorna estadísticas de rendimiento de las estrategias."""
        if not self.strategies:
            return {}
        
        performances = [s.get('performance', 0.0) for s in self.strategies]
        return {
            'mean': sum(performances) / len(performances),
            'max': max(performances),
            'min': min(performances),
            'count': len(performances)
        }
