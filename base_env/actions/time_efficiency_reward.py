# base_env/actions/time_efficiency_reward.py
"""
Revisa que todos los archivos del proyecto tengan bien las importaciones y exportaciones y todo tenga un buen flujo de trabajo y, a la que tengas todo OK, ejecuta un entrenamiento de 100000 steps para comprobar que tenemos buenos archivos, entrena co0rrectamente sin congelaciones, sin runs cortos, con un buen analisis de historico y que recorra los historicos al completo de forma cronologia, que las alineaciones sean correctas, etc. Vamos, los ajustes finos para dejar esto listo para 50M steps de trainSistema de rewards por eficiencia temporal optimizado con potential shaping.

Mejoras implementadas:
- Procesamiento vectorizado para entornos paralelos
- Potential-based shaping para dense guidance
- Decay adaptativo para explotación tardía
- Integración con volatilidad para bonus contextual
- Caching inteligente de configuraciones
- Profiling de rendimiento
"""

from __future__ import annotations
import time
import logging
import numpy as np
from typing import Dict, Any, List, Tuple, Optional, Union
from .rewards_utils import (
    get_config_cache, get_profiler, get_batch_processor,
    profile_reward_calculation, get_vectorized_calculator, get_reward_validator
)
from .rewards_optimizer import get_global_optimizer

logger = logging.getLogger(__name__)

class TimeEfficiencyReward:
    """
    Sistema de rewards por eficiencia temporal optimizado.
    
    Este módulo incentiva exits oportunas y evita over-holding usando
    potential-based shaping y decay adaptativo para 50M steps.
    
    Optimizaciones:
    - Procesamiento vectorizado para entornos paralelos
    - Potential shaping para dense guidance
    - Decay adaptativo basado en progreso del entrenamiento
    - Integración con volatilidad para bonus contextual
    - Caching de configuraciones para eficiencia
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Inicializa el sistema de rewards por eficiencia temporal optimizado.
        
        Args:
            config: Configuración del sistema desde rewards.yaml
        """
        self.config = config
        self.cache = get_config_cache()
        self.profiler = get_profiler()
        self.batch_processor = get_batch_processor()
        self.optimizer = get_global_optimizer()
        self.vectorized_calc = get_vectorized_calculator()
        
        # Cargar configuración con caching
        self._load_config()
        
        # Estado para potential shaping
        self.episode_step = 0
        self.total_episode_steps = 0
        self.efficiency_history = []
        self.volatility_history = []
        
        # Estadísticas de rendimiento
        self.total_rewards_given = 0.0
        self.total_trades_processed = 0
        self.avg_efficiency = 0.0
        
        logger.debug(f"TimeEfficiencyReward inicializado - enabled: {self.enabled}")

    def _load_config(self) -> None:
        """Carga configuración usando utilidades centralizadas."""
        from .rewards_utils import load_config_with_cache, get_config_value
        
        # Cargar configuración con cache
        eff_config = load_config_with_cache(
            self.config.get("time_efficiency", {}), 
            "time_efficiency"
        )
        
        # Aplicar configuración usando utilidades centralizadas
        self.enabled = get_config_value(eff_config, "enabled", True, "time_efficiency")
        self.weight = get_config_value(eff_config, "weight", 0.1, "time_efficiency")
        self.per_bar_cap = get_config_value(eff_config, "per_bar_cap", 0.03, "time_efficiency")
        self.per_trade_cap = get_config_value(eff_config, "per_trade_cap", 0.3, "time_efficiency")
        self.potential_shaping = get_config_value(eff_config, "potential_shaping", True, "time_efficiency")
        self.decay_enabled = get_config_value(eff_config, "decay_enabled", True, "time_efficiency")
        self.decay_rate = get_config_value(eff_config, "decay_rate", 0.99, "time_efficiency")
        self.volatility_bonus = get_config_value(eff_config, "volatility_bonus", True, "time_efficiency")
        self.volatility_threshold = get_config_value(eff_config, "volatility_threshold", 0.02, "time_efficiency")

    @profile_reward_calculation("time_efficiency_calculate")
    def calculate_time_efficiency_reward(self, realized_pnl: float, bars_held: int, 
                                       notional: float, obs: Optional[Dict[str, Any]] = None) -> Tuple[float, Dict[str, float]]:
        """
        Calcula reward por eficiencia temporal con optimizaciones.
        
        Args:
            realized_pnl: PnL realizado del trade
            bars_held: Número de barras que se mantuvo el trade
            notional: Valor nocional del trade
            obs: Observación del entorno (opcional, para volatilidad)
            
        Returns:
            Tupla (reward, componentes_detallados)
        """
        start_time = time.time()
        
        try:
            reward_components = {}
            total_reward = 0.0
            
            if not self.enabled or bars_held <= 0 or notional <= 0:
                return total_reward, reward_components
            
            # Calcular eficiencia base
            efficiency = realized_pnl / max(1, bars_held)
            normalized_efficiency = efficiency / notional
            
            # Aplicar potential-based shaping si está habilitado
            if self.potential_shaping:
                potential_reward = self._calculate_potential_shaping(bars_held, normalized_efficiency)
                total_reward += potential_reward
                reward_components["potential_shaping"] = potential_reward
            
            # Calcular reward base
            raw_reward = normalized_efficiency * self.weight
            
            # Aplicar decay adaptativo si está habilitado
            if self.decay_enabled:
                decay_factor = self._calculate_decay_factor()
                raw_reward *= decay_factor
                reward_components["decay_factor"] = decay_factor
            
            # Aplicar caps
            per_bar_capped = np.clip(raw_reward, -self.per_bar_cap, self.per_bar_cap)
            total_reward += np.clip(per_bar_capped, -self.per_trade_cap, self.per_trade_cap)
            
            # Bonus por volatilidad si está habilitado
            if self.volatility_bonus and obs is not None:
                volatility_bonus = self._calculate_volatility_bonus(obs, normalized_efficiency)
                total_reward += volatility_bonus
                reward_components["volatility_bonus"] = volatility_bonus
            
            # Aplicar optimizaciones globales
            total_reward = self.optimizer.process_reward(
                total_reward, "time_efficiency", time.time() - start_time
            )
            
            # Actualizar estadísticas
            self._update_stats(total_reward, efficiency, normalized_efficiency)
            
            # Preparar componentes detallados
            reward_components.update({
                "time_efficiency": total_reward,
                "efficiency": efficiency,
                "normalized_efficiency": normalized_efficiency,
                "bars_held": bars_held,
                "raw_reward": raw_reward,
                "per_bar_capped": per_bar_capped
            })
            
            return total_reward, reward_components
            
        except Exception as e:
            logger.error(f"Error calculando time efficiency reward: {e}")
            return 0.0, {}

    def _calculate_potential_shaping(self, bars_held: int, normalized_efficiency: float) -> float:
        """
        Calcula potential-based shaping para dense guidance.
        
        Basado en la literatura: F(s) = -log(bars_held) para promover exits tempranos
        cuando la eficiencia es alta.
        """
        if not self.potential_shaping:
            return 0.0
        
        # Potential function: F(s) = -log(bars_held) * efficiency_factor
        efficiency_factor = min(1.0, max(0.0, normalized_efficiency * 100))  # Escalar eficiencia
        potential = -np.log(max(1, bars_held)) * efficiency_factor
        
        # Normalizar potential
        potential_weight = 0.01  # Peso pequeño para no dominar
        return potential * potential_weight

    def _calculate_decay_factor(self) -> float:
        """
        Calcula factor de decay adaptativo basado en progreso del entrenamiento.
        
        Para 50M steps, reduce el peso de la eficiencia temporal en etapas tardías
        para priorizar explotación sobre exploración.
        """
        if not self.decay_enabled:
            return 1.0
        
        # Obtener progreso del entrenamiento desde el optimizador global
        total_steps = self.optimizer.total_steps
        decay_start = 10_000_000  # 10M steps
        decay_end = 50_000_000    # 50M steps
        
        if total_steps < decay_start:
            return 1.0
        
        if total_steps > decay_end:
            return 0.1  # Decay completo
        
        # Decay exponencial entre start y end
        progress = (total_steps - decay_start) / (decay_end - decay_start)
        decay_factor = self.decay_rate ** progress
        
        return max(0.1, decay_factor)

    def _calculate_volatility_bonus(self, obs: Dict[str, Any], normalized_efficiency: float) -> float:
        """
        Calcula bonus por eficiencia en condiciones de baja volatilidad.
        
        Basado en la literatura: recompensar eficiencia cuando el mercado es estable.
        """
        if not self.volatility_bonus:
            return 0.0
        
        try:
            # Extraer volatilidad de la observación
            volatility = self._extract_volatility(obs)
            if volatility is None or volatility <= 0:
                return 0.0
            
            # Bonus si eficiencia alta en baja volatilidad
            if volatility < self.volatility_threshold and normalized_efficiency > 0:
                bonus = normalized_efficiency * 0.1  # Bonus del 10% de la eficiencia
                return min(bonus, 0.05)  # Cap del bonus
            
            return 0.0
            
        except Exception as e:
            logger.debug(f"Error calculando volatility bonus: {e}")
            return 0.0

    def _extract_volatility(self, obs: Dict[str, Any]) -> Optional[float]:
        """Extrae volatilidad de la observación."""
        try:
            # Buscar ATR en diferentes ubicaciones posibles
            atr = obs.get("atr")
            if atr is not None:
                return float(atr)
            
            # Buscar en indicadores técnicos
            indicators = obs.get("indicators", {})
            atr = indicators.get("atr")
            if atr is not None:
                return float(atr)
            
            # Buscar en volatilidad directa
            volatility = obs.get("volatility")
            if volatility is not None:
                return float(volatility)
            
            return None
            
        except Exception:
            return None

    def _update_stats(self, reward: float, efficiency: float, normalized_efficiency: float) -> None:
        """Actualiza estadísticas internas."""
        self.total_rewards_given += reward
        self.total_trades_processed += 1
        self.efficiency_history.append(normalized_efficiency)
        
        # Mantener historial limitado
        if len(self.efficiency_history) > 1000:
            self.efficiency_history = self.efficiency_history[-1000:]
        
        # Actualizar promedio de eficiencia
        self.avg_efficiency = np.mean(self.efficiency_history)

    def calculate_time_efficiency_reward_batch(self, pnl_batch: List[float], 
                                             bars_batch: List[int], 
                                             notional_batch: List[float],
                                             obs_batch: Optional[List[Dict[str, Any]]] = None) -> List[Tuple[float, Dict[str, float]]]:
        """
        Calcula rewards de eficiencia temporal para múltiples trades en lote.
        
        Args:
            pnl_batch: Lista de PnL realizados
            bars_batch: Lista de barras mantenidas
            notional_batch: Lista de valores nocionales
            obs_batch: Lista de observaciones (opcional)
            
        Returns:
            Lista de tuplas (reward, componentes_detallados)
        """
        if not self.batch_processor.should_use_batch(len(pnl_batch)):
            return [self.calculate_time_efficiency_reward(pnl, bars, notional, obs) 
                   for pnl, bars, notional, obs in zip(pnl_batch, bars_batch, notional_batch, obs_batch or [None] * len(pnl_batch))]
        
        try:
            # Convertir a arrays para procesamiento vectorizado
            pnl_array = np.array(pnl_batch)
            bars_array = np.array(bars_batch)
            notional_array = np.array(notional_batch)
            
            # Calcular eficiencias vectorizadas
            efficiency_array = pnl_array / np.maximum(1, bars_array)
            normalized_efficiency_array = efficiency_array / np.maximum(notional_array, 1e-8)
            
            # Aplicar decay factor
            decay_factor = self._calculate_decay_factor()
            
            # Calcular rewards vectorizados
            raw_rewards = normalized_efficiency_array * self.weight * decay_factor
            
            # Aplicar caps vectorizados
            per_bar_capped = np.clip(raw_rewards, -self.per_bar_cap, self.per_bar_cap)
            total_rewards = np.clip(per_bar_capped, -self.per_trade_cap, self.per_trade_cap)
            
            # Aplicar potential shaping si está habilitado
            if self.potential_shaping:
                efficiency_factors = np.minimum(1.0, np.maximum(0.0, normalized_efficiency_array * 100))
                potentials = -np.log(np.maximum(1, bars_array)) * efficiency_factors * 0.01
                total_rewards += potentials
            
            # Procesar con optimizador global
            total_rewards = self.optimizer.batch_process_rewards(total_rewards.tolist(), "time_efficiency")
            
            # Crear resultados
            results = []
            for i, (reward, pnl, bars, notional) in enumerate(zip(total_rewards, pnl_batch, bars_batch, notional_batch)):
                components = {
                    "time_efficiency": reward,
                    "efficiency": pnl / max(1, bars),
                    "normalized_efficiency": (pnl / max(1, bars)) / notional,
                    "bars_held": bars,
                    "decay_factor": decay_factor
                }
                results.append((reward, components))
            
            return results
            
        except Exception as e:
            logger.error(f"Error en batch processing: {e}")
            return [(0.0, {}) for _ in pnl_batch]

    def reset(self) -> None:
        """Resetea el sistema para un nuevo episodio."""
        self.episode_step = 0
        self.efficiency_history.clear()
        self.volatility_history.clear()
        
        logger.debug("TimeEfficiencyReward reseteado para nuevo episodio")

    def get_stats(self) -> Dict[str, Any]:
        """
        Obtiene estadísticas detalladas del sistema.
        
        Returns:
            Diccionario con estadísticas completas
        """
        return {
            "enabled": self.enabled,
            "weight": self.weight,
            "per_bar_cap": self.per_bar_cap,
            "per_trade_cap": self.per_trade_cap,
            "potential_shaping": self.potential_shaping,
            "decay_enabled": self.decay_enabled,
            "decay_rate": self.decay_rate,
            "volatility_bonus": self.volatility_bonus,
            "total_rewards_given": self.total_rewards_given,
            "total_trades_processed": self.total_trades_processed,
            "avg_efficiency": self.avg_efficiency,
            "efficiency_history_size": len(self.efficiency_history)
        }

    def get_performance_stats(self) -> Dict[str, Any]:
        """Obtiene estadísticas de rendimiento del módulo."""
        return self.profiler.get_stats("time_efficiency_calculate")

    def get_efficiency_distribution(self) -> Dict[str, float]:
        """Obtiene distribución de eficiencias."""
        if not self.efficiency_history:
            return {}
        
        efficiency_array = np.array(self.efficiency_history)
        return {
            "mean": np.mean(efficiency_array),
            "std": np.std(efficiency_array),
            "min": np.min(efficiency_array),
            "max": np.max(efficiency_array),
            "median": np.median(efficiency_array),
            "q25": np.percentile(efficiency_array, 25),
            "q75": np.percentile(efficiency_array, 75)
        }
