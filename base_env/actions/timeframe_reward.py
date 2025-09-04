# base_env/actions/timeframe_reward.py
"""
Sistema de rewards por Timeframe.
Recompensa al bot basado en el timeframe usado para la ejecución.
"""

from __future__ import annotations
from typing import Dict, Any, List, Tuple


class TimeframeReward:
    """Sistema de rewards por Timeframe"""
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Inicializa el sistema de rewards por Timeframe
        
        Args:
            config: Configuración del sistema (opcional)
        """
        self.config = config or {}
        self.tf_config = self.config.get("timeframe_config", {})
        self.tf_rewards = self.tf_config.get("timeframe_rewards", {})
    
    def calculate_timeframe_reward(self, timeframe_used: str) -> Tuple[float, Dict[str, float]]:
        """
        Calcula bonus basado en el timeframe usado
        
        Args:
            timeframe_used: Timeframe usado para la ejecución
            
        Returns:
            Tupla (reward_total, componentes_detallados)
        """
        reward_components = {}
        total_reward = 0.0
        
        if self.tf_rewards.get("enabled", False):
            tf_reward = self.tf_rewards.get(timeframe_used, 0.0)
            if tf_reward != 0:
                total_reward += tf_reward
                reward_components["timeframe_reward"] = tf_reward
                reward_components["timeframe_used"] = timeframe_used
        
        return total_reward, reward_components
    
    def get_available_execution_timeframes(self) -> List[str]:
        """Devuelve los timeframes disponibles para ejecución"""
        return self.tf_config.get("execution_tfs", ["1m", "5m"])
    
    def get_trend_timeframes(self) -> List[str]:
        """Devuelve los timeframes para análisis de tendencia"""
        return self.tf_config.get("trend_tfs", ["1h", "4h"])
    
    def get_confirmation_timeframes(self) -> List[str]:
        """Devuelve los timeframes para confirmación"""
        return self.tf_config.get("confirmation_tfs", ["15m"])
    
    def reset(self):
        """Resetea el sistema (no hay estado interno que resetear)"""
        pass
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Obtiene estadísticas del sistema
        
        Returns:
            Diccionario con estadísticas
        """
        return {
            "tf_config": self.tf_config,
            "tf_rewards": self.tf_rewards,
            "execution_tfs": self.get_available_execution_timeframes(),
            "trend_tfs": self.get_trend_timeframes(),
            "confirmation_tfs": self.get_confirmation_timeframes()
        }
