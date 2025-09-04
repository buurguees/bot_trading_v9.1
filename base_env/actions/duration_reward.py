# base_env/actions/duration_reward.py
"""
Sistema de rewards por Duración del Trade.
Recompensa al bot basado en cuánto tiempo mantiene las posiciones.
"""

from __future__ import annotations
from typing import Dict, Any, List, Tuple


class DurationReward:
    """Sistema de rewards por Duración del Trade"""
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Inicializa el sistema de rewards por Duración
        
        Args:
            config: Configuración del sistema (opcional)
        """
        self.config = config or {}
        self.duration_config = self.config.get("duration_rewards", {})
    
    def calculate_duration_reward(self, bars_held: int) -> Tuple[float, Dict[str, float]]:
        """
        Calcula reward basado en la duración del trade (bars_held)
        
        Args:
            bars_held: Número de barras que se mantuvo la posición
            
        Returns:
            Tupla (reward_total, componentes_detallados)
        """
        reward_components = {}
        total_reward = 0.0
        
        if not self.duration_config.get("enabled", False):
            return total_reward, reward_components
        
        # Penalización dura por trades de 0 barras (scalping excesivo)
        if bars_held == 0:
            penalty = self.duration_config.get("zero_bars_penalty", -2.0)
            total_reward += penalty
            reward_components["zero_bars_penalty"] = penalty
        
        # Sin reward/penalty por trades de 1 barra (neutral)
        elif bars_held == 1:
            reward = self.duration_config.get("one_bar_reward", 0.0)
            total_reward += reward
            reward_components["one_bar_reward"] = reward
        
        # Rewards progresivos por trades de 2+ barras
        elif bars_held >= 2:
            reward = self.duration_config.get("two_plus_bars_reward", 0.3)
            total_reward += reward
            reward_components["two_plus_bars_reward"] = reward
            
            # Bonus adicional por trades de larga duración
            long_duration_config = self.duration_config.get("long_duration_bonus", {})
            if long_duration_config.get("enabled", False):
                min_bars = long_duration_config.get("min_bars", 10)
                if bars_held >= min_bars:
                    bonus_per_bar = long_duration_config.get("bonus_per_bar", 0.05)
                    max_bonus = long_duration_config.get("max_bonus", 1.0)
                    
                    # Calcular bonus por barras extra
                    extra_bars = bars_held - min_bars
                    long_duration_bonus = min(extra_bars * bonus_per_bar, max_bonus)
                    total_reward += long_duration_bonus
                    reward_components["long_duration_bonus"] = long_duration_bonus
        
        return total_reward, reward_components
    
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
            "duration_config": self.duration_config,
            "enabled": self.duration_config.get("enabled", False)
        }
