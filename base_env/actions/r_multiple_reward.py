# base_env/actions/r_multiple_reward.py
"""
Sistema de rewards por R-Multiple.
Recompensa al bot basado en el R-Multiple del trade (relación riesgo/recompensa).
"""

from __future__ import annotations
from typing import Dict, Any, List, Tuple


class RMultipleReward:
    """Sistema de rewards por R-Multiple"""
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Inicializa el sistema de rewards por R-Multiple
        
        Args:
            config: Configuración del sistema (opcional)
        """
        self.config = config or {}
        self.tiers_pos = self.config.get("r_multiple_tiers", {}).get("pos", [])
        self.tiers_neg = self.config.get("r_multiple_tiers", {}).get("neg", [])
    
    def calculate_r_multiple_reward(self, r_multiple: float) -> Tuple[float, Dict[str, float]]:
        """
        Calcula el reward basado en R-Multiple del trade
        
        Args:
            r_multiple: R-Multiple del trade
            
        Returns:
            Tupla (reward_total, componentes_detallados)
        """
        reward_components = {}
        total_reward = 0.0
        
        if r_multiple >= 0:
            # R-Multiple positivo
            for min_r, max_r, reward in self.tiers_pos:
                if min_r <= r_multiple <= max_r:
                    total_reward += reward
                    reward_components["r_multiple_reward"] = reward
                    reward_components["r_multiple_tier"] = f"{min_r}-{max_r}R"
                    break
        else:
            # R-Multiple negativo
            r_abs = abs(r_multiple)
            for min_r, max_r, penalty in self.tiers_neg:
                if min_r <= r_abs <= max_r:
                    total_reward += penalty  # penalty ya es negativo
                    reward_components["r_multiple_penalty"] = penalty
                    reward_components["r_multiple_tier"] = f"-{min_r}-{max_r}R"
                    break
        
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
            "tiers_pos": self.tiers_pos,
            "tiers_neg": self.tiers_neg
        }
