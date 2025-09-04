# base_env/actions/holding_reward.py
"""
Sistema de rewards por Holding (mantener posiciones con equity positivo).
Recompensa al bot por mantener posiciones que generan equity positivo.
"""

from __future__ import annotations
from typing import Dict, Any, List, Tuple


class HoldingReward:
    """Sistema de rewards por Holding"""
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Inicializa el sistema de rewards por Holding
        
        Args:
            config: Configuración del sistema (opcional)
        """
        self.config = config or {}
        # Configuración desde rewards.yaml
        holding_config = self.config.get("holding_rewards", {})
        self.holding_reward = holding_config.get("holding_reward", 0.1)
        self.reward_interval = holding_config.get("reward_interval", 10)  # Cada N barras
        
        # Contadores internos
        self.bars_held_with_positive_equity = 0
        self.last_equity_check = 0.0
    
    def calculate_holding_reward(self, obs: Dict[str, Any], 
                               current_equity: float, 
                               initial_balance: float) -> Tuple[float, Dict[str, float]]:
        """
        Calcula el reward por Holding
        
        Args:
            obs: Observación actual del entorno
            current_equity: Equity actual del portfolio
            initial_balance: Balance inicial
            
        Returns:
            Tupla (reward_total, componentes_detallados)
        """
        reward_components = {}
        total_reward = 0.0
        
        # Obtener información de la posición
        position = obs.get("position", {})
        in_position = int(position.get("side", 0)) != 0
        
        # HOLDING REWARD: +0.1 cada 10 barras con equity positivo
        if in_position and current_equity > initial_balance:
            # Contar barras con equity positivo
            if current_equity > self.last_equity_check:
                self.bars_held_with_positive_equity += 1
            else:
                self.bars_held_with_positive_equity = 0  # Reset si equity baja
            
            # Reward cada N barras
            if (self.bars_held_with_positive_equity > 0 and 
                self.bars_held_with_positive_equity % self.reward_interval == 0):
                total_reward += self.holding_reward
                reward_components["holding_reward"] = self.holding_reward
            
            self.last_equity_check = current_equity
        else:
            self.bars_held_with_positive_equity = 0
            self.last_equity_check = current_equity
        
        return total_reward, reward_components
    
    def reset(self):
        """Resetea los contadores internos"""
        self.bars_held_with_positive_equity = 0
        self.last_equity_check = 0.0
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Obtiene estadísticas del sistema
        
        Returns:
            Diccionario con estadísticas
        """
        return {
            "holding_reward": self.holding_reward,
            "reward_interval": self.reward_interval,
            "bars_held_with_positive_equity": self.bars_held_with_positive_equity,
            "last_equity_check": self.last_equity_check
        }
