# base_env/actions/progress_bonus.py
"""
Sistema de bonus por Progreso hacia Objetivo.
Recompensa al bot por progresar hacia el balance objetivo.
"""

from __future__ import annotations
from typing import Dict, Any, List, Tuple


class ProgressBonus:
    """Sistema de bonus por Progreso"""
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Inicializa el sistema de bonus por Progreso
        
        Args:
            config: Configuración del sistema (opcional)
        """
        self.config = config or {}
        self.w_progress = self.config.get("progress_bonus", 0.0)
        self.w_balance_milestone = self.config.get("balance_milestone_reward", 0.0)
        self.w_compound = self.config.get("compound_bonus", 0.0)
    
    def calculate_progress_bonus(self, current_equity: float, 
                               initial_balance: float, 
                               target_balance: float,
                               balance_milestones: int = 0) -> Tuple[float, Dict[str, float]]:
        """
        Calcula bonus por progreso hacia objetivo
        
        Args:
            current_equity: Equity actual del portfolio
            initial_balance: Balance inicial
            target_balance: Balance objetivo
            balance_milestones: Número de milestones alcanzados
            
        Returns:
            Tupla (reward_total, componentes_detallados)
        """
        reward_components = {}
        total_reward = 0.0
        
        # Bonus por milestones de balance
        if balance_milestones > 0:
            milestone_bonus = self.w_balance_milestone * balance_milestones
            total_reward += milestone_bonus
            reward_components["balance_milestone"] = milestone_bonus
        
        # Bonus por progreso hacia objetivo
        if current_equity > initial_balance:
            progress = (current_equity - initial_balance) / (target_balance - initial_balance)
            if progress > 0:
                progress_bonus = self.w_progress * progress
                total_reward += progress_bonus
                reward_components["progress_bonus"] = progress_bonus
        
        # Bonus por compound (crecimiento exponencial)
        if current_equity > initial_balance and self.w_compound > 0:
            growth_factor = current_equity / initial_balance
            compound_bonus = self.w_compound * (growth_factor - 1.0)
            total_reward += compound_bonus
            reward_components["compound_bonus"] = compound_bonus
        
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
            "w_progress": self.w_progress,
            "w_balance_milestone": self.w_balance_milestone,
            "w_compound": self.w_compound
        }
