# base_env/actions/progress_milestone_reward.py
"""
Sistema de rewards por Milestones de Progreso.
Recompensa al bot por alcanzar hitos de progreso hacia el balance objetivo.
Cada milestone solo se puede cobrar una vez por run.
"""

from __future__ import annotations
from typing import Dict, Any, List, Tuple


class ProgressMilestoneReward:
    """Sistema de rewards por Milestones de Progreso"""

    def __init__(self, config: Dict[str, Any]):
        """
        Inicializa el sistema de rewards por milestones de progreso
        
        Args:
            config: Configuración del sistema
        """
        self.config = config
        # Configuración desde rewards.yaml
        milestone_config = self.config.get("progress_milestone_rewards", {})
        self.enabled = milestone_config.get("enabled", True)
        
        # Milestones configurables (porcentaje de progreso -> reward)
        self.milestones = milestone_config.get("milestones", {
            10: 1.0,    # 10% del progreso = +1.0
            25: 1.5,    # 25% del progreso = +1.5
            50: 2.0,    # 50% del progreso = +2.0
            75: 3.0,    # 75% del progreso = +3.0
            100: 10.0   # 100% del progreso = +10.0
        })
        
        # Milestones alcanzados en este run (para evitar duplicados)
        self.achieved_milestones: List[int] = []
        
        # Balance inicial y objetivo para calcular progreso
        self.initial_balance = 0.0
        self.target_balance = 0.0
        
        # Último progreso calculado
        self.last_progress_pct = 0.0

    def initialize_run(self, initial_balance: float, target_balance: float):
        """
        Inicializa el sistema para un nuevo run
        
        Args:
            initial_balance: Balance inicial del run
            target_balance: Balance objetivo
        """
        self.initial_balance = initial_balance
        self.target_balance = target_balance
        self.achieved_milestones.clear()
        self.last_progress_pct = 0.0

    def calculate_progress_milestone_reward(self, current_balance: float) -> Tuple[float, Dict[str, float]]:
        """
        Calcula reward por alcanzar milestones de progreso
        
        Args:
            current_balance: Balance actual del portfolio
            
        Returns:
            Tupla (reward_total, componentes_detallados)
        """
        reward_components = {}
        total_reward = 0.0
        
        if not self.enabled or self.initial_balance <= 0 or self.target_balance <= self.initial_balance:
            return total_reward, reward_components
        
        # Calcular progreso actual (porcentaje del camino hacia el objetivo)
        progress_pct = ((current_balance - self.initial_balance) / (self.target_balance - self.initial_balance)) * 100.0
        
        # Verificar si se alcanzó algún milestone nuevo
        for milestone_pct, reward in self.milestones.items():
            # Solo aplicar si:
            # 1. El progreso actual alcanza o supera el milestone
            # 2. El progreso anterior era menor al milestone (nuevo logro)
            # 3. No se ha cobrado este milestone antes en este run
            if (progress_pct >= milestone_pct and 
                self.last_progress_pct < milestone_pct and 
                milestone_pct not in self.achieved_milestones):
                
                total_reward += reward
                self.achieved_milestones.append(milestone_pct)
                reward_components[f"milestone_{milestone_pct}pct"] = reward
                
                print(f"🎯 MILESTONE ALCANZADO: {milestone_pct}% del progreso → +{reward} reward")
        
        # Actualizar progreso anterior
        self.last_progress_pct = progress_pct
        
        return total_reward, reward_components

    def get_current_progress_pct(self, current_balance: float) -> float:
        """
        Calcula el porcentaje de progreso actual
        
        Args:
            current_balance: Balance actual
            
        Returns:
            Porcentaje de progreso (0-100)
        """
        if self.initial_balance <= 0 or self.target_balance <= self.initial_balance:
            return 0.0
        
        progress_pct = ((current_balance - self.initial_balance) / (self.target_balance - self.initial_balance)) * 100.0
        return max(0.0, min(100.0, progress_pct))

    def get_achieved_milestones(self) -> List[int]:
        """Devuelve la lista de milestones alcanzados en este run"""
        return self.achieved_milestones.copy()

    def get_next_milestone(self, current_balance: float) -> Tuple[int, float]:
        """
        Devuelve el próximo milestone a alcanzar
        
        Args:
            current_balance: Balance actual
            
        Returns:
            Tupla (porcentaje_milestone, reward)
        """
        current_progress = self.get_current_progress_pct(current_balance)
        
        for milestone_pct in sorted(self.milestones.keys()):
            if milestone_pct > current_progress and milestone_pct not in self.achieved_milestones:
                return milestone_pct, self.milestones[milestone_pct]
        
        return 100, self.milestones.get(100, 0.0)

    def reset(self):
        """Resetea el sistema para un nuevo run"""
        self.achieved_milestones.clear()
        self.last_progress_pct = 0.0

    def get_stats(self) -> Dict[str, Any]:
        """
        Obtiene estadísticas del sistema
        
        Returns:
            Diccionario con estadísticas
        """
        return {
            "enabled": self.enabled,
            "milestones": self.milestones,
            "achieved_milestones": self.achieved_milestones,
            "initial_balance": self.initial_balance,
            "target_balance": self.target_balance,
            "last_progress_pct": self.last_progress_pct
        }
