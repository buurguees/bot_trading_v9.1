# base_env/actions/bankruptcy_penalty.py
"""
Sistema de penalties por Bankruptcy.
Penaliza severamente al bot cuando se declara en bancarrota.
"""

from __future__ import annotations
from typing import Dict, Any, List, Tuple


class BankruptcyPenalty:
    """Sistema de penalties por Bankruptcy"""
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Inicializa el sistema de penalties por Bankruptcy
        
        Args:
            config: Configuración del sistema (opcional)
        """
        self.config = config or {}
        # Configuración desde rewards.yaml
        bankruptcy_config = self.config.get("bankruptcy_penalties", {})
        self.bankruptcy_penalty = bankruptcy_config.get("bankruptcy_penalty", -10.0)
        self.survival_bonus = bankruptcy_config.get("survival_bonus", 0.001)
        self.min_survival_equity_pct = bankruptcy_config.get("min_survival_equity_pct", 0.5)
    
    def calculate_bankruptcy_penalty(self, events: List[Dict[str, Any]], 
                                   current_equity: float, 
                                   initial_balance: float) -> Tuple[float, Dict[str, float]]:
        """
        Calcula el penalty por Bankruptcy y bonus por supervivencia
        
        Args:
            events: Lista de eventos del step actual
            current_equity: Equity actual del portfolio
            initial_balance: Balance inicial
            
        Returns:
            Tupla (penalty_total, componentes_detallados)
        """
        penalty_components = {}
        total_penalty = 0.0
        
        # 1. PENALTY POR BANKRUPTCY: -10.0
        bankruptcy_events = [e for e in events if e.get("kind") == "BANKRUPTCY"]
        if bankruptcy_events:
            total_penalty += self.bankruptcy_penalty
            penalty_components["bankruptcy_penalty"] = self.bankruptcy_penalty
        
        # 2. BONUS POR SUPERVIVENCIA (cada step sin quiebra)
        if not bankruptcy_events and current_equity > initial_balance * self.min_survival_equity_pct:
            survival_bonus = self.survival_bonus
            total_penalty += survival_bonus  # Se suma porque es un bonus (positivo)
            penalty_components["survival_bonus"] = survival_bonus
        
        return total_penalty, penalty_components
    
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
            "bankruptcy_penalty": self.bankruptcy_penalty,
            "survival_bonus": self.survival_bonus,
            "min_survival_equity_pct": self.min_survival_equity_pct
        }
