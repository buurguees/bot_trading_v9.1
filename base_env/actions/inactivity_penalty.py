# base_env/actions/inactivity_penalty.py
"""
Sistema de penalties por Inactividad.
Penaliza al bot por no abrir trades durante períodos prolongados.
"""

from __future__ import annotations
from typing import Dict, Any, List, Tuple


class InactivityPenalty:
    """Sistema de penalties por Inactividad"""
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Inicializa el sistema de penalties por Inactividad
        
        Args:
            config: Configuración del sistema (opcional)
        """
        self.config = config or {}
        # Configuración desde rewards.yaml
        inactivity_config = self.config.get("inactivity_penalties", {})
        self.inactivity_penalty = inactivity_config.get("inactivity_penalty", -0.01)
        self.penalty_interval = inactivity_config.get("penalty_interval", 100)  # Cada N pasos
        
        # Contadores internos
        self.steps_since_last_trade = 0
        self.last_trade_step = 0
    
    def calculate_inactivity_penalty(self, events: List[Dict[str, Any]], 
                                   current_step: int) -> Tuple[float, Dict[str, float]]:
        """
        Calcula el penalty por Inactividad
        
        Args:
            events: Lista de eventos del step actual
            current_step: Número de step actual
            
        Returns:
            Tupla (penalty_total, componentes_detallados)
        """
        penalty_components = {}
        total_penalty = 0.0
        
        # PENALTY POR INACTIVIDAD: -0.01 cada 100 pasos sin abrir trade
        if any(e.get("kind") == "OPEN" for e in events):
            self.steps_since_last_trade = 0
            self.last_trade_step = current_step
        else:
            self.steps_since_last_trade += 1
        
        # Aplicar penalty de inactividad
        if self.steps_since_last_trade > self.penalty_interval:
            inactivity_blocks = self.steps_since_last_trade // self.penalty_interval
            inactivity_penalty = self.inactivity_penalty * inactivity_blocks
            total_penalty += inactivity_penalty
            penalty_components["inactivity_penalty"] = inactivity_penalty
        
        return total_penalty, penalty_components
    
    def reset(self):
        """Resetea los contadores internos"""
        self.steps_since_last_trade = 0
        self.last_trade_step = 0
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Obtiene estadísticas del sistema
        
        Returns:
            Diccionario con estadísticas
        """
        return {
            "inactivity_penalty": self.inactivity_penalty,
            "penalty_interval": self.penalty_interval,
            "steps_since_last_trade": self.steps_since_last_trade,
            "last_trade_step": self.last_trade_step
        }
