# base_env/actions/stop_loss_penalty.py
"""
Sistema de penalties por Stop Loss hits.
Penaliza al bot cuando alcanza el Stop Loss.
"""

from __future__ import annotations
from typing import Dict, Any, List, Tuple


class StopLossPenalty:
    """Sistema de penalties por Stop Loss"""
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Inicializa el sistema de penalties por Stop Loss
        
        Args:
            config: Configuración del sistema (opcional)
        """
        self.config = config or {}
        # Configuración desde rewards.yaml
        sl_config = self.config.get("stop_loss_penalties", {})
        self.sl_penalty = sl_config.get("sl_penalty", -0.5)
        self.sl_hit_penalty = sl_config.get("sl_hit_penalty", 0.0)  # Bonus/malus por SL_HIT
    
    def calculate_sl_penalty(self, events: List[Dict[str, Any]]) -> Tuple[float, Dict[str, float]]:
        """
        Calcula el penalty por Stop Loss hits
        
        Args:
            events: Lista de eventos del step actual
            
        Returns:
            Tupla (penalty_total, componentes_detallados)
        """
        penalty_components = {}
        total_penalty = 0.0
        
        # 1. PENALTY POR STOP LOSS: -0.5
        sl_events = [e for e in events if e.get("kind") == "SL_HIT"]
        if sl_events:
            total_penalty += self.sl_penalty
            penalty_components["sl_penalty"] = self.sl_penalty
        
        # 2. BONUS/MALUS ADICIONAL POR SL_HIT (configurable)
        if sl_events and self.sl_hit_penalty != 0:
            total_penalty += self.sl_hit_penalty
            penalty_components["sl_hit_penalty"] = self.sl_hit_penalty
        
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
            "sl_penalty": self.sl_penalty,
            "sl_hit_penalty": self.sl_hit_penalty
        }
