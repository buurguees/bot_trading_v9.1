# base_env/actions/blocked_trade_penalty.py
"""
Sistema de penalties por Trades Bloqueados.
Penaliza al bot cuando los trades son bloqueados por diversas razones.
"""

from __future__ import annotations
from typing import Dict, Any, List, Tuple


class BlockedTradePenalty:
    """Sistema de penalties por Trades Bloqueados"""
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Inicializa el sistema de penalties por Trades Bloqueados
        
        Args:
            config: Configuración del sistema (opcional)
        """
        self.config = config or {}
        # Configuración desde rewards.yaml
        blocked_config = self.config.get("blocked_trade_penalties", {})
        self.block_penalty = blocked_config.get("block_penalty", -0.05)
        self.empty_run_penalty = blocked_config.get("empty_run_penalty", 0.0)
        self.blocked_event_types = blocked_config.get("blocked_event_types", [
            "NO_SL_DISTANCE", 
            "MIN_NOTIONAL_BLOCKED", 
            "DUPLICATE_DECISION",
            "RISK_MANAGER_BLOCKED",
            "INVALID_LEVELS"
        ])
    
    def calculate_blocked_trade_penalty(self, events: List[Dict[str, Any]], 
                                      empty_run: bool = False) -> Tuple[float, Dict[str, float]]:
        """
        Calcula penalty por trades bloqueados
        
        Args:
            events: Lista de eventos del step actual
            empty_run: Si el run está vacío (sin trades)
            
        Returns:
            Tupla (penalty_total, componentes_detallados)
        """
        penalty_components = {}
        total_penalty = 0.0
        
        # PENALTY POR TRADES BLOQUEADOS
        blocked_events = [e for e in events if e.get("kind") in self.blocked_event_types]
        if blocked_events:
            block_penalty = self.block_penalty * len(blocked_events)
            total_penalty += block_penalty
            penalty_components["blocked_trade_penalty"] = block_penalty
            penalty_components["blocked_events_count"] = len(blocked_events)
        
        # PENALTY POR RUNS VACÍOS
        if empty_run and self.empty_run_penalty != 0:
            total_penalty += self.empty_run_penalty
            penalty_components["empty_run_penalty"] = self.empty_run_penalty
        
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
            "block_penalty": self.block_penalty,
            "empty_run_penalty": self.empty_run_penalty
        }
