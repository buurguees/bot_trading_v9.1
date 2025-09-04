# base_env/actions/exploration_bonus.py
"""
Sistema de bonus por exploración acotada de leverage/timeframe.
Pequeño bonus una sola vez por combinación nueva (con decay) para que explore sin sesgarse a lo exótico.
"""

from __future__ import annotations
from typing import Dict, Any, List, Tuple, Optional
from collections import defaultdict


class ExplorationBonus:
    """Sistema de bonus por exploración"""

    def __init__(self, config: Dict[str, Any]):
        """
        Inicializa el sistema de bonus por exploración
        
        Args:
            config: Configuración del sistema
        """
        self.config = config
        # Configuración desde rewards.yaml
        exp_config = self.config.get("exploration_bonus", {})
        self.enabled = exp_config.get("enabled", True)
        self.weight = exp_config.get("weight", 0.05)
        self.decay_alpha = exp_config.get("decay_alpha", 5)
        self.per_trade_cap = exp_config.get("per_trade_cap", 0.1)
        
        # Historial de combinaciones exploradas
        self.seen_combinations = defaultdict(int)  # {(leverage, timeframe): count}

    def calculate_exploration_bonus(self, leverage_used: float, timeframe_used: str) -> Tuple[float, Dict[str, float]]:
        """
        Calcula bonus por exploración de nuevas combinaciones
        
        Args:
            leverage_used: Leverage utilizado en el trade
            timeframe_used: Timeframe utilizado en el trade
            
        Returns:
            Tupla (bonus, componentes_detallados)
        """
        bonus_components = {}
        total_bonus = 0.0
        
        if not self.enabled:
            return total_bonus, bonus_components
        
        # Crear clave de combinación
        combination_key = (round(leverage_used, 1), timeframe_used)
        
        # Incrementar contador de esta combinación
        self.seen_combinations[combination_key] += 1
        count = self.seen_combinations[combination_key]
        
        # Calcular bonus con decay exponencial
        # bonus = w_explore * exp(-count/alpha)
        decay_factor = self.decay_alpha
        raw_bonus = self.weight * (1.0 / (1.0 + count / decay_factor))
        
        # Aplicar cap
        capped_bonus = min(self.per_trade_cap, raw_bonus)
        
        total_bonus = capped_bonus
        bonus_components["exploration_bonus"] = total_bonus
        bonus_components["combination"] = f"{leverage_used}x_{timeframe_used}"
        bonus_components["count"] = count
        bonus_components["decay_factor"] = 1.0 / (1.0 + count / decay_factor)
        
        return total_bonus, bonus_components

    def reset(self):
        """Resetea el sistema para un nuevo run"""
        self.seen_combinations.clear()

    def get_stats(self) -> Dict[str, Any]:
        """
        Obtiene estadísticas del sistema
        
        Returns:
            Diccionario con estadísticas
        """
        return {
            "enabled": self.enabled,
            "weight": self.weight,
            "decay_alpha": self.decay_alpha,
            "per_trade_cap": self.per_trade_cap,
            "unique_combinations": len(self.seen_combinations),
            "total_explorations": sum(self.seen_combinations.values())
        }
