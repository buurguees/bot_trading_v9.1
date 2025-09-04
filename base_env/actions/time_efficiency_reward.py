# base_env/actions/time_efficiency_reward.py
"""
Sistema de rewards por eficiencia temporal (PnL por barra).
Incentiva exits oportunas y evita over-holding.
"""

from __future__ import annotations
from typing import Dict, Any, List, Tuple, Optional


class TimeEfficiencyReward:
    """Sistema de rewards por eficiencia temporal"""

    def __init__(self, config: Dict[str, Any]):
        """
        Inicializa el sistema de rewards por eficiencia temporal
        
        Args:
            config: Configuración del sistema
        """
        self.config = config
        # Configuración desde rewards.yaml
        eff_config = self.config.get("time_efficiency", {})
        self.enabled = eff_config.get("enabled", True)
        self.weight = eff_config.get("weight", 0.1)
        self.per_bar_cap = eff_config.get("per_bar_cap", 0.03)
        self.per_trade_cap = eff_config.get("per_trade_cap", 0.3)

    def calculate_time_efficiency_reward(self, realized_pnl: float, bars_held: int, 
                                       notional: float) -> Tuple[float, Dict[str, float]]:
        """
        Calcula reward por eficiencia temporal
        
        Args:
            realized_pnl: PnL realizado del trade
            bars_held: Número de barras que se mantuvo el trade
            notional: Valor nocional del trade
            
        Returns:
            Tupla (reward, componentes_detallados)
        """
        reward_components = {}
        total_reward = 0.0
        
        if not self.enabled or bars_held <= 0 or notional <= 0:
            return total_reward, reward_components
        
        # Calcular eficiencia: eff = realized_pnl / max(1, bars_held)
        efficiency = realized_pnl / max(1, bars_held)
        
        # Normalizar por notional para hacer comparable entre trades
        normalized_efficiency = efficiency / notional
        
        # Aplicar peso y cap por barra
        raw_reward = normalized_efficiency * self.weight
        per_bar_capped = max(-self.per_bar_cap, min(self.per_bar_cap, raw_reward))
        
        # Aplicar cap total por trade
        total_reward = max(-self.per_trade_cap, min(self.per_trade_cap, per_bar_capped))
        
        reward_components["time_efficiency"] = total_reward
        reward_components["efficiency"] = efficiency
        reward_components["normalized_efficiency"] = normalized_efficiency
        reward_components["bars_held"] = bars_held
        
        return total_reward, reward_components

    def reset(self):
        """Resetea el sistema"""
        pass

    def get_stats(self) -> Dict[str, Any]:
        """
        Obtiene estadísticas del sistema
        
        Returns:
            Diccionario con estadísticas
        """
        return {
            "enabled": self.enabled,
            "weight": self.weight,
            "per_bar_cap": self.per_bar_cap,
            "per_trade_cap": self.per_trade_cap
        }
