# base_env/actions/execution_cost_penalty.py
"""
Sistema de penalización por costes de ejecución explícitos (fees + slippage).
Añade penalty didáctico por coste alto relativo.
"""

from __future__ import annotations
from typing import Dict, Any, List, Tuple, Optional


class ExecutionCostPenalty:
    """Sistema de penalización por costes de ejecución"""

    def __init__(self, config: Dict[str, Any]):
        """
        Inicializa el sistema de penalización por costes
        
        Args:
            config: Configuración del sistema
        """
        self.config = config
        # Configuración desde rewards.yaml
        cost_config = self.config.get("execution_cost_penalty", {})
        self.enabled = cost_config.get("enabled", True)
        self.include_slippage = cost_config.get("include_slippage", True)
        self.est_slippage_bps = cost_config.get("est_slippage_bps", 1.0)
        self.weight = cost_config.get("weight", 0.5)
        self.per_trade_cap = cost_config.get("per_trade_cap", 0.3)

    def calculate_execution_cost_penalty(self, fees_paid: float, notional: float, 
                                       price: float, qty: float) -> Tuple[float, Dict[str, float]]:
        """
        Calcula penalización por costes de ejecución
        
        Args:
            fees_paid: Fees pagados en el trade
            notional: Valor nocional del trade
            price: Precio de ejecución
            qty: Cantidad ejecutada
            
        Returns:
            Tupla (penalty, componentes_detallados)
        """
        penalty_components = {}
        total_penalty = 0.0
        
        if not self.enabled or notional <= 0:
            return total_penalty, penalty_components
        
        # Calcular slippage estimado si está habilitado
        slippage_cost = 0.0
        if self.include_slippage and price > 0 and qty > 0:
            # Slippage en bps convertido a coste absoluto
            slippage_bps = self.est_slippage_bps / 10000.0  # Convertir bps a decimal
            slippage_cost = notional * slippage_bps
        
        # Coste total = fees + slippage
        total_cost = fees_paid + slippage_cost
        
        # Calcular ratio de coste: cost_ratio = (fees+slip)/notional
        cost_ratio = total_cost / notional
        
        # Aplicar penalización: penalty = -w_cost * cost_ratio
        raw_penalty = -self.weight * cost_ratio
        
        # Aplicar cap
        capped_penalty = max(-self.per_trade_cap, raw_penalty)
        
        total_penalty = capped_penalty
        penalty_components["execution_cost_penalty"] = total_penalty
        penalty_components["cost_ratio"] = cost_ratio
        penalty_components["fees_paid"] = fees_paid
        penalty_components["slippage_cost"] = slippage_cost
        penalty_components["total_cost"] = total_cost
        
        return total_penalty, penalty_components

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
            "include_slippage": self.include_slippage,
            "est_slippage_bps": self.est_slippage_bps,
            "weight": self.weight,
            "per_trade_cap": self.per_trade_cap
        }
