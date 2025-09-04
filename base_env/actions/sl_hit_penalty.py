# base_env/actions/sl_hit_penalty.py
"""
Penalización por hit de Stop Loss.
Configurable desde rewards.yaml.
"""

from __future__ import annotations
from typing import Dict, Any, Optional


class SLHitPenalty:
    """Penalización por activación de Stop Loss."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.enabled = config.get("enabled", True)
        self.penalty_amount = config.get("penalty_amount", -5.0)
        self.scale_with_loss = config.get("scale_with_loss", True)
        self.max_penalty = config.get("max_penalty", -20.0)

    def calculate_penalty(self, realized_pnl: float, notional: float, leverage_used: float = 1.0) -> float:
        """
        Calcula penalización por hit de SL.
        
        Args:
            realized_pnl: PnL realizado (negativo en SL)
            notional: Notional del trade
            leverage_used: Leverage usado
            
        Returns:
            Penalización (negativa)
        """
        if not self.enabled:
            return 0.0
        
        # Penalización base
        penalty = self.penalty_amount
        
        # Escalar con la pérdida si está habilitado
        if self.scale_with_loss and realized_pnl < 0:
            # Penalización proporcional a la pérdida
            loss_pct = abs(realized_pnl) / notional if notional > 0 else 0
            penalty *= (1 + loss_pct * leverage_used)
        
        # Aplicar límite máximo
        penalty = max(penalty, self.max_penalty)
        
        return penalty
