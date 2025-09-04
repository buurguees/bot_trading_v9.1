# base_env/actions/tp_hit_reward.py
"""
Reward por hit de Take Profit.
Configurable desde rewards.yaml.
"""

from __future__ import annotations
from typing import Dict, Any, Optional


class TPHitReward:
    """Reward por activación de Take Profit."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.enabled = config.get("enabled", True)
        self.reward_amount = config.get("reward_amount", 5.0)
        self.scale_with_profit = config.get("scale_with_profit", True)
        self.max_reward = config.get("max_reward", 50.0)
        self.leverage_bonus = config.get("leverage_bonus", True)

    def calculate_reward(self, realized_pnl: float, notional: float, leverage_used: float = 1.0) -> float:
        """
        Calcula reward por hit de TP.
        
        Args:
            realized_pnl: PnL realizado (positivo en TP)
            notional: Notional del trade
            leverage_used: Leverage usado
            
        Returns:
            Reward (positivo)
        """
        if not self.enabled:
            return 0.0
        
        # Reward base
        reward = self.reward_amount
        
        # Escalar con el profit si está habilitado
        if self.scale_with_profit and realized_pnl > 0:
            # Reward proporcional al profit
            profit_pct = realized_pnl / notional if notional > 0 else 0
            reward *= (1 + profit_pct)
        
        # Bonus por leverage si está habilitado
        if self.leverage_bonus and leverage_used > 1.0:
            leverage_multiplier = min(leverage_used / 3.0, 2.0)  # Cap en 2x
            reward *= leverage_multiplier
        
        # Aplicar límite máximo
        reward = min(reward, self.max_reward)
        
        return reward
