# base_env/actions/volatility_reward.py
"""
Sistema de rewards por PnL normalizado por volatilidad (Vol-Scaled).
Evita que épocas muy volátiles dominen el gradiente.
"""

from __future__ import annotations
from typing import Dict, Any, List, Tuple, Optional


class VolatilityReward:
    """Sistema de rewards por PnL normalizado por volatilidad"""

    def __init__(self, config: Dict[str, Any]):
        """
        Inicializa el sistema de rewards por volatilidad
        
        Args:
            config: Configuración del sistema
        """
        self.config = config
        # Configuración desde rewards.yaml
        vol_config = self.config.get("volatility_reward", {})
        self.enabled = vol_config.get("enabled", True)
        self.tf = vol_config.get("tf", "1m")
        self.atr_period = vol_config.get("atr_period", 14)
        self.atr_mult = vol_config.get("atr_mult", 1.5)
        self.weight = vol_config.get("weight", 0.2)
        self.per_trade_cap = vol_config.get("per_trade_cap", 0.4)

    def calculate_volatility_reward(self, realized_pnl: float, price: float, 
                                  atr_value: float, notional: float) -> Tuple[float, Dict[str, float]]:
        """
        Calcula reward por PnL normalizado por volatilidad
        
        Args:
            realized_pnl: PnL realizado del trade
            price: Precio de entrada del trade
            atr_value: Valor ATR del timeframe de ejecución
            notional: Valor nocional del trade
            
        Returns:
            Tupla (reward, componentes_detallados)
        """
        reward_components = {}
        total_reward = 0.0
        
        if not self.enabled or atr_value <= 0 or price <= 0:
            return total_reward, reward_components
        
        # Fórmula: reward_vol = (realized_pnl / (price * ATR_k)) * w_vol
        volatility_denominator = price * (atr_value * self.atr_mult)
        vol_scaled_pnl = realized_pnl / volatility_denominator
        
        # Aplicar peso y cap
        raw_reward = vol_scaled_pnl * self.weight
        capped_reward = max(-self.per_trade_cap, min(self.per_trade_cap, raw_reward))
        
        total_reward = capped_reward
        reward_components["volatility_scaled"] = total_reward
        reward_components["vol_scaled_pnl"] = vol_scaled_pnl
        reward_components["atr_used"] = atr_value
        
        return total_reward, reward_components

    def get_atr_from_obs(self, obs: Dict[str, Any]) -> Optional[float]:
        """
        Extrae ATR del timeframe de ejecución desde la observación
        
        Args:
            obs: Observación del entorno
            
        Returns:
            Valor ATR o None si no se encuentra
        """
        try:
            features = obs.get("features", {})
            tf_features = features.get(self.tf, {})
            atr_key = f"atr{self.atr_period}"
            return float(tf_features.get(atr_key, 0.0))
        except (KeyError, ValueError, TypeError):
            return None

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
            "tf": self.tf,
            "atr_period": self.atr_period,
            "atr_mult": self.atr_mult,
            "weight": self.weight,
            "per_trade_cap": self.per_trade_cap
        }
