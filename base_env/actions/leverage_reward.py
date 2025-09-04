# base_env/actions/leverage_reward.py
"""
Sistema de rewards/penalties por Leverage.
Recompensa al bot por usar leverage apropiado y penaliza el uso excesivo.
"""

from __future__ import annotations
from typing import Dict, Any, List, Tuple


class LeverageReward:
    """Sistema de rewards/penalties por Leverage"""
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Inicializa el sistema de rewards/penalties por Leverage
        
        Args:
            config: Configuración del sistema (opcional)
        """
        self.config = config or {}
        self.leverage_config = self.config.get("leverage_rewards", {})
        self._leverage_history: List[float] = []  # Historial de leverage usado
    
    def calculate_leverage_reward(self, realized_pnl: float, leverage_used: float) -> Tuple[float, Dict[str, float]]:
        """
        Calcula reward/penalty basado en leverage usado
        
        Args:
            realized_pnl: PnL realizado del trade
            leverage_used: Leverage usado en el trade
            
        Returns:
            Tupla (reward_total, componentes_detallados)
        """
        reward_components = {}
        total_reward = 0.0
        
        # Reward por leverage alto en trades ganadores
        if realized_pnl > 0:
            high_leverage_config = self.leverage_config.get("high_leverage_bonus", {})
            if high_leverage_config.get("enabled", False):
                min_leverage = high_leverage_config.get("min_leverage", 5.0)
                if leverage_used >= min_leverage:
                    reward_per_leverage = high_leverage_config.get("reward_per_leverage", 0.1)
                    max_bonus = high_leverage_config.get("max_bonus", 2.0)
                    leverage_bonus = min((leverage_used - min_leverage) * reward_per_leverage, max_bonus)
                    total_reward += leverage_bonus
                    reward_components["high_leverage_bonus"] = leverage_bonus
        
        # Penalty por leverage alto en trades perdedores
        elif realized_pnl < 0:
            high_leverage_config = self.leverage_config.get("high_leverage_penalty", {})
            if high_leverage_config.get("enabled", False):
                min_leverage = high_leverage_config.get("min_leverage", 5.0)
                if leverage_used >= min_leverage:
                    penalty_per_leverage = high_leverage_config.get("penalty_per_leverage", 0.2)
                    max_penalty = high_leverage_config.get("max_penalty", 3.0)
                    leverage_penalty = min((leverage_used - min_leverage) * penalty_per_leverage, max_penalty)
                    total_reward -= leverage_penalty
                    reward_components["high_leverage_penalty"] = -leverage_penalty
        
        # Bonus por leverage conservador consistente
        conservative_config = self.leverage_config.get("conservative_bonus", {})
        if conservative_config.get("enabled", False):
            max_leverage = conservative_config.get("max_leverage", 3.0)
            if leverage_used <= max_leverage:
                consistency_bonus = conservative_config.get("consistency_bonus", 0.05)
                total_reward += consistency_bonus
                reward_components["conservative_bonus"] = consistency_bonus
        
        # Actualizar historial de leverage
        self._leverage_history.append(leverage_used)
        max_history = self.leverage_config.get("max_history_length", 100)
        if len(self._leverage_history) > max_history:
            self._leverage_history.pop(0)
        
        return total_reward, reward_components
    
    def get_leverage_consistency_score(self) -> float:
        """Calcula un score de consistencia en el uso de leverage"""
        min_samples = self.leverage_config.get("min_samples_for_consistency", 10)
        if len(self._leverage_history) < min_samples:
            return 0.0
        
        # Calcular desviación estándar del leverage usado
        mean_leverage = sum(self._leverage_history) / len(self._leverage_history)
        variance = sum((x - mean_leverage) ** 2 for x in self._leverage_history) / len(self._leverage_history)
        std_dev = variance ** 0.5
        
        # Score inverso a la desviación (más consistencia = score más alto)
        consistency_score = max(0.0, 1.0 - (std_dev / mean_leverage))
        return consistency_score
    
    def get_recommended_leverage(self, market_volatility: float, confidence: float) -> float:
        """
        Recomienda un leverage basado en volatilidad del mercado y confianza
        
        Args:
            market_volatility: Volatilidad del mercado (0.0 = baja, 1.0 = alta)
            confidence: Confianza en la señal (0.0 = baja, 1.0 = alta)
            
        Returns:
            Leverage recomendado
        """
        # Configuración desde rewards.yaml
        leverage_config = self.config.get("leverage_rewards", {})
        recommendation_config = leverage_config.get("leverage_recommendation", {})
        
        # Lógica configurable: más confianza y menos volatilidad = más leverage
        base_leverage = recommendation_config.get("base_leverage", 3.0)
        volatility_factor = 1.0 - (market_volatility * recommendation_config.get("volatility_reduction", 0.5))
        confidence_factor = 1.0 + (confidence * recommendation_config.get("confidence_boost", 0.5))
        
        recommended = base_leverage * volatility_factor * confidence_factor
        
        # Clamp al rango configurable
        min_leverage = recommendation_config.get("min_leverage", 2.0)
        max_leverage = recommendation_config.get("max_leverage", 10.0)
        return max(min_leverage, min(max_leverage, recommended))
    
    def reset(self):
        """Resetea el historial de leverage"""
        self._leverage_history.clear()
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Obtiene estadísticas del sistema
        
        Returns:
            Diccionario con estadísticas
        """
        return {
            "leverage_config": self.leverage_config,
            "leverage_history_length": len(self._leverage_history),
            "consistency_score": self.get_leverage_consistency_score()
        }
