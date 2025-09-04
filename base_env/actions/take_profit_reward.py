# base_env/actions/take_profit_reward.py
"""
Sistema de rewards por Take Profit hits.
Recompensa al bot cuando alcanza exitosamente el Take Profit.
"""

from __future__ import annotations
from typing import Dict, Any, List, Tuple


class TakeProfitReward:
    """Sistema de rewards por Take Profit"""
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Inicializa el sistema de rewards por Take Profit
        
        Args:
            config: Configuración del sistema (opcional)
        """
        self.config = config or {}
        # Configuración desde rewards.yaml
        tp_config = self.config.get("take_profit_rewards", {})
        self.tp_reward = tp_config.get("tp_reward", 1.0)
        self.efficient_rr_bonus = tp_config.get("efficient_rr_bonus", 0.2)
        self.high_rr_bonus = tp_config.get("high_rr_bonus", 0.5)
        self.high_roi_bonus = tp_config.get("high_roi_bonus", 0.3)
        self.max_drawdown_pct = tp_config.get("max_drawdown_pct", 0.5)
        self.high_rr_threshold = tp_config.get("high_rr_threshold", 2.0)
        self.high_roi_threshold = tp_config.get("high_roi_threshold", 5.0)
        self.rr_bonus_multiplier = tp_config.get("rr_bonus_multiplier", 0.1)
        self.roi_bonus_multiplier = tp_config.get("roi_bonus_multiplier", 0.01)
    
    def calculate_tp_reward(self, events: List[Dict[str, Any]]) -> Tuple[float, Dict[str, float]]:
        """
        Calcula el reward por Take Profit hits
        
        Args:
            events: Lista de eventos del step actual
            
        Returns:
            Tupla (reward_total, componentes_detallados)
        """
        reward_components = {}
        total_reward = 0.0
        
        # 1. REWARD POR TAKE PROFIT: +1.0
        tp_events = [e for e in events if e.get("kind") == "TP_HIT"]
        if tp_events:
            total_reward += self.tp_reward
            reward_components["tp_reward"] = self.tp_reward
        
        # 2. EFFICIENT R/R BONUS: +0.2 si TP con drawdown < 50% del SL
        if tp_events:
            tp_event = tp_events[0]  # Tomar el primer TP event
            entry_price = float(tp_event.get("entry_price", 0.0))
            sl_price = float(tp_event.get("sl", 0.0))
            tp_price = float(tp_event.get("tp", 0.0))
            
            if entry_price > 0 and sl_price > 0 and tp_price > 0:
                # Calcular distancias
                if entry_price > sl_price:  # LONG
                    sl_distance = entry_price - sl_price
                    tp_distance = tp_price - entry_price
                else:  # SHORT
                    sl_distance = sl_price - entry_price
                    tp_distance = entry_price - tp_price
                
                # Verificar si el drawdown fue < X% del SL
                if sl_distance > 0:
                    if tp_distance >= sl_distance * self.max_drawdown_pct:
                        total_reward += self.efficient_rr_bonus
                        reward_components["efficient_rr_bonus"] = self.efficient_rr_bonus
        
        # 3. BONUS ADICIONALES POR CALIDAD DEL TRADE
        close_events = [e for e in events if e.get("kind") == "CLOSE"]
        if close_events:
            close_event = close_events[0]
            roi_pct = float(close_event.get("roi_pct", 0.0))
            r_multiple = float(close_event.get("r_multiple", 0.0))
            
            # Bonus por R-multiple alto
            if r_multiple > self.high_rr_threshold:
                rr_bonus = min(self.high_rr_bonus, (r_multiple - self.high_rr_threshold) * self.rr_bonus_multiplier)
                total_reward += rr_bonus
                reward_components["high_rr_bonus"] = rr_bonus
            
            # Bonus por ROI positivo consistente
            if roi_pct > self.high_roi_threshold:
                roi_bonus = min(self.high_roi_bonus, roi_pct * self.roi_bonus_multiplier)
                total_reward += roi_bonus
                reward_components["high_roi_bonus"] = roi_bonus
        
        return total_reward, reward_components
    
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
            "tp_reward": self.tp_reward,
            "efficient_rr_bonus": self.efficient_rr_bonus,
            "high_rr_bonus": self.high_rr_bonus,
            "high_roi_bonus": self.high_roi_bonus
        }
