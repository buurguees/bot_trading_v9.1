# base_env/actions/roi_reward.py
"""
Sistema de rewards por ROI (Return on Investment).
Recompensa al bot basado en el porcentaje de retorno de la inversión.
"""

from __future__ import annotations
from typing import Dict, Any, List, Tuple


class ROIReward:
    """Sistema de rewards por ROI"""
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Inicializa el sistema de rewards por ROI
        
        Args:
            config: Configuración del sistema (opcional)
        """
        self.config = config or {}
        self.tiers_pos = self.config.get("tiers", {}).get("pos", [])
        self.tiers_neg = self.config.get("tiers", {}).get("neg", [])
    
    def calculate_roi_reward(self, roi_pct: float) -> Tuple[float, Dict[str, float]]:
        """
        Calcula el reward basado en ROI del trade
        
        Args:
            roi_pct: Porcentaje de ROI del trade
            
        Returns:
            Tupla (reward_total, componentes_detallados)
        """
        reward_components = {}
        total_reward = 0.0
        
        if roi_pct >= 0:
            # Trade ganador
            for min_roi, max_roi, reward in self.tiers_pos:
                if min_roi <= roi_pct <= max_roi:
                    total_reward += reward
                    reward_components["roi_reward"] = reward
                    reward_components["roi_tier"] = f"{min_roi}-{max_roi}%"
                    break
        else:
            # Trade perdedor
            roi_abs = abs(roi_pct)
            for min_roi, max_roi, penalty in self.tiers_neg:
                if min_roi <= roi_abs <= max_roi:
                    total_reward += penalty  # penalty ya es negativo
                    reward_components["roi_penalty"] = penalty
                    reward_components["roi_tier"] = f"-{min_roi}-{max_roi}%"
                    break
        
        return total_reward, reward_components
    
    @staticmethod
    def _tier_value(tiers: List[List[float]], pct: float) -> float:
        """Devuelve el valor del tramo donde cae pct (en valor absoluto para tiers_neg)."""
        for lo, hi, val in tiers:
            if lo <= pct <= hi:
                return float(val)
        return float(tiers[-1][2]) if tiers else 0.0
    
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
            "tiers_pos": self.tiers_pos,
            "tiers_neg": self.tiers_neg
        }
