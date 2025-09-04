# base_env/actions/drawdown_penalty.py
"""
Sistema de penalización por drawdown intra-trade.
Premia TPs "limpios" y castiga señales que atraviesan gran DD antes de cerrar.
"""

from __future__ import annotations
from typing import Dict, Any, List, Tuple, Optional


class DrawdownPenalty:
    """Sistema de penalización por drawdown intra-trade"""

    def __init__(self, config: Dict[str, Any]):
        """
        Inicializa el sistema de penalización por drawdown
        
        Args:
            config: Configuración del sistema
        """
        self.config = config
        # Configuración desde rewards.yaml
        dd_config = self.config.get("drawdown_penalty", {})
        self.enabled = dd_config.get("enabled", True)
        self.dd_ratio_threshold = dd_config.get("dd_ratio_threshold", 0.5)
        self.weight = dd_config.get("weight", 0.3)
        self.per_trade_cap = dd_config.get("per_trade_cap", 0.5)

    def calculate_drawdown_penalty(self, realized_pnl: float, max_drawdown_vs_sl: float, 
                                 sl_distance: float, notional: float) -> Tuple[float, Dict[str, float]]:
        """
        Calcula penalización por drawdown intra-trade
        
        Args:
            realized_pnl: PnL realizado del trade
            max_drawdown_vs_sl: Máximo drawdown vs SL durante el trade
            sl_distance: Distancia del SL desde entrada
            notional: Valor nocional del trade
            
        Returns:
            Tupla (penalty, componentes_detallados)
        """
        penalty_components = {}
        total_penalty = 0.0
        
        if not self.enabled or sl_distance <= 0:
            return total_penalty, penalty_components
        
        # Calcular ratio de drawdown: dd_ratio = max_drawdown_vs_SL / sl_distance
        dd_ratio = max_drawdown_vs_sl / sl_distance
        
        # Penalizar solo si supera el threshold
        if dd_ratio > self.dd_ratio_threshold:
            # penalty = -w_dd * max(0, dd_ratio - threshold)
            excess_dd = max(0, dd_ratio - self.dd_ratio_threshold)
            raw_penalty = -self.weight * excess_dd
            
            # Aplicar cap
            capped_penalty = max(-self.per_trade_cap, raw_penalty)
            
            total_penalty = capped_penalty
            penalty_components["drawdown_penalty"] = total_penalty
            penalty_components["dd_ratio"] = dd_ratio
            penalty_components["excess_dd"] = excess_dd
        
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
            "dd_ratio_threshold": self.dd_ratio_threshold,
            "weight": self.weight,
            "per_trade_cap": self.per_trade_cap
        }
