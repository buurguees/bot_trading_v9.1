# base_env/actions/ttl_expiry_penalty.py
"""
Sistema de penalización por expiración TTL.
Si sale por TTL sin tocar SL/TP, usa small_neg para empujar a cerrar con criterio.
"""

from __future__ import annotations
from typing import Dict, Any, List, Tuple, Optional


class TTLExpiryPenalty:
    """Sistema de penalización por expiración TTL"""

    def __init__(self, config: Dict[str, Any]):
        """
        Inicializa el sistema de penalización por TTL
        
        Args:
            config: Configuración del sistema
        """
        self.config = config
        # Configuración desde rewards.yaml
        ttl_config = self.config.get("ttl_expiry", {})
        self.enabled = ttl_config.get("enabled", True)
        self.neutral_penalty = ttl_config.get("neutral_penalty", 0.05)

    def calculate_ttl_expiry_penalty(self, close_reason: str, realized_pnl: float) -> Tuple[float, Dict[str, float]]:
        """
        Calcula penalización por expiración TTL
        
        Args:
            close_reason: Razón del cierre del trade
            realized_pnl: PnL realizado del trade
            
        Returns:
            Tupla (penalty, componentes_detallados)
        """
        penalty_components = {}
        total_penalty = 0.0
        
        if not self.enabled:
            return total_penalty, penalty_components
        
        # Aplicar penalización solo si se cerró por TTL
        if close_reason == "ttl_hit":
            total_penalty = -self.neutral_penalty
            penalty_components["ttl_expiry_penalty"] = total_penalty
            penalty_components["close_reason"] = close_reason
            penalty_components["realized_pnl"] = realized_pnl
        
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
            "neutral_penalty": self.neutral_penalty
        }
