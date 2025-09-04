# base_env/actions/overtrading_penalty.py
"""
Sistema de penalización por sobre-operar (overtrading).
Si abres >N trades en M barras sin mejora de ROI acumulado → penalty.
"""

from __future__ import annotations
from typing import Dict, Any, List, Tuple, Optional
from collections import deque


class OvertradingPenalty:
    """Sistema de penalización por overtrading"""

    def __init__(self, config: Dict[str, Any]):
        """
        Inicializa el sistema de penalización por overtrading
        
        Args:
            config: Configuración del sistema
        """
        self.config = config
        # Configuración desde rewards.yaml
        ot_config = self.config.get("overtrading_penalty", {})
        self.enabled = ot_config.get("enabled", True)
        self.window_bars = ot_config.get("window_bars", 200)
        self.trade_threshold = ot_config.get("trade_threshold", 5)
        self.roi_min_pct = ot_config.get("roi_min_pct", 0.0)
        self.penalty = ot_config.get("penalty", 0.1)
        
        # Historial de trades en ventana deslizante
        self.trade_history = deque(maxlen=self.window_bars)
        self.current_step = 0

    def calculate_overtrading_penalty(self, events: List[Dict[str, Any]], 
                                    current_roi_pct: float) -> Tuple[float, Dict[str, float]]:
        """
        Calcula penalización por overtrading
        
        Args:
            events: Lista de eventos del step actual
            current_roi_pct: ROI acumulado actual en porcentaje
            
        Returns:
            Tupla (penalty, componentes_detallados)
        """
        penalty_components = {}
        total_penalty = 0.0
        
        if not self.enabled:
            return total_penalty, penalty_components
        
        # Incrementar contador de steps
        self.current_step += 1
        
        # Registrar trades en la ventana actual
        for event in events:
            if event.get("kind") == "OPEN":
                self.trade_history.append({
                    "step": self.current_step,
                    "roi_at_open": current_roi_pct
                })
        
        # Verificar condición de overtrading
        if len(self.trade_history) >= self.trade_threshold:
            # Calcular ROI promedio al abrir trades
            roi_at_opens = [trade["roi_at_open"] for trade in self.trade_history]
            avg_roi_at_open = sum(roi_at_opens) / len(roi_at_opens)
            
            # Penalizar si ROI actual no ha mejorado significativamente
            roi_improvement = current_roi_pct - avg_roi_at_open
            
            if roi_improvement <= self.roi_min_pct:
                total_penalty = -self.penalty
                penalty_components["overtrading_penalty"] = total_penalty
                penalty_components["trades_in_window"] = len(self.trade_history)
                penalty_components["roi_improvement"] = roi_improvement
                penalty_components["avg_roi_at_open"] = avg_roi_at_open
        
        return total_penalty, penalty_components

    def reset(self):
        """Resetea el sistema para un nuevo run"""
        self.trade_history.clear()
        self.current_step = 0

    def get_stats(self) -> Dict[str, Any]:
        """
        Obtiene estadísticas del sistema
        
        Returns:
            Diccionario con estadísticas
        """
        return {
            "enabled": self.enabled,
            "window_bars": self.window_bars,
            "trade_threshold": self.trade_threshold,
            "roi_min_pct": self.roi_min_pct,
            "penalty": self.penalty,
            "trades_in_window": len(self.trade_history),
            "current_step": self.current_step
        }
