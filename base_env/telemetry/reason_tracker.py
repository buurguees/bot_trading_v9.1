"""
Sistema unificado de telemetría para razones de no-trade.
"""
from __future__ import annotations
from typing import Dict, List, Optional
from dataclasses import dataclass, field
from enum import Enum


class NoTradeReason(Enum):
    """Razones normalizadas por las que no se opera."""
    COOLDOWN = "COOLDOWN"
    WARMUP = "WARMUP"
    NO_SIGNAL = "NO_SIGNAL"
    POLICY_NO_OPEN = "POLICY_NO_OPEN"
    RISK_BLOCKED = "RISK_BLOCKED"
    NO_SL_DISTANCE = "NO_SL_DISTANCE"
    MIN_NOTIONAL_BLOCKED = "MIN_NOTIONAL_BLOCKED"
    EXPOSURE_LIMIT = "EXPOSURE_LIMIT"
    CIRCUIT_BREAKER = "CIRCUIT_BREAKER"
    BROKER_EMPTY = "BROKER_EMPTY"
    DONE_EARLY = "DONE_EARLY"
    BANKRUPTCY_RESTART = "BANKRUPTCY_RESTART"
    COOLDOWN_AFTER_RESET = "COOLDOWN_AFTER_RESET"
    SHORTS_DISABLED = "SHORTS_DISABLED"
    LOW_EQUITY = "LOW_EQUITY"
    LEVERAGE_CAP = "LEVERAGE_CAP"
    MARGIN_INSUFFICIENT = "MARGIN_INSUFFICIENT"
    POSITION_ALREADY_OPEN = "POSITION_ALREADY_OPEN"
    INVALID_ACTION = "INVALID_ACTION"


@dataclass
class TelemetrySnapshot:
    """Snapshot de telemetría en un momento dado."""
    timestamp: int
    total_events: int
    reasons: Dict[str, int] = field(default_factory=dict)
    top_reasons: List[tuple[str, int]] = field(default_factory=list)
    main_culprit: Optional[str] = None
    main_culprit_pct: float = 0.0


class ReasonTracker:
    """Tracker unificado para razones de no-trade."""
    
    def __init__(self):
        self._counters: Dict[str, int] = {}
        self._total_events = 0
        self._snapshots: List[TelemetrySnapshot] = []
    
    def increment(self, reason: NoTradeReason, count: int = 1) -> None:
        """Incrementa el contador de una razón."""
        reason_str = reason.value
        self._counters[reason_str] = self._counters.get(reason_str, 0) + count
        self._total_events += count
    
    def get_count(self, reason: NoTradeReason) -> int:
        """Obtiene el contador de una razón."""
        return self._counters.get(reason.value, 0)
    
    def get_all_counts(self) -> Dict[str, int]:
        """Obtiene todos los contadores."""
        return self._counters.copy()
    
    def get_total_events(self) -> int:
        """Obtiene el total de eventos."""
        return self._total_events
    
    def get_top_reasons(self, limit: int = 5) -> List[tuple[str, int]]:
        """Obtiene las razones más frecuentes."""
        sorted_reasons = sorted(
            self._counters.items(),
            key=lambda x: x[1],
            reverse=True
        )
        return sorted_reasons[:limit]
    
    def get_main_culprit(self) -> Optional[tuple[str, int, float]]:
        """Obtiene la razón principal (más del 30% de eventos)."""
        if self._total_events == 0:
            return None
        
        sorted_reasons = self.get_top_reasons(1)
        if not sorted_reasons:
            return None
        
        main_reason, count = sorted_reasons[0]
        percentage = (count / self._total_events) * 100.0
        
        if percentage > 30.0:
            return (main_reason, count, percentage)
        
        return None
    
    def create_snapshot(self, timestamp: int) -> TelemetrySnapshot:
        """Crea un snapshot de la telemetría actual."""
        top_reasons = self.get_top_reasons(5)
        main_culprit_info = self.get_main_culprit()
        
        snapshot = TelemetrySnapshot(
            timestamp=timestamp,
            total_events=self._total_events,
            reasons=self._counters.copy(),
            top_reasons=top_reasons,
            main_culprit=main_culprit_info[0] if main_culprit_info else None,
            main_culprit_pct=main_culprit_info[2] if main_culprit_info else 0.0
        )
        
        self._snapshots.append(snapshot)
        return snapshot
    
    def print_summary(self, title: str = "TELEMETRÍA DE RAZONES") -> None:
        """Imprime un resumen de la telemetría."""
        if self._total_events == 0:
            print(f"📊 {title}: Sin eventos registrados")
            return
        
        print(f"\n📊 {title}:")
        print("=" * 60)
        print(f"Total de eventos: {self._total_events}")
        
        # Top razones
        top_reasons = self.get_top_reasons(10)
        if top_reasons:
            print("\n🔝 Top razones:")
            for i, (reason, count) in enumerate(top_reasons, 1):
                percentage = (count / self._total_events) * 100.0
                print(f"   {i:2d}. {reason:<25} {count:6d} ({percentage:5.1f}%)")
        
        # Culpable principal
        main_culprit = self.get_main_culprit()
        if main_culprit:
            reason, count, percentage = main_culprit
            print(f"\n🚨 CULPABLE PRINCIPAL: {reason} ({count} eventos, {percentage:.1f}%)")
            
            # Sugerencias específicas
            suggestions = self._get_suggestions(reason)
            if suggestions:
                print("   💡 SUGERENCIAS:")
                for suggestion in suggestions:
                    print(f"      - {suggestion}")
        
        print("=" * 60)
    
    def _get_suggestions(self, reason: str) -> List[str]:
        """Obtiene sugerencias específicas para una razón."""
        suggestions_map = {
            "RISK_BLOCKED": [
                "Revisar risk_pct_per_trade en risk.yaml",
                "Verificar minNotional en symbols.yaml",
                "Aumentar initial_balance si es necesario"
            ],
            "NO_SIGNAL": [
                "Revisar análisis jerárquico en hierarchical.yaml",
                "Reducir min_confidence temporalmente",
                "Verificar indicadores técnicos"
            ],
            "MIN_NOTIONAL_BLOCKED": [
                "Reducir minNotional en symbols.yaml",
                "Activar train_force_min_notional en risk.yaml",
                "Aumentar risk_pct_per_trade"
            ],
            "LOW_EQUITY": [
                "Aumentar initial_balance en train.yaml",
                "Reducir risk_pct_per_trade",
                "Verificar fees y slippage"
            ],
            "NO_SL_DISTANCE": [
                "Verificar que se aplican SL/TP por defecto",
                "Revisar configuración de default_levels",
                "Asegurar que ATR está disponible"
            ],
            "POLICY_NO_OPEN": [
                "Revisar confidence del análisis jerárquico",
                "Verificar execute_tfs en hierarchical.yaml",
                "Considerar bypass de policy en entrenamiento"
            ],
            "COOLDOWN_AFTER_RESET": [
                "Reducir cooldown_bars en risk.yaml",
                "Verificar configuración de soft_reset",
                "Ajustar post_reset_leverage_cap"
            ]
        }
        
        return suggestions_map.get(reason, [])
    
    def reset(self) -> None:
        """Resetea todos los contadores."""
        self._counters.clear()
        self._total_events = 0
        self._snapshots.clear()
    
    def get_snapshots(self) -> List[TelemetrySnapshot]:
        """Obtiene todos los snapshots."""
        return self._snapshots.copy()
    
    def export_to_dict(self) -> Dict:
        """Exporta la telemetría a un diccionario."""
        return {
            "total_events": self._total_events,
            "counters": self._counters.copy(),
            "top_reasons": self.get_top_reasons(10),
            "main_culprit": self.get_main_culprit(),
            "snapshots_count": len(self._snapshots)
        }


# Instancia global del tracker
reason_tracker = ReasonTracker()
