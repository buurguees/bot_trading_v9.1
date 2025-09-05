"""
Tests para el sistema unificado de telemetría de razones de no-trade.
"""
import pytest
from base_env.telemetry.reason_tracker import ReasonTracker, NoTradeReason, TelemetrySnapshot


def test_reason_tracker_basic_operations():
    """Test operaciones básicas del ReasonTracker."""
    tracker = ReasonTracker()
    
    # Test increment
    tracker.increment(NoTradeReason.RISK_BLOCKED, 5)
    tracker.increment(NoTradeReason.NO_SIGNAL, 3)
    tracker.increment(NoTradeReason.RISK_BLOCKED, 2)
    
    # Test get_count
    assert tracker.get_count(NoTradeReason.RISK_BLOCKED) == 7
    assert tracker.get_count(NoTradeReason.NO_SIGNAL) == 3
    assert tracker.get_count(NoTradeReason.POLICY_NO_OPEN) == 0
    
    # Test get_total_events
    assert tracker.get_total_events() == 10
    
    # Test get_all_counts
    counts = tracker.get_all_counts()
    assert counts["RISK_BLOCKED"] == 7
    assert counts["NO_SIGNAL"] == 3


def test_reason_tracker_top_reasons():
    """Test que se obtienen las razones más frecuentes correctamente."""
    tracker = ReasonTracker()
    
    # Añadir eventos
    tracker.increment(NoTradeReason.RISK_BLOCKED, 10)
    tracker.increment(NoTradeReason.NO_SIGNAL, 5)
    tracker.increment(NoTradeReason.MIN_NOTIONAL_BLOCKED, 3)
    tracker.increment(NoTradeReason.LOW_EQUITY, 1)
    
    # Test get_top_reasons
    top_reasons = tracker.get_top_reasons(3)
    assert len(top_reasons) == 3
    assert top_reasons[0] == ("RISK_BLOCKED", 10)
    assert top_reasons[1] == ("NO_SIGNAL", 5)
    assert top_reasons[2] == ("MIN_NOTIONAL_BLOCKED", 3)


def test_reason_tracker_main_culprit():
    """Test que se identifica correctamente el culpable principal."""
    tracker = ReasonTracker()
    
    # Test sin eventos
    assert tracker.get_main_culprit() is None
    
    # Test con culpable principal (>30%)
    tracker.increment(NoTradeReason.RISK_BLOCKED, 40)
    tracker.increment(NoTradeReason.NO_SIGNAL, 10)
    
    main_culprit = tracker.get_main_culprit()
    assert main_culprit is not None
    reason, count, percentage = main_culprit
    assert reason == "RISK_BLOCKED"
    assert count == 40
    assert percentage == 80.0  # 40/50 * 100
    
    # Test sin culpable principal (<30%)
    tracker.reset()
    tracker.increment(NoTradeReason.RISK_BLOCKED, 10)
    tracker.increment(NoTradeReason.NO_SIGNAL, 10)
    tracker.increment(NoTradeReason.MIN_NOTIONAL_BLOCKED, 10)
    
    assert tracker.get_main_culprit() is None


def test_reason_tracker_snapshot():
    """Test que se crean snapshots correctamente."""
    tracker = ReasonTracker()
    
    # Añadir eventos
    tracker.increment(NoTradeReason.RISK_BLOCKED, 5)
    tracker.increment(NoTradeReason.NO_SIGNAL, 3)
    
    # Crear snapshot
    snapshot = tracker.create_snapshot(1000)
    
    assert snapshot.timestamp == 1000
    assert snapshot.total_events == 8
    assert snapshot.reasons["RISK_BLOCKED"] == 5
    assert snapshot.reasons["NO_SIGNAL"] == 3
    assert len(snapshot.top_reasons) == 2
    assert snapshot.top_reasons[0] == ("RISK_BLOCKED", 5)
    assert snapshot.main_culprit == "RISK_BLOCKED"
    assert snapshot.main_culprit_pct == 62.5  # 5/8 * 100


def test_reason_tracker_reset():
    """Test que el reset funciona correctamente."""
    tracker = ReasonTracker()
    
    # Añadir eventos
    tracker.increment(NoTradeReason.RISK_BLOCKED, 5)
    tracker.create_snapshot(1000)
    
    # Reset
    tracker.reset()
    
    assert tracker.get_total_events() == 0
    assert tracker.get_count(NoTradeReason.RISK_BLOCKED) == 0
    assert len(tracker.get_snapshots()) == 0


def test_reason_tracker_export():
    """Test que la exportación funciona correctamente."""
    tracker = ReasonTracker()
    
    # Añadir eventos
    tracker.increment(NoTradeReason.RISK_BLOCKED, 5)
    tracker.increment(NoTradeReason.NO_SIGNAL, 3)
    tracker.create_snapshot(1000)
    
    # Exportar
    exported = tracker.export_to_dict()
    
    assert exported["total_events"] == 8
    assert exported["counters"]["RISK_BLOCKED"] == 5
    assert exported["counters"]["NO_SIGNAL"] == 3
    assert len(exported["top_reasons"]) == 2
    assert exported["main_culprit"][0] == "RISK_BLOCKED"
    assert exported["snapshots_count"] == 1


def test_telemetry_snapshot():
    """Test que TelemetrySnapshot funciona correctamente."""
    snapshot = TelemetrySnapshot(
        timestamp=1000,
        total_events=10,
        reasons={"RISK_BLOCKED": 7, "NO_SIGNAL": 3},
        top_reasons=[("RISK_BLOCKED", 7), ("NO_SIGNAL", 3)],
        main_culprit="RISK_BLOCKED",
        main_culprit_pct=70.0
    )
    
    assert snapshot.timestamp == 1000
    assert snapshot.total_events == 10
    assert snapshot.reasons["RISK_BLOCKED"] == 7
    assert snapshot.top_reasons[0] == ("RISK_BLOCKED", 7)
    assert snapshot.main_culprit == "RISK_BLOCKED"
    assert snapshot.main_culprit_pct == 70.0


def test_no_trade_reason_enum():
    """Test que el enum NoTradeReason tiene todos los valores esperados."""
    expected_reasons = [
        "COOLDOWN", "WARMUP", "NO_SIGNAL", "POLICY_NO_OPEN", "RISK_BLOCKED",
        "NO_SL_DISTANCE", "MIN_NOTIONAL_BLOCKED", "EXPOSURE_LIMIT", "CIRCUIT_BREAKER",
        "BROKER_EMPTY", "DONE_EARLY", "BANKRUPTCY_RESTART", "COOLDOWN_AFTER_RESET",
        "SHORTS_DISABLED", "LOW_EQUITY", "LEVERAGE_CAP", "MARGIN_INSUFFICIENT",
        "POSITION_ALREADY_OPEN", "INVALID_ACTION"
    ]
    
    for reason in expected_reasons:
        assert hasattr(NoTradeReason, reason), f"NoTradeReason.{reason} no existe"
        assert getattr(NoTradeReason, reason).value == reason


def test_reason_tracker_suggestions():
    """Test que se generan sugerencias correctas para cada razón."""
    tracker = ReasonTracker()
    
    # Test sugerencias para RISK_BLOCKED
    tracker.increment(NoTradeReason.RISK_BLOCKED, 50)
    main_culprit = tracker.get_main_culprit()
    assert main_culprit is not None
    reason, count, percentage = main_culprit
    assert reason == "RISK_BLOCKED"
    
    # Test sugerencias para NO_SIGNAL
    tracker.reset()
    tracker.increment(NoTradeReason.NO_SIGNAL, 50)
    main_culprit = tracker.get_main_culprit()
    assert main_culprit is not None
    reason, count, percentage = main_culprit
    assert reason == "NO_SIGNAL"
    
    # Test sugerencias para MIN_NOTIONAL_BLOCKED
    tracker.reset()
    tracker.increment(NoTradeReason.MIN_NOTIONAL_BLOCKED, 50)
    main_culprit = tracker.get_main_culprit()
    assert main_culprit is not None
    reason, count, percentage = main_culprit
    assert reason == "MIN_NOTIONAL_BLOCKED"


def test_reason_tracker_edge_cases():
    """Test casos edge del ReasonTracker."""
    tracker = ReasonTracker()
    
    # Test increment con count 0
    tracker.increment(NoTradeReason.RISK_BLOCKED, 0)
    assert tracker.get_count(NoTradeReason.RISK_BLOCKED) == 0
    assert tracker.get_total_events() == 0
    
    # Test increment con count negativo
    tracker.increment(NoTradeReason.RISK_BLOCKED, -5)
    assert tracker.get_count(NoTradeReason.RISK_BLOCKED) == -5
    assert tracker.get_total_events() == -5
    
    # Test get_top_reasons con limit 0
    top_reasons = tracker.get_top_reasons(0)
    assert len(top_reasons) == 0
    
    # Test get_top_reasons con limit mayor al número de razones
    tracker.increment(NoTradeReason.NO_SIGNAL, 1)
    top_reasons = tracker.get_top_reasons(10)
    assert len(top_reasons) == 2  # Solo 2 razones únicas


if __name__ == "__main__":
    pytest.main([__file__])
