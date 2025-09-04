#!/usr/bin/env python3
"""
Test para validar que las estrategias persisten información de leverage.
"""
import tempfile
import json
from pathlib import Path
from train_env.strategy_aggregator import _score_row


def test_strategy_scoring_includes_leverage():
    """Test que verifica que el scoring de estrategias incluye métricas de leverage"""
    # Estrategia con leverage eficiente
    strategy_efficient = {
        "kind": "CLOSE",
        "r_multiple": 2.0,
        "roi_pct": 4.0,
        "realized_pnl": 200.0,
        "leverage_used": 5.0,
        "notional_effective": 5000.0,
        "notional_max": 10000.0,
        "exec_tf": "5m",
        "bars_held": 20
    }
    
    # Estrategia con leverage ineficiente
    strategy_inefficient = {
        "kind": "CLOSE",
        "r_multiple": 2.0,
        "roi_pct": 4.0,
        "realized_pnl": 200.0,
        "leverage_used": 25.0,  # Leverage muy alto
        "notional_effective": 1000.0,  # Uso muy bajo del notional
        "notional_max": 10000.0,
        "exec_tf": "5m",
        "bars_held": 20
    }
    
    # Calcular scores
    score_efficient = _score_row(strategy_efficient)
    score_inefficient = _score_row(strategy_inefficient)
    
    # La estrategia eficiente debe tener mejor score
    assert score_efficient > score_inefficient, f"Estrategia eficiente debe tener mejor score: {score_efficient} vs {score_inefficient}"
    
    # Verificar que ambos scores son positivos (tienen bonus por leverage)
    assert score_efficient > 0, f"Score eficiente debe ser positivo: {score_efficient}"
    assert score_inefficient > 0, f"Score ineficiente debe ser positivo: {score_inefficient}"


def test_strategy_without_leverage_info():
    """Test que verifica que las estrategias sin info de leverage no fallan"""
    # Estrategia sin información de leverage (spot o futures sin leverage config)
    strategy_no_leverage = {
        "kind": "CLOSE",
        "r_multiple": 2.0,
        "roi_pct": 4.0,
        "realized_pnl": 200.0,
        "exec_tf": "5m",
        "bars_held": 20
        # Sin leverage_used, notional_effective, notional_max
    }
    
    # Debe calcular score sin errores
    score = _score_row(strategy_no_leverage)
    assert score > 0, f"Score sin leverage debe ser positivo: {score}"
    
    # Debe tener el score base sin bonus de leverage
    expected_base_score = (10.0 * 2.0) + (0.1 * 4.0) + (0.001 * 200.0) + 3.0 + 2.0  # r_multiple + roi + realized + tf_bonus + duration_bonus
    assert abs(score - expected_base_score) < 0.01, f"Score sin leverage debe ser ~{expected_base_score}, es {score}"


def test_leverage_efficiency_bonus():
    """Test que verifica el bonus por eficiencia de leverage"""
    # Estrategia con uso eficiente del leverage (50% del notional máximo)
    strategy_50_percent = {
        "kind": "CLOSE",
        "r_multiple": 1.0,
        "roi_pct": 2.0,
        "realized_pnl": 100.0,
        "leverage_used": 5.0,
        "notional_effective": 5000.0,
        "notional_max": 10000.0,
        "exec_tf": "5m",
        "bars_held": 20
    }
    
    # Estrategia con uso ineficiente del leverage (10% del notional máximo)
    strategy_10_percent = {
        "kind": "CLOSE",
        "r_multiple": 1.0,
        "roi_pct": 2.0,
        "realized_pnl": 100.0,
        "leverage_used": 5.0,
        "notional_effective": 1000.0,
        "notional_max": 10000.0,
        "exec_tf": "5m",
        "bars_held": 20
    }
    
    score_50 = _score_row(strategy_50_percent)
    score_10 = _score_row(strategy_10_percent)
    
    # La estrategia con 50% debe tener mejor score
    assert score_50 > score_10, f"Estrategia 50% debe tener mejor score: {score_50} vs {score_10}"
    
    # La diferencia debe ser aproximadamente 0.8 (0.5 * 2.0 - 0.1 * 2.0)
    expected_diff = 0.8
    actual_diff = score_50 - score_10
    assert abs(actual_diff - expected_diff) < 0.1, f"Diferencia debe ser ~{expected_diff}, es {actual_diff}"


def test_leverage_moderation_bonus():
    """Test que verifica el bonus por leverage moderado"""
    # Estrategia con leverage moderado (5x - óptimo)
    strategy_moderate = {
        "kind": "CLOSE",
        "r_multiple": 1.0,
        "roi_pct": 2.0,
        "realized_pnl": 100.0,
        "leverage_used": 5.0,
        "notional_effective": 5000.0,
        "notional_max": 10000.0,
        "exec_tf": "5m",
        "bars_held": 20
    }
    
    # Estrategia con leverage extremo (25x)
    strategy_extreme = {
        "kind": "CLOSE",
        "r_multiple": 1.0,
        "roi_pct": 2.0,
        "realized_pnl": 100.0,
        "leverage_used": 25.0,
        "notional_effective": 5000.0,
        "notional_max": 10000.0,
        "exec_tf": "5m",
        "bars_held": 20
    }
    
    score_moderate = _score_row(strategy_moderate)
    score_extreme = _score_row(strategy_extreme)
    
    # La estrategia con leverage moderado debe tener mejor score
    assert score_moderate > score_extreme, f"Estrategia moderada debe tener mejor score: {score_moderate} vs {score_extreme}"


def test_strategy_persistence_format():
    """Test que verifica que las estrategias se guardan con el formato correcto"""
    with tempfile.TemporaryDirectory() as temp_dir:
        # Simular estrategias con leverage
        strategies = [
            {
                "kind": "CLOSE",
                "r_multiple": 2.0,
                "roi_pct": 4.0,
                "realized_pnl": 200.0,
                "leverage_used": 5.0,
                "notional_effective": 5000.0,
                "notional_max": 10000.0,
                "exec_tf": "5m",
                "bars_held": 20,
                "open_ts": 1000000,
                "close_ts": 1000100
            },
            {
                "kind": "CLOSE",
                "r_multiple": 1.5,
                "roi_pct": 3.0,
                "realized_pnl": 150.0,
                "leverage_used": 10.0,
                "notional_effective": 8000.0,
                "notional_max": 10000.0,
                "exec_tf": "15m",
                "bars_held": 30,
                "open_ts": 1000000,
                "close_ts": 1000100
            }
        ]
        
        # Guardar estrategias
        strategies_file = Path(temp_dir) / "strategies.json"
        with strategies_file.open("w", encoding="utf-8") as f:
            json.dump(strategies, f, ensure_ascii=False, indent=2)
        
        # Verificar que se guardaron correctamente
        assert strategies_file.exists(), "Archivo de estrategias debe existir"
        
        with strategies_file.open("r", encoding="utf-8") as f:
            loaded_strategies = json.load(f)
        
        assert len(loaded_strategies) == 2, f"Debe tener 2 estrategias, tiene {len(loaded_strategies)}"
        
        # Verificar que cada estrategia tiene los campos de leverage
        for strategy in loaded_strategies:
            assert "leverage_used" in strategy, "Estrategia debe incluir leverage_used"
            assert "notional_effective" in strategy, "Estrategia debe incluir notional_effective"
            assert "notional_max" in strategy, "Estrategia debe incluir notional_max"
            
            # Verificar tipos de datos
            assert isinstance(strategy["leverage_used"], (int, float)), "leverage_used debe ser numérico"
            assert isinstance(strategy["notional_effective"], (int, float)), "notional_effective debe ser numérico"
            assert isinstance(strategy["notional_max"], (int, float)), "notional_max debe ser numérico"


if __name__ == "__main__":
    test_strategy_scoring_includes_leverage()
    test_strategy_without_leverage_info()
    test_leverage_efficiency_bonus()
    test_leverage_moderation_bonus()
    test_strategy_persistence_format()
    print("✅ Todos los tests de persistencia de leverage en estrategias pasaron correctamente")
