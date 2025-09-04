#!/usr/bin/env python3
"""
Test para validar que el leverage se incluye en los logs de trades y agregaciones.
"""
import tempfile
import json
from pathlib import Path
from base_env.logging.run_logger import RunLogger
from base_env.metrics.trade_metrics import TradeRecord, TradeMetrics


def test_trade_record_includes_leverage():
    """Test que verifica que TradeRecord incluye leverage_used"""
    # Crear un TradeRecord con leverage
    trade = TradeRecord(
        entry_price=50000.0,
        exit_price=51000.0,
        qty=0.1,
        side=1,  # LONG
        realized_pnl=100.0,
        bars_held=10,
        leverage_used=5.0,  # ← NUEVO: leverage usado
        open_ts=1000000,
        close_ts=1000100,
        sl=49000.0,
        tp=52000.0,
        roi_pct=2.0,
        r_multiple=1.0,
        risk_pct=2.0
    )
    
    # Verificar que el leverage se guarda correctamente
    assert trade.leverage_used == 5.0, f"Leverage debe ser 5.0, es {trade.leverage_used}"


def test_trade_metrics_calculates_leverage_stats():
    """Test que verifica que TradeMetrics calcula estadísticas de leverage"""
    metrics = TradeMetrics()
    
    # Añadir trades con diferentes leverages
    trades = [
        TradeRecord(
            entry_price=50000.0, exit_price=51000.0, qty=0.1, side=1,
            realized_pnl=100.0, bars_held=10, leverage_used=3.0,
            open_ts=1000000, close_ts=1000100, sl=49000.0, tp=52000.0,
            roi_pct=2.0, r_multiple=1.0, risk_pct=2.0
        ),
        TradeRecord(
            entry_price=50000.0, exit_price=49000.0, qty=0.1, side=1,
            realized_pnl=-100.0, bars_held=5, leverage_used=10.0,
            open_ts=1000000, close_ts=1000100, sl=49000.0, tp=52000.0,
            roi_pct=-2.0, r_multiple=-1.0, risk_pct=2.0
        ),
        TradeRecord(
            entry_price=50000.0, exit_price=52000.0, qty=0.1, side=1,
            realized_pnl=200.0, bars_held=15, leverage_used=15.0,
            open_ts=1000000, close_ts=1000100, sl=49000.0, tp=52000.0,
            roi_pct=4.0, r_multiple=2.0, risk_pct=2.0
        )
    ]
    
    for trade in trades:
        metrics.add_trade(trade)
    
    # Calcular métricas
    result = metrics.calculate_metrics()
    
    # Verificar métricas de leverage
    assert result["avg_leverage"] == 9.33, f"Avg leverage debe ser ~9.33, es {result['avg_leverage']:.2f}"
    assert result["min_leverage"] == 3.0, f"Min leverage debe ser 3.0, es {result['min_leverage']}"
    assert result["max_leverage"] == 15.0, f"Max leverage debe ser 15.0, es {result['max_leverage']}"
    assert result["high_leverage_trades"] == 1, f"High leverage trades debe ser 1 (solo 15.0 > 10.0), es {result['high_leverage_trades']}"
    assert result["high_leverage_pct"] == 33.33, f"High leverage % debe ser ~33.33%, es {result['high_leverage_pct']:.2f}%"


def test_run_logger_includes_leverage_in_trades():
    """Test que verifica que RunLogger incluye leverage en el registro de trades"""
    with tempfile.TemporaryDirectory() as temp_dir:
        logger = RunLogger(
            symbol="TEST",
            models_root=temp_dir,
            max_records=10
        )
        
        # Iniciar run
        logger.start(
            market="futures",
            initial_balance=10000.0,
            target_balance=100000.0,
            initial_equity=10000.0,
            ts_start=1000000
        )
        
        # Simular algunos pasos para evitar filtro de episodio vacío
        logger.update_trades_count(2)
        logger.update_elapsed_steps(1000)
        
        # Registrar trades con leverage
        logger.add_trade_record(
            entry_price=50000.0,
            exit_price=51000.0,
            qty=0.1,
            side=1,
            realized_pnl=100.0,
            bars_held=10,
            leverage_used=5.0,  # ← NUEVO: leverage usado
            open_ts=1000000,
            close_ts=1000100,
            sl=49000.0,
            tp=52000.0,
            roi_pct=2.0,
            r_multiple=1.0,
            risk_pct=2.0
        )
        
        logger.add_trade_record(
            entry_price=50000.0,
            exit_price=49000.0,
            qty=0.1,
            side=1,
            realized_pnl=-100.0,
            bars_held=5,
            leverage_used=10.0,  # ← NUEVO: leverage usado
            open_ts=1000000,
            close_ts=1000100,
            sl=49000.0,
            tp=52000.0,
            roi_pct=-2.0,
            r_multiple=-1.0,
            risk_pct=2.0
        )
        
        # Finalizar run
        logger.finish(
            final_balance=10000.0,
            final_equity=10000.0,
            ts_end=1000000 + 60000  # 60 segundos de duración
        )
        
        # Verificar que el run incluye métricas de leverage
        runs_file = Path(temp_dir) / "TEST" / "TEST_runs.jsonl"
        assert runs_file.exists(), "Archivo de runs debe existir"
        
        runs = []
        with runs_file.open("r", encoding="utf-8") as f:
            for line in f:
                runs.append(json.loads(line))
        
        assert len(runs) == 1, f"Debe tener exactamente 1 run, tiene {len(runs)}"
        
        run = runs[0]
        
        # Verificar métricas de leverage
        assert "avg_leverage" in run, "Run debe incluir avg_leverage"
        assert "min_leverage" in run, "Run debe incluir min_leverage"
        assert "max_leverage" in run, "Run debe incluir max_leverage"
        assert "high_leverage_trades" in run, "Run debe incluir high_leverage_trades"
        assert "high_leverage_pct" in run, "Run debe incluir high_leverage_pct"
        
        # Verificar valores
        assert run["avg_leverage"] == 7.5, f"Avg leverage debe ser 7.5, es {run['avg_leverage']}"
        assert run["min_leverage"] == 5.0, f"Min leverage debe ser 5.0, es {run['min_leverage']}"
        assert run["max_leverage"] == 10.0, f"Max leverage debe ser 10.0, es {run['max_leverage']}"
        assert run["high_leverage_trades"] == 0, f"High leverage trades debe ser 0 (10.0 no es > 10.0), es {run['high_leverage_trades']}"
        assert run["high_leverage_pct"] == 0.0, f"High leverage % debe ser 0.0%, es {run['high_leverage_pct']}%"


def test_leverage_aggregation_edge_cases():
    """Test que verifica casos edge en la agregación de leverage"""
    metrics = TradeMetrics()
    
    # Test 1: Sin trades
    result = metrics.calculate_metrics()
    assert result["avg_leverage"] == 0.0, f"Sin trades, avg leverage debe ser 0.0, es {result['avg_leverage']}"
    assert result["min_leverage"] == 0.0, f"Sin trades, min leverage debe ser 0.0, es {result['min_leverage']}"
    assert result["max_leverage"] == 0.0, f"Sin trades, max leverage debe ser 0.0, es {result['max_leverage']}"
    assert result["high_leverage_trades"] == 0, f"Sin trades, high leverage trades debe ser 0, es {result['high_leverage_trades']}"
    assert result["high_leverage_pct"] == 0.0, f"Sin trades, high leverage % debe ser 0.0%, es {result['high_leverage_pct']}%"
    
    # Test 2: Solo trades con leverage 1.0 (spot)
    trade = TradeRecord(
        entry_price=50000.0, exit_price=51000.0, qty=0.1, side=1,
        realized_pnl=100.0, bars_held=10, leverage_used=1.0,
        open_ts=1000000, close_ts=1000100, sl=49000.0, tp=52000.0,
        roi_pct=2.0, r_multiple=1.0, risk_pct=2.0
    )
    metrics.add_trade(trade)
    
    result = metrics.calculate_metrics()
    assert result["avg_leverage"] == 1.0, f"Con leverage 1.0, avg debe ser 1.0, es {result['avg_leverage']}"
    assert result["min_leverage"] == 1.0, f"Con leverage 1.0, min debe ser 1.0, es {result['min_leverage']}"
    assert result["max_leverage"] == 1.0, f"Con leverage 1.0, max debe ser 1.0, es {result['max_leverage']}"
    assert result["high_leverage_trades"] == 0, f"Con leverage 1.0, high leverage trades debe ser 0, es {result['high_leverage_trades']}"
    assert result["high_leverage_pct"] == 0.0, f"Con leverage 1.0, high leverage % debe ser 0.0%, es {result['high_leverage_pct']}%"


if __name__ == "__main__":
    test_trade_record_includes_leverage()
    test_trade_metrics_calculates_leverage_stats()
    test_run_logger_includes_leverage_in_trades()
    test_leverage_aggregation_edge_cases()
    print("✅ Todos los tests de leverage en logs de trades pasaron correctamente")
