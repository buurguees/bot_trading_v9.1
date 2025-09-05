#!/usr/bin/env python3
"""
Test para validar la retención FIFO de runs en RunLogger.
"""
import tempfile
import json
from pathlib import Path
from base_env.logging.run_logger import RunLogger


def test_fifo_retention():
    """Test que verifica que RunLogger mantiene solo max_records runs usando FIFO"""
    with tempfile.TemporaryDirectory() as temp_dir:
        # Crear RunLogger con límite pequeño para testing
        logger = RunLogger(
            symbol="TEST",
            models_root=temp_dir,
            max_records=3,  # Solo 3 runs
            prune_strategy="fifo"
        )
        
        # Crear 5 runs (más que el límite)
        for i in range(5):
            logger.start(
                market="futures",
                initial_balance=1000.0,
                target_balance=10000.0,
                initial_equity=1000.0,
                ts_start=1000000 + i * 1000
            )
            
            # Simular algunos trades
            logger.update_trades_count(i + 1)
            logger.update_elapsed_steps(1000 + i * 100)  # Más pasos para evitar filtro
            
            # Finalizar run
            logger.finish(
                final_balance=1000.0 + i * 100,
                final_equity=1000.0 + i * 100,
                ts_end=1000000 + i * 1000 + 60000  # 60 segundos de duración
            )
        
        # Verificar que solo se mantienen 3 runs (los últimos 3)
        runs_file = Path(temp_dir) / "TEST" / "TEST_runs.jsonl"
        assert runs_file.exists(), "Archivo de runs debe existir"
        
        runs = []
        with runs_file.open("r", encoding="utf-8") as f:
            for line in f:
                runs.append(json.loads(line))
        
        # Debe tener exactamente 3 runs
        assert len(runs) == 3, f"Debe tener exactamente 3 runs, tiene {len(runs)}"
        
        # Los runs deben ser los últimos 3 (i=2, 3, 4)
        expected_balances = [1200.0, 1300.0, 1400.0]  # 1000 + i*100 para i=2,3,4
        actual_balances = [run["final_balance"] for run in runs]
        
        assert actual_balances == expected_balances, f"Balances incorrectos: {actual_balances} vs {expected_balances}"


def test_no_retention_when_under_limit():
    """Test que verifica que no se eliminan runs cuando están bajo el límite"""
    with tempfile.TemporaryDirectory() as temp_dir:
        logger = RunLogger(
            symbol="TEST",
            models_root=temp_dir,
            max_records=5,  # Límite de 5
            prune_strategy="fifo"
        )
        
        # Crear solo 3 runs (bajo el límite)
        for i in range(3):
            logger.start(
                market="futures",
                initial_balance=1000.0,
                target_balance=10000.0,
                initial_equity=1000.0,
                ts_start=1000000 + i * 1000
            )
            logger.update_trades_count(i + 1)
            logger.update_elapsed_steps(1000 + i * 100)
            logger.finish(
                final_balance=1000.0 + i * 100,
                final_equity=1000.0 + i * 100,
                ts_end=1000000 + i * 1000 + 60000
            )
        
        # Verificar que se mantienen todos los runs
        runs_file = Path(temp_dir) / "TEST" / "TEST_runs.jsonl"
        runs = []
        with runs_file.open("r", encoding="utf-8") as f:
            for line in f:
                runs.append(json.loads(line))
        
        assert len(runs) == 3, f"Debe tener exactamente 3 runs, tiene {len(runs)}"


def test_jsonl_integrity():
    """Test que verifica que la integridad del JSONL se mantiene después de la retención"""
    with tempfile.TemporaryDirectory() as temp_dir:
        logger = RunLogger(
            symbol="TEST",
            models_root=temp_dir,
            max_records=2,
            prune_strategy="fifo"
        )
        
        # Crear 4 runs
        for i in range(4):
            logger.start(
                market="futures",
                initial_balance=1000.0,
                target_balance=10000.0,
                initial_equity=1000.0,
                ts_start=1000000 + i * 1000
            )
            logger.update_trades_count(i + 1)
            logger.update_elapsed_steps(1000 + i * 100)
            logger.finish(
                final_balance=1000.0 + i * 100,
                final_equity=1000.0 + i * 100,
                ts_end=1000000 + i * 1000 + 60000
            )
        
        # Verificar integridad del JSONL
        runs_file = Path(temp_dir) / "TEST" / "TEST_runs.jsonl"
        runs = []
        with runs_file.open("r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                try:
                    run_data = json.loads(line)
                    runs.append(run_data)
                except json.JSONDecodeError as e:
                    assert False, f"Error de JSON en línea {line_num}: {e}"
        
        # Debe tener exactamente 2 runs válidos
        assert len(runs) == 2, f"Debe tener exactamente 2 runs válidos, tiene {len(runs)}"
        
        # Cada run debe tener los campos requeridos
        for run in runs:
            required_fields = ["symbol", "market", "initial_balance", "final_balance", "trades_count", "elapsed_steps"]
            for field in required_fields:
                assert field in run, f"Campo requerido '{field}' faltante en run"


if __name__ == "__main__":
    test_fifo_retention()
    test_no_retention_when_under_limit()
    test_jsonl_integrity()
    print("✅ Todos los tests de retención de runs pasaron correctamente")
