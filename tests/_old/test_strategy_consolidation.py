"""
Test para consolidación de estrategias TOP-1000.
"""
import pytest
import json
import tempfile
from pathlib import Path
from train_env.strategy_aggregator import StrategyAggregator


class TestStrategyConsolidation:
    def test_strategy_consolidation_top_k(self):
        """Test que la consolidación mantiene TOP-K estrategias."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Crear archivos de prueba
            provisional_path = Path(tmpdir) / "provisional.jsonl"
            final_path = Path(tmpdir) / "strategies.json"
            
            # Crear estrategias provisionales con diferentes scores
            provisional_strategies = [
                {
                    "kind": "OPEN",
                    "ts": 1000,
                    "price": 50000,
                    "qty": 0.1,
                    "side": "BUY",
                    "segment_id": 0,
                    "exec_tf": "1m",
                    "bars_held": 10,
                    "pnl": 100,
                    "leverage": 1.0
                },
                {
                    "kind": "CLOSE", 
                    "ts": 2000,
                    "price": 51000,
                    "qty": 0.1,
                    "side": "SELL",
                    "segment_id": 0,
                    "exec_tf": "1m",
                    "bars_held": 10,
                    "pnl": 100,
                    "leverage": 1.0
                },
                {
                    "kind": "BANKRUPTCY",
                    "ts": 3000,
                    "segment_id": 1,
                    "reason": "EQUITY_THRESHOLD"
                }
            ]
            
            # Escribir provisional
            with provisional_path.open("w") as f:
                for strategy in provisional_strategies:
                    f.write(json.dumps(strategy) + "\n")
            
            # Crear estrategias existentes
            existing_strategies = [
                {
                    "signature": "strategy_1",
                    "score": 0.5,
                    "pnl": 50,
                    "exec_tf": "1m",
                    "bars_held": 5
                },
                {
                    "signature": "strategy_2", 
                    "score": 0.3,
                    "pnl": 30,
                    "exec_tf": "5m",
                    "bars_held": 8
                }
            ]
            
            with final_path.open("w") as f:
                json.dump(existing_strategies, f)
            
            # Ejecutar consolidación
            aggregator = StrategyAggregator(
                provisional_path=str(provisional_path),
                final_path=str(final_path),
                top_k=3
            )
            
            result = aggregator.aggregate_top_k()
            
            # Verificar que se consolidaron las estrategias
            assert result["new_strategies"] > 0
            assert result["existing_strategies"] == 2
            assert result["total_after_merge"] >= 2
            
            # Verificar que el archivo final se actualizó
            assert final_path.exists()
            
            # Verificar que el provisional se limpió
            assert not provisional_path.exists() or provisional_path.stat().st_size == 0

    def test_strategy_scoring_futures(self):
        """Test que el scoring de futuros incluye leverage efficiency."""
        with tempfile.TemporaryDirectory() as tmpdir:
            provisional_path = Path(tmpdir) / "provisional.jsonl"
            final_path = Path(tmpdir) / "strategies.json"
            
            # Crear estrategia de futuros con leverage
            futures_strategy = [
                {
                    "kind": "OPEN",
                    "ts": 1000,
                    "price": 50000,
                    "qty": 0.1,
                    "side": "BUY",
                    "segment_id": 0,
                    "exec_tf": "1m",
                    "bars_held": 10,
                    "pnl": 100,
                    "leverage": 5.0  # Leverage alto
                },
                {
                    "kind": "CLOSE",
                    "ts": 2000,
                    "price": 51000,
                    "qty": 0.1,
                    "side": "SELL",
                    "segment_id": 0,
                    "exec_tf": "1m",
                    "bars_held": 10,
                    "pnl": 100,
                    "leverage": 5.0
                }
            ]
            
            with provisional_path.open("w") as f:
                for strategy in futures_strategy:
                    f.write(json.dumps(strategy) + "\n")
            
            # Crear archivo final vacío
            with final_path.open("w") as f:
                json.dump([], f)
            
            # Ejecutar consolidación
            aggregator = StrategyAggregator(
                provisional_path=str(provisional_path),
                final_path=str(final_path),
                top_k=10
            )
            
            result = aggregator.aggregate_top_k()
            
            # Verificar que se procesó la estrategia de futuros
            assert result["new_strategies"] > 0
            
            # Verificar que el archivo final contiene la estrategia
            with final_path.open("r") as f:
                final_strategies = json.load(f)
            
            assert len(final_strategies) > 0
            # Verificar que la estrategia tiene leverage
            assert any("leverage" in strategy for strategy in final_strategies)

    def test_bankruptcy_events_handling(self):
        """Test que los eventos de bancarrota se procesan correctamente."""
        with tempfile.TemporaryDirectory() as tmpdir:
            provisional_path = Path(tmpdir) / "provisional.jsonl"
            final_path = Path(tmpdir) / "strategies.json"
            
            # Crear evento de bancarrota
            bankruptcy_event = {
                "kind": "BANKRUPTCY",
                "ts": 3000,
                "segment_id": 1,
                "reason": "EQUITY_THRESHOLD"
            }
            
            with provisional_path.open("w") as f:
                f.write(json.dumps(bankruptcy_event) + "\n")
            
            # Crear archivo final vacío
            with final_path.open("w") as f:
                json.dump([], f)
            
            # Ejecutar consolidación
            aggregator = StrategyAggregator(
                provisional_path=str(provisional_path),
                final_path=str(final_path),
                top_k=10
            )
            
            result = aggregator.aggregate_top_k()
            
            # Verificar que se procesó el evento de bancarrota
            assert result["new_strategies"] >= 0  # Puede ser 0 si no hay estrategias válidas
            
            # Verificar que el provisional se limpió
            assert not provisional_path.exists() or provisional_path.stat().st_size == 0

    def test_segment_id_tracking(self):
        """Test que el segment_id se mantiene en las estrategias."""
        with tempfile.TemporaryDirectory() as tmpdir:
            provisional_path = Path(tmpdir) / "provisional.jsonl"
            final_path = Path(tmpdir) / "strategies.json"
            
            # Crear estrategia con segment_id
            strategy_with_segment = {
                "kind": "OPEN",
                "ts": 1000,
                "price": 50000,
                "qty": 0.1,
                "side": "BUY",
                "segment_id": 2,  # Segment ID específico
                "exec_tf": "1m",
                "bars_held": 10,
                "pnl": 100,
                "leverage": 1.0
            }
            
            with provisional_path.open("w") as f:
                f.write(json.dumps(strategy_with_segment) + "\n")
            
            # Crear archivo final vacío
            with final_path.open("w") as f:
                json.dump([], f)
            
            # Ejecutar consolidación
            aggregator = StrategyAggregator(
                provisional_path=str(provisional_path),
                final_path=str(final_path),
                top_k=10
            )
            
            result = aggregator.aggregate_top_k()
            
            # Verificar que se procesó la estrategia
            assert result["new_strategies"] > 0
            
            # Verificar que el archivo final contiene la estrategia
            with final_path.open("r") as f:
                final_strategies = json.load(f)
            
            assert len(final_strategies) > 0
            # Verificar que se mantiene el segment_id
            assert any("segment_id" in strategy for strategy in final_strategies)
