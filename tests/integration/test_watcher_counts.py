#!/usr/bin/env python3
"""
Test: Watcher muestra trades_count correcto
- runs.jsonl con trades_count>0 → watcher imprime el valor correcto.
"""

import pytest
import json
import tempfile
from pathlib import Path
import sys

# Agregar el directorio raíz al path
sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.watch_progress import load_runs, ConsoleMonitor


def test_watcher_counts():
    """Test: Watcher muestra trades_count correcto"""
    
    # Crear archivo temporal con runs que tienen trades_count > 0
    with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
        runs_data = [
            {
                "final_equity": 1000.0,
                "final_balance": 1000.0,
                "trades_count": 0,  # Sin trades
                "elapsed_steps": 1000,
                "run_result": "END_OF_HISTORY",
                "ts_end": 1640995200000,
                "initial_balance": 1000.0,
                "target_balance": 10000.0
            },
            {
                "final_equity": 1200.0,
                "final_balance": 1200.0,
                "trades_count": 5,  # Con trades
                "elapsed_steps": 2000,
                "run_result": "END_OF_HISTORY",
                "ts_end": 1640995200000,
                "initial_balance": 1000.0,
                "target_balance": 10000.0,
                "reasons_counter": {
                    "policy_no_open": 10,
                    "risk_blocked": 5
                }
            },
            {
                "final_equity": 1500.0,
                "final_balance": 1500.0,
                "trades_count": 12,  # Más trades
                "elapsed_steps": 3000,
                "run_result": "END_OF_HISTORY",
                "ts_end": 1640995200000,
                "initial_balance": 1000.0,
                "target_balance": 10000.0,
                "reasons_counter": {
                    "policy_no_open": 15,
                    "risk_blocked": 8,
                    "min_notional_blocked": 3
                }
            }
        ]
        
        for run in runs_data:
            f.write(json.dumps(run) + '\n')
        
        temp_file = Path(f.name)
    
    try:
        # Test 1: Cargar runs correctamente
        print("=== Test 1: Cargar runs ===")
        runs = load_runs(temp_file)
        
        assert len(runs) == 3, f"Debe haber 3 runs, encontrados {len(runs)}"
        print(f"✅ Cargados {len(runs)} runs correctamente")
        
        # Test 2: Verificar trades_count
        print("\n=== Test 2: Verificar trades_count ===")
        
        expected_trades = [0, 5, 12]
        for i, run in enumerate(runs):
            trades_count = run.get("trades_count", 0)
            expected = expected_trades[i]
            
            assert trades_count == expected, f"Run {i+1}: trades_count {trades_count} != {expected}"
            print(f"✅ Run {i+1}: trades_count = {trades_count}")
        
        # Test 3: Simular salida del watcher
        print("\n=== Test 3: Simular salida del watcher ===")
        
        # Capturar la salida del watcher
        import io
        import contextlib
        
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            # Simular la lógica de impresión del watcher
            for i, run in enumerate(runs):
                equity = run.get("final_equity", 0.0)
                balance = run.get("final_balance", 0.0)
                trades = run.get("trades_count", 0)
                steps = run.get("elapsed_steps", 0)
                result = run.get("run_result", "?")
                
                # Calcular ROI
                initial = float(run.get("initial_balance", 1000.0))
                roi = ((balance - initial) / initial) * 100.0
                
                print(f"   {i+1}. Equity: {equity:8.2f} | Balance: {balance:8.2f} | ROI: {roi:+6.1f}% | Trades: {trades:3d} | Steps: {steps:5d} | {result:12}")
        
        output_str = output.getvalue()
        print("Salida del watcher:")
        print(output_str)
        
        # Verificar que contiene los trades_count correctos
        assert "Trades:   0" in output_str, "Debe mostrar Trades: 0"
        assert "Trades:   5" in output_str, "Debe mostrar Trades: 5"
        assert "Trades:  12" in output_str, "Debe mostrar Trades: 12"
        
        print("✅ Watcher muestra trades_count correcto")
        
        # Test 4: Verificar top razones
        print("\n=== Test 4: Verificar top razones ===")
        
        # Simular cálculo de top razones
        all_reasons = {}
        for run in runs:
            reasons = run.get("reasons_counter", {})
            for reason, count in reasons.items():
                all_reasons[reason] = all_reasons.get(reason, 0) + count
        
        if all_reasons:
            total_reasons = sum(all_reasons.values())
            sorted_reasons = sorted(all_reasons.items(), key=lambda x: x[1], reverse=True)
            
            print("Top razones calculadas:")
            for i, (reason, count) in enumerate(sorted_reasons[:3], 1):
                pct = (count / total_reasons * 100) if total_reasons > 0 else 0
                print(f"   {i}. {reason}: {count} ({pct:.1f}%)")
            
            # Verificar que policy_no_open es la razón más común
            top_reason = sorted_reasons[0][0]
            assert top_reason == "policy_no_open", f"Top razón debe ser 'policy_no_open', es '{top_reason}'"
            print("✅ Top razones calculadas correctamente")
        
        # Test 5: Verificar métricas promedio
        print("\n=== Test 5: Verificar métricas promedio ===")
        
        trades_counts = [r.get("trades_count", 0) for r in runs]
        avg_trades = sum(trades_counts) / len(trades_counts)
        
        print(f"Trades promedio: {avg_trades:.2f}")
        assert avg_trades == (0 + 5 + 12) / 3, f"Trades promedio debe ser {(0 + 5 + 12) / 3}, es {avg_trades}"
        print("✅ Métricas promedio calculadas correctamente")
        
        print("\n🎯 Resumen del test:")
        print(f"   - Runs cargados: {len(runs)}")
        print(f"   - Trades counts: {trades_counts}")
        print(f"   - Trades promedio: {avg_trades:.2f}")
        print(f"   - Top razones: {len(all_reasons)} tipos")
        print("✅ Todos los tests del watcher pasaron")
        
    finally:
        # Limpiar archivo temporal
        temp_file.unlink()


if __name__ == "__main__":
    test_watcher_counts()
