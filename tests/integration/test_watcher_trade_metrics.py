#!/usr/bin/env python3
"""
Test: Métricas de trades en el watcher
- runs.jsonl con trades_count>0 → watcher muestra trades_count y avg_holding_time
"""

import pytest
import json
import tempfile
from pathlib import Path
import sys

# Agregar el directorio raíz al path
sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.watch_progress import load_runs


def test_watcher_trade_metrics():
    """Test: Watcher debe mostrar métricas de calidad de trades"""
    
    # Crear archivo temporal de runs.jsonl con datos de prueba
    with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
        test_runs = [
            {
                "ts_end": 1743465600000,
                "final_balance": 1050.0,
                "final_equity": 1050.0,
                "initial_balance": 1000.0,
                "target_balance": 1000000.0,
                "trades_count": 3,
                "avg_holding_time": 15.5,
                "trades_with_sl_tp": 2,
                "elapsed_steps": 1000,
                "run_result": "completed",
                "bankruptcy": False,
                "reasons_counter": {
                    "NO_SL_DISTANCE": 5,
                    "MIN_NOTIONAL_BLOCKED": 2
                }
            },
            {
                "ts_end": 1743469200000,
                "final_balance": 980.0,
                "final_equity": 980.0,
                "initial_balance": 1000.0,
                "target_balance": 1000000.0,
                "trades_count": 1,
                "avg_holding_time": 8.0,
                "trades_with_sl_tp": 1,
                "elapsed_steps": 800,
                "run_result": "completed",
                "bankruptcy": False,
                "reasons_counter": {
                    "NO_SL_DISTANCE": 3,
                    "MIN_NOTIONAL_BLOCKED": 1
                }
            },
            {
                "ts_end": 1743472800000,
                "final_balance": 500.0,
                "final_equity": 500.0,
                "initial_balance": 1000.0,
                "target_balance": 1000000.0,
                "trades_count": 0,
                "avg_holding_time": 0.0,
                "trades_with_sl_tp": 0,
                "elapsed_steps": 500,
                "run_result": "bankruptcy",
                "bankruptcy": True,
                "reasons_counter": {
                    "NO_SL_DISTANCE": 10,
                    "MIN_NOTIONAL_BLOCKED": 5
                }
            }
        ]
        
        for run in test_runs:
            f.write(json.dumps(run) + '\n')
        
        temp_file = Path(f.name)
    
    try:
        # Test 1: Cargar runs correctamente
        print("\n=== Test 1: Cargar runs con métricas de trades ===")
        
        runs = load_runs(temp_file)
        assert len(runs) == 3, f"Debe cargar 3 runs, pero cargó {len(runs)}"
        print(f"✅ Cargados {len(runs)} runs correctamente")
        
        # Test 2: Verificar métricas de trades en cada run
        print("\n=== Test 2: Verificar métricas de trades ===")
        
        run1 = runs[0]
        assert run1["trades_count"] == 3, f"Run 1 debe tener 3 trades, pero tiene {run1['trades_count']}"
        assert run1["avg_holding_time"] == 15.5, f"Run 1 debe tener avg_holding_time=15.5, pero tiene {run1['avg_holding_time']}"
        assert run1["trades_with_sl_tp"] == 2, f"Run 1 debe tener 2 trades con SL/TP, pero tiene {run1['trades_with_sl_tp']}"
        print("✅ Run 1: trades_count=3, avg_holding_time=15.5, trades_with_sl_tp=2")
        
        run2 = runs[1]
        assert run2["trades_count"] == 1, f"Run 2 debe tener 1 trade, pero tiene {run2['trades_count']}"
        assert run2["avg_holding_time"] == 8.0, f"Run 2 debe tener avg_holding_time=8.0, pero tiene {run2['avg_holding_time']}"
        assert run2["trades_with_sl_tp"] == 1, f"Run 2 debe tener 1 trade con SL/TP, pero tiene {run2['trades_with_sl_tp']}"
        print("✅ Run 2: trades_count=1, avg_holding_time=8.0, trades_with_sl_tp=1")
        
        run3 = runs[2]
        assert run3["trades_count"] == 0, f"Run 3 debe tener 0 trades, pero tiene {run3['trades_count']}"
        assert run3["avg_holding_time"] == 0.0, f"Run 3 debe tener avg_holding_time=0.0, pero tiene {run3['avg_holding_time']}"
        assert run3["trades_with_sl_tp"] == 0, f"Run 3 debe tener 0 trades con SL/TP, pero tiene {run3['trades_with_sl_tp']}"
        print("✅ Run 3: trades_count=0, avg_holding_time=0.0, trades_with_sl_tp=0")
        
        # Test 3: Verificar cálculo de porcentaje SL/TP
        print("\n=== Test 3: Verificar cálculo de porcentaje SL/TP ===")
        
        # Run 1: 2/3 = 66.7%
        sl_tp_pct_1 = (run1["trades_with_sl_tp"] / max(run1["trades_count"], 1)) * 100.0
        assert abs(sl_tp_pct_1 - 66.7) < 0.1, f"Run 1 debe tener ~66.7% SL/TP, pero tiene {sl_tp_pct_1:.1f}%"
        print(f"✅ Run 1: {sl_tp_pct_1:.1f}% trades con SL/TP")
        
        # Run 2: 1/1 = 100%
        sl_tp_pct_2 = (run2["trades_with_sl_tp"] / max(run2["trades_count"], 1)) * 100.0
        assert sl_tp_pct_2 == 100.0, f"Run 2 debe tener 100% SL/TP, pero tiene {sl_tp_pct_2:.1f}%"
        print(f"✅ Run 2: {sl_tp_pct_2:.1f}% trades con SL/TP")
        
        # Test 4: Verificar razones de bloqueo
        print("\n=== Test 4: Verificar razones de bloqueo ===")
        
        # Agregar todas las razones de los últimos 3 runs
        all_reasons = {}
        for run in runs:
            reasons = run.get("reasons_counter", {})
            for reason, count in reasons.items():
                all_reasons[reason] = all_reasons.get(reason, 0) + count
        
        assert "NO_SL_DISTANCE" in all_reasons, "Debe incluir NO_SL_DISTANCE en razones"
        assert "MIN_NOTIONAL_BLOCKED" in all_reasons, "Debe incluir MIN_NOTIONAL_BLOCKED en razones"
        
        # NO_SL_DISTANCE: 5 + 3 + 10 = 18
        assert all_reasons["NO_SL_DISTANCE"] == 18, f"NO_SL_DISTANCE debe ser 18, pero es {all_reasons['NO_SL_DISTANCE']}"
        # MIN_NOTIONAL_BLOCKED: 2 + 1 + 5 = 8
        assert all_reasons["MIN_NOTIONAL_BLOCKED"] == 8, f"MIN_NOTIONAL_BLOCKED debe ser 8, pero es {all_reasons['MIN_NOTIONAL_BLOCKED']}"
        
        print(f"✅ NO_SL_DISTANCE: {all_reasons['NO_SL_DISTANCE']}")
        print(f"✅ MIN_NOTIONAL_BLOCKED: {all_reasons['MIN_NOTIONAL_BLOCKED']}")
        
        # Test 5: Verificar cálculo de porcentajes de bloqueo
        print("\n=== Test 5: Verificar cálculo de porcentajes de bloqueo ===")
        
        total_reasons = sum(all_reasons.values())
        no_sl_pct = (all_reasons["NO_SL_DISTANCE"] / total_reasons) * 100.0
        min_notional_pct = (all_reasons["MIN_NOTIONAL_BLOCKED"] / total_reasons) * 100.0
        
        # Total: 18 + 8 = 26
        # NO_SL_DISTANCE: 18/26 = 69.2%
        # MIN_NOTIONAL_BLOCKED: 8/26 = 30.8%
        assert abs(no_sl_pct - 69.2) < 0.1, f"NO_SL_DISTANCE debe ser ~69.2%, pero es {no_sl_pct:.1f}%"
        assert abs(min_notional_pct - 30.8) < 0.1, f"MIN_NOTIONAL_BLOCKED debe ser ~30.8%, pero es {min_notional_pct:.1f}%"
        
        print(f"✅ NO_SL_DISTANCE: {no_sl_pct:.1f}%")
        print(f"✅ MIN_NOTIONAL_BLOCKED: {min_notional_pct:.1f}%")
        
    finally:
        # Limpiar archivo temporal
        temp_file.unlink()


if __name__ == "__main__":
    test_watcher_trade_metrics()
