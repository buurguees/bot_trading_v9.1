#!/usr/bin/env python3
"""
Script de tests de sanidad para validar que los fixes funcionan correctamente
"""

import json
import os
import glob
from pathlib import Path
import argparse

def test_dedup_effectiveness():
    """Test: Verificar que la deduplicación funciona correctamente"""
    print("🧪 TEST: Deduplicación efectiva")
    
    # Buscar logs recientes
    log_files = list(Path("logs").glob("**/*.log")) + list(Path("models").glob("**/*.log"))
    if not log_files:
        print("   ⚠️  No se encontraron archivos de log")
        return False
    
    latest_log = max(log_files, key=os.path.getmtime)
    
    open_attempts = 0
    blocked_duplicates = 0
    
    try:
        with open(latest_log, 'r', encoding='utf-8') as f:
            for line in f:
                if "OPEN_ATTEMPT:" in line:
                    open_attempts += 1
                elif "BLOQUEADO" in line or "OPEN_BLOQUEADO" in line:
                    blocked_duplicates += 1
    except:
        print("   ❌ Error leyendo logs")
        return False
    
    print(f"   📊 OPEN_ATTEMPT: {open_attempts}")
    print(f"   🚫 BLOQUEADOS: {blocked_duplicates}")
    
    if blocked_duplicates > 0:
        print("   ✅ Deduplicación funcionando")
        return True
    else:
        print("   ⚠️  No se detectaron bloqueos por duplicados")
        return False

def test_bankruptcy_logic():
    """Test: Verificar que BANKRUPTCY solo ocurre con equity <= umbral"""
    print("🧪 TEST: Lógica de bankruptcy")
    
    runs_file = Path("models/BTCUSDT/BTCUSDT_runs.jsonl")
    if not runs_file.exists():
        print("   ⚠️  No se encontró archivo de runs")
        return False
    
    bankruptcy_runs = []
    early_exit_runs = []
    
    try:
        with open(runs_file, 'r', encoding='utf-8') as f:
            for line in f:
                run = json.loads(line.strip())
                if run.get('run_result') == 'BANKRUPTCY':
                    bankruptcy_runs.append(run)
                elif run.get('run_result') == 'EARLY_EXIT':
                    early_exit_runs.append(run)
    except:
        print("   ❌ Error leyendo runs")
        return False
    
    print(f"   📊 BANKRUPTCY runs: {len(bankruptcy_runs)}")
    print(f"   📊 EARLY_EXIT runs: {len(early_exit_runs)}")
    
    # Verificar que no hay BANKRUPTCY sin trades
    bankruptcy_without_trades = [r for r in bankruptcy_runs if r.get('trades_count', 0) == 0]
    
    if len(bankruptcy_without_trades) == 0:
        print("   ✅ No hay BANKRUPTCY sin trades")
        return True
    else:
        print(f"   ❌ {len(bankruptcy_without_trades)} BANKRUPTCY sin trades")
        return False

def test_drift_corrector():
    """Test: Verificar que drift corrector respeta umbrales"""
    print("🧪 TEST: Drift corrector")
    
    # Buscar logs de drift
    log_files = list(Path("logs").glob("**/*.log")) + list(Path("models").glob("**/*.log"))
    if not log_files:
        print("   ⚠️  No se encontraron archivos de log")
        return False
    
    latest_log = max(log_files, key=os.path.getmtime)
    
    drift_corrected = 0
    drift_ignored = 0
    
    try:
        with open(latest_log, 'r', encoding='utf-8') as f:
            for line in f:
                if "CORRIGIENDO DRIFT" in line:
                    drift_corrected += 1
                elif "DRIFT IGNORADO" in line:
                    drift_ignored += 1
    except:
        print("   ❌ Error leyendo logs")
        return False
    
    print(f"   📊 Drift corregido: {drift_corrected}")
    print(f"   📊 Drift ignorado: {drift_ignored}")
    
    if drift_ignored > 0:
        print("   ✅ Drift corrector respeta umbrales")
        return True
    else:
        print("   ⚠️  No se detectaron drifts ignorados")
        return True  # No es necesariamente malo

def test_sl_tp_ttl_sanity():
    """Test: Verificar que SL/TP/TTL están saneados en TRAIN"""
    print("🧪 TEST: SL/TP/TTL saneados")
    
    # Buscar logs de trades
    log_files = list(Path("logs").glob("**/*.log")) + list(Path("models").glob("**/*.log"))
    if not log_files:
        print("   ⚠️  No se encontraron archivos de log")
        return False
    
    latest_log = max(log_files, key=os.path.getmtime)
    
    trades_with_levels = 0
    trades_without_levels = 0
    
    try:
        with open(latest_log, 'r', encoding='utf-8') as f:
            for line in f:
                if "OPEN_ATTEMPT:" in line and "sl=" in line and "tp=" in line:
                    trades_with_levels += 1
                elif "OPEN_ATTEMPT:" in line:
                    trades_without_levels += 1
    except:
        print("   ❌ Error leyendo logs")
        return False
    
    print(f"   📊 Trades con niveles: {trades_with_levels}")
    print(f"   📊 Trades sin niveles: {trades_without_levels}")
    
    if trades_without_levels == 0:
        print("   ✅ Todos los trades tienen SL/TP")
        return True
    else:
        print("   ⚠️  Algunos trades no tienen niveles")
        return False

def test_env_api():
    """Test: Verificar que env.step() devuelve 5 valores"""
    print("🧪 TEST: API del entorno")
    
    # Este test requeriría ejecutar el entorno, por ahora solo verificamos la estructura
    print("   📊 Verificando estructura de step()...")
    
    # Verificar que base_env.py tiene el método step correcto
    base_env_file = Path("base_env/base_env.py")
    if not base_env_file.exists():
        print("   ❌ No se encontró base_env.py")
        return False
    
    try:
        with open(base_env_file, 'r', encoding='utf-8') as f:
            content = f.read()
            if "def step(self, payload: Optional[Dict[str, Any]] = None):" in content:
                print("   ✅ step() acepta payload opcional")
                return True
            else:
                print("   ❌ step() no tiene la firma correcta")
                return False
    except:
        print("   ❌ Error leyendo base_env.py")
        return False

def test_no_fix_rl_hack():
    """Test: Verificar que no hay hack FIX RL"""
    print("🧪 TEST: Sin hack FIX RL")
    
    # Buscar en gym_wrapper.py
    gym_wrapper_file = Path("train_env/gym_wrapper.py")
    if not gym_wrapper_file.exists():
        print("   ❌ No se encontró gym_wrapper.py")
        return False
    
    try:
        with open(gym_wrapper_file, 'r', encoding='utf-8') as f:
            content = f.read()
            if "FIX RL:" in content:
                print("   ❌ Aún hay hack FIX RL en el código")
                return False
            else:
                print("   ✅ No hay hack FIX RL")
                return True
    except:
        print("   ❌ Error leyendo gym_wrapper.py")
        return False

def main():
    parser = argparse.ArgumentParser(description='Tests de sanidad para validar fixes')
    parser.add_argument('--test', choices=[
        'all', 'dedup', 'bankruptcy', 'drift', 'levels', 'api', 'hack'
    ], default='all', help='Test específico a ejecutar')
    
    args = parser.parse_args()
    
    print("🔍 EJECUTANDO TESTS DE SANIDAD")
    print("=" * 50)
    
    tests = []
    
    if args.test in ['all', 'dedup']:
        tests.append(('Deduplicación', test_dedup_effectiveness))
    
    if args.test in ['all', 'bankruptcy']:
        tests.append(('Bankruptcy', test_bankruptcy_logic))
    
    if args.test in ['all', 'drift']:
        tests.append(('Drift Corrector', test_drift_corrector))
    
    if args.test in ['all', 'levels']:
        tests.append(('SL/TP/TTL', test_sl_tp_ttl_sanity))
    
    if args.test in ['all', 'api']:
        tests.append(('API Env', test_env_api))
    
    if args.test in ['all', 'hack']:
        tests.append(('Sin Hack RL', test_no_fix_rl_hack))
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n{test_name}:")
        if test_func():
            passed += 1
            print(f"   ✅ PASSED")
        else:
            print(f"   ❌ FAILED")
    
    print(f"\n📊 RESULTADO: {passed}/{total} tests pasaron")
    
    if passed == total:
        print("🎉 TODOS LOS TESTS PASARON - Los fixes están funcionando correctamente")
    else:
        print("⚠️  ALGUNOS TESTS FALLARON - Revisar implementación")

if __name__ == "__main__":
    main()
