#!/usr/bin/env python3
"""
Script de diagnóstico detallado para entender por qué no se ejecutan trades.
"""

import json
import sys
import os
from pathlib import Path

def debug_detailed():
    """Diagnóstico detallado del problema."""
    
    print("🔍 DIAGNÓSTICO DETALLADO")
    print("=" * 50)
    
    # 1. Verificar configuración
    print("1. VERIFICANDO CONFIGURACIÓN:")
    
    # Risk config
    risk_file = Path("config/risk.yaml")
    if risk_file.exists():
        with open(risk_file, 'r', encoding='utf-8') as f:
            content = f.read()
            if "risk_pct_per_trade: 3.0" in content:
                print("   ✅ risk_pct_per_trade: 3.0% (futures)")
            else:
                print("   ❌ risk_pct_per_trade no está en 3.0%")
            
            if "train_force_min_notional: true" in content:
                print("   ✅ train_force_min_notional: true")
            else:
                print("   ❌ train_force_min_notional no está habilitado")
            
            if "default_levels:" in content:
                print("   ✅ default_levels configurado")
            else:
                print("   ❌ default_levels no configurado")
    
    # 2. Verificar fix del gym_wrapper
    print("\n2. VERIFICANDO FIX DEL GYM_WRAPPER:")
    gym_file = Path("train_env/gym_wrapper.py")
    if gym_file.exists():
        with open(gym_file, 'r', encoding='utf-8') as f:
            content = f.read()
            if "FIX TEMPORAL: Si RL envía action=0" in content:
                print("   ✅ Fix del gym_wrapper aplicado")
            else:
                print("   ❌ Fix del gym_wrapper NO aplicado")
    
    # 3. Verificar símbolo y filtros
    print("\n3. VERIFICANDO SÍMBOLO Y FILTROS:")
    symbols_file = Path("config/symbols.yaml")
    if symbols_file.exists():
        with open(symbols_file, 'r', encoding='utf-8') as f:
            content = f.read()
            if "BTCUSDT:" in content:
                print("   ✅ BTCUSDT configurado")
                # Buscar minNotional
                lines = content.split('\n')
                in_btcusdt = False
                for line in lines:
                    if "BTCUSDT:" in line:
                        in_btcusdt = True
                    elif in_btcusdt and line.strip().startswith('-'):
                        break
                    elif in_btcusdt and "minNotional:" in line:
                        print(f"   ✅ minNotional: {line.split(':')[1].strip()}")
                        break
            else:
                print("   ❌ BTCUSDT no configurado")
    
    # 4. Verificar logs recientes
    print("\n4. VERIFICANDO LOGS RECIENTES:")
    runs_file = Path("models/BTCUSDT/BTCUSDT_runs.jsonl")
    if runs_file.exists():
        with open(runs_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        if lines:
            last_run = json.loads(lines[-1])
            reasons = last_run.get('reasons_counter', {})
            
            print(f"   📊 Último run:")
            print(f"      - Trades: {last_run.get('trades_count', 0)}")
            print(f"      - Steps: {last_run.get('elapsed_steps', 0)}")
            print(f"      - Result: {last_run.get('run_result', 'UNKNOWN')}")
            
            if reasons:
                print(f"      - Razones principales:")
                sorted_reasons = sorted(reasons.items(), key=lambda x: x[1], reverse=True)
                for reason, count in sorted_reasons[:3]:
                    print(f"        * {reason}: {count}")
    
    # 5. Verificar si hay procesos de entrenamiento
    print("\n5. VERIFICANDO PROCESOS:")
    try:
        import subprocess
        result = subprocess.run(['tasklist', '/FI', 'IMAGENAME eq python.exe'], 
                              capture_output=True, text=True)
        python_processes = len([line for line in result.stdout.split('\n') if 'python.exe' in line])
        print(f"   🔄 Procesos Python activos: {python_processes}")
        
        if python_processes > 0:
            print("   ✅ Entrenamiento ejecutándose")
        else:
            print("   ❌ No hay entrenamiento ejecutándose")
    except:
        print("   ⚠️ No se pudo verificar procesos")
    
    # 6. Recomendaciones
    print("\n6. RECOMENDACIONES:")
    print("   🔧 Si el problema persiste:")
    print("      1. Verificar que el RL esté enviando acciones != 0")
    print("      2. Revisar logs de consola para mensajes de fix")
    print("      3. Verificar que no haya errores en el modelo")
    print("      4. Considerar aumentar risk_pct_per_trade temporalmente")
    print("      5. Verificar que los filtros del símbolo sean correctos")

if __name__ == "__main__":
    debug_detailed()
