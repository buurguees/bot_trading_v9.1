#!/usr/bin/env python3
"""
Script para monitorear las acciones que envía el RL en tiempo real.
"""

import time
import json
from pathlib import Path

def monitor_actions():
    """Monitorea las acciones del RL en tiempo real."""
    
    print("👁️ MONITOREANDO ACCIONES DEL RL")
    print("=" * 40)
    print("Presiona Ctrl+C para detener")
    print()
    
    # Archivo de runs para monitorear
    runs_file = Path("models/BTCUSDT/BTCUSDT_runs.jsonl")
    
    if not runs_file.exists():
        print("❌ No se encontró el archivo de runs")
        return
    
    # Leer el número inicial de runs
    with open(runs_file, 'r') as f:
        initial_runs = len([line for line in f if line.strip()])
    
    print(f"📊 Runs iniciales: {initial_runs}")
    print("🔍 Monitoreando nuevos runs...")
    print()
    
    try:
        while True:
            # Leer el número actual de runs
            with open(runs_file, 'r') as f:
                current_runs = len([line for line in f if line.strip()])
            
            # Si hay nuevos runs, analizarlos
            if current_runs > initial_runs:
                print(f"🆕 Nuevos runs detectados: {current_runs - initial_runs}")
                
                # Leer los últimos runs
                with open(runs_file, 'r') as f:
                    lines = f.readlines()
                
                # Analizar los nuevos runs
                for i in range(initial_runs, current_runs):
                    if i < len(lines) and lines[i].strip():
                        run = json.loads(lines[i])
                        print(f"  Run {i+1}:")
                        print(f"    - Trades: {run.get('trades_count', 0)}")
                        print(f"    - Steps: {run.get('elapsed_steps', 0)}")
                        print(f"    - Result: {run.get('run_result', 'UNKNOWN')}")
                        
                        reasons = run.get('reasons_counter', {})
                        if reasons:
                            print(f"    - Razones:")
                            for reason, count in reasons.items():
                                print(f"      * {reason}: {count}")
                        print()
                
                initial_runs = current_runs
            
            time.sleep(5)  # Verificar cada 5 segundos
            
    except KeyboardInterrupt:
        print("\n⏹️ Monitoreo detenido")

if __name__ == "__main__":
    monitor_actions()
