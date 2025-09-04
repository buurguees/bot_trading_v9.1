#!/usr/bin/env python3
"""
Script para verificar el estado del entrenamiento.
"""

import json
import os
from pathlib import Path

def check_training_status():
    """Verifica el estado del entrenamiento."""
    
    print("🔍 VERIFICANDO ESTADO DEL ENTRENAMIENTO")
    print("=" * 50)
    
    # Verificar archivo de runs
    runs_file = Path("models/BTCUSDT/BTCUSDT_runs.jsonl")
    if runs_file.exists():
        with open(runs_file, 'r') as f:
            lines = f.readlines()
        
        total_runs = len([line for line in lines if line.strip()])
        print(f"📊 Total runs: {total_runs}")
        
        if total_runs > 0:
            # Leer el último run
            last_line = lines[-1].strip()
            if last_line:
                last_run = json.loads(last_line)
                print(f"📈 Último run:")
                print(f"   - Trades: {last_run.get('trades_count', 0)}")
                print(f"   - Steps: {last_run.get('elapsed_steps', 0)}")
                print(f"   - Final Equity: {last_run.get('final_equity', 0):.2f} USDT")
                print(f"   - Result: {last_run.get('run_result', 'UNKNOWN')}")
                
                reasons = last_run.get('reasons_counter', {})
                if reasons:
                    print(f"   - Razones de bloqueo:")
                    for reason, count in reasons.items():
                        print(f"     * {reason}: {count}")
    else:
        print("❌ No se encontró el archivo de runs")
    
    # Verificar si hay procesos de entrenamiento
    print(f"\n🔄 Procesos Python activos: {len([p for p in os.popen('tasklist | findstr python').readlines() if p.strip()])}")
    
    # Verificar logs recientes
    logs_dir = Path("logs/ppo_v1")
    if logs_dir.exists():
        log_files = list(logs_dir.glob("*.0"))
        print(f"📝 Archivos de log: {len(log_files)}")
        
        if log_files:
            latest_log = max(log_files, key=lambda x: x.stat().st_mtime)
            print(f"📄 Log más reciente: {latest_log.name}")
            print(f"⏰ Modificado: {latest_log.stat().st_mtime}")

if __name__ == "__main__":
    check_training_status()
