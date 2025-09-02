#!/usr/bin/env python3
"""
Script de emergencia para limpiar runs duplicados y corregir el archivo runs.jsonl
Elimina runs con valores idénticos y mantiene solo los únicos
"""

import json
from pathlib import Path
from datetime import datetime

def clean_duplicate_runs():
    """Limpia runs duplicados y mantiene solo los únicos"""
    runs_file = Path("models/BTCUSDT/BTCUSDT_runs.jsonl")
    
    if not runs_file.exists():
        print("❌ Archivo de runs no encontrado")
        return
    
    print("🧹 LIMPIANDO RUNS DUPLICADOS...")
    
    # Cargar todos los runs
    runs = []
    with runs_file.open("r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            try:
                run = json.loads(line.strip())
                runs.append(run)
            except json.JSONDecodeError as e:
                print(f"⚠️ Error en línea {line_num}: {e}")
                continue
    
    print(f"📊 Total runs cargados: {len(runs)}")
    
    # Eliminar duplicados basándose en múltiples criterios
    unique_runs = []
    seen_combinations = set()
    
    for run in runs:
        # Crear clave única para identificar duplicados
        balance = float(run.get("final_balance", 0.0))
        equity = float(run.get("final_equity", 0.0))
        ts_end = run.get("ts_end", 0)
        run_result = run.get("run_result", "UNKNOWN")
        
        # Clave más estricta para detectar duplicados
        key = f"{balance:.2f}_{equity:.2f}_{run_result}"
        
        if key not in seen_combinations:
            seen_combinations.add(key)
            unique_runs.append(run)
        else:
            print(f"🗑️ Run duplicado eliminado: Balance {balance:.2f}, Equity {equity:.2f}, Result {run_result}")
    
    print(f"✅ Runs únicos encontrados: {len(unique_runs)}")
    print(f"🗑️ Runs duplicados eliminados: {len(runs) - len(unique_runs)}")
    
    # Ordenar por timestamp de inicio (más antiguos primero)
    unique_runs.sort(key=lambda r: r.get("ts_start", 0))
    
    # Reescribir archivo con runs únicos
    with runs_file.open("w", encoding="utf-8") as f:
        for run in unique_runs:
            f.write(json.dumps(run, ensure_ascii=False) + "\n")
    
    print(f"💾 Archivo limpiado y guardado: {runs_file}")
    
    # Mostrar estadísticas finales
    if unique_runs:
        equities = [float(r.get("final_equity", 0.0)) for r in unique_runs]
        balances = [float(r.get("final_balance", 0.0)) for r in unique_runs]
        
        print(f"\n📊 ESTADÍSTICAS FINALES:")
        print(f"   Total runs únicos: {len(unique_runs)}")
        print(f"   Mejor equity: {max(equities):.2f}")
        print(f"   Peor equity: {min(equities):.2f}")
        print(f"   Mejor balance: {max(balances):.2f}")
        print(f"   Peor balance: {min(balances):.2f}")
        
        # Mostrar últimos 3 runs
        print(f"\n🔄 ÚLTIMOS 3 RUNS:")
        for i, run in enumerate(unique_runs[-3:], 1):
            equity = run.get("final_equity", 0.0)
            balance = run.get("final_balance", 0.0)
            ts = run.get("ts_end", 0)
            
            time_str = "?"
            if ts:
                try:
                    dt = datetime.fromtimestamp(ts / 1000)
                    time_str = dt.strftime("%H:%M:%S")
                except:
                    pass
            
            print(f"   {i}. Equity: {equity:8.2f} | Balance: {balance:8.2f} | {time_str}")

if __name__ == "__main__":
    clean_duplicate_runs()
