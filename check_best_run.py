#!/usr/bin/env python3
"""
Script para verificar el mejor run y diagnosticar problemas de tracking
"""

import json
from pathlib import Path
from datetime import datetime

def load_runs(runs_file: Path):
    """Carga todos los runs desde el archivo JSONL"""
    rows = []
    if not runs_file.exists():
        print(f"❌ Archivo no encontrado: {runs_file}")
        return rows
    
    try:
        with runs_file.open("r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                try:
                    row = json.loads(line.strip())
                    rows.append(row)
                except json.JSONDecodeError as e:
                    print(f"⚠️ Error en línea {line_num}: {e}")
                    continue
        print(f"✅ Cargados {len(rows)} runs exitosamente")
    except Exception as e:
        print(f"❌ Error leyendo archivo: {e}")
    
    return rows

def find_best_run(runs: list) -> dict | None:
    """Encuentra el run con mejor rendimiento (mayor equity final)"""
    if not runs:
        return None
    
    # Buscar el run con mayor equity final (mejor rendimiento)
    best_run = max(runs, key=lambda r: float(r.get("final_equity", 0.0)))
    return best_run

def analyze_runs(runs: list):
    """Analiza los runs y muestra estadísticas"""
    if not runs:
        print("❌ No hay runs para analizar")
        return
    
    print(f"\n📊 ANÁLISIS DE {len(runs)} RUNS:")
    print("=" * 60)
    
    # Estadísticas básicas
    equities = [float(r.get("final_equity", 0.0)) for r in runs]
    balances = [float(r.get("final_balance", 0.0)) for r in runs]
    
    print(f"💰 Equity - Min: {min(equities):.2f}, Max: {max(equities):.2f}, Promedio: {sum(equities)/len(equities):.2f}")
    print(f"💳 Balance - Min: {min(balances):.2f}, Max: {max(balances):.2f}, Promedio: {sum(balances)/len(balances):.2f}")
    
    # Encontrar mejor run
    best_run = find_best_run(runs)
    if best_run:
        print(f"\n🏆 MEJOR RUN:")
        print(f"   Equity: {best_run.get('final_equity', 'N/A')}")
        print(f"   Balance: {best_run.get('final_balance', 'N/A')}")
        print(f"   Run #: {best_run.get('run_number', 'N/A')}")
        
        # Timestamp
        ts = best_run.get("ts_end", 0)
        if ts:
            try:
                dt = datetime.fromtimestamp(ts / 1000)
                print(f"   Fecha: {dt.strftime('%Y-%m-%d %H:%M:%S')}")
            except:
                print(f"   Timestamp: {ts}")
    
    # Mostrar últimos 5 runs
    print(f"\n🔄 ÚLTIMOS 5 RUNS:")
    print("-" * 60)
    for i, run in enumerate(runs[-5:], 1):
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

def main():
    # Configurar paths
    symbol = "BTCUSDT"
    models_root = Path("models")
    runs_file = models_root / symbol / f"{symbol}_runs.jsonl"
    
    print(f"🔍 VERIFICANDO RUNS PARA {symbol}")
    print(f"📁 Archivo: {runs_file}")
    print("=" * 60)
    
    # Cargar runs
    runs = load_runs(runs_file)
    
    if runs:
        # Analizar runs
        analyze_runs(runs)
        
        # Verificar si hay runs positivos
        positive_runs = [r for r in runs if float(r.get("final_equity", 0.0)) > 1000.0]
        if positive_runs:
            print(f"\n✅ ENCONTRADOS {len(positive_runs)} RUNS POSITIVOS!")
            best_positive = max(positive_runs, key=lambda r: float(r.get("final_equity", 0.0)))
            print(f"🏆 Mejor run positivo: Equity {best_positive.get('final_equity', 'N/A')}")
        else:
            print(f"\n⚠️ NO HAY RUNS POSITIVOS (todos por debajo de 1000 USDT)")
    else:
        print("❌ No se pudieron cargar los runs")

if __name__ == "__main__":
    main()
