#!/usr/bin/env python3
"""
Script de monitoreo del entrenamiento del agente de trading.
Muestra métricas en tiempo real y estadísticas de los runs.
"""

import json
import time
import os
from datetime import datetime
from pathlib import Path

def load_jsonl(file_path):
    """Carga un archivo JSONL y retorna una lista de diccionarios."""
    if not os.path.exists(file_path):
        return []
    
    data = []
    with open(file_path, 'r') as f:
        for line in f:
            if line.strip():
                try:
                    data.append(json.loads(line.strip()))
                except json.JSONDecodeError:
                    continue
    return data

def analyze_runs(runs_data):
    """Analiza los datos de runs y retorna estadísticas."""
    if not runs_data:
        return {
            "total_runs": 0,
            "avg_balance": 0,
            "avg_equity": 0,
            "trades_count": 0,
            "bankruptcy_rate": 0,
            "avg_drawdown": 0
        }
    
    total_runs = len(runs_data)
    balances = [run.get("final_balance", 0) for run in runs_data]
    equities = [run.get("final_equity", 0) for run in runs_data]
    trades_counts = [run.get("trades_count", 0) for run in runs_data]
    bankruptcies = [run.get("bankruptcy", False) for run in runs_data]
    drawdowns = [run.get("drawdown_pct", 0) for run in runs_data]
    
    return {
        "total_runs": total_runs,
        "avg_balance": sum(balances) / len(balances) if balances else 0,
        "avg_equity": sum(equities) / len(equities) if equities else 0,
        "avg_trades": sum(trades_counts) / len(trades_counts) if trades_counts else 0,
        "bankruptcy_rate": sum(bankruptcies) / len(bankruptcies) if bankruptcies else 0,
        "avg_drawdown": sum(drawdowns) / len(drawdowns) if drawdowns else 0,
        "max_balance": max(balances) if balances else 0,
        "min_balance": min(balances) if balances else 0
    }

def analyze_metrics(metrics_data):
    """Analiza las métricas de entrenamiento."""
    if not metrics_data:
        return {"iterations": 0, "total_timesteps": 0, "fps": 0}
    
    latest = metrics_data[-1]
    return {
        "iterations": latest.get("iterations", 0),
        "total_timesteps": latest.get("total_timesteps", 0),
        "fps": latest.get("fps", 0),
        "time_elapsed": latest.get("time_elapsed", 0)
    }

def main():
    """Función principal de monitoreo."""
    print("🔍 Monitoreo del Entrenamiento del Agente de Trading")
    print("=" * 60)
    
    # Rutas de archivos
    runs_file = "models/BTCUSDT/BTCUSDT_runs.jsonl"
    metrics_file = "models/BTCUSDT/BTCUSDT_train_metrics.jsonl"
    
    last_runs_count = 0
    last_metrics_timesteps = 0
    
    try:
        while True:
            # Cargar datos
            runs_data = load_jsonl(runs_file)
            metrics_data = load_jsonl(metrics_file)
            
            # Analizar datos
            runs_stats = analyze_runs(runs_data)
            metrics_stats = analyze_metrics(metrics_data)
            
            # Limpiar pantalla (en Windows)
            os.system('cls' if os.name == 'nt' else 'clear')
            
            # Mostrar header
            print("🔍 Monitoreo del Entrenamiento del Agente de Trading")
            print("=" * 60)
            print(f"⏰ {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            print()
            
            # Mostrar métricas de entrenamiento
            print("📊 MÉTRICAS DE ENTRENAMIENTO:")
            print(f"   • Iteraciones: {metrics_stats['iterations']}")
            print(f"   • Timesteps totales: {metrics_stats['total_timesteps']:,}")
            print(f"   • FPS: {metrics_stats['fps']:.1f}")
            if 'time_elapsed' in metrics_stats and metrics_stats['time_elapsed'] > 0:
                print(f"   • Tiempo transcurrido: {metrics_stats['time_elapsed']:.1f}s")
            else:
                print(f"   • Tiempo transcurrido: N/A")
            print()
            
            # Mostrar estadísticas de runs
            print("📈 ESTADÍSTICAS DE RUNS:")
            print(f"   • Total runs: {runs_stats['total_runs']}")
            print(f"   • Balance promedio: ${runs_stats['avg_balance']:.2f}")
            print(f"   • Equity promedio: ${runs_stats['avg_equity']:.2f}")
            print(f"   • Trades promedio: {runs_stats['avg_trades']:.1f}")
            print(f"   • Tasa de bancarrota: {runs_stats['bankruptcy_rate']:.1%}")
            print(f"   • Drawdown promedio: {runs_stats['avg_drawdown']:.2f}%")
            print(f"   • Balance máximo: ${runs_stats['max_balance']:.2f}")
            print(f"   • Balance mínimo: ${runs_stats['min_balance']:.2f}")
            print()
            
            # Mostrar progreso
            new_runs = runs_stats['total_runs'] - last_runs_count
            new_timesteps = metrics_stats['total_timesteps'] - last_metrics_timesteps
            
            if new_runs > 0:
                print(f"🆕 Nuevos runs: {new_runs}")
            if new_timesteps > 0:
                print(f"🆕 Nuevos timesteps: {new_timesteps:,}")
            
            print()
            print("💡 Presiona Ctrl+C para salir")
            
            # Actualizar contadores
            last_runs_count = runs_stats['total_runs']
            last_metrics_timesteps = metrics_stats['total_timesteps']
            
            # Esperar antes de la siguiente actualización
            time.sleep(5)
            
    except KeyboardInterrupt:
        print("\n\n👋 Monitoreo detenido por el usuario")
    except Exception as e:
        print(f"\n❌ Error durante el monitoreo: {e}")

if __name__ == "__main__":
    main()
