#!/usr/bin/env python3
"""
Script de monitoreo en tiempo real para el entrenamiento PPO
Muestra progreso, mejores runs y estadísticas en tiempo real
"""

import json
import time
import os
from pathlib import Path
from datetime import datetime
import threading

class TrainingMonitor:
    def __init__(self, symbol="BTCUSDT", models_root="models", refresh_interval=2):
        self.symbol = symbol
        self.models_root = Path(models_root)
        self.refresh_interval = refresh_interval
        self.runs_file = self.models_root / symbol / f"{symbol}_runs.jsonl"
        self.progress_file = self.models_root / symbol / f"{symbol}_progress.json"
        self.running = False
        
    def start_monitoring(self):
        """Inicia el monitoreo en tiempo real"""
        self.running = True
        print(f"🔍 MONITOREO INICIADO PARA {self.symbol}")
        print(f"📁 Archivo de runs: {self.runs_file}")
        print(f"⏱️  Actualización cada {self.refresh_interval} segundos")
        print("=" * 80)
        
        try:
            while self.running:
                self._display_status()
                time.sleep(self.refresh_interval)
        except KeyboardInterrupt:
            print("\n⏹️  Monitoreo detenido por el usuario")
            self.running = False
    
    def _display_status(self):
        """Muestra el estado actual del entrenamiento"""
        try:
            # Limpiar pantalla (Windows)
            os.system('cls' if os.name == 'nt' else 'clear')
            
            # Mostrar timestamp
            now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            print(f"🕐 {now} | 🔄 Monitoreando entrenamiento de {self.symbol}")
            print("=" * 80)
            
            # Cargar runs actuales
            runs = self._load_runs()
            
            if not runs:
                print("⏳ Esperando primeros runs...")
                return
            
            # Estadísticas generales
            total_runs = len(runs)
            equities = [float(r.get("final_equity", 0.0)) for r in runs]
            balances = [float(r.get("final_balance", 0.0)) for r in runs]
            
            print(f"📊 ESTADÍSTICAS GENERALES:")
            print(f"   Total runs: {total_runs}")
            print(f"   Mejor equity: {max(equities):.2f} USDT")
            print(f"   Peor equity: {min(equities):.2f} USDT")
            print(f"   Promedio equity: {sum(equities)/len(equities):.2f} USDT")
            print(f"   Mejor balance: {max(balances):.2f} USDT")
            print(f"   Peor balance: {min(balances):.2f} USDT")
            
            # Mejor run
            best_run = max(runs, key=lambda r: float(r.get("final_equity", 0.0)))
            print(f"\n🏆 MEJOR RUN:")
            print(f"   Equity: {best_run.get('final_equity', 'N/A')}")
            print(f"   Balance: {best_run.get('final_balance', 'N/A')}")
            print(f"   Estado: {best_run.get('run_result', 'N/A')}")
            
            # Últimos 5 runs
            print(f"\n🔄 ÚLTIMOS 5 RUNS:")
            print("-" * 80)
            for i, run in enumerate(runs[-5:], 1):
                equity = run.get("final_equity", 0.0)
                balance = run.get("final_balance", 0.0)
                result = run.get("run_result", "?")
                ts = run.get("ts_end", 0)
                
                time_str = "?"
                if ts:
                    try:
                        dt = datetime.fromtimestamp(ts / 1000)
                        time_str = dt.strftime("%H:%M:%S")
                    except:
                        pass
                
                # Color según rendimiento
                if equity > 1000:
                    status = "🟢"
                elif equity > 500:
                    status = "🟡"
                else:
                    status = "🔴"
                
                print(f"   {i}. {status} Equity: {equity:8.2f} | Balance: {balance:8.2f} | {result:12} | {time_str}")
            
            # Progreso hacia objetivo
            if runs:
                last_run = runs[-1]
                target = float(last_run.get("target_balance", 1000000))
                best_balance = max(balances)
                progress_pct = (best_balance / target) * 100 if target > 0 else 0
                
                print(f"\n🎯 PROGRESO HACIA OBJETIVO:")
                print(f"   Objetivo: {target:,.0f} USDT")
                print(f"   Mejor logrado: {best_balance:,.2f} USDT")
                print(f"   Progreso: {progress_pct:.2f}%")
                
                # Barra de progreso visual
                bar_length = 40
                filled_length = int(bar_length * progress_pct / 100)
                bar = "█" * filled_length + "░" * (bar_length - filled_length)
                print(f"   [{bar}] {progress_pct:.1f}%")
            
            # Información del archivo
            if self.runs_file.exists():
                file_size = self.runs_file.stat().st_size
                print(f"\n📁 ARCHIVO:")
                print(f"   Tamaño: {file_size:,} bytes")
                print(f"   Última modificación: {datetime.fromtimestamp(self.runs_file.stat().st_mtime).strftime('%H:%M:%S')}")
            
            print(f"\n⏱️  Próxima actualización en {self.refresh_interval} segundos... (Ctrl+C para detener)")
            
        except Exception as e:
            print(f"❌ Error en monitoreo: {e}")
    
    def _load_runs(self):
        """Carga los runs desde el archivo JSONL"""
        runs = []
        if not self.runs_file.exists():
            return runs
        
        try:
            with self.runs_file.open("r", encoding="utf-8") as f:
                for line in f:
                    try:
                        run = json.loads(line.strip())
                        runs.append(run)
                    except json.JSONDecodeError:
                        continue
        except Exception as e:
            print(f"❌ Error leyendo runs: {e}")
        
        return runs

def main():
    import os
    
    print("🚀 MONITOR DE ENTRENAMIENTO PPO")
    print("=" * 50)
    
    monitor = TrainingMonitor(
        symbol="BTCUSDT",
        models_root="models",
        refresh_interval=2
    )
    
    try:
        monitor.start_monitoring()
    except KeyboardInterrupt:
        print("\n👋 Monitoreo detenido")

if __name__ == "__main__":
    main()
