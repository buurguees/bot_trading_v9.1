#!/usr/bin/env python3
"""
Ventana de escritorio con pestañas por símbolo para ver gráficos de Equity/Balance por run.
- Lee símbolos de config/symbols.yaml (modos train_*) y también descubre por models/*
- Cada pestaña refresca sola leyendo models/{symbol}/{symbol}_runs.jsonl
- Línea discontinua = objetivo (target_balance)
- Puedes abrirla en paralelo al entrenamiento
- Muestra el mejor run en la parte superior
- NUEVO: Monitoreo de consola integrado

Uso:
  python scripts/watch_progress.py
Opciones:
  --symbols BTCUSDT,ETHUSDT   # forzar lista (coma-separada)
  --models-root models
  --symbols-yaml config/symbols.yaml
  --refresh 2                 # segundos
  --y-scale linear|log
  --window 0                  # si >0, últimas N runs
  --console                   # NUEVO: habilitar monitoreo de consola
"""

from __future__ import annotations
import argparse, json, time, threading
from pathlib import Path
import tkinter as tk
from tkinter import ttk
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# ---------- utilidades ----------
def load_runs(path: Path):
    rows = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            try:
                rows.append(json.loads(line))
            except Exception:
                continue
    return rows

def discover_symbols(models_root: Path, symbols_yaml: Path | None) -> list[str]:
    found = set()
    # 1) YAML (mira modos train_*)
    if symbols_yaml and symbols_yaml.exists():
        import yaml
        try:
            raw = yaml.safe_load(symbols_yaml.read_text(encoding="utf-8")) or {}
            for it in raw.get("symbols", []):
                mode = (it.get("mode") or "").strip()
                if mode.startswith("train"):
                    s = (it.get("symbol") or it.get("name") or "").strip()
                    if s:
                        found.add(s)
        except Exception:
            pass
    # 2) models/* con runs.jsonl
    if models_root.exists():
        for p in models_root.iterdir():
            if not p.is_dir(): continue
            s = p.name
            runs = p / f"{s}_runs.jsonl"
            if runs.exists():
                found.add(s)
    return sorted(found)

def find_best_run(runs: list) -> dict | None:
    """Encuentra el run con mejor rendimiento (mayor equity final)"""
    if not runs:
        return None
    
    # Buscar el run con mayor equity final (mejor rendimiento)
    best_run = max(runs, key=lambda r: float(r.get("final_equity", 0.0)))
    return best_run

# ---------- pestaña por símbolo ----------
class SymbolTab:
    def __init__(self, parent: ttk.Notebook, symbol: str, models_root: Path, refresh: float, yscale: str, window: int):
        self.symbol = symbol
        self.models_root = models_root
        self.refresh = float(refresh)
        self.yscale = yscale
        self.window = int(window)
        self.runs_file = models_root / symbol / f"{symbol}_runs.jsonl"

        self.frame = ttk.Frame(parent)
        parent.add(self.frame, text=symbol)

        # ← NUEVO: Sección del mejor run en la parte superior
        best_run_frame = ttk.LabelFrame(self.frame, text="🏆 Mejor Run", padding=(8, 6))
        best_run_frame.pack(side=tk.TOP, fill=tk.X, padx=8, pady=(6, 0))
        
        # Información del mejor run
        self.lbl_best_run = ttk.Label(best_run_frame, text="Esperando datos...", font=("Arial", 10, "bold"))
        self.lbl_best_run.pack(side=tk.LEFT)
        
        # Separador
        ttk.Separator(self.frame, orient="horizontal").pack(side=tk.TOP, fill=tk.X, pady=4)

        # barra info
        top = ttk.Frame(self.frame)
        top.pack(side=tk.TOP, fill=tk.X, padx=8, pady=6)
        self.lbl_runs = ttk.Label(top, text="Runs: 0")
        self.lbl_runs.pack(side=tk.LEFT)
        self.lbl_last = ttk.Label(top, text="Última: -")
        self.lbl_last.pack(side=tk.LEFT, padx=12)
        
        # indicador de estado de actualización
        self.lbl_status = ttk.Label(top, text="⏳ Esperando...", foreground="blue")
        self.lbl_status.pack(side=tk.RIGHT)
        
        # botón de actualización manual
        self.btn_refresh = ttk.Button(top, text="🔄 Manual", command=self._manual_refresh)
        self.btn_refresh.pack(side=tk.RIGHT, padx=(8, 0))

        # ← NUEVO: Sección del run actual en la parte inferior
        current_run_frame = ttk.LabelFrame(self.frame, text="🔄 Run Actual", padding=(8, 6))
        current_run_frame.pack(side=tk.BOTTOM, fill=tk.X, padx=8, pady=(0, 6))
        
        # Información del run actual
        self.lbl_current_run = ttk.Label(current_run_frame, text="Esperando run actual...", font=("Arial", 9))
        self.lbl_current_run.pack(side=tk.LEFT)
        
        # Separador antes del gráfico
        ttk.Separator(self.frame, orient="horizontal").pack(side=tk.BOTTOM, fill=tk.X, pady=4)

        # figura
        self.fig = plt.Figure(figsize=(8,4), dpi=100)
        self.ax = self.fig.add_subplot(111)
        self.ax.grid(True, alpha=0.3)
        self.ax.set_yscale(self.yscale)
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        # hilo de refresco
        self._stop = False
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()

    def stop(self):
        self._stop = True
    
    def _manual_refresh(self):
        """Actualización manual del gráfico"""
        try:
            runs = load_runs(self.runs_file)
            self._draw(runs)
            self.lbl_status.config(text=f"🔄 Manual {time.strftime('%H:%M:%S')} | Auto-resume en 1s", foreground="orange")
        except Exception as e:
            self.lbl_status.config(text=f"❌ Error: {str(e)}", foreground="red")

    def _loop(self):
        last_len = -1
        last_content = ""
        while not self._stop:
            # detectar cambios en contenido, no solo en longitud
            try:
                current_content = self.runs_file.read_text(encoding="utf-8") if self.runs_file.exists() else ""
                runs = load_runs(self.runs_file)
                
                # repintar si cambió el número de runs O el contenido
                if len(runs) != last_len or current_content != last_content:
                    last_len = len(runs)
                    last_content = current_content
                    self._draw(runs)
                    # actualizar indicador de estado
                    self.lbl_status.config(text=f"🔄 Auto-actualizado {time.strftime('%H:%M:%S')} | Refresh: {self.refresh}s", foreground="green")
                else:
                    # mostrar estado de espera
                    self.lbl_status.config(text=f"⏳ Auto-refresh {time.strftime('%H:%M:%S')} | Refresh: {self.refresh}s", foreground="blue")
            except Exception as e:
                # si hay error, esperar y continuar
                pass
            
            time.sleep(self.refresh)

    def _draw(self, runs):
        try:
            # ← NUEVO: Actualizar información del mejor run
            self._update_best_run_info(runs)
            
            self.ax.clear()
            self.ax.grid(True, alpha=0.3)
            self.ax.set_yscale(self.yscale)
            
            print(f"🔍 Debug: _draw() ejecutándose con {len(runs)} runs")
        except Exception as e:
            print(f"❌ Error en _draw(): {e}")
            self.lbl_status.config(text=f"❌ Error gráfico: {str(e)}", foreground="red")
            return

        if not runs:
            self.ax.set_title(f"{self.symbol} — esperando primeras runs…")
            self.ax.set_xlabel("Run #"); self.ax.set_ylabel("USDT")
            self.canvas.draw()
            self.lbl_runs.config(text="Runs: 0")
            self.lbl_last.config(text="Última: -")
            return

        # ventana
        view = runs[-self.window:] if self.window > 0 else runs
        x0 = len(runs) - len(view) + 1
        xs = list(range(x0, x0 + len(view)))
        
        print(f"🔍 Debug: Preparando datos - {len(view)} runs, xs={xs[:5]}...")

        y_eq = [float(r.get("final_equity", 0.0)) for r in view]
        y_bal = [float(r.get("final_balance", 0.0)) for r in view]
        target = float(view[-1].get("target_balance", 0.0)) if view else 0.0
        
        print(f"🔍 Debug: Datos preparados - y_eq={y_eq[:5]}, y_bal={y_bal[:5]}, target={target}")

        self.ax.plot(xs, y_eq, marker="o", label="Equity")
        self.ax.plot(xs, y_bal, marker="x", label="Balance")
        if target > 0:
            self.ax.axhline(target, linestyle="--", color="red", label=f"Objetivo {target:.0f}")

        # anotación último punto
        self.ax.annotate(
            f"Run {xs[-1]}\nEq {y_eq[-1]:.2f}\nBal {y_bal[-1]:.2f}",
            xy=(xs[-1], y_eq[-1]),
            xytext=(6,6), textcoords="offset points"
        )

        self.ax.set_title(f"{self.symbol} — runs={len(runs)}")
        self.ax.set_xlabel("Run #"); self.ax.set_ylabel("USDT")
        self.ax.legend()
        self.fig.tight_layout()
        self.canvas.draw()

        self.lbl_runs.config(text=f"Runs: {len(runs)}")
        self.lbl_last.config(text=f"Última: Equity {y_eq[-1]:.2f}, Balance {y_bal[-1]:.2f}")
        
        # mostrar timestamp del último run
        if runs:
            last_ts = runs[-1].get("ts_end", 0)
            if last_ts:
                try:
                    from datetime import datetime
                    dt = datetime.fromtimestamp(last_ts / 1000)  # Convertir de ms a s
                    time_str = dt.strftime("%H:%M:%S")
                    self.lbl_status.config(text=f"🕐 Último run: {time_str}", foreground="black")
                except:
                    pass
        
        print(f"✅ Debug: Gráfico dibujado exitosamente para {len(runs)} runs")
        
        # ← NUEVO: Actualizar información del run actual
        self._update_current_run_info(runs)

    def _update_current_run_info(self, runs: list):
        """Actualiza la información del run actual en la parte inferior"""
        if not runs:
            self.lbl_current_run.config(text="Esperando run actual...", foreground="gray")
            return
        
        # Obtener el último run (actual)
        current_run = runs[-1]
        
        # Extraer información del run actual
        current_balance = float(current_run.get("final_balance", 0.0))
        current_equity = float(current_run.get("final_equity", 0.0))
        current_run_num = len(runs)  # Número del run actual
        current_ts = current_run.get("ts_end", 0)
        current_drawdown = float(current_run.get("drawdown_pct", 0.0))
        
        # Formatear timestamp
        time_str = "?"
        if current_ts:
            try:
                from datetime import datetime
                dt = datetime.fromtimestamp(current_ts / 1000)
                time_str = dt.strftime("%d/%m %H:%M")
            except:
                pass
        
        # Calcular ganancia del run actual
        initial_balance = float(current_run.get("initial_balance", 1000.0))
        gain_pct = ((current_balance - initial_balance) / initial_balance) * 100.0
        
        # Color según rendimiento
        if gain_pct > 0:
            color = "green"
            gain_symbol = "📈"
        elif gain_pct < 0:
            color = "red"
            gain_symbol = "📉"
        else:
            color = "black"
            gain_symbol = "➖"
        
        # Texto del run actual
        current_text = f"{gain_symbol} Run #{current_run_num} | Balance: {current_balance:.2f} | Equity: {current_equity:.2f} | Ganancia: {gain_pct:+.1f}% | Drawdown: {current_drawdown:.1f}% | {time_str}"
        self.lbl_current_run.config(text=current_text, foreground=color)

    def _update_best_run_info(self, runs: list):
        """Actualiza la información del mejor run en la parte superior"""
        best_run = find_best_run(runs)
        
        if not best_run:
            self.lbl_best_run.config(text="Esperando datos...", foreground="gray")
            return
        
        # Extraer información del mejor run
        best_balance = float(best_run.get("final_balance", 0.0))
        best_equity = float(best_run.get("final_equity", 0.0))
        best_run_num = best_run.get("run_number", "?")
        best_ts = best_run.get("ts_end", 0)
        
        # Formatear timestamp
        time_str = "?"
        if best_ts:
            try:
                from datetime import datetime
                dt = datetime.fromtimestamp(best_ts / 1000)
                time_str = dt.strftime("%d/%m %H:%M")
            except:
                pass
        
        # Calcular ganancia
        initial_balance = float(best_run.get("initial_balance", 10000.0))
        gain_pct = ((best_balance - initial_balance) / initial_balance) * 100.0
        
        # Color según rendimiento
        if gain_pct > 0:
            color = "green"
            gain_symbol = "📈"
        elif gain_pct < 0:
            color = "red"
            gain_symbol = "📉"
        else:
            color = "black"
            gain_symbol = "➖"
        
        # Texto del mejor run
        best_text = f"{gain_symbol} Run #{best_run_num} | Balance: {best_balance:.2f} | Equity: {best_equity:.2f} | Ganancia: {gain_pct:+.1f}% | {time_str}"
        self.lbl_best_run.config(text=best_text, foreground=color)

# ---------- NUEVO: Monitoreo de consola integrado ----------
class ConsoleMonitor:
    def __init__(self, symbol="BTCUSDT", models_root="models", refresh_interval=2):
        self.symbol = symbol
        self.models_root = Path(models_root)
        self.refresh_interval = refresh_interval
        self.runs_file = self.models_root / symbol / f"{symbol}_runs.jsonl"
        self.progress_file = self.models_root / symbol / f"{symbol}_progress.json"
        self.running = False
        
    def start_monitoring(self):
        """Inicia el monitoreo en tiempo real en consola"""
        self.running = True
        print(f"🔍 MONITOREO DE CONSOLA INICIADO PARA {self.symbol}")
        print(f"📁 Archivo de runs: {self.runs_file}")
        print(f"⏱️  Actualización cada {self.refresh_interval} segundos")
        print("=" * 80)
        
        try:
            while self.running:
                self._display_status()
                time.sleep(self.refresh_interval)
        except KeyboardInterrupt:
            print("\n⏹️  Monitoreo de consola detenido por el usuario")
            self.running = False
    
    def _display_status(self):
        """Muestra el estado actual del entrenamiento en consola"""
        try:
            # Limpiar pantalla (Windows)
            import os
            os.system('cls' if os.name == 'nt' else 'clear')
            
            # Mostrar timestamp
            from datetime import datetime
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
            print(f"❌ Error en monitoreo de consola: {e}")
    
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

# ---------- app ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbols", default="", help="Lista separada por comas (si se omite, detecta)")
    ap.add_argument("--models-root", default="models")
    ap.add_argument("--symbols-yaml", default="config/symbols.yaml")
    ap.add_argument("--refresh", type=float, default=1.0)
    ap.add_argument("--y-scale", choices=["linear","log"], default="linear")
    ap.add_argument("--window", type=int, default=0)
    ap.add_argument("--console", action="store_true", help="NUEVO: habilitar solo monitoreo de consola")
    args = ap.parse_args()

    # ← NUEVO: Si se especifica --console, solo ejecutar monitoreo de consola
    if args.console:
        print("🚀 MONITOR DE CONSOLA PPO")
        print("=" * 50)
        
        models_root = Path(args.models_root)
        yaml_path = Path(args.symbols_yaml)
        symbols = [s.strip() for s in args.symbols.split(",") if s.strip()] or \
                  discover_symbols(models_root, yaml_path)
        
        if not symbols:
            print("⚠️ No se encontraron símbolos. Define --symbols o crea runs en models/*")
            return
        
        # Usar el primer símbolo para monitoreo de consola
        symbol = symbols[0]
        monitor = ConsoleMonitor(
            symbol=symbol,
            models_root=args.models_root,
            refresh_interval=args.refresh
        )
        
        try:
            monitor.start_monitoring()
        except KeyboardInterrupt:
            print("\n👋 Monitoreo de consola detenido")
        return

    # ← ORIGINAL: Modo gráfico (sin cambios)
    models_root = Path(args.models_root)
    yaml_path = Path(args.symbols_yaml)
    symbols = [s.strip() for s in args.symbols.split(",") if s.strip()] or \
              discover_symbols(models_root, yaml_path)
    if not symbols:
        print("⚠️ No se encontraron símbolos. Define --symbols o crea runs en models/*")
        return

    root = tk.Tk()
    root.title(f"Progreso de entrenamiento — {len(symbols)} símbolos | Refresh: {args.refresh}s | Auto-refresh: 1s")

    notebook = ttk.Notebook(root)
    notebook.pack(fill=tk.BOTH, expand=True)

    tabs: dict[str, SymbolTab] = {}
    for s in symbols:
        tabs[s] = SymbolTab(notebook, s, models_root, args.refresh, args.y_scale, args.window)

    def on_close():
        for t in tabs.values():
            t.stop()
        root.destroy()

    root.protocol("WM_DELETE_WINDOW", on_close)
    root.mainloop()

if __name__ == "__main__":
    main()