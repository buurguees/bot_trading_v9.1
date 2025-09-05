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
    """Carga runs de forma robusta, manejando archivos corruptos o parciales."""
    rows = []
    if not path.exists():
        return rows
    
    try:
        with path.open("r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:  # Saltar líneas vacías
                    continue
                try:
                    run_data = json.loads(line)
                    # Validar que el run tiene campos mínimos
                    if isinstance(run_data, dict) and "final_equity" in run_data:
                        rows.append(run_data)
                except json.JSONDecodeError as e:
                    # Log error pero continuar
                    print(f"⚠️ Error en línea {line_num} de {path}: {e}")
                    continue
                except Exception as e:
                    print(f"⚠️ Error procesando línea {line_num} de {path}: {e}")
                    continue
    except Exception as e:
        print(f"❌ Error leyendo archivo {path}: {e}")
    
    return rows

def read_training_metrics(symbol: str, models_root: Path) -> dict | None:
    """Lee las métricas de entrenamiento más recientes del archivo JSONL."""
    metrics_file = models_root / symbol / f"{symbol}_train_metrics.jsonl"
    
    if not metrics_file.exists():
        return None
    
    try:
        # Leer la última línea válida del archivo
        with metrics_file.open("r", encoding="utf-8") as f:
            lines = f.readlines()
            
        # Buscar la última línea válida (desde el final)
        for line in reversed(lines):
            line = line.strip()
            if not line:
                continue
            try:
                metrics = json.loads(line)
                if isinstance(metrics, dict) and "ts" in metrics:
                    return metrics
            except json.JSONDecodeError:
                continue
                
    except Exception as e:
        print(f"⚠️ Error leyendo métricas de entrenamiento: {e}")
    
    return None

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

        # ← NUEVO: Filtrar runs mal escritos (balance negativo extremo o equity inválido)
        valid_runs = []
        bankruptcy_events = []
        reset_events = []
        
        for r in view:
            equity = float(r.get("final_equity", 0.0))
            balance = float(r.get("final_balance", 0.0))
            run_result = r.get("run_result", "")
            
            # Detectar eventos especiales
            if "BANKRUPTCY" in run_result:
                bankruptcy_events.append(r)
            elif "RESET" in run_result or "SOFT_RESET" in run_result:
                reset_events.append(r)
            
            # Filtrar runs con balance muy negativo o equity inválido
            if balance > -10000 and equity > 0 and equity < 10000000:  # Límites razonables
                valid_runs.append(r)
        
        if not valid_runs:
            self.ax.set_title(f"{self.symbol} — runs válidos: 0 (filtrados runs mal escritos)")
            self.ax.set_xlabel("Run #"); self.ax.set_ylabel("USDT")
            self.canvas.draw()
            self.lbl_runs.config(text="Runs: 0 (válidos)")
            self.lbl_last.config(text="Última: -")
            return
        
        # Usar solo runs válidos
        y_eq = [float(r.get("final_equity", 0.0)) for r in valid_runs]
        y_bal = [float(r.get("final_balance", 0.0)) for r in valid_runs]
        target = float(valid_runs[-1].get("target_balance", 0.0)) if valid_runs else 0.0
        
        print(f"🔍 Debug: Datos preparados - y_eq={y_eq[:5]}, y_bal={y_bal[:5]}, target={target}")

        self.ax.plot(xs, y_eq, marker="o", label="Equity")
        self.ax.plot(xs, y_bal, marker="x", label="Balance")
        if target > 0:
            self.ax.axhline(target, linestyle="--", color="red", label=f"Objetivo {target:.0f}")
        
        # ← NUEVO: Marcadores para eventos especiales
        for event in bankruptcy_events:
            run_idx = valid_runs.index(event) if event in valid_runs else -1
            if run_idx >= 0:
                x_pos = xs[run_idx]
                y_pos = y_eq[run_idx]
                self.ax.scatter(x_pos, y_pos, marker="X", s=100, color="red", 
                              label="Bancarrota" if bankruptcy_events.index(event) == 0 else "")
                self.ax.annotate("💀", xy=(x_pos, y_pos), xytext=(0, 10), 
                               textcoords="offset points", ha="center", fontsize=12)
        
        for event in reset_events:
            run_idx = valid_runs.index(event) if event in valid_runs else -1
            if run_idx >= 0:
                x_pos = xs[run_idx]
                y_pos = y_eq[run_idx]
                self.ax.scatter(x_pos, y_pos, marker="s", s=80, color="orange", 
                              label="Reset" if reset_events.index(event) == 0 else "")
                self.ax.annotate("🔄", xy=(x_pos, y_pos), xytext=(0, -15), 
                               textcoords="offset points", ha="center", fontsize=10)

        # anotación último punto
        self.ax.annotate(
            f"Run {xs[-1]}\nEq {y_eq[-1]:.2f}\nBal {y_bal[-1]:.2f}",
            xy=(xs[-1], y_eq[-1]),
            xytext=(6,6), textcoords="offset points"
        )

        # Título con información de eventos especiales
        title_parts = [f"{self.symbol} — runs={len(runs)} (válidos: {len(valid_runs)})"]
        if bankruptcy_events:
            title_parts.append(f"💀 Bancarrotas: {len(bankruptcy_events)}")
        if reset_events:
            title_parts.append(f"🔄 Resets: {len(reset_events)}")
        
        self.ax.set_title(" | ".join(title_parts))
        self.ax.set_xlabel("Run #"); self.ax.set_ylabel("USDT")
        self.ax.legend()
        self.fig.tight_layout()
        self.canvas.draw()

        # Información de eventos en las etiquetas
        events_info = ""
        if bankruptcy_events:
            events_info += f" | 💀 {len(bankruptcy_events)}"
        if reset_events:
            events_info += f" | 🔄 {len(reset_events)}"
        
        self.lbl_runs.config(text=f"Runs: {len(runs)} (válidos: {len(valid_runs)}){events_info}")
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
        """Muestra el estado actual del entrenamiento en consola con KPIs profesionales"""
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
            
            # Calcular KPIs profesionales
            kpis = self._calculate_kpis(runs)
            
            # Estadísticas generales
            print(f"📊 ESTADÍSTICAS GENERALES:")
            print(f"   Total runs: {kpis['total_runs']}")
            print(f"   Mejor equity: {kpis['best_equity']:.2f} USDT")
            print(f"   Peor equity: {kpis['worst_equity']:.2f} USDT")
            print(f"   Promedio equity: {kpis['avg_equity']:.2f} USDT")
            print(f"   Mejor balance: {kpis['best_balance']:.2f} USDT")
            print(f"   Peor balance: {kpis['worst_balance']:.2f} USDT")
            print(f"   💀 Bancarrotas: {kpis['bankruptcy_count']}")
            print(f"   🔄 Resets: {kpis['reset_count']}")
            
            # KPIs profesionales
            print(f"\n📈 KPIs PROFESIONALES:")
            print(f"   ROI promedio: {kpis['avg_roi']:+.2f}%")
            print(f"   Max Drawdown: {kpis['max_drawdown']:.2f}%")
            print(f"   Win Rate: {kpis['win_rate']:.1f}%")
            print(f"   Profit Factor: {kpis['profit_factor']:.2f}")
            print(f"   Sharpe Ratio: {kpis['sharpe_ratio']:.2f}")
            print(f"   Trades promedio: {kpis['avg_trades']:.1f}")
            print(f"   R-Multiple promedio: {kpis['avg_r_multiple']:.2f}")
            
            # ← NUEVO: KPIs profesionales de trades
            print(f"\n🎯 KPIs PROFESIONALES DE TRADES:")
            if kpis['best_win_rate_trades'] > 0:
                print(f"   Mejor Win Rate: {kpis['best_win_rate_trades']:.1f}%")
                print(f"   Win Rate promedio: {kpis['avg_win_rate_trades']:.1f}%")
            else:
                print(f"   Mejor Win Rate: —")
                print(f"   Win Rate promedio: —")
            
            if kpis['best_avg_trade_pnl'] != 0:
                print(f"   Mejor Avg PnL: {kpis['best_avg_trade_pnl']:+.2f} USDT")
                print(f"   Avg PnL promedio: {kpis['avg_avg_trade_pnl']:+.2f} USDT")
            else:
                print(f"   Mejor Avg PnL: —")
                print(f"   Avg PnL promedio: —")
            
            if kpis['best_profit_factor'] > 0:
                print(f"   Mayor Profit Factor: {kpis['best_profit_factor']:.2f}")
                print(f"   Profit Factor promedio: {kpis['avg_profit_factor']:.2f}")
            else:
                print(f"   Mayor Profit Factor: —")
                print(f"   Profit Factor promedio: —")
            
            if kpis['avg_avg_holding_bars'] > 0:
                print(f"   Holding promedio: {kpis['avg_avg_holding_bars']:.1f} barras")
            else:
                print(f"   Holding promedio: —")
            
            # ← NUEVO: KPIs de leverage
            print(f"\n⚡ KPIs DE LEVERAGE:")
            if kpis['avg_avg_leverage'] > 0:
                print(f"   Leverage promedio: {kpis['avg_avg_leverage']:.1f}x")
                print(f"   Leverage máximo: {kpis['max_leverage']:.1f}x")
                print(f"   % trades high leverage: {kpis['avg_high_leverage_pct']:.1f}%")
            else:
                print(f"   Leverage promedio: —")
                print(f"   Leverage máximo: —")
                print(f"   % trades high leverage: —")
            
            # ← NUEVO: Runs exitosos (no bancarrota)
            successful_runs = len([r for r in runs if "BANKRUPTCY" not in r.get("run_result", "")])
            success_rate = (successful_runs / len(runs)) * 100 if runs else 0
            print(f"   Runs exitosos: {successful_runs}/{len(runs)} ({success_rate:.1f}%)")
            
            # Mejor run
            best_run = max(runs, key=lambda r: float(r.get("final_equity", 0.0)))
            print(f"\n🏆 MEJOR RUN:")
            print(f"   Equity: {best_run.get('final_equity', 'N/A')}")
            print(f"   Balance: {best_run.get('final_balance', 'N/A')}")
            print(f"   ROI: {kpis['best_roi']:+.2f}%")
            print(f"   Estado: {best_run.get('run_result', 'N/A')}")
            
            # Últimos 5 runs con KPIs
            print(f"\n🔄 ÚLTIMOS 5 RUNS:")
            print("-" * 100)
            for i, run in enumerate(runs[-5:], 1):
                equity = run.get("final_equity", 0.0)
                balance = run.get("final_balance", 0.0)
                result = run.get("run_result", "?")
                ts = run.get("ts_end", 0)
                trades = run.get("trades_count", 0)  # Usar trades_count en lugar de trades
                steps = run.get("elapsed_steps", 0)  # Usar elapsed_steps en lugar de steps
                bankruptcy = run.get("bankruptcy", False)
                reasons = run.get("reasons_counter", {})
                
                # ← NUEVO: Métricas profesionales de trades
                avg_holding_time = run.get("avg_holding_time", 0.0)
                trades_with_sl_tp = run.get("trades_with_sl_tp", 0)
                total_trades = max(trades, 1)  # Evitar división por cero
                sl_tp_percentage = (trades_with_sl_tp / total_trades) * 100.0
                
                # ← NUEVO: Métricas profesionales adicionales
                win_rate_trades = run.get("win_rate_trades", 0.0)
                avg_trade_pnl = run.get("avg_trade_pnl", 0.0)
                avg_holding_bars = run.get("avg_holding_bars", 0.0)
                profit_factor = run.get("profit_factor", None)
                max_consecutive_wins = run.get("max_consecutive_wins", 0)
                max_consecutive_losses = run.get("max_consecutive_losses", 0)
                # ← NUEVO: Métricas de leverage
                avg_leverage = run.get("avg_leverage", 0.0)
                max_leverage = run.get("max_leverage", 0.0)
                high_leverage_pct = run.get("high_leverage_pct", 0.0)
                
                # Calcular ROI del run
                initial_balance = float(run.get("initial_balance", 1000.0))
                roi = ((balance - initial_balance) / initial_balance) * 100.0
                
                time_str = "?"
                if ts:
                    try:
                        dt = datetime.fromtimestamp(ts / 1000)
                        time_str = dt.strftime("%H:%M:%S")
                    except:
                        pass
                
                # Color según rendimiento
                if bankruptcy:
                    status = "💀"
                elif roi > 5:
                    status = "🟢"
                elif roi > 0:
                    status = "🟡"
                else:
                    status = "🔴"
                
                # ← NUEVO: Marcar runs exitosos con ⭐
                if not bankruptcy and roi > 0:
                    status += "⭐"
                
                # Mostrar top razón si existe
                top_reason = ""
                if reasons:
                    top_reason_name = max(reasons.items(), key=lambda x: x[1])[0]
                    top_reason_count = max(reasons.values())
                    total_reasons = sum(reasons.values())
                    top_reason_pct = (top_reason_count / total_reasons * 100) if total_reasons > 0 else 0
                    top_reason = f" | Top: {top_reason_name}({top_reason_pct:.1f}%)"
                
                print(f"   {i}. {status} Equity: {equity:8.2f} | Balance: {balance:8.2f} | ROI: {roi:+6.1f}% | Trades: {trades:3d} | Steps: {steps:5d} | {result:12} | {time_str}{top_reason}")
                # ← NUEVO: Mostrar métricas profesionales de trades en línea separada
                if trades > 0:
                    # Usar métricas profesionales si están disponibles, sino calcularlas
                    if win_rate_trades > 0:
                        win_rate_display = win_rate_trades
                    else:
                        # Fallback: calcular WinRate real (trades con ROI positivo)
                        winning_trades = run.get("winning_trades", 0)
                        win_rate_display = (winning_trades / trades) * 100.0 if trades > 0 else 0.0
                    
                    # Usar avg_holding_bars si está disponible, sino avg_holding_time
                    holding_display = avg_holding_bars if avg_holding_bars > 0 else avg_holding_time
                    
                    # Mostrar métricas profesionales
                    pf_str = f"{profit_factor:.2f}" if profit_factor is not None else "N/A"
                    print(f"      📊 WinRate: {win_rate_display:.1f}% | Avg PnL: {avg_trade_pnl:+.2f} | Holding: {holding_display:.1f} bars | PF: {pf_str}")
                    print(f"      🎯 Streaks: +{max_consecutive_wins} / -{max_consecutive_losses} | SL/TP: {sl_tp_percentage:.1f}% ({trades_with_sl_tp}/{trades})")
                    # ← NUEVO: Mostrar métricas de leverage
                    if avg_leverage > 0:
                        print(f"      ⚡ Leverage: {avg_leverage:.1f}x (max: {max_leverage:.1f}x) | High leverage: {high_leverage_pct:.1f}%")
            
            # Progreso hacia objetivo
            if runs:
                last_run = runs[-1]
                target = float(last_run.get("target_balance", 1000000))
                best_balance = max([float(r.get("final_balance", 0.0)) for r in runs])
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
            
            # Top razones de no-trade (últimos 10 runs) - limitado a Top-3
            if len(runs) >= 5:
                all_reasons = {}
                for run in runs[-10:]:  # Últimos 10 runs
                    reasons = run.get("reasons_counter", {})
                    for reason, count in reasons.items():
                        all_reasons[reason] = all_reasons.get(reason, 0) + count
                
                if all_reasons:
                    total_reasons = sum(all_reasons.values())
                    sorted_reasons = sorted(all_reasons.items(), key=lambda x: x[1], reverse=True)
                    
                    print(f"\n🚫 TOP RAZONES DE NO-TRADE (últimos 10 runs):")
                    for i, (reason, count) in enumerate(sorted_reasons[:3], 1):  # ← NUEVO: Solo Top-3
                        pct = (count / total_reasons * 100) if total_reasons > 0 else 0
                        print(f"   {i}. {reason}: {count} ({pct:.1f}%)")
                    
                    # ← NUEVO: Mostrar % acumulado de bloqueos por NO_SL_DISTANCE y MIN_NOTIONAL
                    no_sl_distance_count = all_reasons.get("NO_SL_DISTANCE", 0)
                    min_notional_blocked_count = all_reasons.get("MIN_NOTIONAL_BLOCKED", 0)
                    total_blocked = no_sl_distance_count + min_notional_blocked_count
                    
                    if total_blocked > 0:
                        no_sl_pct = (no_sl_distance_count / total_reasons * 100) if total_reasons > 0 else 0
                        min_notional_pct = (min_notional_blocked_count / total_reasons * 100) if total_reasons > 0 else 0
                        print(f"\n🔒 BLOQUEOS POR NIVELES:")
                        print(f"   NO_SL_DISTANCE: {no_sl_distance_count} ({no_sl_pct:.1f}%)")
                        print(f"   MIN_NOTIONAL_BLOCKED: {min_notional_blocked_count} ({min_notional_pct:.1f}%)")
            
            # Tendencias (últimos 10 runs)
            if len(runs) >= 10:
                recent_runs = runs[-10:]
                recent_rois = []
                for run in recent_runs:
                    initial = float(run.get("initial_balance", 1000.0))
                    final = float(run.get("final_balance", 0.0))
                    roi = ((final - initial) / initial) * 100.0
                    recent_rois.append(roi)
                
                avg_recent_roi = sum(recent_rois) / len(recent_rois)
                print(f"\n📊 TENDENCIAS (últimos 10 runs):")
                print(f"   ROI promedio: {avg_recent_roi:+.2f}%")
                print(f"   Tendencia: {'📈' if avg_recent_roi > 0 else '📉'}")
            
            # ← NUEVO: Métricas de entrenamiento en tiempo real
            training_metrics = read_training_metrics(self.symbol, self.models_root)
            if training_metrics:
                print(f"\n📊 TRAINING METRICS (último snapshot):")
                print(f"   fps: {training_metrics.get('fps', '—'):.1f}")
                print(f"   iterations: {training_metrics.get('iterations', '—')}")
                print(f"   time_elapsed: {training_metrics.get('time_elapsed', '—'):.1f}s")
                print(f"   total_timesteps: {training_metrics.get('total_timesteps', '—'):,}")
                print(f"   approx_kl: {training_metrics.get('approx_kl', '—')}")
                print(f"   clip_fraction: {training_metrics.get('clip_fraction', '—')}")
                print(f"   clip_range: {training_metrics.get('clip_range', '—')}")
                print(f"   entropy_loss: {training_metrics.get('entropy_loss', '—')}")
                print(f"   explained_variance: {training_metrics.get('explained_variance', '—')}")
                print(f"   learning_rate: {training_metrics.get('learning_rate', '—')}")
                print(f"   loss: {training_metrics.get('loss', '—')}")
                print(f"   n_updates: {training_metrics.get('n_updates', '—')}")
                print(f"   policy_gradient_loss: {training_metrics.get('policy_gradient_loss', '—')}")
                print(f"   value_loss: {training_metrics.get('value_loss', '—')}")
            else:
                print(f"\n📊 TRAINING METRICS: — (archivo no encontrado o sin datos)")
            
            # Información del archivo
            if self.runs_file.exists():
                file_size = self.runs_file.stat().st_size
                print(f"\n📁 ARCHIVO:")
                print(f"   Tamaño: {file_size:,} bytes")
                print(f"   Última modificación: {datetime.fromtimestamp(self.runs_file.stat().st_mtime).strftime('%H:%M:%S')}")
            
            print(f"\n⏱️  Próxima actualización en {self.refresh_interval} segundos... (Ctrl+C para detener)")
            
        except Exception as e:
            print(f"❌ Error en monitoreo de consola: {e}")
    
    def _calculate_kpis(self, runs):
        """Calcula KPIs profesionales para los runs"""
        if not runs:
            return {}
        
        # Datos básicos
        equities = [float(r.get("final_equity", 0.0)) for r in runs]
        balances = [float(r.get("final_balance", 0.0)) for r in runs]
        initial_balances = [float(r.get("initial_balance", 1000.0)) for r in runs]
        
        # ROI por run
        rois = []
        for i, run in enumerate(runs):
            initial = initial_balances[i] if i < len(initial_balances) else 1000.0
            final = balances[i]
            roi = ((final - initial) / initial) * 100.0
            rois.append(roi)
        
        # Win rate
        winning_runs = len([r for r in rois if r > 0])
        win_rate = (winning_runs / len(rois)) * 100.0 if rois else 0.0
        
        # Profit Factor
        gross_profit = sum([r for r in rois if r > 0])
        gross_loss = abs(sum([r for r in rois if r < 0]))
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf') if gross_profit > 0 else 0.0
        
        # Max Drawdown
        max_drawdown = 0.0
        peak = initial_balances[0] if initial_balances else 1000.0
        for balance in balances:
            if balance > peak:
                peak = balance
            drawdown = ((peak - balance) / peak) * 100.0
            max_drawdown = max(max_drawdown, drawdown)
        
        # Sharpe Ratio (simplificado)
        if len(rois) > 1:
            avg_roi = sum(rois) / len(rois)
            std_roi = (sum([(r - avg_roi) ** 2 for r in rois]) / len(rois)) ** 0.5
            sharpe_ratio = avg_roi / std_roi if std_roi > 0 else 0.0
        else:
            sharpe_ratio = 0.0
        
        # Trades promedio
        trades_counts = [r.get("trades_count", 0) for r in runs]
        avg_trades = sum(trades_counts) / len(trades_counts) if trades_counts else 0.0
        
        # R-Multiple promedio (si está disponible)
        r_multiples = []
        for run in runs:
            if "r_multiple" in run:
                r_multiples.append(float(run["r_multiple"]))
        avg_r_multiple = sum(r_multiples) / len(r_multiples) if r_multiples else 0.0
        
        # ← NUEVO: Métricas profesionales de trades
        win_rates_trades = [r.get("win_rate_trades", 0.0) for r in runs if r.get("win_rate_trades", 0.0) > 0]
        avg_win_rate_trades = sum(win_rates_trades) / len(win_rates_trades) if win_rates_trades else 0.0
        best_win_rate_trades = max(win_rates_trades) if win_rates_trades else 0.0
        
        avg_trade_pnls = [r.get("avg_trade_pnl", 0.0) for r in runs if r.get("avg_trade_pnl", 0.0) != 0.0]
        avg_avg_trade_pnl = sum(avg_trade_pnls) / len(avg_trade_pnls) if avg_trade_pnls else 0.0
        best_avg_trade_pnl = max(avg_trade_pnls) if avg_trade_pnls else 0.0
        
        profit_factors = [r.get("profit_factor", 0.0) for r in runs if r.get("profit_factor") is not None and r.get("profit_factor", 0.0) > 0]
        avg_profit_factor = sum(profit_factors) / len(profit_factors) if profit_factors else 0.0
        best_profit_factor = max(profit_factors) if profit_factors else 0.0
        
        avg_holding_bars_list = [r.get("avg_holding_bars", 0.0) for r in runs if r.get("avg_holding_bars", 0.0) > 0]
        avg_avg_holding_bars = sum(avg_holding_bars_list) / len(avg_holding_bars_list) if avg_holding_bars_list else 0.0
        
        # ← NUEVO: Métricas de leverage
        avg_leverages = [r.get("avg_leverage", 0.0) for r in runs if r.get("avg_leverage", 0.0) > 0]
        avg_avg_leverage = sum(avg_leverages) / len(avg_leverages) if avg_leverages else 0.0
        max_leverage = max([r.get("max_leverage", 0.0) for r in runs if r.get("max_leverage", 0.0) > 0], default=0.0)
        high_leverage_pcts = [r.get("high_leverage_pct", 0.0) for r in runs if r.get("high_leverage_pct", 0.0) > 0]
        avg_high_leverage_pct = sum(high_leverage_pcts) / len(high_leverage_pcts) if high_leverage_pcts else 0.0
        
        return {
            'total_runs': len(runs),
            'best_equity': max(equities) if equities else 0.0,
            'worst_equity': min(equities) if equities else 0.0,
            'avg_equity': sum(equities) / len(equities) if equities else 0.0,
            'best_balance': max(balances) if balances else 0.0,
            'worst_balance': min(balances) if balances else 0.0,
            'bankruptcy_count': len([r for r in runs if "BANKRUPTCY" in r.get("run_result", "")]),
            'reset_count': len([r for r in runs if "RESET" in r.get("run_result", "") or "SOFT_RESET" in r.get("run_result", "")]),
            'avg_roi': sum(rois) / len(rois) if rois else 0.0,
            'best_roi': max(rois) if rois else 0.0,
            'max_drawdown': max_drawdown,
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'sharpe_ratio': sharpe_ratio,
            'avg_trades': avg_trades,
            'avg_r_multiple': avg_r_multiple,
            # ← NUEVO: Métricas profesionales
            'avg_win_rate_trades': avg_win_rate_trades,
            'best_win_rate_trades': best_win_rate_trades,
            'avg_avg_trade_pnl': avg_avg_trade_pnl,
            'best_avg_trade_pnl': best_avg_trade_pnl,
            'avg_profit_factor': avg_profit_factor,
            'best_profit_factor': best_profit_factor,
            'avg_avg_holding_bars': avg_avg_holding_bars,
            # ← NUEVO: Métricas de leverage
            'avg_avg_leverage': avg_avg_leverage,
            'max_leverage': max_leverage,
            'avg_high_leverage_pct': avg_high_leverage_pct
        }
    
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