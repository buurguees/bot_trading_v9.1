# app.py
"""
App de entrada. Arranca el bot según el modo en YAML y opcionalmente abre la GUI de progreso.

Usos:
  python app.py run                 # ejecuta según config/train.yaml
  python app.py run --gui           # ejecuta + abre ventana equity/balance
  python app.py gui                 # solo ventana de progreso
"""

from __future__ import annotations
import typer
import yaml
import os
import sys
import importlib
from pathlib import Path
from typing import Optional
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # 0=ALL,1=WARNING,2=ERROR,3=FATAL
os.environ["OMP_NUM_THREADS"] = "1"

app = typer.Typer(add_completion=False)

def _load_symbols(path: str = "config/symbols.yaml") -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def _ensure_models_dirs(models_root: str, symbol: str) -> None:
    (Path(models_root) / symbol).mkdir(parents=True, exist_ok=True)

def _import_or_exit(modname: str, attr: Optional[str] = None):
    try:
        mod = importlib.import_module(modname)
        return getattr(mod, attr) if attr else mod
    except Exception as e:
        typer.echo(f"[ERROR] No puedo importar {modname}: {e}")
        sys.exit(1)



@app.command()
def run(
    symbols_path: str = typer.Option("config/symbols.yaml", help="Ruta a symbols.yaml"),
    gui: bool = typer.Option(False, help="Abrir ventana de escritorio"),
):
    """
    Arranca el bot según el modo definido en config/symbols.yaml (por símbolo).
    Ahora: soportado train_spot (BTCUSDT).
    """
    cfg_symbols = _load_symbols(symbols_path)
    # Tomar el primer símbolo (asumiendo que está habilitado)
    if not cfg_symbols["symbols"]:
        typer.echo("[ERROR] No hay símbolos en symbols.yaml")
        raise typer.Exit(code=1)

    sym = cfg_symbols["symbols"][0]
    symbol = sym["symbol"]  # Usar "symbol" en lugar de "name"
    mode = sym.get("mode", "train_spot")
    models_root = "models"  # Por defecto
    _ensure_models_dirs(models_root, symbol)

    typer.echo(f"[INFO] symbol={symbol} mode={mode}")

    if gui:
        import subprocess, sys as _sys
        subprocess.Popen([_sys.executable, "scripts/watch_progress.py", "--symbols", symbol])

    if mode.startswith("train"):  # train_spot o train_futures
        from scripts.train_ppo import main as train_main
        train_main()
    else:
        typer.echo(f"[ERROR] Modo {mode} aún no implementado.")
        raise typer.Exit(code=2)

@app.command()
def gui(
    symbol: str = typer.Option("BTCUSDT", help="Símbolo a visualizar"),
):
    """Abre solo la ventana de escritorio de progreso."""
    import subprocess, sys as _sys
    subprocess.call([_sys.executable, "scripts/watch_progress.py", "--symbol", symbol])

if __name__ == "__main__":
    app()
