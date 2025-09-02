# scripts/show_progress.py
"""
Muestra la evolución de las runs para un símbolo concreto.
Lee:
  - models/{symbol}/{symbol}_runs.jsonl
  - models/{symbol}/{symbol}_progress.json

Genera:
  - print resumen
  - gráfico simple (equity final vs nº de run) con línea de objetivo
"""

from __future__ import annotations
import argparse, json
from pathlib import Path
import matplotlib.pyplot as plt

def load_runs(symbol: str, models_root: str = "models"):
    runs_file = Path(models_root) / symbol / f"{symbol}_runs.jsonl"
    if not runs_file.exists():
        raise FileNotFoundError(f"No existe: {runs_file}")
    runs = []
    with runs_file.open("r", encoding="utf-8") as f:
        for line in f:
            try:
                runs.append(json.loads(line))
            except Exception:
                continue
    return runs

def load_progress(symbol: str, models_root: str = "models"):
    progress_file = Path(models_root) / symbol / f"{symbol}_progress.json"
    if not progress_file.exists():
        return {}
    return json.loads(progress_file.read_text(encoding="utf-8"))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbol", required=True, help="Ej: BTCUSDT")
    parser.add_argument("--models-root", default="models")
    args = parser.parse_args()

    runs = load_runs(args.symbol, args.models_root)
    progress = load_progress(args.symbol, args.models_root)

    if not runs:
        print("⚠️ No hay runs registradas todavía.")
        return

    print(f"📊 Progreso {args.symbol}")
    print(json.dumps(progress, indent=2, ensure_ascii=False))

    # Plot
    x = list(range(1, len(runs)+1))
    y = [r.get("final_equity", 0) for r in runs]
    target = runs[0].get("target_balance", 0)

    plt.figure(figsize=(8,5))
    plt.plot(x, y, marker="o", label="Equity final")
    if target:
        plt.axhline(target, color="red", linestyle="--", label=f"Objetivo {target}")
    plt.xlabel("Run #")
    plt.ylabel("Equity final (USDT)")
    plt.title(f"Evolución de equity - {args.symbol}")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
