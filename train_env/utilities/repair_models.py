#!/usr/bin/env python3
# scripts/repair_models.py
# Limpia JSONL corruptos con "..." y regenera progress.json para cualquier símbolo

import json, os, shutil, time, argparse
from pathlib import Path
from datetime import datetime

def _backup(p: Path):
    if p.exists():
        ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        bak = p.with_suffix(p.suffix + f".bak_{ts}")
        shutil.copy2(p, bak)
        print(f"📦 Backup -> {bak}")

def _load_valid_jsonl(p: Path, required_keys=None):
    out = []
    if not p.exists():
        return out
    with p.open("r", encoding="utf-8", errors="ignore") as f:
        for i, line in enumerate(f, 1):
            s = line.strip()
            if not s or "..." in s:
                continue
            try:
                obj = json.loads(s)
                if required_keys:
                    if any(k not in obj for k in required_keys):
                        continue
                out.append(obj)
            except Exception:
                continue
    return out

def _write_jsonl(p: Path, rows):
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

def _recompute_progress_from_runs(runs, symbol):
    if not runs:
        return {
            "symbol": symbol,
            "runs_completed": 0,
            "best_equity": 0.0,
            "best_balance": 0.0,
            "last_run": None,
            "progress_pct": 0.0
        }
    best_equity = max((r.get("final_equity") or 0.0) for r in runs)
    best_balance = max((r.get("final_balance") or 0.0) for r in runs)
    last_run = runs[-1]
    # Si hay MAX_RECORDS en train.yaml (runs_log.max_records), úsalo para estimar %; si no, 2000 por defecto
    max_records = 2000
    try:
        import yaml
        with open("config/train.yaml","r",encoding="utf-8-sig") as f:
            train_cfg = yaml.safe_load(f) or {}
        max_records = int(train_cfg.get("runs_log",{}).get("max_records", 2000))
    except Exception:
        pass
    progress_pct = round(min(1.0, len(runs)/max_records), 4)
    return {
        "symbol": symbol,
        "runs_completed": len(runs),
        "best_equity": best_equity,
        "best_balance": best_balance,
        "last_run": last_run,
        "progress_pct": progress_pct
    }

def repair_models(symbol: str, verbose: bool = True) -> bool:
    """
    Repara archivos de modelos para un símbolo dado.
    
    Args:
        symbol: Símbolo a reparar (ej: "BTCUSDT")
        verbose: Si mostrar mensajes de progreso
        
    Returns:
        True si la reparación fue exitosa, False si hubo errores
    """
    models_dir = Path(f"models/{symbol}")
    runs_file = models_dir / f"{symbol}_runs.jsonl"
    metrics_file = models_dir / f"{symbol}_train_metrics.jsonl"
    progress_file = models_dir / f"{symbol}_progress.json"
    
    if not models_dir.exists():
        if verbose:
            print(f"❌ No existe models/{symbol}")
        return False

    if verbose:
        print(f"🧹 Reparando models/{symbol} ...")

    # Backups
    for p in [runs_file, metrics_file, progress_file]:
        _backup(p)

    # Cargar válidos
    valid_runs = _load_valid_jsonl(
        runs_file,
        required_keys=["symbol", "final_balance", "final_equity", "ts_start", "ts_end"]
    )
    
    # dedup básico por (ts_start, ts_end, final_balance, final_equity)
    seen = set(); dedup = []
    for r in valid_runs:
        key = (r.get("ts_start"), r.get("ts_end"), round(r.get("final_balance",0.0), 2), round(r.get("final_equity",0.0), 2))
        if key in seen:
            continue
        seen.add(key)
        dedup.append(r)

    # Orden temporal (por ts_start)
    dedup.sort(key=lambda r: (r.get("ts_start") or 0, r.get("ts_end") or 0))

    # Guardar runs limpios
    _write_jsonl(runs_file, dedup)
    if verbose:
        print(f"✅ Runs válidos guardados: {len(dedup)}")

    # Limpiar métricas de entrenamiento (si existen)
    valid_metrics = _load_valid_jsonl(metrics_file, required_keys=["ts","symbol","mode"])
    _write_jsonl(metrics_file, valid_metrics)
    if verbose:
        print(f"✅ Métricas válidas guardadas: {len(valid_metrics)}")

    # Recalcular progress.json
    progress = _recompute_progress_from_runs(dedup, symbol)
    with progress_file.open("w", encoding="utf-8") as f:
        json.dump(progress, f, ensure_ascii=False, indent=2)
    if verbose:
        print(f"📈 progress.json regenerado → runs_completed={progress['runs_completed']} best_equity={progress['best_equity']:.2f}")

    # Validación: verificar que hay al menos una línea válida
    if len(dedup) == 0:
        if verbose:
            print(f"ℹ️  INFO: No hay runs válidos para {symbol}")
            print(f"   Esto es normal en la primera ejecución - se crearán nuevos runs durante el entrenamiento")
        # En primera ejecución, crear un progress.json vacío pero válido
        progress = _recompute_progress_from_runs(dedup, symbol)
        with progress_file.open("w", encoding="utf-8") as f:
            json.dump(progress, f, ensure_ascii=False, indent=2)
        if verbose:
            print(f"📈 progress.json inicial creado → runs_completed=0")
        return True  # No es un error en primera ejecución

    if verbose:
        print("🎉 Reparación terminada exitosamente")
    
    return True

def main():
    parser = argparse.ArgumentParser(description="Repara archivos de modelos corruptos")
    parser.add_argument("symbol", help="Símbolo a reparar (ej: BTCUSDT)")
    parser.add_argument("--verbose", "-v", action="store_true", help="Mostrar mensajes detallados")
    
    args = parser.parse_args()
    
    success = repair_models(args.symbol, verbose=args.verbose)
    
    if not success:
        print(f"❌ La reparación de {args.symbol} falló")
        exit(1)
    else:
        print(f"✅ {args.symbol} reparado exitosamente")

if __name__ == "__main__":
    main()
