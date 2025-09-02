# train_env/strategy_aggregator.py
# Consolida provisional.jsonl -> Top-K en {symbol}_strategies.json y limpia provisional
from __future__ import annotations
from pathlib import Path
import json
from typing import List, Dict, Any

def _score_row(e: Dict[str, Any]) -> float:
    r = float(e.get("r_multiple", 0.0))
    roi = abs(float(e.get("roi_pct", 0.0)))
    realized = float(e.get("realized_pnl", 0.0))
    
    # ← NUEVO: scoring mejorado para futuros con leverage
    base_score = (10.0 * r) + (0.1 * roi) + (0.001 * realized)
    
    # Bonus por uso eficiente de leverage en futuros
    if e.get("leverage_used") and e.get("notional_effective") and e.get("notional_max"):
        leverage_used = float(e.get("leverage_used", 1.0))
        notional_eff = float(e.get("notional_effective", 0.0))
        notional_max = float(e.get("notional_max", 1.0))
        
        # Bonus por uso eficiente del leverage (0.0 a 2.0)
        leverage_efficiency = notional_eff / notional_max if notional_max > 0 else 0.0
        leverage_bonus = leverage_efficiency * 2.0
        
        # Bonus por leverage moderado (evitar extremos)
        leverage_moderation = 1.0 - abs(leverage_used - 5.0) / 25.0  # Máximo en leverage 5x
        leverage_moderation_bonus = max(0.0, leverage_moderation) * 1.0
        
        base_score += leverage_bonus + leverage_moderation_bonus
    
    # ← NUEVO: bonus por timeframes preferidos (1m, 5m, 15m, 1h)
    exec_tf = e.get("exec_tf", "")
    if exec_tf in ["1m", "5m", "15m", "1h"]:
        tf_bonus = 3.0  # Bonus fuerte por timeframes preferidos
        base_score += tf_bonus
    elif exec_tf in ["4h", "1d"]:
        tf_penalty = -2.0  # Penalización por timeframes largos
        base_score += tf_penalty
    
    # ← NUEVO: bonus por duración moderada (evitar estrategias muy cortas o muy largas)
    bars_held = e.get("bars_held", 0)
    if 5 <= bars_held <= 50:  # Rango óptimo: 5-50 barras
        duration_bonus = 2.0
        base_score += duration_bonus
    elif bars_held < 3:  # Muy cortas: penalización
        duration_penalty = -1.0
        base_score += duration_penalty
    elif bars_held > 100:  # Muy largas: penalización
        duration_penalty = -1.5
        base_score += duration_penalty
    
    return base_score

def _load_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            try:
                e = json.loads(line)
                # ← NUEVO: cargar tanto eventos CLOSE como BANKRUPTCY
                if e.get("kind") in ("CLOSE", "BANKRUPTCY"):
                    rows.append(e)
            except Exception:
                continue
    return rows

def _dedupe(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    seen = set()
    out = []
    for e in rows:
        key = (e.get("ts"), e.get("entry_price"), e.get("price"), e.get("entry_qty"))
        if key in seen:
            continue
        seen.add(key)
        out.append(e)
    return out

def aggregate_top_k(provisional_file: str, best_json_file: str, top_k: int = 1000) -> None:
    prov = Path(provisional_file)
    best = Path(best_json_file)
    rows = _load_jsonl(prov)

    # merge con las ya guardadas
    if best.exists():
        try:
            old = json.loads(best.read_text(encoding="utf-8"))
            if isinstance(old, list):
                rows.extend(old)
        except Exception:
            pass

    if not rows:
        prov.unlink(missing_ok=True)
        return

    rows = _dedupe(rows)
    rows.sort(key=_score_row, reverse=True)
    rows = rows[: int(top_k)]

    best.parent.mkdir(parents=True, exist_ok=True)
    best.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")

    # limpiar provisional
    prov.unlink(missing_ok=True)
