# train_env/strategy_logger.py
# Descripción: Append de eventos OPEN/CLOSE en un JSONL provisional.
from __future__ import annotations
from pathlib import Path
import json
from typing import List, Dict, Any

class StrategyLogger:
    def __init__(self, path: str):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def append_many(self, events: List[Dict[str, Any]]):
        with self.path.open("a", encoding="utf-8") as f:
            for e in events:
                # ← NUEVO: capturar eventos de quiebra además de OPEN/CLOSE
                if e.get("kind") in ("OPEN", "CLOSE", "BANKRUPTCY"):
                    f.write(json.dumps(e, ensure_ascii=False) + "\n")
