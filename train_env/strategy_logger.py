# train_env/strategy_logger.py
# Descripción: Append de eventos OPEN/CLOSE en un JSONL provisional.
from __future__ import annotations
from pathlib import Path
import json
import numpy as np
from typing import List, Dict, Any
from base_env.utils.timestamp_utils import add_utc_timestamps

def _convert_numpy_types(obj):
    """Convierte tipos NumPy a tipos nativos de Python para serialización JSON"""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: _convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [_convert_numpy_types(item) for item in obj]
    else:
        return obj

class StrategyLogger:
    def __init__(self, path: str, segment_id: int = 0):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.segment_id = segment_id

    def set_segment_id(self, segment_id: int):
        """Actualiza el segment_id actual."""
        self.segment_id = segment_id

    def append_many(self, events: List[Dict[str, Any]]):
        with self.path.open("a", encoding="utf-8") as f:
            for e in events:
                # ← NUEVO: capturar eventos de quiebra además de OPEN/CLOSE
                if e.get("kind") in ("OPEN", "CLOSE", "BANKRUPTCY"):
                    # Añadir segment_id al evento
                    e_with_segment = e.copy()
                    e_with_segment["segment_id"] = self.segment_id
                    
                    # Añadir timestamps UTC legibles
                    e_with_utc = add_utc_timestamps(e_with_segment)
                    
                    # Convertir tipos NumPy a tipos nativos de Python para serialización JSON
                    converted_event = _convert_numpy_types(e_with_utc)
                    f.write(json.dumps(converted_event, ensure_ascii=False) + "\n")

    def append_single(self, event: Dict[str, Any]):
        """Añade un solo evento."""
        self.append_many([event])
