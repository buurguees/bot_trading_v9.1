from __future__ import annotations
from typing import Dict

def load_symbol_filters() -> Dict[str, dict]:
    # En real se extrae del exchange; aquí tomamos de config/symbols.yaml por simplicidad
    return {
        "BTCUSDT": {"tick_size": 0.1, "lot_step": 0.0001, "min_qty": 0.0001},
        "ETHUSDT": {"tick_size": 0.01, "lot_step": 0.0001, "min_qty": 0.001},
    }
