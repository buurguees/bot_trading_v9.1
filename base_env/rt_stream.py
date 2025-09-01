from __future__ import annotations
import time
import pandas as pd
from typing import Iterator

class RTStream:
    """Proveedor unificado:
       - modo 'replay': yield de barras históricas (para Train/Paper/Event-Driven)
       - modo 'external': placeholder para WS real (se integrará más adelante)
    """
    def replay(self, df: pd.DataFrame, sleep: float = 0.0) -> Iterator[dict]:
        """Itera fila a fila (o chunk a chunk) simulando un stream. sleep=0 para training rápido."""
        for _, row in df.iterrows():
            yield row.to_dict()
            if sleep > 0:
                time.sleep(sleep)
