# base_env/tfs/alignment.py
# Descripción: Verificación de alineación multi-timeframe. En modo estricto, exige todos los TF.
#              El broker ya sirve "última barra cerrada ≤ bar_time"; aquí solo validamos presencia.
from __future__ import annotations
from typing import Dict, Literal, List

TF = Literal["1m","5m","15m","1h","4h","1d"]

class MTFAligner:
    def __init__(self, strict: bool = True) -> None:
        self.strict = strict

    def align(self, broker, required_tfs: List[TF]) -> Dict[TF, Dict[str, float]]:
        aligned = broker.aligned_view(required_tfs)
        if self.strict:
            missing = [tf for tf in required_tfs if tf not in aligned]
            if missing:
                # Podríamos marcar calidad de datos; por simplicidad lanzamos.
                raise RuntimeError(f"Faltan TFs para el bar_time actual: {missing}")
        return aligned
