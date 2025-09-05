
"""
base_env/io/broker.py
Descripción: Interfaz y esqueletos de DataBroker (histórico Parquet / live WS). Entrega OHLCV por TF y cursor temporal.
Datos de entrada: data/{SYMBOL}/{market}/{tf}/... (Parquet) o adapter en vivo.
"""

from __future__ import annotations
from typing import Dict, Optional, Protocol, Literal, Any

TF = Literal["1m", "5m", "15m", "1h", "4h", "1d"]


class DataBroker(Protocol):
    """Contrato mínimo de un broker de datos para el entorno base."""
    def now_ts(self) -> int: ...
    def next(self) -> None: ...
    def reset_to_start(self) -> None: ...
    def get_price(self) -> Optional[float]: ...
    def get_bar(self, tf: TF) -> Optional[Dict[str, float]]: ...
    def aligned_view(self, required_tfs: list[TF]) -> Dict[TF, Dict[str, float]]: ...


class InMemoryBroker:
    """Esqueleto de broker en memoria (histórico). Implementa el contrato y sirve para backtest/train."""

    def __init__(self, series_by_tf: Dict[TF, Dict[int, Dict[str, float]]], base_tf: TF = "1m") -> None:
        self.series_by_tf = series_by_tf
        self.base_tf = base_tf
        self._keys = sorted(series_by_tf.get(base_tf, {}).keys())
        self._i = 0

    def now_ts(self) -> int:
        return self._keys[self._i]

    def next(self) -> None:
        self._i = min(self._i + 1, len(self._keys) - 1)
    
    def reset_to_start(self) -> None:
        """Reinicia el cursor al inicio del histórico (para quiebra)"""
        self._i = 0

    def get_price(self) -> Optional[float]:
        bar = self.get_bar(self.base_tf)
        return None if bar is None else float(bar.get("close", 0.0))

    def get_bar(self, tf: TF) -> Optional[Dict[str, float]]:
        return self.series_by_tf.get(tf, {}).get(self._keys[self._i], None)

    def aligned_view(self, required_tfs: list[TF]) -> Dict[TF, Dict[str, float]]:
        ts = self._keys[self._i]
        out: Dict[TF, Dict[str, float]] = {}
        for tf in required_tfs:
            bar = self.series_by_tf.get(tf, {}).get(ts, None)
            if bar is not None:
                out[tf] = bar
        return out
    
    def is_end_of_data(self) -> bool:
        """Verifica si se ha llegado al final de los datos"""
        return self._i >= len(self._keys) - 1