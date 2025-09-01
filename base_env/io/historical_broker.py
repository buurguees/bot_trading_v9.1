# base_env/io/historical_broker.py
# Descripción:
#   Broker histórico que consume Parquet "aligned" por TF y entrega:
#   - now_ts()
#   - next()
#   - get_price() (close del TF base)
#   - get_bar(tf) (última barra cerrada ≤ now_ts para ese TF)
#   - aligned_view(required_tfs)
#
#   Avanza sobre la secuencia de timestamps del TF base.
#   Para TFs superiores, selecciona la última barra con ts <= now_ts.
#
# Ubicación: base_env/io/historical_broker.py

from __future__ import annotations
from typing import Dict, Optional, Literal, List
from pathlib import Path
from bisect import bisect_right

from .parquet_loader import load_window
from ..tfs.calendar import TF, tf_to_ms

class ParquetHistoricalBroker:
    def __init__(
        self,
        data_root: str | Path,
        symbol: str,
        market: Literal["spot","futures"],
        tfs: List[TF],
        base_tf: TF = "1m",
        ts_from: Optional[int] = None,
        ts_to: Optional[int] = None,
        stage: str = "aligned",
        warmup_bars: int = 2000,
    ) -> None:
        self.root = Path(data_root)
        self.symbol = symbol
        self.market = market
        self.tfs = tfs
        self.base_tf = base_tf
        self.ts_from = ts_from
        self.ts_to = ts_to
        self.stage = stage

        # Cargar ventanas por TF (últimos warmup + rango)
        self.series_by_tf: Dict[TF, Dict[int, Dict[str, float]]] = {}
        self.ts_index_by_tf: Dict[TF, List[int]] = {}

        for tf in tfs:
            series = load_window(self.root, symbol, market, tf, ts_from=ts_from, ts_to=ts_to, stage=stage)
            # Si no hay rango, intenta al menos últimos N para calentar
            if not series:
                series = load_window(self.root, symbol, market, tf, ts_from=None, ts_to=None, stage=stage, limit=warmup_bars)
            # Guardar series ordenadas por ts
            ordered_ts = sorted(series.keys())
            self.series_by_tf[tf] = series
            self.ts_index_by_tf[tf] = ordered_ts

        # Base timeline
        base_ts_list = self.ts_index_by_tf.get(base_tf, [])
        if not base_ts_list:
            raise RuntimeError(f"No hay datos para TF base {base_tf} en {symbol}/{market}/{stage}")
        self._base_ts_list = base_ts_list
        # Cursor en timeline base
        self._i = 0

    # ------------- API -------------
    def now_ts(self) -> int:
        return self._base_ts_list[self._i]

    def next(self) -> None:
        if self._i < len(self._base_ts_list) - 1:
            self._i += 1

    def get_price(self) -> Optional[float]:
        bar = self.get_bar(self.base_tf)
        return None if bar is None else float(bar.get("close", 0.0))

    def get_bar(self, tf: TF) -> Optional[Dict[str, float]]:
        """Devuelve la última barra con ts <= now_ts para el TF dado (cierre disponible)."""
        bars = self.series_by_tf.get(tf, {})
        ts_list = self.ts_index_by_tf.get(tf, [])
        if not ts_list:
            return None
        ts_now = self.now_ts()
        # Encontrar posición de inserción a la derecha y retroceder una
        pos = bisect_right(ts_list, ts_now) - 1
        if pos < 0:
            return None
        ts = ts_list[pos]
        return bars.get(ts, None)

    def aligned_view(self, required_tfs: List[TF]) -> Dict[TF, Dict[str, float]]:
        out: Dict[TF, Dict[str, float]] = {}
        for tf in required_tfs:
            bar = self.get_bar(tf)
            if bar is not None:
                out[tf] = bar
        return out
