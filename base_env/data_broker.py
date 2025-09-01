from __future__ import annotations
from pathlib import Path
import typing as t
import pandas as pd

class DataBroker:
    """Acceso canónico a OHLCV en parquet: load_by_months, load_range, iter_bars."""
    def __init__(self, ohlcv_root: Path):
        self.root = Path(ohlcv_root)

    def _dir(self, symbol: str, timeframe: str) -> Path:
        return self.root / f"symbol={symbol}" / f"timeframe={timeframe}"

    def list_parts(self, symbol: str, timeframe: str) -> list[Path]:
        d = self._dir(symbol, timeframe)
        return sorted(d.glob("part-*.parquet"))

    def load_all(self, symbol: str, timeframe: str) -> pd.DataFrame:
        parts = self.list_parts(symbol, timeframe)
        if not parts:
            raise FileNotFoundError(f"OHLCV not found for {symbol} {timeframe} in {self._dir(symbol,timeframe)}")
        df = pd.concat([pd.read_parquet(p) for p in parts], ignore_index=True).sort_values("timestamp")
        return df

    def load_range(self, symbol: str, timeframe: str, since_ms: int | None = None, until_ms: int | None = None) -> pd.DataFrame:
        df = self.load_all(symbol, timeframe)
        if since_ms is not None:
            df = df[df["timestamp"] >= since_ms]
        if until_ms is not None:
            df = df[df["timestamp"] < until_ms]
        return df.reset_index(drop=True)

    def iter_bars(self, symbol: str, timeframe: str, batch_rows: int = 10_000):
        """Itera en lotes (útil para entrenamientos largos sin cargar todo en RAM)."""
        parts = self.list_parts(symbol, timeframe)
        for p in parts:
            df = pd.read_parquet(p).sort_values("timestamp")
            for i in range(0, len(df), batch_rows):
                yield df.iloc[i:i+batch_rows].copy()
