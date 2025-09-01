from __future__ import annotations
from pathlib import Path
import pandas as pd

class FeatureStore:
    """Gestión de features calculados: lectura/escritura parquet + convención de rutas."""
    def __init__(self, features_root: Path):
        self.root = Path(features_root)

    def path(self, symbol: str, timeframe: str) -> Path:
        d = self.root / f"symbol={symbol}" / f"timeframe={timeframe}"
        d.mkdir(parents=True, exist_ok=True)
        return d

    def write_part(self, symbol: str, timeframe: str, part_name: str, df: pd.DataFrame):
        (self.path(symbol, timeframe) / part_name).with_suffix(".parquet")
        out = self.path(symbol, timeframe) / f"{part_name}.parquet"
        df.to_parquet(out, index=False)
        return out

    def read_all(self, symbol: str, timeframe: str) -> pd.DataFrame:
        d = self.path(symbol, timeframe)
        parts = sorted(d.glob("*.parquet"))
        if not parts:
            raise FileNotFoundError(f"No features in {d}")
        return pd.concat([pd.read_parquet(p) for p in parts], ignore_index=True).sort_values("timestamp")
