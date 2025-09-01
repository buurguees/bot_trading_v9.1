from __future__ import annotations
from pathlib import Path
import polars as pl

_RULE = {"5m":"5m","15m":"15m","1h":"1h","4h":"4h","1d":"1d"}

class Resampler:
    def __init__(self, ohlcv_root: Path):
        self.root = Path(ohlcv_root)

    def resample_symbol(self, symbol: str, from_tf: str = "1m", to_tfs: list[str] = ["5m","15m","1h","4h","1d"]):
        src_dir = self.root / f"symbol={symbol}" / f"timeframe={from_tf}"
        for to_tf in to_tfs:
            dst_dir = self.root / f"symbol={symbol}" / f"timeframe={to_tf}"
            dst_dir.mkdir(parents=True, exist_ok=True)
            for f in sorted(src_dir.glob("part-*.parquet")):
                df = pl.read_parquet(f).with_columns(
                    pl.from_epoch(pl.col("timestamp") // 1000, time_unit="s").alias("datetime")
                )
                out = (
                    df.group_by_dynamic(index_column="datetime", every=_RULE[to_tf], period=_RULE[to_tf], closed="right")
                      .agg([
                        pl.first("open").alias("open"),
                        pl.max("high").alias("high"),
                        pl.min("low").alias("low"),
                        pl.last("close").alias("close"),
                        pl.sum("volume").alias("volume"),
                        pl.first("exchange").alias("exchange"),
                        pl.first("source").alias("source")
                      ])
                      .with_columns((pl.col("datetime").dt.epoch(time_unit="s") * 1000).alias("timestamp"))
                      .drop("datetime").sort("timestamp")
                )
                out.write_parquet(dst_dir / f.name)
