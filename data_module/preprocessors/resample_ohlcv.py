from pathlib import Path
import polars as pl

RULE = {"5m":"5m","15m":"15m","1h":"1h","4h":"4h","1d":"1d"}

def resample_dir(symbol="BTCUSDT", from_tf="1m", to_tf="5m"):
    src_dir = Path(f"data/warehouse/ohlcv/symbol={symbol}/timeframe={from_tf}")
    dst_dir = Path(f"data/warehouse/ohlcv/symbol={symbol}/timeframe={to_tf}")
    dst_dir.mkdir(parents=True, exist_ok=True)
    for f in sorted(src_dir.glob("part-*.parquet")):
        df = pl.read_parquet(f).with_columns((pl.col("timestamp") // 1000).cast(pl.Int64).alias("ts_s"))
        out = (
            df.group_by_dynamic(index_column="ts_s", every=RULE[to_tf], period=RULE[to_tf], closed="right")
              .agg([
                pl.first("open").alias("open"),
                pl.max("high").alias("high"),
                pl.min("low").alias("low"),
                pl.last("close").alias("close"),
                pl.sum("volume").alias("volume"),
                pl.first("exchange").alias("exchange"),
                pl.first("source").alias("source")
              ])
              .with_columns((pl.col("ts_s") * 1000).alias("timestamp"))
              .drop("ts_s")
              .sort("timestamp")
        )
        out.write_parquet(dst_dir / f.name)
        print(f"[OK] resampled {f.name} -> timeframe={to_tf}")
