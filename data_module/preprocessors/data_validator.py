from pathlib import Path
import polars as pl

def validate_ohlcv(dir_path: str) -> dict:
    issues = {"files":0, "dups":0, "nan":0, "unsorted":0, "negatives":0}
    for f in sorted(Path(dir_path).glob("part-*.parquet")):
        issues["files"] += 1
        df = pl.read_parquet(f)
        if not df["timestamp"].is_sorted().all():
            issues["unsorted"] += 1
        dups = df["timestamp"].n_unique() != df.height
        if dups:
            issues["dups"] += 1
        nan = df.select(pl.all().is_null().any()).row(0)[0]
        if nan:
            issues["nan"] += 1
        neg = df.select((pl.col("open")<=0)|(pl.col("high")<=0)|(pl.col("low")<=0)|(pl.col("close")<=0)).any().row(0)[0]
        if neg:
            issues["negatives"] += 1
    return issues
