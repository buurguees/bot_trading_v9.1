from __future__ import annotations
from pathlib import Path
import pandas as pd

def _read_months(dirpath: Path) -> pd.DataFrame:
    dfs = [pd.read_parquet(f) for f in sorted(dirpath.glob("part-*.parquet"))]
    if not dfs:
        return pd.DataFrame(columns=["timestamp","open","high","low","close","volume","exchange","source"])
    return pd.concat(dfs, ignore_index=True).sort_values("timestamp")

def build_mtf_view(symbol: str, ohlcv_root: Path, tfs: dict) -> pd.DataFrame:
    """Vista MTF causal por convención de carpetas parquet."""
    symdir = lambda tf: ohlcv_root / f"symbol={symbol}" / f"timeframe={tf}"
    exec_tf = tfs["execution"][0]
    exec_df = _read_months(symdir(exec_tf)).rename(columns={c: f"{exec_tf}_{c}" for c in ["open","high","low","close","volume"]})
    exec_df = exec_df.rename(columns={f"{exec_tf}_timestamp":"timestamp"}) if "timestamp" in exec_df.columns else exec_df

    def join_asof(base: pd.DataFrame, tf: str, prefix: str) -> pd.DataFrame:
        other = _read_months(symdir(tf))
        if other.empty or base.empty:
            return base
        joined = pd.merge_asof(
            base.sort_values("timestamp"),
            other.sort_values("timestamp")[["timestamp","open","high","low","close","volume"]],
            on="timestamp", direction="backward"
        )
        return joined.rename(columns={k: f"{prefix}{tf}_{k}" for k in ["open","high","low","close","volume"]})

    out = exec_df.copy()
    for tf in tfs.get("direction", []):     out = join_asof(out, tf, "dir_")
    for tf in tfs.get("confirmation", []):  out = join_asof(out, tf, "conf_")
    return out
