from __future__ import annotations
from pathlib import Path
import pandas as pd
from loguru import logger

def _load_exec_df(settings: dict, symbol: str, timeframe: str) -> pd.DataFrame:
    base = Path(settings["paths"]["ohlcv_dir"]) / f"symbol={symbol}" / f"timeframe={timeframe}"
    parts = sorted(base.glob("part-*.parquet"))
    if not parts:
        raise FileNotFoundError(f"No OHLCV files in {base}")
    df = pd.concat([pd.read_parquet(p) for p in parts], ignore_index=True).sort_values("timestamp")
    return df

def run_sma_crossover_backtest(symbol: str, timeframe: str, settings: dict) -> dict:
    df = _load_exec_df(settings, symbol, timeframe)
    close = df["close"]
    df["sma_fast"] = close.rolling(20, min_periods=20).mean()
    df["sma_slow"] = close.rolling(50, min_periods=50).mean()
    df = df.dropna().copy()
    df["signal"] = 0
    df.loc[df["sma_fast"] > df["sma_slow"], "signal"] = 1
    df.loc[df["sma_fast"] < df["sma_slow"], "signal"] = -1
    df["ret"] = close.pct_change().fillna(0.0)
    df["strat"] = df["signal"].shift(1).fillna(0.0) * df["ret"]
    equity = (1 + df["strat"]).cumprod()
    result = {
        "rows": int(len(df)),
        "equity_final": float(equity.iloc[-1]),
        "sharpe_like": float(df["strat"].mean() / (df["strat"].std() + 1e-12) * (252**0.5))
    }
    logger.info({"component":"vec_bt","symbol":symbol,"tf":timeframe, **result})
    return result
