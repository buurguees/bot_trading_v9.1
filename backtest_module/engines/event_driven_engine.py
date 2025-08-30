from __future__ import annotations
from pathlib import Path
import pandas as pd
from loguru import logger
from core.oms.execution_sim import ExecutionSim
from core.oms.router import SimRouter
from core.oms.order import Side
from core.portfolio.ledger import Ledger
from core.market.clocks import now_ms

def _load_df(settings: dict, symbol: str, timeframe: str) -> pd.DataFrame:
    base = Path(settings["paths"]["ohlcv_dir"]) / f"symbol={symbol}" / f"timeframe={timeframe}"
    parts = sorted(base.glob("part-*.parquet"))
    if not parts:
        raise FileNotFoundError(f"No OHLCV files in {base}")
    return pd.concat([pd.read_parquet(p) for p in parts], ignore_index=True).sort_values("timestamp")

def run_paper_loop(symbol: str, timeframe: str, settings: dict, heartbeat):
    df = _load_df(settings, symbol, timeframe).tail(10_000)  # demo
    execsim = ExecutionSim()
    router = SimRouter(execsim)
    ledger = Ledger()

    df["sma20"] = df["close"].rolling(20, min_periods=20).mean()
    df["sma50"] = df["close"].rolling(50, min_periods=50).mean()
    df = df.dropna()
    pos = 0.0

    for _, row in df.iterrows():
        ts = int(row["timestamp"])
        best_bid = row["close"]
        best_ask = row["close"]
        signal = 1 if row["sma20"] > row["sma50"] else -1
        if signal == 1 and pos <= 0:
            f = router.place_market(symbol, Side.BUY, qty=0.001, best_bid=best_bid, best_ask=best_ask, ts_ms=ts)
            ledger.apply_fill(f); pos += f.qty
        elif signal == -1 and pos > 0:
            f = router.place_market(symbol, Side.SELL, qty=pos, best_bid=best_bid, best_ask=best_ask, ts_ms=ts)
            ledger.apply_fill(f); pos = 0.0
        heartbeat.ping()
    logger.info({"component":"paper","symbol":symbol,"tf":timeframe,"fills":len(ledger.lines)})
