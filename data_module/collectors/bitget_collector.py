import time
from datetime import datetime, timezone
from typing import List, Dict
import ccxt
import pandas as pd
from pathlib import Path

EXCHANGE = "bitget"

def to_ms(dt: datetime) -> int:
    return int(dt.replace(tzinfo=timezone.utc).timestamp() * 1000)

def month_slices(years_back: int) -> List[Dict[str, int]]:
    now = datetime.now(timezone.utc)
    end = datetime(now.year, now.month, 1, tzinfo=timezone.utc)
    start = datetime(end.year - years_back, end.month, 1, tzinfo=timezone.utc)
    cur = start
    out = []
    while cur < end:
        nxt_m = 1 if cur.month == 12 else cur.month + 1
        nxt_y = cur.year + 1 if cur.month == 12 else cur.year
        nxt = datetime(nxt_y, nxt_m, 1, tzinfo=timezone.utc)
        out.append({"since": to_ms(cur), "until": to_ms(nxt)})
        cur = nxt
    return out

def fetch_ohlcv(symbol: str, timeframe: str, years_back: int, out_root: Path):
    ex = ccxt.bitget({"enableRateLimit": True})
    out_root.mkdir(parents=True, exist_ok=True)
    for s in month_slices(years_back):
        since, until = s["since"], s["until"]
        all_rows = []
        cursor = since
        while cursor < until:
            try:
                rows = ex.fetch_ohlcv(symbol, timeframe=timeframe, since=cursor, limit=1000)
                if not rows:
                    break
                df = pd.DataFrame(rows, columns=["timestamp", "open", "high", "low", "close", "volume"])
                df = df.drop_duplicates(subset=["timestamp"]).sort_values("timestamp")
                df["exchange"] = EXCHANGE
                df["source"] = "ccxt"
                all_rows.append(df)
                cursor = int(df["timestamp"].iloc[-1]) + 1
                time.sleep(ex.rateLimit / 1000.0)
            except ccxt.RateLimitExceeded:
                time.sleep(2.0)
            except Exception as e:
                print(f"[WARN] chunk failed: {e}")
                time.sleep(1.0)
        if not all_rows:
            continue
        monthly = pd.concat(all_rows, ignore_index=True).sort_values("timestamp")
        month_str = datetime.utcfromtimestamp(monthly['timestamp'].iloc[0]/1000).strftime("%Y-%m")
        symbol_clean = symbol.replace("/", "")
        out_dir = out_root / f"symbol={symbol_clean}" / f"timeframe={timeframe}"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_file = out_dir / f"part-{month_str}.parquet"
        monthly.to_parquet(out_file, index=False)
        print(f"[OK] {symbol} {timeframe} {month_str} -> {out_file}")
