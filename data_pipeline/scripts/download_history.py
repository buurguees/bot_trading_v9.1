# data_pipeline/scripts/download_history.py
# Descripción: Descarga histórico OHLCV (últimos N meses) desde Binance (Spot o Futuros USDT)
# y lo guarda en Parquet particionado por año/mes:
#   data/{SYMBOL}/raw/{tf}/year=YYYY/month=MM/part-YYYY-MM.parquet
#
# Uso (PowerShell/CMD, desde la raíz del repo):
#   python data_pipeline\scripts\download_history.py --symbol BTCUSDT --market spot --tfs 1m,5m,15m,1h,4h,1d --months 6
#   python data_pipeline\scripts\download_history.py --symbol ETHUSDT --market futures --tfs 1m,5m --months 6
#
# Requisitos:
#   pip install requests pyarrow pandas python-dateutil
#
# Notas:
# - Respeta rate limit con pequeñas pausas.
# - Si re-ejecutas, sobreescribe el archivo del mes (idempotente por partición).
# - Estructura compatible con el loader de Parquet del entorno base.

from __future__ import annotations
import argparse
import os
import sys
import time
import math
import json
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from datetime import datetime, timezone
import requests
import pandas as pd

try:
    import pyarrow as pa
    import pyarrow.parquet as pq
except Exception as e:
    print("ERROR: pyarrow es requerido. Instala con: pip install pyarrow", file=sys.stderr)
    raise

from dateutil.relativedelta import relativedelta


BINANCE_SPOT_BASE = "https://api.binance.com"
BINANCE_FUTURES_BASE = "https://fapi.binance.com"

# Mapa TF local → intervalo Binance
INTERVAL_MAP = {
    "1m": "1m",
    "5m": "5m",
    "15m": "15m",
    "1h": "1h",
    "4h": "4h",
    "1d": "1d",
}

# Columnas (esquema base) y dtypes
SCHEMA_COLUMNS = [
    "ts", "open", "high", "low", "close", "volume",
    "symbol", "market", "tf", "ingestion_ts"
]
DTYPES = {
    "ts": "int64",
    "open": "float64",
    "high": "float64",
    "low": "float64",
    "close": "float64",
    "volume": "float64",
    "symbol": "string",
    "market": "string",
    "tf": "string",
    "ingestion_ts": "int64",
}

HEADERS = {
    "User-Agent": "BaseEnvDataDownloader/1.0 (+https://github.com/)",
    "Accept": "application/json",
}


def to_ms(dt: datetime) -> int:
    return int(dt.replace(tzinfo=timezone.utc).timestamp() * 1000)


def month_range_utc(now_utc: datetime, months_back: int) -> List[Tuple[datetime, datetime]]:
    """
    Devuelve pares (start_utc, end_utc_exclusive) por mes, para los últimos 'months_back' meses.
    El mes actual se incluye hasta 'now'.
    """
    out: List[Tuple[datetime, datetime]] = []
    # inicio = primer día del mes de (now - months_back + 1)
    start_month = (now_utc.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
                   - relativedelta(months=months_back - 1))
    cursor = start_month
    while cursor <= now_utc:
        next_month = (cursor + relativedelta(months=1))
        end = min(next_month, now_utc)
        out.append((cursor, end))
        cursor = next_month
    return out


def binance_klines(
    symbol: str,
    interval: str,
    start_ms: int,
    end_ms: int,
    market: str = "spot",
    limit: int = 1000,
    max_retries: int = 5,
    pause_sec: float = 0.25,
) -> List[List]:
    """
    Descarga klines paginando hasta cubrir [start_ms, end_ms].
    Devuelve lista de arrays (cada kline es la lista estándar de Binance).
    """
    base = BINANCE_SPOT_BASE if market == "spot" else BINANCE_FUTURES_BASE
    endpoint = "/api/v3/klines" if market == "spot" else "/fapi/v1/klines"

    out: List[List] = []
    cursor = start_ms

    while cursor < end_ms:
        params = {
            "symbol": symbol,
            "interval": interval,
            "startTime": cursor,
            "endTime": end_ms,
            "limit": limit,
        }

        for attempt in range(1, max_retries + 1):
            try:
                r = requests.get(base + endpoint, params=params, headers=HEADERS, timeout=15)
                if r.status_code == 429:
                    # rate limit
                    time.sleep(pause_sec * attempt)
                    continue
                r.raise_for_status()
                data = r.json()
                if not isinstance(data, list):
                    raise ValueError(f"Respuesta inesperada: {data}")
                if not data:
                    return out
                out.extend(data)
                # Avanzamos el cursor al cierre de la última vela + 1ms para evitar duplicados
                last_close_ms = int(data[-1][6])  # close time is exclusive end (ms)
                # Protección si el servidor devuelve algo raro
                if last_close_ms <= cursor:
                    cursor = cursor + 1
                else:
                    cursor = last_close_ms
                # Rate limit suave
                time.sleep(pause_sec)
                break
            except Exception as e:
                if attempt == max_retries:
                    raise
                time.sleep(pause_sec * attempt)
    return out


def klines_to_df(
    symbol: str,
    market: str,
    tf: str,
    klines: List[List]
) -> pd.DataFrame:
    """
    Convierte la lista de klines Binance a DataFrame con columnas estándar.
    Binance klines fields:
        0: open time (ms)
        1: open
        2: high
        3: low
        4: close
        5: volume
        6: close time (ms)
        7: quote asset volume
        8: number of trades
        9: taker buy base asset volume
        10: taker buy quote asset volume
        11: ignore
    """
    if not klines:
        return pd.DataFrame(columns=SCHEMA_COLUMNS).astype(DTYPES)

    ot = [int(k[0]) for k in klines]
    o = [float(k[1]) for k in klines]
    h = [float(k[2]) for k in klines]
    l = [float(k[3]) for k in klines]
    c = [float(k[4]) for k in klines]
    v = [float(k[5]) for k in klines]

    now_ms = int(time.time() * 1000)
    df = pd.DataFrame({
        "ts": ot,
        "open": o,
        "high": h,
        "low": l,
        "close": c,
        "volume": v,
        "symbol": symbol,
        "market": market,
        "tf": tf,
        "ingestion_ts": now_ms,
    })
    # Ordenar y deduplicar por ts
    df = df.sort_values("ts").drop_duplicates(subset=["ts"], keep="last").reset_index(drop=True)
    # En Binance, el open time define la vela; close time es (open + size - 1). Ya no lo usamos.
    return df.astype(DTYPES)


def write_parquet_partitioned_month(
    root: Path,
    df: pd.DataFrame,
    symbol: str,
    market: str,
    tf: str,
    month_start_utc: datetime
) -> Path:
    """
    Escribe un archivo parquet por mes en:
      data/{SYMBOL}/{market}/raw/{tf}/year=YYYY/month=MM/part-YYYY-MM.parquet
    Sobrescribe el archivo si existe (idempotente).
    """
    year = month_start_utc.year
    month = month_start_utc.month
    out_dir = root / symbol / market / "raw" / tf / f"year={year:04d}" / f"month={month:02d}"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"part-{year:04d}-{month:02d}.parquet"

    # Guardar con Arrow para schema estable
    table = pa.Table.from_pandas(df, preserve_index=False)
    pq.write_table(table, out_file, compression="zstd")
    return out_file


def download_month_for_tf(
    root: Path,
    symbol: str,
    market: str,
    tf: str,
    month_start_utc: datetime,
    month_end_utc: datetime,
    limit_per_req: int = 1000,
) -> Optional[Path]:
    """
    Descarga un mes para un TF concreto y lo guarda.
    """
    interval = INTERVAL_MAP[tf]
    start_ms = to_ms(month_start_utc)
    end_ms = to_ms(month_end_utc)

    kl = binance_klines(
        symbol=symbol,
        interval=interval,
        start_ms=start_ms,
        end_ms=end_ms,
        market=market,
        limit=limit_per_req,
    )
    df = klines_to_df(symbol, market, tf, kl)
    if df.empty:
        return None
    return write_parquet_partitioned_month(root, df, symbol, market, tf, month_start_utc)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Descargar histórico OHLCV (últimos N meses) y guardar en Parquet particionado.")
    p.add_argument("--root", type=str, default="data", help="Directorio raíz de datos (por defecto: data)")
    p.add_argument("--symbol", type=str, required=True, help="Símbolo (ej. BTCUSDT)")
    p.add_argument("--market", type=str, choices=["spot", "futures"], required=True, help="Mercado: spot | futures")
    p.add_argument("--tfs", type=str, default="1m,5m,15m,1h,4h,1d", help="TFs separados por coma (soportados: 1m,5m,15m,1h,4h,1d)")
    p.add_argument("--months", type=int, default=6, help="Meses hacia atrás (incluyendo el mes actual parcial)")
    p.add_argument("--since", type=str, default=None, help="ISO8601 (UTC) opcional para iniciar (e.g., 2024-03-01T00:00:00Z). Ignora --months si se usa.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(args.root)
    symbol = args.symbol.upper()
    market = args.market.lower()
    
    # Debug: mostrar qué se está parseando
    print(f"[DEBUG] args.tfs raw: '{args.tfs}'")
    tfs = [tf.strip() for tf in args.tfs.split(",") if tf.strip()]
    print(f"[DEBUG] tfs parsed: {tfs}")
    print(f"[DEBUG] INTERVAL_MAP keys: {list(INTERVAL_MAP.keys())}")
    
    for tf in tfs:
        print(f"[DEBUG] Checking TF: '{tf}' (type: {type(tf)})")
        if tf not in INTERVAL_MAP:
            raise ValueError(f"TF no soportado: '{tf}' (tipo: {type(tf)})")

    now_utc = datetime.now(timezone.utc)
    if args.since:
        try:
            # parse simple: YYYY-MM-DD o con T...Z
            s = args.since.replace("Z", "+00:00")
            start_utc = datetime.fromisoformat(s)
            if start_utc.tzinfo is None:
                start_utc = start_utc.replace(tzinfo=timezone.utc)
        except Exception as e:
            raise ValueError("--since debe ser ISO8601, ej. 2024-03-01T00:00:00Z") from e

        # construir meses desde since hasta ahora
        months = []
        cursor = start_utc.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
        while cursor <= now_utc:
            next_m = cursor + relativedelta(months=1)
            end = min(next_m, now_utc)
            months.append((cursor, end))
            cursor = next_m
    else:
        months = month_range_utc(now_utc, args.months)

    print(f"[INFO] Descargando {symbol} ({market}) TFs={tfs} meses={len(months)} → raíz={root}")

    for tf in tfs:
        for (m_start, m_end) in months:
            y, m = m_start.year, m_start.month
            print(f"[INFO] {symbol} {market} {tf} {y}-{m:02d} ...", end=" ", flush=True)
            try:
                out = download_month_for_tf(root, symbol, market, tf, m_start, m_end)
                if out is None:
                    print("sin datos")
                else:
                    print(f"OK → {out}")
            except Exception as e:
                print("ERROR")
                print(f"   └─ {type(e).__name__}: {e}")
                # continúa con el siguiente mes/TF
                time.sleep(0.5)

    print("[DONE] Descarga completada.")


if __name__ == "__main__":
    main()
