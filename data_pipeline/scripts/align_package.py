# data_pipeline/scripts/align_package.py
# Descripción:
#   - Lee RAW por TF en data/{SYMBOL}/{market}/raw/{tf}/year=YYYY/month=MM/*.parquet
#   - Normaliza y escribe en ALIGNED (misma granularidad y esquema)
#   - Opcional: genera un "package" por TF (concatenado) en data/{SYMBOL}/{market}/packages/{SYMBOL}_{tf}_{YYYYMM-YYYYMM}.parquet
#
# Nota importante:
#   La "alineación a bar_time del TF base" se resuelve en RUN-TIME por el broker,
#   usando "última barra cerrada ≤ bar_time". Aquí dejaremos los datasets limpios
#   y consistentes por TF (orden, dedupe, tipos).
#
# Uso:
#   python data_pipeline/scripts/align_package.py --symbol BTCUSDT --market spot --tfs 1m,5m,15m,1h --from 2025-03-01 --to 2025-09-01 --make-packages
#
# Requisitos:
#   pip install pyarrow pandas python-dateutil

from __future__ import annotations
import argparse
from pathlib import Path
from datetime import datetime, timezone
from typing import List, Tuple
import pandas as pd
import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.parquet as pq
from dateutil.relativedelta import relativedelta

SCHEMA_COLS = ["ts","open","high","low","close","volume","symbol","market","tf","ingestion_ts"]
DTYPES = {
    "ts":"int64","open":"float64","high":"float64","low":"float64","close":"float64","volume":"float64",
    "symbol":"string","market":"string","tf":"string","ingestion_ts":"int64"
}

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("Alinear/normalizar RAW → ALIGNED y generar packages por TF")
    p.add_argument("--root", type=str, default="data")
    p.add_argument("--symbol", type=str, required=True)
    p.add_argument("--market", type=str, choices=["spot","futures"], required=True)
    p.add_argument("--tfs", type=str, default="1m,5m,15m,1h,4h,1d")
    p.add_argument("--from", dest="date_from", type=str, default=None, help="YYYY-MM-DD (UTC)")
    p.add_argument("--to", dest="date_to", type=str, default=None, help="YYYY-MM-DD (UTC, exclusivo)")
    p.add_argument("--make-packages", action="store_true")
    return p.parse_args()

def month_pairs(date_from: datetime, date_to: datetime) -> List[Tuple[datetime, datetime]]:
    out = []
    cur = date_from.replace(day=1, hour=0, minute=0, second=0, microsecond=0, tzinfo=timezone.utc)
    while cur < date_to:
        nxt = (cur + relativedelta(months=1))
        end = date_to if nxt > date_to else nxt
        out.append((cur, end))
        cur = nxt
    return out

def to_ms(dt: datetime) -> int:
    return int(dt.replace(tzinfo=timezone.utc).timestamp() * 1000)

def normalize_df(df: pd.DataFrame, symbol: str, market: str, tf: str) -> pd.DataFrame:
    if df.empty:
        return df
    # Mantener sólo columnas esperadas y tipos
    cols = {c: df[c] for c in df.columns if c in SCHEMA_COLS}
    df2 = pd.DataFrame(cols)
    # Añadir faltantes
    for c in SCHEMA_COLS:
        if c not in df2.columns:
            if c in ("symbol","market","tf"):
                df2[c] = {"symbol":symbol,"market":market,"tf":tf}[c]
            elif c == "ingestion_ts":
                df2[c] = int(datetime.now(tz=timezone.utc).timestamp() * 1000)
            else:
                raise ValueError(f"Falta columna requerida: {c}")
    # Orden y dedupe
    df2 = df2.sort_values("ts").drop_duplicates(subset=["ts"], keep="last").reset_index(drop=True)
    return df2.astype(DTYPES)

def align_tf(root: Path, symbol: str, market: str, tf: str, y: int, m: int) -> Path | None:
    raw_glob = root / symbol / market / "raw" / tf / f"year={y:04d}" / f"month={m:02d}" / "*.parquet"
    files = list(raw_glob.parent.glob(raw_glob.name))
    if not files:
        return None
    dataset = ds.dataset([str(f) for f in files], format="parquet")
    table = dataset.to_table()
    df = table.to_pandas()
    df = normalize_df(df, symbol, market, tf)

    out_dir = root / symbol / market / "aligned" / tf / f"year={y:04d}" / f"month={m:02d}"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"part-{y:04d}-{m:02d}.parquet"
    pq.write_table(pa.Table.from_pandas(df, preserve_index=False), out_file, compression="zstd")
    return out_file

def package_tf(root: Path, symbol: str, market: str, tf: str, ms_from: int, ms_to: int) -> Path | None:
    # concat por rango en aligned → un sólo parquet por TF
    aligned_glob = root / symbol / market / "aligned" / tf / "year=*" / "month=*" / "*.parquet"
    files = list(aligned_glob.parent.glob(aligned_glob.name))
    if not files:
        return None
    dataset = ds.dataset([str(f) for f in files], format="parquet", partitioning="hive")
    filt = (ds.field("ts") >= pa.scalar(ms_from, pa.int64())) & (ds.field("ts") <= pa.scalar(ms_to, pa.int64()))
    table = dataset.scanner(filter=filt).to_table().sort_by("ts")
    if len(table) == 0:
        return None
    y1 = datetime.utcfromtimestamp(ms_from/1000).strftime("%Y%m")
    y2 = datetime.utcfromtimestamp(ms_to/1000).strftime("%Y%m")
    out_dir = root / symbol / market / "packages"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"{symbol}_{tf}_{y1}-{y2}.parquet"
    pq.write_table(table, out_file, compression="zstd")
    return out_file

def main():
    args = parse_args()
    root = Path(args.root)
    symbol = args.symbol.upper()
    market = args.market.lower()
    tfs = [t.strip() for t in args.tfs.split(",") if t.strip()]

    if args.date_from:
        df = datetime.fromisoformat(args.date_from).replace(tzinfo=timezone.utc)
    else:
        # por defecto últimos 6 meses
        now = datetime.now(timezone.utc)
        df = (now.replace(day=1, hour=0, minute=0, second=0, microsecond=0) - relativedelta(months=5))
    dt = datetime.fromisoformat(args.date_to).replace(tzinfo=timezone.utc) if args.date_to else datetime.now(timezone.utc)

    months = month_pairs(df, dt)
    print(f"[INFO] Alineando {symbol} {market} TFs={tfs} meses={len(months)} → ALIGNED")

    for tf in tfs:
        for (m_start, m_end) in months:
            y, m = m_start.year, m_start.month
            out = align_tf(root, symbol, market, tf, y, m)
            print(f"  - {symbol} {tf} {y}-{m:02d}: {'OK ' + str(out) if out else 'SIN RAW'}")

    if args.make_packages:
        print("[INFO] Generando packages por TF…")
        ms_from, ms_to = int(df.timestamp()*1000), int(dt.timestamp()*1000)
        for tf in tfs:
            p = package_tf(root, symbol, market, tf, ms_from, ms_to)
            print(f"  - package {tf}: {p if p else 'SIN DATOS'}")

    print("[DONE] align_package")
    
if __name__ == "__main__":
    main()
