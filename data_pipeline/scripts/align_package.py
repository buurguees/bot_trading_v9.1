# data_pipeline/scripts/align_package.py
# RAW → ALIGNED (+ packages opcional) con validación, robustez y rendimiento.
# Reqs: pyarrow>=12, pandas, python-dateutil

from __future__ import annotations
import argparse
import logging
import os
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, List, Tuple, Optional, Dict
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.parquet as pq
from dateutil import parser as dtparse
from dateutil.relativedelta import relativedelta

# -----------------------
# Config / Constantes
# -----------------------
SCHEMA = pa.schema([
    pa.field("ts", pa.int64()),
    pa.field("open", pa.float64()),
    pa.field("high", pa.float64()),
    pa.field("low", pa.float64()),
    pa.field("close", pa.float64()),
    pa.field("volume", pa.float64()),
    pa.field("symbol", pa.string()),
    pa.field("market", pa.string()),
    pa.field("tf", pa.string()),
    pa.field("ingestion_ts", pa.int64()),
])

SCHEMA_COLS = [f.name for f in SCHEMA]
NUMERIC_COLS = ("open", "high", "low", "close", "volume")

DEFAULT_TFS = ("1m", "5m", "15m", "1h", "4h", "1d")

# -----------------------
# Utilidades
# -----------------------
def parse_iso_date_or_none(s: Optional[str]) -> Optional[datetime]:
    if not s:
        return None
    dt = dtparse.isoparse(s)
    return dt.astimezone(timezone.utc)

def month_pairs(date_from: datetime, date_to: datetime) -> List[Tuple[datetime, datetime]]:
    out: List[Tuple[datetime, datetime]] = []
    cur = date_from.replace(day=1, hour=0, minute=0, second=0, microsecond=0, tzinfo=timezone.utc)
    end = date_to.astimezone(timezone.utc)
    while cur < end:
        nxt = (cur + relativedelta(months=1))
        out.append((cur, min(nxt, end)))
        cur = nxt
    return out

def to_ms(dt: datetime) -> int:
    return int(dt.astimezone(timezone.utc).timestamp() * 1000)

def utc_now_ms() -> int:
    return int(datetime.now(tz=timezone.utc).timestamp() * 1000)

def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)

def atomic_write_table(table: pa.Table, dest: Path, compression: str, row_group_size: Optional[int]) -> None:
    tmp = dest.with_suffix(dest.suffix + ".tmp")
    if tmp.exists():
        tmp.unlink()
    write_opts = {}
    if row_group_size:
        write_opts["row_group_size"] = int(row_group_size)
    pq.write_table(table, tmp, compression=compression, **write_opts)
    os.replace(tmp, dest)

def bar_relations_ok(df: pd.DataFrame) -> pd.Series:
    # H >= max(O, C) y H >= L ; L <= min(O, C) y L <= H
    h_ok = (df["high"] >= df[["open", "close"]].max(axis=1)) & (df["high"] >= df["low"])
    l_ok = (df["low"] <= df[["open", "close"]].min(axis=1)) & (df["low"] <= df["high"])
    return h_ok & l_ok

@dataclass
class Job:
    tf: str
    year: int
    month: int

# -----------------------
# Normalización mensual
# -----------------------
def normalize_month(
    root: Path,
    symbol: str,
    market: str,
    tf: str,
    year: int,
    month: int,
    *,
    strict: bool,
    dry_run: bool,
    compression: str,
    row_group_size: Optional[int],
    overwrite: bool,
    logger: logging.Logger,
) -> Optional[Path]:
    """
    Lee RAW mensual (por TF) → valida/normaliza → escribe ALIGNED mensual.
    Devuelve ruta escrita o None si no había RAW.
    """
    raw_dir = root / symbol / market / "raw" / tf / f"year={year:04d}" / f"month={month:02d}"
    files = sorted(raw_dir.glob("*.parquet"))
    if not files:
        msg = f"[{symbol} {market} {tf} {year}-{month:02d}] SIN RAW"
        if strict:
            logger.error(msg)
        else:
            logger.info(msg)
        return None

    out_dir = root / symbol / market / "aligned" / tf / f"year={year:04d}" / f"month={month:02d}"
    out_file = out_dir / f"part-{year:04d}-{month:02d}.parquet"

    if out_file.exists() and not overwrite:
        logger.info(f"[SKIP] Ya existe: {out_file}")
        return out_file

    dataset = ds.dataset([str(f) for f in files], format="parquet")
    # Cargar todo el mes a Arrow Table
    table = dataset.to_table()  # deja que Arrow use multi-thread interno

    # Asegurar columnas requeridas + tipos: casteamos/creamos faltantes
    present_cols = set(table.column_names)
    add_cols: Dict[str, pa.Array] = {}

    now_ms = utc_now_ms()

    # Relleno de columnas faltantes
    if "symbol" not in present_cols:
        add_cols["symbol"] = pa.array([symbol] * len(table), type=pa.string())
    if "market" not in present_cols:
        add_cols["market"] = pa.array([market] * len(table), type=pa.string())
    if "tf" not in present_cols:
        add_cols["tf"] = pa.array([tf] * len(table), type=pa.string())
    if "ingestion_ts" not in present_cols:
        add_cols["ingestion_ts"] = pa.array([now_ms] * len(table), type=pa.int64())

    if add_cols:
        table = table.append_columns(list(add_cols.keys()), list(add_cols.values()))

    # Cast seguro a SCHEMA (faltantes → nulos)
    table = table.cast(SCHEMA, safe=False)

    # → pandas para dedupe/validación avanzada por mes (datasets mensuales son manejables)
    df = table.to_pandas(types_mapper=pd.ArrowDtype)

    # Limpieza: drop NaN / inf en numéricos y ts
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=["ts", *NUMERIC_COLS])

    # Tipos robustos
    df["ts"] = df["ts"].astype("int64")
    for c in NUMERIC_COLS:
        df[c] = df[c].astype("float64")
    # Asegurar strings
    for c in ("symbol", "market", "tf"):
        df[c] = df[c].astype("string")
    df["ingestion_ts"] = df["ingestion_ts"].astype("int64")

    # Valores válidos: ts>0, volumen>=0
    mask_valid = (df["ts"] > 0) & (df["volume"] >= 0.0)
    df = df.loc[mask_valid]

    # Validación OHLC relacional (descarta barras imposibles)
    df = df.loc[bar_relations_ok(df)]

    # Orden y dedupe: keep last
    # Ordenamos por ts y, si hay ingestion_ts, el más reciente queda último
    if "ingestion_ts" in df.columns:
        df = df.sort_values(["ts", "ingestion_ts"], kind="mergesort")
    else:
        df = df.sort_values("ts", kind="mergesort")

    before = len(df)
    df = df.drop_duplicates(subset=["ts"], keep="last", ignore_index=True)
    after = len(df)

    # Si quedó vacío, salimos
    if df.empty:
        logger.warning(f"[{symbol} {tf} {year}-{month:02d}] Mes vacío tras limpieza")
        return None

    # Reconstruir Arrow Table limpio y casteado a SCHEMA
    clean_tbl = pa.Table.from_pandas(df, schema=SCHEMA, preserve_index=False)

    # Create output
    if dry_run:
        logger.info(f"[DRY-RUN] {symbol} {tf} {year}-{month:02d} -> {before}->{after} filas (NO escribe)")
        return None

    ensure_dir(out_dir)
    atomic_write_table(clean_tbl, out_file, compression=compression, row_group_size=row_group_size)
    logger.info(f"[OK] {symbol} {tf} {year}-{month:02d}: {before}->{after} filas → {out_file}")
    return out_file

# -----------------------
# Package por TF
# -----------------------
def create_tf_package(
    root: Path,
    symbol: str,
    market: str,
    tf: str,
    ms_from: int,
    ms_to: int,
    *,
    dry_run: bool,
    compression: str,
    row_group_size: Optional[int],
    overwrite: bool,
    logger: logging.Logger,
) -> Optional[Path]:
    aligned_glob = root / symbol / market / "aligned" / tf / "year=*" / "month=*" / "*.parquet"
    files = list(aligned_glob.parent.glob(aligned_glob.name))
    if not files:
        logger.info(f"[PKG] {symbol} {tf}: SIN ALIGNED")
        return None

    package_dir = root / symbol / market / "packages"
    ensure_dir(package_dir)
    y1 = datetime.utcfromtimestamp(ms_from / 1000).strftime("%Y%m")
    y2 = datetime.utcfromtimestamp(ms_to / 1000).strftime("%Y%m")
    out_file = package_dir / f"{symbol}_{tf}_{y1}-{y2}.parquet"

    if out_file.exists() and not overwrite:
        logger.info(f"[PKG-SKIP] Ya existe: {out_file}")
        return out_file

    dataset = ds.dataset([str(f) for f in files], format="parquet", partitioning="hive")
    filt = (ds.field("ts") >= pa.scalar(ms_from, pa.int64())) & (ds.field("ts") < pa.scalar(ms_to, pa.int64()))
    scanner = dataset.scan(filter=filt)
    table = scanner.to_table().sort_by("ts")

    if table.num_rows == 0:
        logger.warning(f"[PKG] {symbol} {tf}: sin filas en rango")
        return None

    if dry_run:
        logger.info(f"[PKG-DRY] {symbol} {tf}: {table.num_rows} filas (NO escribe)")
        return None

    atomic_write_table(table, out_file, compression=compression, row_group_size=row_group_size)
    logger.info(f"[PKG-OK] {symbol} {tf}: {table.num_rows} filas → {out_file}")
    return out_file

# -----------------------
# CLI / Orquestación
# -----------------------
def build_jobs(tfs: Iterable[str], months: List[Tuple[datetime, datetime]]) -> List[Job]:
    jobs: List[Job] = []
    for tf in tfs:
        for m_start, _ in months:
            jobs.append(Job(tf=tf, year=m_start.year, month=m_start.month))
    return jobs

def main() -> None:
    p = argparse.ArgumentParser("RAW → ALIGNED y packages por TF (robusto y rápido)")
    p.add_argument("--root", type=str, default="data")
    p.add_argument("--symbol", type=str, required=True)
    p.add_argument("--market", type=str, choices=["spot", "futures"], required=True)
    p.add_argument("--tfs", type=str, default=",".join(DEFAULT_TFS))
    p.add_argument("--from", dest="date_from", type=str, default=None, help="ISO date UTC (e.g. 2025-03-01)")
    p.add_argument("--to", dest="date_to", type=str, default=None, help="ISO date UTC exclusive (e.g. 2025-09-01)")
    p.add_argument("--make-packages", action="store_true")
    p.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 4), help="Hilos para TF/mes")
    p.add_argument("--compression", type=str, default="zstd", choices=["zstd", "snappy", "gzip", "brotli", "lz4", "none"])
    p.add_argument("--row-group-size", type=int, default=128_000, help="Filas por row group (parquet)")
    p.add_argument("--strict", action="store_true", help="Error si faltan RAWs")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--log-level", type=str, default="INFO", choices=["DEBUG","INFO","WARNING","ERROR"])
    args = p.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s | %(levelname)s | %(message)s",
    )
    logger = logging.getLogger("align_package")

    root = Path(args.root)
    symbol = args.symbol.upper().strip()
    market = args.market.lower().strip()
    tfs = [t.strip() for t in args.tfs.split(",") if t.strip()]

    # Rango temporal por defecto: últimos 6 meses
    now = datetime.now(timezone.utc)
    date_from = parse_iso_date_or_none(args.date_from) or (
        (now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)) - relativedelta(months=5)
    )
    date_to = parse_iso_date_or_none(args.date_to) or now

    if date_from >= date_to:
        logger.error("--from debe ser anterior a --to")
        sys.exit(2)

    months = month_pairs(date_from, date_to)
    logger.info(f"Alineando {symbol} [{market}] TFs={tfs} meses={len(months)} → ALIGNED (workers={args.workers})")

    jobs = build_jobs(tfs, months)
    wrote_any = False

    # Paraleliza por TF/mes (I/O bound → ThreadPool va bien con Arrow)
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futures = [
            ex.submit(
                normalize_month,
                root, symbol, market, job.tf, job.year, job.month,
                strict=args.strict,
                dry_run=args.dry_run,
                compression=args.compression if args.compression != "none" else None,
                row_group_size=args.row_group_size,
                overwrite=args.overwrite,
                logger=logger
            )
            for job in jobs
        ]
        for fut in as_completed(futures):
            try:
                res = fut.result()
                if res:
                    wrote_any = True
            except Exception as e:
                logger.exception(f"Error en tarea de alineación: {e}")
                if args.strict:
                    sys.exit(1)

    # Packages por TF
    if args.make_packages:
        logger.info("Generando packages por TF…")
        ms_from, ms_to = to_ms(date_from), to_ms(date_to)
        for tf in tfs:
            try:
                create_tf_package(
                    root, symbol, market, tf, ms_from, ms_to,
                    dry_run=args.dry_run,
                    compression=args.compression if args.compression != "none" else None,
                    row_group_size=args.row_group_size,
                    overwrite=args.overwrite,
                    logger=logger
                )
            except Exception as e:
                logger.exception(f"Error creando package {tf}: {e}")
                if args.strict:
                    sys.exit(1)

    logger.info("[DONE] align_package")

if __name__ == "__main__":
    main()
