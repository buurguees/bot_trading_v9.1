# base_env/io/parquet_loader.py
# Descripción: Loader histórico basado en Parquet con PyArrow/DuckDB opcional.
# Lee ventanas por símbolo/TF y rango temporal (ts_from, ts_to) y devuelve un dict
# { ts(int) -> {open, high, low, close, volume, ...} } ordenado por ts ascendente.
# Ubicación: base_env/io/parquet_loader.py
#
# Requisitos:
#   pip install pyarrow   # recomendado
# (Opcional para cargas SQL complejas: pip install duckdb)
#
# Convenciones de disco:
#   data/{SYMBOL}/{market}/{stage}/{tf}/year=YYYY/month=MM/*.parquet
#   - stage: raw | aligned | packages
#   - ts en milisegundos UTC (int64)

from __future__ import annotations
from typing import Dict, Iterable, Optional, Tuple, List
from pathlib import Path

try:
    import pyarrow as pa
    import pyarrow.dataset as ds
    _HAS_PA = True
except Exception:
    _HAS_PA = False

# Columnas mínimas esperadas en parquet
REQUIRED_COLS = ("ts", "open", "high", "low", "close", "volume")


def _stage_path(
    root: str | Path,
    symbol: str,
    market: str,
    stage: str,
    tf: str
) -> Path:
    return Path(root) / symbol / market / stage / tf


def list_files(
    root: str | Path,
    symbol: str,
    market: str,
    stage: str,
    tf: str,
) -> Iterable[Path]:
    """Lista TODOS los ficheros parquet para un símbolo/TF/stage (sin filtrar por rango)."""
    base = _stage_path(root, symbol, market, stage, tf)
    # Soporta particionado year=YYYY/month=MM
    yield from base.glob("year=*/month=*/*.parquet")


def _ensure_required_cols(schema: pa.Schema) -> None:
    names = set(schema.names)
    missing = [c for c in REQUIRED_COLS if c not in names]
    if missing:
        raise ValueError(f"Faltan columnas requeridas en parquet: {missing}")


def load_window(
    root: str | Path,
    symbol: str,
    market: str,
    tf: str,
    ts_from: Optional[int] = None,
    ts_to: Optional[int] = None,
    stage: str = "aligned",
    limit: Optional[int] = None,
) -> Dict[int, Dict[str, float]]:
    """
    Carga una ventana temporal de OHLCV y devuelve un dict indexado por ts.

    Args:
        root: directorio raíz de datos (e.g., "data")
        symbol: e.g., "BTCUSDT"
        market: "spot" | "futures"
        tf: "1m" | "5m" | ...
        ts_from: (incl.) límite inferior en ms
        ts_to: (incl.) límite superior en ms
        stage: "raw" | "aligned" | "packages"
        limit: máximo de filas (opcional; útil en pruebas)

    Returns:
        Dict[int, Dict[str, float]]
    """
    if not _HAS_PA:
        raise RuntimeError("pyarrow no está instalado. Ejecuta: pip install pyarrow")

    files = list(list_files(root, symbol, market, stage, tf))
    if not files:
        return {}

    dataset = ds.dataset([str(f) for f in files], format="parquet", partitioning="hive")

    # Validar columnas mínimas
    _ensure_required_cols(dataset.schema)

    # Construir filtro por rango de ts
    filter_expr = None
    if ts_from is not None:
        filter_expr = (ds.field("ts") >= pa.scalar(ts_from, pa.int64()))
    if ts_to is not None:
        expr_to = (ds.field("ts") <= pa.scalar(ts_to, pa.int64()))
        filter_expr = expr_to if filter_expr is None else (filter_expr & expr_to)

    # Proyección de columnas (mínimas)
    cols = [c for c in REQUIRED_COLS if c in dataset.schema.names]

    scanner = dataset.scanner(columns=cols, filter=filter_expr)
    table = scanner.to_table()

    # Ordenar por ts (por seguridad)
    table = table.sort_by("ts")

    # Aplicar limit si procede
    if limit is not None and limit > 0 and len(table) > limit:
        table = table.slice(len(table) - limit)

    # Convertir a dict
    ts_arr = table.column("ts").to_pylist()
    opens = table.column("open").to_pylist()
    highs = table.column("high").to_pylist()
    lows = table.column("low").to_pylist()
    closes = table.column("close").to_pylist()
    vols = table.column("volume").to_pylist()

    out: Dict[int, Dict[str, float]] = {}
    for i, t in enumerate(ts_arr):
        out[int(t)] = {
            "ts": int(t),
            "open": float(opens[i]),
            "high": float(highs[i]),
            "low": float(lows[i]),
            "close": float(closes[i]),
            "volume": float(vols[i]),
        }
    return out


def load_latest_n(
    root: str | Path,
    symbol: str,
    market: str,
    tf: str,
    n: int = 500,
    stage: str = "aligned",
) -> Dict[int, Dict[str, float]]:
    """
    Devuelve los últimos N bares disponibles para el TF dado.
    Útil para calentar ventanas de indicadores.
    """
    if n <= 0:
        return {}
    # Cargamos sin filtro y limit con PyArrow (tomando los últimos N tras ordenar).
    return load_window(root, symbol, market, tf, ts_from=None, ts_to=None, stage=stage, limit=n)


# -------- Validación de alineación y huecos --------
from ..tfs.calendar import tf_to_ms

def validate_alignment_and_gaps(
    root: str | Path,
    symbol: str,
    market: str,
    tfs: List[str],
    stage: str = "aligned",
    allow_gaps: bool = False,
) -> Dict[str, Dict[str, int]]:
    """
    Valida que para cada TF:
      - Existan datos
      - Las marcas de tiempo estén ordenadas y sobre la rejilla del TF (ts % tf_ms == 0)
      - No haya huecos (diff == tf_ms). Si allow_gaps=True, sólo reporta.

    Devuelve resumen por TF: {tf: {rows, gaps, offgrid, duplicates}}.
    Lanza ValueError si se detectan problemas y allow_gaps=False.
    """
    if not _HAS_PA:
        raise RuntimeError("pyarrow no está instalado. Ejecuta: pip install pyarrow")

    summary: Dict[str, Dict[str, int]] = {}
    problems: List[str] = []

    for tf in tfs:
        files = list(list_files(root, symbol, market, stage, tf))
        if not files:
            problems.append(f"[{tf}] No hay ficheros en {root}/{symbol}/{market}/{stage}/{tf}")
            summary[tf] = {"rows": 0, "gaps": 0, "offgrid": 0, "duplicates": 0}
            continue

        dataset = ds.dataset([str(f) for f in files], format="parquet", partitioning="hive")
        if "ts" not in dataset.schema.names:
            problems.append(f"[{tf}] Falta columna 'ts'")
            summary[tf] = {"rows": 0, "gaps": 0, "offgrid": 0, "duplicates": 0}
            continue

        table = dataset.scanner(columns=["ts"]).to_table().sort_by("ts")
        # Convertir a pandas para deduplicar correctamente
        df = table.to_pandas()
        df = df.drop_duplicates(subset=["ts"], keep="last").sort_values("ts")
        ts_arr = df["ts"].tolist()
        rows = len(ts_arr)
        gaps = 0
        offgrid = 0
        duplicates = 0
        if rows == 0:
            problems.append(f"[{tf}] Sin filas tras lectura")
            summary[tf] = {"rows": 0, "gaps": 0, "offgrid": 0, "duplicates": 0}
            continue

        step = tf_to_ms(tf)  # validación de rejilla
        prev = None
        seen = set()
        for t in ts_arr:
            if (t % step) != 0:
                offgrid += 1
            if prev is not None:
                dt = t - prev
                if dt > step:
                    gaps += (dt // step) - 1
                if dt == 0:
                    duplicates += 1
            prev = t
            if t in seen:
                duplicates += 1
            else:
                seen.add(t)

        summary[tf] = {"rows": rows, "gaps": gaps, "offgrid": offgrid, "duplicates": duplicates}
        if (gaps > 0 or offgrid > 0 or duplicates > 0) and not allow_gaps:
            problems.append(f"[{tf}] gaps={gaps}, offgrid={offgrid}, duplicates={duplicates}")

    if problems and not allow_gaps:
        raise ValueError("Validación de histórico fallida:\n- " + "\n- ".join(problems))

    return summary
