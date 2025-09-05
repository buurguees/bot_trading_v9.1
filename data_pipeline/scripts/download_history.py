# data_pipeline/scripts/download_history.py
"""
Descripción: Descarga histórico OHLCV (últimos N meses) desde Binance/Bitget
y lo guarda en Parquet particionado por año/mes con validación de calidad.

Estructura de salida:
  data/{SYMBOL}/raw/{tf}/year=YYYY/month=MM/part-YYYY-MM.parquet

Uso (PowerShell/CMD, desde la raíz del repo):
  python data_pipeline/scripts/download_history.py --symbol BTCUSDT --market spot --tfs 1m,5m,15m,1h,4h,1d --months 36
  python data_pipeline/scripts/download_history.py --symbol ETHUSDT --market futures --tfs 1m,5m --months 36

Requisitos:
  pip install requests pyarrow pandas python-dateutil structlog pydantic

Autor: Trading Bot v9.1
"""

from __future__ import annotations
import argparse
import os
import sys
import time
import json
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Literal
from datetime import datetime, timezone
from dataclasses import dataclass

import requests
import pandas as pd
import yaml
import structlog

try:
    import pyarrow as pa
    import pyarrow.parquet as pq
except ImportError as e:
    print("ERROR: pyarrow es requerido. Instala con: pip install pyarrow", file=sys.stderr)
    raise

from dateutil.relativedelta import relativedelta
from pydantic import BaseModel, Field, validator

# Configurar logging estructurado
structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
        structlog.processors.JSONRenderer()
    ],
    context_class=dict,
    logger_factory=structlog.stdlib.LoggerFactory(),
    wrapper_class=structlog.stdlib.BoundLogger,
    cache_logger_on_first_use=True,
)

logger = structlog.get_logger(__name__)

# Importar collector de Bitget
try:
    from data_pipeline.collectors.bitget_futures_collector import create_bitget_collector
except ImportError:
    logger.warning("bitget_collector_unavailable", message="Solo Binance disponible")
    create_bitget_collector = None


# Configuraciones
BINANCE_SPOT_BASE = "https://api.binance.com"
BINANCE_FUTURES_BASE = "https://fapi.binance.com"

INTERVAL_MAP = {
    "1m": "1m", "5m": "5m", "15m": "15m",
    "1h": "1h", "4h": "4h", "1d": "1d",
}

SCHEMA_COLUMNS = [
    "ts", "open", "high", "low", "close", "volume",
    "symbol", "market", "tf", "ingestion_ts"
]

DTYPES = {
    "ts": "int64", "open": "float64", "high": "float64", "low": "float64",
    "close": "float64", "volume": "float64", "symbol": "string",
    "market": "string", "tf": "string", "ingestion_ts": "int64",
}

HEADERS = {
    "User-Agent": "TradingBot-v9.1/1.0 (+https://github.com/)",
    "Accept": "application/json",
}

# Timeframe intervals en ms para validación
TF_INTERVALS_MS = {
    "1m": 60_000, "5m": 300_000, "15m": 900_000,
    "1h": 3_600_000, "4h": 14_400_000, "1d": 86_400_000,
}


@dataclass
class DataQualityMetrics:
    """Métricas de calidad de datos OHLCV"""
    total_bars: int
    duplicates_removed: int
    zero_volume_bars: int
    invalid_ohlc_bars: int
    time_gaps: int
    data_quality_score: float
    issues: List[str]

    def is_acceptable(self, min_score: float = 85.0) -> bool:
        """Determina si la calidad es aceptable"""
        return self.data_quality_score >= min_score


class DownloadConfig(BaseModel):
    """Configuración validada para descarga"""
    symbol: str = Field(..., min_length=3, max_length=20)
    market: Literal["spot", "futures"]
    timeframes: List[str] = Field(..., min_items=1)
    months: int = Field(default=36, ge=1, le=60)
    exchange: str = Field(default="binance")
    max_retries: int = Field(default=5, ge=1, le=10)
    rate_limit_pause: float = Field(default=0.5, ge=0.1, le=5.0)

    @validator('timeframes')
    def validate_timeframes(cls, v):
        invalid = [tf for tf in v if tf not in INTERVAL_MAP]
        if invalid:
            raise ValueError(f"TFs no soportados: {invalid}")
        return v

    @validator('symbol')
    def validate_symbol(cls, v):
        return v.upper().strip()


class DownloadError(Exception):
    """Error específico de descarga"""
    pass


class DataQualityError(Exception):
    """Error de calidad de datos"""
    pass


def load_exchange_config() -> str:
    """Carga la configuración de exchange desde config/settings.yaml"""
    try:
        settings_path = Path("config/settings.yaml")
        if settings_path.exists():
            with open(settings_path, 'r') as f:
                settings = yaml.safe_load(f)
                exchange = settings.get("exchange", "binance")
                logger.info("config_loaded", exchange=exchange, source="settings.yaml")
                return exchange
        else:
            logger.warning("config_not_found", path=str(settings_path), default="binance")
            return "binance"
    except Exception as e:
        logger.error("config_load_error", error=str(e), default="binance")
        return "binance"


def validate_ohlcv_data(df: pd.DataFrame, tf: str, symbol: str, duplicates_removed: int = 0) -> DataQualityMetrics:
    """Valida integridad y calidad de datos OHLCV"""
    if df.empty:
        return DataQualityMetrics(
            total_bars=0, duplicates_removed=0, zero_volume_bars=0,
            invalid_ohlc_bars=0, time_gaps=0, data_quality_score=0.0,
            issues=["No data received"]
        )

    original_count = len(df)
    issues = []

    # 1. Validar lógica OHLC
    invalid_ohlc = df[
        (df['high'] < df['open']) | (df['high'] < df['close']) |
        (df['low'] > df['open']) | (df['low'] > df['close']) |
        (df['high'] < df['low'])
    ]
    invalid_ohlc_count = len(invalid_ohlc)
    if invalid_ohlc_count > 0:
        issues.append(f"Invalid OHLC logic: {invalid_ohlc_count} bars")

    # 2. Detectar barras con volumen cero
    zero_volume = len(df[df['volume'] <= 0])
    if zero_volume > original_count * 0.05:  # > 5% es sospechoso
        issues.append(f"High zero-volume ratio: {zero_volume}/{original_count}")

    # 3. Detectar gaps temporales críticos
    if len(df) > 1:
        expected_interval = TF_INTERVALS_MS[tf]
        df_sorted = df.sort_values('ts')
        time_diffs = df_sorted['ts'].diff().dropna()
        
        # Gaps > 2x el intervalo esperado
        critical_gaps = time_diffs[time_diffs > expected_interval * 2]
        gap_count = len(critical_gaps)
        
        if gap_count > 0:
            max_gap_hours = critical_gaps.max() / (1000 * 3600)
            issues.append(f"Time gaps detected: {gap_count} (max: {max_gap_hours:.1f}h)")
    else:
        gap_count = 0

    # 4. Duplicados ya están removidos, usar el valor pasado

    # 5. Calcular score de calidad (0-100)
    penalty = 0
    penalty += min(30, invalid_ohlc_count * 2)  # Max -30 por OHLC inválido
    penalty += min(20, (zero_volume / max(1, original_count)) * 100)  # Max -20 por volumen cero
    penalty += min(25, gap_count * 3)  # Max -25 por gaps
    
    quality_score = max(0.0, 100.0 - penalty)

    return DataQualityMetrics(
        total_bars=len(df),
        duplicates_removed=duplicates_removed,
        zero_volume_bars=zero_volume,
        invalid_ohlc_bars=invalid_ohlc_count,
        time_gaps=gap_count,
        data_quality_score=quality_score,
        issues=issues
    )


def to_ms(dt: datetime) -> int:
    """Convierte datetime a timestamp en milisegundos"""
    return int(dt.replace(tzinfo=timezone.utc).timestamp() * 1000)


def month_range_utc(now_utc: datetime, months_back: int) -> List[Tuple[datetime, datetime]]:
    """Genera rangos mensuales para descarga"""
    out: List[Tuple[datetime, datetime]] = []
    start_month = (now_utc.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
                   - relativedelta(months=months_back - 1))
    cursor = start_month
    
    while cursor <= now_utc:
        next_month = cursor + relativedelta(months=1)
        end = min(next_month, now_utc)
        out.append((cursor, end))
        cursor = next_month
    
    return out


def bitget_klines(
    symbol: str,
    interval: str,
    start_ms: int,
    end_ms: int,
    market: str = "futures",
    limit: int = 1000,
    max_retries: int = 5,
    pause_sec: float = 0.1,
) -> List[List]:
    """Descarga klines desde Bitget usando el collector"""
    if create_bitget_collector is None:
        raise ImportError("Bitget collector no disponible")
    
    try:
        collector = create_bitget_collector()
        data = collector.fetch_ohlcv(symbol, interval, start_ms, end_ms, limit, max_retries, pause_sec)
        
        # Convertir a formato compatible
        klines = []
        interval_ms = TF_INTERVALS_MS.get(interval, 60000)  # Usar intervalo real
        for record in data:
            klines.append([
                record["ts"], record["open"], record["high"], record["low"],
                record["close"], record["volume"], record["ts"] + interval_ms,
                record.get("quote_volume", 0.0), 0, 0, 0, 0
            ])
        
        logger.info("bitget_klines_fetched", 
                   symbol=symbol, interval=interval, count=len(klines))
        return klines
        
    except Exception as e:
        logger.error("bitget_fetch_error", 
                    symbol=symbol, interval=interval, error=str(e))
        raise DownloadError(f"Bitget fetch failed: {e}") from e


def binance_klines(
    symbol: str,
    interval: str,
    start_ms: int,
    end_ms: int,
    market: str = "spot",
    limit: int = 1000,
    max_retries: int = 5,
    pause_sec: float = 0.5,
) -> List[List]:
    """Descarga klines desde Binance con manejo robusto de errores"""
    base = BINANCE_SPOT_BASE if market == "spot" else BINANCE_FUTURES_BASE
    endpoint = "/api/v3/klines" if market == "spot" else "/fapi/v1/klines"

    out: List[List] = []
    cursor = start_ms
    request_count = 0

    logger.info("binance_download_started", 
               symbol=symbol, interval=interval, 
               start_ms=start_ms, end_ms=end_ms)

    while cursor < end_ms:
        params = {
            "symbol": symbol, "interval": interval,
            "startTime": cursor, "endTime": end_ms, "limit": limit,
        }

        for attempt in range(1, max_retries + 1):
            try:
                request_count += 1
                
                logger.debug("api_request", 
                           symbol=symbol, cursor=cursor, 
                           attempt=attempt, request_count=request_count)
                
                response = requests.get(
                    base + endpoint, params=params, 
                    headers=HEADERS, timeout=30
                )
                
                # Manejo específico por código de estado
                if response.status_code == 429:
                    retry_after = int(response.headers.get("Retry-After", pause_sec * attempt))
                    logger.warning("rate_limited", 
                                 symbol=symbol, retry_after=retry_after, 
                                 attempt=attempt)
                    time.sleep(retry_after)
                    continue
                
                elif response.status_code >= 500:
                    logger.error("server_error", 
                               symbol=symbol, status_code=response.status_code,
                               attempt=attempt)
                    if attempt == max_retries:
                        raise DownloadError(f"Server error {response.status_code} after {max_retries} attempts")
                    time.sleep(pause_sec * attempt * 2)  # Backoff exponencial
                    continue
                
                response.raise_for_status()
                data = response.json()
                
                if not isinstance(data, list):
                    raise DownloadError(f"Invalid response format: {type(data)}")
                
                if not data:
                    logger.info("no_more_data", symbol=symbol, cursor=cursor)
                    break
                
                out.extend(data)
                
                # Avanzar cursor
                last_close_ms = int(data[-1][6])
                if last_close_ms <= cursor:
                    logger.warning("cursor_not_advancing", 
                                 symbol=symbol, cursor=cursor, last_close=last_close_ms)
                    cursor = cursor + TF_INTERVALS_MS.get(interval, 60000)
                else:
                    cursor = last_close_ms
                
                # Rate limiting preventivo
                time.sleep(pause_sec)
                break
                
            except requests.Timeout:
                logger.error("request_timeout", 
                           symbol=symbol, attempt=attempt, timeout=30)
                if attempt == max_retries:
                    raise DownloadError(f"Timeout after {max_retries} attempts")
                time.sleep(pause_sec * attempt)
                
            except requests.HTTPError as e:
                logger.error("http_error", 
                           symbol=symbol, status_code=e.response.status_code,
                           error=str(e), attempt=attempt)
                if attempt == max_retries:
                    raise DownloadError(f"HTTP error: {e}") from e
                time.sleep(pause_sec * attempt)
                
            except Exception as e:
                logger.error("unexpected_error", 
                           symbol=symbol, error=str(e), 
                           error_type=type(e).__name__, attempt=attempt)
                if attempt == max_retries:
                    raise DownloadError(f"Unexpected error: {e}") from e
                time.sleep(pause_sec * attempt)

    logger.info("binance_download_completed", 
               symbol=symbol, interval=interval, 
               total_bars=len(out), total_requests=request_count)
    
    return out


def klines_to_df(symbol: str, market: str, tf: str, klines: List[List]) -> Tuple[pd.DataFrame, int]:
    """Convierte klines a DataFrame con validación y deduplicación"""
    if not klines:
        logger.warning("empty_klines", symbol=symbol, tf=tf)
        return pd.DataFrame(columns=SCHEMA_COLUMNS).astype(DTYPES), 0

    try:
        # Extraer campos
        timestamps = [int(k[0]) for k in klines]
        opens = [float(k[1]) for k in klines]
        highs = [float(k[2]) for k in klines]
        lows = [float(k[3]) for k in klines]
        closes = [float(k[4]) for k in klines]
        volumes = [float(k[5]) for k in klines]

        now_ms = int(time.time() * 1000)
        
        df = pd.DataFrame({
            "ts": timestamps, "open": opens, "high": highs, "low": lows,
            "close": closes, "volume": volumes, "symbol": symbol,
            "market": market, "tf": tf, "ingestion_ts": now_ms,
        })
        
        # Ordenar y deduplicar
        original_len = len(df)
        df = df.sort_values("ts").drop_duplicates(subset=["ts"], keep="last").reset_index(drop=True)
        duplicates_removed = original_len - len(df)
        
        if duplicates_removed > 0:
            logger.warning("duplicates_removed", 
                         symbol=symbol, tf=tf, 
                         count=duplicates_removed, 
                         original=original_len, final=len(df))

        return df.astype(DTYPES), duplicates_removed
        
    except (ValueError, KeyError, IndexError) as e:
        logger.error("klines_conversion_error", 
                    symbol=symbol, tf=tf, error=str(e))
        raise DownloadError(f"Failed to convert klines: {e}") from e


def write_parquet_partitioned_month(
    root: Path,
    df: pd.DataFrame,
    symbol: str,
    market: str,
    tf: str,
    month_start_utc: datetime
) -> Path:
    """Escribe DataFrame a Parquet particionado por mes"""
    year = month_start_utc.year
    month = month_start_utc.month
    
    out_dir = root / symbol / market / "raw" / tf / f"year={year:04d}" / f"month={month:02d}"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"part-{year:04d}-{month:02d}.parquet"

    try:
        # Guardar con Arrow para schema estable
        table = pa.Table.from_pandas(df, preserve_index=False)
        pq.write_table(table, out_file, compression="zstd")
        
        file_size = out_file.stat().st_size / (1024 * 1024)  # MB
        logger.info("parquet_written", 
                   file=str(out_file), 
                   rows=len(df), 
                   size_mb=round(file_size, 2))
        
        return out_file
        
    except Exception as e:
        logger.error("parquet_write_error", 
                    file=str(out_file), error=str(e))
        raise DownloadError(f"Failed to write parquet: {e}") from e


def download_month_for_tf(
    root: Path,
    config: DownloadConfig,
    tf: str,
    month_start_utc: datetime,
    month_end_utc: datetime,
) -> Optional[Path]:
    """Descarga y valida un mes de datos para un timeframe específico"""
    interval = INTERVAL_MAP[tf]
    start_ms = to_ms(month_start_utc)
    end_ms = to_ms(month_end_utc)

    try:
        # Validar rangos temporales
        if start_ms >= end_ms:
            logger.warning("invalid_time_range", 
                         symbol=config.symbol, tf=tf,
                         start_ms=start_ms, end_ms=end_ms)
            return None
            
        # Descargar según exchange
        if config.exchange == "bitget" and config.market == "futures":
            klines = bitget_klines(
                config.symbol, interval, start_ms, end_ms,
                config.market, 1000, config.max_retries, config.rate_limit_pause
            )
        else:
            klines = binance_klines(
                config.symbol, interval, start_ms, end_ms,
                config.market, 1000, config.max_retries, config.rate_limit_pause
            )

        # Convertir a DataFrame
        df, duplicates_removed = klines_to_df(config.symbol, config.market, tf, klines)
        
        if df.empty:
            logger.info("no_data_for_month", 
                       symbol=config.symbol, tf=tf, 
                       year=month_start_utc.year, month=month_start_utc.month)
            return None

        # Validar calidad
        metrics = validate_ohlcv_data(df, tf, config.symbol, duplicates_removed)
        
        logger.info("data_quality_check", 
                   symbol=config.symbol, tf=tf,
                   quality_score=metrics.data_quality_score,
                   total_bars=metrics.total_bars,
                   issues=len(metrics.issues))

        if not metrics.is_acceptable(min_score=80.0):  # Umbral más estricto
            logger.warning("low_quality_data", 
                         symbol=config.symbol, tf=tf,
                         score=metrics.data_quality_score,
                         issues=metrics.issues)
            # Continuar pero con warning (no es crítico)

        # Escribir archivo
        return write_parquet_partitioned_month(
            root, df, config.symbol, config.market, tf, month_start_utc
        )

    except DownloadError:
        raise
    except Exception as e:
        logger.error("download_month_error", 
                    symbol=config.symbol, tf=tf, 
                    year=month_start_utc.year, month=month_start_utc.month,
                    error=str(e), error_type=type(e).__name__)
        raise DownloadError(f"Failed to download month: {e}") from e


def parse_args() -> argparse.Namespace:
    """Parser de argumentos con validación"""
    p = argparse.ArgumentParser(
        description="Descargar histórico OHLCV con validación de calidad",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ejemplos:
  python data_pipeline/scripts/download_history.py --symbol BTCUSDT --market spot --tfs 1m,5m,1h --months 12
  python data_pipeline/scripts/download_history.py --symbol ETHUSDT --market futures --tfs 1m --months 6 --max-months 3
        """
    )
    
    p.add_argument("--root", type=str, default="data", 
                   help="Directorio raíz de datos")
    p.add_argument("--symbol", type=str, required=True, 
                   help="Símbolo de trading (ej. BTCUSDT)")
    p.add_argument("--market", type=str, choices=["spot", "futures"], required=True,
                   help="Tipo de mercado")
    p.add_argument("--tfs", type=str, default="1m,5m,15m,1h,4h,1d",
                   help="Timeframes separados por coma")
    p.add_argument("--months", type=int, default=36,
                   help="Meses hacia atrás a descargar")
    p.add_argument("--since", type=str, default=None,
                   help="Fecha inicial ISO8601 (ignora --months)")
    p.add_argument("--max-months", type=int, default=None,
                   help="Límite máximo de meses (útil para testing)")
    p.add_argument("--verbose", "-v", action="store_true",
                   help="Logging verbose")
    
    return p.parse_args()


def main() -> None:
    """Función principal con manejo robusto de errores"""
    try:
        args = parse_args()
        
        # Configurar nivel de logging
        if args.verbose:
            import logging
            logging.getLogger().setLevel(logging.DEBUG)

        # Validar configuración
        root = Path(args.root)
        exchange = load_exchange_config()
        
        timeframes = [tf.strip() for tf in args.tfs.split(",") if tf.strip()]
        
        config = DownloadConfig(
            symbol=args.symbol,
            market=args.market,
            timeframes=timeframes,
            months=args.months,
            exchange=exchange
        )
        
        logger.info("download_started", 
                   symbol=config.symbol, market=config.market,
                   timeframes=config.timeframes, months=config.months,
                   exchange=config.exchange, root=str(root))

        # Calcular rangos temporales
        now_utc = datetime.now(timezone.utc)
        
        if args.since:
            try:
                since_str = args.since.replace("Z", "+00:00")
                start_utc = datetime.fromisoformat(since_str)
                if start_utc.tzinfo is None:
                    start_utc = start_utc.replace(tzinfo=timezone.utc)
            except Exception as e:
                logger.error("invalid_since_format", since=args.since, error=str(e))
                raise ValueError("--since debe ser formato ISO8601 (ej. 2024-03-01T00:00:00Z)")

            # Generar meses desde 'since'
            months = []
            cursor = start_utc.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
            while cursor <= now_utc:
                next_month = cursor + relativedelta(months=1)
                end = min(next_month, now_utc)
                months.append((cursor, end))
                cursor = next_month
        else:
            months = month_range_utc(now_utc, config.months)

        # Aplicar límite si se especifica
        if args.max_months and len(months) > args.max_months:
            logger.info("limiting_months", 
                       requested=len(months), limit=args.max_months)
            months = months[:args.max_months]

        # Estadísticas
        total_tasks = len(config.timeframes) * len(months)
        completed_tasks = 0
        failed_tasks = 0
        
        logger.info("download_plan", 
                   total_tasks=total_tasks, 
                   timeframes=len(config.timeframes),
                   months=len(months))

        # Procesar cada combinación TF/mes
        for tf in config.timeframes:
            for month_start, month_end in months:
                completed_tasks += 1
                y, m = month_start.year, month_start.month
                
                logger.info("processing_task", 
                           progress=f"{completed_tasks}/{total_tasks}",
                           symbol=config.symbol, tf=tf, 
                           year=y, month=m)
                
                try:
                    result = download_month_for_tf(
                        root, config, tf, month_start, month_end
                    )
                    
                    if result is None:
                        logger.info("task_completed_no_data", 
                                   symbol=config.symbol, tf=tf, year=y, month=m)
                    else:
                        logger.info("task_completed_success", 
                                   symbol=config.symbol, tf=tf, year=y, month=m,
                                   output_file=str(result))
                        
                except DownloadError as e:
                    failed_tasks += 1
                    logger.error("task_failed", 
                               symbol=config.symbol, tf=tf, year=y, month=m,
                               error=str(e))
                    # Continuar con siguiente tarea
                    
                except Exception as e:
                    failed_tasks += 1
                    logger.error("task_unexpected_error", 
                               symbol=config.symbol, tf=tf, year=y, month=m,
                               error=str(e), error_type=type(e).__name__)
                    # Continuar con siguiente tarea
                
                # Pausa entre tareas para evitar rate limiting
                time.sleep(0.1)

        # Resumen final
        success_rate = ((completed_tasks - failed_tasks) / completed_tasks * 100) if completed_tasks > 0 else 0
        
        logger.info("download_completed", 
                   total_tasks=completed_tasks,
                   failed_tasks=failed_tasks, 
                   success_rate=round(success_rate, 1))
        
        if failed_tasks > 0:
            logger.warning("download_finished_with_errors", failed_count=failed_tasks)
            sys.exit(1)
        else:
            logger.info("download_finished_successfully")

    except KeyboardInterrupt:
        logger.warning("download_interrupted_by_user")
        sys.exit(130)
    except Exception as e:
        logger.error("download_fatal_error", 
                    error=str(e), error_type=type(e).__name__)
        sys.exit(1)


if __name__ == "__main__":
    main()