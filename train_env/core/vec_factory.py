# train_env/vec_factory.py
from __future__ import annotations

import os
import math
import time
import random
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv

# Dependencias internas del proyecto (con tolerancia a entorno)
# BaseTradingEnv y TradingGymWrapper son obligatorios para ambas variantes
from base_env.base_env import BaseTradingEnv
from .gym_wrapper import TradingGymWrapper

# Componentes “cronológicos”; si no existen, se hará fallback al modo simple
try:
    from base_env.io.historical_broker import ParquetHistoricalBroker
except Exception:
    ParquetHistoricalBroker = None  # type: ignore

# Un OMS de prueba ligero; si no existe, se crea uno mínimo
class _MockOMS:
    def __init__(self, now_ts_fn: Callable[[], int]):
        self._now_ts_fn = now_ts_fn

    def now_ts(self) -> int:
        return self._now_ts_fn()


# -------------------------
# Utilidades de configuración
# -------------------------

def _safe_get(d: Dict[str, Any], path: str, default: Any = None) -> Any:
    """Acceso seguro tipo 'a.b.c' en dicts anidados."""
    cur = d
    for key in path.split("."):
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur


def _detect_market(symbol_cfg: Dict[str, Any]) -> str:
    """spot / futures en base a la config del símbolo."""
    mode = (symbol_cfg or {}).get("mode", "spot").lower()
    return "futures" if "future" in mode else "spot"


def _build_leverage_spec(symbol_cfg: Dict[str, Any], models_cfg: Dict[str, Any]) -> Optional[Dict[str, float]]:
    """Construye leverage_spec si el símbolo es futures."""
    market = _detect_market(symbol_cfg)
    if market != "futures":
        return None
    lev_cfg = _safe_get(models_cfg, "leverage", {}) or {}
    mn = float(lev_cfg.get("min", 1.0))
    mx = float(lev_cfg.get("max", 20.0))
    st = float(lev_cfg.get("step", 1.0))
    if mx < mn or st <= 0:
        # sane defaults
        mn, mx, st = 1.0, 20.0, 1.0
    return {"min": mn, "max": mx, "step": st}


def _compose_strategy_log_path(models_root: Union[str, Path], symbol: str) -> str:
    root = Path(models_root)
    sym_dir = root / symbol
    sym_dir.mkdir(parents=True, exist_ok=True)
    return str(sym_dir / f"{symbol}_strategies_provisional.jsonl")


def _ts_now_ms() -> int:
    return int(time.time() * 1000)


# -------------------------
# Validaciones para modo cronológico
# -------------------------

def _read_latest_parquet_dir(data_root: Union[str, Path], symbol: str, market: str, base_tf: str) -> Optional[Path]:
    """
    Espera estructura: {data_root}/{symbol}/{market}/aligned/{base_tf}/
    Devuelve el parquet más reciente (por mtime). Si no existe, None.
    """
    base = Path(data_root) / symbol / market / "aligned" / base_tf
    if not base.exists() or not base.is_dir():
        return None
    files = sorted((p for p in base.glob("*.parquet")), key=lambda p: p.stat().st_mtime, reverse=True)
    return files[0] if files else None


def _validate_min_bars_needed(parquet_file: Path, warmup_bars: int, min_extra: int, verbosity: int = 0) -> Tuple[int, int]:
    """
    Lee parquet para asegurar suficientes barras: warmup_bars + min_extra.
    Devuelve (ts_min, ts_max) en ms.
    """
    df = pd.read_parquet(parquet_file, columns=["ts"])
    if df.empty:
        raise RuntimeError(f"[CHRONO] Parquet vacío: {parquet_file}")
    need = warmup_bars + max(0, min_extra)
    if len(df) < need:
        raise RuntimeError(
            f"[CHRONO] Barras insuficientes: {len(df)} < {need} (warmup={warmup_bars}, extra={min_extra})"
        )
    ts_min, ts_max = int(df["ts"].iloc[0]), int(df["ts"].iloc[-1])
    if verbosity >= 1:
        print(f"[CHRONO] parquet={parquet_file.name} barras={len(df)} ts_min={ts_min} ts_max={ts_max}")
    return ts_min, ts_max


def _slice_ts_range_by_months(ts_min: int, ts_max: int, months_back: Optional[int], verbosity: int = 0) -> Tuple[int, int]:
    """
    Acota [ts_min, ts_max] por months_back (si se pasa).
    """
    if not months_back or months_back <= 0:
        return ts_min, ts_max
    # Aprox 30 días por mes
    ms_month = 30 * 24 * 3600 * 1000
    target_from = ts_max - months_back * ms_month
    ts_from = max(ts_min, target_from)
    if verbosity >= 1:
        print(f"[CHRONO] months_back={months_back} → ts_from={ts_from} ts_to={ts_max}")
    return ts_from, ts_max


# -------------------------
# Factoría de workers
# -------------------------

def _make_worker(
    rank: int,
    seed: int,
    symbol: str,
    tfs: List[str],
    reward_yaml: str,
    data_cfg: Dict[str, Any],
    env_cfg: Dict[str, Any],
    symbol_cfg: Dict[str, Any],
    models_cfg: Dict[str, Any],
    logging_cfg: Dict[str, Any],
    models_root: Union[str, Path],
    chrono: bool,
) -> Callable[[], TradingGymWrapper]:
    """
    Devuelve una función que crea un entorno env + wrapper para Subproc/DummyVecEnv.
    """

    def _init() -> TradingGymWrapper:
        # Semilla por worker
        base_seed = int(seed) + int(rank)
        np.random.seed(base_seed)
        random.seed(base_seed)

        # Verbosidad
        verbosity = int(_safe_get(logging_cfg, "verbosity", 0) or 0)

        # Mercado y leverage
        market = _detect_market(symbol_cfg)
        leverage_spec = _build_leverage_spec(symbol_cfg, models_cfg)

        # Gym wrapper config
        strategy_log_path = _compose_strategy_log_path(models_root, symbol)

        # --- Construcción del entorno base ---
        if chrono and ParquetHistoricalBroker is not None:
            # Config cronológica
            data_root = _safe_get(data_cfg, "root", "data")
            base_tf = _safe_get(data_cfg, "base_tf", tfs[0] if tfs else "1m")
            warmup_bars = int(_safe_get(env_cfg, "warmup_bars", 1000) or 1000)
            min_extra = int(_safe_get(env_cfg, "min_extra_bars", 100) or 100)
            months_back = _safe_get(data_cfg, "months_back", None)

            latest = _read_latest_parquet_dir(data_root, symbol, market, base_tf)
            if latest is None:
                raise RuntimeError(
                    f"[CHRONO] No se encontró parquet en {data_root}/{symbol}/{market}/aligned/{base_tf}"
                )

            ts_min, ts_max = _validate_min_bars_needed(latest, warmup_bars, min_extra, verbosity)
            ts_from, ts_to = _slice_ts_range_by_months(ts_min, ts_max, months_back, verbosity)

            if verbosity >= 1:
                print(f"[CHRONO] Creando broker con ventana ts=[{ts_from}, {ts_to}] base_tf={base_tf}")

            broker = ParquetHistoricalBroker(
                data_root=data_root,
                symbol=symbol,
                market=market,
                base_tf=base_tf,
                ts_from=ts_from,
                ts_to=ts_to,
                seed=base_seed,
                verbosity=verbosity,
            )
            oms = _MockOMS(now_ts_fn=broker.now_ts)

            base_env = BaseTradingEnv(
                symbol=symbol,
                market=market,
                broker=broker,
                oms=oms,
                tfs=tfs,
                config=env_cfg,
                verbosity=verbosity,
            )
        else:
            # Fallback “simple”: BaseTradingEnv debe poder construir sin broker explícito
            if verbosity >= 1:
                note = "simple" if ParquetHistoricalBroker is None else "simple (forzado por chrono=False)"
                print(f"[FACTORY] Modo {note} para symbol={symbol}")
            base_env = BaseTradingEnv(
                symbol=symbol,
                market=_detect_market(symbol_cfg),
                broker=None,
                oms=_MockOMS(now_ts_fn=_ts_now_ms),
                tfs=tfs,
                config=env_cfg,
                verbosity=verbosity,
            )

        # Wrap gym
        env = TradingGymWrapper(
            base_env=base_env,
            reward_yaml=reward_yaml,
            tfs=tfs,
            leverage_spec=leverage_spec,
            strategy_log_path=strategy_log_path,
            verbosity=verbosity,
        )

        # Semilla gym/env
        try:
            env.reset(seed=base_seed)
        except Exception:
            pass

        return env

    return _init


# -------------------------
# API única (reemplaza a ambas fábricas)
# -------------------------

def make_vec_env(
    symbol: str,
    *,
    tfs: List[str],
    reward_yaml: str,
    data_cfg: Dict[str, Any],
    env_cfg: Dict[str, Any],
    symbol_cfg: Dict[str, Any],
    models_cfg: Dict[str, Any],
    logging_cfg: Dict[str, Any],
    models_root: Union[str, Path] = "models",
    n_envs: int = 1,
    seed: int = 42,
    start_method: Optional[str] = "spawn",
    chrono: bool = True,
    use_subproc: bool = True,
):
    """
    Crea un vectorized env unificado que cubre:
      - ✅ Modo cronológico (broker parquet + ventana por months_back)
      - ✅ Modo simple (sin broker explícito)
      - ✅ Detección spot/futures y leverage MultiDiscrete
      - ✅ Logs de estrategias por símbolo
      - ✅ Validaciones de datos en cronológico

    Args:
        symbol: Símbolo (p.ej. "BTCUSDT").
        tfs: Lista de timeframes para observación (p.ej. ["1m", "5m"]).
        reward_yaml: Ruta YAML del shaping de reward.
        data_cfg: Dict con {'root', 'base_tf', 'months_back', ...} para cronológico.
        env_cfg: Dict del entorno (incluye 'warmup_bars', 'min_extra_bars', ...).
        symbol_cfg: Dict de configuración del símbolo (de tus YAMLs).
        models_cfg: Dict (p.ej. leverage, vecnorm, etc.).
        logging_cfg: Dict (verbosity, logs_root, ...).
        models_root: Carpeta base donde guardar modelos/estrategias.
        n_envs: Número de workers.
        seed: Semilla base.
        start_method: 'spawn' recomendado en Windows y con IO pesado.
        chrono: True para entrenamiento “como en real”; False para modo simple.
        use_subproc: True → SubprocVecEnv; False → DummyVecEnv (depurar).

    Returns:
        VecEnv (SubprocVecEnv o DummyVecEnv).
    """
    assert n_envs >= 1, "n_envs debe ser >= 1"
    if chrono and ParquetHistoricalBroker is None:
        # Aviso si se pidió cronológico pero el broker no está disponible
        v = int(_safe_get(logging_cfg, "verbosity", 0) or 0)
        if v >= 1:
            print("[WARN] ParquetHistoricalBroker no disponible: forzando modo simple")
        chrono = False

    workers = [
        _make_worker(
            rank=i,
            seed=seed,
            symbol=symbol,
            tfs=tfs,
            reward_yaml=reward_yaml,
            data_cfg=data_cfg,
            env_cfg=env_cfg,
            symbol_cfg=symbol_cfg,
            models_cfg=models_cfg,
            logging_cfg=logging_cfg,
            models_root=models_root,
            chrono=chrono,
        )
        for i in range(n_envs)
    ]

    if n_envs == 1 or not use_subproc:
        return DummyVecEnv(workers)

    return SubprocVecEnv(workers, start_method=start_method or "spawn")


# -------------------------
# Función de compatibilidad para código existente
# -------------------------

def make_vec_envs_chrono(
    n_envs: int,
    seed: int,
    data_cfg: Dict[str, Any],
    env_cfg: Dict[str, Any],
    logging_cfg: Dict[str, Any],
    models_cfg: Dict[str, Any],
    symbol_cfg: Dict[str, Any],
    runs_log_cfg: Optional[Dict[str, Any]] = None,
    symbol: Optional[str] = None,
    tfs: Optional[List[str]] = None,
    reward_yaml: Optional[str] = None,
    models_root: Union[str, Path] = "models",
    start_method: Optional[str] = "spawn",
    use_subproc: bool = True,
):
    """
    Función de compatibilidad que mapea la API antigua a la nueva API unificada.
    
    Args:
        n_envs: Número de entornos.
        seed: Semilla base.
        data_cfg: Configuración de datos.
        env_cfg: Configuración del entorno.
        logging_cfg: Configuración de logging.
        models_cfg: Configuración de modelos.
        symbol_cfg: Configuración del símbolo.
        runs_log_cfg: Configuración de logs de runs (opcional, no usado).
        symbol: Símbolo de trading (extraído de symbol_cfg si no se proporciona).
        tfs: Lista de timeframes (extraída de env_cfg si no se proporciona).
        reward_yaml: Ruta del YAML de rewards (extraída de models_cfg si no se proporciona).
        models_root: Directorio raíz de modelos.
        start_method: Método de inicio para SubprocVecEnv.
        use_subproc: Si usar SubprocVecEnv o DummyVecEnv.
    
    Returns:
        VecEnv (SubprocVecEnv o DummyVecEnv) en modo cronológico.
    """
    # Extraer parámetros de las configuraciones si no se proporcionan directamente
    if symbol is None:
        symbol = symbol_cfg.get("symbol", "BTCUSDT")
    
    if tfs is None:
        # Intentar extraer tfs de env_cfg, si no existe usar defaults
        tfs = env_cfg.get("tfs", ["1m", "5m"])
        # Si tfs es un string, convertirlo a lista
        if isinstance(tfs, str):
            tfs = [tfs]
    
    if reward_yaml is None:
        # Intentar extraer reward_yaml de models_cfg o usar default
        reward_yaml = models_cfg.get("reward_yaml", "config/reward.yaml")
    
    # Usar la nueva API unificada con chrono=True
    return make_vec_env(
        symbol=symbol,
        tfs=tfs,
        reward_yaml=reward_yaml,
        data_cfg=data_cfg,
        env_cfg=env_cfg,
        symbol_cfg=symbol_cfg,
        models_cfg=models_cfg,
        logging_cfg=logging_cfg,
        models_root=models_root,
        n_envs=n_envs,
        seed=seed,
        start_method=start_method,
        chrono=True,  # Siempre modo cronológico para compatibilidad
        use_subproc=use_subproc,
    )
