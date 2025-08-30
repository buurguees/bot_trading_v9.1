from __future__ import annotations
import json, os, sys, time
from pathlib import Path
import typing as t

import typer
from loguru import logger

# --- Config loading ---
import yaml

APP = typer.Typer(help="Trading Bot v9.1 CLI")

def load_yaml(path: str | Path) -> dict:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Config not found: {p}")
    with p.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}

def setup_logging(settings: dict):
    logger.remove()
    level = settings.get("logging", {}).get("level", "INFO")
    jsonfmt = settings.get("logging", {}).get("json", True)
    path = settings.get("logging", {}).get("path", None)
    if path:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
    logger.add(sys.stdout, level=level, serialize=jsonfmt)
    if path:
        logger.add(path, level=level, serialize=jsonfmt, rotation="10 MB", retention="30 days")

# --- Commands ---

@APP.command("bootstrap-data")
def bootstrap_data():
    """Descarga históricos 1m y guarda Parquet particionado por mes."""
    settings = load_yaml("config/settings.yaml")
    setup_logging(settings)
    from data_module.collectors.bitget_collector import fetch_ohlcv
    data_cfg = load_yaml("config/data.yaml")
    root = Path(settings["paths"]["ohlcv_dir"])
    root.mkdir(parents=True, exist_ok=True)
    for sym in data_cfg["symbols"]:
        fetch_ohlcv(sym, "1m", data_cfg["years_back"], root)

@APP.command("resample")
def resample():
    """Resamplea de 1m a (5m, 15m, 1h, 4h, 1d)."""
    settings = load_yaml("config/settings.yaml")
    setup_logging(settings)
    from data_module.preprocessors.resample_ohlcv import resample_dir
    data_cfg = load_yaml("config/data.yaml")
    for sym in [s.replace("/","") for s in data_cfg["symbols"]]:
        for tf in ["5m","15m","1h","4h","1d"]:
            resample_dir(sym, "1m", tf)

@APP.command("validate-data")
def validate_data():
    """Valida calidad de OHLCV (orden, NaNs, duplicados, negativos)."""
    settings = load_yaml("config/settings.yaml")
    setup_logging(settings)
    from data_module.preprocessors.data_validator import validate_ohlcv
    symcfg = load_yaml("config/symbols.yaml")
    for s in symcfg["symbols"]:
        sid = s["symbol_id"]
        for tf in s["timeframes"]:
            d = f"{settings['paths']['ohlcv_dir']}/symbol={sid}/timeframe={tf}"
            if not Path(d).exists():
                continue
            rep = validate_ohlcv(d)
            logger.info({"component":"validator","symbol":sid,"tf":tf,"report":rep})

@APP.command("build-mtf")
def build_mtf(symbol: str = "BTCUSDT", exec_tf: str = "5m"):
    """Ensambla vista MTF causal para ejecución (features+smc → features_dir)."""
    settings = load_yaml("config/settings.yaml")
    setup_logging(settings)
    from core.market.data_view import build_mtf_view
    from data_module.preprocessors.indicator_calculator import IndicatorCalculator, FeatureConfig
    import pandas as pd

    paths = settings["paths"]
    base = Path(paths["ohlcv_dir"])
    # Carga mínimamente 1m/5m y tfs contexto; generar una vista MTF pequeña de ejemplo
    tfs = {"direction":["1d","4h"], "confirmation":["1h","15m"], "execution":[exec_tf]}
    # Reutilizamos build_mtf_view, que lee Parquet por convención
    out = build_mtf_view(symbol, base, tfs)
    out_dir = Path(paths["features_dir"]) / f"symbol={symbol}" / f"timeframe={exec_tf}"
    out_dir.mkdir(parents=True, exist_ok=True)
    out.to_parquet(out_dir / "mtf_sample.parquet", index=False)
    logger.info({"component":"build-mtf","symbol":symbol,"exec_tf":exec_tf,"rows":len(out)})

@APP.command("backtest-baseline")
def backtest_baseline(symbol: str = "BTCUSDT", tf: str = "5m"):
    """Backtest simple SMA crossover con OMS sim + risk manager."""
    settings = load_yaml("config/settings.yaml")
    setup_logging(settings)
    from backtest_module.engines.vectorized_engine import run_sma_crossover_backtest
    res = run_sma_crossover_backtest(symbol=symbol, timeframe=tf, settings=settings)
    logger.info({"component":"backtest","result":res})

@APP.command("paper")
def run_paper(symbol: str = "BTCUSDT", tf: str = "5m"):
    """Paper trading: usa WS simulado (sin exchange) y OMS sim con latencia/slippage."""
    settings = load_yaml("config/settings.yaml")
    setup_logging(settings)
    from live_module.monitoring.performance_tracker import Heartbeat
    hb = Heartbeat()
    # Simulación mínima: leer últimas barras y ejecutar baseline
    from backtest_module.engines.event_driven_engine import run_paper_loop
    run_paper_loop(symbol=symbol, timeframe=tf, settings=settings, heartbeat=hb)

if __name__ == "__main__":
    APP()
