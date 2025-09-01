from __future__ import annotations
import sys
from pathlib import Path
import typer
from loguru import logger
import yaml

APP = typer.Typer(help="Trading Bot v9.1 CLI")

def load_yaml(path: str | Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}

def setup_logging(settings: dict):
    logger.remove()
    level = settings.get("logging", {}).get("level", "INFO")
    jsonfmt = settings.get("logging", {}).get("json", True)
    path = settings.get("logging", {}).get("path", None)
    logger.add(sys.stdout, level=level, serialize=jsonfmt)
    if path:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        logger.add(path, level=level, serialize=jsonfmt, rotation="10 MB", retention="30 days")

@APP.command("bootstrap-data")
def bootstrap_data():
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
    settings = load_yaml("config/settings.yaml")
    setup_logging(settings)
    from base_env.resampler import Resampler
    data_cfg = load_yaml("config/data.yaml")
    r = Resampler(Path(settings["paths"]["ohlcv_dir"]))
    for sym in [s.replace("/","") for s in data_cfg["symbols"]]:
        r.resample_symbol(sym, from_tf="1m", to_tfs=["5m","15m","1h","4h","1d"])

@APP.command("validate-data")
def validate_data():
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

@APP.command("build-features")
def build_features(symbol: str = "BTCUSDT", exec_tf: str = "5m"):
    """Calcula features técnicos sobre exec_tf y guarda en feature store."""
    settings = load_yaml("config/settings.yaml")
    setup_logging(settings)
    from base_env.context import BaseContext
    from base_env.mtf_view import build_mtf_view
    from base_env.feature_engine import FeatureConfig, IndicatorCalculator

    ctx = BaseContext()
    tfs = {"direction": ["1d","4h"], "confirmation": ["1h","15m"], "execution": [exec_tf]}
    mtf = build_mtf_view(symbol, ctx.ohlcv_dir, tfs)
    exec_cols = [c for c in mtf.columns if c.startswith(exec_tf+"_")]
    df_exec = mtf[["timestamp"] + exec_cols].rename(columns={
        f"{exec_tf}_open":"open", f"{exec_tf}_high":"high", f"{exec_tf}_low":"low",
        f"{exec_tf}_close":"close", f"{exec_tf}_volume":"volume"
    }).copy()

    fcfg = FeatureConfig.from_yaml("config/features.yaml")
    icalc = IndicatorCalculator(fcfg, mode="causal")
    feats = icalc.calculate_all(df_exec)

    out_dir = Path(settings["paths"]["features_dir"]) / f"symbol={symbol}" / f"timeframe={exec_tf}"
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / "features.parquet"
    feats.to_parquet(out, index=False)
    logger.info({"component":"features","symbol":symbol,"tf":exec_tf,"rows":len(feats),"out":str(out)})

@APP.command("build-smc")
def build_smc(symbol: str = "BTCUSDT", exec_tf: str = "5m"):
    """Detecta SMC sobre exec_tf y guarda en feature store."""
    settings = load_yaml("config/settings.yaml")
    setup_logging(settings)
    from base_env.context import BaseContext
    from base_env.mtf_view import build_mtf_view
    from base_env.feature_engine import FeatureConfig, IndicatorCalculator
    from base_env.smc_service import SMCConfig, SMCDetector  # :contentReference[oaicite:5]{index=5}

    ctx = BaseContext()
    tfs = {"direction": ["1d","4h"], "confirmation": ["1h","15m"], "execution": [exec_tf]}
    mtf = build_mtf_view(symbol, ctx.ohlcv_dir, tfs)
    exec_cols = [c for c in mtf.columns if c.startswith(exec_tf+"_")]
    df_exec = mtf[["timestamp"] + exec_cols].rename(columns={
        f"{exec_tf}_open":"open", f"{exec_tf}_high":"high", f"{exec_tf}_low":"low",
        f"{exec_tf}_close":"close", f"{exec_tf}_volume":"volume"
    }).copy()

    fcfg = FeatureConfig.from_yaml("config/features.yaml")
    feats = IndicatorCalculator(fcfg, mode="causal").calculate_all(df_exec)

    scfg = SMCConfig.from_yaml("config/smc.yaml")
    smc = SMCDetector(scfg).detect_all(feats)

    out_dir = Path(settings["paths"]["features_dir"]) / f"symbol={symbol}" / f"timeframe={exec_tf}"
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / "smc.parquet"
    smc.to_parquet(out, index=False)
    logger.info({"component":"smc","symbol":symbol,"tf":exec_tf,"rows":len(smc),"out":str(out)})

@APP.command("build-dataset")
def build_dataset(symbol: str = "BTCUSDT", exec_tf: str = "5m",
                  horizon: int = 48, tp_k_atr: float = 3.0, sl_k_atr: float = 1.0):
    """Crea dataset entrenable (features + SMC + labels triple-barrier)."""
    settings = load_yaml("config/settings.yaml")
    setup_logging(settings)
    from training_module.dataset_builder import build_training_dataset
    out = build_training_dataset(symbol=symbol, exec_tf=exec_tf, horizon=horizon,
                                 tp_k_atr=tp_k_atr, sl_k_atr=sl_k_atr, save_part="train_dataset")
    logger.info({"component":"dataset","out":str(out)})

@APP.command("backtest-baseline")
def backtest_baseline(symbol: str = "BTCUSDT", tf: str = "5m"):
    settings = load_yaml("config/settings.yaml")
    setup_logging(settings)
    from backtest_module.engines.vectorized_engine import run_sma_crossover_backtest
    res = run_sma_crossover_backtest(symbol=symbol, timeframe=tf, settings=settings)
    logger.info({"component":"backtest","result":res})

@APP.command("paper")
def run_paper(symbol: str = "BTCUSDT", tf: str = "5m"):
    settings = load_yaml("config/settings.yaml")
    setup_logging(settings)
    from live_module.monitoring.performance_tracker import Heartbeat
    from backtest_module.engines.event_driven_engine import run_paper_loop
    run_paper_loop(symbol=symbol, timeframe=tf, settings=settings, heartbeat=Heartbeat())

if __name__ == "__main__":
    APP()
