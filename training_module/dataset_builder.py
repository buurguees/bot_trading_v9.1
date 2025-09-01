from __future__ import annotations
from pathlib import Path
import pandas as pd

from base_env import BaseContext, DataBroker, build_mtf_view, FeatureConfig, IndicatorCalculator, SMCConfig, SMCDetector
from training_module.features.labels import triple_barrier_labels

def build_training_dataset(symbol: str = "BTCUSDT", exec_tf: str = "5m",
                           horizon: int = 48, tp_k_atr: float = 3.0, sl_k_atr: float = 1.0,
                           save_part: str = "train_dataset") -> Path:
    """
    Construye dataset entrenable causal:
    - Vista MTF (Dirección 1D/4H, Confirmación 1H/15m, Ejecución exec_tf)
    - Features técnicos (exec_tf)
    - SMC (exec_tf)
    - Labels triple-barrier (exec_tf)
    Guarda en features_dir/symbol=.../timeframe=.../{save_part}.parquet
    """
    ctx = BaseContext()
    db = DataBroker(ctx.ohlcv_dir)

    # Vista MTF (OHLCV)
    tfs = {"direction": ["1d","4h"], "confirmation": ["1h","15m"], "execution": [exec_tf]}
    mtf = build_mtf_view(symbol, ctx.ohlcv_dir, tfs)

    # Cálculo features técnicos sobre exec_tf (columnas exec)
    exec_cols = [c for c in mtf.columns if c.startswith(exec_tf+"_")]
    df_exec = mtf[["timestamp"] + exec_cols].rename(columns={f"{exec_tf}_open":"open", f"{exec_tf}_high":"high",
                                                             f"{exec_tf}_low":"low", f"{exec_tf}_close":"close",
                                                             f"{exec_tf}_volume":"volume"}).copy()

    fcfg = FeatureConfig.from_yaml("config/features.yaml")
    icalc = IndicatorCalculator(fcfg, mode="causal")
    feats = icalc.calculate_all(df_exec)

    # SMC en exec_tf
    scfg = SMCConfig.from_yaml("config/smc.yaml")
    sdet = SMCDetector(scfg)
    smc = sdet.detect_all(feats)

    # Labels
    labeled = triple_barrier_labels(smc, horizon=horizon, tp_k_atr=tp_k_atr, sl_k_atr=sl_k_atr)

    # Unir con contexto MTF (dir_/conf_) sin fuga (asof ya respeta causalidad)
    context_cols = [c for c in mtf.columns if c.startswith("dir_") or c.startswith("conf_")]
    final = labeled.merge(mtf[["timestamp"] + context_cols], on="timestamp", how="left")

    # Guardar
    out_dir = Path(ctx.features_dir) / f"symbol={symbol}" / f"timeframe={exec_tf}"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"{save_part}.parquet"
    final.to_parquet(out_file, index=False)
    return out_file
