# train_env/dataset.py
# Crea brokers históricos por ventanas aleatorias (episodios) sobre 5 años.

from __future__ import annotations
from typing import List, Optional
from pathlib import Path
import random
from base_env.io.historical_broker import ParquetHistoricalBroker
from base_env.config.models import EnvConfig, SymbolMeta, PipelineConfig, HierarchicalConfig, RiskConfig, FeesConfig
from base_env.base_env import BaseTradingEnv

def make_base_env(
    data_root: str | Path,
    symbol: str,
    market: str,
    tfs: List[str],
    base_tf: str,
    stage: str,
    episode_bars: int,
    warmup_bars: int,
) -> BaseTradingEnv:
    # Selecciona una ventana aleatoria por ts usando limit (cargando cola de N barras)
    broker = ParquetHistoricalBroker(
        data_root=data_root,
        symbol=symbol,
        market=market,
        tfs=tfs,
        base_tf=base_tf,
        stage=stage,
        warmup_bars=warmup_bars + episode_bars + 200,  # margen
    )
    # No recortamos el broker; usamos un episodio de longitud fija en wrapper vectorial (n_steps)
    cfg = EnvConfig(
        mode="backtest",
        market=market,
        symbol_meta=SymbolMeta(symbol=symbol, market=market, enabled_tfs=tfs, filters={"minNotional":5.0,"lotStep":0.0001}),
        tfs=tfs,
        pipeline=PipelineConfig(strict_alignment=True),
        hierarchical=HierarchicalConfig(min_confidence=0.0, execute_tfs=[base_tf], confirm_tfs=[tfs[-1]]),
        risk=RiskConfig(),
        fees=FeesConfig(),
    )
    # Aleatoriza el cursor inicial dentro del rango (después del warmup)
    max_i = len(broker._base_ts_list) - (episode_bars + 1)
    if max_i > warmup_bars:
        start_i = random.randint(warmup_bars, max_i)
        broker._i = start_i  # mover cursor de inicio
    return BaseTradingEnv(cfg=cfg, broker=broker, oms=_MockOMS())

class _MockOMS:
    def open(self, side, qty, price_hint, sl, tp): return {"side":1 if side=="LONG" else -1,"qty":qty,"price":price_hint,"fees":0.0,"sl":sl,"tp":tp}
    def close(self, qty, price_hint): return {"qty":qty,"price":price_hint,"fees":0.0}
