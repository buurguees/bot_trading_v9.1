# train_env/vec_factory_chrono.py
from __future__ import annotations
from typing import Callable, Optional
import numpy as np
from stable_baselines3.common.vec_env import SubprocVecEnv
from base_env.io.historical_broker import ParquetHistoricalBroker
from base_env.config.models import EnvConfig, SymbolMeta, PipelineConfig, HierarchicalConfig, RiskConfig, FeesConfig
from base_env.base_env import BaseTradingEnv
from train_env.gym_wrapper import TradingGymWrapper

def make_vec_envs_chrono(n_envs: int, seed: int, data_cfg: dict, env_cfg: dict, logging_cfg: dict, models_cfg: dict, symbol_cfg: dict) -> SubprocVecEnv:
    # Soportamos tanto formato antiguo {"symbol": "BTCUSDT"} como nuevo {"name": "BTCUSDT"}
    symbol = symbol_cfg.get("name", symbol_cfg.get("symbol"))
    leverage_spec = symbol_cfg.get("leverage") if symbol_cfg.get("mode","").endswith("futures") else None

    def make_one(rank: int) -> Callable:
        def _thunk():
            np.random.seed(seed + rank)
            tfs = data_cfg["tfs"]; base_tf = tfs[0]
            broker = ParquetHistoricalBroker(
                data_root=data_cfg["root"], symbol=symbol, market=("futures" if leverage_spec else "spot"),
                tfs=tfs, base_tf=base_tf, stage=data_cfg["stage"], warmup_bars=env_cfg["warmup_bars"]
            )
            # slice por rank
            ts_list = broker._base_ts_list
            n = len(ts_list)
            start = (n * rank) // n_envs
            broker._i = start

            cfg = EnvConfig(
                mode=symbol_cfg["mode"], market=("futures" if leverage_spec else "spot"),
                symbol_meta=SymbolMeta(symbol=symbol, market=("futures" if leverage_spec else "spot"), enabled_tfs=tfs, filters={"minNotional":5.0,"lotStep":0.0001}),
                tfs=tfs, pipeline=PipelineConfig(strict_alignment=True),
                hierarchical=HierarchicalConfig(min_confidence=0.0, execute_tfs=[base_tf], confirm_tfs=[tfs[-1]]),
                risk=RiskConfig(), fees=FeesConfig()
            )

            base = BaseTradingEnv(
                cfg=cfg, broker=broker, oms=_MockOMS(),
                initial_cash=float(env_cfg.get("initial_balance", 1000.0)),
                target_cash=float(env_cfg.get("target_balance", 1_000_000.0)),
                models_root=models_cfg.get("root","models"),
            )
            strat_prov = f"{models_cfg['root']}/{symbol}/{symbol}_strategies_provisional.jsonl"
            return TradingGymWrapper(
                base_env=base,
                reward_yaml=env_cfg["reward_yaml"],
                tfs=tfs,
                leverage_spec=leverage_spec,                  # <— pasa rango desde YAML
                strategy_log_path=strat_prov
            )
        return _thunk
    return SubprocVecEnv([make_one(i) for i in range(n_envs)])

class _MockOMS:
    def open(self, side, qty, price_hint, sl, tp): return {"side": 1 if side=="LONG" else -1, "qty": float(qty), "price": float(price_hint), "fees": 0.0, "sl": sl, "tp": tp}
    def close(self, qty, price_hint): return {"qty": float(qty), "price": float(price_hint), "fees": 0.0}
