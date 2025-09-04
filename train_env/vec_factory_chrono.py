# train_env/vec_factory_chrono.py
from __future__ import annotations
from typing import Callable, Optional
import numpy as np
from stable_baselines3.common.vec_env import SubprocVecEnv
from base_env.io.historical_broker import ParquetHistoricalBroker
from base_env.config.models import EnvConfig, SymbolMeta, PipelineConfig, HierarchicalConfig, RiskConfig, FeesConfig
from base_env.base_env import BaseTradingEnv
from train_env.gym_wrapper import TradingGymWrapper

def make_vec_envs_chrono(n_envs: int, seed: int, data_cfg: dict, env_cfg: dict, logging_cfg: dict, models_cfg: dict, symbol_cfg: dict, runs_log_cfg: dict = None) -> SubprocVecEnv:
    # Soportamos tanto formato antiguo {"symbol": "BTCUSDT"} como nuevo {"name": "BTCUSDT"}
    symbol = symbol_cfg.get("name", symbol_cfg.get("symbol"))
    leverage_spec = symbol_cfg.get("leverage") if symbol_cfg.get("mode","").endswith("futures") else None

    def make_one(rank: int) -> Callable:
        def _thunk():
            np.random.seed(seed + rank)
            tfs = data_cfg["tfs"]; base_tf = tfs[0]
            # Usar datos del mercado correcto (spot o futures)
            market_for_data = symbol_cfg.get("market", "spot")
            # Calcular rango from/to basado en months_back si existe (cronológico)
            ts_from = None; ts_to = None
            months_back = int(data_cfg.get("months_back", 0) or 0)
            if months_back > 0:
                # aproximación: months_back en ms (30 días * months_back)
                approx_ms = months_back * 30 * 24 * 60 * 60 * 1000
                # Usar None aquí, Parquet loader ya recorta por ts_from/ts_to
                # aquí dejamos ts_from=None y ts_to=None para simplificar (puede mejorarse con lectura de último ts)
                pass
            broker = ParquetHistoricalBroker(
                data_root=data_cfg["root"], symbol=symbol, market=market_for_data,
                tfs=tfs, base_tf=base_tf, stage=data_cfg["stage"], warmup_bars=env_cfg["warmup_bars"],
                ts_from=ts_from, ts_to=ts_to
            )
            # Episodios cronológicos: todos comienzan al inicio del histórico (sin slicing por rank)
            broker.reset_to_start()
            
            # ← NUEVO: Validar que hay suficientes datos para warmup
            total_bars = len(broker._base_ts_list)
            warmup_bars = env_cfg["warmup_bars"]
            if total_bars < warmup_bars + 100:  # Mínimo 100 barras después de warmup
                raise RuntimeError(f"Insuficientes datos: {total_bars} barras < {warmup_bars + 100} requeridas")
            
            print(f"[CHRONO] Worker {rank}: {total_bars} barras disponibles, warmup: {warmup_bars}")

            # Determinar market y leverage
            market = ("futures" if leverage_spec else "spot")
            env_leverage = float(leverage_spec.get("default", 1.0)) if leverage_spec else 1.0

            # ← NUEVO: Cargar configuración real de símbolos desde symbols.yaml
            from base_env.config.config_loader import config_loader
            symbols = config_loader.load_symbols()
            symbol_meta = next((s for s in symbols if s.symbol == symbol and s.market == market), None)
            if symbol_meta is None:
                # Fallback si no se encuentra
                symbol_meta = SymbolMeta(symbol=symbol, market=market, enabled_tfs=tfs, filters={"minNotional":5.0,"lotStep":0.0001,"tickSize":0.1}, allow_shorts=bool(symbol_cfg.get("allow_shorts", True)))
            
            # ← NUEVO: Cargar configuración jerárquica real desde hierarchical.yaml
            hier_yaml = config_loader._load_yaml("hierarchical.yaml")
            hier_cfg = HierarchicalConfig(**hier_yaml)
            
            cfg = EnvConfig(
                mode=symbol_cfg["mode"], market=market, leverage=env_leverage,
                symbol_meta=symbol_meta,
                tfs=tfs, pipeline=PipelineConfig(strict_alignment=True),
                hierarchical=hier_cfg,   # ← usa YAML real
                risk=RiskConfig(), fees=FeesConfig(),
                verbosity=logging_cfg.get("train_verbosity", "low")  # ← NUEVO: Pasar verbosity
            )

            base = BaseTradingEnv(
                cfg=cfg, broker=broker, oms=_MockOMS(broker=broker),
                initial_cash=float(env_cfg.get("initial_balance", 1000.0)),
                target_cash=float(env_cfg.get("target_balance", 1_000_000.0)),
                models_root=models_cfg.get("root","models"),
                antifreeze_enabled=env_cfg.get("antifreeze", {}).get("enabled", False),
                runs_log_cfg=runs_log_cfg,
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
    # ← NUEVO: Configurar SubprocVecEnv con método de inicio más robusto
    return SubprocVecEnv([make_one(i) for i in range(n_envs)], start_method='spawn')

class _MockOMS:
    def __init__(self, broker=None):
        self.broker = broker
    
    def open(self, side, qty, price_hint, sl, tp):
        ts = self.broker.now_ts() if self.broker else 0
        return {
            "side": 1 if side in (1, "LONG") else -1,
            "qty": float(qty),
            "price": float(price_hint),
            "fees": 0.0,
            "sl": float(sl) if sl is not None else None,
            "tp": float(tp) if tp is not None else None,
            "ts": int(ts),
        }

    def close(self, qty, price_hint):
        ts = self.broker.now_ts() if self.broker else 0
        return {
            "qty": float(qty),
            "price": float(price_hint),
            "fees": 0.0,
            "ts": int(ts),
        }
