# scripts/backtest_smoke_from_aligned.py
# Descripción: Recorre datos en data/{SYMBOL}/{market}/aligned usando el broker histórico,
#              construye el env y ejecuta unos pasos.
# Uso:
#   python -m scripts.backtest_smoke_from_aligned --symbol BTCUSDT --market spot --tfs 1m,5m --base-tf 1m --steps 200

import argparse
from base_env.base_env import BaseTradingEnv
from base_env.config.models import EnvConfig, SymbolMeta, PipelineConfig, HierarchicalConfig, RiskConfig, FeesConfig
from base_env.io.historical_broker import ParquetHistoricalBroker

class MockOMS:
    def open(self, side, qty, price_hint, sl, tp): return {"side":side,"qty":qty,"price":price_hint,"fees":0.0}
    def close(self, qty, price_hint): return {"qty":qty,"price":price_hint,"fees":0.0}

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=str, default="data")
    p.add_argument("--symbol", type=str, required=True)
    p.add_argument("--market", type=str, choices=["spot","futures"], required=True)
    p.add_argument("--tfs", type=str, default="1m,5m")
    p.add_argument("--base-tf", type=str, default="1m")
    p.add_argument("--steps", type=int, default=200)
    return p.parse_args()

def main():
    args = parse_args()
    tfs = [t.strip() for t in args.tfs.split(",") if t.strip()]

    broker = ParquetHistoricalBroker(
        data_root=args.root,
        symbol=args.symbol.upper(),
        market=args.market.lower(),
        tfs=tfs,
        base_tf=args.base_tf,
        stage="aligned",
        warmup_bars=5000,
    )

    cfg = EnvConfig(
        mode="backtest",
        market=args.market.lower(),
        symbol_meta=SymbolMeta(symbol=args.symbol.upper(), market=args.market.lower(), enabled_tfs=tfs),
        tfs=tfs,
        pipeline=PipelineConfig(strict_alignment=True),
        hierarchical=HierarchicalConfig(min_confidence=0.0),
        risk=RiskConfig(),
        fees=FeesConfig(),
    )

    env = BaseTradingEnv(cfg=cfg, broker=broker, oms=MockOMS())
    obs = env.reset()
    print(f"reset: ts={obs['ts']} base_close={obs['tfs'][tfs[0]]['close']}")

    for i in range(args.steps):
        obs, reward, done, info = env.step()
        base_close = obs["tfs"][tfs[0]]["close"]
        print(f"step {i+1}/{args.steps} ts={obs['ts']} close={base_close:.4f} reward={reward:.6f} ev={len(info.get('events', []))}")
        if done:
            break

if __name__ == "__main__":
    main()
