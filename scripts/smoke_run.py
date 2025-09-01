# scripts/smoke_run.py
# Descripción: Script simple para ejecutar el loop reset/step del entorno y mostrar resúmenes.
# Ubicación: scripts/smoke_run.py

from base_env.config.models import EnvConfig, SymbolMeta, PipelineConfig, HierarchicalConfig, RiskConfig, FeesConfig
from base_env.io.broker import InMemoryBroker
from base_env.base_env import BaseTradingEnv

class MockOMS:
    def open(self, side, qty, price_hint, sl, tp):
        return {"side": side, "qty": qty, "price": price_hint, "sl": sl, "tp": tp, "fees": 0.0}
    def close(self, qty, price_hint):
        return {"qty": qty, "price": price_hint, "fees": 0.0}

def make_series(ts_list, start_price=100.0, step=1.0):
    out = {}
    p = start_price
    for ts in ts_list:
        bar = {
            "ts": ts,
            "open": p,
            "high": p + 0.5,
            "low": p - 0.5,
            "close": p + 0.2,
            "volume": 10.0,
        }
        out[ts] = bar
        p += step
    return out

def main():
    ts_list = [1_000, 2_000, 3_000, 4_000, 5_000]
    series_by_tf = {
        "1m": make_series(ts_list, start_price=100.0, step=1.0),
        "5m": make_series(ts_list, start_price=100.0, step=1.0),
    }
    broker = InMemoryBroker(series_by_tf, base_tf="1m")

    cfg = EnvConfig(
        mode="train",
        market="spot",
        symbol_meta=SymbolMeta(symbol="BTCUSDT", market="spot", enabled_tfs=["1m","5m"]),
        tfs=["1m","5m"],
        pipeline=PipelineConfig(strict_alignment=True),
        hierarchical=HierarchicalConfig(min_confidence=0.0),
        risk=RiskConfig(),
        fees=FeesConfig(),
    )

    env = BaseTradingEnv(cfg=cfg, broker=broker, oms=MockOMS())
    obs = env.reset()
    print(f"reset() ts={obs['ts']} price_1m={obs['tfs']['1m']['close']}")

    steps = 5
    for i in range(steps):
        obs, reward, done, info = env.step()
        ts = obs["ts"]
        price = obs["tfs"]["1m"]["close"]
        print(f"step {i+1}/{steps} ts={ts} price={price:.2f} reward={reward:.6f} done={done} events={len(info.get('events', []))}")
        if done:
            break

if __name__ == "__main__":
    main()
