# tests/test_base_env_smoke.py
# Descripción: Smoke test del entorno base: construye un broker en memoria, instancia el env, hace reset y 3 steps.
# Ubicación: tests/test_base_env_smoke.py

import pytest

from base_env.config.models import EnvConfig, SymbolMeta, PipelineConfig, HierarchicalConfig, RiskConfig, FeesConfig
from base_env.io.broker import InMemoryBroker
from base_env.base_env import BaseTradingEnv

class MockOMS:
    def open(self, side, qty, price_hint, sl, tp):
        # Simula un fill “instantáneo” sin slippage (el slippage se gestiona fuera del core en modos reales)
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

def test_base_env_smoke():
    # Datos de juguete (3 barras)
    ts_list = [1_000, 2_000, 3_000]
    series_by_tf = {
        "1m": make_series(ts_list, start_price=100.0, step=1.0),
        "5m": make_series(ts_list, start_price=100.0, step=1.0),
    }
    broker = InMemoryBroker(series_by_tf, base_tf="1m")

    # Config mínima: usa solo TFs que realmente tenemos en el broker para evitar fallos de alineación
    cfg = EnvConfig(
        mode="train",
        market="spot",
        symbol_meta=SymbolMeta(symbol="BTCUSDT", market="spot", enabled_tfs=["1m","5m"]),
        tfs=["1m","5m"],
        pipeline=PipelineConfig(strict_alignment=True),
        hierarchical=HierarchicalConfig(min_confidence=0.0),  # sin gating para el smoke
        risk=RiskConfig(),
        fees=FeesConfig(),
    )

    env = BaseTradingEnv(cfg=cfg, broker=broker, oms=MockOMS())

    # Reset
    obs0 = env.reset()
    assert "tfs" in obs0 and "features" in obs0 and "analysis" in obs0 and "position" in obs0 and "portfolio" in obs0

    # Tres pasos
    for _ in range(3):
        obs, reward, done, info = env.step()
        assert isinstance(obs, dict)
        assert "tfs" in obs and "analysis" in obs
        assert "events" in info
        assert isinstance(reward, float)
    # No debería crashear; done puede ser True en la última barra según tu implementación futura
