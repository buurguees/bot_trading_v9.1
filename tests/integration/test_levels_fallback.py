"""
Test para validar que el fallback de SL/TP funciona correctamente.
"""
import pytest
from base_env.base_env import BaseTradingEnv
from base_env.policy.gating import Decision
from base_env.config.models import EnvConfig


def test_default_sl_tp_fallback():
    """Test que verifica que se aplican SL/TP por defecto cuando vienen None."""
    # Crear un entorno mock
    cfg = EnvConfig(
        market="futures",
        mode="train_futures",
        symbol_meta=None,  # Se mockeará
        risk=None,  # Se mockeará
        fees=None,  # Se mockeará
        pipeline=None,  # Se mockeará
        hierarchical=None,  # Se mockeará
        tfs=["1h"],
        leverage=2.0
    )
    
    # Mock de broker y OMS
    class MockBroker:
        def get_price(self): return 50000.0
        def now_ts(self): return 1640995200000
        def next(self): pass
        def is_end_of_data(self): return False
        def reset_to_start(self): pass
    
    class MockOMS:
        def open(self, side, qty, price, sl, tp):
            return {"side": 1 if side == "LONG" else -1, "qty": qty, "price": price, "fees": 0.0}
        def close(self, qty, price):
            return {"side": -1, "qty": qty, "price": price, "fees": 0.0}
    
    broker = MockBroker()
    oms = MockOMS()
    
    # Crear entorno
    env = BaseTradingEnv(cfg, broker, oms)
    
    # Test: Decision sin SL/TP debe aplicar fallback
    price = 50000.0
    side = 1  # Long
    
    # Simular observación con features vacías (sin ATR)
    obs = {
        "ts": 1640995200000,
        "tfs": {"1h": {"close": price}},
        "features": {"1h": {}},  # Sin ATR
        "smc": {},
        "analysis": {},
        "position": {"side": 0, "qty": 0.0},
        "portfolio": {"equity_quote": 10000.0, "cash_quote": 10000.0},
        "mode": "train_futures"
    }
    
    # Aplicar fallback
    sl, tp = env._get_default_sl_tp(price, None, side)
    
    # Verificar que se calcularon SL/TP
    assert sl is not None
    assert tp is not None
    assert sl > 0
    assert tp > 0
    
    # Para long: SL < price < TP
    assert sl < price < tp
    
    # Verificar que la distancia es razonable (1% por defecto)
    sl_dist = price - sl
    expected_sl_dist = price * 0.01  # 1%
    assert abs(sl_dist - expected_sl_dist) < price * 0.001  # Tolerancia del 0.1%
    
    # Verificar que TP es 1.5x el riesgo
    tp_dist = tp - price
    expected_tp_dist = sl_dist * 1.5
    assert abs(tp_dist - expected_tp_dist) < price * 0.001


def test_default_sl_tp_short():
    """Test que verifica SL/TP por defecto para posiciones short."""
    # Similar setup pero para short
    cfg = EnvConfig(
        market="futures",
        mode="train_futures",
        symbol_meta=None,
        risk=None,
        fees=None,
        pipeline=None,
        hierarchical=None,
        tfs=["1h"],
        leverage=2.0
    )
    
    class MockBroker:
        def get_price(self): return 50000.0
        def now_ts(self): return 1640995200000
        def next(self): pass
        def is_end_of_data(self): return False
        def reset_to_start(self): pass
    
    class MockOMS:
        def open(self, side, qty, price, sl, tp):
            return {"side": 1 if side == "LONG" else -1, "qty": qty, "price": price, "fees": 0.0}
        def close(self, qty, price):
            return {"side": -1, "qty": qty, "price": price, "fees": 0.0}
    
    broker = MockBroker()
    oms = MockOMS()
    
    env = BaseTradingEnv(cfg, broker, oms)
    
    price = 50000.0
    side = -1  # Short
    
    # Aplicar fallback
    sl, tp = env._get_default_sl_tp(price, None, side)
    
    # Verificar que se calcularon SL/TP
    assert sl is not None
    assert tp is not None
    assert sl > 0
    assert tp > 0
    
    # Para short: TP < price < SL
    assert tp < price < sl
    
    # Verificar que la distancia es razonable (1% por defecto)
    sl_dist = sl - price
    expected_sl_dist = price * 0.01  # 1%
    assert abs(sl_dist - expected_sl_dist) < price * 0.001  # Tolerancia del 0.1%
    
    # Verificar que TP es 1.5x el riesgo
    tp_dist = price - tp
    expected_tp_dist = sl_dist * 1.5
    assert abs(tp_dist - expected_tp_dist) < price * 0.001


if __name__ == "__main__":
    test_default_sl_tp_fallback()
    test_default_sl_tp_short()
    print("✅ Tests de fallback SL/TP pasaron correctamente")
