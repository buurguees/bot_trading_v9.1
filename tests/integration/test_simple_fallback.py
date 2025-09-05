"""
Test simplificado para validar el fallback de SL/TP.
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from base_env.base_env import BaseTradingEnv


def test_default_sl_tp_calculation():
    """Test que verifica que se calculan SL/TP por defecto correctamente."""
    
    # Crear un entorno mock mínimo
    class MockConfig:
        def __init__(self):
            self.risk = MockRiskConfig()
    
    class MockRiskConfig:
        def __init__(self):
            self.common = MockCommonConfig()
    
    class MockCommonConfig:
        def __init__(self):
            self.default_levels = MockDefaultLevels()
    
    class MockDefaultLevels:
        def __init__(self):
            self.use_atr = False
            self.min_sl_pct = 1.0
            self.tp_r_multiple = 1.5
    
    # Crear entorno con config mock
    cfg = MockConfig()
    broker = None
    oms = None
    
    env = BaseTradingEnv.__new__(BaseTradingEnv)  # Crear sin __init__
    env.cfg = cfg
    
    # Test: Decision sin SL/TP debe aplicar fallback
    price = 50000.0
    side = 1  # Long
    atr = None  # Sin ATR
    
    # Aplicar fallback
    sl, tp = env._get_default_sl_tp(price, atr, side)
    
    # Verificar que se calcularon SL/TP
    assert sl is not None, f"SL no debería ser None, obtuvo: {sl}"
    assert tp is not None, f"TP no debería ser None, obtuvo: {tp}"
    assert sl > 0, f"SL debería ser positivo, obtuvo: {sl}"
    assert tp > 0, f"TP debería ser positivo, obtuvo: {tp}"
    
    # Para long: SL < price < TP
    assert sl < price < tp, f"Para long: SL({sl}) < price({price}) < TP({tp})"
    
    # Verificar que la distancia es razonable (1% por defecto)
    sl_dist = price - sl
    expected_sl_dist = price * 0.01  # 1%
    assert abs(sl_dist - expected_sl_dist) < price * 0.001, f"SL distance {sl_dist} no es ~1% de {price}"
    
    # Verificar que TP es 1.5x el riesgo
    tp_dist = tp - price
    expected_tp_dist = sl_dist * 1.5
    assert abs(tp_dist - expected_tp_dist) < price * 0.001, f"TP distance {tp_dist} no es 1.5x SL distance {sl_dist}"
    
    print(f"✅ Test LONG: price={price}, sl={sl:.2f}, tp={tp:.2f}")
    print(f"   SL distance: {sl_dist:.2f} (expected: {expected_sl_dist:.2f})")
    print(f"   TP distance: {tp_dist:.2f} (expected: {expected_tp_dist:.2f})")


def test_default_sl_tp_short():
    """Test que verifica SL/TP por defecto para posiciones short."""
    
    # Crear un entorno mock mínimo
    class MockConfig:
        def __init__(self):
            self.risk = MockRiskConfig()
    
    class MockRiskConfig:
        def __init__(self):
            self.common = MockCommonConfig()
    
    class MockCommonConfig:
        def __init__(self):
            self.default_levels = MockDefaultLevels()
    
    class MockDefaultLevels:
        def __init__(self):
            self.use_atr = False
            self.min_sl_pct = 1.0
            self.tp_r_multiple = 1.5
    
    # Crear entorno con config mock
    cfg = MockConfig()
    broker = None
    oms = None
    
    env = BaseTradingEnv.__new__(BaseTradingEnv)  # Crear sin __init__
    env.cfg = cfg
    
    price = 50000.0
    side = -1  # Short
    atr = None  # Sin ATR
    
    # Aplicar fallback
    sl, tp = env._get_default_sl_tp(price, atr, side)
    
    # Verificar que se calcularon SL/TP
    assert sl is not None, f"SL no debería ser None, obtuvo: {sl}"
    assert tp is not None, f"TP no debería ser None, obtuvo: {tp}"
    assert sl > 0, f"SL debería ser positivo, obtuvo: {sl}"
    assert tp > 0, f"TP debería ser positivo, obtuvo: {tp}"
    
    # Para short: TP < price < SL
    assert tp < price < sl, f"Para short: TP({tp}) < price({price}) < SL({sl})"
    
    # Verificar que la distancia es razonable (1% por defecto)
    sl_dist = sl - price
    expected_sl_dist = price * 0.01  # 1%
    assert abs(sl_dist - expected_sl_dist) < price * 0.001, f"SL distance {sl_dist} no es ~1% de {price}"
    
    # Verificar que TP es 1.5x el riesgo
    tp_dist = price - tp
    expected_tp_dist = sl_dist * 1.5
    assert abs(tp_dist - expected_tp_dist) < price * 0.001, f"TP distance {tp_dist} no es 1.5x SL distance {sl_dist}"
    
    print(f"✅ Test SHORT: price={price}, sl={sl:.2f}, tp={tp:.2f}")
    print(f"   SL distance: {sl_dist:.2f} (expected: {expected_sl_dist:.2f})")
    print(f"   TP distance: {tp_dist:.2f} (expected: {expected_tp_dist:.2f})")


if __name__ == "__main__":
    test_default_sl_tp_calculation()
    test_default_sl_tp_short()
    print("✅ Tests de fallback SL/TP pasaron correctamente")
