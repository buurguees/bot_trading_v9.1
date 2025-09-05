"""
Test para validar que se registran fills correctamente en aperturas/cierres.
"""
import pytest
from base_env.base_env import BaseTradingEnv
from base_env.policy.gating import Decision
from base_env.config.models import EnvConfig


def test_forced_open_fill_registration():
    """Test que verifica que se registra un fill al forzar una apertura."""
    
    # Configuración
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
    
    # Mock de broker
    class MockBroker:
        def __init__(self):
            self.price = 50000.0
            self.step_count = 0
        
        def get_price(self): 
            return self.price
        
        def now_ts(self): 
            return 1640995200000 + self.step_count * 3600000
        
        def next(self): 
            self.step_count += 1
            self.price += 100  # Precio sube gradualmente
        
        def is_end_of_data(self): 
            return self.step_count > 10  # Terminar después de 10 steps
        
        def reset_to_start(self): 
            self.step_count = 0
            self.price = 50000.0
    
    # Mock de OMS que registra fills
    class MockOMS:
        def __init__(self):
            self.fills = []
        
        def open(self, side, qty, price, sl, tp):
            fill = {
                "side": 1 if side == "LONG" else -1,
                "qty": qty,
                "price": price,
                "fees": qty * price * 0.0005,  # 5bps
                "sl": sl,
                "tp": tp
            }
            self.fills.append(("OPEN", fill))
            return fill
        
        def close(self, qty, price):
            fill = {
                "side": -1,
                "qty": qty,
                "price": price,
                "fees": qty * price * 0.0005  # 5bps
            }
            self.fills.append(("CLOSE", fill))
            return fill
    
    broker = MockBroker()
    oms = MockOMS()
    
    # Crear entorno
    env = BaseTradingEnv(cfg, broker, oms, initial_cash=10000.0)
    
    # Reset inicial
    obs = env.reset()
    
    # Verificar estado inicial
    assert env._trades_executed == 0
    assert len(oms.fills) == 0
    assert env.pos.side == 0
    
    # Forzar apertura LONG
    env.set_action_override(3)  # Force LONG
    
    # Ejecutar step
    obs, reward, done, info = env.step()
    
    # Verificar que se registró el fill
    assert env._trades_executed == 1
    assert len(oms.fills) == 1
    assert oms.fills[0][0] == "OPEN"
    
    fill = oms.fills[0][1]
    assert fill["side"] == 1  # Long
    assert fill["qty"] > 0
    assert fill["price"] > 0
    assert fill["sl"] is not None
    assert fill["tp"] is not None
    
    # Verificar que se abrió posición
    assert env.pos.side == 1
    assert env.pos.qty > 0
    assert env.pos.entry_price > 0
    
    print(f"✅ Fill registrado: side={fill['side']}, qty={fill['qty']:.6f}, "
          f"price={fill['price']:.2f}, sl={fill['sl']:.2f}, tp={fill['tp']:.2f}")


def test_forced_close_fill_registration():
    """Test que verifica que se registra un fill al forzar un cierre."""
    
    # Configuración similar
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
    
    # Mock de broker
    class MockBroker:
        def __init__(self):
            self.price = 50000.0
            self.step_count = 0
        
        def get_price(self): 
            return self.price
        
        def now_ts(self): 
            return 1640995200000 + self.step_count * 3600000
        
        def next(self): 
            self.step_count += 1
            self.price += 100
        
        def is_end_of_data(self): 
            return self.step_count > 10
        
        def reset_to_start(self): 
            self.step_count = 0
            self.price = 50000.0
    
    # Mock de OMS
    class MockOMS:
        def __init__(self):
            self.fills = []
        
        def open(self, side, qty, price, sl, tp):
            fill = {
                "side": 1 if side == "LONG" else -1,
                "qty": qty,
                "price": price,
                "fees": qty * price * 0.0005,
                "sl": sl,
                "tp": tp
            }
            self.fills.append(("OPEN", fill))
            return fill
        
        def close(self, qty, price):
            fill = {
                "side": -1,
                "qty": qty,
                "price": price,
                "fees": qty * price * 0.0005
            }
            self.fills.append(("CLOSE", fill))
            return fill
    
    broker = MockBroker()
    oms = MockOMS()
    
    env = BaseTradingEnv(cfg, broker, oms, initial_cash=10000.0)
    
    # Reset inicial
    obs = env.reset()
    
    # 1. Abrir posición
    env.set_action_override(3)  # Force LONG
    obs, reward, done, info = env.step()
    
    # Verificar apertura
    assert env._trades_executed == 1
    assert len(oms.fills) == 1
    assert env.pos.side == 1
    
    # 2. Cerrar posición
    env.set_action_override(1)  # Close all
    obs, reward, done, info = env.step()
    
    # Verificar cierre
    assert env._trades_executed == 2
    assert len(oms.fills) == 2
    assert oms.fills[1][0] == "CLOSE"
    
    close_fill = oms.fills[1][1]
    assert close_fill["side"] == -1  # Cerrar
    assert close_fill["qty"] > 0
    assert close_fill["price"] > 0
    
    # Verificar que se cerró posición
    assert env.pos.side == 0
    assert env.pos.qty == 0.0
    
    print(f"✅ Ciclo completo: {len(oms.fills)} fills, {env._trades_executed} trades ejecutados")


if __name__ == "__main__":
    test_forced_open_fill_registration()
    test_forced_close_fill_registration()
    print("✅ Tests de registro de fills pasaron correctamente")
