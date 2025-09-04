#!/usr/bin/env python3
"""
Test: Flujo completo de abrir y cerrar posición
- Abrir y cerrar posición: margin se reserva/libera; equity==balance al final; sin doble MTM.
"""

import pytest
import numpy as np
from pathlib import Path
import sys

# Agregar el directorio raíz al path
sys.path.insert(0, str(Path(__file__).parent.parent))

from base_env.base_env import BaseTradingEnv
from base_env.config.config_loader import config_loader
from base_env.io.broker import DataBroker
from base_env.accounting.ledger import PortfolioState, PositionState


class MockOMS:
    """OMS mock para testing"""
    def __init__(self):
        self.open_calls = []
        self.close_calls = []
    
    def open(self, side, qty, price, sl=None, tp=None):
        call = {"side": 1 if side == "LONG" else -1, "qty": qty, "price": price, "sl": sl, "tp": tp}
        self.open_calls.append(call)
        return call
    
    def close(self, qty, price):
        call = {"qty": qty, "price": price}
        self.close_calls.append(call)
        return call


class MockBroker:
    """Broker mock que simula cambios de precio controlados"""
    def __init__(self):
        self.prices = [50000.0, 51000.0, 52000.0, 53000.0, 54000.0]  # Precios crecientes
        self.tick_count = 0
        self.max_ticks = 10
    
    def get_price(self):
        price_idx = min(self.tick_count, len(self.prices) - 1)
        return self.prices[price_idx]
    
    def update_price(self):
        # Precio se actualiza en get_price()
        pass
    
    def next(self):
        self.tick_count += 1
    
    def is_end_of_data(self):
        return self.tick_count >= self.max_ticks
    
    def reset_to_start(self):
        self.tick_count = 0


def test_open_close_flow():
    """Test: Flujo completo de abrir y cerrar posición"""
    
    # Configuración mínima para el test
    import yaml
    with open("config/train.yaml", "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    symbol_cfg = cfg["symbols"][0]  # BTCUSDT
    
    # Crear entorno con componentes mock
    broker = MockBroker()
    oms = MockOMS()
    
    env = BaseTradingEnv(
        cfg=cfg,
        broker=broker,
        oms=oms,
        initial_cash=1000.0,
        target_cash=10000.0
    )
    
    # Estado inicial
    initial_cash = env.portfolio.cash_quote
    initial_equity = env.portfolio.equity_quote
    initial_margin = env.portfolio.used_margin
    
    print(f"Estado inicial: cash={initial_cash}, equity={initial_equity}, margin={initial_margin}")
    
    # Tick 1: Abrir posición LONG
    env._action_override = 3  # Force LONG
    obs, reward, done, info = env.step()
    
    # Verificar que se abrió posición
    assert env.pos.side == 1, "Debe haber posición LONG"
    assert env.pos.qty > 0, "Debe haber cantidad en posición"
    assert env.pos.entry_price > 0, "Debe haber precio de entrada"
    assert len(oms.open_calls) == 1, "Debe haber una llamada a open"
    
    # Verificar que se reservó margen
    assert env.portfolio.used_margin > 0, "Debe haber margen usado"
    assert env.portfolio.cash_quote < initial_cash, "Cash debe haber disminuido"
    
    # Verificar que equity incluye unrealized PnL
    assert env.portfolio.equity_quote != env.portfolio.cash_quote, "Equity debe diferir de cash con posición"
    
    print(f"Después de abrir: cash={env.portfolio.cash_quote:.2f}, equity={env.portfolio.equity_quote:.2f}, margin={env.portfolio.used_margin:.2f}")
    print(f"Posición: side={env.pos.side}, qty={env.pos.qty:.6f}, entry={env.pos.entry_price:.2f}")
    
    # Ticks 2-4: Mantener posición (precio sube)
    for i in range(3):
        obs, reward, done, info = env.step()
        
        # Verificar que la posición se mantiene
        assert env.pos.side == 1, f"Posición debe mantenerse en tick {i+2}"
        assert env.pos.qty > 0, f"Debe haber cantidad en tick {i+2}"
        
        # Verificar que unrealized PnL se actualiza
        current_price = broker.get_price()
        expected_unrealized = (current_price - env.pos.entry_price) * env.pos.qty
        assert abs(env.pos.unrealized_pnl - expected_unrealized) < 1e-6, f"Unrealized PnL incorrecto en tick {i+2}"
        
        # Verificar que equity = cash + unrealized PnL
        expected_equity = env.portfolio.cash_quote + env.pos.unrealized_pnl
        assert abs(env.portfolio.equity_quote - expected_equity) < 1e-6, f"Equity incorrecto en tick {i+2}"
    
    print(f"Después de mantener: cash={env.portfolio.cash_quote:.2f}, equity={env.portfolio.equity_quote:.2f}, margin={env.portfolio.used_margin:.2f}")
    print(f"Unrealized PnL: {env.pos.unrealized_pnl:.2f}")
    
    # Tick 5: Cerrar posición
    env._action_override = 1  # Force CLOSE
    obs, reward, done, info = env.step()
    
    # Verificar que se cerró posición
    assert env.pos.side == 0, "Posición debe estar cerrada"
    assert env.pos.qty == 0.0, "Cantidad debe ser 0"
    assert env.pos.entry_price == 0.0, "Precio de entrada debe ser 0"
    assert len(oms.close_calls) == 1, "Debe haber una llamada a close"
    
    # Verificar que se liberó margen
    assert env.portfolio.used_margin == 0.0, "Margen debe estar liberado"
    
    # Verificar que equity == balance después del cierre
    assert abs(env.portfolio.equity_quote - env.portfolio.cash_quote) < 1e-6, "Equity debe igualar cash después del cierre"
    
    # Verificar que no hay unrealized PnL
    assert env.pos.unrealized_pnl == 0.0, "Unrealized PnL debe ser 0"
    
    print(f"Después de cerrar: cash={env.portfolio.cash_quote:.2f}, equity={env.portfolio.equity_quote:.2f}, margin={env.portfolio.used_margin:.2f}")
    
    # Verificar que hubo PnL realizado
    final_cash = env.portfolio.cash_quote
    cash_change = final_cash - initial_cash
    print(f"Cambio en cash: {cash_change:.2f}")
    
    # El PnL debe ser positivo (precio subió de 50000 a 54000)
    assert cash_change > 0, "Debe haber PnL positivo"
    
    print("✅ Test passed: Flujo completo de abrir/cerrar posición funciona correctamente")


if __name__ == "__main__":
    test_open_close_flow()
