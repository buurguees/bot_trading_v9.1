#!/usr/bin/env python3
"""
Test: Verificar que sin posición, equity == balance (sin drift)
- 500 ticks sin posición → equity==balance, sin drift; funding/fees=0.
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
    def open(self, side, qty, price, sl=None, tp=None):
        return {"side": 1 if side == "LONG" else -1, "qty": qty, "price": price, "sl": sl, "tp": tp}
    
    def close(self, qty, price):
        return {"qty": qty, "price": price}


class MockBroker:
    """Broker mock que simula 500 ticks sin cambios de precio"""
    def __init__(self):
        self.price = 50000.0
        self.tick_count = 0
        self.max_ticks = 500
    
    def get_price(self):
        return self.price
    
    def update_price(self):
        # Precio constante para evitar PnL
        pass
    
    def next(self):
        self.tick_count += 1
    
    def is_end_of_data(self):
        return self.tick_count >= self.max_ticks
    
    def reset_to_start(self):
        self.tick_count = 0


def test_ledger_no_position_invariant():
    """Test: 500 ticks sin posición → equity==balance, sin drift"""
    
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
    
    # Verificar estado inicial
    assert env.pos.side == 0, "Posición inicial debe ser 0"
    assert env.portfolio.cash_quote == 1000.0, "Cash inicial debe ser 1000"
    assert env.portfolio.equity_quote == 1000.0, "Equity inicial debe ser 1000"
    assert env.portfolio.used_margin == 0.0, "Used margin inicial debe ser 0"
    
    # Ejecutar 500 ticks sin abrir posición
    drift_count = 0
    for i in range(500):
        obs, reward, done, info = env.step()
        
        # Verificar invariante: sin posición → equity == balance
        if env.pos.side == 0:
            drift = abs(env.portfolio.equity_quote - env.portfolio.cash_quote)
            if drift > 1e-6:
                drift_count += 1
                print(f"⚠️ Drift detectado en tick {i}: equity={env.portfolio.equity_quote:.8f}, cash={env.portfolio.cash_quote:.8f}, drift={drift:.8f}")
        
        # Verificar que no hay margen usado sin posición
        if env.pos.side == 0:
            assert env.portfolio.used_margin == 0.0, f"Used margin debe ser 0 sin posición en tick {i}"
        
        # Verificar que no hay unrealized PnL sin posición
        if env.pos.side == 0:
            assert env.pos.unrealized_pnl == 0.0, f"Unrealized PnL debe ser 0 sin posición en tick {i}"
        
        if done:
            break
    
    # Verificaciones finales
    assert drift_count == 0, f"Se detectaron {drift_count} casos de drift sin posición"
    assert env.portfolio.equity_quote == env.portfolio.cash_quote, "Equity final debe igualar cash"
    assert env.portfolio.used_margin == 0.0, "Used margin final debe ser 0"
    assert env.pos.unrealized_pnl == 0.0, "Unrealized PnL final debe ser 0"
    
    print("✅ Test passed: Sin drift contable en 500 ticks sin posición")


if __name__ == "__main__":
    test_ledger_no_position_invariant()
