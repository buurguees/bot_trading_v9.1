"""
Test simplificado para validar la consistencia del ledger.
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from base_env.accounting.ledger import Accounting, PortfolioState, PositionState


def test_equity_equals_balance_without_position():
    """Test que verifica que equity == balance cuando no hay posición."""
    
    # Crear accounting
    fees_cfg = {
        "spot": {"taker_fee_bps": 10},
        "futures": {"taker_fee_bps": 5}
    }
    accounting = Accounting(fees_cfg, "futures")
    
    # Portfolio inicial
    portfolio = PortfolioState(market="futures")
    portfolio.reset(initial_cash=10000.0)
    
    # Posición vacía
    position = PositionState()
    
    # Mock broker
    class MockBroker:
        def get_price(self):
            return 50000.0
    
    broker = MockBroker()
    
    # Verificar invariante inicial
    assert portfolio.equity_quote == portfolio.cash_quote, f"Equity inicial {portfolio.equity_quote} != cash {portfolio.cash_quote}"
    assert portfolio.used_margin == 0.0, f"Used margin inicial {portfolio.used_margin} != 0"
    assert position.side == 0, f"Position side inicial {position.side} != 0"
    
    # Simular 200 ticks sin posición
    for i in range(200):
        # Actualizar unrealized (no debería cambiar nada sin posición)
        accounting.update_unrealized(broker, position, portfolio)
        
        # Verificar invariante en cada tick
        assert abs(portfolio.equity_quote - portfolio.cash_quote) < 1e-6, f"Tick {i}: equity {portfolio.equity_quote} != cash {portfolio.cash_quote}"
        assert portfolio.used_margin == 0.0, f"Tick {i}: used_margin {portfolio.used_margin} != 0"
        assert position.side == 0, f"Tick {i}: position side {position.side} != 0"
    
    print(f"✅ 200 ticks sin posición: equity={portfolio.equity_quote:.8f}, cash={portfolio.cash_quote:.8f}")


def test_drift_correction():
    """Test que verifica que se corrige el drift de equity automáticamente."""
    
    # Crear accounting
    fees_cfg = {
        "spot": {"taker_fee_bps": 10},
        "futures": {"taker_fee_bps": 5}
    }
    accounting = Accounting(fees_cfg, "futures")
    
    # Portfolio inicial
    portfolio = PortfolioState(market="futures")
    portfolio.reset(initial_cash=10000.0)
    
    # Posición vacía
    position = PositionState()
    
    # Simular drift artificial
    portfolio.equity_quote = 10000.5  # Drift de 0.5 USDT
    
    # Mock broker
    class MockBroker:
        def get_price(self):
            return 50000.0
    
    broker = MockBroker()
    
    # Actualizar unrealized debería corregir el drift
    accounting.update_unrealized(broker, position, portfolio)
    
    # Verificar que se corrigió
    assert abs(portfolio.equity_quote - portfolio.cash_quote) < 1e-6, f"Drift no corregido: equity {portfolio.equity_quote} != cash {portfolio.cash_quote}"
    assert portfolio.used_margin == 0.0, f"Used margin {portfolio.used_margin} != 0"
    
    print(f"✅ Drift corregido: equity={portfolio.equity_quote:.8f}, cash={portfolio.cash_quote:.8f}")


def test_open_close_cycle_consistency():
    """Test que verifica consistencia en un ciclo completo de apertura/cierre."""
    
    # Crear accounting
    fees_cfg = {
        "spot": {"taker_fee_bps": 10},
        "futures": {"taker_fee_bps": 5}
    }
    accounting = Accounting(fees_cfg, "futures")
    
    # Portfolio inicial
    portfolio = PortfolioState(market="futures")
    portfolio.reset(initial_cash=10000.0)
    
    # Posición vacía
    position = PositionState()
    
    # Mock broker
    class MockBroker:
        def __init__(self):
            self.price = 50000.0
        
        def get_price(self):
            return self.price
    
    broker = MockBroker()
    
    # Estado inicial
    initial_cash = portfolio.cash_quote
    initial_equity = portfolio.equity_quote
    
    assert abs(initial_equity - initial_cash) < 1e-6, f"Estado inicial: equity {initial_equity} != cash {initial_cash}"
    assert portfolio.used_margin == 0.0, f"Estado inicial: used_margin {portfolio.used_margin} != 0"
    
    # 1. APERTURA
    fill_open = {
        "side": 1,  # Long
        "qty": 0.1,
        "price": 50000.0,
        "fees": 2.5,  # 0.1 * 50000 * 5bps
        "sl": 49500.0,
        "tp": 50750.0
    }
    
    # Mock config para leverage
    class MockConfig:
        leverage = 2.0
    
    cfg = MockConfig()
    
    accounting.apply_open(fill_open, portfolio, position, cfg)
    
    # Verificar estado después de apertura
    assert position.side == 1, f"Position side después de apertura {position.side} != 1"
    assert position.qty == 0.1, f"Position qty después de apertura {position.qty} != 0.1"
    assert position.entry_price == 50000.0, f"Position entry_price después de apertura {position.entry_price} != 50000.0"
    assert portfolio.used_margin > 0, f"Used margin después de apertura {portfolio.used_margin} <= 0"
    
    # Actualizar unrealized
    accounting.update_unrealized(broker, position, portfolio)
    
    # Verificar que equity incluye PnL no realizado
    expected_equity = portfolio.cash_quote + position.unrealized_pnl
    assert abs(portfolio.equity_quote - expected_equity) < 1e-6, f"Equity {portfolio.equity_quote} != cash {portfolio.cash_quote} + unrealized {position.unrealized_pnl}"
    
    # 2. CIERRE
    broker.price = 51000.0  # Precio subió 2%
    
    fill_close = {
        "side": -1,  # Cerrar long
        "qty": 0.1,
        "price": 51000.0,
        "fees": 2.55  # 0.1 * 51000 * 5bps
    }
    
    realized = accounting.apply_close(fill_close, portfolio, position, cfg)
    
    # Verificar estado después de cierre
    assert position.side == 0, f"Position side después de cierre {position.side} != 0"
    assert position.qty == 0.0, f"Position qty después de cierre {position.qty} != 0.0"
    assert portfolio.used_margin == 0.0, f"Used margin después de cierre {portfolio.used_margin} != 0"
    
    # Actualizar unrealized (no debería cambiar nada)
    accounting.update_unrealized(broker, position, portfolio)
    
    # Verificar invariante final: equity == balance
    assert abs(portfolio.equity_quote - portfolio.cash_quote) < 1e-6, f"Final: equity {portfolio.equity_quote} != cash {portfolio.cash_quote}"
    
    # Verificar que el PnL se reflejó en el balance
    expected_pnl = (51000.0 - 50000.0) * 0.1  # 100 USDT
    expected_fees = 2.5 + 2.55  # 5.05 USDT
    expected_final_cash = initial_cash + expected_pnl - expected_fees
    
    assert abs(portfolio.cash_quote - expected_final_cash) < 1e-6, f"Final cash {portfolio.cash_quote} != expected {expected_final_cash}"
    
    print(f"✅ Ciclo completo: PnL={expected_pnl:.2f}, Fees={expected_fees:.2f}, "
          f"Final cash={portfolio.cash_quote:.2f}, Final equity={portfolio.equity_quote:.2f}")


if __name__ == "__main__":
    test_equity_equals_balance_without_position()
    test_drift_correction()
    test_open_close_cycle_consistency()
    print("✅ Tests de consistencia del ledger pasaron correctamente")
