# tests/test_ledger_consistency.py
"""
Test para validar la consistencia de la contabilidad SPOT/FUTUROS:
- Sin posición: equity == balance siempre
- Open/close libera margen correctamente
- No fugas de equity sin posición
- No valores NaN/Inf
"""

import pytest
import tempfile
from pathlib import Path
from unittest.mock import MagicMock
import sys

# Añadir el directorio raíz al path
sys.path.insert(0, str(Path(__file__).parent.parent))

from base_env.accounting.ledger import Accounting, PortfolioState, PositionState


class TestLedgerConsistency:
    """Test de consistencia de la contabilidad"""
    
    def setup_method(self):
        """Setup para cada test"""
        # Configuración de fees
        self.fees_cfg = {
            "spot": {"taker_fee_bps": 10},  # 0.1%
            "futures": {"taker_fee_bps": 5}  # 0.05%
        }
        
        # Accounting para spot
        self.accounting_spot = Accounting(self.fees_cfg, "spot")
        
        # Accounting para futuros
        self.accounting_futures = Accounting(self.fees_cfg, "futures")
    
    def test_spot_no_position_equity_equals_balance(self):
        """Test que sin posición, equity == balance en spot"""
        portfolio = PortfolioState(market="spot")
        portfolio.reset(initial_cash=1000.0, target_cash=10000.0)
        pos = PositionState()
        
        # Sin posición
        assert pos.side == 0
        assert pos.qty == 0.0
        
        # Verificar que equity == balance
        assert abs(portfolio.equity_quote - portfolio.cash_quote) < 1e-6
    
    def test_futures_no_position_equity_equals_balance(self):
        """Test que sin posición, equity == balance en futuros"""
        portfolio = PortfolioState(market="futures")
        portfolio.reset(initial_cash=1000.0, target_cash=10000.0)
        pos = PositionState()
        
        # Sin posición
        assert pos.side == 0
        assert pos.qty == 0.0
        assert portfolio.used_margin == 0.0
        
        # Verificar que equity == balance
        assert abs(portfolio.equity_quote - portfolio.cash_quote) < 1e-6
    
    def test_spot_long_open_close_cycle(self):
        """Test ciclo completo open/close en spot long"""
        portfolio = PortfolioState(market="spot")
        portfolio.reset(initial_cash=1000.0, target_cash=10000.0)
        pos = PositionState()
        
        initial_cash = portfolio.cash_quote
        initial_equity = portfolio.equity_quote
        
        # Mock de configuración
        cfg = MagicMock()
        cfg.leverage = 1.0
        
        # OPEN: Long 0.01 BTC a 50000 USDT
        fill_open = {
            "side": 1,
            "qty": 0.01,
            "price": 50000.0,
            "fees": 0.0,
            "sl": 49000.0,
            "tp": 52000.0
        }
        
        self.accounting_spot.apply_open(fill_open, portfolio, pos, cfg)
        
        # Verificar estado después de apertura
        assert pos.side == 1
        assert pos.qty == 0.01
        assert pos.entry_price == 50000.0
        assert portfolio.equity_base == 0.01  # Inventario en BTC
        
        # El cash debe haber disminuido por el notional + fees
        expected_cash_decrease = 500.0  # 0.01 * 50000
        assert portfolio.cash_quote < initial_cash
        
        # CLOSE: Cerrar posición a 51000 USDT
        fill_close = {
            "qty": 0.01,
            "price": 51000.0,
            "fees": 0.0
        }
        
        realized = self.accounting_spot.apply_close(fill_close, portfolio, pos, cfg)
        
        # Verificar PnL realizado
        expected_pnl = (51000.0 - 50000.0) * 0.01  # 100 USDT
        assert abs(realized - expected_pnl) < 1e-6
        
        # Verificar que se cerró la posición
        assert pos.side == 0
        assert pos.qty == 0.0
        assert portfolio.equity_base == 0.0
        
        # Verificar que equity == balance al final
        assert abs(portfolio.equity_quote - portfolio.cash_quote) < 1e-6
    
    def test_futures_open_close_margin_cycle(self):
        """Test ciclo completo open/close en futuros con margen"""
        portfolio = PortfolioState(market="futures")
        portfolio.reset(initial_cash=1000.0, target_cash=10000.0)
        pos = PositionState()
        
        initial_cash = portfolio.cash_quote
        initial_equity = portfolio.equity_quote
        
        # Mock de configuración con leverage 2x
        cfg = MagicMock()
        cfg.leverage = 2.0
        
        # OPEN: Long 0.01 BTC a 50000 USDT con 2x leverage
        fill_open = {
            "side": 1,
            "qty": 0.01,
            "price": 50000.0,
            "fees": 0.0,
            "sl": 49000.0,
            "tp": 52000.0
        }
        
        self.accounting_futures.apply_open(fill_open, portfolio, pos, cfg)
        
        # Verificar estado después de apertura
        assert pos.side == 1
        assert pos.qty == 0.01
        assert pos.entry_price == 50000.0
        
        # El margen usado debe ser notional / leverage
        expected_margin = 500.0 / 2.0  # 250 USDT
        assert abs(portfolio.used_margin - expected_margin) < 1e-6
        
        # El cash debe haber disminuido por el margen + fees
        assert portfolio.cash_quote < initial_cash
        
        # CLOSE: Cerrar posición a 51000 USDT
        fill_close = {
            "qty": 0.01,
            "price": 51000.0,
            "fees": 0.0
        }
        
        realized = self.accounting_futures.apply_close(fill_close, portfolio, pos, cfg)
        
        # Verificar PnL realizado
        expected_pnl = (51000.0 - 50000.0) * 0.01  # 100 USDT
        assert abs(realized - expected_pnl) < 1e-6
        
        # Verificar que se cerró la posición
        assert pos.side == 0
        assert pos.qty == 0.0
        assert portfolio.used_margin == 0.0  # Margen liberado
        
        # Verificar que equity == balance al final
        assert abs(portfolio.equity_quote - portfolio.cash_quote) < 1e-6
    
    def test_spot_short_margin_cycle(self):
        """Test ciclo completo open/close en spot short con margen"""
        portfolio = PortfolioState(market="spot")
        portfolio.reset(initial_cash=1000.0, target_cash=10000.0)
        pos = PositionState()
        
        initial_cash = portfolio.cash_quote
        
        # Mock de configuración
        cfg = MagicMock()
        cfg.leverage = 1.0
        
        # OPEN: Short 0.01 BTC a 50000 USDT
        fill_open = {
            "side": -1,
            "qty": 0.01,
            "price": 50000.0,
            "fees": 0.0,
            "sl": 51000.0,
            "tp": 48000.0
        }
        
        self.accounting_spot.apply_open(fill_open, portfolio, pos, cfg)
        
        # Verificar estado después de apertura
        assert pos.side == -1
        assert pos.qty == 0.01
        assert pos.entry_price == 50000.0
        assert portfolio.equity_base == -0.01  # Deuda en BTC
        
        # El margen usado debe ser 50% del notional
        expected_margin = 500.0 * 0.5  # 250 USDT
        assert abs(portfolio.used_margin - expected_margin) < 1e-6
        
        # CLOSE: Cerrar posición a 49000 USDT
        fill_close = {
            "qty": 0.01,
            "price": 49000.0,
            "fees": 0.0
        }
        
        realized = self.accounting_futures.apply_close(fill_close, portfolio, pos, cfg)
        
        # Verificar PnL realizado (positivo para short)
        expected_pnl = (50000.0 - 49000.0) * 0.01  # 100 USDT
        assert abs(realized - expected_pnl) < 1e-6
        
        # Verificar que se cerró la posición
        assert pos.side == 0
        assert pos.qty == 0.0
        assert portfolio.equity_base == 0.0
        assert portfolio.used_margin == 0.0  # Margen liberado
    
    def test_unrealized_pnl_calculation(self):
        """Test cálculo de PnL no realizado"""
        portfolio = PortfolioState(market="spot")
        portfolio.reset(initial_cash=1000.0, target_cash=10000.0)
        pos = PositionState()
        
        # Mock broker con precio actual
        broker = MagicMock()
        broker.get_price.return_value = 51000.0
        
        # Posición long abierta
        pos.side = 1
        pos.qty = 0.01
        pos.entry_price = 50000.0
        
        # Actualizar PnL no realizado
        self.accounting_spot.update_unrealized(broker, pos, portfolio)
        
        # Verificar PnL no realizado
        expected_unrealized = (51000.0 - 50000.0) * 0.01  # 100 USDT
        assert abs(pos.unrealized_pnl - expected_unrealized) < 1e-6
        
        # Verificar que equity incluye PnL no realizado
        expected_equity = portfolio.cash_quote + expected_unrealized
        assert abs(portfolio.equity_quote - expected_equity) < 1e-6
    
    def test_no_nan_inf_values(self):
        """Test que no se generan valores NaN o Inf"""
        portfolio = PortfolioState(market="spot")
        portfolio.reset(initial_cash=1000.0, target_cash=10000.0)
        pos = PositionState()
        
        # Mock de configuración
        cfg = MagicMock()
        cfg.leverage = 1.0
        
        # Operación que podría generar NaN/Inf
        fill_open = {
            "side": 1,
            "qty": 0.01,
            "price": 50000.0,
            "fees": 0.0,
            "sl": 49000.0,
            "tp": 52000.0
        }
        
        self.accounting_spot.apply_open(fill_open, portfolio, pos, cfg)
        
        # Verificar que no hay NaN/Inf
        assert portfolio.equity_quote == portfolio.equity_quote  # No NaN
        assert portfolio.cash_quote == portfolio.cash_quote  # No NaN
        assert portfolio.equity_quote not in (float('inf'), float('-inf'))
        assert portfolio.cash_quote not in (float('inf'), float('-inf'))
        assert pos.unrealized_pnl == pos.unrealized_pnl  # No NaN
        assert pos.unrealized_pnl not in (float('inf'), float('-inf'))
    
    def test_portfolio_consistency_validation(self):
        """Test que la validación de consistencia funciona"""
        portfolio = PortfolioState(market="spot")
        portfolio.reset(initial_cash=1000.0, target_cash=10000.0)
        pos = PositionState()
        
        # Simular inconsistencia: equity != cash sin posición
        portfolio.equity_quote = 1200.0  # Inconsistente
        
        # Aplicar validación
        self.accounting_spot._validate_portfolio_consistency(pos, portfolio)
        
        # Verificar que se corrigió
        assert abs(portfolio.equity_quote - portfolio.cash_quote) < 1e-6
    
    def test_margin_validation(self):
        """Test validación de margen usado"""
        portfolio = PortfolioState(market="futures")
        portfolio.reset(initial_cash=1000.0, target_cash=10000.0)
        pos = PositionState()
        
        # Simular margen negativo (inconsistencia)
        portfolio.used_margin = -100.0
        
        # Aplicar validación
        self.accounting_futures._validate_portfolio_consistency(pos, portfolio)
        
        # Verificar que se corrigió
        assert portfolio.used_margin >= 0.0


if __name__ == "__main__":
    pytest.main([__file__])