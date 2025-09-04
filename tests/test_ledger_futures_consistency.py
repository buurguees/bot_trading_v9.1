# tests/test_ledger_futures_consistency.py
"""
Test para validar la contabilidad correcta de futuros (margen, PnL MTM, fees).
"""

import pytest
from unittest.mock import MagicMock
import sys
from pathlib import Path

# Agregar el directorio raíz al path
sys.path.insert(0, str(Path(__file__).parent.parent))

from base_env.accounting.ledger import Accounting, PortfolioState, PositionState


class TestLedgerFuturesConsistency:
    """Test de consistencia de contabilidad para futuros"""
    
    def setup_method(self):
        """Setup para cada test"""
        # Configuración de fees
        self.fees_cfg = {
            "futures": {
                "taker_fee_bps": 5.0
            }
        }
        
        # Accounting para futuros
        self.accounting = Accounting(self.fees_cfg, "futures")
        
        # Portfolio inicial
        self.portfolio = PortfolioState(market="futures")
        self.portfolio.reset(initial_cash=1000.0, target_cash=1000000.0)
        
        # Posición inicial
        self.position = PositionState()
        
        # Mock de configuración
        self.cfg = MagicMock()
        self.cfg.leverage = 3.0
    
    def test_portfolio_initialization_futures(self):
        """Test que el portfolio se inicializa correctamente para futuros"""
        assert self.portfolio.market == "futures"
        assert self.portfolio.cash_quote == 1000.0
        assert self.portfolio.used_margin == 0.0
        assert self.portfolio.equity_quote == 1000.0
        assert self.portfolio.target_quote == 1000000.0
    
    def test_apply_open_long_futures(self):
        """Test apertura de posición larga en futuros"""
        # Fill para apertura
        fill = {
            "side": 1,
            "qty": 0.1,
            "price": 50000.0,
            "sl": 49000.0,
            "tp": 52000.0
        }
        
        # Aplicar apertura
        self.accounting.apply_open(fill, self.portfolio, self.position, self.cfg)
        
        # Verificar posición
        assert self.position.side == 1
        assert self.position.qty == 0.1
        assert self.position.entry_price == 50000.0
        assert self.position.sl == 49000.0
        assert self.position.tp == 52000.0
        
        # Verificar portfolio
        notional = 0.1 * 50000.0  # 5000 USDT
        margin_required = notional / 3.0  # 1666.67 USDT
        fee = notional * 0.0005  # 2.5 USDT
        
        assert self.portfolio.cash_quote == 1000.0 - margin_required - fee
        assert self.portfolio.used_margin == margin_required
    
    def test_apply_open_short_futures(self):
        """Test apertura de posición corta en futuros"""
        # Fill para apertura corta
        fill = {
            "side": -1,
            "qty": 0.1,
            "price": 50000.0,
            "sl": 51000.0,
            "tp": 48000.0
        }
        
        # Aplicar apertura
        self.accounting.apply_open(fill, self.portfolio, self.position, self.cfg)
        
        # Verificar posición
        assert self.position.side == -1
        assert self.position.qty == 0.1
        assert self.position.entry_price == 50000.0
        
        # Verificar portfolio (mismo margen para long y short en futuros)
        notional = 0.1 * 50000.0  # 5000 USDT
        margin_required = notional / 3.0  # 1666.67 USDT
        fee = notional * 0.0005  # 2.5 USDT
        
        assert self.portfolio.cash_quote == 1000.0 - margin_required - fee
        assert self.portfolio.used_margin == margin_required
    
    def test_apply_close_long_futures(self):
        """Test cierre de posición larga en futuros"""
        # Primero abrir posición
        fill_open = {
            "side": 1,
            "qty": 0.1,
            "price": 50000.0,
            "sl": 49000.0,
            "tp": 52000.0
        }
        self.accounting.apply_open(fill_open, self.portfolio, self.position, self.cfg)
        
        # Estado antes del cierre
        cash_before = self.portfolio.cash_quote
        margin_before = self.portfolio.used_margin
        
        # Fill para cierre
        fill_close = {
            "qty": 0.1,
            "price": 51000.0  # Precio más alto = ganancia
        }
        
        # Aplicar cierre
        realized = self.accounting.apply_close(fill_close, self.portfolio, self.position, self.cfg)
        
        # Verificar PnL realizado
        expected_realized = (51000.0 - 50000.0) * 0.1  # 100 USDT
        assert realized == expected_realized
        
        # Verificar que se liberó el margen
        assert self.portfolio.used_margin == 0.0
        
        # Verificar que se recibió el PnL + margen liberado - fee
        notional = 0.1 * 51000.0  # 5100 USDT
        fee = notional * 0.0005  # 2.55 USDT
        margin_liberated = (50000.0 * 0.1) / 3.0  # 1666.67 USDT
        
        expected_cash = cash_before + expected_realized + margin_liberated - fee
        assert abs(self.portfolio.cash_quote - expected_cash) < 0.01
        
        # Verificar que la posición se cerró
        assert self.position.side == 0
        assert self.position.qty == 0.0
        assert self.position.entry_price == 0.0
    
    def test_apply_close_short_futures(self):
        """Test cierre de posición corta en futuros"""
        # Primero abrir posición corta
        fill_open = {
            "side": -1,
            "qty": 0.1,
            "price": 50000.0,
            "sl": 51000.0,
            "tp": 48000.0
        }
        self.accounting.apply_open(fill_open, self.portfolio, self.position, self.cfg)
        
        # Estado antes del cierre
        cash_before = self.portfolio.cash_quote
        margin_before = self.portfolio.used_margin
        
        # Fill para cierre
        fill_close = {
            "qty": 0.1,
            "price": 49000.0  # Precio más bajo = ganancia para short
        }
        
        # Aplicar cierre
        realized = self.accounting.apply_close(fill_close, self.portfolio, self.position, self.cfg)
        
        # Verificar PnL realizado
        expected_realized = (50000.0 - 49000.0) * 0.1  # 100 USDT
        assert realized == expected_realized
        
        # Verificar que se liberó el margen
        assert self.portfolio.used_margin == 0.0
        
        # Verificar que se recibió el PnL + margen liberado - fee
        notional = 0.1 * 49000.0  # 4900 USDT
        fee = notional * 0.0005  # 2.45 USDT
        margin_liberated = (50000.0 * 0.1) / 3.0  # 1666.67 USDT
        
        expected_cash = cash_before + expected_realized + margin_liberated - fee
        assert abs(self.portfolio.cash_quote - expected_cash) < 0.01
    
    def test_update_unrealized_long_futures(self):
        """Test actualización de PnL no realizado para posición larga"""
        # Abrir posición larga
        fill = {
            "side": 1,
            "qty": 0.1,
            "price": 50000.0,
            "sl": 49000.0,
            "tp": 52000.0
        }
        self.accounting.apply_open(fill, self.portfolio, self.position, self.cfg)
        
        # Mock broker con precio actual
        mock_broker = MagicMock()
        mock_broker.get_price.return_value = 51000.0
        
        # Actualizar PnL no realizado
        self.accounting.update_unrealized(mock_broker, self.position, self.portfolio)
        
        # Verificar PnL no realizado
        expected_unrealized = (51000.0 - 50000.0) * 0.1  # 100 USDT
        assert self.position.unrealized_pnl == expected_unrealized
        
        # Verificar equity
        expected_equity = self.portfolio.cash_quote + expected_unrealized
        assert abs(self.portfolio.equity_quote - expected_equity) < 0.01
    
    def test_update_unrealized_short_futures(self):
        """Test actualización de PnL no realizado para posición corta"""
        # Abrir posición corta
        fill = {
            "side": -1,
            "qty": 0.1,
            "price": 50000.0,
            "sl": 51000.0,
            "tp": 48000.0
        }
        self.accounting.apply_open(fill, self.portfolio, self.position, self.cfg)
        
        # Mock broker con precio actual
        mock_broker = MagicMock()
        mock_broker.get_price.return_value = 49000.0
        
        # Actualizar PnL no realizado
        self.accounting.update_unrealized(mock_broker, self.position, self.portfolio)
        
        # Verificar PnL no realizado
        expected_unrealized = (50000.0 - 49000.0) * 0.1  # 100 USDT
        assert self.position.unrealized_pnl == expected_unrealized
        
        # Verificar equity
        expected_equity = self.portfolio.cash_quote + expected_unrealized
        assert abs(self.portfolio.equity_quote - expected_equity) < 0.01
    
    def test_update_unrealized_no_position(self):
        """Test actualización cuando no hay posición"""
        # Mock broker
        mock_broker = MagicMock()
        mock_broker.get_price.return_value = 50000.0
        
        # Actualizar PnL no realizado
        self.accounting.update_unrealized(mock_broker, self.position, self.portfolio)
        
        # Verificar que no hay PnL no realizado
        assert self.position.unrealized_pnl == 0.0
        
        # Verificar que equity == cash
        assert self.portfolio.equity_quote == self.portfolio.cash_quote
    
    def test_portfolio_consistency_validation(self):
        """Test que las validaciones de consistencia funcionan"""
        # Test 1: Sin posición, equity == cash
        self.accounting._validate_portfolio_consistency(self.position, self.portfolio)
        assert self.portfolio.equity_quote == self.portfolio.cash_quote
        
        # Test 2: Con posición, equity != cash
        fill = {
            "side": 1,
            "qty": 0.1,
            "price": 50000.0,
            "sl": 49000.0,
            "tp": 52000.0
        }
        self.accounting.apply_open(fill, self.portfolio, self.position, self.cfg)
        
        # Mock broker
        mock_broker = MagicMock()
        mock_broker.get_price.return_value = 51000.0
        
        # Actualizar PnL
        self.accounting.update_unrealized(mock_broker, self.position, self.portfolio)
        
        # Verificar que equity != cash (esto es correcto con posición)
        assert self.portfolio.equity_quote != self.portfolio.cash_quote
        
        # Test 3: Cerrar posición y verificar que se libera margen
        fill_close = {
            "qty": 0.1,
            "price": 51000.0
        }
        self.accounting.apply_close(fill_close, self.portfolio, self.position, self.cfg)
        
        # Verificar que se liberó el margen
        assert self.portfolio.used_margin == 0.0
        
        # Verificar que equity == cash después del cierre
        assert abs(self.portfolio.equity_quote - self.portfolio.cash_quote) < 0.01
    
    def test_fees_calculation(self):
        """Test que las fees se calculan correctamente"""
        # Test con diferentes notionales
        notional = 1000.0
        expected_fee = notional * 0.0005  # 0.5 USDT
        
        actual_fee = self.accounting._taker_bps(self.portfolio) / 10000.0 * notional
        assert abs(actual_fee - expected_fee) < 0.001
    
    def test_leverage_variation(self):
        """Test con diferentes niveles de leverage"""
        leverages = [1.0, 2.0, 5.0, 10.0]
        
        for leverage in leverages:
            # Reset portfolio
            self.portfolio.reset(initial_cash=1000.0, target_cash=1000000.0)
            self.position.reset()
            
            # Configurar leverage
            self.cfg.leverage = leverage
            
            # Abrir posición
            fill = {
                "side": 1,
                "qty": 0.1,
                "price": 50000.0,
                "sl": 49000.0,
                "tp": 52000.0
            }
            self.accounting.apply_open(fill, self.portfolio, self.position, self.cfg)
            
            # Verificar margen
            notional = 0.1 * 50000.0  # 5000 USDT
            expected_margin = notional / leverage
            assert abs(self.portfolio.used_margin - expected_margin) < 0.01
            
            # Cerrar posición
            fill_close = {
                "qty": 0.1,
                "price": 51000.0
            }
            self.accounting.apply_close(fill_close, self.portfolio, self.position, self.cfg)
            
            # Verificar que se liberó el margen
            assert self.portfolio.used_margin == 0.0
    
    def test_partial_close(self):
        """Test cierre parcial de posición"""
        # Abrir posición
        fill = {
            "side": 1,
            "qty": 0.2,
            "price": 50000.0,
            "sl": 49000.0,
            "tp": 52000.0
        }
        self.accounting.apply_open(fill, self.portfolio, self.position, self.cfg)
        
        # Cerrar parcialmente
        fill_close = {
            "qty": 0.1,  # Solo la mitad
            "price": 51000.0
        }
        realized = self.accounting.apply_close(fill_close, self.portfolio, self.position, self.cfg)
        
        # Verificar PnL realizado
        expected_realized = (51000.0 - 50000.0) * 0.1  # 100 USDT
        assert realized == expected_realized
        
        # Verificar que la posición se redujo
        assert self.position.qty == 0.1  # Quedó la mitad
        assert self.position.entry_price == 50000.0  # Precio de entrada se mantiene
        
        # Verificar que se liberó margen proporcional
        expected_margin_remaining = (0.1 * 50000.0) / 3.0  # Margen para la posición restante
        assert abs(self.portfolio.used_margin - expected_margin_remaining) < 0.01


if __name__ == "__main__":
    pytest.main([__file__])
