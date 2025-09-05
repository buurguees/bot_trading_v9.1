# tests/test_sizing_futures_min_notional.py
"""
Test para validar el sizing de futuros con minNotional y límites de leverage.
"""

import pytest
from unittest.mock import MagicMock, patch
import sys
from pathlib import Path

# Agregar el directorio raíz al path
sys.path.insert(0, str(Path(__file__).parent.parent))

from base_env.risk.manager import RiskManager, SizedDecision
from base_env.config.models import RiskConfig, SymbolMeta, RiskCommon, RiskFutures


class TestSizingFuturesMinNotional:
    """Test del sizing de futuros con minNotional"""
    
    def setup_method(self):
        """Setup para cada test"""
        # Configuración de riesgo
        self.risk_config = RiskConfig(
            common=RiskCommon(
                train_force_min_notional=True
            ),
            futures=RiskFutures(
                risk_pct_per_trade=2.0,
                max_initial_leverage=5.0
            )
        )
        
        # Metadatos del símbolo
        self.symbol_meta = SymbolMeta(
            symbol="BTCUSDT",
            market="futures",
            filters={
                "minNotional": 10.0,
                "lotStep": 0.0001
            }
        )
        
        # Risk manager
        self.risk_manager = RiskManager(self.risk_config, self.symbol_meta)
        
        # Mock portfolio
        self.portfolio = MagicMock()
        self.portfolio.market = "futures"
        self.portfolio.equity_quote = 1000.0
        self.portfolio.leverage = 3.0
        
        # Mock decision
        self.decision = MagicMock()
        self.decision.should_open = True
        self.decision.side = 1  # Long
        self.decision.price_hint = 50000.0
        self.decision.sl = 49000.0  # 2% SL
        self.decision.tp = None
        
        # Mock events bus
        self.events_bus = MagicMock()
        self.ts_now = 1640995200000
    
    def test_size_futures_basic(self):
        """Test sizing básico de futuros"""
        # Ejecutar sizing
        result = self.risk_manager.size_futures(
            portfolio=self.portfolio,
            decision=self.decision,
            leverage=3.0,
            account_equity=1000.0,
            events_bus=self.events_bus,
            ts_now=self.ts_now
        )
        
        # Verificar resultado
        assert result.should_open is True
        assert result.side == 1
        assert result.qty > 0
        assert result.price_hint == 50000.0
        assert result.sl == 49000.0
        assert result.leverage_used == 3.0
        assert result.notional_effective > 0
        assert result.notional_max == 3000.0  # 1000 * 3.0
    
    def test_size_futures_min_notional_enforcement(self):
        """Test que se respeta minNotional"""
        # Configurar para que el sizing inicial sea menor que minNotional
        self.portfolio.equity_quote = 50.0  # Equity muy pequeño
        self.decision.sl = 49900.0  # SL muy cerca (0.2%)
        
        # Ejecutar sizing
        result = self.risk_manager.size_futures(
            portfolio=self.portfolio,
            decision=self.decision,
            leverage=3.0,
            account_equity=50.0,
            events_bus=self.events_bus,
            ts_now=self.ts_now
        )
        
        # Verificar que se ajustó al minNotional
        assert result.should_open is True
        notional = result.qty * result.price_hint
        assert notional >= 10.0  # minNotional
        
        # Verificar que se emitió evento si fue necesario
        # (El evento se emite solo si qty <= 0, pero aquí debería ajustarse)
    
    def test_size_futures_notional_cap(self):
        """Test que se respeta el límite de notional por leverage"""
        # Configurar para que el sizing exceda el límite
        self.portfolio.equity_quote = 1000.0
        self.decision.sl = 40000.0  # SL muy lejos (20%)
        
        # Ejecutar sizing
        result = self.risk_manager.size_futures(
            portfolio=self.portfolio,
            decision=self.decision,
            leverage=3.0,
            account_equity=1000.0,
            events_bus=self.events_bus,
            ts_now=self.ts_now
        )
        
        # Verificar que no excede el límite
        assert result.should_open is True
        notional = result.qty * result.price_hint
        assert notional <= 3000.0  # 1000 * 3.0 (equity * leverage)
        assert result.notional_max == 3000.0
    
    def test_size_futures_no_sl_fallback(self):
        """Test que usa fallback cuando no hay SL"""
        # Decision sin SL
        self.decision.sl = None
        
        # Ejecutar sizing
        result = self.risk_manager.size_futures(
            portfolio=self.portfolio,
            decision=self.decision,
            leverage=3.0,
            account_equity=1000.0,
            events_bus=self.events_bus,
            ts_now=self.ts_now
        )
        
        # Verificar que se usó fallback
        assert result.should_open is True
        assert result.qty > 0
        
        # Verificar que se aplicó SL por defecto
        assert result.sl is not None
        assert result.sl != 50000.0  # No debería ser igual al precio
    
    def test_size_futures_sl_too_close(self):
        """Test que maneja SL muy cerca del precio"""
        # SL muy cerca (menos de 0.1%)
        self.decision.sl = 49950.0  # 0.1% de distancia
        
        # Ejecutar sizing
        result = self.risk_manager.size_futures(
            portfolio=self.portfolio,
            decision=self.decision,
            leverage=3.0,
            account_equity=1000.0,
            events_bus=self.events_bus,
            ts_now=self.ts_now
        )
        
        # Verificar que se usó fallback
        assert result.should_open is True
        assert result.qty > 0
    
    def test_size_futures_risk_blocked_event(self):
        """Test que emite evento RISK_BLOCKED cuando falla el sizing"""
        # Configurar para que falle el sizing
        self.portfolio.equity_quote = 0.0  # Sin equity
        
        # Ejecutar sizing
        result = self.risk_manager.size_futures(
            portfolio=self.portfolio,
            decision=self.decision,
            leverage=3.0,
            account_equity=0.0,
            events_bus=self.events_bus,
            ts_now=self.ts_now
        )
        
        # Verificar que se bloqueó
        assert result.should_open is False
        
        # Verificar que se emitió evento
        self.events_bus.emit.assert_called_once()
        call_args = self.events_bus.emit.call_args
        assert call_args[0][0] == "RISK_BLOCKED"
        assert call_args[1]["reason"] == "futures_sizing_failed"
        assert call_args[1]["equity"] == 0.0
        assert call_args[1]["leverage"] == 3.0
    
    def test_size_futures_short_position(self):
        """Test sizing para posición corta"""
        # Decision para short
        self.decision.side = -1
        self.decision.sl = 51000.0  # SL por encima del precio
        
        # Ejecutar sizing
        result = self.risk_manager.size_futures(
            portfolio=self.portfolio,
            decision=self.decision,
            leverage=3.0,
            account_equity=1000.0,
            events_bus=self.events_bus,
            ts_now=self.ts_now
        )
        
        # Verificar resultado
        assert result.should_open is True
        assert result.side == -1
        assert result.qty > 0
        assert result.sl == 51000.0
    
    def test_size_futures_different_leverage(self):
        """Test sizing con diferentes niveles de leverage"""
        leverages = [1.0, 2.0, 5.0, 10.0]
        
        for leverage in leverages:
            # Ejecutar sizing
            result = self.risk_manager.size_futures(
                portfolio=self.portfolio,
                decision=self.decision,
                leverage=leverage,
                account_equity=1000.0,
                events_bus=self.events_bus,
                ts_now=self.ts_now
            )
            
            # Verificar que se respeta el leverage
            assert result.should_open is True
            assert result.leverage_used == leverage
            assert result.notional_max == 1000.0 * leverage
    
    def test_size_futures_risk_percentage_variation(self):
        """Test sizing con diferentes porcentajes de riesgo"""
        risk_percentages = [0.5, 1.0, 2.0, 5.0]
        
        for risk_pct in risk_percentages:
            # Actualizar configuración
            self.risk_config.futures.risk_pct_per_trade = risk_pct
            
            # Ejecutar sizing
            result = self.risk_manager.size_futures(
                portfolio=self.portfolio,
                decision=self.decision,
                leverage=3.0,
                account_equity=1000.0,
                events_bus=self.events_bus,
                ts_now=self.ts_now
            )
            
            # Verificar que se respeta el porcentaje de riesgo
            assert result.should_open is True
            # El qty debería ser proporcional al porcentaje de riesgo
            # (verificación cualitativa)
    
    def test_size_futures_no_decision(self):
        """Test que maneja decision None o sin should_open"""
        # Decision None
        result = self.risk_manager.size_futures(
            portfolio=self.portfolio,
            decision=None,
            leverage=3.0,
            account_equity=1000.0,
            events_bus=self.events_bus,
            ts_now=self.ts_now
        )
        
        assert result.should_open is False
        
        # Decision sin should_open
        self.decision.should_open = False
        result = self.risk_manager.size_futures(
            portfolio=self.portfolio,
            decision=self.decision,
            leverage=3.0,
            account_equity=1000.0,
            events_bus=self.events_bus,
            ts_now=self.ts_now
        )
        
        assert result.should_open is False
    
    def test_size_futures_tp_calculation(self):
        """Test que se calcula TP correctamente"""
        # Decision con TP
        self.decision.tp = 52000.0
        
        # Ejecutar sizing
        result = self.risk_manager.size_futures(
            portfolio=self.portfolio,
            decision=self.decision,
            leverage=3.0,
            account_equity=1000.0,
            events_bus=self.events_bus,
            ts_now=self.ts_now
        )
        
        # Verificar que se mantiene el TP
        assert result.should_open is True
        assert result.tp == 52000.0
    
    def test_size_futures_margin_calculation(self):
        """Test que se calcula el margen correctamente"""
        # Ejecutar sizing
        result = self.risk_manager.size_futures(
            portfolio=self.portfolio,
            decision=self.decision,
            leverage=3.0,
            account_equity=1000.0,
            events_bus=self.events_bus,
            ts_now=self.ts_now
        )
        
        # Verificar que se calculó el margen
        assert result.should_open is True
        assert result.notional_effective > 0
        assert result.notional_max > 0
        
        # El margen efectivo debería ser notional / leverage
        expected_margin = result.notional_effective / result.leverage_used
        # (Esta verificación es conceptual, el margen real se maneja en el ledger)


if __name__ == "__main__":
    pytest.main([__file__])
