"""
Test para validar que el sizing respeta los filtros de minNotional, lotStep, etc.
"""
import pytest
from base_env.risk.manager import RiskManager
from base_env.policy.gating import Decision
from base_env.config.models import RiskConfig, SymbolMeta


def test_sizing_with_min_notional_force():
    """Test que verifica que train_force_min_notional funciona correctamente."""
    
    # Configuración de riesgo
    risk_cfg = RiskConfig(
        common={
            "train_force_min_notional": True,
            "default_levels": {
                "use_atr": False,
                "min_sl_pct": 1.0,
                "tp_r_multiple": 1.5
            }
        },
        futures={
            "risk_pct_per_trade": 1.0
        }
    )
    
    # Metadatos del símbolo
    symbol_meta = SymbolMeta(
        symbol="BTCUSDT",
        filters={
            "minNotional": 5.0,
            "lotSizeFilter": {"stepSize": 0.001},
            "priceFilter": {"tickSize": 0.01}
        }
    )
    
    # Crear RiskManager
    risk_manager = RiskManager(risk_cfg, symbol_meta)
    
    # Portfolio mock
    class MockPortfolio:
        def __init__(self):
            self.equity_quote = 1000.0  # Equity bajo para forzar minNotional
            self.market = "futures"
    
    portfolio = MockPortfolio()
    
    # Decision de apertura
    decision = Decision(
        should_open=True,
        side=1,  # Long
        price_hint=50000.0,
        sl=49500.0,  # 1% SL
        tp=50750.0   # 1.5% TP
    )
    
    # Eventos mock
    class MockEventsBus:
        def emit(self, event_type, **kwargs):
            print(f"Event: {event_type} - {kwargs}")
    
    events_bus = MockEventsBus()
    
    # Ejecutar sizing
    sized = risk_manager.size_futures(
        portfolio=portfolio,
        decision=decision,
        leverage=2.0,
        account_equity=1000.0,
        obs={},
        events_bus=events_bus,
        ts_now=1640995200000
    )
    
    # Verificar que se aplicó el sizing
    assert sized.should_open == True
    assert sized.qty > 0
    assert sized.side == 1
    
    # Verificar que el notional cumple minNotional
    notional = sized.qty * sized.price_hint
    assert notional >= 5.0  # minNotional
    
    # Verificar que qty respeta lotStep
    lot_step = 0.001
    assert (sized.qty % lot_step) < 1e-10  # Debe ser múltiplo de lotStep
    
    # Verificar que price_hint respeta tickSize
    tick_size = 0.01
    assert (sized.price_hint % tick_size) < 1e-10  # Debe ser múltiplo de tickSize


def test_sizing_blocked_by_min_notional():
    """Test que verifica que se bloquea cuando no se puede cumplir minNotional."""
    
    # Configuración de riesgo
    risk_cfg = RiskConfig(
        common={
            "train_force_min_notional": True,
            "default_levels": {
                "use_atr": False,
                "min_sl_pct": 1.0,
                "tp_r_multiple": 1.5
            }
        },
        futures={
            "risk_pct_per_trade": 0.1  # Riesgo muy bajo
        }
    )
    
    # Metadatos del símbolo con minNotional alto
    symbol_meta = SymbolMeta(
        symbol="BTCUSDT",
        filters={
            "minNotional": 100.0,  # MinNotional muy alto
            "lotSizeFilter": {"stepSize": 0.001},
            "priceFilter": {"tickSize": 0.01}
        }
    )
    
    # Crear RiskManager
    risk_manager = RiskManager(risk_cfg, symbol_meta)
    
    # Portfolio mock con equity muy bajo
    class MockPortfolio:
        def __init__(self):
            self.equity_quote = 10.0  # Equity muy bajo
            self.market = "futures"
    
    portfolio = MockPortfolio()
    
    # Decision de apertura
    decision = Decision(
        should_open=True,
        side=1,  # Long
        price_hint=50000.0,
        sl=49500.0,  # 1% SL
        tp=50750.0   # 1.5% TP
    )
    
    # Eventos mock
    class MockEventsBus:
        def __init__(self):
            self.events = []
        
        def emit(self, event_type, **kwargs):
            self.events.append((event_type, kwargs))
    
    events_bus = MockEventsBus()
    
    # Ejecutar sizing
    sized = risk_manager.size_futures(
        portfolio=portfolio,
        decision=decision,
        leverage=1.0,  # Sin leverage
        account_equity=10.0,
        obs={},
        events_bus=events_bus,
        ts_now=1640995200000
    )
    
    # Verificar que se bloqueó
    assert sized.should_open == False
    assert sized.qty == 0.0
    
    # Verificar que se emitió evento de bloqueo
    assert len(events_bus.events) > 0
    event_types = [event[0] for event in events_bus.events]
    assert "MIN_NOTIONAL_BLOCKED" in event_types or "RISK_BLOCKED" in event_types


if __name__ == "__main__":
    test_sizing_with_min_notional_force()
    test_sizing_blocked_by_min_notional()
    print("✅ Tests de sizing con filtros pasaron correctamente")
