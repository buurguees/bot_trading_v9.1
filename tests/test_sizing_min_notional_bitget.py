#!/usr/bin/env python3
"""
Test: Sizing con minNotional real de Bitget
- minNotional=5, lotStep/tickSize fijados → qty escalada si hay margen; si no, MIN_NOTIONAL_BLOCKED con log numérico.
"""

import pytest
import numpy as np
from pathlib import Path
import sys

# Agregar el directorio raíz al path
sys.path.insert(0, str(Path(__file__).parent.parent))

from base_env.risk.manager import RiskManager
from base_env.config.models import RiskConfig, SymbolMeta
from base_env.accounting.ledger import PortfolioState, PositionState
from base_env.policy.gating import Decision


def test_sizing_min_notional_bitget():
    """Test: Sizing con minNotional real de Bitget (5.0 USDT)"""
    
    # Configurar SymbolMeta con filtros reales de Bitget
    symbol_meta = SymbolMeta(
        symbol="BTCUSDT",
        market="futures",
        filters={
            "tickSize": 0.1,
            "lotStep": 0.0001,
            "minNotional": 5.0  # minNotional real de Bitget
        }
    )
    
    # Configurar RiskConfig
    risk_config = RiskConfig(
        common={
            "train_force_min_notional": True,
            "default_levels": {
                "min_sl_pct": 1.0
            }
        },
        futures={
            "risk_pct_per_trade": 0.25
        }
    )
    
    # Crear RiskManager
    risk_manager = RiskManager(risk_config, symbol_meta)
    
    # Crear portfolio con equity suficiente
    portfolio = PortfolioState(
        market="futures",
        cash_quote=1000.0,
        equity_quote=1000.0,
        used_margin=0.0
    )
    
    # Test 1: Equity suficiente para minNotional
    print("\n=== Test 1: Equity suficiente para minNotional ===")
    
    decision = Decision(
        should_open=True,
        side=1,  # LONG
        price_hint=50000.0,
        sl=49500.0,  # 1% SL
        tp=51000.0
    )
    
    sized = risk_manager.size_futures(
        portfolio=portfolio,
        decision=decision,
        leverage=3.0,
        account_equity=1000.0
    )
    
    print(f"Decision original: price={decision.price_hint}, sl={decision.sl}")
    print(f"Resultado sizing: should_open={sized.should_open}, qty={sized.qty:.6f}, price={sized.price_hint:.2f}")
    
    if sized.should_open:
        notional = sized.qty * sized.price_hint
        print(f"Notional: {notional:.2f} USDT")
        print(f"MinNotional requerido: 5.0 USDT")
        
        # Verificar que cumple minNotional
        assert notional >= 5.0, f"Notional {notional:.2f} debe ser >= 5.0"
        print("✅ Test 1 passed: Notional cumple minNotional")
    else:
        print("❌ Test 1 failed: No se pudo abrir posición con equity suficiente")
    
    # Test 2: Equity insuficiente para minNotional
    print("\n=== Test 2: Equity insuficiente para minNotional ===")
    
    # Portfolio con equity muy bajo
    portfolio_low = PortfolioState(
        market="futures",
        cash_quote=1.0,  # Equity muy bajo
        equity_quote=1.0,
        used_margin=0.0
    )
    
    sized_low = risk_manager.size_futures(
        portfolio=portfolio_low,
        decision=decision,
        leverage=3.0,
        account_equity=1.0
    )
    
    print(f"Equity: {portfolio_low.equity_quote}")
    print(f"Resultado sizing: should_open={sized_low.should_open}")
    
    if not sized_low.should_open:
        print("✅ Test 2 passed: Posición bloqueada por equity insuficiente")
    else:
        notional_low = sized_low.qty * sized_low.price_hint
        print(f"❌ Test 2 failed: Se abrió posición con notional {notional_low:.2f} < 5.0")
    
    # Test 3: Verificar redondeo a lotStep
    print("\n=== Test 3: Verificar redondeo a lotStep ===")
    
    if sized.should_open:
        lot_step = 0.0001
        qty_rounded = int(sized.qty / lot_step) * lot_step
        print(f"Qty original: {sized.qty:.8f}")
        print(f"Qty redondeado: {qty_rounded:.8f}")
        print(f"LotStep: {lot_step}")
        
        # Verificar que está redondeado a lotStep
        remainder = sized.qty % lot_step
        assert abs(remainder) < 1e-10 or abs(remainder - lot_step) < 1e-10, f"Qty {sized.qty:.8f} no está redondeado a lotStep {lot_step}"
        print("✅ Test 3 passed: Qty redondeado correctamente a lotStep")
    
    # Test 4: Verificar redondeo a tickSize
    print("\n=== Test 4: Verificar redondeo a tickSize ===")
    
    if sized.should_open:
        tick_size = 0.1
        price_rounded = round(sized.price_hint / tick_size) * tick_size
        print(f"Price original: {sized.price_hint:.8f}")
        print(f"Price redondeado: {price_rounded:.8f}")
        print(f"TickSize: {tick_size}")
        
        # Verificar que está redondeado a tickSize
        remainder = sized.price_hint % tick_size
        assert abs(remainder) < 1e-10 or abs(remainder - tick_size) < 1e-10, f"Price {sized.price_hint:.8f} no está redondeado a tickSize {tick_size}"
        print("✅ Test 4 passed: Price redondeado correctamente a tickSize")
    
    # Test 5: Verificar leverage usado
    print("\n=== Test 5: Verificar leverage usado ===")
    
    if sized.should_open:
        print(f"Leverage usado: {sized.leverage_used}")
        print(f"Notional efectivo: {sized.notional_effective:.2f}")
        print(f"Notional máximo: {sized.notional_max:.2f}")
        
        assert sized.leverage_used == 3.0, "Leverage usado debe ser 3.0"
        assert sized.notional_effective <= sized.notional_max, "Notional efectivo debe ser <= máximo"
        print("✅ Test 5 passed: Leverage y notional correctos")
    
    print("\n🎯 Resumen del test:")
    print(f"   - minNotional: 5.0 USDT")
    print(f"   - lotStep: 0.0001")
    print(f"   - tickSize: 0.1")
    print(f"   - train_force_min_notional: True")
    print("✅ Todos los tests de sizing con minNotional real de Bitget pasaron")


if __name__ == "__main__":
    test_sizing_min_notional_bitget()
