#!/usr/bin/env python3
"""
Test: Niveles SL/TP requeridos
- Intentar abrir con sl=None → bloqueado con NO_SL_DISTANCE
- Verificar que tp también sea válido
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


def test_trade_levels_required():
    """Test: Trades sin SL/TP válidos deben ser bloqueados"""
    
    # Configurar SymbolMeta con filtros reales
    symbol_meta = SymbolMeta(
        symbol="BTCUSDT",
        market="futures",
        filters={
            "tickSize": 0.1,
            "lotStep": 0.0001,
            "minNotional": 5.0
        }
    )
    
    # Configurar RiskConfig con niveles mínimos
    from base_env.config.models import DefaultLevelsConfig, RiskCommon, RiskFutures
    risk_config = RiskConfig(
        common=RiskCommon(
            default_levels=DefaultLevelsConfig(
                min_sl_pct=1.0,
                tp_r_multiple=1.5
            ),
            train_force_min_notional=True
        ),
        futures=RiskFutures(
            risk_pct_per_trade=0.25
        )
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
    
    # Test 1: SL=None debe ser bloqueado
    print("\n=== Test 1: SL=None debe ser bloqueado ===")
    
    decision = Decision(
        should_open=True,
        side=1,  # LONG
        price_hint=50000.0,
        sl=None,  # ← SL inválido
        tp=51000.0
    )
    
    sized = risk_manager.size_futures(
        portfolio=portfolio,
        decision=decision,
        leverage=3.0,
        account_equity=1000.0
    )
    
    assert not sized.should_open, f"Trade con SL=None debe ser bloqueado, pero should_open={sized.should_open}"
    print("✅ SL=None correctamente bloqueado")
    
    # Test 2: TP=None debe ser bloqueado
    print("\n=== Test 2: TP=None debe ser bloqueado ===")
    
    decision = Decision(
        should_open=True,
        side=1,  # LONG
        price_hint=50000.0,
        sl=49500.0,  # 1% SL
        tp=None  # ← TP inválido
    )
    
    sized = risk_manager.size_futures(
        portfolio=portfolio,
        decision=decision,
        leverage=3.0,
        account_equity=1000.0
    )
    
    assert not sized.should_open, f"Trade con TP=None debe ser bloqueado, pero should_open={sized.should_open}"
    print("✅ TP=None correctamente bloqueado")
    
    # Test 3: SL con distancia insuficiente debe ser bloqueado
    print("\n=== Test 3: SL con distancia insuficiente debe ser bloqueado ===")
    
    decision = Decision(
        should_open=True,
        side=1,  # LONG
        price_hint=50000.0,
        sl=49900.0,  # Solo 0.2% SL (menos del mínimo 1%)
        tp=51000.0
    )
    
    sized = risk_manager.size_futures(
        portfolio=portfolio,
        decision=decision,
        leverage=3.0,
        account_equity=1000.0
    )
    
    assert not sized.should_open, f"Trade con SL muy cerca debe ser bloqueado, pero should_open={sized.should_open}"
    print("✅ SL con distancia insuficiente correctamente bloqueado")
    
    # Test 4: TP con distancia insuficiente debe ser bloqueado
    print("\n=== Test 4: TP con distancia insuficiente debe ser bloqueado ===")
    
    decision = Decision(
        should_open=True,
        side=1,  # LONG
        price_hint=50000.0,
        sl=49500.0,  # 1% SL
        tp=50050.0  # Solo 0.1% TP (menos del mínimo 1.5% = 1% * 1.5)
    )
    
    sized = risk_manager.size_futures(
        portfolio=portfolio,
        decision=decision,
        leverage=3.0,
        account_equity=1000.0
    )
    
    assert not sized.should_open, f"Trade con TP muy cerca debe ser bloqueado, pero should_open={sized.should_open}"
    print("✅ TP con distancia insuficiente correctamente bloqueado")
    
    # Test 5: SL/TP válidos deben permitir el trade
    print("\n=== Test 5: SL/TP válidos deben permitir el trade ===")
    
    decision = Decision(
        should_open=True,
        side=1,  # LONG
        price_hint=50000.0,
        sl=49500.0,  # 1% SL
        tp=50750.0  # 1.5% TP (1% * 1.5)
    )
    
    sized = risk_manager.size_futures(
        portfolio=portfolio,
        decision=decision,
        leverage=3.0,
        account_equity=1000.0
    )
    
    assert sized.should_open, f"Trade con SL/TP válidos debe ser permitido, pero should_open={sized.should_open}"
    assert sized.qty > 0, f"Trade válido debe tener qty > 0, pero qty={sized.qty}"
    print("✅ SL/TP válidos correctamente permitidos")
    print(f"   Qty: {sized.qty:.6f}, Price: {sized.price_hint:.2f}, SL: {sized.sl:.2f}, TP: {sized.tp:.2f}")


if __name__ == "__main__":
    test_trade_levels_required()
