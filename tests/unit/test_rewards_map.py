#!/usr/bin/env python3
"""
Test: Mapeo de rewards
- Simular cierre por TP → +1.0
- Simular cierre por SL → -0.5
- Bankruptcy → -10
"""

import pytest
import numpy as np
from pathlib import Path
import sys

# Agregar el directorio raíz al path
sys.path.insert(0, str(Path(__file__).parent.parent))

from train_env.utilities.reward_shaper import RewardShaper


def test_rewards_map():
    """Test: Mapeo correcto de rewards según especificación"""
    
    # Crear RewardShaper con configuración básica
    reward_shaper = RewardShaper("config/rewards.yaml")
    
    # Configuración base para todos los tests
    base_obs = {
        "position": {"side": 0, "bars_held": 0},
        "portfolio": {"equity_quote": 1000.0}
    }
    base_reward = 0.0
    initial_balance = 1000.0
    target_balance = 1000000.0
    
    # Test 1: Cierre por TP debe dar +1.0
    print("\n=== Test 1: Cierre por TP → +1.0 ===")
    
    tp_events = [
        {
            "kind": "CLOSE",
            "roi_pct": 1.5,
            "r_multiple": 1.5,
            "risk_pct": 1.0,
            "entry_price": 50000.0,
            "exit_price": 50750.0,
            "sl": 49500.0,
            "tp": 50750.0
        },
        {"kind": "TP_HIT"}
    ]
    
    reward, components = reward_shaper.compute(
        obs=base_obs,
        base_reward=base_reward,
        events=tp_events,
        initial_balance=initial_balance,
        target_balance=target_balance
    )
    
    assert "tp_reward" in components, "Debe incluir tp_reward en componentes"
    assert components["tp_reward"] == 1.0, f"TP debe dar +1.0, pero dio {components['tp_reward']}"
    print(f"✅ TP reward: {components['tp_reward']}")
    
    # Test 2: Cierre por SL debe dar -0.5
    print("\n=== Test 2: Cierre por SL → -0.5 ===")
    
    sl_events = [
        {
            "kind": "CLOSE",
            "roi_pct": -1.0,
            "r_multiple": -1.0,
            "risk_pct": 1.0,
            "entry_price": 50000.0,
            "exit_price": 49500.0,
            "sl": 49500.0,
            "tp": 50750.0
        },
        {"kind": "SL_HIT"}
    ]
    
    reward, components = reward_shaper.compute(
        obs=base_obs,
        base_reward=base_reward,
        events=sl_events,
        initial_balance=initial_balance,
        target_balance=target_balance
    )
    
    assert "sl_reward" in components, "Debe incluir sl_reward en componentes"
    assert components["sl_reward"] == -0.5, f"SL debe dar -0.5, pero dio {components['sl_reward']}"
    print(f"✅ SL reward: {components['sl_reward']}")
    
    # Test 3: Bankruptcy debe dar -10
    print("\n=== Test 3: Bankruptcy → -10 ===")
    
    reward, components = reward_shaper.compute(
        obs=base_obs,
        base_reward=base_reward,
        events=[],
        initial_balance=initial_balance,
        target_balance=target_balance,
        bankruptcy_occurred=True
    )
    
    assert "bankruptcy_penalty" in components, "Debe incluir bankruptcy_penalty en componentes"
    assert components["bankruptcy_penalty"] == -10.0, f"Bankruptcy debe dar -10, pero dio {components['bankruptcy_penalty']}"
    print(f"✅ Bankruptcy penalty: {components['bankruptcy_penalty']}")
    
    # Test 4: ROI escalado proporcionalmente
    print("\n=== Test 4: ROI escalado proporcionalmente ===")
    
    # Cierre parcial (50% del camino entre SL y TP)
    partial_events = [
        {
            "kind": "CLOSE",
            "roi_pct": 0.5,
            "r_multiple": 0.5,
            "risk_pct": 1.0,
            "entry_price": 50000.0,
            "exit_price": 50125.0,  # 50% entre SL (49500) y TP (50750)
            "sl": 49500.0,
            "tp": 50750.0
        }
    ]
    
    reward, components = reward_shaper.compute(
        obs=base_obs,
        base_reward=base_reward,
        events=partial_events,
        initial_balance=initial_balance,
        target_balance=target_balance
    )
    
    assert "roi_scaled_reward" in components, "Debe incluir roi_scaled_reward en componentes"
    # 50% del camino = -0.5 + (1.5 * 0.5) = -0.5 + 0.75 = 0.25
    expected_reward = -0.5 + (1.5 * 0.5)
    assert abs(components["roi_scaled_reward"] - expected_reward) < 0.01, f"ROI escalado debe ser ~{expected_reward}, pero dio {components['roi_scaled_reward']}"
    print(f"✅ ROI escalado: {components['roi_scaled_reward']:.3f} (esperado: {expected_reward:.3f})")
    
    # Test 5: Penalty por inactividad
    print("\n=== Test 5: Penalty por inactividad ===")
    
    reward, components = reward_shaper.compute(
        obs=base_obs,
        base_reward=base_reward,
        events=[],
        initial_balance=initial_balance,
        target_balance=target_balance,
        steps_since_last_trade=150  # > 100 pasos
    )
    
    assert "inactivity_penalty" in components, "Debe incluir inactivity_penalty en componentes"
    # 150 pasos = 1 bloque de 100, penalty = -0.01 * 1 = -0.01
    expected_penalty = -0.01 * (150 // 100)
    assert components["inactivity_penalty"] == expected_penalty, f"Inactividad debe dar {expected_penalty}, pero dio {components['inactivity_penalty']}"
    print(f"✅ Inactividad penalty: {components['inactivity_penalty']}")
    
    # Test 6: Penalty por trades bloqueados
    print("\n=== Test 6: Penalty por trades bloqueados ===")
    
    blocked_events = [
        {"kind": "NO_SL_DISTANCE"},
        {"kind": "MIN_NOTIONAL_BLOCKED"}
    ]
    
    reward, components = reward_shaper.compute(
        obs=base_obs,
        base_reward=base_reward,
        events=blocked_events,
        initial_balance=initial_balance,
        target_balance=target_balance
    )
    
    assert "blocked_trade_penalty" in components, "Debe incluir blocked_trade_penalty en componentes"
    # 2 eventos bloqueados = -0.05 * 2 = -0.1
    expected_penalty = -0.05 * len(blocked_events)
    assert components["blocked_trade_penalty"] == expected_penalty, f"Trades bloqueados debe dar {expected_penalty}, pero dio {components['blocked_trade_penalty']}"
    print(f"✅ Trades bloqueados penalty: {components['blocked_trade_penalty']}")
    
    # Test 7: Bonus por duración de posición
    print("\n=== Test 7: Bonus por duración de posición ===")
    
    position_obs = {
        "position": {"side": 1, "bars_held": 20},  # 20 barras, múltiplo de 10
        "portfolio": {"equity_quote": 1100.0}  # Equity positivo
    }
    
    reward, components = reward_shaper.compute(
        obs=position_obs,
        base_reward=base_reward,
        events=[],
        initial_balance=initial_balance,
        target_balance=target_balance
    )
    
    assert "position_duration_bonus" in components, "Debe incluir position_duration_bonus en componentes"
    assert components["position_duration_bonus"] == 0.05, f"Duración de posición debe dar +0.05, pero dio {components['position_duration_bonus']}"
    print(f"✅ Duración de posición bonus: {components['position_duration_bonus']}")


if __name__ == "__main__":
    test_rewards_map()
