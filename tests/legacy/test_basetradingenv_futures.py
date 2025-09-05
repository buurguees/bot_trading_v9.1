#!/usr/bin/env python3
"""
Script de prueba exhaustivo para verificar que BaseTradingEnv soporte completamente:
- Apertura de posiciones LONG y SHORT en modo futures
- Uso de leverage_override desde gym_wrapper
- Limpieza automática de leverage_override tras cada step
- Integración de size_futures en RiskManager
"""

import json
from pathlib import Path
from unittest.mock import Mock, MagicMock

def test_long_short_positions():
    """Prueba que BaseTradingEnv soporte apertura de posiciones LONG y SHORT en futuros"""
    try:
        print("🧪 Probando apertura de posiciones LONG y SHORT en futuros...")
        
        from base_env.base_env import BaseTradingEnv
        from base_env.config.models import EnvConfig, SymbolMeta
        
        # Crear configuración de futuros
        cfg = EnvConfig(
            market="futures",
            mode="train_futures",
            leverage=5.0,
            symbol_meta=SymbolMeta(
                symbol="BTCUSDT",
                market="futures",
                filters={"minNotional": 5.0, "lotStep": 0.0001}
            )
        )
        
        # Crear mocks para broker y OMS
        mock_broker = Mock()
        mock_broker.now_ts.return_value = 1234567890
        mock_broker.get_price.return_value = 30000.0
        
        mock_oms = Mock()
        mock_oms.open.return_value = {
            "side": "LONG",
            "qty": 0.1,
            "price": 30000.0,
            "sl": 27000.0,
            "tp": 33000.0
        }
        
        # Crear entorno
        env = BaseTradingEnv(cfg, mock_broker, mock_oms, initial_cash=10000.0)
        
        print(f"   ✅ BaseTradingEnv creado en modo futures")
        print(f"   📊 Configuración:")
        print(f"      market: {env.cfg.market}")
        print(f"      mode: {env.cfg.mode}")
        print(f"      leverage: {env.cfg.leverage}")
        
        # Probar apertura LONG
        print(f"\n   🧪 Probando apertura LONG...")
        env.set_action_override(action=3, leverage_override=5.0, leverage_index=3)  # force_long
        
        obs, reward, done, info = env.step()
        
        # Verificar que se llamó a OMS.open con LONG
        mock_oms.open.assert_called()
        call_args = mock_oms.open.call_args
        side = call_args[0][0]  # Primer argumento posicional
        
        print(f"      Acción ejecutada: {side}")
        print(f"      Cantidad: {call_args[0][1]}")
        print(f"      Precio: {call_args[0][2]}")
        print(f"      SL: {call_args[0][3]}")
        print(f"      TP: {call_args[0][4]}")
        
        if side == "LONG":
            print(f"   ✅ Apertura LONG ejecutada correctamente")
        else:
            print(f"   ❌ Apertura LONG falló: {side}")
            return False
        
        # Reset para probar SHORT
        env.reset()
        mock_oms.open.reset_mock()
        
        # Probar apertura SHORT
        print(f"\n   🧪 Probando apertura SHORT...")
        env.set_action_override(action=4, leverage_override=10.0, leverage_index=8)  # force_short
        
        obs, reward, done, info = env.step()
        
        # Verificar que se llamó a OMS.open con SHORT
        mock_oms.open.assert_called()
        call_args = mock_oms.open.call_args
        side = call_args[0][0]
        
        print(f"      Acción ejecutada: {side}")
        print(f"      Cantidad: {call_args[0][1]}")
        print(f"      Precio: {call_args[0][2]}")
        
        if side == "SHORT":
            print(f"   ✅ Apertura SHORT ejecutada correctamente")
        else:
            print(f"   ❌ Apertura SHORT falló: {side}")
            return False
        
        return True
        
    except Exception as e:
        print(f"❌ Error en prueba de posiciones LONG/SHORT: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_leverage_override_integration():
    """Prueba la integración completa de leverage_override desde gym_wrapper"""
    try:
        print("\n🧪 Probando integración de leverage_override desde gym_wrapper...")
        
        from base_env.base_env import BaseTradingEnv
        from base_env.config.models import EnvConfig, SymbolMeta
        
        # Crear configuración
        cfg = EnvConfig(
            market="futures",
            mode="train_futures",
            leverage=5.0,
            symbol_meta=SymbolMeta(
                symbol="BTCUSDT",
                market="futures",
                filters={"minNotional": 5.0, "lotStep": 0.0001}
            )
        )
        
        # Crear mocks
        mock_broker = Mock()
        mock_broker.now_ts.return_value = 1234567890
        mock_broker.get_price.return_value = 30000.0
        
        mock_oms = Mock()
        mock_oms.open.return_value = {
            "side": "LONG",
            "qty": 0.1,
            "price": 30000.0,
            "sl": 27000.0,
            "tp": 33000.0
        }
        
        # Crear entorno
        env = BaseTradingEnv(cfg, mock_broker, mock_oms, initial_cash=10000.0)
        
        print(f"   ✅ Entorno creado")
        
        # Simular diferentes valores de leverage desde gym_wrapper
        test_leverages = [
            {"leverage": 2.0, "index": 0, "expected": 2.0},
            {"leverage": 5.0, "index": 3, "expected": 5.0},
            {"leverage": 10.0, "index": 8, "expected": 10.0},
            {"leverage": 25.0, "index": 23, "expected": 25.0}
        ]
        
        for i, test_case in enumerate(test_leverages):
            print(f"\n   🧪 Caso {i+1}: Leverage {test_case['leverage']}x (índice {test_case['index']})")
            
            # Reset del entorno
            env.reset()
            mock_oms.open.reset_mock()
            
            # Inyectar leverage_override
            env.set_action_override(
                action=3,  # force_long
                leverage_override=test_case["leverage"],
                leverage_index=test_case["index"]
            )
            
            # Ejecutar step
            obs, reward, done, info = env.step()
            
            # Verificar que se usó el leverage correcto
            # Esto se puede verificar indirectamente a través de los eventos
            events = info.get("events", [])
            open_events = [e for e in events if e.get("kind") == "OPEN"]
            
            if open_events:
                event = open_events[0]
                leverage_used = event.get("leverage_used")
                leverage_index = event.get("leverage_index")
                
                print(f"      Leverage usado: {leverage_used}")
                print(f"      Índice leverage: {leverage_index}")
                print(f"      Esperado: {test_case['expected']}")
                
                if leverage_used == test_case["expected"]:
                    print(f"   ✅ Leverage {test_case['leverage']}x aplicado correctamente")
                else:
                    print(f"   ❌ Leverage incorrecto: {leverage_used} vs {test_case['expected']}")
                    return False
            else:
                print(f"   ❌ No se encontraron eventos OPEN")
                return False
        
        return True
        
    except Exception as e:
        print(f"❌ Error en integración de leverage_override: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_leverage_override_cleanup():
    """Prueba la limpieza automática de leverage_override tras cada step"""
    try:
        print("\n🧪 Probando limpieza automática de leverage_override...")
        
        from base_env.base_env import BaseTradingEnv
        from base_env.config.models import EnvConfig, SymbolMeta
        
        # Crear configuración
        cfg = EnvConfig(
            market="futures",
            mode="train_futures",
            leverage=5.0,
            symbol_meta=SymbolMeta(
                symbol="BTCUSDT",
                market="futures",
                filters={"minNotional": 5.0, "lotStep": 0.0001}
            )
        )
        
        # Crear mocks
        mock_broker = Mock()
        mock_broker.now_ts.return_value = 1234567890
        mock_broker.get_price.return_value = 30000.0
        
        mock_oms = Mock()
        mock_oms.open.return_value = {
            "side": "LONG",
            "qty": 0.1,
            "price": 30000.0,
            "sl": 27000.0,
            "tp": 33000.0
        }
        
        # Crear entorno
        env = BaseTradingEnv(cfg, mock_broker, mock_oms, initial_cash=10000.0)
        
        print(f"   ✅ Entorno creado")
        
        # Verificar estado inicial
        print(f"   📊 Estado inicial:")
        print(f"      _action_override: {env._action_override}")
        print(f"      _leverage_override: {env._leverage_override}")
        print(f"      _leverage_index: {env._leverage_index}")
        
        # Inyectar override
        env.set_action_override(action=3, leverage_override=10.0, leverage_index=8)
        
        print(f"\n   📊 Después de set_action_override:")
        print(f"      _action_override: {env._action_override}")
        print(f"      _leverage_override: {env._leverage_override}")
        print(f"      _leverage_index: {env._leverage_index}")
        
        # Verificar que se establecieron
        if env._action_override != 3 or env._leverage_override != 10.0 or env._leverage_index != 8:
            print(f"   ❌ Override no se estableció correctamente")
            return False
        
        # Ejecutar step
        obs, reward, done, info = env.step()
        
        print(f"\n   📊 Después de step():")
        print(f"      _action_override: {env._action_override}")
        print(f"      _leverage_override: {env._leverage_override}")
        print(f"      _leverage_index: {env._leverage_index}")
        
        # Verificar que se limpiaron
        if env._action_override is not None:
            print(f"   ❌ _action_override no se limpió: {env._action_override}")
            return False
        
        if env._leverage_override is not None:
            print(f"   ❌ _leverage_override no se limpió: {env._leverage_override}")
            return False
        
        print(f"   ✅ Limpieza automática funcionando correctamente")
        
        # Probar múltiples steps para asegurar que no hay residuos
        print(f"\n   🧪 Probando múltiples steps...")
        
        for i in range(3):
            # Inyectar override
            env.set_action_override(action=3, leverage_override=15.0, leverage_index=13)
            
            # Ejecutar step
            obs, reward, done, info = env.step()
            
            # Verificar limpieza
            if env._action_override is not None or env._leverage_override is not None:
                print(f"   ❌ Override no se limpió en step {i+1}")
                return False
        
        print(f"   ✅ Limpieza automática consistente en múltiples steps")
        
        return True
        
    except Exception as e:
        print(f"❌ Error en prueba de limpieza de leverage_override: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_size_futures_integration():
    """Prueba la integración de size_futures en RiskManager"""
    try:
        print("\n🧪 Probando integración de size_futures en RiskManager...")
        
        from base_env.risk.manager import RiskManager
        from base_env.config.models import RiskConfig, SymbolMeta
        
        # Crear configuración
        cfg = RiskConfig()
        symbol_meta = SymbolMeta(
            symbol="BTCUSDT",
            market="futures",
            filters={"minNotional": 5.0, "lotStep": 0.0001}
        )
        
        # Crear RiskManager
        risk_manager = RiskManager(cfg, symbol_meta)
        
        print(f"   ✅ RiskManager creado")
        
        # Crear mock decision
        class MockDecision:
            def __init__(self, should_open=True, side=1, price_hint=30000.0, sl=27000.0, tp=33000.0):
                self.should_open = should_open
                self.side = side
                self.price_hint = price_hint
                self.sl = sl
                self.tp = tp
        
        # Crear mock portfolio
        class MockPortfolio:
            def __init__(self, equity=10000.0):
                self.equity_quote = equity
                self.market = "futures"
        
        # Probar diferentes escenarios de leverage
        test_cases = [
            {"leverage": 2.0, "equity": 10000.0, "expected_notional_max": 20000.0},
            {"leverage": 5.0, "equity": 10000.0, "expected_notional_max": 50000.0},
            {"leverage": 10.0, "equity": 10000.0, "expected_notional_max": 100000.0},
            {"leverage": 25.0, "equity": 10000.0, "expected_notional_max": 250000.0}
        ]
        
        for i, test_case in enumerate(test_cases):
            print(f"\n   🧪 Caso {i+1}: Leverage {test_case['leverage']}x")
            
            # Crear decision y portfolio
            decision = MockDecision()
            portfolio = MockPortfolio(test_case["equity"])
            
            # Llamar a size_futures
            sized = risk_manager.size_futures(
                portfolio=portfolio,
                decision=decision,
                leverage=test_case["leverage"],
                account_equity=test_case["equity"]
            )
            
            # Verificar resultado
            print(f"      should_open: {sized.should_open}")
            print(f"      side: {sized.side}")
            print(f"      qty: {sized.qty}")
            print(f"      leverage_used: {sized.leverage_used}")
            print(f"      notional_effective: {sized.notional_effective}")
            print(f"      notional_max: {sized.notional_max}")
            
            # Verificar que se devolvió la información correcta
            if not sized.should_open:
                print(f"   ❌ should_open es False")
                return False
            
            if sized.leverage_used != test_case["leverage"]:
                print(f"   ❌ leverage_used incorrecto: {sized.leverage_used}")
                return False
            
            if sized.notional_max != test_case["expected_notional_max"]:
                print(f"   ❌ notional_max incorrecto: {sized.notional_max}")
                return False
            
            if sized.notional_effective <= 0:
                print(f"   ❌ notional_effective debe ser > 0: {sized.notional_effective}")
                return False
            
            print(f"   ✅ Caso {i+1} funcionando correctamente")
        
        return True
        
    except Exception as e:
        print(f"❌ Error en integración de size_futures: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_complete_futures_workflow():
    """Prueba el flujo completo de futuros desde gym_wrapper hasta BaseTradingEnv"""
    try:
        print("\n🧪 Probando flujo completo de futuros...")
        
        from base_env.base_env import BaseTradingEnv
        from base_env.config.models import EnvConfig, SymbolMeta
        
        # Crear configuración
        cfg = EnvConfig(
            market="futures",
            mode="train_futures",
            leverage=5.0,
            symbol_meta=SymbolMeta(
                symbol="BTCUSDT",
                market="futures",
                filters={"minNotional": 5.0, "lotStep": 0.0001}
            )
        )
        
        # Crear mocks
        mock_broker = Mock()
        mock_broker.now_ts.return_value = 1234567890
        mock_broker.get_price.return_value = 30000.0
        
        mock_oms = Mock()
        mock_oms.open.return_value = {
            "side": "LONG",
            "qty": 0.1,
            "price": 30000.0,
            "sl": 27000.0,
            "tp": 33000.0
        }
        
        # Crear entorno
        env = BaseTradingEnv(cfg, mock_broker, mock_oms, initial_cash=10000.0)
        
        print(f"   ✅ Entorno creado")
        
        # Simular flujo completo desde gym_wrapper
        print(f"\n   🧪 Simulando flujo desde gym_wrapper...")
        
        # 1. Gym wrapper inyecta acción y leverage
        action = [3, 8]  # MultiDiscrete: force_long + leverage 10x
        trade_action = action[0]
        lev_idx = action[1]
        leverage = 2.0 + lev_idx * 1.0  # 2.0 + 8 * 1.0 = 10.0
        
        print(f"      Acción del agente: {action}")
        print(f"      Trade action: {trade_action} (force_long)")
        print(f"      Leverage index: {lev_idx}")
        print(f"      Leverage calculado: {leverage}x")
        
        # 2. Inyectar en BaseTradingEnv
        env.set_action_override(
            action=trade_action,
            leverage_override=leverage,
            leverage_index=lev_idx
        )
        
        # 3. Ejecutar step
        obs, reward, done, info = env.step()
        
        # 4. Verificar resultado
        events = info.get("events", [])
        open_events = [e for e in events if e.get("kind") == "OPEN"]
        
        if open_events:
            event = open_events[0]
            print(f"\n   📊 Evento OPEN generado:")
            print(f"      side: {event.get('side')}")
            print(f"      qty: {event.get('qty')}")
            print(f"      price: {event.get('price')}")
            print(f"      leverage_used: {event.get('leverage_used')}")
            print(f"      notional_effective: {event.get('notional_effective')}")
            print(f"      notional_max: {event.get('notional_max')}")
            print(f"      action_taken: {event.get('action_taken')}")
            print(f"      leverage_index: {event.get('leverage_index')}")
            
            # Verificar que toda la información está presente
            required_fields = [
                "leverage_used", "notional_effective", "notional_max",
                "action_taken", "leverage_index"
            ]
            
            missing_fields = []
            for field in required_fields:
                if event.get(field) is None:
                    missing_fields.append(field)
            
            if missing_fields:
                print(f"   ❌ Campos faltantes: {missing_fields}")
                return False
            
            print(f"   ✅ Flujo completo funcionando correctamente")
            return True
        else:
            print(f"   ❌ No se generaron eventos OPEN")
            return False
        
    except Exception as e:
        print(f"❌ Error en flujo completo de futuros: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("🚀 Verificando soporte completo de futuros en BaseTradingEnv...\n")
    
    tests = [
        test_long_short_positions,
        test_leverage_override_integration,
        test_leverage_override_cleanup,
        test_size_futures_integration,
        test_complete_futures_workflow
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
        print()
    
    print(f"📊 Resultados: {passed}/{total} pruebas pasaron")
    
    if passed == total:
        print("🎉 ¡BaseTradingEnv soporta completamente todas las funcionalidades de futuros!")
        print("\n📋 Funcionalidades verificadas:")
        print("   ✅ Apertura de posiciones LONG y SHORT en modo futures")
        print("   ✅ Uso de leverage_override desde gym_wrapper")
        print("   ✅ Limpieza automática de leverage_override tras cada step")
        print("   ✅ Integración de size_futures en RiskManager")
        print("   ✅ Flujo completo de futuros funcionando")
        print("\n🚀 El sistema está completamente preparado para trading de futuros con leverage dinámico!")
        print("   - Soporte nativo para LONG/SHORT")
        print("   - Integración perfecta con gym_wrapper")
        print("   - Gestión automática de estado")
        print("   - Sizing inteligente para futuros")
    else:
        print("⚠️  Algunas pruebas fallaron. Revisa los errores arriba.")
