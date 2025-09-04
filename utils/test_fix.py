#!/usr/bin/env python3
"""
Script para probar si el fix está funcionando.
"""

import sys
import os
sys.path.append('.')

def test_fix():
    """Prueba si el fix está funcionando."""
    
    print("🧪 PROBANDO EL FIX")
    print("=" * 30)
    
    try:
        # Importar el gym wrapper
        from train_env.gym_wrapper import TradingGymWrapper
        from base_env.base_env import BaseTradingEnv
        
        print("✅ Imports exitosos")
        
        # Crear un entorno base mock
        class MockBaseEnv:
            def __init__(self):
                self._action_override = None
                self._leverage_override = None
                self._leverage_index = None
            
            def set_action_override(self, action, leverage_override=None, leverage_index=None):
                print(f"🎯 set_action_override llamado con: action={action}, leverage={leverage_override}")
                self._action_override = action
                self._leverage_override = leverage_override
                self._leverage_index = leverage_index
            
            def step(self):
                return {}, 0.0, False, {}
            
            def reset(self):
                return {}
        
        # Crear el wrapper
        mock_env = MockBaseEnv()
        wrapper = TradingGymWrapper(mock_env, "config/rewards.yaml", ["1m"])
        
        print("✅ Wrapper creado")
        
        # Probar con action=0 (debería activar el fix)
        print("\n🧪 Probando con action=0 (debería activar el fix):")
        wrapper.step(0)
        
        # Probar con action=3 (no debería activar el fix)
        print("\n🧪 Probando con action=3 (no debería activar el fix):")
        wrapper.step(3)
        
        print("\n✅ Test completado")
        
    except Exception as e:
        print(f"❌ Error en el test: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_fix()
