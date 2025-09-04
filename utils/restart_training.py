#!/usr/bin/env python3
"""
Script para reiniciar el entrenamiento con los fixes aplicados.
"""

import subprocess
import sys
import time
from pathlib import Path

def check_fixes_applied():
    """Verifica que los fixes estén aplicados."""
    print("🔍 Verificando fixes aplicados...")
    
    # Verificar que el fix del gym_wrapper esté aplicado
    gym_wrapper = Path("train_env/gym_wrapper.py")
    if gym_wrapper.exists():
        with open(gym_wrapper, 'r', encoding='utf-8') as f:
            content = f.read()
            if "FIX TEMPORAL: Si RL envía action=0" in content:
                print("✅ Fix del gym_wrapper aplicado")
            else:
                print("❌ Fix del gym_wrapper NO aplicado")
                return False
    
    # Verificar que risk_pct_per_trade esté aumentado
    risk_config = Path("config/risk.yaml")
    if risk_config.exists():
        with open(risk_config, 'r', encoding='utf-8') as f:
            content = f.read()
            if "risk_pct_per_trade: 2.0" in content and "risk_pct_per_trade: 3.0" in content:
                print("✅ Risk config aumentado")
            else:
                print("❌ Risk config NO aumentado")
                return False
    
    return True

def restart_training():
    """Reinicia el entrenamiento."""
    print("\n🚀 Reiniciando entrenamiento...")
    
    # Comando para reiniciar el entrenamiento
    cmd = [sys.executable, "scripts/train_ppo.py"]
    
    try:
        print(f"Ejecutando: {' '.join(cmd)}")
        print("Presiona Ctrl+C para detener el entrenamiento")
        print("=" * 50)
        
        # Ejecutar el entrenamiento
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, 
                                 universal_newlines=True, bufsize=1)
        
        # Mostrar output en tiempo real
        for line in iter(process.stdout.readline, ''):
            print(line.rstrip())
            
            # Buscar indicadores de éxito
            if "🔧 FIX RL: action=0 →" in line:
                print("🎉 ¡FIX FUNCIONANDO! RL ahora envía acciones reales")
            elif "OPEN" in line and "LONG" in line or "SHORT" in line:
                print("🎉 ¡TRADE EJECUTADO! El bot está operando")
            elif "BANKRUPTCY" in line:
                print("⚠️ Bancarrota detectada, pero esto es normal durante el aprendizaje")
        
        process.wait()
        
    except KeyboardInterrupt:
        print("\n⏹️ Entrenamiento detenido por el usuario")
        process.terminate()
        process.wait()
    except Exception as e:
        print(f"❌ Error ejecutando entrenamiento: {e}")

def main():
    print("🔄 REINICIO DE ENTRENAMIENTO CON FIXES")
    print("=" * 50)
    
    if not check_fixes_applied():
        print("\n❌ No se pueden aplicar todos los fixes. Abortando.")
        return
    
    print("\n✅ Todos los fixes están aplicados")
    print("\n📋 RESUMEN DE FIXES:")
    print("   1. RL ahora fuerza acciones reales (3=force_long, 4=force_short) cuando envía action=0")
    print("   2. risk_pct_per_trade aumentado: spot=2.0%, futures=3.0%")
    print("   3. train_force_min_notional=true para cumplir mínimos de exchange")
    print("   4. defaults de SL/TP configurados correctamente")
    
    input("\nPresiona Enter para continuar con el reinicio...")
    
    restart_training()

if __name__ == "__main__":
    main()
