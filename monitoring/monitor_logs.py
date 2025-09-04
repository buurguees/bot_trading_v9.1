#!/usr/bin/env python3
"""
Script para monitorear los logs del entrenamiento en tiempo real.
"""

import time
import subprocess
import sys

def monitor_training_logs():
    """Monitorea los logs del entrenamiento en tiempo real."""
    
    print("👁️ MONITOREANDO LOGS DEL ENTRENAMIENTO")
    print("=" * 50)
    print("Presiona Ctrl+C para detener")
    print()
    
    try:
        # Ejecutar el entrenamiento y capturar output
        process = subprocess.Popen(
            [sys.executable, "scripts/train_ppo.py"],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True,
            bufsize=1
        )
        
        # Monitorear output en tiempo real
        for line in iter(process.stdout.readline, ''):
            print(line.rstrip())
            
            # Buscar indicadores clave
            if "🔧 FIX RL: action=0 →" in line:
                print("🎉 ¡FIX FUNCIONANDO! RL ahora envía acciones reales")
            elif "🎯 OPEN_ATTEMPT:" in line:
                print("🎉 ¡INTENTO DE APERTURA! El bot está intentando operar")
            elif "🔧 DEFAULT_LEVELS_APPLIED:" in line:
                print("🎉 ¡SL/TP POR DEFECTO APLICADOS!")
            elif "FORZANDO_MIN_NOTIONAL" in line:
                print("🎉 ¡MIN NOTIONAL FORZADO!")
            elif "OPEN" in line and ("LONG" in line or "SHORT" in line):
                print("🎉 ¡TRADE EJECUTADO! El bot está operando")
            elif "BANKRUPTCY" in line:
                print("⚠️ Bancarrota detectada, pero esto es normal durante el aprendizaje")
            elif "ERROR" in line or "Traceback" in line:
                print("❌ ERROR DETECTADO!")
                
    except KeyboardInterrupt:
        print("\n⏹️ Monitoreo detenido")
        if 'process' in locals():
            process.terminate()
            process.wait()
    except Exception as e:
        print(f"❌ Error en monitoreo: {e}")

if __name__ == "__main__":
    monitor_training_logs()
