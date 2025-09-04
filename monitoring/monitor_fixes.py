#!/usr/bin/env python3
"""
Script para monitorear específicamente los mensajes de fix del entrenamiento.
"""

import time
import subprocess
import sys
import os

def monitor_fixes():
    """Monitorea los mensajes de fix en tiempo real."""
    
    print("🔍 MONITOREANDO FIXES DEL ENTRENAMIENTO")
    print("=" * 50)
    print("Buscando mensajes de fix implementados...")
    print()
    
    # Buscar el proceso de entrenamiento
    try:
        # Ejecutar ps para encontrar el proceso
        result = subprocess.run(['tasklist', '/FI', 'IMAGENAME eq python.exe'], 
                              capture_output=True, text=True, shell=True)
        
        if 'python.exe' in result.stdout:
            print("✅ Proceso de entrenamiento encontrado")
        else:
            print("❌ No se encontró proceso de entrenamiento")
            return
            
    except Exception as e:
        print(f"❌ Error buscando proceso: {e}")
        return
    
    # Monitorear archivos de log
    log_dir = "logs"
    if not os.path.exists(log_dir):
        print(f"❌ Directorio de logs no encontrado: {log_dir}")
        return
    
    print("📁 Monitoreando directorio de logs...")
    
    # Buscar archivos de log recientes
    log_files = []
    for file in os.listdir(log_dir):
        if file.endswith('.log') or 'events.out.tfevents' in file:
            log_files.append(os.path.join(log_dir, file))
    
    if not log_files:
        print("❌ No se encontraron archivos de log")
        return
    
    # Ordenar por fecha de modificación
    log_files.sort(key=lambda x: os.path.getmtime(x), reverse=True)
    latest_log = log_files[0]
    
    print(f"📄 Monitoreando: {latest_log}")
    print()
    
    # Monitorear el archivo más reciente
    try:
        with open(latest_log, 'r', encoding='utf-8', errors='ignore') as f:
            # Ir al final del archivo
            f.seek(0, 2)
            
            while True:
                line = f.readline()
                if line:
                    line = line.strip()
                    if line:
                        # Buscar mensajes de fix
                        if "FIX RL: action=0 →" in line:
                            print(f"🎉 ¡FIX FUNCIONANDO! {line}")
                        elif "BYPASS POLICY:" in line:
                            print(f"🚀 BYPASS ACTIVO: {line}")
                        elif "FIX DEFAULT_LEVELS_APPLIED:" in line:
                            print(f"🔧 SL/TP APLICADOS: {line}")
                        elif "OPEN_ATTEMPT:" in line:
                            print(f"🎯 INTENTO APERTURA: {line}")
                        elif "FORZANDO_MIN_NOTIONAL" in line:
                            print(f"📏 MIN NOTIONAL: {line}")
                        elif "CORRIGIENDO DRIFT:" in line:
                            print(f"💰 DRIFT CORREGIDO: {line}")
                        elif "ERROR" in line or "Traceback" in line:
                            print(f"❌ ERROR: {line}")
                        elif "BANKRUPTCY" in line:
                            print(f"⚠️ BANCARROTA: {line}")
                else:
                    time.sleep(0.1)
                    
    except KeyboardInterrupt:
        print("\n⏹️ Monitoreo detenido")
    except Exception as e:
        print(f"❌ Error en monitoreo: {e}")

if __name__ == "__main__":
    monitor_fixes()
