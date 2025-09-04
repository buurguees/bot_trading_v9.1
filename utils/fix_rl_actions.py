#!/usr/bin/env python3
"""
Script para solucionar el problema de RL enviando action=0.
Este script modifica temporalmente el gym_wrapper para forzar acciones reales.
"""

import os
import shutil
from pathlib import Path

def backup_original_file():
    """Hace backup del archivo original."""
    original = Path("train_env/gym_wrapper.py")
    backup = Path("train_env/gym_wrapper.py.backup")
    
    if not backup.exists():
        shutil.copy2(original, backup)
        print(f"✅ Backup creado: {backup}")
    else:
        print(f"ℹ️ Backup ya existe: {backup}")

def apply_fix():
    """Aplica el fix para forzar acciones reales."""
    
    # Leer el archivo original
    with open("train_env/gym_wrapper.py", 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Buscar la línea donde se inyecta la acción
    old_line = "        self.env.set_action_override(int(trade_action), leverage_override=leverage, leverage_index=lev_idx if self._lev_spec else None)"
    
    # Nueva lógica: si trade_action es 0, cambiarlo a una acción aleatoria (3 o 4)
    new_logic = '''        # ← FIX TEMPORAL: Si RL envía action=0 (dejar policy), forzar acción real
        if trade_action == 0:
            import random
            # Forzar acciones reales: 3=force_long, 4=force_short
            trade_action = random.choice([3, 4])
            print(f"🔧 FIX RL: action=0 → {trade_action} (force_long/force_short)")
        
        self.env.set_action_override(int(trade_action), leverage_override=leverage, leverage_index=lev_idx if self._lev_spec else None)'''
    
    # Reemplazar la línea
    if old_line in content:
        content = content.replace(old_line, new_logic)
        
        # Escribir el archivo modificado
        with open("train_env/gym_wrapper.py", 'w', encoding='utf-8') as f:
            f.write(content)
        
        print("✅ Fix aplicado: RL ahora forzará acciones reales (3=force_long, 4=force_short)")
        print("   - Si RL envía action=0, se cambiará aleatoriamente a 3 o 4")
        print("   - Esto debería permitir que se ejecuten trades")
        return True
    else:
        print("❌ No se encontró la línea a modificar")
        return False

def restore_original():
    """Restaura el archivo original desde el backup."""
    original = Path("train_env/gym_wrapper.py")
    backup = Path("train_env/gym_wrapper.py.backup")
    
    if backup.exists():
        shutil.copy2(backup, original)
        print(f"✅ Archivo restaurado desde backup")
        return True
    else:
        print("❌ No se encontró el backup")
        return False

def main():
    print("🔧 FIX PARA RL ACTIONS")
    print("=" * 40)
    
    if not Path("train_env/gym_wrapper.py").exists():
        print("❌ No se encontró train_env/gym_wrapper.py")
        return
    
    print("1. Creando backup...")
    backup_original_file()
    
    print("\n2. Aplicando fix...")
    if apply_fix():
        print("\n✅ FIX APLICADO EXITOSAMENTE")
        print("\n📋 PRÓXIMOS PASOS:")
        print("   1. Reiniciar el entrenamiento")
        print("   2. Monitorear que ahora se ejecuten trades")
        print("   3. Una vez que funcione, restaurar el archivo original")
        print("\n🔄 Para restaurar: python fix_rl_actions.py --restore")
    else:
        print("\n❌ No se pudo aplicar el fix")

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "--restore":
        restore_original()
    else:
        main()
