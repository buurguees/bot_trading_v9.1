#!/usr/bin/env python3
"""
Script para ajustar rápidamente los parámetros del sistema.
Permite modificar configuraciones sin editar archivos manualmente.
"""

import yaml
import json
from pathlib import Path

def load_yaml(file_path):
    """Carga un archivo YAML."""
    with open(file_path, 'r') as f:
        return yaml.safe_load(f)

def save_yaml(data, file_path):
    """Guarda datos en un archivo YAML."""
    with open(file_path, 'w') as f:
        yaml.dump(data, f, default_flow_style=False, indent=2)

def tune_rewards():
    """Ajusta los parámetros de reward."""
    print("🎯 AJUSTANDO PARÁMETROS DE REWARD")
    print("=" * 40)
    
    rewards_file = "config/rewards.yaml"
    rewards = load_yaml(rewards_file)
    
    # Mostrar configuración actual
    print("📊 Configuración actual:")
    print(f"   • Bonus por trade: {rewards['trade_activity_daily']['bonus_per_trade']}")
    print(f"   • Penalización por inactividad: {rewards['trade_activity_daily']['shortfall_penalty']}")
    print(f"   • Penalización por exceso: {rewards['trade_activity_daily']['overtrade_penalty']}")
    print(f"   • Warmup days: {rewards['trade_activity_daily']['warmup_days']}")
    print()
    
    # Opciones de ajuste
    print("🔧 Opciones de ajuste:")
    print("1. Más agresivo (más incentivos)")
    print("2. Más conservador (menos incentivos)")
    print("3. Personalizado")
    print("4. Mantener actual")
    
    choice = input("Selecciona una opción (1-4): ").strip()
    
    if choice == "1":  # Más agresivo
        rewards['trade_activity_daily']['bonus_per_trade'] = 0.08
        rewards['trade_activity_daily']['shortfall_penalty'] = 0.12
        rewards['trade_activity_daily']['overtrade_penalty'] = 0.02
        rewards['trade_activity_daily']['warmup_days'] = 2
        print("✅ Configuración más agresiva aplicada")
        
    elif choice == "2":  # Más conservador
        rewards['trade_activity_daily']['bonus_per_trade'] = 0.03
        rewards['trade_activity_daily']['shortfall_penalty'] = 0.05
        rewards['trade_activity_daily']['overtrade_penalty'] = 0.05
        rewards['trade_activity_daily']['warmup_days'] = 5
        print("✅ Configuración más conservadora aplicada")
        
    elif choice == "3":  # Personalizado
        bonus = float(input("Bonus por trade (0.01-0.10): "))
        shortfall = float(input("Penalización por inactividad (0.01-0.20): "))
        overtrade = float(input("Penalización por exceso (0.01-0.10): "))
        warmup = int(input("Días de warmup (1-10): "))
        
        rewards['trade_activity_daily']['bonus_per_trade'] = bonus
        rewards['trade_activity_daily']['shortfall_penalty'] = shortfall
        rewards['trade_activity_daily']['overtrade_penalty'] = overtrade
        rewards['trade_activity_daily']['warmup_days'] = warmup
        print("✅ Configuración personalizada aplicada")
        
    elif choice == "4":
        print("✅ Configuración actual mantenida")
        return
    
    # Guardar cambios
    save_yaml(rewards, rewards_file)
    print(f"💾 Configuración guardada en {rewards_file}")

def tune_training():
    """Ajusta los parámetros de entrenamiento."""
    print("🎓 AJUSTANDO PARÁMETROS DE ENTRENAMIENTO")
    print("=" * 40)
    
    train_file = "config/train.yaml"
    train = load_yaml(train_file)
    
    # Mostrar configuración actual
    print("📊 Configuración actual:")
    print(f"   • Learning rate: {train['ppo']['learning_rate']}")
    print(f"   • Entropy coefficient: {train['ppo']['ent_coef']}")
    print(f"   • Clip range: {train['ppo']['clip_range']}")
    print(f"   • Total timesteps: {train['ppo']['total_timesteps']:,}")
    print()
    
    # Opciones de ajuste
    print("🔧 Opciones de ajuste:")
    print("1. Más exploración (más entropía)")
    print("2. Más explotación (menos entropía)")
    print("3. Más estable (learning rate más bajo)")
    print("4. Más rápido (learning rate más alto)")
    print("5. Personalizado")
    print("6. Mantener actual")
    
    choice = input("Selecciona una opción (1-6): ").strip()
    
    if choice == "1":  # Más exploración
        train['ppo']['ent_coef'] = 0.02
        train['ppo']['clip_range'] = 0.3
        print("✅ Configuración más exploratoria aplicada")
        
    elif choice == "2":  # Más explotación
        train['ppo']['ent_coef'] = 0.005
        train['ppo']['clip_range'] = 0.1
        print("✅ Configuración más explotatoria aplicada")
        
    elif choice == "3":  # Más estable
        train['ppo']['learning_rate'] = 1.0e-4
        train['ppo']['clip_range'] = 0.1
        print("✅ Configuración más estable aplicada")
        
    elif choice == "4":  # Más rápido
        train['ppo']['learning_rate'] = 5.0e-4
        train['ppo']['clip_range'] = 0.3
        print("✅ Configuración más rápida aplicada")
        
    elif choice == "5":  # Personalizado
        lr = float(input("Learning rate (1e-5 a 1e-3): "))
        ent = float(input("Entropy coefficient (0.001 a 0.1): "))
        clip = float(input("Clip range (0.1 a 0.5): "))
        
        train['ppo']['learning_rate'] = lr
        train['ppo']['ent_coef'] = ent
        train['ppo']['clip_range'] = clip
        print("✅ Configuración personalizada aplicada")
        
    elif choice == "6":
        print("✅ Configuración actual mantenida")
        return
    
    # Guardar cambios
    save_yaml(train, train_file)
    print(f"💾 Configuración guardada en {train_file}")

def main():
    """Función principal."""
    print("🔧 AJUSTE RÁPIDO DE PARÁMETROS")
    print("=" * 40)
    print()
    
    while True:
        print("¿Qué quieres ajustar?")
        print("1. Parámetros de reward")
        print("2. Parámetros de entrenamiento")
        print("3. Salir")
        
        choice = input("Selecciona una opción (1-3): ").strip()
        
        if choice == "1":
            tune_rewards()
        elif choice == "2":
            tune_training()
        elif choice == "3":
            print("👋 ¡Hasta luego!")
            break
        else:
            print("❌ Opción no válida")
        
        print()

if __name__ == "__main__":
    main()
