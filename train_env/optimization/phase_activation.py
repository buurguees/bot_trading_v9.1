#!/usr/bin/env python3
"""
Script para activar penalizaciones por fases durante el entrenamiento
Fase A: Después de 2M steps
Fase B: Después de 5-10M steps
"""

import yaml
import argparse
from pathlib import Path

def update_rewards_config(phase, config_path="config/rewards.yaml"):
    """Actualiza config/rewards.yaml según la fase"""
    
    if phase == "A":
        print("🔄 ACTIVANDO FASE A (después de 2M steps)")
        updates = {
            "core_events": {
                "ttl_penalty": -0.02
            },
            "shaping": {
                "inactivity": {
                    "penalty": -0.005
                }
            },
            "volatility_reward": {
                "weight": 0.1
            },
            "drawdown_penalty": {
                "weight": 0.15,
                "per_trade_cap": 0.2
            }
        }
    elif phase == "B":
        print("🔄 ACTIVANDO FASE B (después de 5-10M steps)")
        updates = {
            "core_events": {
                "ttl_penalty": -0.05
            },
            "shaping": {
                "inactivity": {
                    "penalty": -0.01
                }
            },
            "drawdown_penalty": {
                "weight": 0.25,
                "per_trade_cap": 0.4
            },
            "clipping": {
                "per_step": [-0.2, 0.2]
            }
        }
    else:
        print("❌ Fase no válida. Usar 'A' o 'B'")
        return False
    
    # Cargar configuración actual
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    # Aplicar actualizaciones
    for key, value in updates.items():
        if key in config:
            if isinstance(value, dict):
                config[key].update(value)
            else:
                config[key] = value
        else:
            config[key] = value
    
    # Guardar configuración actualizada
    with open(config_path, 'w', encoding='utf-8') as f:
        yaml.dump(config, f, default_flow_style=False, allow_unicode=True)
    
    print(f"✅ Configuración actualizada en {config_path}")
    return True

def update_risk_config(phase, config_path="config/risk.yaml"):
    """Actualiza config/risk.yaml según la fase"""
    
    if phase == "A":
        print("🔄 ACTUALIZANDO RISK CONFIG - FASE A")
        updates = {
            "common": {
                "default_levels": {
                    "min_sl_pct": 0.5,
                    "tp_r_multiple": 1.2
                }
            },
            "spot": {
                "risk_pct_per_trade": 0.6
            }
        }
    elif phase == "B":
        print("🔄 ACTUALIZANDO RISK CONFIG - FASE B")
        updates = {
            "common": {
                "default_levels": {
                    "min_sl_pct": 0.7,
                    "tp_r_multiple": 1.4
                }
            },
            "spot": {
                "risk_pct_per_trade": 0.8
            }
        }
    else:
        print("❌ Fase no válida. Usar 'A' o 'B'")
        return False
    
    # Cargar configuración actual
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    # Aplicar actualizaciones recursivamente
    def deep_update(d, u):
        for k, v in u.items():
            if isinstance(v, dict):
                d[k] = deep_update(d.get(k, {}), v)
            else:
                d[k] = v
        return d
    
    config = deep_update(config, updates)
    
    # Guardar configuración actualizada
    with open(config_path, 'w', encoding='utf-8') as f:
        yaml.dump(config, f, default_flow_style=False, allow_unicode=True)
    
    print(f"✅ Configuración actualizada en {config_path}")
    return True

def show_current_config():
    """Muestra la configuración actual"""
    print("📋 CONFIGURACIÓN ACTUAL:")
    print("-" * 40)
    
    # Mostrar rewards.yaml
    with open("config/rewards.yaml", 'r', encoding='utf-8') as f:
        rewards = yaml.safe_load(f)
    
    print("REWARDS:")
    print(f"  ttl_penalty: {rewards.get('core_events', {}).get('ttl_penalty', 'N/A')}")
    print(f"  inactivity_penalty: {rewards.get('shaping', {}).get('inactivity', {}).get('penalty', 'N/A')}")
    print(f"  drawdown_weight: {rewards.get('drawdown_penalty', {}).get('weight', 'N/A')}")
    
    # Mostrar risk.yaml
    with open("config/risk.yaml", 'r', encoding='utf-8') as f:
        risk = yaml.safe_load(f)
    
    print("\nRISK:")
    print(f"  min_sl_pct: {risk.get('common', {}).get('default_levels', {}).get('min_sl_pct', 'N/A')}")
    print(f"  tp_r_multiple: {risk.get('common', {}).get('default_levels', {}).get('tp_r_multiple', 'N/A')}")
    print(f"  risk_pct_per_trade: {risk.get('spot', {}).get('risk_pct_per_trade', 'N/A')}")

def main():
    parser = argparse.ArgumentParser(description='Activación de penalizaciones por fases')
    parser.add_argument('phase', choices=['A', 'B'], help='Fase a activar (A o B)')
    parser.add_argument('--show', action='store_true', help='Mostrar configuración actual')
    parser.add_argument('--rewards-only', action='store_true', help='Solo actualizar rewards.yaml')
    parser.add_argument('--risk-only', action='store_true', help='Solo actualizar risk.yaml')
    
    args = parser.parse_args()
    
    if args.show:
        show_current_config()
        return
    
    print(f"🚀 ACTIVANDO FASE {args.phase}")
    print("=" * 50)
    
    success = True
    
    if not args.risk_only:
        success &= update_rewards_config(args.phase)
    
    if not args.rewards_only:
        success &= update_risk_config(args.phase)
    
    if success:
        print(f"\n✅ FASE {args.phase} ACTIVADA EXITOSAMENTE")
        print("💡 Reinicia el entrenamiento para aplicar los cambios")
    else:
        print(f"\n❌ ERROR ACTIVANDO FASE {args.phase}")

if __name__ == "__main__":
    main()
