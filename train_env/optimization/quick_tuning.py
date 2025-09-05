#!/usr/bin/env python3
"""
Script para ajustes rápidos de palancas durante el entrenamiento
Permite ajustar parámetros sin reiniciar el entrenamiento
"""

import yaml
import argparse
from pathlib import Path

def adjust_trades_frequency():
    """Ajusta parámetros para aumentar frecuencia de trades"""
    print("🔧 AJUSTANDO PARA MÁS TRADES")
    
    # Actualizar risk.yaml
    with open("config/risk.yaml", 'r', encoding='utf-8') as f:
        risk_config = yaml.safe_load(f)
    
    # Ajustar parámetros para más trades
    risk_config['common']['default_levels']['min_sl_pct'] = 0.4
    risk_config['common']['default_levels']['tp_r_multiple'] = 1.1
    risk_config['spot']['risk_pct_per_trade'] = 0.5
    
    with open("config/risk.yaml", 'w', encoding='utf-8') as f:
        yaml.dump(risk_config, f, default_flow_style=False, allow_unicode=True)
    
    # Reducir penalización de drawdown temporalmente
    with open("config/rewards.yaml", 'r', encoding='utf-8') as f:
        rewards_config = yaml.safe_load(f)
    
    rewards_config['drawdown_penalty']['weight'] = 0.1
    
    with open("config/rewards.yaml", 'w', encoding='utf-8') as f:
        yaml.dump(rewards_config, f, default_flow_style=False, allow_unicode=True)
    
    print("✅ Ajustes aplicados:")
    print("   - min_sl_pct: 0.3 → 0.4")
    print("   - tp_r_multiple: 1.0 → 1.1")
    print("   - risk_pct_per_trade: 0.6 → 0.5")
    print("   - drawdown_penalty: reducido")

def adjust_ttl_issues():
    """Ajusta parámetros para reducir TTLs excesivos"""
    print("🔧 AJUSTANDO PARA REDUCIR TTLs")
    
    # Actualizar risk.yaml
    with open("config/risk.yaml", 'r', encoding='utf-8') as f:
        risk_config = yaml.safe_load(f)
    
    # Aumentar TTL y ajustar TP
    risk_config['common']['default_levels']['ttl_bars_default'] = 200
    risk_config['common']['default_levels']['tp_r_multiple'] = 1.3
    
    with open("config/risk.yaml", 'w', encoding='utf-8') as f:
        yaml.dump(risk_config, f, default_flow_style=False, allow_unicode=True)
    
    print("✅ Ajustes aplicados:")
    print("   - ttl_bars_default: 120 → 200")
    print("   - tp_r_multiple: 1.0 → 1.3")

def adjust_equity_volatility():
    """Ajusta parámetros para reducir volatilidad del equity"""
    print("🔧 AJUSTANDO PARA REDUCIR VOLATILIDAD")
    
    # Actualizar risk.yaml
    with open("config/risk.yaml", 'r', encoding='utf-8') as f:
        risk_config = yaml.safe_load(f)
    
    # Reducir riesgo por trade
    risk_config['spot']['risk_pct_per_trade'] = 0.4
    
    with open("config/risk.yaml", 'w', encoding='utf-8') as f:
        yaml.dump(risk_config, f, default_flow_style=False, allow_unicode=True)
    
    # Actualizar rewards.yaml
    with open("config/rewards.yaml", 'r', encoding='utf-8') as f:
        rewards_config = yaml.safe_load(f)
    
    # Aumentar peso de volatilidad
    rewards_config['volatility_reward']['weight'] = 0.2
    
    with open("config/rewards.yaml", 'w', encoding='utf-8') as f:
        yaml.dump(rewards_config, f, default_flow_style=False, allow_unicode=True)
    
    print("✅ Ajustes aplicados:")
    print("   - risk_pct_per_trade: 0.6 → 0.4")
    print("   - volatility_reward: 0.1 → 0.2")

def adjust_early_bankruptcy():
    """Ajusta parámetros para evitar bancarrota temprana"""
    print("🔧 AJUSTANDO PARA EVITAR BANCARROTA TEMPRANA")
    
    # Actualizar risk.yaml
    with open("config/risk.yaml", 'r', encoding='utf-8') as f:
        risk_config = yaml.safe_load(f)
    
    # Reducir riesgo y ajustar umbrales
    risk_config['spot']['risk_pct_per_trade'] = 0.3
    risk_config['common']['bankruptcy']['threshold_pct'] = 0.05  # Más permisivo
    
    with open("config/risk.yaml", 'w', encoding='utf-8') as f:
        yaml.dump(risk_config, f, default_flow_style=False, allow_unicode=True)
    
    print("✅ Ajustes aplicados:")
    print("   - risk_pct_per_trade: 0.6 → 0.3")
    print("   - bankruptcy_threshold: 0.1% → 0.05%")

def show_current_settings():
    """Muestra la configuración actual"""
    print("📋 CONFIGURACIÓN ACTUAL:")
    print("-" * 40)
    
    # Mostrar risk.yaml
    with open("config/risk.yaml", 'r', encoding='utf-8') as f:
        risk = yaml.safe_load(f)
    
    print("RISK CONFIG:")
    print(f"  min_sl_pct: {risk.get('common', {}).get('default_levels', {}).get('min_sl_pct', 'N/A')}")
    print(f"  tp_r_multiple: {risk.get('common', {}).get('default_levels', {}).get('tp_r_multiple', 'N/A')}")
    print(f"  ttl_bars_default: {risk.get('common', {}).get('default_levels', {}).get('ttl_bars_default', 'N/A')}")
    print(f"  risk_pct_per_trade: {risk.get('spot', {}).get('risk_pct_per_trade', 'N/A')}")
    print(f"  bankruptcy_threshold: {risk.get('common', {}).get('bankruptcy', {}).get('threshold_pct', 'N/A')}%")
    
    # Mostrar rewards.yaml
    with open("config/rewards.yaml", 'r', encoding='utf-8') as f:
        rewards = yaml.safe_load(f)
    
    print("\nREWARDS CONFIG:")
    print(f"  drawdown_weight: {rewards.get('drawdown_penalty', {}).get('weight', 'N/A')}")
    print(f"  volatility_weight: {rewards.get('volatility_reward', {}).get('weight', 'N/A')}")

def main():
    parser = argparse.ArgumentParser(description='Ajustes rápidos de palancas')
    parser.add_argument('action', choices=[
        'more-trades', 'less-ttl', 'less-volatility', 
        'less-bankruptcy', 'show'
    ], help='Acción a realizar')
    
    args = parser.parse_args()
    
    if args.action == 'more-trades':
        adjust_trades_frequency()
    elif args.action == 'less-ttl':
        adjust_ttl_issues()
    elif args.action == 'less-volatility':
        adjust_equity_volatility()
    elif args.action == 'less-bankruptcy':
        adjust_early_bankruptcy()
    elif args.action == 'show':
        show_current_settings()
    
    print("\n💡 Los cambios se aplicarán en el próximo reinicio del entrenamiento")

if __name__ == "__main__":
    main()
