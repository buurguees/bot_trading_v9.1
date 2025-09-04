#!/usr/bin/env python3
"""
Ejemplo de cómo se verán los logs con timestamps UTC legibles.
"""

import json
import time
from base_env.utils.timestamp_utils import add_utc_timestamps, get_current_utc_timestamp

def example_run_log():
    """Ejemplo de un log de run con timestamps UTC"""
    current_ts, current_utc = get_current_utc_timestamp()
    
    run_data = {
        "symbol": "BTCUSDT",
        "market": "futures",
        "initial_balance": 1000.0,
        "target_balance": 1000000.0,
        "final_balance": 1234.56,
        "final_equity": 1234.56,
        "ts_start": current_ts - 3600000,  # 1 hora atrás
        "ts_end": current_ts,
        "trades_count": 5,
        "win_rate_trades": 60.0,
        "avg_trade_pnl": 12.34,
        "profit_factor": 1.5,
        "run_result": "COMPLETED"
    }
    
    # Añadir timestamps UTC
    run_with_utc = add_utc_timestamps(run_data)
    
    print("=== EJEMPLO DE LOG DE RUN ===")
    print(json.dumps(run_with_utc, indent=2, ensure_ascii=False))
    print()

def example_strategy_log():
    """Ejemplo de un log de estrategia con timestamps UTC"""
    current_ts, current_utc = get_current_utc_timestamp()
    
    strategy_events = [
        {
            "kind": "OPEN",
            "side": 1,
            "price": 50000.0,
            "qty": 0.02,
            "leverage": 3.0,
            "ts": current_ts - 1800000,  # 30 min atrás
            "segment_id": 0
        },
        {
            "kind": "CLOSE",
            "side": 1,
            "price": 50500.0,
            "qty": 0.02,
            "realized_pnl": 15.0,
            "bars_held": 5,
            "ts": current_ts,
            "segment_id": 0
        }
    ]
    
    print("=== EJEMPLO DE LOGS DE ESTRATEGIA ===")
    for event in strategy_events:
        event_with_utc = add_utc_timestamps(event)
        print(json.dumps(event_with_utc, indent=2, ensure_ascii=False))
        print()

def example_training_metrics():
    """Ejemplo de métricas de entrenamiento con timestamps UTC"""
    current_ts, current_utc = get_current_utc_timestamp()
    
    metrics_data = {
        "ts": current_ts,
        "symbol": "BTCUSDT",
        "mode": "train",
        "fps": 125.5,
        "iterations": 100,
        "total_timesteps": 204800,
        "ep_reward_mean": 0.45,
        "ep_len_mean": 150.2
    }
    
    metrics_with_utc = add_utc_timestamps(metrics_data)
    
    print("=== EJEMPLO DE MÉTRICAS DE ENTRENAMIENTO ===")
    print(json.dumps(metrics_with_utc, indent=2, ensure_ascii=False))
    print()

def example_trade_metrics():
    """Ejemplo de métricas de trades con timestamps UTC"""
    current_ts, current_utc = get_current_utc_timestamp()
    
    trade_metrics = {
        "trades_count": 10,
        "win_rate_trades": 70.0,
        "avg_trade_pnl": 8.5,
        "profit_factor": 2.1,
        "first_trade_ts": current_ts - 7200000,  # 2 horas atrás
        "last_trade_ts": current_ts,
        "avg_leverage": 3.2,
        "max_leverage": 5.0
    }
    
    # Añadir timestamps UTC
    trade_metrics_with_utc = add_utc_timestamps(trade_metrics)
    
    print("=== EJEMPLO DE MÉTRICAS DE TRADES ===")
    print(json.dumps(trade_metrics_with_utc, indent=2, ensure_ascii=False))
    print()
    
    # Mostrar que los campos UTC se añadieron
    print("Campos UTC añadidos automáticamente:")
    for key, value in trade_metrics_with_utc.items():
        if key.endswith('_utc') or key.endswith('_iso'):
            print(f"  {key}: {value}")
    print()

if __name__ == "__main__":
    print("🕐 EJEMPLOS DE LOGS CON TIMESTAMPS UTC LEGIBLES")
    print("=" * 60)
    print()
    
    example_run_log()
    example_strategy_log()
    example_training_metrics()
    example_trade_metrics()
    
    print("✅ Todos los logs ahora incluyen campos UTC legibles:")
    print("   - ts_utc: Formato YYYY-MM-DD HH:MM:SS UTC")
    print("   - ts_iso: Formato ISO 8601 (YYYY-MM-DDTHH:MM:SS.fffZ)")
    print("   - ts_start_utc, ts_end_utc: Para timestamps de inicio/fin")
    print("   - open_ts_utc, close_ts_utc: Para timestamps de trades")
    print("   - first_trade_ts_utc, last_trade_ts_utc: Para contexto temporal")
