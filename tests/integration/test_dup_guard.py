#!/usr/bin/env python3
"""
Test: Guard de deduplicación
- Bloquea duplicados en la misma barra
- Permite en la siguiente barra
"""

import pytest
import numpy as np
from pathlib import Path
import sys

# Agregar el directorio raíz al path
sys.path.insert(0, str(Path(__file__).parent.parent))

from base_env.policy.rules import dedup_block
from base_env.tfs.calendar import tf_to_ms


def test_dedup_guard():
    """Test: Verificar que la deduplicación funciona correctamente"""
    
    # Configuración de test
    base_tf_ms = tf_to_ms("1m")  # 60,000 ms
    window_bars = 3
    
    # Test 1: Sin última apertura (debe permitir)
    ts_now = 1000000
    last_open_ts = None
    assert not dedup_block(ts_now, last_open_ts, window_bars, base_tf_ms)
    
    # Test 2: Dentro de la ventana (debe bloquear)
    ts_now = 1000000
    last_open_ts = 1000000 - (2 * base_tf_ms)  # 2 barras atrás
    assert dedup_block(ts_now, last_open_ts, window_bars, base_tf_ms)
    
    # Test 3: Fuera de la ventana (debe permitir)
    ts_now = 1000000
    last_open_ts = 1000000 - (4 * base_tf_ms)  # 4 barras atrás
    assert not dedup_block(ts_now, last_open_ts, window_bars, base_tf_ms)
    
    # Test 4: Exactamente en el límite (debe permitir, ya que es >= window_bars)
    ts_now = 1000000
    last_open_ts = 1000000 - (3 * base_tf_ms)  # 3 barras atrás (límite)
    assert not dedup_block(ts_now, last_open_ts, window_bars, base_tf_ms)
    
    # Test 5: Justo después del límite (debe permitir)
    ts_now = 1000000
    last_open_ts = 1000000 - (3 * base_tf_ms) - 1  # 3 barras + 1ms atrás
    assert not dedup_block(ts_now, last_open_ts, window_bars, base_tf_ms)


def test_dedup_guard_different_timeframes():
    """Test: Verificar deduplicación con diferentes timeframes"""
    
    # Test con timeframe de 5m
    base_tf_ms = tf_to_ms("5m")  # 300,000 ms
    window_bars = 2
    
    ts_now = 1000000
    last_open_ts = 1000000 - (1 * base_tf_ms)  # 1 barra de 5m atrás
    assert dedup_block(ts_now, last_open_ts, window_bars, base_tf_ms)
    
    last_open_ts = 1000000 - (3 * base_tf_ms)  # 3 barras de 5m atrás
    assert not dedup_block(ts_now, last_open_ts, window_bars, base_tf_ms)


def test_dedup_guard_edge_cases():
    """Test: Casos límite de deduplicación"""
    
    base_tf_ms = tf_to_ms("1m")
    window_bars = 1
    
    # Test con timestamps iguales (debe bloquear)
    ts_now = 1000000
    last_open_ts = 1000000
    assert dedup_block(ts_now, last_open_ts, window_bars, base_tf_ms)
    
    # Test con timestamp futuro (debe bloquear, ya que la diferencia es negativa)
    ts_now = 1000000
    last_open_ts = 1000000 + base_tf_ms
    assert dedup_block(ts_now, last_open_ts, window_bars, base_tf_ms)
    
    # Test con ventana de 0 barras (siempre debe permitir)
    window_bars = 0
    ts_now = 1000000
    last_open_ts = 1000000 - base_tf_ms
    assert not dedup_block(ts_now, last_open_ts, window_bars, base_tf_ms)


if __name__ == "__main__":
    test_dedup_guard()
    test_dedup_guard_different_timeframes()
    test_dedup_guard_edge_cases()
    print("✅ Todos los tests de deduplicación pasaron")
