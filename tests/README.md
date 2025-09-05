# 🧪 Estructura de Tests

## 📁 **Organización por Categorías**

### **`unit/` - Tests Unitarios**
Prueban componentes individuales de forma aislada:
- `test_ledger_consistency.py` - Consistencia del ledger
- `test_ledger_futures_consistency.py` - Consistencia de futuros
- `test_ledger_no_position_invariant.py` - Invariantes de posición
- `test_reward_logic.py` - Lógica de recompensas
- `test_rewards_map.py` - Mapeo de recompensas
- `test_sizing_filters.py` - Filtros de sizing
- `test_sizing_futures_min_notional.py` - Notional mínimo futuros
- `test_sizing_min_notional_bitget.py` - Notional mínimo Bitget
- `test_wrapper_leverage_mapping.py` - Mapeo de leverage
- `test_strategy_persistence_leverage.py` - Persistencia de estrategias

### **`integration/` - Tests de Integración**
Prueban la interacción entre múltiples componentes:
- `test_open_close_fill.py` - Flujo de apertura/cierre
- `test_open_close_flow.py` - Flujo completo de trades
- `test_leverage_spot_vs_futures.py` - Leverage spot vs futuros
- `test_levels_fallback.py` - Fallback de niveles
- `test_simple_fallback.py` - Fallback simple
- `test_simple_ledger.py` - Ledger simple
- `test_align_no_dupes.py` - Alineación sin duplicados
- `test_dup_guard.py` - Guardia de duplicados
- `test_run_metrics.py` - Métricas de runs
- `test_runs_retention.py` - Retención de runs
- `test_training_metrics_callback.py` - Callback de métricas
- `test_watch_progress_*.py` - Monitoreo de progreso
- `test_watcher_*.py` - Watchers del sistema

### **`e2e/` - Tests End-to-End**
Prueban flujos completos del sistema:
- `test_futures_data_presence.py` - Presencia de datos futuros
- `test_bitget_collector_mapping.py` - Mapeo de collector Bitget
- `test_trade_levels_required.py` - Niveles requeridos de trades
- `test_trade_log_includes_leverage.py` - Logs con leverage

### **`legacy/` - Tests Legacy**
Tests antiguos mantenidos por compatibilidad:
- Tests de la carpeta `_old/` movidos aquí
- Mantenidos para referencia histórica

## 🚀 **Ejecutar Tests**

### **Todos los tests:**
```bash
pytest tests/
```

### **Por categoría:**
```bash
# Tests unitarios
pytest tests/unit/

# Tests de integración
pytest tests/integration/

# Tests E2E
pytest tests/e2e/

# Tests legacy
pytest tests/legacy/
```

### **Por componente específico:**
```bash
# Tests de ledger
pytest tests/unit/test_ledger_*.py

# Tests de rewards
pytest tests/unit/test_reward*.py

# Tests de sizing
pytest tests/unit/test_sizing_*.py
```

## 📊 **Cobertura de Tests**

- **Unitarios**: Componentes individuales (ledger, rewards, sizing, etc.)
- **Integración**: Flujos entre componentes (trades, métricas, monitoreo)
- **E2E**: Flujos completos del sistema (datos, collectors, logs)
- **Legacy**: Tests históricos mantenidos por compatibilidad

## 🔧 **Mantenimiento**

- **Nuevos tests unitarios** → `tests/unit/`
- **Nuevos tests de integración** → `tests/integration/`
- **Nuevos tests E2E** → `tests/e2e/`
- **Tests obsoletos** → `tests/legacy/`
