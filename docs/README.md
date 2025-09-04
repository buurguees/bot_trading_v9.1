# 🤖 Bot Trading v9.1 - Sistema de Trading con IA

Un sistema avanzado de trading automatizado que utiliza **Reinforcement Learning (PPO)** para aprender estrategias de trading en mercados de criptomonedas. El bot puede operar tanto en **spot** como en **futures** con gestión de riesgo avanzada, métricas profesionales y monitoreo en tiempo real.

## 🚀 Características Principales

### 🧠 **Inteligencia Artificial**
- **Algoritmo PPO** (Proximal Policy Optimization) de Stable Baselines3
- **Entrenamiento multi-proceso** con ambientes vectorizados
- **Learning rate annealing** automático (3e-4 → 1e-5)
- **Hiperparámetros optimizados** para trading (ent_coef=0.02, clip_range=0.3)
- **Callbacks avanzados** para checkpoints, estrategias y métricas

### 📊 **Mercados Soportados**
- **Spot Trading**: Leverage fijo 1.0x, trading directo de activos
- **Futures Trading**: Leverage configurable (2x-25x), gestión de margen
- **Símbolos**: BTCUSDT, ETHUSDT y más (configurable en `config/symbols.yaml`)
- **Timeframes**: 1m, 5m, 15m, 1h (datos históricos de Binance)

### ⚡ **Gestión de Riesgo Avanzada**
- **Stop Loss obligatorio**: Mínimo 1% de distancia
- **Take Profit inteligente**: Múltiplo de 1.5x del SL
- **Sizing automático**: Basado en equity y distancia del SL
- **MinNotional**: Configurado para Bitget (5.0 USDT para BTCUSDT)
- **Leverage dinámico**: Clampado al rango del símbolo
- **Bankruptcy protection**: Detección automática de quiebra

### 📈 **Métricas Profesionales**
- **Win Rate**: Porcentaje de trades ganadores
- **Profit Factor**: Ratio ganancias/pérdidas
- **Average Trade PnL**: Beneficio medio por trade
- **Consecutive Streaks**: Rachas de ganancias/pérdidas
- **Holding Time**: Duración media de posiciones
- **Leverage Statistics**: Uso y distribución de leverage

## 🏗️ Arquitectura del Sistema

```
bot_trading_v9.1/
├── 📁 base_env/                 # Entorno base de trading
│   ├── 📁 accounting/          # Sistema contable (ledger, portfolio)
│   ├── 📁 config/              # Configuraciones y modelos de datos
│   ├── 📁 io/                  # Brokers de datos históricos
│   ├── 📁 logging/             # Sistema de logging y métricas
│   ├── 📁 metrics/             # Cálculo de métricas profesionales
│   ├── 📁 risk/                # Gestión de riesgo y sizing
│   └── base_env.py             # Entorno principal de trading
├── 📁 train_env/               # Entorno de entrenamiento
│   ├── 📁 callbacks/           # Callbacks de entrenamiento
│   ├── gym_wrapper.py          # Wrapper para Gym/Stable Baselines3
│   ├── model_manager.py        # Gestión de modelos y artefactos
│   ├── strategy_*.py           # Sistema de estrategias
│   └── vec_factory_*.py        # Factory de ambientes vectorizados
├── 📁 scripts/                 # Scripts principales
│   ├── train_ppo.py            # Entrenamiento principal
│   └── watch_progress.py       # Monitor en tiempo real
├── 📁 config/                  # Configuraciones
│   ├── train.yaml              # Configuración de entrenamiento
│   ├── symbols.yaml            # Configuración de símbolos
│   └── risk.yaml               # Configuración de riesgo
├── 📁 tests/                   # Tests de validación
└── 📁 models/                  # Modelos entrenados y logs
    └── 📁 {SYMBOL}/            # Por cada símbolo
        ├── {SYMBOL}_PPO.zip    # Modelo principal
        ├── {SYMBOL}_runs.jsonl # Historial de runs
        ├── {SYMBOL}_strategies.json # Mejores estrategias
        └── checkpoints/        # Checkpoints periódicos
```

## 🚀 Inicio Rápido

### 1. **Instalación**
```bash
# Clonar el repositorio
git clone <repository-url>
cd bot_trading_v9.1

# Crear entorno virtual
python -m venv venv
source venv/bin/activate  # Linux/Mac
# o
venv\Scripts\activate     # Windows

# Instalar dependencias
pip install -r requirements.txt
```

### 2. **Configuración**
```bash
# Verificar configuración
python -c "import yaml; print(yaml.safe_load(open('config/train.yaml')))"

# Verificar datos disponibles
python -c "from base_env.io.historical_broker import ParquetHistoricalBroker; print(ParquetHistoricalBroker('data', ['1m', '5m', '15m', '1h']).list_symbols())"
```

### 3. **Entrenamiento**
```bash
# Entrenar BTCUSDT en futures
python scripts/train_ppo.py BTCUSDT train_futures

# Entrenar ETHUSDT en spot
python scripts/train_ppo.py ETHUSDT train_spot
```

### 4. **Monitoreo**
```bash
# Ver progreso en tiempo real
python scripts/watch_progress.py BTCUSDT
```

## 📋 Comandos Disponibles

### 🎯 **Entrenamiento**
```bash
# Sintaxis general
python scripts/train_ppo.py <SYMBOL> <MODE>

# Ejemplos
python scripts/train_ppo.py BTCUSDT train_futures    # Futures con leverage
python scripts/train_ppo.py ETHUSDT train_spot       # Spot sin leverage
python scripts/train_ppo.py ADAUSDT train_futures    # Otro símbolo
```

### 📊 **Monitoreo**
```bash
# Ver progreso de entrenamiento
python scripts/watch_progress.py <SYMBOL>

# Ejemplos
python scripts/watch_progress.py BTCUSDT
python scripts/watch_progress.py ETHUSDT
```

### 🧪 **Testing**
```bash
# Ejecutar todos los tests
python -m pytest tests/ -v

# Tests específicos
python -m pytest tests/test_ledger_no_position_invariant.py -v
python -m pytest tests/test_sizing_min_notional_bitget.py -v
python -m pytest tests/test_rewards_map.py -v
```

## ⚙️ Configuración Avanzada

### 📁 **config/train.yaml**
```yaml
# Configuración principal de entrenamiento
env:
  n_envs: 2                    # Número de ambientes paralelos
  initial_balance: 1000.0      # Balance inicial
  target_balance: 1000000.0    # Balance objetivo

ppo:
  total_timesteps: 10000000    # Pasos totales de entrenamiento
  ent_coef: 0.02              # Coeficiente de entropía
  clip_range: 0.3             # Rango de clipping
  anneal_lr: true             # Annealing de learning rate

runs_log:
  max_records: 2000           # Máximo de runs guardados
  prune_strategy: "fifo"      # Estrategia de limpieza
```

### 📁 **config/symbols.yaml**
```yaml
# Configuración por símbolo
BTCUSDT:
  market: "futures"
  leverage:
    min: 2.0
    max: 25.0
    default: 3.0
  filters:
    minNotional: 5.0          # Mínimo notional (Bitget)
    lotStep: 0.001
    tickSize: 0.01
```

### 📁 **config/risk.yaml**
```yaml
# Configuración de riesgo
common:
  default_levels:
    min_sl_pct: 1.0           # Mínimo SL 1%
    tp_r_multiple: 1.5        # TP = 1.5x SL
  train_force_min_notional: true
```

## 📊 Sistema de Métricas

### 🎯 **Métricas por Run**
- **`trades_count`**: Número total de trades
- **`win_rate_trades`**: % de trades ganadores
- **`avg_trade_pnl`**: Beneficio medio por trade (USDT)
- **`avg_holding_bars`**: Duración media en barras
- **`max_consecutive_wins/losses`**: Rachas máximas
- **`profit_factor`**: Ratio ganancias/pérdidas
- **`gross_profit/gross_loss`**: Ganancias/pérdidas totales

### ⚡ **Métricas de Leverage**
- **`avg_leverage`**: Leverage promedio usado
- **`max_leverage`**: Leverage máximo usado
- **`high_leverage_pct`**: % de trades con leverage > 10x

### 📈 **Métricas de Entrenamiento**
- **`fps`**: Frames por segundo
- **`learning_rate`**: Learning rate actual
- **`total_timesteps`**: Pasos totales
- **`approx_kl`**: Divergencia KL aproximada
- **`entropy`**: Entropía de la política

## 🔧 Sistema de Estrategias

### 📝 **Registro de Estrategias**
- **Eventos OPEN/CLOSE**: Registro automático de trades
- **Scoring inteligente**: Basado en R-multiple, ROI, leverage
- **Top-K strategies**: Mejores 1000 estrategias guardadas
- **Deduplicación**: Eliminación de estrategias duplicadas

### 🎯 **Criterios de Scoring**
```python
# Factores de scoring
base_score = r_multiple * 2.0 + roi_pct * 0.1 + realized_pnl * 0.01

# Bonuses
leverage_efficiency_bonus = (notional_eff / notional_max) * 2.0
leverage_moderation_bonus = (1.0 - |leverage - 5.0| / 25.0) * 1.0
timeframe_bonus = tf_multiplier * 0.5
bars_held_bonus = min(bars_held / 10.0, 2.0) * 0.3
```

## 🛡️ Gestión de Riesgo

### ⚠️ **Validaciones Obligatorias**
- **SL requerido**: `sl_distance >= min_sl_pct` (1%)
- **TP requerido**: `tp_distance >= tp_r_multiple * sl_distance` (1.5x)
- **MinNotional**: `notional >= minNotional` (5.0 USDT para BTCUSDT)
- **Leverage válido**: Clampado al rango del símbolo

### 🚫 **Bloqueos de Trading**
- **`NO_SL_DISTANCE`**: SL insuficiente
- **`MIN_NOTIONAL_BLOCKED`**: Notional muy pequeño
- **`BANKRUPTCY`**: Equity <= 0

### 📊 **Sizing Inteligente**
```python
# Cálculo de tamaño
qty_raw = risk_usd / sl_distance
qty = round_down(qty_raw, lotStep)
price = round(price, tickSize)
notional = qty * price

# Escalado si es necesario
if notional < minNotional and train_force_min_notional:
    qty = ceil(minNotional / price) * lotStep
    qty = min(qty, notional_limit / price)
```

## 🎮 Sistema de Rewards

### 🏆 **Rewards por Cierre de Trade**
- **Take Profit**: `+1.0`
- **Stop Loss**: `-0.5`
- **ROI escalado**: Proporcional entre `[-0.5, +1.0]`

### ⚠️ **Penalizaciones**
- **Bankruptcy**: `-10.0` (una vez por run)
- **Inactividad**: `-0.01` cada 100 pasos sin trade
- **Trades bloqueados**: `-0.05` por evento

### 🎁 **Bonuses**
- **Posición mantenida**: `+0.05` cada 10 barras con equity positivo

## 📁 Estructura de Archivos

### 🗂️ **Archivos de Modelo**
```
models/{SYMBOL}/
├── {SYMBOL}_PPO.zip                    # Modelo principal
├── {SYMBOL}_PPO.zip.backup             # Backup del modelo
├── {SYMBOL}_strategies.json            # Top-1000 estrategias
├── {SYMBOL}_strategies_provisional.jsonl # Estrategias provisionales
├── {SYMBOL}_bad_strategies.json        # Estrategias malas
├── {SYMBOL}_progress.json              # Progreso del entrenamiento
├── {SYMBOL}_runs.jsonl                 # Historial de runs
├── {SYMBOL}_train_metrics.jsonl        # Métricas de entrenamiento
└── checkpoints/                        # Checkpoints periódicos
    ├── checkpoint_1000000.zip
    ├── checkpoint_2000000.zip
    └── ...
```

### 📊 **Archivos de Datos**
```
data/
├── 📁 BTCUSDT/                         # Datos por símbolo
│   ├── 1m.parquet                      # Datos de 1 minuto
│   ├── 5m.parquet                      # Datos de 5 minutos
│   ├── 15m.parquet                     # Datos de 15 minutos
│   └── 1h.parquet                      # Datos de 1 hora
├── 📁 ETHUSDT/
└── 📁 ADAUSDT/
```

## 🧪 Testing y Validación

### ✅ **Tests Implementados**
- **`test_ledger_no_position_invariant.py`**: Verifica consistencia contable
- **`test_open_close_flow.py`**: Verifica flujo de apertura/cierre
- **`test_sizing_min_notional_bitget.py`**: Verifica sizing con minNotional
- **`test_trade_levels_required.py`**: Verifica SL/TP obligatorios
- **`test_rewards_map.py`**: Verifica sistema de rewards
- **`test_run_metrics.py`**: Verifica métricas profesionales
- **`test_runs_retention.py`**: Verifica retención FIFO
- **`test_leverage_spot_vs_futures.py`**: Verifica cálculo de leverage
- **`test_trade_log_includes_leverage.py`**: Verifica logging de leverage
- **`test_strategy_persistence_leverage.py`**: Verifica persistencia de estrategias

### 🚀 **Ejecutar Tests**
```bash
# Todos los tests
python -m pytest tests/ -v

# Tests específicos
python -m pytest tests/test_ledger_no_position_invariant.py -v
python -m pytest tests/test_sizing_min_notional_bitget.py -v
python -m pytest tests/test_rewards_map.py -v
```

## 🔍 Monitoreo y Debugging

### 📊 **Watch Progress**
```bash
python scripts/watch_progress.py BTCUSDT
```

**Salida del monitor:**
```
🚀 PROGRESO DE ENTRENAMIENTO: BTCUSDT
═══════════════════════════════════════════════════════════════════════════════

📊 ÚLTIMOS RUNS:
   Run 1: Balance: 1,240.50 | Equity: 1,240.50 | Trades: 42 | Win Rate: 38.1% | Avg PnL: 5.71 | Profit Factor: 1.81
   Run 2: Balance: 1,180.30 | Equity: 1,180.30 | Trades: 38 | Win Rate: 42.1% | Avg PnL: 4.74 | Profit Factor: 1.65

⚡ KPIs DE LEVERAGE:
   Leverage promedio: 3.2x
   Leverage máximo: 15.0x
   % trades high leverage: 12.5%

📈 KPIs PROFESIONALES DE TRADES:
   Mejor Win Rate: 45.2%
   Win Rate promedio: 40.1%
   Mejor Avg PnL: 8.45 USDT
   Avg PnL promedio: 5.12 USDT
   Mayor Profit Factor: 2.15
   Profit Factor promedio: 1.73
```

### 🐛 **Logs de Debugging**
- **`CORRIGIENDO DRIFT…`**: Corrección automática de drift contable
- **`SIZING_BLOCKED`**: Trades bloqueados por sizing
- **`NO_SL_DISTANCE`**: Trades bloqueados por SL insuficiente
- **`MIN_NOTIONAL_BLOCKED`**: Trades bloqueados por notional pequeño

## 🚀 Rendimiento y Escalabilidad

### ⚡ **Optimizaciones**
- **Multiprocessing**: Ambientes paralelos con `SubprocVecEnv`
- **Start method**: `spawn` para estabilidad en Windows
- **Memory management**: FIFO retention de runs (2000 máximo)
- **JSON serialization**: Conversión automática de tipos NumPy

### 📊 **Métricas de Rendimiento**
- **FPS**: Frames por segundo durante entrenamiento
- **Memory usage**: Gestión eficiente de memoria
- **Checkpoint frequency**: Guardado automático cada N pasos
- **Strategy aggregation**: Consolidación eficiente de estrategias

## 🔧 Troubleshooting

### ❌ **Problemas Comunes**

#### **Error de Importación**
```bash
# Error: cannot import name 'PeriodicCheckpoint'
# Solución: Verificar que los callbacks estén en archivos separados
ls train_env/callbacks/
```

#### **Error de Multiprocessing**
```bash
# Error: EOFError, BrokenPipeError
# Solución: Ya configurado con start_method='spawn'
```

#### **Error de Serialización JSON**
```bash
# Error: Object of type 'float32' is not JSON serializable
# Solución: Ya implementada conversión automática de tipos NumPy
```

#### **Error de Configuración**
```bash
# Error: minNotional incorrecto
# Solución: Verificar config/symbols.yaml
python -c "import yaml; print(yaml.safe_load(open('config/symbols.yaml')))"
```

### 🔍 **Verificaciones**
```bash
# Verificar configuración
python -c "import yaml; print(yaml.safe_load(open('config/train.yaml')))"

# Verificar datos
python -c "from base_env.io.historical_broker import ParquetHistoricalBroker; print(ParquetHistoricalBroker('data', ['1m']).list_symbols())"

# Verificar modelo
python -c "from stable_baselines3 import PPO; print('PPO disponible')"
```

## 📚 Dependencias

### 🐍 **Python 3.13+**
- **stable-baselines3**: Algoritmo PPO
- **gymnasium**: Entorno de RL
- **numpy**: Cálculos numéricos
- **pandas**: Manipulación de datos
- **pyarrow**: Lectura de Parquet
- **pyyaml**: Configuraciones
- **pytest**: Testing

### 📦 **Instalación**
```bash
pip install stable-baselines3 gymnasium numpy pandas pyarrow pyyaml pytest
```

## 🤝 Contribución

### 📝 **Guidelines**
1. **Tests**: Añadir tests para nuevas funcionalidades
2. **Documentación**: Actualizar README y comentarios
3. **Configuración**: Mantener compatibilidad con configs existentes
4. **Performance**: Considerar impacto en rendimiento

### 🧪 **Testing**
```bash
# Ejecutar tests antes de commit
python -m pytest tests/ -v

# Verificar linting
python -m flake8 base_env/ train_env/ scripts/
```

## 📄 Licencia

Este proyecto está bajo licencia MIT. Ver `LICENSE` para más detalles.

## 🙏 Agradecimientos

- **Stable Baselines3** por el algoritmo PPO
- **Binance** por los datos históricos
- **OpenAI Gym** por el framework de RL

---

**🚀 ¡Disfruta del trading automatizado con IA! 🚀**
