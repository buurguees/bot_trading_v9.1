# Bot Trading v9.1 - Sistema de Trading con Reinforcement Learning

Sistema avanzado de trading automatizado que utiliza Reinforcement Learning (PPO) para aprender estrategias de trading en mercados de criptomonedas.

## 🚀 Características Principales

- **Reinforcement Learning**: Entrenamiento con PPO (Proximal Policy Optimization)
- **Multi-Timeframe**: Análisis en 1m, 5m, 15m, 1h, 4h
- **Spot y Futuros**: Soporte para ambos mercados
- **Leverage Dinámico**: Selección automática de leverage (2x-10x)
- **Sistema de Rewards Avanzado**: 15+ sistemas de rewards/penalties modulares
- **Risk Management**: Gestión de riesgo con SL/TP automáticos
- **Backtesting**: Validación histórica de estrategias
- **Live Trading**: Ejecución en tiempo real
- **Timestamps UTC**: Logs con fechas legibles en formato UTC
- **Sistema Modular**: Arquitectura completamente modular y configurable

## 📁 Estructura del Proyecto

```
bot_trading_v9.1/
├── app.py                    # Punto de entrada principal
├── requirements.txt          # Dependencias del proyecto
├── README.md                # Este archivo
├── config/                  # Configuraciones YAML
│   ├── symbols.yaml         # Configuración de símbolos
│   ├── rewards.yaml         # Sistema de rewards/penalties
│   ├── risk.yaml           # Gestión de riesgo
│   └── ...
├── base_env/               # Entorno base de trading
│   ├── actions/            # Sistemas de rewards/penalties
│   ├── accounting/         # Contabilidad y PnL
│   ├── analysis/           # Análisis técnico
│   ├── config/             # Modelos de configuración
│   ├── events/             # Sistema de eventos
│   ├── features/           # Pipeline de features
│   ├── io/                 # Ingestión de datos
│   ├── logging/            # Sistema de logs
│   ├── policy/             # Motor de políticas
│   ├── risk/               # Gestión de riesgo
│   ├── smc/                # Smart Money Concepts
│   ├── tfs/                # Multi-timeframe
│   └── telemetry/          # Telemetría
├── train_env/              # Entorno de entrenamiento
│   ├── callbacks/          # Callbacks de entrenamiento
│   ├── utils/              # Utilidades
│   └── ...
├── scripts/                # Scripts de entrenamiento
├── tests/                  # Tests unitarios
├── data/                   # Datos históricos
├── models/                 # Modelos entrenados
├── logs/                   # Logs de entrenamiento
├── utils/                  # Utilidades generales
├── monitoring/             # Scripts de monitoreo
├── docs/                   # Documentación
└── archives/               # Archivos históricos
```

## 🛠️ Instalación y Setup Reproducible

### Requisitos del Sistema
- **Python**: 3.9+ (probado con 3.9, 3.10, 3.11)
- **Dependencias principales**: 
  - `stable-baselines3` (PPO)
  - `gymnasium` (entornos RL)
  - `numpy` (cálculos numéricos)
  - `pyarrow` (datos Parquet)
  - `pyyaml` (configuraciones)
  - `pandas` (manipulación de datos)
  - `torch` (backend de SB3)

### Instalación Paso a Paso

1. **Clonar el repositorio**:
```bash
git clone <repository-url>
cd bot_trading_v9.1
```

2. **Crear entorno virtual**:
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# o
venv\Scripts\activate     # Windows
```

3. **Instalar dependencias**:
```bash
pip install -r requirements.txt
```

4. **Configurar seeds para reproducibilidad**:
```bash
# Fijar seeds para reproducibilidad
export PYTHONHASHSEED=42
python -c "import random; random.seed(42)"
python -c "import numpy; numpy.random.seed(42)"
python -c "import torch; torch.manual_seed(42)"
```

5. **Configurar multiprocessing para Windows**:
```python
# Ya configurado en scripts/train_ppo.py
import multiprocessing as mp
mp.set_start_method('spawn', force=True)
```

6. **Configurar datos**:
   - Colocar datos históricos en `data/BTCUSDT/`
   - Configurar `config/symbols.yaml` para tu símbolo

## 🎯 Quickstart

### Entrenamiento (Futures)
```bash
python app.py --mode train --symbol BTCUSDT --seed 42 --vecenv 4
```

### Backtest Determinista
```bash
python app.py --mode backtest --symbol BTCUSDT --from 2022-01-01 --to 2024-12-31
```

### Live Trading (Paper/Testnet)
```bash
python app.py --mode live --symbol BTCUSDT --account testnet
```

### Comandos Básicos
```bash
# Entrenamiento con GUI
python app.py run --gui

# Solo interfaz gráfica
python app.py gui

# Mostrar configuraciones
python app.py config
```

## 📋 Contratos entre Módulos

### base_env expone

**Interfaz estándar de Gymnasium**:
```python
reset() -> (obs, info)
step(action) -> (obs, reward, terminated, truncated, info)
```

**info contiene**:
- `bar_time`: Timestamp del bar actual
- `price`: Precio actual del símbolo
- `event`: Evento actual `{OPEN, CLOSE, TP, SL, TTL, HOLD}`
- `equity`: Equity actual del portfolio
- `position`: Estado de la posición actual
- `leverage_used`: Leverage utilizado en el trade
- `max_dd_vs_sl`: Máximo drawdown vs Stop Loss
- `fees_paid`: Fees pagados en el trade
- `close_reason`: Razón del cierre `{sl_hit, tp_hit, ttl_hit, manual}`
- `bars_held`: Barras que duró el trade
- `realized_pnl`: PnL realizado del trade

### train_env garantiza

**Normalización de acciones**:
- Acción normalizada `a ∈ [-1,1]` → `side ∈ {-1, 0, +1}`
- `action=0`: Dejar que la política jerárquica decida
- `action=1`: Cerrar todas las posiciones
- `action=3`: Forzar posición larga
- `action=4`: Forzar posición corta

**Orquestación de rewards**:
- Rewards por evento (OPEN, CLOSE, TP, SL, TTL)
- Shaping por step (holding, inactivity, progress)
- Sistema modular de 15+ componentes de reward/penalty

**Deduplicación**:
- Prevención de duplicados por `(bar_time, side)`
- Limpieza automática en nuevo bar y al cerrar posiciones
- Evita bloqueo de señales y mantiene consistencia

## ⚙️ Configuración

### Símbolos (`config/symbols.yaml`) - Ejemplo Mínimo Robusto
```yaml
symbols:
  - symbol: BTCUSDT
    market: futures
    enabled: true
    leverage: 
      min: 2.0
      max: 10.0
      default: 3.0
      dynamic: true
    filters:
      tickSize: 0.1
      lotStep: 0.001
      minNotional: 5.0
```

### Risk Management (`config/risk.yaml`) - Con Flags TRAIN/LIVE
```yaml
common:
  default_levels:
    min_sl_pct: 1.0
    tp_r_multiple: 1.5
    ttl_bars_default: 180
  allow_open_without_levels_train: true   # fallback en TRAIN
  atr_fallback:
    enabled: true
    tf: "1m"
    lookback: 14
    min_sl_atr_mult: 1.2

sizing:
  risk_per_trade_pct: 1.0
  exposure_cap_leverage: 3.0

execution_costs:
  taker_fee_bps: 10
  slippage_bps: 2
```

### Rewards (`config/rewards.yaml`) - Orquestación Clara
```yaml
# Eventos principales
core_events:
  tp_reward: 1.0
  sl_penalty: -0.5
  bankruptcy: -10.0

# Shaping por step
shaping:
  holding_positive_equity:
    enabled: true
    every_bars: 10
    reward: 0.1
  inactivity:
    enabled: true
    every_bars: 100
    penalty: -0.01

# Sistemas avanzados
volatility_scaled_pnl:
  enabled: true
  tf: "1m"
  atr_period: 14
  atr_mult: 1.5
  weight: 0.2

drawdown_penalty:
  enabled: true
  dd_threshold_ratio: 0.5
  weight: 0.3

mtf_alignment:
  enabled: true
  higher_tf: ["1h","4h"]
  agree_bonus: 0.15
  disagree_penalty: 0.075

# Clipping
clipping:
  per_step: [-0.2, 0.2]
  per_close: [-1.0, 1.5]
```

## 📊 Datos

### Descarga de Datos Históricos
```bash
# Script de descarga
python data_pipeline/scripts/download_history.py --symbol BTCUSDT --tfs 1m 5m 15m 1h --years 5
```

### Formato Parquet Esperado por TF
**Columnas requeridas**: `[timestamp, open, high, low, close, volume]`
- **timestamp**: Unix timestamp en milisegundos (UTC)
- **open, high, low, close**: Precios en float64
- **volume**: Volumen en float64
- **tz**: UTC (timezone)
- **epoch**: Unix timestamp en ms

### Validaciones Rápidas
- **Mínimo N filas por TF**: 1000+ barras por timeframe
- **Timestamps contiguos**: Sin gaps mayores a 2x el intervalo del TF
- **Integridad de datos**: OHLC válidos, volumen >= 0

### Estructura de Datos
```
data/
├── BTCUSDT/
│   ├── raw/           # Datos descargados
│   ├── aligned/       # Datos alineados por TF
│   └── packages/      # Paquetes procesados
```

## 🧠 Sistema de Rewards

El bot utiliza un sistema modular de rewards/penalties:

- **Take Profit Rewards**: Recompensas por trades exitosos
- **Stop Loss Penalties**: Penalizaciones por pérdidas
- **Volatility Scaling**: PnL normalizado por volatilidad
- **Drawdown Penalties**: Penalización por drawdown intra-trade
- **Execution Costs**: Penalización por fees y slippage
- **MTF Alignment**: Bonus por alineación multi-timeframe
- **Time Efficiency**: Recompensa por eficiencia temporal
- **Overtrading Penalties**: Penalización por sobre-operar
- **Exploration Bonus**: Bonus por explorar nuevas combinaciones
- **Progress Milestones**: Recompensas por hitos de progreso

## 🔄 Matriz de Modos (Train/Backtest/Live)

| Aspecto | Train | Backtest | Live |
|---------|-------|----------|------|
| **Fuente de datos** | Históricos + vecenv | Históricos deterministas | WS/REST |
| **Fees/Slippage** | Simulados (config) | Simulados realistas | Broker reales |
| **Gating MTF** | Opcional con fallback | Obligatorio | Obligatorio |
| **Fallback SL/TP/TTL** | ON | ON | OFF (recomendado) |
| **Persistencia estrategias** | ON | Opcional | ON |
| **Deduplicación** | ON | ON | ON |
| **Bankruptcy Mode** | Soft Reset | Soft Reset | Hard Reset |
| **Leverage Dinámico** | ON | ON | ON |
| **Rewards Avanzados** | ON | ON | OFF |
| **Logging Detallado** | ON | ON | OFF |

## 📊 Monitoreo

### Scripts de Monitoreo (`monitoring/`)
- `monitor_training.py`: Monitorea el progreso de entrenamiento
- `monitor_logs.py`: Analiza logs de entrenamiento
- `monitor_actions.py`: Monitorea acciones del agente

### Métricas Clave
- **Win Rate**: Porcentaje de trades ganadores
- **Profit Factor**: Ratio ganancias/pérdidas
- **Max Drawdown**: Máxima pérdida consecutiva
- **Sharpe Ratio**: Retorno ajustado por riesgo
- **R-Multiple**: Ratio riesgo/recompensa

## 🧪 Testing - Tests Imprescindibles

### Tests Principales
```bash
# Ejecutar todos los tests
pytest tests/

# Tests específicos imprescindibles
pytest tests/test_rewards_map.py          # Cada evento produce el reward esperado
pytest tests/test_sizing_filters.py       # Respeta minNotional, lotStep, exposición por leverage
pytest tests/test_gate_fallback.py        # Decisiones sin SL/TP/TTL se sanean en TRAIN
pytest tests/test_ledger_invariants.py    # Equity/fees/pnls consistentes
pytest tests/test_mtf_alignment.py        # Bonus/penalty por alineación HTF
pytest tests/test_dup_guard.py            # Deduplicación se limpia en cambio de bar/cierre
```

### Tests de Consistencia
```bash
# Test de consistencia del ledger
pytest tests/test_ledger_consistency.py

# Test de lógica de rewards
pytest tests/test_reward_logic.py

# Test de métricas de runs
pytest tests/test_run_metrics.py

# Test de alineación sin duplicados
pytest tests/test_align_no_dupes.py
```

### Tests de Integración
```bash
# Test de pipeline completo
pytest tests/test_full_pipeline.py

# Test de configuración YAML
pytest tests/test_config_validation.py

# Test de timestamps UTC
pytest tests/test_utc_timestamps.py
```

## 📈 Rendimiento

### Métricas de Entrenamiento
- **Episodios**: Número de runs completados
- **Steps**: Pasos totales de entrenamiento
- **Reward Promedio**: Reward promedio por episodio
- **Trades por Episodio**: Número de trades por run

### Optimización
- **Hyperparameter Tuning**: Usando Optuna
- **Curriculum Learning**: Entrenamiento progresivo
- **Strategy Persistence**: Persistencia de estrategias exitosas

## 🚨 Troubleshooting Real

### Tabla de Síntomas ↔ Causa ↔ Fix

| Síntoma | Causa | Fix |
|---------|-------|-----|
| **BYPASS POLICY: sl=None tp=None ttl_bars=0** | No corre el saneador antes del gate | Activar fallback ATR/%, TTL por defecto y usar la decisión saneada en el gate |
| **MIN_NOTIONAL_BLOCKED continuo** | Qty no alcanza minNotional | Forzar qty al minNotional en TRAIN / revisar lotStep |
| **Few/FPS bajos** | Logging excesivo o entorno lento | Bajar logging a low, usar SubprocVecEnv |
| **No trades ejecutados** | Risk manager bloquea todas las decisiones | Revisar min_sl_pct, tp_r_multiple, allow_open_without_levels_train |
| **Bankruptcy sin trades** | Balance inicial muy bajo o fees altos | Aumentar balance inicial, reducir fees simulados |
| **Rewards inconsistentes** | Configuración YAML incorrecta | Validar rewards.yaml, verificar tipos de datos |
| **Datos no encontrados** | Estructura de datos incorrecta | Verificar data/BTCUSDT/raw/, ejecutar download_history.py |
| **Modelo no carga** | Modelo corrupto o versión incompatible | Eliminar modelo, reentrenar desde cero |
| **Timestamps incorrectos** | Zona horaria no configurada | Verificar UTC en datos, usar timestamp_utils.py |
| **Memory leak** | Objetos no liberados en loops | Revisar garbage collection, cerrar archivos |

### Comandos de Debug
```bash
# Verificar estado del sistema
python utils/check_status.py

# Debug detallado
python utils/debug_detailed.py

# Verificar logs con timestamps UTC
python -m utils.example_utc_logs

# Monitorear acciones en tiempo real
python monitoring/monitor_actions.py
```

## 🔧 Desarrollo

### Estructura de Código
- **Modular**: Cada componente es independiente
- **Configurable**: Todo configurable via YAML
- **Testeable**: Tests unitarios para cada módulo
- **Extensible**: Fácil añadir nuevos sistemas

### Contribuir
1. Fork el proyecto
2. Crear feature branch
3. Añadir tests
4. Hacer pull request

## 📚 Documentación

- `docs/`: Documentación detallada
- `base_env/README.md`: Documentación del entorno base
- `train_env/README.md`: Documentación del entorno de entrenamiento

## 🚨 Disclaimer

Este software es para fines educativos y de investigación. El trading con criptomonedas conlleva riesgos significativos. No se garantiza rentabilidad y se recomienda usar solo capital que puedas permitirte perder.

## 📄 Licencia

[Especificar licencia]

## 📋 Comandos del Sistema

Para una lista completa de comandos disponibles, consulta:
```bash
cat COMANDOS_SISTEMA.txt
```

### Comandos Principales
```bash
# Entrenamiento completo
python app.py run

# Entrenamiento con GUI
python app.py run --gui

# Solo interfaz gráfica
python app.py gui

# Mostrar configuraciones
python app.py config

# Entrenar modelo PPO
python scripts/train_ppo.py

# Monitorear progreso
python scripts/watch_progress.py

# Verificar mejor run
python scripts/check_best_run.py
```

### Gestión de Datos
```bash
# Descargar datos históricos
python data_pipeline/scripts/download_history.py

# Validar datos
python data_pipeline/scripts/validate_history.py

# Alinear paquetes
python data_pipeline/scripts/align_package.py
```

### Monitoreo y Debug
```bash
# Monitorear acciones
python monitoring/monitor_actions.py

# Monitorear logs
python monitoring/monitor_logs.py

# Verificar estado
python utils/check_status.py

# Debug detallado
python utils/debug_detailed.py
```

## 🤝 Soporte

Para soporte técnico o preguntas:
- Crear issue en GitHub
- Revisar documentación en `docs/`
- Consultar logs en `logs/`
- Usar comandos de troubleshooting
- Revisar `COMANDOS_SISTEMA.txt` para referencia completa
