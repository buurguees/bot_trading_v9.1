# 📁 Configuración del Sistema

Este directorio contiene todos los archivos de configuración del sistema de trading bot. Cada archivo YAML define parámetros específicos para diferentes componentes del sistema.

## 🗂️ Estructura de Archivos

### 🔧 **settings.yaml** - Configuración Global
Configuración base del proyecto que afecta a todo el sistema.

```yaml
environment: dev           # dev | paper | live
logging_level: INFO        # DEBUG | INFO | WARNING | ERROR
seed: 42                  # Semilla para reproducibilidad
timezone: "UTC"           # Zona horaria del sistema
results_dir: "runs"       # Directorio para resultados
models_dir: "models"      # Directorio para modelos guardados
```

**Variables:**
- `environment`: Entorno de ejecución (desarrollo, paper trading, live)
- `logging_level`: Nivel de logging del sistema
- `seed`: Semilla para generadores aleatorios
- `timezone`: Zona horaria para timestamps
- `results_dir`: Directorio para almacenar resultados de backtests
- `models_dir`: Directorio para modelos entrenados

### 🚀 **train.yaml** - Configuración de Entrenamiento
Parámetros específicos para el entrenamiento con PPO (Proximal Policy Optimization).

```yaml
seed: 42
log_dir: "logs/ppo_v1"

data:
  root: "data"
  symbols: ["BTCUSDT"]
  market: "spot"
  stage: "aligned"
  tfs: ["1m","5m"]
  months_back: 36

env:
  n_envs: 4
  warmup_bars: 5000
  reward_yaml: "config/rewards.yaml"
  chronological: true
  initial_balance: 1000.0
  target_balance: 1000000.0

ppo:
  total_timesteps: 15000000
  n_steps: 2048
  batch_size: 8192
  learning_rate: 3.0e-4
  gamma: 0.999
  gae_lambda: 0.95
  clip_range: 0.2
  ent_coef: 0.0
  vf_coef: 0.5
  n_epochs: 4
  tensorboard_log: "logs/tb"

models:
  root: "models"
  overwrite: true
  save_every_steps: 500000

logging:
  run_dir: "logs/runs"
  checkpoint_every_steps: 500000
```

**Secciones principales:**

#### 📊 **Data**
- `root`: Directorio raíz de datos
- `symbols`: Lista de símbolos a entrenar
- `market`: Tipo de mercado (spot, futures)
- `stage`: Etapa de datos (raw, aligned)
- `tfs`: Timeframes disponibles
- `months_back`: Cuántos meses de datos históricos usar

#### 🎯 **Environment**
- `n_envs`: Número de entornos paralelos
- `warmup_bars`: Barras de calentamiento antes de empezar
- `reward_yaml`: Archivo de configuración de rewards
- `chronological`: Si usar datos en orden cronológico
- `initial_balance`: Balance inicial para entrenamiento
- `target_balance`: Balance objetivo

#### 🤖 **PPO (Proximal Policy Optimization)**
- `total_timesteps`: Total de pasos de entrenamiento
- `n_steps`: Pasos por episodio
- `batch_size`: Tamaño del batch
- `learning_rate`: Tasa de aprendizaje
- `gamma`: Factor de descuento
- `gae_lambda`: Parámetro GAE (Generalized Advantage Estimation)
- `clip_range`: Rango de clipping para PPO
- `ent_coef`: Coeficiente de entropía
- `vf_coef`: Coeficiente de función de valor
- `n_epochs`: Épocas por batch

#### 💾 **Models**
- `root`: Directorio para guardar modelos
- `overwrite`: Si sobrescribir modelos existentes
- `save_every_steps`: Cada cuántos pasos guardar

#### 📝 **Logging**
- `run_dir`: Directorio para logs de runs
- `checkpoint_every_steps`: Cada cuántos pasos hacer checkpoint

### 🎁 **rewards.yaml** - Configuración de Rewards
Define cómo se calculan los rewards para el entrenamiento de RL.

```yaml
tiers:
  pos: [[0, 1, 0.1], [1, 3, 0.5], [3, 10, 1.0], [10, 100, 2.0]]
  neg: [[0, 1, -0.1], [1, 3, -0.5], [3, 10, -1.0], [10, 100, -2.0]]

bonuses:
  tp_hit: 1.0
  sl_hit: -0.5

weights:
  realized_pnl: 1.0
  unrealized_pnl: 0.1
  r_multiple: 0.5
  risk_efficiency: 0.3
  time_penalty: -0.01
  trade_cost: -0.1
  dd_penalty: -0.2

reward_clip: [-10.0, 10.0]
```

**Componentes:**

#### 🏆 **Tiers (Tramos)**
- `pos`: Rewards por ROI positivo (ganancias)
- `neg`: Rewards por ROI negativo (pérdidas)
- Formato: `[min_roi%, max_roi%, reward_value]`

#### 🎯 **Bonuses**
- `tp_hit`: Bonus por alcanzar take profit
- `sl_hit`: Penalización por alcanzar stop loss

#### ⚖️ **Weights (Pesos)**
- `realized_pnl`: Peso del PnL realizado
- `unrealized_pnl`: Peso del PnL no realizado
- `r_multiple`: Peso del R-multiple
- `risk_efficiency`: Peso de la eficiencia de riesgo
- `time_penalty`: Penalización por tiempo en posición
- `trade_cost`: Coste por operación
- `dd_penalty`: Penalización por drawdown

#### 🔒 **Reward Clip**
- Límites mínimo y máximo para el reward final

### 🎯 **hierarchical.yaml** - Análisis Jerárquico
Configuración del sistema de análisis jerárquico multi-timeframe.

```yaml
direction_tfs: ["1h", "4h", "1d"]
confirm_tfs: ["15m", "1h"]
execute_tfs: ["1m", "5m"]
confidence_threshold: 0.7
```

**Parámetros:**
- `direction_tfs`: Timeframes para determinar dirección
- `confirm_tfs`: Timeframes para confirmación
- `execute_tfs`: Timeframes para ejecución
- `confidence_threshold`: Umbral mínimo de confianza

### 🔄 **pipeline.yaml** - Pipeline de Features
Configuración del pipeline de cálculo de indicadores técnicos.

```yaml
strict_alignment: true
indicators:
  - ema20
  - ema50
  - rsi14
  - atr14
  - macd
  - bollinger_bands
```

**Parámetros:**
- `strict_alignment`: Si exigir alineación estricta de timeframes
- `indicators`: Lista de indicadores a calcular

### ⚠️ **risk.yaml** - Gestión de Riesgo
Parámetros para el sistema de gestión de riesgo.

```yaml
max_position_size: 0.1
max_leverage: 3.0
stop_loss_atr_multiplier: 1.5
take_profit_atr_multiplier: 2.0
max_drawdown: 0.2
```

**Parámetros:**
- `max_position_size`: Tamaño máximo de posición (% del capital)
- `max_leverage`: Apalancamiento máximo
- `stop_loss_atr_multiplier`: Multiplicador ATR para SL
- `take_profit_atr_multiplier`: Multiplicador ATR para TP
- `max_drawdown`: Drawdown máximo permitido

### 💰 **fees.yaml** - Comisiones y Fees
Configuración de comisiones del broker.

```yaml
maker_fee: 0.001
taker_fee: 0.001
min_fee: 0.0001
```

**Parámetros:**
- `maker_fee`: Comisión por órdenes maker
- `taker_fee`: Comisión por órdenes taker
- `min_fee`: Comisión mínima

### 🎯 **symbols.yaml** - Metadatos de Símbolos
Información específica de cada símbolo de trading.

```yaml
BTCUSDT:
  min_qty: 0.001
  price_precision: 2
  qty_precision: 3
  min_notional: 10.0
```

**Parámetros por símbolo:**
- `min_qty`: Cantidad mínima transaccionable
- `price_precision`: Precisión del precio
- `qty_precision`: Precisión de la cantidad
- `min_notional`: Valor mínimo de la operación

### 📊 **oms.yaml** - Order Management System
Configuración del sistema de gestión de órdenes.

```yaml
slippage: 0.0001
execution_delay: 0.1
partial_fills: true
```

**Parámetros:**
- `slippage`: Slippage simulado
- `execution_delay`: Delay de ejecución (segundos)
- `partial_fills`: Si permitir fills parciales

## 🔧 **Uso y Modificación**

### 📝 **Modificar Configuración**
1. **Entrenamiento**: Ajusta `train.yaml` para cambiar parámetros de PPO
2. **Rewards**: Modifica `rewards.yaml` para cambiar la función de reward
3. **Riesgo**: Ajusta `risk.yaml` para cambiar límites de riesgo
4. **Features**: Modifica `pipeline.yaml` para cambiar indicadores

### 🔍 **Validación**
- Todos los archivos YAML deben tener sintaxis válida
- Los valores numéricos deben estar en rangos razonables
- Las rutas de archivos deben existir en el sistema

### 📊 **Monitoreo**
- Los logs se guardan en `logs/` según la configuración
- Los modelos se guardan en `models/` según la configuración
- Los resultados se guardan en `runs/` según la configuración

## 🚀 **Próximos Pasos**

1. **Ajustar parámetros** según tu estrategia
2. **Optimizar rewards** para mejor convergencia
3. **Ajustar gestión de riesgo** según tu perfil
4. **Monitorear logs** durante entrenamiento
5. **Validar configuraciones** antes de live trading
