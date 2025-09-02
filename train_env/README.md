# 🎯 Entorno de Entrenamiento (Training Environment)

Este directorio contiene todos los componentes necesarios para convertir el entorno base de trading en un entorno compatible con Gymnasium y Stable-Baselines3 para entrenamiento de Reinforcement Learning (RL).

## 🗂️ Estructura de Archivos

### 🎮 **gym_wrapper.py** - Wrapper de Gymnasium
Convierte el `BaseTradingEnv` en un entorno compatible con Gymnasium.

**Características principales:**
- **Action Space**: 5 acciones discretas (0-4)
- **Observation Space**: Vector de features aplanado
- **Reward Shaping**: Integración con `RewardShaper`
- **Strategy Logging**: Registro de eventos de trading

**Acciones disponibles:**
- `0`: Dejar que la policy decida (sin override)
- `1`: Cerrar todas las posiciones
- `2`: Bloquear aperturas de nuevas posiciones
- `3`: Forzar apertura LONG con SL/TP automáticos
- `4`: Forzar apertura SHORT con SL/TP automáticos

**Observación aplanada:**
```python
def _obs_dim(self) -> int:
    per_tf = 7      # 7 features por timeframe
    pos = 4         # 4 features de posición
    ana = 2         # 2 features de análisis
    return len(self.tfs)*per_tf + pos + ana
```

**Features por timeframe:**
1. `close` - Precio de cierre
2. `ema20` - Media móvil exponencial 20
3. `ema50` - Media móvil exponencial 50
4. `rsi14` - RSI 14 períodos
5. `atr14` - ATR 14 períodos
6. `macd_hist` - Histograma MACD
7. `bb_p` - Posición en Bandas de Bollinger

**Features de posición:**
1. `side` - Lado de la posición (0, +1, -1)
2. `qty` - Cantidad de la posición
3. `entry_price` - Precio de entrada
4. `unrealized_pnl` - PnL no realizado

**Features de análisis:**
1. `confidence` - Nivel de confianza de la estrategia
2. `side_hint` - Sugerencia de dirección

### 🎁 **reward_shaper.py** - Moldeador de Rewards
Sistema avanzado de cálculo de rewards basado en múltiples factores.

**Componentes del reward:**

#### 🏆 **Tiers por ROI (Tramos)**
```yaml
tiers:
  pos: [[0, 1, 0.1], [1, 3, 0.5], [3, 10, 1.0], [10, 100, 2.0]]
  neg: [[0, 1, -0.1], [1, 3, -0.5], [3, 10, -1.0], [10, 100, -2.0]]
```
- Rewards escalonados según el porcentaje de ROI
- Diferentes escalas para ganancias y pérdidas
- Formato: `[min_roi%, max_roi%, reward_value]`

#### 🎯 **Bonuses por TP/SL**
```yaml
bonuses:
  tp_hit: 1.0      # Bonus por alcanzar take profit
  sl_hit: -0.5     # Penalización por alcanzar stop loss
```

#### ⚖️ **Pesos de componentes**
```yaml
weights:
  realized_pnl: 1.0        # PnL realizado (del entorno)
  unrealized_pnl: 0.1      # PnL no realizado (guía suave)
  r_multiple: 0.5          # R-multiple del trade
  risk_efficiency: 0.3     # Eficiencia de riesgo
  time_penalty: -0.01      # Penalización por tiempo en posición
  trade_cost: -0.1         # Coste por operación
  dd_penalty: -0.2         # Penalización por drawdown
```

**Cálculo de reward:**
```python
def compute(self, obs, base_reward, events):
    # Componentes continuos
    reward = (self.w_realized * realized_usd + 
              self.w_unreal * unreal_usd + 
              self.w_time * time_penalty + 
              self.w_dd * drawdown_penalty)
    
    # Eventos de cierre
    if close_event:
        roi_pct = close_event.get("roi_pct", 0.0)
        r_mult = close_event.get("r_multiple", 0.0)
        risk_pct = close_event.get("risk_pct", 0.0)
        
        # Reward por ROI según tiers
        if roi_pct >= 0:
            reward += self._tier_value(self.tiers_pos, roi_pct)
        else:
            reward += self._tier_value(self.tiers_neg, abs(roi_pct))
        
        # Bonus por TP/SL
        if tp_hit: reward += self.bon_tp
        if sl_hit: reward += self.bon_sl
        
        # Refuerzos por calidad
        reward += self.w_rmult * r_mult
        if risk_pct > 0:
            risk_eff = abs(roi_pct) / risk_pct
            reward += self.w_risk_eff * risk_eff
    
    return self._clip(reward)
```

### 📊 **strategy_logger.py** - Logger de Estrategia
Registra todos los eventos de trading para análisis posterior.

**Funcionalidades:**
- **Append Events**: Añade múltiples eventos al log
- **File Management**: Manejo automático de archivos de log
- **Event Tracking**: Seguimiento de OPEN, CLOSE, TP_HIT, SL_HIT

### 🔄 **vec_factory.py** - Factory de Entornos Vectorizados
Crea entornos vectorizados para entrenamiento paralelo.

**Características:**
- **Multi-Environment**: Múltiples entornos paralelos
- **Chronological**: Opción de datos cronológicos vs aleatorios
- **Warmup**: Barras de calentamiento antes del entrenamiento

### ⏰ **vec_factory_chrono.py** - Factory Cronológica
Versión especializada para entrenamiento con datos cronológicos.

**Ventajas:**
- **Realistic Training**: Entrenamiento más realista
- **Temporal Consistency**: Consistencia temporal entre entornos
- **Market Conditions**: Simula condiciones de mercado reales

### 📈 **dataset.py** - Gestión de Datos
Maneja la carga y preparación de datos para entrenamiento.

**Funcionalidades:**
- **Data Loading**: Carga de datos históricos
- **Timeframe Alignment**: Alineación de múltiples timeframes
- **Feature Computation**: Cálculo de indicadores técnicos
- **Data Validation**: Validación de integridad de datos

### 📝 **callbacks.py** - Callbacks de Entrenamiento
Callbacks personalizados para monitoreo y control del entrenamiento.

**Callbacks incluidos:**
- **Checkpoint Callback**: Guardado automático de modelos
- **Logging Callback**: Logging de métricas de entrenamiento
- **Early Stopping**: Parada temprana basada en criterios
- **Custom Metrics**: Métricas personalizadas de trading

### 🎯 **strategy_aggregator.py** - Agregador de Estrategias
Combina múltiples estrategias o modelos para mejor performance.

**Funcionalidades:**
- **Ensemble Methods**: Métodos de ensemble para múltiples modelos
- **Strategy Selection**: Selección dinámica de estrategias
- **Performance Tracking**: Seguimiento de performance por estrategia
- **Risk Management**: Gestión de riesgo agregada

## 🚀 **Uso del Sistema**

### 1. **Configuración Básica**
```python
from train_env.gym_wrapper import TradingGymWrapper
from train_env.reward_shaper import RewardShaper
from base_env.base_env import BaseTradingEnv

# Crear entorno base
base_env = BaseTradingEnv(cfg, broker, oms)

# Crear wrapper de gym
gym_env = TradingGymWrapper(
    base_env=base_env,
    reward_yaml="config/rewards.yaml",
    tfs=["1m", "5m"],
    strategy_log_path="logs/strategy.log"
)
```

### 2. **Entrenamiento con PPO**
```python
from stable_baselines3 import PPO
from train_env.vec_factory import create_vec_env

# Crear entorno vectorizado
vec_env = create_vec_env(
    env_class=TradingGymWrapper,
    n_envs=4,
    env_kwargs={...}
)

# Entrenar modelo
model = PPO("MlpPolicy", vec_env, verbose=1)
model.learn(total_timesteps=15000000)
```

### 3. **Configuración de Rewards**
```yaml
# config/rewards.yaml
tiers:
  pos: [[0, 1, 0.1], [1, 3, 0.5], [3, 10, 1.0]]
  neg: [[0, 1, -0.1], [1, 3, -0.5], [3, 10, -1.0]]

weights:
  realized_pnl: 1.0
  r_multiple: 0.5
  risk_efficiency: 0.3
```

## 🔧 **Personalización**

### **Modificar Rewards**
1. Ajusta los `tiers` para diferentes niveles de ROI
2. Modifica los `weights` para cambiar la importancia de cada componente
3. Añade nuevos componentes de reward en `reward_shaper.py`

### **Cambiar Features**
1. Modifica `_obs_dim()` en `gym_wrapper.py`
2. Ajusta `_flatten_obs()` para incluir nuevos indicadores
3. Actualiza la configuración en `pipeline.yaml`

### **Añadir Callbacks**
1. Crea nuevos callbacks en `callbacks.py`
2. Implementa lógica personalizada de monitoreo
3. Integra con el sistema de logging existente

## 📊 **Monitoreo y Logging**

### **Logs de Estrategia**
- **Eventos de Trading**: OPEN, CLOSE, TP_HIT, SL_HIT
- **Métricas de Performance**: ROI%, R-multiple, Risk%
- **Análisis de Riesgo**: Drawdown, exposición, eficiencia

### **Métricas de Entrenamiento**
- **Reward Components**: Desglose de cada componente del reward
- **Trading Metrics**: Estadísticas de trading por episodio
- **Risk Metrics**: Métricas de riesgo y gestión

### **TensorBoard Integration**
- **Training Curves**: Curvas de entrenamiento en tiempo real
- **Custom Metrics**: Métricas personalizadas de trading
- **Hyperparameter Tracking**: Seguimiento de hiperparámetros

## 🎯 **Mejores Prácticas**

### **Configuración de Rewards**
1. **Balance**: Equilibra rewards positivos y negativos
2. **Escalado**: Usa escalas apropiadas para cada componente
3. **Consistencia**: Mantén consistencia entre diferentes timeframes

### **Entrenamiento**
1. **Warmup**: Usa suficientes barras de calentamiento
2. **Validation**: Valida en datos no vistos durante entrenamiento
3. **Monitoring**: Monitorea métricas de trading, no solo reward

### **Risk Management**
1. **Position Sizing**: Ajusta tamaños de posición según el modelo
2. **Stop Losses**: Usa SL dinámicos basados en ATR
3. **Drawdown Limits**: Establece límites claros de drawdown

## 🚀 **Próximos Pasos**

1. **Optimizar Rewards**: Ajustar función de reward para mejor convergencia
2. **Feature Engineering**: Añadir indicadores técnicos más sofisticados
3. **Ensemble Methods**: Implementar métodos de ensemble para múltiples modelos
4. **Live Trading**: Preparar para transición a trading en vivo
5. **Performance Analysis**: Análisis profundo de métricas de trading
