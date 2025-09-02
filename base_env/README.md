# BaseEnv v9.1 — Entorno Canónico de Trading (Spot & Futuros)

## 0) Propósito
`BaseEnv` es el **núcleo único** que usan los modos **train / backtest / live**.  
Su misión es construir la observación de mercado multi-timeframe, analizar señales (técnicos + SMC), **tomar decisiones operativas** (abrir/cerrar, TP/SL, trailing, TTL) con **gestión de riesgo autónoma** y llevar la **contabilidad** (balances, PnL, fees, drawdowns).  

> **Nota:** el slippage no se calcula aquí.  
> - En `train` y `backtest` lo aplica el wrapper/adapter.  
> - En `live` llega del exchange.

### 🆕 **Nuevas Funcionalidades (v9.1.1)**
- **Sistema de Logging Completo**: `RunLogger` para tracking de runs completos
- **Gestión de Balances Configurable**: `initial_cash` y `target_cash` configurables
- **Eventos Ultra-Enriquecidos**: Métricas avanzadas en OPEN y CLOSE
- **Tracking de Equity**: Monitoreo de balances y objetivos financieros
- **Logging de Inicio/Fin de Run**: Registro completo de performance

---

## 1) Alcance (qué hace)
- **Ingesta** y **alineación MTF** coherente (1D/4H → 1H/15M → 5M/1M).
- **Features** técnicos (EMA/MA, RSI, MACD, ATR, Bollinger, ADX, OBV, Estocástico, SuperTrend…) y **SMC** (BOS/CHOCH, Order Blocks, FVG, Liquidity zones, POI).
- **Análisis jerárquico**: dirección (macro), confirmación (meso), ejecución (micro) → `confidence` y confluencias.
- **Motor de decisiones** (policy): convierte señales en acción concreta (side, sizing, SL/TP, trailing, TTL).
- **Gestión de riesgo autónoma**: sizing por riesgo, exposición, apalancamiento (Futuros ≤ 3x inicial), circuit breakers.
- **Ejecución lógica** de la operación sobre el **estado interno** y **contabilidad** (balances, PnL, fees).
- **Eventos** y `info` trazables (aperturas, cierres, SL/TP, violaciones de reglas, breakers, etc.).

### Límites (qué NO hace)
- No habla directamente con el exchange: usa **adapters** (OMS sim/paper/live).
- No descarga datos crudos: consume datos ya preparados por un **DataBroker**.
- No calcula **slippage**: es responsabilidad del wrapper/adapter de cada modo.

---

## 2) Principios de diseño
- **Un solo core para todo**: mismo `BaseEnv` en train/backtest/live (cambian adapters).
- **Determinista** en backtest/train (misma semilla ⇒ mismo resultado).
- **Desacoplado por contratos**: DataBroker (entrada), OMS (ejecución), Risk (política interna), EventBus (salida).
- **Fail-safe**: si fallan datos/alineación, bloquea aperturas y prioriza cierres.

---

## 3) Flujo por step
1. **Avance temporal** del `DataBroker` (siguiente bar en TF base).
2. **Alineación MTF** del bar actual para TFs requeridos.
3. **Cálculo de features** (TA + SMC) y normalización.
4. **Análisis jerárquico** → `by_tf` + `confidence` + confluencias.
5. **Gating**: deduplicación y umbrales.  
   Si no cumple, deniega aperturas (cierre puede tener fallback).
6. **Decisión** (policy): `side`, sizing, `SL`, `TP`, `trailing`, `TTL` (o `reduce_only`/`close_all`).
7. **Aplicación** sobre el estado: abrir/cerrar/ajustar; actualizar **balances**, `PnL` R/UR, `MFE/MAE`, `DD`.
8. **Eventos** y `info`: motivos, métricas, calidad de datos; listo para logs/dashboard.

---

## 4) Entradas y salidas

### Entradas
- **Barras OHLCV por TF** alineadas a `bar_time` del TF base.
- **Precio de referencia** del bar en TF base.
- **Filtros de símbolo** (tickSize, lotStep, minNotional, tamaño contrato en Futuros).
- **Parámetros** de riesgo, jerárquico, pipeline, fees, límites y breakers.

### Salidas
- **Acciones** decididas (side, qty, SL, TP, trailing, TTL, parciales/close-only).
- **Eventos de dominio** (aperturas, cierres, SL/TP, fallos de gating, breakers, etc.).
- **Métricas** por step y por trade: PnL R/UR, MFE/MAE, exposure, DD, KPIs agregables.

---

## 5) Multi-Timeframe (MTF) & Jerárquico
- **Capas**:
  - Dirección: **1D, 4H** (regime/tendencia/estructura SMC).
  - Confirmación: **1H, 15M** (momentum, divergencias, fuerza ADX, salud de estructura).
  - Ejecución: **5M, 1M** (timing: OB tap, cierre FVG, break & retest, RSI reset).
- **Alineación**: todos los TF usan su **última barra cerrada ≤ bar_time** del TF base.  
  - Modo **estricto**: faltar un TF requerido ⇒ **no abrir** (cierre puede permitirse con fallback).
- **Confluencias**: reglas tipo **“2-de-3 por capa”** y ponderaciones (1D > 4H, 1H > 15M, 5M > 1M).
- **`confidence ∈ [0,1]`**: agregado de señales ponderadas; **umbral** mínimo de apertura configurable.

---

## 6) Features técnicos & SMC
- **Técnicos**: EMA/MA, RSI, MACD, ATR, Bollinger, ADX/DI, OBV, Estocástico, SuperTrend.
- **SMC**:  
  - **Estructura**: BOS/CHOCH, swing high/low, HH/HL/LH/LL.  
  - **Zonas**: Order Blocks, FVG, liquidity sweeps (EQ highs/lows).  
  - **Contexto**: rango activo, mid-range, imbalance, POI.
- **Normalización**: por TF y ventana; flags booleanos/numéricos para eventos SMC.

---

## 7) Motor de decisiones & Gating
- **Gating**:
  - `confidence` ≥ umbral ⇒ permitir **apertura**.  
  - **Deduplicación**: ventana en barras que impide re-entradas idénticas.  
  - **Fallbacks**: cierre permitido aunque dirección pierda fuerza, para evitar atraparse.
- **Decisión**:
  - **Abrir** long/short si confluencias + riesgo válido.  
  - **Cerrar** por señal contraria, SL/TP, TTL, trailing o circuit breaker.  
  - **Reduce-only** para parciales.  
- **Parámetros**: `TP/SL` fijos o relativos (ATR múltiplos), `TTL` máximo, trailing tras `MFE` umbral.

---

## 8) Sistema de Logging y Eventos Enriquecidos

### 🆕 **RunLogger - Tracking Completo de Runs**
```python
def __init__(self, initial_cash: float = 10000.0, target_cash: float = 1_000_000.0, run_log_dir: str = "logs/runs"):
    self._init_cash = float(initial_cash)
    self._target_cash = float(target_cash)
    self._run_logger = RunLogger(run_log_dir)
```

**Funcionalidades:**
- **Logging de Inicio**: Registra símbolo, mercado, balances iniciales y timestamp
- **Logging de Fin**: Registra balances finales y timestamp de cierre
- **Archivos JSONL**: Un archivo por run con métricas completas
- **Directorio Configurable**: `logs/runs/` por defecto

### 🎯 **Eventos OPEN Enriquecidos**
```python
self.events_bus.emit(
    "OPEN", ts=ts_now, side=("LONG" if sized.side > 0 else "SHORT"),
    qty=self.pos.qty, price=self.pos.entry_price, sl=self.pos.sl, tp=self.pos.tp,
    risk_pct=risk_pct, analysis=obs.get("analysis", {}), indicators=list(feats_exec.keys()),
    used_tfs=used_tfs
)
```

**Métricas incluidas:**
- `risk_pct`: Porcentaje de riesgo basado en distancia SL
- `analysis`: Análisis completo de la observación
- `indicators`: Lista de indicadores disponibles
- `used_tfs`: Timeframes utilizados (direction, confirm, execute)

### 📊 **Eventos CLOSE Ultra-Enriquecidos**
```python
self.events_bus.emit(
    "CLOSE", ts=ts_now, qty=qty_close, price=exit_price,
    realized_pnl=realized, entry_price=entry, entry_qty=qty_now,
    roi_pct=roi_pct, r_multiple=r_multiple, risk_pct=risk_pct,
    reason=("PARTIAL" if sized.should_close_partial else "ALL")
)
```

**Métricas calculadas:**
- `roi_pct`: Porcentaje de retorno sobre el notional
- `r_multiple`: Ratio entre PnL realizado y riesgo inicial
- `risk_pct`: Porcentaje de riesgo inicial del trade
- `realized_pnl`: PnL realizado en USD
- `entry_price/entry_qty`: Precio y cantidad de entrada
- `reason`: Razón del cierre (PARTIAL, ALL, AUTO_PARTIAL, AUTO_ALL)

### 🔧 **Gestión de Balances Configurable**
```python
def reset(self):
    self.portfolio.reset(initial_cash=self._init_cash, target_cash=self._target_cash)
    # ... logging de inicio
    self._run_logger.start(
        symbol=self.cfg.symbol_meta.symbol,
        market=self.cfg.market,
        initial_balance=self.portfolio.cash_quote,
        target_balance=self.portfolio.target_quote,
        initial_equity=self.portfolio.equity_quote,
        ts_start=int(obs["ts"])
    )
```

**Parámetros configurables:**
- `initial_cash`: Balance inicial para entrenamiento/backtest
- `target_cash`: Balance objetivo para tracking de performance
- **Tracking automático**: Equity, balances y objetivos financieros

---

## 9) Gestión de riesgo autónoma

### Común
- **Riesgo por trade** (dinámico): depende de volatilidad (ATR), liquidez y DD.  
- **Exposición máxima** por símbolo y global.  
- **Circuit breakers**:
  - **DD diario** > X% ⇒ **close-only**.  
  - **Calidad de datos** baja ⇒ pause/close-only.  
  - **Señales inconsistentes** ⇒ neutral temporal.

### Sizing (Spot)
- `riesgo_trade = risk_pct * equity_quote`  
- `dist_sl = |entry - SL|`  
- `qty = riesgo_trade / max(ε, dist_sl)`  
- Ajustes a **minNotional** y **lotStep**.

### Sizing (Futuros)
- **Apalancamiento** inicial ≤ **3x** (dinámico según volatilidad/DD).  
- `notional = qty * entry`  
- `margin_used = notional / leverage`  
- Verificar **margen mantenimiento** y límites de riesgo antes de abrir.

---

## 10) 🆕 **Tracking Temporal y Análisis de Timeframes (v9.1.2)**

### **🎯 Tracking Temporal de Posiciones**
```python
@dataclass
class PositionState:
    # ... campos existentes ...
    open_ts: Optional[int] = None      # Timestamp de apertura
    bars_held: int = 0                 # Barras que estuvo realmente abierta
```

**Funcionalidades:**
- **`open_ts`**: Registra timestamp exacto de apertura
- **`bars_held`**: Incrementa automáticamente en cada step
- **Duración real**: Calculada como `ts_close - open_ts`
- **Barras reales**: Contador de barras vs TTL configurado

### **📊 Eventos CLOSE Ultra-Enriquecidos con Temporal**
```python
self.events_bus.emit(
    "CLOSE", 
    # ... campos existentes ...
    # ← NUEVO: información temporal completa
    "open_ts": self.pos.open_ts,           # Timestamp de apertura
    "duration_ms": ts_now - self.pos.open_ts,  # Duración en milisegundos
    "bars_held": self.pos.bars_held,       # Barras que estuvo abierta
    "exec_tf": exec_tf                     # Timeframe de ejecución
)
```

**Nuevas métricas:**
- **`open_ts`**: Timestamp de apertura para correlación OPEN-CLOSE
- **`duration_ms`**: Duración real en milisegundos
- **`bars_held`**: Número de barras que realmente estuvo abierta
- **`exec_tf`**: Timeframe específico donde se ejecutó la estrategia

### **🏆 Scoring Inteligente por Timeframes y Duración**
```python
def _score_row(e: Dict[str, Any]) -> float:
    # ... scoring base ...
    
    # Bonus por timeframes preferidos (1m, 5m, 15m, 1h)
    exec_tf = e.get("exec_tf", "")
    if exec_tf in ["1m", "5m", "15m", "1h"]:
        tf_bonus = 3.0  # Bonus fuerte por timeframes preferidos
    elif exec_tf in ["4h", "1d"]:
        tf_penalty = -2.0  # Penalización por timeframes largos
    
    # Bonus por duración moderada (5-50 barras)
    bars_held = e.get("bars_held", 0)
    if 5 <= bars_held <= 50:
        duration_bonus = 2.0  # Rango óptimo
    elif bars_held < 3:
        duration_penalty = -1.0  # Muy cortas
    elif bars_held > 100:
        duration_penalty = -1.5  # Muy largas
```

**Criterios de Scoring:**
- **Timeframes preferidos**: 1m, 5m, 15m, 1h → **+3.0 bonus**
- **Timeframes evitados**: 4h, 1d → **-2.0 penalización**
- **Duración óptima**: 5-50 barras → **+2.0 bonus**
- **Duración corta**: <3 barras → **-1.0 penalización**
- **Duración larga**: >100 barras → **-1.5 penalización**

### **📈 Beneficios del Tracking Temporal**
1. **Análisis de Duración**: Comparar TTL configurado vs duración real
2. **Optimización de Timeframes**: Identificar qué TFs generan mejores resultados
3. **Estrategias Eficientes**: Favorecer estrategias de duración moderada
4. **Correlación OPEN-CLOSE**: Análisis completo de entrada a salida
5. **Backtesting Avanzado**: Métricas temporales para optimización

### TP/SL & Trailing
- **SL**: `entry ± k_sl * ATR(tf_exec)` ajustado a OB/FVG/liquidez.  
- **TP**: por **R esperado** o niveles SMC.  
- **Trailing**: activar cuando `MFE ≥ umbral` (ej. 1×ATR), con step proporcional a ATR.

---

## 9) Contabilidad, Fees y PnL

### Balances
- **Spot**: `equity_quote`, `equity_base`, `free`, `locked`.  
- **Futuros**: `balance_usdt` (o moneda de margen), **margen inicial** y **margen mantenimiento**.

### Fees
- Modelo **taker** por defecto (ej. 0.1% = 10 bps) aplicado a **open** y **close**.  
- Si el adapter/live devuelve fees exactas, se usan esas.

### Funding (Futuros)
- Aplicación vía adapter (live) o calendario simulado (backtest/train).

### Slippage
- **Nunca** dentro del core.  
- Train/backtest: wrapper aplica slippage ficticio.  
- Live: precio real del exchange.

---

## 10) Uso del Sistema Mejorado

### 🆕 **Inicialización con Nuevos Parámetros**
```python
# Configuración básica
env = BaseTradingEnv(
    cfg=config,
    broker=broker,
    oms=oms,
    initial_cash=10000.0,      # Balance inicial
    target_cash=1000000.0,     # Balance objetivo
    run_log_dir="logs/runs"    # Directorio de logs
)
```

### 📊 **Acceso a Eventos Enriquecidos**
```python
obs, reward, done, info = env.step()

# Eventos disponibles en info["events"]
for event in info["events"]:
    if event["kind"] == "OPEN":
        print(f"Abrió {event['side']} con {event['risk_pct']:.2f}% riesgo")
        print(f"Indicadores: {event['indicators']}")
        print(f"Timeframes: {event['used_tfs']}")
    
    elif event["kind"] == "CLOSE":
        print(f"Cerró con {event['roi_pct']:.2f}% ROI")
        print(f"R-multiple: {event['r_multiple']:.2f}")
        print(f"Razón: {event['reason']}")
```

### 📈 **Monitoreo de Performance**
```python
# Los logs se guardan automáticamente en logs/runs/
# Cada run genera un archivo JSONL con métricas completas
# Formato: run_{timestamp}.jsonl

# Contenido del log:
{
    "symbol": "BTCUSDT",
    "market": "spot",
    "initial_balance": 10000.0,
    "target_balance": 1000000.0,
    "initial_equity": 10000.0,
    "ts_start": 1640995200,
    "ts_end": 1641081600,
    "final_balance": 10500.0,
    "final_equity": 10500.0
}
```

### 🎯 **Integración con RewardShaper**
```python
# Los eventos enriquecidos se pueden usar directamente en el RewardShaper
# El RewardShaper recibe automáticamente:
# - roi_pct: Para cálculo de tiers
# - r_multiple: Para refuerzos de calidad
# - risk_pct: Para eficiencia de riesgo
# - realized_pnl: Para rewards base
```

---

## 11) KPIs y métricas
- **Por step**: exposición, PnL UR, equity, drawdown, confidence, latencia/datos.  
- **Por trade**: R múltiplo, holding time, heat (tiempo en pérdida), eficiencia (MFE/MAE), fees netas.  
- **Por sesión**: PnL total, Sharpe/Sortino/Calmar, win-rate, profit factor, maxDD, exposición/lev. promedio.

---

## 12) Documentos relacionados
- `SPEC.md`: versión extendida de esta especificación.  
- `SCHEMAS.md`: definición de obs/action/eventos.  
- `TEST_PLAN.md`: plan de pruebas unitarias/funcionales.  
- `CHANGELOG.md`: cambios de contrato entre versiones.

---
