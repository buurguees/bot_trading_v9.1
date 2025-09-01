# BaseEnv v9.1 — Entorno Canónico de Trading (Spot & Futuros)

## 0) Propósito
`BaseEnv` es el **núcleo único** que usan los modos **train / backtest / live**.  
Su misión es construir la observación de mercado multi-timeframe, analizar señales (técnicos + SMC), **tomar decisiones operativas** (abrir/cerrar, TP/SL, trailing, TTL) con **gestión de riesgo autónoma** y llevar la **contabilidad** (balances, PnL, fees, drawdowns).  

> **Nota:** el slippage no se calcula aquí.  
> - En `train` y `backtest` lo aplica el wrapper/adapter.  
> - En `live` llega del exchange.

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

## 8) Gestión de riesgo autónoma

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

## 10) KPIs y métricas
- **Por step**: exposición, PnL UR, equity, drawdown, confidence, latencia/datos.  
- **Por trade**: R múltiplo, holding time, heat (tiempo en pérdida), eficiencia (MFE/MAE), fees netas.  
- **Por sesión**: PnL total, Sharpe/Sortino/Calmar, win-rate, profit factor, maxDD, exposición/lev. promedio.

---

## 11) Documentos relacionados
- `SPEC.md`: versión extendida de esta especificación.  
- `SCHEMAS.md`: definición de obs/action/eventos.  
- `TEST_PLAN.md`: plan de pruebas unitarias/funcionales.  
- `CHANGELOG.md`: cambios de contrato entre versiones.

---
