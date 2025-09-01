# SPEC.md
# Descripción: Especificación extendida del entorno base (Spot & Futuros): responsabilidades, flujos, algoritmos y criterios.
# Ubicación: base_env/docs/SPEC.md

# BaseEnv v9.1 — Especificación extendida (Spot & Futuros)

## 0) Propósito
BaseEnv es el núcleo único para train/backtest/live. Construye observaciones MTF, calcula features (TA+SMC), realiza análisis jerárquico (dirección/confirmación/ejecución), decide acciones (abrir/cerrar, TP/SL, trailing, TTL) con gestión de riesgo autónoma, y lleva contabilidad (balances, PnL, fees, DD). **Slippage fuera del core**.

## 1) Responsabilidades
- Ingesta de datos (vía DataBroker) y **alineación multi-timeframe** (1D-4H / 1H-15M / 5M-1M).
- Cálculo de **features técnicos** y **SMC** (normalizados).
- **Análisis jerárquico** → señales por TF + confidence + confluencias.
- **Gating** y política de decisión (deduplicación, umbrales).
- **Gestión de riesgo autónoma** (spot/futuros ≤ 3x): sizing por riesgo/SL, exposición, circuit breakers.
- **Ejecución lógica** sobre el estado interno (posición y cartera) y **contabilidad** (fees y PnL).
- Emisión de **eventos de dominio** y métricas para UI/logs.

### Límites
- No descarga datos crudos; consume de data/{SYMBOL}/ (Parquet) o live adapter.
- No calcula slippage (lo aplica wrapper/OMS).
- No persiste en BBDD (entrega eventos/metricas para que otro módulo persista).

## 2) Flujo por step (resumen)
1. Broker avanza bar del TF base → bar_time.
2. Alineación MTF a ese bar_time.
3. Features TA/SMC (ventanas por TF) → normalización.
4. Jerárquico → señales por TF + confidence + confluencias (2-de-3 por capa).
5. Gating (umbral confidence, dedup, fallbacks).
6. Decisión (side, sizing, SL/TP, trailing, TTL) con límites de riesgo.
7. Aplicación (abrir/cerrar/ajustar) → actualizar pos/cartera, fees, PnL R/UR, MFE/MAE, DD.
8. Emitir eventos + info (motivos y métricas).

## 3) Multi-Timeframe (alineación)
- TFs requeridos: ["1m","5m","15m","1h","4h","1d"] (configurable).
- Alineación: usar **última barra cerrada ≤ bar_time**.
- Modo estricto: si falta un TF requerido → **no abrir** (cierre puede permitir fallback).
- Calidad de datos: marcar huecos y latencias; breaker si degradación persistente.

## 4) Features técnicos y SMC
- Técnicos por TF: EMA/MA, RSI, MACD, ATR, Bollinger (%B/width), ADX/DI, OBV, Estocástico, SuperTrend.
- SMC: Estructura (BOS/CHOCH, HH/HL/LH/LL), Zonas (Order Blocks, FVG, EQ highs/lows), Contexto (rango, mid-range, imbalance, POI).
- Normalización: por TF/ventana; flags booleanos/numéricos para SMC.

## 5) Análisis jerárquico (señales)
- Capas:
  - Dirección: **1D, 4H** (regime, estructura, pendiente EMAs).
  - Confirmación: **1H, 15M** (momentum, divergencias, ADX).
  - Ejecución: **5M, 1M** (timing: OB tap, cierre FVG, break&retest, RSI reset).
- Confluencias: regla **2-de-3 por capa** y ponderaciones (1D>4H, 1H>15M, 5M>1M).
- confidence ∈ [0,1]: agregado ponderado de señales.

## 6) Gating y política
- Umbral mínimo min_confidence para **apertura**.
- Deduplicación por ventanas (open/close).
- Fallbacks: permitir **cierre** aunque no haya confirmación plena (anti-trampa).
- Política: abrir/cerrar, parciales, 
educe_only, TTL, trailing, cierre forzado por breaker.

## 7) Gestión de riesgo (autónoma)
- Común: riesgo por trade dinámico (volatilidad, DD), exposición máxima, breakers.
- Spot (sizing):
  - 
iesgo_trade = risk_pct * equity_quote
  - qty = riesgo_trade / max(ε, |entry - SL|) → aplicar minNotional y lotStep.
- Futuros:
  - Apalancamiento inicial ≤ **3x** (dinámico por volatilidad/DD).
  - 
otional = qty * entry, margin_used = notional / leverage.
  - Verificar **maintenance margin** y límites antes de abrir (denegar o reducir).
- TP/SL:
  - SL = entry ± k_sl * ATR(tf_exec) ajustado a niveles SMC (protegido).
  - TP por R esperado o niveles SMC (cierres de FVG, zonas de liquidez).
  - Trailing tras MFE ≥ umbral (step sobre ATR).

## 8) Contabilidad, fees y PnL
- Balances:
  - Spot: equity_quote, equity_base, fee, locked.
  - Futuros: balance USDT (o moneda de margen), margen inicial/mantenimiento.
- Fees: **taker** por defecto (bps). Si adapter reporta exactas, prevalecen.
- Funding (futuros): por adapter (live) o simulado (opcional en backtest/train).
- PnL:
  - UR Spot: (last - entry) * side * qty
  - R Spot: Σ (close - entry) * side * qty - (fees_open + fees_close)
  - Futuros: igual concepto con contrato/margen.

## 9) Circuit breakers
- DD diario > umbral → **close-only**.
- Degradación de datos/latencia → pausa o cierre seguro.
- Señales inconsistentes graves (ventana corta) → neutral temporal.

## 10) Configuración (ubicación y contratos)
- Ubicación: **config/** (settings.yaml, symbols.yaml, 
isk.yaml, fees.yaml, pipeline.yaml, hierarchical.yaml, oms.yaml).
- Todo lo **configurable** vive en config/. El core lee esos YAML, valida y aplica.

## 11) Integraciones (adapters)
- DataBroker: Parquet (histórico) / WebSocket (live).
- OMS: Sim/Paper/Live. **Slippage siempre en adapter/wrapper**.
- Risk/Policy/Accounting: módulos internos consumen config YAML y devuelven acciones/eventos.

## 12) KPIs clave
- Step: exposure, UR PnL, equity, DD, confidence, latencia.
- Trade: R múltiplo, holding time, heat, eficiencia (MFE/MAE), fees netas.
- Sesión: PnL, Sharpe/Sortino/Calmar, win-rate, profit factor, maxDD, exposición/lev. prom.

## 13) Criterios de aceptación mínimos
- Alineación MTF estable en todos los steps con TFs requeridos.
- Gating respeta umbral/confidencias y deduplicación.
- Sizing respeta minNotional/lotStep y límites de riesgo.
- Fees aplicadas en open y close; PnL cuadra en casos básicos.
- Breakers activan estados seguro (close-only/neutral) y emiten eventos.

