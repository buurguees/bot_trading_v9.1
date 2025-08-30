# 🗺️ Roadmap — Trading Bot v9.1 (Autónomo, Multi-TF, SMC)

## 🎯 Objetivo Global
Construir un bot de trading **totalmente autónomo** que:
- Analice **multi-TF jerárquico**: Dirección (1D, 4H) → Confirmación (1H, 15m) → Ejecución (5m, 1m).
- Combine **indicadores técnicos** + **SMC** (Order Blocks, FVG, Liquidity, BOS/CHOCH).
- Gestione **riesgo de forma autónoma** (sizing dinámico, TP/SL/trailing, circuit breakers).
- Memorice **estrategias ganadoras** (setups + métricas) para referencia futura.
- Ejecute **Backtest / Paper / Live** con **mismo kernel** (equivalencia sim↔live).

---

## 🧭 Secuencia de trabajo (estricta)
> No saltar fases. Cada fase tiene criterios de aceptación.

1) **Fase A — Bases de Trading (Indicadores + SMC)**
2) **Fase B — Base Env** (entorno canónico común a todo)
3) **Fase C — Train Env** (datasets MTF, labeling cost-aware, CV sin leakage) → preparado para **10–30M steps**
4) **Fase D — Backtest** (vectorizado + event-driven, equivalentes)
5) **Fase E — Paper/Live** (router sim↔bitget, risk gates, breakers)
6) **Fase F — Observabilidad** (Prometheus/Grafana), **Memoria de estrategias** y **aprendizaje continuo**

---

## 🔁 Multi-TF Jerárquico (definición operacional)
- **Dirección**: 1D, 4H → determina sesgo macro (alcista/neutral/bajista).
- **Confirmación**: 1H, 15m → valida sesgo y timing.
- **Ejecución**: 5m, 1m → ubica entrada exacta, SL/TP/trailing.
- **Regla**: ninguna operación se abre si **Dirección y Confirmación no están alineadas**.

---

## 🧱 Fase A — Bases del Trading (Indicadores + SMC)
**Objetivo**: disponer de un pipeline de features **determinista y causal** (sin lookahead), alineado por TF.

**Tareas**
- A1. Definir `config/features.yaml` y `config/smc.yaml` (parámetros por TF).
- A2. Implementar **Indicadores técnicos** (RSI, MACD, EMA/SMA, ATR, Bollinger, OBV, VWAP, SuperTrend).
- A3. Implementar **SMC**: swings, BOS/CHOCH, Order Blocks (oferta/demanda), FVG, Liquidity Zones.
- A4. Normalización/estandarización por TF + **as-of join** multi-TF para ventanas de contexto (ventanas mínimas recomendadas):
  - 1D: 180, 4H: 360, 1H: 720, 15m: 960, 5m: 1440, 1m: 3000.
- A5. Validación de datos: gaps/dups, TZ, outliers, consistencia MTF.

**Criterios de aceptación**
- [ ] Features reproducibles (hash del dataset y config).
- [ ] Modos `causal` (sin fuga) y `symmetric` (solo para investigación).
- [ ] Pruebas unitarias de cada indicador/SMC y de alineación MTF.

---

## 🧩 Fase B — Base Env (entorno canónico)
**Objetivo**: un entorno “intocable” que abstrae mercado/órdenes/portafolio de forma consistente para **train, backtest y live**.

**Tareas**
- B1. `core/oms`: `order.py` (Order/Fill/Position), `execution_sim.py` (latencia/slippage/partial fills), `router.py` (sim | bitget).
- B2. `core/portfolio`: `ledger.py` (event sourcing), `accounting.py` (PnL realized/unrealized, fees, funding).
- B3. `core/market`: `assets.py` (filtros minQty/tickSize/lotStep), `clocks.py` (NTP, sesión), `data_view.py` (vista MTF coherente).
- B4. `core/portfolio/risk_manager.py`: sizing dinámico, SL/TP/trailing por ATR + niveles SMC, exposure caps, **circuit breakers**.
- B5. Config tipada (Pydantic/Hydra). Logs **JSON** (niveles, contexto, request-ids).

**Criterios de aceptación**
- [ ] **Equivalencia sim↔live**: mismas comisiones, filtros y redondeos.
- [ ] Golden tests: misma señal → mismos fills en simulador determinista.
- [ ] Circuit breakers activos (DD intradía, latencia, gaps anómalos).

---

## 🧠 Fase C — Train Env (10–30M steps listo)
**Objetivo**: obtener **políticas/modelos cost-aware**, sin leakage, listos para maratón.

**Tareas**
- C1. **Datasets MTF** (por símbolo y TF de ejecución: 5m y 1m): as-of join con contexto de Dirección y Confirmación; masking y padding.
- C2. **Labeling cost-aware** (triple-barrier): horizontes H por TF (p.ej. 5m: 48, 1m: 240), TP/SL en múltiplos de **ATR** + proximidad a niveles SMC; retorno neto (− fees − slippage).
- C3. **CV temporal**: **Purged K-Fold** + **Embargo** (5–10%). **Walk-Forward** con ventanas crecientes.
- C4. Modelos: baselines técnicos; ML (XGBoost, LSTM/TCN); ensembles por régimen (tendencia/sideways/alta vol).
- C5. **Model Registry**: runs (params, seeds, métricas, hashes), artefactos (`models/<symbol>/<id>.*`), reportes `reports/train/<run_id>/`.
- C6. Preparar **lanza-maratón**: configuración de seeds, checkpoints, logging a Postgres, limpieza de memoria, validación periódica.

**Targets/Métricas**
- Métricas objetivo por fold y OOS: **Sharpe**, **Sortino**, **Calmar**, **MaxDD**, **PF**, hit-rate, F1 (señales).
- Umbrales mínimos sugeridos para pasar a Backtest:
  - Sharpe ≥ 1.3, Calmar ≥ 0.8, MaxDD ≤ 12%, PF ≥ 1.2.

**Criterios de aceptación**
- [ ] Sin leakage (tests anti-lookahead).
- [ ] Purged K-Fold + Embargo implementados.
- [ ] Reproducibilidad (hash dataset/features/seeds).
- [ ] Export compatible con **Base Env** (mismo formato de señales).
- [ ] Preparado para **10–30M steps** (config y scripts ready).

---

## 📊 Fase D — Backtest (Vectorizado + Event-Driven)
**Objetivo**: validar la política en simulación rápida (vectorizado) y realista (event-driven) con **mismo kernel**.

**Tareas**
- D1. Motor vectorizado (rápido) con comisiones/slippage fijos.
- D2. Motor event-driven con `execution_sim` (latencia, partials).
- D3. Métricas: Sharpe/Sortino/Calmar, PF, hit-rate, VaR/CVaR, MaxDD, MAR.
- D4. Reportes HTML/PDF (equity, DD, distribución de returns, sensibilidad).

**Criterios de aceptación**
- [ ] Coherencia entre motores en casos simples (tests de contrato).
- [ ] Resultados reproducibles y firmados (hash run).

---

## 💹 Fase E — Paper & Live
**Objetivo**: pasar a stream RT con **router sim** (paper) y luego **router bitget** (live), manteniendo risk gates.

**Tareas**
- E1. WebSocket feeds (libro/ohlcv), reconstrucción de velas 1m/5m.
- E2. Paper: slippage/latencia simulada, fills deterministas.
- E3. Live Spot: capital limitado, `reduce-only` por defecto en cierres.
- E4. Live Futuros: leverage ≤ 3x, funding y protección de liquidación.
- E5. Alertas (Telegram/Email) y pausas automáticas (breakers).

**Criterios de aceptación**
- [ ] Reconciliación de fills con ledger.
- [ ] Alertas y breakers verificados en staging.

---

## 📡 Fase F — Observabilidad, Memoria y Aprendizaje Continuo
**Objetivo**: operar 24/7 con telemetría, memoria de setups y rotación de modelos.

**Tareas**
- F1. Prometheus Exporter (latencias, fills/min, reject rate, DD, VaR intradía).
- F2. Grafana dashboards (Overview, Riesgo, Ejecución, Health WS/REST).
- F3. **Memoria de estrategias ganadoras**: tabla de setups (contexto MTF, indicadores/SMC, SL/TP, métricas) + política de **deprecación** (drift).
- F4. Re-training por **concept drift** (PSI/KS) o deterioro de performance.

---

## ✅ Entregables por fase
- **A**: `config/features.yaml`, `config/smc.yaml`, `data_module/preprocessors/*` con tests; datasets MTF validados.
- **B**: `core/` (OMS, portfolio, market, utils) + `risk_manager.py` con breakers; equivalencia sim↔live.
- **C**: `training_module/` completo (datasets, labeling, CV, tuning, registry); scripts para **10–30M steps**.
- **D**: `backtest_module/` (dos motores) + reportes.
- **E**: `live_module/` (paper/live) + adapters + alertas.
- **F**: `monitoring/` (exporter + dashboards) + memoria de estrategias.

---

## 🧪 Normas de aceptación globales
- Código **tipado** (mypy), **lint** (ruff), **format** (black), **tests** (pytest).
- Logs **JSON** con request-id y contexto (símbolo, TF, run, seed).
- **Equivalencia sim↔live** probada (fees, lotes, redondeos, slippage base).
- **Risk Manager** obligatorio antes de cualquier orden.
- Todo dataset/modelo **firmado por hash** (reproducibilidad).

---

## 🧰 Próximos pasos (local, sin GitHub aún)
1. Crear la estructura vacía `bot_trading_v9_1/` y colocar este `ROADMAP.md`.
2. Añadir `README.md` e `INSTRUCTIONS.md`.
3. Crear `config/` con `features.yaml` y `smc.yaml` (plantillas).
4. Inicializar entorno (poetry/uv) y pre-commit (ruff/black/mypy/pytest).
5. Comenzar **Fase A** (indicadores + SMC) siguiendo los criterios de aceptación.

> Cuando esté lista la Fase A, pasamos a **Base Env (B)**; con B listo, **Train Env (C)** para lanzar la **maratón de 10–30M steps**.
