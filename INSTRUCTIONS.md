# 📑 Instrucciones de Desarrollo — Trading Bot v9.1

Este documento alinea a todos los desarrolladores e IAs que colaboran en el proyecto.

---

## 🧱 Principios Clave
1. **Profesionalismo extremo**: código limpio, tipado, documentado y testeado.
2. **Modularidad**: cada módulo cumple un rol específico y no invade otros dominios.
3. **Equivalencia sim-live**: backtest/paper/live deben compartir el mismo kernel.
4. **Gestión de riesgo no negociable**: ningún trade se ejecuta sin pasar por `risk_manager.py`.
5. **Confiabilidad de datos**: ninguna estrategia se entrena con datos no validados.

---

## 🏗️ Normas de Arquitectura
- Todos los archivos se ubican en la **estructura estricta** definida en `README.md`.
- **Nada de crear carpetas ad-hoc**. Si algo no encaja, se discute antes de añadir.
- Configuración en **YAML** → cargada y validada con Pydantic.
- **Naming convention**:
  - `snake_case` para archivos, funciones y variables.
  - `CamelCase` solo para clases.
- **Imports relativos prohibidos**: usar siempre `bot_trading_v9_1.modulo.submodulo`.

---

## 🧪 Testing
- Cada módulo nuevo debe incluir **tests unitarios** en `tests/unit/`.
- Los flujos completos deben probarse en `tests/integration/`.
- Estrategias deben validarse en `tests/strategies/`.
- Usar `pytest`, `pytest-asyncio`, `hypothesis`.

---

## 📊 Data Workflow
1. Descarga históricos (`bitget_collector.py`).
2. Validación y limpieza (`data_validator.py`).
3. Almacenamiento dual:
   - **Parquet (frío)** con Polars/PyArrow.
   - **InfluxDB (tiempo real)**.
4. Features → técnicos + SMC.
5. Dataset final validado antes de training.

---

## 🛡️ Gestión de Riesgo
- `risk_manager.py` decide:
  - Capital asignado (dinámico).
  - Stop Loss / Take Profit (basados en ATR + niveles SMC).
  - Apalancamiento ≤ 3x en futuros.
- Circuit breakers → bloquean todo el sistema si:
  - DD diario > límite.
  - Latencia > umbral.
  - Señales inconsistentes.

---

## 📈 Estrategias
- **Baseline**: EMA crossover + Order Block confirmación.
- **Técnicas**: momentum, mean reversion, volatility breakout.
- **ML**: LSTM, XGBoost, ensembles.
- Todas deben heredar de `base_strategy.py`.
- Estrategias ganadoras se guardan en **Postgres** con métricas de performance.

---

## 🔔 Observabilidad
- Exportar métricas a **Prometheus**.
- Dashboards en **Grafana** (preconfigurados en `monitoring/dashboards/`).
- Alertas en **Telegram/Email** en:
  - Circuit breaker activado.
  - Pérdida de conexión WS.
  - DD > umbral.

---

## ✅ Checklist por Feature Terminada
- [ ] Código en la carpeta correcta.
- [ ] Documentado y tipado.
- [ ] Tests unitarios y de integración.
- [ ] Validado con `ruff`, `black`, `mypy`.
- [ ] PR revisado y aprobado.
- [ ] Logs exportados en JSON.
