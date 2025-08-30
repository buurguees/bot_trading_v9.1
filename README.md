# 🤖 Trading Bot v9.1 — Arquitectura Autónoma Profesional

## 🎯 Objetivo
Un bot de trading **totalmente autónomo**, con gestión de riesgo integrada y toma de decisiones libre en:
- Selección de capital por operación  
- Colocación dinámica de TP y SL  
- Elección de apalancamiento (modo futuros)  
- Ejecución multi-símbolo y multi-timeframe

## 🧠 Lógica Jerárquica de Timeframes
El bot analiza el mercado de forma piramidal:

- **Dirección principal** → 1D, 4H  
- **Confirmación** → 1H, 15M  
- **Ejecución** → 5M, 1M  

Esto garantiza operaciones alineadas con la **macro-tendencia** y precisión en la entrada.

---

## 📊 Estrategias
- **Indicadores técnicos**: RSI, MACD, EMA, ATR, Bollinger, OBV, SuperTrend  
- **SMC (Smart Money Concepts)**: Order Blocks, FVG, Liquidity Zones, BOS/CHOCH  
- **Confluencias multi-TF**: validación jerárquica obligatoria antes de abrir operación  
- **Gestión de riesgo autónoma**: el bot decide tamaño de posición, TP, SL y trailing  
- **Memoria de estrategias**: guarda las **mejores setups** (PnL, Sharpe, win-rate) en base de datos para usarlas como referencia futura

---

## 🏗️ Estructura de Directorio (Profesional & Estricta)

bot_trading_v9_1/
├── config/ # Configuración central (todo validado)
│ ├── settings.yaml # Globals: entorno, logs, seeds
│ ├── risk.yaml # Límites riesgo: spot/futuros, breakers
│ ├── symbols.yaml # Símbolos + filtros (minQty, lotStep, TF habilitados)
│ └── strategies.yaml # Estrategias habilitadas (indicadores, smc)
│
├── core/ # Núcleo común (idéntico sim / live)
│ ├── oms/ # Order Management System
│ │ ├── order.py # Definición orden/fill/position
│ │ ├── router.py # Enrutador sim | bitget
│ │ └── execution_sim.py # Simulador fills (slippage, latencia, partial)
│ ├── portfolio/
│ │ ├── ledger.py # Libro mayor (event sourcing)
│ │ ├── accounting.py # PnL, fees, funding
│ │ └── risk_manager.py # Gestión riesgo global
│ ├── market/
│ │ ├── clocks.py # Relojes / sesiones / sync NTP
│ │ ├── assets.py # Lot, tickSize, filters símbolo
│ │ └── data_view.py # Vista multi-TF coherente
│ └── utils/ # Utilidades comunes
│ ├── logger.py
│ ├── cache.py
│ └── retry.py
│
├── data_module/ # Datos históricos + tiempo real
│ ├── collectors/ # Descarga y feeds
│ │ ├── bitget_collector.py # OHLCV histórico
│ │ └── websocket_collector.py # WS en vivo
│ ├── preprocessors/
│ │ ├── indicator_calculator.py # RSI, MACD, EMAs, etc.
│ │ └── smc_detector.py # BOS, OB, Liquidity, FVG
│ └── storage/
│ ├── parquet_store.py # Históricos fríos (Parquet/Polars)
│ ├── influx_client.py # Time series en InfluxDB
│ └── postgres_manager.py # Logs, estrategias guardadas
│
├── strategy_engine/ # Donde vive la “inteligencia”
│ ├── base_strategy.py # Clase base
│ ├── ml_strategies/ # Estrategias Machine Learning
│ │ ├── lstm_strategy.py
│ │ └── xgboost_strategy.py
│ └── smc_strategies/ # Estrategias basadas en SMC
│ ├── order_block.py
│ ├── fvg_breaker.py
│ └── liquidity_sweep.py
│
├── training_module/ # Entrenamiento & validación
│ ├── features/ # Features ML (técnicos + smc + volumen)
│ ├── trainers/ # Optuna, walk-forward, cross-validation
│ └── model_selector.py # Ranking mejores modelos
│
├── backtest_module/ # Motores de backtest
│ ├── vectorized_engine.py
│ ├── event_driven_engine.py
│ └── metrics/
│ ├── performance_analyzer.py # Sharpe, Sortino, Calmar
│ ├── risk_calculator.py # VaR, CVaR, DD
│ └── trade_analyzer.py # Win-rate, PF
│
├── live_module/ # Ejecución real
│ ├── spot_trader/ # Spot trading
│ │ ├── executor.py
│ │ └── portfolio_manager.py
│ ├── futures_trader/ # Futures trading
│ │ ├── executor.py
│ │ └── leverage_manager.py
│ └── monitoring/
│ ├── performance_tracker.py
│ └── alert_system.py # Telegram, Email
│
├── monitoring/ # Observabilidad global
│ ├── prometheus_exporter.py
│ └── dashboards/ (Grafana JSONs)
│
├── tests/ # Testeo profesional
│ ├── unit/
│ ├── integration/
│ └── strategies/
│
├── scripts/ # Scripts utilitarios
│ ├── bootstrap_data.sh
│ ├── run_backtest.sh
│ └── run_live.sh
│
├── app.py # CLI Typer: backtest | paper | live
├── pyproject.toml # Dependencias & tooling
└── README.md # Este documento


---

## 🔄 Flujo Operativo

1. **Data Layer**: descarga, limpia y valida históricos (5 años).  
2. **Preprocesamiento**: calcula indicadores y detecta estructuras SMC.  
3. **Strategy Engine**: genera señales → riesgo → OMS.  
4. **Gestión de riesgo**: calcula posición, TP, SL, trailing.  
5. **Backtest/Paper/Live**: mismo kernel, cambia solo el router.  
6. **Memoria de estrategias**: se guardan setups top-PnL para futuras referencias.  

---

## 🛡️ Gestión de Riesgo Autónoma
- **Capital dinámico**: sizing adaptativo por volatilidad, liquidez y DD.  
- **Stop Loss / Take Profit**: calculados automáticamente (ATR + SMC levels).  
- **Circuit Breakers**: pausa si DD diario > X%, latencia > Y ms, o señales inconsistentes.  
- **Futures**: apalancamiento dinámico ≤ 3x inicial.  

---

## 🚀 Próximos Pasos
1. Definir **dataset inicial** (BTC, ETH, SOL, ADA, BNB, DOGE).  
2. Implementar **data pipeline** con indicadores + smc.  
3. Programar **estrategia baseline** (EMA crossover + OB confirmación).  
4. Desarrollar **gestión de riesgo autónoma** en `core/portfolio/risk_manager.py`.  
5. Validar todo con **backtests vectorizados + event-driven**.  
