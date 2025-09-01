# SCHEMAS.md
# Descripción: Esquemas de datos del entorno base (observación, acción, eventos) y tipos esperados.
# Ubicación: base_env/docs/SCHEMAS.md

# BaseEnv v9.1 — SCHEMAS

## 1) Observation (obs)
- 	s (int64, ms UTC): marca temporal del bar del TF base.
- 	fs (dict[str→bar]): barras alineadas por TF requerido.
  - bar: { ts, open, high, low, close, volume, (vwap?), (n_trades?) }
- Features (dict): features normalizadas por TF (TA+SMC).
- Analysis (dict):
  - by_tf (dict): señales por TF (p. ej. direction|confirmation|execution, flags SMC).
  - confidence (float 0..1)
  - confluences (list|int)
- Position (dict):
  - side (int: -1,0,+1)
  - qty (float)
  - entry_price (float)
  - sl (float|null)
  - 	p (float|null)
  - 	rail (float|null)
  - 	tl_bars (int)
  - mfe (float), mae (float)
- portfolio (dict):
  - **Spot**: equity_quote (float), equity_base (float), free (float), locked (float)
  - **Futuros**: balance (float), margin_init (float), margin_maint (float)
  - drawdown_day_pct (float)
  - exposure (float, [-1..1] aprox.)
- quality (dict): flags de datos (huecos/latencia/desync).
- mode (str): 	rain|backtest|live

## 2) Action (action)
- side (int): -1 | 0 | +1
- sizing_mode (str): exposure | risk_amount | qty_absolute
- 	arget_exposure (float) | 
isk_amount (float) | qty (float)  # según modo
- sl (float|null), 	p (float|null)
- 	railing (bool)
- 	tl_bars (int)
- 
educe_only (bool)
- close_all (bool)

## 3) Events (domain)
- OrderOpened: { ts, type: "order_opened", data: { side, qty, price, sl?, tp? } }
- OrderClosed: { ts, type: "order_closed", data: { qty, price, reason } }
- StopHit: { ts, type: "stop_hit", data: { qty, price } }
- TakeProfitHit: { ts, type: "tp_hit", data: { qty, price } }
- TrailMoved: { ts, type: "trail_moved", data: { from, to } }
- RuleViolated: { ts, type: "rule_violated", data: { rule, details } }
- CircuitBreaker: { ts, type: "breaker", data: { kind, state } }
- DataQuality: { ts, type: "data_quality", data: { issue, severity } }

## 4) Config (YAML → objetos)
- config/settings.yaml: entorno, logs, seed, timezone, paths.
- config/symbols.yaml: símbolos, mercados, TFs habilitados, filtros (tickSize, lotStep, minNotional), meta futuros.
- config/risk.yaml: riesgo por trade, exposición máx, apalancamiento máx, breakers.
- config/fees.yaml: taker/maker bps, funding (futuros).
- config/pipeline.yaml: indicadores TA y SMC por TF.
- config/hierarchical.yaml: capas TF, umbral min_confidence, fallbacks, dedup ventanas.
- config/oms.yaml: modo, adapters (sim/live), parámetros wrapper (slippage fuera del core).

## 5) Tipos y convenciones
- Timestamps en **ms UTC** (int64).
- Precios/cantidades en **float64** (normalizados a quote/base según mercado).
- TFs: 1m,5m,15m,1h,4h,1d.
- Slippage: **no** dentro del core.
