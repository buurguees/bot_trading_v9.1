# TEST_PLAN.md
# Descripción: Plan de pruebas para validar el entorno base: alineación MTF, gating, riesgo, contabilidad y breakers.
# Ubicación: base_env/docs/TEST_PLAN.md

# BaseEnv v9.1 — Plan de Pruebas

## 1) Smoke tests (mínimos)
- **Reset/Step**: eset() devuelve obs válida; step() avanza y mantiene tipos/keys de obs.
- **Alineación MTF**: todos los TF requeridos presentes por ar_time (modo estricto).
- **Lectura config**: YAMLs válidos, coerción de tipos correcta.

## 2) Alineación y calidad de datos
- **TF superior**: con ar_time intermedio, el TF superior usa su última barra cerrada ≤ ar_time.
- **Huecos**: si falta TF requerido → no abrir (estricto); cerrar permitido con fallback.
- **Latencia**: simular delay en broker → marcar quality, no abrir.

## 3) Gating y jerárquico
- **Umbral alto** (min_confidence=0.9): no abre; con señales “limpias” baja el umbral y abre.
- **Deduplicación**: re-entradas dentro de la ventana → bloqueadas.
- **Fallback close**: abrir y luego forzar pérdida de confirmación → cierre permitido.

## 4) Riesgo y sizing (Spot)
- **minNotional/lotStep**: calcular qty y redondear; si queda < minNotional, no abrir.
- **risk_pct**: variar SL; qty se ajusta inversamente a la distancia SL.
- **Exposición máx**: si superaría, bloquear.

## 5) Riesgo y sizing (Futuros)
- **Apalancamiento ≤ 3x**: verificar margen inicial y mantenimiento antes de abrir.
- **Sobre-riesgo**: si excede límites → reducir qty o denegar.
- **Breaker DD**: simular DD diario > umbral → **close-only**.

## 6) TP/SL/Trailing/TTL
- **SL/TP**: ejecutar SL/TP en precio; PnL realizado coincide con cálculo esperado ± fees.
- **Trailing**: tras MFE≥umbral, mover trail (event TrailMoved).
- **TTL**: al agotar 	tl_bars, cerrar en el siguiente step.

## 7) Contabilidad y fees
- **Fees**: aplicar taker en open y close (si adapter no da exactas).
- **PnL**: casos básicos: long con subida, short con bajada; parciales; consolidado por trade y sesión.
- **MFE/MAE**: registrar máximos/mínimos intra-trade.

## 8) Eventos y métricas
- **Eventos**: OrderOpened/Closed, StopHit, TakeProfitHit, RuleViolated, CircuitBreaker.
- **Info**: exponer motivos (gating, reglas SMC/TAs activas) y KPIs step/trade.

## 9) Reproducibilidad
- **Semilla fija**: misma corrida = mismos resultados (train/backtest).
- **Determinismo**: sin fuentes no deterministas no controladas.

## 10) Rendimiento (opcional)
- **Ventanas lazy**: no cargar más histórico del necesario por TF.
- **Cache**: no recalcular features duplicadas dentro del mismo ar_time.
