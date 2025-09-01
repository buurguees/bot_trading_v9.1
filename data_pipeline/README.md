# data_pipeline/README.md
# Descripción: Cómo gestionar el histórico (descarga, validación, alineación, empaquetado) y dónde guarda cada fase.
# Ubicación: data_pipeline/README.md
# data_pipeline — Gestión y análisis de histórico

**Objetivo:** centralizar *scripts* y documentación para:
1) **Descargar** histórico en `data/{SYMBOL}/raw/{tf}/year=YYYY/month=MM/*.parquet`
2) **Validar** integridad (orden por `ts`, huecos, tipos) y actualizar **manifest.json**
3) **Alinear** multi-timeframe (bar_time del TF base) a `data/{SYMBOL}/aligned/...`
4) **Empaquetar** datasets grandes en `data/{SYMBOL}/packages/` para consumo rápido de `base_env`

**Convenciones:**
- Timestamps en **ms UTC**.
- Compresión **ZSTD** (~nivel 6–8).
- Columnas: `ts, open, high, low, close, volume, (vwap?), (n_trades?), symbol, market, tf, ingestion_ts`.
- Particionado **year=YYYY/month=MM**.

**Relación con `base_env`:**
- `base_env/io/` consumirá **aligned** y/o **packages** para crear vistas multi-TF eficientes.
- La lógica (SMC, features, indicadores, alineación) vive en `base_env/` (no aquí).
