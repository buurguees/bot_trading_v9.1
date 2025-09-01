# data_pipeline actions — qué haremos a continuación

1) **download_history.py**
   - Implementar descarga por símbolo/TF a `data/{SYMBOL}/raw/{tf}/year=YYYY/month=MM/*.parquet`
   - Esquema `ohlcv_schema.json`, compresión ZSTD, UTC ms
   - Idempotencia por `ts` (no duplicar)

2) **validate_history.py**
   - Comprobar orden `ts` ascendente y tipos
   - Detectar huecos → registrar en `manifest.json` (`issues`)
   - Guardar `first_ts`, `last_ts`, `bar_count`, `files`, `checksums`

3) **align_package.py**
   - Alinear TFs al `bar_time` base
   - Escribir en `data/{SYMBOL}/aligned/...`
   - Generar `data/{SYMBOL}/packages/...` con metadatos

4) **manifest_update.py**
   - Unificar info tras cada proceso y “sellar” el manifiesto

> Nota: la lógica de indicadores/SMC/alineación de entorno vive en `base_env/`, no aquí.
