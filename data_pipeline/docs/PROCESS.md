# data_pipeline/docs/PROCESS.md
# Descripción: Proceso E2E de datos: raw -> aligned -> packages; convenciones y naming.
# Ubicación: data_pipeline/docs/PROCESS.md

# Proceso E2E de datos

## 1) RAW (ingesta)
- Guardar en `data/{SYMBOL}/raw/{tf}/year=YYYY/month=MM/*.parquet`
- Validar: `ts` ascendente, sin duplicados, tipos consistentes.

## 2) VALIDATE
- Reportar huecos, barras corruptas y rangos por TF.
- Escribir/actualizar `data/{SYMBOL}/manifest.json`:
  - `first_ts`, `last_ts`, `bar_count`, `files`, `checksums`, `last_update`, `issues`.

## 3) ALIGN
- Alinear TFs al **bar_time** del TF base (ej. 1m).
- Guardar en `data/{SYMBOL}/aligned/{tf}/year=YYYY/month=MM/*`.
- Marcar `quality_gap=true` cuando falten TFs.

## 4) PACKAGE
- Generar **paquetes grandes** listos para entreno/backtest:
  - `data/{SYMBOL}/packages/{SYMBOL}_spot_MTF_1m-5m-15m_2020-2024.parquet`
- Incluir **metadatos**: columnas disponibles, TFs incluidos, período.

> Nota: `base_env` consumirá **aligned** o **packages** según el modo (rápido vs granular).
