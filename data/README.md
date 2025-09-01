# data/README.md
# Descripción: Estructura de almacenamiento de datos. TODO: no subir Parquet pesados al repo; usar .gitignore.
# Ubicación: data/README.md
# data/ — Almacenamiento de histórico

**Estructura por símbolo**
data/{SYMBOL}/
├── raw/ # descargas sin alinear (por TF, particionado año/mes)
├── aligned/ # barras alineadas por bar_time del TF base
├── packages/ # paquetes grandes (listas para entreno/backtest)
└── manifest.json # índice con rangos, archivos y checksums

markdown
Copiar código

**Notas**
- No subir Parquet al repo (ver `.gitignore`).
- Timestamps ms UTC. Compresión ZSTD.
- `base_env` leerá principalmente `aligned/` o `packages/`.