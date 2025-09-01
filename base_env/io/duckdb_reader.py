# Descripción: Lectura SQL sobre Parquet con DuckDB (útil para joins o rangos grandes).
# Ubicación: base_env/io/duckdb_reader.py
# I/O: SELECT sobre rutas 'data/{SYMBOL}/{market}/{stage}/{tf}/**/*.parquet' → registros ordenados por ts

# TODO IMPLEMENT:
# - query_window(symbol, market, tf, ts_from, ts_to, stage="aligned") -> Iterable[dict]
# - join_multi_tf(symbol, market, tfs, ts_from, ts_to) -> dict[tf -> dict[ts -> bar]]
# - Control de memoria (limitar columnas; seleccionar solo ts, ohlcv)
