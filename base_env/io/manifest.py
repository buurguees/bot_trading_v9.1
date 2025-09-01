# Descripción: Lectura/escritura de manifest.json por símbolo (rangos, archivos, checksums).
# Ubicación: base_env/io/manifest.py
# I/O: data/{SYMBOL}/manifest.json → dict con first_ts, last_ts, bar_count, files, issues

# TODO IMPLEMENT:
# - read_manifest(symbol) -> dict | {}
# - write_manifest(symbol, data: dict) -> None
# - touch_last_update(symbol) -> None
