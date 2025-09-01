# Descripción: Zonas SMC (Order Blocks, FVG, EQ highs/lows) con prioridad/frescura.
# Ubicación: base_env/smc/zones.py
# I/O: barras y/o pivots → lista de zonas con atributos (tf, price_range, freshness, touched?)

# TODO IMPLEMENT:
# - find_fvg(bars) -> lista de gaps
# - find_order_blocks(bars, pivots) -> lista de OB con validez y "mitad" (50%)
# - find_eq_levels(pivots) -> eq_highs/eq_lows detectados
