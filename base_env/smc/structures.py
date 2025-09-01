# Descripción: Detección de estructura de mercado (BOS/CHOCH, HH/HL/LH/LL) por TF.
# Ubicación: base_env/smc/structures.py
# I/O: high/low/close por ventana → flags/últimos puntos de swing

# TODO IMPLEMENT:
# - detect_swings(high, low, win) -> lista de pivots
# - detect_structure(pivots) -> {last_bos: up|down|none, last_choch: ... , last_swing_high/low}
