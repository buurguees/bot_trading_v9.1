# base_env/accounting/fees.py
# Descripción: Helpers de cálculo de fees (bps).

def taker_fee(notional: float, bps: float) -> float:
    return float(notional) * float(bps) / 10_000.0
