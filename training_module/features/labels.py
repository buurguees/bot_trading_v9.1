from __future__ import annotations
import numpy as np
import pandas as pd

def triple_barrier_labels(df: pd.DataFrame, horizon: int, tp_k_atr: float, sl_k_atr: float) -> pd.DataFrame:
    """
    Etiquetado triple-barrier: retorna y in {-1,0,1} y tte (time to event).
    Usa ATR como escala. Requiere columnas: close, ta_atr.
    """
    close = df["close"].values
    atr = df["ta_atr"].values
    n = len(df)
    y = np.zeros(n, dtype=np.int8)
    tte = np.zeros(n, dtype=np.int32)

    for i in range(n):
        hi = min(n - 1, i + horizon)
        up = close[i] + tp_k_atr * atr[i]
        dn = close[i] - sl_k_atr * atr[i]
        hit = 0
        for j in range(i + 1, hi + 1):
            if close[j] >= up:
                y[i] = 1; tte[i] = j - i; hit = 1; break
            if close[j] <= dn:
                y[i] = -1; tte[i] = j - i; hit = 1; break
        if hit == 0:
            # horizonte alcanzado → signo por retorno final (opcional 0)
            ret = (close[hi] - close[i])
            y[i] = 1 if ret > 0 else (-1 if ret < 0 else 0)
            tte[i] = hi - i

    out = df.copy()
    out["y"] = y
    out["tte"] = tte
    return out
