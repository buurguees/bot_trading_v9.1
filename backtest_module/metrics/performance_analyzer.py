from __future__ import annotations
import numpy as np
import pandas as pd

def basic_metrics(returns: pd.Series, periods_per_year: int = 252) -> dict:
    r = returns.fillna(0.0)
    cum = (1 + r).cumprod()
    dd = 1 - cum / cum.cummax()
    sharpe = r.mean() / (r.std() + 1e-12) * np.sqrt(periods_per_year)
    sortino = r.mean() / (r[r<0].std() + 1e-12) * np.sqrt(periods_per_year)
    calmar = (cum.iloc[-1] - 1) / (dd.max() + 1e-12)
    return {
        "sharpe": float(sharpe),
        "sortino": float(sortino),
        "calmar": float(calmar),
        "max_dd": float(dd.max()),
        "equity_final": float(cum.iloc[-1]),
    }
