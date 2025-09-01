from __future__ import annotations
import pandas as pd

class SMCService:
    """Interfaz estable (stub inicial). La lógica real se irá completando."""
    def detect_swings(self, df: pd.DataFrame, lookback_left: int = 5) -> pd.DataFrame:
        # Marca swings simples (máximos/mínimos locales) como demo.
        df = df.copy()
        df["swing_high"] = (df["high"] == df["high"].rolling(lookback_left*2+1, center=True).max()).astype(int)
        df["swing_low"]  = (df["low"]  == df["low"].rolling(lookback_left*2+1, center=True).min()).astype(int)
        return df

    def detect_bos(self, df: pd.DataFrame) -> pd.DataFrame:
        # Demo BOS/CHOCH muy simplificado (placeholder)
        df = df.copy()
        df["bos_up"] = (df["close"] > df["close"].shift(1)) & (df["close"].shift(1) > df["close"].shift(2))
        df["bos_dn"] = (df["close"] < df["close"].shift(1)) & (df["close"].shift(1) < df["close"].shift(2))
        return df
