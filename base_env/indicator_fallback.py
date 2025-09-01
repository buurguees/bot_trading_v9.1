# base_env/indicator_fallback.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Any
import numpy as np
import pandas as pd

@dataclass
class FeatureConfig:
    ema_periods: list[int]
    sma_periods: list[int]
    rsi_period: int
    macd: Dict[str, int]
    atr_period: int
    bbands: Dict[str, Any]
    supertrend: Dict[str, Any]
    obv: bool
    vwap: bool

    @classmethod
    def from_yaml(cls, path: str) -> "FeatureConfig":
        import yaml
        with open(path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)["features"]
        return cls(
            ema_periods=cfg.get("ema", [20, 50, 200]),
            sma_periods=cfg.get("sma", [20, 50, 200]),
            rsi_period=cfg.get("rsi", {}).get("period", 14),
            macd=cfg.get("macd", {"fast": 12, "slow": 26, "signal": 9}),
            atr_period=cfg.get("atr", {}).get("period", 14),
            bbands=cfg.get("bbands", {"period": 20, "dev": 2.0}),
            supertrend=cfg.get("supertrend", {"period": 10, "multiplier": 3.0}),
            obv=cfg.get("obv", True),
            vwap=cfg.get("vwap", True),
        )

class IndicatorCalculator:
    """Fallback sin TA-Lib (numpy/pandas). Causal y vectorizado."""
    def __init__(self, config: FeatureConfig, mode: str = "causal"):
        self.cfg = config
        self.mode = mode

    def _ema(self, s: pd.Series, n: int) -> pd.Series:
        return s.ewm(span=n, adjust=False, min_periods=n).mean()

    def _sma(self, s: pd.Series, n: int) -> pd.Series:
        return s.rolling(n, min_periods=n).mean()

    def _rsi(self, close: pd.Series, n: int) -> pd.Series:
        delta = close.diff()
        up = delta.clip(lower=0.0)
        down = -delta.clip(upper=0.0)
        gain = up.ewm(alpha=1 / n, adjust=False).mean()
        loss = down.ewm(alpha=1 / n, adjust=False).mean()
        rs = gain / loss.replace(0.0, np.nan)
        rsi = 100 - (100 / (1 + rs))
        return rsi

    def _tr(self, high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
        pc = close.shift(1)
        return pd.concat([(high - low).abs(), (high - pc).abs(), (low - pc).abs()], axis=1).max(axis=1)

    def _atr(self, high: pd.Series, low: pd.Series, close: pd.Series, n: int) -> pd.Series:
        return self._tr(high, low, close).ewm(alpha=1 / n, adjust=False).mean()

    def _bb(self, close: pd.Series, n: int, dev: float):
        ma = self._sma(close, n)
        std = close.rolling(n, min_periods=n).std(ddof=0)
        upper, lower = ma + dev * std, ma - dev * std
        return lower, ma, upper

    def calculate_all(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        for n in self.cfg.ema_periods:
            df[f"ta_ema_{n}"] = self._ema(df["close"], n)
        for n in self.cfg.sma_periods:
            df[f"ta_sma_{n}"] = self._sma(df["close"], n)

        # MACD
        f, s, sig = self.cfg.macd["fast"], self.cfg.macd["slow"], self.cfg.macd["signal"]
        macd = self._ema(df["close"], f) - self._ema(df["close"], s)
        macd_signal = macd.ewm(span=sig, adjust=False, min_periods=sig).mean()
        df["ta_macd"], df["ta_macd_signal"], df["ta_macd_histogram"] = macd, macd - macd_signal, macd - macd_signal
        df["ta_macd_cross"] = (df["ta_macd"] > df["ta_macd_signal"]).astype(int)

        # RSI
        df["ta_rsi"] = self._rsi(df["close"], self.cfg.rsi_period)
        df["ta_rsi_overbought"] = (df["ta_rsi"] > 70).astype(int)
        df["ta_rsi_oversold"] = (df["ta_rsi"] < 30).astype(int)

        # ATR
        df["ta_atr"] = self._atr(df["high"], df["low"], df["close"], self.cfg.atr_period)
        df["ta_atr_percent"] = df["ta_atr"] / df["close"]

        # Bollinger
        lo, mid, up = self._bb(df["close"], self.cfg.bbands["period"], self.cfg.bbands["dev"])
        df["ta_bb_lower"], df["ta_bb_middle"], df["ta_bb_upper"] = lo, mid, up
        df["ta_bb_width"] = (up - lo) / mid
        df["ta_bb_position"] = (df["close"] - lo) / (up - lo)

        # OBV
        if self.cfg.obv:
            chg = df["close"].diff().fillna(0)
            direction = np.sign(chg)
            df["ta_obv"] = (direction * df["volume"]).cumsum()
            df["ta_obv_sma"] = df["ta_obv"].rolling(20, min_periods=20).mean()

        # VWAP
        if self.cfg.vwap:
            tp = (df["high"] + df["low"] + df["close"]) / 3
            df["ta_vwap"] = (tp * df["volume"]).cumsum() / df["volume"].cumsum()
            df["ta_vwap_distance"] = (df["close"] - df["ta_vwap"]) / df["ta_vwap"]

        # SuperTrend simple
        period = self.cfg.supertrend.get("period", 10)
        mult = self.cfg.supertrend.get("multiplier", 3.0)
        atr = df["ta_atr"]
        hl2 = (df["high"] + df["low"]) / 2
        upper = hl2 + mult * atr
        lower = hl2 - mult * atr
        st = np.zeros(len(df))
        trend = np.ones(len(df))
        for i in range(1, len(df)):
            upper.iloc[i] = upper.iloc[i] if (upper.iloc[i] < upper.iloc[i-1] or df["close"].iloc[i-1] > upper.iloc[i-1]) else upper.iloc[i-1]
            lower.iloc[i] = lower.iloc[i] if (lower.iloc[i] > lower.iloc[i-1] or df["close"].iloc[i-1] < lower.iloc[i-1]) else lower.iloc[i-1]
            if df["close"].iloc[i] <= lower.iloc[i]:
                trend[i] = -1
            elif df["close"].iloc[i] >= upper.iloc[i]:
                trend[i] = 1
            else:
                trend[i] = trend[i-1]
            st[i] = lower.iloc[i] if trend[i] == 1 else upper.iloc[i]
        df["ta_supertrend"], df["ta_supertrend_trend"] = st, trend
        df["ta_supertrend_signal"] = (df["ta_supertrend_trend"] == 1).astype(int)

        # Limpieza causal
        if self.mode == "causal":
            df = df.ffill()

        return df
