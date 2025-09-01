# base_env/features/pipeline.py
# Descripción: Cálculo de indicadores técnicos por TF con buffers internos por ventana.
# Mantiene historial corto por TF para computar EMA/RSI/ATR/MACD/Bollinger.
# Entrada: barras MTF alineadas (última barra cerrada por TF)
# Salida: dict de features por TF (últimos valores)

from __future__ import annotations
from collections import deque
from typing import Dict, Any, Deque, List

from ..config.models import PipelineConfig
from .indicators import ema, rsi, atr, macd, bollinger

_DEFAULT_WINDOW = 200  # barras por TF


class _TFBuffer:
    def __init__(self, maxlen: int = _DEFAULT_WINDOW) -> None:
        self.ts: Deque[int] = deque(maxlen=maxlen)
        self.open: Deque[float] = deque(maxlen=maxlen)
        self.high: Deque[float] = deque(maxlen=maxlen)
        self.low: Deque[float] = deque(maxlen=maxlen)
        self.close: Deque[float] = deque(maxlen=maxlen)
        self.volume: Deque[float] = deque(maxlen=maxlen)

    def push(self, bar: Dict[str, float]) -> None:
        # Evita duplicados por ts
        if self.ts and self.ts[-1] == int(bar["ts"]):
            return
        self.ts.append(int(bar["ts"]))
        self.open.append(float(bar["open"]))
        self.high.append(float(bar["high"]))
        self.low.append(float(bar["low"]))
        self.close.append(float(bar["close"]))
        self.volume.append(float(bar["volume"]))

    def to_lists(self) -> Dict[str, List[float]]:
        return {
            "open": list(self.open),
            "high": list(self.high),
            "low": list(self.low),
            "close": list(self.close),
            "volume": list(self.volume),
        }


class FeaturePipeline:
    def __init__(self, cfg: PipelineConfig) -> None:
        self.cfg = cfg
        self._buffers: Dict[str, _TFBuffer] = {}

    def compute(self, mtf_bars: Dict[str, Dict[str, float]]) -> Dict[str, Any]:
        # 1) actualizar buffers por TF con la barra actual
        for tf, bar in mtf_bars.items():
            if tf not in self._buffers:
                self._buffers[tf] = _TFBuffer()
            # Asegurar que el bar tiene 'ts'
            if "ts" not in bar:
                # Si el broker no incluye ts en el bar, intenta inferirlo del close (no recomendado)
                # pero en nuestro broker sí va incluido, así que asumimos presente.
                pass
            self._buffers[tf].push(bar)

        # 2) calcular indicadores por TF (último valor)
        feats: Dict[str, Any] = {}
        for tf, buf in self._buffers.items():
            s = buf.to_lists()
            closes = s["close"]
            highs = s["high"]
            lows = s["low"]

            ema20 = ema(closes, 20)
            ema50 = ema(closes, 50)
            rsi14 = rsi(closes, 14)
            atr14 = atr(highs, lows, closes, 14)
            macd_val = macd(closes, 12, 26, 9)
            bb20 = bollinger(closes, 20, 2.0)

            feats[tf] = {
                "ema20": ema20,
                "ema50": ema50,
                "rsi14": rsi14,
                "atr14": atr14,
                "macd": macd_val["macd"] if macd_val else None,
                "macd_signal": macd_val["signal"] if macd_val else None,
                "macd_hist": macd_val["hist"] if macd_val else None,
                "bb_p": bb20["pctB"] if bb20 else None,
                "bb_w": bb20["width"] if bb20 else None,
            }
        return feats
