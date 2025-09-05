# base_env/features/pipeline.py
# Pipeline de features robusto con validación de datos, manejo de duplicados/actualizaciones
# y fallbacks seguros. Mantiene la API de FeaturePipeline para compatibilidad.

from __future__ import annotations
from collections import deque
from dataclasses import dataclass
from typing import Dict, Any, Deque, List, Optional
import logging
import numpy as np

from ..config.models import PipelineConfig

# Import de indicadores (misma ruta que en la versión actual)
from .indicators import ema, rsi, atr, macd, bollinger

_DEFAULT_WINDOW = 200  # barras por TF
_MIN_BARS_FOR_INDICATORS = 50

# -------------------------------
# 1) Validador de calidad de datos
# -------------------------------
class DataQualityValidator:
    """Valida calidad de barras OHLCV + ts."""

    REQUIRED_FIELDS = ("ts", "open", "high", "low", "close", "volume")

    @staticmethod
    def validate_bar(bar: Dict[str, Any]) -> List[str]:
        errors: List[str] = []

        # Campos requeridos y tipos
        for field in DataQualityValidator.REQUIRED_FIELDS:
            if field not in bar:
                errors.append(f"Campo faltante: {field}")
                continue
            if field == "ts":
                if not isinstance(bar[field], (int, np.integer)) or int(bar[field]) <= 0:
                    errors.append(f"Timestamp inválido en ts: {bar[field]}")
            else:
                if not isinstance(bar[field], (int, float, np.floating, np.integer)):
                    errors.append(f"Tipo inválido en {field}: {type(bar[field])}")
                elif float(bar[field]) < 0:
                    errors.append(f"Valor negativo en {field}: {bar[field]}")

        if errors:
            return errors

        # Relaciones OHLC
        o, h, l, c = float(bar["open"]), float(bar["high"]), float(bar["low"]), float(bar["close"])
        if h < max(o, c) or h < l:
            errors.append(f"High inválido: H={h} vs O={o} C={c} L={l}")
        if l > min(o, c) or l > h:
            errors.append(f"Low inválido: L={l} vs O={o} C={c} H={h}")

        return errors


# -------------------------------------
# 2) Buffer con estadísticas y duplicados
# -------------------------------------
@dataclass
class BufferStats:
    total_bars: int = 0
    duplicates_skipped: int = 0
    invalid_bars: int = 0
    last_update_ts: Optional[int] = None


class EnhancedTFBuffer:
    """Buffer por TF con validación, stats y manejo de actualizaciones en la misma barra."""

    def __init__(self, maxlen: int = _DEFAULT_WINDOW, tf_name: str = "unknown") -> None:
        self.tf_name = tf_name
        self.maxlen = maxlen

        self.ts: Deque[int] = deque(maxlen=maxlen)
        self.open: Deque[float] = deque(maxlen=maxlen)
        self.high: Deque[float] = deque(maxlen=maxlen)
        self.low: Deque[float] = deque(maxlen=maxlen)
        self.close: Deque[float] = deque(maxlen=maxlen)
        self.volume: Deque[float] = deque(maxlen=maxlen)

        self.stats = BufferStats()
        self.validator = DataQualityValidator()
        self.logger = logging.getLogger(f"features.buffer.{tf_name}")

    def push(self, bar: Dict[str, Any]) -> bool:
        """Inserta/actualiza una barra tras validar. Devuelve True si actualiza el estado."""
        try:
            errors = self.validator.validate_bar(bar)
            if errors:
                self.logger.warning(f"[{self.tf_name}] Bar inválida: {errors}")
                self.stats.invalid_bars += 1
                return False

            bar_ts = int(bar["ts"])

            # Duplicado exacto (misma barra, mismo close y volumen)
            if self.ts and self.ts[-1] == bar_ts:
                same_close = abs(self.close[-1] - float(bar["close"])) < 1e-8
                same_vol = abs(self.volume[-1] - float(bar["volume"])) < 1e-8
                if same_close and same_vol:
                    self.stats.duplicates_skipped += 1
                    return False
                # Actualización de la barra en curso → reemplazo
                self._replace_last_bar(bar)
                self.stats.last_update_ts = bar_ts
                return True

            # Nueva barra
            self.ts.append(bar_ts)
            self.open.append(float(bar["open"]))
            self.high.append(float(bar["high"]))
            self.low.append(float(bar["low"]))
            self.close.append(float(bar["close"]))
            self.volume.append(float(bar["volume"]))

            self.stats.total_bars += 1
            self.stats.last_update_ts = bar_ts
            return True

        except Exception as e:
            self.logger.error(f"[{self.tf_name}] Error procesando bar: {e}")
            self.stats.invalid_bars += 1
            return False

    def _replace_last_bar(self, bar: Dict[str, Any]) -> None:
        if not self.ts:
            return
        self.open[-1] = float(bar["open"])
        self.high[-1] = float(bar["high"])
        self.low[-1] = float(bar["low"])
        self.close[-1] = float(bar["close"])
        self.volume[-1] = float(bar["volume"])

    def to_numpy(self) -> Dict[str, np.ndarray]:
        return {
            "open": np.asarray(self.open, dtype=float),
            "high": np.asarray(self.high, dtype=float),
            "low": np.asarray(self.low, dtype=float),
            "close": np.asarray(self.close, dtype=float),
            "volume": np.asarray(self.volume, dtype=float),
        }

    def health_report(self) -> Dict[str, Any]:
        size = len(self.ts)
        return {
            "tf": self.tf_name,
            "buffer_size": size,
            "max_size": self.maxlen,
            "utilization_pct": (size / self.maxlen) * 100.0,
            "stats": vars(self.stats),
        }


# -----------------------------------
# 3) Pipeline de features (compatible)
# -----------------------------------
class FeaturePipeline:
    """Mantiene el nombre/clase pública para compatibilidad con el código existente."""

    def __init__(self, cfg: PipelineConfig) -> None:
        self.cfg = cfg
        self.window_size = getattr(cfg, "window", _DEFAULT_WINDOW) if cfg else _DEFAULT_WINDOW
        self._buffers: Dict[str, EnhancedTFBuffer] = {}
        self.logger = logging.getLogger("features.pipeline")
        self._error_counts: Dict[str, int] = {}

    def compute(self, mtf_bars: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """
        Actualiza buffers por TF y computa el último valor de los indicadores.
        Devuelve un dict: { tf: { ema20, ema50, rsi14, atr14, macd, macd_signal, macd_hist, bb_p, bb_w } }
        """
        feats: Dict[str, Any] = {}

        # 1) actualizar buffers con las barras actuales
        for tf, bar in mtf_bars.items():
            try:
                if tf not in self._buffers:
                    self._buffers[tf] = EnhancedTFBuffer(maxlen=self.window_size, tf_name=tf)
                updated = self._buffers[tf].push(bar)
                if not updated:
                    # Si no actualiza, no computamos (duplicado o inválida)
                    continue

                # 2) calcular indicadores para este TF
                tf_feats = self._compute_tf_features(tf)
                if tf_feats:
                    feats[tf] = tf_feats
                    self._error_counts[tf] = 0  # reset en éxito

            except Exception as e:
                self._error_counts[tf] = self._error_counts.get(tf, 0) + 1
                self.logger.error(f"[{tf}] Error en compute(): {e}")

                # Fallback tras demasiados errores seguidos
                if self._error_counts[tf] > 10:
                    feats[tf] = self._default_features()

        return feats

    def _compute_tf_features(self, tf: str) -> Optional[Dict[str, Any]]:
        try:
            buf = self._buffers[tf]
            arrays = buf.to_numpy()

            # Verificar datos suficientes
            if arrays["close"].size < _MIN_BARS_FOR_INDICATORS:
                return None

            closes = arrays["close"]
            highs = arrays["high"]
            lows = arrays["low"]

            features: Dict[str, Any] = {}

            # Cálculo de indicadores con protección de errores
            try:
                features["ema20"] = ema(closes, 20)
                features["ema50"] = ema(closes, 50)
                features["rsi14"] = rsi(closes, 14)
                features["atr14"] = atr(highs, lows, closes, 14)

                macd_result = macd(closes, 12, 26, 9)
                if macd_result:
                    features["macd"] = macd_result["macd"]
                    features["macd_signal"] = macd_result["signal"]
                    features["macd_hist"] = macd_result["hist"]
                else:
                    features["macd"] = features["macd_signal"] = features["macd_hist"] = 0.0

                bb_result = bollinger(closes, 20, 2.0)
                if bb_result:
                    features["bb_p"] = bb_result["pctB"]
                    features["bb_w"] = bb_result["width"]
                else:
                    features["bb_p"] = 0.5
                    features["bb_w"] = 0.02

            except Exception as ind_err:
                self.logger.error(f"[{tf}] Error calculando indicadores: {ind_err}")
                return self._default_features()

            return features

        except Exception as e:
            self.logger.error(f"[{tf}] Error en _compute_tf_features(): {e}")
            return None

    @staticmethod
    def _default_features() -> Dict[str, Any]:
        """Valores neutros/seguros para mantener forma estable."""
        return {
            "ema20": 0.0,
            "ema50": 0.0,
            "rsi14": 50.0,
            "atr14": 0.001,
            "macd": 0.0,
            "macd_signal": 0.0,
            "macd_hist": 0.0,
            "bb_p": 0.5,
            "bb_w": 0.02,
        }

    # --- utilidades opcionales de salud ---
    def system_health(self) -> Dict[str, Any]:
        return {
            "total_timeframes": len(self._buffers),
            "active_buffers": sum(1 for b in self._buffers.values() if len(b.ts) > 0),
            "error_counts": dict(self._error_counts),
            "buffers": {tf: buf.health_report() for tf, buf in self._buffers.items()},
        }
