# base_env/feature_engine.py
from __future__ import annotations

# Intentamos importar tu implementación (TA-Lib)
try:
    from base_env.feature_store import FeatureConfig as _FC_TA, IndicatorCalculator as _IC_TA  # :contentReference[oaicite:3]{index=3}
    _HAS_TALIB = True
except Exception:
    _HAS_TALIB = False

# Fallback puro numpy/pandas
from base_env.indicator_fallback import FeatureConfig as _FC_FB, IndicatorCalculator as _IC_FB

# Export homogéneo
FeatureConfig = _FC_TA if _HAS_TALIB else _FC_FB
IndicatorCalculator = _IC_TA if _HAS_TALIB else _IC_FB
