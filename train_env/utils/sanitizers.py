# bot_trading_v9.1.6/train_env/utils/sanitizers.py
from dataclasses import dataclass
from typing import Optional, Dict, Any
import numpy as np

@dataclass
class OpenDecision:
    should_open: bool
    side: int                 # +1 long, -1 short, 0 hold
    price_hint: Optional[float]
    sl: Optional[float]
    tp: Optional[float]
    ttl_bars: int
    trailing: bool

def _simple_atr(view, lookback: int = 14) -> Optional[float]:
    """ATR simple para fallback. 'view' debe ser lista/array de tuplas (o,h,l,c,...)"""
    try:
        if view is None or len(view) < lookback + 1:
            return None
        tr = []
        prev_close = None
        for row in view:
            o, h, l, c = float(row[0]), float(row[1]), float(row[2]), float(row[3])
            if prev_close is None:
                tr.append(h - l)
            else:
                tr.append(max(h - l, abs(h - prev_close), abs(l - prev_close)))
            prev_close = c
        return float(np.mean(tr[-lookback:]))
    except Exception:
        return None

def sanitize_open_levels(
    decision: OpenDecision,
    price: float,
    dv,                       # tu DataView (para ATR)
    risk_cfg: Dict[str, Any], # config/risk.yaml (common.default_levels.., atr_fallback..)
    is_train: bool = True
) -> OpenDecision:
    """
    Garantiza SL/TP/TTL válidos antes del BYPASS POLICY.
    - Si faltan, los crea con % mínimo y/o ATR fallback.
    - Respeta flag allow_open_without_levels_train (solo TRAIN).
    """
    if not decision.should_open or decision.side == 0:
        return decision

    common = (risk_cfg or {}).get("common", {})
    defaults = common.get("default_levels", {}) or {}
    allow_fb = bool(common.get("allow_open_without_levels_train", True)) if is_train else False
    atr_cfg = common.get("atr_fallback", {}) or {}

    sl, tp, ttl = decision.sl, decision.tp, int(decision.ttl_bars)

    # Si ya están completos y TTL>0, no tocar
    if sl is not None and tp is not None and ttl > 0:
        return decision

    if not allow_fb and (sl is None or tp is None or ttl <= 0):
        # En LIVE puedes impedir el fallback si así lo quieres
        return decision

    min_sl_pct = float(defaults.get("min_sl_pct", 1.0))
    tp_r_mult  = float(defaults.get("tp_r_multiple", 1.5))
    ttl_def    = int(defaults.get("ttl_bars_default", 180))

    sl_dist = price * (min_sl_pct / 100.0)
    print(f"🔧 DISTANCIA SL INICIAL: {sl_dist:.4f} (1% de {price:.2f})")

    # ATR fallback
    if bool(atr_cfg.get("enabled", True)) and dv is not None:
        tf = atr_cfg.get("tf", "1m")
        lb = int(atr_cfg.get("lookback", 14))
        atr = _simple_atr(dv.view(tf, lb + 1), lookback=lb)
        if atr is not None:
            mult = float(atr_cfg.get("min_sl_atr_mult", 1.2))
            atr_dist = atr * mult
            # ← CRÍTICO: Asegurar que ATR no sea menor que la distancia mínima por porcentaje
            sl_dist = max(sl_dist, atr_dist)
            print(f"🔧 ATR FALLBACK: ATR={atr:.4f}, ATR_DIST={atr_dist:.4f}, SL_DIST_FINAL={sl_dist:.4f}")
    
    # ← CRÍTICO: GARANTIZAR que la distancia SL sea al menos 1% del precio
    print(f"🔧 INICIANDO VERIFICACIÓN DE DISTANCIA: sl_dist={sl_dist:.4f}, price={price:.2f}")
    min_sl_dist_absolute = price * (min_sl_pct / 100.0)
    print(f"🔧 VERIFICANDO DISTANCIA: sl_dist={sl_dist:.4f}, min_required={min_sl_dist_absolute:.4f}")
    if sl_dist < min_sl_dist_absolute:
        sl_dist = min_sl_dist_absolute
        print(f"🔧 FORZANDO DISTANCIA MÍNIMA: {sl_dist:.4f} (1% de {price:.2f})")
    else:
        print(f"🔧 DISTANCIA OK: {sl_dist:.4f} >= {min_sl_dist_absolute:.4f}")

    if sl is None:
        sl = price - sl_dist if decision.side > 0 else price + sl_dist
    if tp is None:
        tp = price + tp_r_mult * sl_dist if decision.side > 0 else price - tp_r_mult * sl_dist
    if ttl <= 0:
        ttl = ttl_def

    return OpenDecision(
        should_open=True,
        side=int(np.sign(decision.side)),
        price_hint=price,
        sl=float(sl),
        tp=float(tp),
        ttl_bars=int(ttl),
        trailing=bool(decision.trailing),
    )
