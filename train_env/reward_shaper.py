# train_env/reward_shaper.py
# Lee config/rewards.yaml y calcula el reward con:
# - Tramos por ROI% en el cierre (CLOSE)
# - Bonus/malus por TP_HIT / SL_HIT
# - Refuerzos por R-multiple y eficiencia de riesgo
# - Componentes continuos: realized (entorno), unrealized, time_penalty, trade_cost, dd_penalty

from __future__ import annotations
from typing import Dict, Any, Tuple, List
import yaml
import math


class RewardShaper:
    def __init__(self, yaml_path: str):
        with open(yaml_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        self.cfg = cfg
        self.tiers_pos: List[List[float]] = cfg.get("tiers", {}).get("pos", [])
        self.tiers_neg: List[List[float]] = cfg.get("tiers", {}).get("neg", [])
        self.bon_tp = float(cfg.get("bonuses", {}).get("tp_hit", 0.0))
        self.bon_sl = float(cfg.get("bonuses", {}).get("sl_hit", 0.0))
        w = cfg.get("weights", {})
        self.w_realized = float(w.get("realized_pnl", 0.0))
        self.w_unreal = float(w.get("unrealized_pnl", 0.0))
        self.w_rmult = float(w.get("r_multiple", 0.0))
        self.w_risk_eff = float(w.get("risk_efficiency", 0.0))
        self.w_time = float(w.get("time_penalty", 0.0))
        self.w_trade_cost = float(w.get("trade_cost", 0.0))
        self.w_dd = float(w.get("dd_penalty", 0.0))
        self.clip_lo, self.clip_hi = cfg.get("reward_clip", [-float("inf"), float("inf")])

    # ---------- helpers ----------
    def _clip(self, r: float) -> float:
        return max(self.clip_lo, min(self.clip_hi, r))

    @staticmethod
    def _tier_value(tiers: List[List[float]], pct: float) -> float:
        """Devuelve el valor del tramo donde cae pct (en valor absoluto para tiers_neg)."""
        for lo, hi, val in tiers:
            if lo <= pct <= hi:
                return float(val)
        return float(tiers[-1][2]) if tiers else 0.0

    # ---------- público ----------
    def compute(self, obs: Dict[str, Any], base_reward: float, events: List[Dict[str, Any]]) -> Tuple[float, Dict[str, float]]:
        pos = obs.get("position", {}) or {}
        portfolio = obs.get("portfolio", {}) or {}

        realized_usd = float(base_reward)                           # del env (cierres)
        unreal_usd = float(pos.get("unrealized_pnl", 0.0))          # guía suave
        in_position = int(pos.get("side", 0)) != 0
        dd_day = float(portfolio.get("drawdown_day_pct", 0.0))      # 0..1 si se expone

        reward = 0.0
        # Componentes continuos
        reward += self.w_realized * realized_usd
        reward += self.w_unreal * unreal_usd
        reward += self.w_time * (1.0 if in_position else 0.0)
        reward += self.w_dd * (-abs(dd_day))

        # Coste de apertura (si hay OPEN en este step)
        if any(e.get("kind") == "OPEN" for e in events):
            reward += self.w_trade_cost

        # --- Eventos de cierre: aplicar ROI% por tramos, R-multiple y TP/SL bonus ---
        # Buscamos el primer CLOSE y los flags de TP/SL en el mismo step
        close_ev = next((e for e in events if e.get("kind") == "CLOSE"), None)
        tp_hit = any(e.get("kind") == "TP_HIT" for e in events)
        sl_hit = any(e.get("kind") == "SL_HIT" for e in events)

        if close_ev:
            # Esperamos que el entorno envíe estos campos:
            # entry_price, qty, realized_pnl, roi_pct, r_multiple, risk_pct
            roi_pct = float(close_ev.get("roi_pct", 0.0))
            r_mult = float(close_ev.get("r_multiple", 0.0))
            risk_pct = float(close_ev.get("risk_pct", 0.0))  # % riesgo inicial (distancia SL / entry * 100)

            if roi_pct >= 0:
                reward += self._tier_value(self.tiers_pos, roi_pct)
            else:
                reward += self._tier_value(self.tiers_neg, abs(roi_pct))

            # Bonus por TP/SL
            if tp_hit:
                reward += self.bon_tp
            if sl_hit:
                reward += self.bon_sl

            # Refuerzos por calidad del trade
            reward += self.w_rmult * r_mult
            # Eficiencia de riesgo: ROI% por cada 1% de riesgo usado (cuanto menor riesgo para mismo ROI, mejor)
            if risk_pct > 0:
                risk_eff = (abs(roi_pct) / risk_pct)
                reward += self.w_risk_eff * risk_eff

        return self._clip(reward), {
            "realized_usd": realized_usd,
            "unreal_usd": unreal_usd,
            "dd": dd_day,
        }
