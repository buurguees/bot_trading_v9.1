# train_env/reward_shaper.py
# Lee config/rewards.yaml y calcula el reward con:
# - Tramos por ROI% en el cierre (CLOSE)
# - Bonus/malus por TP_HIT / SL_HIT
# - Refuerzos por R-multiple y eficiencia de riesgo
# - Componentes continuos: realized (entorno), unrealized, time_penalty, trade_cost, dd_penalty
# - NUEVO: Sistema de rewards por leverage y timeframe

from __future__ import annotations
from typing import Dict, Any, Tuple, List
import yaml
import math
from base_env.actions import RewardOrchestrator


class RewardShaper:
    def __init__(self, yaml_path: str):
        with open(yaml_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        self.cfg = cfg
        
        # ← NUEVO: Sistema avanzado de rewards (incluye todos los sistemas)
        self.advanced_rewards = RewardOrchestrator(yaml_path)
        
        # Mantener compatibilidad con sistema anterior
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
        self.w_empty_run = float(w.get("empty_run_penalty", 0.0))
        self.w_balance_milestone = float(w.get("balance_milestone_reward", 0.0))
        self.w_survival = float(w.get("survival_bonus", 0.0))
        self.w_progress = float(w.get("progress_bonus", 0.0))
        self.w_compound = float(w.get("compound_bonus", 0.0))
        self.clip_lo, self.clip_hi = cfg.get("reward_clip", [-float("inf"), float("inf")])
        
        # Contador de steps para el nuevo sistema
        self.current_step = 0

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
    def compute_advanced_trade_reward(self, 
                                    realized_pnl: float,
                                    notional: float,
                                    leverage_used: float,
                                    r_multiple: float,
                                    close_reason: str,
                                    timeframe_used: str,
                                    bars_held: int) -> float:
        """
        Calcula reward avanzado para un trade cerrado usando el nuevo sistema
        
        Args:
            realized_pnl: PnL realizado del trade
            notional: Notional del trade
            leverage_used: Leverage usado en el trade
            r_multiple: R-multiple del trade
            close_reason: Razón del cierre (tp_hit, sl_hit, ttl_hit, manual)
            timeframe_used: Timeframe usado para la ejecución
            bars_held: Número de barras que se mantuvo la posición
            
        Returns:
            Reward total del trade
        """
        return self.advanced_rewards.calculate_trade_reward(
            realized_pnl=realized_pnl,
            notional=notional,
            leverage_used=leverage_used,
            r_multiple=r_multiple,
            close_reason=close_reason,
            timeframe_used=timeframe_used,
            bars_held=bars_held
        )
    
    def compute(self, obs: Dict[str, Any], base_reward: float, events: List[Dict[str, Any]], 
                empty_run: bool = False, balance_milestones: int = 0, 
                initial_balance: float = 1000.0, target_balance: float = 1000000.0,
                steps_since_last_trade: int = 0, bankruptcy_occurred: bool = False) -> Tuple[float, Dict[str, float]]:
        """
        Calcula el reward usando el nuevo sistema granular de rewards.
        """
        # Incrementar contador de steps
        self.current_step += 1
        
        # ← NUEVO: Usar el sistema de rewards orquestado
        granular_reward, granular_components = self.advanced_rewards.compute_reward(
            obs=obs,
            base_reward=0.0,  # El reward base se maneja por separado
            events=events,
            empty_run=empty_run,
            balance_milestones=balance_milestones,
            initial_balance=initial_balance,
            target_balance=target_balance,
            steps_since_last_trade=steps_since_last_trade,
            bankruptcy_occurred=bankruptcy_occurred
        )
        
        # Combinar con sistema anterior para compatibilidad
        legacy_reward = 0.0
        legacy_components = {}
        
        # Mantener algunos componentes del sistema anterior
        pos = obs.get("position", {}) or {}
        portfolio = obs.get("portfolio", {}) or {}
        current_equity = float(portfolio.get("equity_quote", initial_balance))
        
        # Bonus por milestones de balance
        if balance_milestones > 0:
            milestone_bonus = self.w_balance_milestone * balance_milestones
            legacy_reward += milestone_bonus
            legacy_components["balance_milestone"] = milestone_bonus
        
        # Penalty por runs vacíos
        if empty_run:
            empty_penalty = self.w_empty_run
            legacy_reward += empty_penalty
            legacy_components["empty_run_penalty"] = empty_penalty
        
        # Bonus por progreso hacia objetivo
        if current_equity > initial_balance:
            progress = (current_equity - initial_balance) / (target_balance - initial_balance)
            if progress > 0:
                progress_bonus = self.w_progress * progress
                legacy_reward += progress_bonus
                legacy_components["progress_bonus"] = progress_bonus
        
        # Combinar rewards
        total_reward = granular_reward + legacy_reward
        all_components = {**granular_components, **legacy_components}
        
        # Aplicar clipping
        clipped_reward = self._clip(total_reward)
        
        return clipped_reward, all_components
