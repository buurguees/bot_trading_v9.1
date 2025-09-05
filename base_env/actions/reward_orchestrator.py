# base_env/actions/reward_orchestrator.py
"""
Orquestador principal del sistema de rewards/penalties.
Coordina todos los sistemas individuales de rewards y penalties.
"""

from __future__ import annotations
from typing import Dict, Any, List, Tuple, Optional
import yaml
from pathlib import Path

# Importar todos los sistemas de rewards/penalties
from .take_profit_reward import TakeProfitReward
from .stop_loss_penalty import StopLossPenalty
from .bankruptcy_penalty import BankruptcyPenalty
from .holding_reward import HoldingReward
from .inactivity_penalty import InactivityPenalty
from .roi_reward import ROIReward
from .r_multiple_reward import RMultipleReward
from .leverage_reward import LeverageReward
from .timeframe_reward import TimeframeReward
from .duration_reward import DurationReward
from .progress_bonus import ProgressBonus
from .blocked_trade_penalty import BlockedTradePenalty
from .sl_hit_penalty import SLHitPenalty
from .tp_hit_reward import TPHitReward
from .progress_milestone_reward import ProgressMilestoneReward

# Sistemas avanzados
from .volatility_reward import VolatilityReward
from .drawdown_penalty import DrawdownPenalty
from .execution_cost_penalty import ExecutionCostPenalty
from .mtf_alignment_reward import MTFAlignmentReward
from .time_efficiency_reward import TimeEfficiencyReward
from .overtrading_penalty import OvertradingPenalty
from .ttl_expiry_penalty import TTLExpiryPenalty
from .exploration_bonus import ExplorationBonus

# Nuevos módulos de reward shaping
from collections import deque, defaultdict
import numpy as np


class RewardOrchestrator:
    """Orquestador principal del sistema de rewards/penalties"""
    
    def __init__(self, rewards_config_path: str = "config/rewards.yaml"):
        """
        Inicializa el orquestador de rewards
        
        Args:
            rewards_config_path: Ruta al archivo de configuración de rewards
        """
        self.config_path = Path(rewards_config_path)
        self.config = self._load_config()
        
        # Inicializar todos los sistemas de rewards/penalties
        self.tp_reward = TakeProfitReward(self.config)
        self.sl_penalty = StopLossPenalty(self.config)
        self.bankruptcy_penalty = BankruptcyPenalty(self.config)
        self.holding_reward = HoldingReward(self.config)
        self.inactivity_penalty = InactivityPenalty(self.config)
        self.roi_reward = ROIReward(self.config)
        self.r_multiple_reward = RMultipleReward(self.config)
        self.leverage_reward = LeverageReward(self.config)
        self.timeframe_reward = TimeframeReward(self.config)
        self.duration_reward = DurationReward(self.config)
        self.progress_bonus = ProgressBonus(self.config)
        self.blocked_trade_penalty = BlockedTradePenalty(self.config)
        
        # Inicializar sistemas de SL/TP hits
        sl_tp_config = self.config.get("sl_tp_hits", {})
        self.sl_hit_penalty = SLHitPenalty(sl_tp_config.get("sl_hit_penalty", {}))
        self.tp_hit_reward = TPHitReward(sl_tp_config.get("tp_hit_reward", {}))
        
        # Inicializar sistema de milestones de progreso
        self.progress_milestone_reward = ProgressMilestoneReward(self.config)
        
        # Inicializar sistemas avanzados
        self.volatility_reward = VolatilityReward(self.config)
        self.drawdown_penalty = DrawdownPenalty(self.config)
        self.execution_cost_penalty = ExecutionCostPenalty(self.config)
        self.mtf_alignment_reward = MTFAlignmentReward(self.config)
        self.time_efficiency_reward = TimeEfficiencyReward(self.config)
        self.overtrading_penalty = OvertradingPenalty(self.config)
        self.ttl_expiry_penalty = TTLExpiryPenalty(self.config)
        self.exploration_bonus = ExplorationBonus(self.config)
        
        # Configuración de clipping
        self.clip_lo, self.clip_hi = self.config.get("reward_clip", [-float("inf"), float("inf")])
        
        # Contador de steps
        self.current_step = 0
        
        # Estado para nuevos módulos de reward shaping
        self._bars_since_last_trade = 0
        self._current_day_key = None
        self._trades_today = 0
        self._daily_bonus_accum = 0.0
        self._daily_penalty_accum = 0.0
        self._last_side = 0
        self._last_close_ts = None
        self._last_event = None
        self._rollover_reward = 0.0
    
    def initialize_run(self, initial_balance: float, target_balance: float):
        """
        Inicializa el sistema para un nuevo run
        
        Args:
            initial_balance: Balance inicial del run
            target_balance: Balance objetivo
        """
        self.progress_milestone_reward.initialize_run(initial_balance, target_balance)
        self.overtrading_penalty.reset()
        self.exploration_bonus.reset()
        self.current_step = 0
    
    def _load_config(self) -> Dict[str, Any]:
        """Carga la configuración desde rewards.yaml"""
        if not self.config_path.exists():
            raise FileNotFoundError(f"Archivo de configuración no encontrado: {self.config_path}")
        
        with open(self.config_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    
    def _clip(self, r: float) -> float:
        """Aplica clipping al reward"""
        return max(self.clip_lo, min(self.clip_hi, r))
    
    def _day_key(self, info: Dict[str, Any], cfg: Dict[str, Any]) -> int:
        """Calcula la clave del día lógico"""
        ts = int(info.get("bar_time", 0))  # ms
        minutes = int(cfg.get("tf_minutes", 1))
        day_len_ms = 60_000 * 60 * 24
        return ts - (ts % day_len_ms)
    
    def _on_new_bar(self, info: Dict[str, Any]):
        """Llamado en cada step"""
        self._bars_since_last_trade += 1
    
    def _trade_activity_daily_reward(self, info: Dict[str, Any], is_close_event: bool) -> float:
        """Módulo de actividad diaria de trades"""
        c = self.config.get("trade_activity_daily", {})
        if not c.get("enabled", False):
            return 0.0

        # día actual
        day_key = self._day_key(info, c)
        if self._current_day_key is None:
            self._current_day_key = day_key
            self._trades_today = 0
            self._daily_bonus_accum = 0.0
            self._daily_penalty_accum = 0.0
        elif day_key != self._current_day_key:
            # día nuevo: aplicar penalizaciones del día anterior
            target = max(1, int(c.get("target_trades_per_day", 1)))
            warmup_days = int(c.get("warmup_days", 7))
            shortfall_penalty = float(c.get("shortfall_penalty", 0.04))
            overtrade_penalty = float(c.get("overtrade_penalty", 0.02))
            max_p = float(c.get("max_daily_penalty", -0.05))
            
            if self._trades_today < target and warmup_days <= 0:
                missing = target - self._trades_today
                pen = -shortfall_penalty * (missing / target)
                pen = max(max_p, pen)
                self._daily_penalty_accum += pen
                self._rollover_reward = pen
            elif self._trades_today > target:
                excess = self._trades_today - target
                pen = -overtrade_penalty * (excess / target)
                pen = max(max_p, pen)
                self._daily_penalty_accum += pen
                self._rollover_reward = pen
            else:
                self._rollover_reward = 0.0
            
            # reset day counters
            self._current_day_key = day_key
            self._trades_today = 0
            self._daily_bonus_accum = 0.0
            self._daily_penalty_accum = 0.0

        target = max(1, int(c.get("target_trades_per_day", 1)))
        bonus_per_trade = float(c.get("bonus_per_trade", 0.03))
        max_b = float(c.get("max_daily_bonus", 0.05))
        decay_after_hit = float(c.get("decay_after_hit", 0.5))

        r = 0.0

        # bonus por trade cuando ocurre el CIERRE exitoso
        if is_close_event:
            b = bonus_per_trade
            if self._trades_today >= target:
                b *= decay_after_hit
            # acumula con cap diario
            if self._daily_bonus_accum + b > max_b:
                b = max(0.0, max_b - self._daily_bonus_accum)
            self._daily_bonus_accum += b
            self._trades_today += 1
            r += b

        # aplicar rollover reward del día anterior (una sola vez)
        if self._rollover_reward != 0.0:
            r += self._rollover_reward
            self._rollover_reward = 0.0

        return float(r)
    
    def _inactivity_escalator(self) -> float:
        """Penalización escalonada por inactividad"""
        c = self.config.get("inactivity_escalator", {})
        if not c.get("enabled", False):
            return 0.0
        start = int(c.get("start_after_bars", 720))
        step_every = int(c.get("step_every_bars", 180))
        step_pen = float(c.get("step_penalty", -0.002))
        cap = float(c.get("max_penalty", -0.04))

        if self._bars_since_last_trade < start:
            return 0.0
        steps = (self._bars_since_last_trade - start) // step_every + 1
        r = steps * step_pen
        return float(max(cap, r))
    
    def _quality_open_bonus(self, info: Dict[str, Any]) -> float:
        """Bonus por calidad de apertura"""
        c = self.config.get("quality_open_bonus", {})
        if not c.get("enabled", False):
            return 0.0
        if info.get("event") != "OPEN":
            return 0.0

        b = 0.0
        if info.get("mtf_agree", False):
            b += float(c.get("mtf_agree_bonus", 0.01))
        if float(info.get("spread_cost_bps", 99)) <= float(c.get("spread_cost_cap_bps", 2)):
            b += float(c.get("bonus", 0.01))
        if float(info.get("sl_to_atr", 0.0)) >= float(c.get("min_sl_to_atr", 1.0)):
            b += float(c.get("bonus", 0.01))

        cap = float(c.get("per_trade_cap", 0.02))
        return float(min(b, cap))
    
    def _anti_flip_penalty(self, info: Dict[str, Any]) -> float:
        """Penalización anti-ping-pong"""
        c = self.config.get("anti_flip_penalty", {})
        if not c.get("enabled", False):
            return 0.0
        if info.get("event") != "OPEN":
            return 0.0
        side = int(np.sign(info.get("position_side", 0)))
        pen = 0.0
        if self._last_close_ts is not None and side != 0 and self._last_side != 0 and side != self._last_side:
            # flip
            bars = int(info.get("bars_since_last_close", 1e9))
            if bars <= int(c.get("window_bars", 30)):
                if not (c.get("ignore_if_profit_tp", True) and self._last_event == "TP"):
                    pen = float(c.get("penalty", -0.01))
        return float(pen)
    
    def _policy_signal_reward(self, obs: Dict[str, Any], events: List[Dict[str, Any]]) -> float:
        """Reward por seguir las señales de la política"""
        c = self.config.get("policy_signal_reward", {})
        if not c.get("enabled", False):
            return 0.0
        
        reward = 0.0
        
        # Obtener información de la política
        policy_signal = obs.get("policy_signal", {})
        should_open = policy_signal.get("should_open", False)
        side_hint = policy_signal.get("side_hint", 0)
        confidence = policy_signal.get("confidence", 0.0)
        
        # Verificar si hay eventos de apertura
        for event in events:
            if event.get("event") == "OPEN":
                # Reward por abrir cuando la política dice que debe abrir
                if should_open and side_hint != 0:
                    reward += float(c.get("follow_signal_bonus", 0.05))
                    # Bonus adicional por alta confianza
                    if confidence > 0.5:
                        reward += 0.02
                break
        
        # Penalty por no abrir cuando la política dice que debe abrir
        if should_open and side_hint != 0 and not any(e.get("event") == "OPEN" for e in events):
            reward += float(c.get("ignore_signal_penalty", -0.02))
        
        # Verificar señales de cierre
        should_close = policy_signal.get("should_close", False)
        for event in events:
            if event.get("event") in ("TP", "SL", "CLOSE", "TTL"):
                if should_close:
                    reward += float(c.get("follow_close_signal_bonus", 0.03))
                break
        
        return float(reward)
    
    def calculate_trade_reward(self, 
                             realized_pnl: float,
                             notional: float,
                             leverage_used: float,
                             r_multiple: float,
                             close_reason: str,
                             timeframe_used: str,
                             bars_held: int) -> float:
        """
        Calcula el reward total para un trade cerrado
        
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
        reward = 0.0
        
        # 1. Reward por ROI
        roi_pct = (realized_pnl / notional) * 100.0 if notional > 0 else 0.0
        roi_reward, _ = self.roi_reward.calculate_roi_reward(roi_pct)
        reward += roi_reward
        
        # 2. Reward por R-Multiple
        r_reward, _ = self.r_multiple_reward.calculate_r_multiple_reward(r_multiple)
        reward += r_reward
        
        # 3. Reward/Penalty por leverage
        leverage_reward, _ = self.leverage_reward.calculate_leverage_reward(realized_pnl, leverage_used)
        reward += leverage_reward
        
        # 4. Bonus por timeframe usado
        tf_reward, _ = self.timeframe_reward.calculate_timeframe_reward(timeframe_used)
        reward += tf_reward
        
        # 5. Reward por duración del trade (bars_held)
        duration_reward, _ = self.duration_reward.calculate_duration_reward(bars_held)
        reward += duration_reward
        
        # 6. Reward/Penalty por SL/TP hits según close_reason
        if close_reason == "sl_hit":
            sl_penalty = self.sl_hit_penalty.calculate_penalty(realized_pnl, notional, leverage_used)
            reward += sl_penalty
        elif close_reason == "tp_hit":
            tp_reward = self.tp_hit_reward.calculate_reward(realized_pnl, notional, leverage_used)
            reward += tp_reward
        
        # 7. Penalización por expiración TTL
        ttl_penalty, _ = self.ttl_expiry_penalty.calculate_ttl_expiry_penalty(close_reason, realized_pnl)
        reward += ttl_penalty
        
        # 8. Bonus por exploración de leverage/timeframe
        exploration_bonus, _ = self.exploration_bonus.calculate_exploration_bonus(leverage_used, timeframe_used)
        reward += exploration_bonus
        
        return reward
    
    def calculate_advanced_trade_reward(self, realized_pnl: float, notional: float, 
                                      leverage_used: float, r_multiple: float, 
                                      close_reason: str, timeframe_used: str, 
                                      bars_held: int, price: float, 
                                      max_drawdown_vs_sl: float, sl_distance: float,
                                      fees_paid: float, qty: float, 
                                      trade_side: int, obs: Dict[str, Any]) -> Tuple[float, Dict[str, float]]:
        """
        Calcula reward avanzado para un trade cerrado con todos los sistemas
        
        Args:
            realized_pnl: PnL realizado del trade
            notional: Notional del trade
            leverage_used: Leverage usado en el trade
            r_multiple: R-multiple del trade
            close_reason: Razón del cierre
            timeframe_used: Timeframe usado
            bars_held: Número de barras que se mantuvo
            price: Precio de entrada
            max_drawdown_vs_sl: Máximo drawdown vs SL
            sl_distance: Distancia del SL
            fees_paid: Fees pagados
            qty: Cantidad ejecutada
            trade_side: Dirección del trade (+1/-1)
            obs: Observación del entorno
            
        Returns:
            Tupla (reward_total, componentes_detallados)
        """
        all_components = {}
        total_reward = 0.0
        
        # 1. Reward base del trade (sistemas existentes)
        base_trade_reward = self.calculate_trade_reward(
            realized_pnl, notional, leverage_used, r_multiple, 
            close_reason, timeframe_used, bars_held
        )
        total_reward += base_trade_reward
        all_components["base_trade_reward"] = base_trade_reward
        
        # 2. PnL normalizado por volatilidad
        atr_value = self.volatility_reward.get_atr_from_obs(obs)
        if atr_value and atr_value > 0:
            vol_reward, vol_components = self.volatility_reward.calculate_volatility_reward(
                realized_pnl, price, atr_value, notional
            )
            total_reward += vol_reward
            all_components.update(vol_components)
        
        # 3. Penalización por drawdown intra-trade
        if sl_distance > 0:
            dd_penalty, dd_components = self.drawdown_penalty.calculate_drawdown_penalty(
                realized_pnl, max_drawdown_vs_sl, sl_distance, notional
            )
            total_reward += dd_penalty
            all_components.update(dd_components)
        
        # 4. Costes de ejecución
        cost_penalty, cost_components = self.execution_cost_penalty.calculate_execution_cost_penalty(
            fees_paid, notional, price, qty
        )
        total_reward += cost_penalty
        all_components.update(cost_components)
        
        # 5. Alineación multi-timeframe
        mtf_reward, mtf_components = self.mtf_alignment_reward.calculate_mtf_alignment_reward(
            trade_side, obs
        )
        total_reward += mtf_reward
        all_components.update(mtf_components)
        
        # 6. Eficiencia temporal
        time_eff_reward, time_eff_components = self.time_efficiency_reward.calculate_time_efficiency_reward(
            realized_pnl, bars_held, notional
        )
        total_reward += time_eff_reward
        all_components.update(time_eff_components)
        
        return total_reward, all_components
    
    def compute_reward(self, obs: Dict[str, Any], base_reward: float, events: List[Dict[str, Any]], 
                      empty_run: bool = False, balance_milestones: int = 0, 
                      initial_balance: float = 1000.0, target_balance: float = 1000000.0,
                      steps_since_last_trade: int = 0, bankruptcy_occurred: bool = False) -> Tuple[float, Dict[str, float]]:
        """
        Calcula el reward total usando todos los sistemas
        
        Args:
            obs: Observación actual del entorno
            base_reward: Reward base del entorno
            events: Lista de eventos del step actual
            empty_run: Si el run está vacío
            balance_milestones: Número de milestones alcanzados
            initial_balance: Balance inicial
            target_balance: Balance objetivo
            steps_since_last_trade: Pasos desde el último trade
            bankruptcy_occurred: Si ocurrió bancarrota
            
        Returns:
            Tupla (reward_total, componentes_detallados)
        """
        # Incrementar contador de steps
        self.current_step += 1
        
        # Llamar _on_new_bar para actualizar contadores
        self._on_new_bar(obs)
        
        # Obtener información del estado actual
        position = obs.get("position", {})
        portfolio = obs.get("portfolio", {})
        current_equity = float(portfolio.get("equity_quote", initial_balance))
        
        all_components = {}
        total_reward = base_reward
        
        # Detectar eventos de cierre para actualizar estado
        is_close_event = False
        for event in events:
            if event.get("event") in ("TP", "SL", "CLOSE", "TTL"):
                is_close_event = True
                self._last_close_ts = int(obs.get("bar_time", 0))
                self._last_event = event.get("event")
                self._last_side = int(np.sign(event.get("closed_side", 0)))
                self._bars_since_last_trade = 0
                break
        
        # 1. Progress Milestone Rewards (una vez por run)
        milestone_reward, milestone_components = self.progress_milestone_reward.calculate_progress_milestone_reward(current_equity)
        total_reward += milestone_reward
        all_components.update(milestone_components)
        
        # 2. Take Profit Rewards
        tp_reward, tp_components = self.tp_reward.calculate_tp_reward(events)
        total_reward += tp_reward
        all_components.update(tp_components)
        
        # 3. Stop Loss Penalties
        sl_penalty, sl_components = self.sl_penalty.calculate_sl_penalty(events)
        total_reward += sl_penalty
        all_components.update(sl_components)
        
        # 4. Bankruptcy Penalties y Survival Bonus
        bankruptcy_penalty, bankruptcy_components = self.bankruptcy_penalty.calculate_bankruptcy_penalty(
            events, current_equity, initial_balance
        )
        total_reward += bankruptcy_penalty
        all_components.update(bankruptcy_components)
        
        # 5. Holding Rewards
        holding_reward, holding_components = self.holding_reward.calculate_holding_reward(
            obs, current_equity, initial_balance
        )
        total_reward += holding_reward
        all_components.update(holding_components)
        
        # 6. Inactivity Penalties
        inactivity_penalty, inactivity_components = self.inactivity_penalty.calculate_inactivity_penalty(
            events, self.current_step
        )
        total_reward += inactivity_penalty
        all_components.update(inactivity_components)
        
        # 7. Progress Bonus
        progress_bonus, progress_components = self.progress_bonus.calculate_progress_bonus(
            current_equity, initial_balance, target_balance, balance_milestones
        )
        total_reward += progress_bonus
        all_components.update(progress_components)
        
        # 8. Blocked Trade Penalties
        blocked_penalty, blocked_components = self.blocked_trade_penalty.calculate_blocked_trade_penalty(
            events, empty_run
        )
        total_reward += blocked_penalty
        all_components.update(blocked_components)
        
        # 9. Penalización por Overtrading
        overtrading_penalty, overtrading_components = self.overtrading_penalty.calculate_overtrading_penalty(
            events, ((current_equity - initial_balance) / initial_balance) * 100.0
        )
        total_reward += overtrading_penalty
        all_components.update(overtrading_components)
        
        # 10. Nuevos módulos de reward shaping
        # 10.1 Trade Activity Daily (objetivo ~1 trade/día)
        trade_activity_reward = self._trade_activity_daily_reward(obs, is_close_event)
        total_reward += trade_activity_reward
        all_components["trade_activity_daily"] = trade_activity_reward
        
        # 10.2 Inactivity Escalator (penalización suave por inactividad)
        inactivity_escalator_penalty = self._inactivity_escalator()
        total_reward += inactivity_escalator_penalty
        all_components["inactivity_escalator"] = inactivity_escalator_penalty
        
        # 10.3 Quality Open Bonus (solo en aperturas)
        for event in events:
            if event.get("event") == "OPEN":
                quality_bonus = self._quality_open_bonus(event)
                total_reward += quality_bonus
                all_components["quality_open_bonus"] = quality_bonus
                
                # Anti-flip penalty (solo en aperturas)
                anti_flip_penalty = self._anti_flip_penalty(event)
                total_reward += anti_flip_penalty
                all_components["anti_flip_penalty"] = anti_flip_penalty
                break
        
        # 10.4 Policy Signal Reward (seguir señales de la política)
        policy_signal_reward = self._policy_signal_reward(obs, events)
        total_reward += policy_signal_reward
        all_components["policy_signal_reward"] = policy_signal_reward
        
        # Aplicar clipping global
        clipping_config = self.config.get("clipping", {})
        per_step_clip = clipping_config.get("per_step", [-0.15, 0.15])
        clipped_reward = max(per_step_clip[0], min(per_step_clip[1], total_reward))
        
        return clipped_reward, all_components
    
    def reset(self):
        """Resetea todos los sistemas"""
        self.tp_reward.reset()
        self.sl_penalty.reset()
        self.bankruptcy_penalty.reset()
        self.holding_reward.reset()
        self.inactivity_penalty.reset()
        self.roi_reward.reset()
        self.r_multiple_reward.reset()
        self.leverage_reward.reset()
        self.timeframe_reward.reset()
        self.duration_reward.reset()
        self.progress_bonus.reset()
        self.blocked_trade_penalty.reset()
        self.current_step = 0
        
        # Resetear estado de nuevos módulos
        self._bars_since_last_trade = 0
        self._current_day_key = None
        self._trades_today = 0
        self._daily_bonus_accum = 0.0
        self._daily_penalty_accum = 0.0
        self._last_side = 0
        self._last_close_ts = None
        self._last_event = None
        self._rollover_reward = 0.0
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Obtiene estadísticas de todos los sistemas
        
        Returns:
            Diccionario con estadísticas
        """
        return {
            "current_step": self.current_step,
            "config_path": str(self.config_path),
            "clip_range": [self.clip_lo, self.clip_hi],
            "systems": {
                "take_profit": self.tp_reward.get_stats(),
                "stop_loss": self.sl_penalty.get_stats(),
                "bankruptcy": self.bankruptcy_penalty.get_stats(),
                "holding": self.holding_reward.get_stats(),
                "inactivity": self.inactivity_penalty.get_stats(),
                "roi": self.roi_reward.get_stats(),
                "r_multiple": self.r_multiple_reward.get_stats(),
                "leverage": self.leverage_reward.get_stats(),
                "timeframe": self.timeframe_reward.get_stats(),
                "duration": self.duration_reward.get_stats(),
                "progress": self.progress_bonus.get_stats(),
                "blocked_trade": self.blocked_trade_penalty.get_stats()
            }
        }
