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
        
        # Obtener información del estado actual
        position = obs.get("position", {})
        portfolio = obs.get("portfolio", {})
        current_equity = float(portfolio.get("equity_quote", initial_balance))
        
        all_components = {}
        total_reward = base_reward
        
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
        
        # Aplicar clipping
        clipped_reward = self._clip(total_reward)
        
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
