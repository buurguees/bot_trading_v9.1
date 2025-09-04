# base_env/actions/__init__.py
"""
Sistema modular de rewards y penalties para trading algorítmico.
Cada componente de reward/penalty está en su propio archivo para facilitar
el mantenimiento y la configuración.
"""

# Sistemas de rewards/penalties individuales
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

# Orquestador principal
from .reward_orchestrator import RewardOrchestrator

# Sistema de selección de leverage y timeframe
from .leverage_timeframe_selector import LeverageTimeframeSelector, LeverageTimeframeAction

# Gestor de bancarrota
from .bankruptcy_manager import BankruptcyManager

# Rewards/Penalties por SL/TP hits
from .sl_hit_penalty import SLHitPenalty
from .tp_hit_reward import TPHitReward

# Sistema de milestones de progreso
from .progress_milestone_reward import ProgressMilestoneReward

# Sistemas avanzados de rewards/penalties
from .volatility_reward import VolatilityReward
from .drawdown_penalty import DrawdownPenalty
from .execution_cost_penalty import ExecutionCostPenalty
from .mtf_alignment_reward import MTFAlignmentReward
from .time_efficiency_reward import TimeEfficiencyReward
from .overtrading_penalty import OvertradingPenalty
from .ttl_expiry_penalty import TTLExpiryPenalty
from .exploration_bonus import ExplorationBonus

__all__ = [
    # Sistemas individuales
    "TakeProfitReward",
    "StopLossPenalty", 
    "BankruptcyPenalty",
    "HoldingReward",
    "InactivityPenalty",
    "ROIReward",
    "RMultipleReward",
    "LeverageReward",
    "TimeframeReward",
    "DurationReward",
    "ProgressBonus",
    "BlockedTradePenalty",
    
    # Orquestador
    "RewardOrchestrator",
    
    # Selector de leverage/timeframe
    "LeverageTimeframeSelector",
    "LeverageTimeframeAction",
    
    # Gestor de bancarrota
    "BankruptcyManager",
    
    # Rewards/Penalties por SL/TP hits
    "SLHitPenalty",
    "TPHitReward",
    
    # Sistema de milestones de progreso
    "ProgressMilestoneReward",
    
    # Sistemas avanzados de rewards/penalties
    "VolatilityReward",
    "DrawdownPenalty",
    "ExecutionCostPenalty",
    "MTFAlignmentReward",
    "TimeEfficiencyReward",
    "OvertradingPenalty",
    "TTLExpiryPenalty",
    "ExplorationBonus"
]
