# train_env/core/__init__.py
"""
Componentes principales del sistema de entrenamiento.
"""

from .training_orchestrator import TrainingOrchestrator
from .gym_wrapper import TradingGymWrapper
from .vec_factory import make_vec_envs_chrono
from .vec_factory import make_vec_env
from .model_manager import ModelManager
from .worker_manager import WorkerManager, get_optimal_worker_config

__all__ = [
    'TrainingOrchestrator',
    'TradingGymWrapper', 
    'make_vec_envs_chrono',
    'make_vec_env',
    'ModelManager',
    'WorkerManager',
    'get_optimal_worker_config'
]
