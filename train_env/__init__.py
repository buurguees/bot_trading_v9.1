# train_env/__init__.py
"""
Sistema de entrenamiento PPO modular y organizado.

Estructura:
- core/: Componentes principales (orchestrator, wrappers, managers)
- scripts/: Scripts de entrenamiento y monitoreo
- optimization/: Herramientas de optimización y tuning
- analysis/: Análisis y métricas de entrenamiento
- utilities/: Utilidades y herramientas auxiliares
- config/: Configuraciones específicas
- callbacks/: Callbacks de entrenamiento
- monitoring/: Monitoreo en tiempo real
- utils/: Utilidades generales
"""

# Importar componentes principales
from .core import (
    TrainingOrchestrator,
    TradingGymWrapper,
    make_vec_envs_chrono,
    make_vec_env,
    ModelManager,
    WorkerManager,
    get_optimal_worker_config
)

from .utilities import (
    RewardShaper,
    aggregate_top_k,
    StrategyCurriculum,
    StrategyLogger,
    StrategyPersistence,
    LearningRateResetCallback
)

from .monitoring import RealTimeMonitor, MetricPoint, HealthReport

# Importar callbacks
from .callbacks import (
    PeriodicCheckpoint,
    StrategyKeeper,
    StrategyConsultant,
    AntiBadStrategy,
    MainModelSaver,
    TrainingMetricsCallback
)

__all__ = [
    # Core
    'TrainingOrchestrator',
    'TradingGymWrapper',
    'make_vec_envs_chrono',
    'make_vec_env',
    'ModelManager',
    'WorkerManager',
    'get_optimal_worker_config',
    
    # Utilities
    'RewardShaper',
    'aggregate_top_k',
    'StrategyCurriculum',
    'StrategyLogger',
    'StrategyPersistence',
    'LearningRateResetCallback',
    
    # Monitoring
    'RealTimeMonitor',
    'MetricPoint',
    'HealthReport',
    
    # Callbacks
    'PeriodicCheckpoint',
    'StrategyKeeper',
    'StrategyConsultant',
    'AntiBadStrategy',
    'MainModelSaver',
    'TrainingMetricsCallback'
]
# Empaqueta utilidades de entrenamiento (wrapper Gym, dataset, reward shaper).
