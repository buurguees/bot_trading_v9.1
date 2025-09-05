# train_env/training_orchestrator.py
"""
Training Orchestrator - Coordina todos los aspectos del entrenamiento PPO.
Refactoriza train_ppo.py en un sistema modular y robusto.

Mejoras implementadas:
- Robustez: Manejo de excepciones granular, reintentos, validaciones estrictas
- Optimización: Monitoreo dinámico, fallbacks, mejor uso de recursos
- Lógica SB3: EvalCallback, VecMonitor, LinearSchedule, mejor integración
"""

from __future__ import annotations
import os
import sys
import yaml
import numpy as np
import multiprocessing as mp
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple, Union
import logging
from dataclasses import dataclass
from logging.handlers import RotatingFileHandler
import time
import traceback

# Añadir el directorio raíz al path
sys.path.insert(0, str(Path(__file__).parent.parent))

from stable_baselines3 import PPO
from stable_baselines3.common.logger import configure
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv, VecMonitor
from stable_baselines3.common.callbacks import EvalCallback
try:
    from stable_baselines3.common.schedules import LinearSchedule
except ImportError:
    # Fallback para versiones más antiguas de SB3
    from stable_baselines3.common.utils import LinearSchedule

from .vec_factory import make_vec_envs_chrono, make_vec_env
from ..callbacks import (
    PeriodicCheckpoint, StrategyKeeper, StrategyConsultant, 
    AntiBadStrategy, MainModelSaver, TrainingMetricsCallback
)
from ..utilities.learning_rate_reset_callback import LearningRateResetCallback
from ..utilities.strategy_aggregator import aggregate_top_k
from .model_manager import ModelManager
from .worker_manager import WorkerManager, get_optimal_worker_config, print_worker_recommendation
from base_env.config.config_validator import TrainingConfig, validate_training_config
from base_env.config.config_loader import config_loader
from ..utilities.repair_models import repair_models

logger = logging.getLogger(__name__)

@dataclass
class TrainingMetrics:
    """Métricas de entrenamiento con smoothing y tracking avanzado"""
    total_timesteps: int
    current_timestep: int = 0
    episodes_completed: int = 0
    best_reward: float = float('-inf')
    current_reward: float = 0.0
    learning_rate: float = 0.0
    policy_loss: float = 0.0
    value_loss: float = 0.0
    entropy_loss: float = 0.0
    smoothed_reward: float = 0.0  # Promedio rodante
    reward_history: List[float] = None  # Historial para smoothing
    window_size: int = 100  # Ventana para smoothing
    
    def __post_init__(self):
        if self.reward_history is None:
            self.reward_history = []
    
    def update_smoothed_reward(self, new_reward: float, alpha: float = 0.1):
        """Actualiza reward suavizado con promedio exponencial"""
        if self.smoothed_reward == 0.0:
            self.smoothed_reward = new_reward
        else:
            self.smoothed_reward = alpha * new_reward + (1 - alpha) * self.smoothed_reward
    
    def update_reward_history(self, new_reward: float):
        """Actualiza historial de rewards para smoothing"""
        self.reward_history.append(new_reward)
        if len(self.reward_history) > self.window_size:
            self.reward_history.pop(0)
        
        # Calcular promedio rodante
        if len(self.reward_history) > 0:
            self.smoothed_reward = np.mean(self.reward_history)
    
    def get_reward_trend(self) -> str:
        """Retorna tendencia del reward basada en los últimos valores"""
        if len(self.reward_history) < 10:
            return "insufficient_data"
        
        recent = self.reward_history[-10:]
        if len(recent) < 2:
            return "insufficient_data"
        
        trend = np.polyfit(range(len(recent)), recent, 1)[0]
        if trend > 0.01:
            return "improving"
        elif trend < -0.01:
            return "declining"
        else:
            return "stable"

class TrainingOrchestrator:
    """Orquestador principal del entrenamiento PPO con mejoras de robustez y optimización"""
    
    def __init__(self, config_path: str = "config/train.yaml"):
        """
        Inicializa el orquestador de entrenamiento
        
        Args:
            config_path: Ruta al archivo de configuración
        """
        self.config_path = config_path
        self.config: Optional[TrainingConfig] = None
        self.model_manager: Optional[ModelManager] = None
        self.worker_manager: Optional[WorkerManager] = None
        self.environment: Optional[Union[SubprocVecEnv, DummyVecEnv]] = None
        self.eval_environment: Optional[Union[SubprocVecEnv, DummyVecEnv]] = None  # Nueva: para evaluación
        self.model: Optional[PPO] = None
        self.callbacks: List[Any] = []
        self.metrics = TrainingMetrics(total_timesteps=0)
        self.training_start_time: Optional[float] = None
        self.last_checkpoint_time: Optional[float] = None
        
        # Configurar multiprocessing cross-platform
        self._setup_multiprocessing()
        
    def _setup_multiprocessing(self):
        """Configura multiprocessing cross-platform con fallbacks"""
        try:
            available_methods = mp.get_all_start_methods()
            if 'spawn' in available_methods:
                mp.set_start_method('spawn', force=True)
                logger.info("✅ Multiprocessing configurado con método 'spawn'")
            else:
                logger.warning(f"⚠️ Método 'spawn' no disponible. Métodos disponibles: {available_methods}")
                logger.warning("   Usando método por defecto (puede causar problemas en Windows)")
        except RuntimeError as e:
            logger.warning(f"⚠️ Error configurando multiprocessing: {e}")
            logger.warning("   Continuando con configuración por defecto")
    
    def setup_training(self, skip_repair: bool = False) -> bool:
        """
        Configura todos los componentes de entrenamiento con reintentos y validaciones robustas
        
        Args:
            skip_repair: Si saltar la reparación de archivos
            
        Returns:
            True si la configuración fue exitosa
        """
        max_retries = 3
        retry_delay = 2.0
        
        for attempt in range(max_retries):
            try:
                logger.info(f"🔧 Configurando entrenamiento (intento {attempt + 1}/{max_retries})")
                
                # 1. Cargar y validar configuración
                if not self._load_and_validate_config():
                    logger.error("❌ Falló la carga de configuración")
                    continue
                
                # 2. Configurar worker manager
                self._setup_worker_manager()
                
                # 3. Configurar entorno de entrenamiento
                if not self._setup_environment(eval_mode=False):
                    logger.error("❌ Falló la configuración del entorno de entrenamiento")
                    continue
                
                # 4. Configurar entorno de evaluación
                if not self._setup_environment(eval_mode=True):
                    logger.warning("⚠️ Falló la configuración del entorno de evaluación, continuando sin evaluación")
                
                # 5. Configurar modelo
                if not self._setup_model():
                    logger.error("❌ Falló la configuración del modelo")
                    continue
                
                # 6. Configurar callbacks
                self._setup_callbacks()
                
                # 7. Configurar logging
                self._setup_logging()
                
                # 8. Reparar archivos si es necesario
                if not skip_repair:
                    if not self._repair_model_files():
                        logger.warning("⚠️ Falló la reparación de archivos, continuando...")
                
                logger.info("✅ Configuración de entrenamiento completada exitosamente")
                return True
                
            except Exception as e:
                logger.error(f"❌ Error configurando entrenamiento (intento {attempt + 1}): {e}")
                logger.error(f"   Traceback: {traceback.format_exc()}")
                
                if attempt < max_retries - 1:
                    logger.info(f"⏳ Reintentando en {retry_delay} segundos...")
                    time.sleep(retry_delay)
                    retry_delay *= 1.5  # Backoff exponencial
                else:
                    logger.error("❌ Máximo número de reintentos alcanzado")
                    return False
        
        return False
    
    def _load_and_validate_config(self) -> bool:
        """Carga y valida la configuración con validaciones estrictas"""
        try:
            # Verificar que el archivo de configuración existe
            if not Path(self.config_path).exists():
                logger.error(f"❌ Archivo de configuración no encontrado: {self.config_path}")
                return False
            
            # Cargar configuración YAML cruda
            with open(self.config_path, "r", encoding="utf-8") as f:
                raw_config = yaml.safe_load(f)
            
            if not raw_config:
                logger.error("❌ Archivo de configuración vacío o inválido")
                return False
            
            # Obtener símbolo desde symbols.yaml
            syms = config_loader.load_symbols()
            if not syms:
                logger.error("❌ No se encontraron símbolos en symbols.yaml")
                return False
            
            sym0 = next((s for s in syms if s.market == "futures"), None)
            if sym0 is None:
                sym0 = next((s for s in syms if s.market == "spot"), syms[0])
            
            if not sym0:
                logger.error("❌ No se pudo seleccionar un símbolo válido")
                return False
            
            # Crear configuración de entorno
            env_config = {
                "symbol": sym0.symbol,
                "mode": f"train_{sym0.market}",
                "leverage": (sym0.leverage.__dict__ if sym0.leverage else None),
                "tfs": raw_config.get("tfs", ["1m", "5m", "15m", "1h", "4h"]),
                "n_envs": raw_config.get("env", {}).get("n_envs", 4),
                "seed": raw_config.get("seed", 42),
                "data_root": raw_config.get("data_root", "data"),
                "market": sym0.market,
                "base_tf": raw_config.get("base_tf", "1m"),
                "warmup_bars": raw_config.get("warmup_bars", 5000)
            }
            
            # Crear configuración validada
            self.config = TrainingConfig(
                ppo=raw_config.get("ppo", {}),
                env=env_config,
                logging=raw_config.get("logging", {}),
                models=raw_config.get("models", {})
            )
            
            # Nueva: Detectar y configurar GPU
            self._detect_and_configure_gpu()
            
            # Validar directorios necesarios
            self._validate_directories()
            
            logger.info(f"✅ Configuración validada para {sym0.symbol} ({sym0.market})")
            return True
            
        except yaml.YAMLError as e:
            logger.error(f"❌ Error parseando YAML: {e}")
            return False
        except Exception as e:
            logger.error(f"❌ Error cargando configuración: {e}")
            logger.error(f"   Traceback: {traceback.format_exc()}")
            return False
    
    def _detect_and_configure_gpu(self):
        """Detecta GPU y configura device automáticamente"""
        try:
            import torch
            if torch.cuda.is_available():
                gpu_count = torch.cuda.device_count()
                gpu_name = torch.cuda.get_device_name(0)
                gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                
                logger.info(f"🟢 GPU disponible: {gpu_name} ({gpu_memory:.1f}GB)")
                logger.info(f"   Dispositivos CUDA: {gpu_count}")
                
                # Configurar device en la configuración PPO
                if hasattr(self.config.ppo, 'device'):
                    self.config.ppo.device = 'cuda'
                    logger.info("✅ Device configurado como 'cuda'")
                else:
                    logger.warning("⚠️ No se pudo configurar device en PPO config")
            else:
                logger.info("🟡 GPU no disponible, usando CPU")
                if hasattr(self.config.ppo, 'device'):
                    self.config.ppo.device = 'cpu'
        except ImportError:
            logger.warning("⚠️ PyTorch no disponible, no se puede detectar GPU")
        except Exception as e:
            logger.warning(f"⚠️ Error detectando GPU: {e}")
    
    def _validate_directories(self):
        """Valida que los directorios necesarios existan"""
        required_dirs = [
            self.config.env.data_root,
            self.config.models.root,
            self.config.logging.tensorboard_log or "logs"
        ]
        
        for dir_path in required_dirs:
            if dir_path and not Path(dir_path).exists():
                try:
                    Path(dir_path).mkdir(parents=True, exist_ok=True)
                    logger.info(f"📁 Directorio creado: {dir_path}")
                except Exception as e:
                    logger.warning(f"⚠️ No se pudo crear directorio {dir_path}: {e}")
    
    def _setup_worker_manager(self):
        """Configura el worker manager"""
        self.worker_manager = WorkerManager()
        
        # Obtener recomendación de workers
        recommendation = get_optimal_worker_config(
            target_workers=self.config.env.n_envs,
            memory_per_worker_mb=1024
        )
        
        # Actualizar configuración con workers óptimos
        self.config.env.n_envs = recommendation.optimal_workers
        self.config.ppo.batch_size = recommendation.recommended_batch_size
        
        # Imprimir recomendación
        print_worker_recommendation(recommendation)
        
        logger.info(f"Worker manager configurado: {recommendation.optimal_workers} workers")
    
    def _setup_environment(self, eval_mode: bool = False) -> bool:
        """Configura el entorno vectorizado con soporte para evaluación y fallbacks"""
        try:
            # Preparar configuración para make_vec_envs_chrono
            data_cfg = {
                "data_root": self.config.env.data_root,
                "symbol": self.config.env.symbol,
                "market": self.config.env.market,
                "tfs": self.config.env.tfs,
                "base_tf": self.config.env.base_tf,
                "warmup_bars": self.config.env.warmup_bars
            }
            
            # Para evaluación, usar menos workers y datos diferentes si es posible
            n_envs = 1 if eval_mode else self.config.env.n_envs
            seed_offset = 1000 if eval_mode else 0  # Diferente semilla para eval
            
            env_cfg = {
                "n_envs": n_envs,
                "seed": self.config.env.seed + seed_offset,
                "mode": f"train_{self.config.env.market}",
                "leverage": getattr(self.config.env, 'leverage', None)
            }
            
            logging_cfg = self.config.logging.dict()
            models_cfg = self.config.models.dict()
            symbol_cfg = {
                "symbol": self.config.env.symbol,
                "mode": f"train_{self.config.env.market}",
                "leverage": getattr(self.config.env, 'leverage', None)
            }
            
            # Intentar crear entorno con SubprocVecEnv primero
            try:
                env = make_vec_envs_chrono(
                    n_envs=n_envs,
                    seed=self.config.env.seed + seed_offset,
                    data_cfg=data_cfg,
                    env_cfg=env_cfg,
                    logging_cfg=logging_cfg,
                    models_cfg=models_cfg,
                    symbol_cfg=symbol_cfg
                )
                
                # Envolver con VecMonitor para métricas automáticas
                env = VecMonitor(env)
                
                if eval_mode:
                    self.eval_environment = env
                    logger.info(f"✅ Entorno de evaluación creado: {n_envs} workers")
                else:
                    self.environment = env
                    logger.info(f"✅ Entorno de entrenamiento creado: {n_envs} workers")
                
                return True
                
            except Exception as subproc_error:
                logger.warning(f"⚠️ SubprocVecEnv falló: {subproc_error}")
                logger.warning("   Intentando con DummyVecEnv como fallback...")
                
                # Fallback a DummyVecEnv
                try:
                    # Usar la nueva API unificada con DummyVecEnv
                    env = make_vec_env(
                        symbol=self.config.env.symbol,
                        tfs=self.config.env.tfs,
                        reward_yaml="config/reward.yaml",  # Default
                        data_cfg=data_cfg,
                        env_cfg=env_cfg,
                        symbol_cfg=symbol_cfg,
                        models_cfg=models_cfg,
                        logging_cfg=logging_cfg,
                        models_root=self.config.models.root,
                        n_envs=n_envs,
                        seed=self.config.env.seed + seed_offset,
                        chrono=True,
                        use_subproc=False  # Forzar DummyVecEnv
                    )
                    
                    # Envolver con VecMonitor
                    env = VecMonitor(env)
                    
                    if eval_mode:
                        self.eval_environment = env
                        logger.info(f"✅ Entorno de evaluación (DummyVecEnv) creado: {n_envs} workers")
                    else:
                        self.environment = env
                        logger.info(f"✅ Entorno de entrenamiento (DummyVecEnv) creado: {n_envs} workers")
                    
                    return True
                    
                except Exception as dummy_error:
                    logger.error(f"❌ DummyVecEnv también falló: {dummy_error}")
                    return False
            
        except Exception as e:
            logger.error(f"❌ Error creando entorno: {e}")
            logger.error(f"   Traceback: {traceback.format_exc()}")
            return False
    
    def _setup_model(self) -> bool:
        """Configura el modelo PPO"""
        try:
            # Crear model manager
            self.model_manager = ModelManager(
                symbol=self.config.env.symbol,
                models_root=self.config.models.root,
                overwrite=self.config.models.overwrite
            )
            
            # Mostrar resumen del modelo
            self.model_manager.print_summary()
            
            # Cargar o crear modelo
            ppo_kwargs = self.config.ppo.dict()
            self.model = self.model_manager.load_model(
                env=self.environment,
                **ppo_kwargs
            )
            
            if self.model is None:
                logger.error("❌ No se pudo cargar o crear el modelo")
                return False
            
            # Actualizar métricas
            self.metrics.total_timesteps = self.config.ppo.total_timesteps
            self.metrics.learning_rate = self.config.ppo.learning_rate
            
            logger.info(f"✅ Modelo PPO configurado: {self.config.env.symbol}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error configurando modelo: {e}")
            return False
    
    def _setup_callbacks(self):
        """Configura los callbacks de entrenamiento con EvalCallback y LinearSchedule"""
        try:
            # Obtener rutas de archivos
            file_paths = self.model_manager.get_file_paths()
            
            # Callbacks básicos
            self.callbacks = [
                PeriodicCheckpoint(
                    save_freq=self.config.logging.save_freq,
                    name_prefix=f"{self.config.env.symbol}_checkpoint",
                    save_path=file_paths['checkpoints_dir']
                ),
                StrategyKeeper(
                    strategy_file=file_paths['strategies_provisional'],
                    save_freq=self.config.logging.save_freq
                ),
                StrategyConsultant(
                    strategy_file=file_paths['strategies_best'],
                    consult_freq=self.config.logging.eval_freq
                ),
                AntiBadStrategy(
                    bad_strategy_file=file_paths['bad_strategies'],
                    check_freq=self.config.logging.eval_freq
                ),
                MainModelSaver(
                    model_path=file_paths['model_path'],
                    save_freq=self.config.logging.save_freq
                ),
                TrainingMetricsCallback(
                    metrics_file=file_paths['progress_path'],
                    log_freq=self.config.logging.log_interval
                )
            ]
            
            # Nueva: EvalCallback para evaluación periódica
            if self.eval_environment is not None:
                eval_callback = EvalCallback(
                    self.eval_environment,
                    best_model_save_path=str(Path(file_paths['checkpoints_dir']) / "best_model"),
                    log_path=str(Path(self.config.logging.tensorboard_log) / "eval_logs"),
                    eval_freq=self.config.logging.eval_freq,
                    deterministic=True,
                    render=False,
                    verbose=1
                )
                self.callbacks.append(eval_callback)
                logger.info(f"✅ EvalCallback configurado (freq: {self.config.logging.eval_freq})")
            else:
                logger.warning("⚠️ Entorno de evaluación no disponible, saltando EvalCallback")
            
            # Callback de reset de learning rate si está habilitado
            lr_reset_cfg = getattr(self.config.ppo, 'lr_reset', {"enabled": False})
            if lr_reset_cfg.get('enabled', False):
                self.callbacks.append(LearningRateResetCallback(
                    env=self.environment,
                    reset_threshold=lr_reset_cfg.get('threshold_runs', 30),
                    verbose=1
                ))
                logger.info(f"✅ Learning rate reset habilitado: {lr_reset_cfg.get('threshold_runs', 30)} runs")
            
            logger.info(f"✅ {len(self.callbacks)} callbacks configurados")
            
        except Exception as e:
            logger.error(f"❌ Error configurando callbacks: {e}")
            logger.error(f"   Traceback: {traceback.format_exc()}")
    
    def _setup_learning_rate_schedule(self):
        """Configura LinearSchedule para learning rate si está habilitado"""
        try:
            lr_schedule_cfg = getattr(self.config.ppo, 'lr_schedule', {"enabled": False})
            if lr_schedule_cfg.get('enabled', False):
                initial_lr = self.config.ppo.learning_rate
                final_lr = lr_schedule_cfg.get('final_lr', initial_lr * 0.01)
                total_timesteps = self.config.ppo.total_timesteps
                
                # Crear LinearSchedule
                lr_schedule = LinearSchedule(
                    total_timesteps=total_timesteps,
                    initial_p=1.0,
                    final_p=final_lr / initial_lr
                )
                
                # Aplicar al modelo
                if self.model is not None:
                    self.model.lr_schedule = lr_schedule
                    logger.info(f"✅ Learning rate schedule configurado: {initial_lr} → {final_lr}")
                else:
                    logger.warning("⚠️ Modelo no disponible para configurar lr_schedule")
                    
        except Exception as e:
            logger.warning(f"⚠️ Error configurando lr_schedule: {e}")
    
    def _setup_logging(self):
        """Configura el sistema de logging con rotación y mejor configuración"""
        try:
            # Configurar rotación de logs
            log_dir = Path(self.config.logging.tensorboard_log or "logs")
            log_dir.mkdir(parents=True, exist_ok=True)
            
            # Configurar handler de rotación
            log_file = log_dir / f"{self.config.env.symbol}_training.log"
            file_handler = RotatingFileHandler(
                log_file,
                maxBytes=10*1024*1024,  # 10MB
                backupCount=5
            )
            file_handler.setLevel(logging.INFO)
            file_handler.setFormatter(logging.Formatter(
                '%(asctime)s | %(levelname)s | %(name)s | %(message)s'
            ))
            
            # Añadir handler al logger principal
            root_logger = logging.getLogger()
            root_logger.addHandler(file_handler)
            
            # Configurar logger de PPO
            if self.config.logging.tensorboard_log:
                configure(
                    folder=self.config.logging.tensorboard_log,
                    format_strings=['stdout', 'log', 'tensorboard']
                )
            else:
                configure(folder=None, format_strings=['stdout', 'log'])
            
            # Configurar variables de entorno para estabilidad
            os.environ['PYTHONHASHSEED'] = str(self.config.env.seed)
            os.environ['OMP_NUM_THREADS'] = '1'
            
            # Configurar nivel de logging basado en verbosidad
            verbosity = getattr(self.config.logging, 'verbosity', 1)
            if verbosity >= 2:
                logging.getLogger().setLevel(logging.DEBUG)
            elif verbosity >= 1:
                logging.getLogger().setLevel(logging.INFO)
            else:
                logging.getLogger().setLevel(logging.WARNING)
            
            logger.info("✅ Sistema de logging configurado con rotación")
            logger.info(f"   Log file: {log_file}")
            logger.info(f"   Verbosity: {verbosity}")
            
        except Exception as e:
            logger.error(f"❌ Error configurando logging: {e}")
            logger.error(f"   Traceback: {traceback.format_exc()}")
    
    def _repair_model_files(self) -> bool:
        """Repara archivos de modelos si es necesario"""
        try:
            logger.info(f"🧹 Reparando archivos de {self.config.env.symbol}...")
            repair_success = repair_models(self.config.env.symbol, verbose=True)
            
            if not repair_success:
                logger.warning(f"⚠️ No se pudo reparar archivos de {self.config.env.symbol}")
                logger.warning("   Continuando con entrenamiento...")
                return True  # No es crítico, continuar
            
            logger.info(f"✅ Archivos de {self.config.env.symbol} reparados exitosamente")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error reparando archivos: {e}")
            return False
    
    def run_training(self) -> bool:
        """
        Ejecuta el entrenamiento con manejo robusto de errores y graceful shutdown
        
        Returns:
            True si el entrenamiento fue exitoso
        """
        if not all([self.config, self.model, self.environment, self.callbacks]):
            logger.error("❌ Entrenamiento no configurado correctamente")
            return False
        
        try:
            # Configurar learning rate schedule si está habilitado
            self._setup_learning_rate_schedule()
            
            # Marcar inicio del entrenamiento
            self.training_start_time = time.time()
            self.last_checkpoint_time = self.training_start_time
            
            logger.info("🚀 INICIANDO ENTRENAMIENTO PPO")
            logger.info(f"📊 Configuración: {self.config.ppo.n_steps} steps/env, "
                       f"{self.config.ppo.batch_size} batch, {self.config.ppo.learning_rate} lr")
            logger.info(f"🔧 Workers: {self.config.env.n_envs}")
            logger.info(f"📝 Timesteps totales: {self.config.ppo.total_timesteps:,}")
            logger.info(f"🎯 Device: {getattr(self.config.ppo, 'device', 'auto')}")
            
            # Ejecutar entrenamiento
            self.model.learn(
                total_timesteps=self.config.ppo.total_timesteps,
                callback=self.callbacks,
                reset_num_timesteps=False  # Para continuation
            )
            
            # Guardado final
            self._finalize_training()
            
            # Calcular tiempo total
            total_time = time.time() - self.training_start_time
            logger.info(f"✅ Entrenamiento completado exitosamente en {total_time/3600:.2f} horas")
            return True
            
        except KeyboardInterrupt:
            logger.info("⏹️ Entrenamiento interrumpido por usuario")
            self._finalize_training()
            return True
            
        except (EOFError, BrokenPipeError) as e:
            logger.error(f"❌ ERROR DE MULTIPROCESSING: {type(e).__name__}: {e}")
            logger.error("🔧 Esto puede ser causado por:")
            logger.error("   - Demasiados workers para los recursos disponibles")
            logger.error("   - Problemas de comunicación entre procesos")
            logger.error("   - Memoria insuficiente")
            self._graceful_shutdown()
            return False
            
        except Exception as e:
            logger.error(f"❌ Error durante entrenamiento: {e}")
            logger.error(f"   Traceback: {traceback.format_exc()}")
            self._graceful_shutdown()
            return False
        
        finally:
            # Siempre cerrar entornos
            self._close_environments()
    
    def _graceful_shutdown(self):
        """Cierra recursos de manera segura en caso de error"""
        try:
            logger.info("🔄 Iniciando shutdown graceful...")
            
            # Guardar modelo en estado actual
            if self.model is not None:
                try:
                    file_paths = self.model_manager.get_file_paths()
                    self.model.save(file_paths['model_path'])
                    logger.info(f"💾 Modelo guardado en estado de error: {file_paths['model_path']}")
                except Exception as save_error:
                    logger.error(f"❌ No se pudo guardar el modelo: {save_error}")
            
            # Cerrar entornos
            self._close_environments()
            
            logger.info("✅ Shutdown graceful completado")
            
        except Exception as e:
            logger.error(f"❌ Error durante shutdown graceful: {e}")
    
    def _close_environments(self):
        """Cierra todos los entornos de manera segura"""
        try:
            if self.environment is not None:
                self.environment.close()
                self.environment = None
                logger.info("✅ Entorno de entrenamiento cerrado")
            
            if self.eval_environment is not None:
                self.eval_environment.close()
                self.eval_environment = None
                logger.info("✅ Entorno de evaluación cerrado")
                
        except Exception as e:
            logger.warning(f"⚠️ Error cerrando entornos: {e}")
    
    def _finalize_training(self):
        """Finaliza el entrenamiento guardando modelos y estrategias"""
        try:
            # Guardar modelo principal
            file_paths = self.model_manager.get_file_paths()
            self.model.save(file_paths['model_path'])
            logger.info(f"💾 Modelo guardado: {file_paths['model_path']}")
            
            # Consolidar estrategias
            aggregate_top_k(
                file_paths['strategies_provisional'],
                file_paths['strategies_best'],
                1000  # top_k
            )
            logger.info(f"📈 Estrategias consolidadas: {file_paths['strategies_best']}")
            
        except Exception as e:
            logger.error(f"❌ Error finalizando entrenamiento: {e}")
    
    def get_training_status(self) -> Dict[str, Any]:
        """Retorna el estado actual del entrenamiento con métricas avanzadas"""
        # Calcular tiempo transcurrido
        elapsed_time = None
        if self.training_start_time:
            elapsed_time = time.time() - self.training_start_time
        
        # Calcular progreso
        progress = 0.0
        if self.metrics.total_timesteps > 0:
            progress = (self.metrics.current_timestep / self.metrics.total_timesteps) * 100
        
        return {
            "configured": all([self.config, self.model, self.environment]),
            "training_active": self.training_start_time is not None,
            "elapsed_time_seconds": elapsed_time,
            "elapsed_time_hours": elapsed_time / 3600 if elapsed_time else None,
            "progress_percentage": progress,
            "metrics": {
                "total_timesteps": self.metrics.total_timesteps,
                "current_timestep": self.metrics.current_timestep,
                "episodes_completed": self.metrics.episodes_completed,
                "best_reward": self.metrics.best_reward,
                "current_reward": self.metrics.current_reward,
                "smoothed_reward": self.metrics.smoothed_reward,
                "reward_trend": self.metrics.get_reward_trend(),
                "learning_rate": self.metrics.learning_rate,
                "policy_loss": self.metrics.policy_loss,
                "value_loss": self.metrics.value_loss,
                "entropy_loss": self.metrics.entropy_loss
            },
            "environment_info": {
                "training_envs": self.config.env.n_envs if self.config else None,
                "eval_envs": 1 if self.eval_environment else 0,
                "env_type": type(self.environment).__name__ if self.environment else None
            },
            "config": self.config.dict() if self.config else None,
            "system_info": self.worker_manager.get_system_info() if self.worker_manager else None
        }

def main():
    """Función principal para ejecutar el orquestador con mejor configuración"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Training Orchestrator para PPO - Versión Mejorada")
    parser.add_argument("--config", type=str, default="config/train.yaml",
                       help="Ruta al archivo de configuración")
    parser.add_argument("--skip-repair", action="store_true",
                       help="Saltar reparación de archivos de modelos")
    parser.add_argument("--verbose", "-v", action="count", default=0,
                       help="Nivel de verbosidad (-v, -vv, -vvv)")
    parser.add_argument("--dry-run", action="store_true",
                       help="Solo validar configuración sin entrenar")
    
    args = parser.parse_args()
    
    # Configurar logging basado en verbosidad
    log_levels = [logging.WARNING, logging.INFO, logging.DEBUG]
    log_level = log_levels[min(args.verbose, len(log_levels) - 1)]
    
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s | %(levelname)s | %(name)s | %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    logger.info("🚀 Iniciando Training Orchestrator - Versión Mejorada")
    logger.info(f"📁 Configuración: {args.config}")
    logger.info(f"🔧 Verbosidad: {log_level}")
    logger.info(f"🧪 Dry run: {args.dry_run}")
    
    try:
        # Crear orquestador
        orchestrator = TrainingOrchestrator(args.config)
        
        # Configurar entrenamiento
        if not orchestrator.setup_training(skip_repair=args.skip_repair):
            logger.error("❌ Falló la configuración del entrenamiento")
            sys.exit(1)
        
        # Mostrar estado inicial
        status = orchestrator.get_training_status()
        logger.info(f"📊 Estado inicial: {status['environment_info']}")
        
        if args.dry_run:
            logger.info("✅ Dry run completado - configuración válida")
            return
        
        # Ejecutar entrenamiento
        if not orchestrator.run_training():
            logger.error("❌ Falló el entrenamiento")
            sys.exit(1)
        
        # Mostrar estado final
        final_status = orchestrator.get_training_status()
        logger.info(f"📈 Estado final: {final_status['metrics']}")
        logger.info("🎉 Entrenamiento completado exitosamente")
        
    except KeyboardInterrupt:
        logger.info("⏹️ Interrumpido por usuario")
        sys.exit(0)
    except Exception as e:
        logger.error(f"❌ Error fatal: {e}")
        logger.error(f"   Traceback: {traceback.format_exc()}")
        sys.exit(1)

if __name__ == "__main__":
    main()
