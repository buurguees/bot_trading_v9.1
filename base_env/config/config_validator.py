# base_env/config/config_validator.py
"""
Sistema de validación de configuración con Pydantic.
Valida configuraciones de entrenamiento antes de ejecutar.
"""

from __future__ import annotations
from typing import Optional, Dict, Any, List, Union, Literal
from pydantic import BaseModel, validator, Field, model_validator
import yaml
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

class PPOConfig(BaseModel):
    """Configuración validada para PPO"""
    learning_rate: float = Field(3e-4, ge=1e-6, le=1.0, description="Learning rate entre 1e-6 y 1.0")
    n_steps: int = Field(2048, ge=64, le=8192, description="Steps por rollout entre 64 y 8192")
    batch_size: int = Field(64, ge=8, le=1024, description="Batch size entre 8 y 1024")
    n_epochs: int = Field(10, ge=1, le=50, description="Epochs por update entre 1 y 50")
    gamma: float = Field(0.99, ge=0.8, le=1.0, description="Discount factor entre 0.8 y 1.0")
    gae_lambda: float = Field(0.95, ge=0.8, le=1.0, description="GAE lambda entre 0.8 y 1.0")
    clip_range: float = Field(0.2, ge=0.05, le=0.5, description="Clip range entre 0.05 y 0.5")
    ent_coef: float = Field(0.0, ge=0.0, le=1.0, description="Entropy coefficient entre 0.0 y 1.0")
    vf_coef: float = Field(0.5, ge=0.0, le=2.0, description="Value function coefficient entre 0.0 y 2.0")
    max_grad_norm: float = Field(0.5, ge=0.1, le=5.0, description="Max gradient norm entre 0.1 y 5.0")
    total_timesteps: int = Field(1000000, ge=10000, le=100000000, description="Total timesteps entre 10K y 100M")
    anneal_lr: bool = Field(True, description="Si debe anealar learning rate")
    verbose: int = Field(1, ge=0, le=2, description="Verbosity level entre 0 y 2")
    
    @validator('batch_size')
    def validate_batch_size(cls, v, values):
        """Valida que batch_size sea compatible con n_steps"""
        if 'n_steps' in values and v > values['n_steps']:
            raise ValueError(f'batch_size ({v}) no puede ser mayor que n_steps ({values["n_steps"]})')
        return v
    
    @validator('learning_rate')
    def validate_learning_rate(cls, v):
        """Valida learning rate con advertencias"""
        if v > 0.01:
            logger.warning(f"Learning rate muy alto ({v}), puede causar inestabilidad")
        elif v < 1e-5:
            logger.warning(f"Learning rate muy bajo ({v}), puede causar convergencia lenta")
        return v

class EnvironmentConfig(BaseModel):
    """Configuración validada para entornos"""
    n_envs: int = Field(4, ge=1, le=32, description="Número de entornos entre 1 y 32")
    seed: int = Field(42, ge=0, le=2**31-1, description="Seed entre 0 y 2^31-1")
    data_root: str = Field("data", description="Directorio raíz de datos")
    symbol: str = Field(..., min_length=3, max_length=20, description="Símbolo de trading")
    market: Literal["spot", "futures"] = Field(..., description="Tipo de mercado")
    tfs: List[str] = Field(..., min_items=1, max_items=10, description="Lista de timeframes")
    base_tf: str = Field("1m", description="Timeframe base")
    warmup_bars: int = Field(5000, ge=100, le=50000, description="Barras de warmup entre 100 y 50K")
    
    @validator('tfs')
    def validate_tfs(cls, v):
        """Valida timeframes válidos"""
        valid_tfs = {"1m", "5m", "15m", "30m", "1h", "2h", "4h", "8h", "12h", "1d", "1w", "1M"}
        invalid_tfs = [tf for tf in v if tf not in valid_tfs]
        if invalid_tfs:
            raise ValueError(f"Timeframes inválidos: {invalid_tfs}. Válidos: {sorted(valid_tfs)}")
        return v
    
    @validator('base_tf')
    def validate_base_tf(cls, v, values):
        """Valida que base_tf esté en la lista de tfs"""
        if 'tfs' in values and v not in values['tfs']:
            logger.warning(f"base_tf ({v}) no está en la lista de tfs, será agregado automáticamente")
        return v

class LoggingConfig(BaseModel):
    """Configuración validada para logging"""
    tensorboard_log: Optional[str] = Field(None, description="Directorio para logs de TensorBoard")
    log_interval: int = Field(10, ge=1, le=1000, description="Intervalo de logging entre 1 y 1000")
    save_freq: int = Field(10000, ge=1000, le=1000000, description="Frecuencia de guardado entre 1K y 1M")
    eval_freq: int = Field(50000, ge=1000, le=1000000, description="Frecuencia de evaluación entre 1K y 1M")
    verbose: int = Field(1, ge=0, le=2, description="Verbosity level entre 0 y 2")
    print_interval: int = Field(1000, ge=100, le=10000, description="Intervalo de print entre 100 y 10K")
    
    @validator('tensorboard_log')
    def validate_tensorboard_log(cls, v):
        """Valida directorio de TensorBoard"""
        if v is not None:
            path = Path(v)
            if not path.parent.exists():
                logger.warning(f"Directorio padre de tensorboard_log no existe: {path.parent}")
        return v

class ModelConfig(BaseModel):
    """Configuración validada para modelos"""
    root: str = Field("models", description="Directorio raíz de modelos")
    overwrite: bool = Field(False, description="Si sobrescribir modelos existentes")
    checkpoint_freq: int = Field(100000, ge=10000, le=1000000, description="Frecuencia de checkpoints entre 10K y 1M")
    keep_checkpoints: int = Field(5, ge=1, le=20, description="Checkpoints a mantener entre 1 y 20")
    backup_freq: int = Field(50000, ge=10000, le=500000, description="Frecuencia de backup entre 10K y 500K")

class TrainingConfig(BaseModel):
    """Configuración completa validada para entrenamiento"""
    ppo: PPOConfig = Field(default_factory=PPOConfig)
    env: EnvironmentConfig = Field(..., description="Configuración de entorno")
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    models: ModelConfig = Field(default_factory=ModelConfig)
    
    @model_validator(mode='after')
    def validate_consistency(self):
        """Valida consistencia entre configuraciones"""
        # Validar que n_envs sea compatible con batch_size
        if self.env.n_envs > self.ppo.batch_size:
            logger.warning(f"n_envs ({self.env.n_envs}) > batch_size ({self.ppo.batch_size}), puede causar problemas")
        
        # Validar que save_freq sea múltiplo de eval_freq
        if self.logging.save_freq % self.logging.eval_freq != 0:
            logger.warning("save_freq debería ser múltiplo de eval_freq para mejor rendimiento")
        
        return self
    
    def get_effective_config(self) -> Dict[str, Any]:
        """Devuelve configuración procesada y validada"""
        return self.dict()
    
    def to_yaml(self, path: Union[str, Path]) -> None:
        """Guarda configuración validada a YAML"""
        config_dict = self.dict()
        with open(path, 'w', encoding='utf-8') as f:
            yaml.dump(config_dict, f, default_flow_style=False, allow_unicode=True)
    
    @classmethod
    def from_yaml(cls, path: Union[str, Path]) -> 'TrainingConfig':
        """Carga configuración desde YAML con validación"""
        with open(path, 'r', encoding='utf-8') as f:
            config_dict = yaml.safe_load(f)
        return cls(**config_dict)

class ConfigValidator:
    """Validador de configuración con utilidades adicionales"""
    
    @staticmethod
    def validate_file(path: Union[str, Path]) -> Tuple[bool, Optional[str], Optional[TrainingConfig]]:
        """
        Valida archivo de configuración
        
        Returns:
            Tuple[is_valid, error_message, config_object]
        """
        try:
            config = TrainingConfig.from_yaml(path)
            return True, None, config
        except Exception as e:
            return False, str(e), None
    
    @staticmethod
    def create_template_config(symbol: str, market: str, tfs: List[str]) -> TrainingConfig:
        """Crea configuración template para un símbolo específico"""
        env_config = EnvironmentConfig(
            symbol=symbol,
            market=market,
            tfs=tfs
        )
        
        return TrainingConfig(env=env_config)
    
    @staticmethod
    def get_recommended_config(system_resources: Dict[str, Any]) -> Dict[str, Any]:
        """Genera configuración recomendada basada en recursos del sistema"""
        cpu_count = system_resources.get('cpu_physical', 4)
        memory_gb = system_resources.get('memory_total_gb', 8)
        
        # Ajustar configuración según recursos
        if memory_gb < 8:
            n_envs = max(1, cpu_count // 2)
            batch_size = 32
            n_steps = 1024
        elif memory_gb < 16:
            n_envs = max(2, cpu_count - 1)
            batch_size = 64
            n_steps = 2048
        else:
            n_envs = cpu_count
            batch_size = 128
            n_steps = 4096
        
        return {
            'env': {'n_envs': n_envs},
            'ppo': {
                'batch_size': batch_size,
                'n_steps': n_steps,
                'learning_rate': 3e-4 if memory_gb >= 8 else 1e-4
            }
        }

def validate_training_config(config_path: Union[str, Path]) -> TrainingConfig:
    """
    Función de conveniencia para validar configuración de entrenamiento
    
    Args:
        config_path: Ruta al archivo de configuración
        
    Returns:
        TrainingConfig validada
        
    Raises:
        ValueError: Si la configuración es inválida
    """
    is_valid, error_msg, config = ConfigValidator.validate_file(config_path)
    
    if not is_valid:
        raise ValueError(f"Configuración inválida: {error_msg}")
    
    return config

def create_optimized_config(symbol: str, 
                          market: str, 
                          tfs: List[str],
                          system_resources: Optional[Dict[str, Any]] = None) -> TrainingConfig:
    """
    Crea configuración optimizada para un símbolo específico
    
    Args:
        symbol: Símbolo de trading
        market: Tipo de mercado
        tfs: Lista de timeframes
        system_resources: Recursos del sistema (opcional)
        
    Returns:
        TrainingConfig optimizada
    """
    config = ConfigValidator.create_template_config(symbol, market, tfs)
    
    if system_resources:
        recommendations = ConfigValidator.get_recommended_config(system_resources)
        # Aplicar recomendaciones
        for section, values in recommendations.items():
            if hasattr(config, section):
                for key, value in values.items():
                    setattr(getattr(config, section), key, value)
    
    return config
