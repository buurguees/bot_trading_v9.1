# base_env/actions/reward_decomposition.py
"""
Sistema de reward decomposition basado en R-DDQN para 50M steps.

Implementa una red de rewards entrenada con demos expertos para generar
señales dinámicos que mejoran la convergencia en entrenamientos largos.

Basado en: "R-DDQN: Optimizing Algorithmic Trading Strategies Using a Reward Decomposition Double DQN" (2024)
"""

from __future__ import annotations
import time
import logging
import numpy as np
from typing import Dict, Any, List, Tuple, Optional, Union
from dataclasses import dataclass
from collections import deque
import json
from pathlib import Path

# Importar utilidades de optimización
from .rewards_utils import (
    get_config_cache, get_profiler, get_batch_processor,
    profile_reward_calculation, get_vectorized_calculator, get_reward_validator
)
from .rewards_optimizer import get_global_optimizer

logger = logging.getLogger(__name__)

@dataclass
class ExpertDemo:
    """Demo experto para entrenar la red de rewards."""
    obs: Dict[str, Any]
    action: int
    reward: float
    next_obs: Dict[str, Any]
    done: bool
    timestamp: float

@dataclass
class RewardComponent:
    """Componente de reward con peso dinámico."""
    name: str
    base_weight: float
    current_weight: float
    importance: float
    last_update: float

class RewardDecompositionNetwork:
    """
    Red de reward decomposition basada en R-DDQN.
    
    Esta red aprende a predecir rewards agregados basándose en demos expertos,
    reduciendo la necesidad de tuning manual y mejorando la escalabilidad.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Inicializa la red de reward decomposition.
        
        Args:
            config: Configuración del sistema
        """
        self.config = config
        self.cache = get_config_cache()
        self.profiler = get_profiler()
        self.batch_processor = get_batch_processor()
        self.optimizer = get_global_optimizer()
        
        # Cargar configuración
        self._load_config()
        
        # Componentes de reward con pesos dinámicos
        self.reward_components = self._initialize_components()
        
        # Historial de demos expertos
        self.expert_demos = deque(maxlen=self.max_demos)
        self.demo_weights = deque(maxlen=self.max_demos)
        
        # Estado de la red
        self.is_trained = False
        self.training_loss = 0.0
        self.prediction_accuracy = 0.0
        
        # Estadísticas
        self.total_predictions = 0
        self.correct_predictions = 0
        self.avg_prediction_error = 0.0
        
        logger.debug(f"RewardDecompositionNetwork inicializado - enabled: {self.enabled}")

    def _load_config(self) -> None:
        """Carga configuración del sistema."""
        decomp_config = self.config.get("reward_decomposition", {})
        self.enabled = decomp_config.get("enabled", False)
        self.learning_rate = decomp_config.get("learning_rate", 0.001)
        self.batch_size = decomp_config.get("batch_size", 32)
        self.max_demos = decomp_config.get("max_demos", 10000)
        self.update_frequency = decomp_config.get("update_frequency", 1000)
        self.prediction_threshold = decomp_config.get("prediction_threshold", 0.1)
        self.weight_decay = decomp_config.get("weight_decay", 0.99)
        self.expert_weight = decomp_config.get("expert_weight", 0.7)
        self.model_weight = decomp_config.get("model_weight", 0.3)

    def _initialize_components(self) -> Dict[str, RewardComponent]:
        """Inicializa componentes de reward con pesos dinámicos."""
        components = {}
        
        # Componentes principales basados en la literatura
        component_configs = [
            ("profit_loss", 0.3, 0.3),
            ("risk_adjusted", 0.2, 0.2),
            ("time_efficiency", 0.15, 0.15),
            ("volatility_adaptation", 0.1, 0.1),
            ("drawdown_control", 0.1, 0.1),
            ("exploration_bonus", 0.05, 0.05),
            ("consistency", 0.1, 0.1)
        ]
        
        for name, base_weight, importance in component_configs:
            components[name] = RewardComponent(
                name=name,
                base_weight=base_weight,
                current_weight=base_weight,
                importance=importance,
                last_update=time.time()
            )
        
        return components

    def add_expert_demo(self, demo: ExpertDemo) -> None:
        """
        Añade un demo experto para entrenar la red.
        
        Args:
            demo: Demo experto con observación, acción, reward, etc.
        """
        if not self.enabled:
            return
        
        self.expert_demos.append(demo)
        
        # Calcular peso del demo basado en calidad
        demo_weight = self._calculate_demo_weight(demo)
        self.demo_weights.append(demo_weight)
        
        # Entrenar la red si hay suficientes demos
        if len(self.expert_demos) >= self.batch_size:
            self._train_network()

    def _calculate_demo_weight(self, demo: ExpertDemo) -> float:
        """
        Calcula peso del demo basado en calidad del reward.
        
        Basado en la literatura: demos con rewards más altos y consistentes
        tienen mayor peso en el entrenamiento.
        """
        # Peso base basado en magnitud del reward
        reward_magnitude = abs(demo.reward)
        base_weight = min(1.0, reward_magnitude / 10.0)  # Normalizar a [0, 1]
        
        # Bonus por consistencia (rewards positivos)
        consistency_bonus = 1.0 if demo.reward > 0 else 0.5
        
        # Penalty por demos muy antiguos
        age_penalty = 1.0 - (time.time() - demo.timestamp) / (24 * 3600)  # 24 horas
        age_penalty = max(0.1, age_penalty)
        
        return base_weight * consistency_bonus * age_penalty

    def _train_network(self) -> None:
        """
        Entrena la red de reward decomposition.
        
        Implementa el algoritmo R-DDQN para aprender pesos dinámicos
        basándose en demos expertos.
        """
        if len(self.expert_demos) < self.batch_size:
            return
        
        try:
            # Seleccionar batch de demos con pesos
            batch_indices = np.random.choice(
                len(self.expert_demos), 
                size=min(self.batch_size, len(self.expert_demos)),
                replace=False,
                p=np.array(self.demo_weights) / sum(self.demo_weights)
            )
            
            batch_demos = [self.expert_demos[i] for i in batch_indices]
            batch_weights = [self.demo_weights[i] for i in batch_indices]
            
            # Entrenar componentes individualmente
            total_loss = 0.0
            for component_name, component in self.reward_components.items():
                component_loss = self._train_component(component, batch_demos, batch_weights)
                total_loss += component_loss
            
            # Actualizar estado de la red
            self.training_loss = total_loss / len(self.reward_components)
            self.is_trained = True
            
            logger.debug(f"Red entrenada - Loss: {self.training_loss:.4f}")
            
        except Exception as e:
            logger.error(f"Error entrenando red de decomposition: {e}")

    def _train_component(self, component: RewardComponent, 
                        demos: List[ExpertDemo], 
                        weights: List[float]) -> float:
        """
        Entrena un componente específico de reward.
        
        Args:
            component: Componente a entrenar
            demos: Demos de entrenamiento
            weights: Pesos de los demos
            
        Returns:
            Loss del componente
        """
        try:
            # Extraer features relevantes para el componente
            features = self._extract_component_features(component.name, demos)
            targets = [demo.reward for demo in demos]
            
            # Calcular nuevo peso basado en correlación con target
            if len(features) > 1:
                correlation = np.corrcoef(features, targets)[0, 1]
                if not np.isnan(correlation):
                    # Actualizar peso basado en correlación
                    new_weight = component.base_weight * (1 + correlation * 0.5)
                    component.current_weight = max(0.01, min(1.0, new_weight))
                    component.last_update = time.time()
            
            # Calcular loss (MSE ponderado)
            predictions = features * component.current_weight
            weighted_errors = [(pred - target) ** 2 * weight 
                             for pred, target, weight in zip(predictions, targets, weights)]
            loss = np.mean(weighted_errors)
            
            return loss
            
        except Exception as e:
            logger.error(f"Error entrenando componente {component.name}: {e}")
            return 0.0

    def _extract_component_features(self, component_name: str, 
                                  demos: List[ExpertDemo]) -> np.ndarray:
        """
        Extrae features relevantes para un componente específico.
        
        Args:
            component_name: Nombre del componente
            demos: Demos de entrenamiento
            
        Returns:
            Array de features extraídas
        """
        features = []
        
        for demo in demos:
            obs = demo.obs
            
            if component_name == "profit_loss":
                # Features relacionadas con P&L
                pnl = obs.get("realized_pnl", 0.0)
                notional = obs.get("notional", 1.0)
                feature = pnl / max(notional, 1e-8)
                
            elif component_name == "risk_adjusted":
                # Features relacionadas con riesgo
                pnl = obs.get("realized_pnl", 0.0)
                drawdown = obs.get("max_drawdown", 0.0)
                feature = pnl / max(drawdown, 1e-8) if drawdown > 0 else pnl
                
            elif component_name == "time_efficiency":
                # Features relacionadas con eficiencia temporal
                pnl = obs.get("realized_pnl", 0.0)
                bars_held = obs.get("bars_held", 1)
                feature = pnl / max(bars_held, 1)
                
            elif component_name == "volatility_adaptation":
                # Features relacionadas con volatilidad
                atr = obs.get("atr", 0.0)
                price = obs.get("price", 1.0)
                feature = atr / max(price, 1e-8)
                
            elif component_name == "drawdown_control":
                # Features relacionadas con control de drawdown
                drawdown = obs.get("max_drawdown", 0.0)
                equity = obs.get("equity", 1.0)
                feature = -drawdown / max(equity, 1e-8)
                
            elif component_name == "exploration_bonus":
                # Features relacionadas con exploración
                leverage = obs.get("leverage_used", 1.0)
                timeframe = obs.get("timeframe_used", "1m")
                feature = leverage * (1.0 if timeframe == "1m" else 0.5)
                
            elif component_name == "consistency":
                # Features relacionadas con consistencia
                recent_rewards = obs.get("recent_rewards", [])
                if recent_rewards:
                    feature = np.std(recent_rewards) if len(recent_rewards) > 1 else 0.0
                else:
                    feature = 0.0
                    
            else:
                feature = 0.0
            
            features.append(feature)
        
        return np.array(features)

    @profile_reward_calculation("reward_decomposition_predict")
    def predict_reward(self, obs: Dict[str, Any], action: int) -> Tuple[float, Dict[str, float]]:
        """
        Predice reward usando la red de decomposition.
        
        Args:
            obs: Observación actual
            action: Acción a tomar
            
        Returns:
            Tupla (reward_predicho, componentes_detallados)
        """
        if not self.enabled or not self.is_trained:
            return 0.0, {}
        
        try:
            total_reward = 0.0
            components = {}
            
            # Predecir cada componente
            for component_name, component in self.reward_components.items():
                # Extraer features para el componente
                features = self._extract_component_features(component_name, [ExpertDemo(
                    obs=obs, action=action, reward=0.0, next_obs={}, done=False, timestamp=time.time()
                )])
                
                if len(features) > 0:
                    # Predecir reward del componente
                    component_reward = features[0] * component.current_weight
                    total_reward += component_reward
                    components[f"{component_name}_prediction"] = component_reward
            
            # Aplicar peso de la red vs expertos
            final_reward = (total_reward * self.model_weight + 
                          self._get_expert_reward(obs, action) * self.expert_weight)
            
            # Actualizar estadísticas
            self.total_predictions += 1
            
            return final_reward, components
            
        except Exception as e:
            logger.error(f"Error prediciendo reward: {e}")
            return 0.0, {}

    def _get_expert_reward(self, obs: Dict[str, Any], action: int) -> float:
        """
        Obtiene reward de demos expertos similares.
        
        Args:
            obs: Observación actual
            action: Acción a tomar
            
        Returns:
            Reward promedio de demos similares
        """
        if not self.expert_demos:
            return 0.0
        
        try:
            # Buscar demos similares
            similar_demos = []
            for demo in self.expert_demos:
                similarity = self._calculate_similarity(obs, demo.obs)
                if similarity > 0.7:  # Umbral de similitud
                    similar_demos.append(demo.reward)
            
            if similar_demos:
                return np.mean(similar_demos)
            else:
                return 0.0
                
        except Exception as e:
            logger.error(f"Error obteniendo reward experto: {e}")
            return 0.0

    def _calculate_similarity(self, obs1: Dict[str, Any], obs2: Dict[str, Any]) -> float:
        """
        Calcula similitud entre dos observaciones.
        
        Args:
            obs1: Primera observación
            obs2: Segunda observación
            
        Returns:
            Score de similitud entre 0 y 1
        """
        try:
            # Features clave para similitud
            key_features = ["price", "atr", "leverage_used", "timeframe_used"]
            
            similarities = []
            for feature in key_features:
                val1 = obs1.get(feature, 0.0)
                val2 = obs2.get(feature, 0.0)
                
                if val1 == 0 and val2 == 0:
                    similarities.append(1.0)
                elif val1 == 0 or val2 == 0:
                    similarities.append(0.0)
                else:
                    # Similitud basada en diferencia relativa
                    diff = abs(val1 - val2) / max(abs(val1), abs(val2))
                    similarities.append(1.0 - diff)
            
            return np.mean(similarities)
            
        except Exception:
            return 0.0

    def update_weights(self, actual_reward: float, predicted_reward: float) -> None:
        """
        Actualiza pesos de la red basándose en error de predicción.
        
        Args:
            actual_reward: Reward real observado
            predicted_reward: Reward predicho por la red
        """
        if not self.enabled:
            return
        
        try:
            # Calcular error de predicción
            prediction_error = abs(actual_reward - predicted_reward)
            self.avg_prediction_error = (self.avg_prediction_error * 0.9 + 
                                       prediction_error * 0.1)
            
            # Actualizar pesos de componentes basándose en error
            for component in self.reward_components.values():
                if prediction_error < self.prediction_threshold:
                    # Predicción buena, mantener peso
                    component.current_weight = component.current_weight
                else:
                    # Predicción mala, ajustar peso
                    adjustment = 0.1 if actual_reward > predicted_reward else -0.1
                    component.current_weight = max(0.01, min(1.0, 
                        component.current_weight + adjustment))
                
                # Aplicar decay de peso
                component.current_weight *= self.weight_decay
                component.current_weight = max(0.01, component.current_weight)
            
            # Actualizar precisión
            if prediction_error < self.prediction_threshold:
                self.correct_predictions += 1
            
            self.prediction_accuracy = self.correct_predictions / max(self.total_predictions, 1)
            
        except Exception as e:
            logger.error(f"Error actualizando pesos: {e}")

    def get_stats(self) -> Dict[str, Any]:
        """
        Obtiene estadísticas de la red de decomposition.
        
        Returns:
            Diccionario con estadísticas completas
        """
        return {
            "enabled": self.enabled,
            "is_trained": self.is_trained,
            "training_loss": self.training_loss,
            "prediction_accuracy": self.prediction_accuracy,
            "total_predictions": self.total_predictions,
            "avg_prediction_error": self.avg_prediction_error,
            "expert_demos_count": len(self.expert_demos),
            "components": {
                name: {
                    "base_weight": comp.base_weight,
                    "current_weight": comp.current_weight,
                    "importance": comp.importance,
                    "last_update": comp.last_update
                }
                for name, comp in self.reward_components.items()
            }
        }

    def save_model(self, file_path: str) -> None:
        """Guarda el modelo de decomposition."""
        if not self.enabled:
            return
        
        try:
            model_data = {
                "is_trained": self.is_trained,
                "training_loss": self.training_loss,
                "prediction_accuracy": self.prediction_accuracy,
                "components": {
                    name: {
                        "base_weight": comp.base_weight,
                        "current_weight": comp.current_weight,
                        "importance": comp.importance
                    }
                    for name, comp in self.reward_components.items()
                }
            }
            
            with open(file_path, 'w') as f:
                json.dump(model_data, f, indent=2)
            
            logger.info(f"Modelo de decomposition guardado en {file_path}")
            
        except Exception as e:
            logger.error(f"Error guardando modelo: {e}")

    def load_model(self, file_path: str) -> None:
        """Carga el modelo de decomposition."""
        if not self.enabled:
            return
        
        try:
            with open(file_path, 'r') as f:
                model_data = json.load(f)
            
            self.is_trained = model_data.get("is_trained", False)
            self.training_loss = model_data.get("training_loss", 0.0)
            self.prediction_accuracy = model_data.get("prediction_accuracy", 0.0)
            
            # Cargar pesos de componentes
            components_data = model_data.get("components", {})
            for name, comp_data in components_data.items():
                if name in self.reward_components:
                    self.reward_components[name].current_weight = comp_data.get("current_weight", comp_data.get("base_weight", 0.1))
            
            logger.info(f"Modelo de decomposition cargado desde {file_path}")
            
        except Exception as e:
            logger.error(f"Error cargando modelo: {e}")

    def reset(self) -> None:
        """Resetea la red de decomposition."""
        self.expert_demos.clear()
        self.demo_weights.clear()
        self.is_trained = False
        self.training_loss = 0.0
        self.prediction_accuracy = 0.0
        self.total_predictions = 0
        self.correct_predictions = 0
        self.avg_prediction_error = 0.0
        
        # Resetear pesos a valores base
        for component in self.reward_components.values():
            component.current_weight = component.base_weight
            component.last_update = time.time()
        
        logger.debug("RewardDecompositionNetwork reseteado")
