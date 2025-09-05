# train_env/monitoring/real_time_monitor.py
"""
Sistema de monitoreo en tiempo real para entrenamiento PPO.
Detecta problemas early y proporciona métricas detalladas.
"""

from __future__ import annotations
import time
import psutil
import threading
import queue
from typing import Dict, Any, Optional, List, Callable
from dataclasses import dataclass, field
from collections import deque
import logging
import json
from pathlib import Path

logger = logging.getLogger(__name__)

@dataclass
class MetricPoint:
    """Punto de métrica con timestamp"""
    name: str
    value: float
    timestamp: float
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class HealthReport:
    """Reporte de salud del entrenamiento"""
    timestamp: float
    overall_health: str  # "healthy", "warning", "critical"
    learning_progress: Dict[str, Any]
    memory_usage: Dict[str, Any]
    convergence_risk: Dict[str, Any]
    exploration_balance: Dict[str, Any]
    recommendations: List[str] = field(default_factory=list)

class RealTimeMonitor:
    """Monitor en tiempo real para entrenamiento PPO"""
    
    def __init__(self, 
                 update_interval: int = 10,
                 max_history: int = 1000,
                 alert_thresholds: Optional[Dict[str, float]] = None):
        """
        Inicializa el monitor en tiempo real
        
        Args:
            update_interval: Intervalo de actualización en segundos
            max_history: Máximo número de puntos históricos
            alert_thresholds: Umbrales para alertas
        """
        self.update_interval = update_interval
        self.max_history = max_history
        self.alert_thresholds = alert_thresholds or self._get_default_thresholds()
        
        # Almacenamiento de métricas
        self.metrics: Dict[str, deque] = {}
        self.alerts: List[str] = []
        self.health_history: deque = deque(maxlen=100)
        
        # Estado del monitor
        self.is_running = False
        self.monitor_thread: Optional[threading.Thread] = None
        self.metric_queue = queue.Queue()
        
        # Callbacks
        self.alert_callbacks: List[Callable[[str, Dict], None]] = []
        self.health_callbacks: List[Callable[[HealthReport], None]] = []
        
        # Métricas de entrenamiento
        self.training_metrics = {
            'episode_reward': deque(maxlen=100),
            'policy_loss': deque(maxlen=100),
            'value_loss': deque(maxlen=100),
            'entropy_loss': deque(maxlen=100),
            'learning_rate': deque(maxlen=100),
            'explained_variance': deque(maxlen=100)
        }
        
    def _get_default_thresholds(self) -> Dict[str, float]:
        """Retorna umbrales por defecto para alertas"""
        return {
            'memory_usage_percent': 85.0,
            'cpu_usage_percent': 90.0,
            'reward_stagnation_steps': 1000,
            'loss_explosion_threshold': 10.0,
            'learning_rate_min': 1e-6,
            'learning_rate_max': 1e-2
        }
    
    def start_monitoring(self):
        """Inicia el monitoreo en tiempo real"""
        if self.is_running:
            logger.warning("Monitor ya está ejecutándose")
            return
        
        self.is_running = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()
        logger.info(f"🔍 Monitor iniciado (intervalo: {self.update_interval}s)")
    
    def stop_monitoring(self):
        """Detiene el monitoreo"""
        self.is_running = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5)
        logger.info("⏹️ Monitor detenido")
    
    def _monitor_loop(self):
        """Loop principal del monitor"""
        while self.is_running:
            try:
                # Procesar métricas en cola
                self._process_metric_queue()
                
                # Recolectar métricas del sistema
                self._collect_system_metrics()
                
                # Verificar salud del entrenamiento
                health_report = self.check_training_health()
                if health_report:
                    self.health_history.append(health_report)
                    self._notify_health_callbacks(health_report)
                
                # Verificar alertas
                self._check_alerts()
                
                time.sleep(self.update_interval)
                
            except Exception as e:
                logger.error(f"Error en loop de monitoreo: {e}")
                time.sleep(self.update_interval)
    
    def track_metric(self, name: str, value: float, timestamp: float = None, metadata: Dict[str, Any] = None):
        """Registra una métrica con timestamp"""
        if timestamp is None:
            timestamp = time.time()
        
        metric_point = MetricPoint(
            name=name,
            value=value,
            timestamp=timestamp,
            metadata=metadata or {}
        )
        
        self.metric_queue.put(metric_point)
    
    def _process_metric_queue(self):
        """Procesa métricas en cola"""
        while not self.metric_queue.empty():
            try:
                metric = self.metric_queue.get_nowait()
                
                if metric.name not in self.metrics:
                    self.metrics[metric.name] = deque(maxlen=self.max_history)
                
                self.metrics[metric.name].append(metric)
                
                # Actualizar métricas de entrenamiento específicas
                if metric.name in self.training_metrics:
                    self.training_metrics[metric.name].append(metric.value)
                
            except queue.Empty:
                break
            except Exception as e:
                logger.error(f"Error procesando métrica: {e}")
    
    def _collect_system_metrics(self):
        """Recolecta métricas del sistema"""
        try:
            # CPU
            cpu_percent = psutil.cpu_percent(interval=1)
            self.track_metric('cpu_usage_percent', cpu_percent)
            
            # Memoria
            memory = psutil.virtual_memory()
            self.track_metric('memory_usage_percent', memory.percent)
            self.track_metric('memory_available_gb', memory.available / (1024**3))
            self.track_metric('memory_used_gb', memory.used / (1024**3))
            
            # Disco
            disk = psutil.disk_usage('/')
            self.track_metric('disk_usage_percent', (disk.used / disk.total) * 100)
            self.track_metric('disk_free_gb', disk.free / (1024**3))
            
        except Exception as e:
            logger.error(f"Error recolectando métricas del sistema: {e}")
    
    def check_training_health(self) -> Optional[HealthReport]:
        """Evalúa la salud del entrenamiento"""
        try:
            # Verificar que tenemos métricas suficientes
            if not any(self.training_metrics.values()):
                return None
            
            # Análisis de progreso de aprendizaje
            learning_progress = self._check_learning_progress()
            
            # Análisis de uso de memoria
            memory_usage = self._check_memory_usage()
            
            # Análisis de riesgo de convergencia
            convergence_risk = self._check_convergence_risk()
            
            # Análisis de balance de exploración
            exploration_balance = self._check_exploration_balance()
            
            # Determinar salud general
            overall_health = self._determine_overall_health(
                learning_progress, memory_usage, convergence_risk, exploration_balance
            )
            
            # Generar recomendaciones
            recommendations = self._generate_recommendations(
                learning_progress, memory_usage, convergence_risk, exploration_balance
            )
            
            return HealthReport(
                timestamp=time.time(),
                overall_health=overall_health,
                learning_progress=learning_progress,
                memory_usage=memory_usage,
                convergence_risk=convergence_risk,
                exploration_balance=exploration_balance,
                recommendations=recommendations
            )
            
        except Exception as e:
            logger.error(f"Error verificando salud del entrenamiento: {e}")
            return None
    
    def _check_learning_progress(self) -> Dict[str, Any]:
        """Verifica el progreso del aprendizaje"""
        episode_rewards = list(self.training_metrics['episode_reward'])
        
        if not episode_rewards:
            return {"status": "no_data", "trend": "unknown"}
        
        # Calcular tendencia
        if len(episode_rewards) >= 10:
            recent_avg = sum(episode_rewards[-10:]) / 10
            older_avg = sum(episode_rewards[-20:-10]) / 10 if len(episode_rewards) >= 20 else recent_avg
            trend = "improving" if recent_avg > older_avg else "stagnant" if abs(recent_avg - older_avg) < 0.01 else "declining"
        else:
            trend = "insufficient_data"
        
        # Calcular estadísticas
        current_reward = episode_rewards[-1] if episode_rewards else 0
        best_reward = max(episode_rewards) if episode_rewards else 0
        avg_reward = sum(episode_rewards) / len(episode_rewards)
        
        return {
            "status": "healthy" if trend == "improving" else "warning" if trend == "stagnant" else "critical",
            "trend": trend,
            "current_reward": current_reward,
            "best_reward": best_reward,
            "avg_reward": avg_reward,
            "episodes_tracked": len(episode_rewards)
        }
    
    def _check_memory_usage(self) -> Dict[str, Any]:
        """Verifica el uso de memoria"""
        memory_metrics = self.metrics.get('memory_usage_percent', deque())
        
        if not memory_metrics:
            return {"status": "no_data"}
        
        current_usage = memory_metrics[-1].value
        avg_usage = sum(m.value for m in memory_metrics) / len(memory_metrics)
        max_usage = max(m.value for m in memory_metrics)
        
        if current_usage > self.alert_thresholds['memory_usage_percent']:
            status = "critical"
        elif current_usage > self.alert_thresholds['memory_usage_percent'] * 0.8:
            status = "warning"
        else:
            status = "healthy"
        
        return {
            "status": status,
            "current_percent": current_usage,
            "avg_percent": avg_usage,
            "max_percent": max_usage,
            "threshold": self.alert_thresholds['memory_usage_percent']
        }
    
    def _check_convergence_risk(self) -> Dict[str, Any]:
        """Verifica riesgo de convergencia prematura"""
        policy_losses = list(self.training_metrics['policy_loss'])
        value_losses = list(self.training_metrics['value_loss'])
        
        if not policy_losses or not value_losses:
            return {"status": "no_data"}
        
        # Verificar si las pérdidas están estancadas
        if len(policy_losses) >= 50:
            recent_policy_var = self._calculate_variance(policy_losses[-50:])
            recent_value_var = self._calculate_variance(value_losses[-50:])
            
            if recent_policy_var < 0.001 and recent_value_var < 0.001:
                status = "critical"
                risk = "high"
            elif recent_policy_var < 0.01 and recent_value_var < 0.01:
                status = "warning"
                risk = "medium"
            else:
                status = "healthy"
                risk = "low"
        else:
            status = "insufficient_data"
            risk = "unknown"
        
        return {
            "status": status,
            "risk_level": risk,
            "policy_loss_variance": recent_policy_var if len(policy_losses) >= 50 else None,
            "value_loss_variance": recent_value_var if len(value_losses) >= 50 else None
        }
    
    def _check_exploration_balance(self) -> Dict[str, Any]:
        """Verifica el balance de exploración"""
        entropies = list(self.training_metrics['entropy_loss'])
        learning_rates = list(self.training_metrics['learning_rate'])
        
        if not entropies or not learning_rates:
            return {"status": "no_data"}
        
        current_entropy = entropies[-1] if entropies else 0
        current_lr = learning_rates[-1] if learning_rates else 0
        
        # Verificar si la entropía es muy baja (sobre-explotación)
        if current_entropy < 0.01:
            status = "critical"
            balance = "over_exploiting"
        elif current_entropy < 0.05:
            status = "warning"
            balance = "low_exploration"
        else:
            status = "healthy"
            balance = "balanced"
        
        # Verificar learning rate
        lr_status = "healthy"
        if current_lr < self.alert_thresholds['learning_rate_min']:
            lr_status = "too_low"
        elif current_lr > self.alert_thresholds['learning_rate_max']:
            lr_status = "too_high"
        
        return {
            "status": status,
            "balance": balance,
            "current_entropy": current_entropy,
            "current_learning_rate": current_lr,
            "lr_status": lr_status
        }
    
    def _calculate_variance(self, values: List[float]) -> float:
        """Calcula la varianza de una lista de valores"""
        if len(values) < 2:
            return 0.0
        
        mean = sum(values) / len(values)
        variance = sum((x - mean) ** 2 for x in values) / len(values)
        return variance
    
    def _determine_overall_health(self, learning_progress: Dict, memory_usage: Dict, 
                                 convergence_risk: Dict, exploration_balance: Dict) -> str:
        """Determina la salud general del entrenamiento"""
        statuses = [
            learning_progress.get('status', 'unknown'),
            memory_usage.get('status', 'unknown'),
            convergence_risk.get('status', 'unknown'),
            exploration_balance.get('status', 'unknown')
        ]
        
        if 'critical' in statuses:
            return 'critical'
        elif 'warning' in statuses:
            return 'warning'
        else:
            return 'healthy'
    
    def _generate_recommendations(self, learning_progress: Dict, memory_usage: Dict,
                                 convergence_risk: Dict, exploration_balance: Dict) -> List[str]:
        """Genera recomendaciones basadas en el análisis"""
        recommendations = []
        
        # Recomendaciones de progreso de aprendizaje
        if learning_progress.get('trend') == 'stagnant':
            recommendations.append("Considera aumentar el learning rate o ajustar la exploración")
        elif learning_progress.get('trend') == 'declining':
            recommendations.append("El rendimiento está empeorando, revisa los hiperparámetros")
        
        # Recomendaciones de memoria
        if memory_usage.get('status') == 'critical':
            recommendations.append("Uso de memoria crítico, reduce el batch_size o número de workers")
        elif memory_usage.get('status') == 'warning':
            recommendations.append("Uso de memoria alto, monitorea de cerca")
        
        # Recomendaciones de convergencia
        if convergence_risk.get('risk_level') == 'high':
            recommendations.append("Riesgo alto de convergencia prematura, aumenta la exploración")
        elif convergence_risk.get('risk_level') == 'medium':
            recommendations.append("Riesgo medio de convergencia, considera ajustar la entropía")
        
        # Recomendaciones de exploración
        if exploration_balance.get('balance') == 'over_exploiting':
            recommendations.append("Sobre-explotación detectada, aumenta el coeficiente de entropía")
        elif exploration_balance.get('balance') == 'low_exploration':
            recommendations.append("Exploración baja, considera aumentar la entropía o learning rate")
        
        return recommendations
    
    def _check_alerts(self):
        """Verifica condiciones de alerta"""
        current_time = time.time()
        
        # Verificar uso de memoria
        memory_metrics = self.metrics.get('memory_usage_percent', deque())
        if memory_metrics and memory_metrics[-1].value > self.alert_thresholds['memory_usage_percent']:
            alert_msg = f"Uso de memoria crítico: {memory_metrics[-1].value:.1f}%"
            self._trigger_alert(alert_msg, {'type': 'memory', 'value': memory_metrics[-1].value})
        
        # Verificar CPU
        cpu_metrics = self.metrics.get('cpu_usage_percent', deque())
        if cpu_metrics and cpu_metrics[-1].value > self.alert_thresholds['cpu_usage_percent']:
            alert_msg = f"Uso de CPU alto: {cpu_metrics[-1].value:.1f}%"
            self._trigger_alert(alert_msg, {'type': 'cpu', 'value': cpu_metrics[-1].value})
    
    def _trigger_alert(self, message: str, metadata: Dict[str, Any]):
        """Dispara una alerta"""
        self.alerts.append(f"{time.strftime('%H:%M:%S')} - {message}")
        logger.warning(f"🚨 ALERTA: {message}")
        
        # Notificar callbacks
        for callback in self.alert_callbacks:
            try:
                callback(message, metadata)
            except Exception as e:
                logger.error(f"Error en callback de alerta: {e}")
    
    def add_alert_callback(self, callback: Callable[[str, Dict], None]):
        """Añade callback para alertas"""
        self.alert_callbacks.append(callback)
    
    def add_health_callback(self, callback: Callable[[HealthReport], None]):
        """Añade callback para reportes de salud"""
        self.health_callbacks.append(callback)
    
    def _notify_health_callbacks(self, health_report: HealthReport):
        """Notifica callbacks de salud"""
        for callback in self.health_callbacks:
            try:
                callback(health_report)
            except Exception as e:
                logger.error(f"Error en callback de salud: {e}")
    
    def get_metrics_summary(self) -> Dict[str, Any]:
        """Retorna resumen de métricas"""
        summary = {}
        
        for name, metric_deque in self.metrics.items():
            if metric_deque:
                values = [m.value for m in metric_deque]
                summary[name] = {
                    'current': values[-1],
                    'min': min(values),
                    'max': max(values),
                    'avg': sum(values) / len(values),
                    'count': len(values)
                }
        
        return summary
    
    def export_metrics(self, filepath: str):
        """Exporta métricas a archivo JSON"""
        try:
            export_data = {
                'timestamp': time.time(),
                'metrics': self.get_metrics_summary(),
                'health_history': [
                    {
                        'timestamp': hr.timestamp,
                        'overall_health': hr.overall_health,
                        'recommendations': hr.recommendations
                    }
                    for hr in self.health_history
                ],
                'alerts': self.alerts[-100:]  # Últimas 100 alertas
            }
            
            with open(filepath, 'w') as f:
                json.dump(export_data, f, indent=2)
            
            logger.info(f"📊 Métricas exportadas a {filepath}")
            
        except Exception as e:
            logger.error(f"Error exportando métricas: {e}")
    
    def auto_adjust_parameters(self) -> Dict[str, float]:
        """Ajusta parámetros automáticamente basado en métricas"""
        adjustments = {}
        
        # Ajustar learning rate basado en progreso
        learning_progress = self._check_learning_progress()
        if learning_progress.get('trend') == 'stagnant':
            current_lr = self.training_metrics['learning_rate'][-1] if self.training_metrics['learning_rate'] else 3e-4
            adjustments['learning_rate'] = min(current_lr * 1.2, 1e-3)
        
        # Ajustar entropía basado en exploración
        exploration_balance = self._check_exploration_balance()
        if exploration_balance.get('balance') == 'over_exploiting':
            adjustments['ent_coef'] = 0.01  # Aumentar exploración
        
        return adjustments
