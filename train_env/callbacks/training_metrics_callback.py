#!/usr/bin/env python3
"""
TrainingMetricsCallback - Callback para escribir métricas de entrenamiento en tiempo real.

Escribe snapshots periódicos de métricas de entrenamiento SB3 a un archivo JSONL
para monitoreo en tiempo real desde watch_progress.py.
"""

from __future__ import annotations
import json
import time
import os
import threading
from pathlib import Path
from typing import Dict, Any, Optional
from stable_baselines3.common.callbacks import BaseCallback
from base_env.utils.timestamp_utils import add_utc_timestamps


class TrainingMetricsCallback(BaseCallback):
    """
    Callback que escribe métricas de entrenamiento SB3 a un archivo JSONL.
    
    Métricas capturadas:
    - fps, iterations, time_elapsed, total_timesteps
    - approx_kl, clip_fraction, clip_range, entropy_loss
    - explained_variance, learning_rate, loss, n_updates
    - policy_gradient_loss, value_loss
    """
    
    def __init__(
        self,
        symbol: str,
        mode: str,
        metrics_path: str,
        log_interval: int = 2048,
        verbose: int = 0
    ):
        super().__init__(verbose)
        self.symbol = symbol
        self.mode = mode
        self.metrics_path = Path(metrics_path)
        self.log_interval = log_interval
        self.iterations = 0
        self.last_log_time = time.time()
        self.last_timesteps = 0
        self._file_lock = threading.Lock()
        
        # Crear directorio si no existe
        self.metrics_path.parent.mkdir(parents=True, exist_ok=True)
        
        if self.verbose > 0:
            print(f"📊 TrainingMetricsCallback inicializado:")
            print(f"   Symbol: {self.symbol}")
            print(f"   Mode: {self.mode}")
            print(f"   Metrics path: {self.metrics_path}")
            print(f"   Log interval: {self.log_interval}")
    
    def _on_step(self) -> bool:
        """Se ejecuta en cada step del entrenamiento."""
        # Solo loguear cada log_interval timesteps
        if self.model.num_timesteps % self.log_interval == 0:
            self._log_metrics()
        return True
    
    def _log_metrics(self):
        """Escribe las métricas actuales al archivo JSONL."""
        try:
            current_time = time.time()
            time_elapsed = current_time - self.last_log_time
            
            # Calcular FPS
            timesteps_delta = self.model.num_timesteps - self.last_timesteps
            fps = timesteps_delta / time_elapsed if time_elapsed > 0 else 0.0
            
            # Extraer métricas del logger de SB3
            metrics = self._extract_metrics_from_logger()
            
            # Crear snapshot de métricas
            snapshot = {
                "ts": int(current_time * 1000),  # Unix timestamp en ms
                "symbol": self.symbol,
                "mode": self.mode,
                "fps": fps,
                "iterations": self.iterations,
                "time_elapsed": time_elapsed,
                "total_timesteps": self.model.num_timesteps,
                **metrics
            }
            
            # Añadir timestamps UTC legibles
            snapshot = add_utc_timestamps(snapshot)
            
            # Escribir al archivo JSONL (thread-safe)
            self._write_snapshot(snapshot)
            
            # Actualizar contadores
            self.iterations += 1
            self.last_log_time = current_time
            self.last_timesteps = self.model.num_timesteps
            
            if self.verbose > 0:
                print(f"📊 Métricas guardadas: {self.iterations} iteraciones, "
                      f"{self.model.num_timesteps} timesteps, {fps:.1f} fps")
                
        except Exception as e:
            print(f"⚠️ Error guardando métricas: {e}")
    
    def _extract_metrics_from_logger(self) -> Dict[str, Any]:
        """Extrae métricas del logger de SB3."""
        metrics = {}
        
        if hasattr(self.model, 'logger') and self.model.logger is not None:
            # Acceder a las métricas del logger
            name_to_value = getattr(self.model.logger, 'name_to_value', {})
            
            # Mapear nombres de métricas de SB3 a nuestros nombres
            metric_mapping = {
                'train/approx_kl': 'approx_kl',
                'train/clip_fraction': 'clip_fraction',
                'train/clip_range': 'clip_range',
                'train/entropy_loss': 'entropy_loss',
                'train/explained_variance': 'explained_variance',
                'train/learning_rate': 'learning_rate',
                'train/loss': 'loss',
                'train/n_updates': 'n_updates',
                'train/policy_gradient_loss': 'policy_gradient_loss',
                'train/value_loss': 'value_loss'
            }
            
            for sb3_key, our_key in metric_mapping.items():
                if sb3_key in name_to_value:
                    metrics[our_key] = name_to_value[sb3_key]
                else:
                    metrics[our_key] = None
        
        # Si no hay logger o métricas, inicializar con None
        if not metrics:
            for key in ['approx_kl', 'clip_fraction', 'clip_range', 'entropy_loss',
                       'explained_variance', 'learning_rate', 'loss', 'n_updates',
                       'policy_gradient_loss', 'value_loss']:
                metrics[key] = None
        
        return metrics
    
    def _write_snapshot(self, snapshot: Dict[str, Any]):
        """Escribe un snapshot al archivo JSONL de forma thread-safe."""
        with self._file_lock:
            try:
                # Escribir línea JSON al archivo
                with open(self.metrics_path, 'a', encoding='utf-8') as f:
                    json.dump(snapshot, f, ensure_ascii=False)
                    f.write('\n')
                    f.flush()  # Asegurar que se escriba inmediatamente
                    
            except Exception as e:
                print(f"⚠️ Error escribiendo snapshot: {e}")
    
    def _on_training_start(self):
        """Se ejecuta al inicio del entrenamiento."""
        if self.verbose > 0:
            print(f"🚀 Iniciando monitoreo de métricas para {self.symbol}")
        
        # Inicializar contadores
        self.last_log_time = time.time()
        self.last_timesteps = self.model.num_timesteps
        self.iterations = 0
    
    def _on_training_end(self):
        """Se ejecuta al final del entrenamiento."""
        if self.verbose > 0:
            print(f"🏁 Finalizando monitoreo de métricas para {self.symbol}")
            print(f"   Total iteraciones: {self.iterations}")
            print(f"   Archivo: {self.metrics_path}")
