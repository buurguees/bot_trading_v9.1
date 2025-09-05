#!/usr/bin/env python3
"""
Tests para TrainingMetricsCallback
"""

import json
import tempfile
import time
from pathlib import Path
from unittest.mock import Mock, patch
import pytest

# Añadir el directorio raíz al path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from train_env.callbacks.training_metrics_callback import TrainingMetricsCallback


class TestTrainingMetricsCallback:
    """Tests para el callback de métricas de entrenamiento."""
    
    def test_callback_initialization(self):
        """Test de inicialización del callback."""
        with tempfile.TemporaryDirectory() as temp_dir:
            metrics_path = Path(temp_dir) / "test_metrics.jsonl"
            
            callback = TrainingMetricsCallback(
                symbol="BTCUSDT",
                mode="train_futures",
                metrics_path=str(metrics_path),
                log_interval=100,
                verbose=0
            )
            
            assert callback.symbol == "BTCUSDT"
            assert callback.mode == "train_futures"
            assert callback.metrics_path == metrics_path
            assert callback.log_interval == 100
            assert callback.iterations == 0
    
    def test_metrics_file_creation(self):
        """Test de creación del archivo de métricas."""
        with tempfile.TemporaryDirectory() as temp_dir:
            metrics_path = Path(temp_dir) / "test_metrics.jsonl"
            
            callback = TrainingMetricsCallback(
                symbol="BTCUSDT",
                mode="train_futures",
                metrics_path=str(metrics_path),
                log_interval=100,
                verbose=0
            )
            
            # Simular modelo con logger
            mock_model = Mock()
            mock_model.num_timesteps = 0
            mock_model.logger = Mock()
            mock_model.logger.name_to_value = {
                'train/approx_kl': 0.01,
                'train/clip_fraction': 0.1,
                'train/learning_rate': 0.0003,
                'train/loss': 0.5
            }
            
            callback.model = mock_model
            
            # Simular entrenamiento
            callback._on_training_start()
            
            # Simular algunos steps
            for i in range(1, 301):  # 300 steps
                mock_model.num_timesteps = i
                callback._on_step()
            
            # Verificar que se creó el archivo
            assert metrics_path.exists()
            
            # Leer y verificar contenido
            with open(metrics_path, 'r') as f:
                lines = f.readlines()
            
            # Debería haber 3 líneas (steps 100, 200, 300)
            assert len(lines) == 3
            
            # Verificar estructura de la primera línea
            first_metrics = json.loads(lines[0])
            assert first_metrics['symbol'] == 'BTCUSDT'
            assert first_metrics['mode'] == 'train_futures'
            assert first_metrics['total_timesteps'] == 100
            assert first_metrics['iterations'] == 1
            assert 'ts' in first_metrics
            assert 'fps' in first_metrics
            assert first_metrics['approx_kl'] == 0.01
            assert first_metrics['learning_rate'] == 0.0003
    
    def test_metrics_extraction(self):
        """Test de extracción de métricas del logger."""
        with tempfile.TemporaryDirectory() as temp_dir:
            metrics_path = Path(temp_dir) / "test_metrics.jsonl"
            
            callback = TrainingMetricsCallback(
                symbol="BTCUSDT",
                mode="train_futures",
                metrics_path=str(metrics_path),
                log_interval=100,
                verbose=0
            )
            
            # Simular modelo con logger completo
            mock_model = Mock()
            mock_model.num_timesteps = 100
            mock_model.logger = Mock()
            mock_model.logger.name_to_value = {
                'train/approx_kl': 0.015,
                'train/clip_fraction': 0.12,
                'train/clip_range': 0.2,
                'train/entropy_loss': -0.1,
                'train/explained_variance': 0.8,
                'train/learning_rate': 0.0003,
                'train/loss': 0.45,
                'train/n_updates': 5,
                'train/policy_gradient_loss': 0.2,
                'train/value_loss': 0.25
            }
            
            callback.model = mock_model
            
            # Extraer métricas
            metrics = callback._extract_metrics_from_logger()
            
            # Verificar que se extrajeron todas las métricas
            assert metrics['approx_kl'] == 0.015
            assert metrics['clip_fraction'] == 0.12
            assert metrics['clip_range'] == 0.2
            assert metrics['entropy_loss'] == -0.1
            assert metrics['explained_variance'] == 0.8
            assert metrics['learning_rate'] == 0.0003
            assert metrics['loss'] == 0.45
            assert metrics['n_updates'] == 5
            assert metrics['policy_gradient_loss'] == 0.2
            assert metrics['value_loss'] == 0.25
    
    def test_metrics_with_missing_values(self):
        """Test con métricas faltantes en el logger."""
        with tempfile.TemporaryDirectory() as temp_dir:
            metrics_path = Path(temp_dir) / "test_metrics.jsonl"
            
            callback = TrainingMetricsCallback(
                symbol="BTCUSDT",
                mode="train_futures",
                metrics_path=str(metrics_path),
                log_interval=100,
                verbose=0
            )
            
            # Simular modelo con logger parcial
            mock_model = Mock()
            mock_model.num_timesteps = 100
            mock_model.logger = Mock()
            mock_model.logger.name_to_value = {
                'train/learning_rate': 0.0003,
                'train/loss': 0.5
                # Faltan otras métricas
            }
            
            callback.model = mock_model
            
            # Extraer métricas
            metrics = callback._extract_metrics_from_logger()
            
            # Verificar que las métricas disponibles están presentes
            assert metrics['learning_rate'] == 0.0003
            assert metrics['loss'] == 0.5
            
            # Verificar que las métricas faltantes son None
            assert metrics['approx_kl'] is None
            assert metrics['clip_fraction'] is None
            assert metrics['entropy_loss'] is None
    
    def test_fps_calculation(self):
        """Test de cálculo de FPS."""
        with tempfile.TemporaryDirectory() as temp_dir:
            metrics_path = Path(temp_dir) / "test_metrics.jsonl"
            
            callback = TrainingMetricsCallback(
                symbol="BTCUSDT",
                mode="train_futures",
                metrics_path=str(metrics_path),
                log_interval=100,
                verbose=0
            )
            
            # Simular modelo
            mock_model = Mock()
            mock_model.num_timesteps = 0
            mock_model.logger = Mock()
            mock_model.logger.name_to_value = {}
            
            callback.model = mock_model
            callback._on_training_start()
            
            # Simular tiempo transcurrido
            time.sleep(0.1)  # 100ms
            
            # Simular 100 timesteps
            mock_model.num_timesteps = 100
            callback._on_step()
            
            # Verificar que se calculó FPS
            with open(metrics_path, 'r') as f:
                line = f.readline()
                metrics = json.loads(line)
                
                assert 'fps' in metrics
                assert metrics['fps'] > 0  # Debería ser positivo
                assert metrics['total_timesteps'] == 100
    
    def test_file_lock_safety(self):
        """Test de seguridad del file lock."""
        with tempfile.TemporaryDirectory() as temp_dir:
            metrics_path = Path(temp_dir) / "test_metrics.jsonl"
            
            callback = TrainingMetricsCallback(
                symbol="BTCUSDT",
                mode="train_futures",
                metrics_path=str(metrics_path),
                log_interval=100,
                verbose=0
            )
            
            # Simular modelo
            mock_model = Mock()
            mock_model.num_timesteps = 100
            mock_model.logger = Mock()
            mock_model.logger.name_to_value = {}
            
            callback.model = mock_model
            
            # Escribir múltiples snapshots simultáneamente
            import threading
            
            def write_snapshot():
                callback._log_metrics()
            
            threads = []
            for _ in range(5):
                thread = threading.Thread(target=write_snapshot)
                threads.append(thread)
                thread.start()
            
            for thread in threads:
                thread.join()
            
            # Verificar que el archivo tiene exactamente 5 líneas
            with open(metrics_path, 'r') as f:
                lines = f.readlines()
                assert len(lines) == 5
                
                # Verificar que todas las líneas son JSON válido
                for line in lines:
                    metrics = json.loads(line)
                    assert metrics['symbol'] == 'BTCUSDT'


if __name__ == "__main__":
    pytest.main([__file__])
