#!/usr/bin/env python3
"""
Tests para el bloque de métricas de entrenamiento en watch_progress.py
"""

import json
import tempfile
from pathlib import Path
from unittest.mock import patch, Mock
import pytest

# Añadir el directorio raíz al path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.watch_progress import read_training_metrics, ConsoleMonitor


class TestWatchProgressTrainingBlock:
    """Tests para el bloque de métricas de entrenamiento en watch_progress."""
    
    def test_read_training_metrics_success(self):
        """Test de lectura exitosa de métricas de entrenamiento."""
        with tempfile.TemporaryDirectory() as temp_dir:
            models_root = Path(temp_dir)
            symbol = "BTCUSDT"
            metrics_file = models_root / symbol / f"{symbol}_train_metrics.jsonl"
            
            # Crear directorio y archivo
            metrics_file.parent.mkdir(parents=True, exist_ok=True)
            
            # Escribir métricas de prueba
            test_metrics = [
                {
                    "ts": 1640995200000,
                    "symbol": "BTCUSDT",
                    "mode": "train_futures",
                    "fps": 150.5,
                    "iterations": 1,
                    "time_elapsed": 10.2,
                    "total_timesteps": 2048,
                    "approx_kl": 0.01,
                    "clip_fraction": 0.1,
                    "clip_range": 0.2,
                    "entropy_loss": -0.05,
                    "explained_variance": 0.8,
                    "learning_rate": 0.0003,
                    "loss": 0.45,
                    "n_updates": 5,
                    "policy_gradient_loss": 0.2,
                    "value_loss": 0.25
                },
                {
                    "ts": 1640995260000,
                    "symbol": "BTCUSDT",
                    "mode": "train_futures",
                    "fps": 160.2,
                    "iterations": 2,
                    "time_elapsed": 20.5,
                    "total_timesteps": 4096,
                    "approx_kl": 0.015,
                    "clip_fraction": 0.12,
                    "clip_range": 0.2,
                    "entropy_loss": -0.06,
                    "explained_variance": 0.82,
                    "learning_rate": 0.0003,
                    "loss": 0.42,
                    "n_updates": 10,
                    "policy_gradient_loss": 0.18,
                    "value_loss": 0.24
                }
            ]
            
            with open(metrics_file, 'w') as f:
                for metrics in test_metrics:
                    f.write(json.dumps(metrics) + '\n')
            
            # Leer métricas
            result = read_training_metrics(symbol, models_root)
            
            # Verificar que se leyó la última línea
            assert result is not None
            assert result['iterations'] == 2
            assert result['total_timesteps'] == 4096
            assert result['fps'] == 160.2
            assert result['approx_kl'] == 0.015
    
    def test_read_training_metrics_file_not_found(self):
        """Test cuando el archivo de métricas no existe."""
        with tempfile.TemporaryDirectory() as temp_dir:
            models_root = Path(temp_dir)
            symbol = "BTCUSDT"
            
            result = read_training_metrics(symbol, models_root)
            
            assert result is None
    
    def test_read_training_metrics_corrupted_file(self):
        """Test con archivo corrupto o líneas inválidas."""
        with tempfile.TemporaryDirectory() as temp_dir:
            models_root = Path(temp_dir)
            symbol = "BTCUSDT"
            metrics_file = models_root / symbol / f"{symbol}_train_metrics.jsonl"
            
            # Crear directorio y archivo
            metrics_file.parent.mkdir(parents=True, exist_ok=True)
            
            # Escribir archivo con líneas corruptas y válidas
            with open(metrics_file, 'w') as f:
                f.write('{"ts": 1640995200000, "symbol": "BTCUSDT", "iterations": 1}\n')  # Válida
                f.write('{"corrupted": json}\n')  # Corrupta
                f.write('\n')  # Vacía
                f.write('{"ts": 1640995260000, "symbol": "BTCUSDT", "iterations": 2}\n')  # Válida
                f.write('{"incomplete": "json"')  # Incompleta
            
            # Leer métricas
            result = read_training_metrics(symbol, models_root)
            
            # Debería devolver la última línea válida
            assert result is not None
            assert result['iterations'] == 2
            assert result['ts'] == 1640995260000
    
    def test_read_training_metrics_empty_file(self):
        """Test con archivo vacío."""
        with tempfile.TemporaryDirectory() as temp_dir:
            models_root = Path(temp_dir)
            symbol = "BTCUSDT"
            metrics_file = models_root / symbol / f"{symbol}_train_metrics.jsonl"
            
            # Crear directorio y archivo vacío
            metrics_file.parent.mkdir(parents=True, exist_ok=True)
            metrics_file.touch()
            
            result = read_training_metrics(symbol, models_root)
            
            assert result is None
    
    @patch('scripts.watch_progress.read_training_metrics')
    def test_console_monitor_with_metrics(self, mock_read_metrics):
        """Test del ConsoleMonitor mostrando métricas de entrenamiento."""
        # Mock de métricas
        mock_metrics = {
            "ts": 1640995200000,
            "symbol": "BTCUSDT",
            "mode": "train_futures",
            "fps": 150.5,
            "iterations": 10,
            "time_elapsed": 100.2,
            "total_timesteps": 20480,
            "approx_kl": 0.01,
            "clip_fraction": 0.1,
            "clip_range": 0.2,
            "entropy_loss": -0.05,
            "explained_variance": 0.8,
            "learning_rate": 0.0003,
            "loss": 0.45,
            "n_updates": 50,
            "policy_gradient_loss": 0.2,
            "value_loss": 0.25
        }
        mock_read_metrics.return_value = mock_metrics
        
        with tempfile.TemporaryDirectory() as temp_dir:
            models_root = Path(temp_dir)
            symbol = "BTCUSDT"
            
            # Crear archivo de runs vacío
            runs_file = models_root / symbol / f"{symbol}_runs.jsonl"
            runs_file.parent.mkdir(parents=True, exist_ok=True)
            runs_file.touch()
            
            # Crear monitor
            monitor = ConsoleMonitor(
                symbol=symbol,
                models_root=str(models_root),
                refresh_interval=1
            )
            
            # Capturar output
            import io
            import contextlib
            
            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                monitor._display_status()
            
            output_text = output.getvalue()
            
            # Verificar que se muestran las métricas
            assert "TRAINING METRICS" in output_text
            assert "fps: 150.5" in output_text
            assert "iterations: 10" in output_text
            assert "total_timesteps: 20,480" in output_text
            assert "approx_kl: 0.01" in output_text
            assert "learning_rate: 0.0003" in output_text
    
    @patch('scripts.watch_progress.read_training_metrics')
    def test_console_monitor_without_metrics(self, mock_read_metrics):
        """Test del ConsoleMonitor sin métricas de entrenamiento."""
        mock_read_metrics.return_value = None
        
        with tempfile.TemporaryDirectory() as temp_dir:
            models_root = Path(temp_dir)
            symbol = "BTCUSDT"
            
            # Crear archivo de runs vacío
            runs_file = models_root / symbol / f"{symbol}_runs.jsonl"
            runs_file.parent.mkdir(parents=True, exist_ok=True)
            runs_file.touch()
            
            # Crear monitor
            monitor = ConsoleMonitor(
                symbol=symbol,
                models_root=str(models_root),
                refresh_interval=1
            )
            
            # Capturar output
            import io
            import contextlib
            
            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                monitor._display_status()
            
            output_text = output.getvalue()
            
            # Verificar que se muestra mensaje de no datos
            assert "TRAINING METRICS: —" in output_text
            assert "archivo no encontrado o sin datos" in output_text
    
    def test_metrics_formatting(self):
        """Test del formateo de métricas en la consola."""
        with tempfile.TemporaryDirectory() as temp_dir:
            models_root = Path(temp_dir)
            symbol = "BTCUSDT"
            metrics_file = models_root / symbol / f"{symbol}_train_metrics.jsonl"
            
            # Crear directorio y archivo
            metrics_file.parent.mkdir(parents=True, exist_ok=True)
            
            # Escribir métricas con valores None
            test_metrics = {
                "ts": 1640995200000,
                "symbol": "BTCUSDT",
                "mode": "train_futures",
                "fps": 150.5,
                "iterations": 1,
                "time_elapsed": 10.2,
                "total_timesteps": 2048,
                "approx_kl": None,  # Valor faltante
                "clip_fraction": 0.1,
                "clip_range": 0.2,
                "entropy_loss": -0.05,
                "explained_variance": None,  # Valor faltante
                "learning_rate": 0.0003,
                "loss": 0.45,
                "n_updates": 5,
                "policy_gradient_loss": 0.2,
                "value_loss": 0.25
            }
            
            with open(metrics_file, 'w') as f:
                f.write(json.dumps(test_metrics) + '\n')
            
            # Leer métricas
            result = read_training_metrics(symbol, models_root)
            
            # Verificar que se manejan los valores None
            assert result is not None
            assert result['approx_kl'] is None
            assert result['explained_variance'] is None
            assert result['fps'] == 150.5
            assert result['learning_rate'] == 0.0003


if __name__ == "__main__":
    pytest.main([__file__])
