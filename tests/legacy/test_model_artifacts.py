"""
Test para gestión de artefactos de modelo por símbolo.
"""
import pytest
import tempfile
import json
from pathlib import Path
from train_env.core.model_manager import ModelManager


class TestModelArtifacts:
    def test_model_manager_initialization(self):
        """Test que ModelManager se inicializa correctamente."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = ModelManager(
                symbol="BTCUSDT",
                models_root=tmpdir,
                overwrite=False
            )
            
            # Verificar estructura de directorios
            assert manager.symbol_dir.exists()
            assert (manager.symbol_dir / "checkpoints").exists()
            
            # Verificar rutas de archivos
            assert manager.model_path.name == "BTCUSDT_PPO.zip"
            assert manager.backup_path.name == "BTCUSDT_PPO.zip.backup"
            assert manager.strategies_path.name == "BTCUSDT_strategies.json"
            assert manager.provisional_path.name == "BTCUSDT_strategies_provisional.jsonl"
            assert manager.bad_strategies_path.name == "BTCUSDT_bad_strategies.json"
            assert manager.progress_path.name == "BTCUSDT_progress.json"
            assert manager.runs_path.name == "BTCUSDT_runs.jsonl"

    def test_model_info_empty(self):
        """Test que get_model_info funciona con directorio vacío."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = ModelManager("BTCUSDT", tmpdir)
            
            info = manager.get_model_info()
            
            assert info["symbol"] == "BTCUSDT"
            assert info["model_exists"] is False
            assert info["backup_exists"] is False
            assert info["strategies_count"] == 0
            assert info["provisional_count"] == 0
            assert info["bad_strategies_count"] == 0
            assert info["checkpoints_count"] == 0
            assert info["runs_count"] == 0

    def test_file_paths(self):
        """Test que get_file_paths devuelve todas las rutas correctas."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = ModelManager("BTCUSDT", tmpdir)
            
            paths = manager.get_file_paths()
            
            assert "model" in paths
            assert "backup" in paths
            assert "strategies" in paths
            assert "provisional" in paths
            assert "bad_strategies" in paths
            assert "progress" in paths
            assert "runs" in paths
            assert "checkpoints_dir" in paths
            
            # Verificar que todas las rutas son Path objects
            for path in paths.values():
                assert isinstance(path, Path)

    def test_cleanup_provisional(self):
        """Test que cleanup_provisional elimina el archivo provisional."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = ModelManager("BTCUSDT", tmpdir)
            
            # Crear archivo provisional
            with manager.provisional_path.open("w") as f:
                f.write('{"test": "data"}\n')
            
            assert manager.provisional_path.exists()
            
            # Limpiar provisional
            manager.cleanup_provisional()
            
            assert not manager.provisional_path.exists()

    def test_model_info_with_files(self):
        """Test que get_model_info cuenta archivos existentes correctamente."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = ModelManager("BTCUSDT", tmpdir)
            
            # Crear archivos de prueba
            with manager.strategies_path.open("w") as f:
                json.dump([{"test": 1}, {"test": 2}], f)
            
            with manager.provisional_path.open("w") as f:
                f.write('{"test": "data1"}\n')
                f.write('{"test": "data2"}\n')
            
            with manager.bad_strategies_path.open("w") as f:
                json.dump([{"bad": 1}], f)
            
            with manager.runs_path.open("w") as f:
                f.write('{"run": 1}\n')
                f.write('{"run": 2}\n')
                f.write('{"run": 3}\n')
            
            # Crear checkpoints
            checkpoints_dir = manager.symbol_dir / "checkpoints"
            (checkpoints_dir / "checkpoint_1000000.zip").touch()
            (checkpoints_dir / "checkpoint_2000000.zip").touch()
            
            info = manager.get_model_info()
            
            assert info["strategies_count"] == 2
            assert info["provisional_count"] == 2
            assert info["bad_strategies_count"] == 1
            assert info["checkpoints_count"] == 2
            assert info["runs_count"] == 3

    def test_cleanup_old_checkpoints(self):
        """Test que cleanup_old_checkpoints mantiene solo los últimos N checkpoints."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = ModelManager("BTCUSDT", tmpdir)
            
            checkpoints_dir = manager.symbol_dir / "checkpoints"
            
            # Crear 7 checkpoints
            for i in range(1, 8):
                (checkpoints_dir / f"checkpoint_{i * 1000000}.zip").touch()
            
            assert len(list(checkpoints_dir.glob("checkpoint_*.zip"))) == 7
            
            # Limpiar, manteniendo solo los últimos 5
            manager.cleanup_old_checkpoints(keep_last=5)
            
            remaining = list(checkpoints_dir.glob("checkpoint_*.zip"))
            assert len(remaining) == 5
            
            # Verificar que se mantuvieron los más recientes
            timesteps = [int(f.stem.split("_")[1]) for f in remaining]
            timesteps.sort(reverse=True)
            assert timesteps == [7000000, 6000000, 5000000, 4000000, 3000000]

    def test_different_symbols(self):
        """Test que diferentes símbolos tienen directorios separados."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager1 = ModelManager("BTCUSDT", tmpdir)
            manager2 = ModelManager("ETHUSDT", tmpdir)
            
            # Verificar que tienen directorios diferentes
            assert manager1.symbol_dir != manager2.symbol_dir
            assert manager1.symbol_dir.name == "BTCUSDT"
            assert manager2.symbol_dir.name == "ETHUSDT"
            
            # Verificar que ambos directorios existen
            assert manager1.symbol_dir.exists()
            assert manager2.symbol_dir.exists()
            
            # Verificar que los archivos tienen nombres diferentes
            assert manager1.model_path.name == "BTCUSDT_PPO.zip"
            assert manager2.model_path.name == "ETHUSDT_PPO.zip"

    def test_overwrite_flag(self):
        """Test que el flag overwrite se almacena correctamente."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager1 = ModelManager("BTCUSDT", tmpdir, overwrite=False)
            manager2 = ModelManager("ETHUSDT", tmpdir, overwrite=True)
            
            # El flag overwrite se usa internamente en load_model
            # Aquí solo verificamos que se almacena correctamente
            assert hasattr(manager1, 'overwrite')
            assert hasattr(manager2, 'overwrite')
            assert manager1.overwrite is False
            assert manager2.overwrite is True
