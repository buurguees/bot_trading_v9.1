# tests/test_watcher_monitor.py
"""
Test para validar el watch_progress.py mejorado
"""

import pytest
import tempfile
import json
from pathlib import Path
from unittest.mock import patch
import sys

# Añadir el directorio raíz al path
sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.watch_progress import ConsoleMonitor


class TestWatcherMonitor:
    """Test del monitor de consola mejorado"""
    
    def setup_method(self):
        """Setup para cada test"""
        self.temp_dir = tempfile.mkdtemp()
        self.models_dir = Path(self.temp_dir) / "models" / "BTCUSDT"
        self.models_dir.mkdir(parents=True, exist_ok=True)
        
        # Crear monitor
        self.monitor = ConsoleMonitor(
            symbol="BTCUSDT",
            models_root=str(self.temp_dir),
            refresh_interval=1.0
        )
    
    def create_test_runs(self, runs_data):
        """Crea archivo de runs de prueba"""
        runs_file = self.models_dir / "BTCUSDT_runs.jsonl"
        
        with open(runs_file, 'w') as f:
            for run in runs_data:
                f.write(json.dumps(run) + '\n')
        
        return runs_file
    
    def test_calculate_kpis_basic(self):
        """Test cálculo básico de KPIs"""
        # Crear runs de prueba
        runs_data = [
            {
                "final_equity": 1200.0,
                "final_balance": 1200.0,
                "initial_balance": 1000.0,
                "trades_count": 5,
                "steps": 1000,
                "run_result": "END_OF_HISTORY"
            },
            {
                "final_equity": 800.0,
                "final_balance": 800.0,
                "initial_balance": 1000.0,
                "trades_count": 3,
                "steps": 800,
                "run_result": "END_OF_HISTORY"
            }
        ]
        
        self.create_test_runs(runs_data)
        
        # Cargar runs
        runs = self.monitor._load_runs()
        
        # Calcular KPIs
        kpis = self.monitor._calculate_kpis(runs)
        
        # Verificar KPIs básicos
        assert kpis['total_runs'] == 2
        assert kpis['best_equity'] == 1200.0
        assert kpis['worst_equity'] == 800.0
        assert kpis['avg_equity'] == 1000.0
        
        # Verificar win rate
        assert kpis['win_rate'] == 50.0  # 1 de 2 runs ganadores
    
    def test_load_runs_empty_file(self):
        """Test carga de runs con archivo vacío"""
        # Crear archivo vacío
        runs_file = self.models_dir / "BTCUSDT_runs.jsonl"
        runs_file.touch()
        
        # Cargar runs
        runs = self.monitor._load_runs()
        
        # Verificar que está vacío
        assert len(runs) == 0


if __name__ == "__main__":
    pytest.main([__file__])