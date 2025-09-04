# tests/test_watch_progress_parser.py
"""
Test para validar el parser de watch_progress.py con KPIs de futuros.
"""

import pytest
import tempfile
import json
from pathlib import Path
from unittest.mock import patch, MagicMock
import sys

# Agregar el directorio raíz al path
sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.watch_progress import ConsoleMonitor, load_runs, find_best_run


class TestWatchProgressParser:
    """Test del parser de watch_progress.py"""
    
    def setup_method(self):
        """Setup para cada test"""
        self.temp_dir = tempfile.mkdtemp()
        self.models_root = Path(self.temp_dir) / "models"
        self.models_root.mkdir(parents=True, exist_ok=True)
        
        self.symbol = "BTCUSDT"
        self.symbol_dir = self.models_root / self.symbol
        self.symbol_dir.mkdir(parents=True, exist_ok=True)
        
        self.runs_file = self.symbol_dir / f"{self.symbol}_runs.jsonl"
        
        # Monitor de consola
        self.monitor = ConsoleMonitor(
            symbol=self.symbol,
            models_root=str(self.models_root),
            refresh_interval=1
        )
    
    def teardown_method(self):
        """Cleanup después de cada test"""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def _create_test_runs(self, runs_data):
        """Crea archivo de runs de prueba"""
        with self.runs_file.open("w", encoding="utf-8") as f:
            for run in runs_data:
                f.write(json.dumps(run) + "\n")
    
    def test_load_runs_basic(self):
        """Test carga básica de runs"""
        # Datos de prueba
        runs_data = [
            {
                "final_equity": 1200.0,
                "final_balance": 1150.0,
                "initial_balance": 1000.0,
                "trades_count": 5,
                "steps": 1000,
                "run_result": "END_OF_HISTORY",
                "ts_end": 1640995200000
            },
            {
                "final_equity": 1100.0,
                "final_balance": 1050.0,
                "initial_balance": 1000.0,
                "trades_count": 3,
                "steps": 800,
                "run_result": "BANKRUPTCY",
                "ts_end": 1640995300000
            }
        ]
        
        # Crear archivo
        self._create_test_runs(runs_data)
        
        # Cargar runs
        runs = load_runs(self.runs_file)
        
        # Verificar
        assert len(runs) == 2
        assert runs[0]["final_equity"] == 1200.0
        assert runs[1]["run_result"] == "BANKRUPTCY"
    
    def test_load_runs_corrupted_file(self):
        """Test que maneja archivos corruptos correctamente"""
        # Crear archivo con líneas corruptas
        with self.runs_file.open("w", encoding="utf-8") as f:
            f.write('{"final_equity": 1200.0, "final_balance": 1150.0}\n')  # Válido
            f.write('{"corrupted": json}\n')  # Inválido
            f.write('{"final_equity": 1100.0, "final_balance": 1050.0}\n')  # Válido
            f.write('\n')  # Línea vacía
            f.write('{"incomplete": "data"\n')  # Incompleto
        
        # Cargar runs
        runs = load_runs(self.runs_file)
        
        # Verificar que solo se cargaron los válidos
        assert len(runs) == 2
        assert runs[0]["final_equity"] == 1200.0
        assert runs[1]["final_equity"] == 1100.0
    
    def test_load_runs_empty_file(self):
        """Test que maneja archivos vacíos correctamente"""
        # Crear archivo vacío
        self.runs_file.touch()
        
        # Cargar runs
        runs = load_runs(self.runs_file)
        
        # Verificar
        assert len(runs) == 0
    
    def test_load_runs_nonexistent_file(self):
        """Test que maneja archivos inexistentes correctamente"""
        # Archivo que no existe
        nonexistent_file = self.models_root / "nonexistent" / "runs.jsonl"
        
        # Cargar runs
        runs = load_runs(nonexistent_file)
        
        # Verificar
        assert len(runs) == 0
    
    def test_find_best_run(self):
        """Test que encuentra el mejor run correctamente"""
        # Datos de prueba
        runs_data = [
            {
                "final_equity": 1200.0,
                "final_balance": 1150.0,
                "initial_balance": 1000.0,
                "trades_count": 5,
                "steps": 1000,
                "run_result": "END_OF_HISTORY"
            },
            {
                "final_equity": 1100.0,
                "final_balance": 1050.0,
                "initial_balance": 1000.0,
                "trades_count": 3,
                "steps": 800,
                "run_result": "BANKRUPTCY"
            },
            {
                "final_equity": 1300.0,  # Mejor equity
                "final_balance": 1250.0,
                "initial_balance": 1000.0,
                "trades_count": 8,
                "steps": 1200,
                "run_result": "END_OF_HISTORY"
            }
        ]
        
        # Crear archivo
        self._create_test_runs(runs_data)
        
        # Cargar runs
        runs = load_runs(self.runs_file)
        
        # Encontrar mejor run
        best_run = find_best_run(runs)
        
        # Verificar
        assert best_run is not None
        assert best_run["final_equity"] == 1300.0
        assert best_run["trades_count"] == 8
    
    def test_find_best_run_empty(self):
        """Test que maneja lista vacía correctamente"""
        best_run = find_best_run([])
        assert best_run is None
    
    def test_calculate_kpis_basic(self):
        """Test cálculo básico de KPIs"""
        # Datos de prueba
        runs_data = [
            {
                "final_equity": 1200.0,
                "final_balance": 1150.0,
                "initial_balance": 1000.0,
                "trades_count": 5,
                "steps": 1000,
                "run_result": "END_OF_HISTORY"
            },
            {
                "final_equity": 1100.0,
                "final_balance": 1050.0,
                "initial_balance": 1000.0,
                "trades_count": 3,
                "steps": 800,
                "run_result": "BANKRUPTCY"
            },
            {
                "final_equity": 1300.0,
                "final_balance": 1250.0,
                "initial_balance": 1000.0,
                "trades_count": 8,
                "steps": 1200,
                "run_result": "END_OF_HISTORY"
            }
        ]
        
        # Crear archivo
        self._create_test_runs(runs_data)
        
        # Cargar runs
        runs = load_runs(self.runs_file)
        
        # Calcular KPIs
        kpis = self.monitor._calculate_kpis(runs)
        
        # Verificar KPIs básicos
        assert kpis['total_runs'] == 3
        assert kpis['best_equity'] == 1300.0
        assert kpis['worst_equity'] == 1100.0
        assert kpis['avg_equity'] == 1200.0
        assert kpis['best_balance'] == 1250.0
        assert kpis['worst_balance'] == 1050.0
        
        # Verificar KPIs profesionales
        assert kpis['avg_roi'] > 0  # ROI promedio positivo
        assert kpis['max_drawdown'] >= 0  # Drawdown no negativo
        assert kpis['win_rate'] >= 0  # Win rate entre 0 y 100
        assert kpis['win_rate'] <= 100
        assert kpis['profit_factor'] >= 0  # Profit factor no negativo
        assert kpis['sharpe_ratio'] >= 0  # Sharpe ratio no negativo
        assert kpis['avg_trades'] > 0  # Trades promedio positivo
    
    def test_calculate_kpis_bankruptcy_detection(self):
        """Test detección de bancarrotas en KPIs"""
        # Datos de prueba con bancarrotas
        runs_data = [
            {
                "final_equity": 1200.0,
                "final_balance": 1150.0,
                "initial_balance": 1000.0,
                "trades_count": 5,
                "steps": 1000,
                "run_result": "END_OF_HISTORY"
            },
            {
                "final_equity": 800.0,
                "final_balance": 750.0,
                "initial_balance": 1000.0,
                "trades_count": 3,
                "steps": 800,
                "run_result": "BANKRUPTCY"
            },
            {
                "final_equity": 900.0,
                "final_balance": 850.0,
                "initial_balance": 1000.0,
                "trades_count": 2,
                "steps": 600,
                "run_result": "SOFT_RESET"
            }
        ]
        
        # Crear archivo
        self._create_test_runs(runs_data)
        
        # Cargar runs
        runs = load_runs(self.runs_file)
        
        # Calcular KPIs
        kpis = self.monitor._calculate_kpis(runs)
        
        # Verificar detección de bancarrotas
        assert kpis['bankruptcy_count'] == 1  # Solo BANKRUPTCY
        assert kpis['reset_count'] == 1  # SOFT_RESET
    
    def test_calculate_kpis_roi_calculation(self):
        """Test cálculo de ROI en KPIs"""
        # Datos de prueba con ROI conocido
        runs_data = [
            {
                "final_equity": 1200.0,
                "final_balance": 1200.0,  # ROI = 20%
                "initial_balance": 1000.0,
                "trades_count": 5,
                "steps": 1000,
                "run_result": "END_OF_HISTORY"
            },
            {
                "final_equity": 900.0,
                "final_balance": 900.0,   # ROI = -10%
                "initial_balance": 1000.0,
                "trades_count": 3,
                "steps": 800,
                "run_result": "END_OF_HISTORY"
            }
        ]
        
        # Crear archivo
        self._create_test_runs(runs_data)
        
        # Cargar runs
        runs = load_runs(self.runs_file)
        
        # Calcular KPIs
        kpis = self.monitor._calculate_kpis(runs)
        
        # Verificar ROI
        assert abs(kpis['avg_roi'] - 5.0) < 0.1  # (20% + (-10%)) / 2 = 5%
        assert kpis['best_roi'] == 20.0
        assert kpis['win_rate'] == 50.0  # 1 de 2 runs ganadores
    
    def test_calculate_kpis_profit_factor(self):
        """Test cálculo de profit factor"""
        # Datos de prueba para profit factor
        runs_data = [
            {
                "final_equity": 1200.0,
                "final_balance": 1200.0,  # ROI = 20%
                "initial_balance": 1000.0,
                "trades_count": 5,
                "steps": 1000,
                "run_result": "END_OF_HISTORY"
            },
            {
                "final_equity": 1100.0,
                "final_balance": 1100.0,  # ROI = 10%
                "initial_balance": 1000.0,
                "trades_count": 3,
                "steps": 800,
                "run_result": "END_OF_HISTORY"
            },
            {
                "final_equity": 900.0,
                "final_balance": 900.0,   # ROI = -10%
                "initial_balance": 1000.0,
                "trades_count": 2,
                "steps": 600,
                "run_result": "END_OF_HISTORY"
            }
        ]
        
        # Crear archivo
        self._create_test_runs(runs_data)
        
        # Cargar runs
        runs = load_runs(self.runs_file)
        
        # Calcular KPIs
        kpis = self.monitor._calculate_kpis(runs)
        
        # Verificar profit factor
        # Gross profit = 20% + 10% = 30%
        # Gross loss = 10%
        # Profit factor = 30% / 10% = 3.0
        assert abs(kpis['profit_factor'] - 3.0) < 0.1
    
    def test_calculate_kpis_drawdown(self):
        """Test cálculo de drawdown máximo"""
        # Datos de prueba para drawdown
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
                "final_equity": 1100.0,
                "final_balance": 1100.0,
                "initial_balance": 1000.0,
                "trades_count": 3,
                "steps": 800,
                "run_result": "END_OF_HISTORY"
            },
            {
                "final_equity": 800.0,   # Drawdown desde 1200 = 33.33%
                "final_balance": 800.0,
                "initial_balance": 1000.0,
                "trades_count": 2,
                "steps": 600,
                "run_result": "END_OF_HISTORY"
            }
        ]
        
        # Crear archivo
        self._create_test_runs(runs_data)
        
        # Cargar runs
        runs = load_runs(self.runs_file)
        
        # Calcular KPIs
        kpis = self.monitor._calculate_kpis(runs)
        
        # Verificar drawdown
        # Peak = 1200, lowest = 800
        # Drawdown = (1200 - 800) / 1200 * 100 = 33.33%
        assert abs(kpis['max_drawdown'] - 33.33) < 0.1
    
    def test_calculate_kpis_sharpe_ratio(self):
        """Test cálculo de Sharpe ratio"""
        # Datos de prueba para Sharpe ratio
        runs_data = [
            {
                "final_equity": 1200.0,
                "final_balance": 1200.0,  # ROI = 20%
                "initial_balance": 1000.0,
                "trades_count": 5,
                "steps": 1000,
                "run_result": "END_OF_HISTORY"
            },
            {
                "final_equity": 1100.0,
                "final_balance": 1100.0,  # ROI = 10%
                "initial_balance": 1000.0,
                "trades_count": 3,
                "steps": 800,
                "run_result": "END_OF_HISTORY"
            },
            {
                "final_equity": 1000.0,
                "final_balance": 1000.0,  # ROI = 0%
                "initial_balance": 1000.0,
                "trades_count": 2,
                "steps": 600,
                "run_result": "END_OF_HISTORY"
            }
        ]
        
        # Crear archivo
        self._create_test_runs(runs_data)
        
        # Cargar runs
        runs = load_runs(self.runs_file)
        
        # Calcular KPIs
        kpis = self.monitor._calculate_kpis(runs)
        
        # Verificar Sharpe ratio
        # ROI promedio = 10%, desviación estándar > 0
        # Sharpe ratio = 10% / std > 0
        assert kpis['sharpe_ratio'] >= 0
    
    def test_calculate_kpis_empty_runs(self):
        """Test que maneja lista vacía correctamente"""
        kpis = self.monitor._calculate_kpis([])
        assert kpis == {}
    
    def test_load_runs_robustness(self):
        """Test robustez del parser con datos reales"""
        # Datos más realistas
        runs_data = [
            {
                "final_equity": 1250.0,
                "final_balance": 1200.0,
                "initial_balance": 1000.0,
                "target_balance": 1000000.0,
                "trades_count": 15,
                "steps": 2500,
                "run_result": "END_OF_HISTORY",
                "ts_start": 1640995000000,
                "ts_end": 1640995200000,
                "cumulative_reward": 125.5,
                "reasons_counter": {
                    "OPEN": 8,
                    "CLOSE": 7,
                    "TP_HIT": 3,
                    "SL_HIT": 2
                },
                "bankruptcy": False,
                "segment_id": 1
            },
            {
                "final_equity": 950.0,
                "final_balance": 900.0,
                "initial_balance": 1000.0,
                "target_balance": 1000000.0,
                "trades_count": 8,
                "steps": 1200,
                "run_result": "BANKRUPTCY",
                "ts_start": 1640995200000,
                "ts_end": 1640995300000,
                "cumulative_reward": -45.2,
                "reasons_counter": {
                    "OPEN": 4,
                    "CLOSE": 4,
                    "SL_HIT": 4
                },
                "bankruptcy": True,
                "segment_id": 1
            }
        ]
        
        # Crear archivo
        self._create_test_runs(runs_data)
        
        # Cargar runs
        runs = load_runs(self.runs_file)
        
        # Verificar
        assert len(runs) == 2
        assert runs[0]["trades_count"] == 15
        assert runs[1]["bankruptcy"] is True
        assert "reasons_counter" in runs[0]
        assert runs[0]["reasons_counter"]["OPEN"] == 8
        
        # Calcular KPIs
        kpis = self.monitor._calculate_kpis(runs)
        
        # Verificar KPIs
        assert kpis['total_runs'] == 2
        assert kpis['bankruptcy_count'] == 1
        assert kpis['avg_trades'] == 11.5  # (15 + 8) / 2


if __name__ == "__main__":
    pytest.main([__file__])
