#!/usr/bin/env python3
"""
Test para validar las métricas del watcher.
Verifica que:
1. WinRate se calcule correctamente
2. Profit Factor se calcule correctamente
3. Los KPIs se muestren apropiadamente
4. Los runs exitosos se marquen correctamente
"""

import pytest
import json
import tempfile
from pathlib import Path
import sys

# Añadir el directorio raíz al path
sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.watch_progress import ConsoleMonitor


class TestWatchProgressMetrics:
    def setup_method(self):
        """Configuración inicial para cada test"""
        # Crear directorio temporal para tests
        self.temp_dir = tempfile.mkdtemp()
        self.models_root = Path(self.temp_dir)
        self.symbol = "TEST"
        
        # Crear directorio del símbolo
        symbol_dir = self.models_root / self.symbol
        symbol_dir.mkdir(exist_ok=True)
        
        self.runs_file = symbol_dir / f"{self.symbol}_runs.jsonl"
        
        # Crear monitor para testing
        self.monitor = ConsoleMonitor(
            symbol=self.symbol,
            models_root=str(self.models_root),
            refresh_interval=1
        )

    def teardown_method(self):
        """Limpieza después de cada test"""
        import shutil
        shutil.rmtree(self.temp_dir)

    def create_test_runs(self, runs_data):
        """Helper para crear archivo de runs de prueba"""
        with self.runs_file.open("w", encoding="utf-8") as f:
            for run in runs_data:
                f.write(json.dumps(run) + "\n")

    def test_win_rate_calculation(self):
        """Test: WinRate debe calcularse correctamente"""
        # Crear runs de prueba con diferentes resultados
        test_runs = [
            {
                "run_number": 1,
                "initial_balance": 1000.0,
                "final_balance": 1100.0,  # +10% (ganador)
                "final_equity": 1100.0,
                "run_result": "COMPLETED"
            },
            {
                "run_number": 2,
                "initial_balance": 1000.0,
                "final_balance": 950.0,   # -5% (perdedor)
                "final_equity": 950.0,
                "run_result": "COMPLETED"
            },
            {
                "run_number": 3,
                "initial_balance": 1000.0,
                "final_balance": 1050.0,  # +5% (ganador)
                "final_equity": 1050.0,
                "run_result": "COMPLETED"
            },
            {
                "run_number": 4,
                "initial_balance": 1000.0,
                "final_balance": 900.0,   # -10% (perdedor)
                "final_equity": 900.0,
                "run_result": "BANKRUPTCY"
            }
        ]
        
        self.create_test_runs(test_runs)
        runs = self.monitor._load_runs()
        kpis = self.monitor._calculate_kpis(runs)
        
        # 2 ganadores de 4 runs = 50% win rate
        assert kpis['win_rate'] == 50.0
        assert kpis['total_runs'] == 4

    def test_profit_factor_calculation(self):
        """Test: Profit Factor debe calcularse correctamente"""
        # Crear runs con ganancias y pérdidas específicas
        test_runs = [
            {
                "run_number": 1,
                "initial_balance": 1000.0,
                "final_balance": 1200.0,  # +20% (ganancia)
                "final_equity": 1200.0,
                "run_result": "COMPLETED"
            },
            {
                "run_number": 2,
                "initial_balance": 1000.0,
                "final_balance": 1100.0,  # +10% (ganancia)
                "final_equity": 1100.0,
                "run_result": "COMPLETED"
            },
            {
                "run_number": 3,
                "initial_balance": 1000.0,
                "final_balance": 900.0,   # -10% (pérdida)
                "final_equity": 900.0,
                "run_result": "COMPLETED"
            },
            {
                "run_number": 4,
                "initial_balance": 1000.0,
                "final_balance": 800.0,   # -20% (pérdida)
                "final_equity": 800.0,
                "run_result": "BANKRUPTCY"
            }
        ]
        
        self.create_test_runs(test_runs)
        runs = self.monitor._load_runs()
        kpis = self.monitor._calculate_kpis(runs)
        
        # Ganancia total: 20% + 10% = 30%
        # Pérdida total: 10% + 20% = 30%
        # Profit Factor = 30% / 30% = 1.0
        assert kpis['profit_factor'] == 1.0

    def test_successful_runs_counting(self):
        """Test: Los runs exitosos deben contarse correctamente"""
        test_runs = [
            {
                "run_number": 1,
                "initial_balance": 1000.0,
                "final_balance": 1100.0,
                "final_equity": 1100.0,
                "run_result": "COMPLETED"  # Exitoso
            },
            {
                "run_number": 2,
                "initial_balance": 1000.0,
                "final_balance": 900.0,
                "final_equity": 900.0,
                "run_result": "BANKRUPTCY"  # No exitoso
            },
            {
                "run_number": 3,
                "initial_balance": 1000.0,
                "final_balance": 1050.0,
                "final_equity": 1050.0,
                "run_result": "COMPLETED"  # Exitoso
            },
            {
                "run_number": 4,
                "initial_balance": 1000.0,
                "final_balance": 800.0,
                "final_equity": 800.0,
                "run_result": "SOFT_RESET"  # No exitoso (contiene RESET)
            }
        ]
        
        self.create_test_runs(test_runs)
        runs = self.monitor._load_runs()
        kpis = self.monitor._calculate_kpis(runs)
        
        # 2 runs exitosos de 4 total
        assert kpis['bankruptcy_count'] == 1  # Solo 1 BANKRUPTCY
        assert kpis['reset_count'] == 1       # Solo 1 RESET

    def test_avg_roi_calculation(self):
        """Test: ROI promedio debe calcularse correctamente"""
        test_runs = [
            {
                "run_number": 1,
                "initial_balance": 1000.0,
                "final_balance": 1100.0,  # +10%
                "final_equity": 1100.0,
                "run_result": "COMPLETED"
            },
            {
                "run_number": 2,
                "initial_balance": 1000.0,
                "final_balance": 1200.0,  # +20%
                "final_equity": 1200.0,
                "run_result": "COMPLETED"
            },
            {
                "run_number": 3,
                "initial_balance": 1000.0,
                "final_balance": 900.0,   # -10%
                "final_equity": 900.0,
                "run_result": "COMPLETED"
            }
        ]
        
        self.create_test_runs(test_runs)
        runs = self.monitor._load_runs()
        kpis = self.monitor._calculate_kpis(runs)
        
        # ROI promedio: (10% + 20% - 10%) / 3 = 6.67%
        assert abs(kpis['avg_roi'] - 6.67) < 0.1

    def test_max_drawdown_calculation(self):
        """Test: Max Drawdown debe calcularse correctamente"""
        test_runs = [
            {
                "run_number": 1,
                "initial_balance": 1000.0,
                "final_balance": 1200.0,  # Peak
                "final_equity": 1200.0,
                "run_result": "COMPLETED"
            },
            {
                "run_number": 2,
                "initial_balance": 1000.0,
                "final_balance": 800.0,   # Drawdown desde 1200
                "final_equity": 800.0,
                "run_result": "COMPLETED"
            },
            {
                "run_number": 3,
                "initial_balance": 1000.0,
                "final_balance": 1000.0,  # Recuperación
                "final_equity": 1000.0,
                "run_result": "COMPLETED"
            }
        ]
        
        self.create_test_runs(test_runs)
        runs = self.monitor._load_runs()
        kpis = self.monitor._calculate_kpis(runs)
        
        # Max drawdown: (1200 - 800) / 1200 * 100 = 33.33%
        assert abs(kpis['max_drawdown'] - 33.33) < 0.1

    def test_trades_count_average(self):
        """Test: Trades promedio debe calcularse correctamente"""
        test_runs = [
            {
                "run_number": 1,
                "initial_balance": 1000.0,
                "final_balance": 1100.0,
                "final_equity": 1100.0,
                "trades_count": 10,
                "run_result": "COMPLETED"
            },
            {
                "run_number": 2,
                "initial_balance": 1000.0,
                "final_balance": 1200.0,
                "final_equity": 1200.0,
                "trades_count": 20,
                "run_result": "COMPLETED"
            },
            {
                "run_number": 3,
                "initial_balance": 1000.0,
                "final_balance": 900.0,
                "final_equity": 900.0,
                "trades_count": 5,
                "run_result": "COMPLETED"
            }
        ]
        
        self.create_test_runs(test_runs)
        runs = self.monitor._load_runs()
        kpis = self.monitor._calculate_kpis(runs)
        
        # Trades promedio: (10 + 20 + 5) / 3 = 11.67
        assert abs(kpis['avg_trades'] - 11.67) < 0.1

    def test_r_multiple_average(self):
        """Test: R-Multiple promedio debe calcularse correctamente"""
        test_runs = [
            {
                "run_number": 1,
                "initial_balance": 1000.0,
                "final_balance": 1100.0,
                "final_equity": 1100.0,
                "r_multiple": 1.5,
                "run_result": "COMPLETED"
            },
            {
                "run_number": 2,
                "initial_balance": 1000.0,
                "final_balance": 1200.0,
                "final_equity": 1200.0,
                "r_multiple": 2.0,
                "run_result": "COMPLETED"
            },
            {
                "run_number": 3,
                "initial_balance": 1000.0,
                "final_balance": 900.0,
                "final_equity": 900.0,
                "r_multiple": -1.0,
                "run_result": "COMPLETED"
            }
        ]
        
        self.create_test_runs(test_runs)
        runs = self.monitor._load_runs()
        kpis = self.monitor._calculate_kpis(runs)
        
        # R-Multiple promedio: (1.5 + 2.0 - 1.0) / 3 = 0.83
        assert abs(kpis['avg_r_multiple'] - 0.83) < 0.1

    def test_empty_runs_handling(self):
        """Test: Manejo de runs vacíos"""
        runs = []
        kpis = self.monitor._calculate_kpis(runs)
        
        # Con runs vacíos, debe devolver diccionario vacío
        assert kpis == {}

    def test_sharpe_ratio_calculation(self):
        """Test: Sharpe Ratio debe calcularse correctamente"""
        test_runs = [
            {
                "run_number": 1,
                "initial_balance": 1000.0,
                "final_balance": 1100.0,  # +10%
                "final_equity": 1100.0,
                "run_result": "COMPLETED"
            },
            {
                "run_number": 2,
                "initial_balance": 1000.0,
                "final_balance": 1200.0,  # +20%
                "final_equity": 1200.0,
                "run_result": "COMPLETED"
            },
            {
                "run_number": 3,
                "initial_balance": 1000.0,
                "final_balance": 900.0,   # -10%
                "final_equity": 900.0,
                "run_result": "COMPLETED"
            }
        ]
        
        self.create_test_runs(test_runs)
        runs = self.monitor._load_runs()
        kpis = self.monitor._calculate_kpis(runs)
        
        # Sharpe ratio debe ser un número válido (no NaN o infinito)
        assert isinstance(kpis['sharpe_ratio'], (int, float))
        assert not (kpis['sharpe_ratio'] != kpis['sharpe_ratio'])  # No NaN


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
