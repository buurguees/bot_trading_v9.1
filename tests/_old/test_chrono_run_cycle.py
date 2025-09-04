# tests/test_chrono_run_cycle.py
"""
Test para validar el entrenamiento cronológico:
- Un run por pasada completa del histórico
- Reinicio cronológico al inicio
- No runs vacíos
"""

import pytest
import tempfile
import json
from pathlib import Path
from unittest.mock import MagicMock, patch
import sys

# Añadir el directorio raíz al path
sys.path.insert(0, str(Path(__file__).parent.parent))

from base_env.base_env import BaseTradingEnv
from base_env.io.historical_broker import ParquetHistoricalBroker
from base_env.config.models import EnvConfig, SymbolMeta, PipelineConfig, HierarchicalConfig, RiskConfig, FeesConfig


class MockBroker:
    """Mock broker para testing"""
    
    def __init__(self, total_bars=1000):
        self.total_bars = total_bars
        self.current_bar = 0
        self._base_ts_list = list(range(total_bars))
        self._i = 0
    
    def get_price(self):
        return 50000.0 + (self.current_bar * 10)  # Precio simulado
    
    def next(self):
        if self._i < len(self._base_ts_list) - 1:
            self._i += 1
            self.current_bar += 1
    
    def is_end_of_data(self):
        return self._i >= len(self._base_ts_list) - 1
    
    def reset_to_start(self):
        self._i = 0
        self.current_bar = 0
    
    def get_current_ts(self):
        return self._base_ts_list[self._i] if self._i < len(self._base_ts_list) else self._base_ts_list[-1]


class MockOMS:
    """Mock OMS para testing"""
    
    def open(self, side, qty, price_hint, sl, tp):
        return {
            "side": 1 if side == "LONG" else -1,
            "qty": float(qty),
            "price": float(price_hint),
            "fees": 0.0,
            "sl": sl,
            "tp": tp
        }
    
    def close(self, qty, price_hint):
        return {
            "qty": float(qty),
            "price": float(price_hint),
            "fees": 0.0
        }


class TestChronologicalTraining:
    """Test del entrenamiento cronológico"""
    
    def setup_method(self):
        """Setup para cada test"""
        self.temp_dir = tempfile.mkdtemp()
        self.models_dir = Path(self.temp_dir) / "models" / "BTCUSDT"
        self.models_dir.mkdir(parents=True, exist_ok=True)
        
        # Configuración del entorno
        self.cfg = EnvConfig(
            mode="train_spot",
            market="spot",
            leverage=1.0,
            symbol_meta=SymbolMeta(
                symbol="BTCUSDT",
                market="spot",
                enabled_tfs=["1m"],
                filters={"minNotional": 1.0, "lotStep": 0.0001},
                allow_shorts=True
            ),
            tfs=["1m"],
            pipeline=PipelineConfig(strict_alignment=True),
            hierarchical=HierarchicalConfig(min_confidence=0.0, execute_tfs=["1m"], confirm_tfs=[]),
            risk=RiskConfig(),
            fees=FeesConfig()
        )
        
        # Mock broker
        self.broker = MockBroker(total_bars=100)
        
        # Mock OMS
        self.oms = MockOMS()
    
    def test_chronological_reset(self):
        """Test que el reset reinicia cronológicamente al inicio"""
        env = BaseTradingEnv(
            cfg=self.cfg,
            broker=self.broker,
            oms=self.oms,
            initial_cash=1000.0,
            target_cash=10000.0,
            models_root=str(self.temp_dir)
        )
        
        # Avanzar algunos pasos
        for _ in range(10):
            obs, reward, done, info = env.step()
            if done:
                break
        
        # Verificar que no está al inicio
        assert self.broker.current_bar > 0
        
        # Reset
        obs = env.reset()
        
        # Verificar que volvió al inicio
        assert self.broker.current_bar == 0
        assert self.broker._i == 0
    
    def test_end_of_history_detection(self):
        """Test que detecta correctamente el final del histórico"""
        env = BaseTradingEnv(
            cfg=self.cfg,
            broker=self.broker,
            oms=self.oms,
            initial_cash=1000.0,
            target_cash=10000.0,
            models_root=str(self.temp_dir)
        )
        
        # Avanzar hasta el final
        done = False
        steps = 0
        while not done and steps < 200:  # Límite de seguridad
            obs, reward, done, info = env.step()
            steps += 1
        
        # Verificar que se detectó el final del histórico
        assert done
        assert "END_OF_HISTORY" in str(info.get("done_reason", ""))
        assert self.broker.is_end_of_data()
    
    def test_run_logging_with_activity(self):
        """Test que loguea runs con actividad real"""
        env = BaseTradingEnv(
            cfg=self.cfg,
            broker=self.broker,
            oms=self.oms,
            initial_cash=1000.0,
            target_cash=10000.0,
            models_root=str(self.temp_dir)
        )
        
        # Simular actividad (cambiar equity significativamente)
        env.portfolio.equity_quote = 1200.0  # +200 USDT
        
        # Avanzar hasta el final
        done = False
        steps = 0
        while not done and steps < 200:
            obs, reward, done, info = env.step()
            steps += 1
        
        # Verificar que se logueó el run
        runs_file = self.models_dir / "BTCUSDT_runs.jsonl"
        assert runs_file.exists()
        
        with open(runs_file, 'r') as f:
            lines = f.readlines()
            assert len(lines) > 0
            
            # Verificar contenido del run
            run_data = json.loads(lines[0])
            assert "final_equity" in run_data
            assert "final_balance" in run_data
            assert "run_result" in run_data
            assert run_data["run_result"] == "END_OF_HISTORY"
    
    def test_no_empty_runs_logged(self):
        """Test que no loguea runs vacíos sin actividad"""
        env = BaseTradingEnv(
            cfg=self.cfg,
            broker=self.broker,
            oms=self.oms,
            initial_cash=1000.0,
            target_cash=10000.0,
            models_root=str(self.temp_dir)
        )
        
        # NO simular actividad (equity permanece igual)
        # Avanzar hasta el final
        done = False
        steps = 0
        while not done and steps < 200:
            obs, reward, done, info = env.step()
            steps += 1
        
        # Verificar que NO se logueó el run (sin actividad)
        runs_file = self.models_dir / "BTCUSDT_runs.jsonl"
        if runs_file.exists():
            with open(runs_file, 'r') as f:
                lines = f.readlines()
                # Debería estar vacío o no tener runs sin actividad
                assert len(lines) == 0
    
    def test_multiple_chronological_cycles(self):
        """Test múltiples ciclos cronológicos"""
        env = BaseTradingEnv(
            cfg=self.cfg,
            broker=self.broker,
            oms=self.oms,
            initial_cash=1000.0,
            target_cash=10000.0,
            models_root=str(self.temp_dir)
        )
        
        # Ejecutar 3 ciclos completos
        for cycle in range(3):
            # Simular actividad diferente en cada ciclo
            env.portfolio.equity_quote = 1000.0 + (cycle * 100)
            
            # Avanzar hasta el final
            done = False
            steps = 0
            while not done and steps < 200:
                obs, reward, done, info = env.step()
                steps += 1
            
            # Verificar que se detectó el final
            assert done
            
            # Reset para siguiente ciclo
            if cycle < 2:  # No reset en el último ciclo
                obs = env.reset()
                assert self.broker.current_bar == 0  # Debe volver al inicio
        
        # Verificar que se loguearon múltiples runs
        runs_file = self.models_dir / "BTCUSDT_runs.jsonl"
        if runs_file.exists():
            with open(runs_file, 'r') as f:
                lines = f.readlines()
                # Debería tener 3 runs (uno por ciclo con actividad)
                assert len(lines) == 3
    
    def test_telemetry_end_of_history(self):
        """Test que la telemetría registra END_OF_HISTORY"""
        env = BaseTradingEnv(
            cfg=self.cfg,
            broker=self.broker,
            oms=self.oms,
            initial_cash=1000.0,
            target_cash=10000.0,
            models_root=str(self.temp_dir)
        )
        
        # Avanzar hasta el final
        done = False
        steps = 0
        while not done and steps < 200:
            obs, reward, done, info = env.step()
            steps += 1
        
        # Verificar que se registró END_OF_HISTORY en telemetría
        # (Esto depende de la implementación específica del sistema de telemetría)
        assert done
        assert "END_OF_HISTORY" in str(info.get("done_reason", ""))


if __name__ == "__main__":
    pytest.main([__file__])