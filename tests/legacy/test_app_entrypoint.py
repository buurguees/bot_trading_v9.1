# tests/test_app_entrypoint.py
"""
Test para validar que app.py funciona como director de orquesta
y carga correctamente todas las configuraciones YAML.
"""

import pytest
import yaml
import tempfile
import os
from pathlib import Path
from unittest.mock import patch, MagicMock

# Importar el módulo app
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from app import ConfigOrchestrator


class TestConfigOrchestrator:
    """Test del director de orquesta de configuraciones"""
    
    def setup_method(self):
        """Setup para cada test"""
        self.orchestrator = ConfigOrchestrator()
    
    def test_load_all_configs_success(self):
        """Test que carga todas las configuraciones correctamente"""
        # Mock de config_loader
        with patch('app.config_loader') as mock_loader:
            # Mock de símbolos
            mock_symbol = MagicMock()
            mock_symbol.symbol = "BTCUSDT"
            mock_symbol.market = "spot"
            mock_symbol.leverage = None
            mock_symbol.allow_shorts = True
            mock_symbol.filters = {"minNotional": 1.0, "lotStep": 0.0001}
            mock_symbol.enabled_tfs = ["1m", "5m", "15m", "1h"]
            mock_loader.load_symbols.return_value = [mock_symbol]
            
            # Mock de archivos YAML
            with patch('builtins.open', create=True) as mock_open:
                mock_open.return_value.__enter__.return_value.read.return_value = "test: data"
                
                configs = self.orchestrator.load_all_configs()
                
                # Verificar que se cargaron todas las configuraciones
                assert "symbols" in configs
                assert "train" in configs
                assert "risk" in configs
                assert "rewards" in configs
                assert "hierarchical" in configs
                assert "fees" in configs
    
    def test_get_symbols_for_training(self):
        """Test que obtiene símbolos para entrenamiento correctamente"""
        # Mock de símbolos
        mock_symbol = MagicMock()
        mock_symbol.symbol = "BTCUSDT"
        mock_symbol.market = "spot"
        mock_symbol.leverage = None
        mock_symbol.allow_shorts = True
        mock_symbol.filters = {"minNotional": 1.0, "lotStep": 0.0001}
        mock_symbol.enabled_tfs = ["1m", "5m", "15m", "1h"]
        
        self.orchestrator.symbols_config = [mock_symbol]
        
        symbols = self.orchestrator.get_symbols_for_training()
        
        assert len(symbols) == 1
        assert symbols[0]["symbol"] == "BTCUSDT"
        assert symbols[0]["mode"] == "train_spot"
        assert symbols[0]["market"] == "spot"
        assert symbols[0]["allow_shorts"] == True
    
    def test_get_symbols_for_training_futures(self):
        """Test que maneja correctamente símbolos de futuros"""
        # Mock de símbolo de futuros
        mock_leverage = MagicMock()
        mock_leverage.model_dump.return_value = {"min": 1.0, "max": 5.0, "step": 1.0}
        
        mock_symbol = MagicMock()
        mock_symbol.symbol = "BTCUSDT"
        mock_symbol.market = "futures"
        mock_symbol.leverage = mock_leverage
        mock_symbol.allow_shorts = True
        mock_symbol.filters = {"minNotional": 1.0, "lotStep": 0.0001}
        mock_symbol.enabled_tfs = ["1m", "5m", "15m", "1h"]
        
        self.orchestrator.symbols_config = [mock_symbol]
        
        symbols = self.orchestrator.get_symbols_for_training()
        
        assert len(symbols) == 1
        assert symbols[0]["symbol"] == "BTCUSDT"
        assert symbols[0]["mode"] == "train_futures"
        assert symbols[0]["market"] == "futures"
        assert symbols[0]["leverage"] == {"min": 1.0, "max": 5.0, "step": 1.0}
    
    def test_print_config_summary(self, capsys):
        """Test que imprime resumen de configuración correctamente"""
        # Mock de configuraciones
        mock_symbol = MagicMock()
        mock_symbol.symbol = "BTCUSDT"
        mock_symbol.market = "spot"
        mock_symbol.leverage = None
        mock_symbol.allow_shorts = True
        mock_symbol.filters = {"minNotional": 1.0, "lotStep": 0.0001}
        mock_symbol.enabled_tfs = ["1m", "5m", "15m", "1h"]
        
        self.orchestrator.symbols_config = [mock_symbol]
        self.orchestrator.train_config = {
            "ppo": {"total_timesteps": 50000000},
            "env": {
                "n_envs": 4,
                "episode_length": 365,
                "warmup_bars": 2000,
                "antifreeze": {"enabled": False},
                "chronological": True,
                "initial_balance": 1000.0,
                "target_balance": 1000000.0
            },
            "data": {
                "months_back": 60,
                "tfs": ["1m", "5m", "15m", "1h"],
                "stage": "aligned"
            }
        }
        self.orchestrator.risk_config = {
            "common": {
                "bankruptcy": {
                    "mode": "end",
                    "threshold_pct": 20.0
                },
                "train_force_min_notional": True
            },
            "spot": {"risk_pct_per_trade": 2.0},
            "futures": {"risk_pct_per_trade": 2.0}
        }
        self.orchestrator.hierarchical_config = {
            "gating": {"min_confidence": 0.0},
            "layers": {
                "execute_tfs": ["1m"],
                "confirm_tfs": []
            }
        }
        
        self.orchestrator.print_config_summary()
        
        captured = capsys.readouterr()
        assert "RESUMEN DE CONFIGURACIÓN DEL SISTEMA" in captured.out
        assert "BTCUSDT" in captured.out
        assert "train_spot" in captured.out
        assert "50000000" in captured.out
        assert "end" in captured.out


class TestAppIntegration:
    """Test de integración del app.py"""
    
    def test_config_command(self, capsys):
        """Test del comando config"""
        from app import config
        
        with patch('app.ConfigOrchestrator') as mock_orchestrator_class:
            mock_orchestrator = MagicMock()
            mock_orchestrator_class.return_value = mock_orchestrator
            
            # Mock de configuraciones
            mock_symbol = MagicMock()
            mock_symbol.symbol = "BTCUSDT"
            mock_symbol.market = "spot"
            mock_symbol.leverage = None
            mock_symbol.allow_shorts = True
            mock_symbol.filters = {"minNotional": 1.0, "lotStep": 0.0001}
            mock_symbol.enabled_tfs = ["1m", "5m", "15m", "1h"]
            
            mock_orchestrator.symbols_config = [mock_symbol]
            mock_orchestrator.train_config = {
                "ppo": {"total_timesteps": 50000000},
                "env": {
                    "n_envs": 4,
                    "episode_length": 365,
                    "warmup_bars": 2000,
                    "antifreeze": {"enabled": False},
                    "chronological": True,
                    "initial_balance": 1000.0,
                    "target_balance": 1000000.0
                },
                "data": {
                    "months_back": 60,
                    "tfs": ["1m", "5m", "15m", "1h"],
                    "stage": "aligned"
                }
            }
            mock_orchestrator.risk_config = {
                "common": {
                    "bankruptcy": {
                        "mode": "end",
                        "threshold_pct": 20.0
                    },
                    "train_force_min_notional": True
                },
                "spot": {"risk_pct_per_trade": 2.0},
                "futures": {"risk_pct_per_trade": 2.0}
            }
            mock_orchestrator.hierarchical_config = {
                "gating": {"min_confidence": 0.0},
                "layers": {
                    "execute_tfs": ["1m"],
                    "confirm_tfs": []
                }
            }
            
            config()
            
            captured = capsys.readouterr()
            assert "RESUMEN DE CONFIGURACIÓN DEL SISTEMA" in captured.out


if __name__ == "__main__":
    pytest.main([__file__])
