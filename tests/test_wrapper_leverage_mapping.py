# tests/test_wrapper_leverage_mapping.py
"""
Test para validar el mapeo de leverage en el gym wrapper para futuros.
"""

import pytest
import numpy as np
from unittest.mock import MagicMock, patch
import sys
from pathlib import Path

# Agregar el directorio raíz al path
sys.path.insert(0, str(Path(__file__).parent.parent))

from train_env.gym_wrapper import TradingGymWrapper


class TestWrapperLeverageMapping:
    """Test del mapeo de leverage en el gym wrapper"""
    
    def setup_method(self):
        """Setup para cada test"""
        # Mock del entorno base
        self.mock_base_env = MagicMock()
        self.mock_base_env.reset.return_value = {
            "tfs": {"1h": {"close": 50000.0}},
            "features": {"1h": {"ema20": 49000.0}},
            "position": {"side": 0, "qty": 0.0},
            "analysis": {"confidence": 0.5}
        }
        self.mock_base_env.step.return_value = (
            {"tfs": {"1h": {"close": 51000.0}}},
            0.1,
            False,
            {"events": []}
        )
        
        # Configuración de leverage
        self.leverage_spec = {
            "min": 1.0,
            "max": 5.0,
            "step": 1.0,
            "default": 2.0
        }
        
        # Crear wrapper
        self.wrapper = TradingGymWrapper(
            base_env=self.mock_base_env,
            reward_yaml="config/rewards.yaml",
            tfs=["1h"],
            leverage_spec=self.leverage_spec
        )
    
    def test_leverage_spec_initialization(self):
        """Test que el leverage spec se inicializa correctamente"""
        assert self.wrapper._lev_spec is not None
        mn, mx, st, n = self.wrapper._lev_spec
        assert mn == 1.0
        assert mx == 5.0
        assert st == 1.0
        assert n == 5  # (5.0 - 1.0) / 1.0 + 1 = 5 niveles
    
    def test_action_space_multidiscrete(self):
        """Test que el action space es MultiDiscrete para futuros"""
        assert hasattr(self.wrapper.action_space, 'nvec')
        assert len(self.wrapper.action_space.nvec) == 2
        assert self.wrapper.action_space.nvec[0] == 5  # trade actions
        assert self.wrapper.action_space.nvec[1] == 5  # leverage levels
    
    def test_action_space_discrete_spot(self):
        """Test que el action space es Discrete para spot"""
        wrapper_spot = TradingGymWrapper(
            base_env=self.mock_base_env,
            reward_yaml="config/rewards.yaml",
            tfs=["1h"],
            leverage_spec=None  # Sin leverage = spot
        )
        
        assert hasattr(wrapper_spot.action_space, 'n')
        assert wrapper_spot.action_space.n == 5  # Solo trade actions
    
    def test_lev_from_idx_mapping(self):
        """Test que el mapeo de índice a leverage funciona correctamente"""
        # Test índices válidos
        assert self.wrapper._lev_from_idx(0) == 1.0  # min
        assert self.wrapper._lev_from_idx(1) == 2.0
        assert self.wrapper._lev_from_idx(2) == 3.0
        assert self.wrapper._lev_from_idx(3) == 4.0
        assert self.wrapper._lev_from_idx(4) == 5.0  # max
    
    def test_lev_from_idx_edge_cases(self):
        """Test que el mapeo maneja casos edge correctamente"""
        # Test índice fuera de rango (debería ajustarse)
        with patch('builtins.print') as mock_print:
            # Índice negativo
            result = self.wrapper._lev_from_idx(-1)
            assert result == 1.0  # Ajustado a min
            mock_print.assert_called()
            
            # Índice mayor al máximo
            result = self.wrapper._lev_from_idx(10)
            assert result == 5.0  # Ajustado a max
            mock_print.assert_called()
    
    def test_step_with_leverage_action(self):
        """Test que step procesa acciones con leverage correctamente"""
        # Acción: [trade_action=2, leverage_idx=3] -> leverage=4.0x
        action = np.array([2, 3])
        
        # Mock set_action_override
        self.mock_base_env.set_action_override = MagicMock()
        
        # Ejecutar step
        obs, reward, done, truncated, info = self.wrapper.step(action)
        
        # Verificar que se llamó set_action_override con los parámetros correctos
        self.mock_base_env.set_action_override.assert_called_once_with(
            int(2),  # trade_action
            leverage_override=4.0,  # leverage calculado
            leverage_index=3  # leverage_idx
        )
        
        # Verificar que se llamó step del entorno base
        self.mock_base_env.step.assert_called_once()
    
    def test_step_with_spot_action(self):
        """Test que step procesa acciones de spot correctamente"""
        # Crear wrapper para spot (sin leverage)
        wrapper_spot = TradingGymWrapper(
            base_env=self.mock_base_env,
            reward_yaml="config/rewards.yaml",
            tfs=["1h"],
            leverage_spec=None
        )
        
        # Acción simple: trade_action=1
        action = 1
        
        # Mock set_action_override
        self.mock_base_env.set_action_override = MagicMock()
        
        # Ejecutar step
        obs, reward, done, truncated, info = wrapper_spot.step(action)
        
        # Verificar que se llamó set_action_override sin leverage
        self.mock_base_env.set_action_override.assert_called_once_with(
            int(1),  # trade_action
            leverage_override=None,  # Sin leverage
            leverage_index=None  # Sin leverage_idx
        )
    
    def test_step_leverage_validation(self):
        """Test que step valida límites de leverage"""
        # Acción con leverage_idx fuera de rango
        action = np.array([1, 10])  # leverage_idx=10 (fuera de rango)
        
        # Mock set_action_override
        self.mock_base_env.set_action_override = MagicMock()
        
        # Mock print para capturar el warning
        with patch('builtins.print') as mock_print:
            # Ejecutar step
            obs, reward, done, truncated, info = self.wrapper.step(action)
            
            # Verificar que se imprimió el warning
            mock_print.assert_called_with("⚠️ Leverage index fuera de rango, ajustado a 4 → 5.0x")
            
            # Verificar que se ajustó el leverage
            self.mock_base_env.set_action_override.assert_called_once_with(
                int(1),  # trade_action
                leverage_override=5.0,  # leverage ajustado a max
                leverage_index=4  # leverage_idx ajustado
            )
    
    def test_curriculum_learning_integration(self):
        """Test que el curriculum learning se integra correctamente"""
        # Mock del curriculum
        mock_curriculum = MagicMock()
        mock_curriculum.suggest_action_modification.return_value = 3
        
        # Asignar curriculum al wrapper
        self.wrapper.curriculum = mock_curriculum
        
        # Acción original
        action = np.array([1, 2])
        
        # Mock set_action_override
        self.mock_base_env.set_action_override = MagicMock()
        
        # Mock random para que siempre active curriculum (5% chance)
        with patch('random.random', return_value=0.01):  # < 0.05
            # Ejecutar step
            obs, reward, done, truncated, info = self.wrapper.step(action)
            
            # Verificar que se llamó el curriculum
            mock_curriculum.suggest_action_modification.assert_called_once()
            
            # Verificar que se usó la acción modificada
            self.mock_base_env.set_action_override.assert_called_once_with(
                int(3),  # acción modificada por curriculum
                leverage_override=3.0,  # leverage original
                leverage_index=2  # leverage_idx original
            )
    
    def test_observation_flattening(self):
        """Test que las observaciones se aplanan correctamente"""
        # Mock observación compleja
        complex_obs = {
            "tfs": {
                "1h": {"close": 50000.0}
            },
            "features": {
                "1h": {
                    "ema20": 49000.0,
                    "ema50": 48000.0,
                    "rsi14": 60.0,
                    "atr14": 1000.0,
                    "macd_hist": 50.0,
                    "bb_p": 0.7
                }
            },
            "position": {
                "side": 1,
                "qty": 0.1,
                "entry_price": 49000.0,
                "unrealized_pnl": 100.0
            },
            "analysis": {
                "confidence": 0.8,
                "side_hint": 1
            }
        }
        
        # Aplanar observación
        flattened = self.wrapper._flatten_obs(complex_obs)
        
        # Verificar dimensiones
        expected_dim = len(self.wrapper.tfs) * 7 + 4 + 2  # per_tf=7, pos=4, ana=2
        assert len(flattened) == expected_dim
        
        # Verificar algunos valores específicos
        assert flattened[0] == 50000.0  # close
        assert flattened[1] == 49000.0  # ema20
        assert flattened[2] == 48000.0  # ema50
        assert flattened[3] == 60.0     # rsi14
        assert flattened[4] == 1000.0   # atr14
        assert flattened[5] == 50.0     # macd_hist
        assert flattened[6] == 0.7      # bb_p
        
        # Verificar posición
        assert flattened[7] == 1.0      # side
        assert flattened[8] == 0.1      # qty
        assert flattened[9] == 49000.0  # entry_price
        assert flattened[10] == 100.0   # unrealized_pnl
        
        # Verificar análisis
        assert flattened[11] == 0.8     # confidence
        assert flattened[12] == 1.0     # side_hint
    
    def test_observation_space_dimensions(self):
        """Test que el observation space tiene las dimensiones correctas"""
        expected_dim = len(self.wrapper.tfs) * 7 + 4 + 2
        assert self.wrapper.observation_space.shape[0] == expected_dim
    
    def test_reset_functionality(self):
        """Test que reset funciona correctamente"""
        # Ejecutar reset
        obs, info = self.wrapper.reset()
        
        # Verificar que se llamó reset del entorno base
        self.mock_base_env.reset.assert_called_once()
        
        # Verificar que se devolvió observación aplanada
        assert isinstance(obs, np.ndarray)
        assert len(obs) == self.wrapper.observation_space.shape[0]
        assert info == {}


if __name__ == "__main__":
    pytest.main([__file__])
