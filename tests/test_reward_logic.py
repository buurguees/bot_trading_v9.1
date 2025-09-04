#!/usr/bin/env python3
"""
Test para validar la lógica de recompensas del nuevo sistema.
Verifica que:
1. Los trades con TP den +1.0 reward
2. Los trades con SL den -0.5 reward  
3. El ROI proporcional funcione correctamente
4. No se use PnL instantáneo
5. Los bonus por duración de posiciones funcionen
"""

import pytest
import sys
from pathlib import Path

# Añadir el directorio raíz al path
sys.path.insert(0, str(Path(__file__).parent.parent))

from train_env.reward_shaper import RewardShaper


class TestRewardLogic:
    def setup_method(self):
        """Configuración inicial para cada test"""
        # Crear configuración de rewards para testing
        self.reward_config = {
            "tiers": {
                "pos": [[0, 100, 0.0]],  # No usar tiers en tests
                "neg": [[0, 100, 0.0]]
            },
            "bonuses": {
                "tp_hit": 0.0,  # No usar bonuses en tests
                "sl_hit": 0.0
            },
            "weights": {
                "realized_pnl": 0.0,      # ← NUEVO: No usar PnL instantáneo
                "unrealized_pnl": 0.0,    # ← NUEVO: No usar PnL no realizado
                "r_multiple": 0.0,        # Simplificar para tests
                "risk_efficiency": 0.0,   # Simplificar para tests
                "time_penalty": 0.0,      # ← NUEVO: No penalizar duración
                "trade_cost": 0.0,        # Simplificar para tests
                "dd_penalty": 0.0,        # ← NUEVO: No penalizar drawdowns
                "survival_bonus": 0.0,    # Simplificar para tests
                "progress_bonus": 0.0,    # Simplificar para tests
                "compound_bonus": 0.0,    # Simplificar para tests
                "empty_run_penalty": 0.0, # Simplificar para tests
                "balance_milestone_reward": 0.0  # Simplificar para tests
            },
            "reward_clip": [-10.0, 10.0]
        }
        
        # Crear archivo temporal de configuración
        import tempfile
        import yaml
        self.temp_config = tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False)
        yaml.dump(self.reward_config, self.temp_config)
        self.temp_config.close()
        
        self.shaper = RewardShaper(self.temp_config.name)

    def teardown_method(self):
        """Limpieza después de cada test"""
        import os
        os.unlink(self.temp_config.name)

    def test_tp_hit_reward(self):
        """Test: Trade con TP debe dar +1.0 reward"""
        obs = {
            "position": {"side": 0},  # Sin posición
            "portfolio": {"equity_quote": 1000.0}
        }
        
        events = [
            {
                "kind": "CLOSE",
                "roi_pct": 2.0,
                "r_multiple": 1.5,
                "risk_pct": 1.0
            },
            {
                "kind": "TP_HIT"
            }
        ]
        
        reward, details = self.shaper.compute(obs, 0.0, events)
        
        # Debe dar +1.0 por TP + ROI proporcional
        expected_reward = 1.0 + (2.0 * 0.2)  # +1.0 TP + 0.4 ROI
        assert reward == expected_reward
        assert details["tp_sl_reward"] == 1.0
        assert details["roi_reward"] == 0.4

    def test_sl_hit_reward(self):
        """Test: Trade con SL debe dar -0.5 reward"""
        obs = {
            "position": {"side": 0},  # Sin posición
            "portfolio": {"equity_quote": 1000.0}
        }
        
        events = [
            {
                "kind": "CLOSE",
                "roi_pct": -1.5,
                "r_multiple": -1.5,
                "risk_pct": 1.0
            },
            {
                "kind": "SL_HIT"
            }
        ]
        
        reward, details = self.shaper.compute(obs, 0.0, events)
        
        # Debe dar -0.5 por SL + ROI proporcional negativo
        expected_reward = -0.5 + (-1.5 * 0.2)  # -0.5 SL + -0.3 ROI
        assert abs(reward - expected_reward) < 0.001
        assert details["tp_sl_reward"] == -0.5
        assert abs(details["roi_reward"] - (-0.3)) < 0.001

    def test_no_pnl_instantaneous(self):
        """Test: No debe usar PnL instantáneo"""
        obs = {
            "position": {"side": 1, "unrealized_pnl": 50.0},  # Con posición y PnL
            "portfolio": {"equity_quote": 1050.0}
        }
        
        events = []  # Sin eventos de cierre
        
        reward, details = self.shaper.compute(obs, 100.0, events)  # base_reward = 100
        
        # El reward debe ser solo el bonus por duración de posición (0.05)
        # porque no hay eventos de cierre y no debe usar PnL instantáneo
        assert reward == 0.05  # Solo bonus por duración de posición
        assert "realized_usd" not in details  # No debe incluir PnL en detalles

    def test_position_duration_bonus(self):
        """Test: Bonus por duración de posiciones positivas"""
        obs = {
            "position": {"side": 1},  # Con posición
            "portfolio": {"equity_quote": 1100.0}  # Equity > initial_balance
        }
        
        events = []  # Sin eventos de cierre
        
        reward, details = self.shaper.compute(obs, 0.0, events, initial_balance=1000.0)
        
        # Debe dar +0.05 por duración de posición positiva
        assert reward == 0.05
        assert details["position_duration_bonus"] == 0.05

    def test_inactivity_penalty(self):
        """Test: Penalty por exceso de inactividad"""
        obs = {
            "position": {"side": 0},  # Sin posición
            "portfolio": {"equity_quote": 1000.0}
        }
        
        events = []  # Sin eventos de trading
        
        reward, details = self.shaper.compute(obs, 0.0, events)
        
        # Debe dar -0.01 por inactividad
        assert reward == -0.01
        assert details["inactivity_penalty"] == -0.01

    def test_roi_proportional_scaling(self):
        """Test: ROI proporcional debe escalar correctamente"""
        obs = {
            "position": {"side": 0},
            "portfolio": {"equity_quote": 1000.0}
        }
        
        # Test con diferentes ROIs
        test_cases = [
            (5.0, 1.0),    # +5% ROI = +1.0 reward
            (-3.0, -0.6),  # -3% ROI = -0.6 reward
            (0.0, 0.0),    # 0% ROI = 0.0 reward
        ]
        
        for roi_pct, expected_roi_reward in test_cases:
            events = [
                {
                    "kind": "CLOSE",
                    "roi_pct": roi_pct,
                    "r_multiple": 0.0,
                    "risk_pct": 1.0
                }
            ]
            
            reward, details = self.shaper.compute(obs, 0.0, events)
            
            assert abs(details["roi_reward"] - expected_roi_reward) < 0.001
            assert abs(reward - expected_roi_reward) < 0.001  # Solo ROI reward en este caso

    def test_no_trade_events_no_penalty(self):
        """Test: Si hay eventos de trading, no debe aplicar penalty de inactividad"""
        obs = {
            "position": {"side": 0},  # Sin posición
            "portfolio": {"equity_quote": 1000.0}
        }
        
        events = [
            {"kind": "OPEN"}  # Hay evento de trading
        ]
        
        reward, details = self.shaper.compute(obs, 0.0, events)
        
        # No debe aplicar penalty de inactividad
        assert details["inactivity_penalty"] == 0.0
        assert reward == 0.0  # Solo trade_cost que está en 0.0

    def test_reward_clipping(self):
        """Test: Los rewards deben respetar el clipping"""
        obs = {
            "position": {"side": 0},
            "portfolio": {"equity_quote": 1000.0}
        }
        
        # Test con ROI muy alto que debería ser recortado
        events = [
            {
                "kind": "CLOSE",
                "roi_pct": 1000.0,  # ROI muy alto
                "r_multiple": 0.0,
                "risk_pct": 1.0
            },
            {
                "kind": "TP_HIT"
            }
        ]
        
        reward, details = self.shaper.compute(obs, 0.0, events)
        
        # El reward debe estar dentro del rango de clipping
        assert -10.0 <= reward <= 10.0
        assert reward == self.shaper._clip(reward)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
