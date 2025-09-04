# tests/test_rewards_pipeline.py
"""
Test para validar los rewards informativos moldeados por YAML
"""

import pytest
import tempfile
import yaml
from pathlib import Path
import sys

# Añadir el directorio raíz al path
sys.path.insert(0, str(Path(__file__).parent.parent))

from train_env.reward_shaper import RewardShaper


class TestRewardsPipeline:
    """Test del pipeline de rewards"""
    
    def setup_method(self):
        """Setup para cada test"""
        self.temp_dir = tempfile.mkdtemp()
        
        # Crear configuración de rewards para testing
        self.rewards_config = {
            "tiers": {
                "pos": [[0, 5, 0.2], [6, 15, 0.5], [16, 40, 1.0]],
                "neg": [[0, 5, -0.5], [6, 15, -1.0], [16, 40, -1.5]]
            },
            "bonuses": {"tp_hit": 0.5, "sl_hit": -0.5},
            "weights": {
                "realized_pnl": 1.0,
                "unrealized_pnl": 0.02,
                "time_penalty": -0.0001,
                "survival_bonus": 0.001
            },
            "reward_clip": [-5.0, 5.0]
        }
        
        # Crear archivo YAML temporal
        self.rewards_yaml = Path(self.temp_dir) / "rewards.yaml"
        with open(self.rewards_yaml, 'w') as f:
            yaml.dump(self.rewards_config, f)
        
        # Crear RewardShaper
        self.shaper = RewardShaper(str(self.rewards_yaml))
    
    def test_tp_sl_bonuses(self):
        """Test bonuses por TP/SL hits"""
        obs = {
            "position": {"side": 0, "qty": 0.0, "unrealized_pnl": 0.0},
            "portfolio": {"equity_quote": 1000.0, "drawdown_day_pct": 0.0}
        }
        
        # Evento de cierre con TP hit
        events_tp = [{
            "kind": "CLOSE",
            "roi_pct": 10.0,
            "r_multiple": 2.0,
            "risk_pct": 5.0
        }, {"kind": "TP_HIT"}]
        
        reward_tp, _ = self.shaper.compute(
            obs, base_reward=0.0, events=events_tp,
            initial_balance=1000.0, target_balance=10000.0
        )
        
        # Evento de cierre con SL hit
        events_sl = [{
            "kind": "CLOSE",
            "roi_pct": -5.0,
            "r_multiple": -1.0,
            "risk_pct": 5.0
        }, {"kind": "SL_HIT"}]
        
        reward_sl, _ = self.shaper.compute(
            obs, base_reward=0.0, events=events_sl,
            initial_balance=1000.0, target_balance=10000.0
        )
        
        # Verificar que TP tiene bonus positivo y SL tiene malus
        assert reward_tp > reward_sl
    
    def test_roi_tiers(self):
        """Test buckets por ROI%"""
        obs = {
            "position": {"side": 0, "qty": 0.0, "unrealized_pnl": 0.0},
            "portfolio": {"equity_quote": 1000.0, "drawdown_day_pct": 0.0}
        }
        
        # Test ROI positivo
        events = [{
            "kind": "CLOSE",
            "roi_pct": 10.0,
            "r_multiple": 2.0,
            "risk_pct": 5.0
        }]
        
        reward, _ = self.shaper.compute(
            obs, base_reward=0.0, events=events,
            initial_balance=1000.0, target_balance=10000.0
        )
        
        # Verificar que el reward es positivo
        assert reward > 0
    
    def test_reward_clipping(self):
        """Test que el reward se clipa correctamente"""
        obs = {
            "position": {"side": 0, "qty": 0.0, "unrealized_pnl": 0.0},
            "portfolio": {"equity_quote": 1000.0, "drawdown_day_pct": 0.0}
        }
        
        # Evento con ROI muy alto
        events_high_roi = [{
            "kind": "CLOSE",
            "roi_pct": 1000.0,
            "r_multiple": 10.0,
            "risk_pct": 1.0
        }]
        
        reward, _ = self.shaper.compute(
            obs, base_reward=0.0, events=events_high_roi,
            initial_balance=1000.0, target_balance=10000.0
        )
        
        # Verificar que el reward está clipado
        assert reward <= 5.0
        assert reward >= -5.0


if __name__ == "__main__":
    pytest.main([__file__])