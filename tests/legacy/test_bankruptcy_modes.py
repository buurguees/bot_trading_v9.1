# tests/test_bankruptcy_modes.py
"""
Test para validar la bancarrota configurable por YAML:
- Modo "end": termina el episodio
- Modo "soft_reset": reinicia balance pero continúa
- Cooldown y leverage cap después de soft reset
- Fallback a "end" si se excede max_resets_per_run
"""

import pytest
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch
import sys

# Añadir el directorio raíz al path
sys.path.insert(0, str(Path(__file__).parent.parent))

from base_env.base_env import BaseTradingEnv
from base_env.config.models import EnvConfig, SymbolMeta, PipelineConfig, HierarchicalConfig, RiskConfig, FeesConfig


class MockBroker:
    """Mock broker para testing"""
    
    def __init__(self):
        self.current_price = 50000.0
        self.current_ts = 1640995200000
    
    def get_price(self):
        return self.current_price
    
    def get_current_ts(self):
        return self.current_ts
    
    def next(self):
        pass
    
    def is_end_of_data(self):
        return False
    
    def reset_to_start(self):
        pass


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


class TestBankruptcyModes:
    """Test de modos de bancarrota"""
    
    def setup_method(self):
        """Setup para cada test"""
        self.temp_dir = tempfile.mkdtemp()
        self.models_dir = Path(self.temp_dir) / "models" / "BTCUSDT"
        self.models_dir.mkdir(parents=True, exist_ok=True)
        
        # Mock broker
        self.broker = MockBroker()
        
        # Mock OMS
        self.oms = MockOMS()
    
    def create_env_with_bankruptcy_config(self, mode="end", threshold_pct=20.0, max_resets=1):
        """Crea entorno con configuración de bancarrota específica"""
        # Mock de configuración de riesgo
        mock_risk_config = MagicMock()
        mock_risk_config.common.bankruptcy.enabled = True
        mock_risk_config.common.bankruptcy.mode = mode
        mock_risk_config.common.bankruptcy.threshold_pct = threshold_pct
        mock_risk_config.common.bankruptcy.penalty_reward = -10.0
        mock_risk_config.common.bankruptcy.restart_on_bankruptcy = True
        mock_risk_config.common.bankruptcy.soft_reset.max_resets_per_run = max_resets
        mock_risk_config.common.bankruptcy.soft_reset.post_reset_leverage_cap = 2.0
        mock_risk_config.common.bankruptcy.soft_reset.cooldown_bars = 50
        mock_risk_config.common.bankruptcy.soft_reset.label_segment = True
        
        # Configuración del entorno
        cfg = EnvConfig(
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
            risk=mock_risk_config,
            fees=FeesConfig()
        )
        
        env = BaseTradingEnv(
            cfg=cfg,
            broker=self.broker,
            oms=self.oms,
            initial_cash=1000.0,
            target_cash=10000.0,
            models_root=str(self.temp_dir)
        )
        
        return env
    
    def test_bankruptcy_end_mode(self):
        """Test modo 'end': termina el episodio"""
        env = self.create_env_with_bankruptcy_config(mode="end", threshold_pct=20.0)
        
        # Reset del entorno
        obs = env.reset()
        
        # Simular bancarrota: equity por debajo del umbral
        env.portfolio.equity_quote = 150.0  # 15% del initial (1000), por debajo del 20%
        
        # Ejecutar step
        obs, reward, done, info = env.step()
        
        # Verificar que se detectó bancarrota y terminó el episodio
        assert done
        assert "BANKRUPTCY" in str(info.get("done_reason", ""))
        assert info.get("bankruptcy", False)
        assert info.get("penalty_reward", 0) == -10.0
        
        # Verificar que se aplicó la penalización
        assert reward < 0  # Debe incluir la penalización
    
    def test_bankruptcy_soft_reset_mode(self):
        """Test modo 'soft_reset': reinicia balance pero continúa"""
        env = self.create_env_with_bankruptcy_config(mode="soft_reset", threshold_pct=20.0)
        
        # Reset del entorno
        obs = env.reset()
        
        # Simular bancarrota: equity por debajo del umbral
        env.portfolio.equity_quote = 150.0  # 15% del initial (1000), por debajo del 20%
        
        # Ejecutar step
        obs, reward, done, info = env.step()
        
        # Verificar que NO terminó el episodio (soft reset)
        assert not done
        assert "SOFT_RESET" in str(info.get("done_reason", ""))
        assert info.get("soft_reset", False)
        assert info.get("reset_count", 0) == 1
        assert info.get("segment_id", 0) == 1
        
        # Verificar que se reinició el balance
        assert env.portfolio.cash_quote == 1000.0  # Initial cash
        assert env.portfolio.equity_quote == 1000.0  # Initial equity
        assert env.portfolio.used_margin == 0.0  # Sin margen usado
        
        # Verificar que se activó cooldown y leverage cap
        assert env._cooldown_bars_remaining == 50
        assert env._leverage_cap_active == 2.0
        assert env._soft_reset_count == 1
        assert env._current_segment_id == 1
    
    def test_soft_reset_cooldown_blocks_trading(self):
        """Test que el cooldown bloquea trading después de soft reset"""
        env = self.create_env_with_bankruptcy_config(mode="soft_reset", threshold_pct=20.0)
        
        # Reset del entorno
        obs = env.reset()
        
        # Simular bancarrota y soft reset
        env.portfolio.equity_quote = 150.0
        obs, reward, done, info = env.step()
        
        # Verificar que se activó cooldown
        assert env._cooldown_bars_remaining == 50
        
        # Intentar trading durante cooldown
        env.set_action_override(3)  # Force LONG
        obs, reward, done, info = env.step()
        
        # Verificar que se bloqueó el trading
        events = info.get("events", [])
        cooldown_events = [e for e in events if e.get("kind") == "COOLDOWN_AFTER_RESET"]
        assert len(cooldown_events) > 0
        
        # Verificar que el cooldown disminuyó
        assert env._cooldown_bars_remaining == 49
    
    def test_soft_reset_leverage_cap(self):
        """Test que el leverage cap se aplica después de soft reset"""
        env = self.create_env_with_bankruptcy_config(mode="soft_reset", threshold_pct=20.0)
        
        # Reset del entorno
        obs = env.reset()
        
        # Simular bancarrota y soft reset
        env.portfolio.equity_quote = 150.0
        obs, reward, done, info = env.step()
        
        # Verificar que se activó leverage cap
        assert env._leverage_cap_active == 2.0
        
        # Intentar usar leverage alto
        env.set_leverage_override(5.0)  # 5x leverage
        env.set_action_override(3)  # Force LONG
        
        # El leverage debería estar limitado a 2.0x
        assert env._leverage_override == 2.0
    
    def test_max_resets_fallback_to_end(self):
        """Test que exceder max_resets_per_run causa fallback a modo 'end'"""
        env = self.create_env_with_bankruptcy_config(mode="soft_reset", threshold_pct=20.0, max_resets=2)
        
        # Reset del entorno
        obs = env.reset()
        
        # Simular múltiples bancarrotas
        for reset_count in range(3):  # 3 resets (excede el máximo de 2)
            env.portfolio.equity_quote = 150.0  # Bancarrota
            obs, reward, done, info = env.step()
            
            if reset_count < 2:
                # Primeros 2 resets: soft reset
                assert not done
                assert "SOFT_RESET" in str(info.get("done_reason", ""))
                assert info.get("reset_count", 0) == reset_count + 1
            else:
                # Tercer reset: fallback a end
                assert done
                assert "BANKRUPTCY_MAX_RESETS" in str(info.get("done_reason", ""))
                assert info.get("bankruptcy", False)
                break
    
    def test_bankruptcy_disabled(self):
        """Test que la bancarrota se puede desactivar"""
        env = self.create_env_with_bankruptcy_config(mode="end", threshold_pct=20.0)
        
        # Desactivar bancarrota
        env.cfg.risk.common.bankruptcy.enabled = False
        
        # Reset del entorno
        obs = env.reset()
        
        # Simular bancarrota: equity por debajo del umbral
        env.portfolio.equity_quote = 150.0  # 15% del initial (1000), por debajo del 20%
        
        # Ejecutar step
        obs, reward, done, info = env.step()
        
        # Verificar que NO se detectó bancarrota
        assert not done
        assert "BANKRUPTCY" not in str(info.get("done_reason", ""))
        assert not info.get("bankruptcy", False)
    
    def test_bankruptcy_threshold_calculation(self):
        """Test cálculo correcto del umbral de bancarrota"""
        env = self.create_env_with_bankruptcy_config(mode="end", threshold_pct=30.0)
        
        # Reset del entorno
        obs = env.reset()
        
        # Umbral debería ser 30% de 1000 = 300
        # Equity por encima del umbral: no bancarrota
        env.portfolio.equity_quote = 350.0
        obs, reward, done, info = env.step()
        assert not done
        
        # Equity por debajo del umbral: bancarrota
        env.portfolio.equity_quote = 250.0
        obs, reward, done, info = env.step()
        assert done
        assert "BANKRUPTCY" in str(info.get("done_reason", ""))
    
    def test_penalty_reward_application(self):
        """Test que se aplica la penalización de bancarrota"""
        env = self.create_env_with_bankruptcy_config(mode="end", threshold_pct=20.0)
        
        # Reset del entorno
        obs = env.reset()
        
        # Simular bancarrota
        env.portfolio.equity_quote = 150.0
        obs, reward, done, info = env.step()
        
        # Verificar que se aplicó la penalización
        assert reward < 0
        assert info.get("penalty_reward", 0) == -10.0
    
    def test_segment_id_increment(self):
        """Test que el segment_id se incrementa en soft reset"""
        env = self.create_env_with_bankruptcy_config(mode="soft_reset", threshold_pct=20.0)
        
        # Reset del entorno
        obs = env.reset()
        initial_segment_id = env._current_segment_id
        
        # Simular bancarrota y soft reset
        env.portfolio.equity_quote = 150.0
        obs, reward, done, info = env.step()
        
        # Verificar que se incrementó el segment_id
        assert env._current_segment_id == initial_segment_id + 1
        assert info.get("segment_id", 0) == initial_segment_id + 1


if __name__ == "__main__":
    pytest.main([__file__])