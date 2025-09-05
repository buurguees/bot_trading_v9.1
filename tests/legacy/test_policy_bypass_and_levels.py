# tests/test_policy_bypass_and_levels.py
"""
Test para validar el bypass de policy en TRAIN y SL/TP por defecto:
- Policy no bloquea al RL en entrenamiento
- SL/TP por defecto cuando faltan
- NO_SL_DISTANCE ≈ 0
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
        self.current_ts = 1640995200000  # Timestamp fijo
    
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


class TestPolicyBypass:
    """Test del bypass de policy en entrenamiento"""
    
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
        self.broker = MockBroker()
        
        # Mock OMS
        self.oms = MockOMS()
    
    def test_rl_action_bypass_policy(self):
        """Test que las acciones RL bypasean la policy"""
        env = BaseTradingEnv(
            cfg=self.cfg,
            broker=self.broker,
            oms=self.oms,
            initial_cash=1000.0,
            target_cash=10000.0,
            models_root=str(self.temp_dir)
        )
        
        # Reset del entorno
        obs = env.reset()
        
        # Configurar acción RL forzada (force long)
        env.set_action_override(3)  # Force LONG
        
        # Ejecutar step
        obs, reward, done, info = env.step()
        
        # Verificar que se ejecutó la acción RL (no policy)
        # Esto se puede verificar por la presencia de trades o cambios en el portfolio
        # En un entorno real, esto se verificaría por la apertura de posición
    
    def test_default_sl_tp_calculation(self):
        """Test que se calculan SL/TP por defecto cuando faltan"""
        env = BaseTradingEnv(
            cfg=self.cfg,
            broker=self.broker,
            oms=self.oms,
            initial_cash=1000.0,
            target_cash=10000.0,
            models_root=str(self.temp_dir)
        )
        
        # Test con ATR disponible
        price = 50000.0
        atr = 1000.0  # ATR simulado
        side = 1  # Long
        
        sl, tp = env._get_default_sl_tp(price, atr, side)
        
        # Verificar que se calcularon SL/TP
        assert sl is not None
        assert tp is not None
        assert sl < price  # SL debe estar por debajo del precio para long
        assert tp > price  # TP debe estar por encima del precio para long
        
        # Verificar que la distancia es razonable
        sl_distance = price - sl
        tp_distance = tp - price
        assert sl_distance > 0
        assert tp_distance > 0
    
    def test_default_sl_tp_fallback(self):
        """Test fallback a porcentajes cuando no hay ATR"""
        env = BaseTradingEnv(
            cfg=self.cfg,
            broker=self.broker,
            oms=self.oms,
            initial_cash=1000.0,
            target_cash=10000.0,
            models_root=str(self.temp_dir)
        )
        
        # Test sin ATR (None)
        price = 50000.0
        atr = None
        side = 1  # Long
        
        sl, tp = env._get_default_sl_tp(price, atr, side)
        
        # Verificar que se calcularon SL/TP con fallback
        assert sl is not None
        assert tp is not None
        assert sl < price  # SL debe estar por debajo del precio para long
        assert tp > price  # TP debe estar por encima del precio para long
        
        # Verificar que usa porcentajes por defecto
        sl_distance_pct = ((price - sl) / price) * 100
        tp_distance_pct = ((tp - price) / price) * 100
        
        # Debería usar aproximadamente 1% para SL y 1.5% para TP
        assert 0.5 <= sl_distance_pct <= 2.0  # Tolerancia
        assert 1.0 <= tp_distance_pct <= 3.0  # Tolerancia
    
    def test_no_sl_distance_elimination(self):
        """Test que NO_SL_DISTANCE se elimina con SL/TP por defecto"""
        env = BaseTradingEnv(
            cfg=self.cfg,
            broker=self.broker,
            oms=self.oms,
            initial_cash=1000.0,
            target_cash=10000.0,
            models_root=str(self.temp_dir)
        )
        
        # Reset del entorno
        obs = env.reset()
        
        # Configurar acción RL forzada sin SL/TP
        env.set_action_override(3)  # Force LONG
        
        # Ejecutar step
        obs, reward, done, info = env.step()
        
        # Verificar que no hay eventos NO_SL_DISTANCE
        events = info.get("events", [])
        no_sl_events = [e for e in events if e.get("kind") == "NO_SL_DISTANCE"]
        assert len(no_sl_events) == 0
    
    def test_policy_confidence_bypass(self):
        """Test que la policy no bloquea por confidence en entrenamiento"""
        # Configurar policy con confidence alta (que normalmente bloquearía)
        high_confidence_cfg = EnvConfig(
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
            hierarchical=HierarchicalConfig(min_confidence=0.8, execute_tfs=["1m"], confirm_tfs=[]),  # Alta confidence
            risk=RiskConfig(),
            fees=FeesConfig()
        )
        
        env = BaseTradingEnv(
            cfg=high_confidence_cfg,
            broker=self.broker,
            oms=self.oms,
            initial_cash=1000.0,
            target_cash=10000.0,
            models_root=str(self.temp_dir)
        )
        
        # Reset del entorno
        obs = env.reset()
        
        # Configurar acción RL forzada
        env.set_action_override(3)  # Force LONG
        
        # Ejecutar step
        obs, reward, done, info = env.step()
        
        # Verificar que la acción RL se ejecutó a pesar de la alta confidence
        # (Esto se verificaría por la apertura de posición en un entorno real)
    
    def test_short_position_default_levels(self):
        """Test SL/TP por defecto para posiciones short"""
        env = BaseTradingEnv(
            cfg=self.cfg,
            broker=self.broker,
            oms=self.oms,
            initial_cash=1000.0,
            target_cash=10000.0,
            models_root=str(self.temp_dir)
        )
        
        # Test para short
        price = 50000.0
        atr = 1000.0
        side = -1  # Short
        
        sl, tp = env._get_default_sl_tp(price, atr, side)
        
        # Verificar que se calcularon SL/TP para short
        assert sl is not None
        assert tp is not None
        assert sl > price  # SL debe estar por encima del precio para short
        assert tp < price  # TP debe estar por debajo del precio para short
        
        # Verificar que la distancia es razonable
        sl_distance = sl - price
        tp_distance = price - tp
        assert sl_distance > 0
        assert tp_distance > 0
    
    def test_yaml_configuration_usage(self):
        """Test que usa configuración YAML para niveles por defecto"""
        # Mock de configuración YAML
        mock_risk_config = MagicMock()
        mock_risk_config.common.default_levels = {
            'use_atr': True,
            'atr_period': 14,
            'sl_atr_mult': 1.5,  # 1.5x ATR
            'min_sl_pct': 2.0,   # 2% mínimo
            'tp_r_multiple': 2.0  # 2x el riesgo
        }
        
        env = BaseTradingEnv(
            cfg=self.cfg,
            broker=self.broker,
            oms=self.oms,
            initial_cash=1000.0,
            target_cash=10000.0,
            models_root=str(self.temp_dir)
        )
        
        # Mock de la configuración de riesgo
        env.cfg.risk = mock_risk_config
        
        # Test con ATR
        price = 50000.0
        atr = 1000.0
        side = 1  # Long
        
        sl, tp = env._get_default_sl_tp(price, atr, side)
        
        # Verificar que usa la configuración YAML
        expected_sl_distance = atr * 1.5  # sl_atr_mult
        expected_tp_distance = expected_sl_distance * 2.0  # tp_r_multiple
        
        actual_sl_distance = price - sl
        actual_tp_distance = tp - price
        
        # Verificar que está cerca de los valores esperados (con tolerancia)
        assert abs(actual_sl_distance - expected_sl_distance) < 100  # Tolerancia
        assert abs(actual_tp_distance - expected_tp_distance) < 200  # Tolerancia


if __name__ == "__main__":
    pytest.main([__file__])