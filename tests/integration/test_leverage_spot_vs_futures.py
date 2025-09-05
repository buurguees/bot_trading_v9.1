#!/usr/bin/env python3
"""
Test para validar el cálculo de leverage en spot vs futures.
"""
import tempfile
from pathlib import Path
from base_env.base_env import BaseTradingEnv, OMSAdapter
from base_env.io.historical_broker import ParquetHistoricalBroker
from base_env.config.models import EnvConfig, SymbolMeta, LeverageConfig
from base_env.config.models import RiskConfig, RiskCommon, RiskFutures, DefaultLevelsConfig


def test_spot_leverage_always_one():
    """Test que verifica que en spot siempre se usa leverage = 1.0"""
    with tempfile.TemporaryDirectory() as temp_dir:
        # Configurar para spot
        symbol_meta = SymbolMeta(
            symbol="BTCUSDT",
            market="spot",
            filters={}
        )
        
        env_config = EnvConfig(
            symbol_meta=symbol_meta,
            market="spot"
        )
        
        # Crear broker mock
        broker = ParquetHistoricalBroker(
            data_root="data",
            symbol="BTCUSDT",
            market="spot",
            tfs=["1m", "5m", "15m", "1h"]
        )
        
        # Crear OMS mock
        class MockOMS:
            def open(self, side, qty, price_hint, sl, tp):
                return {"price": price_hint, "qty": qty}
            def close(self, qty, price_hint):
                return {"price": price_hint, "qty": qty}
        
        oms = MockOMS()
        
        # Crear entorno
        env = BaseTradingEnv(
            cfg=env_config,
            broker=broker,
            oms=oms,
            initial_cash=10000.0,
            target_cash=100000.0,
            models_root=temp_dir
        )
        
        # Verificar que el leverage calculado es siempre 1.0
        leverage = env._calculate_leverage_used()
        assert leverage == 1.0, f"En spot, leverage debe ser 1.0, es {leverage}"
        
        # Verificar que override de leverage se ignora en spot
        env._leverage_override = 10.0  # Intentar usar leverage 10x
        leverage = env._calculate_leverage_used()
        assert leverage == 1.0, f"En spot, override de leverage debe ignorarse, es {leverage}"


def test_futures_leverage_clamping():
    """Test que verifica que en futures el leverage se clampa correctamente"""
    with tempfile.TemporaryDirectory() as temp_dir:
        # Configurar para futures con leverage
        symbol_meta = SymbolMeta(
            symbol="BTCUSDT",
            market="futures",
            filters={},
            leverage=LeverageConfig(
                min=2.0,
                max=25.0,
                step=1.0,
                default=3.0
            )
        )
        
        env_config = EnvConfig(
            symbol_meta=symbol_meta,
            market="futures"
        )
        
        # Crear broker mock
        broker = ParquetHistoricalBroker(
            data_root="data",
            symbol="BTCUSDT",
            market="futures",
            tfs=["1m", "5m", "15m", "1h"]
        )
        
        # Crear OMS mock
        class MockOMS:
            def open(self, side, qty, price_hint, sl, tp):
                return {"price": price_hint, "qty": qty}
            def close(self, qty, price_hint):
                return {"price": price_hint, "qty": qty}
        
        oms = MockOMS()
        
        # Crear entorno
        env = BaseTradingEnv(
            cfg=env_config,
            broker=broker,
            oms=oms,
            initial_cash=10000.0,
            target_cash=100000.0,
            models_root=temp_dir
        )
        
        # Test 1: Sin override, debe usar default (3.0)
        leverage = env._calculate_leverage_used()
        assert leverage == 3.0, f"Sin override, debe usar default 3.0, es {leverage}"
        
        # Test 2: Override dentro del rango, debe usar el override
        env._leverage_override = 10.0
        leverage = env._calculate_leverage_used()
        assert leverage == 10.0, f"Override 10.0 dentro del rango, debe usar 10.0, es {leverage}"
        
        # Test 3: Override por debajo del mínimo, debe clamp a 2.0
        env._leverage_override = 1.0
        leverage = env._calculate_leverage_used()
        assert leverage == 2.0, f"Override 1.0 por debajo del mínimo, debe clamp a 2.0, es {leverage}"
        
        # Test 4: Override por encima del máximo, debe clamp a 25.0
        env._leverage_override = 50.0
        leverage = env._calculate_leverage_used()
        assert leverage == 25.0, f"Override 50.0 por encima del máximo, debe clamp a 25.0, es {leverage}"


def test_futures_leverage_edge_cases():
    """Test que verifica casos edge del leverage en futures"""
    with tempfile.TemporaryDirectory() as temp_dir:
        # Configurar para futures con leverage extremo
        symbol_meta = SymbolMeta(
            symbol="BTCUSDT",
            market="futures",
            filters={},
            leverage=LeverageConfig(
                min=1.0,
                max=100.0,
                step=0.1,
                default=5.0
            )
        )
        
        env_config = EnvConfig(
            symbol_meta=symbol_meta,
            market="futures"
        )
        
        # Crear broker mock
        broker = ParquetHistoricalBroker(
            data_root="data",
            symbol="BTCUSDT",
            market="futures",
            tfs=["1m", "5m", "15m", "1h"]
        )
        
        # Crear OMS mock
        class MockOMS:
            def open(self, side, qty, price_hint, sl, tp):
                return {"price": price_hint, "qty": qty}
            def close(self, qty, price_hint):
                return {"price": price_hint, "qty": qty}
        
        oms = MockOMS()
        
        # Crear entorno
        env = BaseTradingEnv(
            cfg=env_config,
            broker=broker,
            oms=oms,
            initial_cash=10000.0,
            target_cash=100000.0,
            models_root=temp_dir
        )
        
        # Test 1: Override exactamente en el mínimo
        env._leverage_override = 1.0
        leverage = env._calculate_leverage_used()
        assert leverage == 1.0, f"Override en mínimo 1.0, debe ser 1.0, es {leverage}"
        
        # Test 2: Override exactamente en el máximo
        env._leverage_override = 100.0
        leverage = env._calculate_leverage_used()
        assert leverage == 100.0, f"Override en máximo 100.0, debe ser 100.0, es {leverage}"
        
        # Test 3: Override con decimales
        env._leverage_override = 15.7
        leverage = env._calculate_leverage_used()
        assert leverage == 15.7, f"Override con decimales 15.7, debe ser 15.7, es {leverage}"


def test_futures_without_leverage_config():
    """Test que verifica el comportamiento cuando no hay configuración de leverage"""
    with tempfile.TemporaryDirectory() as temp_dir:
        # Configurar para futures sin configuración de leverage
        symbol_meta = SymbolMeta(
            symbol="BTCUSDT",
            market="futures",
            filters={}
            # Sin leverage config
        )
        
        env_config = EnvConfig(
            symbol_meta=symbol_meta,
            market="futures"
        )
        
        # Crear broker mock
        broker = ParquetHistoricalBroker(
            data_root="data",
            symbol="BTCUSDT",
            market="futures",
            tfs=["1m", "5m", "15m", "1h"]
        )
        
        # Crear OMS mock
        class MockOMS:
            def open(self, side, qty, price_hint, sl, tp):
                return {"price": price_hint, "qty": qty}
            def close(self, qty, price_hint):
                return {"price": price_hint, "qty": qty}
        
        oms = MockOMS()
        
        # Crear entorno
        env = BaseTradingEnv(
            cfg=env_config,
            broker=broker,
            oms=oms,
            initial_cash=10000.0,
            target_cash=100000.0,
            models_root=temp_dir
        )
        
        # Debe usar valores por defecto
        leverage = env._calculate_leverage_used()
        assert leverage == 3.0, f"Sin config de leverage, debe usar default 3.0, es {leverage}"
        
        # Override debe funcionar pero sin clamping
        env._leverage_override = 50.0
        leverage = env._calculate_leverage_used()
        assert leverage == 50.0, f"Sin config de leverage, override debe usarse sin clamping, es {leverage}"


if __name__ == "__main__":
    test_spot_leverage_always_one()
    test_futures_leverage_clamping()
    test_futures_leverage_edge_cases()
    test_futures_without_leverage_config()
    print("✅ Todos los tests de leverage spot vs futures pasaron correctamente")
