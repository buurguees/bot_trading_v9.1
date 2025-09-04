# tests/test_bitget_collector_mapping.py
"""
Test para validar el collector de Bitget para futuros:
- Mapeo de TFs y símbolos
- Normalización de columnas
- Interfaz fetch_ohlcv
"""

import pytest
from unittest.mock import patch, MagicMock
from data_pipeline.collectors.bitget_futures_collector import BitgetFuturesCollector, create_bitget_collector


class TestBitgetFuturesCollector:
    """Test del collector de Bitget para futuros"""
    
    def setup_method(self):
        """Setup para cada test"""
        self.collector = BitgetFuturesCollector()
    
    def test_symbol_mapping(self):
        """Test que mapea símbolos correctamente"""
        # Test símbolo estándar
        assert self.collector._map_symbol("BTCUSDT") == "BTCUSDT"
        assert self.collector._map_symbol("ethusdt") == "ETHUSDT"
        assert self.collector._map_symbol("ADAUSDT") == "ADAUSDT"
    
    def test_interval_mapping(self):
        """Test que mapea TFs correctamente"""
        # Test TFs soportados
        assert self.collector._map_interval("1m") == "1m"
        assert self.collector._map_interval("5m") == "5m"
        assert self.collector._map_interval("15m") == "15m"
        assert self.collector._map_interval("1h") == "1H"
        assert self.collector._map_interval("4h") == "4H"
        assert self.collector._map_interval("1d") == "1D"
        
        # Test TF no soportado
        with pytest.raises(ValueError, match="TF no soportado"):
            self.collector._map_interval("30m")
    
    def test_normalize_klines(self):
        """Test que normaliza klines correctamente"""
        # Mock klines de Bitget
        mock_klines = [
            [1640995200000, 47000.0, 48000.0, 46000.0, 47500.0, 100.0, 1640995260000, 4750000.0, 50, 60.0, 2850000.0, 0],
            [1640995260000, 47500.0, 48500.0, 47000.0, 48000.0, 120.0, 1640995320000, 5760000.0, 60, 70.0, 3360000.0, 0]
        ]
        
        normalized = self.collector._normalize_klines(mock_klines, "BTCUSDT", "1h")
        
        assert len(normalized) == 2
        assert normalized[0]["ts"] == 1640995200000
        assert normalized[0]["open"] == 47000.0
        assert normalized[0]["high"] == 48000.0
        assert normalized[0]["low"] == 46000.0
        assert normalized[0]["close"] == 47500.0
        assert normalized[0]["volume"] == 100.0
        assert normalized[0]["quote_volume"] == 4750000.0
        assert normalized[0]["symbol"] == "BTCUSDT"
        assert normalized[0]["tf"] == "1h"
        assert normalized[0]["market"] == "futures"
        assert normalized[0]["exchange"] == "bitget"
        assert normalized[0]["product_type"] == "umcbl"
    
    @patch('data_pipeline.collectors.bitget_futures_collector.requests.Session.get')
    def test_fetch_ohlcv_success(self, mock_get):
        """Test que fetch_ohlcv funciona correctamente"""
        # Mock response de Bitget
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "code": "00000",
            "data": [
                [1640995200000, 47000.0, 48000.0, 46000.0, 47500.0, 100.0, 1640995260000, 4750000.0, 50, 60.0, 2850000.0, 0]
            ]
        }
        mock_get.return_value = mock_response
        
        # Test fetch
        result = self.collector.fetch_ohlcv("BTCUSDT", "1h", 1640995200000, 1640995260000)
        
        assert len(result) == 1
        assert result[0]["symbol"] == "BTCUSDT"
        assert result[0]["tf"] == "1h"
        assert result[0]["market"] == "futures"
    
    @patch('data_pipeline.collectors.bitget_futures_collector.requests.Session.get')
    def test_fetch_ohlcv_api_error(self, mock_get):
        """Test que maneja errores de API correctamente"""
        # Mock error response
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "code": "40001",
            "msg": "Invalid symbol"
        }
        mock_get.return_value = mock_response
        
        # Test que lanza excepción
        with pytest.raises(ValueError, match="Error API Bitget"):
            self.collector.fetch_ohlcv("INVALID", "1h", 1640995200000, 1640995260000)
    
    @patch('data_pipeline.collectors.bitget_futures_collector.requests.Session.get')
    def test_fetch_ohlcv_rate_limit(self, mock_get):
        """Test que maneja rate limits correctamente"""
        # Mock rate limit response
        mock_response = MagicMock()
        mock_response.status_code = 429
        mock_get.return_value = mock_response
        
        # Test que reintenta con rate limit
        with pytest.raises(ValueError):  # Fallará después de reintentos
            self.collector.fetch_ohlcv("BTCUSDT", "1h", 1640995200000, 1640995260000, max_retries=1)
    
    @patch('data_pipeline.collectors.bitget_futures_collector.requests.Session.get')
    def test_get_symbol_info(self, mock_get):
        """Test que obtiene información del símbolo correctamente"""
        # Mock response con información del símbolo
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "code": "00000",
            "data": [
                {
                    "symbol": "BTCUSDT",
                    "baseCoin": "BTC",
                    "quoteCoin": "USDT",
                    "minTradeNum": "0.00001",
                    "maxTradeNum": "1000",
                    "priceScale": "1",
                    "minTradeUSDT": "5",
                    "status": "online"
                }
            ]
        }
        mock_get.return_value = mock_response
        
        # Test get_symbol_info
        info = self.collector.get_symbol_info("BTCUSDT")
        
        assert info["symbol"] == "BTCUSDT"
        assert info["base_asset"] == "BTC"
        assert info["quote_asset"] == "USDT"
        assert info["min_qty"] == 0.00001
        assert info["max_qty"] == 1000.0
        assert info["tick_size"] == 1.0
        assert info["min_notional"] == 5.0
        assert info["status"] == "online"
        assert info["exchange"] == "bitget"
        assert info["market"] == "futures"
        assert info["product_type"] == "umcbl"
    
    def test_create_bitget_collector(self):
        """Test que la factory function funciona correctamente"""
        collector = create_bitget_collector()
        assert isinstance(collector, BitgetFuturesCollector)
        assert collector.product_type == "umcbl"
        
        # Test con product_type personalizado
        collector_custom = create_bitget_collector("custom")
        assert collector_custom.product_type == "custom"


class TestBitgetCollectorIntegration:
    """Test de integración del collector de Bitget"""
    
    def test_collector_initialization(self):
        """Test que el collector se inicializa correctamente"""
        collector = BitgetFuturesCollector()
        assert collector.product_type == "umcbl"
        assert collector.base_url == "https://api.bitget.com"
        assert collector.session is not None
    
    def test_collector_with_custom_product_type(self):
        """Test que el collector acepta product_type personalizado"""
        collector = BitgetFuturesCollector("custom_type")
        assert collector.product_type == "custom_type"
    
    def test_normalize_empty_klines(self):
        """Test que normaliza klines vacíos correctamente"""
        collector = BitgetFuturesCollector()
        result = collector._normalize_klines([], "BTCUSDT", "1h")
        assert result == []
    
    def test_normalize_klines_with_missing_fields(self):
        """Test que normaliza klines con campos faltantes correctamente"""
        collector = BitgetFuturesCollector()
        # Kline con menos campos de los esperados
        mock_klines = [
            [1640995200000, 47000.0, 48000.0, 46000.0, 47500.0, 100.0, 1640995260000]
        ]
        
        normalized = collector._normalize_klines(mock_klines, "BTCUSDT", "1h")
        
        assert len(normalized) == 1
        assert normalized[0]["quote_volume"] == 0.0  # Campo faltante se llena con 0


if __name__ == "__main__":
    pytest.main([__file__])
