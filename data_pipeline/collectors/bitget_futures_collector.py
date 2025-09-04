# data_pipeline/collectors/bitget_futures_collector.py
# Descripción: Collector para descarga de datos históricos de futuros desde Bitget
# Interfaz: fetch_ohlcv(symbol: str, interval: str, start: int, end: int) -> list[dict]
# 
# Características:
# - Usa API oficial Bitget Perpetual USDT-M
# - Mapea TFs internos → API Bitget
# - Symbol mapping: interno "BTCUSDT" → Bitget "BTCUSDT" + productType "umcbl"
# - Normaliza OHLCV a columnas estándar: ts, open, high, low, close, volume, quote_volume

from __future__ import annotations
import time
import requests
from typing import List, Dict, Optional
from datetime import datetime, timezone

# API Bitget Perpetual USDT-M
BITGET_FUTURES_BASE = "https://api.bitget.com"

# Mapa TF interno → intervalo Bitget
BITGET_INTERVAL_MAP = {
    "1m": "1m",
    "5m": "5m", 
    "15m": "15m",
    "1h": "1H",
    "4h": "4H",
    "1d": "1D",
}

# Headers estándar
HEADERS = {
    "User-Agent": "BaseEnvDataDownloader/1.0 (+https://github.com/)",
    "Accept": "application/json",
    "Content-Type": "application/json",
}


class BitgetFuturesCollector:
    """
    Collector para descarga de datos históricos de futuros desde Bitget.
    
    Características:
    - Usa API oficial Bitget Perpetual USDT-M
    - Mapea símbolos internos a símbolos Bitget
    - Normaliza datos a formato estándar
    - Maneja rate limits y reintentos
    """
    
    def __init__(self, product_type: str = "umcbl"):
        """
        Inicializa el collector.
        
        Args:
            product_type: Tipo de producto Bitget (por defecto "umcbl" para USDT-M perpetual)
        """
        self.product_type = product_type
        self.base_url = BITGET_FUTURES_BASE
        self.session = requests.Session()
        self.session.headers.update(HEADERS)
    
    def _map_symbol(self, symbol: str) -> str:
        """
        Mapea símbolo interno a símbolo Bitget.
        
        Args:
            symbol: Símbolo interno (ej. "BTCUSDT")
            
        Returns:
            Símbolo Bitget (ej. "BTCUSDT")
        """
        # Por ahora, los símbolos son iguales
        # En el futuro se podría agregar mapeo más complejo
        return symbol.upper()
    
    def _map_interval(self, tf: str) -> str:
        """
        Mapea TF interno a intervalo Bitget.
        
        Args:
            tf: Timeframe interno (ej. "1m", "1h")
            
        Returns:
            Intervalo Bitget (ej. "1m", "1H")
        """
        if tf not in BITGET_INTERVAL_MAP:
            raise ValueError(f"TF no soportado: {tf}. Soportados: {list(BITGET_INTERVAL_MAP.keys())}")
        return BITGET_INTERVAL_MAP[tf]
    
    def _normalize_klines(self, klines: List[List], symbol: str, tf: str) -> List[Dict]:
        """
        Normaliza klines de Bitget a formato estándar.
        
        Bitget klines fields:
            0: open time (ms)
            1: open
            2: high  
            3: low
            4: close
            5: volume
            6: close time (ms)
            7: quote asset volume
            8: number of trades
            9: taker buy base asset volume
            10: taker buy quote asset volume
            11: ignore
        
        Args:
            klines: Lista de klines de Bitget
            symbol: Símbolo
            tf: Timeframe
            
        Returns:
            Lista de diccionarios normalizados
        """
        normalized = []
        for kline in klines:
            normalized.append({
                "ts": int(kline[0]),  # open time
                "open": float(kline[1]),
                "high": float(kline[2]),
                "low": float(kline[3]),
                "close": float(kline[4]),
                "volume": float(kline[5]),
                "quote_volume": float(kline[7]) if len(kline) > 7 else 0.0,
                "symbol": symbol,
                "tf": tf,
                "market": "futures",
                "exchange": "bitget",
                "product_type": self.product_type,
            })
        return normalized
    
    def fetch_ohlcv(
        self, 
        symbol: str, 
        interval: str, 
        start: int, 
        end: int,
        limit: int = 1000,
        max_retries: int = 5,
        pause_sec: float = 0.1
    ) -> List[Dict]:
        """
        Descarga datos OHLCV desde Bitget.
        
        Args:
            symbol: Símbolo (ej. "BTCUSDT")
            interval: Intervalo (ej. "1m", "5m", "1h")
            start: Timestamp inicio (ms)
            end: Timestamp fin (ms)
            limit: Límite por request (máx 1000)
            max_retries: Máximo reintentos
            pause_sec: Pausa entre requests
            
        Returns:
            Lista de diccionarios con datos OHLCV normalizados
        """
        # Mapear símbolo e intervalo
        bitget_symbol = self._map_symbol(symbol)
        bitget_interval = self._map_interval(interval)
        
        # Endpoint Bitget para klines
        endpoint = "/api/spot/v1/market/candles"
        
        all_klines = []
        cursor = start
        
        while cursor < end:
            params = {
                "symbol": bitget_symbol,
                "granularity": bitget_interval,
                "startTime": str(cursor),
                "endTime": str(end),
                "limit": str(min(limit, 1000)),
            }
            
            for attempt in range(1, max_retries + 1):
                try:
                    response = self.session.get(
                        self.base_url + endpoint,
                        params=params,
                        timeout=15
                    )
                    
                    if response.status_code == 429:
                        # Rate limit
                        time.sleep(pause_sec * attempt)
                        continue
                    
                    response.raise_for_status()
                    data = response.json()
                    
                    if data.get("code") != "00000":
                        raise ValueError(f"Error API Bitget: {data.get('msg', 'Unknown error')}")
                    
                    klines = data.get("data", [])
                    if not klines:
                        return all_klines
                    
                    # Normalizar y agregar
                    normalized = self._normalize_klines(klines, symbol, interval)
                    all_klines.extend(normalized)
                    
                    # Avanzar cursor al close time de la última vela + 1ms
                    last_close_ms = int(klines[-1][6])  # close time
                    if last_close_ms <= cursor:
                        cursor = cursor + 1
                    else:
                        cursor = last_close_ms
                    
                    # Rate limit suave
                    time.sleep(pause_sec)
                    break
                    
                except Exception as e:
                    if attempt == max_retries:
                        raise
                    time.sleep(pause_sec * attempt)
        
        return all_klines
    
    def get_symbol_info(self, symbol: str) -> Dict:
        """
        Obtiene información del símbolo desde Bitget.
        
        Args:
            symbol: Símbolo
            
        Returns:
            Diccionario con información del símbolo
        """
        bitget_symbol = self._map_symbol(symbol)
        endpoint = "/api/spot/v1/public/symbols"
        
        response = self.session.get(self.base_url + endpoint, timeout=15)
        response.raise_for_status()
        data = response.json()
        
        if data.get("code") != "00000":
            raise ValueError(f"Error API Bitget: {data.get('msg', 'Unknown error')}")
        
        symbols = data.get("data", [])
        for sym_info in symbols:
            if sym_info.get("symbol") == bitget_symbol:
                return {
                    "symbol": symbol,
                    "base_asset": sym_info.get("baseCoin"),
                    "quote_asset": sym_info.get("quoteCoin"),
                    "min_qty": float(sym_info.get("minTradeNum", 0)),
                    "max_qty": float(sym_info.get("maxTradeNum", 0)),
                    "tick_size": float(sym_info.get("priceScale", 0)),
                    "min_notional": float(sym_info.get("minTradeUSDT", 0)),
                    "status": sym_info.get("status"),
                    "exchange": "bitget",
                    "market": "futures",
                    "product_type": self.product_type,
                }
        
        raise ValueError(f"Símbolo no encontrado: {symbol}")


def create_bitget_collector(product_type: str = "umcbl") -> BitgetFuturesCollector:
    """
    Factory function para crear un collector de Bitget.
    
    Args:
        product_type: Tipo de producto Bitget
        
    Returns:
        Instancia de BitgetFuturesCollector
    """
    return BitgetFuturesCollector(product_type=product_type)


# Función de conveniencia para uso directo
def fetch_ohlcv(
    symbol: str, 
    interval: str, 
    start: int, 
    end: int,
    product_type: str = "umcbl"
) -> List[Dict]:
    """
    Función de conveniencia para descargar datos OHLCV.
    
    Args:
        symbol: Símbolo
        interval: Intervalo
        start: Timestamp inicio (ms)
        end: Timestamp fin (ms)
        product_type: Tipo de producto Bitget
        
    Returns:
        Lista de diccionarios con datos OHLCV
    """
    collector = create_bitget_collector(product_type=product_type)
    return collector.fetch_ohlcv(symbol, interval, start, end)


if __name__ == "__main__":
    # Test básico
    import sys
    from datetime import datetime, timezone, timedelta
    
    if len(sys.argv) < 2:
        print("Uso: python bitget_futures_collector.py <symbol>")
        sys.exit(1)
    
    symbol = sys.argv[1].upper()
    
    # Test con últimos 7 días
    end_time = datetime.now(timezone.utc)
    start_time = end_time - timedelta(days=7)
    
    start_ms = int(start_time.timestamp() * 1000)
    end_ms = int(end_time.timestamp() * 1000)
    
    print(f"Descargando {symbol} desde {start_time} hasta {end_time}")
    
    try:
        collector = create_bitget_collector()
        data = collector.fetch_ohlcv(symbol, "1h", start_ms, end_ms)
        
        print(f"Descargados {len(data)} registros")
        if data:
            print(f"Primer registro: {data[0]}")
            print(f"Último registro: {data[-1]}")
        
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)
