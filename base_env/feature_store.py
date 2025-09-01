# base_env\feature_store.py
# Features Foundation
from __future__ import annotations
import pandas as pd
import numpy as np
import talib
from typing import Dict, Any, Optional
from dataclasses import dataclass
import yaml
from pathlib import Path

@dataclass
class FeatureConfig:
    """Configuración tipada para features técnicos."""
    ema_periods: list[int]
    sma_periods: list[int]
    rsi_period: int
    macd: dict[str, int]
    atr_period: int
    bbands: dict[str, Any]
    supertrend: dict[str, Any]
    obv: bool
    vwap: bool
    stoch: dict[str, int]
    williams_r: int
    cci_period: int
    
    @classmethod
    def from_yaml(cls, path: str) -> "FeatureConfig":
        with open(path, 'r') as f:
            cfg = yaml.safe_load(f)['features']
        return cls(
            ema_periods=cfg.get('ema', [20, 50, 200]),
            sma_periods=cfg.get('sma', [20, 50, 200]),
            rsi_period=cfg.get('rsi', {}).get('period', 14),
            macd=cfg.get('macd', {'fast': 12, 'slow': 26, 'signal': 9}),
            atr_period=cfg.get('atr', {}).get('period', 14),
            bbands=cfg.get('bbands', {'period': 20, 'dev': 2.0}),
            supertrend=cfg.get('supertrend', {'period': 10, 'multiplier': 3.0}),
            obv=cfg.get('obv', True),
            vwap=cfg.get('vwap', True),
            stoch=cfg.get('stoch', {'k_period': 14, 'd_period': 3}),
            williams_r=cfg.get('williams_r', 14),
            cci_period=cfg.get('cci', 20)
        )

class IndicatorCalculator:
    """
    Calculadora completa de indicadores técnicos con soporte causal/no-causal.
    Optimizada para training masivo con vectorización numpy/talib.
    """
    
    def __init__(self, config: FeatureConfig, mode: str = "causal"):
        self.config = config
        self.mode = mode  # "causal" o "symmetric"
        
    def calculate_all(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calcula todos los indicadores configurados."""
        df = df.copy()
        
        # Validate required columns
        required = ['open', 'high', 'low', 'close', 'volume']
        if not all(col in df.columns for col in required):
            raise ValueError(f"Missing required columns. Need: {required}")
        
        # Convert to numpy for talib (más rápido)
        open_prices = df['open'].values
        high_prices = df['high'].values
        low_prices = df['low'].values
        close_prices = df['close'].values
        volume = df['volume'].values
        
        # Moving Averages
        df = self._add_moving_averages(df, close_prices)
        
        # Momentum Indicators  
        df = self._add_momentum_indicators(df, open_prices, high_prices, low_prices, close_prices)
        
        # Volume Indicators
        df = self._add_volume_indicators(df, close_prices, volume)
        
        # Volatility Indicators
        df = self._add_volatility_indicators(df, high_prices, low_prices, close_prices)
        
        # Trend Indicators
        df = self._add_trend_indicators(df, high_prices, low_prices, close_prices)
        
        # Custom Indicators
        df = self._add_custom_indicators(df, open_prices, high_prices, low_prices, close_prices, volume)
        
        # Handle NaNs based on mode
        if self.mode == "causal":
            # Forward fill para training (no lookahead)
            df = df.fillna(method='ffill')
        
        return df
    
    def _add_moving_averages(self, df: pd.DataFrame, close_prices: np.ndarray) -> pd.DataFrame:
        """EMAs y SMAs optimizadas."""
        for period in self.config.ema_periods:
            df[f'ta_ema_{period}'] = talib.EMA(close_prices, timeperiod=period)
            
        for period in self.config.sma_periods:
            df[f'ta_sma_{period}'] = talib.SMA(close_prices, timeperiod=period)
            
        # EMA crossovers (señales importantes)
        if 20 in self.config.ema_periods and 50 in self.config.ema_periods:
            df['ta_ema_cross_20_50'] = (df['ta_ema_20'] > df['ta_ema_50']).astype(int)
            
        return df
    
    def _add_momentum_indicators(self, df: pd.DataFrame, open_p: np.ndarray, 
                               high_p: np.ndarray, low_p: np.ndarray, close_p: np.ndarray) -> pd.DataFrame:
        """RSI, MACD, Stochastic, Williams %R, CCI."""
        
        # RSI
        df['ta_rsi'] = talib.RSI(close_p, timeperiod=self.config.rsi_period)
        df['ta_rsi_overbought'] = (df['ta_rsi'] > 70).astype(int)
        df['ta_rsi_oversold'] = (df['ta_rsi'] < 30).astype(int)
        
        # MACD
        macd, macd_signal, macd_hist = talib.MACD(
            close_p, 
            fastperiod=self.config.macd['fast'],
            slowperiod=self.config.macd['slow'], 
            signalperiod=self.config.macd['signal']
        )
        df['ta_macd'] = macd
        df['ta_macd_signal'] = macd_signal
        df['ta_macd_histogram'] = macd_hist
        df['ta_macd_cross'] = (macd > macd_signal).astype(int)
        
        # Stochastic
        slowk, slowd = talib.STOCH(
            high_p, low_p, close_p,
            fastk_period=self.config.stoch['k_period'],
            slowk_period=self.config.stoch['d_period'],
            slowd_period=self.config.stoch['d_period']
        )
        df['ta_stoch_k'] = slowk
        df['ta_stoch_d'] = slowd
        df['ta_stoch_overbought'] = (slowk > 80).astype(int)
        df['ta_stoch_oversold'] = (slowk < 20).astype(int)
        
        # Williams %R
        df['ta_williams_r'] = talib.WILLR(high_p, low_p, close_p, timeperiod=self.config.williams_r)
        
        # CCI
        df['ta_cci'] = talib.CCI(high_p, low_p, close_p, timeperiod=self.config.cci_period)
        
        return df
    
    def _add_volume_indicators(self, df: pd.DataFrame, close_prices: np.ndarray, volume: np.ndarray) -> pd.DataFrame:
        """OBV, VWAP, Volume Profile."""
        
        if self.config.obv:
            df['ta_obv'] = talib.OBV(close_prices, volume)
            df['ta_obv_sma'] = talib.SMA(df['ta_obv'].values, timeperiod=20)
        
        if self.config.vwap:
            # VWAP calculation
            typical_price = (df['high'] + df['low'] + df['close']) / 3
            vwap_num = (typical_price * df['volume']).cumsum()
            vwap_den = df['volume'].cumsum()
            df['ta_vwap'] = vwap_num / vwap_den
            df['ta_vwap_distance'] = (df['close'] - df['ta_vwap']) / df['ta_vwap']
        
        return df
    
    def _add_volatility_indicators(self, df: pd.DataFrame, high_prices: np.ndarray, 
                                 low_prices: np.ndarray, close_prices: np.ndarray) -> pd.DataFrame:
        """ATR, Bollinger Bands, Volatility."""
        
        # ATR
        df['ta_atr'] = talib.ATR(high_prices, low_prices, close_prices, timeperiod=self.config.atr_period)
        df['ta_atr_percent'] = df['ta_atr'] / df['close']
        
        # Bollinger Bands
        bb_upper, bb_middle, bb_lower = talib.BBANDS(
            close_prices,
            timeperiod=self.config.bbands['period'],
            nbdevup=self.config.bbands['dev'],
            nbdevdn=self.config.bbands['dev']
        )
        df['ta_bb_upper'] = bb_upper
        df['ta_bb_middle'] = bb_middle
        df['ta_bb_lower'] = bb_lower
        df['ta_bb_width'] = (bb_upper - bb_lower) / bb_middle
        df['ta_bb_position'] = (close_prices - bb_lower) / (bb_upper - bb_lower)
        
        # Volatility (rolling standard deviation)
        df['ta_volatility'] = df['close'].rolling(window=20).std()
        
        return df
    
    def _add_trend_indicators(self, df: pd.DataFrame, high_prices: np.ndarray, 
                            low_prices: np.ndarray, close_prices: np.ndarray) -> pd.DataFrame:
        """SuperTrend, Parabolic SAR, ADX."""
        
        # SuperTrend (custom implementation)
        df = self._calculate_supertrend(df, high_prices, low_prices, close_prices)
        
        # Parabolic SAR
        df['ta_sar'] = talib.SAR(high_prices, low_prices)
        
        # ADX (trend strength)
        df['ta_adx'] = talib.ADX(high_prices, low_prices, close_prices, timeperiod=14)
        df['ta_plus_di'] = talib.PLUS_DI(high_prices, low_prices, close_prices, timeperiod=14)
        df['ta_minus_di'] = talib.MINUS_DI(high_prices, low_prices, close_prices, timeperiod=14)
        
        return df
    
    def _calculate_supertrend(self, df: pd.DataFrame, high_prices: np.ndarray, 
                            low_prices: np.ndarray, close_prices: np.ndarray) -> pd.DataFrame:
        """SuperTrend indicator implementation."""
        period = self.config.supertrend['period']
        multiplier = self.config.supertrend['multiplier']
        
        # ATR for SuperTrend
        atr = talib.ATR(high_prices, low_prices, close_prices, timeperiod=period)
        hl2 = (high_prices + low_prices) / 2
        
        # Basic upper and lower bands
        upper_band = hl2 + (multiplier * atr)
        lower_band = hl2 - (multiplier * atr)
        
        # Initialize arrays
        supertrend = np.zeros_like(close_prices)
        trend = np.ones_like(close_prices)
        
        for i in range(1, len(close_prices)):
            # Update bands
            if upper_band[i] < upper_band[i-1] or close_prices[i-1] > upper_band[i-1]:
                upper_band[i] = upper_band[i]
            else:
                upper_band[i] = upper_band[i-1]
                
            if lower_band[i] > lower_band[i-1] or close_prices[i-1] < lower_band[i-1]:
                lower_band[i] = lower_band[i]
            else:
                lower_band[i] = lower_band[i-1]
            
            # Determine trend
            if close_prices[i] <= lower_band[i]:
                trend[i] = -1
            elif close_prices[i] >= upper_band[i]:
                trend[i] = 1
            else:
                trend[i] = trend[i-1]
            
            # Set SuperTrend value
            if trend[i] == 1:
                supertrend[i] = lower_band[i]
            else:
                supertrend[i] = upper_band[i]
        
        df['ta_supertrend'] = supertrend
        df['ta_supertrend_trend'] = trend
        df['ta_supertrend_signal'] = (trend == 1).astype(int)
        
        return df
    
    def _add_custom_indicators(self, df: pd.DataFrame, open_p: np.ndarray, 
                             high_p: np.ndarray, low_p: np.ndarray, 
                             close_p: np.ndarray, volume: np.ndarray) -> pd.DataFrame:
        """Indicadores custom relevantes para crypto trading."""
        
        # Price momentum
        df['ta_momentum_5'] = close_p / np.roll(close_p, 5) - 1
        df['ta_momentum_10'] = close_p / np.roll(close_p, 10) - 1
        df['ta_momentum_20'] = close_p / np.roll(close_p, 20) - 1
        
        # Volume momentum
        df['ta_volume_sma'] = talib.SMA(volume, timeperiod=20)
        df['ta_volume_ratio'] = volume / df['ta_volume_sma'].values
        
        # Price action patterns
        df['ta_body_size'] = np.abs(close_p - open_p) / open_p
        df['ta_upper_shadow'] = (high_p - np.maximum(open_p, close_p)) / np.maximum(open_p, close_p)
        df['ta_lower_shadow'] = (np.minimum(open_p, close_p) - low_p) / np.minimum(open_p, close_p)
        
        # Gap detection
        df['ta_gap_up'] = (open_p > np.roll(high_p, 1)).astype(int)
        df['ta_gap_down'] = (open_p < np.roll(low_p, 1)).astype(int)
        
        return df
    
    def get_feature_names(self) -> list[str]:
        """Retorna lista de todos los nombres de features que se generarán."""
        features = []
        
        # Moving averages
        for period in self.config.ema_periods:
            features.append(f'ta_ema_{period}')
        for period in self.config.sma_periods:
            features.append(f'ta_sma_{period}')
        
        # Momentum
        features.extend([
            'ta_rsi', 'ta_rsi_overbought', 'ta_rsi_oversold',
            'ta_macd', 'ta_macd_signal', 'ta_macd_histogram', 'ta_macd_cross',
            'ta_stoch_k', 'ta_stoch_d', 'ta_stoch_overbought', 'ta_stoch_oversold',
            'ta_williams_r', 'ta_cci'
        ])
        
        # Volume
        if self.config.obv:
            features.extend(['ta_obv', 'ta_obv_sma'])
        if self.config.vwap:
            features.extend(['ta_vwap', 'ta_vwap_distance'])
        
        # Volatility
        features.extend([
            'ta_atr', 'ta_atr_percent',
            'ta_bb_upper', 'ta_bb_middle', 'ta_bb_lower', 'ta_bb_width', 'ta_bb_position',
            'ta_volatility'
        ])
        
        # Trend
        features.extend([
            'ta_supertrend', 'ta_supertrend_trend', 'ta_supertrend_signal',
            'ta_sar', 'ta_adx', 'ta_plus_di', 'ta_minus_di'
        ])
        
        # Custom
        features.extend([
            'ta_momentum_5', 'ta_momentum_10', 'ta_momentum_20',
            'ta_volume_sma', 'ta_volume_ratio',
            'ta_body_size', 'ta_upper_shadow', 'ta_lower_shadow',
            'ta_gap_up', 'ta_gap_down'
        ])
        
        return features


def validate_features(df: pd.DataFrame) -> dict:
    """Valida calidad de features calculados."""
    issues = {
        'total_features': len([c for c in df.columns if c.startswith('ta_')]),
        'nan_features': len([c for c in df.columns if c.startswith('ta_') and df[c].isna().any()]),
        'infinite_features': 0,
        'constant_features': 0
    }
    
    for col in df.columns:
        if col.startswith('ta_'):
            if np.isinf(df[col]).any():
                issues['infinite_features'] += 1
            if df[col].nunique() == 1:
                issues['constant_features'] += 1
    
    return issues


if __name__ == "__main__":
    # Test básico
    config = FeatureConfig.from_yaml("config/features.yaml")
    calc = IndicatorCalculator(config, mode="causal")
    
    # Sample data
    np.random.seed(42)
    dates = pd.date_range('2020-01-01', periods=1000, freq='1H')
    df = pd.DataFrame({
        'timestamp': [int(d.timestamp() * 1000) for d in dates],
        'open': 100 + np.cumsum(np.random.randn(1000) * 0.1),
        'high': 0,
        'low': 0,
        'close': 0,
        'volume': np.random.randint(1000, 10000, 1000)
    })
    
    # Generate realistic OHLC
    df['close'] = df['open'] + np.random.randn(1000) * 0.5
    df['high'] = df[['open', 'close']].max(axis=1) + np.abs(np.random.randn(1000) * 0.3)
    df['low'] = df[['open', 'close']].min(axis=1) - np.abs(np.random.randn(1000) * 0.3)
    
    # Calculate features
    df_with_features = calc.calculate_all(df)
    validation = validate_features(df_with_features)
    
    print(f"Features calculados: {validation['total_features']}")
    print(f"Features con NaN: {validation['nan_features']}")
    print(f"Lista de features: {calc.get_feature_names()[:10]}...")  # Solo primeros 10