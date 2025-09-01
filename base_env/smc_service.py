from __future__ import annotations
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import yaml
from enum import Enum

class SwingType(Enum):
    HIGH = "high"
    LOW = "low"

@dataclass 
class SMCConfig:
    """Configuración para Smart Money Concepts."""
    # Swing detection
    swing_lookback_left: int
    swing_min_atr_k: float
    
    # BOS/CHOCH
    bos_buffer_atr_k: float
    bos_require_close: bool
    
    # Fair Value Gaps
    fvg_min_gap_atr_k: float
    fvg_mode: str  # "classic" or "refined"
    
    # Order Blocks
    ob_extend_until_mitigated: bool
    ob_max_age_bars: int
    
    # Liquidity
    liquidity_lookback: int
    liquidity_buffer_atr_k: float
    
    # Runtime
    mode: str  # "causal" or "symmetric"
    
    @classmethod
    def from_yaml(cls, path: str) -> "SMCConfig":
        with open(path, 'r') as f:
            cfg = yaml.safe_load(f)
        
        return cls(
            swing_lookback_left=cfg.get('swings', {}).get('lookback_left', 5),
            swing_min_atr_k=cfg.get('swings', {}).get('min_atr_k', 0.5),
            bos_buffer_atr_k=cfg.get('bos', {}).get('buffer_atr_k', 0.1),
            bos_require_close=cfg.get('bos', {}).get('require_close', True),
            fvg_min_gap_atr_k=cfg.get('fvg', {}).get('min_gap_atr_k', 0.1),
            fvg_mode=cfg.get('fvg', {}).get('mode', 'classic'),
            ob_extend_until_mitigated=cfg.get('order_blocks', {}).get('extend_until_mitigated', True),
            ob_max_age_bars=cfg.get('order_blocks', {}).get('max_age_bars', 1500),
            liquidity_lookback=cfg.get('liquidity', {}).get('lookback', 50),
            liquidity_buffer_atr_k=cfg.get('liquidity', {}).get('buffer_atr_k', 0.1),
            mode=cfg.get('runtime', {}).get('mode', 'causal')
        )

@dataclass
class Swing:
    """Representa un swing high/low."""
    index: int
    price: float
    type: SwingType
    timestamp: int
    atr_distance: float

@dataclass  
class OrderBlock:
    """Representa un Order Block."""
    start_idx: int
    end_idx: int
    high: float
    low: float
    type: str  # "bullish" or "bearish"
    origin_swing: Swing
    mitigated: bool = False
    mitigation_idx: Optional[int] = None

@dataclass
class FairValueGap:
    """Representa un Fair Value Gap."""
    start_idx: int
    top: float
    bottom: float
    type: str  # "bullish" or "bearish"
    filled: bool = False
    fill_idx: Optional[int] = None

class SMCDetector:
    """
    Implementación completa de Smart Money Concepts.
    Detecta swings, BOS/CHOCH, Order Blocks, FVGs y zonas de liquidez.
    """
    
    def __init__(self, config: SMCConfig):
        self.config = config
        
    def detect_all(self, df: pd.DataFrame) -> pd.DataFrame:
        """Pipeline completo de detección SMC."""
        df = df.copy()
        
        # Validar columnas requeridas
        required = ['open', 'high', 'low', 'close', 'volume']
        if not all(col in df.columns for col in required):
            raise ValueError(f"Missing columns. Required: {required}")
        
        # Calcular ATR si no existe
        if 'ta_atr' not in df.columns:
            df['ta_atr'] = self._calculate_atr(df)
        
        # 1. Detectar swings
        swings = self._detect_swings(df)
        df = self._add_swing_columns(df, swings)
        
        # 2. Detectar BOS/CHOCH
        df = self._detect_bos_choch(df, swings)
        
        # 3. Detectar Order Blocks  
        order_blocks = self._detect_order_blocks(df, swings)
        df = self._add_order_block_columns(df, order_blocks)
        
        # 4. Detectar Fair Value Gaps
        fvgs = self._detect_fair_value_gaps(df)
        df = self._add_fvg_columns(df, fvgs)
        
        # 5. Detectar zonas de liquidez
        df = self._detect_liquidity_zones(df, swings)
        
        # 6. Añadir features derivados
        df = self._add_derived_features(df)
        
        return df
    
    def _calculate_atr(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calcula ATR si no está disponible."""
        high_low = df['high'] - df['low']
        high_close = np.abs(df['high'] - df['close'].shift(1))
        low_close = np.abs(df['low'] - df['close'].shift(1))
        
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        return tr.rolling(window=period).mean()
    
    def _detect_swings(self, df: pd.DataFrame) -> List[Swing]:
        """Detecta swing highs y lows usando método de ventana deslizante."""
        swings = []
        lookback = self.config.swing_lookback_left
        
        # Usar rolling windows para encontrar máximos y mínimos locales
        window_size = 2 * lookback + 1
        
        for i in range(lookback, len(df) - lookback):
            window_high = df['high'].iloc[i-lookback:i+lookback+1]
            window_low = df['low'].iloc[i-lookback:i+lookback+1]
            
            current_high = df['high'].iloc[i]
            current_low = df['low'].iloc[i]
            atr = df['ta_atr'].iloc[i]
            
            # Swing High
            if current_high == window_high.max():
                # Verificar que supere el threshold mínimo ATR
                if self._is_significant_swing(window_high, current_high, atr, SwingType.HIGH):
                    swings.append(Swing(
                        index=i,
                        price=current_high,
                        type=SwingType.HIGH,
                        timestamp=int(df['timestamp'].iloc[i]) if 'timestamp' in df.columns else i,
                        atr_distance=atr * self.config.swing_min_atr_k
                    ))
            
            # Swing Low  
            if current_low == window_low.min():
                if self._is_significant_swing(window_low, current_low, atr, SwingType.LOW):
                    swings.append(Swing(
                        index=i,
                        price=current_low,
                        type=SwingType.LOW,
                        timestamp=int(df['timestamp'].iloc[i]) if 'timestamp' in df.columns else i,
                        atr_distance=atr * self.config.swing_min_atr_k
                    ))
        
        return swings
    
    def _is_significant_swing(self, window: pd.Series, current_price: float, 
                            atr: float, swing_type: SwingType) -> bool:
        """Verifica si el swing supera el threshold mínimo de ATR."""
        if atr <= 0:
            return False
            
        min_distance = atr * self.config.swing_min_atr_k
        
        if swing_type == SwingType.HIGH:
            # Para swing high, verificar distancia con el segundo más alto
            sorted_highs = window.nlargest(2)
            if len(sorted_highs) > 1:
                return current_price - sorted_highs.iloc[1] >= min_distance
        else:
            # Para swing low, verificar distancia con el segundo más bajo
            sorted_lows = window.nsmallest(2)
            if len(sorted_lows) > 1:
                return sorted_lows.iloc[1] - current_price >= min_distance
                
        return True
    
    def _add_swing_columns(self, df: pd.DataFrame, swings: List[Swing]) -> pd.DataFrame:
        """Añade columnas de swings al DataFrame."""
        df['smc_swing_high'] = 0
        df['smc_swing_low'] = 0
        df['smc_swing_high_price'] = np.nan
        df['smc_swing_low_price'] = np.nan
        
        for swing in swings:
            if swing.type == SwingType.HIGH:
                df.loc[swing.index, 'smc_swing_high'] = 1
                df.loc[swing.index, 'smc_swing_high_price'] = swing.price
            else:
                df.loc[swing.index, 'smc_swing_low'] = 1
                df.loc[swing.index, 'smc_swing_low_price'] = swing.price
        
        return df
    
    def _detect_bos_choch(self, df: pd.DataFrame, swings: List[Swing]) -> pd.DataFrame:
        """Detecta Break of Structure (BOS) y Change of Character (CHOCH)."""
        df['smc_bos_bullish'] = 0
        df['smc_bos_bearish'] = 0
        df['smc_choch_bullish'] = 0
        df['smc_choch_bearish'] = 0
        
        # Separar swings por tipo
        swing_highs = [s for s in swings if s.type == SwingType.HIGH]
        swing_lows = [s for s in swings if s.type == SwingType.LOW]
        
        buffer_atr = self.config.bos_buffer_atr_k
        
        # Detectar BOS Bullish (ruptura de swing high previo)
        for i, current_high in enumerate(swing_highs[1:], 1):
            prev_high = swing_highs[i-1]
            atr = df['ta_atr'].iloc[current_high.index]
            threshold = prev_high.price + (atr * buffer_atr)
            
            if current_high.price > threshold:
                # Verificar que el precio cierre por encima si está configurado
                if self.config.bos_require_close:
                    close_price = df['close'].iloc[current_high.index]
                    if close_price > threshold:
                        df.loc[current_high.index, 'smc_bos_bullish'] = 1
                else:
                    df.loc[current_high.index, 'smc_bos_bullish'] = 1
        
        # Detectar BOS Bearish (ruptura de swing low previo)
        for i, current_low in enumerate(swing_lows[1:], 1):
            prev_low = swing_lows[i-1]
            atr = df['ta_atr'].iloc[current_low.index]
            threshold = prev_low.price - (atr * buffer_atr)
            
            if current_low.price < threshold:
                if self.config.bos_require_close:
                    close_price = df['close'].iloc[current_low.index]
                    if close_price < threshold:
                        df.loc[current_low.index, 'smc_bos_bearish'] = 1
                else:
                    df.loc[current_low.index, 'smc_bos_bearish'] = 1
        
        # CHOCH: cambio de carácter (higher low después de lower high, etc.)
        self._detect_choch(df, swing_highs, swing_lows)
        
        return df
    
    def _detect_choch(self, df: pd.DataFrame, swing_highs: List[Swing], swing_lows: List[Swing]):
        """Detecta Change of Character patterns."""
        # Combinar y ordenar todos los swings por índice
        all_swings = sorted(swing_highs + swing_lows, key=lambda x: x.index)
        
        for i in range(2, len(all_swings)):
            prev2 = all_swings[i-2]
            prev1 = all_swings[i-1]
            current = all_swings[i]
            
            # CHOCH Bullish: Lower High + Higher Low
            if (prev2.type == SwingType.HIGH and prev1.type == SwingType.LOW and 
                current.type == SwingType.HIGH):
                
                if (current.price < prev2.price and  # Lower High
                    prev1.price > swing_lows[max(0, swing_lows.index(prev1)-1)].price):  # Higher Low
                    df.loc[current.index, 'smc_choch_bullish'] = 1
            
            # CHOCH Bearish: Higher Low + Lower High  
            if (prev2.type == SwingType.LOW and prev1.type == SwingType.HIGH and 
                current.type == SwingType.LOW):
                
                if (current.price > prev2.price and  # Higher Low
                    prev1.price < swing_highs[max(0, swing_highs.index(prev1)-1)].price):  # Lower High
                    df.loc[current.index, 'smc_choch_bearish'] = 1
    
    def _detect_order_blocks(self, df: pd.DataFrame, swings: List[Swing]) -> List[OrderBlock]:
        """Detecta Order Blocks basados en swings y volumen."""
        order_blocks = []
        
        for swing in swings:
            # Buscar el candle de origen (último candle opuesto antes del swing)
            origin_candle = self._find_origin_candle(df, swing)
            
            if origin_candle is not None:
                ob_type = "bullish" if swing.type == SwingType.LOW else "bearish"
                
                # Definir el rango del Order Block
                if ob_type == "bullish":
                    ob_high = df['high'].iloc[origin_candle]
                    ob_low = df['low'].iloc[origin_candle]
                else:
                    ob_high = df['high'].iloc[origin_candle] 
                    ob_low = df['low'].iloc[origin_candle]
                
                order_block = OrderBlock(
                    start_idx=origin_candle,
                    end_idx=swing.index,
                    high=ob_high,
                    low=ob_low,
                    type=ob_type,
                    origin_swing=swing
                )
                
                order_blocks.append(order_block)
        
        # Verificar mitigación de Order Blocks
        self._check_ob_mitigation(df, order_blocks)
        
        return order_blocks
    
    def _find_origin_candle(self, df: pd.DataFrame, swing: Swing) -> Optional[int]:
        """Encuentra el candle de origen para un Order Block."""
        lookback = min(20, swing.index)  # Buscar máximo 20 candles atrás
        
        if swing.type == SwingType.LOW:
            # Para swing low, buscar el último candle bearish antes del movimiento
            for i in range(swing.index - 1, swing.index - lookback - 1, -1):
                if i < 0:
                    break
                if df['close'].iloc[i] < df['open'].iloc[i]:  # Candle bearish
                    return i
        else:
            # Para swing high, buscar el último candle bullish  
            for i in range(swing.index - 1, swing.index - lookback - 1, -1):
                if i < 0:
                    break
                if df['close'].iloc[i] > df['open'].iloc[i]:  # Candle bullish
                    return i
        
        return None
    
    def _check_ob_mitigation(self, df: pd.DataFrame, order_blocks: List[OrderBlock]):
        """Verifica si los Order Blocks han sido mitigados."""
        for ob in order_blocks:
            if ob.mitigated:
                continue
                
            # Buscar mitigación después del swing
            max_search = min(len(df), ob.end_idx + self.config.ob_max_age_bars)
            
            for i in range(ob.end_idx + 1, max_search):
                if ob.type == "bullish":
                    # OB bullish se mitiga cuando el precio rompe por debajo
                    if df['low'].iloc[i] < ob.low:
                        ob.mitigated = True
                        ob.mitigation_idx = i
                        break
                else:
                    # OB bearish se mitiga cuando el precio rompe por encima
                    if df['high'].iloc[i] > ob.high:
                        ob.mitigated = True
                        ob.mitigation_idx = i
                        break
    
    def _add_order_block_columns(self, df: pd.DataFrame, order_blocks: List[OrderBlock]) -> pd.DataFrame:
        """Añade columnas de Order Blocks al DataFrame."""
        df['smc_ob_bullish'] = 0
        df['smc_ob_bearish'] = 0
        df['smc_ob_active_bullish'] = 0
        df['smc_ob_active_bearish'] = 0
        df['smc_ob_high'] = np.nan
        df['smc_ob_low'] = np.nan
        
        for ob in order_blocks:
            # Marcar origen del Order Block
            if ob.type == "bullish":
                df.loc[ob.start_idx, 'smc_ob_bullish'] = 1
                df.loc[ob.start_idx, 'smc_ob_high'] = ob.high
                df.loc[ob.start_idx, 'smc_ob_low'] = ob.low
                
                # Marcar como activo si no está mitigado
                if not ob.mitigated and self.config.ob_extend_until_mitigated:
                    end_idx = ob.mitigation_idx if ob.mitigation_idx else min(len(df), ob.end_idx + self.config.ob_max_age_bars)
                    df.loc[ob.start_idx:end_idx, 'smc_ob_active_bullish'] = 1
                    
            else:
                df.loc[ob.start_idx, 'smc_ob_bearish'] = 1
                df.loc[ob.start_idx, 'smc_ob_high'] = ob.high
                df.loc[ob.start_idx, 'smc_ob_low'] = ob.low
                
                if not ob.mitigated and self.config.ob_extend_until_mitigated:
                    end_idx = ob.mitigation_idx if ob.mitigation_idx else min(len(df), ob.end_idx + self.config.ob_max_age_bars)
                    df.loc[ob.start_idx:end_idx, 'smc_ob_active_bearish'] = 1
        
        return df
    
    def _detect_fair_value_gaps(self, df: pd.DataFrame) -> List[FairValueGap]:
        """Detecta Fair Value Gaps (FVGs)."""
        fvgs = []
        
        for i in range(2, len(df)):
            candle1 = i - 2  # Primer candle
            candle2 = i - 1  # Candle del medio  
            candle3 = i      # Último candle
            
            atr = df['ta_atr'].iloc[i]
            min_gap = atr * self.config.fvg_min_gap_atr_k
            
            # FVG Bullish: gap entre candle1.high y candle3.low
            if (df['low'].iloc[candle3] > df['high'].iloc[candle1] and
                df['low'].iloc[candle3] - df['high'].iloc[candle1] >= min_gap):
                
                fvg = FairValueGap(
                    start_idx=candle1,
                    top=df['low'].iloc[candle3],
                    bottom=df['high'].iloc[candle1],
                    type="bullish"
                )
                fvgs.append(fvg)
            
            # FVG Bearish: gap entre candle1.low y candle3.high
            elif (df['high'].iloc[candle3] < df['low'].iloc[candle1] and
                  df['low'].iloc[candle1] - df['high'].iloc[candle3] >= min_gap):
                
                fvg = FairValueGap(
                    start_idx=candle1,
                    top=df['low'].iloc[candle1],
                    bottom=df['high'].iloc[candle3],
                    type="bearish"
                )
                fvgs.append(fvg)
        
        # Verificar fill de FVGs
        self._check_fvg_fill(df, fvgs)
        
        return fvgs
    
    def _check_fvg_fill(self, df: pd.DataFrame, fvgs: List[FairValueGap]):
        """Verifica si los FVGs han sido llenados."""
        for fvg in fvgs:
            if fvg.filled:
                continue
                
            # Buscar fill después de la formación
            for i in range(fvg.start_idx + 3, len(df)):
                if fvg.type == "bullish":
                    # FVG bullish se llena cuando el precio baja al rango
                    if df['low'].iloc[i] <= fvg.bottom:
                        fvg.filled = True
                        fvg.fill_idx = i
                        break
                else:
                    # FVG bearish se llena cuando el precio sube al rango
                    if df['high'].iloc[i] >= fvg.top:
                        fvg.filled = True
                        fvg.fill_idx = i
                        break
    
    def _add_fvg_columns(self, df: pd.DataFrame, fvgs: List[FairValueGap]) -> pd.DataFrame:
        """Añade columnas de Fair Value Gaps al DataFrame."""
        df['smc_fvg_bullish'] = 0
        df['smc_fvg_bearish'] = 0
        df['smc_fvg_active_bullish'] = 0
        df['smc_fvg_active_bearish'] = 0
        df['smc_fvg_top'] = np.nan
        df['smc_fvg_bottom'] = np.nan
        
        for fvg in fvgs:
            if fvg.type == "bullish":
                df.loc[fvg.start_idx, 'smc_fvg_bullish'] = 1
                df.loc[fvg.start_idx, 'smc_fvg_top'] = fvg.top
                df.loc[fvg.start_idx, 'smc_fvg_bottom'] = fvg.bottom
                
                # Marcar como activo hasta que se llene
                if not fvg.filled:
                    df.loc[fvg.start_idx:, 'smc_fvg_active_bullish'] = 1
                else:
                    df.loc[fvg.start_idx:fvg.fill_idx, 'smc_fvg_active_bullish'] = 1
                    
            else:
                df.loc[fvg.start_idx, 'smc_fvg_bearish'] = 1
                df.loc[fvg.start_idx, 'smc_fvg_top'] = fvg.top
                df.loc[fvg.start_idx, 'smc_fvg_bottom'] = fvg.bottom
                
                if not fvg.filled:
                    df.loc[fvg.start_idx:, 'smc_fvg_active_bearish'] = 1
                else:
                    df.loc[fvg.start_idx:fvg.fill_idx, 'smc_fvg_active_bearish'] = 1
        
        return df
    
    def _detect_liquidity_zones(self, df: pd.DataFrame, swings: List[Swing]) -> pd.DataFrame:
        """Detecta zonas de liquidez basadas en swing highs/lows."""
        df['smc_liquidity_high'] = 0
        df['smc_liquidity_low'] = 0
        df['smc_liquidity_swept_high'] = 0
        df['smc_liquidity_swept_low'] = 0
        
        lookback = self.config.liquidity_lookback
        buffer_atr = self.config.liquidity_buffer_atr_k
        
        swing_highs = [s for s in swings if s.type == SwingType.HIGH]
        swing_lows = [s for s in swings if s.type == SwingType.LOW]
        
        # Marcar zonas de liquidez en swing highs
        for swing in swing_highs:
            df.loc[swing.index, 'smc_liquidity_high'] = 1
            
            # Verificar si fue barrida posteriormente
            atr = df['ta_atr'].iloc[swing.index]
            threshold = swing.price + (atr * buffer_atr)
            
            end_search = min(len(df), swing.index + lookback)
            for i in range(swing.index + 1, end_search):
                if df['high'].iloc[i] > threshold:
                    df.loc[i, 'smc_liquidity_swept_high'] = 1
                    break
        
        # Marcar zonas de liquidez en swing lows
        for swing in swing_lows:
            df.loc[swing.index, 'smc_liquidity_low'] = 1
            
            # Verificar si fue barrida posteriormente
            atr = df['ta_atr'].iloc[swing.index]
            threshold = swing.price - (atr * buffer_atr)
            
            end_search = min(len(df), swing.index + lookback)
            for i in range(swing.index + 1, end_search):
                if df['low'].iloc[i] < threshold:
                    df.loc[i, 'smc_liquidity_swept_low'] = 1
                    break
        
        return df
    
    def _add_derived_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Añade features derivados y combinaciones SMC."""
        # Confluencias
        df['smc_bullish_confluence'] = (
            df['smc_ob_active_bullish'] + 
            df['smc_fvg_active_bullish'] + 
            df['smc_bos_bullish']
        )
        
        df['smc_bearish_confluence'] = (
            df['smc_ob_active_bearish'] + 
            df['smc_fvg_active_bearish'] + 
            df['smc_bos_bearish']  
        )
        
        # Señales combinadas
        df['smc_entry_long'] = (
            (df['smc_ob_active_bullish'] == 1) & 
            (df['smc_bos_bullish'] == 1) |
            (df['smc_fvg_active_bullish'] == 1) & 
            (df['smc_liquidity_swept_low'] == 1)
        ).astype(int)
        
        df['smc_entry_short'] = (
            (df['smc_ob_active_bearish'] == 1) & 
            (df['smc_bos_bearish'] == 1) |
            (df['smc_fvg_active_bearish'] == 1) & 
            (df['smc_liquidity_swept_high'] == 1)
        ).astype(int)
        
        return df
    
    def get_smc_feature_names(self) -> List[str]:
        """Retorna lista de todos los nombres de features SMC generados."""
        return [
            # Swings
            'smc_swing_high', 'smc_swing_low',
            'smc_swing_high_price', 'smc_swing_low_price',
            
            # BOS/CHOCH
            'smc_bos_bullish', 'smc_bos_bearish',
            'smc_choch_bullish', 'smc_choch_bearish',
            
            # Order Blocks
            'smc_ob_bullish', 'smc_ob_bearish',
            'smc_ob_active_bullish', 'smc_ob_active_bearish',
            'smc_ob_high', 'smc_ob_low',
            
            # Fair Value Gaps
            'smc_fvg_bullish', 'smc_fvg_bearish',
            'smc_fvg_active_bullish', 'smc_fvg_active_bearish',
            'smc_fvg_top', 'smc_fvg_bottom',
            
            # Liquidity
            'smc_liquidity_high', 'smc_liquidity_low',
            'smc_liquidity_swept_high', 'smc_liquidity_swept_low',
            
            # Derived
            'smc_bullish_confluence', 'smc_bearish_confluence',
            'smc_entry_long', 'smc_entry_short'
        ]


def validate_smc_features(df: pd.DataFrame) -> Dict[str, any]:
    """Valida features SMC calculados."""
    smc_cols = [c for c in df.columns if c.startswith('smc_')]
    
    validation = {
        'total_smc_features': len(smc_cols),
        'swing_count': df['smc_swing_high'].sum() + df['smc_swing_low'].sum(),
        'bos_count': df['smc_bos_bullish'].sum() + df['smc_bos_bearish'].sum(),
        'ob_count': df['smc_ob_bullish'].sum() + df['smc_ob_bearish'].sum(),
        'fvg_count': df['smc_fvg_bullish'].sum() + df['smc_fvg_bearish'].sum(),
        'entry_signals': df['smc_entry_long'].sum() + df['smc_entry_short'].sum(),
        'nan_features': len([c for c in smc_cols if df[c].isna().any()])
    }
    
    return validation


if __name__ == "__main__":
    # Test básico
    config = SMCConfig.from_yaml("config/smc.yaml")
    detector = SMCDetector(config)
    
    # Sample data más realista
    np.random.seed(42)
    n_bars = 2000
    
    # Generate trending price data
    trend = np.cumsum(np.random.randn(n_bars) * 0.001)
    noise = np.random.randn(n_bars) * 0.002
    base_price = 50000
    
    df = pd.DataFrame({
        'timestamp': range(n_bars),
        'open': base_price + trend + noise,
        'high': 0,
        'low': 0, 
        'close': 0,
        'volume': np.random.randint(1000, 10000, n_bars)
    })
    
    # Generate realistic OHLC with gaps and trends
    for i in range(n_bars):
        open_price = df['open'].iloc[i]
        close_change = np.random.randn() * 50
        close_price = open_price + close_change
        
        high_ext = np.abs(np.random.randn()) * 30
        low_ext = np.abs(np.random.randn()) * 30
        
        df.loc[i, 'close'] = close_price
        df.loc[i, 'high'] = max(open_price, close_price) + high_ext
        df.loc[i, 'low'] = min(open_price, close_price) - low_ext
    
    # Detectar SMC
    df_with_smc = detector.detect_all(df)
    validation = validate_smc_features(df_with_smc)
    
    print("=== SMC Detection Results ===")
    for key, value in validation.items():
        print(f"{key}: {value}")
    
    print(f"\nSample SMC features:")
    smc_cols = [c for c in df_with_smc.columns if c.startswith('smc_')][:10]
    print(df_with_smc[smc_cols].head())