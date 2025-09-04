# base_env/actions/leverage_timeframe_selector.py
# Sistema de selección dinámica de leverage y timeframe

from __future__ import annotations
from typing import Dict, Any, List, Tuple, Optional
from dataclasses import dataclass
import numpy as np


@dataclass
class LeverageTimeframeAction:
    """Acción que incluye leverage y timeframe seleccionados"""
    action_type: int  # 0=hold, 1=close_all, 2=no_open, 3=force_long, 4=force_short
    leverage: float
    timeframe: str
    confidence: float


class LeverageTimeframeSelector:
    """Selector dinámico de leverage y timeframe basado en condiciones del mercado"""
    
    def __init__(self, 
                 available_leverages: List[float],
                 available_timeframes: List[str],
                 symbol_config: Dict[str, Any]):
        self.available_leverages = available_leverages
        self.available_timeframes = available_timeframes
        self.symbol_config = symbol_config
        self.leverage_config = symbol_config.get("leverage", {})
        
        # Configuración de leverage (manejar tanto dict como objeto)
        if hasattr(self.leverage_config, 'min'):
            # Es un objeto dataclass
            self.min_leverage = getattr(self.leverage_config, 'min', 2.0)
            self.max_leverage = getattr(self.leverage_config, 'max', 25.0)
            self.leverage_step = getattr(self.leverage_config, 'step', 1.0)
        else:
            # Es un diccionario
            self.min_leverage = self.leverage_config.get("min", 2.0)
            self.max_leverage = self.leverage_config.get("max", 25.0)
            self.leverage_step = self.leverage_config.get("step", 1.0)
        
    def select_leverage_timeframe(self, 
                                 action: int,
                                 market_conditions: Dict[str, Any],
                                 confidence: float) -> LeverageTimeframeAction:
        """
        Selecciona leverage y timeframe basado en condiciones del mercado
        
        Args:
            action: Acción base del RL (0-4)
            market_conditions: Condiciones del mercado (volatilidad, tendencia, etc.)
            confidence: Confianza en la señal (0.0-1.0)
            
        Returns:
            Acción con leverage y timeframe seleccionados
        """
        # Seleccionar leverage
        leverage = self._select_leverage(market_conditions, confidence)
        
        # Seleccionar timeframe
        timeframe = self._select_timeframe(market_conditions, confidence)
        
        return LeverageTimeframeAction(
            action_type=action,
            leverage=leverage,
            timeframe=timeframe,
            confidence=confidence
        )
    
    def _select_leverage(self, market_conditions: Dict[str, Any], confidence: float) -> float:
        """Selecciona leverage basado en condiciones del mercado"""
        volatility = market_conditions.get("volatility", 0.5)
        trend_strength = market_conditions.get("trend_strength", 0.5)
        market_regime = market_conditions.get("regime", "normal")  # normal, trending, ranging
        
        # Leverage base
        base_leverage = 3.0
        
        # Ajustes basados en volatilidad
        if volatility < 0.3:  # Baja volatilidad
            volatility_multiplier = 1.2
        elif volatility > 0.7:  # Alta volatilidad
            volatility_multiplier = 0.8
        else:  # Volatilidad normal
            volatility_multiplier = 1.0
        
        # Ajustes basados en confianza
        confidence_multiplier = 0.8 + (confidence * 0.4)  # 0.8 a 1.2
        
        # Ajustes basados en régimen del mercado
        if market_regime == "trending":
            regime_multiplier = 1.1  # Más leverage en tendencias claras
        elif market_regime == "ranging":
            regime_multiplier = 0.9  # Menos leverage en rangos
        else:
            regime_multiplier = 1.0
        
        # Ajustes basados en fuerza de tendencia
        trend_multiplier = 0.9 + (trend_strength * 0.2)  # 0.9 a 1.1
        
        # Calcular leverage final
        final_leverage = (base_leverage * 
                         volatility_multiplier * 
                         confidence_multiplier * 
                         regime_multiplier * 
                         trend_multiplier)
        
        # Clamp al rango disponible
        final_leverage = max(self.min_leverage, min(self.max_leverage, final_leverage))
        
        # Redondear al step más cercano
        final_leverage = round(final_leverage / self.leverage_step) * self.leverage_step
        
        return final_leverage
    
    def _select_timeframe(self, market_conditions: Dict[str, Any], confidence: float) -> str:
        """Selecciona timeframe basado en condiciones del mercado"""
        volatility = market_conditions.get("volatility", 0.5)
        trend_strength = market_conditions.get("trend_strength", 0.5)
        market_regime = market_conditions.get("regime", "normal")
        
        # Lógica de selección de timeframe
        if market_regime == "trending" and trend_strength > 0.7:
            # Mercado en tendencia fuerte -> usar timeframes más largos
            if "5m" in self.available_timeframes:
                return "5m"
            else:
                return self.available_timeframes[0]
        
        elif volatility > 0.7:
            # Alta volatilidad -> usar timeframes más cortos para mejor control
            if "1m" in self.available_timeframes:
                return "1m"
            else:
                return self.available_timeframes[0]
        
        elif confidence > 0.8:
            # Alta confianza -> usar timeframes más largos
            if "5m" in self.available_timeframes:
                return "5m"
            else:
                return self.available_timeframes[0]
        
        else:
            # Condiciones normales -> usar timeframe por defecto
            return self.available_timeframes[0]
    
    def get_leverage_range(self) -> Tuple[float, float]:
        """Devuelve el rango de leverage disponible"""
        return self.min_leverage, self.max_leverage
    
    def get_available_timeframes(self) -> List[str]:
        """Devuelve los timeframes disponibles"""
        return self.available_timeframes.copy()
    
    def calculate_market_conditions(self, obs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Calcula condiciones del mercado basado en observaciones
        
        Args:
            obs: Observaciones del entorno
            
        Returns:
            Diccionario con condiciones del mercado
        """
        features = obs.get("features", {})
        analysis = obs.get("analysis", {})
        
        # Calcular volatilidad (usando ATR normalizado)
        volatility = 0.5  # Default
        for tf in ["1m", "5m", "15m", "1h"]:
            if tf in features:
                atr = features[tf].get("atr14", 0.0)
                close = obs.get("tfs", {}).get(tf, {}).get("close", 1.0)
                if close > 0 and atr is not None and atr > 0:
                    atr_pct = (atr / close) * 100.0
                    volatility = max(volatility, min(1.0, atr_pct / 2.0))  # Normalizar
        
        # Calcular fuerza de tendencia
        trend_strength = 0.5  # Default
        side_hint = analysis.get("side_hint", 0.0)
        confidence = analysis.get("confidence", 0.0)
        trend_strength = abs(side_hint) * confidence
        
        # Determinar régimen del mercado
        regime = "normal"
        if trend_strength > 0.7:
            regime = "trending"
        elif volatility > 0.6 and trend_strength < 0.3:
            regime = "ranging"
        
        return {
            "volatility": volatility,
            "trend_strength": trend_strength,
            "regime": regime,
            "confidence": confidence
        }
