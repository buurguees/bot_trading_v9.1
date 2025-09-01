# base_env\risck_manager-py

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from enum import Enum
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import yaml

class RiskLevel(Enum):
    LOW = "low"
    MEDIUM = "medium" 
    HIGH = "high"
    CRITICAL = "critical"

class MarketRegime(Enum):
    TRENDING = "trending"
    SIDEWAYS = "sideways"
    HIGH_VOLATILITY = "high_vol"
    LOW_VOLATILITY = "low_vol"

@dataclass
class RiskConfig:
    """Configuración completa de gestión de riesgo."""
    # Position Sizing
    base_risk_per_trade_pct: float
    max_risk_per_trade_pct: float
    min_position_usdt: float
    max_position_usdt: float
    
    # Portfolio Risk
    max_portfolio_risk_pct: float
    max_daily_loss_pct: float
    max_drawdown_pct: float
    max_open_positions: int
    max_correlated_positions: int
    
    # Leverage & Futures
    max_leverage_spot: float
    max_leverage_futures: float
    leverage_scaling_factor: float
    
    # Stop Loss & Take Profit
    atr_sl_multiplier: float
    atr_tp_multiplier: float
    min_risk_reward_ratio: float
    trailing_stop_activation_rr: float
    
    # Circuit Breakers
    max_latency_ms: int
    max_slippage_bp: float
    max_consecutive_losses: int
    cooldown_after_breaker_minutes: int
    
    # Dynamic Scaling
    volatility_scaling: bool
    regime_scaling: bool
    correlation_scaling: bool
    kelly_sizing: bool
    
    @classmethod
    def from_yaml(cls, path: str) -> "RiskConfig":
        with open(path, 'r') as f:
            cfg = yaml.safe_load(f)
        
        return cls(
            base_risk_per_trade_pct=cfg.get('sizing', {}).get('risk_per_trade_pct', 0.01),
            max_risk_per_trade_pct=cfg.get('sizing', {}).get('max_risk_per_trade_pct', 0.03),
            min_position_usdt=cfg.get('sizing', {}).get('min_notional_usdt', 10.0),
            max_position_usdt=cfg.get('sizing', {}).get('max_position_usdt', 10000.0),
            
            max_portfolio_risk_pct=cfg.get('portfolio', {}).get('max_risk_pct', 0.15),
            max_daily_loss_pct=cfg.get('portfolio', {}).get('max_daily_loss_pct', 0.05),
            max_drawdown_pct=cfg.get('portfolio', {}).get('max_drawdown_pct', 0.20),
            max_open_positions=cfg.get('portfolio', {}).get('max_open_positions', 5),
            max_correlated_positions=cfg.get('portfolio', {}).get('max_correlated_positions', 2),
            
            max_leverage_spot=cfg.get('futures', {}).get('max_leverage_spot', 1.0),
            max_leverage_futures=cfg.get('futures', {}).get('max_leverage_futures', 3.0),
            leverage_scaling_factor=cfg.get('futures', {}).get('leverage_scaling_factor', 0.8),
            
            atr_sl_multiplier=cfg.get('tp_sl', {}).get('atr_sl_k', 1.5),
            atr_tp_multiplier=cfg.get('tp_sl', {}).get('atr_tp_k', 3.0),
            min_risk_reward_ratio=cfg.get('tp_sl', {}).get('min_rr', 1.5),
            trailing_stop_activation_rr=cfg.get('tp_sl', {}).get('trailing_activation_rr', 1.0),
            
            max_latency_ms=cfg.get('circuit_breakers', {}).get('latency_ms', 1500),
            max_slippage_bp=cfg.get('circuit_breakers', {}).get('max_slippage_bp', 50.0),
            max_consecutive_losses=cfg.get('circuit_breakers', {}).get('max_consecutive_losses', 5),
            cooldown_after_breaker_minutes=cfg.get('circuit_breakers', {}).get('cooldown_minutes', 15),
            
            volatility_scaling=cfg.get('dynamic', {}).get('volatility_scaling', True),
            regime_scaling=cfg.get('dynamic', {}).get('regime_scaling', True),
            correlation_scaling=cfg.get('dynamic', {}).get('correlation_scaling', True),
            kelly_sizing=cfg.get('dynamic', {}).get('kelly_sizing', False)
        )

@dataclass
class PositionRequest:
    """Solicitud de posición para evaluación de riesgo."""
    symbol: str
    side: str  # "long" or "short"
    entry_price: float
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    strategy_confidence: float = 0.5  # 0-1
    market_regime: MarketRegime = MarketRegime.TRENDING
    correlation_group: Optional[str] = None

@dataclass
class RiskAssessment:
    """Resultado de evaluación de riesgo."""
    approved: bool
    position_size_usdt: float
    leverage: float
    stop_loss: float
    take_profit: float
    risk_reward_ratio: float
    portfolio_risk_pct: float
    risk_level: RiskLevel
    rejection_reason: Optional[str] = None
    warnings: List[str] = None

@dataclass
class CircuitBreakerState:
    """Estado de los circuit breakers."""
    active: bool = False
    trigger_time: Optional[datetime] = None
    trigger_reason: str = ""
    cooldown_until: Optional[datetime] = None

class AdvancedRiskManager:
    """
    Gestor de riesgo avanzado con:
    - Position sizing dinámico basado en volatilidad y régimen de mercado
    - Circuit breakers automáticos
    - Gestión de correlaciones
    - Kelly criterion opcional
    - Trailing stops dinámicos
    """
    
    def __init__(self, config: RiskConfig):
        self.config = config
        self.circuit_breaker = CircuitBreakerState()
        self.trade_history: List[Dict] = []
        self.daily_pnl: Dict[str, float] = {}  # date -> pnl
        self.current_positions: Dict[str, Dict] = {}  # symbol -> position_info
        self.correlation_matrix: Optional[pd.DataFrame] = None
        
    def evaluate_position(self, request: PositionRequest, 
                         current_equity: float, 
                         market_data: Dict) -> RiskAssessment:
        """Evaluación completa de riesgo para una nueva posición."""
        
        # 1. Verificar circuit breakers
        if self._check_circuit_breakers():
            return RiskAssessment(
                approved=False,
                position_size_usdt=0.0,
                leverage=1.0,
                stop_loss=request.entry_price,
                take_profit=request.entry_price,
                risk_reward_ratio=0.0,
                portfolio_risk_pct=0.0,
                risk_level=RiskLevel.CRITICAL,
                rejection_reason="Circuit breaker active"
            )
        
        # 2. Verificar límites básicos de portfolio
        portfolio_check = self._check_portfolio_limits(current_equity)
        if not portfolio_check[0]:
            return RiskAssessment(
                approved=False,
                position_size_usdt=0.0,
                leverage=1.0,
                stop_loss=request.entry_price,
                take_profit=request.entry_price,
                risk_reward_ratio=0.0,
                portfolio_risk_pct=0.0,
                risk_level=RiskLevel.CRITICAL,
                rejection_reason=portfolio_check[1]
            )
        
        # 3. Calcular position sizing dinámico
        base_size = self._calculate_dynamic_position_size(
            request, current_equity, market_data
        )
        
        # 4. Optimizar stop loss y take profit
        sl, tp = self._optimize_sl_tp(request, market_data)
        
        # 5. Calcular leverage óptimo
        leverage = self._calculate_optimal_leverage(request, market_data)
        
        # 6. Ajustar por correlaciones
        size_adjustment = self._correlation_adjustment(request)
        final_size = base_size * size_adjustment
        
        # 7. Aplicar límites finales
        final_size = max(self.config.min_position_usdt, 
                        min(final_size, self.config.max_position_usdt))
        
        # 8. Calcular risk-reward ratio
        rr_ratio = self._calculate_rr_ratio(request.entry_price, sl, tp, request.side)
        
        # 9. Verificar ratio mínimo
        if rr_ratio < self.config.min_risk_reward_ratio:
            return RiskAssessment(
                approved=False,
                position_size_usdt=0.0,
                leverage=1.0,
                stop_loss=sl,
                take_profit=tp,
                risk_reward_ratio=rr_ratio,
                portfolio_risk_pct=0.0,
                risk_level=RiskLevel.HIGH,
                rejection_reason=f"RR ratio {rr_ratio:.2f} below minimum {self.config.min_risk_reward_ratio}"
            )
        
        # 10. Calcular riesgo final del portfolio
        portfolio_risk = self._calculate_portfolio_risk_impact(final_size, current_equity)
        
        # 11. Determinar nivel de riesgo
        risk_level = self._assess_risk_level(portfolio_risk, rr_ratio, market_data)
        
        # 12. Generar warnings
        warnings = self._generate_warnings(request, market_data, portfolio_risk)
        
        return RiskAssessment(
            approved=True,
            position_size_usdt=final_size,
            leverage=leverage,
            stop_loss=sl,
            take_profit=tp,
            risk_reward_ratio=rr_ratio,
            portfolio_risk_pct=portfolio_risk,
            risk_level=risk_level,
            warnings=warnings
        )
    
    def _check_circuit_breakers(self) -> bool:
        """Verifica si hay circuit breakers activos."""
        if not self.circuit_breaker.active:
            return False
            
        # Verificar si ya pasó el cooldown
        if (self.circuit_breaker.cooldown_until and 
            datetime.now() > self.circuit_breaker.cooldown_until):
            self._reset_circuit_breakers()
            return False
            
        return True
    
    def _check_portfolio_limits(self, current_equity: float) -> Tuple[bool, str]:
        """Verifica límites básicos del portfolio."""
        
        # Verificar máximo número de posiciones
        if len(self.current_positions) >= self.config.max_open_positions:
            return False, "Maximum open positions reached"
        
        # Verificar pérdida diaria
        today = datetime.now().strftime('%Y-%m-%d')
        daily_loss = self.daily_pnl.get(today, 0.0)
        max_daily_loss = current_equity * self.config.max_daily_loss_pct
        
        if daily_loss < -max_daily_loss:
            return False, f"Daily loss limit exceeded: {daily_loss:.2f}"
        
        # Verificar drawdown máximo
        if self._calculate_current_drawdown(current_equity) > self.config.max_drawdown_pct:
            return False, "Maximum drawdown exceeded"
        
        return True, ""
    
    def _calculate_dynamic_position_size(self, request: PositionRequest, 
                                       current_equity: float, 
                                       market_data: Dict) -> float:
        """Calcula position sizing dinámico basado en múltiples factores."""
        
        # Base risk por trade
        base_risk_pct = self.config.base_risk_per_trade_pct
        
        # 1. Ajuste por volatilidad
        if self.config.volatility_scaling:
            vol_adjustment = self._volatility_scaling_factor(market_data)
            base_risk_pct *= vol_adjustment
        
        # 2. Ajuste por régimen de mercado
        if self.config.regime_scaling:
            regime_adjustment = self._regime_scaling_factor(request.market_regime)
            base_risk_pct *= regime_adjustment
        
        # 3. Ajuste por confianza de la estrategia
        confidence_adjustment = 0.5 + (request.strategy_confidence * 0.5)  # 0.5-1.0
        base_risk_pct *= confidence_adjustment
        
        # 4. Kelly criterion (opcional)
        if self.config.kelly_sizing:
            kelly_factor = self._calculate_kelly_factor(request.symbol)
            base_risk_pct = min(base_risk_pct, kelly_factor)
        
        # 5. Limitar por máximos
        base_risk_pct = min(base_risk_pct, self.config.max_risk_per_trade_pct)
        
        # Convertir a USDT
        risk_usdt = current_equity * base_risk_pct
        
        # Calcular posición basada en stop loss
        if request.stop_loss:
            stop_distance = abs(request.entry_price - request.stop_loss)
            position_size = risk_usdt / stop_distance if stop_distance > 0 else 0
        else:
            # Usar ATR como stop loss estimado
            atr = market_data.get('atr', request.entry_price * 0.02)  # 2% default
            stop_distance = atr * self.config.atr_sl_multiplier
            position_size = risk_usdt / stop_distance
        
        return position_size
    
    def _volatility_scaling_factor(self, market_data: Dict) -> float:
        """Factor de escalado basado en volatilidad actual vs histórica."""
        current_vol = market_data.get('volatility_20d', 0.02)
        avg_vol = market_data.get('volatility_avg', 0.02)
        
        if avg_vol <= 0:
            return 1.0
        
        vol_ratio = current_vol / avg_vol
        
        # Escalar inversamente: más volatilidad = menos riesgo
        if vol_ratio > 1.5:
            return 0.6  # Alta volatilidad -> reducir posición
        elif vol_ratio > 1.2:
            return 0.8
        elif vol_ratio < 0.7:
            return 1.3  # Baja volatilidad -> aumentar posición
        elif vol_ratio < 0.9:
            return 1.1
        else:
            return 1.0  # Volatilidad normal
    
    def _regime_scaling_factor(self, regime: MarketRegime) -> float:
        """Factor de escalado basado en régimen de mercado."""
        scaling_factors = {
            MarketRegime.TRENDING: 1.2,      # Trending favorece momentum
            MarketRegime.SIDEWAYS: 0.8,      # Sideways es más difícil
            MarketRegime.HIGH_VOLATILITY: 0.7,  # Alta vol = más riesgo
            MarketRegime.LOW_VOLATILITY: 1.1    # Baja vol = más estable
        }
        return scaling_factors.get(regime, 1.0)
    
    def _optimize_sl_tp(self, request: PositionRequest, 
                       market_data: Dict) -> Tuple[float, float]:
        """Optimiza stop loss y take profit basado en ATR y niveles SMC."""
        
        entry = request.entry_price
        atr = market_data.get('atr', entry * 0.02)
        
        # Stop Loss base en ATR
        sl_distance = atr * self.config.atr_sl_multiplier
        
        if request.side == "long":
            base_sl = entry - sl_distance
            base_tp = entry + (sl_distance * self.config.atr_tp_multiplier)
        else:
            base_sl = entry + sl_distance  
            base_tp = entry - (sl_distance * self.config.atr_tp_multiplier)
        
        # Ajustar por niveles SMC si están disponibles
        adjusted_sl = self._adjust_sl_for_smc(base_sl, market_data, request.side)
        adjusted_tp = self._adjust_tp_for_smc(base_tp, market_data, request.side)
        
        # Usar valores de request si son mejores
        final_sl = request.stop_loss if request.stop_loss else adjusted_sl
        final_tp = request.take_profit if request.take_profit else adjusted_tp
        
        return final_sl, final_tp
    
    def _adjust_sl_for_smc(self, base_sl: float, market_data: Dict, side: str) -> float:
        """Ajusta SL considerando niveles SMC (Order Blocks, Liquidity)."""
        
        # Obtener niveles SMC cercanos
        ob_levels = market_data.get('order_block_levels', [])
        liquidity_levels = market_data.get('liquidity_levels', [])
        
        if side == "long":
            # Para long, buscar niveles de soporte debajo del SL base
            support_levels = [lvl for lvl in ob_levels + liquidity_levels if lvl < base_sl]
            if support_levels:
                # Colocar SL ligeramente debajo del soporte más cercano
                closest_support = max(support_levels)
                return closest_support * 0.998  # 0.2% buffer
        else:
            # Para short, buscar niveles de resistencia encima del SL base
            resistance_levels = [lvl for lvl in ob_levels + liquidity_levels if lvl > base_sl]
            if resistance_levels:
                # Colocar SL ligeramente encima de la resistencia más cercana
                closest_resistance = min(resistance_levels)
                return closest_resistance * 1.002  # 0.2% buffer
        
        return base_sl
    
    def _adjust_tp_for_smc(self, base_tp: float, market_data: Dict, side: str) -> float:
        """Ajusta TP considerando niveles SMC y FVGs."""
        
        fvg_levels = market_data.get('fvg_levels', [])
        ob_levels = market_data.get('order_block_levels', [])
        
        if side == "long":
            # Para long, buscar resistencias encima del TP base
            resistance_levels = [lvl for lvl in fvg_levels + ob_levels if lvl > base_tp]
            if resistance_levels:
                # Colocar TP ligeramente debajo de la resistencia más cercana
                closest_resistance = min(resistance_levels)
                return closest_resistance * 0.998
        else:
            # Para short, buscar soportes debajo del TP base
            support_levels = [lvl for lvl in fvg_levels + ob_levels if lvl < base_tp]
            if support_levels:
                # Colocar TP ligeramente encima del soporte más cercano
                closest_support = max(support_levels)
                return closest_support * 1.002
        
        return base_tp
    
    def _calculate_optimal_leverage(self, request: PositionRequest, 
                                  market_data: Dict) -> float:
        """Calcula leverage óptimo basado en volatilidad y tipo de instrumento."""
        
        # Leverage máximo por tipo
        if request.symbol.endswith('USDT'):  # Spot
            max_lev = self.config.max_leverage_spot
        else:  # Futures
            max_lev = self.config.max_leverage_futures
        
        # Ajustar por volatilidad
        current_vol = market_data.get('volatility_20d', 0.02)
        
        if current_vol > 0.05:  # Alta volatilidad (>5%)
            vol_factor = 0.5
        elif current_vol > 0.03:  # Media volatilidad (3-5%)
            vol_factor = 0.7
        else:  # Baja volatilidad (<3%)
            vol_factor = 1.0
        
        # Aplicar factor de escalado
        optimal_leverage = max_lev * vol_factor * self.config.leverage_scaling_factor
        
        return max(1.0, min(optimal_leverage, max_lev))
    
    def _correlation_adjustment(self, request: PositionRequest) -> float:
        """Ajusta position size basado en correlaciones existentes."""
        
        if not self.config.correlation_scaling or not self.correlation_matrix:
            return 1.0
        
        # Contar posiciones correlacionadas
        correlated_positions = 0
        for symbol in self.current_positions:
            if self._are_correlated(request.symbol, symbol):
                correlated_positions += 1
        
        # Reducir tamaño si hay muchas posiciones correlacionadas
        if correlated_positions >= self.config.max_correlated_positions:
            return 0.0  # No permitir más posiciones correlacionadas
        elif correlated_positions > 0:
            return 1.0 - (0.2 * correlated_positions)  # Reducir 20% por cada correlación
        
        return 1.0
    
    def _are_correlated(self, symbol1: str, symbol2: str, threshold: float = 0.7) -> bool:
        """Verifica si dos símbolos están correlacionados."""
        if not self.correlation_matrix or symbol1 == symbol2:
            return False
            
        try:
            correlation = abs(self.correlation_matrix.loc[symbol1, symbol2])
            return correlation > threshold
        except (KeyError, IndexError):
            return False
    
    def _calculate_kelly_factor(self, symbol: str) -> float:
        """Calcula Kelly criterion factor basado en historial de trades."""
        
        # Obtener trades recientes del símbolo
        symbol_trades = [t for t in self.trade_history[-50:] if t.get('symbol') == symbol]
        
        if len(symbol_trades) < 10:
            return self.config.base_risk_per_trade_pct  # No suficiente historia
        
        # Calcular win rate y avg win/loss
        wins = [t['pnl'] for t in symbol_trades if t['pnl'] > 0]
        losses = [abs(t['pnl']) for t in symbol_trades if t['pnl'] < 0]
        
        if not wins or not losses:
            return self.config.base_risk_per_trade_pct
        
        win_rate = len(wins) / len(symbol_trades)
        avg_win = np.mean(wins)
        avg_loss = np.mean(losses)
        
        # Kelly fraction: f* = (bp - q) / b
        # donde b = avg_win/avg_loss, p = win_rate, q = 1-win_rate
        if avg_loss == 0:
            return self.config.base_risk_per_trade_pct
            
        b = avg_win / avg_loss
        kelly_fraction = (b * win_rate - (1 - win_rate)) / b
        
        # Ser conservador: usar solo 25% del Kelly
        conservative_kelly = kelly_fraction * 0.25
        
        # Limitar entre valores razonables
        return max(0.001, min(conservative_kelly, self.config.max_risk_per_trade_pct))
    
    def _calculate_rr_ratio(self, entry: float, sl: float, tp: float, side: str) -> float:
        """Calcula risk-reward ratio."""
        if side == "long":
            risk = entry - sl
            reward = tp - entry
        else:
            risk = sl - entry
            reward = entry - tp
        
        return reward / risk if risk > 0 else 0.0
    
    def _calculate_portfolio_risk_impact(self, position_size: float, 
                                       current_equity: float) -> float:
        """Calcula el impacto de riesgo en el portfolio."""
        return position_size / current_equity
    
    def _calculate_current_drawdown(self, current_equity: float) -> float:
        """Calcula drawdown actual desde peak equity."""
        # Simplificado - en implementación real rastrear peak equity
        return 0.0  # Placeholder
    
    def _assess_risk_level(self, portfolio_risk: float, rr_ratio: float, 
                          market_data: Dict) -> RiskLevel:
        """Evalúa nivel de riesgo general."""
        
        if portfolio_risk > 0.05 or rr_ratio < 1.2:
            return RiskLevel.HIGH
        elif portfolio_risk > 0.03 or rr_ratio < 1.5:
            return RiskLevel.MEDIUM
        else:
            return RiskLevel.LOW
    
    def _generate_warnings(self, request: PositionRequest, 
                          market_data: Dict, portfolio_risk: float) -> List[str]:
        """Genera warnings relevantes."""
        warnings = []
        
        if portfolio_risk > 0.04:
            warnings.append("High portfolio risk exposure")
        
        if market_data.get('volatility_20d', 0.02) > 0.05:
            warnings.append("High market volatility detected")
        
        if len(self.current_positions) > 3:
            warnings.append("Multiple open positions - monitor correlation")
            
        return warnings
    
    def trigger_circuit_breaker(self, reason: str):
        """Activa circuit breaker con razón específica."""
        self.circuit_breaker.active = True
        self.circuit_breaker.trigger_time = datetime.now()
        self.circuit_breaker.trigger_reason = reason
        self.circuit_breaker.cooldown_until = (
            datetime.now() + timedelta(minutes=self.config.cooldown_after_breaker_minutes)
        )
    
    def _reset_circuit_breakers(self):
        """Resetea circuit breakers."""
        self.circuit_breaker = CircuitBreakerState()
    
    def should_trail_stop(self, symbol: str, current_price: float) -> Tuple[bool, float]:
        """Determina si se debe activar trailing stop."""
        position = self.current_positions.get(symbol)
        if not position:
            return False, 0.0
        
        entry_price = position['entry_price']
        stop_loss = position['stop_loss']
        side = position['side']
        
        # Calcular profit actual
        if side == "long":
            profit_ratio = (current_price - entry_price) / (entry_price - stop_loss)
        else:
            profit_ratio = (entry_price - current_price) / (stop_loss - entry_price)
        
        # Activar trailing si se alcanzó la ratio configurada
        if profit_ratio >= self.config.trailing_stop_activation_rr:
            atr = position.get('atr', entry_price * 0.02)
            trail_distance = atr * self.config.atr_sl_multiplier
            
            if side == "long":
                new_sl = current_price - trail_distance
                return new_sl > stop_loss, new_sl
            else:
                new_sl = current_price + trail_distance
                return new_sl < stop_loss, new_sl
        
        return False, stop_loss
    
    def update_position(self, symbol: str, position_data: Dict):
        """Actualiza información de posición existente."""
        self.current_positions[symbol] = position_data
    
    def close_position(self, symbol: str, pnl: float):
        """Cierra posición y actualiza estadísticas."""
        if symbol in self.current_positions:
            del self.current_positions[symbol]
        
        # Registrar trade
        trade_record = {
            'symbol': symbol,
            'pnl': pnl,
            'timestamp': datetime.now(),
            'date': datetime.now().strftime('%Y-%m-%d')
        }
        self.trade_history.append(trade_record)
        
        # Actualizar PnL diario
        today = trade_record['date']
        self.daily_pnl[today] = self.daily_pnl.get(today, 0.0) + pnl
        
        # Verificar circuit breakers
        self._check_consecutive_losses()
    
    def _check_consecutive_losses(self):
        """Verifica pérdidas consecutivas para circuit breaker."""
        recent_trades = self.trade_history[-self.config.max_consecutive_losses:]
        
        if (len(recent_trades) >= self.config.max_consecutive_losses and
            all(t['pnl'] < 0 for t in recent_trades)):
            self.trigger_circuit_breaker(
                f"Consecutive losses: {self.config.max_consecutive_losses}"
            )
    
    def get_portfolio_summary(self) -> Dict:
        """Retorna resumen del estado del portfolio."""
        return {
            'open_positions': len(self.current_positions),
            'circuit_breaker_active': self.circuit_breaker.active,
            'circuit_breaker_reason': self.circuit_breaker.trigger_reason,
            'daily_pnl': sum(self.daily_pnl.values()),
            'total_trades': len(self.trade_history),
            'recent_win_rate': self._calculate_recent_win_rate()
        }
    
    def _calculate_recent_win_rate(self, lookback: int = 20) -> float:
        """Calcula win rate reciente."""
        recent_trades = self.trade_history[-lookback:]
        if not recent_trades:
            return 0.0
        
        wins = sum(1 for t in recent_trades if t['pnl'] > 0)
        return wins / len(recent_trades)


if __name__ == "__main__":
    # Test básico del risk manager
    config = RiskConfig.from_yaml("config/risk.yaml")
    risk_manager = AdvancedRiskManager(config)
    
    # Sample request
    request = PositionRequest(
        symbol="BTCUSDT",
        side="long",
        entry_price=50000.0,
        strategy_confidence=0.8,
        market_regime=MarketRegime.TRENDING
    )
    
    # Sample market data
    market_data = {
        'atr': 1000.0,
        'volatility_20d': 0.03,
        'volatility_avg': 0.025,
        'order_block_levels': [49500, 50500],
        'fvg_levels': [51000, 52000]
    }
    
    # Evaluate position
    assessment = risk_manager.evaluate_position(
        request=request,
        current_equity=10000.0,
        market_data=market_data
    )
    
    print("=== Risk Assessment ===")
    print(f"Approved: {assessment.approved}")
    print(f"Position Size: ${assessment.position_size_usdt:.2f}")
    print(f"Leverage: {assessment.leverage:.2f}x")
    print(f"Stop Loss: ${assessment.stop_loss:.2f}")
    print(f"Take Profit: ${assessment.take_profit:.2f}")
    print(f"Risk/Reward: {assessment.risk_reward_ratio:.2f}")
    print(f"Portfolio Risk: {assessment.portfolio_risk_pct:.2%}")
    print(f"Risk Level: {assessment.risk_level.value}")
    if assessment.warnings:
        print(f"Warnings: {assessment.warnings}")