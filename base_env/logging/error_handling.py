"""
Módulo centralizado para logging estructurado, manejo de errores y validación de estado.
Utilidades transversales para el sistema de trading.
"""

import json
import logging
import time
from typing import Dict, Any, List, Optional, Callable, Union
from dataclasses import dataclass, asdict
from functools import wraps
from enum import Enum

# ===== EXCEPCIONES DEL DOMINIO =====

class TradingError(Exception):
    """Excepción base para errores de trading"""
    def __init__(self, message: str, context: Optional[Dict[str, Any]] = None):
        super().__init__(message)
        self.context = context or {}
        self.timestamp = time.time()

class InsufficientBalanceError(TradingError):
    """Error cuando no hay balance suficiente para una operación"""
    pass

class InvalidPositionStateError(TradingError):
    """Error cuando el estado de la posición es inválido"""
    pass

class DataIntegrityError(TradingError):
    """Error cuando hay problemas de integridad de datos"""
    pass

class RiskManagementError(TradingError):
    """Error cuando el risk manager bloquea una operación"""
    pass

class OMSExecutionError(TradingError):
    """Error en la ejecución de órdenes en el OMS"""
    pass

class ConfigurationError(TradingError):
    """Error en la configuración del sistema"""
    pass

# ===== LOGGER ESPECIALIZADO =====

class TradingLogger:
    """Logger especializado para trading con formato JSON estructurado"""
    
    def __init__(self, name: str, level: int = logging.INFO):
        self.logger = logging.getLogger(name)
        self.logger.setLevel(level)
        
        # Configurar handler si no existe
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
    
    def _log_structured(self, level: str, event: str, data: Dict[str, Any]):
        """Log estructurado en formato JSON"""
        log_entry = {
            "timestamp": time.time(),
            "level": level,
            "event": event,
            "data": data
        }
        
        if level == "ERROR":
            self.logger.error(json.dumps(log_entry, default=str))
        elif level == "WARNING":
            self.logger.warning(json.dumps(log_entry, default=str))
        elif level == "INFO":
            self.logger.info(json.dumps(log_entry, default=str))
        else:
            self.logger.debug(json.dumps(log_entry, default=str))
    
    def trade_executed(self, side: str, qty: float, price: float, 
                      sl: Optional[float] = None, tp: Optional[float] = None,
                      notional: Optional[float] = None, fees: Optional[float] = None):
        """Log de trade ejecutado"""
        data = {
            "side": side,
            "qty": qty,
            "price": price,
            "sl": sl,
            "tp": tp,
            "notional": notional,
            "fees": fees
        }
        self._log_structured("INFO", "TRADE_EXECUTED", data)
    
    def trade_rejected(self, reason: str, side: str, qty: float, price: float):
        """Log de trade rechazado"""
        data = {
            "reason": reason,
            "side": side,
            "qty": qty,
            "price": price
        }
        self._log_structured("WARNING", "TRADE_REJECTED", data)
    
    def position_opened(self, side: str, qty: float, entry_price: float, 
                       sl: Optional[float] = None, tp: Optional[float] = None):
        """Log de posición abierta"""
        data = {
            "side": side,
            "qty": qty,
            "entry_price": entry_price,
            "sl": sl,
            "tp": tp
        }
        self._log_structured("INFO", "POSITION_OPENED", data)
    
    def position_closed(self, side: str, qty: float, exit_price: float, 
                       pnl: float, duration_bars: int):
        """Log de posición cerrada"""
        data = {
            "side": side,
            "qty": qty,
            "exit_price": exit_price,
            "pnl": pnl,
            "duration_bars": duration_bars
        }
        self._log_structured("INFO", "POSITION_CLOSED", data)
    
    def risk_event(self, event_type: str, details: Dict[str, Any]):
        """Log de evento de riesgo"""
        data = {
            "event_type": event_type,
            "details": details
        }
        self._log_structured("WARNING", "RISK_EVENT", data)
    
    def system_error(self, error: Exception, context: Dict[str, Any]):
        """Log de error del sistema"""
        data = {
            "error_type": type(error).__name__,
            "error_message": str(error),
            "context": context
        }
        self._log_structured("ERROR", "SYSTEM_ERROR", data)
    
    def performance_metric(self, metric_name: str, value: float, unit: str = ""):
        """Log de métrica de rendimiento"""
        data = {
            "metric_name": metric_name,
            "value": value,
            "unit": unit
        }
        self._log_structured("INFO", "PERFORMANCE_METRIC", data)

# ===== DECORADOR PARA OPERACIONES CRÍTICAS =====

def critical_operation(operation_name: str, logger: Optional[TradingLogger] = None):
    """Decorador para capturar errores en operaciones críticas"""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            if logger is None:
                tlogger = TradingLogger(func.__name__)
            else:
                tlogger = logger
            
            try:
                tlogger._log_structured("INFO", "CRITICAL_OPERATION_START", {
                    "operation": operation_name,
                    "function": func.__name__
                })
                
                result = func(*args, **kwargs)
                
                tlogger._log_structured("INFO", "CRITICAL_OPERATION_SUCCESS", {
                    "operation": operation_name,
                    "function": func.__name__
                })
                
                return result
                
            except Exception as e:
                tlogger.system_error(e, {
                    "operation": operation_name,
                    "function": func.__name__,
                    "args": str(args)[:200],  # Limitar tamaño
                    "kwargs": str(kwargs)[:200]
                })
                raise
        
        return wrapper
    return decorator

# ===== VALIDADOR DE ESTADO =====

@dataclass
class ValidationResult:
    """Resultado de validación de estado"""
    is_valid: bool
    errors: List[str]
    warnings: List[str]
    timestamp: float = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = time.time()

class StateValidator:
    """Validador de consistencia de estado del sistema"""
    
    @staticmethod
    def validate_portfolio_consistency(portfolio, position) -> List[str]:
        """Valida consistencia entre portfolio y posición"""
        errors = []
        
        try:
            # Verificar que el equity sea consistente
            if hasattr(portfolio, 'equity_quote') and hasattr(position, 'unrealized_pnl'):
                expected_equity = portfolio.cash_quote + (position.unrealized_pnl or 0)
                if abs(portfolio.equity_quote - expected_equity) > 0.01:
                    errors.append(f"Inconsistencia equity: {portfolio.equity_quote} vs {expected_equity}")
            
            # Verificar que la posición sea válida
            if hasattr(position, 'side') and position.side != 0:
                if not hasattr(position, 'qty') or position.qty <= 0:
                    errors.append("Posición activa sin cantidad válida")
                
                if not hasattr(position, 'entry_price') or position.entry_price <= 0:
                    errors.append("Posición activa sin precio de entrada válido")
            
            # Verificar que el balance no sea negativo
            if hasattr(portfolio, 'cash_quote') and portfolio.cash_quote < 0:
                errors.append(f"Balance negativo: {portfolio.cash_quote}")
            
        except Exception as e:
            errors.append(f"Error en validación de portfolio: {e}")
        
        return errors
    
    @staticmethod
    def validate_position_state(position) -> List[str]:
        """Valida estado de la posición"""
        errors = []
        
        try:
            if hasattr(position, 'side'):
                if position.side not in [-1, 0, 1]:
                    errors.append(f"Side inválido: {position.side}")
                
                if position.side != 0:  # Posición activa
                    if not hasattr(position, 'qty') or position.qty <= 0:
                        errors.append("Cantidad inválida para posición activa")
                    
                    if not hasattr(position, 'entry_price') or position.entry_price <= 0:
                        errors.append("Precio de entrada inválido para posición activa")
                    
                    # Verificar SL/TP si están definidos
                    if hasattr(position, 'sl') and position.sl is not None:
                        if position.sl <= 0:
                            errors.append("Stop Loss inválido")
                    
                    if hasattr(position, 'tp') and position.tp is not None:
                        if position.tp <= 0:
                            errors.append("Take Profit inválido")
            
        except Exception as e:
            errors.append(f"Error en validación de posición: {e}")
        
        return errors
    
    @staticmethod
    def validate_risk_limits(portfolio, position, risk_config) -> List[str]:
        """Valida límites de riesgo"""
        errors = []
        
        try:
            if hasattr(portfolio, 'equity_quote') and portfolio.equity_quote > 0:
                # Verificar exposición máxima
                if hasattr(risk_config, 'exposure') and hasattr(risk_config.exposure, 'max_notional_pct'):
                    if hasattr(position, 'qty') and hasattr(position, 'entry_price'):
                        notional = abs(position.qty * position.entry_price)
                        max_notional = portfolio.equity_quote * (risk_config.exposure.max_notional_pct / 100)
                        if notional > max_notional:
                            errors.append(f"Exposición excede límite: {notional} > {max_notional}")
                
                # Verificar drawdown máximo
                if hasattr(risk_config, 'circuit_breakers') and hasattr(risk_config.circuit_breakers, 'daily_dd_pct'):
                    if hasattr(portfolio, 'max_equity') and hasattr(portfolio, 'equity_quote'):
                        dd_pct = ((portfolio.max_equity - portfolio.equity_quote) / portfolio.max_equity) * 100
                        if dd_pct > risk_config.circuit_breakers.daily_dd_pct:
                            errors.append(f"Drawdown excede límite: {dd_pct:.2f}% > {risk_config.circuit_breakers.daily_dd_pct}%")
        
        except Exception as e:
            errors.append(f"Error en validación de riesgo: {e}")
        
        return errors
    
    @staticmethod
    def validate_complete_state(portfolio, position, risk_config) -> ValidationResult:
        """Validación completa del estado del sistema"""
        errors = []
        warnings = []
        
        # Validar portfolio
        portfolio_errors = StateValidator.validate_portfolio_consistency(portfolio, position)
        errors.extend(portfolio_errors)
        
        # Validar posición
        position_errors = StateValidator.validate_position_state(position)
        errors.extend(position_errors)
        
        # Validar límites de riesgo
        if risk_config:
            risk_errors = StateValidator.validate_risk_limits(portfolio, position, risk_config)
            errors.extend(risk_errors)
        
        # Añadir warnings para casos no críticos
        if hasattr(portfolio, 'equity_quote') and portfolio.equity_quote < 100:
            warnings.append("Equity muy bajo")
        
        return ValidationResult(
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=warnings
        )

# ===== CIRCUIT BREAKER =====

class CircuitBreakerState(Enum):
    """Estados del circuit breaker"""
    CLOSED = "closed"      # Funcionamiento normal
    OPEN = "open"          # Circuito abierto, bloqueando operaciones
    HALF_OPEN = "half_open"  # Probando si el servicio se recuperó

@dataclass
class CircuitBreakerConfig:
    """Configuración del circuit breaker"""
    failure_threshold: int = 5        # Número de fallos antes de abrir
    recovery_timeout: float = 60.0    # Tiempo en segundos antes de intentar recuperación
    success_threshold: int = 3        # Número de éxitos para cerrar en half_open

class CircuitBreaker:
    """Circuit breaker para control de fallos repetidos"""
    
    def __init__(self, name: str, config: CircuitBreakerConfig = None):
        self.name = name
        self.config = config or CircuitBreakerConfig()
        self.state = CircuitBreakerState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time = None
        self.logger = TradingLogger(f"CircuitBreaker.{name}")
    
    def call(self, func: Callable, *args, **kwargs):
        """Ejecuta función con protección de circuit breaker"""
        if self.state == CircuitBreakerState.OPEN:
            if self._should_attempt_reset():
                self.state = CircuitBreakerState.HALF_OPEN
                self.success_count = 0
                self.logger._log_structured("INFO", "CIRCUIT_BREAKER_HALF_OPEN", {
                    "name": self.name
                })
            else:
                raise OMSExecutionError(f"Circuit breaker {self.name} is OPEN")
        
        try:
            result = func(*args, **kwargs)
            self._on_success()
            return result
        except Exception as e:
            self._on_failure()
            raise
    
    def _should_attempt_reset(self) -> bool:
        """Determina si se debe intentar reset del circuit breaker"""
        if self.last_failure_time is None:
            return True
        
        return time.time() - self.last_failure_time >= self.config.recovery_timeout
    
    def _on_success(self):
        """Maneja éxito de operación"""
        if self.state == CircuitBreakerState.HALF_OPEN:
            self.success_count += 1
            if self.success_count >= self.config.success_threshold:
                self.state = CircuitBreakerState.CLOSED
                self.failure_count = 0
                self.logger._log_structured("INFO", "CIRCUIT_BREAKER_CLOSED", {
                    "name": self.name
                })
        else:
            self.failure_count = 0
    
    def _on_failure(self):
        """Maneja fallo de operación"""
        self.failure_count += 1
        self.last_failure_time = time.time()
        
        if self.failure_count >= self.config.failure_threshold:
            self.state = CircuitBreakerState.OPEN
            self.logger._log_structured("ERROR", "CIRCUIT_BREAKER_OPEN", {
                "name": self.name,
                "failure_count": self.failure_count
            })
    
    def reset(self):
        """Resetea el circuit breaker manualmente"""
        self.state = CircuitBreakerState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time = None
        self.logger._log_structured("INFO", "CIRCUIT_BREAKER_RESET", {
            "name": self.name
        })
    
    def get_status(self) -> Dict[str, Any]:
        """Retorna estado actual del circuit breaker"""
        return {
            "name": self.name,
            "state": self.state.value,
            "failure_count": self.failure_count,
            "success_count": self.success_count,
            "last_failure_time": self.last_failure_time
        }

# ===== UTILIDADES ADICIONALES =====

def create_trading_logger(name: str) -> TradingLogger:
    """Factory function para crear logger de trading"""
    return TradingLogger(name)

def log_performance(func: Callable) -> Callable:
    """Decorador para loggear métricas de rendimiento"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        logger = TradingLogger(func.__name__)
        start_time = time.time()
        
        try:
            result = func(*args, **kwargs)
            execution_time = time.time() - start_time
            logger.performance_metric(f"{func.__name__}_execution_time", execution_time, "seconds")
            return result
        except Exception as e:
            execution_time = time.time() - start_time
            logger.performance_metric(f"{func.__name__}_error_time", execution_time, "seconds")
            logger.system_error(e, {"function": func.__name__})
            raise
        
    return wrapper