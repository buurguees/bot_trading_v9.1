#!/usr/bin/env python3
"""
Ejemplo de uso del módulo de error handling y logging estructurado.
"""

import sys
import time
from pathlib import Path

# Añadir el directorio raíz al path
sys.path.append(str(Path(__file__).parent.parent))

from base_env.logging.error_handling import (
    TradingLogger,
    TradingError,
    InsufficientBalanceError,
    InvalidPositionStateError,
    DataIntegrityError,
    critical_operation,
    StateValidator,
    CircuitBreaker,
    CircuitBreakerConfig,
    log_performance
)

def example_trading_logger():
    """Ejemplo de uso del TradingLogger"""
    print("🔍 Ejemplo de TradingLogger")
    print("=" * 40)
    
    # Crear logger
    logger = TradingLogger("ExampleTrading")
    
    # Log de trade ejecutado
    logger.trade_executed(
        side="LONG",
        qty=0.1,
        price=50000.0,
        sl=49500.0,
        tp=50500.0,
        notional=5000.0,
        fees=5.0
    )
    
    # Log de trade rechazado
    logger.trade_rejected(
        reason="Insufficient balance",
        side="SHORT",
        qty=0.2,
        price=50000.0
    )
    
    # Log de posición abierta
    logger.position_opened(
        side="LONG",
        qty=0.1,
        entry_price=50000.0,
        sl=49500.0,
        tp=50500.0
    )
    
    # Log de posición cerrada
    logger.position_closed(
        side="LONG",
        qty=0.1,
        exit_price=50500.0,
        pnl=50.0,
        duration_bars=15
    )
    
    # Log de evento de riesgo
    logger.risk_event("HIGH_DRAWDOWN", {
        "current_dd": 15.5,
        "max_allowed": 10.0
    })
    
    # Log de métrica de rendimiento
    logger.performance_metric("execution_time", 0.025, "seconds")
    
    print("✅ Ejemplos de logging completados")

def example_critical_operation():
    """Ejemplo de uso del decorador critical_operation"""
    print("\n🔍 Ejemplo de critical_operation")
    print("=" * 40)
    
    logger = TradingLogger("CriticalExample")
    
    @critical_operation("trade_execution", logger)
    def execute_trade(side: str, qty: float, price: float):
        """Simula ejecución de trade"""
        print(f"Ejecutando trade: {side} {qty} @ {price}")
        time.sleep(0.1)  # Simular procesamiento
        return {"success": True, "order_id": "12345"}
    
    @critical_operation("risk_check")
    def check_risk(side: str, qty: float, price: float):
        """Simula verificación de riesgo"""
        print(f"Verificando riesgo: {side} {qty} @ {price}")
        if qty > 1.0:
            raise RiskManagementError("Cantidad excede límite de riesgo")
        return True
    
    # Ejecutar operaciones críticas
    try:
        result = execute_trade("LONG", 0.1, 50000.0)
        print(f"Resultado: {result}")
        
        check_risk("LONG", 0.1, 50000.0)
        print("Verificación de riesgo exitosa")
        
    except Exception as e:
        print(f"Error capturado: {e}")
    
    print("✅ Ejemplos de operaciones críticas completados")

def example_state_validator():
    """Ejemplo de uso del StateValidator"""
    print("\n🔍 Ejemplo de StateValidator")
    print("=" * 40)
    
    # Mock de portfolio y posición
    class MockPortfolio:
        def __init__(self):
            self.cash_quote = 1000.0
            self.equity_quote = 1000.0
            self.max_equity = 1000.0
    
    class MockPosition:
        def __init__(self):
            self.side = 0
            self.qty = 0.0
            self.entry_price = 0.0
            self.unrealized_pnl = 0.0
            self.sl = None
            self.tp = None
    
    class MockRiskConfig:
        def __init__(self):
            self.exposure = type('obj', (object,), {'max_notional_pct': 100.0})()
            self.circuit_breakers = type('obj', (object,), {'daily_dd_pct': 10.0})()
    
    # Crear instancias
    portfolio = MockPortfolio()
    position = MockPosition()
    risk_config = MockRiskConfig()
    
    # Validar estado
    result = StateValidator.validate_complete_state(portfolio, position, risk_config)
    
    print(f"Estado válido: {result.is_valid}")
    print(f"Errores: {result.errors}")
    print(f"Warnings: {result.warnings}")
    
    # Simular posición activa
    position.side = 1
    position.qty = 0.1
    position.entry_price = 50000.0
    position.unrealized_pnl = 50.0
    
    result = StateValidator.validate_complete_state(portfolio, position, risk_config)
    
    print(f"\nCon posición activa:")
    print(f"Estado válido: {result.is_valid}")
    print(f"Errores: {result.errors}")
    print(f"Warnings: {result.warnings}")
    
    print("✅ Ejemplos de validación de estado completados")

def example_circuit_breaker():
    """Ejemplo de uso del CircuitBreaker"""
    print("\n🔍 Ejemplo de CircuitBreaker")
    print("=" * 40)
    
    # Crear circuit breaker
    config = CircuitBreakerConfig(
        failure_threshold=3,
        recovery_timeout=5.0,
        success_threshold=2
    )
    cb = CircuitBreaker("OMS", config)
    
    def failing_function():
        """Función que falla"""
        raise Exception("Simulated failure")
    
    def working_function():
        """Función que funciona"""
        return "Success"
    
    # Probar con función que falla
    print("Probando con función que falla...")
    for i in range(5):
        try:
            cb.call(failing_function)
        except Exception as e:
            print(f"Intento {i+1}: {e}")
            print(f"Estado: {cb.get_status()}")
    
    # Probar con función que funciona (debería fallar por circuit breaker abierto)
    print("\nProbando con función que funciona (circuit breaker abierto)...")
    try:
        cb.call(working_function)
    except Exception as e:
        print(f"Error esperado: {e}")
    
    # Esperar y probar recuperación
    print("\nEsperando recuperación...")
    time.sleep(6)  # Más que recovery_timeout
    
    print("Probando recuperación...")
    for i in range(3):
        try:
            result = cb.call(working_function)
            print(f"Intento {i+1}: {result}")
            print(f"Estado: {cb.get_status()}")
        except Exception as e:
            print(f"Intento {i+1}: {e}")
    
    print("✅ Ejemplos de circuit breaker completados")

def example_performance_logging():
    """Ejemplo de uso del decorador log_performance"""
    print("\n🔍 Ejemplo de log_performance")
    print("=" * 40)
    
    @log_performance
    def fast_operation():
        """Operación rápida"""
        time.sleep(0.01)
        return "Fast operation completed"
    
    @log_performance
    def slow_operation():
        """Operación lenta"""
        time.sleep(0.1)
        return "Slow operation completed"
    
    @log_performance
    def failing_operation():
        """Operación que falla"""
        time.sleep(0.05)
        raise Exception("Operation failed")
    
    # Ejecutar operaciones
    try:
        result1 = fast_operation()
        print(f"Resultado 1: {result1}")
        
        result2 = slow_operation()
        print(f"Resultado 2: {result2}")
        
        failing_operation()
    except Exception as e:
        print(f"Error esperado: {e}")
    
    print("✅ Ejemplos de logging de rendimiento completados")

def main():
    """Función principal"""
    print("🚀 Ejemplos de Error Handling y Logging")
    print("=" * 60)
    
    try:
        example_trading_logger()
        example_critical_operation()
        example_state_validator()
        example_circuit_breaker()
        example_performance_logging()
        
        print("\n🎉 Todos los ejemplos completados exitosamente")
        
    except Exception as e:
        print(f"\n❌ Error en ejemplos: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
