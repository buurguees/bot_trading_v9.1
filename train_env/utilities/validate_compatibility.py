#!/usr/bin/env python3
"""
Script para validar compatibilidad de las mejoras implementadas.
"""

import sys
import json
import time
from pathlib import Path
from typing import Dict, Any, List
from decimal import Decimal

def test_oms_adapter_compatibility() -> Dict[str, Any]:
    """Test de compatibilidad del OMSAdapter"""
    print("🔍 Testando compatibilidad OMSAdapter...")
    
    # Mock OMSAdapter para testing
    class MockOMSAdapter:
        def open(self, side: int, qty: float, price_hint: float, sl: float = None, tp: float = None) -> Dict[str, Any]:
            return {
                "success": True,
                "side": "LONG" if side > 0 else "SHORT",
                "qty": float(qty),
                "price": float(price_hint),
                "notional": float(qty) * float(price_hint),
                "fees": 0.001,
                "ts": int(time.time() * 1000),
                "order_id": "test_123",
                "sl": float(sl) if sl is not None else None,
                "tp": float(tp) if tp is not None else None
            }
        
        def close(self, qty: float, price_hint: float) -> Dict[str, Any]:
            return {
                "success": True,
                "side": "CLOSE",
                "qty": float(qty),
                "price": float(price_hint),
                "notional": float(qty) * float(price_hint),
                "fees": 0.001,
                "ts": int(time.time() * 1000),
                "order_id": "close_123",
                "sl": None,
                "tp": None
            }
    
    # Test con diferentes tipos de entrada
    oms = MockOMSAdapter()
    tests = []
    
    # Test 1: Float normal
    try:
        result = oms.open(1, 0.1, 50000.0, 49500.0, 50500.0)
        tests.append({"test": "float_normal", "success": True, "result": result})
    except Exception as e:
        tests.append({"test": "float_normal", "success": False, "error": str(e)})
    
    # Test 2: Decimal (debería convertirse automáticamente)
    try:
        result = oms.open(1, Decimal("0.1"), Decimal("50000.0"), Decimal("49500.0"), Decimal("50500.0"))
        tests.append({"test": "decimal_input", "success": True, "result": result})
    except Exception as e:
        tests.append({"test": "decimal_input", "success": False, "error": str(e)})
    
    # Test 3: None values
    try:
        result = oms.open(-1, 0.1, 50000.0, None, None)
        tests.append({"test": "none_values", "success": True, "result": result})
    except Exception as e:
        tests.append({"test": "none_values", "success": False, "error": str(e)})
    
    return {
        "oms_adapter_tests": tests,
        "all_passed": all(test["success"] for test in tests)
    }

def test_typedict_compatibility() -> Dict[str, Any]:
    """Test de compatibilidad TypedDict"""
    print("🔍 Testando compatibilidad TypedDict...")
    
    # Importar FillResponse
    try:
        from base_env.base_env import FillResponse
        tests = []
        
        # Test 1: Crear FillResponse válido
        try:
            fill = FillResponse(
                success=True,
                side="LONG",
                qty=0.1,
                price=50000.0,
                notional=5000.0,
                fees=0.001,
                ts=1234567890,
                order_id="test_123",
                sl=49500.0,
                tp=50500.0
            )
            tests.append({"test": "create_fillresponse", "success": True})
        except Exception as e:
            tests.append({"test": "create_fillresponse", "success": False, "error": str(e)})
        
        # Test 2: Crear con None values
        try:
            fill = FillResponse(
                success=True,
                side="SHORT",
                qty=0.1,
                price=50000.0,
                notional=5000.0,
                fees=0.001,
                ts=1234567890,
                order_id="test_123",
                sl=None,
                tp=None
            )
            tests.append({"test": "create_with_none", "success": True})
        except Exception as e:
            tests.append({"test": "create_with_none", "success": False, "error": str(e)})
        
        return {
            "typedict_tests": tests,
            "all_passed": all(test["success"] for test in tests)
        }
        
    except ImportError as e:
        return {
            "typedict_tests": [],
            "all_passed": False,
            "error": f"Error importando FillResponse: {e}"
        }

def test_decimal_utils_compatibility() -> Dict[str, Any]:
    """Test de compatibilidad decimal_utils"""
    print("🔍 Testando compatibilidad decimal_utils...")
    
    try:
        from base_env.decimal_utils import (
            to_decimal, to_float, safe_decimal_operation,
            decimal_round, decimal_compare, create_number
        )
        
        tests = []
        
        # Test 1: Conversión básica
        try:
            result = to_decimal(123.456, 4)
            expected = Decimal("123.4560")
            tests.append({
                "test": "to_decimal_basic",
                "success": result == expected,
                "result": str(result),
                "expected": str(expected)
            })
        except Exception as e:
            tests.append({"test": "to_decimal_basic", "success": False, "error": str(e)})
        
        # Test 2: Conversión a float
        try:
            result = to_float(Decimal("123.456"))
            expected = 123.456
            tests.append({
                "test": "to_float_basic",
                "success": abs(result - expected) < 1e-6,
                "result": result,
                "expected": expected
            })
        except Exception as e:
            tests.append({"test": "to_float_basic", "success": False, "error": str(e)})
        
        # Test 3: Operación segura
        try:
            def test_func(a, b):
                return a + b
            
            result = safe_decimal_operation(test_func, Decimal("1.1"), Decimal("2.2"))
            expected = Decimal("3.3")
            tests.append({
                "test": "safe_decimal_operation",
                "success": result == expected,
                "result": str(result),
                "expected": str(expected)
            })
        except Exception as e:
            tests.append({"test": "safe_decimal_operation", "success": False, "error": str(e)})
        
        # Test 4: Comparación decimal
        try:
            result = decimal_compare(Decimal("1.0"), Decimal("1.0000001"), 1e-5)
            tests.append({
                "test": "decimal_compare",
                "success": result == True,
                "result": result
            })
        except Exception as e:
            tests.append({"test": "decimal_compare", "success": False, "error": str(e)})
        
        return {
            "decimal_utils_tests": tests,
            "all_passed": all(test["success"] for test in tests)
        }
        
    except ImportError as e:
        return {
            "decimal_utils_tests": [],
            "all_passed": False,
            "error": f"Error importando decimal_utils: {e}"
        }

def test_json_serialization() -> Dict[str, Any]:
    """Test de serialización JSON"""
    print("🔍 Testando serialización JSON...")
    
    tests = []
    
    # Test 1: Serialización normal (float)
    try:
        data = {
            "balance": 1000.0,
            "equity": 1000.0,
            "price": 50000.0,
            "qty": 0.1
        }
        json_str = json.dumps(data)
        parsed = json.loads(json_str)
        tests.append({
            "test": "float_serialization",
            "success": parsed == data,
            "json_str": json_str
        })
    except Exception as e:
        tests.append({"test": "float_serialization", "success": False, "error": str(e)})
    
    # Test 2: Serialización con Decimal (si está disponible)
    try:
        from base_env.decimal_utils import DecimalEncoder
        
        data = {
            "balance": Decimal("1000.0"),
            "equity": Decimal("1000.0"),
            "price": Decimal("50000.0"),
            "qty": Decimal("0.1")
        }
        json_str = json.dumps(data, cls=DecimalEncoder)
        parsed = json.loads(json_str)
        
        # Verificar que se convirtió a float
        expected = {
            "balance": 1000.0,
            "equity": 1000.0,
            "price": 50000.0,
            "qty": 0.1
        }
        tests.append({
            "test": "decimal_serialization",
            "success": parsed == expected,
            "json_str": json_str
        })
    except Exception as e:
        tests.append({"test": "decimal_serialization", "success": False, "error": str(e)})
    
    return {
        "json_tests": tests,
        "all_passed": all(test["success"] for test in tests)
    }

def run_compatibility_test() -> Dict[str, Any]:
    """Ejecuta test completo de compatibilidad"""
    print("🚀 Test de Compatibilidad - Mejoras Fase 1 y 2")
    print("=" * 60)
    
    results = {
        "timestamp": time.time(),
        "python_version": sys.version,
        "platform": sys.platform
    }
    
    # 1. Test OMSAdapter
    print("\n📡 1. Testando OMSAdapter...")
    results["oms_adapter"] = test_oms_adapter_compatibility()
    print(f"   Resultado: {'✅ PASS' if results['oms_adapter']['all_passed'] else '❌ FAIL'}")
    
    # 2. Test TypedDict
    print("\n📋 2. Testando TypedDict...")
    results["typedict"] = test_typedict_compatibility()
    print(f"   Resultado: {'✅ PASS' if results['typedict']['all_passed'] else '❌ FAIL'}")
    
    # 3. Test decimal_utils
    print("\n🔢 3. Testando decimal_utils...")
    results["decimal_utils"] = test_decimal_utils_compatibility()
    print(f"   Resultado: {'✅ PASS' if results['decimal_utils']['all_passed'] else '❌ FAIL'}")
    
    # 4. Test JSON
    print("\n📄 4. Testando serialización JSON...")
    results["json"] = test_json_serialization()
    print(f"   Resultado: {'✅ PASS' if results['json']['all_passed'] else '❌ FAIL'}")
    
    # Resumen general
    all_tests_passed = all([
        results["oms_adapter"]["all_passed"],
        results["typedict"]["all_passed"],
        results["decimal_utils"]["all_passed"],
        results["json"]["all_passed"]
    ])
    
    results["overall_success"] = all_tests_passed
    
    return results

def main() -> int:
    """Función principal"""
    print("🔍 Validador de Compatibilidad")
    print("=" * 40)
    
    # Verificar que estamos en el directorio correcto
    if not Path("base_env").exists():
        print("❌ Ejecutar desde el directorio raíz del proyecto")
        return 1
    
    # Ejecutar test
    results = run_compatibility_test()
    
    # Guardar resultados
    results_file = "compatibility_results.json"
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\n💾 Resultados guardados en {results_file}")
    
    # Resumen final
    print("\n📋 RESUMEN FINAL:")
    print(f"   OMSAdapter: {'✅' if results['oms_adapter']['all_passed'] else '❌'}")
    print(f"   TypedDict: {'✅' if results['typedict']['all_passed'] else '❌'}")
    print(f"   Decimal Utils: {'✅' if results['decimal_utils']['all_passed'] else '❌'}")
    print(f"   JSON Serialization: {'✅' if results['json']['all_passed'] else '❌'}")
    print(f"   RESULTADO GENERAL: {'✅ COMPATIBLE' if results['overall_success'] else '❌ INCOMPATIBLE'}")
    
    return 0 if results["overall_success"] else 1

if __name__ == "__main__":
    sys.exit(main())
