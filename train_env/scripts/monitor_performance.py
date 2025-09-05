#!/usr/bin/env python3
"""
Script para monitorear rendimiento y detectar impactos de las mejoras.
"""

import time
import sys
import json
from pathlib import Path
from typing import Dict, Any, List
import psutil
import os

def measure_step_performance(env, num_steps: int = 100) -> Dict[str, float]:
    """Mide rendimiento del método step()"""
    print(f"🔍 Midiendo rendimiento de {num_steps} steps...")
    
    # Reset del entorno
    env.reset()
    
    # Medir tiempo
    start_time = time.time()
    start_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
    
    for i in range(num_steps):
        if i % 20 == 0:
            print(f"   Step {i}/{num_steps}")
        
        # Ejecutar step
        obs, reward, done, info = env.step()
        
        if done:
            env.reset()
    
    end_time = time.time()
    end_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
    
    total_time = end_time - start_time
    avg_time_per_step = total_time / num_steps
    memory_used = end_memory - start_memory
    
    return {
        "total_time": total_time,
        "avg_time_per_step": avg_time_per_step,
        "steps_per_second": num_steps / total_time,
        "memory_used_mb": memory_used,
        "num_steps": num_steps
    }

def measure_oms_conversion_performance() -> Dict[str, float]:
    """Mide rendimiento de conversiones OMS"""
    from decimal import Decimal
    
    # Datos de prueba
    test_data = [
        (123.456789, 0.00123456, 98765.4321),
        (Decimal("123.456789"), Decimal("0.00123456"), Decimal("98765.4321")),
        (123.456789, None, 98765.4321),
    ]
    
    print("🔍 Midiendo rendimiento de conversiones OMS...")
    
    # Medir conversión float
    start_time = time.time()
    for _ in range(10000):
        for qty, sl, price in test_data:
            float_qty = float(qty) if qty is not None else None
            float_sl = float(sl) if sl is not None else None
            float_price = float(price) if price is not None else None
    float_time = time.time() - start_time
    
    # Medir conversión con Decimal
    start_time = time.time()
    for _ in range(10000):
        for qty, sl, price in test_data:
            if isinstance(qty, Decimal):
                float_qty = float(qty)
            else:
                float_qty = float(qty) if qty is not None else None
            
            if isinstance(sl, Decimal):
                float_sl = float(sl)
            else:
                float_sl = float(sl) if sl is not None else None
            
            if isinstance(price, Decimal):
                float_price = float(price)
            else:
                float_price = float(price) if price is not None else None
    decimal_time = time.time() - start_time
    
    return {
        "float_conversion_time": float_time,
        "decimal_conversion_time": decimal_time,
        "overhead_ratio": decimal_time / float_time if float_time > 0 else 1.0
    }

def check_memory_usage() -> Dict[str, float]:
    """Verifica uso de memoria del sistema"""
    process = psutil.Process()
    memory_info = process.memory_info()
    
    return {
        "rss_mb": memory_info.rss / 1024 / 1024,
        "vms_mb": memory_info.vms / 1024 / 1024,
        "percent": process.memory_percent(),
        "available_mb": psutil.virtual_memory().available / 1024 / 1024
    }

def analyze_log_performance() -> Dict[str, Any]:
    """Analiza rendimiento de logging"""
    runs_dir = Path("models/BTCUSDT")
    if not runs_dir.exists():
        return {"error": "No se encontró directorio de runs"}
    
    log_files = list(runs_dir.glob("*.jsonl")) + list(runs_dir.glob("*.json"))
    total_size = sum(f.stat().st_size for f in log_files)
    
    return {
        "log_files_count": len(log_files),
        "total_size_mb": total_size / 1024 / 1024,
        "avg_file_size_mb": (total_size / len(log_files)) / 1024 / 1024 if log_files else 0
    }

def run_performance_test() -> Dict[str, Any]:
    """Ejecuta test completo de rendimiento"""
    print("🚀 Test de Rendimiento - Mejoras Fase 1 y 2")
    print("=" * 60)
    
    results = {
        "timestamp": time.time(),
        "system_info": {
            "python_version": sys.version,
            "platform": sys.platform,
            "cpu_count": psutil.cpu_count()
        }
    }
    
    # 1. Verificar uso de memoria
    print("\n📊 1. Verificando uso de memoria...")
    results["memory"] = check_memory_usage()
    print(f"   Memoria RSS: {results['memory']['rss_mb']:.1f} MB")
    print(f"   Memoria VMS: {results['memory']['vms_mb']:.1f} MB")
    print(f"   Porcentaje: {results['memory']['percent']:.1f}%")
    
    # 2. Medir conversiones OMS
    print("\n🔄 2. Midiendo conversiones OMS...")
    results["oms_conversion"] = measure_oms_conversion_performance()
    print(f"   Conversión float: {results['oms_conversion']['float_conversion_time']:.4f}s")
    print(f"   Conversión Decimal: {results['oms_conversion']['decimal_conversion_time']:.4f}s")
    print(f"   Overhead: {results['oms_conversion']['overhead_ratio']:.2f}x")
    
    # 3. Analizar logs
    print("\n📝 3. Analizando rendimiento de logs...")
    results["logs"] = analyze_log_performance()
    if "error" not in results["logs"]:
        print(f"   Archivos de log: {results['logs']['log_files_count']}")
        print(f"   Tamaño total: {results['logs']['total_size_mb']:.2f} MB")
        print(f"   Tamaño promedio: {results['logs']['avg_file_size_mb']:.2f} MB")
    
    # 4. Test de step (si el entorno está disponible)
    print("\n🏃 4. Test de step() (opcional)...")
    try:
        # Intentar importar y crear entorno
        sys.path.append(".")
        from base_env.base_env import BaseTradingEnv
        from base_env.config.models import EnvConfig
        
        # Crear configuración mínima para test
        config = EnvConfig(
            market="futures",
            tfs=["1m", "5m"],
            months_back=1
        )
        
        env = BaseTradingEnv(config)
        results["step_performance"] = measure_step_performance(env, 50)
        
        print(f"   Tiempo promedio por step: {results['step_performance']['avg_time_per_step']:.4f}s")
        print(f"   Steps por segundo: {results['step_performance']['steps_per_second']:.2f}")
        print(f"   Memoria usada: {results['step_performance']['memory_used_mb']:.1f} MB")
        
    except Exception as e:
        print(f"   ⚠️  No se pudo ejecutar test de step: {e}")
        results["step_performance"] = {"error": str(e)}
    
    return results

def main() -> int:
    """Función principal"""
    print("🔍 Monitor de Rendimiento")
    print("=" * 40)
    
    # Verificar que estamos en el directorio correcto
    if not Path("base_env").exists():
        print("❌ Ejecutar desde el directorio raíz del proyecto")
        return 1
    
    # Ejecutar test
    results = run_performance_test()
    
    # Guardar resultados
    results_file = "performance_results.json"
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\n💾 Resultados guardados en {results_file}")
    
    # Resumen
    print("\n📋 RESUMEN:")
    print(f"   Memoria actual: {results['memory']['rss_mb']:.1f} MB")
    if "step_performance" in results and "error" not in results["step_performance"]:
        print(f"   Rendimiento step: {results['step_performance']['steps_per_second']:.2f} steps/s")
    print(f"   Overhead Decimal: {results['oms_conversion']['overhead_ratio']:.2f}x")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
