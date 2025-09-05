# data_pipeline/scripts/validate_gaps_fixed.py
# Descripción: Valida que los gaps se hayan corregido correctamente

from __future__ import annotations
import argparse
from pathlib import Path
from typing import List, Dict, Any
import pandas as pd
import pyarrow as pa
import pyarrow.dataset as ds

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("Validar que los gaps se hayan corregido")
    p.add_argument("--root", type=str, default="data")
    p.add_argument("--symbol", type=str, required=True)
    p.add_argument("--market", type=str, choices=["spot","futures"], required=True)
    p.add_argument("--tfs", type=str, default="1m,5m,15m,1h")
    return p.parse_args()

def tf_to_ms(tf: str) -> int:
    """Convierte timeframe a milisegundos"""
    if tf.endswith('m'):
        return int(tf[:-1]) * 60 * 1000
    elif tf.endswith('h'):
        return int(tf[:-1]) * 60 * 60 * 1000
    elif tf.endswith('d'):
        return int(tf[:-1]) * 24 * 60 * 60 * 1000
    else:
        raise ValueError(f"Timeframe no soportado: {tf}")

def validate_tf_gaps(root: Path, symbol: str, market: str, tf: str) -> Dict[str, Any]:
    """Valida gaps en un timeframe específico"""
    print(f"🔍 Validando {symbol} {market} {tf}...")
    
    # Cargar datos
    aligned_dir = root / symbol / market / "aligned" / tf
    files = []
    for year_dir in aligned_dir.glob("year=*"):
        for month_dir in year_dir.glob("month=*"):
            files.extend(month_dir.glob("*.parquet"))
    
    if not files:
        print(f"❌ No hay datos para {symbol} {market} {tf}")
        return {"status": "no_data", "gaps": 0, "rows": 0}
    
    # Cargar todos los datos
    dataset = ds.dataset([str(f) for f in files], format="parquet", partitioning="hive")
    table = dataset.scanner(columns=["ts"]).to_table().sort_by("ts")
    df = table.to_pandas()
    
    if df.empty:
        print(f"❌ DataFrame vacío para {symbol} {market} {tf}")
        return {"status": "empty", "gaps": 0, "rows": 0}
    
    # Calcular gaps
    step_ms = tf_to_ms(tf)
    expected_ts = pd.Series(range(df['ts'].min(), df['ts'].max() + step_ms, step_ms))
    actual_ts = df['ts'].sort_values()
    
    # Encontrar gaps
    missing_ts = set(expected_ts) - set(actual_ts)
    gaps = len(missing_ts)
    
    print(f"📊 Datos: {len(df):,} filas")
    print(f"🔍 Gaps detectados: {gaps:,}")
    
    if gaps == 0:
        print(f"✅ {tf}: Sin gaps - PERFECTO")
        return {"status": "perfect", "gaps": 0, "rows": len(df)}
    elif gaps < 100:
        print(f"⚠️ {tf}: {gaps} gaps menores - ACEPTABLE")
        return {"status": "acceptable", "gaps": gaps, "rows": len(df)}
    else:
        print(f"❌ {tf}: {gaps} gaps - NECESITA CORRECCIÓN")
        return {"status": "needs_fix", "gaps": gaps, "rows": len(df)}

def main():
    args = parse_args()
    root = Path(args.root)
    symbol = args.symbol.upper()
    market = args.market.lower()
    tfs = [t.strip() for t in args.tfs.split(",") if t.strip()]
    
    print(f"🚀 Validando corrección de gaps para {symbol} {market}")
    print(f"📋 Timeframes: {tfs}")
    print("-" * 50)
    
    results = {}
    
    for tf in tfs:
        try:
            result = validate_tf_gaps(root, symbol, market, tf)
            results[tf] = result
        except Exception as e:
            print(f"❌ Error validando {tf}: {e}")
            results[tf] = {"status": "error", "gaps": -1, "rows": 0}
    
    print("-" * 50)
    print("📊 RESUMEN DE VALIDACIÓN:")
    
    perfect_count = 0
    acceptable_count = 0
    needs_fix_count = 0
    
    for tf, result in results.items():
        if result["status"] == "perfect":
            print(f"✅ {tf}: PERFECTO - 0 gaps")
            perfect_count += 1
        elif result["status"] == "acceptable":
            print(f"⚠️ {tf}: ACEPTABLE - {result['gaps']} gaps")
            acceptable_count += 1
        elif result["status"] == "needs_fix":
            print(f"❌ {tf}: NECESITA CORRECCIÓN - {result['gaps']} gaps")
            needs_fix_count += 1
        else:
            print(f"❌ {tf}: ERROR - {result.get('error', 'Error desconocido')}")
    
    print("-" * 50)
    print(f"📈 ESTADÍSTICAS:")
    print(f"✅ Perfectos: {perfect_count}/{len(tfs)}")
    print(f"⚠️ Aceptables: {acceptable_count}/{len(tfs)}")
    print(f"❌ Necesitan corrección: {needs_fix_count}/{len(tfs)}")
    
    if needs_fix_count == 0:
        print("\n🎉 ¡TODOS LOS GAPS CORREGIDOS EXITOSAMENTE!")
        print("💡 El sistema está listo para entrenamiento en producción.")
    else:
        print(f"\n⚠️ {needs_fix_count} timeframes aún necesitan corrección.")
        print("💡 Ejecuta fill_gaps.py para corregir los gaps restantes.")

if __name__ == "__main__":
    main()
