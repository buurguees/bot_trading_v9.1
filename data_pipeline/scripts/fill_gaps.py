# data_pipeline/scripts/fill_gaps.py
# Descripción: Rellena gaps en datos históricos usando interpolación y forward-fill
# Uso: python data_pipeline/scripts/fill_gaps.py --symbol BTCUSDT --market futures --tfs 1m,5m

from __future__ import annotations
import argparse
from pathlib import Path
from datetime import datetime, timezone
from typing import List, Dict, Any
import pandas as pd
import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.parquet as pq
import numpy as np

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("Rellenar gaps en datos históricos")
    p.add_argument("--root", type=str, default="data")
    p.add_argument("--symbol", type=str, required=True)
    p.add_argument("--market", type=str, choices=["spot","futures"], required=True)
    p.add_argument("--tfs", type=str, default="1m,5m")
    p.add_argument("--method", type=str, choices=["interpolate", "forward_fill", "linear"], default="interpolate")
    p.add_argument("--dry-run", action="store_true", help="Solo mostrar qué gaps se rellenarían")
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

def create_complete_timeline(start_ts: int, end_ts: int, tf: str) -> pd.DataFrame:
    """Crea una timeline completa sin gaps para el timeframe dado"""
    step_ms = tf_to_ms(tf)
    timestamps = list(range(start_ts, end_ts + step_ms, step_ms))
    
    return pd.DataFrame({
        'ts': timestamps,
        'tf': tf
    })

def fill_gaps_in_tf(root: Path, symbol: str, market: str, tf: str, method: str, dry_run: bool = False) -> Dict[str, Any]:
    """Rellena gaps en un timeframe específico"""
    print(f"🔍 Procesando {symbol} {market} {tf}...")
    
    # Cargar datos existentes
    aligned_dir = root / symbol / market / "aligned" / tf
    files = []
    for year_dir in aligned_dir.glob("year=*"):
        for month_dir in year_dir.glob("month=*"):
            files.extend(month_dir.glob("*.parquet"))
    
    if not files:
        print(f"❌ No hay datos para {symbol} {market} {tf}")
        return {"status": "no_data", "gaps_filled": 0, "original_rows": 0}
    
    # Cargar todos los datos
    dataset = ds.dataset([str(f) for f in files], format="parquet", partitioning="hive")
    table = dataset.scanner(columns=["ts", "open", "high", "low", "close", "volume"]).to_table().sort_by("ts")
    df = table.to_pandas()
    
    if df.empty:
        print(f"❌ DataFrame vacío para {symbol} {market} {tf}")
        return {"status": "empty", "gaps_filled": 0, "original_rows": 0}
    
    original_rows = len(df)
    print(f"📊 Datos originales: {original_rows:,} filas")
    
    # Crear timeline completa
    start_ts = df['ts'].min()
    end_ts = df['ts'].max()
    complete_timeline = create_complete_timeline(start_ts, end_ts, tf)
    
    print(f"📅 Timeline completa: {len(complete_timeline):,} filas esperadas")
    
    # Merge con datos existentes
    merged = complete_timeline.merge(df, on='ts', how='left', suffixes=('', '_existing'))
    
    # Identificar gaps
    gaps_mask = merged['open'].isna()
    gaps_count = gaps_mask.sum()
    
    print(f"🔍 Gaps detectados: {gaps_count:,}")
    
    if gaps_count == 0:
        print(f"✅ No hay gaps en {tf}")
        return {"status": "no_gaps", "gaps_filled": 0, "original_rows": original_rows}
    
    if dry_run:
        print(f"🔍 DRY RUN: Se rellenarían {gaps_count:,} gaps usando método '{method}'")
        return {"status": "dry_run", "gaps_filled": gaps_count, "original_rows": original_rows}
    
    # Rellenar gaps según método
    if method == "interpolate":
        # Interpolación lineal para OHLCV
        merged['open'] = merged['open'].interpolate(method='linear')
        merged['high'] = merged['high'].interpolate(method='linear')
        merged['low'] = merged['low'].interpolate(method='linear')
        merged['close'] = merged['close'].interpolate(method='linear')
        merged['volume'] = merged['volume'].fillna(0)  # Volume = 0 para gaps
        
    elif method == "forward_fill":
        # Forward fill (último valor conocido)
        merged['open'] = merged['open'].fillna(method='ffill')
        merged['high'] = merged['high'].fillna(method='ffill')
        merged['low'] = merged['low'].fillna(method='ffill')
        merged['close'] = merged['close'].fillna(method='ffill')
        merged['volume'] = merged['volume'].fillna(0)
        
    elif method == "linear":
        # Interpolación lineal más robusta
        for col in ['open', 'high', 'low', 'close']:
            merged[col] = merged[col].interpolate(method='linear', limit_direction='both')
        merged['volume'] = merged['volume'].fillna(0)
    
    # Añadir metadatos
    merged['symbol'] = symbol
    merged['market'] = market
    merged['tf'] = tf
    merged['ingestion_ts'] = int(datetime.now(tz=timezone.utc).timestamp() * 1000)
    
    # Eliminar columnas duplicadas
    merged = merged.drop(columns=['tf'], errors='ignore')
    
    # Guardar datos rellenados
    output_dir = root / symbol / market / "aligned_filled" / tf
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Dividir por año/mes para mantener estructura
    merged['year'] = pd.to_datetime(merged['ts'], unit='ms').dt.year
    merged['month'] = pd.to_datetime(merged['ts'], unit='ms').dt.month
    
    for (year, month), group in merged.groupby(['year', 'month']):
        month_dir = output_dir / f"year={year:04d}" / f"month={month:02d}"
        month_dir.mkdir(parents=True, exist_ok=True)
        
        group_clean = group.drop(columns=['year', 'month']).sort_values('ts')
        
        output_file = month_dir / f"part-{year:04d}-{month:02d}.parquet"
        pq.write_table(
            pa.Table.from_pandas(group_clean, preserve_index=False), 
            output_file, 
            compression="zstd"
        )
    
    print(f"✅ Gaps rellenados: {gaps_count:,} → {len(merged):,} filas totales")
    print(f"💾 Datos guardados en: {output_dir}")
    
    return {
        "status": "success", 
        "gaps_filled": gaps_count, 
        "original_rows": original_rows,
        "final_rows": len(merged),
        "output_dir": str(output_dir)
    }

def main():
    args = parse_args()
    root = Path(args.root)
    symbol = args.symbol.upper()
    market = args.market.lower()
    tfs = [t.strip() for t in args.tfs.split(",") if t.strip()]
    
    print(f"🚀 Iniciando relleno de gaps para {symbol} {market}")
    print(f"📋 Timeframes: {tfs}")
    print(f"🔧 Método: {args.method}")
    print(f"🔍 Modo: {'DRY RUN' if args.dry_run else 'EJECUCIÓN'}")
    print("-" * 50)
    
    results = {}
    
    for tf in tfs:
        try:
            result = fill_gaps_in_tf(root, symbol, market, tf, args.method, args.dry_run)
            results[tf] = result
        except Exception as e:
            print(f"❌ Error procesando {tf}: {e}")
            results[tf] = {"status": "error", "error": str(e)}
    
    print("-" * 50)
    print("📊 RESUMEN DE RESULTADOS:")
    for tf, result in results.items():
        if result["status"] == "success":
            print(f"✅ {tf}: {result['gaps_filled']:,} gaps rellenados ({result['original_rows']:,} → {result['final_rows']:,} filas)")
        elif result["status"] == "dry_run":
            print(f"🔍 {tf}: {result['gaps_filled']:,} gaps detectados (DRY RUN)")
        elif result["status"] == "no_gaps":
            print(f"✅ {tf}: Sin gaps")
        else:
            print(f"❌ {tf}: {result.get('error', 'Error desconocido')}")
    
    if not args.dry_run and any(r["status"] == "success" for r in results.values()):
        print("\n💡 PRÓXIMOS PASOS:")
        print("1. Verificar datos rellenados en data/{symbol}/{market}/aligned_filled/")
        print("2. Reemplazar datos originales si están correctos:")
        print(f"   mv data/{symbol}/{market}/aligned_filled data/{symbol}/{market}/aligned")
        print("3. Revalidar datos:")
        print(f"   python -m app run --symbol {symbol} --market {market}")

if __name__ == "__main__":
    main()
