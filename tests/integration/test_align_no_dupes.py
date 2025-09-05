# tests/test_align_no_dupes.py
"""
Test para validar que align_package.py no produce duplicados y mantiene orden cronológico.
"""

import pytest
import tempfile
import shutil
from pathlib import Path
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from datetime import datetime, timezone
import subprocess
import sys


class TestAlignNoDupes:
    """Test de alineado sin duplicados"""
    
    def setup_method(self):
        """Setup para cada test"""
        self.temp_dir = tempfile.mkdtemp()
        self.data_root = Path(self.temp_dir) / "data"
        self.data_root.mkdir(parents=True, exist_ok=True)
        
        # Crear estructura de directorios
        self.symbol = "BTCUSDT"
        self.market = "futures"
        self.tf = "1h"
        
        self.raw_path = self.data_root / self.symbol / self.market / "raw" / self.tf
        self.aligned_path = self.data_root / self.symbol / self.market / "aligned" / self.tf
    
    def teardown_method(self):
        """Cleanup después de cada test"""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def _create_test_parquet(self, path: Path, data: list, year: int, month: int):
        """Crea un archivo parquet de prueba"""
        year_month_path = path / f"year={year:04d}" / f"month={month:02d}"
        year_month_path.mkdir(parents=True, exist_ok=True)
        
        df = pd.DataFrame(data)
        table = pa.Table.from_pandas(df)
        pq.write_table(table, year_month_path / f"part-{year:04d}-{month:02d}.parquet")
    
    def test_align_no_duplicates(self):
        """Test que align_package.py no produce duplicados"""
        # Crear datos de prueba con duplicados potenciales
        base_ts = int(datetime(2024, 1, 1, tzinfo=timezone.utc).timestamp() * 1000)
        
        # Datos para enero 2024
        jan_data = [
            {
                "ts": base_ts,
                "open": 47000.0,
                "high": 48000.0,
                "low": 46000.0,
                "close": 47500.0,
                "volume": 100.0,
                "symbol": self.symbol,
                "market": self.market,
                "tf": self.tf,
                "ingestion_ts": base_ts
            },
            {
                "ts": base_ts + 3600000,  # +1 hora
                "open": 47500.0,
                "high": 48500.0,
                "low": 47000.0,
                "close": 48000.0,
                "volume": 120.0,
                "symbol": self.symbol,
                "market": self.market,
                "tf": self.tf,
                "ingestion_ts": base_ts + 3600000
            },
            # Duplicado del primer timestamp (debería ser eliminado)
            {
                "ts": base_ts,
                "open": 47100.0,  # Diferente precio
                "high": 48100.0,
                "low": 46100.0,
                "close": 47600.0,
                "volume": 110.0,
                "symbol": self.symbol,
                "market": self.market,
                "tf": self.tf,
                "ingestion_ts": base_ts + 1000  # Diferente ingestion_ts
            }
        ]
        
        # Crear archivo raw
        self._create_test_parquet(self.raw_path, jan_data, 2024, 1)
        
        # Ejecutar align_package.py
        cmd = [
            sys.executable,
            "data_pipeline/scripts/align_package.py",
            "--root", str(self.data_root),
            "--symbol", self.symbol,
            "--market", self.market,
            "--tfs", self.tf,
            "--from", "2024-01-01",
            "--to", "2024-02-01"
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True, cwd=Path(__file__).parent.parent.parent)
        
        # Verificar que el comando fue exitoso
        assert result.returncode == 0, f"align_package.py failed: {result.stderr}"
        
        # Verificar que se creó el archivo aligned
        aligned_file = self.aligned_path / "year=2024" / "month=01" / "part-2024-01.parquet"
        assert aligned_file.exists(), "Aligned file was not created"
        
        # Leer y verificar el archivo aligned
        table = pq.read_table(aligned_file)
        df = table.to_pandas()
        
        # Verificar que no hay duplicados por timestamp
        assert len(df) == 2, f"Expected 2 rows, got {len(df)}"
        assert len(df['ts'].unique()) == 2, "Found duplicate timestamps"
        
        # Verificar que se mantuvo el último registro para cada timestamp
        first_ts_data = df[df['ts'] == base_ts].iloc[0]
        assert first_ts_data['open'] == 47100.0, "Should keep the last record for duplicate timestamp"
        assert first_ts_data['close'] == 47600.0, "Should keep the last record for duplicate timestamp"
    
    def test_align_chronological_order(self):
        """Test que align_package.py mantiene orden cronológico"""
        # Crear datos desordenados
        base_ts = int(datetime(2024, 1, 1, tzinfo=timezone.utc).timestamp() * 1000)
        
        # Datos en orden desordenado
        unordered_data = [
            {
                "ts": base_ts + 7200000,  # +2 horas
                "open": 48000.0,
                "high": 49000.0,
                "low": 47500.0,
                "close": 48500.0,
                "volume": 130.0,
                "symbol": self.symbol,
                "market": self.market,
                "tf": self.tf,
                "ingestion_ts": base_ts + 7200000
            },
            {
                "ts": base_ts,  # Primera hora
                "open": 47000.0,
                "high": 48000.0,
                "low": 46000.0,
                "close": 47500.0,
                "volume": 100.0,
                "symbol": self.symbol,
                "market": self.market,
                "tf": self.tf,
                "ingestion_ts": base_ts
            },
            {
                "ts": base_ts + 3600000,  # +1 hora
                "open": 47500.0,
                "high": 48500.0,
                "low": 47000.0,
                "close": 48000.0,
                "volume": 120.0,
                "symbol": self.symbol,
                "market": self.market,
                "tf": self.tf,
                "ingestion_ts": base_ts + 3600000
            }
        ]
        
        # Crear archivo raw
        self._create_test_parquet(self.raw_path, unordered_data, 2024, 1)
        
        # Ejecutar align_package.py
        cmd = [
            sys.executable,
            "data_pipeline/scripts/align_package.py",
            "--root", str(self.data_root),
            "--symbol", self.symbol,
            "--market", self.market,
            "--tfs", self.tf,
            "--from", "2024-01-01",
            "--to", "2024-02-01"
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True, cwd=Path(__file__).parent.parent.parent)
        
        # Verificar que el comando fue exitoso
        assert result.returncode == 0, f"align_package.py failed: {result.stderr}"
        
        # Leer y verificar el archivo aligned
        aligned_file = self.aligned_path / "year=2024" / "month=01" / "part-2024-01.parquet"
        table = pq.read_table(aligned_file)
        df = table.to_pandas()
        
        # Verificar que está ordenado cronológicamente
        assert len(df) == 3, f"Expected 3 rows, got {len(df)}"
        assert df['ts'].is_monotonic_increasing, "Data is not in chronological order"
        
        # Verificar timestamps específicos
        assert df.iloc[0]['ts'] == base_ts
        assert df.iloc[1]['ts'] == base_ts + 3600000
        assert df.iloc[2]['ts'] == base_ts + 7200000
    
    def test_align_timestamp_range(self):
        """Test que align_package.py respeta el rango de timestamps"""
        # Crear datos que van más allá del rango especificado
        base_ts = int(datetime(2024, 1, 1, tzinfo=timezone.utc).timestamp() * 1000)
        
        # Datos que incluyen diciembre 2023 y febrero 2024
        extended_data = [
            {
                "ts": base_ts - 86400000,  # 1 día antes (diciembre 2023)
                "open": 46000.0,
                "high": 47000.0,
                "low": 45000.0,
                "close": 46500.0,
                "volume": 90.0,
                "symbol": self.symbol,
                "market": self.market,
                "tf": self.tf,
                "ingestion_ts": base_ts - 86400000
            },
            {
                "ts": base_ts,  # Enero 2024
                "open": 47000.0,
                "high": 48000.0,
                "low": 46000.0,
                "close": 47500.0,
                "volume": 100.0,
                "symbol": self.symbol,
                "market": self.market,
                "tf": self.tf,
                "ingestion_ts": base_ts
            },
            {
                "ts": base_ts + 2678400000,  # 31 días después (febrero 2024)
                "open": 48000.0,
                "high": 49000.0,
                "low": 47500.0,
                "close": 48500.0,
                "volume": 110.0,
                "symbol": self.symbol,
                "market": self.market,
                "tf": self.tf,
                "ingestion_ts": base_ts + 2678400000
            }
        ]
        
        # Crear archivo raw
        self._create_test_parquet(self.raw_path, extended_data, 2024, 1)
        
        # Ejecutar align_package.py con rango específico
        cmd = [
            sys.executable,
            "data_pipeline/scripts/align_package.py",
            "--root", str(self.data_root),
            "--symbol", self.symbol,
            "--market", self.market,
            "--tfs", self.tf,
            "--from", "2024-01-01",
            "--to", "2024-02-01"
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True, cwd=Path(__file__).parent.parent.parent)
        
        # Verificar que el comando fue exitoso
        assert result.returncode == 0, f"align_package.py failed: {result.stderr}"
        
        # Leer y verificar el archivo aligned
        aligned_file = self.aligned_path / "year=2024" / "month=01" / "part-2024-01.parquet"
        table = pq.read_table(aligned_file)
        df = table.to_pandas()
        
        # Verificar que solo se incluyó el dato de enero 2024
        assert len(df) == 1, f"Expected 1 row, got {len(df)}"
        assert df.iloc[0]['ts'] == base_ts, "Should only include data from January 2024"
    
    def test_align_schema_consistency(self):
        """Test que align_package.py mantiene consistencia del esquema"""
        # Crear datos con esquema completo
        base_ts = int(datetime(2024, 1, 1, tzinfo=timezone.utc).timestamp() * 1000)
        
        data = [
            {
                "ts": base_ts,
                "open": 47000.0,
                "high": 48000.0,
                "low": 46000.0,
                "close": 47500.0,
                "volume": 100.0,
                "symbol": self.symbol,
                "market": self.market,
                "tf": self.tf,
                "ingestion_ts": base_ts
            }
        ]
        
        # Crear archivo raw
        self._create_test_parquet(self.raw_path, data, 2024, 1)
        
        # Ejecutar align_package.py
        cmd = [
            sys.executable,
            "data_pipeline/scripts/align_package.py",
            "--root", str(self.data_root),
            "--symbol", self.symbol,
            "--market", self.market,
            "--tfs", self.tf,
            "--from", "2024-01-01",
            "--to", "2024-02-01"
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True, cwd=Path(__file__).parent.parent.parent)
        
        # Verificar que el comando fue exitoso
        assert result.returncode == 0, f"align_package.py failed: {result.stderr}"
        
        # Leer y verificar el archivo aligned
        aligned_file = self.aligned_path / "year=2024" / "month=01" / "part-2024-01.parquet"
        table = pq.read_table(aligned_file)
        df = table.to_pandas()
        
        # Verificar que el esquema es correcto
        expected_columns = ["ts", "open", "high", "low", "close", "volume", "symbol", "market", "tf", "ingestion_ts"]
        assert list(df.columns) == expected_columns, f"Schema mismatch: {list(df.columns)}"
        
        # Verificar tipos de datos
        assert df['ts'].dtype == 'int64', "Timestamp should be int64"
        assert df['open'].dtype == 'float64', "Open should be float64"
        assert df['symbol'].dtype == 'string', "Symbol should be string"
        assert df['market'].dtype == 'string', "Market should be string"
        assert df['tf'].dtype == 'string', "TF should be string"
        
        # Verificar valores
        assert df.iloc[0]['symbol'] == self.symbol
        assert df.iloc[0]['market'] == self.market
        assert df.iloc[0]['tf'] == self.tf


if __name__ == "__main__":
    pytest.main([__file__])
