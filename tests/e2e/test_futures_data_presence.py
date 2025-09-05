# tests/test_futures_data_presence.py
"""
Test para validar que app.py aborta con mensaje claro si faltan datos aligned futures.
"""

import pytest
from unittest.mock import patch, MagicMock
from pathlib import Path
import tempfile
import os
import sys

# Agregar el directorio raíz al path para importar app
sys.path.insert(0, str(Path(__file__).parent.parent))
from app import _validate_historical_data


class TestFuturesDataPresence:
    """Test de validación de presencia de datos futures"""
    
    def setup_method(self):
        """Setup para cada test"""
        self.temp_dir = tempfile.mkdtemp()
        self.data_root = Path(self.temp_dir) / "data"
        self.data_root.mkdir(parents=True, exist_ok=True)
    
    def teardown_method(self):
        """Cleanup después de cada test"""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_validate_historical_data_missing_raw(self, capsys):
        """Test que detecta cuando faltan datos RAW"""
        symbol = "BTCUSDT"
        market = "futures"
        data_config = {
            "tfs": ["1m", "5m", "15m", "1h"],
            "stage": "aligned"
        }
        
        # Cambiar al directorio temporal
        original_cwd = os.getcwd()
        os.chdir(self.temp_dir)
        
        try:
            # Mock validate_alignment_and_gaps para que falle
            with patch('app.validate_alignment_and_gaps') as mock_validate:
                mock_validate.side_effect = Exception("No data found")
                
                result = _validate_historical_data(symbol, market, data_config)
                
                # Debería retornar False
                assert result is False
                
                # Verificar que se imprimió el mensaje correcto
                captured = capsys.readouterr()
                assert "No se encontraron datos RAW para BTCUSDT (futures)" in captured.out
                assert "COMANDOS SUGERIDOS PARA DESCARGAR:" in captured.out
                assert "python data_pipeline/scripts/download_history.py" in captured.out
                assert "python data_pipeline/scripts/align_package.py" in captured.out
                
        finally:
            os.chdir(original_cwd)
    
    def test_validate_historical_data_missing_aligned(self, capsys):
        """Test que detecta cuando faltan datos ALIGNED"""
        symbol = "BTCUSDT"
        market = "futures"
        data_config = {
            "tfs": ["1m", "5m", "15m", "1h"],
            "stage": "aligned"
        }
        
        # Crear estructura de datos RAW pero no ALIGNED
        raw_path = self.data_root / symbol / market / "raw"
        raw_path.mkdir(parents=True, exist_ok=True)
        
        # Crear un archivo parquet dummy en raw
        (raw_path / "1m" / "year=2024" / "month=01" / "part-2024-01.parquet").parent.mkdir(parents=True, exist_ok=True)
        (raw_path / "1m" / "year=2024" / "month=01" / "part-2024-01.parquet").touch()
        
        # Cambiar al directorio temporal
        original_cwd = os.getcwd()
        os.chdir(self.temp_dir)
        
        try:
            # Mock validate_alignment_and_gaps para que falle
            with patch('app.validate_alignment_and_gaps') as mock_validate:
                mock_validate.side_effect = Exception("No aligned data found")
                
                result = _validate_historical_data(symbol, market, data_config)
                
                # Debería retornar False
                assert result is False
                
                # Verificar que se imprimió el mensaje correcto
                captured = capsys.readouterr()
                assert "No se encontraron datos ALIGNED para BTCUSDT (futures)" in captured.out
                assert "COMANDO SUGERIDO PARA ALINEAR:" in captured.out
                assert "python data_pipeline/scripts/align_package.py" in captured.out
                
        finally:
            os.chdir(original_cwd)
    
    def test_validate_historical_data_success(self):
        """Test que valida correctamente cuando los datos están presentes"""
        symbol = "BTCUSDT"
        market = "futures"
        data_config = {
            "tfs": ["1m", "5m", "15m", "1h"],
            "stage": "aligned"
        }
        
        # Cambiar al directorio temporal
        original_cwd = os.getcwd()
        os.chdir(self.temp_dir)
        
        try:
            # Mock validate_alignment_and_gaps para que funcione
            with patch('app.validate_alignment_and_gaps') as mock_validate:
                mock_validate.return_value = "Data validation successful"
                
                result = _validate_historical_data(symbol, market, data_config)
                
                # Debería retornar True
                assert result is True
                
                # Verificar que se llamó con los parámetros correctos
                mock_validate.assert_called_once_with(
                    root="data",
                    symbol=symbol,
                    market=market,
                    tfs=data_config["tfs"],
                    stage=data_config["stage"],
                    allow_gaps=False
                )
                
        finally:
            os.chdir(original_cwd)
    
    def test_validate_historical_data_autocorrection_success(self):
        """Test que la autocorrección funciona correctamente"""
        symbol = "BTCUSDT"
        market = "futures"
        data_config = {
            "tfs": ["1m", "5m", "15m", "1h"],
            "stage": "aligned"
        }
        
        # Cambiar al directorio temporal
        original_cwd = os.getcwd()
        os.chdir(self.temp_dir)
        
        try:
            # Mock validate_alignment_and_gaps para que falle primero y luego funcione
            with patch('app.validate_alignment_and_gaps') as mock_validate:
                mock_validate.side_effect = [
                    Exception("No data found"),  # Primera llamada falla
                    "Data validation successful"  # Segunda llamada funciona
                ]
                
                # Mock os.system para que simule éxito
                with patch('app.os.system') as mock_system:
                    mock_system.return_value = 0
                    
                    result = _validate_historical_data(symbol, market, data_config)
                    
                    # Debería retornar True después de la autocorrección
                    assert result is True
                    
                    # Verificar que se ejecutó el comando de alineación
                    mock_system.assert_called_once()
                    call_args = mock_system.call_args[0][0]
                    assert "align_package.py" in call_args
                    assert f"--symbol {symbol}" in call_args
                    assert f"--market {market}" in call_args
                    
        finally:
            os.chdir(original_cwd)
    
    def test_validate_historical_data_autocorrection_failure(self, capsys):
        """Test que maneja correctamente el fallo de autocorrección"""
        symbol = "BTCUSDT"
        market = "futures"
        data_config = {
            "tfs": ["1m", "5m", "15m", "1h"],
            "stage": "aligned"
        }
        
        # Cambiar al directorio temporal
        original_cwd = os.getcwd()
        os.chdir(self.temp_dir)
        
        try:
            # Mock validate_alignment_and_gaps para que siempre falle
            with patch('app.validate_alignment_and_gaps') as mock_validate:
                mock_validate.side_effect = Exception("No data found")
                
                # Mock os.system para que simule fallo
                with patch('app.os.system') as mock_system:
                    mock_system.return_value = 1  # Código de error
                    
                    result = _validate_historical_data(symbol, market, data_config)
                    
                    # Debería retornar False
                    assert result is False
                    
                    # Verificar que se imprimió el mensaje de error
                    captured = capsys.readouterr()
                    assert "Alineación falló con código 1" in captured.out
                    
        finally:
            os.chdir(original_cwd)
    
    def test_validate_historical_data_different_tfs(self, capsys):
        """Test que maneja diferentes configuraciones de TFs"""
        symbol = "ETHUSDT"
        market = "futures"
        data_config = {
            "tfs": ["1m", "5m", "1h"],  # TFs diferentes
            "stage": "aligned"
        }
        
        # Cambiar al directorio temporal
        original_cwd = os.getcwd()
        os.chdir(self.temp_dir)
        
        try:
            # Mock validate_alignment_and_gaps para que falle
            with patch('app.validate_alignment_and_gaps') as mock_validate:
                mock_validate.side_effect = Exception("No data found")
                
                result = _validate_historical_data(symbol, market, data_config)
                
                # Debería retornar False
                assert result is False
                
                # Verificar que se usaron los TFs correctos en el mensaje
                captured = capsys.readouterr()
                assert "1m,5m,1h" in captured.out  # TFs en el comando sugerido
                
        finally:
            os.chdir(original_cwd)


if __name__ == "__main__":
    pytest.main([__file__])
