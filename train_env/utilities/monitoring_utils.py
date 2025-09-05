# train_env/utilities/monitoring_utils.py
"""
Utilidades comunes para scripts de monitoreo de entrenamiento.
Funciones compartidas para carga de datos, validación y logging.
"""

from __future__ import annotations
import json
import logging
from pathlib import Path
from typing import List, Dict, Optional, Any, Union
from datetime import datetime
import pandas as pd
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class RunData:
    """Estructura de datos para un run de entrenamiento."""
    final_balance: float
    final_equity: float
    trades_count: int
    bankruptcy: bool
    drawdown_pct: float
    ts_start: Optional[int] = None
    ts_end: Optional[int] = None
    target_balance: Optional[float] = None
    elapsed_steps: Optional[int] = None
    raw_data: Optional[Dict[str, Any]] = None

class MonitoringError(Exception):
    """Excepción base para errores de monitoreo."""
    pass

class DataValidationError(MonitoringError):
    """Error en validación de datos."""
    pass

class FileNotFoundError(MonitoringError):
    """Error cuando no se encuentra un archivo."""
    pass

def setup_monitoring_logging(level: int = logging.INFO, log_file: Optional[str] = None) -> None:
    """
    Configura logging para scripts de monitoreo.
    
    Args:
        level: Nivel de logging.
        log_file: Archivo de log opcional.
    """
    handlers = [logging.StreamHandler()]
    if log_file:
        handlers.append(logging.FileHandler(log_file))
    
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=handlers
    )

def load_jsonl_safe(file_path: Path, required_fields: Optional[List[str]] = None) -> List[Dict[str, Any]]:
    """
    Carga un archivo JSONL de forma segura con validación.
    
    Args:
        file_path: Ruta al archivo JSONL.
        required_fields: Campos requeridos en cada línea.
        
    Returns:
        Lista de diccionarios con datos válidos.
        
    Raises:
        FileNotFoundError: Si el archivo no existe.
        DataValidationError: Si hay problemas de validación.
    """
    if not file_path.exists():
        raise FileNotFoundError(f"Archivo no encontrado: {file_path}")
    
    data: List[Dict[str, Any]] = []
    errors = []
    
    try:
        with file_path.open('r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                
                try:
                    run_data = json.loads(line)
                    if not isinstance(run_data, dict):
                        errors.append(f"Línea {line_num}: No es un objeto JSON válido")
                        continue
                    
                    # Validar campos requeridos
                    if required_fields:
                        missing_fields = [field for field in required_fields if field not in run_data]
                        if missing_fields:
                            errors.append(f"Línea {line_num}: Campos faltantes: {missing_fields}")
                            continue
                    
                    data.append(run_data)
                    
                except json.JSONDecodeError as e:
                    errors.append(f"Línea {line_num}: Error JSON: {e}")
                    continue
    
    except Exception as e:
        raise DataValidationError(f"Error leyendo archivo {file_path}: {e}")
    
    if errors:
        logger.warning(f"Encontrados {len(errors)} errores en {file_path}:")
        for error in errors[:5]:  # Mostrar solo los primeros 5 errores
            logger.warning(f"  {error}")
        if len(errors) > 5:
            logger.warning(f"  ... y {len(errors) - 5} errores más")
    
    logger.info(f"Cargados {len(data)} registros válidos de {file_path}")
    return data

def validate_run_data(run_data: Dict[str, Any]) -> RunData:
    """
    Valida y convierte datos de run a estructura tipada.
    
    Args:
        run_data: Diccionario con datos del run.
        
    Returns:
        RunData validado y tipado.
        
    Raises:
        DataValidationError: Si los datos no son válidos.
    """
    try:
        # Campos requeridos
        required_fields = ['final_balance', 'final_equity', 'trades_count', 'bankruptcy', 'drawdown_pct']
        missing_fields = [field for field in required_fields if field not in run_data]
        if missing_fields:
            raise DataValidationError(f"Campos faltantes: {missing_fields}")
        
        # Convertir tipos de forma segura
        final_balance = float(run_data.get('final_balance', 0))
        final_equity = float(run_data.get('final_equity', 0))
        trades_count = int(run_data.get('trades_count', 0))
        bankruptcy = bool(run_data.get('bankruptcy', False))
        drawdown_pct = float(run_data.get('drawdown_pct', 0))
        
        # Campos opcionales
        ts_start = run_data.get('ts_start')
        if ts_start is not None:
            ts_start = int(ts_start)
        
        ts_end = run_data.get('ts_end')
        if ts_end is not None:
            ts_end = int(ts_end)
        
        target_balance = run_data.get('target_balance')
        if target_balance is not None:
            target_balance = float(target_balance)
        
        elapsed_steps = run_data.get('elapsed_steps')
        if elapsed_steps is not None:
            elapsed_steps = int(elapsed_steps)
        
        return RunData(
            final_balance=final_balance,
            final_equity=final_equity,
            trades_count=trades_count,
            bankruptcy=bankruptcy,
            drawdown_pct=drawdown_pct,
            ts_start=ts_start,
            ts_end=ts_end,
            target_balance=target_balance,
            elapsed_steps=elapsed_steps,
            raw_data=run_data
        )
        
    except (ValueError, TypeError) as e:
        raise DataValidationError(f"Error convirtiendo tipos: {e}")

def load_runs_data(symbol: str, models_root: str = "models", 
                  required_fields: Optional[List[str]] = None) -> List[RunData]:
    """
    Carga y valida datos de runs para un símbolo.
    
    Args:
        symbol: Símbolo a cargar.
        models_root: Directorio raíz de modelos.
        required_fields: Campos requeridos.
        
    Returns:
        Lista de RunData validados.
    """
    models_path = Path(models_root) / symbol
    runs_file = models_path / f"{symbol}_runs.jsonl"
    
    if not runs_file.exists():
        logger.warning(f"Archivo de runs no encontrado: {runs_file}")
        return []
    
    # Campos requeridos por defecto
    if required_fields is None:
        required_fields = ['final_balance', 'final_equity', 'trades_count', 'bankruptcy', 'drawdown_pct']
    
    try:
        raw_data = load_jsonl_safe(runs_file, required_fields)
        runs_data = []
        
        for i, run_dict in enumerate(raw_data):
            try:
                run_data = validate_run_data(run_dict)
                runs_data.append(run_data)
            except DataValidationError as e:
                logger.warning(f"Run {i} inválido: {e}")
                continue
        
        logger.info(f"Cargados {len(runs_data)} runs válidos para {symbol}")
        return runs_data
        
    except Exception as e:
        logger.error(f"Error cargando runs para {symbol}: {e}")
        return []

def load_metrics_data(symbol: str, models_root: str = "models") -> List[Dict[str, Any]]:
    """
    Carga datos de métricas de entrenamiento.
    
    Args:
        symbol: Símbolo a cargar.
        models_root: Directorio raíz de modelos.
        
    Returns:
        Lista de métricas.
    """
    models_path = Path(models_root) / symbol
    metrics_file = models_path / f"{symbol}_train_metrics.jsonl"
    
    if not metrics_file.exists():
        logger.warning(f"Archivo de métricas no encontrado: {metrics_file}")
        return []
    
    try:
        return load_jsonl_safe(metrics_file)
    except Exception as e:
        logger.error(f"Error cargando métricas para {symbol}: {e}")
        return []

def calculate_trading_stats(runs_data: List[RunData]) -> Dict[str, Any]:
    """
    Calcula estadísticas de trading a partir de runs.
    
    Args:
        runs_data: Lista de runs validados.
        
    Returns:
        Diccionario con estadísticas.
    """
    if not runs_data:
        return {
            "total_runs": 0,
            "avg_balance": 0.0,
            "avg_equity": 0.0,
            "avg_trades": 0.0,
            "bankruptcy_rate": 0.0,
            "avg_drawdown": 0.0,
            "best_balance": 0.0,
            "worst_balance": 0.0
        }
    
    balances = [run.final_balance for run in runs_data]
    equities = [run.final_equity for run in runs_data]
    trades = [run.trades_count for run in runs_data]
    bankruptcies = [run.bankruptcy for run in runs_data]
    drawdowns = [run.drawdown_pct for run in runs_data]
    
    return {
        "total_runs": len(runs_data),
        "avg_balance": sum(balances) / len(balances),
        "avg_equity": sum(equities) / len(equities),
        "avg_trades": sum(trades) / len(trades),
        "bankruptcy_rate": sum(bankruptcies) / len(bankruptcies),
        "avg_drawdown": sum(drawdowns) / len(drawdowns),
        "best_balance": max(balances),
        "worst_balance": min(balances),
        "balance_std": pd.Series(balances).std() if len(balances) > 1 else 0.0,
        "equity_std": pd.Series(equities).std() if len(equities) > 1 else 0.0
    }

def format_currency(value: float, currency: str = "USDT") -> str:
    """Formatea un valor como moneda."""
    return f"{value:,.2f} {currency}"

def format_percentage(value: float, decimals: int = 2) -> str:
    """Formatea un valor como porcentaje."""
    return f"{value:.{decimals}f}%"

def format_timestamp(timestamp: int) -> str:
    """Formatea un timestamp a string legible."""
    try:
        dt = datetime.fromtimestamp(timestamp / 1000)
        return dt.strftime("%Y-%m-%d %H:%M:%S")
    except (ValueError, OSError):
        return str(timestamp)

def discover_symbols(models_root: str = "models") -> List[str]:
    """
    Descubre símbolos disponibles en el directorio de modelos.
    
    Args:
        models_root: Directorio raíz de modelos.
        
    Returns:
        Lista de símbolos encontrados.
    """
    models_path = Path(models_root)
    if not models_path.exists():
        logger.warning(f"Directorio de modelos no encontrado: {models_path}")
        return []
    
    symbols = []
    for item in models_path.iterdir():
        if item.is_dir() and (item / f"{item.name}_runs.jsonl").exists():
            symbols.append(item.name)
    
    logger.info(f"Símbolos encontrados: {symbols}")
    return symbols

def export_data_to_csv(data: List[Dict[str, Any]], output_path: Path) -> None:
    """
    Exporta datos a CSV.
    
    Args:
        data: Datos a exportar.
        output_path: Ruta de salida.
    """
    if not data:
        logger.warning("No hay datos para exportar")
        return
    
    try:
        df = pd.DataFrame(data)
        df.to_csv(output_path, index=False)
        logger.info(f"Datos exportados a {output_path}")
    except Exception as e:
        logger.error(f"Error exportando datos: {e}")

def create_summary_report(runs_data: List[RunData], 
                         metrics_data: List[Dict[str, Any]],
                         symbol: str) -> str:
    """
    Crea un reporte resumido en texto.
    
    Args:
        runs_data: Datos de runs.
        metrics_data: Datos de métricas.
        symbol: Símbolo analizado.
        
    Returns:
        Reporte en formato texto.
    """
    stats = calculate_trading_stats(runs_data)
    
    report = f"""
=== REPORTE DE ENTRENAMIENTO - {symbol} ===
Fecha: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

ESTADÍSTICAS DE RUNS:
- Total de runs: {stats['total_runs']}
- Balance promedio: {format_currency(stats['avg_balance'])}
- Equity promedio: {format_currency(stats['avg_equity'])}
- Trades promedio: {stats['avg_trades']:.1f}
- Tasa de bancarrota: {format_percentage(stats['bankruptcy_rate'] * 100)}
- Drawdown promedio: {format_percentage(stats['avg_drawdown'])}
- Mejor balance: {format_currency(stats['best_balance'])}
- Peor balance: {format_currency(stats['worst_balance'])}

MÉTRICAS DE ENTRENAMIENTO:
- Total de métricas: {len(metrics_data)}
"""
    
    if metrics_data:
        # Agregar estadísticas de métricas si están disponibles
        report += f"- Última métrica: {metrics_data[-1].get('timestamp', 'N/A')}\n"
    
    return report
