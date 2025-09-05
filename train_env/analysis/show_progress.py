# train_env/analysis/show_progress.py
"""
Muestra la evolución de las runs para un símbolo concreto con visualizaciones avanzadas.

Mejoras implementadas:
- Logging estructurado en lugar de print
- Type hints completos
- Validación robusta de datos
- Múltiples tipos de gráficos
- Exportación de gráficos
- Manejo de errores mejorado
- Funciones modulares reutilizables

Lee:
  - models/{symbol}/{symbol}_runs.jsonl
  - models/{symbol}/{symbol}_progress.json

Genera:
  - Reporte detallado con logging
  - Gráficos múltiples (equity, trades, drawdown)
  - Exportación opcional de gráficos
"""

from __future__ import annotations
import argparse
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime
import numpy as np

# Importar utilidades comunes
import sys
sys.path.append(str(Path(__file__).parent.parent))
from utilities.monitoring_utils import (
    setup_monitoring_logging, load_runs_data, load_metrics_data,
    format_currency, format_percentage, RunData, MonitoringError
)

logger = logging.getLogger(__name__)

def load_progress(symbol: str, models_root: str = "models") -> Dict[str, Any]:
    """
    Carga datos de progreso de entrenamiento.
    
    Args:
        symbol: Símbolo a cargar.
        models_root: Directorio raíz de modelos.
        
    Returns:
        Diccionario con datos de progreso.
    """
    progress_file = Path(models_root) / symbol / f"{symbol}_progress.json"
    if not progress_file.exists():
        logger.warning(f"Archivo de progreso no encontrado: {progress_file}")
        return {}
    
    try:
        import json
        return json.loads(progress_file.read_text(encoding="utf-8"))
    except Exception as e:
        logger.error(f"Error cargando progreso para {symbol}: {e}")
        return {}

def create_equity_plot(runs_data: List[RunData], symbol: str, 
                      target_balance: Optional[float] = None,
                      save_path: Optional[Path] = None) -> None:
    """
    Crea gráfico de evolución de equity.
    
    Args:
        runs_data: Lista de runs validados.
        symbol: Símbolo a graficar.
        target_balance: Balance objetivo opcional.
        save_path: Ruta para guardar el gráfico.
    """
    if not runs_data:
        logger.warning("No hay datos para graficar")
        return
    
    try:
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Preparar datos
        x = list(range(1, len(runs_data) + 1))
        y_equity = [run.final_equity for run in runs_data]
        y_balance = [run.final_balance for run in runs_data]
        
        # Gráfico principal
        ax.plot(x, y_equity, marker="o", label="Equity Final", linewidth=2, markersize=4)
        ax.plot(x, y_balance, marker="s", label="Balance Final", linewidth=2, markersize=4, alpha=0.7)
        
        # Línea objetivo
        if target_balance:
            ax.axhline(target_balance, color="red", linestyle="--", 
                      label=f"Objetivo {format_currency(target_balance)}", linewidth=2)
        
        # Líneas de tendencia
        if len(runs_data) > 5:
            z_equity = np.polyfit(x, y_equity, 1)
            p_equity = np.poly1d(z_equity)
            ax.plot(x, p_equity(x), "b--", alpha=0.5, label="Tendencia Equity")
        
        # Configuración del gráfico
        ax.set_xlabel("Número de Run", fontsize=12)
        ax.set_ylabel("Valor (USDT)", fontsize=12)
        ax.set_title(f"Evolución de Equity y Balance - {symbol}", fontsize=14, fontweight="bold")
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        
        # Formatear eje Y
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: format_currency(x)))
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Gráfico guardado en {save_path}")
        
        plt.show()
        
    except Exception as e:
        logger.error(f"Error creando gráfico de equity: {e}")

def create_trades_plot(runs_data: List[RunData], symbol: str,
                      save_path: Optional[Path] = None) -> None:
    """
    Crea gráfico de evolución de trades.
    
    Args:
        runs_data: Lista de runs validados.
        symbol: Símbolo a graficar.
        save_path: Ruta para guardar el gráfico.
    """
    if not runs_data:
        return
    
    try:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
        
        x = list(range(1, len(runs_data) + 1))
        trades = [run.trades_count for run in runs_data]
        bankruptcies = [run.bankruptcy for run in runs_data]
        
        # Gráfico de trades
        ax1.plot(x, trades, marker="o", color="green", linewidth=2, markersize=4)
        ax1.set_ylabel("Número de Trades", fontsize=12)
        ax1.set_title(f"Evolución de Trades - {symbol}", fontsize=14, fontweight="bold")
        ax1.grid(True, alpha=0.3)
        
        # Línea de tendencia para trades
        if len(runs_data) > 5:
            z_trades = np.polyfit(x, trades, 1)
            p_trades = np.poly1d(z_trades)
            ax1.plot(x, p_trades(x), "g--", alpha=0.5, label="Tendencia")
            ax1.legend()
        
        # Gráfico de bancarrotas
        bankruptcy_indices = [i+1 for i, b in enumerate(bankruptcies) if b]
        ax2.scatter(bankruptcy_indices, [1]*len(bankruptcy_indices), 
                   color="red", marker="x", s=100, label="Bancarrota")
        ax2.set_xlabel("Número de Run", fontsize=12)
        ax2.set_ylabel("Bancarrota", fontsize=12)
        ax2.set_title("Ocurrencias de Bancarrota", fontsize=12)
        ax2.set_ylim(-0.1, 1.1)
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Gráfico de trades guardado en {save_path}")
        
        plt.show()
        
    except Exception as e:
        logger.error(f"Error creando gráfico de trades: {e}")

def create_drawdown_plot(runs_data: List[RunData], symbol: str,
                        save_path: Optional[Path] = None) -> None:
    """
    Crea gráfico de evolución de drawdown.
    
    Args:
        runs_data: Lista de runs validados.
        symbol: Símbolo a graficar.
        save_path: Ruta para guardar el gráfico.
    """
    if not runs_data:
        return
    
    try:
        fig, ax = plt.subplots(figsize=(12, 6))
        
        x = list(range(1, len(runs_data) + 1))
        drawdowns = [run.drawdown_pct for run in runs_data]
        
        # Gráfico de drawdown
        colors = ['red' if dd > 50 else 'orange' if dd > 20 else 'green' for dd in drawdowns]
        ax.scatter(x, drawdowns, c=colors, alpha=0.7, s=50)
        
        # Línea de tendencia
        if len(runs_data) > 5:
            z_dd = np.polyfit(x, drawdowns, 1)
            p_dd = np.poly1d(z_dd)
            ax.plot(x, p_dd(x), "b--", alpha=0.5, label="Tendencia")
            ax.legend()
        
        # Líneas de referencia
        ax.axhline(20, color="orange", linestyle=":", alpha=0.7, label="Advertencia (20%)")
        ax.axhline(50, color="red", linestyle=":", alpha=0.7, label="Crítico (50%)")
        
        ax.set_xlabel("Número de Run", fontsize=12)
        ax.set_ylabel("Drawdown (%)", fontsize=12)
        ax.set_title(f"Evolución de Drawdown - {symbol}", fontsize=14, fontweight="bold")
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Gráfico de drawdown guardado en {save_path}")
        
        plt.show()
        
    except Exception as e:
        logger.error(f"Error creando gráfico de drawdown: {e}")

def generate_progress_report(runs_data: List[RunData], progress_data: Dict[str, Any], 
                           symbol: str) -> None:
    """
    Genera reporte de progreso detallado.
    
    Args:
        runs_data: Lista de runs validados.
        progress_data: Datos de progreso.
        symbol: Símbolo analizado.
    """
    if not runs_data:
        logger.warning("No hay datos de runs para reportar")
        return
    
    logger.info(f"Generando reporte de progreso para {symbol}")
    
    # Estadísticas básicas
    total_runs = len(runs_data)
    avg_equity = sum(run.final_equity for run in runs_data) / total_runs
    avg_balance = sum(run.final_balance for run in runs_data) / total_runs
    avg_trades = sum(run.trades_count for run in runs_data) / total_runs
    bankruptcy_count = sum(1 for run in runs_data if run.bankruptcy)
    bankruptcy_rate = bankruptcy_count / total_runs
    
    # Mejor y peor run
    best_run = max(runs_data, key=lambda r: r.final_equity)
    worst_run = min(runs_data, key=lambda r: r.final_equity)
    
    # Reporte
    report_lines = [
        f"📊 REPORTE DE PROGRESO - {symbol}",
        "=" * 50,
        f"📅 Generado: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "📈 ESTADÍSTICAS GENERALES:",
        f"   • Total de runs: {total_runs:,}",
        f"   • Equity promedio: {format_currency(avg_equity)}",
        f"   • Balance promedio: {format_currency(avg_balance)}",
        f"   • Trades promedio: {avg_trades:.1f}",
        f"   • Tasa de bancarrota: {format_percentage(bankruptcy_rate * 100)}",
        "",
        "🏆 MEJOR RUN:",
        f"   • Run #{runs_data.index(best_run) + 1}",
        f"   • Equity: {format_currency(best_run.final_equity)}",
        f"   • Balance: {format_currency(best_run.final_balance)}",
        f"   • Trades: {best_run.trades_count}",
        f"   • Drawdown: {format_percentage(best_run.drawdown_pct)}",
        "",
        "📉 PEOR RUN:",
        f"   • Run #{runs_data.index(worst_run) + 1}",
        f"   • Equity: {format_currency(worst_run.final_equity)}",
        f"   • Balance: {format_currency(worst_run.final_balance)}",
        f"   • Trades: {worst_run.trades_count}",
        f"   • Drawdown: {format_percentage(worst_run.drawdown_pct)}"
    ]
    
    # Datos de progreso si están disponibles
    if progress_data:
        report_lines.extend([
            "",
            "📊 DATOS DE PROGRESO:",
            f"   • Episodios completados: {progress_data.get('episodes_completed', 'N/A')}",
            f"   • Mejor reward: {progress_data.get('best_reward', 'N/A')}",
            f"   • Reward actual: {progress_data.get('current_reward', 'N/A')}",
            f"   • Learning rate: {progress_data.get('learning_rate', 'N/A')}"
        ])
    
    report_text = "\n".join(report_lines)
    logger.info("Reporte de progreso generado:")
    print(report_text)

def main():
    """Función principal con argumentos de línea de comandos mejorados."""
    parser = argparse.ArgumentParser(
        description="Muestra la evolución de runs con visualizaciones avanzadas",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ejemplos de uso:
  python show_progress.py --symbol BTCUSDT
  python show_progress.py --symbol ETHUSDT --models-root /path/to/models
  python show_progress.py --symbol BTCUSDT --save-plots --output-dir plots/
  python show_progress.py --symbol BTCUSDT --plot-type all --verbose
        """
    )
    
    parser.add_argument("--symbol", required=True, help="Símbolo a analizar (ej: BTCUSDT)")
    parser.add_argument("--models-root", default="models", help="Directorio raíz de modelos")
    parser.add_argument("--plot-type", choices=["equity", "trades", "drawdown", "all"], 
                       default="all", help="Tipo de gráfico a mostrar")
    parser.add_argument("--save-plots", action="store_true", 
                       help="Guardar gráficos en archivos")
    parser.add_argument("--output-dir", type=Path, default=Path("plots"),
                       help="Directorio para guardar gráficos")
    parser.add_argument("--verbose", "-v", action="store_true",
                       help="Mostrar información detallada")
    parser.add_argument("--log-file", type=Path,
                       help="Archivo de log")
    
    args = parser.parse_args()
    
    # Configurar logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    setup_monitoring_logging(level=log_level, log_file=args.log_file)
    
    logger.info(f"Iniciando visualización de progreso para {args.symbol}")
    logger.info(f"Directorio de modelos: {args.models_root}")
    
    try:
        # Cargar datos usando utilidades comunes
        logger.info("Cargando datos de runs...")
        runs_data = load_runs_data(args.symbol, args.models_root)
        
        logger.info("Cargando datos de progreso...")
        progress_data = load_progress(args.symbol, args.models_root)
        
        if not runs_data:
            logger.warning("No hay runs registradas todavía")
            return 1
        
        # Generar reporte de progreso
        generate_progress_report(runs_data, progress_data, args.symbol)
        
        # Crear directorio de salida si se solicita guardar
        if args.save_plots:
            args.output_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"Grabando gráficos en {args.output_dir}")
        
        # Determinar balance objetivo
        target_balance = None
        if runs_data and runs_data[0].target_balance:
            target_balance = runs_data[0].target_balance
        
        # Crear gráficos según el tipo solicitado
        if args.plot_type in ["equity", "all"]:
            equity_path = args.output_dir / f"{args.symbol}_equity.png" if args.save_plots else None
            create_equity_plot(runs_data, args.symbol, target_balance, equity_path)
        
        if args.plot_type in ["trades", "all"]:
            trades_path = args.output_dir / f"{args.symbol}_trades.png" if args.save_plots else None
            create_trades_plot(runs_data, args.symbol, trades_path)
        
        if args.plot_type in ["drawdown", "all"]:
            drawdown_path = args.output_dir / f"{args.symbol}_drawdown.png" if args.save_plots else None
            create_drawdown_plot(runs_data, args.symbol, drawdown_path)
        
        logger.info("Visualización completada exitosamente")
        return 0
        
    except MonitoringError as e:
        logger.error(f"Error de monitoreo: {e}")
        return 1
    except Exception as e:
        logger.error(f"Error inesperado: {e}", exc_info=True)
        return 1

if __name__ == "__main__":
    exit(main())
