#!/usr/bin/env python3
"""
Script de análisis de resultados del entrenamiento.
Genera reportes detallados sobre el rendimiento del agente.

Mejoras implementadas:
- Logging estructurado en lugar de print
- Type hints completos
- Parametrización completa (símbolo, paths)
- Validación robusta de datos
- Manejo de errores mejorado
- Funciones modulares reutilizables
"""

import argparse
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

# Importar utilidades comunes
import sys
sys.path.append(str(Path(__file__).parent.parent))
from utilities.monitoring_utils import (
    setup_monitoring_logging, load_runs_data, load_metrics_data,
    calculate_trading_stats, create_summary_report, export_data_to_csv,
    format_currency, format_percentage, RunData, MonitoringError
)

logger = logging.getLogger(__name__)

def analyze_trading_performance(runs_data: List[RunData]) -> Dict[str, Any]:
    """
    Analiza el rendimiento de trading con estadísticas avanzadas.
    
    Args:
        runs_data: Lista de runs validados.
        
    Returns:
        Diccionario con estadísticas detalladas.
    """
    if not runs_data:
        logger.warning("No hay datos de runs para analizar")
        return {}
    
    try:
        # Convertir a DataFrame para análisis
        df_data = []
        for run in runs_data:
            row = {
                'final_balance': run.final_balance,
                'final_equity': run.final_equity,
                'trades_count': run.trades_count,
                'bankruptcy': run.bankruptcy,
                'drawdown_pct': run.drawdown_pct,
                'ts_start': run.ts_start,
                'ts_end': run.ts_end,
                'target_balance': run.target_balance,
                'elapsed_steps': run.elapsed_steps
            }
            df_data.append(row)
        
        df = pd.DataFrame(df_data)
        
        # Estadísticas básicas con manejo de NaN
        stats = {
            "total_runs": len(df),
            "avg_balance": df['final_balance'].mean(),
            "std_balance": df['final_balance'].std(),
            "avg_equity": df['final_equity'].mean(),
            "std_equity": df['final_equity'].std(),
            "avg_trades": df['trades_count'].mean(),
            "std_trades": df['trades_count'].std(),
            "bankruptcy_rate": df['bankruptcy'].mean(),
            "avg_drawdown": df['drawdown_pct'].mean(),
            "max_balance": df['final_balance'].max(),
            "min_balance": df['final_balance'].min(),
            "balance_range": df['final_balance'].max() - df['final_balance'].min(),
            "median_balance": df['final_balance'].median(),
            "balance_skewness": df['final_balance'].skew(),
            "balance_kurtosis": df['final_balance'].kurtosis()
        }
        
        # Análisis de evolución temporal si hay timestamps
        if 'ts_start' in df.columns and df['ts_start'].notna().any():
            try:
                df['ts_start'] = pd.to_datetime(df['ts_start'], unit='ms', errors='coerce')
                df = df.sort_values('ts_start').dropna(subset=['ts_start'])
                
                if len(df) >= 10:
                    # Calcular rolling averages
                    df['balance_ma_10'] = df['final_balance'].rolling(window=10, min_periods=1).mean()
                    df['trades_ma_10'] = df['trades_count'].rolling(window=10, min_periods=1).mean()
                    
                    # Tendencias
                    if len(df) >= 20:
                        stats['balance_trend'] = df['balance_ma_10'].iloc[-10:].mean() - df['balance_ma_10'].iloc[:10].mean()
                        stats['trades_trend'] = df['trades_ma_10'].iloc[-10:].mean() - df['trades_ma_10'].iloc[:10].mean()
                    
                    # Análisis de volatilidad temporal
                    stats['balance_volatility'] = df['final_balance'].rolling(window=10).std().mean()
                    
            except Exception as e:
                logger.warning(f"Error en análisis temporal: {e}")
        
        # Análisis de percentiles
        stats['balance_p25'] = df['final_balance'].quantile(0.25)
        stats['balance_p75'] = df['final_balance'].quantile(0.75)
        stats['balance_p90'] = df['final_balance'].quantile(0.90)
        stats['balance_p95'] = df['final_balance'].quantile(0.95)
        
        logger.info(f"Análisis completado para {len(runs_data)} runs")
        return stats
        
    except Exception as e:
        logger.error(f"Error analizando rendimiento de trading: {e}")
        return {}

def analyze_learning_curve(metrics_data: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Analiza la curva de aprendizaje con métricas avanzadas.
    
    Args:
        metrics_data: Lista de métricas de entrenamiento.
        
    Returns:
        Diccionario con estadísticas de aprendizaje.
    """
    if not metrics_data:
        logger.warning("No hay datos de métricas para analizar")
        return {}
    
    try:
        df = pd.DataFrame(metrics_data)
        
        # Estadísticas básicas
        stats = {
            "total_iterations": len(df),
            "total_timesteps": df['total_timesteps'].max() if 'total_timesteps' in df.columns else 0,
            "avg_fps": df['fps'].mean() if 'fps' in df.columns else 0,
            "total_time": df['time_elapsed'].max() if 'time_elapsed' in df.columns else 0
        }
        
        # Análisis de métricas de PPO si están disponibles
        ppo_metrics = ['policy_loss', 'value_loss', 'entropy_loss', 'learning_rate']
        for metric in ppo_metrics:
            if metric in df.columns:
                stats[f"avg_{metric}"] = df[metric].mean()
                stats[f"final_{metric}"] = df[metric].iloc[-1] if len(df) > 0 else 0
                stats[f"min_{metric}"] = df[metric].min()
                stats[f"max_{metric}"] = df[metric].max()
        
        # Análisis de tendencias de aprendizaje
        if 'ep_reward_mean' in df.columns and len(df) > 10:
            # Calcular tendencia de reward
            recent_rewards = df['ep_reward_mean'].iloc[-10:].mean()
            early_rewards = df['ep_reward_mean'].iloc[:10].mean()
            stats['reward_improvement'] = recent_rewards - early_rewards
            stats['reward_trend'] = "improving" if stats['reward_improvement'] > 0 else "declining"
        
        # Análisis de estabilidad
        if 'policy_loss' in df.columns and len(df) > 5:
            policy_loss_std = df['policy_loss'].rolling(window=5).std().mean()
            stats['policy_stability'] = 1.0 / (1.0 + policy_loss_std)  # Más alto = más estable
        
        logger.info(f"Análisis de curva de aprendizaje completado para {len(metrics_data)} iteraciones")
        return stats
        
    except Exception as e:
        logger.error(f"Error analizando curva de aprendizaje: {e}")
        return {}

def generate_report(runs_data: List[RunData], metrics_data: List[Dict[str, Any]], 
                   symbol: str, output_file: Optional[Path] = None) -> None:
    """
    Genera un reporte completo con análisis detallado.
    
    Args:
        runs_data: Lista de runs validados.
        metrics_data: Lista de métricas de entrenamiento.
        symbol: Símbolo analizado.
        output_file: Archivo de salida opcional.
    """
    logger.info(f"Generando reporte para {symbol}")
    
    # Crear reporte base
    report_lines = [
        "📊 REPORTE DE ANÁLISIS DEL ENTRENAMIENTO",
        "=" * 60,
        f"📅 Generado: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"📈 Símbolo: {symbol}",
        ""
    ]
    
    # Análisis de rendimiento de trading
    trading_stats = analyze_trading_performance(runs_data)
    if trading_stats:
        report_lines.extend([
            "📈 RENDIMIENTO DE TRADING:",
            f"   • Total de runs: {trading_stats['total_runs']:,}",
            f"   • Balance promedio: {format_currency(trading_stats['avg_balance'])} ± {format_currency(trading_stats['std_balance'])}",
            f"   • Equity promedio: {format_currency(trading_stats['avg_equity'])} ± {format_currency(trading_stats['std_equity'])}",
            f"   • Trades promedio: {trading_stats['avg_trades']:.1f} ± {trading_stats['std_trades']:.1f}",
            f"   • Tasa de bancarrota: {format_percentage(trading_stats['bankruptcy_rate'] * 100)}",
            f"   • Drawdown promedio: {format_percentage(trading_stats['avg_drawdown'])}",
            f"   • Balance máximo: {format_currency(trading_stats['max_balance'])}",
            f"   • Balance mínimo: {format_currency(trading_stats['min_balance'])}",
            f"   • Rango de balance: {format_currency(trading_stats['balance_range'])}",
            f"   • Mediana de balance: {format_currency(trading_stats['median_balance'])}",
            f"   • P25: {format_currency(trading_stats['balance_p25'])}, P75: {format_currency(trading_stats['balance_p75'])}",
            f"   • P90: {format_currency(trading_stats['balance_p90'])}, P95: {format_currency(trading_stats['balance_p95'])}"
        ])
        
        if 'balance_trend' in trading_stats:
            trend_emoji = "📈" if trading_stats['balance_trend'] > 0 else "📉"
            report_lines.append(f"   • Tendencia de balance: {trend_emoji} {format_currency(trading_stats['balance_trend'])}")
        
        if 'trades_trend' in trading_stats:
            report_lines.append(f"   • Tendencia de trades: {trading_stats['trades_trend']:.1f}")
        
        if 'balance_volatility' in trading_stats:
            report_lines.append(f"   • Volatilidad de balance: {format_currency(trading_stats['balance_volatility'])}")
        
        report_lines.append("")
    
    # Análisis de curva de aprendizaje
    learning_stats = analyze_learning_curve(metrics_data)
    if learning_stats:
        report_lines.extend([
            "🎓 CURVA DE APRENDIZAJE:",
            f"   • Iteraciones totales: {learning_stats['total_iterations']:,}",
            f"   • Timesteps totales: {learning_stats['total_timesteps']:,}",
            f"   • FPS promedio: {learning_stats['avg_fps']:.1f}",
            f"   • Tiempo total: {learning_stats['total_time']:.1f}s"
        ])
        
        # Métricas de PPO
        ppo_metrics = ['policy_loss', 'value_loss', 'entropy_loss', 'learning_rate']
        for metric in ppo_metrics:
            if f"avg_{metric}" in learning_stats:
                report_lines.append(f"   • {metric.replace('_', ' ').title()}: {learning_stats[f'avg_{metric}']:.4f}")
        
        if 'reward_improvement' in learning_stats:
            trend_emoji = "📈" if learning_stats['reward_improvement'] > 0 else "📉"
            report_lines.append(f"   • Mejora de reward: {trend_emoji} {learning_stats['reward_improvement']:.4f}")
        
        if 'policy_stability' in learning_stats:
            stability = "Alta" if learning_stats['policy_stability'] > 0.5 else "Media" if learning_stats['policy_stability'] > 0.3 else "Baja"
            report_lines.append(f"   • Estabilidad de política: {stability} ({learning_stats['policy_stability']:.3f})")
        
        report_lines.append("")
    
    # Recomendaciones inteligentes
    recommendations = generate_recommendations(trading_stats, learning_stats)
    if recommendations:
        report_lines.extend(["💡 RECOMENDACIONES:"] + recommendations + [""])
    
    # Escribir reporte
    report_text = "\n".join(report_lines)
    
    if output_file:
        try:
            output_file.write_text(report_text, encoding='utf-8')
            logger.info(f"Reporte guardado en {output_file}")
        except Exception as e:
            logger.error(f"Error guardando reporte: {e}")
    
    # Mostrar en consola
    logger.info("Reporte generado:")
    print(report_text)

def generate_recommendations(trading_stats: Dict[str, Any], 
                           learning_stats: Dict[str, Any]) -> List[str]:
    """
    Genera recomendaciones inteligentes basadas en las estadísticas.
    
    Args:
        trading_stats: Estadísticas de trading.
        learning_stats: Estadísticas de aprendizaje.
        
    Returns:
        Lista de recomendaciones.
    """
    recommendations = []
    
    if trading_stats:
        # Análisis de bancarrota
        if trading_stats['bankruptcy_rate'] > 0.5:
            recommendations.append("   ⚠️  Tasa de bancarrota alta - considerar ajustar risk management")
        elif trading_stats['bankruptcy_rate'] > 0.2:
            recommendations.append("   ⚠️  Tasa de bancarrota moderada - monitorear de cerca")
        
        # Análisis de actividad
        if trading_stats['avg_trades'] < 0.5:
            recommendations.append("   ⚠️  Pocos trades - considerar aumentar incentivos de actividad")
        elif trading_stats['avg_trades'] > 5:
            recommendations.append("   ⚠️  Muchos trades - considerar penalizar overtrading")
        
        # Análisis de tendencias
        if 'balance_trend' in trading_stats and trading_stats['balance_trend'] < 0:
            recommendations.append("   ⚠️  Tendencia negativa en balance - revisar estrategia")
        
        # Análisis de volatilidad
        if trading_stats['std_balance'] > 100:
            recommendations.append("   ⚠️  Alta volatilidad en balance - considerar estabilizar")
        
        # Análisis de distribución
        if 'balance_skewness' in trading_stats:
            if trading_stats['balance_skewness'] < -1:
                recommendations.append("   ⚠️  Distribución muy sesgada hacia la izquierda - revisar estrategia")
            elif trading_stats['balance_skewness'] > 1:
                recommendations.append("   ⚠️  Distribución muy sesgada hacia la derecha - posible overfitting")
    
    if learning_stats:
        # Análisis de estabilidad
        if 'policy_stability' in learning_stats and learning_stats['policy_stability'] < 0.3:
            recommendations.append("   ⚠️  Baja estabilidad de política - considerar reducir learning rate")
        
        # Análisis de mejora
        if 'reward_improvement' in learning_stats and learning_stats['reward_improvement'] < 0:
            recommendations.append("   ⚠️  Sin mejora en rewards - revisar configuración de entrenamiento")
    
    # Recomendaciones generales
    recommendations.extend([
        "   ✅ Revisar logs de entrenamiento para más detalles",
        "   ✅ Considerar ajustar parámetros de reward según comportamiento",
        "   ✅ Monitorear métricas de PPO (loss, entropy, etc.)",
        "   ✅ Usar análisis de correlación entre métricas para insights"
    ])
    
    return recommendations

def main():
    """Función principal con argumentos de línea de comandos."""
    parser = argparse.ArgumentParser(
        description="Analiza resultados de entrenamiento con métricas avanzadas",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ejemplos de uso:
  python analyze_results.py --symbol BTCUSDT
  python analyze_results.py --symbol ETHUSDT --models_root /path/to/models
  python analyze_results.py --symbol BTCUSDT --output report.txt --export-csv
  python analyze_results.py --symbol BTCUSDT --verbose
        """
    )
    
    parser.add_argument("--symbol", default="BTCUSDT", 
                       help="Símbolo a analizar (default: BTCUSDT)")
    parser.add_argument("--models_root", default="models",
                       help="Directorio raíz de modelos (default: models)")
    parser.add_argument("--output", type=Path,
                       help="Archivo de salida para el reporte")
    parser.add_argument("--export-csv", action="store_true",
                       help="Exportar datos a CSV")
    parser.add_argument("--verbose", "-v", action="store_true",
                       help="Mostrar información detallada")
    parser.add_argument("--log-file", type=Path,
                       help="Archivo de log")
    
    args = parser.parse_args()
    
    # Configurar logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    setup_monitoring_logging(level=log_level, log_file=args.log_file)
    
    logger.info(f"Iniciando análisis para {args.symbol}")
    logger.info(f"Directorio de modelos: {args.models_root}")
    
    try:
        # Cargar datos usando utilidades comunes
        logger.info("Cargando datos de runs...")
        runs_data = load_runs_data(args.symbol, args.models_root)
        
        logger.info("Cargando datos de métricas...")
        metrics_data = load_metrics_data(args.symbol, args.models_root)
        
        if not runs_data and not metrics_data:
            logger.error("No se encontraron datos para analizar")
            return 1
        
        # Generar reporte
        output_file = args.output
        if output_file and not output_file.suffix:
            output_file = output_file.with_suffix('.txt')
        
        generate_report(runs_data, metrics_data, args.symbol, output_file)
        
        # Exportar a CSV si se solicita
        if args.export_csv:
            if runs_data:
                runs_csv = Path(f"{args.symbol}_runs_analysis.csv")
                runs_df_data = []
                for run in runs_data:
                    runs_df_data.append({
                        'final_balance': run.final_balance,
                        'final_equity': run.final_equity,
                        'trades_count': run.trades_count,
                        'bankruptcy': run.bankruptcy,
                        'drawdown_pct': run.drawdown_pct,
                        'ts_start': run.ts_start,
                        'ts_end': run.ts_end,
                        'target_balance': run.target_balance,
                        'elapsed_steps': run.elapsed_steps
                    })
                export_data_to_csv(runs_df_data, runs_csv)
            
            if metrics_data:
                metrics_csv = Path(f"{args.symbol}_metrics_analysis.csv")
                export_data_to_csv(metrics_data, metrics_csv)
        
        logger.info("Análisis completado exitosamente")
        return 0
        
    except MonitoringError as e:
        logger.error(f"Error de monitoreo: {e}")
        return 1
    except Exception as e:
        logger.error(f"Error inesperado: {e}", exc_info=True)
        return 1

if __name__ == "__main__":
    exit(main())
