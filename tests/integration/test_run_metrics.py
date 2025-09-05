# tests/test_run_metrics.py
# Descripción: Tests para validar métricas profesionales de runs

import pytest
import json
import tempfile
from pathlib import Path
from base_env.metrics.trade_metrics import TradeMetrics, TradeRecord
from base_env.logging.run_logger import RunLogger

class TestTradeMetrics:
    """Tests para el cálculo de métricas profesionales de trades"""
    
    def test_empty_metrics(self):
        """Test con métricas vacías"""
        metrics = TradeMetrics()
        result = metrics.calculate_metrics()
        
        assert result["trades_count"] == 0
        assert result["win_rate_trades"] == 0.0
        assert result["avg_trade_pnl"] == 0.0
        assert result["avg_holding_bars"] == 0.0
        assert result["max_consecutive_wins"] == 0
        assert result["max_consecutive_losses"] == 0
        assert result["gross_profit"] == 0.0
        assert result["gross_loss"] == 0.0
        assert result["profit_factor"] is None
    
    def test_win_rate_calculation(self):
        """Test cálculo de win rate con 6 ganadores de 10 trades"""
        metrics = TradeMetrics()
        
        # 6 trades ganadores
        for i in range(6):
            trade = TradeRecord(
                entry_price=100.0,
                exit_price=110.0,  # +10% ganancia
                qty=1.0,
                side=1,
                realized_pnl=10.0,
                bars_held=5
            )
            metrics.add_trade(trade)
        
        # 4 trades perdedores
        for i in range(4):
            trade = TradeRecord(
                entry_price=100.0,
                exit_price=95.0,  # -5% pérdida
                qty=1.0,
                side=1,
                realized_pnl=-5.0,
                bars_held=3
            )
            metrics.add_trade(trade)
        
        result = metrics.calculate_metrics()
        
        assert result["trades_count"] == 10
        assert result["win_rate_trades"] == 60.0  # 6/10 * 100
        assert result["winning_trades"] == 6
        assert result["losing_trades"] == 4
    
    def test_avg_trade_pnl_calculation(self):
        """Test cálculo de PnL promedio por trade"""
        metrics = TradeMetrics()
        
        # Trade ganador: +50 USDT
        trade1 = TradeRecord(
            entry_price=100.0,
            exit_price=150.0,
            qty=1.0,
            side=1,
            realized_pnl=50.0,
            bars_held=10
        )
        metrics.add_trade(trade1)
        
        # Trade perdedor: -20 USDT
        trade2 = TradeRecord(
            entry_price=100.0,
            exit_price=80.0,
            qty=1.0,
            side=1,
            realized_pnl=-20.0,
            bars_held=5
        )
        metrics.add_trade(trade2)
        
        result = metrics.calculate_metrics()
        
        # PnL promedio: (50 + (-20)) / 2 = 15.0
        assert result["avg_trade_pnl"] == 15.0
        assert result["total_pnl"] == 30.0
    
    def test_consecutive_streaks(self):
        """Test cálculo de rachas consecutivas"""
        metrics = TradeMetrics()
        
        # Secuencia: W-W-L-L-L-W-W-W-L
        trades_data = [
            (100.0, 110.0, 10.0),   # W
            (100.0, 110.0, 10.0),   # W
            (100.0, 90.0, -10.0),   # L
            (100.0, 90.0, -10.0),   # L
            (100.0, 90.0, -10.0),   # L
            (100.0, 110.0, 10.0),   # W
            (100.0, 110.0, 10.0),   # W
            (100.0, 110.0, 10.0),   # W
            (100.0, 90.0, -10.0),   # L
        ]
        
        for entry, exit, pnl in trades_data:
            trade = TradeRecord(
                entry_price=entry,
                exit_price=exit,
                qty=1.0,
                side=1,
                realized_pnl=pnl,
                bars_held=5
            )
            metrics.add_trade(trade)
        
        result = metrics.calculate_metrics()
        
        # Máxima racha de ganancias: 3 (al final)
        # Máxima racha de pérdidas: 3 (en el medio)
        assert result["max_consecutive_wins"] == 3
        assert result["max_consecutive_losses"] == 3
    
    def test_profit_factor_calculation(self):
        """Test cálculo de profit factor"""
        metrics = TradeMetrics()
        
        # Gross profit: 100 USDT (2 trades de +50 cada uno)
        for i in range(2):
            trade = TradeRecord(
                entry_price=100.0,
                exit_price=150.0,
                qty=1.0,
                side=1,
                realized_pnl=50.0,
                bars_held=5
            )
            metrics.add_trade(trade)
        
        # Gross loss: 50 USDT (2 trades de -25 cada uno)
        for i in range(2):
            trade = TradeRecord(
                entry_price=100.0,
                exit_price=75.0,
                qty=1.0,
                side=1,
                realized_pnl=-25.0,
                bars_held=3
            )
            metrics.add_trade(trade)
        
        result = metrics.calculate_metrics()
        
        # Profit factor = abs(100 / 50) = 2.0
        assert result["gross_profit"] == 100.0
        assert result["gross_loss"] == 50.0
        assert result["profit_factor"] == 2.0
    
    def test_profit_factor_only_wins(self):
        """Test profit factor cuando solo hay ganancias"""
        metrics = TradeMetrics()
        
        # Solo trades ganadores
        for i in range(3):
            trade = TradeRecord(
                entry_price=100.0,
                exit_price=110.0,
                qty=1.0,
                side=1,
                realized_pnl=10.0,
                bars_held=5
            )
            metrics.add_trade(trade)
        
        result = metrics.calculate_metrics()
        
        assert result["gross_profit"] == 30.0
        assert result["gross_loss"] == 0.0
        assert result["profit_factor"] == float('inf')
    
    def test_avg_holding_bars(self):
        """Test cálculo de duración promedio de trades"""
        metrics = TradeMetrics()
        
        # Trades con diferentes duraciones
        holding_times = [5, 10, 15, 20]
        for holding in holding_times:
            trade = TradeRecord(
                entry_price=100.0,
                exit_price=110.0,
                qty=1.0,
                side=1,
                realized_pnl=10.0,
                bars_held=holding
            )
            metrics.add_trade(trade)
        
        result = metrics.calculate_metrics()
        
        # Promedio: (5 + 10 + 15 + 20) / 4 = 12.5
        assert result["avg_holding_bars"] == 12.5

class TestRunLoggerIntegration:
    """Tests para la integración con RunLogger"""
    
    def test_run_logger_with_trade_metrics(self):
        """Test que RunLogger incluya métricas profesionales en el run final"""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Crear RunLogger
            logger = RunLogger("TEST", models_root=temp_dir)
            
            # Iniciar run
            logger.start(
                market="futures",
                initial_balance=1000.0,
                target_balance=10000.0,
                initial_equity=1000.0,
                ts_start=1000000
            )
            
            # Simular algunos trades
            logger.add_trade_record(
                entry_price=100.0,
                exit_price=110.0,
                qty=1.0,
                side=1,
                realized_pnl=10.0,
                bars_held=5,
                open_ts=1000000,
                close_ts=1001000
            )
            
            logger.add_trade_record(
                entry_price=100.0,
                exit_price=95.0,
                qty=1.0,
                side=1,
                realized_pnl=-5.0,
                bars_held=3,
                open_ts=1002000,
                close_ts=1003000
            )
            
            # Actualizar contador de trades para que no se considere episodio vacío
            logger.update_trades_count(2)
            logger.update_elapsed_steps(100)
            
            # Finalizar run
            logger.finish(
                final_balance=1005.0,
                final_equity=1005.0,
                ts_end=1004000
            )
            
            # Verificar que el archivo se creó
            runs_file = Path(temp_dir) / "TEST" / "TEST_runs.jsonl"
            assert runs_file.exists()
            
            # Leer el run guardado
            with runs_file.open("r") as f:
                run_data = json.loads(f.readline())
            
            # Verificar métricas profesionales
            assert run_data["trades_count"] == 2
            assert run_data["win_rate_trades"] == 50.0  # 1 de 2 trades ganador
            assert run_data["avg_trade_pnl"] == 2.5  # (10 + (-5)) / 2
            assert run_data["avg_holding_bars"] == 4.0  # (5 + 3) / 2
            assert run_data["max_consecutive_wins"] == 1
            assert run_data["max_consecutive_losses"] == 1
            assert run_data["gross_profit"] == 10.0
            assert run_data["gross_loss"] == 5.0
            assert run_data["profit_factor"] == 2.0  # 10 / 5
            assert run_data["winning_trades"] == 1
            assert run_data["losing_trades"] == 1
            assert run_data["total_pnl"] == 5.0

if __name__ == "__main__":
    pytest.main([__file__])
