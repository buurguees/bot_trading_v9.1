# base_env/metrics/trade_metrics.py
# Descripción: Recolector de métricas profesionales de trades para backtesting

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
import statistics
from ..utils.timestamp_utils import timestamp_to_utc_string

@dataclass
class TradeRecord:
    """Registro individual de un trade cerrado"""
    entry_price: float
    exit_price: float
    qty: float
    side: int  # 1 = LONG, -1 = SHORT
    realized_pnl: float
    bars_held: int
    leverage_used: float = 1.0  # ← NUEVO: Leverage usado en el trade
    open_ts: Optional[int] = None
    close_ts: Optional[int] = None
    sl: Optional[float] = None
    tp: Optional[float] = None
    roi_pct: float = 0.0
    r_multiple: float = 0.0
    risk_pct: float = 0.0
    
    @property
    def is_winner(self) -> bool:
        """True si el trade fue ganador"""
        return self.realized_pnl > 0
    
    @property
    def is_loser(self) -> bool:
        """True si el trade fue perdedor"""
        return self.realized_pnl < 0
    
    @property
    def open_ts_utc(self) -> Optional[str]:
        """Timestamp de apertura en formato UTC legible"""
        return timestamp_to_utc_string(self.open_ts)
    
    @property
    def close_ts_utc(self) -> Optional[str]:
        """Timestamp de cierre en formato UTC legible"""
        return timestamp_to_utc_string(self.close_ts)

@dataclass
class TradeMetrics:
    """Métricas profesionales calculadas a partir de una lista de trades"""
    trades: List[TradeRecord] = field(default_factory=list)
    
    def add_trade(self, trade: TradeRecord):
        """Añade un trade al registro"""
        self.trades.append(trade)
    
    def calculate_metrics(self) -> Dict[str, Any]:
        """Calcula todas las métricas profesionales"""
        if not self.trades:
            return self._empty_metrics()
        
        # Métricas básicas
        trades_count = len(self.trades)
        winning_trades = [t for t in self.trades if t.is_winner]
        losing_trades = [t for t in self.trades if t.is_loser]
        
        # Win rate
        win_rate_trades = (len(winning_trades) / trades_count) * 100.0 if trades_count > 0 else 0.0
        
        # PnL promedio por trade
        total_pnl = sum(t.realized_pnl for t in self.trades)
        avg_trade_pnl = total_pnl / trades_count if trades_count > 0 else 0.0
        
        # Duración promedio de trades
        avg_holding_bars = statistics.mean([t.bars_held for t in self.trades]) if self.trades else 0.0
        
        # Rachas de ganancias y pérdidas
        max_consecutive_wins, max_consecutive_losses = self._calculate_consecutive_streaks()
        
        # Gross profit y gross loss
        gross_profit = sum(t.realized_pnl for t in winning_trades) if winning_trades else 0.0
        gross_loss = abs(sum(t.realized_pnl for t in losing_trades)) if losing_trades else 0.0
        
        # Profit factor
        profit_factor = None
        if gross_loss != 0:
            profit_factor = abs(gross_profit / gross_loss)
        elif gross_profit > 0:
            profit_factor = float('inf')  # Solo ganancias
        
        # ← NUEVO: Métricas de leverage
        leverages = [t.leverage_used for t in self.trades]
        avg_leverage = statistics.mean(leverages) if leverages else 0.0
        min_leverage = min(leverages) if leverages else 0.0
        max_leverage = max(leverages) if leverages else 0.0
        high_leverage_trades = len([l for l in leverages if l > 10.0])
        high_leverage_pct = (high_leverage_trades / trades_count) * 100.0 if trades_count > 0 else 0.0

        # Obtener timestamps del primer y último trade para contexto temporal
        first_trade_ts = self.trades[0].open_ts if self.trades and self.trades[0].open_ts else None
        last_trade_ts = self.trades[-1].close_ts if self.trades and self.trades[-1].close_ts else None
        
        return {
            "trades_count": trades_count,
            "win_rate_trades": round(win_rate_trades, 2),
            "avg_trade_pnl": round(avg_trade_pnl, 2),
            "avg_holding_bars": round(avg_holding_bars, 1),
            "max_consecutive_wins": max_consecutive_wins,
            "max_consecutive_losses": max_consecutive_losses,
            "gross_profit": round(gross_profit, 2),
            "gross_loss": round(gross_loss, 2),
            "profit_factor": round(profit_factor, 2) if profit_factor is not None else None,
            "winning_trades": len(winning_trades),
            "losing_trades": len(losing_trades),
            "total_pnl": round(total_pnl, 2),
            # ← NUEVO: Métricas de leverage
            "avg_leverage": round(avg_leverage, 2),
            "min_leverage": round(min_leverage, 2),
            "max_leverage": round(max_leverage, 2),
            "high_leverage_trades": high_leverage_trades,
            "high_leverage_pct": round(high_leverage_pct, 2),
            # ← NUEVO: Timestamps UTC para contexto temporal
            "first_trade_ts": first_trade_ts,
            "last_trade_ts": last_trade_ts,
            "first_trade_ts_utc": timestamp_to_utc_string(first_trade_ts),
            "last_trade_ts_utc": timestamp_to_utc_string(last_trade_ts)
        }
    
    def _calculate_consecutive_streaks(self) -> tuple[int, int]:
        """Calcula las rachas máximas de ganancias y pérdidas consecutivas"""
        if not self.trades:
            return 0, 0
        
        max_wins = 0
        max_losses = 0
        current_wins = 0
        current_losses = 0
        
        for trade in self.trades:
            if trade.is_winner:
                current_wins += 1
                current_losses = 0
                max_wins = max(max_wins, current_wins)
            elif trade.is_loser:
                current_losses += 1
                current_wins = 0
                max_losses = max(max_losses, current_losses)
            else:  # Trade neutro (PnL = 0)
                current_wins = 0
                current_losses = 0
        
        return max_wins, max_losses
    
    def _empty_metrics(self) -> Dict[str, Any]:
        """Retorna métricas vacías cuando no hay trades"""
        return {
            "trades_count": 0,
            "win_rate_trades": 0.0,
            "avg_trade_pnl": 0.0,
            "avg_holding_bars": 0.0,
            "max_consecutive_wins": 0,
            "max_consecutive_losses": 0,
            "gross_profit": 0.0,
            "gross_loss": 0.0,
            "profit_factor": None,
            "winning_trades": 0,
            "losing_trades": 0,
            "total_pnl": 0.0,
            # ← NUEVO: Métricas de leverage vacías
            "avg_leverage": 0.0,
            "min_leverage": 0.0,
            "max_leverage": 0.0,
            "high_leverage_trades": 0,
            "high_leverage_pct": 0.0,
            # ← NUEVO: Timestamps UTC vacíos
            "first_trade_ts": None,
            "last_trade_ts": None,
            "first_trade_ts_utc": None,
            "last_trade_ts_utc": None
        }
    
    def reset(self):
        """Limpia todos los trades registrados"""
        self.trades.clear()
    
    def get_trade_summary(self) -> str:
        """Retorna un resumen legible de los trades"""
        if not self.trades:
            return "No trades executed"
        
        metrics = self.calculate_metrics()
        return (f"Trades: {metrics['trades_count']} | "
                f"Win Rate: {metrics['win_rate_trades']:.1f}% | "
                f"Avg PnL: {metrics['avg_trade_pnl']:.2f} | "
                f"Profit Factor: {metrics['profit_factor']:.2f if metrics['profit_factor'] else 'N/A'}")
