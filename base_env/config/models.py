
"""
base_env/config/models.py
Descripción: Modelos/contratos para cargar y validar la configuración YAML de config/.
Origen esperado: config/settings.yaml, symbols.yaml, risk.yaml, fees.yaml, pipeline.yaml, hierarchical.yaml, oms.yaml
NOTA: Aquí definimos tipos y defaults. La carga real de YAML puedes hacerla en tu bootstraper.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Dict, Literal, Optional, Any

TF = Literal["1m", "5m", "15m", "1h", "4h", "1d"]
MarketType = Literal["spot", "futures"]


@dataclass
class PipelineConfig:
    indicators: Dict[str, Any] = field(default_factory=dict)
    smc: Dict[str, Any] = field(default_factory=dict)
    strict_alignment: bool = True


@dataclass
class HierarchicalConfig:
    direction_tfs: List[TF] = field(default_factory=lambda: ["1d", "4h"])
    confirm_tfs: List[TF] = field(default_factory=lambda: ["1h", "15m"])
    execute_tfs: List[TF] = field(default_factory=lambda: ["5m", "1m"])
    min_confidence: float = 0.0
    allow_fallback_open: bool = True
    allow_fallback_close: bool = True
    dedup_open_window_bars: int = 3
    dedup_close_window_bars: int = 1


@dataclass
class RiskCommon:
    daily_max_drawdown_pct: float = 5.0
    exposure_max_abs: float = 1.0
    circuit_breakers: Dict[str, bool] = field(default_factory=lambda: {"data_quality_pause": True, "inconsistent_signals_pause": True})


@dataclass
class RiskSpot:
    risk_pct_per_trade: float = 0.5
    trailing: Dict[str, Any] = field(default_factory=lambda: {"enabled": True, "atr_multiple": 1.0})


@dataclass
class RiskFutures:
    max_initial_leverage: int = 3
    risk_pct_per_trade: float = 0.5
    margin_buffer_pct: float = 10.0
    trailing: Dict[str, Any] = field(default_factory=lambda: {"enabled": True, "atr_multiple": 1.0})


@dataclass
class RiskConfig:
    common: RiskCommon = field(default_factory=RiskCommon)
    spot: RiskSpot = field(default_factory=RiskSpot)
    futures: RiskFutures = field(default_factory=RiskFutures)


@dataclass
class FeesConfig:
    spot: Dict[str, float] = field(default_factory=lambda: {"taker_fee_bps": 10.0, "maker_fee_bps": 8.0})
    futures: Dict[str, Any] = field(default_factory=lambda: {"taker_fee_bps": 5.0, "maker_fee_bps": 2.0, "funding": {"simulate_in_backtest": False, "schedule_hours": 8}})


@dataclass
class SymbolMeta:
    symbol: str = "BTCUSDT"
    market: MarketType = "spot"
    enabled_tfs: List[TF] = field(default_factory=lambda: ["1m", "5m", "15m", "1h", "4h", "1d"])
    filters: Dict[str, float] = field(default_factory=lambda: {"tickSize": 0.1, "lotStep": 0.0001, "minNotional": 5.0})
    futures_meta: Dict[str, float] = field(default_factory=dict)


@dataclass
class EnvConfig:
    mode: Literal["train", "backtest", "live"] = "train"
    market: MarketType = "spot"
    leverage: float = 1.0    # si futures, rango 2.0–25.0
    symbol_meta: SymbolMeta = field(default_factory=SymbolMeta)
    tfs: List[TF] = field(default_factory=lambda: ["1m", "5m", "15m", "1h", "4h", "1d"])

    pipeline: PipelineConfig = field(default_factory=PipelineConfig)
    hierarchical: HierarchicalConfig = field(default_factory=HierarchicalConfig)
    risk: RiskConfig = field(default_factory=RiskConfig)
    fees: FeesConfig = field(default_factory=FeesConfig)
