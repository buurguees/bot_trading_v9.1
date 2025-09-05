"""
Loader centralizado para toda la configuración YAML.
"""
from __future__ import annotations
import yaml
from pathlib import Path
from typing import Dict, Any, Optional
from .models import EnvConfig, SymbolMeta, RiskConfig, FeesConfig, PipelineConfig, HierarchicalConfig, LeverageConfig


class ConfigLoader:
    """Loader centralizado para toda la configuración del bot."""
    
    def __init__(self, config_root: str = "config"):
        self.config_root = Path(config_root)
        self._cache: Dict[str, Any] = {}
    
    def load_yaml(self, filename: str) -> Dict[str, Any]:
        """Carga un archivo YAML con cache."""
        if filename in self._cache:
            return self._cache[filename]
        
        filepath = self.config_root / filename
        if not filepath.exists():
            raise FileNotFoundError(f"Archivo de configuración no encontrado: {filepath}")
        
        with open(filepath, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f)
        
        self._cache[filename] = data
        return data
    
    def load_symbols(self) -> list[SymbolMeta]:
        """Carga la configuración de símbolos."""
        data = self.load_yaml("symbols.yaml")
        symbols = []
        
        for sym_data in data.get("symbols", []):
            # Cargar leverage si existe
            leverage = None
            if "leverage" in sym_data:
                lev_data = sym_data["leverage"]
                leverage = LeverageConfig(
                    min=lev_data.get("min", 1.0),
                    max=lev_data.get("max", 25.0),
                    step=lev_data.get("step", 1.0),
                    default=lev_data.get("default", 2.0)
                )
            
            symbol = SymbolMeta(
                symbol=sym_data["symbol"],
                market="futures" if sym_data.get("mode", "").endswith("futures") else "spot",
                enabled_tfs=sym_data.get("enabled_tfs", ["1m", "5m", "15m", "1h"]),
                filters=sym_data.get("filters", {}),
                allow_shorts=sym_data.get("allow_shorts", True),
                leverage=leverage
            )
            symbols.append(symbol)
        
        return symbols
    
    def load_risk_config(self) -> RiskConfig:
        """Carga la configuración de riesgo."""
        data = self.load_yaml("risk.yaml")
        
        # Cargar bankruptcy config
        bankruptcy_data = data.get("common", {}).get("bankruptcy", {})
        from .models import BankruptcyConfig, SoftResetConfig
        
        soft_reset_data = bankruptcy_data.get("soft_reset", {})
        soft_reset = SoftResetConfig(
            max_resets_per_run=soft_reset_data.get("max_resets_per_run", 2),
            post_reset_leverage_cap=soft_reset_data.get("post_reset_leverage_cap", 2.0),
            cooldown_bars=soft_reset_data.get("cooldown_bars", 50),
            label_segment=soft_reset_data.get("label_segment", True)
        )
        
        bankruptcy = BankruptcyConfig(
            enabled=bankruptcy_data.get("enabled", False),  # Cambiado de True a False
            threshold_pct=bankruptcy_data.get("threshold_pct", 20.0),
            penalty_reward=bankruptcy_data.get("penalty_reward", -10.0),
            mode=bankruptcy_data.get("mode", "end"),
            restart_on_bankruptcy=bankruptcy_data.get("restart_on_bankruptcy", True),
            soft_reset=soft_reset
        )
        
        # Cargar default levels
        from .models import DefaultLevelsConfig
        default_levels_data = data.get("common", {}).get("default_levels", {})
        default_levels = DefaultLevelsConfig(
            use_atr=default_levels_data.get("use_atr", True),
            atr_period=default_levels_data.get("atr_period", 14),
            sl_atr_mult=default_levels_data.get("sl_atr_mult", 1.0),
            min_sl_pct=default_levels_data.get("min_sl_pct", 1.0),
            tp_r_multiple=default_levels_data.get("tp_r_multiple", 1.5)
        )
        
        # Cargar risk common
        from .models import RiskCommon
        common_data = data.get("common", {})
        risk_common = RiskCommon(
            daily_max_drawdown_pct=common_data.get("daily_max_drawdown_pct", 5.0),
            exposure_max_abs=common_data.get("exposure_max_abs", 1.0),
            circuit_breakers=common_data.get("circuit_breakers", {}),
            bankruptcy=bankruptcy,
            default_levels=default_levels,
            train_force_min_notional=common_data.get("train_force_min_notional", True)
        )
        
        # Cargar risk spot
        from .models import RiskSpot
        spot_data = data.get("spot", {})
        risk_spot = RiskSpot(
            risk_pct_per_trade=spot_data.get("risk_pct_per_trade", 2.0),
            trailing=spot_data.get("trailing", {"enabled": True, "atr_multiple": 1.0})
        )
        
        # Cargar risk futures
        from .models import RiskFutures
        futures_data = data.get("futures", {})
        risk_futures = RiskFutures(
            max_initial_leverage=futures_data.get("max_initial_leverage", 3),
            risk_pct_per_trade=futures_data.get("risk_pct_per_trade", 2.0),
            margin_buffer_pct=futures_data.get("margin_buffer_pct", 10.0),
            trailing=futures_data.get("trailing", {"enabled": True, "atr_multiple": 1.0})
        )
        
        return RiskConfig(
            common=risk_common,
            spot=risk_spot,
            futures=risk_futures
        )
    
    def load_fees_config(self) -> FeesConfig:
        """Carga la configuración de fees."""
        data = self.load_yaml("fees.yaml")
        
        return FeesConfig(
            spot=data.get("spot", {"taker_fee_bps": 10.0, "maker_fee_bps": 8.0}),
            futures=data.get("futures", {"taker_fee_bps": 5.0, "maker_fee_bps": 2.0, "funding": {"simulate_in_backtest": False, "schedule_hours": 8}})
        )
    
    def load_hierarchical_config(self) -> HierarchicalConfig:
        """Carga la configuración jerárquica."""
        data = self.load_yaml("hierarchical.yaml")
        
        # Extraer configuración de gating
        gating = data.get("gating", {})
        dedup = gating.get("dedup", {})
        
        return HierarchicalConfig(
            direction_tfs=data.get("layers", {}).get("direction_tfs", ["1d", "4h"]),
            confirm_tfs=data.get("layers", {}).get("confirm_tfs", ["1h", "15m"]),
            execute_tfs=data.get("layers", {}).get("execute_tfs", ["5m", "1m"]),
            min_confidence=gating.get("min_confidence", 0.0),
            allow_fallback_open=gating.get("allow_fallback_open", True),
            allow_fallback_close=gating.get("allow_fallback_close", True),
            dedup_open_window_bars=dedup.get("open_window_bars", 3),
            dedup_close_window_bars=dedup.get("close_window_bars", 1)
        )
    
    def load_pipeline_config(self) -> PipelineConfig:
        """Carga la configuración del pipeline."""
        data = self.load_yaml("pipeline.yaml")
        
        return PipelineConfig(
            indicators=data.get("indicators", {}),
            smc=data.get("smc", {}),
            strict_alignment=data.get("strict_alignment", True)
        )
    
    def load_train_config(self) -> Dict[str, Any]:
        """Carga la configuración de entrenamiento."""
        return self.load_yaml("train.yaml")
    
    def load_rewards_config(self) -> Dict[str, Any]:
        """Carga la configuración de rewards."""
        return self.load_yaml("rewards.yaml")
    
    def create_env_config(self, symbol: str, mode: str) -> EnvConfig:
        """Crea una configuración completa del entorno para un símbolo."""
        symbols = self.load_symbols()
        symbol_meta = next((s for s in symbols if s.symbol == symbol), None)
        if not symbol_meta:
            raise ValueError(f"Símbolo {symbol} no encontrado en symbols.yaml")
        
        # Determinar market y leverage
        market = "futures" if mode.endswith("futures") else "spot"
        leverage = 1.0
        if market == "futures" and symbol_meta.leverage:
            leverage = symbol_meta.leverage.default
        
        return EnvConfig(
            mode=mode,
            market=market,
            leverage=leverage,
            symbol_meta=symbol_meta,
            tfs=symbol_meta.enabled_tfs,
            pipeline=self.load_pipeline_config(),
            hierarchical=self.load_hierarchical_config(),
            risk=self.load_risk_config(),
            fees=self.load_fees_config()
        )
    
    def print_config_summary(self):
        """Imprime un resumen de la configuración cargada."""
        print("📋 RESUMEN DE CONFIGURACIÓN:")
        print("=" * 50)
        
        # Símbolos
        symbols = self.load_symbols()
        print(f"🔤 Símbolos configurados: {len(symbols)}")
        for sym in symbols:
            mode = "futures" if sym.market == "futures" else "spot"
            leverage_info = f" (leverage: {sym.leverage.min}-{sym.leverage.max}x)" if sym.leverage else ""
            print(f"   - {sym.symbol}: {mode}{leverage_info}")
        
        # Risk
        risk = self.load_risk_config()
        print(f"⚠️  Risk configurado:")
        print(f"   - Bankruptcy: {risk.common.bankruptcy.mode} (threshold: {risk.common.bankruptcy.threshold_pct}%)")
        print(f"   - Spot risk: {risk.spot.risk_pct_per_trade}% por trade")
        print(f"   - Futures risk: {risk.futures.risk_pct_per_trade}% por trade")
        
        # Hierarchical
        hier = self.load_hierarchical_config()
        print(f"🏗️  Análisis jerárquico:")
        print(f"   - Min confidence: {hier.min_confidence}")
        print(f"   - Trend TFs: {hier.direction_tfs}")
        print(f"   - Confirm TFs: {hier.confirm_tfs}")
        print(f"   - Execute TFs: {hier.execute_tfs}")
        
        print("=" * 50)
    
    def clear_cache(self):
        """Limpia el cache de configuración."""
        self._cache.clear()


# Instancia global del loader
config_loader = ConfigLoader()
