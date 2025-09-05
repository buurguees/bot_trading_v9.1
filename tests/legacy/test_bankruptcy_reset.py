from __future__ import annotations

import os
import sys
from pathlib import Path
import pytest

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from base_env.base_env import BaseTradingEnv
from base_env.config.models import EnvConfig, SymbolMeta, PipelineConfig, HierarchicalConfig, RiskConfig, FeesConfig


class _DropPriceBroker:
    def __init__(self, tfs: list[str], base_tf: str = "1m", start_ts: int = 1_700_000_000_000):
        self.tfs = tfs
        self.base_tf = base_tf
        self._ts = int(start_ts)
        self._p = 100.0
        self._ticks = 0
        self._initial_ts = int(start_ts)

    def now_ts(self) -> int: return self._ts
    def get_price(self) -> float: return self._p
    def _bar(self, tf: str) -> dict:
        p = self._p
        return {"ts": self._ts, "open": p, "high": p, "low": p, "close": p, "volume": 0.0}
    def get_bar(self, tf: str): return self._bar(tf)
    def aligned_view(self, required_tfs: list[str]): return {tf: self._bar(tf) for tf in required_tfs}
    def next(self) -> None:
        self._ts += 60_000
        self._ticks += 1
        # Caída gradual y continua para disparar bancarrota
        if self._ticks > 3:  # Empezar a caer después de 3 ticks
            # Caída del 10% por tick para asegurar bankruptcy
            self._p *= 0.9
    def reset_to_start(self) -> None:
        self._ts = self._initial_ts
        self._p = 100.0
        self._ticks = 0
    def is_end_of_data(self) -> bool:
        return False  # Para tests, nunca termina


class _MockOMS:
    def open(self, side, qty, price_hint, sl, tp): return {"side": 1 if side=="LONG" else -1, "qty": float(qty), "price": float(price_hint), "fees": 0.0, "sl": sl, "tp": tp}
    def close(self, qty, price_hint): return {"qty": float(qty), "price": float(price_hint), "fees": 0.0}


def _env_with_bankruptcy(tmp_models: Path) -> BaseTradingEnv:
    tfs = ["1m","5m"]
    broker = _DropPriceBroker(tfs=tfs, base_tf="1m")
    risk = RiskConfig()
    # forzar bancarrota activa y umbral 20%
    risk.common.bankruptcy.enabled = True
    risk.common.bankruptcy.threshold_pct = 20.0
    risk.common.bankruptcy.penalty_reward = -10.0
    risk.common.bankruptcy.restart_on_bankruptcy = True
    # aumentar riesgo en futuros para abrir tamaño grande
    risk.futures.risk_pct_per_trade = 50.0

    cfg = EnvConfig(
        mode="train", market="spot", leverage=1.0,
        symbol_meta=SymbolMeta(symbol="TEST", market="spot", enabled_tfs=tfs, filters={"minNotional":1.0,"lotStep":0.0001}),
        tfs=tfs, pipeline=PipelineConfig(strict_alignment=True),
        hierarchical=HierarchicalConfig(min_confidence=0.0, execute_tfs=["1m"], confirm_tfs=["5m"]),
        risk=risk, fees=FeesConfig(),
    )
    env = BaseTradingEnv(cfg=cfg, broker=broker, oms=_MockOMS(), initial_cash=1000.0, target_cash=10_000.0, models_root=str(tmp_models))
    return env


def test_dispara_bancarrota(tmp_path: Path):
    """Test directo de bankruptcy sin depender del entorno completo."""
    from base_env.risk.manager import RiskManager
    from base_env.accounting.ledger import PortfolioState, PositionState, Accounting
    from base_env.events.bus import SimpleEventBus
    
    # Configurar bankruptcy
    risk = RiskConfig()
    risk.common.bankruptcy.enabled = True
    risk.common.bankruptcy.threshold_pct = 20.0
    risk.common.bankruptcy.penalty_reward = -10.0
    
    # Crear componentes
    risk_manager = RiskManager(risk, SymbolMeta(symbol="TEST", market="spot"))
    portfolio = PortfolioState(market="spot", cash_quote=1000.0, equity_quote=1000.0)
    position = PositionState()
    fees_cfg = FeesConfig()
    accounting = Accounting(fees_cfg.model_dump() if hasattr(fees_cfg, "model_dump") else fees_cfg.__dict__, "spot")
    events_bus = SimpleEventBus()
    
    # Abrir posición larga
    entry = 100.0
    qty = 10.0
    fill = {"side": 1, "qty": qty, "price": entry, "fees": 0.0, "sl": None, "tp": None}
    accounting.apply_open(fill, portfolio, position, None)
    
    print(f"💰 Balance inicial: $1000.0")
    print(f"📈 Posición abierta: {qty} @ ${entry} (notional: ${entry * qty})")
    print(f"💼 Equity tras apertura: ${portfolio.equity_quote}")
    
    # Simular caída de precio y verificar bankruptcy
    initial_balance = 1000.0
    bankruptcy_detected = False
    
    # Crear broker una sola vez
    broker = _DropPriceBroker([], base_tf="1m")
    
    for step in range(10):
        # Caída del 20% por step
        current_price = entry * (0.8 ** (step + 1))
        broker._p = current_price
        
        # Actualizar PnL
        accounting.update_unrealized(broker, position, portfolio)
        
        # Verificar bankruptcy
        bankruptcy_occurred = risk_manager.check_bankruptcy(
            portfolio, initial_balance, events_bus, 1700000000000 + step * 60000
        )
        
        print(f"Step {step+1}: Precio=${current_price:.2f}, Equity=${portfolio.equity_quote:.2f}")
        print(f"   - Cash: ${portfolio.cash_quote:.2f}, PnL: ${position.unrealized_pnl:.2f}, Side: {position.side}")
        
        if bankruptcy_occurred:
            bankruptcy_detected = True
            print(f"🚨 BANCARROTA DETECTADA en step {step+1}")
            print(f"   - Equity final: ${portfolio.equity_quote:.2f}")
            print(f"   - Threshold (20%): ${initial_balance * 0.2:.2f}")
            break
    
    assert bankruptcy_detected, f"Debe dispararse bancarrota. Equity final: ${portfolio.equity_quote:.2f}, Threshold: ${initial_balance * 0.2:.2f}"


def test_bankruptcy_soft_reset(tmp_path: Path):
    """Test bankruptcy en modo soft_reset: resetea balance/equity y continúa el run."""
    env = _env_with_bankruptcy(tmp_path / "models")
    
    # Configurar modo soft_reset
    env.cfg.risk.common.bankruptcy.mode = "soft_reset"
    env.cfg.risk.common.bankruptcy.soft_reset.max_resets_per_run = 2
    env.cfg.risk.common.bankruptcy.soft_reset.cooldown_bars = 5
    env.cfg.risk.common.bankruptcy.soft_reset.post_reset_leverage_cap = 2.0
    
    _ = env.reset()
    
    initial_balance = env.portfolio.cash_quote
    print(f"💰 Balance inicial: ${initial_balance}")
    
    # Abrir posición
    entry = float(env.broker.get_price())
    qty = 10.0
    fill = {"side": 1, "qty": qty, "price": entry, "fees": 0.0, "sl": None, "tp": None}
    env.accounting.apply_open(fill, env.portfolio, env.pos, env.cfg)
    
    # Avanzar hasta bankruptcy
    soft_reset_detected = False
    for step in range(20):
        obs, r, done, info = env.step()
        
        if info.get("soft_reset", False):
            soft_reset_detected = True
            print(f"🔄 SOFT RESET DETECTADO en step {step+1}")
            print(f"   - Reset count: {info.get('reset_count', 0)}")
            print(f"   - Segment ID: {info.get('segment_id', 0)}")
            print(f"   - Cooldown: {info.get('cooldown_remaining', 0)} barras")
            print(f"   - Leverage cap: {info.get('leverage_cap', None)}x")
            
            # Verificar que NO termina el episodio
            assert not done, "Soft reset NO debe terminar el episodio"
            
            # Verificar que el balance se resetea
            assert abs(env.portfolio.cash_quote - initial_balance) < 0.01, f"Balance debe resetearse a ${initial_balance}"
            assert abs(env.portfolio.equity_quote - initial_balance) < 0.01, f"Equity debe resetearse a ${initial_balance}"
            assert env.portfolio.used_margin == 0.0, "Used margin debe ser 0 tras reset"
            
            # Verificar cooldown activo
            assert env._cooldown_bars_remaining > 0, "Cooldown debe estar activo"
            assert env._leverage_cap_active == 2.0, "Leverage cap debe estar activo"
            
            break
    
    assert soft_reset_detected, "Debe detectarse soft reset"


def test_soft_reset_exhausted_fallback_to_end(tmp_path: Path):
    """Test que cuando se agotan los soft resets, fallback a modo 'end'."""
    env = _env_with_bankruptcy(tmp_path / "models")
    
    # Configurar modo soft_reset con solo 1 reset permitido
    env.cfg.risk.common.bankruptcy.mode = "soft_reset"
    env.cfg.risk.common.bankruptcy.soft_reset.max_resets_per_run = 1
    
    _ = env.reset()
    
    # Primera bankruptcy (soft reset)
    entry = float(env.broker.get_price())
    qty = 10.0
    fill = {"side": 1, "qty": qty, "price": entry, "fees": 0.0, "sl": None, "tp": None}
    env.accounting.apply_open(fill, env.portfolio, env.pos, env.cfg)
    
    first_soft_reset = False
    for step in range(20):
        obs, r, done, info = env.step()
        if info.get("soft_reset", False):
            first_soft_reset = True
            print(f"🔄 Primer soft reset en step {step+1}")
            break
    
    assert first_soft_reset, "Debe ocurrir primer soft reset"
    
    # Segunda bankruptcy (debe fallback a 'end')
    entry = float(env.broker.get_price())
    qty = 10.0
    fill = {"side": 1, "qty": qty, "price": entry, "fees": 0.0, "sl": None, "tp": None}
    env.accounting.apply_open(fill, env.portfolio, env.pos, env.cfg)
    
    fallback_to_end = False
    for step in range(20):
        obs, r, done, info = env.step()
        if info.get("bankruptcy", False) and not info.get("soft_reset", False):
            fallback_to_end = True
            print(f"🚨 Fallback a modo 'end' en step {step+1}")
            assert done, "Debe terminar el episodio en fallback a 'end'"
            break
    
    assert fallback_to_end, "Debe hacer fallback a modo 'end' cuando se agotan los soft resets"



