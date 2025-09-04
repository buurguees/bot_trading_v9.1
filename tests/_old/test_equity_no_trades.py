from __future__ import annotations

import sys
import os
# Asegura import de paquete del proyecto al ejecutar pytest desde cualquier cwd
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import os
import json
from pathlib import Path

import pytest

from base_env.base_env import BaseTradingEnv
from base_env.config.models import EnvConfig, SymbolMeta, PipelineConfig, HierarchicalConfig, RiskConfig, FeesConfig


class _FakeBroker:
    """Broker mínimo para tests: genera barras constantes por TF.

    Invariante para estos tests: el precio no cambia y no hay trades →
    equity debe permanecer igual al balance.
    """

    def __init__(self, tfs: list[str], base_tf: str = "1m", start_ts: int = 1_700_000_000_000):
        self.tfs = tfs
        self.base_tf = base_tf
        self._ts = int(start_ts)

    def now_ts(self) -> int:
        return self._ts

    def get_price(self) -> float:
        return 100.0  # precio constante

    def _bar(self, tf: str) -> dict:
        p = self.get_price()
        return {"ts": self._ts, "open": p, "high": p, "low": p, "close": p, "volume": 0.0}

    def get_bar(self, tf: str) -> dict:
        return self._bar(tf)

    def aligned_view(self, required_tfs: list[str]) -> dict:
        return {tf: self._bar(tf) for tf in required_tfs}

    def next(self) -> None:
        # avanzar 1m
        self._ts += 60_000

    def reset_to_start(self) -> None:
        pass


def _make_env(tmp_models: Path, market: str) -> BaseTradingEnv:
    tfs = ["1m", "5m"]
    broker = _FakeBroker(tfs=tfs, base_tf="1m")
    cfg = EnvConfig(
        mode="train",
        market=market,  # "spot" | "futures"
        leverage=(1.0 if market == "spot" else 2.0),
        symbol_meta=SymbolMeta(symbol="TEST", market=market, enabled_tfs=tfs, filters={"minNotional": 1.0, "lotStep": 0.0001}),
        tfs=tfs,
        pipeline=PipelineConfig(strict_alignment=True),
        hierarchical=HierarchicalConfig(min_confidence=1.0, execute_tfs=["1m"], confirm_tfs=["5m"]),
        risk=RiskConfig(),
        fees=FeesConfig(),
    )
    env = BaseTradingEnv(cfg=cfg, broker=broker, oms=_MockOMS(), initial_cash=1000.0, target_cash=10_000.0, models_root=str(tmp_models))
    return env


class _MockOMS:
    def open(self, side, qty, price_hint, sl, tp):
        return {"side": 1 if side == "LONG" else -1, "qty": float(qty), "price": float(price_hint), "fees": 0.0, "sl": sl, "tp": tp}

    def close(self, qty, price_hint):
        return {"qty": float(qty), "price": float(price_hint), "fees": 0.0}


@pytest.mark.parametrize("market", ["spot"])  # caso 1
def test_spot_no_pos_no_move(tmp_path: Path, market: str):
    env = _make_env(tmp_path / "models", market)
    obs = env.reset()
    eq0 = env.portfolio.equity_quote
    bal0 = env.portfolio.cash_quote
    total_r = 0.0

    for _ in range(100):
        # 2 = block_open (ver set_action_override docstring)
        env.set_action_override(2)
        obs, r, done, info = env.step()
        total_r += float(r)
        assert int(info.get("equity_drift_without_position", 0)) == 0
        assert env.pos.side == 0 and (env.pos.qty or 0.0) == 0.0
        assert abs(env.portfolio.equity_quote - env.portfolio.cash_quote) < 1e-6

    assert abs(env.portfolio.equity_quote - eq0) < 1e-6
    assert abs(env.portfolio.cash_quote - bal0) < 1e-6
    assert abs(total_r) < 1e-6


@pytest.mark.parametrize("market", ["futures"])  # caso 2
def test_futures_no_pos_no_move(tmp_path: Path, market: str):
    env = _make_env(tmp_path / "models", market)
    obs = env.reset()
    eq0 = env.portfolio.equity_quote
    bal0 = env.portfolio.cash_quote

    for _ in range(100):
        env.set_action_override(2)
        obs, r, done, info = env.step()
        assert int(info.get("equity_drift_without_position", 0)) == 0
        assert env.pos.side == 0 and (env.pos.qty or 0.0) == 0.0
        assert abs(env.portfolio.equity_quote - env.portfolio.cash_quote) < 1e-6
        assert abs(env.portfolio.used_margin) < 1e-12

    assert abs(env.portfolio.equity_quote - eq0) < 1e-6
    assert abs(env.portfolio.cash_quote - bal0) < 1e-6


def test_no_log_empty_run(tmp_path: Path):
    models_dir = tmp_path / "models"
    env = _make_env(models_dir, "spot")
    _ = env.reset()

    # 50 pasos sin trades
    for _ in range(50):
        env.set_action_override(2)
        _ = env.step()

    # simular cierre de run manual (no debería escribirse por ser vacío y corto)
    runs_file = models_dir / env.cfg.symbol_meta.symbol / f"{env.cfg.symbol_meta.symbol}_runs.jsonl"
    if runs_file.exists():
        os.remove(runs_file)

    env._run_logger.finish(
        final_balance=float(env.portfolio.cash_quote),
        final_equity=float(env.portfolio.equity_quote),
        ts_end=int(env.broker.now_ts()),
    )

    # No debe existir el archivo o debe estar vacío
    if runs_file.exists():
        with runs_file.open("r", encoding="utf-8") as f:
            lines = [ln for ln in f.readlines() if ln.strip()]
        assert len(lines) == 0
    else:
        assert True


