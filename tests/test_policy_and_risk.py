# tests/test_policy_and_risk.py
from base_env.policy.gating import PolicyEngine
from base_env.config.models import HierarchicalConfig, RiskConfig, SymbolMeta
from base_env.risk.manager import RiskManager

def test_policy_open_and_risk_sizing(env_cfg):
    policy = PolicyEngine(env_cfg.hierarchical, base_tf="1m")
    risk = RiskManager(env_cfg.risk, env_cfg.symbol_meta)

    # Obs con confluencia y ATR presente (ejecución 1m)
    obs = {
        "ts": 1743465600000,
        "tfs": {"1m": {"close": 100, "ts":1743465600000}, "5m": {"close": 100, "ts":1743465600000}},
        "features": {"1m": {"ema20": 101, "ema50": 99, "atr14": 1.5}},
        "analysis": {"confidence": 1.0, "side_hint": 1},
        "position": {"side": 0}
    }
    d = policy.decide(obs)
    assert d.should_open and d.side == 1 and d.sl is not None and d.tp is not None

    class P:
        market="spot"
        equity_quote=10_000.0
    class Pos:
        side=0; qty=0.0
    sized = risk.apply(P, Pos, d, obs)
    assert sized.should_open and sized.qty > 0.0

def test_policy_close_on_reverse(env_cfg):
    policy = PolicyEngine(env_cfg.hierarchical, base_tf="1m")
    # posición long abierta
    obs = {
        "ts": 1743465660000,
        "tfs": {"1m": {"close": 100, "ts":1743465660000}, "5m": {"close": 100, "ts":1743465660000}},
        "features": {"1m": {"ema20": 98, "ema50": 100, "atr14": 1.5}},
        "analysis": {"confidence": 1.0, "side_hint": -1},
        "position": {"side": 1}
    }
    d = policy.decide(obs)
    assert d.should_close_all is True
