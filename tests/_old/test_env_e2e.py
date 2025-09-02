# tests/test_env_e2e.py
# E2E: Broker histórico → BaseTradingEnv → Policy+Risk → Accounting → eventos.

from base_env.io.historical_broker import ParquetHistoricalBroker
from base_env.base_env import BaseTradingEnv

def test_e2e_aligned(tmp_data_root, env_cfg, mock_oms):
    broker = ParquetHistoricalBroker(
        data_root=tmp_data_root, symbol="BTCUSDT", market="spot",
        tfs=["1m","5m"], base_tf="1m", stage="aligned", warmup_bars=5000
    )
    env = BaseTradingEnv(cfg=env_cfg, broker=broker, oms=mock_oms)
    obs = env.reset()
    steps = 1500  # suficiente para tener ATR y cruce EMAs varias veces
    total_reward = 0.0
    opened = 0
    closed = 0

    for i in range(steps):
        obs, reward, done, info = env.step()
        total_reward += reward
        for ev in info.get("events", []):
            if ev["kind"] == "OPEN": opened += 1
            if ev["kind"] == "CLOSE": closed += 1
        if done:
            break

    # Debe haber al menos una apertura y un cierre en 1500 pasos
    assert opened >= 1
    assert closed >= 1
    # reward puede ser + o -, pero flotante; validamos que se sumó
    assert isinstance(total_reward, float)
