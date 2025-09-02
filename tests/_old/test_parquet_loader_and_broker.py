# tests/test_parquet_loader_and_broker.py
from base_env.io.parquet_loader import load_window, load_latest_n
from base_env.io.historical_broker import ParquetHistoricalBroker

def test_parquet_loader_window(tmp_data_root):
    out = load_window(tmp_data_root, "BTCUSDT", "spot", "1m", stage="aligned")
    assert isinstance(out, dict) and len(out) > 0
    # claves ts ordenables
    ts_list = sorted(out.keys())
    assert ts_list[0] < ts_list[-1]
    sample = out[ts_list[10]]
    for k in ["open","high","low","close","volume"]:
        assert k in sample

def test_historical_broker_aligns(tmp_data_root):
    broker = ParquetHistoricalBroker(
        data_root=tmp_data_root, symbol="BTCUSDT", market="spot",
        tfs=["1m","5m"], base_tf="1m", stage="aligned", warmup_bars=1000
    )
    # base now_ts está en 1m timeline
    t1 = broker.now_ts()
    b1m = broker.get_bar("1m")
    b5m = broker.get_bar("5m")
    assert b1m and b5m
    assert b1m["ts"] <= t1 and b5m["ts"] <= t1
    broker.next()
    assert broker.now_ts() > t1
