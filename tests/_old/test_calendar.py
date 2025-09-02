# tests/test_calendar.py
from base_env.tfs.calendar import tf_to_ms, floor_ts_to_tf, prev_closed_bar_end, is_close_boundary

def test_tf_to_ms():
    assert tf_to_ms("1m") == 60_000
    assert tf_to_ms("5m") == 300_000
    assert tf_to_ms("1h") == 3_600_000
    assert tf_to_ms("1d") == 86_400_000

def test_floor_and_prev_close():
    ts = 1_000_000  # ms
    assert floor_ts_to_tf(ts, "1m") % 60_000 == 0
    # prev close: si no está justo en cierre, devuelve inicio de la barra actual
    pc = prev_closed_bar_end(ts, "5m")
    assert pc % (5*60_000) == 0

def test_close_boundary():
    assert is_close_boundary(300_000, "5m") is True
    assert is_close_boundary(301_000, "5m") is False
