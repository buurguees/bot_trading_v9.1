# tests/test_indicators.py
from base_env.features.indicators import ema, rsi, atr, macd, bollinger

def test_ema_basic():
    vals = [1,2,3,4,5,6]
    v = ema(vals, 3)
    assert v is not None and v > 0

def test_rsi_bounds():
    vals = [i for i in range(1,40)]
    r = rsi(vals, 14)
    assert r is not None and 0 <= r <= 100

def test_atr_sane():
    highs = [10,11,12,13,14,15,16,17,18,19,20,21,22,23,24]
    lows  = [ 9, 9,10,11,12,13,14,15,16,17,18,19,20,21,22]
    closes= [ 9.5,10,11,12,13,14,15,16,17,18,19,20,21,22,23]
    a = atr(highs, lows, closes, 14)
    assert a is not None and a > 0

def test_macd_and_bb():
    closes = [i + (0.1 if i%3==0 else -0.1) for i in range(1,60)]
    m = macd(closes, 12, 26, 9)
    bb = bollinger(closes, 20, 2.0)
    assert m and "macd" in m and "signal" in m and "hist" in m
    assert bb and "upper" in bb and "lower" in bb and bb["upper"] > bb["lower"]
