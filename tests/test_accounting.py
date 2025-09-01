# tests/test_accounting.py
from base_env.accounting.ledger import Accounting, PortfolioState, PositionState

def test_open_close_spot_updates_equity():
    fees_cfg = {"spot":{"taker_fee_bps":10.0},"futures":{"taker_fee_bps":2.0}}
    acc = Accounting(fees_cfg, market="spot")
    port = PortfolioState(market="spot")
    port.reset()
    pos = PositionState()

    # open long 0.1 @ 100
    fill_open = {"side": 1, "qty": 0.1, "price": 100.0}
    acc.apply_open(fill_open, port, pos, cfg=None)
    # equity_quote decreased, equity_base increased
    assert port.equity_quote < 10_000 and port.equity_base > 0.0

    # close @ 110
    fill_close = {"qty": 0.1, "price": 110.0}
    realized = acc.apply_close(fill_close, port, pos, cfg=None)
    assert realized > 0.0
    assert pos.qty == 0.0
