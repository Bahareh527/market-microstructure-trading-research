import pandas as pd

from market_microstructure.backtest import run_backtest


def test_signal_is_applied_one_snapshot_later():
    price = pd.Series([100.0, 101.0, 102.0])
    signal = pd.Series([1, 1, 1])
    frame, _ = run_backtest(price, signal, transaction_cost_bps=0.0)
    assert frame.loc[0, "position"] == 0.0
    assert frame.loc[1, "position"] == 1.0


def test_costs_reduce_or_equal_return():
    price = pd.Series([100.0, 101.0, 100.0, 101.0, 100.0])
    signal = pd.Series([1, -1, 1, -1, 1])
    _, free = run_backtest(price, signal, transaction_cost_bps=0.0)
    _, costly = run_backtest(price, signal, transaction_cost_bps=2.0)
    assert costly["total_return"] <= free["total_return"]
