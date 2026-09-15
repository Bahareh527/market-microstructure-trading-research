"""A small, transparent, cost-aware research backtest."""

from __future__ import annotations

import numpy as np
import pandas as pd


def run_backtest(
    mid_price: pd.Series,
    signal: pd.Series | np.ndarray,
    transaction_cost_bps: float = 1.0,
) -> tuple[pd.DataFrame, dict[str, float]]:
    """Apply each signal from the next snapshot onward to avoid lookahead."""

    if transaction_cost_bps < 0:
        raise ValueError("transaction_cost_bps cannot be negative")
    price = pd.Series(mid_price, copy=True).astype(float).reset_index(drop=True)
    raw_signal = pd.Series(signal, copy=True).astype(float).reset_index(drop=True)
    if len(price) != len(raw_signal):
        raise ValueError("mid_price and signal must have equal length")
    if (price <= 0).any():
        raise ValueError("mid prices must be positive")

    position = raw_signal.clip(-1.0, 1.0).shift(1).fillna(0.0)
    market_return = price.pct_change().fillna(0.0)
    turnover = position.diff().abs().fillna(position.abs())
    gross_return = position * market_return
    cost = turnover * transaction_cost_bps / 10_000.0
    net_return = gross_return - cost
    equity_curve = (1.0 + net_return).cumprod()
    drawdown = equity_curve / equity_curve.cummax() - 1.0

    result = pd.DataFrame(
        {
            "mid_price": price,
            "signal": raw_signal,
            "position": position,
            "market_return": market_return,
            "turnover": turnover,
            "gross_return": gross_return,
            "cost": cost,
            "net_return": net_return,
            "equity_curve": equity_curve,
            "drawdown": drawdown,
        }
    )
    active = position.ne(0.0)
    metrics = {
        "total_return": float(equity_curve.iloc[-1] - 1.0),
        "gross_total_return": float((1.0 + gross_return).prod() - 1.0),
        "max_drawdown": float(drawdown.min()),
        "turnover": float(turnover.sum()),
        "hit_rate_active": float((net_return[active] > 0).mean()) if active.any() else 0.0,
        "transaction_cost_bps": float(transaction_cost_bps),
    }
    return result, metrics
