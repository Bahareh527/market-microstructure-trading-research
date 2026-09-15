"""Forward labels kept separate from causal feature engineering."""

from __future__ import annotations

import numpy as np
import pandas as pd


def add_future_move_label(
    frame: pd.DataFrame,
    horizon: int = 10,
    threshold_bps: float = 0.5,
    price_column: str = "mid_price",
) -> pd.DataFrame:
    """Label future mid-price movement as down (-1), flat (0), or up (1)."""

    if horizon < 1:
        raise ValueError("horizon must be positive")
    if threshold_bps < 0:
        raise ValueError("threshold_bps cannot be negative")
    if price_column not in frame:
        raise ValueError(f"missing price column: {price_column}")

    result = frame.copy()
    future_price = result[price_column].shift(-horizon)
    future_return_bps = (future_price / result[price_column] - 1.0) * 10_000.0
    valid = future_price.notna()
    label = pd.Series(pd.NA, index=result.index, dtype="Int64")
    label.loc[valid] = np.select(
        [
            future_return_bps.loc[valid] > threshold_bps,
            future_return_bps.loc[valid] < -threshold_bps,
        ],
        [1, -1],
        default=0,
    )
    result["future_return_bps"] = future_return_bps
    result["target"] = label
    return result
