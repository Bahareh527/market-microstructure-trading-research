"""Feature engineering for L2 limit-order-book snapshots."""

from __future__ import annotations

import numpy as np
import pandas as pd


def validate_lob(frame: pd.DataFrame, levels: int = 5) -> None:
    required = {"timestamp", "bid_price_1", "ask_price_1", "bid_size_1", "ask_size_1"}
    for level in range(1, levels + 1):
        required.update(
            {
                f"bid_price_{level}",
                f"ask_price_{level}",
                f"bid_size_{level}",
                f"ask_size_{level}",
            }
        )
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"missing LOB columns: {missing}")
    if (frame["ask_price_1"] < frame["bid_price_1"]).any():
        raise ValueError("crossed quotes are not supported")
    for level in range(1, levels):
        if (frame[f"bid_price_{level + 1}"] > frame[f"bid_price_{level}"]).any():
            raise ValueError("bid prices must decrease away from the touch")
        if (frame[f"ask_price_{level + 1}"] < frame[f"ask_price_{level}"]).any():
            raise ValueError("ask prices must increase away from the touch")


def order_flow_imbalance(frame: pd.DataFrame) -> pd.Series:
    """Compute event-level top-of-book order-flow imbalance.

    The sign convention follows the standard event accounting identity: added bid
    liquidity and removed ask liquidity are positive; the opposite events are negative.
    """

    bid_price = frame["bid_price_1"]
    ask_price = frame["ask_price_1"]
    bid_size = frame["bid_size_1"]
    ask_size = frame["ask_size_1"]
    previous_bid_price = bid_price.shift(1)
    previous_ask_price = ask_price.shift(1)
    previous_bid_size = bid_size.shift(1)
    previous_ask_size = ask_size.shift(1)

    values = (
        (bid_price >= previous_bid_price).astype(float) * bid_size
        - (bid_price <= previous_bid_price).astype(float) * previous_bid_size
        - (ask_price <= previous_ask_price).astype(float) * ask_size
        + (ask_price >= previous_ask_price).astype(float) * previous_ask_size
    )
    return values.fillna(0.0).rename("order_flow_imbalance")


def engineer_features(
    frame: pd.DataFrame, levels: int = 5, rolling_window: int = 20
) -> pd.DataFrame:
    """Return a copy with causal, snapshot-level features."""

    validate_lob(frame, levels=levels)
    if rolling_window < 2:
        raise ValueError("rolling_window must be at least 2")

    result = frame.copy()
    result["timestamp"] = pd.to_datetime(result["timestamp"], utc=True)
    result = result.sort_values("timestamp", kind="stable").reset_index(drop=True)
    result["mid_price"] = (result["bid_price_1"] + result["ask_price_1"]) / 2.0
    result["spread"] = result["ask_price_1"] - result["bid_price_1"]
    result["relative_spread_bps"] = result["spread"] / result["mid_price"] * 10_000.0

    top_total = result["bid_size_1"] + result["ask_size_1"]
    result["top_imbalance"] = (
        result["bid_size_1"] - result["ask_size_1"]
    ) / top_total.replace(0.0, np.nan)
    result["weighted_midprice"] = (
        result["ask_price_1"] * result["bid_size_1"]
        + result["bid_price_1"] * result["ask_size_1"]
    ) / top_total.replace(0.0, np.nan)
    result["weighted_midprice_distance_bps"] = (
        (result["weighted_midprice"] - result["mid_price"])
        / result["mid_price"]
        * 10_000.0
    )

    bid_depth = sum(result[f"bid_size_{level}"] for level in range(1, levels + 1))
    ask_depth = sum(result[f"ask_size_{level}"] for level in range(1, levels + 1))
    total_depth = bid_depth + ask_depth
    result["depth_imbalance"] = (bid_depth - ask_depth) / total_depth.replace(0.0, np.nan)
    result["top_depth_concentration"] = top_total / total_depth.replace(0.0, np.nan)
    result["order_flow_imbalance"] = order_flow_imbalance(result)
    result["signed_trade_flow"] = result["aggressor_side"] * result["trade_size"]

    log_mid = np.log(result["mid_price"])
    result["log_return_1"] = log_mid.diff()
    result["momentum"] = log_mid.diff(rolling_window)
    result["rolling_volatility"] = result["log_return_1"].rolling(
        rolling_window, min_periods=rolling_window
    ).std()
    result["rolling_ofi"] = result["order_flow_imbalance"].rolling(
        rolling_window, min_periods=rolling_window
    ).sum()
    return result
