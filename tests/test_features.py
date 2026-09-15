import pandas as pd
import pytest

from market_microstructure.features import engineer_features, order_flow_imbalance
from market_microstructure.synthetic import SyntheticLOBConfig, generate_synthetic_lob


def test_imbalance_bounds_and_weighted_midprice():
    raw = generate_synthetic_lob(SyntheticLOBConfig(rows=140, levels=5))
    result = engineer_features(raw, levels=5, rolling_window=10)
    assert result["top_imbalance"].between(-1.0, 1.0).all()
    assert result["depth_imbalance"].between(-1.0, 1.0).all()
    first = result.iloc[0]
    expected = (
        first["ask_price_1"] * first["bid_size_1"]
        + first["bid_price_1"] * first["ask_size_1"]
    ) / (first["bid_size_1"] + first["ask_size_1"])
    assert first["weighted_midprice"] == pytest.approx(expected)


def test_order_flow_imbalance_sign_convention():
    frame = pd.DataFrame(
        {
            "bid_price_1": [100.0, 100.0, 100.1],
            "ask_price_1": [100.2, 100.2, 100.2],
            "bid_size_1": [10.0, 14.0, 8.0],
            "ask_size_1": [12.0, 9.0, 9.0],
        }
    )
    values = order_flow_imbalance(frame)
    assert values.iloc[0] == 0.0
    assert values.iloc[1] == pytest.approx(7.0)
    assert values.iloc[2] == pytest.approx(8.0)
