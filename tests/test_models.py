import pandas as pd

from market_microstructure.models import chronological_split


def test_chronological_split_does_not_shuffle():
    frame = pd.DataFrame(
        {
            "timestamp": pd.date_range("2026-01-01", periods=40, freq="s", tz="UTC"),
            "value": range(40),
        }
    )
    split = chronological_split(frame, test_fraction=0.25)
    assert split.train["timestamp"].max() < split.test["timestamp"].min()
    assert len(split.train) == 30
    assert len(split.test) == 10
