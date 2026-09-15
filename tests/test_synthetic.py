import pandas as pd

from market_microstructure.synthetic import SyntheticLOBConfig, generate_synthetic_lob


def test_generator_is_deterministic():
    config = SyntheticLOBConfig(rows=120, levels=3, seed=9)
    pd.testing.assert_frame_equal(generate_synthetic_lob(config), generate_synthetic_lob(config))


def test_book_is_ordered_and_spread_is_nonnegative():
    frame = generate_synthetic_lob(SyntheticLOBConfig(rows=120, levels=5))
    assert (frame["ask_price_1"] >= frame["bid_price_1"]).all()
    for level in range(1, 5):
        assert (frame[f"bid_price_{level}"] >= frame[f"bid_price_{level + 1}"]).all()
        assert (frame[f"ask_price_{level}"] <= frame[f"ask_price_{level + 1}"]).all()
