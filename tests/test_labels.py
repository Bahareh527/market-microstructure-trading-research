import pandas as pd

from market_microstructure.labels import add_future_move_label


def test_labels_use_only_future_horizon_and_leave_tail_unknown():
    frame = pd.DataFrame({"mid_price": [100.0, 100.0, 100.02, 99.98, 100.0]})
    result = add_future_move_label(frame, horizon=2, threshold_bps=1.0)
    assert result.loc[0, "target"] == 1
    assert result.loc[1, "target"] == -1
    assert result["target"].tail(2).isna().all()
