"""Leakage-aware baseline modeling utilities."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


FEATURE_COLUMNS = [
    "relative_spread_bps",
    "top_imbalance",
    "weighted_midprice_distance_bps",
    "depth_imbalance",
    "top_depth_concentration",
    "order_flow_imbalance",
    "signed_trade_flow",
    "momentum",
    "rolling_volatility",
    "rolling_ofi",
]


@dataclass(frozen=True)
class ChronologicalSplit:
    train: pd.DataFrame
    test: pd.DataFrame


def chronological_split(frame: pd.DataFrame, test_fraction: float = 0.25) -> ChronologicalSplit:
    if not 0.0 < test_fraction < 1.0:
        raise ValueError("test_fraction must be between zero and one")
    if len(frame) < 20:
        raise ValueError("at least 20 observations are required")
    ordered = frame.sort_values("timestamp", kind="stable").reset_index(drop=True)
    split_at = int(len(ordered) * (1.0 - test_fraction))
    return ChronologicalSplit(ordered.iloc[:split_at].copy(), ordered.iloc[split_at:].copy())


def build_baseline(random_state: int = 42) -> Pipeline:
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            (
                "classifier",
                LogisticRegression(
                    max_iter=1_500,
                    class_weight="balanced",
                    random_state=random_state,
                ),
            ),
        ]
    )


def classification_metrics(y_true: pd.Series, y_pred: pd.Series) -> dict[str, object]:
    labels = [-1, 0, 1]
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
        "label_order": labels,
    }
