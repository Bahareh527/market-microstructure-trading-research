"""End-to-end reproducible experiment."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from .backtest import run_backtest
from .features import engineer_features
from .labels import add_future_move_label
from .models import FEATURE_COLUMNS, build_baseline, chronological_split, classification_metrics
from .synthetic import SyntheticLOBConfig, generate_synthetic_lob


def run_experiment(
    results_dir: str | Path,
    data_dir: str | Path,
    rows: int = 6_000,
    seed: int = 42,
) -> dict[str, object]:
    results_path = Path(results_dir)
    data_path = Path(data_dir)
    results_path.mkdir(parents=True, exist_ok=True)
    data_path.mkdir(parents=True, exist_ok=True)

    raw = generate_synthetic_lob(SyntheticLOBConfig(rows=rows, seed=seed))
    raw.head(1_000).to_csv(data_path / "sample_lob.csv", index=False)
    featured = engineer_features(raw)
    labeled = add_future_move_label(featured, horizon=10, threshold_bps=0.5)
    model_frame = labeled.dropna(subset=FEATURE_COLUMNS + ["target"]).copy()
    model_frame["target"] = model_frame["target"].astype(int)
    split = chronological_split(model_frame, test_fraction=0.25)

    model = build_baseline(random_state=seed)
    model.fit(split.train[FEATURE_COLUMNS], split.train["target"])
    predictions = pd.Series(
        model.predict(split.test[FEATURE_COLUMNS]), index=split.test.index, name="prediction"
    )
    classification = classification_metrics(split.test["target"], predictions)
    backtest_frame, backtest = run_backtest(
        split.test["mid_price"], predictions, transaction_cost_bps=1.0
    )

    label_distribution = {
        str(int(label)): int(count)
        for label, count in model_frame["target"].value_counts().sort_index().items()
    }
    metrics: dict[str, object] = {
        "experiment": "synthetic_lob_logistic_baseline",
        "seed": seed,
        "raw_rows": len(raw),
        "model_rows": len(model_frame),
        "train_rows": len(split.train),
        "test_rows": len(split.test),
        "features": FEATURE_COLUMNS,
        "label_distribution": label_distribution,
        "classification": classification,
        "backtest": backtest,
        "disclaimer": "Synthetic research result; not evidence of live-trading performance.",
    }
    (results_path / "baseline_metrics.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )

    figure, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    display_slice = model_frame.iloc[:700]
    axes[0, 0].plot(display_slice["timestamp"], display_slice["mid_price"], linewidth=1.0)
    axes[0, 0].set_title("Synthetic mid-price")
    axes[0, 0].set_ylabel("Price units")
    axes[0, 1].plot(
        display_slice["timestamp"], display_slice["depth_imbalance"],
        linewidth=0.9, color="#b45309"
    )
    axes[0, 1].axhline(0.0, color="black", linewidth=0.6)
    axes[0, 1].set_title("Depth imbalance")
    counts = model_frame["target"].value_counts().reindex([-1, 0, 1], fill_value=0)
    axes[1, 0].bar(["Down", "Flat", "Up"], counts.values, color=["#b91c1c", "#64748b", "#047857"])
    axes[1, 0].set_title("Label distribution")
    axes[1, 1].plot(backtest_frame["equity_curve"], color="#1d4ed8", linewidth=1.1)
    axes[1, 1].axhline(1.0, color="black", linewidth=0.6)
    axes[1, 1].set_title("Cost-aware test equity curve")
    axes[1, 1].set_xlabel("Test snapshot")
    figure.suptitle("Reproducible synthetic baseline", fontsize=14)
    figure.savefig(results_path / "baseline_overview.png", dpi=160)
    plt.close(figure)
    return metrics
