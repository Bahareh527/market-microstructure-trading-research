"""Deterministic synthetic limit-order-book snapshots."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class SyntheticLOBConfig:
    rows: int = 6_000
    levels: int = 5
    seed: int = 42
    start_price: float = 100.0
    tick_size: float = 0.01
    frequency: str = "1s"

    def validate(self) -> None:
        if self.rows < 100:
            raise ValueError("rows must be at least 100")
        if self.levels < 1:
            raise ValueError("levels must be positive")
        if self.start_price <= 0 or self.tick_size <= 0:
            raise ValueError("prices and tick size must be positive")


def _sigmoid(value: float) -> float:
    return 1.0 / (1.0 + np.exp(-value))


def generate_synthetic_lob(config: SyntheticLOBConfig | None = None) -> pd.DataFrame:
    """Generate a reproducible L2 snapshot stream with no real market data.

    A persistent latent pressure process influences both displayed depth and later
    price changes. The relationship exists solely to make the research workflow
    testable; it is not intended to simulate a particular venue or instrument.
    """

    config = config or SyntheticLOBConfig()
    config.validate()
    rng = np.random.default_rng(config.seed)

    rows: list[dict[str, float | int | str]] = []
    price = config.start_price
    pressure = 0.0
    regime = 0.0
    timestamps = pd.date_range(
        "2026-01-02 09:30:00", periods=config.rows, freq=config.frequency, tz="UTC"
    )

    for index, timestamp in enumerate(timestamps):
        if index % 350 == 0:
            regime = float(rng.normal(0.0, 0.55))

        pressure = 0.91 * pressure + 0.10 * regime + float(rng.normal(0.0, 0.38))
        upward_probability = _sigmoid(0.68 * pressure)
        movement_probability = min(0.58, 0.11 + 0.10 * abs(pressure))
        draw = float(rng.random())
        if draw < movement_probability:
            price += config.tick_size * (1.0 if rng.random() < upward_probability else -1.0)
        price = max(price, 10.0 * config.tick_size)

        spread_ticks = int(rng.choice([1, 2, 3], p=[0.72, 0.23, 0.05]))
        best_bid = price - spread_ticks * config.tick_size / 2.0
        best_ask = price + spread_ticks * config.tick_size / 2.0
        imbalance = float(np.tanh(pressure / 2.2))

        row: dict[str, float | int | str] = {
            "timestamp": timestamp.isoformat(),
            "mid_price": round((best_bid + best_ask) / 2.0, 6),
        }
        for level in range(1, config.levels + 1):
            decay = 1.0 + 0.16 * (level - 1)
            base_depth = float(rng.lognormal(mean=4.15, sigma=0.33)) * decay
            bid_multiplier = max(0.20, 1.0 + 0.48 * imbalance)
            ask_multiplier = max(0.20, 1.0 - 0.48 * imbalance)
            row[f"bid_price_{level}"] = round(
                best_bid - (level - 1) * config.tick_size, 6
            )
            row[f"ask_price_{level}"] = round(
                best_ask + (level - 1) * config.tick_size, 6
            )
            row[f"bid_size_{level}"] = round(
                max(1.0, base_depth * bid_multiplier * rng.lognormal(0.0, 0.12)), 3
            )
            row[f"ask_size_{level}"] = round(
                max(1.0, base_depth * ask_multiplier * rng.lognormal(0.0, 0.12)), 3
            )

        aggressor_side = 1 if rng.random() < _sigmoid(0.60 * pressure) else -1
        row["aggressor_side"] = aggressor_side
        row["trade_price"] = row["ask_price_1"] if aggressor_side == 1 else row["bid_price_1"]
        row["trade_size"] = round(float(rng.lognormal(mean=2.8, sigma=0.55)), 3)
        rows.append(row)

    return pd.DataFrame(rows)
