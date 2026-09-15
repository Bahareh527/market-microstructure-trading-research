"""Command-line entry point."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .pipeline import run_experiment


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the synthetic LOB research baseline")
    parser.add_argument("--rows", type=int, default=6_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, default=Path("results"))
    parser.add_argument("--data", type=Path, default=Path("data"))
    args = parser.parse_args()
    metrics = run_experiment(args.output, args.data, rows=args.rows, seed=args.seed)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
