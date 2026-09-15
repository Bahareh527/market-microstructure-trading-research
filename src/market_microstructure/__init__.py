"""Clean-room market-microstructure research utilities."""

from .backtest import run_backtest
from .features import engineer_features
from .labels import add_future_move_label
from .synthetic import SyntheticLOBConfig, generate_synthetic_lob

__all__ = [
    "SyntheticLOBConfig",
    "add_future_move_label",
    "engineer_features",
    "generate_synthetic_lob",
    "run_backtest",
]
__version__ = "0.1.0"
