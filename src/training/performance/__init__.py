"""Phase F performance contracts and data-materialization primitives."""

from .metrics import PerformanceBaseline, PerformanceProbe, write_baseline
from .rl_dataset import (
    RLMatrixBundle,
    find_rl_matrices,
    load_rl_matrices,
    materialize_rl_matrices,
    predict_fixed_windows,
)

__all__ = [
    "PerformanceBaseline",
    "PerformanceProbe",
    "RLMatrixBundle",
    "find_rl_matrices",
    "load_rl_matrices",
    "materialize_rl_matrices",
    "predict_fixed_windows",
    "write_baseline",
]
