"""
Empirical model evaluation: backtest, error correlation, meta-ensemble.

Answers the only question that matters:
  1. Does the raw-sequence model beat the existing ensemble on live data?
  2. Are their errors decorrelated?

If (1) and (2) → meta-weighted stacked ensemble.
If (1) and not (2) → candidate replacement (soak decides).
If not (1) → interesting experiment, archive it.
"""

from __future__ import annotations

from .model_comparison import ModelComparisonResult, compare_models_on_history
from .error_correlation import ErrorCorrelation, analyze_error_correlation
from .meta_ensemble import MetaEnsemble, compute_meta_weights
from .soak_runner import SoakResult, run_soak_comparison

__all__ = [
    "ModelComparisonResult",
    "compare_models_on_history",
    "ErrorCorrelation",
    "analyze_error_correlation",
    "MetaEnsemble",
    "compute_meta_weights",
    "SoakResult",
    "run_soak_comparison",
]
