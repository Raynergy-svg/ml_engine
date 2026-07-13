"""Realized-outcome label generators for supervised training heads.

This package replaces the closed-form synthetic labels that previously
contaminated the confidence head (`load_ridge_data`) with labels derived
from actual trade outcomes (real journal labels) and triple-barrier
forward simulation (pseudo-labels for unlabeled rows).

Public API:
    compute_realized_confidence_labels — generate per-row labels in [0, 1]
    load_all_journal_trades — pool live + archive (salvage-tolerant) trades
"""

from __future__ import annotations

from src.training.labels.realized_confidence_label import (
    compute_realized_confidence_labels,
)
from src.training.labels.realized_volatility_regime_label import (
    NAN_SENTINEL,
    compute_realized_volatility_regime_labels,
    compute_forward_realized_volatility,
)
from src.training.labels.forward_drawdown_state_label import (
    DRAWDOWN_NAN_SENTINEL,
    compute_forward_drawdown_state_labels,
)
from src.training.labels.journal_loader import (
    load_all_journal_trades,
    salvage_corrupt_journal,
)

__all__ = [
    "compute_realized_confidence_labels",
    "compute_realized_volatility_regime_labels",
    "compute_forward_realized_volatility",
    "NAN_SENTINEL",
    "compute_forward_drawdown_state_labels",
    "DRAWDOWN_NAN_SENTINEL",
    "load_all_journal_trades",
    "salvage_corrupt_journal",
]
