"""Deterministic heuristic bridges for leaky ML heads.

Re-exports the bridge functions that replace leaky training-time formulas
at runtime consumer sites. See each module's docstring for the bridge-
promotion gate.

Modules:
    volatility_regime: 4-class regime classifier (replaces TCN volatility head)
    momentum: momentum_score regression + acceleration binary (replaces LightGBM momentum head)
    streak_prob: streak probability regression (replaces LightGBM risk streak_prob output)
"""
from __future__ import annotations

from src.scanner.heuristics.momentum import (
    compute_acceleration,
    compute_momentum_score,
)
from src.scanner.heuristics.streak_prob import compute_streak_prob
from src.scanner.heuristics.volatility_regime import compute_volatility_regime

__all__ = [
    "compute_acceleration",
    "compute_momentum_score",
    "compute_streak_prob",
    "compute_volatility_regime",
]
