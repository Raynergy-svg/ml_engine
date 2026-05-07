"""Deterministic heuristic bridges for leaky ML heads.

Re-exports the bridge functions that replace leaky training-time formulas
at runtime consumer sites. See each module's docstring for the bridge-
promotion gate.

Modules (state):
    volatility_regime: 4-class regime classifier (replaces TCN volatility head) — landed
    momentum: momentum_score regression + acceleration binary — landed
    streak_prob: streak probability regression — pending
"""
from __future__ import annotations

from src.scanner.heuristics.momentum import (
    compute_acceleration,
    compute_momentum_score,
)
from src.scanner.heuristics.volatility_regime import compute_volatility_regime

__all__ = [
    "compute_acceleration",
    "compute_momentum_score",
    "compute_volatility_regime",
]
