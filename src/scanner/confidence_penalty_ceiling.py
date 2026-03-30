"""Confidence Penalty Ceiling — Phase 57 US-352.

Prevents stacked confidence penalties from pushing all signals below
execution threshold. Three independent penalty systems (overconfidence,
concept drift, ensemble disagreement) currently operate without awareness
of each other. This module enforces a hard floor: no pair can be penalized
below confidence=40.0 (on 0-100 scale) by stacking alone.

The ceiling is NOT a gate bypass — a pair still must clear all gates.
It prevents denominator collapse where 0.70→0.67 from overconfidence, then
0.67*0.88=0.59 from drift, suppressing a signal that may have been valid.

Usage (0-1 scale):
    ceiling = ConfidencePenaltyCeiling()
    # Before applying -0.03 overconfidence subtraction:
    result = ceiling.check_subtraction(confidence, 0.03)
    confidence -= result.applied_penalty
    # Before applying drift multiplier:
    result = ceiling.check_multiplier(confidence, drift_mult)
    confidence *= result.applied_penalty
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)

# Default floor: pairs cannot be penalized below this confidence value (0-1 scale)
# Phase 79: Fixed from 40.0 (0-100 scale) to 0.40 (0-1 scale) to match actual confidence domain
DEFAULT_CEILING_FLOOR = 0.40

# Same floor in 0-1 normalised scale (kept for backward compat with multiplier checks)
DEFAULT_CEILING_FLOOR_NORM = 0.40


@dataclass
class CeilingResult:
    """Result of a ceiling check."""
    attempted_penalty: float
    applied_penalty: float
    ceiling_active: bool
    confidence_before: float
    confidence_after: float


class ConfidencePenaltyCeiling:
    """Enforces a hard floor on per-scan cumulative confidence penalties.

    Prevents three independent penalty systems (overconfidence, concept drift,
    ensemble disagreement) from stacking to suppress all signals. Each system
    calls this class before applying its penalty; the ceiling adjusts the
    penalty downward if the pair is already near the floor.

    The floor is expressed on the 0-1 scale (matching engine.py's confidence).
    Phase 79: Fixed from 0-100 to 0-1 scale.

    Usage (0-1 scale):
        ceiling = ConfidencePenaltyCeiling()
        # Additive penalty (subtraction):
        result = ceiling.check_subtraction(current_confidence=0.70, proposed_sub=0.03)
        confidence -= result.applied_penalty

        # Multiplicative penalty (multiplier < 1.0):
        result = ceiling.check_multiplier(current_confidence=0.70, proposed_mult=0.85)
        confidence *= result.applied_penalty  # applied_penalty is the safe multiplier
    """

    def __init__(
        self,
        ceiling_floor: float = DEFAULT_CEILING_FLOOR,
        ceiling_floor_norm: float = DEFAULT_CEILING_FLOOR_NORM,
    ):
        self.ceiling_floor = float(ceiling_floor)
        self.ceiling_floor_norm = float(ceiling_floor_norm)

        # Counters for monitoring
        self._total_checks: int = 0
        self._ceiling_activations: int = 0

    # ── Public API ─────────────────────────────────────────────────────────

    def check_subtraction(
        self,
        current_confidence: float,
        proposed_sub: float,
        pair: str = "",
        source: str = "",
    ) -> CeilingResult:
        """Check an additive subtraction penalty against the ceiling floor.

        Args:
            current_confidence: Current confidence on 0-100 scale.
            proposed_sub: Amount to subtract (positive number, e.g. 3.0).
            pair: Pair name for logging.
            source: Penalty source name for logging.

        Returns:
            CeilingResult with applied_penalty capped so confidence stays >= floor.
        """
        self._total_checks += 1
        proposed_sub = max(0.0, float(proposed_sub))
        current_confidence = float(current_confidence)

        # How much room before we hit the floor?
        headroom = max(0.0, current_confidence - self.ceiling_floor)
        applied_sub = min(proposed_sub, headroom)
        ceiling_active = applied_sub < proposed_sub

        if ceiling_active:
            self._ceiling_activations += 1
            logger.debug(
                "%s: Penalty ceiling active [%s] — attempted=%.2f applied=%.2f "
                "(conf=%.1f floor=%.1f)",
                pair or "?", source or "unknown",
                proposed_sub, applied_sub, current_confidence, self.ceiling_floor,
            )

        return CeilingResult(
            attempted_penalty=proposed_sub,
            applied_penalty=applied_sub,
            ceiling_active=ceiling_active,
            confidence_before=current_confidence,
            confidence_after=current_confidence - applied_sub,
        )

    def check_multiplier(
        self,
        current_confidence: float,
        proposed_mult: float,
        pair: str = "",
        source: str = "",
        scale: str = "0-100",
    ) -> CeilingResult:
        """Check a multiplicative penalty (multiplier < 1.0) against the ceiling floor.

        The multiplier is safe if: current_confidence * proposed_mult >= floor.
        If not, we compute the minimum multiplier that keeps us at the floor.

        Args:
            current_confidence: Current confidence on 0-100 scale (or 0-1 if scale='0-1').
            proposed_mult: Proposed confidence multiplier (0 < mult <= 1.0).
            pair: Pair name for logging.
            source: Penalty source name for logging.
            scale: '0-100' (default) or '0-1' — determines which floor to use.

        Returns:
            CeilingResult where applied_penalty is the safe multiplier to use.
        """
        self._total_checks += 1
        proposed_mult = max(0.0, min(1.0, float(proposed_mult)))
        current_confidence = float(current_confidence)

        floor = self.ceiling_floor if scale == "0-100" else self.ceiling_floor_norm
        projected = current_confidence * proposed_mult

        if projected >= floor or current_confidence <= floor:
            # Either multiplier is safe, or confidence is already at/below floor
            # (in the latter case, don't apply further multiplier reduction)
            safe_mult = proposed_mult if current_confidence > floor else 1.0
            ceiling_active = current_confidence <= floor and proposed_mult < 1.0
        else:
            # Compute minimum safe multiplier: floor / current_confidence
            safe_mult = floor / current_confidence if current_confidence > 0 else 1.0
            safe_mult = max(safe_mult, 0.0)
            ceiling_active = True

        if ceiling_active:
            self._ceiling_activations += 1
            logger.debug(
                "%s: Multiplier ceiling active [%s] — attempted=%.3f applied=%.3f "
                "(conf=%.3f floor=%.3f scale=%s)",
                pair or "?", source or "unknown",
                proposed_mult, safe_mult, current_confidence, floor, scale,
            )

        return CeilingResult(
            attempted_penalty=proposed_mult,
            applied_penalty=safe_mult,
            ceiling_active=ceiling_active,
            confidence_before=current_confidence,
            confidence_after=current_confidence * safe_mult,
        )

    def get_stats(self) -> dict:
        """Return ceiling activation statistics."""
        activation_rate = (
            self._ceiling_activations / self._total_checks
            if self._total_checks > 0 else 0.0
        )
        return {
            "total_checks": self._total_checks,
            "ceiling_activations": self._ceiling_activations,
            "activation_rate": round(activation_rate, 4),
            "ceiling_floor": self.ceiling_floor,
            "ceiling_floor_norm": self.ceiling_floor_norm,
        }

    def reset_stats(self) -> None:
        """Reset per-session counters."""
        self._total_checks = 0
        self._ceiling_activations = 0
