"""
Ensemble Model Conflict Resolution System.

Detects and penalizes directional conflicts between TCN, Ridge, and RF ensemble models.
Provides graduated penalty curves for model disagreement and direction conflict detection.

Key Features:
- Direction extraction from prediction scores (BUY/SELL/HOLD thresholds)
- Graduated penalty curve based on disagreement magnitude
- Direction conflict detection (when models suggest opposite directions)
- Consensus direction calculation via majority vote
- Graceful fallback to flat -15% penalty on error
- Config-driven customization with validation
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class ModelPrediction:
    """Single model's prediction for ensemble conflict analysis."""

    model_name: str
    """Model identifier: 'TCN', 'Ridge', 'RF'."""

    score: float
    """Raw prediction score (-1.0 to 1.0). Negative = SELL, Positive = BUY."""

    direction: str
    """Extracted direction: 'BUY', 'SELL', or 'HOLD'."""

    confidence: float
    """Model confidence in prediction (0.0 to 1.0)."""


@dataclass
class ConflictResult:
    """Result of ensemble conflict analysis."""

    has_direction_conflict: bool
    """True if models disagree on direction (one BUY, another SELL)."""

    disagreement_score: float
    """Numeric disagreement (0.0 to 1.0). Higher = more disagreement."""

    penalty: float
    """Penalty to apply to confidence (-1.0 to 0.0). E.g., -0.20 = 20% penalty."""

    should_block: bool
    """True if trade should be blocked entirely (hard gate)."""

    direction_votes: Dict[str, int]
    """Vote counts: {'BUY': 2, 'SELL': 1, 'HOLD': 0}."""

    consensus_direction: str
    """Majority vote direction: 'BUY', 'SELL', or 'HOLD'."""

    details: dict
    """Full breakdown for logging: prediction_scores, heuristics, etc."""


@dataclass
class ConflictConfig:
    """Configuration for ensemble conflict resolver."""

    enabled: bool = True
    """Enable conflict resolution. If False, returns zero penalty."""

    direction_threshold: float = 0.1
    """Score magnitude needed for BUY/SELL direction (vs HOLD).
    score > threshold → BUY, score < -threshold → SELL, else HOLD."""

    penalty_curve: List[Tuple[float, float]] = field(
        default_factory=lambda: [
            (0.30, -0.08),   # disagreement 0.30 → -8% penalty
            (0.35, -0.12),   # disagreement 0.35 → -12% penalty
            (0.40, -0.20),   # disagreement 0.40 → -20% penalty
            (0.45, -0.30),   # disagreement 0.45 → -30% penalty
            (0.50, -1.00),   # disagreement 0.50+ → block trade
        ]
    )
    """Graduated penalty curve: list of (disagreement_threshold, penalty)."""

    block_on_direction_conflict: bool = True
    """Block trade if 2+ active directions detected (BUY vs SELL)."""

    block_on_high_disagreement: bool = True
    """Block trade if disagreement >= 0.50."""


def create_default_conflict_resolver() -> EnsembleConflictResolver:
    """Factory: create resolver with default config."""
    return EnsembleConflictResolver(ConflictConfig())


class EnsembleConflictResolver:
    """Detects and penalizes directional conflicts between ensemble models."""

    def __init__(self, config: Optional[ConflictConfig] = None):
        """Initialize resolver with optional custom config.

        Args:
            config: ConflictConfig instance. If None, uses defaults.

        Raises:
            ValueError: If config validation fails.
        """
        self.config = config or ConflictConfig()
        self._validate_config()
        logger.debug(
            f"EnsembleConflictResolver initialized: enabled={self.config.enabled}, "
            f"direction_threshold={self.config.direction_threshold}"
        )

    def _validate_config(self) -> None:
        """Validate config at load time.

        Raises:
            ValueError: If any validation check fails.
        """
        errors = []

        # Check direction_threshold
        if not (0.0 <= self.config.direction_threshold <= 1.0):
            errors.append(
                f"direction_threshold must be [0.0, 1.0], got {self.config.direction_threshold}"
            )

        # Check penalty_curve structure
        if not self.config.penalty_curve:
            errors.append("penalty_curve cannot be empty")
        else:
            prev_disagreement = -1.0
            for i, (disagreement, penalty) in enumerate(self.config.penalty_curve):
                if not (0.0 <= disagreement <= 1.0):
                    errors.append(
                        f"penalty_curve[{i}] disagreement must be [0.0, 1.0], got {disagreement}"
                    )
                if not (-1.0 <= penalty <= 0.0):
                    errors.append(
                        f"penalty_curve[{i}] penalty must be [-1.0, 0.0], got {penalty}"
                    )
                if disagreement <= prev_disagreement:
                    errors.append(
                        f"penalty_curve[{i}] disagreement must be strictly increasing, "
                        f"got {disagreement} after {prev_disagreement}"
                    )
                prev_disagreement = disagreement

        if errors:
            msg = "ConflictConfig validation failed:\n" + "\n".join(errors)
            logger.error(msg)
            raise ValueError(msg)

        logger.debug("ConflictConfig validation passed")

    def resolve(self, model_predictions: List[ModelPrediction]) -> ConflictResult:
        """Analyze model predictions for conflicts and compute penalty.

        Args:
            model_predictions: List of ModelPrediction from ensemble models.

        Returns:
            ConflictResult with penalty, direction conflict info, and metadata.

        Note:
            - If config.enabled is False, returns zero penalty.
            - On exception, logs error and returns graceful -15% fallback penalty.
        """
        try:
            if not self.config.enabled:
                logger.debug("Conflict resolver disabled; returning zero penalty")
                return ConflictResult(
                    has_direction_conflict=False,
                    disagreement_score=0.0,
                    penalty=0.0,
                    should_block=False,
                    direction_votes={"BUY": 0, "SELL": 0, "HOLD": 0},
                    consensus_direction="HOLD",
                    details={"reason": "resolver_disabled"},
                )

            if not model_predictions:
                logger.warning("resolve() called with empty predictions list")
                return ConflictResult(
                    has_direction_conflict=False,
                    disagreement_score=0.0,
                    penalty=0.0,
                    should_block=False,
                    direction_votes={"BUY": 0, "SELL": 0, "HOLD": 0},
                    consensus_direction="HOLD",
                    details={"reason": "empty_predictions"},
                )

            # Extract directions from predictions
            directions = self._extract_directions(model_predictions)

            # Detect direction conflict
            has_direction_conflict = self._detect_direction_conflict(directions)

            # Calculate numeric disagreement
            disagreement_score = self._calculate_disagreement(model_predictions)

            # Compute penalty via curve interpolation
            penalty = self._interpolate_penalty(disagreement_score)

            # Decide whether to block
            should_block = self._should_block(directions, disagreement_score)

            # If blocking, override penalty to hard block (-1.0)
            if should_block:
                penalty = -1.0

            # Calculate consensus direction
            direction_votes = self._count_direction_votes(directions)
            consensus_direction = self._consensus_direction(direction_votes)

            result = ConflictResult(
                has_direction_conflict=has_direction_conflict,
                disagreement_score=disagreement_score,
                penalty=penalty,
                should_block=should_block,
                direction_votes=direction_votes,
                consensus_direction=consensus_direction,
                details={
                    "model_predictions": [
                        {
                            "model": p.model_name,
                            "score": p.score,
                            "direction": p.direction,
                            "confidence": p.confidence,
                        }
                        for p in model_predictions
                    ],
                    "directions_extracted": directions,
                    "has_direction_conflict": has_direction_conflict,
                    "disagreement_score": disagreement_score,
                    "penalty": penalty,
                    "should_block": should_block,
                },
            )

            logger.debug(
                f"Conflict resolution: disagreement={disagreement_score:.3f}, "
                f"penalty={penalty:.3f}, block={should_block}, "
                f"direction_conflict={has_direction_conflict}"
            )

            return result

        except Exception as e:
            # Catch all exceptions: ValueError, TypeError, KeyError, RuntimeError, etc.
            # This is a financial calculation path, so errors must be surfaced
            logger.error(
                f"Exception in resolve() during conflict analysis: {type(e).__name__}: {e}",
                exc_info=True,
            )
            # Graceful fallback: -15% penalty (backward compatible)
            fallback_penalty = -0.15
            logger.warning(
                f"Returning fallback penalty={fallback_penalty} due to resolution error"
            )
            return ConflictResult(
                has_direction_conflict=False,
                disagreement_score=0.0,
                penalty=fallback_penalty,
                should_block=False,
                direction_votes={"BUY": 0, "SELL": 0, "HOLD": 0},
                consensus_direction="HOLD",
                details={
                    "reason": "fallback_on_error",
                    "error": f"{type(e).__name__}: {e}",
                },
            )

    def _extract_directions(self, predictions: List[ModelPrediction]) -> Dict[str, str]:
        """Extract BUY/SELL/HOLD from each model's prediction score.

        Rules:
        - score > direction_threshold → BUY
        - score < -direction_threshold → SELL
        - else → HOLD

        Args:
            predictions: List of ModelPrediction.

        Returns:
            Dict[model_name] = direction string.
        """
        directions = {}
        for pred in predictions:
            score = float(pred.score)
            if score > self.config.direction_threshold:
                direction = "BUY"
            elif score < -self.config.direction_threshold:
                direction = "SELL"
            else:
                direction = "HOLD"
            directions[pred.model_name] = direction
            logger.debug(
                f"Model {pred.model_name}: score={score:.3f} → {direction}"
            )
        return directions

    def _detect_direction_conflict(self, directions: Dict[str, str]) -> bool:
        """Detect if any model says BUY while another says SELL.

        Args:
            directions: Dict[model_name] = direction.

        Returns:
            True if conflict detected (BUY and SELL both present).
        """
        has_buy = "BUY" in directions.values()
        has_sell = "SELL" in directions.values()
        conflict = has_buy and has_sell
        if conflict:
            logger.debug(
                f"Direction conflict detected: some models BUY, others SELL. "
                f"Directions: {directions}"
            )
        return conflict

    def _calculate_disagreement(self, predictions: List[ModelPrediction]) -> float:
        """Calculate numeric disagreement from prediction spread.

        Formula:
        disagreement = std(scores) / max(abs(mean(scores)), 0.01)
        Then clamp to [0.0, 1.0].

        Rationale:
        - High std + low mean → high disagreement (models spread out)
        - Low std → low disagreement (models clustered)
        - Normalize by mean to avoid scale bias

        Args:
            predictions: List of ModelPrediction.

        Returns:
            Disagreement score [0.0, 1.0].
        """
        if not predictions:
            return 0.0

        scores = np.array([float(p.score) for p in predictions])

        if len(scores) == 1:
            # Only one model: no disagreement possible
            return 0.0

        mean_score = float(np.mean(scores))
        std_score = float(np.std(scores))

        # Avoid division by zero
        denominator = max(abs(mean_score), 0.01)
        disagreement = std_score / denominator

        # Clamp to [0, 1]
        disagreement = float(np.clip(disagreement, 0.0, 1.0))

        logger.debug(
            f"Disagreement calculation: scores={scores}, mean={mean_score:.3f}, "
            f"std={std_score:.3f}, disagreement={disagreement:.3f}"
        )

        return disagreement

    def _interpolate_penalty(self, disagreement: float) -> float:
        """Interpolate penalty from graduated curve.

        If disagreement falls between two curve points, linearly interpolate.
        If above max, use max penalty. If below min, use 0 (no penalty).

        Args:
            disagreement: Disagreement score [0.0, 1.0].

        Returns:
            Penalty [-1.0, 0.0].
        """
        disagreement = float(np.clip(disagreement, 0.0, 1.0))

        # Find bounding curve points
        curve = self.config.penalty_curve
        if not curve:
            logger.warning("Empty penalty curve; returning 0.0")
            return 0.0

        penalty = 0.0  # Default: no penalty below first point

        if disagreement < curve[0][0]:
            # Below first curve point: no penalty
            penalty = 0.0
        elif disagreement >= curve[-1][0]:
            # At or above last point: use last penalty
            penalty = curve[-1][1]
        else:
            # Interpolate between two curve points
            for i in range(len(curve) - 1):
                d1, p1 = curve[i]
                d2, p2 = curve[i + 1]
                if d1 <= disagreement < d2:
                    # Linear interpolation between points
                    t = (disagreement - d1) / (d2 - d1)
                    penalty = p1 + t * (p2 - p1)
                    break
                elif disagreement == d2:
                    # Exactly at this point
                    penalty = p2
                    break

        penalty = float(np.clip(penalty, -1.0, 0.0))
        logger.debug(
            f"Penalty interpolation: disagreement={disagreement:.3f} → penalty={penalty:.3f}"
        )
        return penalty

    def _should_block(self, directions: Dict[str, str], disagreement: float) -> bool:
        """Decide whether to block trade entirely.

        Block if:
        1. block_on_direction_conflict=True AND 2+ active directions (BUY vs SELL)
        2. block_on_high_disagreement=True AND disagreement >= 0.50

        Args:
            directions: Dict[model_name] = direction.
            disagreement: Numeric disagreement [0.0, 1.0].

        Returns:
            True if trade should be blocked.
        """
        active_directions = {d for d in directions.values() if d != "HOLD"}

        block_by_conflict = (
            self.config.block_on_direction_conflict and len(active_directions) > 1
        )
        block_by_disagreement = (
            self.config.block_on_high_disagreement and disagreement >= 0.50
        )

        should_block = block_by_conflict or block_by_disagreement

        if should_block:
            reason_parts = []
            if block_by_conflict:
                reason_parts.append(
                    f"direction_conflict ({len(active_directions)} active directions)"
                )
            if block_by_disagreement:
                reason_parts.append(f"high_disagreement ({disagreement:.3f} >= 0.50)")
            logger.debug(f"Trade block triggered: {', '.join(reason_parts)}")

        return should_block

    def _count_direction_votes(self, directions: Dict[str, str]) -> Dict[str, int]:
        """Count votes for each direction.

        Args:
            directions: Dict[model_name] = direction.

        Returns:
            Dict with vote counts: {'BUY': int, 'SELL': int, 'HOLD': int}.
        """
        votes = {"BUY": 0, "SELL": 0, "HOLD": 0}
        for direction in directions.values():
            if direction in votes:
                votes[direction] += 1
        return votes

    def _consensus_direction(self, direction_votes: Dict[str, int]) -> str:
        """Calculate consensus direction via majority vote.

        Priority: BUY > SELL > HOLD (if tie).
        HOLD is returned only if no BUY or SELL votes.

        Args:
            direction_votes: Dict with vote counts.

        Returns:
            Consensus direction: 'BUY', 'SELL', or 'HOLD'.
        """
        if direction_votes["BUY"] > 0 and direction_votes["BUY"] >= direction_votes["SELL"]:
            return "BUY"
        elif direction_votes["SELL"] > 0:
            return "SELL"
        else:
            return "HOLD"
