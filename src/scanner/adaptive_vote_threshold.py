"""Phase 66 US-390: AdaptiveVoteThreshold.

Mirrors AdaptiveMomentumFloor (Phase 62 US-375) for the agent consensus domain.

When VoteGapAnalyzer reports NEAR_THRESHOLD (gap_at_p50 <= 0.05), pairs
are consistently scoring just below weighted_vote_threshold — a micro-calibration
issue, not structural agent weakness. In that case, a small automatic reduction
in weighted_vote_threshold (0.02 per adaptation) can unlock trades without
materially reducing signal quality.

Guard conditions:
  - classification must be NEAR_THRESHOLD
  - cooldown_remaining must be 0
  - adaptation_count < max_adaptations (default 5)
  - current_threshold - step >= floor (default 0.40)

Vote score is 0.0-1.0 scale (agent team weighted average), so:
  step  = 0.02  (same scale as momentum)
  floor = 0.40  (config.py validation already clamps to [0.40, 0.95])
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

# ─── constants ────────────────────────────────────────────────────────────────

CLASS_NEAR_THRESHOLD = "NEAR_THRESHOLD"

DEFAULT_STEP = 0.02
DEFAULT_COOLDOWN = 20
DEFAULT_MAX_ADAPTATIONS = 5
DEFAULT_FLOOR = 0.40
DEFAULT_STATE_PATH = Path("trained_data/adaptive_vote_threshold_state.json")


class AdaptiveVoteThreshold:
    """Auto-reduce weighted_vote_threshold by a small step when NEAR_THRESHOLD.

    Usage:
        avt = AdaptiveVoteThreshold()
        new_threshold = avt.tick(classification, current_threshold)
        if new_threshold is not None:
            config.weighted_vote_threshold = new_threshold
    """

    def __init__(
        self,
        step: float = DEFAULT_STEP,
        cooldown: int = DEFAULT_COOLDOWN,
        max_adaptations: int = DEFAULT_MAX_ADAPTATIONS,
        floor: float = DEFAULT_FLOOR,
        state_path: Path = DEFAULT_STATE_PATH,
    ) -> None:
        self._step = float(step)
        self._cooldown = int(cooldown)
        self._max_adaptations = int(max_adaptations)
        self._floor = float(floor)
        self._state_path = Path(state_path)

        # Mutable state — persisted
        self._adaptation_count: int = 0
        self._cooldown_remaining: int = 0
        self._last_adapted_threshold: Optional[float] = None
        self._total_reduction: float = 0.0

        self._load_state()

    # ------------------------------------------------------------------ #
    # Core tick                                                            #
    # ------------------------------------------------------------------ #

    def tick(self, classification: str, current_threshold: float) -> Optional[float]:
        """Evaluate one scan cycle.

        Args:
            classification: VoteGapAnalyzer classification string.
            current_threshold: The current config.weighted_vote_threshold value.

        Returns:
            New threshold float if adaptation fired, else None.
        """
        # Decrement cooldown on every call (regardless of classification)
        if self._cooldown_remaining > 0:
            self._cooldown_remaining -= 1

        # Check all guard conditions
        if classification != CLASS_NEAR_THRESHOLD:
            return None

        if self._cooldown_remaining > 0:
            logger.debug(
                "AdaptiveVoteThreshold: NEAR_THRESHOLD but cooldown=%d remaining",
                self._cooldown_remaining,
            )
            return None

        if self._adaptation_count >= self._max_adaptations:
            logger.debug(
                "AdaptiveVoteThreshold: max adaptations (%d) exhausted",
                self._max_adaptations,
            )
            return None

        candidate = round(float(current_threshold) - self._step, 4)
        if candidate < self._floor:
            logger.debug(
                "AdaptiveVoteThreshold: candidate=%.4f < floor=%.4f — blocked",
                candidate, self._floor,
            )
            return None

        # Fire adaptation
        self._adaptation_count += 1
        self._cooldown_remaining = self._cooldown
        self._last_adapted_threshold = candidate
        self._total_reduction = round(self._total_reduction + self._step, 4)

        logger.info(
            "AdaptiveVoteThreshold: weighted_vote_threshold reduced %.4f → %.4f "
            "(adaptation %d/%d, total_reduction=%.4f)",
            current_threshold, candidate,
            self._adaptation_count, self._max_adaptations,
            self._total_reduction,
        )
        return candidate

    # ------------------------------------------------------------------ #
    # Status                                                               #
    # ------------------------------------------------------------------ #

    def get_status(self) -> Dict[str, Any]:
        """Return a status dict for logging and external inspection."""
        return {
            "adaptation_count": self._adaptation_count,
            "cooldown_remaining": self._cooldown_remaining,
            "max_adaptations": self._max_adaptations,
            "floor": self._floor,
            "step": self._step,
            "total_reduction": self._total_reduction,
            "last_adapted_threshold": self._last_adapted_threshold,
        }

    # ------------------------------------------------------------------ #
    # Persistence                                                          #
    # ------------------------------------------------------------------ #

    def save_state(self, path: Optional[Path] = None) -> None:
        target = Path(path) if path else self._state_path
        target.parent.mkdir(parents=True, exist_ok=True)
        state = {
            "adaptation_count": self._adaptation_count,
            "cooldown_remaining": self._cooldown_remaining,
            "last_adapted_threshold": self._last_adapted_threshold,
            "total_reduction": self._total_reduction,
        }
        try:
            fd, tmp = tempfile.mkstemp(dir=str(target.parent), suffix=".tmp")
            with os.fdopen(fd, "w") as f:
                json.dump(state, f, indent=2, sort_keys=True)
            os.rename(tmp, str(target))
        except Exception as exc:
            logger.error("AdaptiveVoteThreshold.save_state failed: %s", exc)
            try:
                os.unlink(tmp)
            except Exception:
                pass

    def _load_state(self) -> None:
        if not self._state_path.exists():
            return
        try:
            with open(self._state_path) as f:
                state = json.load(f)
            self._adaptation_count = int(state.get("adaptation_count", 0))
            self._cooldown_remaining = int(state.get("cooldown_remaining", 0))
            self._last_adapted_threshold = state.get("last_adapted_threshold")
            self._total_reduction = float(state.get("total_reduction", 0.0))
        except Exception as exc:
            logger.error("AdaptiveVoteThreshold._load_state failed: %s", exc)
