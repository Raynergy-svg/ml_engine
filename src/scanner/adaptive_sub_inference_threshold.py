"""Phase 67 US-393: AdaptiveSubInferenceThreshold.

Mirrors AdaptiveMomentumFloor (Phase 62) for the sub-inference domain.

When SubInferenceGapAnalyzer reports NEAR_THRESHOLD, the sub-inference
window checks are close to consensus but just failing the vote count gate.
A small reduction in sub_inference_vote_threshold can unlock trades.

The sub_inference_vote_threshold controls:
  vote_required = ceil(total * threshold)
  agent_passed = votes >= vote_required

For window_checks=3:
  threshold 0.66 → need 2/3 votes
  threshold 0.50 → need 2/3 votes (ceil rounds up)
  threshold 0.34 → need 1/3 votes

Guard conditions:
  - classification must be NEAR_THRESHOLD
  - cooldown_remaining must be 0
  - adaptation_count < max_adaptations (default 3, more conservative)
  - current_threshold - step >= floor (default 0.40)

Sub-inference threshold adaptations are more conservative than other gates
because lowering it too much effectively removes the consensus check.

step  = 0.05  (coarser than momentum/vote 0.02 — threshold is quantized)
floor = 0.40  (config.py validates [0.34, 1.0])
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

DEFAULT_STEP = 0.05
DEFAULT_COOLDOWN = 30       # More conservative cooldown (vs 20 for other gates)
DEFAULT_MAX_ADAPTATIONS = 3  # Fewer adaptations allowed (vs 5 for other gates)
DEFAULT_FLOOR = 0.40
DEFAULT_STATE_PATH = Path("trained_data/adaptive_sub_inference_threshold_state.json")


class AdaptiveSubInferenceThreshold:
    """Auto-reduce sub_inference_vote_threshold when NEAR_THRESHOLD.

    Usage:
        asit = AdaptiveSubInferenceThreshold()
        new_threshold = asit.tick(classification, current_threshold)
        if new_threshold is not None:
            config.sub_inference_vote_threshold = new_threshold
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

    def tick(self, classification: str, current_threshold: float, total_agents: int = 3) -> Optional[float]:
        """Evaluate one scan cycle.

        Args:
            classification: SubInferenceGapAnalyzer classification string.
            current_threshold: The current config.sub_inference_vote_threshold value.
            total_agents: Number of window checks (used for quantization awareness).

        Returns:
            New threshold float if adaptation fired, else None.
        """
        import math

        if self._cooldown_remaining > 0:
            self._cooldown_remaining -= 1

        if classification != CLASS_NEAR_THRESHOLD:
            return None

        if self._cooldown_remaining > 0:
            logger.debug(
                "AdaptiveSubInferenceThreshold: NEAR_THRESHOLD but cooldown=%d remaining",
                self._cooldown_remaining,
            )
            return None

        if self._adaptation_count >= self._max_adaptations:
            logger.debug(
                "AdaptiveSubInferenceThreshold: max adaptations (%d) exhausted",
                self._max_adaptations,
            )
            return None

        candidate = round(float(current_threshold) - self._step, 4)
        if candidate < self._floor:
            logger.debug(
                "AdaptiveSubInferenceThreshold: candidate=%.4f < floor=%.4f — blocked",
                candidate, self._floor,
            )
            return None

        # Quantization awareness: check if adaptation actually changes vote_required
        curr_required = max(1, math.ceil(total_agents * current_threshold))
        cand_required = max(1, math.ceil(total_agents * candidate))

        if cand_required == curr_required:
            # Standard step has no effect due to ceil() quantization.
            # Jump to the boundary where vote_required actually drops.
            boundary = (curr_required - 1) / total_agents
            candidate = round(boundary - 0.001, 4)
            if candidate < self._floor:
                logger.info(
                    "AdaptiveSubInferenceThreshold: quantization boundary %.4f < floor %.4f — blocked",
                    candidate, self._floor,
                )
                return None
            cand_required = max(1, math.ceil(total_agents * candidate))
            if cand_required == curr_required:
                # Still no change -- guard anyway
                return None

        actual_reduction = round(float(current_threshold) - candidate, 4)

        # Fire adaptation
        self._adaptation_count += 1
        self._cooldown_remaining = self._cooldown
        self._last_adapted_threshold = candidate
        self._total_reduction = round(self._total_reduction + actual_reduction, 4)

        logger.info(
            "AdaptiveSubInferenceThreshold: sub_inference_vote_threshold reduced %.4f → %.4f "
            "(adaptation %d/%d, total_reduction=%.4f, vote_required %d→%d)",
            current_threshold, candidate,
            self._adaptation_count, self._max_adaptations,
            self._total_reduction, curr_required, cand_required,
        )
        return candidate

    # ------------------------------------------------------------------ #
    # Status                                                               #
    # ------------------------------------------------------------------ #

    def get_status(self) -> Dict[str, Any]:
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
            logger.error("AdaptiveSubInferenceThreshold.save_state failed: %s", exc)
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
            logger.error("AdaptiveSubInferenceThreshold._load_state failed: %s", exc)
