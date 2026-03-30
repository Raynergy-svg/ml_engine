"""Phase 75: AdaptiveDisagreementFloor (quantization-aware).

When most blocked trades have disagreement just above the hard floor,
raise disagreement_hard_floor to the next quantized boundary to unlock trades.

model_disagreement with 5 heuristics (Phase 76) is quantized: 0.0, 0.2, 0.4, 0.6, 0.8, 1.0.
The default hard_floor=0.30 means 1/5 disagreement (0.20) passes, 2/5 (0.40) blocks.

QUANTIZATION-AWARE (Phase 72 pattern):
  With N heuristics, the only meaningful boundaries are k/N for k in 0..N.
  Linear stepping (0.05) wastes adaptation budget crossing the same boundary
  or landing between boundaries with zero behavioral effect.
  Instead, compute the next quantized boundary and jump directly to it + epsilon.

  Example with N=5, floor=0.30:
    Boundary values: 0.000, 0.200, 0.400, 0.600, 0.800, 1.000
    Step 1: 0.30 → 0.41 (just above 0.400 — allows 2/5 disagreement)
    Step 2: 0.41 → 0.61 (just above 0.600 — allows 3/5 disagreement)
    Each step changes behavior. No wasted adaptations.

IMPORTANT: This is a safety gate. Adaptations are conservative:
  ceiling = 0.50 (never above half the agents disagreeing — blocks step 2 by default)
  max_adaptations = 2
  cooldown = 50 (wait 50 scans between adaptations)

Direction: threshold goes UP (more permissive), opposite of confidence/momentum floors.
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

CLASS_NEAR_THRESHOLD = "NEAR_THRESHOLD"

DEFAULT_COOLDOWN = 50
DEFAULT_MAX_ADAPTATIONS = 2
DEFAULT_FLOOR = 0.30       # Never below original hard floor
DEFAULT_CEILING = 0.50     # Never allow more than half disagreeing
DEFAULT_NUM_HEURISTICS = 5  # Number of heuristics in uncertainty agent (Phase 76: expanded from 3)
DEFAULT_BOUNDARY_EPSILON = 0.01  # Offset above quantized boundary
DEFAULT_STATE_PATH = Path("trained_data/adaptive_disagreement_floor_state.json")


def _compute_quantized_boundaries(num_heuristics: int) -> list[float]:
    """Return sorted list of all possible disagreement values for N heuristics."""
    return [k / num_heuristics for k in range(num_heuristics + 1)]


def _next_boundary_step(current_floor: float, boundaries: list[float], epsilon: float) -> Optional[float]:
    """Find the next quantized boundary above current_floor and return boundary + epsilon.

    If current_floor is already above a boundary (e.g. 0.34 > 0.333),
    find the NEXT boundary above current_floor (0.667) and return 0.667 + epsilon.
    Returns None if no higher boundary exists.
    """
    for b in boundaries:
        if b > current_floor + 1e-9:
            return round(b + epsilon, 4)
    return None


class AdaptiveDisagreementFloor:
    """Auto-raise disagreement_hard_floor when NEAR_THRESHOLD (quantization-aware).

    Instead of fixed linear steps, jumps directly to the next quantized boundary
    where behavior actually changes. With 3 heuristics, boundaries are at
    0.000, 0.333, 0.667, 1.000. Each adaptation crosses exactly one boundary.

    Usage:
        adf = AdaptiveDisagreementFloor()
        new_floor = adf.tick(classification, current_floor)
        if new_floor is not None:
            config.disagreement_hard_floor = new_floor
    """

    def __init__(
        self,
        cooldown: int = DEFAULT_COOLDOWN,
        max_adaptations: int = DEFAULT_MAX_ADAPTATIONS,
        floor: float = DEFAULT_FLOOR,
        ceiling: float = DEFAULT_CEILING,
        num_heuristics: int = DEFAULT_NUM_HEURISTICS,
        boundary_epsilon: float = DEFAULT_BOUNDARY_EPSILON,
        state_path: Path = DEFAULT_STATE_PATH,
        # Legacy compat: accept and ignore step= kwarg
        step: float = 0.0,
    ) -> None:
        self._cooldown = int(cooldown)
        self._max_adaptations = int(max_adaptations)
        self._floor = float(floor)
        self._ceiling = float(ceiling)
        self._num_heuristics = int(num_heuristics)
        self._boundary_epsilon = float(boundary_epsilon)
        self._boundaries = _compute_quantized_boundaries(self._num_heuristics)
        self._state_path = Path(state_path)

        self._adaptation_count: int = 0
        self._cooldown_remaining: int = 0
        self._last_adapted_threshold: Optional[float] = None
        self._total_increase: float = 0.0

        self._load_state()

    def tick(self, classification: str, current_floor: float) -> Optional[float]:
        """Evaluate one scan cycle (quantization-aware).

        Jumps directly to next quantized boundary + epsilon instead of fixed steps.
        Returns new threshold if adaptation fires, else None.
        Direction: UP (more permissive).
        """
        if self._cooldown_remaining > 0:
            self._cooldown_remaining -= 1

        if classification != CLASS_NEAR_THRESHOLD:
            return None

        if self._cooldown_remaining > 0:
            return None

        if self._adaptation_count >= self._max_adaptations:
            return None

        candidate = _next_boundary_step(
            float(current_floor), self._boundaries, self._boundary_epsilon,
        )
        if candidate is None:
            logger.debug(
                "AdaptiveDisagreementFloor: no higher boundary exists above %.4f",
                current_floor,
            )
            return None

        if candidate > self._ceiling:
            logger.debug(
                "AdaptiveDisagreementFloor: candidate=%.4f (boundary+eps) > ceiling=%.4f — blocked",
                candidate, self._ceiling,
            )
            return None

        actual_step = round(candidate - float(current_floor), 4)
        self._adaptation_count += 1
        self._cooldown_remaining = self._cooldown
        self._last_adapted_threshold = candidate
        self._total_increase = round(self._total_increase + actual_step, 4)

        logger.info(
            "AdaptiveDisagreementFloor: disagreement_hard_floor raised %.4f → %.4f "
            "(crossed boundary at %.4f, adaptation %d/%d, total_increase=%.4f)",
            current_floor, candidate,
            candidate - self._boundary_epsilon,
            self._adaptation_count, self._max_adaptations,
            self._total_increase,
        )
        return candidate

    def get_status(self) -> Dict[str, Any]:
        return {
            "adaptation_count": self._adaptation_count,
            "cooldown_remaining": self._cooldown_remaining,
            "max_adaptations": self._max_adaptations,
            "floor": self._floor,
            "ceiling": self._ceiling,
            "num_heuristics": self._num_heuristics,
            "boundaries": [round(b, 4) for b in self._boundaries],
            "boundary_epsilon": self._boundary_epsilon,
            "total_increase": self._total_increase,
            "last_adapted_threshold": self._last_adapted_threshold,
        }

    def save_state(self, path: Optional[Path] = None) -> None:
        target = Path(path) if path else self._state_path
        target.parent.mkdir(parents=True, exist_ok=True)
        state = {
            "adaptation_count": self._adaptation_count,
            "cooldown_remaining": self._cooldown_remaining,
            "last_adapted_threshold": self._last_adapted_threshold,
            "total_increase": self._total_increase,
        }
        try:
            fd, tmp = tempfile.mkstemp(dir=str(target.parent), suffix=".tmp")
            with os.fdopen(fd, "w") as f:
                json.dump(state, f, indent=2, sort_keys=True)
            os.rename(tmp, str(target))
        except Exception as exc:
            logger.error("AdaptiveDisagreementFloor.save_state failed: %s", exc)
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
            self._total_increase = float(state.get("total_increase", 0.0))
        except Exception as exc:
            logger.error("AdaptiveDisagreementFloor._load_state failed: %s", exc)
