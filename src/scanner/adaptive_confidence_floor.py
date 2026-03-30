"""Phase 58 US-361: AdaptiveConfidenceFloor.

Conservative auto-reduction of min_confidence when the system is stalled
AND ThresholdGapAnalyzer + GatePassPredictor confirm signals exist but
fall just below threshold.

Safety constraints:
  - Step: -1.0 pt per adaptation (small, conservative)
  - Cooldown: COOLDOWN_SCANS=20 must elapse between adaptations
  - Max: MAX_TOTAL_ADAPTATIONS=5 — after 5 reductions, requires manual review
  - Floor: SAFETY_FLOOR=45.0 — never lower below this (5 pts above penalty ceiling)
  - Stall guard: only adapts when scans_since_last_trade >= 50

This is a last-resort mechanism. Phase 57 penalty fixes are given priority.
The floor activates only when: penalties are NOT the root cause AND the
gap analysis confirms the threshold is structurally too high.
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

DEFAULT_STATE_PATH = Path("trained_data/adaptive_floor_state.json")
SAFETY_FLOOR = 45.0
ADAPTATION_STEP = 1.0
COOLDOWN_SCANS = 20
MAX_TOTAL_ADAPTATIONS = 5
STALL_THRESHOLD = 50


@dataclass
class AdaptiveFloorState:
    """Persisted state for AdaptiveConfidenceFloor."""
    current_threshold: float
    total_adaptations: int
    cooldown_remaining: int
    last_adaptation_ts: Optional[str]
    version: str = "1.0"


class AdaptiveConfidenceFloor:
    """Conservative threshold reducer for stall recovery.

    Lifecycle:
      - tick() called every scan cycle (decrements cooldown)
      - should_adapt() called when stalled flag fires
      - apply_adjustment() mutates config.min_confidence if should_adapt()
      - get_state() exposes current status for monitoring
    """

    def __init__(
        self,
        safety_floor: float = SAFETY_FLOOR,
        adaptation_step: float = ADAPTATION_STEP,
        cooldown_scans: int = COOLDOWN_SCANS,
        max_total_adaptations: int = MAX_TOTAL_ADAPTATIONS,
        stall_threshold: int = STALL_THRESHOLD,
        state_path: Path = DEFAULT_STATE_PATH,
        initial_threshold: float = 58.0,
    ) -> None:
        self._safety_floor = safety_floor
        self._adaptation_step = adaptation_step
        self._cooldown_scans = cooldown_scans
        self._max_total_adaptations = max_total_adaptations
        self._stall_threshold = stall_threshold
        self._state_path = Path(state_path)

        # Internal state
        self._current_threshold = initial_threshold
        self._total_adaptations = 0
        self._cooldown_remaining = 0
        self._last_adaptation_ts: Optional[str] = None

        self._load_state()

    # ------------------------------------------------------------------ #
    # Per-scan tick                                                        #
    # ------------------------------------------------------------------ #

    def tick(self) -> None:
        """Decrement cooldown counter. Call once per scan cycle."""
        if self._cooldown_remaining > 0:
            self._cooldown_remaining -= 1

    # ------------------------------------------------------------------ #
    # Adaptation decision                                                  #
    # ------------------------------------------------------------------ #

    def should_adapt(
        self,
        gap_report: object,       # ThresholdGapReport
        predictor_report: Dict[str, Any],
        scans_since_last_trade: int,
    ) -> bool:
        """Return True when all conditions justify a threshold reduction.

        Conditions:
          1. System stalled (scans_since_last_trade >= stall_threshold)
          2. Cooldown elapsed (cooldown_remaining == 0)
          3. Under max adaptations cap
          4. Gap analyzer says threshold is structurally too high
          5. Predictor says signals exist (penalties alone not sufficient to explain stall)
             OR gap analyzer alone is sufficient when predictor has no data

        Note: If predictor says signal_exists=True, penalties are already the issue
        and Phase 57 fixes should handle it. We only lower threshold when
        signals genuinely don't pass even under penalty-free conditions.
        """
        if scans_since_last_trade < self._stall_threshold:
            return False

        if self._cooldown_remaining > 0:
            return False

        if self._total_adaptations >= self._max_total_adaptations:
            logger.warning(
                "AdaptiveConfidenceFloor: max adaptations (%d) reached — "
                "manual review required",
                self._max_total_adaptations,
            )
            return False

        # Gap report must recommend threshold reduction
        rec = getattr(gap_report, "recommendation", None)
        if rec not in ("THRESHOLD_TOO_HIGH", "NEAR_THRESHOLD"):
            return False

        # Only adapt when predictor confirms signals don't pass even penalty-free
        # (if signals pass penalty-free → Phase 57 is the fix, not threshold)
        signal_exists = predictor_report.get("signal_exists", False)
        if signal_exists:
            # Signals exist penalty-free → Phase 57 remediation is sufficient
            # Don't lower threshold; let penalty fixes work first
            return False

        return True

    def compute_adjustment(
        self,
        gap_report: object,
        predictor_report: Dict[str, Any],
    ) -> float:
        """Return the signed delta to apply to min_confidence.

        Returns negative float (e.g. -1.0). Always respects safety_floor.
        """
        proposed_new = self._current_threshold - self._adaptation_step
        if proposed_new < self._safety_floor:
            # Would breach floor — clamp to floor
            delta = -(self._current_threshold - self._safety_floor)
            if delta == 0.0:
                return 0.0
            return round(delta, 4)
        return -self._adaptation_step

    def apply_adjustment(
        self,
        config: object,
        delta: float,
    ) -> None:
        """Mutate config.min_confidence and persist state.

        Args:
            config: ScannerConfig instance with min_confidence attribute.
            delta: Signed delta (negative = reduction).
        """
        old_threshold = float(getattr(config, "min_confidence", self._current_threshold))
        new_threshold = max(old_threshold + delta, self._safety_floor)

        logger.warning(
            "AdaptiveConfidenceFloor: min_confidence reduced from %.1f to %.1f "
            "(delta=%.2f, adaptation=%d/%d)",
            old_threshold, new_threshold, delta,
            self._total_adaptations + 1, self._max_total_adaptations,
        )

        setattr(config, "min_confidence", new_threshold)
        self._current_threshold = new_threshold
        self._total_adaptations += 1
        self._cooldown_remaining = self._cooldown_scans
        self._last_adaptation_ts = datetime.now(timezone.utc).isoformat()
        self.save_state()

    # ------------------------------------------------------------------ #
    # Status                                                               #
    # ------------------------------------------------------------------ #

    def get_state(self) -> Dict[str, Any]:
        return {
            "current_threshold": self._current_threshold,
            "total_adaptations": self._total_adaptations,
            "cooldown_remaining": self._cooldown_remaining,
            "last_adaptation_ts": self._last_adaptation_ts,
            "max_adaptations_reached": (
                self._total_adaptations >= self._max_total_adaptations
            ),
            "safety_floor": self._safety_floor,
        }

    # ------------------------------------------------------------------ #
    # Persistence                                                          #
    # ------------------------------------------------------------------ #

    def save_state(self) -> None:
        """Atomically persist internal state."""
        self._state_path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "version": "1.0",
            "current_threshold": self._current_threshold,
            "total_adaptations": self._total_adaptations,
            "cooldown_remaining": self._cooldown_remaining,
            "last_adaptation_ts": self._last_adaptation_ts,
            "last_updated": datetime.now(timezone.utc).isoformat(),
        }
        try:
            fd, tmp_path = tempfile.mkstemp(
                dir=str(self._state_path.parent), suffix=".tmp"
            )
            with os.fdopen(fd, "w") as f:
                json.dump(data, f, indent=2, sort_keys=True)
            os.rename(tmp_path, str(self._state_path))
        except Exception as exc:
            logger.error("AdaptiveConfidenceFloor.save_state failed: %s", exc)
            try:
                os.unlink(tmp_path)
            except Exception:
                pass

    def _load_state(self) -> None:
        if not self._state_path.exists():
            return
        try:
            with open(self._state_path) as f:
                data = json.load(f)
            self._current_threshold = float(data.get("current_threshold", self._current_threshold))
            self._total_adaptations = int(data.get("total_adaptations", 0))
            self._cooldown_remaining = int(data.get("cooldown_remaining", 0))
            self._last_adaptation_ts = data.get("last_adaptation_ts")
            logger.debug(
                "AdaptiveConfidenceFloor: loaded state — threshold=%.1f "
                "adaptations=%d cooldown=%d",
                self._current_threshold, self._total_adaptations, self._cooldown_remaining,
            )
        except Exception as exc:
            logger.error("AdaptiveConfidenceFloor._load_state failed: %s", exc)
