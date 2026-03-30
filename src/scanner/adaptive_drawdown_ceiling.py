"""Phase 64 US-382: AdaptiveDrawdownCeiling.

Auto-raise max_drawdown_pct when RiskGateAnalyzer classifies NEAR_THRESHOLD.
Mirrors AdaptiveMomentumFloor (Phase 62) but for the risk gate domain.

DIRECTION IS INVERTED vs confidence/momentum:
  - Confidence/Momentum: threshold goes DOWN to be more permissive
  - Drawdown ceiling:    threshold goes UP to be more permissive
  (higher max_drawdown_pct = more RF drawdown predictions allowed through)

Guard conditions are more conservative than confidence/momentum because
raising the drawdown ceiling directly increases portfolio risk exposure:
  step=0.001  (vs 0.02 for momentum — 20x smaller)
  cooldown=30  (vs 20 — 50% longer)
  max_adaptations=3  (vs 5 — 40% fewer)
  ceiling=0.040  (absolute cap at 4% drawdown threshold)

Trigger: RiskGateAnalyzer.classification == NEAR_THRESHOLD
  (gap_at_p50 <= 0.005 with >= 3 fail records — pairs barely over threshold)

State: trained_data/adaptive_drawdown_ceiling_state.json
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

DEFAULT_STATE_PATH = Path("trained_data/adaptive_drawdown_ceiling_state.json")
CLASS_NEAR_THRESHOLD = "NEAR_THRESHOLD"


class AdaptiveDrawdownCeiling:
    """Auto-raise max_drawdown_pct by step when risk gate is NEAR_THRESHOLD.

    Designed to unlock trades where RF drawdown predictions sit just barely
    over the max_drawdown_pct threshold — a situation where a small threshold
    increase resolves the block without structural model retraining.

    Usage:
        ceiling = AdaptiveDrawdownCeiling()
        # Called every 50 scans inside the Phase 63 periodic block:
        new_threshold = ceiling.tick(_rga_cls, float(self.config.max_drawdown_pct))
        if new_threshold is not None:
            self.config.max_drawdown_pct = new_threshold
    """

    def __init__(
        self,
        step: float = 0.001,
        cooldown: int = 30,
        max_adaptations: int = 3,
        ceiling: float = 0.040,
        state_path: Path = DEFAULT_STATE_PATH,
    ) -> None:
        self._step = float(step)
        self._cooldown = int(cooldown)
        self._max_adaptations = int(max_adaptations)
        self._ceiling = float(ceiling)
        self._state_path = Path(state_path)

        self._adaptation_count: int = 0
        self._cooldown_remaining: int = 0
        self._last_adapted_threshold: Optional[float] = None
        self._total_increase: float = 0.0

        self._load_state()

    # ------------------------------------------------------------------ #
    # Core tick logic                                                      #
    # ------------------------------------------------------------------ #

    def tick(
        self,
        classification: str,
        current_max_drawdown_pct: float,
    ) -> Optional[float]:
        """Evaluate whether to raise max_drawdown_pct.

        Cooldown decrements on every call (regardless of adaptation).
        Returns new threshold float if adaptation fires, else None.

        Cooldown semantics (decrement-before-check, mirrors Phase 62):
          After adaptation sets cooldown=30:
            tick+1: decrement→29, 29>0 → block
            ...
            tick+30: decrement→0, 0>0 → False → fires (30 ticks block)
        """
        # Decrement cooldown FIRST on every tick
        if self._cooldown_remaining > 0:
            self._cooldown_remaining -= 1

        # Guard 1: classification must be NEAR_THRESHOLD
        if classification != CLASS_NEAR_THRESHOLD:
            return None

        # Guard 2: cooldown must be exhausted
        if self._cooldown_remaining > 0:
            return None

        # Guard 3: max adaptations not exhausted
        if self._adaptation_count >= self._max_adaptations:
            return None

        # Guard 4: candidate must not exceed ceiling
        candidate = round(float(current_max_drawdown_pct) + self._step, 4)
        if candidate > self._ceiling:
            return None

        # All guards passed — fire adaptation
        self._adaptation_count += 1
        self._cooldown_remaining = self._cooldown
        self._last_adapted_threshold = candidate
        self._total_increase = round(self._total_increase + self._step, 4)

        logger.info(
            "AdaptiveDrawdownCeiling: adaptation %d/%d — "
            "max_drawdown_pct raised %.4f → %.4f (total_increase=%.4f)",
            self._adaptation_count, self._max_adaptations,
            float(current_max_drawdown_pct), candidate, self._total_increase,
        )
        return candidate

    # ------------------------------------------------------------------ #
    # Status                                                               #
    # ------------------------------------------------------------------ #

    def get_status(self) -> Dict[str, Any]:
        """Return current state for logging or external inspection."""
        return {
            "adaptation_count": self._adaptation_count,
            "cooldown_remaining": self._cooldown_remaining,
            "max_adaptations": self._max_adaptations,
            "ceiling": self._ceiling,
            "step": self._step,
            "total_increase": self._total_increase,
            "last_adapted_threshold": self._last_adapted_threshold,
        }

    # ------------------------------------------------------------------ #
    # Persistence                                                          #
    # ------------------------------------------------------------------ #

    def save_state(self, path: Optional[Path] = None) -> None:
        """Atomically persist state to JSON."""
        target = Path(path) if path else self._state_path
        target.parent.mkdir(parents=True, exist_ok=True)
        state = {
            "adaptation_count": self._adaptation_count,
            "cooldown_remaining": self._cooldown_remaining,
            "last_adapted_threshold": self._last_adapted_threshold,
            "total_increase": self._total_increase,
        }
        tmp_path: Optional[str] = None
        try:
            fd, tmp_path = tempfile.mkstemp(dir=str(target.parent), suffix=".tmp")
            with os.fdopen(fd, "w") as f:
                json.dump(state, f, indent=2, sort_keys=True)
            os.rename(tmp_path, str(target))
            tmp_path = None
        except Exception as exc:
            logger.error("AdaptiveDrawdownCeiling.save_state failed: %s", exc)
            if tmp_path is not None:
                try:
                    os.unlink(tmp_path)
                except Exception:
                    pass

    def _load_state(self) -> None:
        """Restore persisted state on init."""
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
            logger.error("AdaptiveDrawdownCeiling._load_state failed: %s", exc)
