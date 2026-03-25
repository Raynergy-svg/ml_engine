"""Regime Transition Tracker with Markov chain probabilities.

Tracks volatility regime transitions (LOW → NORMAL → HIGH → EXTREME)
as a first-order Markov chain. Uses observed transition frequencies to
predict next-regime probabilities for proactive risk adjustment.

Usage:
    tracker = RegimeTransitionTracker()
    tracker.update("HIGH")           # called each scan cycle
    probs = tracker.predict_next()   # {"LOW": 0.1, "NORMAL": 0.2, ...}
    mod = tracker.get_regime_risk_modifier()  # 0.85 if spike likely
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# Canonical regime order
REGIMES: List[str] = ["LOW", "NORMAL", "HIGH", "EXTREME"]
_REGIME_INDEX = {r: i for i, r in enumerate(REGIMES)}

# Default project root (same convention as config.py)
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
_DEFAULT_PATH = _PROJECT_ROOT / "trained_data" / "regime_transitions.json"


class RegimeTransitionTracker:
    """Tracks regime transitions and predicts next-state probabilities.

    Maintains a 4×4 transition count matrix persisted to JSON.
    Each call to update(regime) records the transition from previous → current.
    predict_next() returns the row-normalized probability distribution.
    """

    def __init__(
        self,
        path: Optional[Path] = None,
        spike_threshold: float = 0.30,
        spike_modifier: float = 0.85,
    ):
        """Initialize tracker.

        Args:
            path: Path to persist transition counts (default: trained_data/regime_transitions.json)
            spike_threshold: P(EXTREME|current) threshold to trigger risk reduction
            spike_modifier: Position size multiplier when spike risk detected
        """
        self.path = Path(path) if path else _DEFAULT_PATH
        self.spike_threshold = spike_threshold
        self.spike_modifier = spike_modifier

        # 4×4 transition count matrix [from_regime][to_regime]
        self._counts: List[List[int]] = [[0] * 4 for _ in range(4)]
        self._current_regime: Optional[str] = None
        self._total_updates: int = 0

        self._load()

    def _load(self) -> None:
        """Load persisted transition counts."""
        try:
            from src.scanner.automation.safe_json import safe_json_read
            data = safe_json_read(self.path, default=None)
        except ImportError:
            import json
            data = None
            if self.path.exists():
                try:
                    data = json.loads(self.path.read_text(encoding="utf-8"))
                except Exception:
                    pass

        if data and isinstance(data, dict):
            matrix = data.get("transition_counts")
            if matrix and len(matrix) == 4 and all(len(row) == 4 for row in matrix):
                self._counts = [[int(c) for c in row] for row in matrix]
            self._current_regime = data.get("current_regime")
            self._total_updates = int(data.get("total_updates", 0))
            logger.debug(
                f"Regime tracker loaded: {self._total_updates} updates, "
                f"current regime: {self._current_regime}"
            )

    def _save(self) -> None:
        """Persist transition counts to JSON."""
        data = {
            "transition_counts": self._counts,
            "current_regime": self._current_regime,
            "total_updates": self._total_updates,
            "regimes": REGIMES,
        }
        try:
            from src.scanner.automation.safe_json import safe_json_write
            safe_json_write(self.path, data)
        except ImportError:
            import json
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_text(json.dumps(data, indent=2), encoding="utf-8")

    def update(self, regime: str) -> None:
        """Record a regime observation and update transition matrix.

        Args:
            regime: Current volatility regime (LOW/NORMAL/HIGH/EXTREME)
        """
        regime = str(regime).upper().strip()
        if regime not in _REGIME_INDEX:
            logger.warning(f"Unknown regime '{regime}', skipping transition update")
            return

        if self._current_regime is not None and self._current_regime in _REGIME_INDEX:
            from_idx = _REGIME_INDEX[self._current_regime]
            to_idx = _REGIME_INDEX[regime]
            self._counts[from_idx][to_idx] += 1

        self._current_regime = regime
        self._total_updates += 1

        # Persist every 5 updates to avoid excessive writes
        if self._total_updates % 5 == 0:
            self._save()

    def predict_next(self) -> Dict[str, float]:
        """Predict next regime probabilities from current state.

        Returns:
            Dict mapping regime name → probability (sums to 1.0).
            Returns uniform distribution if no data for current state.
        """
        if self._current_regime is None or self._current_regime not in _REGIME_INDEX:
            # Uniform prior
            return {r: 0.25 for r in REGIMES}

        row_idx = _REGIME_INDEX[self._current_regime]
        row = self._counts[row_idx]
        total = sum(row)

        if total == 0:
            # No observations from this state — uniform prior
            return {r: 0.25 for r in REGIMES}

        return {REGIMES[i]: row[i] / total for i in range(4)}

    def get_regime_risk_modifier(self) -> float:
        """Get position size modifier based on regime spike risk.

        Returns:
            1.0 if no spike risk, self.spike_modifier (default 0.85) if
            probability of transitioning to EXTREME exceeds threshold.
        """
        probs = self.predict_next()
        p_extreme = probs.get("EXTREME", 0.0)

        if p_extreme >= self.spike_threshold:
            logger.info(
                f"Regime spike risk: P(EXTREME|{self._current_regime}) = {p_extreme:.2f} "
                f"≥ {self.spike_threshold} → applying {self.spike_modifier}x size modifier"
            )
            return self.spike_modifier

        return 1.0

    def get_predictive_size_modifier(self) -> float:
        """Get forward-looking position size modifier from Markov predictions.

        Uses predicted regime probabilities to proactively reduce position size
        when HIGH or EXTREME volatility is likely, even before it arrives.

        Returns:
            Float 0.5-1.0:
            - P(HIGH) > 0.3 → 0.85 modifier
            - P(EXTREME) > 0.3 → 0.70 modifier
            - Both compound: 0.85 * 0.70 = 0.595
            - Otherwise 1.0 (no reduction)
        """
        probs = self.predict_next()
        modifier = 1.0

        p_high = probs.get("HIGH", 0.0)
        p_extreme = probs.get("EXTREME", 0.0)

        if p_high > 0.3:
            modifier *= 0.85
            logger.info(
                f"Predictive sizing: P(HIGH|{self._current_regime}) = {p_high:.2f} > 0.3 "
                f"→ 0.85x modifier"
            )

        if p_extreme > 0.3:
            modifier *= 0.70
            logger.info(
                f"Predictive sizing: P(EXTREME|{self._current_regime}) = {p_extreme:.2f} > 0.3 "
                f"→ 0.70x modifier"
            )

        return round(max(0.5, modifier), 4)

    def get_status(self) -> Dict[str, Any]:
        """Get tracker status for observability."""
        probs = self.predict_next()
        return {
            "current_regime": self._current_regime,
            "total_updates": self._total_updates,
            "next_regime_probs": probs,
            "spike_risk": probs.get("EXTREME", 0.0) >= self.spike_threshold,
            "risk_modifier": self.get_regime_risk_modifier(),
            "predictive_size_modifier": self.get_predictive_size_modifier(),
        }

    def flush(self) -> None:
        """Force-save current state (call on clean shutdown)."""
        self._save()
