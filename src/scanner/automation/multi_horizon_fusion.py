"""Multi-horizon signal fusion with Bayesian model averaging.

US-081: Fuse signals across 4 timeframes (1m, 5m, 1h, 4h) using
Bayesian model averaging. Each timeframe gets a prior weight that
updates based on recent accuracy. Agreement across timeframes
produces an A/B/C grade that modifies trade confidence.

Grade system:
- Grade A: All 4 timeframes agree → +10% confidence boost
- Grade B: 3 of 4 agree → no change
- Grade C: 2 or fewer agree → -15% confidence penalty

Bayesian weights: prior proportional to inverse error rate,
updated every 50 trades with exponential decay on old observations.
"""

from __future__ import annotations

import json
import logging
import math
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
_FUSION_PATH = _PROJECT_ROOT / "trained_data" / "multi_horizon_weights.json"

# Supported timeframes
TIMEFRAMES = ["M1", "M5", "H1", "H4"]

# Default prior weights (equal)
DEFAULT_PRIORS: Dict[str, float] = {
    "M1": 0.15,   # 1-minute: noisy, low weight
    "M5": 0.25,   # 5-minute: primary scanning TF
    "H1": 0.35,   # 1-hour: strong trend signal
    "H4": 0.25,   # 4-hour: macro trend
}

# Confidence grade adjustments
GRADE_ADJUSTMENTS: Dict[str, float] = {
    "A": 0.10,    # +10% boost
    "B": 0.00,    # No change
    "C": -0.15,   # -15% penalty
}

# Update interval for Bayesian weight updates
BAYESIAN_UPDATE_INTERVAL = 50


@dataclass
class TimeframeSignal:
    """Signal from a single timeframe."""

    timeframe: str = ""
    direction: str = "HOLD"  # LONG, SHORT, HOLD
    confidence: float = 0.0
    features: Dict[str, float] = field(default_factory=dict)
    timestamp: str = ""

    def to_dict(self) -> dict:
        return {
            "timeframe": self.timeframe,
            "direction": self.direction,
            "confidence": round(self.confidence, 4),
            "features": {k: round(v, 4) for k, v in self.features.items()},
            "timestamp": self.timestamp,
        }


@dataclass
class FusionResult:
    """Result of multi-horizon signal fusion."""

    fused_direction: str = "HOLD"
    fused_confidence: float = 0.0
    agreement_grade: str = "C"  # A, B, or C
    agreement_count: int = 0  # How many TFs agree with fused direction
    total_signals: int = 0
    confidence_adjustment: float = 0.0
    timeframe_weights: Dict[str, float] = field(default_factory=dict)
    signal_details: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "fused_direction": self.fused_direction,
            "fused_confidence": round(self.fused_confidence, 4),
            "agreement_grade": self.agreement_grade,
            "agreement_count": self.agreement_count,
            "total_signals": self.total_signals,
            "confidence_adjustment": round(self.confidence_adjustment, 4),
            "timeframe_weights": {k: round(v, 4) for k, v in self.timeframe_weights.items()},
            "signal_details": self.signal_details,
        }


class MultiHorizonFusion:
    """Multi-horizon signal fusion with Bayesian model averaging.

    Pipeline:
    1. Collect signals from multiple timeframes
    2. Weight each signal by Bayesian prior (updated by historical accuracy)
    3. Compute weighted directional consensus
    4. Grade agreement (A/B/C)
    5. Apply confidence adjustment based on grade
    """

    def __init__(
        self,
        priors: Optional[Dict[str, float]] = None,
        persistence_path: Optional[Path] = None,
        update_interval: int = BAYESIAN_UPDATE_INTERVAL,
    ):
        self._priors = priors or dict(DEFAULT_PRIORS)
        self.persistence_path = persistence_path or _FUSION_PATH
        self.update_interval = update_interval
        self._signals: Dict[str, TimeframeSignal] = {}
        self._accuracy_history: Dict[str, List[int]] = defaultdict(list)
        self._trades_since_update: int = 0

        # Load persisted weights
        self._load_weights()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def add_signal(
        self,
        timeframe: str,
        direction: str,
        confidence: float,
        features: Optional[Dict[str, float]] = None,
    ) -> None:
        """Register a signal from a specific timeframe.

        Args:
            timeframe: Timeframe code (M1, M5, H1, H4)
            direction: Signal direction (LONG, SHORT, HOLD)
            confidence: Signal confidence (0.0-1.0)
            features: Optional feature dict for this timeframe
        """
        tf = timeframe.upper()
        self._signals[tf] = TimeframeSignal(
            timeframe=tf,
            direction=direction.upper(),
            confidence=max(0.0, min(1.0, confidence)),
            features=features or {},
            timestamp=datetime.now(timezone.utc).isoformat(),
        )

    def clear_signals(self) -> None:
        """Clear all registered signals for a new scan cycle."""
        self._signals.clear()

    def fuse(self) -> FusionResult:
        """Fuse all registered timeframe signals into a single decision.

        Returns:
            FusionResult with fused direction, confidence, and grade
        """
        result = FusionResult(total_signals=len(self._signals))

        if not self._signals:
            return result

        # Normalize weights for available timeframes
        available_tfs = list(self._signals.keys())
        raw_weights = {tf: self._priors.get(tf, 0.2) for tf in available_tfs}
        total_weight = sum(raw_weights.values())
        if total_weight <= 0:
            total_weight = 1.0
        weights = {tf: w / total_weight for tf, w in raw_weights.items()}
        result.timeframe_weights = weights

        # Compute weighted directional scores
        long_score = 0.0
        short_score = 0.0
        hold_score = 0.0

        for tf, signal in self._signals.items():
            w = weights.get(tf, 0.0)
            weighted_conf = w * signal.confidence

            if signal.direction == "LONG":
                long_score += weighted_conf
            elif signal.direction == "SHORT":
                short_score += weighted_conf
            else:
                hold_score += weighted_conf

            result.signal_details.append(signal.to_dict())

        # Determine fused direction
        scores = {"LONG": long_score, "SHORT": short_score, "HOLD": hold_score}
        fused_dir = max(scores, key=scores.get)
        fused_conf = scores[fused_dir]
        result.fused_direction = fused_dir
        result.fused_confidence = min(1.0, fused_conf)

        # Count agreement
        if fused_dir != "HOLD":
            agree_count = sum(
                1 for s in self._signals.values()
                if s.direction == fused_dir
            )
        else:
            agree_count = sum(
                1 for s in self._signals.values()
                if s.direction == "HOLD"
            )
        result.agreement_count = agree_count

        # Grade agreement
        n = len(self._signals)
        if n >= 4 and agree_count >= 4:
            result.agreement_grade = "A"
        elif n >= 3 and agree_count >= 3:
            result.agreement_grade = "B" if n > 3 else "A"
        elif n >= 3 and agree_count >= 2:
            result.agreement_grade = "B"
        else:
            result.agreement_grade = "C"

        # Special case: only 2 signals
        if n == 2:
            result.agreement_grade = "A" if agree_count == 2 else "C"

        # Apply confidence adjustment
        result.confidence_adjustment = GRADE_ADJUSTMENTS.get(
            result.agreement_grade, 0.0
        )

        logger.info(
            f"MultiHorizonFusion: {n} TFs → {fused_dir} "
            f"(conf={fused_conf:.3f}, grade={result.agreement_grade}, "
            f"agree={agree_count}/{n})"
        )

        return result

    def record_outcome(
        self,
        timeframe: str,
        was_correct: bool,
    ) -> None:
        """Record a trade outcome for Bayesian weight updates.

        Args:
            timeframe: Which timeframe's signal to update
            was_correct: Whether the signal predicted correctly
        """
        tf = timeframe.upper()
        self._accuracy_history[tf].append(1 if was_correct else 0)
        self._trades_since_update += 1

        # Check if time to update weights
        if self._trades_since_update >= self.update_interval:
            self._update_bayesian_weights()
            self._trades_since_update = 0

    # ------------------------------------------------------------------
    # Bayesian weight updates
    # ------------------------------------------------------------------

    def _update_bayesian_weights(self) -> None:
        """Update timeframe priors based on historical accuracy.

        Uses inverse error rate as the prior: better-performing
        timeframes get higher weight.
        """
        if not self._accuracy_history:
            return

        new_weights: Dict[str, float] = {}
        for tf in TIMEFRAMES:
            history = self._accuracy_history.get(tf, [])
            if len(history) < 5:
                # Not enough data, keep default prior
                new_weights[tf] = self._priors.get(tf, 0.25)
                continue

            # Use recent window with exponential decay
            recent = history[-200:]  # Cap at 200 observations
            n = len(recent)
            # Exponential weights: newer observations matter more
            decay = np.exp(np.linspace(-2.0, 0.0, n))
            weighted_accuracy = float(np.average(recent, weights=decay))

            # Inverse error rate as prior (avoid division by zero)
            error_rate = max(1.0 - weighted_accuracy, 0.05)
            new_weights[tf] = 1.0 / error_rate

        # Normalize
        total = sum(new_weights.values())
        if total > 0:
            self._priors = {tf: w / total for tf, w in new_weights.items()}

        # Persist
        self._save_weights()

        logger.info(
            f"MultiHorizonFusion: Updated Bayesian weights: "
            f"{', '.join(f'{tf}={w:.3f}' for tf, w in self._priors.items())}"
        )

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _save_weights(self) -> None:
        """Persist weights and accuracy history."""
        try:
            data = {
                "version": 1,
                "priors": {k: round(v, 6) for k, v in self._priors.items()},
                "accuracy_history": {
                    tf: hist[-200:]  # Keep last 200
                    for tf, hist in self._accuracy_history.items()
                },
                "last_updated": datetime.now(timezone.utc).isoformat(),
            }
            self.persistence_path.parent.mkdir(parents=True, exist_ok=True)
            try:
                from src.scanner.automation.safe_json import safe_json_write
                safe_json_write(self.persistence_path, data)
            except ImportError:
                self.persistence_path.write_text(json.dumps(data, indent=2))
        except Exception as e:
            logger.debug(f"MultiHorizonFusion: Failed to save weights: {e}")

    def _load_weights(self) -> None:
        """Load persisted weights."""
        if not self.persistence_path.exists():
            return
        try:
            try:
                from src.scanner.automation.safe_json import safe_json_read
                data = safe_json_read(self.persistence_path, default={})
            except ImportError:
                data = json.loads(self.persistence_path.read_text())

            priors = data.get("priors", {})
            if priors:
                self._priors = priors
                logger.info(
                    f"MultiHorizonFusion: Loaded weights: "
                    f"{', '.join(f'{tf}={w:.3f}' for tf, w in self._priors.items())}"
                )

            history = data.get("accuracy_history", {})
            for tf, hist in history.items():
                self._accuracy_history[tf] = hist

        except Exception as e:
            logger.debug(f"MultiHorizonFusion: Failed to load weights: {e}")
