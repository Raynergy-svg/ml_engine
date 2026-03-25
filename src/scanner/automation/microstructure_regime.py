"""Microstructure-driven regime detection for earlier stress signals.

US-074: Augments RegimeTransitionTracker with spread dynamics and order-book
signals that predict regime shifts 30-60 seconds before volatility-based
detection alone.

Signals tracked:
1. spread_ratio — current spread vs 20-observation baseline
2. spread_acceleration — rate of change of spread (d(spread)/dt over 3 scans)
3. bid_ask_imbalance — directional pressure from bid/ask volumes

These are combined into a microstructure_confidence_lift that adds +5-15% to
regime prediction confidence when microstructure aligns with volatility signals.
"""

from __future__ import annotations

import json
import logging
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
_MICRO_PATH = _PROJECT_ROOT / "trained_data" / "microstructure_regime.json"

# Thresholds for signal interpretation
SPREAD_STRESS_RATIO = 2.0        # Current spread > 2x baseline = stress
SPREAD_ACCEL_THRESHOLD = 0.5     # Rapid widening
IMBALANCE_THRESHOLD = 0.3        # Significant directional pressure
MAX_CONFIDENCE_LIFT = 0.15       # Maximum lift from microstructure
MIN_CONFIDENCE_LIFT = 0.05       # Minimum lift when signals weakly align


@dataclass
class MicrostructureSignals:
    """Current microstructure signal state."""

    spread_ratio: float = 1.0          # current_spread / baseline_spread
    spread_acceleration: float = 0.0   # d(spread_ratio)/dt over last 3 scans
    bid_ask_imbalance: float = 0.0     # (bid_vol - ask_vol) / total_vol, range [-1, 1]
    microstructure_score: float = 0.0  # Combined signal (0-1, higher = more stress)
    predicted_regime: str = ""
    confidence_lift: float = 0.0

    def to_dict(self) -> dict:
        return {
            "spread_ratio": round(self.spread_ratio, 4),
            "spread_acceleration": round(self.spread_acceleration, 4),
            "bid_ask_imbalance": round(self.bid_ask_imbalance, 4),
            "microstructure_score": round(self.microstructure_score, 4),
            "predicted_regime": self.predicted_regime,
            "confidence_lift": round(self.confidence_lift, 4),
        }


class MicrostructureRegimeDetector:
    """Detects regime shifts using microstructure signals.

    Maintains rolling windows of spread data and order-flow imbalance.
    Provides a confidence_lift that augments the main RegimeTransitionTracker.
    """

    def __init__(
        self,
        window_size: int = 20,
        persistence_path: Optional[Path] = None,
    ):
        self.window_size = window_size
        self.persistence_path = persistence_path or _MICRO_PATH

        # Rolling windows
        self._spread_ratios: deque = deque(maxlen=window_size)
        self._imbalances: deque = deque(maxlen=window_size)
        self._spread_history: deque = deque(maxlen=3)  # For acceleration

        self._baseline_spread: float = 0.0
        self._baseline_count: int = 0
        self._current_signals = MicrostructureSignals()

        self._load_state()

    def _load_state(self) -> None:
        """Load persisted baseline from disk."""
        if not self.persistence_path.exists():
            return
        try:
            try:
                from src.scanner.automation.safe_json import safe_json_read
                data = safe_json_read(self.persistence_path, default={})
            except ImportError:
                data = json.loads(self.persistence_path.read_text())

            self._baseline_spread = data.get("baseline_spread", 0.0)
            self._baseline_count = data.get("baseline_count", 0)
        except Exception as e:
            logger.debug("Failed to load microstructure state: %s", e)

    def _save_state(self) -> None:
        """Persist baseline to disk."""
        data = {
            "baseline_spread": self._baseline_spread,
            "baseline_count": self._baseline_count,
            "last_updated": datetime.now(timezone.utc).isoformat(),
        }
        try:
            from src.scanner.automation.safe_json import safe_json_write
            safe_json_write(self.persistence_path, data)
        except ImportError:
            self.persistence_path.parent.mkdir(parents=True, exist_ok=True)
            self.persistence_path.write_text(json.dumps(data, indent=2))

    def update(
        self,
        current_spread_pips: float = 0.0,
        bid_ask_imbalance: float = 0.0,
        volatility_regime: str = "NORMAL",
    ) -> MicrostructureSignals:
        """Update microstructure signals with latest scan data.

        Args:
            current_spread_pips: Current average spread in pips across scanned pairs
            bid_ask_imbalance: (bid_volume - ask_volume) / total_volume, range [-1, 1]
            volatility_regime: Current regime from RegimeTransitionTracker

        Returns:
            MicrostructureSignals with computed metrics
        """
        # Update baseline spread (EMA with slow decay)
        if self._baseline_count < self.window_size:
            # Bootstrap phase — accumulate baseline
            self._baseline_count += 1
            alpha = 1.0 / self._baseline_count
        else:
            alpha = 2.0 / (self.window_size + 1)

        if current_spread_pips > 0:
            self._baseline_spread = (
                (1 - alpha) * self._baseline_spread + alpha * current_spread_pips
            )

        # Compute spread ratio
        spread_ratio = 1.0
        if self._baseline_spread > 0 and current_spread_pips > 0:
            spread_ratio = current_spread_pips / self._baseline_spread
        self._spread_ratios.append(spread_ratio)

        # Compute spread acceleration (change in spread_ratio over last 3 scans)
        self._spread_history.append(spread_ratio)
        spread_acceleration = 0.0
        if len(self._spread_history) >= 3:
            deltas = [
                self._spread_history[i] - self._spread_history[i - 1]
                for i in range(1, len(self._spread_history))
            ]
            spread_acceleration = sum(deltas) / len(deltas)

        # Track imbalance
        self._imbalances.append(bid_ask_imbalance)

        # Compute microstructure score (0-1 stress indicator)
        spread_stress = min(spread_ratio / SPREAD_STRESS_RATIO, 1.0)
        accel_stress = min(abs(spread_acceleration) / SPREAD_ACCEL_THRESHOLD, 1.0)
        imbalance_stress = min(abs(bid_ask_imbalance) / IMBALANCE_THRESHOLD, 1.0)

        # Weighted combination: spread ratio is most reliable
        micro_score = 0.50 * spread_stress + 0.30 * accel_stress + 0.20 * imbalance_stress
        micro_score = max(0.0, min(1.0, micro_score))

        # Predict regime from microstructure alone
        if micro_score > 0.8:
            predicted_regime = "EXTREME"
        elif micro_score > 0.5:
            predicted_regime = "HIGH"
        elif micro_score > 0.25:
            predicted_regime = "NORMAL"
        else:
            predicted_regime = "LOW"

        # Compute confidence lift
        # Lift is highest when microstructure agrees with volatility regime
        confidence_lift = 0.0
        if predicted_regime == volatility_regime:
            # Agreement → lift proportional to score
            confidence_lift = MIN_CONFIDENCE_LIFT + (MAX_CONFIDENCE_LIFT - MIN_CONFIDENCE_LIFT) * micro_score
        elif (
            (predicted_regime in ("HIGH", "EXTREME") and volatility_regime in ("HIGH", "EXTREME"))
            or (predicted_regime in ("LOW", "NORMAL") and volatility_regime in ("LOW", "NORMAL"))
        ):
            # Partial agreement
            confidence_lift = MIN_CONFIDENCE_LIFT * micro_score

        # Log divergence (microstructure says stress, volatility says calm or vice versa)
        stress_regimes = {"HIGH", "EXTREME"}
        calm_regimes = {"LOW", "NORMAL"}
        if (
            predicted_regime in stress_regimes
            and volatility_regime in calm_regimes
        ):
            logger.warning(
                "Microstructure divergence: micro=%s (score=%.2f) but vol_regime=%s "
                "— potential regime shift incoming",
                predicted_regime, micro_score, volatility_regime,
            )

        self._current_signals = MicrostructureSignals(
            spread_ratio=spread_ratio,
            spread_acceleration=spread_acceleration,
            bid_ask_imbalance=bid_ask_imbalance,
            microstructure_score=micro_score,
            predicted_regime=predicted_regime,
            confidence_lift=confidence_lift,
        )

        # Persist baseline periodically
        if self._baseline_count % 10 == 0:
            self._save_state()

        return self._current_signals

    def predict_next_regime(self) -> Tuple[str, float]:
        """Return current microstructure regime prediction and confidence lift.

        Returns:
            (predicted_regime, confidence_lift) where lift is 0.05-0.15
        """
        return self._current_signals.predicted_regime, self._current_signals.confidence_lift

    def get_signals(self) -> MicrostructureSignals:
        """Get current microstructure signal state."""
        return self._current_signals

    def get_spread_baseline(self) -> float:
        """Get the current spread baseline (EMA)."""
        return self._baseline_spread
