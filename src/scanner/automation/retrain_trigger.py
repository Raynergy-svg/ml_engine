"""
Drift-Triggered Retraining — Monitors model accuracy and triggers retraining
when performance degrades below acceptable thresholds.

Features:
- Tracks rolling accuracy per pair from closed trade outcomes
- Detects per-pair drift and global performance degradation
- Auto-generates retrain requests with priority levels
- Cooldown period prevents redundant retraining
- Integrates with observation_log for pattern capture
- File-locked writes for safe concurrent access
"""

from __future__ import annotations

import fcntl
import json
import logging
import os
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class RetrainRequest:
    """Request object for triggered retraining."""

    pairs: list  # List of pair codes to retrain
    reason: str  # Human-readable reason (e.g., "Drift detected: 3 pair(s) below accuracy threshold")
    accuracy_snapshot: dict  # Current accuracy per pair at trigger time
    triggered_at: str  # ISO timestamp when drift was detected
    priority: str  # "urgent" or "normal"


class RetrainTrigger:
    """Monitors model accuracy and triggers retraining when drift detected."""

    def __init__(
        self,
        request_path: str = "trained_data/retrain_request.json",
        retrain_summary_path: str = "trained_data/retrain_all_summary.json",
        observation_log=None,
        per_pair_threshold: float = 0.50,
        global_threshold: float = 0.51,
        rolling_window: int = 50,
        cooldown_hours: int = 24,
        trend_window: int = 20,
        trend_decline_threshold: float = 0.05,
    ):
        """Initialize RetrainTrigger.

        Args:
            request_path: Path to write retrain request JSON (read by scheduled_retrain.py)
            retrain_summary_path: Path to retrain status file (for cooldown checking)
            observation_log: Optional ObservationLog instance for pattern capture
            per_pair_threshold: Accuracy below this triggers per-pair retrain (0.50 = coin-flip)
            global_threshold: Global accuracy below this triggers full retrain
            rolling_window: Number of recent predictions to track (limits memory)
            cooldown_hours: Hours to wait after retrain before checking again
            trend_window: Recent window to compare against full window for decline detection
            trend_decline_threshold: Trigger if recent accuracy is this much below full window (e.g. 0.05 = 5pp decline)
        """
        self.request_path = Path(request_path)
        self.retrain_summary_path = Path(retrain_summary_path)
        self.observation_log = observation_log
        self.per_pair_threshold = per_pair_threshold
        self.global_threshold = global_threshold
        self.rolling_window = rolling_window
        self.cooldown_hours = cooldown_hours
        self.trend_window = trend_window
        self.trend_decline_threshold = trend_decline_threshold
        self._prediction_history: dict[str, list] = {}  # pair -> list of bool

    def record_prediction(self, pair: str, correct: bool) -> None:
        """Record a prediction outcome for drift tracking.

        Args:
            pair: Currency pair (e.g., "EUR_USD")
            correct: True if prediction was correct, False otherwise
        """
        if pair not in self._prediction_history:
            self._prediction_history[pair] = []

        self._prediction_history[pair].append(correct)

        # Keep only rolling_window entries to prevent unbounded memory growth
        if len(self._prediction_history[pair]) > self.rolling_window:
            self._prediction_history[pair] = self._prediction_history[pair][
                -self.rolling_window :
            ]

        logger.debug(f"{pair}: recorded prediction outcome={correct}")

    def get_pair_accuracy(self, pair: str) -> Optional[float]:
        """Get rolling accuracy for a pair.

        Args:
            pair: Currency pair code

        Returns:
            Accuracy as float (0.0-1.0), or None if insufficient data (< 10 samples)
        """
        history = self._prediction_history.get(pair, [])
        if len(history) < 10:  # Minimum samples for meaningful accuracy
            return None
        return sum(history) / len(history)

    def get_global_accuracy(self) -> Optional[float]:
        """Get global accuracy across all pairs.

        Returns:
            Accuracy as float (0.0-1.0), or None if insufficient data (< 20 samples)
        """
        all_outcomes = []
        for history in self._prediction_history.values():
            all_outcomes.extend(history)

        if len(all_outcomes) < 20:  # Minimum samples for global evaluation
            return None
        return sum(all_outcomes) / len(all_outcomes)

    def check_drift(self) -> Optional[RetrainRequest]:
        """Check if any pair or global accuracy has drifted below threshold.

        Returns:
            RetrainRequest if retrain needed, None otherwise.
            Side effects: Writes retrain_request.json if drift detected.
        """
        # Check if we're in cooldown period after last retrain
        if self._is_in_cooldown():
            logger.debug("Retrain cooldown active, skipping drift check")
            return None

        pairs_to_retrain = []
        accuracy_snapshot = {}

        # Check per-pair accuracy (threshold + trend decline)
        for pair, history in self._prediction_history.items():
            accuracy = self.get_pair_accuracy(pair)
            if accuracy is not None:
                accuracy_snapshot[pair] = {
                    "accuracy": round(accuracy, 4),
                    "trades": len(history),
                }
                # Absolute threshold trigger
                if accuracy < self.per_pair_threshold:
                    pairs_to_retrain.append(pair)
                    logger.warning(
                        f"Drift detected for {pair}: accuracy={accuracy:.3f} "
                        f"< {self.per_pair_threshold}"
                    )
                # Trend decline trigger: recent accuracy declining vs full window
                elif len(history) >= self.rolling_window:
                    recent = history[-self.trend_window:]
                    recent_acc = sum(recent) / len(recent)
                    full_acc = sum(history) / len(history)
                    decline = full_acc - recent_acc
                    if decline >= self.trend_decline_threshold:
                        pairs_to_retrain.append(pair)
                        accuracy_snapshot[pair]["trend_decline"] = round(decline, 4)
                        logger.warning(
                            f"Accuracy trend decline for {pair}: recent={recent_acc:.3f} "
                            f"vs full={full_acc:.3f} (decline={decline:.3f}pp)"
                        )

        # Check global accuracy
        global_acc = self.get_global_accuracy()
        if global_acc is not None:
            accuracy_snapshot["_global"] = {"accuracy": round(global_acc, 4)}
            if global_acc < self.global_threshold:
                # Global drift: retrain all pairs
                from src.scanner.config import DEFAULT_PAIRS

                pairs_to_retrain = list(DEFAULT_PAIRS)
                logger.warning(
                    f"Global drift detected: accuracy={global_acc:.3f} "
                    f"< {self.global_threshold}. Full retrain requested."
                )

        if not pairs_to_retrain:
            logger.debug("No drift detected")
            return None

        # Determine priority based on severity
        priority = "urgent" if (global_acc and global_acc < self.global_threshold) else "normal"

        request = RetrainRequest(
            pairs=list(set(pairs_to_retrain)),
            reason=f"Drift detected: {len(pairs_to_retrain)} pair(s) below accuracy threshold",
            accuracy_snapshot=accuracy_snapshot,
            triggered_at=datetime.utcnow().isoformat() + "Z",
            priority=priority,
        )

        # Write request to file (for scheduled_retrain.py to pick up)
        self._write_request(request)

        # Log observation to ObservationLog if available
        if self.observation_log:
            try:
                self.observation_log.log_observation(
                    pair="_ALL",
                    category="retrain_trigger",
                    description=request.reason,
                    metadata=accuracy_snapshot,
                )
            except Exception as e:
                logger.debug(f"Failed to log observation: {e}")

        return request

    def _is_in_cooldown(self) -> bool:
        """Check if we're within cooldown period since last retrain.

        Returns:
            True if retrain completed recently (within cooldown_hours)
        """
        try:
            if self.retrain_summary_path.exists():
                data = json.loads(self.retrain_summary_path.read_text())
                # Support both 'timestamp' and 'completed_at' keys
                last_time = data.get("timestamp") or data.get("completed_at", "")

                if last_time:
                    # Parse ISO timestamp safely
                    try:
                        last_dt = datetime.fromisoformat(last_time.replace("Z", "+00:00"))
                    except (ValueError, AttributeError):
                        # Fallback if timestamp is malformed
                        logger.debug(f"Could not parse timestamp: {last_time}")
                        return False

                    # Compare with now (handle timezone-aware datetimes)
                    now = (
                        datetime.now(last_dt.tzinfo)
                        if last_dt.tzinfo
                        else datetime.utcnow()
                    )

                    time_since_retrain = now - last_dt.replace(tzinfo=None)
                    if time_since_retrain < timedelta(hours=self.cooldown_hours):
                        logger.debug(
                            f"Cooldown active: {time_since_retrain.total_seconds():.0f}s "
                            f"since last retrain (cooldown={self.cooldown_hours}h)"
                        )
                        return True
        except Exception as e:
            logger.debug(f"Cooldown check failed (non-fatal): {e}")

        return False

    def _write_request(self, request: RetrainRequest) -> None:
        """Write retrain request file (read by scheduled_retrain.py).

        Uses fcntl file locking to ensure safe concurrent access.

        Args:
            request: RetrainRequest object to persist
        """
        try:
            self.request_path.parent.mkdir(parents=True, exist_ok=True)

            with open(self.request_path, "w") as f:
                # Acquire exclusive lock
                fcntl.flock(f.fileno(), fcntl.LOCK_EX)
                try:
                    json.dump(asdict(request), f, indent=2)
                    f.flush()
                    os.fsync(f.fileno())
                finally:
                    # Release lock
                    fcntl.flock(f.fileno(), fcntl.LOCK_UN)

            logger.info(
                f"Retrain request written: {len(request.pairs)} pair(s), "
                f"priority={request.priority}"
            )
        except Exception as e:
            logger.error(f"Failed to write retrain request: {e}")

    def clear_request(self) -> None:
        """Clear the retrain request after it's been processed.

        Called by scheduled_retrain.py after completing retrain.
        """
        try:
            if self.request_path.exists():
                self.request_path.unlink()
                logger.debug("Retrain request cleared")
        except Exception as e:
            logger.error(f"Failed to clear retrain request: {e}")

    def get_stats(self) -> dict:
        """Get current drift monitoring statistics.

        Returns:
            Dict with pair counts, accuracy ranges, and trigger status
        """
        stats = {
            "pairs_tracked": len(self._prediction_history),
            "total_predictions": sum(
                len(h) for h in self._prediction_history.values()
            ),
            "per_pair_accuracies": {},
            "global_accuracy": self.get_global_accuracy(),
            "in_cooldown": self._is_in_cooldown(),
        }

        for pair, history in self._prediction_history.items():
            acc = self.get_pair_accuracy(pair)
            if acc is not None:
                stats["per_pair_accuracies"][pair] = {
                    "accuracy": round(acc, 4),
                    "samples": len(history),
                    "below_threshold": acc < self.per_pair_threshold,
                }

        return stats
