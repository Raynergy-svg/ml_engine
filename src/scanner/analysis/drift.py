"""
Model Drift Detection Module.

Monitors model accuracy over time and detects performance degradation.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

logger = logging.getLogger(__name__)


@dataclass
class DriftResult:
    """Result of drift detection.

    Attributes:
        drift_detected: Whether significant drift was detected
        current_accuracy: Current model accuracy estimate
        baseline_accuracy: Baseline accuracy from training
        drift_amount: Absolute difference between current and baseline
        needs_retrain: Whether retraining is recommended
        retrain_reason: Reason for retraining recommendation
        last_check: Timestamp of last check
    """
    drift_detected: bool = False
    current_accuracy: float = 0.0
    baseline_accuracy: float = 0.0
    drift_amount: float = 0.0
    needs_retrain: bool = False
    retrain_reason: Optional[str] = None
    last_check: datetime = None

    def __post_init__(self):
        if self.last_check is None:
            self.last_check = datetime.now(timezone.utc)


class DriftDetector:
    """Detects model performance drift.

    Monitors model accuracy over time by comparing current performance
    against baseline metrics from training. Triggers retrain recommendations
    when drift exceeds threshold.

    Features:
    - Baseline accuracy from model metadata
    - Live accuracy tracking via MemoryClient
    - Configurable drift threshold
    - Per-pair and global drift detection
    """

    def __init__(
        self,
        threshold: float = 0.03,
        model_dir: Optional[Path] = None,
        min_trades_for_check: int = 20,
    ):
        """Initialize drift detector.

        Args:
            threshold: Accuracy drop threshold for triggering retrain (default 3%)
            model_dir: Path to model artifacts directory
            min_trades_for_check: Minimum trades needed for valid drift check
        """
        self.threshold = threshold
        self.model_dir = model_dir or Path("trained_data/models")
        self.min_trades_for_check = min_trades_for_check

        # Cached data
        self._model_meta: Optional[Dict[str, Any]] = None
        self._memory_client = None
        self._baseline_accuracy: float = 0.0
        self._last_check_time: Optional[datetime] = None

        # Per-pair tracking
        self._pair_accuracies: Dict[str, float] = {}

    def _load_model_meta(self) -> bool:
        """Load model metadata for baseline accuracy.

        Returns:
            True if metadata loaded successfully
        """
        if self._model_meta is not None:
            return True

        meta_path = self.model_dir / "modular_ensemble.meta.json"

        if not meta_path.exists():
            logger.debug(f"Model meta not found: {meta_path}")
            return False

        try:
            self._model_meta = json.loads(meta_path.read_text())

            # Extract baseline accuracy
            # Try multiple paths in meta structure
            baseline = 0.0

            if "results" in self._model_meta:
                baseline = self._model_meta.get("results", {}).get(
                    "transformer", {}
                ).get("val_accuracy", 0)

            if baseline == 0 and "models" in self._model_meta:
                baseline = self._model_meta.get("models", {}).get(
                    "direction", {}
                ).get("metrics", {}).get("val_accuracy", 0)

            if baseline == 0 and "metrics" in self._model_meta:
                baseline = self._model_meta.get("metrics", {}).get("val_accuracy", 0)

            self._baseline_accuracy = baseline
            logger.debug(f"Loaded baseline accuracy: {baseline:.2%}")
            return True

        except Exception as e:
            logger.warning(f"Failed to load model meta: {e}")
            return False

    def _init_memory_client(self) -> bool:
        """Initialize memory client for live accuracy tracking.

        Returns:
            True if client initialized successfully
        """
        if self._memory_client is not None:
            return True

        try:
            from memory_client import MLEngineMemory
            self._memory_client = MLEngineMemory()
            return True
        except ImportError:
            logger.debug("MemoryClient not available")
            return False

    def get_baseline_accuracy(self) -> float:
        """Get baseline accuracy from model training.

        Returns:
            Baseline validation accuracy (0.0 if unavailable)
        """
        self._load_model_meta()
        return self._baseline_accuracy

    def get_live_accuracy(self, pair: Optional[str] = None) -> Tuple[float, int]:
        """Get live accuracy from recent trades.

        Args:
            pair: Optional pair to filter by (None = all pairs)

        Returns:
            Tuple of (accuracy, num_trades)
        """
        if not self._init_memory_client():
            return 0.0, 0

        try:
            # Get recent trades from memory
            trades = self._memory_client.get_recent_trades(limit=100)

            if pair:
                trades = [t for t in trades if t.get("instrument") == pair]

            if len(trades) < self.min_trades_for_check:
                return 0.0, len(trades)

            # Calculate win rate as proxy for accuracy
            wins = sum(1 for t in trades if t.get("pnl", 0) > 0)
            total = len(trades)
            accuracy = wins / total if total > 0 else 0.0

            return accuracy, total

        except Exception as e:
            logger.debug(f"Failed to get live accuracy: {e}")
            return 0.0, 0

    def detect(self, pair: Optional[str] = None) -> DriftResult:
        """Detect model drift.

        Compares current live accuracy against baseline from training.
        Returns drift result with recommendations.

        Args:
            pair: Optional pair to check (None = global check)

        Returns:
            DriftResult with drift detection details
        """
        self._load_model_meta()

        baseline = self._baseline_accuracy
        current, num_trades = self.get_live_accuracy(pair)

        # If not enough trades, use baseline as current (no drift)
        if num_trades < self.min_trades_for_check:
            return DriftResult(
                drift_detected=False,
                current_accuracy=baseline,
                baseline_accuracy=baseline,
                drift_amount=0.0,
                needs_retrain=False,
                retrain_reason=f"Insufficient trades ({num_trades} < {self.min_trades_for_check})",
            )

        # Calculate drift
        drift_amount = baseline - current  # Positive = accuracy dropped
        drift_detected = drift_amount > self.threshold

        # Determine retrain recommendation
        needs_retrain = False
        retrain_reason = None

        if drift_detected:
            needs_retrain = True
            retrain_reason = (
                f"Accuracy dropped {drift_amount:.1%} "
                f"(from {baseline:.1%} to {current:.1%})"
            )
        elif current < 0.5:
            needs_retrain = True
            retrain_reason = f"Accuracy below 50% ({current:.1%})"

        # Update per-pair tracking
        if pair:
            self._pair_accuracies[pair] = current

        self._last_check_time = datetime.now(timezone.utc)

        return DriftResult(
            drift_detected=drift_detected,
            current_accuracy=current,
            baseline_accuracy=baseline,
            drift_amount=drift_amount,
            needs_retrain=needs_retrain,
            retrain_reason=retrain_reason,
        )

    def detect_for_pairs(
        self,
        pairs: list,
    ) -> Dict[str, DriftResult]:
        """Detect drift for multiple pairs.

        Args:
            pairs: List of pair names to check

        Returns:
            Dict mapping pair -> DriftResult
        """
        results = {}

        for pair in pairs:
            results[pair] = self.detect(pair)

        return results

    def get_retrain_candidates(
        self,
        results: Dict[str, DriftResult],
    ) -> list:
        """Get list of pairs that need retraining.

        Args:
            results: Dict from detect_for_pairs()

        Returns:
            List of pair names needing retrain
        """
        return [
            pair for pair, result in results.items()
            if result.needs_retrain
        ]

    def check_model_age(self, max_age_days: int = 30) -> Tuple[bool, int]:
        """Check if model is too old and needs refresh.

        Args:
            max_age_days: Maximum age in days before recommending retrain

        Returns:
            Tuple of (needs_retrain, age_days)
        """
        self._load_model_meta()

        if not self._model_meta:
            return True, -1

        try:
            trained_at = self._model_meta.get("trained_at")
            if not trained_at:
                return True, -1

            trained_date = datetime.fromisoformat(trained_at.replace("Z", "+00:00"))
            age = datetime.now(timezone.utc) - trained_date
            age_days = age.days

            needs_retrain = age_days > max_age_days

            return needs_retrain, age_days

        except Exception as e:
            logger.debug(f"Failed to check model age: {e}")
            return True, -1
