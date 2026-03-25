"""Per-Pair Model Selector — Automatically switches between joint and per-pair
fine-tuned models based on rolling accuracy comparison.

Maintains a registry of model selections per pair with rolling accuracy tracking.
Automatically recommends switches when a per-pair model outperforms the joint model
by a configurable threshold, and vice versa.

Key features:
- Tracks joint vs per-pair rolling accuracy per pair
- Automatic model switching recommendation based on accuracy improvement
- File-locked JSON persistence (safe for concurrent access)
- Graceful handling of missing/corrupted JSON files
- JSONL event log for audit trail of all switches
"""

from __future__ import annotations

import fcntl
import json
import logging
import os
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class PairModelRecord:
    """Record of model selection status for a single pair."""

    pair: str
    active_model: str  # "joint" or "per_pair"
    joint_accuracy: float  # rolling accuracy of joint model
    per_pair_accuracy: float  # rolling accuracy of per-pair model
    joint_trades: int  # number of trades evaluated on joint model
    per_pair_trades: int  # number of trades evaluated on per-pair model
    last_switch: str  # ISO timestamp of last model switch (empty if never switched)
    reason: str  # human-readable reason for current selection


class PairModelSelector:
    """Selects the best model for each pair based on rolling accuracy."""

    def __init__(
        self,
        registry_path: str = "trained_data/models/pair_model_registry.json",
        selections_log_path: str = "trained_data/models/pair_model_selections.jsonl",
        models_dir: str = "trained_data/models",
        rolling_window: int = 30,
        min_trades: int = 30,
        switch_threshold: float = 0.02,
    ):
        """Initialize PairModelSelector.

        Args:
            registry_path: Path to JSON registry file (persisted state)
            selections_log_path: Path to JSONL audit log of all switches
            models_dir: Directory containing model files
            rolling_window: Number of trades for exponential moving average
            min_trades: Minimum trades on both models before recommending a switch
            switch_threshold: Minimum accuracy improvement to switch models (0-1 scale)
                For example, 0.02 means per-pair must be 2% better than joint
        """
        self.registry_path = Path(registry_path)
        self.selections_log_path = Path(selections_log_path)
        self.models_dir = Path(models_dir)
        self.rolling_window = rolling_window
        self.min_trades = min_trades
        self.switch_threshold = switch_threshold
        self._registry: Dict[str, PairModelRecord] = {}
        self._load_registry()

    def _load_registry(self) -> None:
        """Load pair model registry. Handle missing/corrupted gracefully."""
        try:
            if self.registry_path.exists():
                data = json.loads(self.registry_path.read_text())
                for pair, rec in data.items():
                    self._registry[pair] = PairModelRecord(**rec)
                logger.debug(f"Loaded pair model registry: {len(self._registry)} pairs")
        except (json.JSONDecodeError, TypeError, KeyError, ValueError) as e:
            logger.warning(f"Registry load failed: {e}. Starting fresh.")
            self._registry = {}

    def _save_registry(self) -> None:
        """Save registry with file locking."""
        try:
            self.registry_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.registry_path, "w") as f:
                fcntl.flock(f.fileno(), fcntl.LOCK_EX)
                try:
                    json.dump(
                        {p: asdict(r) for p, r in self._registry.items()},
                        f,
                        indent=2,
                    )
                    f.flush()
                    os.fsync(f.fileno())
                finally:
                    fcntl.flock(f.fileno(), fcntl.LOCK_UN)
        except Exception as e:
            logger.error(f"Registry save failed: {e}")

    def get_active_model(self, pair: str) -> str:
        """Get which model to use for a pair.

        Args:
            pair: Currency pair (e.g., "EUR_USD")

        Returns:
            "joint" or "per_pair"
        """
        if pair in self._registry:
            return self._registry[pair].active_model
        return "joint"  # Default to joint for new pairs

    def has_per_pair_model(self, pair: str) -> bool:
        """Check if a per-pair fine-tuned model exists on disk.

        Args:
            pair: Currency pair

        Returns:
            True if per-pair model directory with .keras or .pkl files exists
        """
        pair_dir = self.models_dir / pair
        if not pair_dir.exists():
            return False
        # Check for .keras or .pkl files in pair directory
        return bool(list(pair_dir.glob("*.keras")) or list(pair_dir.glob("*.pkl")))

    def record_prediction(
        self,
        pair: str,
        model_type: str,
        predicted_direction: str,
        actual_direction: str,
    ) -> None:
        """Record a prediction outcome for accuracy tracking.

        Args:
            pair: Currency pair
            model_type: "joint" or "per_pair"
            predicted_direction: What the model predicted ("LONG", "SHORT", "HOLD")
            actual_direction: What actually happened ("LONG", "SHORT", "HOLD")

        Returns:
            None. Updates internal registry and persists to disk.
        """
        # Validate inputs
        if not pair or not isinstance(pair, str):
            logger.warning(f"Invalid pair: {pair}")
            return

        if model_type not in ("joint", "per_pair"):
            logger.warning(f"Invalid model_type: {model_type}")
            return

        try:
            # Initialize record if needed
            if pair not in self._registry:
                self._registry[pair] = PairModelRecord(
                    pair=pair,
                    active_model="joint",
                    joint_accuracy=0.0,
                    per_pair_accuracy=0.0,
                    joint_trades=0,
                    per_pair_trades=0,
                    last_switch="",
                    reason="default",
                )

            rec = self._registry[pair]
            correct = (predicted_direction == actual_direction)

            # Update rolling accuracy using exponential moving average
            # This allows recent performance to matter more than old performance
            alpha = 2.0 / (self.rolling_window + 1)

            if model_type == "joint":
                rec.joint_trades += 1
                if rec.joint_trades == 1:
                    # First trade: set accuracy to 1.0 (correct) or 0.0 (wrong)
                    rec.joint_accuracy = 1.0 if correct else 0.0
                else:
                    # Update with exponential moving average
                    rec.joint_accuracy = (
                        alpha * (1.0 if correct else 0.0)
                        + (1.0 - alpha) * rec.joint_accuracy
                    )

            elif model_type == "per_pair":
                rec.per_pair_trades += 1
                if rec.per_pair_trades == 1:
                    rec.per_pair_accuracy = 1.0 if correct else 0.0
                else:
                    rec.per_pair_accuracy = (
                        alpha * (1.0 if correct else 0.0)
                        + (1.0 - alpha) * rec.per_pair_accuracy
                    )

            self._save_registry()

            # Log at debug level for high-frequency events
            logger.debug(
                f"{pair}/{model_type}: recorded {'correct' if correct else 'incorrect'} "
                f"({rec.joint_trades} joint, {rec.per_pair_trades} per_pair)"
            )

        except Exception as e:
            logger.error(f"Error recording prediction for {pair}: {e}")

    def check_switch(self, pair: str) -> Optional[str]:
        """Check if a pair should switch models.

        Recommends a switch only when:
        - Both models have minimum trades (min_trades)
        - The candidate model exceeds the current model by switch_threshold

        Args:
            pair: Currency pair

        Returns:
            New model type ("joint" or "per_pair") to switch to, or None if no switch
        """
        if pair not in self._registry:
            return None

        rec = self._registry[pair]

        # Need minimum trades on both models before switching
        if rec.joint_trades < self.min_trades or rec.per_pair_trades < self.min_trades:
            return None

        current = rec.active_model
        if current == "joint":
            # Switch to per_pair if significantly better
            if rec.per_pair_accuracy > rec.joint_accuracy + self.switch_threshold:
                return "per_pair"
        else:  # current == "per_pair"
            # Switch back to joint if per_pair degraded
            if rec.joint_accuracy > rec.per_pair_accuracy + self.switch_threshold:
                return "joint"

        return None

    def execute_switch(self, pair: str, new_model: str) -> None:
        """Execute a model switch for a pair.

        Args:
            pair: Currency pair
            new_model: "joint" or "per_pair"

        Returns:
            None. Updates registry, logs the switch, and persists to disk.
        """
        if pair not in self._registry:
            return

        rec = self._registry[pair]
        old_model = rec.active_model

        # Only proceed if actually changing
        if old_model == new_model:
            return

        rec.active_model = new_model
        rec.last_switch = datetime.utcnow().isoformat() + "Z"
        rec.reason = (
            f"Switched from {old_model} to {new_model}: "
            f"joint={rec.joint_accuracy:.3f} per_pair={rec.per_pair_accuracy:.3f}"
        )

        # Log the switch to audit trail
        self._log_selection(pair, old_model, new_model, rec)
        self._save_registry()

        logger.info(
            f"Model switch for {pair}: {old_model} → {new_model} ({rec.reason})"
        )

    def _log_selection(
        self,
        pair: str,
        old_model: str,
        new_model: str,
        rec: PairModelRecord,
    ) -> None:
        """Append switch event to JSONL log with file locking."""
        try:
            entry = {
                "timestamp": datetime.utcnow().isoformat() + "Z",
                "pair": pair,
                "old_model": old_model,
                "new_model": new_model,
                "joint_accuracy": rec.joint_accuracy,
                "per_pair_accuracy": rec.per_pair_accuracy,
                "joint_trades": rec.joint_trades,
                "per_pair_trades": rec.per_pair_trades,
                "reason": rec.reason,
            }

            self.selections_log_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.selections_log_path, "a") as f:
                fcntl.flock(f.fileno(), fcntl.LOCK_EX)
                try:
                    f.write(json.dumps(entry) + "\n")
                    f.flush()
                    os.fsync(f.fileno())
                finally:
                    fcntl.flock(f.fileno(), fcntl.LOCK_UN)
        except Exception as e:
            logger.error(f"Selection log failed: {e}")

    def get_status(self) -> Dict[str, Dict]:
        """Get current model selection status for all pairs.

        Returns:
            Dict mapping pair -> record dict with all status fields
        """
        return {pair: asdict(rec) for pair, rec in self._registry.items()}

    def get_summary(self) -> str:
        """Get human-readable summary of model selections.

        Returns:
            Formatted string with pair, active model, and accuracy stats
        """
        if not self._registry:
            return "No pair model selections recorded yet."

        lines = [
            "PairModelSelector Summary",
            "=" * 100,
            (
                f"{'Pair':12} {'Active':10} {'Joint Acc':>12} {'Per-Pair Acc':>12} "
                f"{'Joint Trades':>13} {'Per-Pair':>10}"
            ),
            "-" * 100,
        ]

        for pair, rec in sorted(self._registry.items()):
            lines.append(
                f"{pair:12} {rec.active_model:10} "
                f"{rec.joint_accuracy:12.3f} {rec.per_pair_accuracy:12.3f} "
                f"{rec.joint_trades:13} {rec.per_pair_trades:>10}"
            )

        # Count active models
        joint_count = sum(1 for r in self._registry.values() if r.active_model == "joint")
        per_pair_count = sum(
            1 for r in self._registry.values() if r.active_model == "per_pair"
        )

        lines.extend(
            [
                "=" * 100,
                f"Total pairs: {len(self._registry)} "
                f"(Joint: {joint_count}, Per-Pair: {per_pair_count})",
            ]
        )

        return "\n".join(lines)

    def reset_pair(self, pair: str) -> None:
        """Reset model selection history for a specific pair.

        Use after retraining a pair model to start accuracy evaluation fresh.

        Args:
            pair: Currency pair to reset
        """
        if pair in self._registry:
            del self._registry[pair]
            self._save_registry()
            logger.info(f"Reset model selection history for {pair}")

    def get_pair_stats(self, pair: str) -> Optional[Dict]:
        """Get detailed statistics for a specific pair.

        Args:
            pair: Currency pair

        Returns:
            Dict with accuracy, trades, and recommendation, or None if not found
        """
        if pair not in self._registry:
            return None

        rec = self._registry[pair]
        recommendation = self.check_switch(pair)

        return {
            "pair": pair,
            "active_model": rec.active_model,
            "joint_accuracy": rec.joint_accuracy,
            "per_pair_accuracy": rec.per_pair_accuracy,
            "joint_trades": rec.joint_trades,
            "per_pair_trades": rec.per_pair_trades,
            "last_switch": rec.last_switch,
            "reason": rec.reason,
            "recommended_model": recommendation,
            "has_per_pair_model_on_disk": self.has_per_pair_model(pair),
        }
