"""Agent accuracy matrix per regime and pair.

US-100: Track per-agent directional accuracy sliced by regime and pair.
Identify which agents are calibrated for which market conditions.
Dynamically weight agents — boost agents with >60% accuracy in current
regime/pair, penalize those <45%.

Based on 2024 multi-agent ensemble calibration research.

Matrix structure: {regime: {pair: {agent: AccuracyRecord}}}
Each AccuracyRecord tracks correct/total predictions in a rolling window.
"""

from __future__ import annotations

import json
import logging
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
_MATRIX_PATH = _PROJECT_ROOT / "trained_data" / "agent_accuracy_matrix.json"

# Weight adjustment thresholds
HIGH_ACCURACY_THRESHOLD = 0.60  # Boost agents above this
LOW_ACCURACY_THRESHOLD = 0.45  # Penalize agents below this
MIN_SAMPLES = 10  # Minimum verdicts to form an opinion
WEIGHT_BOOST = 1.15  # Multiplier for high-accuracy agents
WEIGHT_PENALTY = 0.80  # Multiplier for low-accuracy agents
ROLLING_WINDOW = 50  # Rolling window for accuracy calculation


@dataclass
class AccuracyRecord:
    """Rolling accuracy record for one agent-pair-regime combination."""

    correct: int = 0
    total: int = 0

    @property
    def accuracy(self) -> float:
        return self.correct / self.total if self.total > 0 else 0.5

    @property
    def has_sufficient_data(self) -> bool:
        return self.total >= MIN_SAMPLES


class AgentAccuracyMatrix:
    """Track and query per-agent accuracy across regimes and pairs.

    Provides a lookup table: given (agent, pair, regime), return
    historical accuracy and a weight multiplier for dynamic weighting.

    Usage:
        matrix = AgentAccuracyMatrix()
        matrix.record_verdict("trend", "EUR_USD", "HIGH", "BUY", "win")
        weights = matrix.get_regime_weights("HIGH", "EUR_USD")
        # weights = {"trend": 1.15, "mean_reversion": 0.80, ...}
    """

    def __init__(
        self,
        window: int = ROLLING_WINDOW,
        persistence_path: Optional[Path] = None,
    ):
        self.window = window
        self.persistence_path = persistence_path or _MATRIX_PATH

        # Matrix: regime → pair → agent → (correct_list, total_list)
        # Using lists for rolling window support
        self._correct: Dict[str, Dict[str, Dict[str, List[bool]]]] = defaultdict(
            lambda: defaultdict(lambda: defaultdict(list))
        )

        self._total_records: int = 0

        self._load_state()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def record_verdict(
        self,
        agent_name: str,
        pair: str,
        regime: str,
        predicted_direction: str,
        actual_outcome: str,
    ) -> None:
        """Record an agent verdict and its outcome.

        Args:
            agent_name: Name of the agent (e.g., "trend", "momentum").
            pair: Currency pair (e.g., "EUR_USD").
            regime: Volatility regime at time of verdict.
            predicted_direction: What the agent predicted ("BUY" or "SELL").
            actual_outcome: Trade outcome ("win", "loss", "breakeven").
        """
        is_correct = actual_outcome in ("win", "breakeven")

        window = self._correct[regime][pair][agent_name]
        window.append(is_correct)

        # Rolling window trim
        if len(window) > self.window:
            self._correct[regime][pair][agent_name] = window[-self.window:]

        self._total_records += 1

    def get_agent_accuracy(
        self,
        agent_name: str,
        pair: Optional[str] = None,
        regime: Optional[str] = None,
        window: Optional[int] = None,
    ) -> Tuple[float, int]:
        """Get agent accuracy for a specific context.

        Args:
            agent_name: Agent name.
            pair: Filter by pair (None = all pairs).
            regime: Filter by regime (None = all regimes).
            window: Override rolling window size.

        Returns:
            Tuple of (accuracy, sample_count).
        """
        _window = window or self.window
        correct_total = []

        regimes = [regime] if regime else list(self._correct.keys())
        for r in regimes:
            if r not in self._correct:
                continue
            pairs = [pair] if pair else list(self._correct[r].keys())
            for p in pairs:
                if p not in self._correct[r]:
                    continue
                if agent_name in self._correct[r][p]:
                    records = self._correct[r][p][agent_name][-_window:]
                    correct_total.extend(records)

        if not correct_total:
            return 0.5, 0  # Neutral when no data

        accuracy = sum(correct_total) / len(correct_total)
        return accuracy, len(correct_total)

    def get_regime_weights(
        self,
        regime: str,
        pair: Optional[str] = None,
    ) -> Dict[str, float]:
        """Get weight multipliers for all agents in a regime/pair context.

        Args:
            regime: Current volatility regime.
            pair: Currency pair (None = regime-wide).

        Returns:
            Dict mapping agent_name → weight multiplier.
        """
        weights: Dict[str, float] = {}

        # Collect all agents we have data for
        all_agents: set = set()
        if regime in self._correct:
            if pair and pair in self._correct[regime]:
                all_agents = set(self._correct[regime][pair].keys())
            else:
                for p in self._correct[regime]:
                    all_agents.update(self._correct[regime][p].keys())

        for agent in all_agents:
            accuracy, count = self.get_agent_accuracy(agent, pair, regime)

            if count < MIN_SAMPLES:
                weights[agent] = 1.0  # Neutral when insufficient data
                continue

            if accuracy >= HIGH_ACCURACY_THRESHOLD:
                # Boost proportionally to accuracy above threshold
                excess = accuracy - HIGH_ACCURACY_THRESHOLD
                boost = 1.0 + (WEIGHT_BOOST - 1.0) * (excess / (1.0 - HIGH_ACCURACY_THRESHOLD))
                weights[agent] = min(WEIGHT_BOOST, boost)
            elif accuracy <= LOW_ACCURACY_THRESHOLD:
                # Penalize proportionally to accuracy below threshold
                deficit = LOW_ACCURACY_THRESHOLD - accuracy
                penalty = 1.0 - (1.0 - WEIGHT_PENALTY) * (deficit / LOW_ACCURACY_THRESHOLD)
                weights[agent] = max(WEIGHT_PENALTY, penalty)
            else:
                weights[agent] = 1.0  # Neutral zone

        return weights

    def get_best_agents(
        self,
        regime: str,
        pair: Optional[str] = None,
        n: int = 3,
    ) -> List[Tuple[str, float, int]]:
        """Get top-N agents by accuracy for a regime/pair.

        Returns:
            List of (agent_name, accuracy, sample_count) tuples.
        """
        results = []
        all_agents: set = set()

        if regime in self._correct:
            if pair and pair in self._correct[regime]:
                all_agents = set(self._correct[regime][pair].keys())
            else:
                for p in self._correct[regime]:
                    all_agents.update(self._correct[regime][p].keys())

        for agent in all_agents:
            acc, count = self.get_agent_accuracy(agent, pair, regime)
            if count >= MIN_SAMPLES:
                results.append((agent, acc, count))

        results.sort(key=lambda x: x[1], reverse=True)
        return results[:n]

    def get_status(self) -> Dict[str, Any]:
        """Get matrix status for observability."""
        regime_stats: Dict[str, int] = {}
        for regime in self._correct:
            count = 0
            for pair in self._correct[regime]:
                for agent in self._correct[regime][pair]:
                    count += len(self._correct[regime][pair][agent])
            regime_stats[regime] = count

        return {
            "total_records": self._total_records,
            "regimes_tracked": list(self._correct.keys()),
            "records_per_regime": regime_stats,
        }

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save_state(self) -> None:
        """Persist accuracy matrix."""
        try:
            data = {
                "version": 1,
                "total_records": self._total_records,
                "matrix": {},
                "last_updated": datetime.now(timezone.utc).isoformat(),
            }

            for regime in self._correct:
                data["matrix"][regime] = {}
                for pair in self._correct[regime]:
                    data["matrix"][regime][pair] = {}
                    for agent in self._correct[regime][pair]:
                        records = self._correct[regime][pair][agent][-self.window:]
                        data["matrix"][regime][pair][agent] = {
                            "correct": sum(records),
                            "total": len(records),
                            "accuracy": round(sum(records) / max(len(records), 1), 4),
                        }

            self.persistence_path.parent.mkdir(parents=True, exist_ok=True)
            try:
                from src.scanner.automation.safe_json import safe_json_write
                safe_json_write(self.persistence_path, data)
            except ImportError:
                self.persistence_path.write_text(json.dumps(data, indent=2))
        except Exception as e:
            logger.debug(f"AgentAccuracyMatrix: Failed to save: {e}")

    def _load_state(self) -> None:
        """Load persisted accuracy matrix."""
        if not self.persistence_path.exists():
            return
        try:
            try:
                from src.scanner.automation.safe_json import safe_json_read
                data = safe_json_read(self.persistence_path, default={})
            except ImportError:
                data = json.loads(self.persistence_path.read_text(encoding="utf-8"))

            if not isinstance(data, dict):
                return

            self._total_records = data.get("total_records", 0)

            for regime, pairs in data.get("matrix", {}).items():
                for pair, agents in pairs.items():
                    for agent, stats in agents.items():
                        correct = stats.get("correct", 0)
                        total = stats.get("total", 0)
                        # Reconstruct rolling window approximation
                        records = [True] * correct + [False] * (total - correct)
                        self._correct[regime][pair][agent] = records

            logger.debug(
                f"AgentAccuracyMatrix: Loaded {self._total_records} records "
                f"across {len(self._correct)} regimes"
            )
        except Exception as e:
            logger.debug(f"AgentAccuracyMatrix: Failed to load: {e}")
