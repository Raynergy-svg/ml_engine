"""Pair-regime-agent interaction matrix.

US-113: Track the 3-way interaction between pairs, regimes, and agents.
Some agents may excel at EUR_USD in HIGH regime but fail at GBP_JPY in
the same regime. This module captures that granularity for targeted
agent weighting.

Approach:
1. Record (pair, regime, agent, vote, actual_outcome) for each prediction
2. Maintain 3-way accuracy matrix: pair × regime × agent
3. Provide agent weight multiplier based on historical accuracy
4. Generate specialization reports: which agents are best for which contexts
"""

from __future__ import annotations

import json
import logging
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
_LOG_PATH = _PROJECT_ROOT / "trained_data" / "pair_regime_agent_matrix_log.json"

# Matrix parameters
ROLLING_WINDOW = 60  # Maximum samples per cell
MIN_SAMPLES = 8      # Minimum samples for a cell to produce a weight
DEFAULT_WEIGHT = 1.0  # Weight when insufficient data

# Weight multiplier bounds
BOOST_THRESHOLD = 0.60   # Accuracy above this → weight boost
PENALIZE_THRESHOLD = 0.40  # Accuracy below this → weight penalty
MAX_WEIGHT = 1.40
MIN_WEIGHT = 0.50


class PairRegimeAgentMatrix:
    """Tracks 3-way pair×regime×agent accuracy for targeted weighting.

    Extends agent_accuracy_matrix (US-100) by adding the pair dimension.
    Reveals agent specialization patterns — e.g., momentum agent excels
    on GBP_JPY in HIGH regime but underperforms on EUR_CHF in LOW regime.

    Usage:
        matrix = PairRegimeAgentMatrix()
        matrix.record_vote("EUR_USD", "HIGH", "momentum", "BUY", won=True)
        weight = matrix.get_agent_weight("EUR_USD", "HIGH", "momentum")
        report = matrix.get_specialization_report()
    """

    def __init__(self, persistence_path: Optional[Path] = None):
        self.persistence_path = persistence_path or _LOG_PATH

        # 3-way matrix: {pair: {regime: {agent: [records]}}}
        self._matrix: Dict[str, Dict[str, Dict[str, List[Dict[str, Any]]]]] = (
            defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
        )

        # Aggregated stats cache
        self._stats_cache: Dict[str, Dict[str, float]] = {}
        self._total_votes: int = 0

        self._load_state()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def record_vote(
        self,
        pair: str,
        regime: str,
        agent: str,
        vote: str,
        won: bool,
    ) -> None:
        """Record an agent vote and its outcome.

        Args:
            pair: Currency pair.
            regime: Market regime.
            agent: Agent name.
            vote: Agent's vote (BUY/SELL/HOLD).
            won: Whether the trade was a winner.
        """
        record = {
            "vote": vote,
            "won": won,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

        self._matrix[pair][regime][agent].append(record)

        # Trim
        if len(self._matrix[pair][regime][agent]) > ROLLING_WINDOW * 2:
            self._matrix[pair][regime][agent] = (
                self._matrix[pair][regime][agent][-ROLLING_WINDOW:]
            )

        # Invalidate cache
        cache_key = f"{pair}_{regime}_{agent}"
        self._stats_cache.pop(cache_key, None)

        self._total_votes += 1

    def get_agent_weight(
        self, pair: str, regime: str, agent: str
    ) -> float:
        """Get accuracy-based weight multiplier for agent in context.

        Args:
            pair: Currency pair.
            regime: Market regime.
            agent: Agent name.

        Returns:
            Weight multiplier [0.50, 1.40].
        """
        cache_key = f"{pair}_{regime}_{agent}"
        if cache_key in self._stats_cache:
            return self._stats_cache[cache_key]

        records = self._matrix.get(pair, {}).get(regime, {}).get(agent, [])

        if len(records) < MIN_SAMPLES:
            self._stats_cache[cache_key] = DEFAULT_WEIGHT
            return DEFAULT_WEIGHT

        recent = records[-ROLLING_WINDOW:]
        accuracy = sum(1 for r in recent if r["won"]) / len(recent)

        # Compute weight multiplier
        if accuracy > BOOST_THRESHOLD:
            intensity = (accuracy - BOOST_THRESHOLD) / (1.0 - BOOST_THRESHOLD)
            weight = 1.0 + (MAX_WEIGHT - 1.0) * intensity
        elif accuracy < PENALIZE_THRESHOLD:
            intensity = (PENALIZE_THRESHOLD - accuracy) / PENALIZE_THRESHOLD
            weight = 1.0 - (1.0 - MIN_WEIGHT) * intensity
        else:
            weight = DEFAULT_WEIGHT

        weight = round(float(np.clip(weight, MIN_WEIGHT, MAX_WEIGHT)), 4)
        self._stats_cache[cache_key] = weight
        return weight

    def get_cell_accuracy(
        self, pair: str, regime: str, agent: str
    ) -> Optional[float]:
        """Get raw accuracy for a specific cell."""
        records = self._matrix.get(pair, {}).get(regime, {}).get(agent, [])
        if len(records) < MIN_SAMPLES:
            return None
        recent = records[-ROLLING_WINDOW:]
        return round(sum(1 for r in recent if r["won"]) / len(recent), 4)

    def get_specialization_report(self, top_n: int = 15) -> Dict[str, Any]:
        """Get top agent×pair×regime combinations by accuracy.

        Returns:
            Dict with top performers, worst performers, and matrix summary.
        """
        cells = []

        for pair in self._matrix:
            for regime in self._matrix[pair]:
                for agent in self._matrix[pair][regime]:
                    records = self._matrix[pair][regime][agent]
                    if len(records) < MIN_SAMPLES:
                        continue

                    recent = records[-ROLLING_WINDOW:]
                    accuracy = sum(1 for r in recent if r["won"]) / len(recent)

                    cells.append({
                        "pair": pair,
                        "regime": regime,
                        "agent": agent,
                        "accuracy": round(accuracy, 4),
                        "n_samples": len(recent),
                        "weight": self.get_agent_weight(pair, regime, agent),
                    })

        if not cells:
            return {"top_performers": [], "worst_performers": [], "total_cells": 0}

        cells.sort(key=lambda c: c["accuracy"], reverse=True)

        return {
            "top_performers": cells[:top_n],
            "worst_performers": cells[-top_n:] if len(cells) > top_n else [],
            "total_cells": len(cells),
            "avg_accuracy": round(
                float(np.mean([c["accuracy"] for c in cells])), 4
            ),
        }

    def get_status(self) -> Dict[str, Any]:
        """Get matrix status for observability."""
        pair_counts = {}
        for pair in self._matrix:
            total = 0
            for regime in self._matrix[pair]:
                for agent in self._matrix[pair][regime]:
                    total += len(self._matrix[pair][regime][agent])
            pair_counts[pair] = total

        return {
            "total_votes": self._total_votes,
            "pairs_tracked": list(self._matrix.keys()),
            "pair_vote_counts": pair_counts,
            "cache_size": len(self._stats_cache),
        }

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save_state(self) -> None:
        """Persist matrix state."""
        try:
            # Convert nested defaultdict to plain dict
            matrix_plain: Dict[str, Any] = {}
            for pair in self._matrix:
                matrix_plain[pair] = {}
                for regime in self._matrix[pair]:
                    matrix_plain[pair][regime] = {}
                    for agent in self._matrix[pair][regime]:
                        matrix_plain[pair][regime][agent] = (
                            self._matrix[pair][regime][agent][-ROLLING_WINDOW:]
                        )

            data = {
                "version": 1,
                "total_votes": self._total_votes,
                "matrix": matrix_plain,
                "last_updated": datetime.now(timezone.utc).isoformat(),
            }
            self.persistence_path.parent.mkdir(parents=True, exist_ok=True)
            try:
                from src.scanner.automation.safe_json import safe_json_write
                safe_json_write(self.persistence_path, data)
            except ImportError:
                self.persistence_path.write_text(json.dumps(data, indent=2))
        except Exception as e:
            logger.warning(f"PairRegimeAgentMatrix: Failed to save: {e}")

    def _load_state(self) -> None:
        """Load persisted state."""
        if not self.persistence_path.exists():
            return
        try:
            try:
                from src.scanner.automation.safe_json import safe_json_read
                data = safe_json_read(self.persistence_path, default={})
            except ImportError:
                data = json.loads(self.persistence_path.read_text(encoding="utf-8"))

            if isinstance(data, dict):
                self._total_votes = data.get("total_votes", 0)
                raw_matrix = data.get("matrix", {})
                for pair in raw_matrix:
                    if not isinstance(raw_matrix[pair], dict):
                        continue
                    for regime in raw_matrix[pair]:
                        if not isinstance(raw_matrix[pair][regime], dict):
                            continue
                        for agent in raw_matrix[pair][regime]:
                            records = raw_matrix[pair][regime][agent]
                            if not isinstance(records, list):
                                continue
                            valid = [
                                r for r in records
                                if isinstance(r, dict) and "won" in r
                            ]
                            self._matrix[pair][regime][agent] = valid
        except Exception as e:
            logger.warning(f"PairRegimeAgentMatrix: Failed to load: {e}")
