"""Entropy-based position sizing from ensemble signals.

US-098: Measure Shannon entropy across the 12-agent ensemble vote
distribution. High entropy (disagreement) reduces position size by
30-50%. Low entropy (consensus) allows full sizing or slight boost.
Provides information-theoretic bridge between agent consensus and
risk allocation.

Based on 2024 entropy portfolio research.

Shannon Entropy: H = -Σ(p_i * log2(p_i))
- 12 agents with binary votes: max entropy = log2(2) = 1.0 bit
- Normalized entropy: H_norm = H / log2(K) where K = number of categories
- Size multiplier: mult = 1.0 + boost_at_low_entropy - penalty_at_high_entropy
"""

from __future__ import annotations

import json
import logging
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
_ENTROPY_LOG_PATH = _PROJECT_ROOT / "trained_data" / "entropy_sizing_log.json"

# Default sizing parameters
DEFAULT_MIN_MULTIPLIER = 0.30  # Minimum position multiplier at max entropy
DEFAULT_MAX_MULTIPLIER = 1.20  # Maximum position multiplier at min entropy
DEFAULT_NEUTRAL_ENTROPY = 0.50  # Entropy at which multiplier = 1.0

# Regime-conditioned entropy thresholds
# Higher volatility regimes have stricter (lower) entropy tolerance
REGIME_ENTROPY_THRESHOLDS: Dict[str, Dict[str, float]] = {
    "LOW": {
        "neutral": 0.55,  # More tolerant in low vol
        "max_boost": 1.20,
        "max_penalty": 0.40,  # Gentler penalty
    },
    "NORMAL": {
        "neutral": 0.50,
        "max_boost": 1.15,
        "max_penalty": 0.35,
    },
    "HIGH": {
        "neutral": 0.40,  # Stricter in high vol
        "max_boost": 1.10,
        "max_penalty": 0.30,  # Harsher minimum
    },
    "EXTREME": {
        "neutral": 0.30,  # Very strict in extreme vol
        "max_boost": 1.05,
        "max_penalty": 0.25,  # Very harsh minimum
    },
}


@dataclass
class EntropyResult:
    """Result of entropy-based position sizing."""

    raw_entropy: float = 0.0
    normalized_entropy: float = 0.0
    size_multiplier: float = 1.0
    regime: str = "NORMAL"
    vote_distribution: Dict[str, int] = field(default_factory=dict)
    n_agents: int = 0

    def to_dict(self) -> dict:
        return {
            "raw_entropy": round(self.raw_entropy, 6),
            "normalized_entropy": round(self.normalized_entropy, 4),
            "size_multiplier": round(self.size_multiplier, 4),
            "regime": self.regime,
            "vote_distribution": self.vote_distribution,
            "n_agents": self.n_agents,
        }


class EntropyPositionSizer:
    """Information-theoretic position sizing from agent ensemble entropy.

    Measures how much agents agree (low entropy) or disagree (high
    entropy) and scales position size accordingly. Provides a
    principled, continuous measure of signal quality.

    Usage:
        sizer = EntropyPositionSizer()
        result = sizer.compute_ensemble_entropy(agent_votes)
        risk_pct *= result.size_multiplier
    """

    def __init__(
        self,
        min_multiplier: float = DEFAULT_MIN_MULTIPLIER,
        max_multiplier: float = DEFAULT_MAX_MULTIPLIER,
        persistence_path: Optional[Path] = None,
    ):
        self.min_multiplier = min_multiplier
        self.max_multiplier = max_multiplier
        self.persistence_path = persistence_path or _ENTROPY_LOG_PATH

        # History for diagnostics
        self._history: deque = deque(maxlen=200)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def compute_ensemble_entropy(
        self,
        agent_votes: Dict[str, Any],
    ) -> EntropyResult:
        """Compute Shannon entropy of agent vote distribution.

        Args:
            agent_votes: Dict mapping agent_name → vote.
                Vote can be: bool (True=for, False=against),
                str ("BUY"/"SELL"/"HOLD"), int (1/-1/0),
                or float (confidence score).

        Returns:
            EntropyResult with raw and normalized entropy.
        """
        result = EntropyResult()

        if not agent_votes:
            return result

        result.n_agents = len(agent_votes)

        # Categorize votes into discrete buckets
        vote_counts: Dict[str, int] = {"FOR": 0, "AGAINST": 0, "NEUTRAL": 0}

        for agent, vote in agent_votes.items():
            if isinstance(vote, bool):
                vote_counts["FOR" if vote else "AGAINST"] += 1
            elif isinstance(vote, str):
                v = vote.upper()
                if v in ("BUY", "SELL", "FOR", "TRUE", "1"):
                    vote_counts["FOR"] += 1
                elif v in ("AGAINST", "FALSE", "0", "HOLD"):
                    vote_counts["AGAINST"] += 1
                else:
                    vote_counts["NEUTRAL"] += 1
            elif isinstance(vote, (int, float)):
                if vote > 0.1:
                    vote_counts["FOR"] += 1
                elif vote < -0.1:
                    vote_counts["AGAINST"] += 1
                else:
                    vote_counts["NEUTRAL"] += 1
            else:
                vote_counts["NEUTRAL"] += 1

        result.vote_distribution = vote_counts

        # Compute Shannon entropy
        total = sum(vote_counts.values())
        if total == 0:
            return result

        probs = []
        for count in vote_counts.values():
            if count > 0:
                p = count / total
                probs.append(p)

        if not probs:
            return result

        # H = -Σ(p * log2(p))
        probs_arr = np.array(probs, dtype=np.float64)
        result.raw_entropy = -float(np.sum(probs_arr * np.log2(probs_arr)))

        # Normalize by max possible entropy (uniform over K categories)
        k = len([c for c in vote_counts.values() if c > 0])
        max_entropy = np.log2(max(k, 2))
        result.normalized_entropy = result.raw_entropy / max_entropy if max_entropy > 0 else 0.0

        return result

    def get_size_multiplier(
        self,
        entropy_result: Optional[EntropyResult] = None,
        normalized_entropy: Optional[float] = None,
        regime: str = "NORMAL",
    ) -> float:
        """Compute position size multiplier from entropy.

        Args:
            entropy_result: Result from compute_ensemble_entropy.
            normalized_entropy: Directly pass normalized entropy (0-1).
            regime: Current volatility regime.

        Returns:
            Position size multiplier in [min_multiplier, max_multiplier].
        """
        if entropy_result is not None:
            h = entropy_result.normalized_entropy
        elif normalized_entropy is not None:
            h = normalized_entropy
        else:
            return 1.0

        # Get regime-specific thresholds
        thresholds = REGIME_ENTROPY_THRESHOLDS.get(
            regime, REGIME_ENTROPY_THRESHOLDS["NORMAL"]
        )
        neutral = thresholds["neutral"]
        max_boost = thresholds["max_boost"]
        max_penalty = thresholds["max_penalty"]

        if h <= neutral:
            # Low entropy (consensus) → boost
            if neutral > 0:
                boost_fraction = 1.0 - (h / neutral)
            else:
                boost_fraction = 1.0
            multiplier = 1.0 + (max_boost - 1.0) * boost_fraction
        else:
            # High entropy (disagreement) → penalize
            remaining = 1.0 - neutral
            if remaining > 0:
                penalty_fraction = (h - neutral) / remaining
            else:
                penalty_fraction = 1.0
            multiplier = 1.0 - (1.0 - max_penalty) * penalty_fraction

        # Clamp
        multiplier = max(self.min_multiplier, min(self.max_multiplier, multiplier))

        return multiplier

    def size_from_votes(
        self,
        agent_votes: Dict[str, Any],
        regime: str = "NORMAL",
    ) -> EntropyResult:
        """Convenience: compute entropy and size multiplier in one call.

        Args:
            agent_votes: Agent vote dict.
            regime: Current volatility regime.

        Returns:
            EntropyResult with size_multiplier populated.
        """
        result = self.compute_ensemble_entropy(agent_votes)
        result.regime = regime
        result.size_multiplier = self.get_size_multiplier(
            entropy_result=result, regime=regime
        )

        # Log
        self._history.append({
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "entropy": round(result.normalized_entropy, 4),
            "multiplier": round(result.size_multiplier, 4),
            "regime": regime,
            "n_agents": result.n_agents,
            "distribution": result.vote_distribution,
        })

        return result

    def get_status(self) -> Dict[str, Any]:
        """Get sizer status for observability."""
        recent = list(self._history)[-20:]
        avg_entropy = 0.0
        avg_mult = 1.0
        if recent:
            avg_entropy = sum(h["entropy"] for h in recent) / len(recent)
            avg_mult = sum(h["multiplier"] for h in recent) / len(recent)

        return {
            "total_computations": len(self._history),
            "avg_entropy_recent": round(avg_entropy, 4),
            "avg_multiplier_recent": round(avg_mult, 4),
            "regime_thresholds": REGIME_ENTROPY_THRESHOLDS,
        }

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save_log(self) -> None:
        """Persist entropy sizing history."""
        try:
            data = {
                "version": 1,
                "history": list(self._history)[-100:],
                "last_updated": datetime.now(timezone.utc).isoformat(),
            }
            self.persistence_path.parent.mkdir(parents=True, exist_ok=True)
            try:
                from src.scanner.automation.safe_json import safe_json_write
                safe_json_write(self.persistence_path, data)
            except ImportError:
                self.persistence_path.write_text(json.dumps(data, indent=2))
        except Exception as e:
            logger.debug(f"EntropySizer: Failed to save: {e}")
