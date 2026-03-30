"""
Phase 55 (US-343): Agent Disagreement Pattern Feedback.

Correlates agent disagreement patterns with trade outcomes to identify
"toxic" disagreement patterns — specific agent pairs that, when they
disagree, predict losses with high confidence.

Architecture:
    1. After each trade close: record agent verdict vector + outcome
    2. Compute disagreement_score = std(scores) / mean(scores)
    3. Track per-bucket (0-0.2, 0.2-0.4, 0.4-0.6, 0.6+) win rates
    4. Identify toxic agent-pair conflicts (loss rate > 70%)
    5. Apply confidence penalty when toxic pattern is active

Usage:
    tracker = DisagreementTracker()
    tracker.record_trade(agent_scores, won=True, pnl=50.0)
    penalty = tracker.get_confidence_penalty(agent_scores)
    report = tracker.generate_report()
"""

from __future__ import annotations

import json
import logging
import os
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

_DEFAULT_PERSISTENCE_PATH = "trained_data/disagreement_patterns.json"
_REANALYZE_INTERVAL = 50  # Re-analyze every N closed trades
_TOXIC_LOSS_RATE_THRESHOLD = 0.70  # Agent pair conflict → loss > 70%
_TOXIC_MIN_OBSERVATIONS = 5  # Minimum conflicts to flag as toxic
_MAX_CONFIDENCE_PENALTY = 0.10  # Cap penalty at -10% confidence
_DISAGREEMENT_BUCKETS = [
    (0.0, 0.2, "low"),
    (0.2, 0.4, "moderate"),
    (0.4, 0.6, "high"),
    (0.6, float("inf"), "extreme"),
]


@dataclass
class TradeDisagreementRecord:
    """Single trade's disagreement snapshot."""
    agent_scores: Dict[str, float]
    disagreement_score: float
    bucket: str
    won: bool
    pnl: float
    timestamp: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "agent_scores": {k: round(v, 4) for k, v in self.agent_scores.items()},
            "disagreement_score": round(self.disagreement_score, 4),
            "bucket": self.bucket,
            "won": self.won,
            "pnl": round(self.pnl, 2),
            "timestamp": self.timestamp,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "TradeDisagreementRecord":
        return cls(
            agent_scores=d.get("agent_scores", {}),
            disagreement_score=d.get("disagreement_score", 0.0),
            bucket=d.get("bucket", "low"),
            won=d.get("won", False),
            pnl=d.get("pnl", 0.0),
            timestamp=d.get("timestamp", ""),
        )


@dataclass
class AgentPairConflict:
    """Tracks conflict stats between two agents."""
    agent_a: str
    agent_b: str
    conflict_count: int = 0
    loss_count: int = 0
    total_pnl: float = 0.0

    @property
    def loss_rate(self) -> float:
        return self.loss_count / self.conflict_count if self.conflict_count > 0 else 0.0

    @property
    def is_toxic(self) -> bool:
        return (
            self.conflict_count >= _TOXIC_MIN_OBSERVATIONS
            and self.loss_rate >= _TOXIC_LOSS_RATE_THRESHOLD
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "agent_a": self.agent_a,
            "agent_b": self.agent_b,
            "conflict_count": self.conflict_count,
            "loss_count": self.loss_count,
            "loss_rate": round(self.loss_rate, 4),
            "total_pnl": round(self.total_pnl, 2),
            "is_toxic": self.is_toxic,
        }


class DisagreementTracker:
    """Tracks agent disagreement patterns and correlates with trade outcomes.

    Implements a feedback loop:
        record_trade() → accumulate → analyze (every 50 trades) → penalties
    """

    def __init__(
        self,
        persistence_path: str = _DEFAULT_PERSISTENCE_PATH,
        reanalyze_interval: int = _REANALYZE_INTERVAL,
        toxic_threshold: float = _TOXIC_LOSS_RATE_THRESHOLD,
        min_observations: int = _TOXIC_MIN_OBSERVATIONS,
        max_penalty: float = _MAX_CONFIDENCE_PENALTY,
    ):
        self.persistence_path = persistence_path
        self.reanalyze_interval = reanalyze_interval
        self.toxic_threshold = toxic_threshold
        self.min_observations = min_observations
        self.max_penalty = max_penalty

        self._records: List[TradeDisagreementRecord] = []
        self._bucket_stats: Dict[str, Dict[str, int]] = {
            b[2]: {"wins": 0, "losses": 0} for b in _DISAGREEMENT_BUCKETS
        }
        self._pair_conflicts: Dict[Tuple[str, str], AgentPairConflict] = {}
        self._toxic_patterns: List[AgentPairConflict] = []
        self._trades_since_analysis = 0
        self._total_trades = 0

        self._load_state()

    # ----- Core API -----

    def record_trade(
        self,
        agent_scores: Dict[str, float],
        won: bool,
        pnl: float = 0.0,
    ) -> None:
        """Record a closed trade's agent score vector and outcome."""
        if not agent_scores or len(agent_scores) < 2:
            return

        # Normalize scores to 0-1
        scores = {k: max(0.0, min(1.0, float(v))) for k, v in agent_scores.items()}

        # Compute disagreement score
        d_score = self._compute_disagreement(scores)
        bucket = self._classify_bucket(d_score)

        record = TradeDisagreementRecord(
            agent_scores=scores,
            disagreement_score=d_score,
            bucket=bucket,
            won=won,
            pnl=pnl,
            timestamp=datetime.now(timezone.utc).isoformat(),
        )
        self._records.append(record)

        # Update bucket stats
        if won:
            self._bucket_stats[bucket]["wins"] += 1
        else:
            self._bucket_stats[bucket]["losses"] += 1

        # Update pairwise conflict tracking
        self._update_pair_conflicts(scores, won, pnl)

        self._trades_since_analysis += 1
        self._total_trades += 1

        # Re-analyze periodically
        if self._trades_since_analysis >= self.reanalyze_interval:
            self._analyze_patterns()
            self._trades_since_analysis = 0
            self.save_state()

    # Alias expected by acceptance criteria (US-343)
    def record_outcome(
        self,
        agent_scores: Dict[str, float],
        won: bool,
        pnl: float = 0.0,
    ) -> None:
        """Alias for record_trade() — matches US-343 acceptance criteria naming."""
        return self.record_trade(agent_scores=agent_scores, won=won, pnl=pnl)

    def get_confidence_penalty(self, agent_scores: Dict[str, float]) -> float:
        """Return a confidence penalty (negative float) if toxic patterns are active.

        Returns 0.0 if no toxic patterns match, or a negative value up to -max_penalty.
        """
        if not self._toxic_patterns or not agent_scores or len(agent_scores) < 2:
            return 0.0

        scores = {k: max(0.0, min(1.0, float(v))) for k, v in agent_scores.items()}
        penalty = 0.0

        for toxic in self._toxic_patterns:
            a, b = toxic.agent_a, toxic.agent_b
            if a in scores and b in scores:
                # Check if these two agents are currently in disagreement
                # Disagreement = one is high (>0.6) and other is low (<0.4)
                score_a, score_b = scores[a], scores[b]
                if (score_a > 0.6 and score_b < 0.4) or (score_b > 0.6 and score_a < 0.4):
                    # Scale penalty by loss rate and conflict count confidence
                    confidence_factor = min(toxic.conflict_count / 20.0, 1.0)
                    pair_penalty = toxic.loss_rate * confidence_factor * self.max_penalty
                    penalty += pair_penalty
                    logger.info(
                        "Phase 55 (US-343): Toxic disagreement detected: %s vs %s "
                        "(scores=%.2f/%.2f, loss_rate=%.0f%%, penalty=%.4f)",
                        a, b, score_a, score_b, toxic.loss_rate * 100, pair_penalty,
                    )

        # Cap total penalty
        return -min(penalty, self.max_penalty)

    def generate_report(self) -> Dict[str, Any]:
        """Generate a disagreement analysis report."""
        bucket_report = {}
        for bucket_name, stats in self._bucket_stats.items():
            total = stats["wins"] + stats["losses"]
            win_rate = stats["wins"] / total if total > 0 else 0.0
            bucket_report[bucket_name] = {
                "total": total,
                "wins": stats["wins"],
                "losses": stats["losses"],
                "win_rate": round(win_rate, 4),
            }

        toxic_report = [t.to_dict() for t in self._toxic_patterns]

        return {
            "total_trades": self._total_trades,
            "bucket_win_rates": bucket_report,
            "toxic_patterns": toxic_report,
            "toxic_count": len(self._toxic_patterns),
            "total_pair_conflicts_tracked": len(self._pair_conflicts),
            "last_analysis_at": datetime.now(timezone.utc).isoformat(),
        }

    # ----- Internal Analysis -----

    @staticmethod
    def _compute_disagreement(scores: Dict[str, float]) -> float:
        """Compute disagreement = std(scores) / mean(scores).

        Returns 0.0 if mean is 0 (all agents scored 0).
        """
        if not scores:
            return 0.0
        vals = list(scores.values())
        n = len(vals)
        if n < 2:
            return 0.0

        mean = sum(vals) / n
        if mean <= 0.0:
            return 0.0

        variance = sum((v - mean) ** 2 for v in vals) / n
        std = variance ** 0.5
        return std / mean

    @staticmethod
    def _classify_bucket(d_score: float) -> str:
        """Classify disagreement score into a bucket."""
        for low, high, name in _DISAGREEMENT_BUCKETS:
            if low <= d_score < high:
                return name
        return "extreme"

    def _update_pair_conflicts(
        self,
        scores: Dict[str, float],
        won: bool,
        pnl: float,
    ) -> None:
        """Track pairwise agent conflicts."""
        agents = sorted(scores.keys())
        for a, b in combinations(agents, 2):
            score_a, score_b = scores[a], scores[b]
            # Conflict = one scores high (>0.6) and other scores low (<0.4)
            if (score_a > 0.6 and score_b < 0.4) or (score_b > 0.6 and score_a < 0.4):
                key = (a, b)
                if key not in self._pair_conflicts:
                    self._pair_conflicts[key] = AgentPairConflict(agent_a=a, agent_b=b)
                conflict = self._pair_conflicts[key]
                conflict.conflict_count += 1
                conflict.total_pnl += pnl
                if not won:
                    conflict.loss_count += 1

    def _analyze_patterns(self) -> None:
        """Re-analyze toxic patterns from accumulated data."""
        self._toxic_patterns = [
            conflict for conflict in self._pair_conflicts.values()
            if conflict.is_toxic
        ]
        if self._toxic_patterns:
            logger.info(
                "Phase 55 (US-343): Found %d toxic disagreement patterns from %d pair conflicts",
                len(self._toxic_patterns), len(self._pair_conflicts),
            )
            for tp in self._toxic_patterns[:5]:
                logger.info(
                    "  Toxic: %s vs %s — %d conflicts, %.0f%% loss rate, PnL=%.2f",
                    tp.agent_a, tp.agent_b, tp.conflict_count,
                    tp.loss_rate * 100, tp.total_pnl,
                )

    # ----- Persistence -----

    def save_state(self) -> None:
        """Persist disagreement state with resilient save."""
        path = self.persistence_path
        try:
            os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
            data = {
                "version": 1,
                "updated_at": datetime.now(timezone.utc).isoformat(),
                "total_trades": self._total_trades,
                "trades_since_analysis": self._trades_since_analysis,
                "bucket_stats": self._bucket_stats,
                "pair_conflicts": {
                    f"{k[0]}:{k[1]}": {
                        "agent_a": v.agent_a,
                        "agent_b": v.agent_b,
                        "conflict_count": v.conflict_count,
                        "loss_count": v.loss_count,
                        "total_pnl": v.total_pnl,
                    }
                    for k, v in self._pair_conflicts.items()
                },
                "toxic_patterns": [t.to_dict() for t in self._toxic_patterns],
                "recent_records": [r.to_dict() for r in self._records[-200:]],
            }
            try:
                from src.scanner.safe_io import resilient_save
                resilient_save(path, data, context="disagreement_tracker")
            except ImportError:
                tmp_path = path + ".tmp"
                with open(tmp_path, "w") as f:
                    json.dump(data, f, indent=2, sort_keys=True)
                os.rename(tmp_path, path)
        except Exception as e:
            logger.error("Phase 55 (US-343): Failed to save disagreement state: %s", e)

    def _load_state(self) -> None:
        """Load persisted state."""
        path = self.persistence_path
        try:
            if not os.path.exists(path):
                return
            with open(path, "r") as f:
                data = json.load(f)
            if not isinstance(data, dict):
                return

            self._total_trades = data.get("total_trades", 0)
            self._trades_since_analysis = data.get("trades_since_analysis", 0)

            raw_bucket = data.get("bucket_stats", {})
            if isinstance(raw_bucket, dict):
                for k, v in raw_bucket.items():
                    if k in self._bucket_stats and isinstance(v, dict):
                        self._bucket_stats[k] = {
                            "wins": v.get("wins", 0),
                            "losses": v.get("losses", 0),
                        }

            raw_conflicts = data.get("pair_conflicts", {})
            if isinstance(raw_conflicts, dict):
                for key_str, v in raw_conflicts.items():
                    parts = key_str.split(":", 1)
                    if len(parts) == 2 and isinstance(v, dict):
                        key = (parts[0], parts[1])
                        self._pair_conflicts[key] = AgentPairConflict(
                            agent_a=v.get("agent_a", parts[0]),
                            agent_b=v.get("agent_b", parts[1]),
                            conflict_count=v.get("conflict_count", 0),
                            loss_count=v.get("loss_count", 0),
                            total_pnl=v.get("total_pnl", 0.0),
                        )

            raw_toxic = data.get("toxic_patterns", [])
            if isinstance(raw_toxic, list):
                self._toxic_patterns = [
                    AgentPairConflict(
                        agent_a=t.get("agent_a", ""),
                        agent_b=t.get("agent_b", ""),
                        conflict_count=t.get("conflict_count", 0),
                        loss_count=t.get("loss_count", 0),
                        total_pnl=t.get("total_pnl", 0.0),
                    )
                    for t in raw_toxic if isinstance(t, dict)
                ]

            raw_records = data.get("recent_records", [])
            if isinstance(raw_records, list):
                self._records = [
                    TradeDisagreementRecord.from_dict(r)
                    for r in raw_records if isinstance(r, dict)
                ]

            logger.info(
                "Phase 55 (US-343): Loaded disagreement state — %d trades, %d pair conflicts, %d toxic",
                self._total_trades, len(self._pair_conflicts), len(self._toxic_patterns),
            )
        except (json.JSONDecodeError, OSError) as e:
            logger.warning("Phase 55 (US-343): Could not load disagreement state: %s", e)
