"""Observation logging for interesting market patterns (even when no trade is taken).

US-008: Add observation logging for interesting market patterns.
"""

from __future__ import annotations

import json
import logging
import glob as _glob
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

OBSERVATIONS_PATH = Path("trained_data/observations.jsonl")

# Phase 30 (US-182): Log rotation settings
_LOG_MAX_BYTES = 100_000  # 100KB trigger for rotation
_LOG_MAX_ROTATED_FILES = 7  # Keep at most 7 rotated files

OBSERVATION_CATEGORIES = [
    "regime_change",
    "unusual_spread",
    "agent_disagreement",
    "near_miss",
    "correlation_break",
    "group_momentum",       # US-059: cross-pair currency group signals
    "correlation_conflict",  # US-059: correlated pair double-exposure blocks
    "gate_degradation",     # US-063: gate model accuracy dropped below threshold
    "macro_stress",         # US-066: macro stress level elevated
]


class ObservationLog:
    """Captures interesting market observations during scans."""

    def __init__(self, log_path: Optional[Path] = None):
        self.log_path = log_path or OBSERVATIONS_PATH

    def _maybe_rotate(self) -> None:
        """Phase 30 (US-182): Rotate observation log when it exceeds size limit."""
        try:
            if not self.log_path.exists():
                return
            size = self.log_path.stat().st_size
            if size < _LOG_MAX_BYTES:
                return

            # Rotate to dated file
            ts = datetime.now(timezone.utc).strftime("%Y-%m-%d_%H%M%S")
            rotated = self.log_path.with_suffix(f".{ts}.jsonl")
            self.log_path.rename(rotated)
            logger.info("Observation log rotated: %s (%d bytes)", rotated.name, size)

            # Prune old rotated files (keep newest N)
            pattern = str(self.log_path.with_suffix("")) + ".*.jsonl"
            rotated_files = sorted(_glob.glob(pattern))
            while len(rotated_files) > _LOG_MAX_ROTATED_FILES:
                oldest = Path(rotated_files.pop(0))
                oldest.unlink(missing_ok=True)
                logger.info("Pruned old observation log: %s", oldest.name)
        except Exception as e:
            logger.debug(f"Observation log rotation skipped: {e}")

    def log_observation(
        self,
        pair: str,
        category: str,
        description: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Append an observation to trained_data/observations.jsonl."""
        self._maybe_rotate()  # Phase 30 (US-182)
        entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "pair": pair,
            "category": category,
            "description": description,
            "metadata": metadata or {},
        }
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.log_path, "a") as f:
            f.write(json.dumps(entry, default=str) + "\n")
        logger.debug("Observation: %s %s — %s", pair, category, description)

    def log_from_analysis(self, analysis: Any) -> int:
        """Detect and log observations from a scan analysis result.

        Checks for:
        1. Regime change (volatility regime != historical mode)
        2. Agent disagreement (agents split ~50/50)
        3. Near miss (confidence within 2% of threshold)

        Returns count of observations logged.
        """
        count = 0
        pair = getattr(analysis, "pair", "UNKNOWN")

        # Agent disagreement: agents split roughly 50/50
        agent_votes = getattr(analysis, "agent_votes", 0)
        agent_total = getattr(analysis, "agent_total", 0)
        if agent_total > 0 and agent_votes == agent_total // 2:
            self.log_observation(
                pair=pair,
                category="agent_disagreement",
                description=f"Agents split {agent_votes}/{agent_total}",
                metadata={"votes": agent_votes, "total": agent_total},
            )
            count += 1

        # Near miss: confidence within 2% of min_confidence
        confidence = getattr(analysis, "confidence", 0) or 0
        min_conf = 0.55  # default threshold
        if not getattr(analysis, "is_tradeable", False) and abs(confidence - min_conf) < 0.02:
            self.log_observation(
                pair=pair,
                category="near_miss",
                description=f"Confidence {confidence:.0%} within 2% of threshold {min_conf:.0%}",
                metadata={"confidence": confidence, "threshold": min_conf},
            )
            count += 1

        # Regime observation
        regime = str(getattr(analysis, "volatility_regime", "") or "")
        if regime in ("EXTREME", "HIGH"):
            self.log_observation(
                pair=pair,
                category="regime_change",
                description=f"Volatility regime: {regime}",
                metadata={"regime": regime},
            )
            count += 1

        return count

    def log_group_momentum(
        self,
        pair: str,
        action: str,
        alignment: float,
        base_score: float,
        quote_score: float,
        reason: str = "",
    ) -> None:
        """Log a group momentum observation (block, boost, or notable alignment).

        US-059: Captures cross-pair currency group signals for pattern analysis.
        """
        self.log_observation(
            pair=pair,
            category="group_momentum",
            description=f"Group momentum {action}: alignment={alignment:.2f} ({reason})",
            metadata={
                "action": action,
                "alignment": alignment,
                "base_score": base_score,
                "quote_score": quote_score,
                "reason": reason,
            },
        )

    def log_correlation_conflict(
        self,
        pair: str,
        conflicting_pair: str,
        correlation: float,
        reason: str = "",
    ) -> None:
        """Log a correlation conflict observation (double-exposure block).

        US-059: Captures when correlated pairs would create excessive exposure.
        """
        self.log_observation(
            pair=pair,
            category="correlation_conflict",
            description=(
                f"Correlation conflict with {conflicting_pair} "
                f"(r={correlation:.2f}): {reason}"
            ),
            metadata={
                "conflicting_pair": conflicting_pair,
                "correlation": correlation,
                "reason": reason,
            },
        )

    def get_observation_summary(
        self, limit: int = 100
    ) -> Dict[str, int]:
        """Get category counts from recent observations.

        Returns:
            Dict mapping category name → count of observations.
        """
        recent = self.get_recent(limit=limit)
        counts: Dict[str, int] = {}
        for entry in recent:
            cat = entry.get("category", "unknown")
            counts[cat] = counts.get(cat, 0) + 1
        return counts

    def get_recent(
        self,
        pair: Optional[str] = None,
        category: Optional[str] = None,
        limit: int = 20,
    ) -> List[Dict[str, Any]]:
        """Query recent observations with optional filters."""
        if not self.log_path.exists():
            return []

        results: List[Dict[str, Any]] = []
        for line in reversed(self.log_path.read_text().strip().split("\n")):
            if not line.strip():
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue

            if pair and entry.get("pair") != pair:
                continue
            if category and entry.get("category") != category:
                continue

            results.append(entry)
            if len(results) >= limit:
                break

        return results
