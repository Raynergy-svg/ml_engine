"""Observation logging for interesting market patterns (even when no trade is taken).

US-008: Add observation logging for interesting market patterns.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

OBSERVATIONS_PATH = Path("trained_data/observations.jsonl")

OBSERVATION_CATEGORIES = [
    "regime_change",
    "unusual_spread",
    "agent_disagreement",
    "near_miss",
    "correlation_break",
]


class ObservationLog:
    """Captures interesting market observations during scans."""

    def __init__(self, log_path: Optional[Path] = None):
        self.log_path = log_path or OBSERVATIONS_PATH

    def log_observation(
        self,
        pair: str,
        category: str,
        description: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Append an observation to trained_data/observations.jsonl."""
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
