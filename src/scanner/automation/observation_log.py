"""Observation logging for interesting market patterns.

Captures observations during scans even when no trade is taken.
Implementation target: PRD US-008.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional


class ObservationLog:
    """Logs market observations for future pattern matching."""

    LOG_FILE = "trained_data/observations.jsonl"

    # Categories for structured observation tagging
    CATEGORIES = {
        "regime_shift",     # Volatility regime changed
        "correlation",      # Unusual pair correlation detected
        "divergence",       # Model/agent disagreement on direction
        "breakout",         # Price broke S/R level
        "session_anomaly",  # Unusual session timing behavior
        "agent_conflict",   # Agents strongly disagree
        "near_miss",        # Almost tradeable but gated out
        "pattern",          # Recurring price pattern detected
    }

    def __init__(self, project_root: Optional[str] = None):
        self._root = Path(project_root) if project_root else Path.cwd()
        self._log_path = self._root / self.LOG_FILE
        self._log_path.parent.mkdir(parents=True, exist_ok=True)

    def log_observation(
        self,
        pair: str,
        category: str,
        description: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Log a market observation.

        Args:
            pair: Currency pair (e.g., "EUR_USD")
            category: One of CATEGORIES or free-form string
            description: Human-readable description of the observation
            metadata: Optional structured data (scores, prices, etc.)
        """
        entry = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "pair": pair,
            "category": category,
            "description": description,
            "metadata": metadata or {},
        }
        with open(self._log_path, "a") as f:
            f.write(json.dumps(entry) + "\n")

    def get_recent(
        self,
        pair: Optional[str] = None,
        category: Optional[str] = None,
        limit: int = 20,
    ) -> List[Dict[str, Any]]:
        """Get recent observations with optional filters.

        Args:
            pair: Filter by pair (None = all pairs)
            category: Filter by category (None = all categories)
            limit: Max entries to return (newest first)

        Returns:
            List of observation dicts, newest first.
        """
        if not self._log_path.exists():
            return []

        entries = []
        for line in self._log_path.read_text().strip().split("\n"):
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue
            if pair and entry.get("pair") != pair:
                continue
            if category and entry.get("category") != category:
                continue
            entries.append(entry)

        # Return newest first, limited
        return list(reversed(entries[-limit:]))

    def get_pair_summary(self, pair: str) -> Dict[str, Any]:
        """Get observation summary for a specific pair.

        Returns:
            Dict with category counts, recent observations, and patterns.
        """
        observations = self.get_recent(pair=pair, limit=100)
        if not observations:
            return {"pair": pair, "total": 0, "categories": {}}

        categories: Dict[str, int] = {}
        for obs in observations:
            cat = obs.get("category", "unknown")
            categories[cat] = categories.get(cat, 0) + 1

        return {
            "pair": pair,
            "total": len(observations),
            "categories": categories,
            "latest": observations[0] if observations else None,
            "oldest": observations[-1] if observations else None,
        }

    def log_scan_observations(self, scan_analyses: list) -> int:
        """Extract and log observations from a scan result.

        Automatically detects interesting patterns from scan analyses
        and logs them. Returns count of observations logged.

        Args:
            scan_analyses: List of PairAnalysis-like dicts or objects

        Returns:
            Number of observations logged.
        """
        count = 0
        for a in scan_analyses:
            pair = getattr(a, "pair", None) or a.get("pair", "UNKNOWN") if isinstance(a, dict) else getattr(a, "pair", "UNKNOWN")
            confidence = getattr(a, "confidence", 0) if not isinstance(a, dict) else a.get("confidence", 0)
            gates = getattr(a, "gates_passed", False) if not isinstance(a, dict) else a.get("gates_passed", False)
            vote = getattr(a, "weighted_vote_score", 0) if not isinstance(a, dict) else a.get("weighted_vote_score", 0)
            regime = getattr(a, "volatility_regime", "") if not isinstance(a, dict) else a.get("volatility_regime", "")

            # Near miss: high confidence but gates failed
            if confidence > 0.55 and not gates:
                self.log_observation(
                    pair=pair,
                    category="near_miss",
                    description=f"High confidence ({confidence:.1%}) but gates failed. Vote: {vote:.1%}",
                    metadata={"confidence": confidence, "vote_score": vote, "regime": str(regime)},
                )
                count += 1

            # Regime anomaly: EXTREME volatility
            if str(regime).upper() == "EXTREME":
                self.log_observation(
                    pair=pair,
                    category="regime_shift",
                    description=f"EXTREME volatility regime detected",
                    metadata={"regime": str(regime), "confidence": confidence},
                )
                count += 1

        return count
