"""Buddy's forward model of its own performance.

Reads the latest projections and produces a human-readable summary
of where Buddy thinks its accuracy is headed.

Used by: orchestrator status display, daily briefings.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional, List

logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
_PROJECTION_CACHE_PATH = _PROJECT_ROOT / "trained_data" / "drift_projections.json"


class SelfModelState:
    """Buddy's forward model of its own performance.

    Reads the latest drift projections and provides:
    1. Summary of current state across all pairs
    2. Identification of worst-performing pairs
    3. Flag for immediate action needed
    4. Human-readable risk assessment
    """

    CRITICAL_URGENCY_PAIRS_THRESHOLD = 1  # Alert if 1+ pair in critical state

    def __init__(self, projection_cache_path: Optional[Path] = None):
        """Initialize SelfModelState.

        Args:
            projection_cache_path: Path to drift_projections.json
        """
        self.projection_cache_path = projection_cache_path or _PROJECTION_CACHE_PATH

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_summary(self) -> Dict[str, Any]:
        """Return current self-model state across all pairs.

        Returns:
            Dict with:
            - timestamp: when projections were generated
            - pairs_monitored: number of pairs with projections
            - pairs_critical: list of pairs in critical urgency
            - pairs_warning: list of pairs in high/medium urgency
            - worst_pair: pair with most urgent degradation
            - hours_until_worst_critical: hours until worst pair hits critical
            - overall_health: "stable", "monitoring", "warning", or "critical"
            - summary: human-readable one-liner
        """
        projections = self._load_projections()

        if not projections:
            return {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "pairs_monitored": 0,
                "pairs_critical": [],
                "pairs_warning": [],
                "worst_pair": None,
                "hours_until_worst_critical": None,
                "overall_health": "unknown",
                "summary": "No projections available",
            }

        # Classify pairs by urgency
        critical_pairs = []
        warning_pairs = []

        for pair, proj in projections.items():
            urgency = proj.get("action_urgency", "low")
            if urgency == "critical":
                critical_pairs.append(pair)
            elif urgency in ["high", "medium"]:
                warning_pairs.append(pair)

        # Find worst pair
        worst_pair = None
        worst_hours = None
        worst_proj = None

        for pair, proj in projections.items():
            hours = proj.get("hours_until_critical")
            if hours is not None:
                if worst_hours is None or hours < worst_hours:
                    worst_pair = pair
                    worst_hours = hours
                    worst_proj = proj

        # Determine overall health
        if critical_pairs:
            overall_health = "critical"
        elif warning_pairs:
            overall_health = "warning"
        elif worst_hours is not None and worst_hours <= 24:
            overall_health = "monitoring"
        else:
            overall_health = "stable"

        # Generate summary
        summary = self._generate_summary(
            critical_pairs, warning_pairs, worst_pair, worst_hours, overall_health
        )

        return {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "pairs_monitored": len(projections),
            "pairs_critical": critical_pairs,
            "pairs_warning": warning_pairs,
            "worst_pair": worst_pair,
            "hours_until_worst_critical": worst_hours,
            "overall_health": overall_health,
            "summary": summary,
        }

    def get_worst_pair(self) -> Optional[str]:
        """Return the pair with the most urgent projected degradation.

        Returns:
            Pair name or None if all pairs stable.
        """
        projections = self._load_projections()

        if not projections:
            return None

        worst_pair = None
        worst_urgency_order = {"low": 0, "medium": 1, "high": 2, "critical": 3}

        for pair, proj in projections.items():
            urgency = proj.get("action_urgency", "low")
            action = proj.get("recommended_action", "monitor")

            # Rank by urgency first, then by hours_until_critical
            urgency_rank = worst_urgency_order.get(urgency, 0)

            if worst_pair is None:
                worst_pair = (pair, urgency_rank, proj.get("hours_until_critical"))
            else:
                current_rank = (worst_pair[1], worst_pair[2] or float("inf"))
                new_rank = (urgency_rank, proj.get("hours_until_critical") or float("inf"))

                if new_rank > current_rank:
                    worst_pair = (pair, urgency_rank, proj.get("hours_until_critical"))

        return worst_pair[0] if worst_pair else None

    def needs_immediate_action(self) -> bool:
        """Return True if any pair has critical urgency projection.

        Returns:
            True if any pair in critical state, False otherwise.
        """
        projections = self._load_projections()

        for pair, proj in projections.items():
            if proj.get("action_urgency") == "critical":
                return True

        return False

    def get_pair_detail(self, pair: str) -> Optional[Dict[str, Any]]:
        """Get detailed projection for a specific pair.

        Args:
            pair: Currency pair

        Returns:
            Projection dict or None if not found.
        """
        projections = self._load_projections()
        return projections.get(pair)

    # ------------------------------------------------------------------
    # Internal methods
    # ------------------------------------------------------------------

    def _load_projections(self) -> Dict[str, Dict[str, Any]]:
        """Load projection cache from JSON.

        Returns:
            Dict[pair, projection_dict]. Empty dict if unavailable.
        """
        if not self.projection_cache_path.exists():
            return {}

        try:
            try:
                from src.scanner.automation.safe_json import safe_json_read

                data = safe_json_read(self.projection_cache_path, default={})
            except ImportError:
                data = json.loads(
                    self.projection_cache_path.read_text(encoding="utf-8")
                )

            if not isinstance(data, dict):
                return {}

            projections = data.get("projections", {})
            if not isinstance(projections, dict):
                return {}

            return projections

        except Exception as e:
            logger.debug(f"SelfModelState: Failed to load projections: {e}")
            return {}

    def _generate_summary(
        self,
        critical_pairs: List[str],
        warning_pairs: List[str],
        worst_pair: Optional[str],
        worst_hours: Optional[int],
        overall_health: str,
    ) -> str:
        """Generate human-readable summary of system state.

        Args:
            critical_pairs: List of critical pairs
            warning_pairs: List of warning pairs
            worst_pair: Worst-performing pair
            worst_hours: Hours until worst pair critical
            overall_health: Overall health status

        Returns:
            Human-readable string
        """
        if overall_health == "critical":
            if critical_pairs:
                return f"CRITICAL: {len(critical_pairs)} pair(s) degrading rapidly ({', '.join(critical_pairs[:3])}). Immediate retrain required."
            return "CRITICAL: System accuracy degradation detected. Action required."

        if overall_health == "warning":
            return f"WARNING: {len(warning_pairs)} pair(s) showing drift. Monitor closely and prepare retrain."

        if overall_health == "monitoring":
            if worst_hours is not None:
                return f"MONITORING: {worst_pair} projected to degrade within {worst_hours}h. Prepare contingency."
            return "MONITORING: Mild drift detected on some pairs. Continue observation."

        return "STABLE: All pairs performing as expected. No immediate action required."
