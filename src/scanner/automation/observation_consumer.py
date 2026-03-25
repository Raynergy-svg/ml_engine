"""Observation consumer with pattern detection and alerts.

US-119: Analyze the 97+ observations from observation_log.py. Detect
recurring patterns, generate config recommendations, and raise alerts
when observation categories spike.

Approach:
1. Read observations.jsonl (near_miss, regime_change, agent_disagreement, etc.)
2. Categorize and count by type, pair, regime
3. Detect patterns: frequency spikes, temporal clustering
4. Recommend config adjustments based on patterns
5. Alert when category count exceeds threshold in time window
"""

from __future__ import annotations

import json
import logging
from collections import Counter, defaultdict
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
_OBS_PATH = _PROJECT_ROOT / "trained_data" / "observations.jsonl"
_LOG_PATH = _PROJECT_ROOT / "trained_data" / "observation_consumer_log.json"

# Pattern detection parameters
MIN_PATTERN_COUNT = 3      # Minimum occurrences to be a pattern
PATTERN_WINDOW_HOURS = 24  # Time window for pattern detection
SPIKE_THRESHOLD = 5        # N occurrences in window = spike alert

# Recommendation mapping: observation type → config adjustment suggestion
OBSERVATION_RECOMMENDATIONS = {
    "near_miss": {
        "action": "lower_confidence_threshold",
        "delta": -0.01,
        "reason": "Frequent near-misses suggest threshold is slightly too tight",
    },
    "agent_disagreement": {
        "action": "increase_uncertainty_weight",
        "delta": 0.05,
        "reason": "Agent disagreement correlates with poor outcomes — weight uncertainty higher",
    },
    "correlation_conflict": {
        "action": "tighten_correlation_filter",
        "delta": -0.05,
        "reason": "Correlation conflicts suggest filter threshold too permissive",
    },
    "regime_change": {
        "action": "reduce_position_size",
        "delta": -0.10,
        "reason": "Frequent regime changes indicate unstable market — reduce exposure",
    },
    "group_momentum": {
        "action": "boost_momentum_gate",
        "delta": 0.05,
        "reason": "Group momentum signals being generated frequently — raise gate sensitivity",
    },
}


class ObservationConsumer:
    """Consumes and analyzes observations for pattern detection.

    Bridges observation_log.py (which writes but nobody reads) with
    actionable system insights. Detects recurring patterns, recommends
    config changes, and alerts on observation spikes.

    Usage:
        consumer = ObservationConsumer()
        consumer.consume_observations()
        patterns = consumer.detect_patterns()
        adjustments = consumer.recommend_adjustments()
        alerts = consumer.check_alerts(window_hours=6)
    """

    def __init__(
        self,
        observations_path: Optional[Path] = None,
        persistence_path: Optional[Path] = None,
    ):
        self.observations_path = observations_path or _OBS_PATH
        self.persistence_path = persistence_path or _LOG_PATH

        # Loaded observations
        self._observations: List[Dict[str, Any]] = []
        self._last_consumed_count: int = 0

        # Analysis results
        self._patterns: List[Dict[str, Any]] = []
        self._recommendations: List[Dict[str, Any]] = []
        self._alerts: List[Dict[str, Any]] = []
        self._total_consumed: int = 0

        self._load_state()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def consume_observations(
        self, path: Optional[Path] = None
    ) -> int:
        """Read and parse observations from jsonl file.

        Args:
            path: Override observations file path.

        Returns:
            Number of new observations consumed.
        """
        obs_path = path or self.observations_path
        if not obs_path.exists():
            return 0

        try:
            observations = []
            with open(obs_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            observations.append(json.loads(line))
                        except json.JSONDecodeError:
                            continue

            new_count = len(observations) - self._last_consumed_count
            self._observations = observations
            self._last_consumed_count = len(observations)
            self._total_consumed += max(0, new_count)

            return max(0, new_count)

        except Exception as e:
            logger.debug(f"ObservationConsumer: Failed to read: {e}")
            return 0

    def detect_patterns(self) -> List[Dict[str, Any]]:
        """Detect recurring observation patterns.

        Returns:
            List of detected patterns with type, count, and confidence.
        """
        if not self._observations:
            self.consume_observations()

        if not self._observations:
            return []

        patterns = []

        # Count by observation type
        type_counts = Counter()
        pair_type_counts: Dict[str, Counter] = defaultdict(Counter)
        regime_type_counts: Dict[str, Counter] = defaultdict(Counter)

        for obs in self._observations:
            obs_type = obs.get("type", obs.get("category", "unknown"))
            pair = obs.get("pair", "unknown")
            regime = obs.get("regime", "unknown")

            type_counts[obs_type] += 1
            pair_type_counts[pair][obs_type] += 1
            regime_type_counts[regime][obs_type] += 1

        # Detect type-level patterns
        total = len(self._observations) or 1
        for obs_type, count in type_counts.most_common():
            if count >= MIN_PATTERN_COUNT:
                patterns.append({
                    "type": obs_type,
                    "count": count,
                    "frequency": round(count / total, 4),
                    "level": "type",
                    "details": None,
                })

        # Detect pair-specific patterns
        for pair, counts in pair_type_counts.items():
            for obs_type, count in counts.most_common():
                if count >= MIN_PATTERN_COUNT:
                    patterns.append({
                        "type": obs_type,
                        "count": count,
                        "frequency": round(count / total, 4),
                        "level": "pair",
                        "details": {"pair": pair},
                    })

        # Detect regime-specific patterns
        for regime, counts in regime_type_counts.items():
            for obs_type, count in counts.most_common():
                if count >= MIN_PATTERN_COUNT:
                    patterns.append({
                        "type": obs_type,
                        "count": count,
                        "frequency": round(count / total, 4),
                        "level": "regime",
                        "details": {"regime": regime},
                    })

        # Sort by count descending
        patterns.sort(key=lambda p: p["count"], reverse=True)
        self._patterns = patterns[:30]  # Keep top 30

        return self._patterns

    def recommend_adjustments(self) -> List[Dict[str, Any]]:
        """Generate config change recommendations from detected patterns.

        Returns:
            List of recommended adjustments with action and reason.
        """
        if not self._patterns:
            self.detect_patterns()

        recommendations = []

        for pattern in self._patterns:
            obs_type = pattern["type"]
            if obs_type in OBSERVATION_RECOMMENDATIONS:
                rec = dict(OBSERVATION_RECOMMENDATIONS[obs_type])
                rec["pattern_count"] = pattern["count"]
                rec["pattern_frequency"] = pattern["frequency"]
                rec["pattern_level"] = pattern["level"]
                rec["pattern_details"] = pattern.get("details")
                rec["timestamp"] = datetime.now(timezone.utc).isoformat()
                recommendations.append(rec)

        self._recommendations = recommendations
        return recommendations

    def check_alerts(self, window_hours: int = 6) -> List[Dict[str, Any]]:
        """Check for observation category spikes in recent window.

        Args:
            window_hours: Time window to check for spikes.

        Returns:
            List of alerts for categories exceeding spike threshold.
        """
        if not self._observations:
            self.consume_observations()

        alerts = []
        cutoff = datetime.now(timezone.utc) - timedelta(hours=window_hours)

        # Count recent observations by type
        recent_counts: Counter = Counter()
        for obs in self._observations:
            timestamp_str = obs.get("timestamp", "")
            try:
                obs_time = datetime.fromisoformat(
                    timestamp_str.replace("Z", "+00:00")
                )
                if obs_time >= cutoff:
                    obs_type = obs.get("type", obs.get("category", "unknown"))
                    recent_counts[obs_type] += 1
            except (ValueError, TypeError):
                continue

        for obs_type, count in recent_counts.items():
            if count >= SPIKE_THRESHOLD:
                alerts.append({
                    "type": obs_type,
                    "count": count,
                    "window_hours": window_hours,
                    "severity": "critical" if count >= SPIKE_THRESHOLD * 2 else "warning",
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                })

        self._alerts = alerts
        return alerts

    def get_status(self) -> Dict[str, Any]:
        """Get consumer status for observability."""
        return {
            "total_consumed": self._total_consumed,
            "observations_loaded": len(self._observations),
            "patterns_detected": len(self._patterns),
            "recommendations": len(self._recommendations),
            "active_alerts": len(self._alerts),
            "recent_patterns": self._patterns[:5],
            "recent_alerts": self._alerts[:5],
        }

    # ------------------------------------------------------------------
    # Phase 23 (US-143): Immediate gate adjustments
    # ------------------------------------------------------------------

    def get_immediate_adjustments(
        self, current_cycle: int = 0
    ) -> Dict[str, Any]:
        """Produce real-time gate overrides based on recent observation patterns.

        Unlike recommend_adjustments() which produces passive config recommendations,
        this method returns concrete key-value overrides that should be applied to
        scanner.config BEFORE the next scan cycle.

        Rate-limited: max 1 adjustment per key per 5 cycles.

        Args:
            current_cycle: Current scan cycle number for rate-limiting.

        Returns:
            Dict of config key → new value to apply before next scan.
        """
        if not self._observations:
            self.consume_observations()

        adjustments: Dict[str, Any] = {}

        # Track rate limits: {key: last_cycle_applied}
        if not hasattr(self, "_adjustment_rate_limits"):
            self._adjustment_rate_limits: Dict[str, int] = {}

        def _rate_ok(key: str) -> bool:
            last = self._adjustment_rate_limits.get(key, -999)
            return (current_cycle - last) >= 5

        # Count recent patterns by category (last 20 observations)
        recent = self._observations[-20:] if len(self._observations) > 20 else self._observations
        category_counts: Counter = Counter()
        for obs in recent:
            cat = obs.get("category", obs.get("type", "unknown"))
            category_counts[cat] += 1

        # Pattern 1: Repeated agent_disagreement → lower vote threshold
        if category_counts.get("consensus_penalty", 0) + category_counts.get("agent_disagreement", 0) >= 3:
            if _rate_ok("sub_inference_vote_threshold"):
                adjustments["sub_inference_vote_threshold"] = 0.55  # from 0.66
                self._adjustment_rate_limits["sub_inference_vote_threshold"] = current_cycle
                logger.info(
                    "US-143: Lowering sub_inference_vote_threshold → 0.55 "
                    f"(agent disagreement count: {category_counts.get('consensus_penalty', 0) + category_counts.get('agent_disagreement', 0)})"
                )

        # Pattern 2: Near-misses (trade_block_reason with confidence close to threshold)
        near_miss_count = 0
        for obs in recent:
            if obs.get("category") == "trade_block_reason" or obs.get("type") == "trade_block_reason":
                meta = obs.get("metadata", {})
                conf = meta.get("confidence", 0) if isinstance(meta, dict) else 0
                if conf and 0.48 <= float(conf) <= 0.57:
                    near_miss_count += 1
        if near_miss_count >= 3:
            if _rate_ok("min_confidence"):
                # Temporarily lower min_confidence by 2 percentage points
                adjustments["min_confidence_offset"] = -2.0
                self._adjustment_rate_limits["min_confidence"] = current_cycle
                logger.info(
                    f"US-143: Applying min_confidence offset -2.0 "
                    f"(near-miss count: {near_miss_count})"
                )

        # Pattern 3: Repeated regime_default → system is always UNKNOWN
        if category_counts.get("regime_default", 0) >= 5:
            if _rate_ok("regime_consensus_map"):
                # When regime is always defaulting, loosen consensus for NORMAL
                adjustments["regime_consensus_map_normal"] = 0.25  # from 0.33
                self._adjustment_rate_limits["regime_consensus_map"] = current_cycle
                logger.info(
                    "US-143: Loosening NORMAL regime consensus → 0.25 "
                    f"(regime_default count: {category_counts.get('regime_default', 0)})"
                )

        return adjustments

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save_state(self) -> None:
        """Persist consumer state."""
        try:
            data = {
                "version": 1,
                "total_consumed": self._total_consumed,
                "last_consumed_count": self._last_consumed_count,
                "patterns": self._patterns[:30],
                "recommendations": self._recommendations[:20],
                "alerts": self._alerts[:20],
                "last_updated": datetime.now(timezone.utc).isoformat(),
            }
            self.persistence_path.parent.mkdir(parents=True, exist_ok=True)
            try:
                from src.scanner.automation.safe_json import safe_json_write
                safe_json_write(self.persistence_path, data)
            except ImportError:
                self.persistence_path.write_text(json.dumps(data, indent=2))
        except Exception as e:
            logger.debug(f"ObservationConsumer: Failed to save: {e}")

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
                self._total_consumed = data.get("total_consumed", 0)
                self._last_consumed_count = data.get("last_consumed_count", 0)
                self._patterns = data.get("patterns", [])
                self._recommendations = data.get("recommendations", [])
                self._alerts = data.get("alerts", [])
        except Exception as e:
            logger.debug(f"ObservationConsumer: Failed to load: {e}")
