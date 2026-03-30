"""
Dynamic Gate Threshold Optimizer — Phase 54 (US-334).

Consumes GateAttributionEngine reports to auto-tune gate thresholds.
Gates killing good trades (good_rejection_rate < 0.30) get relaxed;
gates protecting profit (good_rejection_rate > 0.70) get tightened.

Safeguards:
  - Max ±15% adjustment per cycle
  - Minimum 50 rejections before adjusting (prevents small-sample bias)
  - Adjustment history persisted for audit trail
  - Recommendations logged before application

Usage:
    optimizer = GateThresholdOptimizer()
    recommendations = optimizer.compute_recommendations(gate_report)
    applied = optimizer.apply_recommendations(config)
"""

import json
import logging
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

_DEFAULT_PERSISTENCE_PATH = "trained_data/gate_threshold_history.json"


@dataclass
class ThresholdRecommendation:
    """Single gate threshold adjustment recommendation."""
    gate_name: str = ""
    current_threshold: float = 0.0
    recommended_threshold: float = 0.0
    adjustment_pct: float = 0.0
    reason: str = ""
    good_rejection_rate: float = 0.0
    evidence_count: int = 0
    applied: bool = False
    timestamp: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "gate_name": self.gate_name,
            "current_threshold": round(self.current_threshold, 4),
            "recommended_threshold": round(self.recommended_threshold, 4),
            "adjustment_pct": round(self.adjustment_pct, 4),
            "reason": self.reason,
            "good_rejection_rate": round(self.good_rejection_rate, 4),
            "evidence_count": self.evidence_count,
            "applied": self.applied,
            "timestamp": self.timestamp,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ThresholdRecommendation":
        return cls(
            gate_name=str(d.get("gate_name", "")),
            current_threshold=float(d.get("current_threshold", 0.0)),
            recommended_threshold=float(d.get("recommended_threshold", 0.0)),
            adjustment_pct=float(d.get("adjustment_pct", 0.0)),
            reason=str(d.get("reason", "")),
            good_rejection_rate=float(d.get("good_rejection_rate", 0.0)),
            evidence_count=int(d.get("evidence_count", 0)),
            applied=bool(d.get("applied", False)),
            timestamp=str(d.get("timestamp", "")),
        )


@dataclass
class GateThresholdOptimizerConfig:
    """Configuration for gate threshold optimizer."""
    relax_threshold: float = 0.30       # good_rejection_rate below this → relax gate
    tighten_threshold: float = 0.70     # good_rejection_rate above this → tighten gate
    relax_pct: float = 0.10             # Relax by 10% when killing good trades
    tighten_pct: float = 0.05           # Tighten by 5% when protecting profit
    max_adjustment_pct: float = 0.15    # Cap any single adjustment to ±15%
    min_evidence: int = 50              # Minimum rejections before adjusting
    max_history_per_gate: int = 10      # Keep last 10 adjustments per gate
    persistence_path: str = _DEFAULT_PERSISTENCE_PATH
    auto_apply: bool = False            # If True, apply without manual review


# Known gates and their default threshold field names
_GATE_THRESHOLD_MAP = {
    "drawdown_adapter": {"field": "halt_threshold", "default": 0.08, "direction": "lower_is_stricter"},
    "spread_filter": {"field": "max_spread_atr_pct", "default": 0.20, "direction": "lower_is_stricter"},
    "calendar_filter": {"field": "blackout_minutes_high", "default": 30, "direction": "higher_is_stricter"},
    "fitness_detector": {"field": "min_fitness_score", "default": 0.40, "direction": "higher_is_stricter"},
    "setup_quality": {"field": "min_quality_score", "default": 0.35, "direction": "higher_is_stricter"},
    "staleness_guard": {"field": "stale_threshold", "default": 15.0, "direction": "lower_is_stricter"},
    "pattern_gate": {"field": "confidence_boost", "default": 0.05, "direction": "higher_is_more_permissive"},
}


class GateThresholdOptimizer:
    """Auto-tunes gate thresholds based on attribution data.

    Consumes GateAttributionEngine reports and recommends threshold
    adjustments to reduce false rejections while preserving profitable gates.
    """

    def __init__(self, config: Optional[GateThresholdOptimizerConfig] = None):
        self.config = config or GateThresholdOptimizerConfig()
        self._history: Dict[str, List[ThresholdRecommendation]] = {}
        self._current_thresholds: Dict[str, float] = {}
        # Phase 50 (US-308-ext): Agent feedback log for attribution integration
        self._agent_feedback: List[Dict[str, Any]] = []
        self._load_state()

    def compute_recommendations(
        self,
        gate_report: Dict[str, Any],
        current_thresholds: Optional[Dict[str, float]] = None,
    ) -> List[ThresholdRecommendation]:
        """Compute threshold adjustment recommendations from gate attribution report.

        Args:
            gate_report: Output from GateAttributionEngine.generate_report()
                         {gate_name: {rejections, passes, rejection_pct, good_rejection_rate, ...}}
            current_thresholds: {gate_name: current_threshold_value}

        Returns:
            List of ThresholdRecommendation objects
        """
        if current_thresholds:
            self._current_thresholds.update(current_thresholds)

        recommendations = []
        now = datetime.now(timezone.utc).isoformat()

        for gate_name, stats in gate_report.items():
            if not isinstance(stats, dict):
                continue

            rejections = int(stats.get("rejections", 0))
            good_rejection_rate = float(stats.get("good_rejection_rate", 0.5))

            # Skip gates with insufficient evidence
            if rejections < self.config.min_evidence:
                continue

            # Get current threshold
            gate_info = _GATE_THRESHOLD_MAP.get(gate_name, {})
            current = self._current_thresholds.get(
                gate_name,
                gate_info.get("default", 0.0),
            )

            if current == 0.0:
                continue  # No threshold to adjust

            direction = gate_info.get("direction", "higher_is_stricter")
            rec = None

            if good_rejection_rate < self.config.relax_threshold:
                # Gate is killing good trades → relax
                adj_pct = min(self.config.relax_pct, self.config.max_adjustment_pct)

                if direction == "higher_is_stricter":
                    # Lower threshold = more permissive
                    new_threshold = current * (1 - adj_pct)
                elif direction == "lower_is_stricter":
                    # Higher threshold = more permissive
                    new_threshold = current * (1 + adj_pct)
                else:
                    new_threshold = current * (1 + adj_pct)

                rec = ThresholdRecommendation(
                    gate_name=gate_name,
                    current_threshold=current,
                    recommended_threshold=new_threshold,
                    adjustment_pct=adj_pct,
                    reason=f"Killing good trades: good_rejection_rate={good_rejection_rate:.3f} < {self.config.relax_threshold}",
                    good_rejection_rate=good_rejection_rate,
                    evidence_count=rejections,
                    timestamp=now,
                )

            elif good_rejection_rate > self.config.tighten_threshold:
                # Gate is protecting profit → tighten slightly
                adj_pct = min(self.config.tighten_pct, self.config.max_adjustment_pct)

                if direction == "higher_is_stricter":
                    new_threshold = current * (1 + adj_pct)
                elif direction == "lower_is_stricter":
                    new_threshold = current * (1 - adj_pct)
                else:
                    new_threshold = current * (1 - adj_pct)

                rec = ThresholdRecommendation(
                    gate_name=gate_name,
                    current_threshold=current,
                    recommended_threshold=new_threshold,
                    adjustment_pct=-adj_pct,
                    reason=f"Protecting profit: good_rejection_rate={good_rejection_rate:.3f} > {self.config.tighten_threshold}",
                    good_rejection_rate=good_rejection_rate,
                    evidence_count=rejections,
                    timestamp=now,
                )

            if rec is not None:
                recommendations.append(rec)
                logger.info(
                    "Phase 54 (US-334): Gate '%s' recommendation: %.4f → %.4f (%+.1f%%) — %s",
                    rec.gate_name, rec.current_threshold,
                    rec.recommended_threshold, rec.adjustment_pct * 100,
                    rec.reason,
                )

        return recommendations

    def apply_recommendations(
        self,
        recommendations: List[ThresholdRecommendation],
    ) -> List[ThresholdRecommendation]:
        """Apply recommendations and record in history.

        Returns the list of applied recommendations.
        """
        applied = []
        for rec in recommendations:
            if rec.evidence_count < self.config.min_evidence:
                continue

            # Record in history
            if rec.gate_name not in self._history:
                self._history[rec.gate_name] = []

            rec.applied = True
            self._history[rec.gate_name].append(rec)

            # Trim history
            max_hist = self.config.max_history_per_gate
            if len(self._history[rec.gate_name]) > max_hist:
                self._history[rec.gate_name] = self._history[rec.gate_name][-max_hist:]

            # Update current thresholds
            self._current_thresholds[rec.gate_name] = rec.recommended_threshold

            applied.append(rec)
            logger.info(
                "Phase 54 (US-334): Applied gate adjustment: %s %.4f → %.4f",
                rec.gate_name, rec.current_threshold, rec.recommended_threshold,
            )

        return applied

    def get_recommendations_report(
        self, gate_report: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Generate full recommendations report.

        Returns:
            {gate_name: {current_threshold, recommended_threshold, reason, evidence_count}}
        """
        recs = self.compute_recommendations(gate_report)
        report = {}
        for rec in recs:
            report[rec.gate_name] = rec.to_dict()
        return report

    def get_adjustment_history(self, gate_name: Optional[str] = None) -> Dict[str, Any]:
        """Get adjustment history for one or all gates."""
        if gate_name:
            entries = self._history.get(gate_name, [])
            return {gate_name: [r.to_dict() for r in entries]}

        result = {}
        for gn, entries in self._history.items():
            result[gn] = [r.to_dict() for r in entries]
        return result

    def get_current_thresholds(self) -> Dict[str, float]:
        """Return current threshold values (possibly adjusted)."""
        return dict(self._current_thresholds)

    def record_agent_feedback(
        self,
        agent_name: str,
        weight_delta: float,
        source: str = "unknown",
    ) -> None:
        """Phase 50 (US-308-ext): Record agent weight feedback from attribution engine.

        Integrates attribution weight recommendations into threshold optimization.
        Weight deltas > 1.0 indicate agent should be weighted more; < 1.0 indicate less.

        Args:
            agent_name: Name of the agent (e.g., "trend_detector", "momentum_agent")
            weight_delta: Recommended weight multiplier (e.g., 1.15 = increase 15%)
            source: Source of the feedback (e.g., "trade_attribution")
        """
        try:
            feedback = {
                "agent_name": str(agent_name),
                "weight_delta": float(weight_delta),
                "source": str(source),
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
            self._agent_feedback.append(feedback)

            # Trim history to last 200 entries
            if len(self._agent_feedback) > 200:
                self._agent_feedback = self._agent_feedback[-100:]

            direction = "increase" if weight_delta > 1.0 else "decrease"
            magnitude = abs((weight_delta - 1.0) * 100)
            logger.debug(
                "Phase 50 (US-308-ext): Recorded agent feedback: %s %s %.1f%% (delta=%.3f)",
                agent_name, direction, magnitude, weight_delta,
            )
        except Exception as e:
            logger.debug(f"Phase 50: Failed to record agent feedback: {e}")

    def get_agent_feedback(self) -> List[Dict[str, Any]]:
        """Return list of recorded agent feedback entries."""
        return list(self._agent_feedback)

    def save_state(self) -> None:
        """Persist optimizer state."""
        path = self.config.persistence_path
        try:
            os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
            data = {
                "version": 1,
                "updated_at": datetime.now(timezone.utc).isoformat(),
                "current_thresholds": {
                    k: round(v, 6) for k, v in self._current_thresholds.items()
                },
                "history": {
                    gate: [r.to_dict() for r in recs[-self.config.max_history_per_gate:]]
                    for gate, recs in self._history.items()
                },
                # Phase 50 (US-308-ext): Persist agent feedback
                "agent_feedback": self._agent_feedback[-100:] if self._agent_feedback else [],
            }
            tmp_path = path + ".tmp"
            with open(tmp_path, "w") as f:
                json.dump(data, f, indent=2, sort_keys=True)
            os.rename(tmp_path, path)
        except Exception as e:
            logger.error("Phase 54 (US-334): Failed to save gate threshold optimizer: %s", e)

    def _load_state(self) -> None:
        """Load persisted state."""
        path = self.config.persistence_path
        if not os.path.exists(path):
            return
        try:
            with open(path, "r") as f:
                data = json.load(f)
            if isinstance(data, dict):
                thresholds = data.get("current_thresholds", {})
                if isinstance(thresholds, dict):
                    self._current_thresholds = {
                        k: float(v) for k, v in thresholds.items()
                        if isinstance(v, (int, float))
                    }
                history = data.get("history", {})
                if isinstance(history, dict):
                    for gate, recs in history.items():
                        if isinstance(recs, list):
                            self._history[gate] = [
                                ThresholdRecommendation.from_dict(r)
                                for r in recs if isinstance(r, dict)
                            ]
                # Phase 50 (US-308-ext): Load agent feedback
                feedback = data.get("agent_feedback", [])
                if isinstance(feedback, list):
                    self._agent_feedback = [
                        f for f in feedback if isinstance(f, dict)
                    ]
                logger.info(
                    "Phase 54 (US-334): Loaded gate threshold optimizer — %d gates, %d thresholds, %d agent feedbacks",
                    len(self._history), len(self._current_thresholds), len(self._agent_feedback),
                )
        except (json.JSONDecodeError, OSError) as e:
            logger.warning("Phase 54 (US-334): Could not load threshold optimizer: %s", e)
