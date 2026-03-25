"""Tier 3 Pattern Engine — Monthly narrative arc detection.

PRD v2.2 §7.1 Phase 4: "T3 pattern engine (monthly narrative arcs)"

T3 operates on the longest timescale — looking across weeks and months
to detect behavioral arcs that T1 (daily) and T2 (weekly) can't see.

A "narrative arc" is a multi-phase behavioral trajectory:
  1. BUILDING  — A trend is forming (e.g., gradually increasing stress)
  2. PEAK      — The pattern reaches maximum intensity
  3. RESOLVING — The pattern is fading or being addressed
  4. RESOLVED  — The arc is complete

Types of narrative arcs detected:
  - Emotional drift: Gradual shift from one emotional baseline to another
  - Confidence evolution: How trading confidence changes over weeks
  - Override trajectory: Whether overrides are increasing or decreasing
  - Readiness trend: Long-term readiness trajectory (improving/declining)
  - Performance cycle: Recurring win/loss patterns at monthly granularity
  - Stress accumulation: Compounding stress that doesn't resolve between sessions

Design principles:
  - Requires 4+ weeks of data to detect any arc
  - Uses moving averages and trend detection (linear regression)
  - Arcs have phases (building → peak → resolving → resolved)
  - Each arc includes a "so what" — actionable insight
  - Cloud synthesis optionally adds narrative depth

Zero-dependency — pure Python, no numpy/scipy.
"""

from __future__ import annotations

import json
import logging
import math
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from src.aura.patterns.base import (
    DetectedPattern,
    EvidenceItem,
    PatternDomain,
    PatternStatus,
    PatternTier,
)

logger = logging.getLogger(__name__)

# --- Configuration ---
MIN_WEEKS_FOR_ARC = 4          # Minimum weeks of data to detect an arc
TREND_SIGNIFICANCE = 0.15      # Minimum slope magnitude (per-week change) to be notable
DRIFT_THRESHOLD = 0.20         # Minimum shift in weekly average to qualify as drift
MOVING_AVG_WINDOW = 3          # Weeks for moving average smoothing


class ArcPhase:
    """Narrative arc phase labels."""
    BUILDING = "building"
    PEAK = "peak"
    RESOLVING = "resolving"
    RESOLVED = "resolved"
    STABLE = "stable"           # No clear arc — flat trend


@dataclass
class NarrativeArc:
    """A detected long-term behavioral trajectory."""

    arc_id: str
    arc_type: str               # "emotional_drift", "confidence_evolution", etc.
    phase: str                  # ArcPhase value
    description: str
    trend_slope: float          # Rate of change per week
    current_value: float        # Latest smoothed value
    peak_value: float           # Maximum value observed in this arc
    start_week: str             # ISO week when arc started
    peak_week: str              # ISO week of peak
    duration_weeks: int
    data_points: List[Tuple[str, float]] = field(default_factory=list)  # (week, value)
    actionable_insight: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "arc_id": self.arc_id,
            "arc_type": self.arc_type,
            "phase": self.phase,
            "description": self.description,
            "trend_slope": round(self.trend_slope, 4),
            "current_value": round(self.current_value, 3),
            "peak_value": round(self.peak_value, 3),
            "start_week": self.start_week,
            "peak_week": self.peak_week,
            "duration_weeks": self.duration_weeks,
            "data_points": [(w, round(v, 3)) for w, v in self.data_points],
            "actionable_insight": self.actionable_insight,
        }


def _linear_regression(x: List[float], y: List[float]) -> Tuple[float, float, float]:
    """Compute simple linear regression: y = slope * x + intercept.

    Returns:
        (slope, intercept, r_squared) tuple. Pure Python, no numpy.
    """
    n = len(x)
    if n < 2 or len(y) != n:
        return 0.0, 0.0, 0.0

    mean_x = sum(x) / n
    mean_y = sum(y) / n

    ss_xx = sum((xi - mean_x) ** 2 for xi in x)
    ss_yy = sum((yi - mean_y) ** 2 for yi in y)
    ss_xy = sum((xi - mean_x) * (yi - mean_y) for xi, yi in zip(x, y))

    if ss_xx < 1e-10:
        return 0.0, mean_y, 0.0

    slope = ss_xy / ss_xx
    intercept = mean_y - slope * mean_x
    r_squared = (ss_xy ** 2) / (ss_xx * ss_yy) if ss_yy > 1e-10 else 0.0

    return slope, intercept, r_squared


def _moving_average(values: List[float], window: int = MOVING_AVG_WINDOW) -> List[float]:
    """Compute moving average for smoothing."""
    if len(values) <= window:
        return values[:]

    result = []
    for i in range(len(values)):
        start = max(0, i - window + 1)
        window_values = values[start:i + 1]
        result.append(sum(window_values) / len(window_values))
    return result


def _detect_phase(
    smoothed: List[float],
    slope: float,
    r_squared: float,
) -> str:
    """Determine the narrative arc phase from smoothed trend data."""
    if len(smoothed) < 3:
        return ArcPhase.STABLE

    # Check recent direction (last 3 values)
    recent = smoothed[-3:]
    recent_slope = (recent[-1] - recent[0]) / max(len(recent) - 1, 1)
    peak_idx = smoothed.index(max(smoothed))

    # If peak is at the end and trend is up → building
    if peak_idx >= len(smoothed) - 2 and slope > TREND_SIGNIFICANCE:
        return ArcPhase.BUILDING

    # If peak is in the middle and recent trend is down → resolving
    if peak_idx < len(smoothed) - 2 and recent_slope < -TREND_SIGNIFICANCE * 0.5:
        return ArcPhase.RESOLVING

    # If the value has returned to near-start levels → resolved
    if len(smoothed) >= 4:
        start_avg = sum(smoothed[:2]) / 2
        end_avg = sum(smoothed[-2:]) / 2
        peak_val = max(smoothed)
        if peak_val > start_avg * 1.2 and abs(end_avg - start_avg) < 0.1:
            return ArcPhase.RESOLVED

    # If value is near peak → peak
    if peak_idx >= len(smoothed) - 2 and abs(slope) < TREND_SIGNIFICANCE:
        return ArcPhase.PEAK

    # If strong upward trend → building
    if slope > TREND_SIGNIFICANCE and r_squared > 0.3:
        return ArcPhase.BUILDING

    # If strong downward trend and values are above baseline → resolving
    if slope < -TREND_SIGNIFICANCE and r_squared > 0.3:
        return ArcPhase.RESOLVING

    return ArcPhase.STABLE


class Tier3NarrativeArcDetector:
    """Monthly narrative arc detector.

    Analyzes multi-week behavioral data to detect long-term arcs:
    emotional drift, confidence evolution, override trajectories,
    readiness trends, and performance cycles.

    Args:
        patterns_dir: Where to persist detected T3 patterns
    """

    def __init__(self, patterns_dir: Optional[Path] = None):
        self.patterns_dir = patterns_dir or Path(".aura/patterns")
        self.patterns_dir.mkdir(parents=True, exist_ok=True)
        self._patterns_file = self.patterns_dir / "t3_patterns.json"
        self._active_patterns: Dict[str, DetectedPattern] = {}
        self._load_patterns()

    def _load_patterns(self) -> None:
        """Load persisted T3 patterns."""
        if not self._patterns_file.exists():
            return
        try:
            data = json.loads(self._patterns_file.read_text())
            for pid, pdata in data.items():
                evidence = [EvidenceItem(**e) for e in pdata.pop("evidence", [])]
                pdata["tier"] = PatternTier(pdata["tier"])
                pdata["domain"] = PatternDomain(pdata["domain"])
                pdata["status"] = PatternStatus(pdata["status"])
                self._active_patterns[pid] = DetectedPattern(
                    evidence=evidence, **pdata
                )
        except Exception as e:
            logger.warning(f"T3: Failed to load patterns: {e}")

    def _save_patterns(self) -> None:
        """Persist T3 patterns to disk."""
        try:
            data = {
                pid: p.to_dict()
                for pid, p in self._active_patterns.items()
                if p.status not in (PatternStatus.ARCHIVED, PatternStatus.INVALIDATED)
            }
            self._patterns_file.write_text(json.dumps(data, indent=2, default=str))
        except Exception as e:
            logger.error(f"T3: Failed to save patterns: {e}")

    # --- Main Detection ---

    def detect(
        self,
        conversations: List[Dict[str, Any]],
        readiness_history: List[Dict[str, Any]],
        trade_outcomes: List[Dict[str, Any]],
        override_events: List[Dict[str, Any]],
    ) -> List[DetectedPattern]:
        """Run all T3 narrative arc analyses.

        Should be called monthly or on-demand via CLI.

        Args:
            conversations: Full conversation history (multi-week window)
            readiness_history: Readiness score history
            trade_outcomes: Trade journal entries
            override_events: Override events with outcomes

        Returns:
            List of detected or updated T3 narrative arc patterns
        """
        all_patterns: List[DetectedPattern] = []

        all_patterns.extend(
            self._detect_emotional_drift(conversations)
        )
        all_patterns.extend(
            self._detect_confidence_evolution(trade_outcomes)
        )
        all_patterns.extend(
            self._detect_override_trajectory(override_events)
        )
        all_patterns.extend(
            self._detect_readiness_trend(readiness_history)
        )
        all_patterns.extend(
            self._detect_performance_cycle(trade_outcomes)
        )
        all_patterns.extend(
            self._detect_stress_accumulation(conversations, readiness_history)
        )

        self._save_patterns()

        if all_patterns:
            logger.info(f"T3: Detected {len(all_patterns)} narrative arc patterns")

        return all_patterns

    # --- Arc Detectors ---

    def _detect_emotional_drift(
        self,
        conversations: List[Dict[str, Any]],
    ) -> List[DetectedPattern]:
        """Detect gradual shift in emotional baseline over weeks.

        Example: "Your emotional state has shifted from mostly calm (weeks 1-4)
        to frequently anxious (weeks 5-8), suggesting building stress."
        """
        weekly_emotions = self._group_by_week_emotion(conversations)
        if len(weekly_emotions) < MIN_WEEKS_FOR_ARC:
            return []

        # Map emotional states to a valence score
        valence_map = {
            "calm": 0.8, "energized": 0.9, "confident": 0.85, "focused": 0.8,
            "neutral": 0.5,
            "stressed": 0.2, "anxious": 0.15, "fatigued": 0.25,
            "frustrated": 0.1, "overwhelmed": 0.05, "revenge": 0.0,
        }

        weeks = sorted(weekly_emotions.keys())
        valences = []
        for week in weeks:
            states = weekly_emotions[week]
            avg_valence = sum(
                valence_map.get(s.lower(), 0.5) for s in states
            ) / max(len(states), 1)
            valences.append(avg_valence)

        smoothed = _moving_average(valences)
        x_vals = list(range(len(smoothed)))
        slope, intercept, r_sq = _linear_regression(x_vals, smoothed)

        if abs(slope) < TREND_SIGNIFICANCE * 0.5:
            return []  # No meaningful drift

        phase = _detect_phase(smoothed, slope, r_sq)
        direction = "declining" if slope < 0 else "improving"
        start_avg = sum(smoothed[:2]) / min(2, len(smoothed))
        end_avg = sum(smoothed[-2:]) / min(2, len(smoothed))

        arc = NarrativeArc(
            arc_id="emotional_drift",
            arc_type="emotional_drift",
            phase=phase,
            description=(
                f"Emotional baseline {direction} over {len(weeks)} weeks "
                f"(valence {start_avg:.2f} → {end_avg:.2f}, "
                f"slope={slope:.3f}/week, R²={r_sq:.2f})"
            ),
            trend_slope=slope,
            current_value=smoothed[-1],
            peak_value=max(smoothed),
            start_week=weeks[0],
            peak_week=weeks[smoothed.index(max(smoothed))],
            duration_weeks=len(weeks),
            data_points=list(zip(weeks, smoothed)),
            actionable_insight=(
                f"Emotional valence is {direction}. "
                + (
                    "Consider stress management techniques and potential trading breaks."
                    if slope < 0
                    else "Positive trend — current routines appear beneficial."
                )
            ),
        )

        return self._upsert_arc_pattern(arc)

    def _detect_confidence_evolution(
        self,
        trade_outcomes: List[Dict[str, Any]],
    ) -> List[DetectedPattern]:
        """Detect changes in trading confidence (via Buddy confidence scores) over weeks."""
        weekly_conf = self._group_by_week_field(
            trade_outcomes, "buddy_confidence",
            ts_field="timestamp",
        )
        if len(weekly_conf) < MIN_WEEKS_FOR_ARC:
            return []

        weeks = sorted(weekly_conf.keys())
        avg_confs = [
            sum(weekly_conf[w]) / len(weekly_conf[w]) for w in weeks
        ]

        smoothed = _moving_average(avg_confs)
        x_vals = list(range(len(smoothed)))
        slope, intercept, r_sq = _linear_regression(x_vals, smoothed)

        if abs(slope) < TREND_SIGNIFICANCE * 0.3:
            return []

        phase = _detect_phase(smoothed, slope, r_sq)
        direction = "increasing" if slope > 0 else "decreasing"

        arc = NarrativeArc(
            arc_id="confidence_evolution",
            arc_type="confidence_evolution",
            phase=phase,
            description=(
                f"Trading confidence {direction} over {len(weeks)} weeks "
                f"({smoothed[0]:.2f} → {smoothed[-1]:.2f}, "
                f"slope={slope:.3f}/week)"
            ),
            trend_slope=slope,
            current_value=smoothed[-1],
            peak_value=max(smoothed),
            start_week=weeks[0],
            peak_week=weeks[smoothed.index(max(smoothed))],
            duration_weeks=len(weeks),
            data_points=list(zip(weeks, smoothed)),
            actionable_insight=(
                f"Confidence is {direction}. "
                + (
                    "Rising confidence may indicate model improvement or market alignment."
                    if slope > 0
                    else "Declining confidence may signal model degradation or regime change."
                )
            ),
        )

        return self._upsert_arc_pattern(arc)

    def _detect_override_trajectory(
        self,
        override_events: List[Dict[str, Any]],
    ) -> List[DetectedPattern]:
        """Detect whether overrides are increasing or decreasing over weeks."""
        weekly_counts = self._count_by_week(
            override_events, ts_field="timestamp"
        )
        if len(weekly_counts) < MIN_WEEKS_FOR_ARC:
            return []

        weeks = sorted(weekly_counts.keys())
        counts = [weekly_counts[w] for w in weeks]

        smoothed = _moving_average(counts)
        x_vals = list(range(len(smoothed)))
        slope, intercept, r_sq = _linear_regression(x_vals, smoothed)

        if abs(slope) < 0.3:  # Less than ~0.3 overrides/week change
            return []

        phase = _detect_phase(smoothed, slope, r_sq)
        direction = "increasing" if slope > 0 else "decreasing"

        # Also compute win rate trend for overrides
        weekly_wins = defaultdict(int)
        weekly_total = defaultdict(int)
        for event in override_events:
            ts = event.get("timestamp", "")
            outcome = event.get("outcome", "")
            try:
                dt = datetime.fromisoformat(str(ts).replace("Z", "+00:00"))
                week_key = f"{dt.isocalendar()[0]}-W{dt.isocalendar()[1]:02d}"
            except (ValueError, AttributeError):
                continue
            if outcome in ("win", "loss"):
                weekly_total[week_key] += 1
                if outcome == "win":
                    weekly_wins[week_key] += 1

        arc = NarrativeArc(
            arc_id="override_trajectory",
            arc_type="override_trajectory",
            phase=phase,
            description=(
                f"Override frequency {direction} over {len(weeks)} weeks "
                f"({smoothed[0]:.1f} → {smoothed[-1]:.1f}/week, "
                f"slope={slope:.2f}/week)"
            ),
            trend_slope=slope,
            current_value=smoothed[-1],
            peak_value=max(smoothed),
            start_week=weeks[0],
            peak_week=weeks[smoothed.index(max(smoothed))],
            duration_weeks=len(weeks),
            data_points=list(zip(weeks, smoothed)),
            actionable_insight=(
                f"Overrides are {direction}. "
                + (
                    "Increasing overrides may indicate growing distrust of the bot — "
                    "review recent accuracy and recalibrate if needed."
                    if slope > 0
                    else "Decreasing overrides suggests growing trust in bot signals."
                )
            ),
        )

        return self._upsert_arc_pattern(arc)

    def _detect_readiness_trend(
        self,
        readiness_history: List[Dict[str, Any]],
    ) -> List[DetectedPattern]:
        """Detect long-term readiness score trajectory."""
        weekly_readiness = self._group_by_week_field(
            readiness_history, "score",
            ts_field="timestamp",
        )
        if len(weekly_readiness) < MIN_WEEKS_FOR_ARC:
            return []

        weeks = sorted(weekly_readiness.keys())
        avg_scores = [
            sum(weekly_readiness[w]) / len(weekly_readiness[w]) for w in weeks
        ]

        smoothed = _moving_average(avg_scores)
        x_vals = list(range(len(smoothed)))
        slope, intercept, r_sq = _linear_regression(x_vals, smoothed)

        if abs(slope) < TREND_SIGNIFICANCE:
            return []

        phase = _detect_phase(smoothed, slope, r_sq)
        direction = "improving" if slope > 0 else "declining"

        arc = NarrativeArc(
            arc_id="readiness_trend",
            arc_type="readiness_trend",
            phase=phase,
            description=(
                f"Readiness {direction} over {len(weeks)} weeks "
                f"({smoothed[0]:.1f} → {smoothed[-1]:.1f}, "
                f"slope={slope:.2f}/week, R²={r_sq:.2f})"
            ),
            trend_slope=slope,
            current_value=smoothed[-1],
            peak_value=max(smoothed),
            start_week=weeks[0],
            peak_week=weeks[smoothed.index(max(smoothed))],
            duration_weeks=len(weeks),
            data_points=list(zip(weeks, smoothed)),
            actionable_insight=(
                f"Readiness is {direction}. "
                + (
                    "Improving readiness correlates with better trading — maintain current practices."
                    if slope > 0
                    else "Declining readiness is a risk factor — investigate stress sources and recovery habits."
                )
            ),
        )

        return self._upsert_arc_pattern(arc)

    def _detect_performance_cycle(
        self,
        trade_outcomes: List[Dict[str, Any]],
    ) -> List[DetectedPattern]:
        """Detect recurring monthly performance patterns (good weeks / bad weeks)."""
        weekly_pnl = self._group_by_week_field(
            trade_outcomes, "pnl_pips",
            ts_field="timestamp",
            fallback_field="pnl",
        )
        if len(weekly_pnl) < MIN_WEEKS_FOR_ARC:
            return []

        weeks = sorted(weekly_pnl.keys())
        total_pnls = [sum(weekly_pnl[w]) for w in weeks]

        smoothed = _moving_average(total_pnls)
        x_vals = list(range(len(smoothed)))
        slope, intercept, r_sq = _linear_regression(x_vals, smoothed)

        # Also detect volatility (consistency)
        pnl_mean = sum(total_pnls) / len(total_pnls)
        pnl_std = math.sqrt(
            sum((p - pnl_mean) ** 2 for p in total_pnls) / len(total_pnls)
        ) if len(total_pnls) > 1 else 0.0

        # Count positive vs negative weeks
        pos_weeks = sum(1 for p in total_pnls if p > 0)
        neg_weeks = sum(1 for p in total_pnls if p <= 0)
        win_pct = pos_weeks / len(total_pnls)

        direction = "improving" if slope > 0 else "declining" if slope < -0.5 else "stable"

        arc = NarrativeArc(
            arc_id="performance_cycle",
            arc_type="performance_cycle",
            phase=_detect_phase(smoothed, slope, r_sq) if abs(slope) > 0.5 else ArcPhase.STABLE,
            description=(
                f"Performance {direction} over {len(weeks)} weeks "
                f"({pos_weeks} positive, {neg_weeks} negative, "
                f"win%={win_pct:.0%}, weekly PnL σ={pnl_std:.1f} pips)"
            ),
            trend_slope=slope,
            current_value=smoothed[-1],
            peak_value=max(smoothed),
            start_week=weeks[0],
            peak_week=weeks[smoothed.index(max(smoothed))],
            duration_weeks=len(weeks),
            data_points=list(zip(weeks, smoothed)),
            actionable_insight=(
                f"Performance is {direction} with {win_pct:.0%} positive weeks. "
                + (
                    "Strong consistency — maintain current strategy."
                    if win_pct >= 0.6
                    else "Consider reviewing strategy adjustments or risk sizing."
                )
            ),
        )

        return self._upsert_arc_pattern(arc)

    def _detect_stress_accumulation(
        self,
        conversations: List[Dict[str, Any]],
        readiness_history: List[Dict[str, Any]],
    ) -> List[DetectedPattern]:
        """Detect compounding stress that doesn't resolve between sessions.

        Unlike emotional drift (which tracks valence), this specifically
        detects when stress indicators accumulate without recovery.
        """
        # Build weekly stress score (0-1, higher = more stressed)
        negative_states = {"stressed", "anxious", "fatigued", "frustrated", "overwhelmed", "revenge"}
        weekly_stress: Dict[str, List[float]] = defaultdict(list)

        for conv in conversations:
            ts = conv.get("timestamp", "")
            state = conv.get("emotional_state", "neutral").lower()
            try:
                dt = datetime.fromisoformat(str(ts).replace("Z", "+00:00"))
                week_key = f"{dt.isocalendar()[0]}-W{dt.isocalendar()[1]:02d}"
            except (ValueError, AttributeError):
                continue
            stress_score = 1.0 if state in negative_states else 0.0
            weekly_stress[week_key].append(stress_score)

        if len(weekly_stress) < MIN_WEEKS_FOR_ARC:
            return []

        weeks = sorted(weekly_stress.keys())
        avg_stress = [
            sum(weekly_stress[w]) / len(weekly_stress[w]) for w in weeks
        ]

        smoothed = _moving_average(avg_stress)
        x_vals = list(range(len(smoothed)))
        slope, intercept, r_sq = _linear_regression(x_vals, smoothed)

        # Only report if stress is BUILDING (increasing trend)
        if slope < TREND_SIGNIFICANCE * 0.5:
            return []

        # Check if stress ever "resets" (drops significantly between weeks)
        recovery_gaps = []
        for i in range(1, len(smoothed)):
            if smoothed[i] < smoothed[i - 1] * 0.7:  # 30%+ drop
                recovery_gaps.append(i)

        has_recovery = len(recovery_gaps) > 0
        phase = _detect_phase(smoothed, slope, r_sq)

        arc = NarrativeArc(
            arc_id="stress_accumulation",
            arc_type="stress_accumulation",
            phase=phase,
            description=(
                f"Stress accumulating over {len(weeks)} weeks "
                f"(stress index {smoothed[0]:.2f} → {smoothed[-1]:.2f}, "
                f"slope={slope:.3f}/week). "
                + (
                    f"Some recovery detected ({len(recovery_gaps)} reset weeks)."
                    if has_recovery
                    else "No significant recovery periods detected — compounding risk."
                )
            ),
            trend_slope=slope,
            current_value=smoothed[-1],
            peak_value=max(smoothed),
            start_week=weeks[0],
            peak_week=weeks[smoothed.index(max(smoothed))],
            duration_weeks=len(weeks),
            data_points=list(zip(weeks, smoothed)),
            actionable_insight=(
                "Stress is accumulating without adequate recovery. "
                + (
                    "Consider a structured break or reduced trading activity."
                    if not has_recovery
                    else "Recovery periods exist but stress trend is still rising."
                )
            ),
        )

        return self._upsert_arc_pattern(arc)

    # --- Pattern Management ---

    def _upsert_arc_pattern(self, arc: NarrativeArc) -> List[DetectedPattern]:
        """Create or update a T3 pattern from a detected narrative arc."""
        existing = self._active_patterns.get(arc.arc_id)

        evidence = EvidenceItem(
            source_type="narrative_arc",
            source_id=arc.arc_id,
            timestamp=datetime.now(timezone.utc).isoformat(),
            summary=arc.description[:200],
            data=arc.to_dict(),
        )

        # US-203: Compute actual R² from arc data_points instead of broken proxy.
        # The old formula (slope²/(slope²+0.01)) always returned ~1.0 for any
        # non-trivial slope, giving false confidence in weak trends.
        r_sq = 0.0
        if len(arc.data_points) >= 2:
            _x = list(range(len(arc.data_points)))
            _y = [v for _, v in arc.data_points]
            _, _, r_sq = _linear_regression(
                [float(xi) for xi in _x], _y
            )
        duration_factor = min(1.0, arc.duration_weeks / 12)  # More weeks = more confident
        confidence = min(0.90, 0.40 + duration_factor * 0.3 + r_sq * 0.2)

        if existing and existing.status not in (
            PatternStatus.ARCHIVED, PatternStatus.INVALIDATED
        ):
            existing.add_evidence(evidence)
            existing.confidence = confidence
            existing.description = arc.description
            if arc.actionable_insight:
                existing.suggested_rule = arc.actionable_insight
            return [existing]

        pattern = DetectedPattern(
            pattern_id=arc.arc_id,
            tier=PatternTier.T3_MONTHLY,
            domain=(
                PatternDomain.HUMAN
                if arc.arc_type in ("emotional_drift", "stress_accumulation")
                else PatternDomain.CROSS_ENGINE
            ),
            description=arc.description,
            evidence=[evidence],
            observation_count=1,
            confidence=confidence,
            suggested_rule=arc.actionable_insight,
        )
        self._active_patterns[arc.arc_id] = pattern
        logger.info(f"T3: New narrative arc: '{arc.arc_id}' ({arc.phase})")
        return [pattern]

    # --- Data Grouping Helpers ---

    def _group_by_week_emotion(
        self,
        records: List[Dict[str, Any]],
    ) -> Dict[str, List[str]]:
        """Group conversation emotional states by ISO week."""
        result: Dict[str, List[str]] = defaultdict(list)
        for record in records:
            ts = record.get("timestamp", "")
            state = record.get("emotional_state", "neutral")
            try:
                dt = datetime.fromisoformat(str(ts).replace("Z", "+00:00"))
                week_key = f"{dt.isocalendar()[0]}-W{dt.isocalendar()[1]:02d}"
            except (ValueError, AttributeError):
                continue
            result[week_key].append(state)
        return dict(result)

    def _group_by_week_field(
        self,
        records: List[Dict[str, Any]],
        value_field: str,
        ts_field: str = "timestamp",
        fallback_field: Optional[str] = None,
    ) -> Dict[str, List[float]]:
        """Group a numeric field by ISO week."""
        result: Dict[str, List[float]] = defaultdict(list)
        for record in records:
            ts = record.get(ts_field, record.get("close_time", ""))
            value = record.get(value_field)
            if value is None and fallback_field:
                value = record.get(fallback_field)
            if value is None:
                continue
            try:
                dt = datetime.fromisoformat(str(ts).replace("Z", "+00:00"))
                week_key = f"{dt.isocalendar()[0]}-W{dt.isocalendar()[1]:02d}"
            except (ValueError, AttributeError):
                continue
            try:
                result[week_key].append(float(value))
            except (ValueError, TypeError):
                continue
        return dict(result)

    def _count_by_week(
        self,
        records: List[Dict[str, Any]],
        ts_field: str = "timestamp",
    ) -> Dict[str, int]:
        """Count records per ISO week."""
        result: Dict[str, int] = defaultdict(int)
        for record in records:
            ts = record.get(ts_field, "")
            try:
                dt = datetime.fromisoformat(str(ts).replace("Z", "+00:00"))
                week_key = f"{dt.isocalendar()[0]}-W{dt.isocalendar()[1]:02d}"
            except (ValueError, AttributeError):
                continue
            result[week_key] += 1
        return dict(result)

    # --- Query API ---

    def get_active_patterns(self) -> List[DetectedPattern]:
        """Return all active T3 patterns."""
        return [
            p for p in self._active_patterns.values()
            if p.status not in (PatternStatus.ARCHIVED, PatternStatus.INVALIDATED)
        ]

    def get_promotable_patterns(self) -> List[DetectedPattern]:
        """Return T3 patterns ready for rule promotion."""
        return [p for p in self._active_patterns.values() if p.is_promotable()]

    def get_arcs_summary(self) -> List[Dict[str, Any]]:
        """Get a summary of all active narrative arcs for CLI display."""
        arcs = []
        for pattern in self.get_active_patterns():
            # Extract arc data from latest evidence
            latest_evidence = pattern.evidence[-1] if pattern.evidence else None
            arc_data = latest_evidence.data if latest_evidence else {}

            arcs.append({
                "arc_type": arc_data.get("arc_type", pattern.pattern_id),
                "phase": arc_data.get("phase", "unknown"),
                "description": pattern.description[:120],
                "confidence": pattern.confidence,
                "duration_weeks": arc_data.get("duration_weeks", 0),
                "trend_slope": arc_data.get("trend_slope", 0),
                "insight": pattern.suggested_rule or "",
            })
        return arcs
