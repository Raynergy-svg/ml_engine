"""Phase 58 US-359: ThresholdGapAnalyzer.

Detects when the current min_confidence threshold structurally exceeds
the natural signal distribution of the ensemble models.

Key metric: gap_at_p75 = current_threshold - p75_of_raw_confidence
If the top-quartile signals still fall more than 5 pts below threshold,
this is a structural mismatch — not noise. Recommendation: THRESHOLD_TOO_HIGH.

Design: pure analysis class — no side effects, no persistence.
Wiring: called by AdaptiveConfidenceFloor (US-361) and diagnostic paths.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)

# Recommendation constants
REC_OK = "OK"
REC_THRESHOLD_TOO_HIGH = "THRESHOLD_TOO_HIGH"
REC_NEAR_THRESHOLD = "NEAR_THRESHOLD"
REC_INSUFFICIENT_DATA = "INSUFFICIENT_DATA"

# Thresholds
MIN_COUNT_FOR_ANALYSIS = 20
GAP_TOO_HIGH_CUTOFF = 5.0      # gap_at_p75 > 5.0 → THRESHOLD_TOO_HIGH
PASS_RATE_OK_CUTOFF = 10.0     # signals_above_threshold_pct >= 10 → OK
PASS_RATE_CONCERN = 5.0        # < 5% → flag even if gap is small
SAFETY_FLOOR = 45.0


@dataclass
class ThresholdGapReport:
    """Result of a threshold gap analysis pass."""
    count: int
    mean_raw: float
    median_raw: float
    p75_raw: float
    current_threshold: float
    gap_at_median: float          # threshold - median (positive = threshold above median)
    gap_at_p75: float             # threshold - p75 (positive = threshold above p75)
    signals_above_threshold_pct: float
    recommendation: str


class ThresholdGapAnalyzer:
    """Analyze the structural gap between min_confidence and raw signal distribution.

    Usage:
        analyzer = ThresholdGapAnalyzer()
        report = analyzer.analyze(recorder, min_confidence=58.0)
        if analyzer.should_lower_threshold(report, scans_since_last_trade=55):
            new_threshold = analyzer.compute_suggested_threshold(report)
    """

    def __init__(self, safety_floor: float = SAFETY_FLOOR) -> None:
        self._safety_floor = safety_floor

    def analyze(
        self,
        recorder: object,  # RawConfidenceRecorder
        min_confidence: float,
    ) -> ThresholdGapReport:
        """Compute distribution stats and classify gap severity.

        Args:
            recorder: RawConfidenceRecorder instance.
            min_confidence: Current engine min_confidence value (0-100 scale).

        Returns:
            ThresholdGapReport with recommendation.
        """
        dist = recorder.get_distribution()
        count = dist.get("count", 0)

        if count < MIN_COUNT_FOR_ANALYSIS:
            logger.debug(
                "ThresholdGapAnalyzer: insufficient data (count=%d < %d)",
                count, MIN_COUNT_FOR_ANALYSIS,
            )
            return ThresholdGapReport(
                count=count,
                mean_raw=dist.get("mean", 0.0),
                median_raw=dist.get("median", 0.0),
                p75_raw=dist.get("p75", 0.0),
                current_threshold=min_confidence,
                gap_at_median=0.0,
                gap_at_p75=0.0,
                signals_above_threshold_pct=0.0,
                recommendation=REC_INSUFFICIENT_DATA,
            )

        mean_raw = dist.get("mean", 0.0)
        median_raw = dist.get("median", 0.0)
        p75_raw = dist.get("p75", 0.0)

        gap_at_median = round(min_confidence - median_raw, 4)
        gap_at_p75 = round(min_confidence - p75_raw, 4)

        # Compute fraction of raw scores above threshold
        recent = recorder.get_recent(n=count)
        above = sum(
            1 for r in recent
            if r.get("raw_confidence", 0.0) >= min_confidence
        )
        signals_above_pct = round((above / max(count, 1)) * 100.0, 2)

        # Classify
        if signals_above_pct >= PASS_RATE_OK_CUTOFF:
            recommendation = REC_OK
        elif gap_at_p75 > GAP_TOO_HIGH_CUTOFF:
            recommendation = REC_THRESHOLD_TOO_HIGH
        elif signals_above_pct < PASS_RATE_CONCERN or gap_at_p75 > 0:
            recommendation = REC_NEAR_THRESHOLD
        else:
            recommendation = REC_OK

        report = ThresholdGapReport(
            count=count,
            mean_raw=mean_raw,
            median_raw=median_raw,
            p75_raw=p75_raw,
            current_threshold=min_confidence,
            gap_at_median=gap_at_median,
            gap_at_p75=gap_at_p75,
            signals_above_threshold_pct=signals_above_pct,
            recommendation=recommendation,
        )
        logger.debug(
            "ThresholdGapAnalyzer: threshold=%.1f p75=%.1f gap=%.1f "
            "above_pct=%.1f%% rec=%s",
            min_confidence, p75_raw, gap_at_p75, signals_above_pct, recommendation,
        )
        return report

    def should_lower_threshold(
        self,
        report: ThresholdGapReport,
        scans_since_last_trade: int,
        stall_threshold: int = 50,
    ) -> bool:
        """Return True when conditions justify a threshold reduction.

        Requirements:
          1. System is stalled (scans_since_last_trade >= stall_threshold)
          2. Recommendation suggests threshold is the blocker
        """
        if scans_since_last_trade < stall_threshold:
            return False
        return report.recommendation in (REC_THRESHOLD_TOO_HIGH, REC_NEAR_THRESHOLD)

    def compute_suggested_threshold(
        self,
        report: ThresholdGapReport,
        step: float = 1.0,
        safety_floor: Optional[float] = None,
    ) -> float:
        """Return suggested new threshold (current - step, floored at safety_floor)."""
        floor = safety_floor if safety_floor is not None else self._safety_floor
        return max(report.current_threshold - step, floor)
