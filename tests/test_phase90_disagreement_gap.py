"""Phase 90 US-401: Tests for disagreement_gap_analyzer.py constant recalibration."""
from __future__ import annotations

import threading
from unittest.mock import MagicMock

import pytest

from src.scanner.disagreement_gap_analyzer import (
    NEAR_MISS_GAP,
    MODERATE_GAP,
    CLASS_NEAR_THRESHOLD,
    CLASS_MODERATE,
    CLASS_FAR_ABOVE,
    CLASS_INSUFFICIENT_DATA,
    MIN_FAIL_COUNT,
    analyze,
)


def _make_recorder(fail_p50: float, fail_count: int = MIN_FAIL_COUNT, all_count: int = 10,
                   pass_rate: float = 0.7, fail_p75: float | None = None) -> MagicMock:
    """Build a minimal mock recorder for analyze()."""
    if fail_p75 is None:
        fail_p75 = fail_p50 + 0.1
    recorder = MagicMock()
    recorder.get_fail_distribution.return_value = {
        "count": fail_count, "p50": fail_p50, "p75": fail_p75
    }
    recorder.get_pass_rate.return_value = pass_rate
    recorder.get_distribution.return_value = {"count": all_count}
    recorder._lock = threading.Lock()
    recorder._records = []
    return recorder


# ── Constant sanity checks ─────────────────────────────────────────────────


def test_near_miss_gap_value():
    """NEAR_MISS_GAP must be 0.35 after Phase 90 recalibration."""
    assert NEAR_MISS_GAP == 0.35


def test_moderate_gap_value():
    """MODERATE_GAP must be 0.50 after Phase 90 recalibration."""
    assert MODERATE_GAP == 0.50


# ── Classification at floor=0.50 ──────────────────────────────────────────


def test_fail_p50_060_floor_050_near_threshold():
    """3/5 disagreement=0.60 with floor=0.50 → gap=0.10 → NEAR_THRESHOLD."""
    recorder = _make_recorder(fail_p50=0.60)
    result = analyze(recorder, disagreement_hard_floor=0.50)
    assert result["gap_at_p50"] == pytest.approx(0.10, abs=1e-6)
    assert result["classification"] == CLASS_NEAR_THRESHOLD


def test_fail_p50_080_floor_050_near_threshold():
    """4/5 disagreement=0.80 with floor=0.50 → gap=0.30 → NEAR_THRESHOLD (was MODERATE with old threshold)."""
    recorder = _make_recorder(fail_p50=0.80)
    result = analyze(recorder, disagreement_hard_floor=0.50)
    assert result["gap_at_p50"] == pytest.approx(0.30, abs=1e-6)
    assert result["classification"] == CLASS_NEAR_THRESHOLD, (
        f"Expected NEAR_THRESHOLD (gap=0.30 <= NEAR_MISS_GAP=0.35) but got {result['classification']}"
    )


def test_fail_p50_100_floor_050_moderate():
    """5/5 disagreement=1.00 with floor=0.50 → gap=0.50 → MODERATE."""
    recorder = _make_recorder(fail_p50=1.00)
    result = analyze(recorder, disagreement_hard_floor=0.50)
    assert result["gap_at_p50"] == pytest.approx(0.50, abs=1e-6)
    assert result["classification"] == CLASS_MODERATE


def test_fail_p50_producing_gap_above_050_far_above():
    """gap > 0.50 → FAR_ABOVE classification."""
    # With floor=0.30, p50=0.90 → gap=0.60 > MODERATE_GAP=0.50
    recorder = _make_recorder(fail_p50=0.90)
    result = analyze(recorder, disagreement_hard_floor=0.30)
    assert result["gap_at_p50"] == pytest.approx(0.60, abs=1e-6)
    assert result["classification"] == CLASS_FAR_ABOVE


def test_insufficient_data_below_min_fail_count():
    """Fewer than MIN_FAIL_COUNT failures → INSUFFICIENT_DATA."""
    recorder = _make_recorder(fail_p50=0.60, fail_count=MIN_FAIL_COUNT - 1)
    result = analyze(recorder, disagreement_hard_floor=0.50)
    assert result["classification"] == CLASS_INSUFFICIENT_DATA


def test_near_miss_count_uses_updated_gap():
    """Near-miss count includes all blocks within NEAR_MISS_GAP=0.35 of floor."""
    floor = 0.50
    recorder = MagicMock()
    recorder.get_fail_distribution.return_value = {
        "count": 5, "p50": 0.60, "p75": 0.80
    }
    recorder.get_pass_rate.return_value = 0.5
    recorder.get_distribution.return_value = {"count": 10}
    # Simulate 3 hard_blocked records: two within 0.35 of floor, one far
    recorder._lock = threading.Lock()
    recorder._records = [
        {"hard_blocked": True, "disagreement": 0.60},  # gap=0.10 ≤ 0.35 → near-miss
        {"hard_blocked": True, "disagreement": 0.80},  # gap=0.30 ≤ 0.35 → near-miss
        {"hard_blocked": True, "disagreement": 1.00},  # gap=0.50 ≤ 0.35? No → not near-miss
        {"hard_blocked": False, "disagreement": 0.20},
        {"hard_blocked": True, "disagreement": 0.55},  # gap=0.05 ≤ 0.35 → near-miss
    ]
    result = analyze(recorder, disagreement_hard_floor=floor)
    assert result["near_miss_count"] == 3
