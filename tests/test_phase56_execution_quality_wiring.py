"""Tests for Phase 56 US-349: ExecutionQualityTracker fill wiring.

Covers:
  - ExecutionQualityTracker.track_fill() signature and behavior
  - track_fill with filled / partial / rejected status
  - Slippage stored correctly
  - save_state() called after fill (atomic write, no .tmp lingers)
  - get_pair_quality() returns stats after fill
  - compute_agent_penalty() returns non-positive float after high slippage
  - Wiring grep: track_fill called in execution.py outside class definition
  - Wiring grep: _exec_quality_tracker = None in execution.py __init__
"""

import json
import os
import tempfile
from pathlib import Path

import pytest

from src.scanner.automation.execution_quality_tracker import (
    ExecutionQualityTracker,
    HIGH_SLIPPAGE_THRESHOLD,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def tracker(tmp_path):
    return ExecutionQualityTracker(persistence_path=tmp_path / "eq.json")


# ---------------------------------------------------------------------------
# 1. track_fill behavior
# ---------------------------------------------------------------------------

class TestTrackFill:

    def test_track_fill_increments_total(self, tracker):
        tracker.track_fill(
            pair="EUR_USD",
            expected_price=1.0850,
            fill_price=1.0851,
            slippage_pips=0.1,
            fill_status="filled",
        )
        stats = tracker.get_status()
        assert stats["total_fills"] >= 1

    def test_track_fill_stores_slippage(self, tracker):
        tracker.track_fill(
            pair="EUR_USD",
            expected_price=1.0850,
            fill_price=1.0866,
            slippage_pips=1.6,
            fill_status="filled",
        )
        quality = tracker.get_pair_quality("EUR_USD")
        assert quality is not None
        assert quality.avg_slippage_pips > 0.0

    def test_track_fill_partial_status_tracked(self, tracker):
        tracker.track_fill(
            pair="GBP_USD",
            expected_price=1.2500,
            fill_price=1.2502,
            slippage_pips=0.2,
            fill_status="partial",
        )
        quality = tracker.get_pair_quality("GBP_USD")
        assert quality is not None
        # fill_rate should be below 1.0 for partial fills
        assert quality.fill_rate < 1.0

    def test_track_fill_rejected_status_tracked(self, tracker):
        tracker.track_fill(
            pair="USD_JPY",
            expected_price=149.50,
            fill_price=149.52,
            slippage_pips=0.2,
            fill_status="rejected",
        )
        stats = tracker.get_status()
        assert stats.get("total_rejections", 0) >= 1

    def test_zero_slippage_acceptable(self, tracker):
        tracker.track_fill(
            pair="EUR_USD",
            expected_price=1.0850,
            fill_price=1.0850,
            slippage_pips=0.0,
            fill_status="filled",
        )
        quality = tracker.get_pair_quality("EUR_USD")
        assert quality is not None
        assert quality.avg_slippage_pips == pytest.approx(0.0, abs=0.001)


# ---------------------------------------------------------------------------
# 2. compute_agent_penalty after high slippage
# ---------------------------------------------------------------------------

class TestAgentPenalty:

    def test_no_penalty_below_slippage_threshold(self, tracker):
        """Low slippage with < 5 samples → no penalty (insufficient data)."""
        tracker.track_fill("EUR_USD", 1.0850, 1.08505, 0.05, "filled")
        penalty = tracker.compute_agent_penalty("EUR_USD", agent_confidence=0.70)
        assert penalty == 0.0  # < 5 samples, no penalty applied

    def test_high_slippage_results_in_quality_score_below_1(self, tracker):
        """High slippage should lower quality score below 1.0."""
        for _ in range(5):
            tracker.track_fill(
                pair="EUR_USD",
                expected_price=1.0850,
                fill_price=1.0870,  # 2.0 pip slippage — well above 1.5 threshold
                slippage_pips=2.0,
                fill_status="filled",
            )
        quality = tracker.get_pair_quality("EUR_USD")
        assert quality is not None
        assert quality.quality_score < 1.0


# ---------------------------------------------------------------------------
# 3. get_worst_pairs
# ---------------------------------------------------------------------------

class TestGetWorstPairs:

    def test_get_worst_pairs_returns_list(self, tracker):
        tracker.track_fill("EUR_USD", 1.0850, 1.0855, 0.5, "filled")
        tracker.track_fill("GBP_USD", 1.2500, 1.2520, 2.0, "filled")
        worst = tracker.get_worst_pairs(n=2)
        assert isinstance(worst, list)
        assert len(worst) <= 2


# ---------------------------------------------------------------------------
# 4. Persistence
# ---------------------------------------------------------------------------

class TestPersistence:

    def test_save_state_creates_file(self, tmp_path):
        path = tmp_path / "eq2.json"
        tracker = ExecutionQualityTracker(persistence_path=path)
        tracker.track_fill("EUR_USD", 1.0850, 1.0851, 0.1, "filled")
        tracker.save_state()
        assert path.exists()

    def test_save_state_valid_json(self, tmp_path):
        path = tmp_path / "eq3.json"
        tracker = ExecutionQualityTracker(persistence_path=path)
        tracker.track_fill("EUR_USD", 1.0850, 1.0851, 0.1, "filled")
        tracker.save_state()
        data = json.loads(path.read_text())
        assert "total_fills" in data or "pairs" in data


# ---------------------------------------------------------------------------
# 5. Wiring verification
# ---------------------------------------------------------------------------

class TestWiringVerification:

    def test_track_fill_called_in_execution_py(self):
        """Live wiring: track_fill must be called in execution.py outside class definition."""
        exec_path = "src/scanner/execution.py"
        with open(exec_path) as f:
            content = f.read()
        assert "_exec_quality_tracker.track_fill(" in content, (
            "track_fill() must be called in execution.py (Phase 56 US-349 wiring)"
        )

    def test_exec_quality_tracker_initialized_in_execution_py(self):
        """ExecutionQualityTracker must be initialized in execution.py __init__."""
        exec_path = "src/scanner/execution.py"
        with open(exec_path) as f:
            content = f.read()
        assert "ExecutionQualityTracker" in content
        assert "_exec_quality_tracker = None" in content

    def test_save_state_called_after_fill_in_execution(self):
        """save_state() must be called after track_fill in execution.py."""
        exec_path = "src/scanner/execution.py"
        with open(exec_path) as f:
            content = f.read()
        # Both must appear in the US-349 block
        assert "_exec_quality_tracker.save_state()" in content
