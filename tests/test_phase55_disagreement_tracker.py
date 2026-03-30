"""
Tests for Phase 55 US-343: Agent Disagreement Pattern Feedback.

Tests cover:
  - disagreement_score calculation (CV: std/mean)
  - Bucket classification (low/moderate/high/extreme)
  - Pairwise conflict tracking
  - Toxic pattern detection at 70% loss-rate threshold
  - Confidence penalty computation and capping
  - record_trade() and record_outcome() alias
  - Periodic re-analysis at 50-trade interval
  - Persistence: save_state / _load_state round-trip
  - Edge cases: zero scores, single agent, empty inputs
  - generate_report() structure
"""

import json
import os
import tempfile
from unittest.mock import patch, MagicMock

import pytest

from src.scanner.disagreement_tracker import (
    DisagreementTracker,
    AgentPairConflict,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_tracker(tmp_path=None) -> DisagreementTracker:
    """Create a tracker with a temp persistence path so tests don't stomp prod data."""
    if tmp_path is None:
        tmp_path = os.path.join(tempfile.mkdtemp(), "disagreement_patterns.json")
    return DisagreementTracker(persistence_path=tmp_path)


def _scores_uniform(n: int = 5, value: float = 0.7) -> dict:
    """All agents score the same — zero disagreement."""
    return {f"agent_{i}": value for i in range(n)}


def _scores_split(high: float = 0.9, low: float = 0.1) -> dict:
    """Half agents score high, half low — extreme disagreement."""
    return {
        "trend": high,
        "momentum": high,
        "risk_sentinel": low,
        "uncertainty": low,
    }


# ---------------------------------------------------------------------------
# 1. Disagreement score calculation
# ---------------------------------------------------------------------------

class TestDisagreementScoreCalculation:
    """Tests for _compute_disagreement() static method."""

    def test_zero_disagreement_uniform_scores(self):
        """All agents score identically → std = 0 → disagreement = 0."""
        tracker = make_tracker()
        scores = _scores_uniform(5, 0.7)
        result = tracker._compute_disagreement(scores)
        assert result == 0.0

    def test_high_disagreement_polar_scores(self):
        """Half high, half low → CV should be significant (> 0.3)."""
        tracker = make_tracker()
        scores = {"a": 0.9, "b": 0.9, "c": 0.1, "d": 0.1}
        result = tracker._compute_disagreement(scores)
        # mean = 0.5, std ≈ 0.4, CV ≈ 0.8
        assert result > 0.3, f"Expected high disagreement, got {result}"

    def test_single_agent_returns_zero(self):
        """Single agent → cannot compute CV, should return 0."""
        tracker = make_tracker()
        result = tracker._compute_disagreement({"only_agent": 0.8})
        assert result == 0.0

    def test_all_zero_scores_returns_zero(self):
        """Mean = 0 → CV undefined, should return 0.0 (not divide-by-zero)."""
        tracker = make_tracker()
        result = tracker._compute_disagreement({"a": 0.0, "b": 0.0, "c": 0.0})
        assert result == 0.0

    def test_moderate_disagreement(self):
        """Scores spread around 0.5 with small variance → CV in moderate range."""
        tracker = make_tracker()
        scores = {"a": 0.6, "b": 0.5, "c": 0.4}
        result = tracker._compute_disagreement(scores)
        # mean=0.5, std≈0.082, CV≈0.163 — should be "low" bucket
        assert 0.0 < result < 0.3, f"Expected low-to-moderate CV, got {result}"


# ---------------------------------------------------------------------------
# 2. Bucket classification
# ---------------------------------------------------------------------------

class TestBucketClassification:
    """Tests for _classify_bucket() static method."""

    def test_low_bucket(self):
        tracker = make_tracker()
        assert tracker._classify_bucket(0.0) == "low"
        assert tracker._classify_bucket(0.15) == "low"
        assert tracker._classify_bucket(0.19) == "low"

    def test_moderate_bucket(self):
        tracker = make_tracker()
        assert tracker._classify_bucket(0.2) == "moderate"
        assert tracker._classify_bucket(0.35) == "moderate"

    def test_high_bucket(self):
        tracker = make_tracker()
        assert tracker._classify_bucket(0.4) == "high"
        assert tracker._classify_bucket(0.55) == "high"

    def test_extreme_bucket(self):
        tracker = make_tracker()
        assert tracker._classify_bucket(0.6) == "extreme"
        assert tracker._classify_bucket(1.5) == "extreme"
        assert tracker._classify_bucket(99.9) == "extreme"

    def test_bucket_stats_updated_on_record(self):
        """record_trade() should increment the correct bucket's win/loss counter."""
        tracker = make_tracker()
        # Force high disagreement (extreme bucket)
        scores = _scores_split(high=0.9, low=0.1)
        tracker.record_trade(scores, won=False)
        # Extreme bucket should have 1 loss
        assert tracker._bucket_stats["extreme"]["losses"] == 1
        assert tracker._bucket_stats["extreme"]["wins"] == 0


# ---------------------------------------------------------------------------
# 3. Pairwise conflict tracking
# ---------------------------------------------------------------------------

class TestPairwiseConflictTracking:
    """Tests for _update_pair_conflicts()."""

    def test_conflict_registered_when_polar(self):
        """Agents at >0.6 vs <0.4 should register as a conflict."""
        tracker = make_tracker()
        scores = {"trend": 0.8, "risk_sentinel": 0.2}
        tracker._update_pair_conflicts(scores, won=False, pnl=0.0)
        key = ("risk_sentinel", "trend")
        assert key in tracker._pair_conflicts
        assert tracker._pair_conflicts[key].conflict_count == 1

    def test_no_conflict_when_close(self):
        """Both agents scoring similarly should not trigger a conflict."""
        tracker = make_tracker()
        scores = {"a": 0.7, "b": 0.65}
        tracker._update_pair_conflicts(scores, won=True, pnl=10.0)
        # No pair should be registered (difference < threshold)
        assert len(tracker._pair_conflicts) == 0

    def test_loss_count_incremented(self):
        """Loss trades should increment loss_count on the conflict."""
        tracker = make_tracker()
        scores = {"a": 0.9, "b": 0.1}
        tracker._update_pair_conflicts(scores, won=False, pnl=-50.0)
        tracker._update_pair_conflicts(scores, won=True, pnl=30.0)
        key = ("a", "b")
        conflict = tracker._pair_conflicts[key]
        assert conflict.conflict_count == 2
        assert conflict.loss_count == 1


# ---------------------------------------------------------------------------
# 4. Toxic pattern detection
# ---------------------------------------------------------------------------

class TestToxicPatternDetection:
    """Tests for _analyze_patterns() and is_toxic property."""

    def test_pair_becomes_toxic_above_threshold(self):
        """A pair with >70% loss rate and >= 5 observations should be toxic."""
        tracker = make_tracker()
        scores = {"trend": 0.9, "risk_sentinel": 0.1}
        # 7 losses, 2 wins (77.7% loss rate)
        for _ in range(7):
            tracker._update_pair_conflicts(scores, won=False, pnl=-20.0)
        for _ in range(2):
            tracker._update_pair_conflicts(scores, won=True, pnl=15.0)
        tracker._analyze_patterns()
        assert len(tracker._toxic_patterns) == 1
        assert tracker._toxic_patterns[0].agent_a in {"trend", "risk_sentinel"}

    def test_pair_not_toxic_below_min_observations(self):
        """Fewer than 5 observations should never be flagged as toxic."""
        tracker = make_tracker()
        scores = {"a": 0.9, "b": 0.1}
        for _ in range(4):  # Only 4 — below MIN_OBSERVATIONS
            tracker._update_pair_conflicts(scores, won=False, pnl=-10.0)
        tracker._analyze_patterns()
        assert len(tracker._toxic_patterns) == 0

    def test_pair_not_toxic_below_loss_threshold(self):
        """60% loss rate (below 70%) should not be flagged as toxic."""
        tracker = make_tracker()
        scores = {"a": 0.9, "b": 0.1}
        for _ in range(6):
            tracker._update_pair_conflicts(scores, won=False, pnl=-10.0)
        for _ in range(4):
            tracker._update_pair_conflicts(scores, won=True, pnl=8.0)
        tracker._analyze_patterns()
        assert len(tracker._toxic_patterns) == 0

    def test_reanalysis_triggered_every_50_trades(self):
        """_analyze_patterns should be called after 50 record_trade() calls."""
        tracker = make_tracker()
        tracker.reanalyze_interval = 50

        with patch.object(tracker, '_analyze_patterns', wraps=tracker._analyze_patterns) as mock_analyze:
            # Record 49 trades — should not trigger
            scores = {"a": 0.5, "b": 0.5}
            for _ in range(49):
                tracker.record_trade(scores, won=True)
            assert mock_analyze.call_count == 0

            # Record 50th trade — should trigger
            tracker.record_trade(scores, won=True)
            assert mock_analyze.call_count == 1


# ---------------------------------------------------------------------------
# 5. Confidence penalty
# ---------------------------------------------------------------------------

class TestConfidencePenalty:
    """Tests for get_confidence_penalty()."""

    def test_no_penalty_when_no_toxic_patterns(self):
        """No toxic patterns → penalty = 0.0."""
        tracker = make_tracker()
        scores = _scores_split()
        assert tracker.get_confidence_penalty(scores) == 0.0

    def test_penalty_applied_when_toxic_pair_active(self):
        """Active toxic pair + current disagreement → negative penalty."""
        tracker = make_tracker()
        # Force toxic pattern
        conflict = AgentPairConflict(
            agent_a="trend",
            agent_b="risk_sentinel",
            conflict_count=10,
            loss_count=8,  # 80% loss rate — toxic
        )
        tracker._toxic_patterns = [conflict]

        scores = {"trend": 0.85, "risk_sentinel": 0.15}  # Clearly in disagreement
        penalty = tracker.get_confidence_penalty(scores)
        assert penalty < 0.0, f"Expected negative penalty, got {penalty}"
        assert penalty >= -tracker.max_penalty, f"Penalty exceeded max: {penalty}"

    def test_penalty_capped_at_max(self):
        """Multiple toxic patterns should not exceed max_penalty cap."""
        tracker = make_tracker()
        # Add many toxic patterns
        tracker._toxic_patterns = [
            AgentPairConflict("a", "b", conflict_count=10, loss_count=9),
            AgentPairConflict("c", "d", conflict_count=10, loss_count=9),
            AgentPairConflict("e", "f", conflict_count=10, loss_count=9),
        ]
        scores = {"a": 0.9, "b": 0.1, "c": 0.9, "d": 0.1, "e": 0.9, "f": 0.1}
        penalty = tracker.get_confidence_penalty(scores)
        assert penalty >= -tracker.max_penalty

    def test_no_penalty_when_agents_agree(self):
        """If both agents in a toxic pair are currently in agreement, no penalty."""
        tracker = make_tracker()
        conflict = AgentPairConflict(
            agent_a="trend",
            agent_b="risk_sentinel",
            conflict_count=10,
            loss_count=8,
        )
        tracker._toxic_patterns = [conflict]

        # Both agents scoring similarly — no active conflict
        scores = {"trend": 0.7, "risk_sentinel": 0.65}
        penalty = tracker.get_confidence_penalty(scores)
        assert penalty == 0.0


# ---------------------------------------------------------------------------
# 6. record_outcome alias
# ---------------------------------------------------------------------------

class TestRecordOutcomeAlias:
    """record_outcome() should be a working alias for record_trade()."""

    def test_record_outcome_increments_total(self):
        tracker = make_tracker()
        assert tracker._total_trades == 0
        tracker.record_outcome({"a": 0.7, "b": 0.3}, won=True, pnl=20.0)
        assert tracker._total_trades == 1

    def test_record_outcome_updates_bucket_stats(self):
        tracker = make_tracker()
        # Low-disagreement scores → "low" bucket
        tracker.record_outcome({"a": 0.6, "b": 0.65}, won=True)
        assert tracker._bucket_stats["low"]["wins"] == 1


# ---------------------------------------------------------------------------
# 7. Persistence round-trip
# ---------------------------------------------------------------------------

class TestPersistence:
    """Tests for save_state() / _load_state() round-trip."""

    def test_state_survives_save_and_reload(self, tmp_path):
        path = str(tmp_path / "test_state.json")
        tracker = make_tracker(path)

        # Populate with some data
        conflict = AgentPairConflict("trend", "risk_sentinel", conflict_count=8, loss_count=6)
        tracker._pair_conflicts[("risk_sentinel", "trend")] = conflict
        tracker._total_trades = 42
        tracker._bucket_stats["moderate"]["wins"] = 7
        tracker._bucket_stats["moderate"]["losses"] = 3

        tracker.save_state()
        assert os.path.exists(path)

        # Reload into fresh tracker
        tracker2 = make_tracker(path)
        assert tracker2._total_trades == 42
        assert tracker2._bucket_stats["moderate"]["wins"] == 7
        assert tracker2._bucket_stats["moderate"]["losses"] == 3
        assert ("risk_sentinel", "trend") in tracker2._pair_conflicts

    def test_corrupt_state_file_handled_gracefully(self, tmp_path):
        path = str(tmp_path / "corrupt.json")
        with open(path, "w") as f:
            f.write("NOT VALID JSON {{{{")
        # Should not raise — just start fresh
        tracker = make_tracker(path)
        assert tracker._total_trades == 0
        assert len(tracker._pair_conflicts) == 0

    def test_atomic_write_uses_tmp_rename(self, tmp_path):
        """Verify atomic write: .tmp file is written then renamed (no partial writes)."""
        path = str(tmp_path / "atomic.json")
        tracker = make_tracker(path)
        tracker._total_trades = 5

        tracker.save_state()
        # File should exist and be valid JSON
        with open(path) as f:
            data = json.load(f)
        assert data.get("total_trades") == 5
        # No .tmp file should linger
        assert not os.path.exists(path + ".tmp")


# ---------------------------------------------------------------------------
# 8. generate_report structure
# ---------------------------------------------------------------------------

class TestGenerateReport:
    """Tests for generate_report() output structure."""

    def test_report_has_required_keys(self):
        tracker = make_tracker()
        report = tracker.generate_report()
        assert "total_trades" in report
        assert "bucket_win_rates" in report
        assert "toxic_patterns" in report
        assert "toxic_count" in report
        assert "total_pair_conflicts_tracked" in report

    def test_report_win_rates_correct(self):
        """Win rates in report should match bucket stats."""
        tracker = make_tracker()
        tracker._bucket_stats["low"] = {"wins": 3, "losses": 1}
        tracker._total_trades = 4
        report = tracker.generate_report()
        assert report["bucket_win_rates"]["low"]["win_rate"] == pytest.approx(0.75, abs=0.01)
