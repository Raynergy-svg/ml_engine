"""Tests for Phase 58 US-361: AdaptiveConfidenceFloor.

Covers:
  - should_adapt False when cooldown active
  - should_adapt False when not stalled
  - should_adapt False when max adaptations reached
  - should_adapt False when signal_exists (Phase 57 should handle)
  - should_adapt True when all conditions met
  - apply_adjustment respects safety_floor
  - apply_adjustment mutates config.min_confidence
  - cooldown decrements correctly via tick()
  - state persistence round-trip
  - max_adaptations_reached flag in get_state
  - compute_adjustment clamps at safety_floor
"""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from src.scanner.adaptive_confidence_floor import AdaptiveConfidenceFloor
from src.scanner.threshold_gap_analyzer import ThresholdGapReport, REC_THRESHOLD_TOO_HIGH, REC_NEAR_THRESHOLD, REC_OK


def _make_gap_report(rec=REC_THRESHOLD_TOO_HIGH, threshold=58.0):
    return ThresholdGapReport(
        count=50,
        mean_raw=50.0,
        median_raw=50.0,
        p75_raw=52.0,
        current_threshold=threshold,
        gap_at_median=8.0,
        gap_at_p75=6.0,
        signals_above_threshold_pct=0.0,
        recommendation=rec,
    )


def _predictor_no_signal():
    return {"signal_exists": False, "penalty_free_pass_rate_pct": 0.0}


def _predictor_signal_exists():
    return {"signal_exists": True, "penalty_free_pass_rate_pct": 15.0}


class TestShouldAdapt:

    def test_false_when_not_stalled(self, tmp_path):
        floor = AdaptiveConfidenceFloor(
            stall_threshold=50, state_path=tmp_path / "af.json"
        )
        report = _make_gap_report()
        assert floor.should_adapt(report, _predictor_no_signal(), scans_since_last_trade=30) is False

    def test_false_when_cooldown_active(self, tmp_path):
        floor = AdaptiveConfidenceFloor(
            cooldown_scans=20, state_path=tmp_path / "af.json"
        )
        floor._cooldown_remaining = 5
        report = _make_gap_report()
        assert floor.should_adapt(report, _predictor_no_signal(), scans_since_last_trade=55) is False

    def test_false_when_max_adaptations_reached(self, tmp_path):
        floor = AdaptiveConfidenceFloor(
            max_total_adaptations=3, state_path=tmp_path / "af.json"
        )
        floor._total_adaptations = 3
        report = _make_gap_report()
        assert floor.should_adapt(report, _predictor_no_signal(), scans_since_last_trade=55) is False

    def test_false_when_ok_recommendation(self, tmp_path):
        floor = AdaptiveConfidenceFloor(state_path=tmp_path / "af.json")
        report = _make_gap_report(rec=REC_OK)
        assert floor.should_adapt(report, _predictor_no_signal(), scans_since_last_trade=55) is False

    def test_false_when_signal_exists_penalty_phase57_is_fix(self, tmp_path):
        """If signals exist penalty-free, Phase 57 handles it — don't lower threshold."""
        floor = AdaptiveConfidenceFloor(state_path=tmp_path / "af.json")
        report = _make_gap_report(rec=REC_THRESHOLD_TOO_HIGH)
        assert floor.should_adapt(report, _predictor_signal_exists(), scans_since_last_trade=55) is False

    def test_true_when_all_conditions_met(self, tmp_path):
        floor = AdaptiveConfidenceFloor(
            stall_threshold=50, cooldown_scans=5,
            state_path=tmp_path / "af.json",
        )
        report = _make_gap_report(rec=REC_THRESHOLD_TOO_HIGH)
        assert floor.should_adapt(report, _predictor_no_signal(), scans_since_last_trade=55) is True

    def test_true_for_near_threshold_recommendation(self, tmp_path):
        floor = AdaptiveConfidenceFloor(state_path=tmp_path / "af.json")
        report = _make_gap_report(rec=REC_NEAR_THRESHOLD)
        assert floor.should_adapt(report, _predictor_no_signal(), scans_since_last_trade=55) is True


class TestApplyAdjustment:

    def test_mutates_config_min_confidence(self, tmp_path):
        floor = AdaptiveConfidenceFloor(
            initial_threshold=58.0, state_path=tmp_path / "af.json"
        )
        config = SimpleNamespace(min_confidence=58.0)
        floor.apply_adjustment(config, delta=-1.0)
        assert config.min_confidence == pytest.approx(57.0, abs=0.01)

    def test_respects_safety_floor(self, tmp_path):
        floor = AdaptiveConfidenceFloor(
            safety_floor=45.0, initial_threshold=45.5,
            state_path=tmp_path / "af.json",
        )
        config = SimpleNamespace(min_confidence=45.5)
        floor.apply_adjustment(config, delta=-1.0)
        assert config.min_confidence == pytest.approx(45.0, abs=0.01)

    def test_increments_total_adaptations(self, tmp_path):
        floor = AdaptiveConfidenceFloor(state_path=tmp_path / "af.json")
        config = SimpleNamespace(min_confidence=58.0)
        floor.apply_adjustment(config, delta=-1.0)
        assert floor._total_adaptations == 1

    def test_sets_cooldown(self, tmp_path):
        floor = AdaptiveConfidenceFloor(
            cooldown_scans=20, state_path=tmp_path / "af.json"
        )
        config = SimpleNamespace(min_confidence=58.0)
        floor.apply_adjustment(config, delta=-1.0)
        assert floor._cooldown_remaining == 20


class TestTick:

    def test_tick_decrements_cooldown(self, tmp_path):
        floor = AdaptiveConfidenceFloor(state_path=tmp_path / "af.json")
        floor._cooldown_remaining = 5
        floor.tick()
        assert floor._cooldown_remaining == 4

    def test_tick_does_not_go_below_zero(self, tmp_path):
        floor = AdaptiveConfidenceFloor(state_path=tmp_path / "af.json")
        floor._cooldown_remaining = 0
        floor.tick()
        assert floor._cooldown_remaining == 0

    def test_cooldown_reaches_zero_after_n_ticks(self, tmp_path):
        floor = AdaptiveConfidenceFloor(
            cooldown_scans=5, state_path=tmp_path / "af.json"
        )
        config = SimpleNamespace(min_confidence=58.0)
        floor.apply_adjustment(config, delta=-1.0)
        for _ in range(5):
            floor.tick()
        assert floor._cooldown_remaining == 0


class TestGetState:

    def test_get_state_all_keys(self, tmp_path):
        floor = AdaptiveConfidenceFloor(state_path=tmp_path / "af.json")
        state = floor.get_state()
        for key in ["current_threshold", "total_adaptations", "cooldown_remaining",
                    "last_adaptation_ts", "max_adaptations_reached", "safety_floor"]:
            assert key in state

    def test_max_adaptations_reached_flag(self, tmp_path):
        floor = AdaptiveConfidenceFloor(
            max_total_adaptations=2, state_path=tmp_path / "af.json"
        )
        floor._total_adaptations = 2
        state = floor.get_state()
        assert state["max_adaptations_reached"] is True


class TestPersistence:

    def test_save_load_round_trip(self, tmp_path):
        p = tmp_path / "af.json"
        f1 = AdaptiveConfidenceFloor(initial_threshold=58.0, state_path=p)
        config = SimpleNamespace(min_confidence=58.0)
        f1.apply_adjustment(config, delta=-1.0)
        f1.save_state()

        f2 = AdaptiveConfidenceFloor(initial_threshold=58.0, state_path=p)
        assert f2._current_threshold == pytest.approx(57.0, abs=0.01)
        assert f2._total_adaptations == 1
        assert f2._cooldown_remaining == 20  # default cooldown

    def test_save_writes_valid_json(self, tmp_path):
        p = tmp_path / "af.json"
        floor = AdaptiveConfidenceFloor(state_path=p)
        floor.save_state()
        data = json.loads(p.read_text())
        assert "current_threshold" in data
        assert "total_adaptations" in data

    def test_load_missing_file_no_error(self, tmp_path):
        floor = AdaptiveConfidenceFloor(state_path=tmp_path / "nonexistent.json")
        assert floor._total_adaptations == 0
