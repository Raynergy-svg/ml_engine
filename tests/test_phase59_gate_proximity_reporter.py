"""Tests for Phase 59 US-364: GateProximityReporter."""

import pytest
from types import SimpleNamespace
from src.scanner.gate_proximity_reporter import (
    GateProximityReporter,
    MISS_CLASS_NEAR, MISS_CLASS_MODERATE, MISS_CLASS_FAR,
)


def _make_conf_reject(pair, confidence):
    """Pair that failed confidence gate."""
    return SimpleNamespace(
        pair=pair,
        confidence=confidence,
        confidence_gate_passed=False,
        momentum_gate_passed=False,
        risk_gate_passed=False,
        is_tradeable=False,
    )


def _make_conf_pass(pair, confidence):
    """Pair that passed confidence gate."""
    return SimpleNamespace(
        pair=pair,
        confidence=confidence,
        confidence_gate_passed=True,
        momentum_gate_passed=True,
        risk_gate_passed=True,
        is_tradeable=True,
    )


class TestMissClassification:

    def test_near_miss_classification(self, tmp_path):
        reporter = GateProximityReporter(state_path=tmp_path / "gp.jsonl")
        # gap = 58.0 - 56.0 = 2.0 → NEAR_MISS
        reporter.record_scan([_make_conf_reject("EUR_USD", 56.0)], min_confidence=58.0)
        summary = reporter.get_proximity_summary()
        assert summary["near_miss_count"] == 1
        assert summary["near_miss_rate_pct"] == pytest.approx(100.0, abs=0.1)

    def test_moderate_miss_classification(self, tmp_path):
        reporter = GateProximityReporter(state_path=tmp_path / "gp.jsonl")
        # gap = 58.0 - 52.0 = 6.0 → MODERATE_MISS
        reporter.record_scan([_make_conf_reject("EUR_USD", 52.0)], min_confidence=58.0)
        summary = reporter.get_proximity_summary()
        assert summary["near_miss_count"] == 0

    def test_far_miss_classification(self, tmp_path):
        reporter = GateProximityReporter(state_path=tmp_path / "gp.jsonl")
        # gap = 58.0 - 45.0 = 13.0 → FAR_MISS
        reporter.record_scan([_make_conf_reject("EUR_USD", 45.0)], min_confidence=58.0)
        summary = reporter.get_proximity_summary()
        assert summary["near_miss_count"] == 0

    def test_skips_conf_pass_pairs(self, tmp_path):
        reporter = GateProximityReporter(state_path=tmp_path / "gp.jsonl")
        # Mix of passes and rejections — only rejects should be recorded
        analyses = [
            _make_conf_pass("EUR_USD", 62.0),
            _make_conf_reject("GBP_USD", 56.5),  # near miss
        ]
        reporter.record_scan(analyses, min_confidence=58.0)
        summary = reporter.get_proximity_summary()
        assert summary["total_conf_rejects"] == 1


class TestGetNearMissRate:

    def test_near_miss_rate_correct_fraction(self, tmp_path):
        reporter = GateProximityReporter(state_path=tmp_path / "gp.jsonl")
        # 3 near misses, 3 far misses → 50%
        near = [_make_conf_reject(f"P{i}", 56.0) for i in range(3)]  # gap=2
        far = [_make_conf_reject(f"P{i+3}", 45.0) for i in range(3)]  # gap=13
        reporter.record_scan(near + far, min_confidence=58.0)
        rate = reporter.get_near_miss_rate()
        assert rate == pytest.approx(0.5, abs=0.01)

    def test_empty_reporter_returns_zero(self, tmp_path):
        reporter = GateProximityReporter(state_path=tmp_path / "gp.jsonl")
        assert reporter.get_near_miss_rate() == 0.0


class TestGetProximitySummary:

    def test_summary_all_required_fields(self, tmp_path):
        reporter = GateProximityReporter(state_path=tmp_path / "gp.jsonl")
        reporter.record_scan([_make_conf_reject("EUR_USD", 55.0)], min_confidence=58.0)
        summary = reporter.get_proximity_summary()
        for key in ["total_conf_rejects", "near_miss_count", "near_miss_rate_pct",
                    "avg_gap", "closest_gap", "closest_pair"]:
            assert key in summary

    def test_closest_gap_and_pair(self, tmp_path):
        reporter = GateProximityReporter(state_path=tmp_path / "gp.jsonl")
        analyses = [
            _make_conf_reject("EUR_USD", 57.0),  # gap=1.0
            _make_conf_reject("GBP_USD", 50.0),  # gap=8.0
        ]
        reporter.record_scan(analyses, min_confidence=58.0)
        summary = reporter.get_proximity_summary()
        assert summary["closest_gap"] == pytest.approx(1.0, abs=0.01)
        assert summary["closest_pair"] == "EUR_USD"

    def test_empty_summary_safe(self, tmp_path):
        reporter = GateProximityReporter(state_path=tmp_path / "gp.jsonl")
        summary = reporter.get_proximity_summary()
        assert summary["total_conf_rejects"] == 0


class TestGetActionableThreshold:

    def test_threshold_below_lowest_reject(self, tmp_path):
        reporter = GateProximityReporter(state_path=tmp_path / "gp.jsonl")
        # 10 pairs rejected at 55.0 → unblock 10% at ~54.9
        for _ in range(10):
            reporter.record_scan([_make_conf_reject("EUR_USD", 55.0)], min_confidence=58.0)
        t = reporter.get_actionable_threshold(target_near_miss_unblock=0.10)
        assert t <= 55.0 + 0.1

    def test_empty_returns_zero(self, tmp_path):
        reporter = GateProximityReporter(state_path=tmp_path / "gp.jsonl")
        assert reporter.get_actionable_threshold() == 0.0


class TestPersistence:

    def test_save_load_round_trip(self, tmp_path):
        p = tmp_path / "gp.jsonl"
        r1 = GateProximityReporter(state_path=p)
        r1.record_scan([_make_conf_reject("EUR_USD", 56.0)], min_confidence=58.0)
        r1.save_state()
        r2 = GateProximityReporter(state_path=p)
        assert r2.get_proximity_summary()["total_conf_rejects"] == 1
