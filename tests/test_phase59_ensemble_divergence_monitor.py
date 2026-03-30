"""Tests for Phase 59 US-365: EnsembleDivergenceMonitor."""

import pytest
from src.scanner.ensemble_divergence_monitor import (
    EnsembleDivergenceMonitor,
    REC_HIGH_DISAGREEMENT, REC_OK,
)


class TestRecord:

    def test_record_computes_divergence(self, tmp_path):
        mon = EnsembleDivergenceMonitor(state_path=tmp_path / "ed.jsonl")
        div = mon.record("EUR_USD", tcn_confidence=72.0, ridge_confidence=55.0)
        assert div == pytest.approx(17.0, abs=0.01)

    def test_mean_divergence_after_records(self, tmp_path):
        mon = EnsembleDivergenceMonitor(state_path=tmp_path / "ed.jsonl")
        mon.record("EUR_USD", 70.0, 50.0)  # div=20
        mon.record("EUR_USD", 60.0, 60.0)  # div=0
        assert mon.get_mean_divergence() == pytest.approx(10.0, abs=0.01)

    def test_empty_mean_divergence_zero(self, tmp_path):
        mon = EnsembleDivergenceMonitor(state_path=tmp_path / "ed.jsonl")
        assert mon.get_mean_divergence() == 0.0


class TestIsHighDivergence:

    def test_false_when_low_divergence(self, tmp_path):
        mon = EnsembleDivergenceMonitor(high_divergence_threshold=20.0, state_path=tmp_path / "ed.jsonl")
        mon.record("EUR_USD", 60.0, 58.0)  # div=2
        assert mon.is_high_divergence() is False

    def test_true_when_high_divergence(self, tmp_path):
        mon = EnsembleDivergenceMonitor(high_divergence_threshold=20.0, state_path=tmp_path / "ed.jsonl")
        for _ in range(5):
            mon.record("EUR_USD", 75.0, 50.0)  # div=25
        assert mon.is_high_divergence() is True

    def test_custom_threshold(self, tmp_path):
        mon = EnsembleDivergenceMonitor(state_path=tmp_path / "ed.jsonl")
        mon.record("EUR_USD", 70.0, 60.0)  # div=10
        assert mon.is_high_divergence(threshold=5.0) is True
        assert mon.is_high_divergence(threshold=15.0) is False


class TestGetDivergenceSummary:

    def test_ok_recommendation_when_low_rate(self, tmp_path):
        mon = EnsembleDivergenceMonitor(
            high_divergence_threshold=20.0,
            high_rate_threshold=0.30,
            state_path=tmp_path / "ed.jsonl",
        )
        # 5 low, 1 high → rate = 1/6 ≈ 17% < 30%
        for _ in range(5):
            mon.record("EUR_USD", 60.0, 58.0)  # div=2
        mon.record("EUR_USD", 75.0, 50.0)  # div=25
        summary = mon.get_divergence_summary()
        assert summary["recommendation"] == REC_OK

    def test_high_disagreement_recommendation(self, tmp_path):
        mon = EnsembleDivergenceMonitor(
            high_divergence_threshold=20.0,
            high_rate_threshold=0.30,
            state_path=tmp_path / "ed.jsonl",
        )
        # 10 high divergence → rate = 100% > 30%
        for _ in range(10):
            mon.record("EUR_USD", 80.0, 55.0)  # div=25
        summary = mon.get_divergence_summary()
        assert summary["recommendation"] == REC_HIGH_DISAGREEMENT
        assert summary["high_divergence_rate_pct"] == pytest.approx(100.0, abs=0.1)

    def test_summary_all_required_keys(self, tmp_path):
        mon = EnsembleDivergenceMonitor(state_path=tmp_path / "ed.jsonl")
        mon.record("EUR_USD", 70.0, 55.0)
        summary = mon.get_divergence_summary()
        for key in ["count", "mean_divergence", "max_divergence",
                    "high_divergence_rate_pct", "recommendation"]:
            assert key in summary

    def test_empty_summary_safe(self, tmp_path):
        mon = EnsembleDivergenceMonitor(state_path=tmp_path / "ed.jsonl")
        summary = mon.get_divergence_summary()
        assert summary["count"] == 0
        assert summary["recommendation"] == REC_OK


class TestPerPairSummary:

    def test_per_pair_filtering(self, tmp_path):
        mon = EnsembleDivergenceMonitor(state_path=tmp_path / "ed.jsonl")
        mon.record("EUR_USD", 75.0, 50.0)   # div=25
        mon.record("GBP_USD", 60.0, 58.0)   # div=2
        eur = mon.get_per_pair_summary("EUR_USD")
        assert eur["count"] == 1
        assert eur["mean_divergence"] == pytest.approx(25.0, abs=0.01)

    def test_per_pair_empty(self, tmp_path):
        mon = EnsembleDivergenceMonitor(state_path=tmp_path / "ed.jsonl")
        summary = mon.get_per_pair_summary("EUR_USD")
        assert summary["count"] == 0


class TestPersistence:

    def test_save_load_round_trip(self, tmp_path):
        p = tmp_path / "ed.jsonl"
        m1 = EnsembleDivergenceMonitor(state_path=p)
        m1.record("EUR_USD", 75.0, 50.0)
        m1.save_state()
        m2 = EnsembleDivergenceMonitor(state_path=p)
        assert m2.get_divergence_summary()["count"] == 1
        assert m2.get_mean_divergence() == pytest.approx(25.0, abs=0.01)
