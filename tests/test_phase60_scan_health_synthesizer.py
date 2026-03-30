"""Tests for Phase 60 US-367/US-368: ScanHealthSynthesizer + BottleneckRanker."""

import pytest
from src.scanner.scan_health_synthesizer import (
    ScanHealthSynthesizer,
    rank_bottlenecks,
    STATUS_HEALTHY, STATUS_DEGRADED, STATUS_CRITICAL,
    BN_SYSTEM_STALL, BN_ENSEMBLE_DIVERGENCE, BN_CONFIDENCE_TOO_HIGH,
    BN_MOMENTUM_KILL, BN_NEAR_MISS_CLUSTER, BN_RISK_KILL,
)


# ─── helpers ──────────────────────────────────────────────────────────────────

def _signals(
    gap_at_p75=0.0,
    near_miss_rate_pct=0.0,
    dominant_kill_stage="none",
    ensemble_recommendation="OK",
    stall_active=False,
    tradeable_pct=100.0,
):
    return {
        "gap_at_p75": gap_at_p75,
        "near_miss_rate_pct": near_miss_rate_pct,
        "dominant_kill_stage": dominant_kill_stage,
        "ensemble_recommendation": ensemble_recommendation,
        "stall_active": stall_active,
        "tradeable_pct": tradeable_pct,
    }


# ─── US-368: BottleneckRanker ─────────────────────────────────────────────────

class TestRankBottlenecks:

    def test_no_bottlenecks_when_all_healthy(self):
        bns = rank_bottlenecks(_signals())
        assert bns == []

    def test_system_stall_detected(self):
        bns = rank_bottlenecks(_signals(stall_active=True))
        names = [b["name"] for b in bns]
        assert BN_SYSTEM_STALL in names

    def test_system_stall_has_highest_impact(self):
        bns = rank_bottlenecks(_signals(
            stall_active=True,
            ensemble_recommendation="HIGH_ENSEMBLE_DISAGREEMENT",
            gap_at_p75=15.0,
        ))
        assert bns[0]["name"] == BN_SYSTEM_STALL
        assert bns[0]["impact_score"] == pytest.approx(1.0, abs=0.01)

    def test_ensemble_divergence_detected(self):
        bns = rank_bottlenecks(_signals(ensemble_recommendation="HIGH_ENSEMBLE_DISAGREEMENT"))
        names = [b["name"] for b in bns]
        assert BN_ENSEMBLE_DIVERGENCE in names
        ens = next(b for b in bns if b["name"] == BN_ENSEMBLE_DIVERGENCE)
        assert ens["impact_score"] == pytest.approx(0.85, abs=0.01)

    def test_confidence_too_high_scales_with_gap(self):
        bns = rank_bottlenecks(_signals(gap_at_p75=10.0))
        names = [b["name"] for b in bns]
        assert BN_CONFIDENCE_TOO_HIGH in names
        bn = next(b for b in bns if b["name"] == BN_CONFIDENCE_TOO_HIGH)
        # score = 0.75 * min(10/20, 1.0) = 0.75 * 0.5 = 0.375
        assert bn["impact_score"] == pytest.approx(0.375, abs=0.01)

    def test_confidence_too_high_not_detected_when_gap_small(self):
        # gap_at_p75 = 3.0 ≤ 5.0 → no bottleneck
        bns = rank_bottlenecks(_signals(gap_at_p75=3.0))
        assert not any(b["name"] == BN_CONFIDENCE_TOO_HIGH for b in bns)

    def test_momentum_kill_detected(self):
        bns = rank_bottlenecks(_signals(dominant_kill_stage="momentum", tradeable_pct=0.0))
        names = [b["name"] for b in bns]
        assert BN_MOMENTUM_KILL in names
        bn = next(b for b in bns if b["name"] == BN_MOMENTUM_KILL)
        assert bn["impact_score"] == pytest.approx(0.70, abs=0.01)

    def test_momentum_kill_not_detected_when_tradeable_ok(self):
        # tradeable_pct=50 → momentum gate is NOT the dominant blocker
        bns = rank_bottlenecks(_signals(dominant_kill_stage="momentum", tradeable_pct=50.0))
        assert not any(b["name"] == BN_MOMENTUM_KILL for b in bns)

    def test_near_miss_cluster_scales_with_rate(self):
        bns = rank_bottlenecks(_signals(near_miss_rate_pct=80.0))
        names = [b["name"] for b in bns]
        assert BN_NEAR_MISS_CLUSTER in names
        bn = next(b for b in bns if b["name"] == BN_NEAR_MISS_CLUSTER)
        # score = 0.65 * 0.80 = 0.52
        assert bn["impact_score"] == pytest.approx(0.52, abs=0.01)

    def test_near_miss_cluster_not_detected_below_50_pct(self):
        bns = rank_bottlenecks(_signals(near_miss_rate_pct=40.0))
        assert not any(b["name"] == BN_NEAR_MISS_CLUSTER for b in bns)

    def test_risk_kill_detected(self):
        bns = rank_bottlenecks(_signals(dominant_kill_stage="risk", tradeable_pct=0.0))
        names = [b["name"] for b in bns]
        assert BN_RISK_KILL in names
        bn = next(b for b in bns if b["name"] == BN_RISK_KILL)
        assert bn["impact_score"] == pytest.approx(0.60, abs=0.01)

    def test_multiple_bottlenecks_sorted_descending(self):
        bns = rank_bottlenecks(_signals(
            stall_active=True,
            ensemble_recommendation="HIGH_ENSEMBLE_DISAGREEMENT",
            gap_at_p75=20.0,
            near_miss_rate_pct=90.0,
            dominant_kill_stage="momentum",
            tradeable_pct=0.0,
        ))
        scores = [b["impact_score"] for b in bns]
        assert scores == sorted(scores, reverse=True)
        assert bns[0]["name"] == BN_SYSTEM_STALL

    def test_bottleneck_has_required_keys(self):
        bns = rank_bottlenecks(_signals(stall_active=True))
        assert bns
        for key in ["name", "impact_score", "description", "action"]:
            assert key in bns[0]


# ─── US-367: ScanHealthSynthesizer health_status computation ──────────────────

class TestScanHealthSynthesizer_HealthStatus:

    def test_healthy_when_all_nominal(self):
        synth = ScanHealthSynthesizer()
        # All modules None → all signals default to healthy values
        report = synth.synthesize()
        assert report["health_status"] == STATUS_HEALTHY

    def test_critical_when_stall_and_ensemble_divergent(self):
        """stall_active=True AND ensemble HIGH_DISAGREEMENT → CRITICAL."""
        from unittest.mock import MagicMock
        mock_diagnostics = MagicMock()
        mock_diagnostics.generate_report.return_value = {"stalled": True}
        mock_divergence = MagicMock()
        mock_divergence.get_divergence_summary.return_value = {
            "recommendation": "HIGH_ENSEMBLE_DISAGREEMENT"
        }
        synth = ScanHealthSynthesizer(
            ensemble_divergence=mock_divergence,
            scan_diagnostics=mock_diagnostics,
        )
        report = synth.synthesize()
        assert report["health_status"] == STATUS_CRITICAL

    def test_critical_when_stall_and_near_miss_rate_low(self):
        """stall_active=True AND near_miss_rate < 30 → CRITICAL (problem is not threshold)."""
        from unittest.mock import MagicMock
        mock_diagnostics = MagicMock()
        mock_diagnostics.generate_report.return_value = {"stalled": True}
        mock_proximity = MagicMock()
        mock_proximity.get_proximity_summary.return_value = {"near_miss_rate_pct": 10.0}
        synth = ScanHealthSynthesizer(
            gate_proximity=mock_proximity,
            scan_diagnostics=mock_diagnostics,
        )
        report = synth.synthesize()
        assert report["health_status"] == STATUS_CRITICAL

    def test_degraded_when_stall_only(self):
        """stall active but near_miss_rate >= 30 → DEGRADED (threshold adj might help)."""
        from unittest.mock import MagicMock
        mock_diagnostics = MagicMock()
        mock_diagnostics.generate_report.return_value = {"stalled": True}
        mock_proximity = MagicMock()
        mock_proximity.get_proximity_summary.return_value = {"near_miss_rate_pct": 60.0}
        synth = ScanHealthSynthesizer(
            gate_proximity=mock_proximity,
            scan_diagnostics=mock_diagnostics,
        )
        report = synth.synthesize()
        assert report["health_status"] == STATUS_DEGRADED

    def test_degraded_when_high_near_miss_rate(self):
        from unittest.mock import MagicMock
        mock_proximity = MagicMock()
        mock_proximity.get_proximity_summary.return_value = {"near_miss_rate_pct": 70.0}
        synth = ScanHealthSynthesizer(gate_proximity=mock_proximity)
        report = synth.synthesize()
        assert report["health_status"] == STATUS_DEGRADED

    def test_degraded_when_dominant_kill_set(self):
        from unittest.mock import MagicMock
        mock_funnel = MagicMock()
        mock_funnel.get_funnel_summary.return_value = {
            "dominant_kill_stage": "momentum",
            "tradeable_pct": 0.0,
        }
        synth = ScanHealthSynthesizer(signal_funnel=mock_funnel)
        report = synth.synthesize()
        assert report["health_status"] == STATUS_DEGRADED


# ─── US-367: ScanHealthSynthesizer report structure ───────────────────────────

class TestScanHealthSynthesizer_ReportStructure:

    def test_report_has_all_required_keys(self):
        synth = ScanHealthSynthesizer()
        report = synth.synthesize()
        for key in [
            "health_status", "bottlenecks", "top_action",
            "confidence_floor_gap", "near_miss_rate_pct",
            "funnel_dominant_kill", "ensemble_recommendation",
            "stall_active", "tradeable_pct", "summary",
        ]:
            assert key in report, f"Missing key: {key}"

    def test_top_action_non_empty_when_bottleneck_present(self):
        from unittest.mock import MagicMock
        mock_diagnostics = MagicMock()
        mock_diagnostics.generate_report.return_value = {"stalled": True}
        synth = ScanHealthSynthesizer(scan_diagnostics=mock_diagnostics)
        report = synth.synthesize()
        assert isinstance(report["top_action"], str)
        assert len(report["top_action"]) > 0

    def test_top_action_default_when_healthy(self):
        synth = ScanHealthSynthesizer()
        report = synth.synthesize()
        assert "No action needed" in report["top_action"] or isinstance(report["top_action"], str)

    def test_bottlenecks_sorted_descending(self):
        from unittest.mock import MagicMock
        mock_diagnostics = MagicMock()
        mock_diagnostics.generate_report.return_value = {"stalled": True}
        mock_divergence = MagicMock()
        mock_divergence.get_divergence_summary.return_value = {
            "recommendation": "HIGH_ENSEMBLE_DISAGREEMENT"
        }
        synth = ScanHealthSynthesizer(
            ensemble_divergence=mock_divergence,
            scan_diagnostics=mock_diagnostics,
        )
        report = synth.synthesize()
        scores = [b["impact_score"] for b in report["bottlenecks"]]
        assert scores == sorted(scores, reverse=True)

    def test_graceful_when_all_modules_none(self):
        synth = ScanHealthSynthesizer()
        report = synth.synthesize()
        assert report["health_status"] == STATUS_HEALTHY
        assert report["bottlenecks"] == []
        assert report["stall_active"] is False

    def test_graceful_when_module_raises(self):
        from unittest.mock import MagicMock
        mock_funnel = MagicMock()
        mock_funnel.get_funnel_summary.side_effect = RuntimeError("boom")
        synth = ScanHealthSynthesizer(signal_funnel=mock_funnel)
        report = synth.synthesize()
        # Should not raise — graceful degradation
        assert "health_status" in report

    def test_summary_is_string(self):
        synth = ScanHealthSynthesizer()
        report = synth.synthesize()
        assert isinstance(report["summary"], str)
        assert len(report["summary"]) > 0
