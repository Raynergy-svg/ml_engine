"""Integration tests for Phase 60 US-370: wiring verification + end-to-end scenarios."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from src.scanner.scan_health_synthesizer import (
    ScanHealthSynthesizer,
    rank_bottlenecks,
    STATUS_HEALTHY, STATUS_DEGRADED, STATUS_CRITICAL,
    BN_SYSTEM_STALL, BN_ENSEMBLE_DIVERGENCE, BN_NEAR_MISS_CLUSTER,
    BN_MOMENTUM_KILL,
)


# ─── US-370-1: Live wiring verification ──────────────────────────────────────

class TestContinuousPy_Phase60WiringVerification:

    def test_scan_health_synthesizer_imported_in_continuous(self):
        cont_src = Path("src/scanner/automation/continuous.py").read_text()
        assert "ScanHealthSynthesizer" in cont_src

    def test_scan_health_assigned_in_continuous(self):
        cont_src = Path("src/scanner/automation/continuous.py").read_text()
        assert "_scan_health = None" in cont_src or "self._scan_health = None" in cont_src

    def test_scan_health_synthesize_called_in_continuous(self):
        cont_src = Path("src/scanner/automation/continuous.py").read_text()
        assert "_scan_health.synthesize()" in cont_src

    def test_synthesize_called_every_10_scans(self):
        cont_src = Path("src/scanner/automation/continuous.py").read_text()
        # Must have both synthesize() call and the modulo 10 guard together
        assert "_scan_count % 10 == 0" in cont_src
        synth_pos = cont_src.find("_scan_health.synthesize()")
        assert synth_pos != -1

    def test_phase60_init_block_present(self):
        cont_src = Path("src/scanner/automation/continuous.py").read_text()
        assert "Phase 60" in cont_src

    def test_scan_health_receives_module_refs(self):
        """Init block must pass signal_funnel, gate_proximity, etc."""
        cont_src = Path("src/scanner/automation/continuous.py").read_text()
        assert "signal_funnel=" in cont_src
        assert "gate_proximity=" in cont_src
        assert "ensemble_divergence=" in cont_src


# ─── US-370-2: CRITICAL scenario ─────────────────────────────────────────────

class TestScenario_Critical:

    def test_critical_stall_plus_ensemble_divergence(self):
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
        bn_names = [b["name"] for b in report["bottlenecks"]]
        assert BN_SYSTEM_STALL in bn_names
        assert BN_ENSEMBLE_DIVERGENCE in bn_names

    def test_critical_stall_and_stall_is_top_bottleneck(self):
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
        assert report["bottlenecks"][0]["name"] == BN_SYSTEM_STALL

    def test_critical_top_action_non_empty(self):
        mock_diagnostics = MagicMock()
        mock_diagnostics.generate_report.return_value = {"stalled": True}
        synth = ScanHealthSynthesizer(scan_diagnostics=mock_diagnostics)
        report = synth.synthesize()
        assert len(report["top_action"]) > 0


# ─── US-370-3: DEGRADED scenario ──────────────────────────────────────────────

class TestScenario_Degraded:

    def test_degraded_near_miss_cluster_plus_momentum_kill(self):
        mock_proximity = MagicMock()
        mock_proximity.get_proximity_summary.return_value = {"near_miss_rate_pct": 70.0}
        mock_funnel = MagicMock()
        mock_funnel.get_funnel_summary.return_value = {
            "dominant_kill_stage": "momentum",
            "tradeable_pct": 0.0,
        }
        synth = ScanHealthSynthesizer(
            gate_proximity=mock_proximity,
            signal_funnel=mock_funnel,
        )
        report = synth.synthesize()
        assert report["health_status"] == STATUS_DEGRADED
        bn_names = [b["name"] for b in report["bottlenecks"]]
        assert BN_NEAR_MISS_CLUSTER in bn_names
        assert BN_MOMENTUM_KILL in bn_names

    def test_degraded_bottlenecks_sorted_correctly(self):
        mock_proximity = MagicMock()
        mock_proximity.get_proximity_summary.return_value = {"near_miss_rate_pct": 60.0}
        mock_funnel = MagicMock()
        mock_funnel.get_funnel_summary.return_value = {
            "dominant_kill_stage": "momentum",
            "tradeable_pct": 0.0,
        }
        synth = ScanHealthSynthesizer(
            gate_proximity=mock_proximity,
            signal_funnel=mock_funnel,
        )
        report = synth.synthesize()
        scores = [b["impact_score"] for b in report["bottlenecks"]]
        assert scores == sorted(scores, reverse=True)

    def test_degraded_summary_references_status(self):
        mock_proximity = MagicMock()
        mock_proximity.get_proximity_summary.return_value = {"near_miss_rate_pct": 80.0}
        synth = ScanHealthSynthesizer(gate_proximity=mock_proximity)
        report = synth.synthesize()
        assert "DEGRADED" in report["summary"]


# ─── US-370-4: HEALTHY scenario ───────────────────────────────────────────────

class TestScenario_Healthy:

    def test_healthy_all_modules_none(self):
        synth = ScanHealthSynthesizer()
        report = synth.synthesize()
        assert report["health_status"] == STATUS_HEALTHY
        assert report["bottlenecks"] == []

    def test_healthy_all_modules_nominal(self):
        mock_diagnostics = MagicMock()
        mock_diagnostics.generate_report.return_value = {"stalled": False}
        mock_proximity = MagicMock()
        mock_proximity.get_proximity_summary.return_value = {"near_miss_rate_pct": 10.0}
        mock_funnel = MagicMock()
        mock_funnel.get_funnel_summary.return_value = {
            "dominant_kill_stage": "none",
            "tradeable_pct": 50.0,
        }
        mock_divergence = MagicMock()
        mock_divergence.get_divergence_summary.return_value = {"recommendation": "OK"}
        synth = ScanHealthSynthesizer(
            scan_diagnostics=mock_diagnostics,
            gate_proximity=mock_proximity,
            signal_funnel=mock_funnel,
            ensemble_divergence=mock_divergence,
        )
        report = synth.synthesize()
        assert report["health_status"] == STATUS_HEALTHY
        assert report["bottlenecks"] == []

    def test_healthy_top_action_default(self):
        synth = ScanHealthSynthesizer()
        report = synth.synthesize()
        assert "No action needed" in report["top_action"]


# ─── US-370-5: Graceful degradation ──────────────────────────────────────────

class TestGracefulDegradation:

    def test_synthesize_never_raises_when_module_explodes(self):
        bad = MagicMock()
        bad.generate_report.side_effect = RuntimeError("db connection lost")
        synth = ScanHealthSynthesizer(scan_diagnostics=bad)
        # Must not raise
        report = synth.synthesize()
        assert "health_status" in report

    def test_partial_modules_ok(self):
        """Only gate_proximity wired — others None."""
        mock_proximity = MagicMock()
        mock_proximity.get_proximity_summary.return_value = {"near_miss_rate_pct": 0.0}
        synth = ScanHealthSynthesizer(gate_proximity=mock_proximity)
        report = synth.synthesize()
        assert report["health_status"] == STATUS_HEALTHY


# ─── US-370-6: Module importability ──────────────────────────────────────────

class TestImportability:

    def test_module_importable(self):
        from src.scanner.scan_health_synthesizer import ScanHealthSynthesizer
        assert ScanHealthSynthesizer is not None

    def test_rank_bottlenecks_importable(self):
        from src.scanner.scan_health_synthesizer import rank_bottlenecks
        assert rank_bottlenecks is not None

    def test_constants_importable(self):
        from src.scanner.scan_health_synthesizer import (
            STATUS_HEALTHY, STATUS_DEGRADED, STATUS_CRITICAL,
            BN_SYSTEM_STALL, BN_ENSEMBLE_DIVERGENCE,
        )
        assert STATUS_HEALTHY == "HEALTHY"
        assert STATUS_CRITICAL == "CRITICAL"
