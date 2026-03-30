"""Tests for Phase 57 US-356: PenaltyAuditor.

Covers:
  - run_audit with no monitors returns OK recommendation
  - run_audit with high breach rate returns CEILING_NEEDED
  - run_audit with overconfidence as top source + low predictions returns CALIBRATION_GUARD_NEEDED
  - run_audit with very low gate_pass_rate returns THRESHOLD_RELAXATION_NEEDED
  - save_audit writes valid JSON
  - save_audit no .tmp file lingers
  - run_audit returns all required keys
"""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.scanner.penalty_audit import PenaltyAuditor


class TestRunAudit:

    def test_no_monitors_returns_ok(self, tmp_path):
        auditor = PenaltyAuditor(audit_path=tmp_path / "audit.json")
        audit = auditor.run_audit()
        assert audit["recommendation"] == "OK"

    def test_returns_all_required_keys(self, tmp_path):
        auditor = PenaltyAuditor(audit_path=tmp_path / "audit.json")
        audit = auditor.run_audit()
        for key in ["top_penalty_source", "breach_rate_pct", "avg_cumulative_penalty",
                    "pairs_above_ceiling", "top_blocking_gate", "gate_pass_rate_pct",
                    "recommendation", "timestamp"]:
            assert key in audit, f"Missing key: {key}"

    def test_high_breach_rate_returns_ceiling_needed(self, tmp_path):
        """breach_rate > 50% → CEILING_NEEDED."""
        auditor = PenaltyAuditor(audit_path=tmp_path / "audit.json")
        # Mock a CPM with high breach_rate
        cpm = MagicMock()
        cpm.get_breach_rate.return_value = 0.85
        cpm.generate_report.return_value = {}
        audit = auditor.run_audit(confidence_penalty_monitor=cpm)
        assert audit["recommendation"] == "CEILING_NEEDED"

    def test_overconfidence_top_source_low_predictions_returns_calibration_guard_needed(self, tmp_path):
        """top_source=overconfidence + calibration_n_predictions < 20 → CALIBRATION_GUARD_NEEDED."""
        auditor = PenaltyAuditor(audit_path=tmp_path / "audit.json")
        cpm = MagicMock()
        cpm.get_breach_rate.return_value = 0.20  # Below CEILING threshold
        cpm.generate_report.return_value = {
            "EUR_USD": {
                "total_penalty": 0.15,
                "sources": {"overconfidence": 0.12, "concept_drift": 0.03},
                "floor_breach": False,
            }
        }
        audit = auditor.run_audit(
            confidence_penalty_monitor=cpm,
            calibration_n_predictions=6,  # < 20
        )
        assert audit["recommendation"] == "CALIBRATION_GUARD_NEEDED"

    def test_low_gate_pass_rate_returns_threshold_relaxation(self, tmp_path):
        """gate_pass_rate_pct < 5% → THRESHOLD_RELAXATION_NEEDED."""
        auditor = PenaltyAuditor(audit_path=tmp_path / "audit.json")
        scan_report = {"gate_pass_rate_pct": 2.0, "stalled": True}
        audit = auditor.run_audit(scan_diagnostics_report=scan_report)
        assert audit["recommendation"] == "THRESHOLD_RELAXATION_NEEDED"

    def test_gate_rejection_tracker_top_gate_used(self, tmp_path):
        auditor = PenaltyAuditor(audit_path=tmp_path / "audit.json")
        grt = MagicMock()
        grt.get_top_blocking_gate.return_value = "momentum_gate"
        audit = auditor.run_audit(gate_rejection_tracker=grt)
        assert audit["top_blocking_gate"] == "momentum_gate"


class TestSaveAudit:

    def test_save_audit_creates_file(self, tmp_path):
        auditor = PenaltyAuditor(audit_path=tmp_path / "audit.json")
        audit = auditor.run_audit()
        auditor.save_audit(audit)
        assert (tmp_path / "audit.json").exists()

    def test_save_audit_valid_json(self, tmp_path):
        auditor = PenaltyAuditor(audit_path=tmp_path / "audit.json")
        audit = auditor.run_audit()
        auditor.save_audit(audit)
        data = json.loads((tmp_path / "audit.json").read_text())
        assert "recommendation" in data

    def test_save_audit_no_tmp_file(self, tmp_path):
        auditor = PenaltyAuditor(audit_path=tmp_path / "audit.json")
        audit = auditor.run_audit()
        auditor.save_audit(audit)
        assert not (tmp_path / "audit.json.tmp").exists()

    def test_save_audit_custom_path(self, tmp_path):
        auditor = PenaltyAuditor()
        audit = auditor.run_audit()
        custom_path = tmp_path / "custom_audit.json"
        auditor.save_audit(audit, path=custom_path)
        assert custom_path.exists()
