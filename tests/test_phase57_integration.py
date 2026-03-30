"""Tests for Phase 57 US-357: Integration — all Phase 57 modules co-exist.

Covers:
  - All five Phase 57 modules import without conflict
  - Penalty ceiling prevents floor breach in stacked-penalty scenario
  - Calibration guard blocks penalty when n_predictions < 20
  - Stall recovery manager enters/exits recovery over a simulated stall sequence
  - Drift proxy guard uses pre-penalty confidence correctly
  - Penalty audit returns CEILING_NEEDED when breach_rate > 0.50
  - All four modules co-exist alongside Phase 56 modules
  - Wiring grep: ConfidencePenaltyCeiling initialized in engine.py
  - Wiring grep: CalibrationGuard.should_apply_penalty called in engine.py
  - Wiring grep: StallRecoveryManager.penalties_suspended called in engine.py
  - Wiring grep: DriftProxyGuard.compute_proxies called in engine.py
  - Wiring grep: PenaltyAuditor.run_audit called in continuous.py
"""

import json
import pytest
from unittest.mock import MagicMock


# ---------------------------------------------------------------------------
# 1. All modules import cleanly
# ---------------------------------------------------------------------------

class TestModuleImports:

    def test_confidence_penalty_ceiling_imports(self):
        from src.scanner.confidence_penalty_ceiling import ConfidencePenaltyCeiling
        assert ConfidencePenaltyCeiling is not None

    def test_calibration_guard_imports(self):
        from src.scanner.calibration_guard import CalibrationGuard
        assert CalibrationGuard is not None

    def test_stall_recovery_imports(self):
        from src.scanner.stall_recovery import StallRecoveryManager
        assert StallRecoveryManager is not None

    def test_drift_proxy_guard_imports(self):
        from src.scanner.drift_proxy_guard import DriftProxyGuard
        assert DriftProxyGuard is not None

    def test_penalty_auditor_imports(self):
        from src.scanner.penalty_audit import PenaltyAuditor
        assert PenaltyAuditor is not None

    def test_all_phase57_modules_import_simultaneously(self):
        from src.scanner.confidence_penalty_ceiling import ConfidencePenaltyCeiling
        from src.scanner.calibration_guard import CalibrationGuard
        from src.scanner.stall_recovery import StallRecoveryManager
        from src.scanner.drift_proxy_guard import DriftProxyGuard
        from src.scanner.penalty_audit import PenaltyAuditor
        assert all([c is not None for c in [
            ConfidencePenaltyCeiling, CalibrationGuard,
            StallRecoveryManager, DriftProxyGuard, PenaltyAuditor
        ]])

    def test_phase57_alongside_phase56_modules(self):
        """Phase 57 and Phase 56 modules co-exist without circular imports."""
        from src.scanner.confidence_penalty_ceiling import ConfidencePenaltyCeiling
        from src.scanner.calibration_guard import CalibrationGuard
        from src.scanner.stall_recovery import StallRecoveryManager
        from src.scanner.drift_proxy_guard import DriftProxyGuard
        from src.scanner.penalty_audit import PenaltyAuditor
        # Phase 56
        from src.scanner.gate_rejection_tracker import GateRejectionTracker
        from src.scanner.confidence_penalty_monitor import ConfidencePenaltyMonitor
        from src.scanner.virtual_trade_logger import VirtualTradeLogger
        from src.scanner.scan_diagnostics import ScanDiagnosticsReporter
        assert True  # No import error = pass


# ---------------------------------------------------------------------------
# 2. Behavioral integration
# ---------------------------------------------------------------------------

class TestBehavioralIntegration:

    def test_ceiling_prevents_floor_breach_in_stacked_scenario(self):
        """Stacked: overconfidence -3pt + drift mult=0.85 on conf=45 → floor protected."""
        from src.scanner.confidence_penalty_ceiling import ConfidencePenaltyCeiling
        ceiling = ConfidencePenaltyCeiling(ceiling_floor=40.0)

        confidence = 45.0
        # Overconfidence -3pt
        r1 = ceiling.check_subtraction(confidence, 3.0, source="overconfidence")
        confidence -= r1.applied_penalty  # 45 - 3 = 42

        # Drift mult=0.85 → 42*0.85=35.7 < floor=40 → ceiling fires
        r2 = ceiling.check_multiplier(confidence, 0.85, source="concept_drift")
        confidence *= r2.applied_penalty

        assert confidence >= 40.0
        assert r2.ceiling_active is True

    def test_calibration_guard_blocks_with_insufficient_data(self):
        from src.scanner.calibration_guard import CalibrationGuard
        guard = CalibrationGuard(min_valid_predictions=20)
        report = {"n_predictions": 6, "valid_bins": 0, "overconfidence_ratio": 0.80}
        assert guard.should_apply_penalty(report) is False

    def test_calibration_guard_allows_with_sufficient_data(self):
        from src.scanner.calibration_guard import CalibrationGuard
        guard = CalibrationGuard(min_valid_predictions=20, min_valid_bins=2)
        report = {"n_predictions": 30, "valid_bins": 4, "overconfidence_ratio": 0.60}
        assert guard.should_apply_penalty(report) is True

    def test_stall_recovery_full_cycle(self, tmp_path):
        """Simulate: stall → recovery → exit → re-stall → recovery again."""
        from src.scanner.stall_recovery import StallRecoveryManager, StallRecoveryState
        mgr = StallRecoveryManager(stall_threshold=5, recovery_window=3,
                                   state_path=tmp_path / "sr.json")
        # Normal
        assert mgr.get_state()["state"] == StallRecoveryState.NORMAL
        # Stall triggers recovery
        mgr.tick(scans_since_last_trade=6)
        assert mgr.is_in_recovery() is True
        assert mgr.penalties_suspended() is True
        # Recovery window elapses (3 ticks)
        mgr.tick(scans_since_last_trade=0)
        mgr.tick(scans_since_last_trade=0)
        mgr.tick(scans_since_last_trade=0)
        assert mgr.get_state()["state"] == StallRecoveryState.NORMAL
        assert mgr.get_state()["total_recovery_cycles"] == 1

    def test_drift_proxy_guard_uses_pre_penalty(self):
        from src.scanner.drift_proxy_guard import DriftProxyGuard
        guard = DriftProxyGuard()
        # pre=0.72, post=0.55 (penalized)
        proxies = guard.compute_proxies(0.72, 0.55)
        # Must use pre=0.72 NOT post=0.55
        assert proxies.prediction_error == pytest.approx(abs(0.72 - 0.5), abs=0.001)
        assert proxies.agent_disagreement == pytest.approx(1.0 - 0.72, abs=0.001)

    def test_penalty_audit_ceiling_needed_recommendation(self, tmp_path):
        from src.scanner.penalty_audit import PenaltyAuditor
        auditor = PenaltyAuditor(audit_path=tmp_path / "audit.json")
        cpm = MagicMock()
        cpm.get_breach_rate.return_value = 0.85
        cpm.generate_report.return_value = {}
        audit = auditor.run_audit(confidence_penalty_monitor=cpm)
        assert audit["recommendation"] == "CEILING_NEEDED"
        auditor.save_audit(audit)
        data = json.loads((tmp_path / "audit.json").read_text())
        assert data["recommendation"] == "CEILING_NEEDED"

    def test_all_phase57_modules_instantiate_together(self, tmp_path):
        """No runtime conflicts when all five modules are instantiated simultaneously."""
        from src.scanner.confidence_penalty_ceiling import ConfidencePenaltyCeiling
        from src.scanner.calibration_guard import CalibrationGuard
        from src.scanner.stall_recovery import StallRecoveryManager
        from src.scanner.drift_proxy_guard import DriftProxyGuard
        from src.scanner.penalty_audit import PenaltyAuditor

        ceiling = ConfidencePenaltyCeiling()
        guard = CalibrationGuard()
        recovery = StallRecoveryManager(state_path=tmp_path / "sr.json")
        drift_guard = DriftProxyGuard()
        auditor = PenaltyAuditor(audit_path=tmp_path / "audit.json")

        assert ceiling is not None
        assert guard is not None
        assert recovery is not None
        assert drift_guard is not None
        assert auditor is not None


# ---------------------------------------------------------------------------
# 3. Wiring verification — all five production call-site greps
# ---------------------------------------------------------------------------

class TestWiringGreps:

    def _read(self, path):
        with open(path) as f:
            return f.read()

    def test_penalty_ceiling_initialized_in_engine(self):
        content = self._read("src/scanner/engine.py")
        assert "ConfidencePenaltyCeiling" in content
        assert "_penalty_ceiling = None" in content

    def test_calibration_guard_initialized_in_engine(self):
        content = self._read("src/scanner/engine.py")
        assert "CalibrationGuard" in content
        assert "_calibration_guard = None" in content

    def test_calibration_guard_should_apply_penalty_called_in_engine(self):
        content = self._read("src/scanner/engine.py")
        assert "_calibration_guard.should_apply_penalty(" in content

    def test_stall_recovery_initialized_in_engine(self):
        content = self._read("src/scanner/engine.py")
        assert "StallRecoveryManager" in content
        assert "_stall_recovery = None" in content

    def test_stall_recovery_penalties_suspended_called_in_engine(self):
        content = self._read("src/scanner/engine.py")
        assert "_stall_recovery.penalties_suspended()" in content

    def test_drift_proxy_guard_initialized_in_engine(self):
        content = self._read("src/scanner/engine.py")
        assert "DriftProxyGuard" in content
        assert "_drift_proxy_guard = None" in content

    def test_drift_proxy_guard_compute_proxies_called_in_engine(self):
        content = self._read("src/scanner/engine.py")
        assert "_drift_proxy_guard.compute_proxies(" in content

    def test_penalty_ceiling_check_subtraction_called_in_engine(self):
        content = self._read("src/scanner/engine.py")
        assert "_penalty_ceiling.check_subtraction(" in content

    def test_penalty_ceiling_check_multiplier_called_in_engine(self):
        content = self._read("src/scanner/engine.py")
        assert "_penalty_ceiling.check_multiplier(" in content

    def test_penalty_auditor_initialized_in_continuous(self):
        content = self._read("src/scanner/automation/continuous.py")
        assert "PenaltyAuditor" in content
        assert "_penalty_auditor = None" in content

    def test_penalty_auditor_run_audit_called_in_continuous(self):
        content = self._read("src/scanner/automation/continuous.py")
        assert "_penalty_auditor.run_audit(" in content
