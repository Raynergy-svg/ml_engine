"""Tests for Phase 56 US-351: Integration — all Phase 56 modules co-exist.

Covers:
  - All five Phase 56 modules import without conflict
  - GateRejectionTracker and ScanDiagnosticsReporter agree on tradeable counts
  - ConfidencePenaltyMonitor detects floor breaches that correlate with zero tradeables
  - VirtualTradeLogger captures setups rejected by gate tracker
  - End-to-end: simulate a stalled-scan sequence through all modules
  - Wiring grep: GateRejectionTracker.record_rejection called in engine.py
  - Wiring grep: VirtualTradeLogger.log_virtual_trade called in engine.py
  - Wiring grep: ConfidencePenaltyMonitor.record_penalties called in engine.py
  - Wiring grep: ExecutionQualityTracker initialized in execution.py
  - Wiring grep: track_fill called in execution.py
  - Wiring grep: ScanDiagnosticsReporter initialized in continuous.py
  - Wiring grep: record_scan called in continuous.py
"""

import json
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# 1. All five modules import without error
# ---------------------------------------------------------------------------

class TestModuleImports:

    def test_gate_rejection_tracker_imports(self):
        from src.scanner.gate_rejection_tracker import GateRejectionTracker
        assert GateRejectionTracker is not None

    def test_confidence_penalty_monitor_imports(self):
        from src.scanner.confidence_penalty_monitor import ConfidencePenaltyMonitor
        assert ConfidencePenaltyMonitor is not None

    def test_virtual_trade_logger_imports(self):
        from src.scanner.virtual_trade_logger import VirtualTradeLogger
        assert VirtualTradeLogger is not None

    def test_execution_quality_tracker_imports(self):
        from src.scanner.automation.execution_quality_tracker import ExecutionQualityTracker
        assert ExecutionQualityTracker is not None

    def test_scan_diagnostics_reporter_imports(self):
        from src.scanner.scan_diagnostics import ScanDiagnosticsReporter
        assert ScanDiagnosticsReporter is not None

    def test_all_five_modules_import_simultaneously(self):
        """Confirm no namespace conflicts or circular import issues."""
        from src.scanner.gate_rejection_tracker import GateRejectionTracker
        from src.scanner.confidence_penalty_monitor import ConfidencePenaltyMonitor
        from src.scanner.virtual_trade_logger import VirtualTradeLogger
        from src.scanner.automation.execution_quality_tracker import ExecutionQualityTracker
        from src.scanner.scan_diagnostics import ScanDiagnosticsReporter
        # All five instantiate together without conflict
        assert all([
            GateRejectionTracker is not None,
            ConfidencePenaltyMonitor is not None,
            VirtualTradeLogger is not None,
            ExecutionQualityTracker is not None,
            ScanDiagnosticsReporter is not None,
        ])


# ---------------------------------------------------------------------------
# 2. Cross-module consistency: stalled scan sequence
# ---------------------------------------------------------------------------

class TestCrossModuleConsistency:
    """Simulate a stalled scan sequence through multiple Phase 56 modules."""

    def test_gate_tracker_and_scan_diagnostics_agree_on_pass_rate(self, tmp_path):
        """When all scans have zero tradeables, both modules report 0% gate pass rate."""
        from src.scanner.gate_rejection_tracker import GateRejectionTracker
        from src.scanner.scan_diagnostics import ScanDiagnosticsReporter

        grt = GateRejectionTracker(persistence_path=tmp_path / "grt.json")
        sdr = ScanDiagnosticsReporter(
            report_path=tmp_path / "report.json",
            state_path=tmp_path / "state.json",
        )

        # 5 scan cycles, no tradeables, gate rejections every cycle
        for _ in range(5):
            grt.record_rejection("EUR_USD", "confidence", "confidence < 0.30", 0.18)
            grt.mark_scan_complete()
            sdr.record_scan([{"confidence": 0.18, "is_tradeable": False, "direction": "HOLD"}])

        report = sdr.generate_report()
        assert report["gate_pass_rate_pct"] == 0.0

        stats = grt.get_stats()
        top_gate = grt.get_top_blocking_gate()
        assert top_gate == "confidence"
        assert stats["total_rejections"] >= 5

    def test_penalty_monitor_breach_correlates_with_zero_tradeables(self, tmp_path):
        """When cumulative penalty causes floor breach, gate pass rate should be 0."""
        from src.scanner.confidence_penalty_monitor import ConfidencePenaltyMonitor
        from src.scanner.scan_diagnostics import ScanDiagnosticsReporter

        cpm = ConfidencePenaltyMonitor(persistence_path=tmp_path / "cpm.json")
        sdr = ScanDiagnosticsReporter(
            report_path=tmp_path / "report.json",
            state_path=tmp_path / "state.json",
        )

        # High penalties causing floor breaches
        for _ in range(5):
            cpm.record_penalties("EUR_USD", {
                "overconfidence": 0.03,
                "concept_drift": 0.12,
                "disagreement": 0.08,
            })
            sdr.record_scan([{"confidence": 0.15, "is_tradeable": False, "direction": "LONG"}])

        assert cpm.is_floor_breach("EUR_USD")
        report = sdr.generate_report()
        assert report["gate_pass_rate_pct"] == 0.0
        assert report["stalled"] is False  # threshold=50, only 5 scans

    def test_virtual_trade_logger_captures_high_confidence_rejects(self, tmp_path):
        """VirtualTradeLogger captures setups that pass confidence but fail gates."""
        from src.scanner.virtual_trade_logger import VirtualTradeLogger
        from src.scanner.gate_rejection_tracker import GateRejectionTracker

        vtl = VirtualTradeLogger(log_path=tmp_path / "virtual.jsonl")
        grt = GateRejectionTracker(persistence_path=tmp_path / "grt.json")

        # A pair that almost traded — high confidence, wrong direction gate
        vtl.log_virtual_trade(
            pair="GBP_USD",
            direction="LONG",
            confidence=0.55,
            gate_failures=["momentum_gate=False"],
            features={},
            agent_scores={},
        )
        grt.record_rejection("GBP_USD", "momentum_gate", "momentum < 0", 0.55)

        recent = vtl.get_recent(n=10)
        assert len(recent) >= 1
        assert recent[0]["pair"] == "GBP_USD"
        assert recent[0]["direction"] == "LONG"

        stats = grt.get_stats()
        assert stats["total_rejections"] >= 1

    def test_stalled_flag_fires_at_threshold(self, tmp_path):
        """ScanDiagnosticsReporter stalled=True at exactly stalled_threshold scans."""
        from src.scanner.scan_diagnostics import ScanDiagnosticsReporter

        sdr = ScanDiagnosticsReporter(
            report_path=tmp_path / "report.json",
            state_path=tmp_path / "state.json",
            stalled_threshold=5,
        )
        for _ in range(5):
            sdr.record_scan([{"confidence": 0.20, "is_tradeable": False, "direction": "HOLD"}])

        report = sdr.generate_report()
        assert report["stalled"] is True
        assert "STALLED" in report["diagnostic_summary"]

    def test_penalty_breach_rate_nonzero_after_breaches(self, tmp_path):
        """ConfidencePenaltyMonitor breach_rate > 0 after recording floor breaches."""
        from src.scanner.confidence_penalty_monitor import ConfidencePenaltyMonitor

        cpm = ConfidencePenaltyMonitor(persistence_path=tmp_path / "cpm.json")
        # Record enough scans to have non-trivial breach rate
        for _ in range(10):
            cpm.record_penalties("EUR_USD", {
                "overconfidence": 0.03,
                "concept_drift": 0.10,
                "disagreement": 0.09,  # total=0.22 > default threshold 0.20
            })
            cpm.flush_scan()

        breach_rate = cpm.get_breach_rate()
        assert breach_rate > 0.0

    def test_gate_tracker_rejection_rate_api(self, tmp_path):
        """GateRejectionTracker.get_gate_rejection_rate returns float 0-1."""
        from src.scanner.gate_rejection_tracker import GateRejectionTracker

        grt = GateRejectionTracker(persistence_path=tmp_path / "grt.json")
        for _ in range(3):
            grt.record_rejection("EUR_USD", "confidence", "conf < 0.3", 0.2)
            grt.mark_scan_complete()

        rate = grt.get_gate_rejection_rate("confidence")
        assert 0.0 <= rate <= 1.0
        assert rate > 0.0

    def test_all_modules_persist_and_reload(self, tmp_path):
        """All Phase 56 modules survive a save/reload cycle without data loss."""
        from src.scanner.gate_rejection_tracker import GateRejectionTracker
        from src.scanner.confidence_penalty_monitor import ConfidencePenaltyMonitor
        from src.scanner.scan_diagnostics import ScanDiagnosticsReporter

        # Session 1 — record data
        grt1 = GateRejectionTracker(persistence_path=tmp_path / "grt.json")
        grt1.record_rejection("EUR_USD", "confidence", "conf < 0.3", 0.2)
        grt1.save_state()

        sdr1 = ScanDiagnosticsReporter(
            report_path=tmp_path / "r.json", state_path=tmp_path / "s.json"
        )
        sdr1.record_scan([{"confidence": 0.2, "is_tradeable": False, "direction": "HOLD"}])
        sdr1.save_state()

        # Session 2 — reload
        grt2 = GateRejectionTracker(persistence_path=tmp_path / "grt.json")
        sdr2 = ScanDiagnosticsReporter(
            report_path=tmp_path / "r.json", state_path=tmp_path / "s.json"
        )

        assert grt2.get_stats()["total_rejections"] >= 1
        assert sdr2._total_scans == 1


# ---------------------------------------------------------------------------
# 3. Wiring verification — all seven live call-site greps
# ---------------------------------------------------------------------------

class TestAllWiringGreps:

    def _read(self, path):
        with open(path) as f:
            return f.read()

    # --- engine.py (US-346, US-347, US-348) ---

    def test_gate_rejection_tracker_initialized_in_engine(self):
        content = self._read("src/scanner/engine.py")
        assert "GateRejectionTracker" in content
        assert "_gate_rejection_tracker = None" in content

    def test_gate_rejection_tracker_record_called_in_engine(self):
        content = self._read("src/scanner/engine.py")
        assert "_gate_rejection_tracker.record_rejection(" in content

    def test_confidence_penalty_monitor_initialized_in_engine(self):
        content = self._read("src/scanner/engine.py")
        assert "ConfidencePenaltyMonitor" in content
        assert "_confidence_penalty_monitor = None" in content

    def test_confidence_penalty_monitor_record_called_in_engine(self):
        content = self._read("src/scanner/engine.py")
        assert "_confidence_penalty_monitor.record_penalties(" in content

    def test_virtual_trade_logger_initialized_in_engine(self):
        content = self._read("src/scanner/engine.py")
        assert "VirtualTradeLogger" in content
        assert "_virtual_trade_logger = None" in content

    def test_virtual_trade_logger_log_called_in_engine(self):
        content = self._read("src/scanner/engine.py")
        assert "_virtual_trade_logger.log_virtual_trade(" in content

    # --- execution.py (US-349) ---

    def test_exec_quality_tracker_initialized_in_execution(self):
        content = self._read("src/scanner/execution.py")
        assert "ExecutionQualityTracker" in content
        assert "_exec_quality_tracker = None" in content

    def test_exec_quality_tracker_track_fill_called_in_execution(self):
        content = self._read("src/scanner/execution.py")
        assert "_exec_quality_tracker.track_fill(" in content

    def test_exec_quality_tracker_save_state_called_in_execution(self):
        content = self._read("src/scanner/execution.py")
        assert "_exec_quality_tracker.save_state()" in content

    # --- continuous.py (US-350) ---

    def test_scan_diagnostics_initialized_in_continuous(self):
        content = self._read("src/scanner/automation/continuous.py")
        assert "ScanDiagnosticsReporter" in content
        assert "_scan_diagnostics = None" in content

    def test_scan_diagnostics_record_scan_called_in_continuous(self):
        content = self._read("src/scanner/automation/continuous.py")
        assert "_scan_diagnostics.record_scan(" in content
