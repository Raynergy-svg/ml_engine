"""Tests for Phase 58 US-362: Integration — all Phase 58 modules co-exist.

Covers:
  - All four Phase 58 modules import without conflict
  - All four alongside Phase 57 and Phase 56 modules (no circular imports)
  - Full pipeline: signals below threshold → THRESHOLD_TOO_HIGH → no_signal → adaptive adapts
  - Full pipeline: signals above threshold → OK → signal_exists → adaptive does NOT adapt
  - Stall recovery full cycle: stall → tick → exit
  - Wiring grep: RawConfidenceRecorder initialized in engine.py
  - Wiring grep: _raw_conf_recorder.record( called in engine.py
  - Wiring grep: GatePassPredictor initialized in continuous.py
  - Wiring grep: _gate_pass_predictor.generate_report( called in continuous.py
  - Wiring grep: AdaptiveConfidenceFloor initialized in continuous.py
  - Wiring grep: _adaptive_floor.tick() called in continuous.py
  - Wiring grep: _adaptive_floor.should_adapt( called in continuous.py
"""

import json
from types import SimpleNamespace

import pytest


# ---------------------------------------------------------------------------
# 1. Module imports
# ---------------------------------------------------------------------------

class TestModuleImports:

    def test_raw_confidence_recorder_imports(self):
        from src.scanner.raw_confidence_recorder import RawConfidenceRecorder
        assert RawConfidenceRecorder is not None

    def test_threshold_gap_analyzer_imports(self):
        from src.scanner.threshold_gap_analyzer import ThresholdGapAnalyzer
        assert ThresholdGapAnalyzer is not None

    def test_gate_pass_predictor_imports(self):
        from src.scanner.gate_pass_predictor import GatePassPredictor
        assert GatePassPredictor is not None

    def test_adaptive_confidence_floor_imports(self):
        from src.scanner.adaptive_confidence_floor import AdaptiveConfidenceFloor
        assert AdaptiveConfidenceFloor is not None

    def test_all_phase58_modules_import_simultaneously(self):
        from src.scanner.raw_confidence_recorder import RawConfidenceRecorder
        from src.scanner.threshold_gap_analyzer import ThresholdGapAnalyzer
        from src.scanner.gate_pass_predictor import GatePassPredictor
        from src.scanner.adaptive_confidence_floor import AdaptiveConfidenceFloor
        assert all(m is not None for m in [
            RawConfidenceRecorder, ThresholdGapAnalyzer,
            GatePassPredictor, AdaptiveConfidenceFloor,
        ])

    def test_phase58_alongside_phase57_modules(self):
        # Phase 58
        from src.scanner.raw_confidence_recorder import RawConfidenceRecorder
        from src.scanner.threshold_gap_analyzer import ThresholdGapAnalyzer
        from src.scanner.gate_pass_predictor import GatePassPredictor
        from src.scanner.adaptive_confidence_floor import AdaptiveConfidenceFloor
        # Phase 57
        from src.scanner.confidence_penalty_ceiling import ConfidencePenaltyCeiling
        from src.scanner.calibration_guard import CalibrationGuard
        from src.scanner.stall_recovery import StallRecoveryManager
        from src.scanner.drift_proxy_guard import DriftProxyGuard
        from src.scanner.penalty_audit import PenaltyAuditor
        # Phase 56
        from src.scanner.gate_rejection_tracker import GateRejectionTracker
        from src.scanner.confidence_penalty_monitor import ConfidencePenaltyMonitor
        from src.scanner.scan_diagnostics import ScanDiagnosticsReporter
        assert True  # No import error = pass


# ---------------------------------------------------------------------------
# 2. Behavioral integration
# ---------------------------------------------------------------------------

class TestBehavioralIntegration:

    def test_full_pipeline_signals_below_threshold_adaptive_adapts(self, tmp_path):
        """50 signals at 50.0, threshold=58.0, stall=55 → adaptive floor adapts."""
        from src.scanner.raw_confidence_recorder import RawConfidenceRecorder
        from src.scanner.threshold_gap_analyzer import ThresholdGapAnalyzer
        from src.scanner.gate_pass_predictor import GatePassPredictor
        from src.scanner.adaptive_confidence_floor import AdaptiveConfidenceFloor

        recorder = RawConfidenceRecorder(max_size=200, state_path=tmp_path / "rc.jsonl")
        for _ in range(50):
            recorder.record("EUR_USD", 50.0, scan_id="s1")

        analyzer = ThresholdGapAnalyzer()
        predictor = GatePassPredictor()
        floor = AdaptiveConfidenceFloor(
            initial_threshold=58.0,
            stall_threshold=50,
            state_path=tmp_path / "af.json",
        )

        gap_report = analyzer.analyze(recorder, 58.0)
        pred_report = predictor.generate_report(recorder, 58.0)

        assert gap_report.recommendation == "THRESHOLD_TOO_HIGH"
        assert pred_report["signal_exists"] is False
        assert floor.should_adapt(gap_report, pred_report, scans_since_last_trade=55) is True

        delta = floor.compute_adjustment(gap_report, pred_report)
        config = SimpleNamespace(min_confidence=58.0)
        floor.apply_adjustment(config, delta)
        assert config.min_confidence == pytest.approx(57.0, abs=0.01)

    def test_full_pipeline_signals_above_threshold_no_adapt(self, tmp_path):
        """50 signals at 65.0, threshold=58.0 → OK → adaptive does NOT adapt."""
        from src.scanner.raw_confidence_recorder import RawConfidenceRecorder
        from src.scanner.threshold_gap_analyzer import ThresholdGapAnalyzer
        from src.scanner.gate_pass_predictor import GatePassPredictor
        from src.scanner.adaptive_confidence_floor import AdaptiveConfidenceFloor

        recorder = RawConfidenceRecorder(max_size=200, state_path=tmp_path / "rc.jsonl")
        for _ in range(50):
            recorder.record("EUR_USD", 65.0, scan_id="s1")

        analyzer = ThresholdGapAnalyzer()
        predictor = GatePassPredictor()
        floor = AdaptiveConfidenceFloor(
            initial_threshold=58.0,
            stall_threshold=50,
            state_path=tmp_path / "af.json",
        )

        gap_report = analyzer.analyze(recorder, 58.0)
        pred_report = predictor.generate_report(recorder, 58.0)

        assert gap_report.recommendation == "OK"
        assert floor.should_adapt(gap_report, pred_report, scans_since_last_trade=60) is False

    def test_signal_exists_prevents_adaptive_even_when_stalled(self, tmp_path):
        """If penalty-free pass rate >= 5%, Phase 57 should fix it — don't lower threshold."""
        from src.scanner.raw_confidence_recorder import RawConfidenceRecorder
        from src.scanner.threshold_gap_analyzer import ThresholdGapAnalyzer
        from src.scanner.gate_pass_predictor import GatePassPredictor
        from src.scanner.adaptive_confidence_floor import AdaptiveConfidenceFloor

        recorder = RawConfidenceRecorder(max_size=200, state_path=tmp_path / "rc.jsonl")
        # 3 signals above threshold out of 25 → 12% → signal_exists=True
        for _ in range(3):
            recorder.record("EUR_USD", 62.0, scan_id="s1")
        for _ in range(22):
            recorder.record("EUR_USD", 50.0, scan_id="s1")

        analyzer = ThresholdGapAnalyzer()
        predictor = GatePassPredictor()
        floor = AdaptiveConfidenceFloor(initial_threshold=58.0, state_path=tmp_path / "af.json")

        gap_report = analyzer.analyze(recorder, 58.0)
        pred_report = predictor.generate_report(recorder, 58.0)

        assert pred_report["signal_exists"] is True
        assert floor.should_adapt(gap_report, pred_report, scans_since_last_trade=55) is False

    def test_adaptive_floor_cooldown_prevents_rapid_adaptation(self, tmp_path):
        from src.scanner.adaptive_confidence_floor import AdaptiveConfidenceFloor
        from src.scanner.threshold_gap_analyzer import ThresholdGapReport

        floor = AdaptiveConfidenceFloor(
            cooldown_scans=5, initial_threshold=58.0,
            state_path=tmp_path / "af.json",
        )
        config = SimpleNamespace(min_confidence=58.0)
        floor.apply_adjustment(config, delta=-1.0)
        assert floor._cooldown_remaining == 5

        # Immediately try again — cooldown blocks it
        gap_report = ThresholdGapReport(
            count=50, mean_raw=50.0, median_raw=50.0, p75_raw=50.0,
            current_threshold=57.0, gap_at_median=7.0, gap_at_p75=7.0,
            signals_above_threshold_pct=0.0, recommendation="THRESHOLD_TOO_HIGH",
        )
        pred_report = {"signal_exists": False}
        assert floor.should_adapt(gap_report, pred_report, scans_since_last_trade=55) is False

        # After cooldown elapses
        for _ in range(5):
            floor.tick()
        assert floor.should_adapt(gap_report, pred_report, scans_since_last_trade=55) is True

    def test_all_phase58_modules_instantiate_together(self, tmp_path):
        from src.scanner.raw_confidence_recorder import RawConfidenceRecorder
        from src.scanner.threshold_gap_analyzer import ThresholdGapAnalyzer
        from src.scanner.gate_pass_predictor import GatePassPredictor
        from src.scanner.adaptive_confidence_floor import AdaptiveConfidenceFloor

        recorder = RawConfidenceRecorder(state_path=tmp_path / "rc.jsonl")
        analyzer = ThresholdGapAnalyzer()
        predictor = GatePassPredictor()
        floor = AdaptiveConfidenceFloor(state_path=tmp_path / "af.json")

        assert recorder is not None
        assert analyzer is not None
        assert predictor is not None
        assert floor is not None

    def test_persistence_across_all_modules(self, tmp_path):
        """RawConfidenceRecorder and AdaptiveConfidenceFloor survive save/load."""
        from src.scanner.raw_confidence_recorder import RawConfidenceRecorder
        from src.scanner.adaptive_confidence_floor import AdaptiveConfidenceFloor

        # Recorder save/load
        p1 = tmp_path / "rc.jsonl"
        r1 = RawConfidenceRecorder(state_path=p1)
        for _ in range(5):
            r1.record("EUR_USD", 55.0, "s1")
        r1.save_state()
        r2 = RawConfidenceRecorder(state_path=p1)
        assert r2.count() == 5

        # Adaptive floor save/load
        p2 = tmp_path / "af.json"
        f1 = AdaptiveConfidenceFloor(initial_threshold=58.0, state_path=p2)
        config = SimpleNamespace(min_confidence=58.0)
        f1.apply_adjustment(config, delta=-1.0)
        f2 = AdaptiveConfidenceFloor(initial_threshold=58.0, state_path=p2)
        assert f2._total_adaptations == 1
        assert f2._current_threshold == pytest.approx(57.0, abs=0.01)


# ---------------------------------------------------------------------------
# 3. Wiring greps
# ---------------------------------------------------------------------------

class TestWiringGreps:

    def _read(self, path):
        with open(path) as f:
            return f.read()

    def test_raw_conf_recorder_initialized_in_engine(self):
        content = self._read("src/scanner/engine.py")
        assert "_raw_conf_recorder = None" in content
        assert "RawConfidenceRecorder" in content

    def test_raw_conf_recorder_record_called_in_engine(self):
        content = self._read("src/scanner/engine.py")
        assert "_raw_conf_recorder.record(" in content

    def test_gate_pass_predictor_initialized_in_continuous(self):
        content = self._read("src/scanner/automation/continuous.py")
        assert "_gate_pass_predictor = None" in content
        assert "GatePassPredictor" in content

    def test_gate_pass_predictor_generate_report_called_in_continuous(self):
        content = self._read("src/scanner/automation/continuous.py")
        assert "_gate_pass_predictor.generate_report(" in content

    def test_adaptive_floor_initialized_in_continuous(self):
        content = self._read("src/scanner/automation/continuous.py")
        assert "_adaptive_floor = None" in content
        assert "AdaptiveConfidenceFloor" in content

    def test_adaptive_floor_tick_called_in_continuous(self):
        content = self._read("src/scanner/automation/continuous.py")
        assert "_adaptive_floor.tick()" in content

    def test_adaptive_floor_should_adapt_called_in_continuous(self):
        content = self._read("src/scanner/automation/continuous.py")
        assert "_adaptive_floor.should_adapt(" in content
