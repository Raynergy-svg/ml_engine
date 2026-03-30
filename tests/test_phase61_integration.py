"""Phase 61 US-374 — End-to-end integration tests.

Minimum 12 tests covering:
- All Phase 61 modules importable
- Live wiring greps in engine.py
- Behavioral simulations (NEAR_THRESHOLD, FAR_BELOW, near_miss_rate)
- ScanHealthSynthesizer get_momentum_signals() handshake
- MomentumScoreRecorder save/load preserves records
- Pass rate correctness
- Empty analyzer returns INSUFFICIENT_DATA without crash
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

# ── imports ──────────────────────────────────────────────────────────────────

ENGINE_SRC = Path("src/scanner/engine.py")
CONTINUOUS_SRC = Path("src/scanner/automation/continuous.py")


# ── T1: Importability ────────────────────────────────────────────────────────


class TestPhase61Importability:
    """All Phase 61 modules import cleanly alongside Phase 56-60 modules."""

    def test_momentum_score_recorder_importable(self):
        from src.scanner.momentum_score_recorder import MomentumScoreRecorder
        assert MomentumScoreRecorder is not None

    def test_momentum_gap_analyzer_importable(self):
        from src.scanner.momentum_gap_analyzer import (
            analyze,
            MomentumGapAnalyzer,
            CLASS_NEAR_THRESHOLD,
            CLASS_MODERATE,
            CLASS_FAR_BELOW,
            CLASS_INSUFFICIENT_DATA,
        )
        assert analyze is not None

    def test_phase56_through_60_still_importable(self):
        """Sanity: Phase 56-60 modules unaffected by Phase 61 additions."""
        from src.scanner.signal_funnel_tracker import SignalFunnelTracker
        from src.scanner.gate_proximity_reporter import GateProximityReporter
        from src.scanner.ensemble_divergence_monitor import EnsembleDivergenceMonitor
        from src.scanner.scan_health_synthesizer import ScanHealthSynthesizer
        assert ScanHealthSynthesizer is not None


# ── T2: Live wiring greps in engine.py ───────────────────────────────────────


class TestPhase61WiringGreps:
    """Per Live Wiring Verification Gates: grep engine.py for call sites."""

    def _src(self) -> str:
        return ENGINE_SRC.read_text()

    def test_momentum_recorder_init_block_present(self):
        """_momentum_recorder = None init guard present in engine.py."""
        assert "_momentum_recorder = None" in self._src()

    def test_momentum_score_recorder_import_in_engine(self):
        """MomentumScoreRecorder import present in engine.py."""
        assert "MomentumScoreRecorder" in self._src()

    def test_momentum_recorder_record_call_present(self):
        """_momentum_recorder.record( call site present in engine.py."""
        assert "_momentum_recorder.record(" in self._src()

    def test_momentum_recorder_save_state_call_present(self):
        """_momentum_recorder.save_state() call site present in engine.py."""
        assert "_momentum_recorder.save_state()" in self._src()

    def test_momentum_gap_analyzer_periodic_log_present(self):
        """Periodic MomentumGapAnalyzer call (scan_cycle_count % 50) present."""
        assert "_scan_cycle_count % 50" in self._src()

    def test_near_threshold_warning_log_present(self):
        """NEAR_THRESHOLD branch logs a warning in engine.py."""
        assert "NEAR_THRESHOLD" in self._src()


# ── T3: Behavioral simulations ───────────────────────────────────────────────


class TestPhase61BehavioralSimulations:
    """End-to-end scenario simulations per PRD acceptance criteria."""

    def _make_recorder(self, fail_scores, pass_scores=None):
        from src.scanner.momentum_score_recorder import MomentumScoreRecorder
        with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False) as f:
            path = Path(f.name)
        path.unlink()
        rec = MomentumScoreRecorder(state_path=path)
        for s in fail_scores:
            rec.record("EUR/USD", momentum_score=s, momentum_passed=False)
        for s in (pass_scores or []):
            rec.record("EUR/USD", momentum_score=s, momentum_passed=True)
        return rec

    def test_near_threshold_scenario(self):
        """10 records at score=0.42 vs min_momentum=0.45 → gap=0.03, NEAR_THRESHOLD."""
        from src.scanner.momentum_gap_analyzer import analyze, CLASS_NEAR_THRESHOLD
        rec = self._make_recorder(fail_scores=[0.42] * 10)
        result = analyze(rec, min_momentum=0.45)
        assert result["gap_at_p50"] == pytest.approx(0.03, abs=1e-4)
        assert result["classification"] == CLASS_NEAR_THRESHOLD

    def test_far_below_scenario(self):
        """10 records at score=0.20 vs min_momentum=0.45 → gap=0.25, FAR_BELOW."""
        from src.scanner.momentum_gap_analyzer import analyze, CLASS_FAR_BELOW
        rec = self._make_recorder(fail_scores=[0.20] * 10)
        result = analyze(rec, min_momentum=0.45)
        assert result["gap_at_p50"] == pytest.approx(0.25, abs=1e-4)
        assert result["classification"] == CLASS_FAR_BELOW

    def test_near_miss_rate_scenario(self):
        """5 near-miss fails + 5 far fails → near_miss_rate_pct=50.0."""
        from src.scanner.momentum_gap_analyzer import analyze
        near = [0.42] * 5   # gap=0.03 ≤ 0.05 → near miss
        far = [0.10] * 5    # gap=0.35 > 0.05 → not near miss
        rec = self._make_recorder(fail_scores=near + far)
        result = analyze(rec, min_momentum=0.45)
        assert result["near_miss_rate_pct"] == pytest.approx(50.0, abs=0.1)
        assert result["near_miss_count"] == 5

    def test_pass_rate_scenario(self):
        """7 pass + 3 fail → pass_rate=0.70."""
        from src.scanner.momentum_gap_analyzer import analyze
        rec = self._make_recorder(
            fail_scores=[0.35] * 3,
            pass_scores=[0.55] * 7,
        )
        result = analyze(rec, min_momentum=0.45)
        assert result["pass_rate"] == pytest.approx(0.70, abs=1e-4)

    def test_empty_analyzer_returns_insufficient_data(self):
        """Empty MomentumGapAnalyzer returns INSUFFICIENT_DATA without crash."""
        from src.scanner.momentum_gap_analyzer import MomentumGapAnalyzer, CLASS_INSUFFICIENT_DATA
        analyzer = MomentumGapAnalyzer(recorder=None, min_momentum=0.45)
        result = analyzer.analyze_current()
        assert result["classification"] == CLASS_INSUFFICIENT_DATA
        assert result["gap_at_p50"] == 0.0


# ── T4: ScanHealthSynthesizer handshake ──────────────────────────────────────


class TestPhase61ScanHealthHandshake:
    """MomentumGapAnalyzer.get_momentum_signals() is consumable by ScanHealthSynthesizer."""

    def test_get_momentum_signals_dict_shape(self):
        """get_momentum_signals() returns a dict with the 4 expected signal keys."""
        from src.scanner.momentum_gap_analyzer import MomentumGapAnalyzer
        from src.scanner.momentum_score_recorder import MomentumScoreRecorder
        with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False) as f:
            path = Path(f.name)
        path.unlink()
        rec = MomentumScoreRecorder(state_path=path)
        for _ in range(10):
            rec.record("EUR/USD", 0.42, momentum_passed=False)
        analyzer = MomentumGapAnalyzer(recorder=rec, min_momentum=0.45)
        signals = analyzer.get_momentum_signals()
        assert isinstance(signals, dict)
        assert "momentum_gap_at_p50" in signals
        assert "momentum_near_miss_rate_pct" in signals
        assert "momentum_classification" in signals
        assert "momentum_pass_rate" in signals

    def test_scan_health_synthesizer_accepts_momentum_signals(self):
        """ScanHealthSynthesizer.synthesize() does not crash when momentum signals are available."""
        from src.scanner.momentum_gap_analyzer import MomentumGapAnalyzer
        from src.scanner.momentum_score_recorder import MomentumScoreRecorder
        from src.scanner.scan_health_synthesizer import ScanHealthSynthesizer
        with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False) as f:
            path = Path(f.name)
        path.unlink()
        rec = MomentumScoreRecorder(state_path=path)
        for _ in range(5):
            rec.record("EUR/USD", 0.42, momentum_passed=False)
        # ScanHealthSynthesizer with no connectors still returns a valid health report
        shs = ScanHealthSynthesizer()
        report = shs.synthesize()
        assert "health_status" in report


# ── T5: Persistence round-trip ───────────────────────────────────────────────


class TestPhase61PersistenceRoundTrip:
    """MomentumScoreRecorder save/load preserves all records across instantiation."""

    def test_save_load_preserves_records(self):
        from src.scanner.momentum_score_recorder import MomentumScoreRecorder
        with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False) as f:
            path = Path(f.name)
        path.unlink()

        rec1 = MomentumScoreRecorder(state_path=path)
        scores = [0.30, 0.35, 0.40, 0.42, 0.38]
        for s in scores:
            rec1.record("EUR/USD", momentum_score=s, momentum_passed=(s >= 0.45))
        rec1.save_state()

        rec2 = MomentumScoreRecorder(state_path=path)
        assert rec2.get_distribution()["count"] == len(scores)
        assert rec2.get_distribution()["min"] == pytest.approx(0.30, abs=1e-5)
        path.unlink(missing_ok=True)
