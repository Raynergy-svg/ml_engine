"""Integration tests for Phase 59 US-366: wiring verification.

Confirms that:
  1. SignalFunnelTracker is initialised in continuous.py and its record_scan()
     is called every scan cycle.
  2. GateProximityReporter is initialised in engine.py and its record_scan()
     is called after every pair scan.
  3. EnsembleDivergenceMonitor is initialised in engine.py and its record()
     is called where tcn_conf / ridge_conf are available.
  4. The three modules are importable and compose correctly end-to-end.
  5. All three modules persist and reload state without data loss.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

# ─── helpers ──────────────────────────────────────────────────────────────────

def _make_analysis(conf_passed, mom_passed, risk_passed, tradeable, confidence=55.0, pair="EUR_USD"):
    return SimpleNamespace(
        pair=pair,
        confidence=confidence,
        confidence_gate_passed=conf_passed,
        momentum_gate_passed=mom_passed,
        risk_gate_passed=risk_passed,
        is_tradeable=tradeable,
    )


# ─── US-366-1: Live wiring — call site verification (engine.py) ───────────────

class TestEnginePy_WiringVerification:

    def test_gate_proximity_init_block_exists_in_engine(self):
        """engine.py must initialise _gate_proximity with GateProximityReporter."""
        engine_src = Path("src/scanner/engine.py").read_text()
        assert "GateProximityReporter" in engine_src, (
            "_gate_proximity init block missing — GateProximityReporter never imported in engine.py"
        )
        assert "_gate_proximity" in engine_src

    def test_gate_proximity_record_scan_called_in_engine(self):
        """engine.py must call _gate_proximity.record_scan() at end of scan."""
        engine_src = Path("src/scanner/engine.py").read_text()
        assert "_gate_proximity.record_scan(" in engine_src, (
            "record_scan() call missing — GateProximityReporter write-only dead code"
        )

    def test_ensemble_divergence_init_block_exists_in_engine(self):
        """engine.py must initialise _ensemble_divergence with EnsembleDivergenceMonitor."""
        engine_src = Path("src/scanner/engine.py").read_text()
        assert "EnsembleDivergenceMonitor" in engine_src
        assert "_ensemble_divergence" in engine_src

    def test_ensemble_divergence_record_called_in_engine(self):
        """engine.py must call _ensemble_divergence.record() at tcn/ridge site."""
        engine_src = Path("src/scanner/engine.py").read_text()
        assert "_ensemble_divergence.record(" in engine_src, (
            "record() call missing — EnsembleDivergenceMonitor write-only dead code"
        )

    def test_ensemble_divergence_record_after_tcn_conf(self):
        """_ensemble_divergence.record() must appear after tcn_conf assignment."""
        engine_src = Path("src/scanner/engine.py").read_text()
        tcn_pos = engine_src.find("tcn_conf = float(abs(float(signal.tcn_probability)")
        record_pos = engine_src.find("_ensemble_divergence.record(")
        assert tcn_pos != -1, "tcn_conf assignment not found"
        assert record_pos != -1, "_ensemble_divergence.record() not found"
        assert record_pos > tcn_pos, (
            "_ensemble_divergence.record() must appear AFTER tcn_conf assignment"
        )


# ─── US-366-2: Live wiring — call site verification (continuous.py) ──────────

class TestContinuousPy_WiringVerification:

    def test_signal_funnel_init_block_exists_in_continuous(self):
        """continuous.py must initialise _signal_funnel with SignalFunnelTracker."""
        cont_src = Path("src/scanner/automation/continuous.py").read_text()
        assert "SignalFunnelTracker" in cont_src, (
            "_signal_funnel init block missing — SignalFunnelTracker never imported in continuous.py"
        )
        assert "_signal_funnel" in cont_src

    def test_signal_funnel_record_scan_called_in_continuous(self):
        """continuous.py must call _signal_funnel.record_scan() each cycle."""
        cont_src = Path("src/scanner/automation/continuous.py").read_text()
        assert "_signal_funnel.record_scan(" in cont_src, (
            "record_scan() call missing — SignalFunnelTracker write-only dead code"
        )

    def test_signal_funnel_save_state_called_in_continuous(self):
        """continuous.py must persist funnel state each cycle."""
        cont_src = Path("src/scanner/automation/continuous.py").read_text()
        assert "_signal_funnel.save_state()" in cont_src


# ─── US-366-3: Behavioural integration — SignalFunnelTracker ─────────────────

class TestSignalFunnelIntegration:

    def test_full_pass_funnel_integration(self, tmp_path):
        from src.scanner.signal_funnel_tracker import SignalFunnelTracker, KILL_STAGE_NONE
        tracker = SignalFunnelTracker(state_path=tmp_path / "sf.jsonl")
        analyses = [_make_analysis(True, True, True, True) for _ in range(8)]
        funnel = tracker.record_scan(analyses)
        assert funnel["total_evaluated"] == 8
        assert funnel["is_tradeable"] == 8
        assert funnel["tradeable_pct"] == pytest.approx(100.0, abs=0.1)
        assert funnel["dominant_kill_stage"] == KILL_STAGE_NONE

    def test_mixed_funnel_identifies_dominant_kill(self, tmp_path):
        from src.scanner.signal_funnel_tracker import SignalFunnelTracker, KILL_STAGE_CONFIDENCE
        tracker = SignalFunnelTracker(state_path=tmp_path / "sf.jsonl")
        # 9 fail at confidence, 1 passes everything
        analyses = (
            [_make_analysis(False, False, False, False)] * 9
            + [_make_analysis(True, True, True, True)] * 1
        )
        funnel = tracker.record_scan(analyses)
        assert funnel["dominant_kill_stage"] == KILL_STAGE_CONFIDENCE

    def test_funnel_aggregates_across_multiple_scans(self, tmp_path):
        from src.scanner.signal_funnel_tracker import SignalFunnelTracker
        tracker = SignalFunnelTracker(state_path=tmp_path / "sf.jsonl")
        for _ in range(3):
            tracker.record_scan([_make_analysis(True, True, True, True)] * 5)
        summary = tracker.get_funnel_summary()
        assert summary["scan_count"] == 3
        assert summary["total_evaluated"] == 15

    def test_funnel_persist_and_reload(self, tmp_path):
        from src.scanner.signal_funnel_tracker import SignalFunnelTracker
        p = tmp_path / "sf.jsonl"
        t1 = SignalFunnelTracker(state_path=p)
        t1.record_scan([_make_analysis(True, True, True, True)])
        t1.save_state()
        t2 = SignalFunnelTracker(state_path=p)
        assert t2.get_funnel_summary()["scan_count"] == 1


# ─── US-366-4: Behavioural integration — GateProximityReporter ───────────────

class TestGateProximityIntegration:

    def test_near_miss_identified(self, tmp_path):
        from src.scanner.gate_proximity_reporter import GateProximityReporter, MISS_CLASS_NEAR
        reporter = GateProximityReporter(state_path=tmp_path / "gp.jsonl")
        analyses = [_make_analysis(False, False, False, False, confidence=56.5)]
        reporter.record_scan(analyses, min_confidence=58.0)
        summary = reporter.get_proximity_summary()
        assert summary["near_miss_count"] == 1
        assert summary["closest_pair"] == "EUR_USD"

    def test_passed_pairs_not_counted_as_rejects(self, tmp_path):
        from src.scanner.gate_proximity_reporter import GateProximityReporter
        reporter = GateProximityReporter(state_path=tmp_path / "gp.jsonl")
        # Mix: 1 pass + 1 near-miss reject
        analyses = [
            _make_analysis(True, True, True, True, confidence=62.0, pair="EUR_USD"),
            _make_analysis(False, False, False, False, confidence=57.0, pair="GBP_USD"),
        ]
        reporter.record_scan(analyses, min_confidence=58.0)
        summary = reporter.get_proximity_summary()
        assert summary["total_conf_rejects"] == 1

    def test_proximity_persist_and_reload(self, tmp_path):
        from src.scanner.gate_proximity_reporter import GateProximityReporter
        p = tmp_path / "gp.jsonl"
        r1 = GateProximityReporter(state_path=p)
        r1.record_scan(
            [_make_analysis(False, False, False, False, confidence=57.0)],
            min_confidence=58.0,
        )
        r1.save_state()
        r2 = GateProximityReporter(state_path=p)
        assert r2.get_proximity_summary()["total_conf_rejects"] == 1

    def test_actionable_threshold_below_rejects(self, tmp_path):
        from src.scanner.gate_proximity_reporter import GateProximityReporter
        reporter = GateProximityReporter(state_path=tmp_path / "gp.jsonl")
        for _ in range(10):
            reporter.record_scan(
                [_make_analysis(False, False, False, False, confidence=56.0)],
                min_confidence=58.0,
            )
        t = reporter.get_actionable_threshold(target_near_miss_unblock=0.10)
        assert t <= 56.1  # must be at or below the reject confidence


# ─── US-366-5: Behavioural integration — EnsembleDivergenceMonitor ───────────

class TestEnsembleDivergenceIntegration:

    def test_divergence_recorded_correctly(self, tmp_path):
        from src.scanner.ensemble_divergence_monitor import EnsembleDivergenceMonitor
        mon = EnsembleDivergenceMonitor(state_path=tmp_path / "ed.jsonl")
        div = mon.record("EUR_USD", tcn_confidence=72.0, ridge_confidence=55.0)
        assert div == pytest.approx(17.0, abs=0.01)
        assert mon.get_mean_divergence() == pytest.approx(17.0, abs=0.01)

    def test_high_divergence_detection(self, tmp_path):
        from src.scanner.ensemble_divergence_monitor import (
            EnsembleDivergenceMonitor, REC_HIGH_DISAGREEMENT,
        )
        mon = EnsembleDivergenceMonitor(
            high_divergence_threshold=20.0,
            high_rate_threshold=0.30,
            state_path=tmp_path / "ed.jsonl",
        )
        for _ in range(10):
            mon.record("EUR_USD", tcn_confidence=80.0, ridge_confidence=50.0)
        summary = mon.get_divergence_summary()
        assert summary["recommendation"] == REC_HIGH_DISAGREEMENT
        assert mon.is_high_divergence() is True

    def test_divergence_ok_when_models_agree(self, tmp_path):
        from src.scanner.ensemble_divergence_monitor import EnsembleDivergenceMonitor, REC_OK
        mon = EnsembleDivergenceMonitor(state_path=tmp_path / "ed.jsonl")
        for _ in range(5):
            mon.record("EUR_USD", tcn_confidence=60.0, ridge_confidence=59.0)
        assert mon.get_divergence_summary()["recommendation"] == REC_OK

    def test_divergence_persist_and_reload(self, tmp_path):
        from src.scanner.ensemble_divergence_monitor import EnsembleDivergenceMonitor
        p = tmp_path / "ed.jsonl"
        m1 = EnsembleDivergenceMonitor(state_path=p)
        m1.record("EUR_USD", 75.0, 50.0)
        m1.save_state()
        m2 = EnsembleDivergenceMonitor(state_path=p)
        assert m2.get_mean_divergence() == pytest.approx(25.0, abs=0.01)
        assert m2.get_divergence_summary()["count"] == 1

    def test_per_pair_summary_filtered(self, tmp_path):
        from src.scanner.ensemble_divergence_monitor import EnsembleDivergenceMonitor
        mon = EnsembleDivergenceMonitor(state_path=tmp_path / "ed.jsonl")
        mon.record("EUR_USD", 75.0, 50.0)   # div=25
        mon.record("GBP_USD", 60.0, 59.0)   # div=1
        eur = mon.get_per_pair_summary("EUR_USD")
        gbp = mon.get_per_pair_summary("GBP_USD")
        assert eur["mean_divergence"] == pytest.approx(25.0, abs=0.01)
        assert gbp["mean_divergence"] == pytest.approx(1.0, abs=0.01)


# ─── US-366-6: Cross-module composition ──────────────────────────────────────

class TestCrossModuleComposition:

    def test_all_three_modules_importable_together(self):
        from src.scanner.signal_funnel_tracker import SignalFunnelTracker
        from src.scanner.gate_proximity_reporter import GateProximityReporter
        from src.scanner.ensemble_divergence_monitor import EnsembleDivergenceMonitor
        assert SignalFunnelTracker is not None
        assert GateProximityReporter is not None
        assert EnsembleDivergenceMonitor is not None

    def test_full_scan_pipeline_simulation(self, tmp_path):
        """Simulate a full scan: analyses flow through all three Phase 59 modules."""
        from src.scanner.signal_funnel_tracker import SignalFunnelTracker
        from src.scanner.gate_proximity_reporter import GateProximityReporter
        from src.scanner.ensemble_divergence_monitor import EnsembleDivergenceMonitor

        tracker = SignalFunnelTracker(state_path=tmp_path / "sf.jsonl")
        reporter = GateProximityReporter(state_path=tmp_path / "gp.jsonl")
        monitor = EnsembleDivergenceMonitor(state_path=tmp_path / "ed.jsonl")

        # Simulate 10 pairs: 3 tradeable, 4 near-miss rejects, 3 far rejects
        tradeable = [_make_analysis(True, True, True, True, confidence=65.0, pair=f"P{i}") for i in range(3)]
        near_rejects = [_make_analysis(False, False, False, False, confidence=57.0, pair=f"N{i}") for i in range(4)]
        far_rejects = [_make_analysis(False, False, False, False, confidence=45.0, pair=f"F{i}") for i in range(3)]
        analyses = tradeable + near_rejects + far_rejects

        # SignalFunnelTracker
        funnel = tracker.record_scan(analyses)
        assert funnel["total_evaluated"] == 10
        assert funnel["is_tradeable"] == 3

        # GateProximityReporter
        reporter.record_scan(analyses, min_confidence=58.0)
        proximity = reporter.get_proximity_summary()
        assert proximity["total_conf_rejects"] == 7
        assert proximity["near_miss_count"] == 4

        # EnsembleDivergenceMonitor (simulate per-pair TCN/Ridge recording)
        for a in analyses:
            monitor.record(a.pair, tcn_confidence=60.0, ridge_confidence=58.0)
        div_summary = monitor.get_divergence_summary()
        assert div_summary["count"] == 10
