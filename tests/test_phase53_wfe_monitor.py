"""
Tests for Phase 53 US-332: Walk-Forward Efficiency Stability Monitor.

Tests cover:
  - WFE recording and history management
  - Stable → Unstable transition (3+ consecutive declines >5%)
  - Unstable → Recovering transition (first positive cycle)
  - Recovering → Stable transition (2+ consecutive positive cycles)
  - Recovering → Unstable relapse (decline during recovery)
  - Weight freezing and retrieval
  - Trend summary report
  - Persistence save/load roundtrip
  - should_retrain() gating
  - Custom config thresholds
  - Lazy imports and wiring
"""

import json
import os
import tempfile

import pytest

from src.scanner.wfe_monitor import (
    WFEStabilityMonitor,
    WFEMonitorConfig,
    WFESnapshot,
)


@pytest.fixture
def tmp_dir():
    with tempfile.TemporaryDirectory() as d:
        yield d


@pytest.fixture
def monitor(tmp_dir):
    config = WFEMonitorConfig(
        persistence_path=os.path.join(tmp_dir, "wfe_history.json"),
    )
    return WFEStabilityMonitor(config)


class TestWFERecording:

    def test_record_single_wfe(self, monitor):
        """Single WFE recording adds to history."""
        state = monitor.record_wfe(0.72, epoch_count=5, accepted_count=3)
        assert state == "STABLE"
        assert len(monitor.history) == 1
        assert monitor.history[0].wfe == 0.72

    def test_history_trimmed_to_max(self, tmp_dir):
        """History trimmed to max_history."""
        config = WFEMonitorConfig(
            max_history=5,
            persistence_path=os.path.join(tmp_dir, "wfe.json"),
        )
        mon = WFEStabilityMonitor(config)
        for i in range(10):
            mon.record_wfe(0.70 + i * 0.01)
        assert len(mon.history) == 5
        # Most recent should be last
        assert abs(mon.history[-1].wfe - 0.79) < 0.001

    def test_snapshot_to_from_dict(self):
        """WFESnapshot round-trips through dict."""
        snap = WFESnapshot(wfe=0.68, timestamp="2026-03-24T12:00:00Z",
                           epoch_count=4, accepted_count=2)
        d = snap.to_dict()
        restored = WFESnapshot.from_dict(d)
        assert abs(restored.wfe - 0.68) < 0.001
        assert restored.epoch_count == 4
        assert restored.accepted_count == 2


class TestStabilityTransitions:

    def test_stable_stays_stable_on_flat(self, monitor):
        """Flat WFE keeps state STABLE."""
        monitor.record_wfe(0.70)
        monitor.record_wfe(0.70)
        monitor.record_wfe(0.70)
        assert monitor.state == "STABLE"
        assert monitor.should_retrain() is True

    def test_stable_to_unstable_on_declines(self, monitor):
        """3+ consecutive >5% declines → UNSTABLE."""
        monitor.record_wfe(0.80)
        monitor.record_wfe(0.74)  # -7.5%
        monitor.record_wfe(0.68)  # -8.1%
        monitor.record_wfe(0.62)  # -8.8%  → 3 consecutive declines
        assert monitor.state == "UNSTABLE"
        assert monitor.should_retrain() is False

    def test_unstable_to_recovering(self, monitor):
        """First positive cycle in UNSTABLE → RECOVERING."""
        # Drive to UNSTABLE
        monitor.record_wfe(0.80)
        monitor.record_wfe(0.74)
        monitor.record_wfe(0.68)
        monitor.record_wfe(0.62)
        assert monitor.state == "UNSTABLE"

        # One positive cycle
        monitor.record_wfe(0.65)  # +4.8% → positive
        assert monitor.state == "RECOVERING"
        assert monitor.should_retrain() is False  # Still paused

    def test_recovering_to_stable(self, monitor):
        """2+ consecutive positive cycles in RECOVERING → STABLE."""
        # Drive to UNSTABLE
        monitor.record_wfe(0.80)
        monitor.record_wfe(0.74)
        monitor.record_wfe(0.68)
        monitor.record_wfe(0.62)
        assert monitor.state == "UNSTABLE"

        # Recovery
        monitor.record_wfe(0.65)  # → RECOVERING
        monitor.record_wfe(0.70)  # 2nd positive → STABLE
        assert monitor.state == "STABLE"
        assert monitor.should_retrain() is True

    def test_recovering_relapse_to_unstable(self, monitor):
        """Decline during RECOVERING → back to UNSTABLE."""
        # Drive to UNSTABLE
        monitor.record_wfe(0.80)
        monitor.record_wfe(0.74)
        monitor.record_wfe(0.68)
        monitor.record_wfe(0.62)

        # Start recovery
        monitor.record_wfe(0.65)  # → RECOVERING
        assert monitor.state == "RECOVERING"

        # Relapse
        monitor.record_wfe(0.58)  # decline → UNSTABLE
        assert monitor.state == "UNSTABLE"

    def test_small_decline_not_triggered(self, monitor):
        """Declines < 5% don't count as declining."""
        monitor.record_wfe(0.80)
        monitor.record_wfe(0.78)  # -2.5% — below threshold
        monitor.record_wfe(0.76)  # -2.6%
        monitor.record_wfe(0.75)  # -1.3%
        monitor.record_wfe(0.74)  # -1.3%
        assert monitor.state == "STABLE"


class TestWeightFreezing:

    def test_freeze_and_retrieve_weights(self, monitor):
        """Frozen weights can be retrieved."""
        weights = {"trend": 0.12, "momentum": 0.10}
        monitor.freeze_weights(weights)
        frozen = monitor.get_frozen_weights()
        assert frozen is not None
        assert abs(frozen["trend"] - 0.12) < 0.001

    def test_no_frozen_weights_initially(self, monitor):
        """No frozen weights before freeze_weights() is called."""
        assert monitor.get_frozen_weights() is None

    def test_frozen_weights_are_copies(self, monitor):
        """Frozen weights are independent of original dict."""
        weights = {"trend": 0.12}
        monitor.freeze_weights(weights)
        weights["trend"] = 0.99  # Mutate original
        assert monitor.get_frozen_weights()["trend"] == 0.12


class TestTrendSummary:

    def test_summary_with_data(self, monitor):
        """Trend summary returns correct structure."""
        monitor.record_wfe(0.70)
        monitor.record_wfe(0.75)
        summary = monitor.get_trend_summary()
        assert summary["state"] == "STABLE"
        assert abs(summary["wfe_current"] - 0.75) < 0.001
        assert abs(summary["wfe_previous"] - 0.70) < 0.001
        assert summary["history_length"] == 2
        assert summary["has_frozen_weights"] is False

    def test_summary_insufficient_data(self, monitor):
        """Summary with no data returns safe defaults."""
        summary = monitor.get_trend_summary()
        assert summary["trend_direction"] == "insufficient_data"
        assert summary["history_length"] == 0


class TestPersistence:

    def test_save_load_roundtrip(self, tmp_dir):
        """Monitor state round-trips through save/load."""
        path = os.path.join(tmp_dir, "wfe_rt.json")
        config = WFEMonitorConfig(persistence_path=path)

        # Create and populate
        mon1 = WFEStabilityMonitor(config)
        mon1.record_wfe(0.80)
        mon1.record_wfe(0.74)
        mon1.record_wfe(0.68)
        mon1.record_wfe(0.62)  # → UNSTABLE
        mon1.freeze_weights({"trend": 0.15})
        mon1.save_state()

        # Reload
        mon2 = WFEStabilityMonitor(config)
        assert mon2.state == "UNSTABLE"
        assert len(mon2.history) == 4
        assert abs(mon2.history[-1].wfe - 0.62) < 0.001
        frozen = mon2.get_frozen_weights()
        assert frozen is not None
        assert abs(frozen["trend"] - 0.15) < 0.001

    def test_load_missing_file(self, tmp_dir):
        """Missing file starts fresh."""
        config = WFEMonitorConfig(
            persistence_path=os.path.join(tmp_dir, "nonexistent.json"),
        )
        mon = WFEStabilityMonitor(config)
        assert mon.state == "STABLE"
        assert len(mon.history) == 0

    def test_load_corrupted_file(self, tmp_dir):
        """Corrupted file gracefully falls back."""
        path = os.path.join(tmp_dir, "corrupt.json")
        with open(path, "w") as f:
            f.write("{bad json!!!}")
        config = WFEMonitorConfig(persistence_path=path)
        mon = WFEStabilityMonitor(config)
        assert mon.state == "STABLE"


class TestCustomConfig:

    def test_custom_thresholds(self, tmp_dir):
        """Custom decline_cycles_trigger changes sensitivity."""
        config = WFEMonitorConfig(
            decline_cycles_trigger=2,  # More sensitive: only 2 cycles
            decline_threshold=0.03,     # Lower threshold
            persistence_path=os.path.join(tmp_dir, "wfe.json"),
        )
        mon = WFEStabilityMonitor(config)
        mon.record_wfe(0.80)
        mon.record_wfe(0.76)  # -5% > 3%
        mon.record_wfe(0.72)  # -5.3% > 3% → 2 consecutive = UNSTABLE
        assert mon.state == "UNSTABLE"

    def test_custom_recovery_cycles(self, tmp_dir):
        """Custom recovery_cycles_required changes recovery speed."""
        config = WFEMonitorConfig(
            decline_cycles_trigger=2,
            recovery_cycles_required=3,  # Slower recovery
            persistence_path=os.path.join(tmp_dir, "wfe.json"),
        )
        mon = WFEStabilityMonitor(config)
        # Drive to UNSTABLE
        mon.record_wfe(0.80)
        mon.record_wfe(0.70)
        mon.record_wfe(0.60)
        assert mon.state == "UNSTABLE"

        # Recovery needs 3 positive cycles
        mon.record_wfe(0.62)  # → RECOVERING
        mon.record_wfe(0.65)  # 2nd positive — still RECOVERING
        assert mon.state == "RECOVERING"
        mon.record_wfe(0.68)  # 3rd positive → STABLE
        assert mon.state == "STABLE"


class TestWFEMonitorLazyImports:

    def test_wfe_stability_monitor_import(self):
        from src.scanner import WFEStabilityMonitor
        assert WFEStabilityMonitor is not None

    def test_wfe_monitor_config_import(self):
        from src.scanner import WFEMonitorConfig
        assert WFEMonitorConfig is not None


class TestWFEMonitorWiring:

    def test_execution_manager_has_wfe_monitor(self):
        try:
            from src.scanner.execution import ExecutionManager
            import inspect
            source = inspect.getsource(ExecutionManager.__init__)
            assert "_wfe_monitor" in source
            assert "WFEStabilityMonitor" in source
        except ImportError:
            pytest.skip("ExecutionManager not importable")

    def test_wfe_check_in_sync(self):
        try:
            from src.scanner.execution import ExecutionManager
            import inspect
            source = inspect.getsource(ExecutionManager)
            assert "should_retrain" in source
            assert "_wfe_monitor" in source
        except ImportError:
            pytest.skip("ExecutionManager not importable")
