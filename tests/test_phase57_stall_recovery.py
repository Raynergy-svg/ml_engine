"""Tests for Phase 57 US-354: StallRecoveryManager.

Covers:
  - Starts in NORMAL state
  - Enters RECOVERY at stall threshold
  - penalties_suspended() True in RECOVERY
  - penalties_suspended() False in NORMAL
  - is_in_recovery() alias matches penalties_suspended()
  - Recovery exits after RECOVERY_WINDOW ticks
  - Re-enters RECOVERY if still stalled after window
  - Does not enter recovery if below threshold
  - save_state and _load_state round-trip
  - Corrupt state handled gracefully
  - get_state() returns all required keys
"""

import json
import pytest
from pathlib import Path

from src.scanner.stall_recovery import StallRecoveryManager, StallRecoveryState


def make_manager(tmp_path, stall_threshold=5, recovery_window=3):
    return StallRecoveryManager(
        stall_threshold=stall_threshold,
        recovery_window=recovery_window,
        state_path=tmp_path / "stall_recovery.json",
    )


class TestNormalState:

    def test_starts_in_normal_state(self, tmp_path):
        mgr = make_manager(tmp_path)
        assert mgr.get_state()["state"] == StallRecoveryState.NORMAL

    def test_penalties_not_suspended_in_normal(self, tmp_path):
        mgr = make_manager(tmp_path)
        assert mgr.penalties_suspended() is False

    def test_below_threshold_stays_normal(self, tmp_path):
        mgr = make_manager(tmp_path, stall_threshold=5)
        for _ in range(4):
            mgr.tick(scans_since_last_trade=4)
        assert mgr.get_state()["state"] == StallRecoveryState.NORMAL


class TestRecoveryEntry:

    def test_enters_recovery_at_threshold(self, tmp_path):
        mgr = make_manager(tmp_path, stall_threshold=5)
        mgr.tick(scans_since_last_trade=5)
        assert mgr.is_in_recovery() is True

    def test_penalties_suspended_in_recovery(self, tmp_path):
        mgr = make_manager(tmp_path, stall_threshold=5)
        mgr.tick(scans_since_last_trade=5)
        assert mgr.penalties_suspended() is True

    def test_recovery_window_remaining_set(self, tmp_path):
        mgr = make_manager(tmp_path, stall_threshold=5, recovery_window=3)
        mgr.tick(scans_since_last_trade=5)
        assert mgr.get_state()["recovery_scans_remaining"] == 3

    def test_total_recovery_cycles_increments(self, tmp_path):
        mgr = make_manager(tmp_path, stall_threshold=5, recovery_window=3)
        mgr.tick(scans_since_last_trade=5)
        assert mgr.get_state()["total_recovery_cycles"] == 1


class TestRecoveryExit:

    def test_exits_after_recovery_window(self, tmp_path):
        mgr = make_manager(tmp_path, stall_threshold=5, recovery_window=3)
        mgr.tick(scans_since_last_trade=5)   # Enter recovery
        assert mgr.is_in_recovery() is True
        mgr.tick(scans_since_last_trade=0)   # Tick 1 (no longer stalled)
        mgr.tick(scans_since_last_trade=0)   # Tick 2
        mgr.tick(scans_since_last_trade=0)   # Tick 3 → exits
        assert mgr.get_state()["state"] == StallRecoveryState.NORMAL
        assert mgr.penalties_suspended() is False

    def test_re_enters_if_still_stalled_after_window(self, tmp_path):
        mgr = make_manager(tmp_path, stall_threshold=5, recovery_window=2)
        mgr.tick(scans_since_last_trade=5)   # Enter recovery
        mgr.tick(scans_since_last_trade=5)   # Tick 1
        mgr.tick(scans_since_last_trade=5)   # Tick 2 → exits AND re-enters (still stalled)
        assert mgr.is_in_recovery() is True
        assert mgr.get_state()["total_recovery_cycles"] == 2


class TestPersistence:

    def test_state_survives_save_reload(self, tmp_path):
        mgr1 = make_manager(tmp_path, stall_threshold=5, recovery_window=3)
        mgr1.tick(scans_since_last_trade=5)  # Enter recovery
        mgr1.save_state()

        mgr2 = make_manager(tmp_path, stall_threshold=5, recovery_window=3)
        assert mgr2.is_in_recovery() is True
        assert mgr2.get_state()["total_recovery_cycles"] == 1

    def test_corrupt_state_handled_gracefully(self, tmp_path):
        state_path = tmp_path / "stall_recovery.json"
        state_path.write_text("NOT JSON {{{")
        mgr = StallRecoveryManager(
            stall_threshold=5, recovery_window=3, state_path=state_path
        )
        assert mgr.get_state()["state"] == StallRecoveryState.NORMAL


class TestGetState:

    def test_get_state_has_all_required_keys(self, tmp_path):
        mgr = make_manager(tmp_path)
        state = mgr.get_state()
        for key in ["state", "recovery_scans_remaining", "total_recovery_cycles",
                    "total_ticks", "stall_threshold", "recovery_window", "timestamp"]:
            assert key in state, f"Missing key: {key}"
