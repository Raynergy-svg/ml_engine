"""Tier 2 T8: Config-write triggers immediate scanner reload.

No mocks. Real classes against real disk via tmp_path per the No-Mock Rule
in .claude/rules/improvement.md (promoted 2026-05-01).

Three load-bearing assertions:
  1. AdjustmentApprover._save_approved sets `config_dirty=true` in state.json
  2. _consume_config_dirty_flag(state_path) returns True and clears the flag
  3. _consume_config_dirty_flag returns False when flag is absent or false
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.scanner.automation.adjustment_approver import AdjustmentApprover


def test_save_approved_sets_config_dirty(tmp_path: Path):
    """Writing approved config adjustments must flip state.json:config_dirty=true.

    Drives the real `_save_approved` write path. The shrink-guard tripwire
    requires a dict payload with a `history` list; we hand it one entry so
    the proposed history (n=1) is never smaller than the on-disk history (n=0).
    """
    approved = tmp_path / "config_adjustments.json"
    approved.write_text(json.dumps({
        "version": 1,
        "history": [],
        "pending": {},
        "last_applied": {},
    }))
    state = tmp_path / "state.json"
    state.write_text(json.dumps({"halted": False, "mode": "demo"}))

    approver = AdjustmentApprover(
        approved_path=approved,
        state_path=state,
    )
    approver._save_approved({
        "version": 1,
        "history": [{
            "key": "atr_sl_multiplier",
            "old_value": 1.0,
            "new_value": 1.3,
            "source": "test",
            "reason": "T8 test",
            "timestamp": "2026-05-12T00:00:00Z",
        }],
        "pending": {},
        "last_applied": {},
        "total_adjustments": 1,
    })

    state_data = json.loads(state.read_text())
    assert state_data.get("config_dirty") is True, (
        "AdjustmentApprover._save_approved must mark config_dirty=true "
        "so the scanner can pick up the new approved adjustments at the "
        "next cycle boundary."
    )
    # Sanity: other fields preserved (atomic write must not stomp state).
    assert state_data.get("halted") is False
    assert state_data.get("mode") == "demo"


def test_dirty_flag_cleared_after_consume(tmp_path: Path):
    """`_consume_config_dirty_flag` must return True AND atomically clear flag."""
    state = tmp_path / "state.json"
    state.write_text(json.dumps({
        "halted": False,
        "mode": "demo",
        "config_dirty": True,
    }))

    from src.tui.embedded_scanner import _consume_config_dirty_flag
    dirty = _consume_config_dirty_flag(state)
    assert dirty is True

    after = json.loads(state.read_text())
    assert after.get("config_dirty") is False, (
        "After consume, config_dirty must be cleared so subsequent cycles "
        "do not re-reload unnecessarily."
    )
    # Sanity: other fields preserved.
    assert after.get("halted") is False
    assert after.get("mode") == "demo"


def test_dirty_flag_false_when_not_set(tmp_path: Path):
    """No `config_dirty` key, or false value, means no reload needed."""
    from src.tui.embedded_scanner import _consume_config_dirty_flag

    # Case 1: key absent.
    state = tmp_path / "state.json"
    state.write_text(json.dumps({"halted": False, "mode": "demo"}))
    assert _consume_config_dirty_flag(state) is False

    # Case 2: key present but false.
    state.write_text(json.dumps({
        "halted": False,
        "mode": "demo",
        "config_dirty": False,
    }))
    assert _consume_config_dirty_flag(state) is False

    # Case 3: state file missing entirely — graceful no-reload, no crash.
    missing = tmp_path / "nonexistent_state.json"
    assert _consume_config_dirty_flag(missing) is False
