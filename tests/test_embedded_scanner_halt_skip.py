"""Mythos audit 2026-04-30 — halt-aware scan-skip in EmbeddedScanner.

Pre-fix evidence (verified on disk PID 16596):
  * state.json halted=True set at 14:30:21 by auto-halt commit eacb617
  * virtual_trades.jsonl entries at 14:33:05 / 14:33:23 / 14:33:33
    proving the TUI kept scanning AFTER halt was set
  * grep "halted|get_halted" of src/tui/embedded_scanner.py +
    src/scanner/engine.py returned NOTHING

Same class of bug as f070d39: the state flag was set on one path but
not READ on the live runtime path. continuous.py:_run_smart_loop reads
halted (CLI scan path), but EmbeddedScanner.run_one_cycle (TUI path)
did not.

The fix: check StateEngine.get_halted() at the top of run_one_cycle,
return None to skip the cycle, emit a single brain log on first
halt-skip + reset latch on un-halt so operator sees state changes.
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest


def _make_embedded(halted_value: bool):
    """Build a bare EmbeddedScanner instance for the halt-skip test.
    Bypasses __init__ which has heavy deps."""
    from src.tui.embedded_scanner import EmbeddedScanner
    es = EmbeddedScanner.__new__(EmbeddedScanner)
    es._running = True
    es._scanner = MagicMock()  # truthy, so the `is None` guard doesn't trigger
    es._scanner.scan = MagicMock()
    es._scan_count = 0
    es._brain = MagicMock()
    return es


def testrun_one_cycle_skips_when_halted():
    """When StateEngine.halted=True, run_one_cycle must return None
    BEFORE incrementing _scan_count and BEFORE calling Scanner.scan."""
    es = _make_embedded(halted_value=True)
    with patch(
        "src.scanner.automation.state_engine.StateEngine"
    ) as MockSE:
        MockSE.return_value.get_halted.return_value = True
        result = es.run_one_cycle()

    assert result is None
    assert es._scan_count == 0  # not incremented
    es._scanner.scan.assert_not_called()  # scan NOT invoked


def testrun_one_cycle_emits_brain_message_once_per_halt(monkeypatch):
    """Operator sees the halt state via brain feed. To avoid spamming
    the feed every 5 seconds, the message should fire once per
    transition into halted state."""
    es = _make_embedded(halted_value=True)
    with patch(
        "src.scanner.automation.state_engine.StateEngine"
    ) as MockSE:
        MockSE.return_value.get_halted.return_value = True
        es.run_one_cycle()
        es.run_one_cycle()
        es.run_one_cycle()

    halt_messages = [
        c for c in es._brain.call_args_list
        if any("HALTED" in str(arg) for arg in c.args)
    ]
    assert len(halt_messages) == 1, (
        f"Expected ONE halt brain message across 3 halted cycles, "
        f"got {len(halt_messages)}"
    )


def test_unhalt_resets_latch_so_re_halt_fires_again():
    """After operator un-halts, scan resumes. If the bot then re-halts,
    a fresh brain message must fire — operator needs to see the new
    halt event, not silence."""
    es = _make_embedded(halted_value=True)
    halted_states = [True, False, True]  # halt, unhalt, re-halt

    with patch(
        "src.scanner.automation.state_engine.StateEngine"
    ) as MockSE:
        # Provide fresh instance per call so get_halted() advances through halted_states
        MockSE.return_value.get_halted.side_effect = halted_states
        es.run_one_cycle()  # halted → emit message
        # Cycle 2 will run scan but our scanner mock returns MagicMock — fine.
        es.run_one_cycle()  # unhalted → resets latch, runs scan
        es.run_one_cycle()  # re-halted → emit message AGAIN

    halt_messages = [
        c for c in es._brain.call_args_list
        if any("HALTED" in str(arg) for arg in c.args)
    ]
    assert len(halt_messages) == 2, (
        f"Expected 2 halt brain messages (initial + re-halt), got {len(halt_messages)}"
    )


def testrun_one_cycle_proceeds_when_not_halted():
    """Normal operation: halted=False, _scan_count increments,
    Scanner.scan() is called."""
    es = _make_embedded(halted_value=False)
    # Stub the heavy scan path - we just want to verify it gets called
    es._scanner.scan.return_value = MagicMock(tradeable=[], analyses=[])
    es._auto_execute = False
    es._post_scan_automation = MagicMock()
    es._memory_guard = MagicMock()
    es._autonomy = None
    es._build_enrichment = MagicMock()
    es._maybe_route_to_meta_per_cycle = MagicMock()
    es._config = MagicMock(pairs=["EUR_USD"], default_pairs=["EUR_USD"], blocked_pairs=set())
    es._interval_minutes = 5

    with patch(
        "src.scanner.automation.state_engine.StateEngine"
    ) as MockSE:
        MockSE.return_value.get_halted.return_value = False
        es.run_one_cycle()

    assert es._scan_count == 1
    es._scanner.scan.assert_called_once()


def test_state_engine_failure_does_not_block_scanning():
    """Defensive: if StateEngine itself fails, fall through to normal
    scan rather than locking up the TUI on a degraded state-engine."""
    es = _make_embedded(halted_value=False)
    es._scanner.scan.return_value = MagicMock(tradeable=[], analyses=[])
    es._auto_execute = False
    es._post_scan_automation = MagicMock()
    es._memory_guard = MagicMock()
    es._autonomy = None
    es._build_enrichment = MagicMock()
    es._maybe_route_to_meta_per_cycle = MagicMock()
    es._config = MagicMock(pairs=["EUR_USD"], default_pairs=["EUR_USD"], blocked_pairs=set())
    es._interval_minutes = 5

    with patch(
        "src.scanner.automation.state_engine.StateEngine",
        side_effect=OSError("disk error"),
    ):
        # Must not raise; scan should still proceed.
        es.run_one_cycle()

    assert es._scan_count == 1
    es._scanner.scan.assert_called_once()
