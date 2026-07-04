"""Regression test for the close_trade halt guard (operator-directed safety-ADD, 2026-06-24).

FINDING: execute_trade has a mid-cycle halt guard (execution.py ~2093) but close_trade had NONE, so
an autonomous/programmatic close could reach the broker while state.halted=True. This adds a
deterministic code-layer guard: BLOCK autonomous closes while halted; allow an EXPLICIT
operator-initiated close via operator_override=True (e.g. the TUI's halt-aware confirm). Safety-ADD
only — it can only make close MORE restrictive, never less.

REPRODUCE-THEN-RESOLVE: `test_close_trade_blocks_when_halted` asserts the exact `BLOCKED:
state.halted=True` return, which ONLY the new guard produces — so it FAILS without the guard
(pre-fix close_trade proceeds to the broker) and PASSES with it.

No mocks: real ExecutionManager, real StateEngine, real disk via tmp_path. The "not blocked" cases
point OANDA_API_URL at an unreachable local address so close_trade proceeds PAST the guard without
hitting the real OANDA practice API (the failure is then a connection error, NOT the halt block).
"""
from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from src.scanner.execution import ExecutionConfig, ExecutionManager
from src.scanner.automation.state_engine import StateEngine

_BLOCKED = "BLOCKED: state.halted=True"


def _close(mgr, trade_id="TEST_CLOSE_1", **kw):
    return asyncio.run(mgr.close_trade(trade_id, **kw))


def test_close_trade_blocks_when_halted(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """An autonomous close MUST be blocked when state.halted=True (before any broker call)."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / ".claude").mkdir()
    (tmp_path / "trained_data").mkdir()
    se = StateEngine()
    se.set_halted(True)
    assert se.get_halted() is True  # fixture sanity

    mgr = ExecutionManager(config=ExecutionConfig())
    result = _close(mgr)

    assert result["success"] is False
    assert result["error"] == _BLOCKED            # blocked by the halt guard...
    assert result["trade_id"] == "TEST_CLOSE_1"   # ...before any broker call (no network needed)


def test_close_trade_operator_override_passes_when_halted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """An EXPLICIT operator-initiated close (operator_override=True) bypasses the halt guard."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / ".claude").mkdir()
    (tmp_path / "trained_data").mkdir()
    monkeypatch.setenv("OANDA_API_URL", "http://127.0.0.1:1")  # avoid the real OANDA API
    StateEngine().set_halted(True)

    mgr = ExecutionManager(config=ExecutionConfig())
    result = _close(mgr, operator_override=True)

    assert result["success"] is False
    assert result["error"] != _BLOCKED                # got PAST the guard...
    assert "halted" not in result["error"].lower()    # ...failed downstream, not on the halt


def test_close_trade_proceeds_when_not_halted(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """With halted=False, the halt guard must NOT short-circuit (close behavior unchanged)."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / ".claude").mkdir()
    (tmp_path / "trained_data").mkdir()
    monkeypatch.setenv("OANDA_API_URL", "http://127.0.0.1:1")
    se = StateEngine()
    se.set_halted(False)
    assert se.get_halted() is False

    mgr = ExecutionManager(config=ExecutionConfig())
    result = _close(mgr)

    assert result["success"] is False           # fails downstream (no reachable broker), but...
    assert result["error"] != _BLOCKED          # ...NOT the halt guard
    assert "halted" not in result["error"].lower()


def test_close_trade_halt_check_blocks_when_state_unreadable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """2026-07-02 (operator decision): fail-CLOSED, matching execute_trade. A corrupt state.json
    must BLOCK an autonomous close, not silently permit it. Never raises out of the guard."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / ".claude").mkdir()
    (tmp_path / ".claude" / "state.json").write_text("{not valid json")
    (tmp_path / "trained_data").mkdir()
    monkeypatch.setenv("OANDA_API_URL", "http://127.0.0.1:1")

    mgr = ExecutionManager(config=ExecutionConfig())
    result = _close(mgr)  # get_halted_strict() reads the file directly -> unreadable -> fail-closed

    assert result["success"] is False
    assert result["error"] == _BLOCKED
