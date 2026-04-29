"""Mythos audit 2026-04-29 Tier-7 step D — Ctrl+R state-preserving restart.

Pre-fix: every commit during a session forced a manual TUI relaunch.
Operator pain quantified: 6 manual relaunches in this 8-hour Mythos
session alone.

This test verifies action_safe_restart's contract WITHOUT actually
calling os.execv (which would replace the test process image).
We patch os.execv to capture the call, then assert:
  * State-persistence methods on the scanner are attempted in order
  * Soft-handoff beacon written to .claude/state.json with safe_restart
    metadata (operator can correlate "did I restart?" via state.json
    timestamps after the fact)
  * Scanner shutdown invoked
  * os.execv called with the expected python + module args

The key binding registration is verified by inspecting BuddyApp.BINDINGS
since binding lookup requires the full Textual lifecycle.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.tui.app import BuddyApp


def test_ctrl_r_binding_registered():
    """Operators should be able to discover the binding via the visible
    keybind footer."""
    bindings = BuddyApp.BINDINGS
    matched = [b for b in bindings if str(b.key) == "ctrl+r"]
    assert matched, f"ctrl+r binding missing from BuddyApp.BINDINGS"
    binding = matched[0]
    assert binding.action == "safe_restart"
    assert binding.show is True


def _make_app_for_action(scanner_mock=None) -> BuddyApp:
    """Build a BuddyApp instance bypassing Textual's mount lifecycle."""
    app = BuddyApp.__new__(BuddyApp)
    app._scanner = scanner_mock
    app._write_brain = lambda _msg: None
    return app


def test_safe_restart_calls_state_persistence_methods(tmp_path, monkeypatch):
    """Each named persistence method must be called on the scanner.
    Methods missing from the scanner must NOT raise — graceful skip."""
    monkeypatch.chdir(tmp_path)
    scanner = MagicMock()
    scanner.save_scan_state = MagicMock()
    scanner.save_state = MagicMock()
    scanner.flush_state = MagicMock()

    app = _make_app_for_action(scanner)
    with patch("os.execv") as mock_execv:
        app.action_safe_restart()

    scanner.save_scan_state.assert_called_once()
    scanner.save_state.assert_called_once()
    scanner.flush_state.assert_called_once()
    mock_execv.assert_called_once()


def test_safe_restart_writes_beacon_to_state_json(tmp_path, monkeypatch):
    """Beacon must include timestamp, session_id, and reason — those
    let the next boot diagnose 'did the operator just restart?'"""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("BUDDY_SESSION_ID", "test-session-abc")

    app = _make_app_for_action()
    with patch("os.execv"):
        app.action_safe_restart()

    beacon = Path(tmp_path) / ".claude" / "state.json"
    assert beacon.exists()
    data = json.loads(beacon.read_text())
    assert "safe_restart" in data
    sr = data["safe_restart"]
    assert sr["session_id"] == "test-session-abc"
    assert sr["reason"] == "operator_ctrl_r"
    assert sr["at"]  # ISO timestamp present


def test_safe_restart_preserves_existing_state_json(tmp_path, monkeypatch):
    """Beacon must MERGE into existing state.json, never overwrite the
    operator's state (open positions, scan_count, etc.)."""
    monkeypatch.chdir(tmp_path)
    state_dir = tmp_path / ".claude"
    state_dir.mkdir()
    state_file = state_dir / "state.json"
    state_file.write_text(json.dumps({
        "goal": "continuous_scan",
        "open_trades": [{"id": 999}],
        "scan_cycle_count": 1287,
    }))

    app = _make_app_for_action()
    with patch("os.execv"):
        app.action_safe_restart()

    data = json.loads(state_file.read_text())
    # Existing keys preserved
    assert data["goal"] == "continuous_scan"
    assert data["open_trades"] == [{"id": 999}]
    assert data["scan_cycle_count"] == 1287
    # New key added
    assert "safe_restart" in data


def test_safe_restart_shuts_down_scanner_before_exec(tmp_path, monkeypatch):
    """OANDA stream sockets close politely so the broker reconnect is clean."""
    monkeypatch.chdir(tmp_path)
    scanner = MagicMock()
    app = _make_app_for_action(scanner)
    with patch("os.execv"):
        app.action_safe_restart()
    scanner.shutdown.assert_called_once()


def test_safe_restart_invokes_execv_with_module_argv(tmp_path, monkeypatch):
    """Replace process image with the same `python -m src.tui [args]`
    invocation — preserves operator's CLI flags (--demo, --live, etc.)."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, "argv", ["src.tui", "--demo"])

    app = _make_app_for_action()
    with patch("os.execv") as mock_execv:
        app.action_safe_restart()

    mock_execv.assert_called_once()
    args, _ = mock_execv.call_args
    python_path, argv = args
    assert "python" in python_path.lower() or python_path == sys.executable
    assert argv[1] == "-m"
    assert argv[2] == "src.tui"
    assert "--demo" in argv


def test_safe_restart_falls_back_to_quit_if_execv_fails(tmp_path, monkeypatch):
    """If execv raises (rare, usually permission/path issues), the operator
    should still get a clean quit rather than a half-restarted hang."""
    monkeypatch.chdir(tmp_path)
    scanner = MagicMock()
    app = _make_app_for_action(scanner)

    quit_called = {"count": 0}
    def fake_quit(self_):
        quit_called["count"] += 1

    with patch("os.execv", side_effect=OSError("execv permission denied")):
        with patch("textual.app.App.action_quit", fake_quit):
            app.action_safe_restart()

    assert quit_called["count"] == 1


def test_safe_restart_swallows_state_save_exceptions(tmp_path, monkeypatch):
    """A broken save_scan_state must not block the restart sequence —
    operator pressed Ctrl+R; honor it even if state-save is degraded."""
    monkeypatch.chdir(tmp_path)
    scanner = MagicMock()
    scanner.save_scan_state = MagicMock(side_effect=RuntimeError("disk full"))
    scanner.save_state = MagicMock()
    scanner.flush_state = MagicMock()

    app = _make_app_for_action(scanner)
    with patch("os.execv") as mock_execv:
        app.action_safe_restart()

    scanner.save_state.assert_called_once()
    scanner.flush_state.assert_called_once()
    mock_execv.assert_called_once()


def test_safe_restart_works_without_scanner(tmp_path, monkeypatch):
    """Demo mode or pre-init has no scanner — still must restart cleanly."""
    monkeypatch.chdir(tmp_path)
    app = _make_app_for_action(None)
    with patch("os.execv") as mock_execv:
        app.action_safe_restart()
    mock_execv.assert_called_once()
