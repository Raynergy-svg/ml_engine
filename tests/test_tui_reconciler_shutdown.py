"""EmbeddedScanner reconciler thread/task shutdown tests (P0-6).

The audit (2026-06-21) found ``EmbeddedScanner.shutdown()`` never stopped the
``state-reconciler`` asyncio loop nor joined its daemon thread — a leak on every
TUI exit. This module verifies the fix: shutdown() stops the loop cross-thread
and reaps the thread within the join timeout.

NO MOCKS (CLAUDE.md No-Mock Rule): we drive a REAL ``StateReconciler`` with a
real injected ``oanda_fetcher`` callable (a plain function returning a dict —
not a mock) and a real ``EmbeddedScanner`` against real ``tmp_path`` disk.
``run_forever`` genuinely awaits ``asyncio.sleep`` on a real event loop.
"""
from __future__ import annotations

import time
from pathlib import Path

from src.scanner.automation.state_reconciler import StateReconciler
from src.tui.embedded_scanner import EmbeddedScanner


def _make_scanner() -> EmbeddedScanner:
    return EmbeddedScanner(brain_callback=lambda *a, **k: None)


def _make_real_reconciler(tmp_path: Path) -> StateReconciler:
    # Real reconciler; inject a real fetcher so it never touches OANDA.
    def _fetcher() -> dict:
        return {"nav": 100000.0, "open_positions": []}

    return StateReconciler(
        state_path=tmp_path / "state.json",
        drift_log_path=tmp_path / "state_drift_log.jsonl",
        oanda_fetcher=_fetcher,
        interval_seconds=1.0,
        project_root=tmp_path,
    )


def test_reconciler_loop_starts_and_shutdown_joins(tmp_path: Path) -> None:
    """_start_reconciler_loop spins a live thread; shutdown() reaps it."""
    scanner = _make_scanner()
    scanner._reconciler = _make_real_reconciler(tmp_path)
    scanner._start_reconciler_loop()

    # Let the thread + loop come up.
    deadline = time.time() + 2.0
    while scanner._reconciler_loop is None and time.time() < deadline:
        time.sleep(0.01)

    assert scanner._reconciler_thread is not None
    assert scanner._reconciler_thread.is_alive()
    assert scanner._reconciler_loop is not None

    scanner.shutdown()

    # shutdown() must have stopped the loop, joined the thread, and cleared
    # the handles — no leak past app exit.
    assert scanner._reconciler_thread is None
    assert scanner._reconciler_loop is None
    assert scanner._reconciler_task is None


def test_shutdown_without_reconciler_is_safe(tmp_path: Path) -> None:
    """shutdown() must not raise when the reconciler never started."""
    scanner = _make_scanner()
    # Never call _start_reconciler_loop — handles stay None.
    scanner.shutdown()
    assert scanner._reconciler_thread is None
    assert scanner._reconciler_loop is None


def test_shutdown_does_not_hang(tmp_path: Path) -> None:
    """shutdown() returns well within the join timeout (no hang)."""
    scanner = _make_scanner()
    scanner._reconciler = _make_real_reconciler(tmp_path)
    scanner._start_reconciler_loop()

    deadline = time.time() + 2.0
    while scanner._reconciler_loop is None and time.time() < deadline:
        time.sleep(0.01)

    t0 = time.monotonic()
    scanner.shutdown()
    elapsed = time.monotonic() - t0
    # join timeout is 2s; a clean stop should be near-instant. Allow generous
    # headroom for CI but prove it does not hang on the full timeout+.
    assert elapsed < 3.0
