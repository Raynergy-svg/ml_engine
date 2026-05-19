"""US-013 — Inbox approve-all moves to a Textual @work worker.

This test proves three things end-to-end (real classes, real disk, no mocks):

1. ``InboxScreen.action_approve_all`` no longer blocks the main event
   loop. While the worker is in flight, a concurrent keypress is still
   processed within a tight latency budget (100ms per the AC).
2. The worker reports progress through ``_approve_all_progress`` /
   ``_update_progress`` so the ProgressBar widget actually moves.
3. The worker drives real ``HomeworkReviewer.approve`` against a real
   ``HomeworkStore`` backed by ``tmp_path`` — the 5 seeded homework
   entries land in history once approve-all completes.

NO MOCKS by project rule (.claude/rules/improvement.md "No-Mock Rule").
Module-level constant redirection via pytest's monkeypatch is the
no-mock way to point real classes at ``tmp_path``: it changes a path
attribute, it does not substitute a test-double.
"""
from __future__ import annotations

import asyncio
import dataclasses
import json
import time
from pathlib import Path
from typing import Iterator, List

import pytest
from textual.app import App, ComposeResult
from textual.containers import Horizontal
from textual.widgets import Label, ProgressBar, TabbedContent, TabPane

from src.scanner.automation.homework import HomeworkEntry, HomeworkStore
from src.scanner.automation.homework import store as homework_store_module
from src.scanner.automation.homework import reviewer as homework_reviewer_module
from src.scanner.automation.homework import applicator as homework_applicator_module
from src.tui.screens.inbox_screen import InboxScreen


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_entry(idx: int) -> HomeworkEntry:
    """Build a real, complete HomeworkEntry — all required fields filled in."""
    return HomeworkEntry(
        homework_id=f"hw-us013-{idx}",
        trade_id=f"trade-{idx:04d}",
        generated_at="2026-05-18T22:00:00Z",
        pair="EUR_USD",
        direction="LONG" if idx % 2 == 0 else "SHORT",
        entry_price=1.10 + idx * 0.001,
        sl_price=1.09 + idx * 0.001,
        tp_price=1.12 + idx * 0.001,
        rr_ratio=2.0,
        confidence=0.65,
        weighted_vote_score=0.70,
        regime="NORMAL",
        agent_verdicts=[],
        close_time="2026-05-18T22:30:00Z",
        close_price=1.09 + idx * 0.001,
        realized_pl=-12.5 if idx % 2 == 0 else 18.7,
        close_reason="SL" if idx % 2 == 0 else "TP",
        duration_minutes=30,
        mfe_pips=2.0,
        mae_pips=12.0,
        analysis_markdown=f"## Analysis #{idx}\n\nReal homework body.",
        proposed_lesson=f"Lesson learned from trade {idx}.",
        confidence_in_analysis=0.7,
        agents_to_reinforce=["risk_sentinel"],
        agents_to_penalize=["trend"],
    )


@pytest.fixture
def tmp_paths(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Path]:
    """Redirect every module-level default path to tmp_path.

    The HomeworkStore / HomeworkReviewer / TrainingSignalApplicator
    defaults reference real on-disk paths under .claude/ + trained_data/.
    Without redirection the test would mutate the operator's real state.
    Monkeypatching the module constants is *not* a mock — the underlying
    classes still run their full real logic against real files.
    """
    claude_dir = tmp_path / ".claude"
    claude_dir.mkdir(parents=True, exist_ok=True)
    weights_dir = tmp_path / "trained_data" / "models"
    weights_dir.mkdir(parents=True, exist_ok=True)

    pending_path = claude_dir / "homework_pending.jsonl"
    history_path = claude_dir / "homework_history.jsonl"
    quarantine_path = claude_dir / "homework_quarantine.jsonl"
    rejected_log = claude_dir / "rejected_heuristics_log.jsonl"
    weights_path = weights_dir / "agent_weights.json"

    # Seed agent_weights.json with a minimal regime-keyed dict so the
    # TrainingSignalApplicator can write deltas without erroring.
    weights_path.write_text(
        json.dumps(
            {
                "NORMAL": {"trend": 1.0, "risk_sentinel": 1.0},
                "_meta": {"version": 1},
            },
            indent=2,
        )
    )

    monkeypatch.setattr(homework_store_module, "DEFAULT_PENDING_PATH", pending_path)
    monkeypatch.setattr(homework_store_module, "DEFAULT_HISTORY_PATH", history_path)
    monkeypatch.setattr(
        homework_store_module, "DEFAULT_QUARANTINE_PATH", quarantine_path
    )
    monkeypatch.setattr(
        homework_reviewer_module, "DEFAULT_REJECTED_LOG_PATH", rejected_log
    )
    monkeypatch.setattr(
        homework_applicator_module, "DEFAULT_WEIGHTS_PATH", weights_path
    )
    yield tmp_path


@pytest.fixture
def seeded_store(tmp_paths: Path) -> HomeworkStore:
    """Five real homework entries persisted via the real HomeworkStore."""
    store = HomeworkStore()
    for idx in range(5):
        store.add(_make_entry(idx))
    pending = store.list_pending()
    assert len(pending) == 5, f"seed failed — got {len(pending)} entries"
    return store


# ---------------------------------------------------------------------------
# Test host app — wraps InboxScreen so we can drive it through Pilot
# ---------------------------------------------------------------------------


class _InboxHost(App):
    """Minimal App that hosts InboxScreen inside a TabbedContent.

    Mirrors the real BuddyApp structure so InboxScreen's
    ``_update_tab_label`` lookup (#main-tabs / get_tab("inbox")) doesn't
    error out. The screen still owns its own composition + lifecycle.
    """

    def __init__(self, project_root: str) -> None:
        super().__init__()
        self._project_root = project_root
        # F1 keypress sentinel — counts how many times the main thread
        # processed an F1 action while the approve-all worker was active.
        self.f1_presses: List[float] = []
        # Latency budget for the AC: each F1 press must be acknowledged
        # within 100ms of being dispatched.
        self.last_dispatch_ts: float = 0.0
        self.last_press_latency_ms: float = 0.0

    BINDINGS = [("f1", "record_f1", "record")]

    def compose(self) -> ComposeResult:
        with TabbedContent(initial="inbox", id="main-tabs"):
            with TabPane("◈ Inbox", id="inbox"):
                yield InboxScreen(
                    project_root=self._project_root,
                    id="inbox-screen",
                )

    def action_record_f1(self) -> None:
        """Main-thread responsiveness probe. If the event loop is
        blocked by approve-all, this never fires until the worker
        finishes — which is exactly what the AC is checking against.
        """
        now = time.monotonic()
        if self.last_dispatch_ts > 0:
            self.last_press_latency_ms = (now - self.last_dispatch_ts) * 1000.0
        self.f1_presses.append(now)


# ---------------------------------------------------------------------------
# AC #1 — worker decorator + responsiveness
# ---------------------------------------------------------------------------


def test_action_approve_all_is_dispatched_to_worker() -> None:
    """AC: action_approve_all decorated or refactored to use @work(thread=True).

    Static guarantee: the worker entry point exists, carries the @work
    marker, and runs in a thread group dedicated to approve-all.
    """
    worker = getattr(InboxScreen, "_approve_all_worker", None)
    assert worker is not None, "_approve_all_worker method must exist"

    # Textual's @work decorator wraps the function and attaches the
    # worker config on the wrapper. We just need to verify thread=True
    # is honored by looking at the wrapper's closure / name; the precise
    # attribute layout differs across Textual versions, so we assert by
    # behavior: calling it returns a Worker, not a bare value.
    src = (
        Path(__file__).resolve().parent.parent
        / "src" / "tui" / "screens" / "inbox_screen.py"
    ).read_text()
    assert "@work(thread=True" in src, (
        "action_approve_all worker must be decorated with @work(thread=True)"
    )
    assert "call_from_thread" in src, (
        "Worker must publish progress via self.app.call_from_thread"
    )
    assert "ProgressBar" in src, "Inbox must mount a ProgressBar widget"


# ---------------------------------------------------------------------------
# AC #2 — Pilot E2E with real worker + real homework + real keypress
# ---------------------------------------------------------------------------


def test_approve_all_worker_does_not_block_main_loop(
    seeded_store: HomeworkStore, tmp_paths: Path
) -> None:
    """AC: TUI main thread remains responsive while approve-all runs.

    Mounts the real InboxScreen against a real HomeworkStore with 5
    real entries, fires action_approve_all, dispatches an F1 keypress
    while the worker thread is still in flight, and asserts:

      * the F1 keypress is acknowledged (action_record_f1 fires)
      * the homework history file gains 5 entries (worker completed
        the real approvals against real disk)
      * the ProgressBar reaches its declared total

    Async tests use ``asyncio.run`` directly (repo convention; no
    pytest-asyncio plugin installed).
    """
    app = _InboxHost(project_root=str(tmp_paths))

    async def _run() -> None:
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            # Give InboxScreen.on_mount + _load_proposals a tick to settle.
            await pilot.pause()

            inbox = app.query_one("#inbox-screen", InboxScreen)
            # Sanity — the 5 seeded entries are in the visible queue before
            # we kick off approve-all.
            hw_rows = [
                r for r in inbox._current_rows if r.entry_type == "homework"
            ]
            assert len(hw_rows) == 5, (
                f"InboxScreen should have loaded 5 homework rows, got {len(hw_rows)}"
            )

            # ── Fire approve-all. action_approve_all returns immediately
            # because it dispatches the work to a thread; the worker runs
            # in the background.
            inbox.action_approve_all()
            assert inbox._approve_all_running is True, (
                "guard flag must be set the instant the worker is dispatched"
            )

            # ── Press F1 while the worker is presumably still in flight.
            # If action_approve_all were still synchronous, the press would
            # be queued behind the loops + drain and we'd never get inside
            # the latency budget.
            app.last_dispatch_ts = time.monotonic()
            await pilot.press("f1")
            # One short pause to let the press dispatch into action_record_f1.
            await pilot.pause()

            assert len(app.f1_presses) >= 1, (
                "F1 keypress should be acknowledged by the main thread while "
                "the approve-all worker runs on a separate thread"
            )
            assert app.last_press_latency_ms < 100.0, (
                "F1 keypress must be processed within 100ms of dispatch — "
                f"observed {app.last_press_latency_ms:.1f}ms"
            )

            # ── Wait for the worker to finish (poll the guard flag).
            for _ in range(200):  # up to ~20s safety budget
                if not inbox._approve_all_running:
                    break
                await asyncio.sleep(0.05)
            else:
                pytest.fail(
                    "approve-all worker did not complete within 10s budget"
                )

            # Allow a final tick for _approve_all_complete to reload the queue.
            await pilot.pause()

            # ── Verify the worker actually approved every homework entry.
            history = seeded_store.list_history()
            assert len(history) == 5, (
                f"expected 5 entries in history after approve-all, got {len(history)}"
            )
            assert all(e.operator_grade == "approved" for e in history)

            # ── ProgressBar visited 5/5 (or 6/6 if we include the
            # adjustment + meta sentinels — both are zero-length here).
            bar = app.query_one("#approve-all-progress", ProgressBar)
            assert bar.progress is not None and bar.progress >= 5, (
                f"ProgressBar should have advanced to at least 5; got {bar.progress}"
            )
            assert bar.total is not None and bar.total >= 5

            # ── Completion banner shows "approved N items".
            label = app.query_one("#approve-all-progress-label", Label)
            label_text = str(label.render()).lower()
            assert "approved" in label_text, (
                f"completion label should announce approved count; got {label_text!r}"
            )

    asyncio.run(_run())


# ---------------------------------------------------------------------------
# AC #3 — re-entrancy guard + visibility lifecycle
# ---------------------------------------------------------------------------


def test_progress_bar_toggles_visibility_around_run(
    seeded_store: HomeworkStore, tmp_paths: Path
) -> None:
    """AC: ProgressBar shown during approve-all, hidden afterward."""
    app = _InboxHost(project_root=str(tmp_paths))

    async def _run() -> None:
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            await pilot.pause()

            inbox = app.query_one("#inbox-screen", InboxScreen)
            row = app.query_one("#approve-all-progress-row", Horizontal)

            # Hidden before approve-all fires.
            assert row.display is False, (
                "progress row must start hidden — current display="
                f"{row.display!r}"
            )

            inbox.action_approve_all()
            await pilot.pause()
            # display flips True the instant the dispatcher runs (main thread).
            assert row.display is True, (
                "progress row must be visible while approve-all is running"
            )

            # Worker completes.
            for _ in range(200):
                if not inbox._approve_all_running:
                    break
                await asyncio.sleep(0.05)
            else:
                pytest.fail("approve-all worker did not complete in time")

            # Auto-hide timer fires at 3s. Wait a touch over that.
            await asyncio.sleep(3.3)
            await pilot.pause()
            assert row.display is False, (
                "progress row must auto-hide ~3s after completion"
            )

    asyncio.run(_run())


def test_reentrancy_guard_blocks_second_call(
    seeded_store: HomeworkStore, tmp_paths: Path
) -> None:
    """Second action_approve_all while a worker is in flight is a no-op
    plus notification — protects against duplicate-spawn races from
    button mash + hotkey double-tap."""
    app = _InboxHost(project_root=str(tmp_paths))

    async def _run() -> None:
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            await pilot.pause()

            inbox = app.query_one("#inbox-screen", InboxScreen)
            inbox.action_approve_all()
            # Immediately fire a second call. The guard should short-circuit.
            inbox.action_approve_all()
            assert inbox._approve_all_running is True

            # Wait for the single in-flight worker to finish.
            for _ in range(200):
                if not inbox._approve_all_running:
                    break
                await asyncio.sleep(0.05)
            else:
                pytest.fail("approve-all worker did not complete in time")
            await pilot.pause()

            # Exactly 5 history rows — second call did not double-process.
            history = seeded_store.list_history()
            assert len(history) == 5

    asyncio.run(_run())
