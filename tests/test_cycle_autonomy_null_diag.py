"""Behavioral tests for the null-diag short-circuit (Angle 3', 2026-05-10).

Hard rules from the brief:

  - No mocks. No `unittest.mock`, no `MagicMock`, no `patch`. Tests use the
    real `CycleAutonomyTriggers` class against `tmp_path`-redirected disk.
  - No test-doubling of `_invoke` or `invoke_claude_reflection`. Instead
    we set `BUDDY_REFLECTION_DRY_RUN=1` (a real prod flag) so the heavy
    path early-returns without spawning Claude.
  - "Did NOT route" is verified by checking that the on-disk
    `reflection_events.jsonl` got the line AND the heavy path's brain log
    line ("autonomy reflection complete"/"firing") was NOT emitted.
  - "DID route" is verified by checking that no event was written AND
    that the heavy-path brain log line WAS emitted (dry-run variant).

These tests do not exercise `MetaManager` because this branch's
`cycle_autonomy.py` does not depend on it (the heavy path here is
`_invoke` → `invoke_claude_reflection`). The brief's MetaManager test
shape was written against the sister branch with route_incident; the
behavioral contract — "null-diag fires write to the events sink and do
NOT engage the heavy path" — is identical and that is what we verify.
"""
from __future__ import annotations

import json
import multiprocessing as mp
import os
import threading
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import List

import pytest


# ── Real-class fixtures (no mocks) ─────────────────────────────────────


class _BrainCapture:
    """Real callable that records the lines cycle_autonomy emits to the
    brain feed. NOT a mock — a plain Python class with a __call__ that
    appends to a list. Tests inspect `.lines` to verify which code path
    ran. The shape of this object matches what `EmbeddedScanner._write_brain`
    presents to `CycleAutonomyTriggers`.
    """

    def __init__(self) -> None:
        self.lines: List[str] = []

    def __call__(self, line: str) -> None:
        self.lines.append(str(line))


@pytest.fixture(autouse=True)
def _isolate_env(monkeypatch, tmp_path):
    """Each test runs in its own tmp_path with a quiet env. We force the
    reflection_events writer to a tmp_path-backed instance so production
    `.claude/reflection_events.jsonl` is never touched."""
    # Disable any cached singleton from a prior test
    from src.scanner.automation import reflection_events as re_mod

    events_path = tmp_path / ".claude" / "reflection_events.jsonl"
    re_mod.reset_default_writer_for_tests(
        path=events_path,
        alarm_threshold_per_24h=re_mod.ALARM_THRESHOLD_PER_24H,
    )

    # Redirect cwd so any code that uses Path("logs/...") writes inside
    # tmp_path and we can assert on absence.
    monkeypatch.chdir(tmp_path)

    # Ensure feature flags default to enabled (tests opt out individually)
    for key in (
        "BUDDY_DISABLE_PERIODIC_REFLECTION",
        "BUDDY_DISABLE_SELF_HEAL_REFLECTION",
        "BUDDY_DISABLE_REJECTION_REFLECTION",
        "BUDDY_DISABLE_LOSING_STREAK_REFLECTION",
    ):
        monkeypatch.delenv(key, raising=False)

    # Mode breadcrumb the writer records into events
    monkeypatch.setenv("BUDDY_MODE", "DRY_RUN")
    yield events_path

    # Reset the singleton between tests so leftover state doesn't bleed
    re_mod._DEFAULT_WRITER = None


def _make_triggers(**kwargs):
    """Construct a real CycleAutonomyTriggers with a real brain capture."""
    from src.scanner.automation.cycle_autonomy import CycleAutonomyTriggers

    brain = _BrainCapture()
    kwargs.setdefault("brain_callback", brain)
    t = CycleAutonomyTriggers(**kwargs)
    # Stash for inspection (not a mock — it's the same object the trigger
    # already holds via `self._brain`)
    t._capture = brain  # type: ignore[attr-defined]
    return t


def _read_events(path: Path) -> List[dict]:
    if not path.exists():
        return []
    out: List[dict] = []
    for raw in path.read_text().splitlines():
        if not raw.strip():
            continue
        out.append(json.loads(raw))
    return out


def _heavy_path_engaged(brain: _BrainCapture, trigger: str) -> bool:
    """Did the heavy reflection path (Claude spawn) actually engage?

    The heavy path emits a `▸ <Trigger> reflection firing ...` line
    BEFORE calling `_invoke`. The null-diag short-circuit emits a
    `◌ <trigger> fire skipped (null-diag: ...)` line and never reaches
    the firing line. Reading the brain capture is a real disk-equivalent
    check (it's exactly what the TUI sees).
    """
    needle = "reflection firing" if trigger != "self_heal" else "Self-heal reflection firing"
    return any(needle in line for line in brain.lines)


def _short_circuit_logged(brain: _BrainCapture) -> bool:
    return any("null-diag" in line for line in brain.lines)


# ── Test 1: periodic with None diag short-circuits ─────────────────────


def test_periodic_with_none_diag_writes_event_no_changepackage(
    monkeypatch, _isolate_env: Path
) -> None:
    """Brief test #1: fire periodic with `_pending_diag` unset (None);
    assert reflection_events.jsonl has 1 line and the heavy path did
    NOT engage. The "no ChangePackage" assertion in the brief maps here
    to "no Claude reflection log entry" because this branch's heavy path
    is a Claude subprocess, not the meta-pipeline.
    """
    # Belt-and-suspenders: even if the short-circuit fails and the heavy
    # path engages, the dry-run flag prevents a real Claude subprocess.
    monkeypatch.setenv("BUDDY_REFLECTION_DRY_RUN", "1")

    t = _make_triggers(periodic_every_n=1)

    # `_pending_diag` is never set on a fresh instance — assert that
    # explicitly so the test guards against a future regression where
    # the constructor pre-populates it.
    assert getattr(t, "_pending_diag", None) is None

    short_circuited = t._fire_periodic(
        scan_count=361,
        scan_result=SimpleNamespace(tradeable=[], analyses=[1, 2, 3]),
        trades_executed=0,
    )
    assert short_circuited is True, "periodic with no diag must short-circuit"

    events = _read_events(_isolate_env)
    assert len(events) == 1, f"expected 1 event, got {len(events)}: {events}"
    ev = events[0]
    assert ev["trigger"] == "periodic"
    assert ev["cycle"] == 361
    assert ev["mode"] == "DRY_RUN"
    assert ev["null_diag_reason"] == "diag_is_none"
    assert ev["context"]["trades_executed"] == 0
    assert ev["context"]["tradeable_count"] == 0
    assert ev["context"]["scanned_count"] == 3
    assert ev["context"]["rejection_streak"] is None
    assert ev["context"]["losing_streak"] is None
    # ts_utc parses as ISO 8601
    parsed = datetime.fromisoformat(ev["ts_utc"])
    assert parsed.tzinfo is not None

    # Heavy path did not engage (no "Periodic reflection firing" line)
    assert not _heavy_path_engaged(t._capture, "periodic")
    # Short-circuit breadcrumb was logged
    assert _short_circuit_logged(t._capture)


# ── Test 2: rejection with empty recommended_actions short-circuits ────


def test_rejection_with_empty_recommended_actions_writes_event_no_changepackage(
    monkeypatch, _isolate_env: Path
) -> None:
    """Brief test #2: caller sets `_pending_diag = {"recommended_actions": []}`
    — the empty-list case. Must short-circuit with reason
    `empty_recommended_actions`.
    """
    monkeypatch.setenv("BUDDY_REFLECTION_DRY_RUN", "1")

    t = _make_triggers(rejection_streak_threshold=3)
    # Simulate the streak threshold having been reached
    t._rejection_streak = 3
    # Caller has populated diag but with no actions — the bug case
    t._pending_diag = {"status": "WARN", "recommended_actions": []}

    short_circuited = t._fire_rejection(scan_count=42, tradeable_count=4)
    assert short_circuited is True

    events = _read_events(_isolate_env)
    assert len(events) == 1
    ev = events[0]
    assert ev["trigger"] == "rejection"
    assert ev["cycle"] == 42
    assert ev["null_diag_reason"] == "empty_recommended_actions"
    assert ev["context"]["rejection_streak"] == 3
    assert ev["context"]["tradeable_count"] == 4
    # rejection streak counter was reset (existing behavior preserved)
    assert t._rejection_streak == 0

    assert not _heavy_path_engaged(t._capture, "rejection")


# ── Test 3: losing_streak with None diag short-circuits ────────────────


def test_losing_streak_with_none_diag_writes_event_no_changepackage(
    monkeypatch, _isolate_env: Path
) -> None:
    """Brief test #3: fire losing_streak without diag → short-circuit."""
    monkeypatch.setenv("BUDDY_REFLECTION_DRY_RUN", "1")

    t = _make_triggers(losing_streak_threshold=3)
    # Simulate `_should_fire_losing_streak()` having stashed the recent losses
    t._pending_losing_streak = [
        {"trade_id": "T1", "pair": "EUR_USD"},
        {"trade_id": "T2", "pair": "GBP_USD"},
        {"trade_id": "T3", "pair": "USD_JPY"},
    ]
    assert getattr(t, "_pending_diag", None) is None

    short_circuited = t._fire_losing_streak(scan_count=999)
    assert short_circuited is True

    events = _read_events(_isolate_env)
    assert len(events) == 1
    ev = events[0]
    assert ev["trigger"] == "losing_streak"
    assert ev["cycle"] == 999
    assert ev["null_diag_reason"] == "diag_is_none"
    assert ev["context"]["losing_streak"] == 3
    assert ev["context"]["trades_executed"] == 3

    assert not _heavy_path_engaged(t._capture, "losing_streak")


# ── Test 4: self-heal with diag DOES route (heavy path) ────────────────


def test_self_heal_with_diag_DOES_route(
    monkeypatch, _isolate_env: Path
) -> None:
    """Brief test #4: self-heal with populated diag must NOT short-circuit
    — the operator must still get the deep reflection. We set
    BUDDY_REFLECTION_DRY_RUN=1 so `_invoke` early-returns without
    spawning Claude, but the heavy-path brain log line ('Self-heal
    reflection firing') still fires BEFORE the dry-run check.

    Negative assertion: no entry was added to reflection_events.jsonl
    (the new sink is for null-diag only; self-heal is unchanged).
    """
    monkeypatch.setenv("BUDDY_REFLECTION_DRY_RUN", "1")

    t = _make_triggers()
    t._pending_diag = {
        "status": "DEGRADED",
        "issues": [{"check": "model_age", "severity": "WARN", "detail": "30d"}],
        "recommended_actions": ["tighten:min_confidence"],
    }

    # _fire_self_heal returns None (unchanged signature). The test verifies
    # behavior via brain log lines + disk state.
    result = t._fire_self_heal(scan_count=7)
    assert result is None  # signature preserved

    # Self-heal must NOT write to the null-diag sink
    events = _read_events(_isolate_env)
    assert events == [], f"self-heal must not emit null-diag events, got {events}"

    # Heavy path engaged — the "firing" brain line was emitted
    assert _heavy_path_engaged(t._capture, "self_heal"), (
        f"self-heal must route to heavy path; brain lines: {t._capture.lines}"
    )
    # And the dry-run breadcrumb fired (proves _invoke was actually called
    # but the dry-run flag short-circuited the subprocess spawn — exactly
    # what we want for this test).
    assert any(
        "dry-run" in line and "SELFHEAL" in line for line in t._capture.lines
    ), f"_invoke should have been called and dry-run-skipped; brain lines: {t._capture.lines}"


# ── Test 5: concurrent flock-protected writers ─────────────────────────


def _writer_worker(path_str: str, trigger: str, cycle_offset: int, n: int) -> None:
    """Runs in a child process; appends N events as fast as possible."""
    # Clean import in the subprocess
    from src.scanner.automation.reflection_events import ReflectionEventsWriter

    writer = ReflectionEventsWriter(path=Path(path_str))
    for i in range(n):
        writer.append(
            trigger=trigger,
            cycle=cycle_offset + i,
            mode="DRY_RUN",
            null_diag_reason="diag_is_none",
            context={"i": i, "pid": os.getpid()},
        )


def test_concurrent_flock_event_writers(_isolate_env: Path) -> None:
    """Brief test #5: two processes append concurrently; every line must
    be intact JSON and total count must be exact. flock + atomic append
    is what guarantees this; the test is a real multi-process race.
    """
    path = _isolate_env

    n_per = 25
    p1 = mp.Process(target=_writer_worker, args=(str(path), "periodic", 0, n_per))
    p2 = mp.Process(target=_writer_worker, args=(str(path), "rejection", 1000, n_per))

    p1.start()
    p2.start()
    p1.join(timeout=30)
    p2.join(timeout=30)
    assert p1.exitcode == 0, f"writer 1 failed (exit={p1.exitcode})"
    assert p2.exitcode == 0, f"writer 2 failed (exit={p2.exitcode})"

    events = _read_events(path)
    assert len(events) == 2 * n_per, (
        f"expected {2 * n_per} events, got {len(events)}: races dropped lines"
    )

    # Each line must be a complete JSON object — flock prevented torn writes
    triggers = [e["trigger"] for e in events]
    assert triggers.count("periodic") == n_per
    assert triggers.count("rejection") == n_per

    # Every entry has the required schema fields
    for ev in events:
        assert "ts_utc" in ev
        assert "trigger" in ev
        assert "cycle" in ev
        assert "mode" in ev
        assert "null_diag_reason" in ev
        assert "context" in ev
        assert isinstance(ev["context"], dict)


# ── Test 6: alarm fires at 100/24h threshold ────────────────────────────


def test_event_alarm_at_100_per_24h(_isolate_env: Path) -> None:
    """Brief test #6: pre-seed the JSONL with 100 events timestamped in
    the last 24h, fire one more, assert WARN-class alarm fired.

    structlog isn't routed to stdlib logging in this codebase by default,
    so capturing via `caplog` is unreliable. Instead we verify the alarm
    fired by inspecting the writer's real `_alarmed_in_window` latch
    (a real attribute on a real instance — not a mock).
    """
    from src.scanner.automation.reflection_events import (
        ALARM_THRESHOLD_PER_24H,
        ReflectionEventsWriter,
    )

    path = _isolate_env
    path.parent.mkdir(parents=True, exist_ok=True)

    # Seed exactly the threshold count, all within the last 24h.
    now = datetime.now(timezone.utc)
    seeded: List[str] = []
    for i in range(ALARM_THRESHOLD_PER_24H):
        ts = now - timedelta(minutes=i + 1)
        seeded.append(
            json.dumps(
                {
                    "ts_utc": ts.isoformat(),
                    "trigger": "periodic",
                    "cycle": 100 + i,
                    "mode": "DRY_RUN",
                    "null_diag_reason": "diag_is_none",
                    "context": {},
                },
                sort_keys=True,
                separators=(",", ":"),
            )
        )
    path.write_text("\n".join(seeded) + "\n")

    writer = ReflectionEventsWriter(
        path=path, alarm_threshold_per_24h=ALARM_THRESHOLD_PER_24H
    )
    assert writer._alarmed_in_window is False, "alarm latch should start unset"

    # Fire one more — total now exceeds threshold
    writer.append(
        trigger="periodic",
        cycle=999,
        mode="DRY_RUN",
        null_diag_reason="diag_is_none",
        context={},
    )

    assert writer._alarmed_in_window is True, (
        "alarm must fire once total > threshold within 24h"
    )

    # Firing again should NOT re-alarm (latched until window quiets)
    writer.append(
        trigger="periodic",
        cycle=1000,
        mode="DRY_RUN",
        null_diag_reason="diag_is_none",
        context={},
    )
    assert writer._alarmed_in_window is True, "latch must persist within window"


# ── Test 7 (defensive): periodic WITH populated diag goes to heavy path ─


def test_periodic_with_recommended_actions_DOES_route(
    monkeypatch, _isolate_env: Path
) -> None:
    """Backwards-compatibility guard: if a future caller pre-populates
    `_pending_diag` with a non-empty `recommended_actions` BEFORE firing
    periodic, the heavy path engages and NO null-diag event is written.
    This is what makes the change non-destructive for callers that learn
    to provide structured input later.
    """
    monkeypatch.setenv("BUDDY_REFLECTION_DRY_RUN", "1")

    t = _make_triggers(periodic_every_n=1)
    t._pending_diag = {
        "status": "WARN",
        "recommended_actions": ["raise:min_confidence"],
    }

    short_circuited = t._fire_periodic(
        scan_count=10,
        scan_result=SimpleNamespace(tradeable=[], analyses=[]),
        trades_executed=0,
    )
    assert short_circuited is False, "populated diag must engage heavy path"

    events = _read_events(_isolate_env)
    assert events == [], "heavy-path fires must not emit null-diag events"

    assert _heavy_path_engaged(t._capture, "periodic")


# ── Test 8 (concurrent threads + JSON validity guard) ──────────────────


def test_concurrent_thread_writers_produce_valid_json(_isolate_env: Path) -> None:
    """Lighter-weight companion to test 5 — two threads in the same
    process. Catches lock-skip regressions where flock works between
    processes but the writer doesn't take a thread-level lock. Threads
    share the same FD-table so the test exercises the OS lock semantics
    directly.
    """
    from src.scanner.automation.reflection_events import ReflectionEventsWriter

    path = _isolate_env
    writer = ReflectionEventsWriter(path=path)

    n_per_thread = 50
    errors: List[Exception] = []

    def worker(trigger: str, base_cycle: int) -> None:
        try:
            for i in range(n_per_thread):
                writer.append(
                    trigger=trigger,
                    cycle=base_cycle + i,
                    mode="DRY_RUN",
                    null_diag_reason="diag_is_none",
                    context={"i": i, "thread": threading.current_thread().name},
                )
        except Exception as exc:  # pragma: no cover
            errors.append(exc)

    t1 = threading.Thread(target=worker, args=("periodic", 0), name="W1")
    t2 = threading.Thread(target=worker, args=("rejection", 5000), name="W2")
    t1.start()
    t2.start()
    t1.join(timeout=30)
    t2.join(timeout=30)

    assert not errors, f"thread writers raised: {errors}"

    # All 100 lines parse as JSON
    events = _read_events(path)
    assert len(events) == 2 * n_per_thread

    # No torn lines (each non-empty line is valid JSON)
    raw_lines = [
        line for line in path.read_text().splitlines() if line.strip()
    ]
    for line in raw_lines:
        json.loads(line)  # raises if torn
