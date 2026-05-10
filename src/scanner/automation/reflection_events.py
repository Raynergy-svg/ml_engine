"""Operator-readable event sink for reflection triggers that have NO
structured input to act on.

Background (Angle 3' of the reflection-subsystem hardening, 2026-05-10):

`cycle_autonomy.CycleAutonomyTriggers` fires four reflection types:
`periodic` (every N cycles), `rejection` (consecutive no-execute streak),
`losing_streak` (consecutive losses), and `self_heal` (PostTradeDiagnostics
returns DEGRADED/CRITICAL). All four currently call the heavy reflection
spawn path (in this snapshot: `_invoke` → `invoke_claude_reflection`; in the
sister meta-pipeline branch: `MetaManager.route_incident`) regardless of
whether they actually have something for the surgeon/Claude to act on.

Three of the four (`periodic`, `rejection`, `losing_streak`) routinely fire
with `_pending_diag = None` or `recommended_actions = []`. Without a
structured payload there is nothing to mutate or reason about, so the
downstream pipeline produces audit-trail noise (`scorecard_failed`,
`rejected`) without actuation. Disk evidence (Agent A, 2026-05-10): 145
ChangePackages in the prior 6h on the parent ml_engine main branch, 100%
rejected. Same shape applies to wasted Claude subprocesses on this worktree.

This module is the "stop the noise" sink. When periodic/rejection/
losing_streak fires WITHOUT a populated diag, `cycle_autonomy` calls
`ReflectionEventsWriter.append(...)` here instead of routing — preserving
operator-visible signal (someone CAN inspect what would have triggered)
without spending a Claude subprocess or auto-rejected ChangePackage on it.

Self-heal is unchanged. Self-heal always has structured input.

This file does NOT route to the meta-pipeline, does NOT call Claude, does
NOT mutate config. Pure observability.

Schema (one line per fire, JSON-Lines):

    {
      "ts_utc": "2026-05-10T18:30:00+00:00",
      "trigger": "periodic" | "rejection" | "losing_streak",
      "cycle": 361,
      "mode": "DRY_RUN" | "LIVE" | "OBSERVE" | ...,
      "null_diag_reason": "diag_is_none" | "empty_recommended_actions",
      "context": {
          "rejection_streak": int | null,
          "losing_streak": int | null,
          "trades_executed": int | null,
          "tradeable_count": int | null,
          "scanned_count": int | null
      }
    }

Concurrency: writes use `fcntl.flock` (LOCK_EX) so multi-process scanner
runs (e.g. supervisor + cycle worker) cannot interleave a partial line.
Atomicity: each `append` opens the file in 'a' binary mode, takes the lock,
writes the line + newline, fsyncs, then releases. A truncated write would
leave the file with an incomplete line; the lock prevents that.

Counter alarm: the writer keeps a 24h rolling count. If it exceeds
`ALARM_THRESHOLD_PER_24H` (default 100), the next `append` logs WARN once
("periodic noise leaking through new sink"). Pure observability — no
action is taken; the operator decides what to do.
"""
from __future__ import annotations

import json
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, Optional

import structlog

try:  # POSIX file-locking. Worktree is darwin/linux only.
    import fcntl
    _HAVE_FLOCK = True
except ImportError:  # pragma: no cover - Windows fallback (no live tests)
    fcntl = None  # type: ignore[assignment]
    _HAVE_FLOCK = False

logger = structlog.get_logger(__name__)


DEFAULT_PATH = Path(".claude") / "reflection_events.jsonl"

# 100 events / 24h is generous: with the default 12-cycle periodic trigger
# at 5min/cycle, periodic alone would emit ~24/day. Add rejection +
# losing_streak and we expect 30-60 events/day in steady state. Crossing
# 100 means something is firing far more than scheduled — surface a WARN.
ALARM_THRESHOLD_PER_24H = 100


VALID_TRIGGERS = {"periodic", "rejection", "losing_streak"}
VALID_NULL_DIAG_REASONS = {"diag_is_none", "empty_recommended_actions"}


class ReflectionEventsWriter:
    """Append-only JSONL writer for null-diag reflection events.

    One instance per process is fine; the file lock makes concurrent
    appenders (across threads or processes) safe. Construction is cheap;
    the file is opened lazily on each append (rather than holding an FD)
    so log rotation tooling that moves the file works.
    """

    def __init__(
        self,
        path: Optional[Path] = None,
        alarm_threshold_per_24h: int = ALARM_THRESHOLD_PER_24H,
    ) -> None:
        self._path = Path(path) if path is not None else DEFAULT_PATH
        self._alarm_threshold = int(alarm_threshold_per_24h)
        # We've already alarmed once — do not spam the logs every append.
        self._alarmed_in_window: bool = False
        self._alarm_window_start: Optional[datetime] = None

    @property
    def path(self) -> Path:
        return self._path

    # ── public API ─────────────────────────────────────────────────────

    def append(
        self,
        *,
        trigger: str,
        cycle: int,
        mode: str,
        null_diag_reason: str,
        context: Optional[Dict[str, Any]] = None,
        ts_utc: Optional[datetime] = None,
    ) -> Dict[str, Any]:
        """Append one event to the JSONL sink.

        Returns the event dict that was written (so callers can log /
        inspect / pass to the brain feed).

        Raises:
            ValueError: trigger or null_diag_reason not in the allowed set.
            OSError: if the file cannot be opened or locked. The caller
                should NOT swallow this — a write failure here is a
                degradation in operator-visible signal and must surface.
        """
        if trigger not in VALID_TRIGGERS:
            raise ValueError(
                f"trigger must be one of {sorted(VALID_TRIGGERS)}, got {trigger!r}"
            )
        if null_diag_reason not in VALID_NULL_DIAG_REASONS:
            raise ValueError(
                f"null_diag_reason must be one of {sorted(VALID_NULL_DIAG_REASONS)},"
                f" got {null_diag_reason!r}"
            )

        ts = ts_utc if ts_utc is not None else datetime.now(timezone.utc)
        event: Dict[str, Any] = {
            "ts_utc": ts.isoformat(),
            "trigger": str(trigger),
            "cycle": int(cycle),
            "mode": str(mode),
            "null_diag_reason": str(null_diag_reason),
            "context": dict(context or {}),
        }

        line = json.dumps(event, sort_keys=True, separators=(",", ":")) + "\n"

        # Ensure parent dir exists. mkdir(parents=True, exist_ok=True) is
        # idempotent and atomic for the leaf, which is all we need.
        self._path.parent.mkdir(parents=True, exist_ok=True)

        # Open in append-binary so we can fsync; 'a' guarantees the OS
        # appends at end-of-file even with concurrent writers (POSIX).
        # The flock provides additional within-process and across-process
        # mutual exclusion, which protects the rolling-count read step too.
        with open(self._path, "ab") as fh:
            if _HAVE_FLOCK:
                fcntl.flock(fh.fileno(), fcntl.LOCK_EX)
            try:
                fh.write(line.encode("utf-8"))
                fh.flush()
                try:
                    os.fsync(fh.fileno())
                except OSError:
                    # fsync can fail on some filesystems (tmpfs, sshfs);
                    # the data is in the page cache either way and the
                    # next reader will see it.
                    pass
            finally:
                if _HAVE_FLOCK:
                    fcntl.flock(fh.fileno(), fcntl.LOCK_UN)

        # Counter alarm — pure observability. We re-read the file to count
        # within the 24h window because that is the only multi-process-safe
        # way to do it (per-instance state would miss sibling processes).
        try:
            self._maybe_alarm(now_utc=ts)
        except Exception as exc:  # pragma: no cover - never block writes
            logger.debug("reflection_events.alarm_check_failed", error=str(exc))

        return event

    # ── alarm helpers ──────────────────────────────────────────────────

    def _maybe_alarm(self, *, now_utc: datetime) -> None:
        """If the rolling 24h count crosses the threshold, log WARN once."""
        count = self._count_in_window(end=now_utc, hours=24)
        if count <= self._alarm_threshold:
            # Reset latch when the window quiets back down so a future
            # spike re-alarms.
            if (
                self._alarm_window_start is not None
                and now_utc - self._alarm_window_start > timedelta(hours=24)
            ):
                self._alarmed_in_window = False
                self._alarm_window_start = None
            return

        if self._alarmed_in_window:
            return

        logger.warning(
            "reflection_events.threshold_exceeded",
            count=count,
            threshold=self._alarm_threshold,
            window_hours=24,
            message="periodic noise leaking through new sink",
            path=str(self._path),
        )
        self._alarmed_in_window = True
        self._alarm_window_start = now_utc

    def _count_in_window(self, *, end: datetime, hours: int) -> int:
        """Count events with ts_utc within (end - hours, end]."""
        if not self._path.exists():
            return 0
        cutoff = end - timedelta(hours=hours)
        count = 0
        try:
            with open(self._path, "rb") as fh:
                if _HAVE_FLOCK:
                    fcntl.flock(fh.fileno(), fcntl.LOCK_SH)
                try:
                    for raw in fh:
                        try:
                            entry = json.loads(raw)
                        except json.JSONDecodeError:
                            continue
                        ts_str = entry.get("ts_utc")
                        if not isinstance(ts_str, str):
                            continue
                        try:
                            ts = datetime.fromisoformat(ts_str)
                        except ValueError:
                            continue
                        if ts.tzinfo is None:
                            ts = ts.replace(tzinfo=timezone.utc)
                        if cutoff < ts <= end:
                            count += 1
                finally:
                    if _HAVE_FLOCK:
                        fcntl.flock(fh.fileno(), fcntl.LOCK_UN)
        except OSError as exc:
            logger.debug("reflection_events.count_read_failed", error=str(exc))
            return 0
        return count


# Module-level singleton so cycle_autonomy doesn't need to thread an
# instance through. The default path is process-cwd-relative, matching
# the rest of the .claude/ persistence convention (state.json, etc.).
_DEFAULT_WRITER: Optional[ReflectionEventsWriter] = None


def get_default_writer() -> ReflectionEventsWriter:
    global _DEFAULT_WRITER
    if _DEFAULT_WRITER is None:
        _DEFAULT_WRITER = ReflectionEventsWriter()
    return _DEFAULT_WRITER


def reset_default_writer_for_tests(
    path: Optional[Path] = None,
    alarm_threshold_per_24h: int = ALARM_THRESHOLD_PER_24H,
) -> ReflectionEventsWriter:
    """Test-only helper: rebind the module singleton to a tmp_path-backed
    instance. Keeps production code from accidentally using a real
    `.claude/reflection_events.jsonl` during pytest runs."""
    global _DEFAULT_WRITER
    _DEFAULT_WRITER = ReflectionEventsWriter(
        path=path,
        alarm_threshold_per_24h=alarm_threshold_per_24h,
    )
    return _DEFAULT_WRITER


__all__ = [
    "ReflectionEventsWriter",
    "DEFAULT_PATH",
    "ALARM_THRESHOLD_PER_24H",
    "VALID_TRIGGERS",
    "VALID_NULL_DIAG_REASONS",
    "get_default_writer",
    "reset_default_writer_for_tests",
]
