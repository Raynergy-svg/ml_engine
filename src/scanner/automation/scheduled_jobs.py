"""Scheduled jobs registry (pick #5).

Lets operators schedule recurring tasks (homework batches, daily exports,
weekly audits) without code changes. Config lives in `.claude/jobs.json`
(operator-edited); runtime state lives in `trained_data/jobs_runtime_state.json`
(atomic via safe_json_write).

The registry is **Claude-free** by design: jobs run shell commands or import
Python callables, no LLM in the hot path (per CLAUDE.md trading invariants).

Scheduler is invoked from `orchestrator.run_cycle()` as a non-critical
DispatchStep — see `_build_dispatch_table()`. Job execution happens in a
daemon thread so a slow/hung job cannot block the scan loop.

Schedule grammar (deliberately minimal — no croniter dependency):
    daily_HH:MM          e.g. "daily_00:00"  (UTC)
    weekly_DOW_HH:MM     e.g. "weekly_SUN_02:00"  (DOW = MON|TUE|...|SUN)
    every_N_minutes      e.g. "every_30_minutes"  (N >= 1)
    every_N_hours        e.g. "every_6_hours"     (N >= 1)
"""
from __future__ import annotations

import logging
import re
import shlex
import subprocess
import threading
from dataclasses import dataclass, field
from datetime import datetime, time, timedelta, timezone
from pathlib import Path
from typing import Any, Optional, Union

from src.scanner.automation.safe_json import safe_json_read, safe_json_write

logger = logging.getLogger(__name__)

_REPO_ROOT = Path(__file__).resolve().parents[3]
_JOBS_CONFIG_PATH = _REPO_ROOT / ".claude" / "jobs.json"
_JOBS_STATE_PATH = _REPO_ROOT / "trained_data" / "jobs_runtime_state.json"

_DOW = ["MON", "TUE", "WED", "THU", "FRI", "SAT", "SUN"]  # Python weekday() order


# ────────────────────────────────────────────────────────────────────────
# Data classes
# ────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class JobConfig:
    """One scheduled job, loaded from `.claude/jobs.json`."""

    job_id: str
    name: str
    schedule: str
    command: str  # shell command — split via shlex, never run with shell=True
    enabled: bool = False
    description: str = ""

    @classmethod
    def from_dict(cls, raw: dict) -> "JobConfig":
        return cls(
            job_id=str(raw["job_id"]),
            name=str(raw.get("name", raw["job_id"])),
            schedule=str(raw["schedule"]),
            command=str(raw["command"]),
            enabled=bool(raw.get("enabled", False)),
            description=str(raw.get("description", "")),
        )


@dataclass
class JobRuntimeState:
    """Per-job runtime state. Persisted to trained_data/jobs_runtime_state.json."""

    last_run_at: Optional[str] = None    # ISO 8601 UTC
    last_status: str = "pending"          # "pending" | "success" | "failure"
    last_error: Optional[str] = None
    run_count: int = 0
    # T1 additions:
    state: str = "active"                 # "active" | "paused"
    next_run_at_iso: Optional[str] = None  # ISO 8601 UTC of next scheduled run
    last_status_at: Optional[str] = None   # ISO 8601 UTC of last status update

    @classmethod
    def from_dict(cls, raw: dict) -> "JobRuntimeState":
        return cls(
            last_run_at=raw.get("last_run_at"),
            last_status=str(raw.get("last_status", "pending")),
            last_error=raw.get("last_error"),
            run_count=int(raw.get("run_count", 0)),
            state=str(raw.get("state", "active")),
            next_run_at_iso=raw.get("next_run_at_iso"),
            last_status_at=raw.get("last_status_at"),
        )

    def to_dict(self) -> dict:
        return {
            "last_run_at": self.last_run_at,
            "last_status": self.last_status,
            "last_error": self.last_error,
            "run_count": self.run_count,
            "state": self.state,
            "next_run_at_iso": self.next_run_at_iso,
            "last_status_at": self.last_status_at,
        }


# ────────────────────────────────────────────────────────────────────────
# Schedule parsing
# ────────────────────────────────────────────────────────────────────────


_DAILY = re.compile(r"^daily_(\d{2}):(\d{2})$")
_WEEKLY = re.compile(r"^weekly_([A-Z]{3})_(\d{2}):(\d{2})$")
_EVERY_MIN = re.compile(r"^every_(\d+)_minutes$")
_EVERY_HR = re.compile(r"^every_(\d+)_hours$")


def compute_next_run(
    schedule: str,
    *,
    last_run: Optional[datetime] = None,
    now: Optional[datetime] = None,
) -> datetime:
    """Return the next time `schedule` should fire after `last_run` (or `now`).

    All times are UTC. Raises ValueError on unparseable schedules so config
    typos surface loudly rather than silently producing dead jobs (per
    improvement.md's "Config Validation Gates").
    """
    now = now or datetime.now(timezone.utc)
    if now.tzinfo is None:
        now = now.replace(tzinfo=timezone.utc)
    anchor = last_run if last_run is not None else now
    if anchor.tzinfo is None:
        anchor = anchor.replace(tzinfo=timezone.utc)

    m = _DAILY.match(schedule)
    if m:
        hh, mm = int(m.group(1)), int(m.group(2))
        _validate_clock(hh, mm)
        target = now.replace(hour=hh, minute=mm, second=0, microsecond=0)
        # If we've already passed today's slot — OR we ran at it — advance one day.
        if target <= anchor:
            target += timedelta(days=1)
        return target

    m = _WEEKLY.match(schedule)
    if m:
        dow_str, hh, mm = m.group(1), int(m.group(2)), int(m.group(3))
        if dow_str not in _DOW:
            raise ValueError(f"unknown day-of-week {dow_str!r}; expected one of {_DOW}")
        _validate_clock(hh, mm)
        target_dow = _DOW.index(dow_str)
        days_ahead = (target_dow - anchor.weekday()) % 7
        target = anchor.replace(hour=hh, minute=mm, second=0, microsecond=0) + timedelta(days=days_ahead)
        if target <= anchor:
            target += timedelta(days=7)
        return target

    m = _EVERY_MIN.match(schedule)
    if m:
        n = int(m.group(1))
        if n < 1:
            raise ValueError(f"every_N_minutes requires N >= 1, got {n}")
        return anchor + timedelta(minutes=n)

    m = _EVERY_HR.match(schedule)
    if m:
        n = int(m.group(1))
        if n < 1:
            raise ValueError(f"every_N_hours requires N >= 1, got {n}")
        return anchor + timedelta(hours=n)

    raise ValueError(
        f"unparseable schedule {schedule!r} — expected "
        "daily_HH:MM | weekly_DOW_HH:MM | every_N_minutes | every_N_hours"
    )


def _validate_clock(hh: int, mm: int) -> None:
    if not (0 <= hh <= 23 and 0 <= mm <= 59):
        raise ValueError(f"invalid HH:MM ({hh:02d}:{mm:02d})")


# ────────────────────────────────────────────────────────────────────────
# Registry
# ────────────────────────────────────────────────────────────────────────


def default_jobs() -> list[JobConfig]:
    """Built-in jobs surfaced when `.claude/jobs.json` is missing.

    All disabled by default — operator opts in by editing the JSON (or, in
    a future TUI iteration, toggling from the Config screen).
    """
    return [
        JobConfig(
            job_id="homework_weekly",
            name="Weekly homework batch",
            schedule="weekly_SUN_02:00",
            command="python buddy_scanner.py homework --generate-batch --last 100",
            enabled=False,
            description="Regenerate the last 100 trade homework entries every Sunday 02:00 UTC.",
        ),
    ]


def _executor_default(cmd: str, *, cwd: Path) -> subprocess.Popen:
    """Default subprocess spawner — split via shlex so we never invoke a shell."""
    return subprocess.Popen(
        shlex.split(cmd),
        cwd=str(cwd),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
    )


class ScheduledJobsRegistry:
    """Tracks job configs + runtime state; tick() spawns due jobs.

    Lifecycle:
        registry = ScheduledJobsRegistry()
        registry.load()       # read jobs.json + jobs_runtime_state.json
        registry.tick()       # call every cycle — fires due jobs
    """

    def __init__(
        self,
        *,
        config_path: Path = _JOBS_CONFIG_PATH,
        state_path: Path = _JOBS_STATE_PATH,
        executor: Any = _executor_default,
        repo_root: Path = _REPO_ROOT,
    ) -> None:
        self._config_path = Path(config_path)
        self._state_path = Path(state_path)
        self._executor = executor
        self._repo_root = Path(repo_root)
        self._jobs: dict[str, JobConfig] = {}
        self._state: dict[str, JobRuntimeState] = {}
        self._lock = threading.Lock()

    # ── persistence ────────────────────────────────────────────────────

    def load(self) -> None:
        """Read config + state from disk. Safe on missing/corrupt files."""
        with self._lock:
            self._jobs = {j.job_id: j for j in self._load_config()}
            self._state = self._load_state()
            # Ensure every job has a state row.
            for jid in self._jobs:
                self._state.setdefault(jid, JobRuntimeState())

    def _load_config(self) -> list[JobConfig]:
        if not self._config_path.exists():
            return list(default_jobs())
        raw = safe_json_read(self._config_path, default=None)
        if not isinstance(raw, dict) or not isinstance(raw.get("jobs"), list):
            logger.warning(
                "ScheduledJobs: %s has wrong shape — using defaults", self._config_path
            )
            return list(default_jobs())
        out: list[JobConfig] = []
        for entry in raw["jobs"]:
            if not isinstance(entry, dict):
                continue
            try:
                out.append(JobConfig.from_dict(entry))
            except (KeyError, ValueError) as e:
                logger.warning("ScheduledJobs: skipping malformed job: %s", e)
        return out

    def _load_state(self) -> dict[str, JobRuntimeState]:
        raw = safe_json_read(self._state_path, default=None)
        if not isinstance(raw, dict):
            return {}
        out: dict[str, JobRuntimeState] = {}
        for jid, payload in raw.items():
            if isinstance(payload, dict):
                out[str(jid)] = JobRuntimeState.from_dict(payload)
        return out

    def _persist_state(self) -> None:
        payload = {jid: s.to_dict() for jid, s in self._state.items()}
        if not safe_json_write(self._state_path, payload, sort_keys=True):
            logger.error("ScheduledJobs: failed to persist state to %s", self._state_path)

    # ── queries ────────────────────────────────────────────────────────

    def jobs(self) -> list[JobConfig]:
        with self._lock:
            return list(self._jobs.values())

    def state(self, job_id: str) -> JobRuntimeState:
        with self._lock:
            return self._state.get(job_id, JobRuntimeState())

    def due_jobs(self, *, now: Optional[datetime] = None) -> list[JobConfig]:
        """List enabled jobs whose next_run_at has passed.

        Pure function modulo (a) the snapshot of jobs/state captured under
        the lock and (b) `now` (defaults to UTC now()).
        """
        now = now or datetime.now(timezone.utc)
        out: list[JobConfig] = []
        with self._lock:
            for jid, cfg in self._jobs.items():
                if not cfg.enabled:
                    continue
                state = self._state.get(jid, JobRuntimeState())
                if state.state == "paused":
                    continue
                try:
                    last = _parse_iso(state.last_run_at)
                    nxt = compute_next_run(cfg.schedule, last_run=last, now=now)
                except ValueError as e:
                    logger.warning("ScheduledJobs[%s]: schedule error: %s", jid, e)
                    continue
                if nxt <= now:
                    out.append(cfg)
        return out

    # ── execution ──────────────────────────────────────────────────────

    def tick(self, *, now: Optional[datetime] = None) -> list[str]:
        """Fire any due jobs. Returns list of job_ids that were started.

        Each job runs in its own daemon thread; tick() returns promptly so
        the scan loop is never blocked.
        """
        now = now or datetime.now(timezone.utc)
        fired: list[str] = []
        for job in self.due_jobs(now=now):
            t = threading.Thread(
                target=self._run_job,
                args=(job, now),
                daemon=True,
                name=f"job_{job.job_id}",
            )
            t.start()
            fired.append(job.job_id)
        return fired

    def _run_job(self, job: JobConfig, started_at: datetime) -> None:
        """Run one job synchronously inside its own thread."""
        status = "success"
        error: Optional[str] = None
        try:
            proc = self._executor(job.command, cwd=self._repo_root)
            _stdout, stderr = proc.communicate()
            if proc.returncode != 0:
                status = "failure"
                snippet = (stderr or b"").decode("utf-8", errors="replace")[:500]
                error = f"exit {proc.returncode}: {snippet.strip()}"
                logger.error("ScheduledJobs[%s] failed: %s", job.job_id, error)
            else:
                logger.info("ScheduledJobs[%s] completed", job.job_id)
        except (OSError, subprocess.SubprocessError) as e:
            status = "failure"
            error = f"{type(e).__name__}: {e}"
            logger.exception("ScheduledJobs[%s] crashed", job.job_id)

        with self._lock:
            state = self._state.setdefault(job.job_id, JobRuntimeState())
            state.last_run_at = started_at.replace(microsecond=0).isoformat()
            state.last_status = status
            state.last_error = error
            state.last_status_at = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
            state.run_count += 1
            try:
                nxt = compute_next_run(job.schedule, last_run=started_at)
                state.next_run_at_iso = nxt.replace(microsecond=0).isoformat()
            except ValueError:
                state.next_run_at_iso = None
            self._persist_state()

    # ── T1: pause / resume / trigger / next_run_at persistence ─────────

    def pause_job(self, job_id: str) -> bool:
        """Mark a job paused. Returns True if changed; False if unknown id."""
        with self._lock:
            if job_id not in self._jobs:
                return False
            s = self._state.setdefault(job_id, JobRuntimeState())
            s.state = "paused"
            self._persist_state()
            return True

    def resume_job(self, job_id: str) -> bool:
        """Mark a job active. Returns True if changed; False if unknown id."""
        with self._lock:
            if job_id not in self._jobs:
                return False
            s = self._state.setdefault(job_id, JobRuntimeState())
            s.state = "active"
            self._persist_state()
            return True

    def trigger_now(self, job_id: str) -> bool:
        """Run a job immediately in a daemon thread. Returns True if started."""
        with self._lock:
            job = self._jobs.get(job_id)
        if job is None:
            return False
        now = datetime.now(timezone.utc)
        t = threading.Thread(
            target=self._run_job, args=(job, now),
            daemon=True, name=f"job_{job.job_id}_manual",
        )
        t.start()
        return True


# ────────────────────────────────────────────────────────────────────────
# Helpers
# ────────────────────────────────────────────────────────────────────────


def _parse_iso(s: Optional[str]) -> Optional[datetime]:
    if not s:
        return None
    try:
        dt = datetime.fromisoformat(s)
    except (TypeError, ValueError):
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt
