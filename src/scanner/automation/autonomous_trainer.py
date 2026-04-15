"""Autonomous trainer — actually retrains models when they go stale.

The pipeline that already existed (`scripts/scheduled_retrain.py`) was
designed for launchd cron + an Orchestrator class that engine.py never
instantiates. Result: models hadn't been retrained autonomously in 28+
days even though every other piece of the loop existed.

This module bridges the gap. When CycleAutonomyTriggers detects model
staleness (via model_freshness), it calls `maybe_spawn_autonomous_retrain`
which directly subprocess-spawns scheduled_retrain.py — no cron, no
Orchestrator, no waiting for the 50-trade drift window.

Safety:
  - 24h cooldown between retrain spawns (no thrashing)
  - Single-flight: skip if a retrain process is already running
  - Persistent state in trained_data/.autonomous_retrain_state.json so
    cooldown survives process restarts
  - Background subprocess (non-blocking) — the scan loop continues
  - Captures PID + start time so the next cycle can poll completion
  - Emits brain-log lines and a reflection_log JSONL entry so the user
    sees it in the TUI
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import structlog

logger = structlog.get_logger(__name__)

# Default 12h cooldown — retraining costs ~10min and isn't free, but
# Mon+Fri scheduling means we may legitimately want two training sessions
# in the same week. 12h still prevents thrashing within a single day.
DEFAULT_COOLDOWN_SECONDS = 12 * 3600

# Days of the week (UTC) when scheduled training fires. Forex convention:
#   Monday — train on weekend's data + last week, before NY open
#   Friday — train mid-day, before weekend gap
# Override via env: BUDDY_TRAIN_DAYS=mon,fri or "0,4" (0=Mon)
_DEFAULT_TRAIN_DAYS = "mon,fri"


def _parse_train_days(spec: str) -> set:
    """Parse 'mon,fri' or '0,4' into a set of weekday integers (0=Mon..6=Sun)."""
    name_to_idx = {
        "mon": 0, "tue": 1, "wed": 2, "thu": 3,
        "fri": 4, "sat": 5, "sun": 6,
    }
    out: set = set()
    for tok in spec.split(","):
        tok = tok.strip().lower()
        if not tok:
            continue
        if tok in name_to_idx:
            out.add(name_to_idx[tok])
        else:
            try:
                idx = int(tok)
                if 0 <= idx <= 6:
                    out.add(idx)
            except ValueError:
                continue
    return out


SCHEDULED_TRAIN_DAYS = _parse_train_days(
    os.environ.get("BUDDY_TRAIN_DAYS", _DEFAULT_TRAIN_DAYS)
)

# State file persists cooldown across process restarts.
STATE_PATH = Path("trained_data/.autonomous_retrain_state.json")

# Reflection log shared with the rest of the autonomy loop so the TUI
# Reflection Stream panel can show retrain events in real time.
REFLECTION_LOG = Path("logs/reflection_log.jsonl")

DEFAULT_PAIRS = (
    "EUR_USD,GBP_USD,USD_JPY,USD_CHF,AUD_USD,EUR_GBP,EUR_JPY"
)


@dataclass
class RetrainState:
    last_started_at: Optional[float] = None  # epoch seconds
    last_completed_at: Optional[float] = None
    last_pid: Optional[int] = None
    last_pairs: Optional[str] = None
    last_status: Optional[str] = None  # "running" | "success" | "failed" | "timeout"
    last_reason: Optional[str] = None  # what triggered the retrain
    last_oldest_age_days: Optional[float] = None
    last_schedule_day_iso: Optional[str] = None  # "YYYY-MM-DD" — last scheduled fire

    def to_dict(self) -> Dict[str, Any]:
        return {
            "last_started_at": self.last_started_at,
            "last_completed_at": self.last_completed_at,
            "last_pid": self.last_pid,
            "last_pairs": self.last_pairs,
            "last_status": self.last_status,
            "last_reason": self.last_reason,
            "last_oldest_age_days": self.last_oldest_age_days,
            "last_schedule_day_iso": self.last_schedule_day_iso,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "RetrainState":
        return cls(
            last_started_at=d.get("last_started_at"),
            last_completed_at=d.get("last_completed_at"),
            last_pid=d.get("last_pid"),
            last_pairs=d.get("last_pairs"),
            last_status=d.get("last_status"),
            last_reason=d.get("last_reason"),
            last_oldest_age_days=d.get("last_oldest_age_days"),
            last_schedule_day_iso=d.get("last_schedule_day_iso"),
        )


def _load_state() -> RetrainState:
    try:
        if STATE_PATH.exists():
            return RetrainState.from_dict(json.loads(STATE_PATH.read_text()))
    except (json.JSONDecodeError, OSError) as e:
        logger.warning("autonomous_trainer.state_load_failed", error=str(e))
    return RetrainState()


def _save_state(state: RetrainState) -> None:
    try:
        STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
        STATE_PATH.write_text(json.dumps(state.to_dict(), indent=2, sort_keys=True))
    except OSError as e:
        logger.warning("autonomous_trainer.state_save_failed", error=str(e))


def _append_reflection_log(payload: Dict[str, Any]) -> None:
    """Mirror the format used by claude_subprocess._append_reflection_log so
    the TUI Reflection Stream panel renders these as a 'system' line."""
    try:
        REFLECTION_LOG.parent.mkdir(parents=True, exist_ok=True)
        with open(REFLECTION_LOG, "a") as f:
            f.write(json.dumps(payload) + "\n")
    except OSError as e:
        logger.warning("autonomous_trainer.reflection_log_append_failed", error=str(e))


def _is_process_running(pid: Optional[int]) -> bool:
    """Cross-platform best-effort process existence check."""
    if pid is None or pid <= 0:
        return False
    try:
        os.kill(pid, 0)
        return True
    except (OSError, ProcessLookupError):
        return False


# Singleton process handle held in module scope so we can poll completion
# across cycles without re-importing.
_active_retrain_process: Optional[subprocess.Popen] = None


def get_active_process() -> Optional[subprocess.Popen]:
    return _active_retrain_process


def is_retrain_running() -> bool:
    """True if the in-process Popen handle is alive OR the persisted PID
    from a prior session is still running."""
    global _active_retrain_process
    if _active_retrain_process is not None and _active_retrain_process.poll() is None:
        return True
    state = _load_state()
    return _is_process_running(state.last_pid) and state.last_status == "running"


def maybe_spawn_autonomous_retrain(
    freshness: Dict[str, Any],
    brain_callback: Callable[[str], None],
    pairs: Optional[str] = None,
    cooldown_seconds: int = DEFAULT_COOLDOWN_SECONDS,
    force: bool = False,
    dry_run: bool = False,
) -> Dict[str, Any]:
    """Spawn scheduled_retrain.py as a background subprocess if conditions met.

    Conditions:
      - freshness.status is CRITICAL (always retrain)
      - freshness.status is STALE and force=True
      - cooldown_seconds since last retrain has elapsed
      - no other retrain currently running

    Returns:
        {"spawned": bool, "skipped_reason": str | None, "pid": int | None,
         "pairs": str | None}

    Never raises — caller can fire-and-forget.
    """
    global _active_retrain_process
    result: Dict[str, Any] = {"spawned": False, "skipped_reason": None, "pid": None, "pairs": None}

    status = str(freshness.get("status", "UNKNOWN"))
    oldest = freshness.get("oldest_age_days")

    # Schedule check FIRST — Mon/Fri once per day, regardless of staleness.
    # Forex models genuinely need weekly retraining; the scheduled cadence
    # is the canonical trigger. Staleness fires only fill gaps when the
    # schedule was missed (process down, weekend rollover, etc.).
    state = _load_state()
    now = time.time()
    today_iso = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    weekday = datetime.now(timezone.utc).weekday()
    on_train_day = weekday in SCHEDULED_TRAIN_DAYS
    schedule_already_fired_today = state.last_schedule_day_iso == today_iso

    schedule_fire = on_train_day and not schedule_already_fired_today
    triggered_by = None  # "schedule" | "stale" | "critical" | "force"

    if force:
        triggered_by = "force"
    elif schedule_fire:
        triggered_by = "schedule"
    elif status == "CRITICAL":
        triggered_by = "critical"
    elif status == "STALE":
        # STALE now fires by default — forex models past 5d old need refresh.
        # Opt out via BUDDY_DISABLE_AUTOTRAIN_ON_STALE=1 if you want to
        # hold off until CRITICAL.
        if os.environ.get("BUDDY_DISABLE_AUTOTRAIN_ON_STALE", "").lower() in ("1", "true", "yes"):
            result["skipped_reason"] = f"status=STALE (autotrain on stale disabled)"
            return result
        triggered_by = "stale"
    else:
        result["skipped_reason"] = (
            f"status={status} weekday={weekday} (no trigger condition met)"
        )
        return result

    # Cooldown — but scheduled fires get a softer cooldown (6h) so a Mon
    # fire doesn't get blocked by a Sat manual run.
    effective_cooldown = cooldown_seconds
    if triggered_by == "schedule":
        effective_cooldown = min(cooldown_seconds, 6 * 3600)
    if state.last_started_at and not force:
        elapsed = now - float(state.last_started_at)
        if elapsed < effective_cooldown:
            remaining = effective_cooldown - elapsed
            result["skipped_reason"] = (
                f"cooldown ({remaining/3600:.1f}h remaining, trigger={triggered_by})"
            )
            brain_callback(
                f"[dim]  ▸ retrain cooldown ({remaining/3600:.1f}h remaining, trigger={triggered_by})[/]"
            )
            return result

    # Single-flight
    if is_retrain_running():
        result["skipped_reason"] = "retrain already running"
        brain_callback("[dim]  ▸ retrain already in flight, skipping spawn[/]")
        return result

    pairs_str = pairs or DEFAULT_PAIRS
    cmd = [
        sys.executable,
        str(Path("scripts/scheduled_retrain.py")),
        "--pairs",
        pairs_str,
    ]
    weekday_name = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"][weekday]
    reason = (
        f"trigger={triggered_by}, weekday={weekday_name}, "
        f"freshness={status}, oldest={oldest}d"
    )

    if dry_run:
        brain_callback(f"[yellow]  ▸ AUTONOMOUS RETRAIN dry-run: would spawn {cmd}[/]")
        result["spawned"] = False
        result["skipped_reason"] = "dry_run"
        return result

    # Verify the script actually exists before spawning
    script_path = Path("scripts/scheduled_retrain.py")
    if not script_path.exists():
        result["skipped_reason"] = f"missing script: {script_path}"
        brain_callback(f"[red]  ✗ retrain script missing: {script_path}[/]")
        return result

    # Spawn detached, capture stdout/stderr to a log file for postmortem
    log_dir = Path("logs/autonomous_retrain")
    log_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    log_path = log_dir / f"retrain_{ts}.log"

    try:
        log_file = open(log_path, "ab", buffering=0)
        proc = subprocess.Popen(
            cmd,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            cwd=str(Path.cwd()),
            env=os.environ.copy(),
        )
    except (OSError, FileNotFoundError) as e:
        result["skipped_reason"] = f"spawn_failed: {e}"
        brain_callback(f"[red]  ✗ retrain spawn failed: {e}[/]")
        return result

    _active_retrain_process = proc

    # Persist state
    state.last_started_at = now
    state.last_pid = proc.pid
    state.last_pairs = pairs_str
    state.last_status = "running"
    state.last_reason = reason
    state.last_oldest_age_days = oldest
    if triggered_by == "schedule":
        state.last_schedule_day_iso = today_iso
    _save_state(state)

    brain_callback(
        f"[green]▸ AUTONOMOUS RETRAIN STARTED — PID {proc.pid} — "
        f"pairs={pairs_str.split(',')[0]}+{len(pairs_str.split(','))-1} more — "
        f"reason: {reason}[/]"
    )
    brain_callback(f"[dim]    log: {log_path}[/]")

    # Reflection log entry so TUI Reflection Stream shows it
    _append_reflection_log({
        "ts": datetime.now(timezone.utc).isoformat(),
        "trade_id": f"AUTOTRAIN-{ts}",
        "mode": "system",
        "success": True,
        "duration_seconds": 0,
        "stdout_len": 0,
        "returncode": None,
        "promise_found": False,
        "artifacts_written": [],
        "hypothesis": f"Autonomous retrain started: {reason}",
        "cost_usd": 0.0,
        "error": None,
        "prompt_preview": f"freshness_status={status} oldest={oldest}d pairs={pairs_str}",
    })

    result.update({
        "spawned": True,
        "pid": proc.pid,
        "pairs": pairs_str,
    })
    return result


def poll_completion(brain_callback: Callable[[str], None]) -> Optional[str]:
    """Check if the active retrain finished. Updates state + emits brain
    line on completion. Returns "success" / "failed" / "running" / None.

    Call from each scan cycle to surface completion ASAP.
    """
    global _active_retrain_process

    state = _load_state()

    proc = _active_retrain_process
    if proc is None and state.last_pid:
        # State says running but we don't have a handle — process started
        # in a prior session. Check if PID is still alive.
        if not _is_process_running(state.last_pid) and state.last_status == "running":
            # Stale state — process died without updating
            state.last_status = "unknown"
            state.last_completed_at = time.time()
            _save_state(state)
            brain_callback(
                f"[yellow]  ▸ retrain PID {state.last_pid} no longer running "
                f"(status unknown — check logs/autonomous_retrain/)[/]"
            )
        return state.last_status

    if proc is None:
        return state.last_status

    rc = proc.poll()
    if rc is None:
        return "running"

    # Completed
    status = "success" if rc == 0 else "failed"
    state.last_status = status
    state.last_completed_at = time.time()
    duration = (
        state.last_completed_at - (state.last_started_at or state.last_completed_at)
    )
    _save_state(state)

    color = "green" if rc == 0 else "red"
    icon = "✓" if rc == 0 else "✗"
    brain_callback(
        f"[{color}]▸ AUTONOMOUS RETRAIN {icon} {status.upper()} "
        f"(rc={rc}, {duration/60:.1f}min, pairs={state.last_pairs})[/]"
    )

    _append_reflection_log({
        "ts": datetime.now(timezone.utc).isoformat(),
        "trade_id": f"AUTOTRAIN-DONE-{state.last_pid}",
        "mode": "system",
        "success": rc == 0,
        "duration_seconds": duration,
        "stdout_len": 0,
        "returncode": rc,
        "promise_found": False,
        "artifacts_written": [],
        "hypothesis": (
            f"Autonomous retrain completed: status={status}, "
            f"duration={duration/60:.1f}min, pairs={state.last_pairs}"
        ),
        "cost_usd": 0.0,
        "error": None if rc == 0 else f"non-zero exit {rc}",
        "prompt_preview": "",
    })

    _active_retrain_process = None
    return status


__all__ = [
    "maybe_spawn_autonomous_retrain",
    "poll_completion",
    "is_retrain_running",
    "get_active_process",
    "RetrainState",
    "DEFAULT_COOLDOWN_SECONDS",
    "DEFAULT_PAIRS",
    "STATE_PATH",
]
