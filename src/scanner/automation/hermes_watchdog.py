"""Hermes watchdog — 30-min in-process anomaly detector.

Subprocess entry point fired by ScheduledJobsRegistry. Reads scanner
state files, decides watch/silent/no-op, appends to digest, persists
dedup state.

Buddy policies honored (per CLAUDE.md + .claude/rules/improvement.md):
- Claude-free: no LLM, no external network calls.
- JSON reads wrapped in try/except, graceful fallback to defaults.
- Atomic writes via HermesDigest (markdown) + HermesState (JSON).
- Specific exception types (OSError, json.JSONDecodeError) — no bare except.
- Watchdog NEVER alerts on its own job_id (recursive-failure guard).
- Dedup: same alert_type or job_id within 30 min  no re-write.
- Silent rate-limit: max one silent entry per 4h.
"""
from __future__ import annotations

import logging
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

from src.scanner.automation.brain_caps import caps as brain_caps
from src.scanner.automation.hermes_digest import DigestEntry, HermesDigest
from src.scanner.automation.hermes_state import HermesState
from src.scanner.automation.safe_json import safe_json_read


logger = logging.getLogger(__name__)


_OWN_JOB_ID = "hermes_watchdog"
_HEARTBEAT_STALE_THRESHOLD_SEC = 60.0
_SILENT_RATE_LIMIT_HOURS = 4.0
_ALERT_DEDUP_WINDOW_MIN = 30.0
_JOB_FAILURE_DEDUP_WINDOW_MIN = 30.0


@dataclass(frozen=True)
class WatchdogContext:
    repo_root: Path
    now: datetime  # tz-aware UTC


@dataclass
class WatchdogResult:
    entries_written: List[str] = field(default_factory=list)  # "watch" | "silent"


@dataclass
class _Trigger:
    kind: str
    body: List[str]
    dedup_alert_key: Optional[str] = None
    dedup_job_key: Optional[str] = None


def run_once(ctx: WatchdogContext) -> WatchdogResult:
    """One tick. Returns what got written.

    Never raises — failures are logged and the next tick (30 min later)
    has a fresh chance. This is intentional: the watchdog must NOT take
    the scan loop down by itself.
    """
    try:
        return _run_once_impl(ctx)
    except Exception as e:  # noqa: BLE001 — top-level safety net only
        logger.error("hermes_watchdog crashed: %s", e, exc_info=True)
        return WatchdogResult(entries_written=[])


def _run_once_impl(ctx: WatchdogContext) -> WatchdogResult:
    state_path = ctx.repo_root / ".claude" / "hermes_state.json"
    state = HermesState.from_path(state_path)

    alert_state = _read_alert_state(ctx.repo_root)
    heartbeat = _read_heartbeat(ctx.repo_root)
    jobs_state = _read_jobs_state(ctx.repo_root)

    triggers: List[_Trigger] = []

    # 1. Active unacknowledged alerts (from AlertManager)
    for alert in alert_state.get("active_alerts", []):
        if not isinstance(alert, dict):
            continue
        if alert.get("acknowledged"):
            continue
        alert_type = str(alert.get("alert_type", "unknown"))
        if _within_dedup_window(state.last_alert_keys.get(alert_type),
                                ctx.now, _ALERT_DEDUP_WINDOW_MIN):
            continue
        message = str(alert.get("message", ""))[:200]
        severity = str(alert.get("severity", ""))
        triggers.append(_Trigger(
            kind="watch",
            body=[
                f"trigger: alert_type={alert_type}  severity={severity}",
                f"detail: {message}",
            ],
            dedup_alert_key=alert_type,
        ))

    # 2. Heartbeat staleness
    hb_age = _heartbeat_age_sec(heartbeat, ctx.now)
    if hb_age is not None and hb_age > _HEARTBEAT_STALE_THRESHOLD_SEC:
        scanner_alive = bool(heartbeat.get("scanner_alive", False))
        key = "heartbeat_stale"
        if not _within_dedup_window(state.last_alert_keys.get(key),
                                    ctx.now, _ALERT_DEDUP_WINDOW_MIN):
            triggers.append(_Trigger(
                kind="watch",
                body=[
                    f"trigger: heartbeat_stale  age={hb_age:.0f}s  scanner_alive_flag={scanner_alive}",
                    f"detail: heartbeat.ts_iso older than {int(_HEARTBEAT_STALE_THRESHOLD_SEC)}s; scanner process likely dead",
                ],
                dedup_alert_key=key,
            ))

    # 3. Job failures (T1 surface) — exclude self
    for job_id, jstate in jobs_state.items():
        if job_id == _OWN_JOB_ID:
            # Recursive-failure guard — never alert on our own job.
            continue
        if not isinstance(jstate, dict):
            continue
        if jstate.get("state") != "active":
            continue
        if jstate.get("last_status") != "failure":
            continue
        if _within_dedup_window(state.last_job_failure_keys.get(job_id),
                                ctx.now, _JOB_FAILURE_DEDUP_WINDOW_MIN):
            continue
        last_error = str(jstate.get("last_error", ""))[:200]
        last_status_at = jstate.get("last_status_at", "")
        triggers.append(_Trigger(
            kind="watch",
            body=[
                f"trigger: scheduled_jobs failure  job_id={job_id}",
                f"detail: last_status_at={last_status_at}  last_error={last_error}",
            ],
            dedup_job_key=job_id,
        ))

    # 4. Silent (if no triggers and rate-limit passed)
    if not triggers:
        last_silent = _parse_iso(state.last_silent_at_iso)
        elapsed_h = ((ctx.now - last_silent).total_seconds() / 3600.0
                     if last_silent else float("inf"))
        if elapsed_h >= _SILENT_RATE_LIMIT_HOURS:
            triggers.append(_Trigger(
                kind="silent",
                body=[
                    "no alert; logged for completeness (proof-of-life)",
                ],
            ))
            state.last_silent_at_iso = ctx.now.isoformat()

    if not triggers:
        return WatchdogResult(entries_written=[])

    # Write all triggers
    digest = _make_digest(ctx)
    iso_now = ctx.now.isoformat()
    entries_written: List[str] = []
    for t in triggers:
        digest.append(DigestEntry(at=ctx.now, kind=t.kind, body=t.body))
        entries_written.append(t.kind)
        if t.dedup_alert_key:
            state.last_alert_keys[t.dedup_alert_key] = iso_now
        if t.dedup_job_key:
            state.last_job_failure_keys[t.dedup_job_key] = iso_now

    state.save_to(state_path)
    return WatchdogResult(entries_written=entries_written)


# helpers


def _read_alert_state(root: Path) -> dict:
    p = root / ".claude" / "alert_state.json"
    raw = safe_json_read(p, default=None)
    return raw if isinstance(raw, dict) else {}


def _read_heartbeat(root: Path) -> dict:
    p = root / ".claude" / "heartbeat.json"
    raw = safe_json_read(p, default=None)
    return raw if isinstance(raw, dict) else {}


def _read_jobs_state(root: Path) -> dict:
    p = root / "trained_data" / "jobs_runtime_state.json"
    raw = safe_json_read(p, default=None)
    return raw if isinstance(raw, dict) else {}


def _heartbeat_age_sec(heartbeat: dict, now: datetime) -> Optional[float]:
    ts_iso = heartbeat.get("ts_iso")
    ts = _parse_iso(ts_iso)
    if ts is None:
        return None
    return (now - ts).total_seconds()


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


def _within_dedup_window(last_iso: Optional[str], now: datetime,
                         minutes: float) -> bool:
    last = _parse_iso(last_iso)
    if last is None:
        return False
    return (now - last).total_seconds() < minutes * 60


def _make_digest(ctx: WatchdogContext) -> HermesDigest:
    digest_path = ctx.repo_root / ".claude" / "brain" / "hermes_watchdog.md"
    archive_dir = ctx.repo_root / ".claude" / "brain" / ".archive"
    hard_cap, _warn_ratio = brain_caps().get("hermes_watchdog.md", (8000, 1.15))
    return HermesDigest(
        digest_path=digest_path,
        archive_dir=archive_dir,
        hard_cap=hard_cap,
    )


def main() -> int:
    """Subprocess entry — called by `python -m src.scanner.automation.hermes_watchdog`."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    repo_root = Path(__file__).resolve().parents[3]
    ctx = WatchdogContext(repo_root=repo_root, now=datetime.now(timezone.utc))
    result = run_once(ctx)
    if result.entries_written:
        logger.info("hermes_watchdog wrote: %s", result.entries_written)
    return 0


if __name__ == "__main__":
    sys.exit(main())
