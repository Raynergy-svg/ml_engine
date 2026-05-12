"""Hermes daily brief — fixed-schema 07:00 UTC summary.

Subprocess entry point fired by ScheduledJobsRegistry daily at 07:00 UTC.
Composes a structured brief (no LLM, deterministic priority for the
`notable` line) and appends it to .claude/brain/hermes_watchdog.md.

Buddy policies honored:
- No LLM in the loop (deterministic priority for `notable`).
- JSON reads via safe_json_read with try/except + graceful fallback.
- Atomic markdown append via HermesDigest.
- State persistence via HermesState (cycle_count_at_last_brief for delta).
- Specific exception types — no bare except.
- Trade-journal load handles list shape (current contract) gracefully;
  unexpected shapes degrade to empty results, not crashes.
"""
from __future__ import annotations

import logging
import re
import sys
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, List, Optional, Tuple

from src.scanner.automation.brain_caps import caps as brain_caps
from src.scanner.automation.hermes_digest import DigestEntry, HermesDigest
from src.scanner.automation.hermes_state import HermesState
from src.scanner.automation.safe_json import safe_json_read

logger = logging.getLogger(__name__)


_OWN_JOB_ID = "hermes_daily_brief"
_DAY_HEADER_RE = re.compile(r"^## (\d{4}-\d{2}-\d{2})\s*$", re.MULTILINE)
_WATCH_ENTRY_RE = re.compile(r"^### (\d{2}):(\d{2})Z — watch\s*$", re.MULTILINE)


@dataclass(frozen=True)
class BriefContext:
    repo_root: Path
    now: datetime  # tz-aware UTC


def run_once(ctx: BriefContext) -> None:
    """Compose + append the daily brief. Never raises."""
    try:
        _run_once_impl(ctx)
    except Exception as e:  # noqa: BLE001 — top-level safety net
        logger.error("hermes_daily_brief crashed: %s", e, exc_info=True)


def _run_once_impl(ctx: BriefContext) -> None:
    state_path = ctx.repo_root / ".claude" / "hermes_state.json"
    state = HermesState.from_path(state_path)

    inputs = _gather_inputs(ctx)
    body = _compose_body(ctx, inputs, state)

    digest = _make_digest(ctx)
    digest.append(DigestEntry(at=ctx.now, kind="brief", body=body))

    state.last_brief_at_iso = ctx.now.isoformat()
    state.cycle_count_at_last_brief = inputs.cycle_count
    state.save_to(state_path)


# inputs


@dataclass
class _Inputs:
    halted: bool
    mode: str
    cycle_count: Optional[int]
    trades_today_count: int
    trades_today_pnl: float
    active_unack_alerts: List[dict]
    failing_jobs_24h: List[Tuple[str, str]]
    model_ages: List[Tuple[str, float, str]]
    alerts_24h_count: int


def _gather_inputs(ctx: BriefContext) -> _Inputs:
    root = ctx.repo_root
    state_json = _read_json(root / ".claude" / "state.json")
    heartbeat = _read_json(root / ".claude" / "heartbeat.json")
    alert_state = _read_json(root / ".claude" / "alert_state.json")
    journal = _read_journal(root / "trained_data" / "trade_journal_rl.json")
    jobs = _read_json(root / "trained_data" / "jobs_runtime_state.json")

    halted = bool(state_json.get("halted", False)) if isinstance(state_json, dict) else False
    mode = str(state_json.get("mode", "unknown")) if isinstance(state_json, dict) else "unknown"
    cycle_count: Optional[int] = None
    if isinstance(heartbeat, dict):
        cc = heartbeat.get("cycle_count")
        if isinstance(cc, int):
            cycle_count = cc

    trades_today_count, trades_today_pnl = _trades_in_today(journal, ctx.now)
    active_unack = _active_unacknowledged_alerts(alert_state)
    failing_jobs_24h = _failing_jobs_in_last_24h(jobs, ctx.now, exclude={_OWN_JOB_ID})
    model_ages = _model_ages(root, ctx.now)
    alerts_24h_count = _count_watch_entries_in_last_24h(
        root / ".claude" / "brain" / "hermes_watchdog.md", ctx.now,
    )

    return _Inputs(
        halted=halted,
        mode=mode,
        cycle_count=cycle_count,
        trades_today_count=trades_today_count,
        trades_today_pnl=trades_today_pnl,
        active_unack_alerts=active_unack,
        failing_jobs_24h=failing_jobs_24h,
        model_ages=model_ages,
        alerts_24h_count=alerts_24h_count,
    )


# composition


def _compose_body(ctx: BriefContext, inp: _Inputs, state: HermesState) -> List[str]:
    cycles_today_str = _format_cycles_today(inp.cycle_count, state)
    pnl_str = f"P&L ${inp.trades_today_pnl:.2f}"
    model_str = _format_model_ages(inp.model_ages)
    job_counts = _job_counts_summary(ctx, inp.failing_jobs_24h)
    notable = _select_notable(inp)

    return [
        f"halted: {str(inp.halted).lower()} · mode: {inp.mode} · cycles_today: {cycles_today_str}",
        f"trades_24h: {inp.trades_today_count} trades · {pnl_str}",
        f"model ages: {model_str}",
        f"jobs: {job_counts}",
        f"alerts_24h: {inp.alerts_24h_count} watch entries",
        f"notable: {notable}",
    ]


def _select_notable(inp: _Inputs) -> str:
    """Fixed priority — first match wins."""
    has_loss_streak = any(
        a.get("alert_type") == "consecutive_losses"
        for a in inp.active_unack_alerts
    )
    if inp.halted and has_loss_streak:
        return "halted on loss streak — operator review required"
    if inp.halted:
        return "halted; operator un-halt required to resume"
    if inp.failing_jobs_24h:
        job_id, err_snippet = inp.failing_jobs_24h[0]
        return f"scheduled job {job_id} failed: {err_snippet[:80]}"
    stale = [(p, age, g) for p, age, g in inp.model_ages if age > 30.0]
    if stale:
        p, age, _g = stale[0]
        return f"model staleness — {p} is {age:.1f}d old"
    return "all systems nominal"


# helpers


def _read_json(path: Path) -> Any:
    raw = safe_json_read(path, default=None)
    return raw if raw is not None else {}


def _read_journal(path: Path) -> List[dict]:
    raw = safe_json_read(path, default=None)
    if isinstance(raw, list):
        return [r for r in raw if isinstance(r, dict)]
    if isinstance(raw, dict) and isinstance(raw.get("trades"), list):
        return [r for r in raw["trades"] if isinstance(r, dict)]
    return []


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


def _midnight_utc(now: datetime) -> datetime:
    return now.replace(hour=0, minute=0, second=0, microsecond=0)


def _trades_in_today(journal: List[dict], now: datetime) -> Tuple[int, float]:
    midnight = _midnight_utc(now)
    count, pnl_sum = 0, 0.0
    for t in journal:
        ts = _parse_iso(t.get("close_time"))
        if ts is None or ts < midnight or ts >= now:
            continue
        count += 1
        try:
            pnl_sum += float(t.get("pnl", 0.0))
        except (TypeError, ValueError):
            pass
    return count, pnl_sum


def _active_unacknowledged_alerts(alert_state: Any) -> List[dict]:
    if not isinstance(alert_state, dict):
        return []
    alerts = alert_state.get("active_alerts", [])
    if not isinstance(alerts, list):
        return []
    return [a for a in alerts if isinstance(a, dict) and not a.get("acknowledged")]


def _failing_jobs_in_last_24h(
    jobs: Any, now: datetime, *, exclude: set,
) -> List[Tuple[str, str]]:
    if not isinstance(jobs, dict):
        return []
    cutoff = now - timedelta(hours=24)
    out: List[Tuple[str, str]] = []
    for job_id, jstate in jobs.items():
        if job_id in exclude or not isinstance(jstate, dict):
            continue
        if jstate.get("state") != "active":
            continue
        if jstate.get("last_status") != "failure":
            continue
        ts = _parse_iso(jstate.get("last_status_at"))
        if ts is None or ts < cutoff:
            continue
        err = str(jstate.get("last_error", ""))[:200]
        out.append((job_id, err))
    return out


def _model_ages(root: Path, now: datetime) -> List[Tuple[str, float, str]]:
    """Walk trained_data/models/<PAIR>/transformer_direction.meta.pkl.

    Uses joblib.load (sklearn convention) — the meta files are written by
    the trainer using joblib.dump. Returns empty list on any failure;
    brief shows (unknown) in that case.
    """
    out: List[Tuple[str, float, str]] = []
    models_root = root / "trained_data" / "models"
    if not models_root.exists():
        return out
    try:
        import joblib
    except ImportError:
        return out
    for pair_dir in sorted(models_root.iterdir()):
        if not pair_dir.is_dir():
            continue
        meta_path = pair_dir / "transformer_direction.meta.pkl"
        if not meta_path.exists():
            continue
        try:
            payload = joblib.load(meta_path)
        except (OSError, EOFError, KeyError, ValueError):
            continue
        if not isinstance(payload, dict):
            continue
        trained_at = _parse_iso(payload.get("trained_at"))
        if trained_at is None:
            continue
        age_days = (now - trained_at).total_seconds() / 86400.0
        granularity = str(payload.get("granularity", ""))
        out.append((pair_dir.name, age_days, granularity))
    return out


def _format_model_ages(ages: List[Tuple[str, float, str]]) -> str:
    if not ages:
        return "(unknown — no meta sidecars found)"
    parts = [f"{p} {a:.1f}d" for p, a, _g in ages]
    granularities = sorted({g for _p, _a, g in ages if g})
    suffix = f" ({'/'.join(granularities)})" if granularities else ""
    return " · ".join(parts) + suffix


def _format_cycles_today(cycle_count: Optional[int], state: HermesState) -> str:
    if cycle_count is None:
        return "unknown"
    last = state.cycle_count_at_last_brief
    if last is None:
        return "unknown"
    delta = cycle_count - int(last)
    return f"{delta}"


def _job_counts_summary(ctx: BriefContext, failing: List[Tuple[str, str]]) -> str:
    jobs = _read_json(ctx.repo_root / "trained_data" / "jobs_runtime_state.json")
    if not isinstance(jobs, dict):
        return "0 active, 0 paused, 0 failures in last 24h"
    active = sum(1 for j in jobs.values() if isinstance(j, dict) and j.get("state") == "active")
    paused = sum(1 for j in jobs.values() if isinstance(j, dict) and j.get("state") == "paused")
    return f"{active} active, {paused} paused, {len(failing)} failures in last 24h"


def _count_watch_entries_in_last_24h(digest_path: Path, now: datetime) -> int:
    if not digest_path.exists():
        return 0
    try:
        text = digest_path.read_text(encoding="utf-8")
    except OSError:
        return 0
    cutoff = now - timedelta(hours=24)
    count = 0
    current_date: Optional[datetime] = None
    for line in text.splitlines():
        m_day = _DAY_HEADER_RE.match(line)
        if m_day:
            try:
                current_date = datetime.strptime(m_day.group(1), "%Y-%m-%d").replace(tzinfo=timezone.utc)
            except ValueError:
                current_date = None
            continue
        m_watch = _WATCH_ENTRY_RE.match(line)
        if m_watch and current_date is not None:
            hh, mm = int(m_watch.group(1)), int(m_watch.group(2))
            entry_at = current_date.replace(hour=hh, minute=mm)
            if entry_at >= cutoff:
                count += 1
    return count


def _make_digest(ctx: BriefContext) -> HermesDigest:
    digest_path = ctx.repo_root / ".claude" / "brain" / "hermes_watchdog.md"
    archive_dir = ctx.repo_root / ".claude" / "brain" / ".archive"
    hard_cap, _ = brain_caps().get("hermes_watchdog.md", (8000, 1.15))
    return HermesDigest(
        digest_path=digest_path,
        archive_dir=archive_dir,
        hard_cap=hard_cap,
    )


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    repo_root = Path(__file__).resolve().parents[3]
    ctx = BriefContext(repo_root=repo_root, now=datetime.now(timezone.utc))
    run_once(ctx)
    return 0


if __name__ == "__main__":
    sys.exit(main())
