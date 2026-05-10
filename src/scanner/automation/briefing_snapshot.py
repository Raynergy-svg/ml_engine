"""Deterministic briefing snapshot writer (Angle 1').

Emits `.claude/brain/snapshot.md` — a per-cycle deterministic mirror of the
runtime state pulled from on-disk surfaces. **NO LLM**, **NO mutation of
human-curated brain files** (`briefing.md`, `learnings.md`, `strategic_log.md`,
`open_questions.md`, `trade_narrative.md` are NEVER touched).

Why a separate file:
- `briefing.md` is operator-owned narrative. Per-cycle programmatic rewrites
  would erase context the operator hand-curated.
- A separate `snapshot.md` lets the operator (and Claude) cross-reference
  ground truth without touching the narrative file.

Source surfaces (all optional, fail-closed):
- `.claude/state.json` — halted, mode, scan_cycle_count, safe_restart
- `.claude/heartbeat.json` — pid, ts_iso, scanner_alive, cycle_count
- `.claude/alert_state.json` — last_fired alerts, active_alerts
- `.claude/meta/changes/*.json` — last 5 ChangePackages by mtime
- `trained_data/trade_journal_rl.json` — last 24h closed trades
- `model_freshness.get_model_freshness_for_pairs()` — per-pair model age
- `trained_data/virtual_trades.jsonl` — gate-rejected setups (line count)
- `git rev-parse HEAD` — short hash

Atomic-write contract:
- Every write goes to a temp file in the same dir, then `os.replace()`.
- A heartbeat sidecar (`.claude/snapshot_heartbeat.json`) records every
  attempt + last error so the operator can see if the writer itself died.
- Diff-and-write semantics: identical *meaningful* state (modulo monotonic
  counters in `_DIFF_IGNORE_KEYS`) skips disk replace to avoid mtime churn.
"""
from __future__ import annotations

import json
import logging
import os
import subprocess
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


# Default project root. Tests pass `project_root=tmp_path`; the live runtime
# uses the cwd which is always the repo root in our launchers.
_DEFAULT_PROJECT_ROOT = Path(".")

# Keys that monotonically tick every cycle and should NOT trigger a write
# when they are the only thing that changed (would churn the file every run).
_DIFF_IGNORE_KEYS = frozenset({
    "ts_utc",
    "ts_local",
    "git_head",  # changes only on commit, but cheap to ignore in diff
    "scan_count",
    "heartbeat_cycle_count",
    "heartbeat_ts_iso",
    "trigger",  # the trigger label itself shouldn't gate the write
})


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _atomic_write_text(target: Path, content: str) -> None:
    """Atomic write: tempfile in same dir + os.replace.

    Raises on failure (caller decides how to log/heartbeat).
    """
    target.parent.mkdir(parents=True, exist_ok=True)
    # tempfile in the same dir guarantees os.replace is atomic on POSIX/NTFS.
    fd, tmp_path = tempfile.mkstemp(
        prefix=f".{target.name}.",
        suffix=".tmp",
        dir=str(target.parent),
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(content)
            fh.flush()
            try:
                os.fsync(fh.fileno())
            except OSError:
                # fsync not supported on all FS (tmpfs); the rename is still atomic.
                pass
        os.replace(tmp_path, target)
    except Exception:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise


def _safe_read_json(path: Path) -> Optional[Any]:
    """Read JSON, return None on any error. Logs WARN on corruption."""
    try:
        if not path.exists():
            return None
    except OSError:
        return None
    try:
        with path.open("r", encoding="utf-8") as fh:
            return json.load(fh)
    except (json.JSONDecodeError, ValueError) as exc:
        logger.warning("briefing_snapshot: corrupt JSON at %s: %s", path, exc)
        return None
    except OSError as exc:
        logger.warning("briefing_snapshot: cannot read %s: %s", path, exc)
        return None


def _git_head_short(project_root: Path) -> str:
    """Best-effort short HEAD via `git rev-parse --short HEAD`."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=str(project_root),
            capture_output=True,
            text=True,
            timeout=2,
        )
        if result.returncode == 0:
            return (result.stdout or "").strip() or "unknown"
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError) as exc:
        logger.warning("briefing_snapshot: git rev-parse failed: %s", exc)
    return "unknown"


def _format_dt(dt_str: Any) -> str:
    """Best-effort ISO -> short rendering. Returns the input on failure."""
    if not dt_str:
        return "unknown"
    try:
        s = str(dt_str).replace("Z", "+00:00")
        dt = datetime.fromisoformat(s)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc).strftime("%Y-%m-%d %H:%M:%SZ")
    except (ValueError, TypeError):
        return str(dt_str)


def _parse_ts(value: Any) -> Optional[datetime]:
    if not value:
        return None
    try:
        s = str(value).replace("Z", "+00:00")
        dt = datetime.fromisoformat(s)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt
    except (ValueError, TypeError):
        return None


class BriefingSnapshotWriter:
    """Deterministic snapshot.md writer. NEVER touches briefing.md.

    Usage:
        writer = BriefingSnapshotWriter(project_root="/path/to/repo")
        writer.write_now(trigger="boot")               # immediate
        writer.maybe_write(cycle_count, every_n=12)    # cadence-gated
    """

    SCHEMA_VERSION = "1"

    def __init__(
        self,
        project_root: Path | str | None = None,
    ) -> None:
        self._root = Path(project_root) if project_root else _DEFAULT_PROJECT_ROOT
        self._snapshot_path = self._root / ".claude" / "brain" / "snapshot.md"
        self._heartbeat_path = self._root / ".claude" / "snapshot_heartbeat.json"
        # Cache for diff detection: last successfully written meaningful state.
        self._last_meaningful_state: Optional[Dict[str, Any]] = None

    # ── Public API ────────────────────────────────────────────────────

    def maybe_write(self, cycle_count: int, every_n: int = 12) -> bool:
        """Write the snapshot if `cycle_count % every_n == 0`. Idempotent."""
        if every_n <= 0:
            return False
        if cycle_count % every_n != 0:
            return False
        return self.write_now(trigger=f"cycle {cycle_count}")

    def write_now(self, trigger: str = "manual") -> bool:
        """Compute + write the snapshot. Returns True on success.

        Diff-and-write: if the meaningful state matches the last successful
        write, skips the disk replace (avoids mtime churn).
        """
        attempt_ts = _utcnow().isoformat()
        try:
            state = self._collect_state(trigger=trigger)
            content = self._render_markdown(state)
            meaningful = self._meaningful_subset(state)

            if self._last_meaningful_state is not None and meaningful == self._last_meaningful_state:
                # Nothing meaningful changed; record heartbeat success but skip rewrite.
                self._write_heartbeat(
                    last_attempt_utc=attempt_ts,
                    last_success_utc=attempt_ts,
                    last_error=None,
                    skipped="no_meaningful_change",
                )
                return True

            _atomic_write_text(self._snapshot_path, content)
            self._last_meaningful_state = meaningful
            self._write_heartbeat(
                last_attempt_utc=attempt_ts,
                last_success_utc=attempt_ts,
                last_error=None,
            )
            return True

        except Exception as exc:
            logger.error(
                "briefing_snapshot: write failed (trigger=%s): %s",
                trigger,
                exc,
                exc_info=True,
            )
            self._write_heartbeat(
                last_attempt_utc=attempt_ts,
                last_success_utc=None,
                last_error=f"{type(exc).__name__}: {exc}",
            )
            return False

    # ── State collection (fail-closed per source) ─────────────────────

    def _collect_state(self, *, trigger: str) -> Dict[str, Any]:
        """Read every source surface; each is wrapped in try/except via the
        helper functions, so any single failure produces a placeholder
        instead of crashing the writer."""
        now = _utcnow()
        state_json = _safe_read_json(self._root / ".claude" / "state.json") or {}
        heartbeat = _safe_read_json(self._root / ".claude" / "heartbeat.json") or {}
        alert_state = _safe_read_json(self._root / ".claude" / "alert_state.json") or {}

        return {
            "trigger": trigger,
            "ts_utc": now.isoformat(),
            "ts_local": now.astimezone().strftime("%Y-%m-%d %H:%M:%S %Z"),
            "git_head": _git_head_short(self._root),
            "schema_version": self.SCHEMA_VERSION,
            "halted": bool(state_json.get("halted", False)),
            "mode": state_json.get("mode") or "unknown",
            "scan_count": int(state_json.get("scan_cycle_count") or 0),
            "safe_restart": state_json.get("safe_restart"),
            "heartbeat_pid": heartbeat.get("pid"),
            "heartbeat_ts_iso": heartbeat.get("ts_iso"),
            "heartbeat_scanner_alive": heartbeat.get("scanner_alive"),
            "heartbeat_cycle_count": heartbeat.get("cycle_count"),
            "alerts": self._collect_alerts(alert_state),
            "halt_reason": self._derive_halt_reason(state_json, alert_state),
            "recent_trades_24h": self._collect_recent_trades(now),
            "recent_changes": self._collect_recent_changes(),
            "model_freshness": self._collect_model_freshness(),
            "virtual_trades_lines": self._count_virtual_trades(),
        }

    @staticmethod
    def _collect_alerts(alert_state: Dict[str, Any]) -> Dict[str, Any]:
        active = alert_state.get("active_alerts") or []
        last_fired = alert_state.get("last_fired") or {}
        types: List[str] = []
        for entry in active:
            if isinstance(entry, dict):
                t = entry.get("type") or entry.get("kind") or entry.get("name")
                if t:
                    types.append(str(t))
            elif isinstance(entry, str):
                types.append(entry)
        return {
            "active_count": len(active),
            "active_types": types,
            "last_fired_keys": sorted(last_fired.keys()) if isinstance(last_fired, dict) else [],
        }

    def _derive_halt_reason(
        self,
        state_json: Dict[str, Any],
        alert_state: Dict[str, Any],
    ) -> Optional[str]:
        """Pull a halt reason from latest alert or last meta change kind."""
        if not state_json.get("halted"):
            return None
        # Prefer active alert types
        active = alert_state.get("active_alerts") or []
        for entry in active:
            if isinstance(entry, dict):
                t = entry.get("type") or entry.get("kind") or entry.get("name")
                if t:
                    return str(t)
            elif isinstance(entry, str):
                return entry
        # Fall back to most recent meta change kind
        recent = self._collect_recent_changes(limit=1)
        if recent:
            kind = recent[0].get("kind")
            if kind:
                return f"meta:{kind}"
        return "unknown"

    def _collect_recent_trades(self, now: datetime) -> List[Dict[str, Any]]:
        """Return trades closed in the last 24h. Best-effort schema."""
        path = self._root / "trained_data" / "trade_journal_rl.json"
        data = _safe_read_json(path)
        if not isinstance(data, list):
            return []
        cutoff = now - timedelta(hours=24)
        recent: List[Dict[str, Any]] = []
        for entry in data:
            if not isinstance(entry, dict):
                continue
            # Filter to closed trades only (best effort)
            outcome = entry.get("outcome")
            close_ts_raw = (
                entry.get("close_time")
                or entry.get("closed_at")
                or entry.get("exit_time")
                or entry.get("timestamp")
            )
            close_ts = _parse_ts(close_ts_raw)
            if close_ts is None or close_ts < cutoff:
                continue
            recent.append({
                "pair": entry.get("pair") or "?",
                "direction": entry.get("direction") or "?",
                "pnl_pips": entry.get("pnl_pips") or entry.get("pnl_R") or entry.get("pnl") or 0.0,
                "confidence": entry.get("confidence") or 0.0,
                "exit_reason": entry.get("exit_reason") or outcome or "?",
                "close_ts": close_ts.isoformat(),
            })
        # Sort newest first, cap at 10 to keep snapshot bounded.
        recent.sort(key=lambda r: r.get("close_ts", ""), reverse=True)
        return recent[:10]

    def _collect_recent_changes(self, limit: int = 5) -> List[Dict[str, Any]]:
        """Read last `limit` ChangePackages from .claude/meta/changes/, sorted by mtime desc."""
        changes_dir = self._root / ".claude" / "meta" / "changes"
        try:
            files = [
                p for p in changes_dir.iterdir()
                if p.is_file() and p.suffix == ".json"
            ]
        except (OSError, FileNotFoundError):
            return []
        # Sort by mtime descending. Wrap in try/except since stat() can fail mid-iteration.
        try:
            files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        except OSError:
            pass
        results: List[Dict[str, Any]] = []
        for p in files[:limit]:
            data = _safe_read_json(p)
            if not isinstance(data, dict):
                continue
            results.append({
                "change_id": data.get("change_id") or p.stem,
                "kind": (data.get("incident") or {}).get("kind") or data.get("kind") or "?",
                "stage": data.get("stage") or "?",
                "updated_at": data.get("updated_at") or data.get("created_at") or "?",
            })
        return results

    def _collect_model_freshness(self) -> Optional[Dict[str, Any]]:
        """Best-effort wrapper around model_freshness.

        Prefers `get_model_freshness_for_pairs` (Tier-7 per-pair scoping)
        when available; falls back to `get_model_freshness` on older
        branches that don't have the per-pair variant. Either failure
        mode is non-fatal — a missing freshness section is preferable
        to a snapshot write that crashes the cycle.
        """
        models_dir = self._root / "trained_data" / "models"
        try:
            try:
                from src.scanner.automation.model_freshness import (
                    get_model_freshness_for_pairs,
                )
                return get_model_freshness_for_pairs(models_dir=models_dir)
            except ImportError:
                from src.scanner.automation.model_freshness import (
                    get_model_freshness,
                )
                return get_model_freshness(models_dir=models_dir)
        except Exception as exc:  # pragma: no cover — defensive belt
            logger.warning("briefing_snapshot: model_freshness failed: %s", exc)
            return None

    def _count_virtual_trades(self) -> int:
        path = self._root / "trained_data" / "virtual_trades.jsonl"
        try:
            if not path.exists():
                return 0
            with path.open("r", encoding="utf-8") as fh:
                return sum(1 for _ in fh)
        except OSError as exc:
            logger.warning("briefing_snapshot: virtual_trades count failed: %s", exc)
            return 0

    # ── Diff detection ────────────────────────────────────────────────

    @staticmethod
    def _meaningful_subset(state: Dict[str, Any]) -> Dict[str, Any]:
        """Return state minus monotonic-counter / cosmetic keys."""
        return {k: v for k, v in state.items() if k not in _DIFF_IGNORE_KEYS}

    # ── Markdown rendering ────────────────────────────────────────────

    def _render_markdown(self, state: Dict[str, Any]) -> str:
        """Render the full snapshot.md. Pure f-string templating, no jinja."""
        lines: List[str] = []
        # 1. Header
        halted = state["halted"]
        lines.append(
            f"# Buddy Snapshot — {state['ts_utc']} (UTC) / {state['ts_local']}"
        )
        lines.append("")
        lines.append(f"- HEAD: `{state['git_head']}`")
        lines.append(f"- halted: `{halted}`")
        lines.append(f"- last-write-trigger: `{state['trigger']}`")
        lines.append(f"- schema: `v{state['schema_version']}`")
        lines.append("")

        # 2. Halt context
        lines.append("## Halt context")
        if halted:
            reason = state.get("halt_reason") or "unknown"
            since = "unknown"
            sr = state.get("safe_restart")
            if isinstance(sr, dict):
                since = _format_dt(sr.get("at"))
            lines.append(f"- HALTED — reason: `{reason}` — since: `{since}`")
        else:
            lines.append(f"- running (mode=`{state.get('mode')}`)")
        lines.append("")

        # 3. Recent trades 24h
        lines.append("## Recent trade outcomes (24h)")
        trades = state.get("recent_trades_24h") or []
        if not trades:
            lines.append("- no closed trades in last 24h")
        else:
            lines.append("")
            lines.append("| pair | dir | pnl_pips | conf | exit_reason |")
            lines.append("|---|---|---|---|---|")
            for t in trades:
                lines.append(
                    f"| {t['pair']} | {t['direction']} | {t['pnl_pips']} "
                    f"| {t['confidence']} | {t['exit_reason']} |"
                )
        lines.append("")

        # 4. Active alerts
        lines.append("## Active alerts")
        alerts = state.get("alerts") or {}
        active_count = alerts.get("active_count", 0)
        types = alerts.get("active_types") or []
        lines.append(f"- active count: `{active_count}`")
        if types:
            lines.append(f"- types: {', '.join(f'`{t}`' for t in types)}")
        last_fired_keys = alerts.get("last_fired_keys") or []
        if last_fired_keys:
            lines.append(
                f"- last_fired (any time): {', '.join(f'`{k}`' for k in last_fired_keys)}"
            )
        lines.append("")

        # 5. Recent ChangePackages
        lines.append("## Recent ChangePackages")
        changes = state.get("recent_changes") or []
        if not changes:
            lines.append("- no ChangePackages on disk")
        else:
            lines.append("")
            lines.append("| change_id | kind | stage | updated_at |")
            lines.append("|---|---|---|---|")
            for c in changes:
                lines.append(
                    f"| `{c['change_id']}` | {c['kind']} | {c['stage']} "
                    f"| {_format_dt(c['updated_at'])} |"
                )
        lines.append("")

        # 6. Model freshness
        lines.append("## Model freshness")
        freshness = state.get("model_freshness")
        if not freshness:
            lines.append("- model_freshness unavailable (no data yet)")
        else:
            lines.append(
                f"- status: `{freshness.get('status', 'UNKNOWN')}` "
                f"— oldest_age_days: `{freshness.get('oldest_age_days')}`"
            )
            stale = freshness.get("stale_models") or []
            if stale:
                lines.append(f"- stale: {', '.join(stale)}")
            # Per-pair table
            per_pair: List[Dict[str, Any]] = []
            for g in freshness.get("groups", []):
                if g.get("name") == "per_pair_models":
                    per_pair = g.get("per_pair_ages") or []
                    break
            if per_pair:
                lines.append("")
                lines.append("| pair | trained_at | age_days | stale (>7d) |")
                lines.append("|---|---|---|---|")
                for entry in per_pair:
                    age = entry.get("age_days")
                    stale_mark = "yes" if (age is not None and age > 7) else "no"
                    lines.append(
                        f"| {entry.get('pair', '?')} | "
                        f"{_format_dt(entry.get('trained_at'))} | "
                        f"{age} | {stale_mark} |"
                    )
        lines.append("")

        # 7. Scan stats
        lines.append("## Scan stats")
        lines.append(f"- last cycle ts (heartbeat): `{state.get('heartbeat_ts_iso') or 'unknown'}`")
        lines.append(f"- scan_count (state.json): `{state.get('scan_count', 0)}`")
        lines.append(
            f"- heartbeat cycle_count: `{state.get('heartbeat_cycle_count')}` "
            f"(pid=`{state.get('heartbeat_pid')}`, alive=`{state.get('heartbeat_scanner_alive')}`)"
        )
        lines.append(f"- virtual_trades.jsonl lines: `{state.get('virtual_trades_lines', 0)}`")
        lines.append("")

        # 8. Footer
        lines.append(
            "<!-- generated by BriefingSnapshotWriter; this file is gitignored; "
            "briefing.md is human-owned -->"
        )
        lines.append("")
        return "\n".join(lines)

    # ── Heartbeat sidecar ─────────────────────────────────────────────

    def _write_heartbeat(
        self,
        *,
        last_attempt_utc: str,
        last_success_utc: Optional[str],
        last_error: Optional[str],
        skipped: Optional[str] = None,
    ) -> None:
        """Persist heartbeat sidecar. Best-effort — never raises."""
        payload: Dict[str, Any] = {
            "last_attempt_utc": last_attempt_utc,
            "last_success_utc": last_success_utc,
            "last_error": last_error,
        }
        if skipped:
            payload["skipped"] = skipped
        try:
            content = json.dumps(payload, indent=2, sort_keys=True)
            _atomic_write_text(self._heartbeat_path, content)
        except Exception as exc:
            # Never crash on heartbeat write — log and continue.
            logger.warning("briefing_snapshot: heartbeat write failed: %s", exc)


__all__ = ["BriefingSnapshotWriter"]
