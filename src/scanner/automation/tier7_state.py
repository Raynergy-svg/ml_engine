"""Tier 7 (autonomous self-healing loop) live state — READ-ONLY export for AXIOM.

Consolidates the Tier 7 control-plane state that already lives across several disk
files into one stable, dashboard-consumable snapshot. This is a DISPLAY contract:
the dashboard reads it (or calls ``build_tier7_state`` directly) to show what the
autonomous loop is doing. There is NO control path here — nothing in this module
mutates state, halts/unhalts, or touches execution.

Honest ``running`` detection (L-017: shipped/last-known != running): the loop is
``running`` only if the heartbeat is FRESH (within ``fresh_seconds``) AND its pid
is alive. A stale heartbeat or dead pid => running:false with the reason, never a
cheerful "alive" from a 4-day-old beacon.

Sources (all read-only): ``.claude/heartbeat.json`` (pid/cycle/ts), ``.claude/state.json``
(halted/mode/status/goal/next/scan_cycle_count), ``.claude/meta/changes.jsonl``
(last meta-pipeline decision/action), ``.claude/self_heal_*.json`` (self-heal budget/debounce).
"""
from __future__ import annotations

import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

DEFAULT_FRESH_SECONDS = 90
TIER7_STATE_REL = ".claude/tier7_state.json"


def _read_json(path: Path) -> Optional[Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def _pid_alive(pid: Optional[int]) -> bool:
    """True if the pid is alive. PermissionError => alive but not ours (still alive)."""
    if not pid or int(pid) <= 0:
        return False
    try:
        os.kill(int(pid), 0)
    except PermissionError:
        return True
    except (OSError, ProcessLookupError, ValueError):
        return False
    return True


def _last_meta_event(root: Path) -> Optional[Dict[str, Any]]:
    path = root / ".claude" / "meta" / "changes.jsonl"
    try:
        lines = [ln for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]
    except OSError:
        return None
    for ln in reversed(lines):
        try:
            return json.loads(ln)
        except ValueError:
            continue
    return None


def _recent_selfheal_events(root: Path, *, limit: int = 10) -> list:
    """Last ``limit`` self-heal events written by the Tier 7 supervisor (real apply()
    results — no fabricated 'healed' claims; empty if none)."""
    path = root / ".claude" / "tier7_selfheal_events.jsonl"
    try:
        lines = [ln for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]
    except OSError:
        return []
    out = []
    for ln in lines[-limit:]:
        try:
            out.append(json.loads(ln))
        except ValueError:
            continue
    return out


def build_tier7_state(project_root: Path = Path("."), *, now: Optional[datetime] = None,
                      fresh_seconds: int = DEFAULT_FRESH_SECONDS) -> Dict[str, Any]:
    """Build the consolidated, read-only Tier 7 state snapshot (honest running flag)."""
    root = Path(project_root)
    now = now or datetime.now(timezone.utc)
    hb = _read_json(root / ".claude" / "heartbeat.json") or {}
    st = _read_json(root / ".claude" / "state.json") or {}

    pid = hb.get("pid")
    ts_iso = hb.get("ts_iso")
    age_seconds: Optional[float] = None
    if ts_iso:
        try:
            hb_ts = datetime.fromisoformat(str(ts_iso))
            if hb_ts.tzinfo is None:
                hb_ts = hb_ts.replace(tzinfo=timezone.utc)
            age_seconds = (now.astimezone(timezone.utc) - hb_ts.astimezone(timezone.utc)).total_seconds()
        except (ValueError, TypeError):
            age_seconds = None

    pid_alive = _pid_alive(pid) if pid else False
    fresh = age_seconds is not None and age_seconds <= fresh_seconds
    running = bool(fresh and pid_alive)
    if running:
        reason = "heartbeat fresh + pid alive"
    elif not pid_alive:
        reason = f"pid {pid} not alive"
    elif age_seconds is None:
        reason = "no heartbeat timestamp"
    else:
        reason = f"heartbeat stale ({int(age_seconds)}s > {fresh_seconds}s)"

    meta = _last_meta_event(root) or {}
    sh_budget = _read_json(root / ".claude" / "self_heal_action_budget.json")
    sh_debounce = _read_json(root / ".claude" / "self_heal_debounce.json")
    sh_recent = _recent_selfheal_events(root, limit=10)

    return {
        "generated_at": now.astimezone(timezone.utc).isoformat(),
        "running": running,
        "running_reason": reason,
        "halted": bool(st.get("halted", False)),
        "mode": st.get("mode"),
        "status": st.get("status"),
        "goal": st.get("goal"),
        "improvement_focus": st.get("improvement_focus"),
        "scan_cycle_count": st.get("scan_cycle_count"),
        "current_action": st.get("next") or meta.get("event"),
        "last_cycle": {
            "heartbeat_ts": ts_iso,
            "age_seconds": round(age_seconds, 1) if age_seconds is not None else None,
            "cycle_count": hb.get("cycle_count"),
            "pid": pid,
            "pid_alive": pid_alive,
            "scanner_alive_beacon": hb.get("scanner_alive"),
        },
        "self_heal": {
            "action_budget": sh_budget,
            "debounce": sh_debounce,
            "recent_events": sh_recent,
        },
        "meta_last_event": {
            k: meta.get(k) for k in ("change_id", "stage", "event", "kind", "deploy_target", "updated_at")
        } if meta else None,
        # 2026-07-03 additive: the supervisor's per-tick maintenance/diagnostics
        # snapshot (alerts, adjustment backlog, lane liveness, needs_attention).
        # None when the supervisor hasn't written one yet — display-optional.
        "maintenance": _read_json(root / ".claude" / "maintenance_report.json"),
        "note": "READ-ONLY display snapshot for AXIOM. No control path; nothing here mutates state.",
    }


def write_tier7_state(project_root: Path = Path("."), *, out_rel: str = TIER7_STATE_REL,
                      now: Optional[datetime] = None) -> Dict[str, Any]:
    """Persist the snapshot atomically to ``<root>/.claude/tier7_state.json``; return it."""
    root = Path(project_root)
    snap = build_tier7_state(root, now=now)
    out = root / out_rel
    out.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(out.parent), suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(snap, fh, indent=2, sort_keys=True)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, out)
    except OSError:
        Path(tmp).unlink(missing_ok=True)
        raise
    return snap
