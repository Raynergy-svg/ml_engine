"""Durable session and ledger helpers for AXIOM operator epochs."""
from __future__ import annotations

import json
import os
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


REPO_ROOT = Path(__file__).resolve().parents[2]
AXIOM_DIR = REPO_ROOT / "trained_data" / "axiom"
SESSION_PATH = AXIOM_DIR / "operator_session.json"
DECISIONS_PATH = AXIOM_DIR / "operator_decisions.jsonl"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _atomic_write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2, sort_keys=True)
            fh.write("\n")
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, path)
    finally:
        try:
            if os.path.exists(tmp):
                os.unlink(tmp)
        except OSError:
            pass


def append_jsonl(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(payload, sort_keys=True) + "\n")


def _iter_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    if not path.exists():
        return []
    rows: List[Dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            payload = json.loads(line)
        except ValueError:
            continue
        if isinstance(payload, dict):
            rows.append(payload)
    return rows


def tail_jsonl(path: Path, limit: int = 10) -> List[Dict[str, Any]]:
    rows = list(_iter_jsonl(path))
    return list(reversed(rows[-limit:]))


def default_session(*, session_id: Optional[str] = None) -> Dict[str, Any]:
    now = utc_now()
    return {
        "session_id": session_id or f"axiom-operator-{int(time.time())}",
        "status": "idle",
        "provider": "claude_subscription",
        "epoch": 0,
        "context_budget_chars": 120000,
        "rollover_threshold": 0.8,
        "context_used_chars": 0,
        "context_used_pct": 0.0,
        "handoff_summary": "No operator epoch has run yet.",
        "open_incidents": [],
        "last_tools": [],
        "blocked_reasons": [],
        "next_action": "observe",
        "last_action": None,
        "last_reasoning": None,
        "last_error": None,
        "last_started_at": None,
        "last_completed_at": None,
        "updated_at": now,
        "api_key_refused": False,
        "cli_available": None,
        "mcp_config": "mcp.json",
        "mode": "shadow",
    }


def load_session(path: Path = SESSION_PATH) -> Dict[str, Any]:
    if not path.exists():
        return default_session()
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            return default_session()
    except (OSError, ValueError):
        return default_session()
    base = default_session(session_id=str(payload.get("session_id") or "axiom-operator"))
    base.update(payload)
    return base


def save_session(payload: Dict[str, Any], path: Path = SESSION_PATH) -> None:
    payload = dict(payload)
    payload["updated_at"] = utc_now()
    _atomic_write_json(path, payload)


def read_operator_snapshot(
    *,
    session_path: Path = SESSION_PATH,
    decisions_path: Path = DECISIONS_PATH,
    decision_limit: int = 10,
) -> Dict[str, Any]:
    def display_path(path: Path) -> str:
        try:
            return str(path.relative_to(REPO_ROOT))
        except ValueError:
            return str(path)

    session = load_session(session_path)
    decisions = tail_jsonl(decisions_path, decision_limit)
    return {
        "has_run": bool(decisions) or bool(session_path.exists()),
        "session": session,
        "last_decision": decisions[0] if decisions else None,
        "recent_decisions": decisions,
        "source": {
            "session": display_path(session_path),
            "decisions": display_path(decisions_path),
        },
    }
