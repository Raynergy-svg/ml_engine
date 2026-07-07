"""AXIOM Agent Runtime — operator disposition store for ESCALATION proposals.

Tracks accept/deny decisions the OPERATOR makes on proposals surfaced by the
resident loop's mind-window, keyed by the proposal's stable ``proposal_id``
(``policy.compute_proposal_id`` -- a hash of action+params, so the identical
proposal re-surfacing across cycles maps to the same disposition record
instead of asking the operator again every 5 minutes).

This module NEVER executes an action itself and NEVER decides what "accepted"
means for a given action type -- see ``dashboard/server/axiom_proposals.py``
for the accept-dispatch table that routes an accepted proposal to its
EXISTING, independently-safeguarded execution path (StateEngine.set_halted,
LiveGate.arm, control_safety.set_override, AdjustmentApprover.approve). This
module only answers: has the operator already decided on this one, and what.
"""
from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any, Dict, Optional

from src.axiom_operator.session import utc_now

REPO_ROOT = Path(__file__).resolve().parents[2]
DISPOSITIONS_PATH = REPO_ROOT / "trained_data" / "axiom" / "proposal_dispositions.json"

_VALID_STATUSES = ("accepted", "denied")


def _atomic_write(path: Path, data: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(data, fh, indent=2, sort_keys=True)
            fh.write("\n")
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def _load(path: Path) -> Dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


def get_disposition(proposal_id: str, *, path: Optional[Path] = None) -> Optional[Dict[str, Any]]:
    """None means: no operator decision recorded yet -- still pending.

    ``path`` defaults to the CURRENT value of module-level ``DISPOSITIONS_PATH``,
    resolved inside the function body (NOT as the literal default value) --
    a default bound at def-time would ignore a test's
    ``monkeypatch.setattr(proposal_store, "DISPOSITIONS_PATH", tmp_path)``."""
    resolved = Path(path) if path is not None else DISPOSITIONS_PATH
    return _load(resolved).get(proposal_id)


def record_disposition(
    proposal_id: str,
    *,
    status: str,
    actor: str,
    reason: str = "",
    detail: Optional[Dict[str, Any]] = None,
    path: Optional[Path] = None,
) -> Dict[str, Any]:
    """Atomic read-modify-write. ``status`` must be "accepted" or "denied" --
    anything else is a caller bug, not a valid disposition, and raises."""
    if status not in _VALID_STATUSES:
        raise ValueError(f"invalid disposition status {status!r}; must be one of {_VALID_STATUSES}")
    if not proposal_id:
        raise ValueError("proposal_id is required")
    if not actor:
        raise ValueError("actor is required -- every disposition must be attributable")

    resolved = Path(path) if path is not None else DISPOSITIONS_PATH
    data = _load(resolved)
    entry = {
        "proposal_id": proposal_id,
        "status": status,
        "actor": actor,
        "reason": reason,
        "detail": detail or {},
        "decided_at": utc_now(),
    }
    data[proposal_id] = entry
    _atomic_write(resolved, data)
    return entry


def list_dispositions(*, path: Optional[Path] = None) -> Dict[str, Any]:
    resolved = Path(path) if path is not None else DISPOSITIONS_PATH
    return _load(resolved)
