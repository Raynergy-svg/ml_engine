"""AXIOM Agent Runtime — audit trail.

Extends the EXISTING dashboard control-audit log (``dashboard.server.control_safety``,
``trained_data/axiom/control_audit.jsonl``) rather than inventing a second log file:
every agent-runtime action attempt (allowed, denied, or proposed-for-operator) is
appended to the SAME atomic, append-only JSONL file the dashboard's halt/unhalt/
flatten/arm controls already write to, tagged ``source: "agent_runtime"`` so the two
streams can be told apart while remaining one unified, operator-reviewable trail.

Reuses ``control_safety.audit()`` verbatim (stamps ``ts`` + requires ``actor``) —
this module adds no new atomic-write mechanism, per "do not reimplement".
"""
from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

from dashboard.server import control_safety as cs

SOURCE = "agent_runtime"


def _json_safe(value: Any) -> Any:
    """Recursively coerce a value into something ``json.dumps`` accepts.

    Action params/results occasionally carry non-primitive objects (e.g. a
    ``Path`` a test injects for isolation) — the audit log must never fail to
    write because of that; anything not a plain JSON type is stringified.
    """
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def record(
    *,
    action: str,
    tier: Optional[str],
    actor: str,
    allowed: bool,
    outcome: str,
    reason: str = "",
    params: Optional[Dict[str, Any]] = None,
    result: Optional[Dict[str, Any]] = None,
) -> None:
    """Append one audit line for an agent-runtime action attempt.

    ``tier`` is ``None`` only for an unclassifiable/unknown action (fail-closed
    deny) — every classified action carries its ``ActionTier.value``.
    """
    entry: Dict[str, Any] = {
        "source": SOURCE,
        "action": action,
        "tier": tier,
        "allowed": bool(allowed),
        "outcome": outcome,
        "reason": reason,
        "params": _json_safe(params or {}),
    }
    if result is not None:
        entry["result"] = _json_safe(result)
    cs.audit(entry, actor=actor)


def record_cycle(
    *,
    cycle_id: str,
    actor: str,
    cli_available: bool,
    autonomy_enabled: bool,
    reasoning_summary: str,
    proposed_count: int,
    executed_count: int,
    shadow_count: int,
    proposal_count: int,
    denied_count: int,
    verified: bool,
) -> None:
    """One cycle-level audit line for the resident loop (OBSERVE->...->LOG),
    in addition to the per-action lines ``record()`` already writes for each
    action attempted during that cycle. ``tier`` is None — a cycle isn't a
    single-tier action."""
    record(
        action="resident_loop_cycle",
        tier=None,
        actor=actor,
        allowed=True,
        outcome="cycle_complete",
        reason=reasoning_summary[:500],
        params={
            "cycle_id": cycle_id,
            "cli_available": cli_available,
            "autonomy_enabled": autonomy_enabled,
            "proposed_count": proposed_count,
            "executed_count": executed_count,
            "shadow_count": shadow_count,
            "proposal_count": proposal_count,
            "denied_count": denied_count,
            "verified": verified,
        },
    )


def read_recent(limit: int = 50) -> List[Dict[str, Any]]:
    """Read-only tail of the shared audit log, filtered to agent-runtime entries.

    Most-recent-first, mirroring ``dashboard.server.control.audit_log``'s contract.
    """
    if not cs.AUDIT_PATH.exists():
        return []
    rows: List[Dict[str, Any]] = []
    for line in cs.AUDIT_PATH.read_text(encoding="utf-8").splitlines():
        try:
            row = json.loads(line)
        except ValueError:
            continue
        if row.get("source") == SOURCE:
            rows.append(row)
    rows.reverse()
    return rows[: int(limit)]
