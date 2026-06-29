"""AXIOM Phase 2 — bounded control router. BUILT DISABLED.

Mounted by app.py ONLY when ``AXIOM_CONTROL_ENABLED`` is truthy (default OFF → these
routes do not exist → 404). Every handler: requires an explicit per-action confirm header,
runs the structural guards in ``control_safety.enforce`` BEFORE any effect, and audit-logs
the attempt (allowed AND denied). See dashboard/CONTROL_DESIGN.md.

Implementation status (scaffold): halt/unhalt are functional via the bot's StateEngine
(safe, grounded). set_gross_leverage / start_loop / stop_loop pass all guards + audit but
return 501 — their exact effect mechanism (live gross-leverage re-read; loop supervision)
is deferred to the enable step, after operator confirm, to avoid an unverified write or a
process-exec surface.
"""
from __future__ import annotations

import logging
from typing import Any, Dict

from fastapi import APIRouter, Header, HTTPException
from pydantic import BaseModel

from dashboard.server import control_safety as cs

logger = logging.getLogger("axiom.control")
router = APIRouter(prefix="/api/control", tags=["control"])


class ActionBody(BaseModel):
    params: Dict[str, Any] = {}


def _confirm(action: str, header_val: str | None) -> None:
    """Per-action explicit confirm — not a blanket session grant."""
    if header_val != action:
        raise HTTPException(status_code=403, detail="missing/incorrect X-AXIOM-Confirm header")


def _run(action: str, params: Dict[str, Any]):
    """Guard → audit → effect. Denials are audited and surfaced as 403."""
    try:
        normalized = cs.enforce(action, params)
    except cs.ControlDenied as exc:
        cs.audit({"action": action, "params": params, "allowed": False, "reason": str(exc), "result": "denied"})
        raise HTTPException(status_code=403, detail=str(exc))

    # --- effects (only reached after ALL immutables passed) ---
    # Only HALT is functional — it is fail-safe (it can only STOP trading). Every other
    # action is guarded + audited but its effect is DEFERRED (501) until the enable step:
    #   unhalt           — needs the unhalt-eligibility gate (model freshness / drawdown /
    #                       ship-gate) wired first; resuming a possibly-broken system on a
    #                       single confirm is unsafe (verifier MED-1). Stays 501 until wired.
    #   set_gross_leverage / start_loop / stop_loop — exact mechanism pending operator confirm.
    if action == "halt":
        from src.scanner.automation.state_engine import StateEngine
        StateEngine().set_halted(True)
        cs.audit({"action": action, "params": normalized, "allowed": True,
                  "reason": "guards passed", "result": "halted"})
        return {"ok": True, "action": action, "result": "halted"}

    cs.audit({"action": action, "params": normalized, "allowed": True,
              "reason": "guards passed", "result": "accepted_not_wired"})
    deferred_note = {
        "unhalt": "unhalt requires the unhalt-eligibility gate (freshness/drawdown/ship-gate) "
                  "to be wired before it can resume PRACTICE trading",
    }.get(action, "effect not wired in scaffold (pending operator confirm of mechanism)")
    raise HTTPException(status_code=501, detail=f"{action} accepted + audited — {deferred_note}")


@router.post("/halt")
def halt(body: ActionBody, x_axiom_confirm: str | None = Header(default=None)):
    _confirm("halt", x_axiom_confirm)
    return _run("halt", body.params)


@router.post("/unhalt")
def unhalt(body: ActionBody, x_axiom_confirm: str | None = Header(default=None)):
    _confirm("unhalt", x_axiom_confirm)
    return _run("unhalt", body.params)


@router.post("/set_gross_leverage")
def set_gross_leverage(body: ActionBody, x_axiom_confirm: str | None = Header(default=None)):
    _confirm("set_gross_leverage", x_axiom_confirm)
    return _run("set_gross_leverage", body.params)


@router.post("/start_loop")
def start_loop(body: ActionBody, x_axiom_confirm: str | None = Header(default=None)):
    _confirm("start_loop", x_axiom_confirm)
    return _run("start_loop", body.params)


@router.post("/stop_loop")
def stop_loop(body: ActionBody, x_axiom_confirm: str | None = Header(default=None)):
    _confirm("stop_loop", x_axiom_confirm)
    return _run("stop_loop", body.params)
