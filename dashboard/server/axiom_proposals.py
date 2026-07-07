"""AXIOM Phase 3 — Activity panel Accept/Deny for resident-loop proposals.

Mounted by app.py ONLY when ``AXIOM_CONTROL_ENABLED`` is truthy (same gate as
control.py — these endpoints can execute real, if practice-only, state
changes). Every ESCALATION action the resident loop proposes
(src/agent_runtime/policy.py's 6 ESCALATION actions) is structurally
INCAPABLE of auto-executing (see policy.py's ActionSpec/PolicyEngine
docstrings) — it can only ever become a Proposal routed to this panel.

Accept NEVER invents a new execution path. It dispatches to the SAME
pre-existing, independently-safeguarded function each action type already
has:
  unhalt_lane            -> dashboard.server.control._run("unhalt", ...) --
                             same eligibility gate (drawdown/verify_gate/
                             model-freshness) + ARM requirement as the
                             existing /api/control/unhalt endpoint.
  increase_gross_leverage -> control._run("set_gross_leverage", ...), with an
                             ADDED strict-increase check (mirroring
                             policy.py's reduce_gross_leverage) so an
                             "increase" proposal can never silently decrease
                             leverage or no-op.
  enable_new_exposure,
  change_strategy_or_code -> ConfigAdjuster.collect_adjustment() (schema
                             validation against ScannerConfig field names)
                             immediately followed by
                             AdjustmentApprover.approve() -- the SAME
                             two-stage flow any other config-adjustment
                             source uses; no bypass of that validation.
  arm_live_gate,
  promote_model           -> DELIBERATELY NOT WIRED this session. Getting
                             LiveGate.arm()'s expected_universe_hash or the
                             brain loop's promotion-decision context subtly
                             wrong is a worse failure mode than leaving these
                             two as an explicit "not yet automated -- do this
                             manually" response. See NOTES.md.

Deny NEVER executes anything -- it only records a disposition.
"""
from __future__ import annotations

from typing import Any, Callable, Dict, Optional

from fastapi import APIRouter, Header, HTTPException, Request

from dashboard.server import control_safety as cs
from dashboard.server.control import _actor, _confirm, _run as _control_run
from src.agent_runtime import loop as agent_loop
from src.agent_runtime import proposal_store

router = APIRouter(prefix="/api/axiom_proposals", tags=["axiom_proposals"])

# Action types with NO wired accept-handler yet -- Accept returns 501 with a
# clear explanation, never a silent no-op and never a half-verified attempt.
_NOT_YET_WIRED = frozenset({"arm_live_gate", "promote_model"})


def _read_mind_window() -> Dict[str, Any]:
    # Look up CYCLES_PATH/AUTONOMY_CONFIG_PATH on the module at CALL time (not
    # read_mind_window's own default-arg binding, which is fixed at import
    # time) so tests can monkeypatch agent_loop.CYCLES_PATH and have it apply.
    return agent_loop.read_mind_window(
        cycles_path=agent_loop.CYCLES_PATH, autonomy_path=agent_loop.AUTONOMY_CONFIG_PATH,
    )


def _find_proposal(proposal_id: str) -> Dict[str, Any]:
    window = _read_mind_window()
    for entry in window.get("pending_operator_proposals", []):
        if entry.get("proposal_id") == proposal_id:
            return entry
    raise HTTPException(status_code=404, detail=f"no pending proposal with id {proposal_id!r}")


def _accept_unhalt_lane(params: Dict[str, Any], *, actor: str) -> Dict[str, Any]:
    lane = params.get("lane")
    extra = {k: v for k, v in params.items() if k != "lane"}
    if extra:
        raise cs.ControlDenied(f"unhalt_lane proposal has unexpected params {sorted(extra)}")
    control_params = {"lane": lane} if lane else {}
    return _control_run("unhalt", control_params, actor=actor)


def _accept_increase_gross_leverage(params: Dict[str, Any], *, actor: str) -> Dict[str, Any]:
    try:
        new_value = float(params.get("new_value"))
    except (TypeError, ValueError):
        raise cs.ControlDenied(f"increase_gross_leverage proposal has no numeric 'new_value': {params!r}")

    current = cs.read_overrides().get("gross_leverage")
    if current is None:
        from src.equity.oanda_trend import DEFAULT_GROSS_LEVERAGE

        current = DEFAULT_GROSS_LEVERAGE
    validated = cs.validate_leverage(new_value)
    if not (validated > float(current)):
        raise cs.ControlDenied(
            f"increase_gross_leverage refused: new={validated} is not strictly greater than "
            f"current={current} -- an 'increase' proposal must actually increase"
        )
    return _control_run("set_gross_leverage", {"gross_leverage": validated}, actor=actor)


def _accept_via_config_adjuster(source_label: str, params: Dict[str, Any], *, actor: str) -> Dict[str, Any]:
    from src.scanner.automation.adjustment_approver import AdjustmentApprover
    from src.scanner.automation.config_adjuster import ConfigAdjuster

    key = params.get("key")
    if not key:
        raise cs.ControlDenied(f"{source_label} proposal has no 'key' param -- cannot translate to a config adjustment")
    value = params.get("value")
    reason = str(params.get("reason") or f"resident-loop {source_label} proposal, operator-accepted via Activity panel")

    adjuster_proposal_id = ConfigAdjuster().collect_adjustment(
        source=f"resident_loop:{source_label}", key=key, value=value, reason=reason,
    )
    if not adjuster_proposal_id:
        raise cs.ControlDenied(
            f"{key!r} failed ConfigAdjuster schema validation (not a known ScannerConfig field, "
            "or an invalid value) -- refusing to create a pending adjustment"
        )
    approved = AdjustmentApprover().approve(adjuster_proposal_id)
    if not approved:
        raise cs.ControlDenied(f"AdjustmentApprover refused proposal {adjuster_proposal_id}")
    return {
        "result": "config_adjustment_approved", "key": key, "value": value,
        "adjuster_proposal_id": adjuster_proposal_id,
    }


def _accept_enable_new_exposure(params: Dict[str, Any], *, actor: str) -> Dict[str, Any]:
    return _accept_via_config_adjuster("enable_new_exposure", params, actor=actor)


def _accept_change_strategy_or_code(params: Dict[str, Any], *, actor: str) -> Dict[str, Any]:
    return _accept_via_config_adjuster("change_strategy_or_code", params, actor=actor)


_ACCEPT_HANDLERS: Dict[str, Callable[..., Dict[str, Any]]] = {
    "unhalt_lane": _accept_unhalt_lane,
    "increase_gross_leverage": _accept_increase_gross_leverage,
    "enable_new_exposure": _accept_enable_new_exposure,
    "change_strategy_or_code": _accept_change_strategy_or_code,
}


@router.get("")
def list_proposals() -> Dict[str, Any]:
    """Pending + resolved proposals, each joined with its operator disposition."""
    window = _read_mind_window()
    return {
        "pending": window.get("pending_operator_proposals", []),
        "resolved": window.get("resolved_proposals", []),
    }


@router.post("/{proposal_id}/deny")
def deny_proposal(
    proposal_id: str,
    request: Request,
    *,
    x_axiom_actor: Optional[str] = Header(default=None),
) -> Dict[str, Any]:
    entry = _find_proposal(proposal_id)
    actor = _actor(request, x_axiom_actor)
    disposition = proposal_store.record_disposition(
        proposal_id, status="denied", actor=actor,
        reason="operator denied via Activity panel", detail={"action": entry.get("action")},
        path=proposal_store.DISPOSITIONS_PATH,
    )
    cs.audit(
        {"action": entry.get("action"), "outcome": "proposal_denied", "proposal_id": proposal_id,
         "allowed": False, "result": "denied"},
        actor=actor,
    )
    return {"ok": True, "disposition": disposition}


@router.post("/{proposal_id}/accept")
def accept_proposal(
    proposal_id: str,
    request: Request,
    *,
    x_axiom_confirm: Optional[str] = Header(default=None),
    x_axiom_actor: Optional[str] = Header(default=None),
) -> Dict[str, Any]:
    entry = _find_proposal(proposal_id)
    action = entry.get("action") or ""

    if action in _NOT_YET_WIRED:
        raise HTTPException(
            status_code=501,
            detail=(
                f"{action!r} has no wired accept-handler yet -- getting its safety-critical "
                "context (ship-gate universe hash / brain-loop promotion decision) subtly wrong "
                "is worse than not automating it. Perform this action manually via its existing "
                "dedicated path and then Deny this proposal so it stops resurfacing."
            ),
        )

    handler = _ACCEPT_HANDLERS.get(action)
    if handler is None:
        raise HTTPException(status_code=501, detail=f"{action!r} has no wired accept-handler")

    _confirm(action, x_axiom_confirm)
    actor = _actor(request, x_axiom_actor)
    # Params live under detail.params (matches ActionOutcome's real shape --
    # see loop.py::_act_escalation -- NOT a top-level "params" key on the outcome).
    params = (entry.get("detail") or {}).get("params") or {}

    try:
        result = handler(params, actor=actor)
    except cs.ControlDenied as exc:
        proposal_store.record_disposition(
            proposal_id, status="denied", actor=actor,
            reason=f"accept-time guard refused: {exc}", detail={"action": action},
            path=proposal_store.DISPOSITIONS_PATH,
        )
        raise HTTPException(status_code=403, detail=str(exc))
    except HTTPException as exc:
        # _control_run (control.py's reused _run) catches cs.ControlDenied itself
        # and re-raises AS an HTTPException(403, ...) -- by the time it's out
        # here it's no longer a ControlDenied, so this is the branch that
        # actually catches "the underlying safeguard (ARM/eligibility/etc)
        # refused". A 403/other-4xx from the safeguard is still a REAL refusal
        # and must be recorded, same as the ControlDenied branch above; only a
        # 5xx (unexpected breakage, not a policy refusal) is left unrecorded so
        # the operator can retry after investigating rather than have it look
        # like a considered "denied".
        if 400 <= exc.status_code < 500:
            proposal_store.record_disposition(
                proposal_id, status="denied", actor=actor,
                reason=f"accept-time guard refused: {exc.detail}", detail={"action": action},
                path=proposal_store.DISPOSITIONS_PATH,
            )
        raise
    except Exception as exc:  # noqa: BLE001 -- guards passed but the effect failed; never crash the route
        raise HTTPException(status_code=500, detail=f"accept effect failed: {exc!r}")

    disposition = proposal_store.record_disposition(
        proposal_id, status="accepted", actor=actor, detail={"action": action, "result": result},
        path=proposal_store.DISPOSITIONS_PATH,
    )
    return {"ok": True, "result": result, "disposition": disposition}
