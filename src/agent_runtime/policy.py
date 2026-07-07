"""AXIOM Agent Runtime — action-policy engine.

Classifies every action a future resident agent loop might invoke into exactly
one of four tiers and enforces the tier's contract:

  OPERATIONAL      — restart a daemon, refresh data, run a diagnostic, clear a
                      stale flag. Allowed after a preflight check, audited.
  DEESCALATION     — halt a lane, de-risk, reduce size. Allowed AUTONOMOUSLY,
                      but preflighted + audited, and structurally
                      risk-DECREASING only (see the ``ActionSpec`` contract
                      below and the individual action implementations).
  SELF_IMPROVE     — a fixed menu of narrow, pre-coded data/doctrine edits
                      (see ``src/agent_runtime/self_improve.py``) against a
                      hardcoded target path per action, gated by apply -> test
                      -> verify -> commit-or-revert. Autonomy-flag-gated like
                      OPERATIONAL/DEESCALATION. A second, independent
                      allow/deny check in that module refuses any target
                      outside a tiny explicit allowlist (data-hygiene JSON,
                      LESSONS.md, NOTES.md) and refuses a STRUCTURAL denylist
                      unconditionally (trade/execution/gate/policy/autonomy-
                      flag code, the practice pin, LiveGate/ARM code) —
                      those stay ESCALATION regardless of any flag.
  ESCALATION       — arm, unhalt, size up, enable new exposure, promote a
                      model, change strategy/code. NEVER autonomous — the
                      engine can only ever return a ``Proposal`` for the
                      operator. This is enforced STRUCTURALLY:
                      ``ActionSpec.__post_init__`` refuses to construct an
                      ESCALATION spec that carries an ``execute`` callable at
                      all, and ``PolicyEngine.submit`` never reads ``.execute``
                      on the ESCALATION branch. There is no code path from
                      ``submit()`` to a state mutation for this tier.

Fail-closed: an action name absent from the registry is DENIED, not silently
allowed and not defaulted to any tier.

Wired to the EXISTING infra, not reimplemented: ``StateEngine.set_halted`` for
halts, ``dashboard.server.control_safety`` for leverage overrides + the shared
audit log, ``src.scanner.config.ScannerConfig`` for the practice-pin re-derive.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, Optional


class ActionTier(str, Enum):
    OPERATIONAL = "operational"
    DEESCALATION = "deescalation"
    SELF_IMPROVE = "self_improve"
    ESCALATION = "escalation"


class PolicyDenied(RuntimeError):
    """Raised when an action is refused — unknown action, failed preflight,
    or a de-escalation action that would not actually decrease risk."""


class PolicyRegistrationError(RuntimeError):
    """Raised when an ActionSpec / registry violates a structural rule at
    construction time (e.g. an ESCALATION spec carrying an execute callable,
    or two specs claiming the same action name)."""


@dataclass(frozen=True)
class Proposal:
    """What an ESCALATION action produces instead of an effect. Routed to the
    operator; nothing in this codebase auto-approves one."""

    action: str
    tier: ActionTier
    params: Dict[str, Any]
    rationale: str
    proposal_id: str
    requires: str = "explicit operator approval (typed confirmation / dashboard ARM)"


def compute_proposal_id(action: str, params: Dict[str, Any]) -> str:
    """Stable id for a proposal, derived from (action, params) -- NOT random and
    NOT time-based, so the identical proposal re-surfacing across resident-loop
    cycles (e.g. the LLM re-noticing the same unresolved defect) maps to the
    SAME id, and an operator's accept/deny decision (keyed by this id in
    ``src.agent_runtime.proposal_store``) survives across cycles instead of
    being asked again every 5 minutes."""
    canonical = json.dumps({"action": action, "params": params}, sort_keys=True, default=str)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]


@dataclass(frozen=True)
class PolicyResult:
    action: str
    tier: ActionTier
    executed: bool
    outcome: Dict[str, Any] = field(default_factory=dict)
    proposal: Optional[Proposal] = None


@dataclass(frozen=True)
class ActionSpec:
    """One entry in the policy registry.

    The structural guarantee lives here: ``tier is ESCALATION`` and
    ``execute is not None`` can never both be true — the dataclass refuses to
    construct. Every non-escalation action MUST carry an execute callable
    (there is no such thing as an inert operational/de-escalation entry that
    silently no-ops).
    """

    name: str
    tier: ActionTier
    description: str
    execute: Optional[Callable[..., Dict[str, Any]]] = None
    preflight: Optional[Callable[[Dict[str, Any]], None]] = None
    params_hint: Optional[str] = None
    """Optional one-line description of the EXACT accepted kwargs, surfaced to the
    diagnosing LLM in loop.py's prompt. Without this, the model has only the bare
    action name to go on and will guess plausible-sounding param names (observed
    live: "tiers", "regimes" instead of the actual "keys") -- execute()'s strict
    signature then rejects the unexpected kwarg and the action is denied every
    cycle. This closes that gap; it is documentation only, never a trust boundary
    (the structural denylist/allowlist and hardcoded target paths don't depend on
    the LLM getting this right)."""

    def __post_init__(self) -> None:
        if self.tier is ActionTier.ESCALATION:
            if self.execute is not None:
                raise PolicyRegistrationError(
                    f"structural violation: escalation action {self.name!r} must not "
                    "carry an execute callable — escalation can only ever be proposed"
                )
        elif self.execute is None:
            raise PolicyRegistrationError(
                f"{self.tier.value} action {self.name!r} requires an execute callable"
            )


def _current_oanda_environment() -> str:
    from src.scanner.config import ScannerConfig

    return getattr(ScannerConfig(), "oanda_environment", "practice")


def _preflight_practice_pin(_params: Dict[str, Any]) -> None:
    """Every action, every tier — the practice pin is re-derived from disk,
    never assumed. (L-003: oanda_environment stays "practice", forever.)"""
    env = _current_oanda_environment()
    if env != "practice":
        raise PolicyDenied(
            f"HARD LINE: practice pin violated (oanda_environment={env!r}) — refusing all actions"
        )


class PolicyEngine:
    """The one entry point a future agent loop calls to attempt an action."""

    def __init__(self, registry: Dict[str, ActionSpec], *, audit_fn: Callable[..., None]):
        self._registry = dict(registry)
        self._audit_fn = audit_fn

    def classify(self, action: str) -> Optional[ActionTier]:
        spec = self._registry.get(action)
        return spec.tier if spec is not None else None

    def submit(self, action: str, params: Optional[Dict[str, Any]] = None, *, actor: str) -> PolicyResult:
        params = dict(params or {})
        spec = self._registry.get(action)

        if spec is None:
            self._audit_fn(
                action=action, tier=None, actor=actor, allowed=False,
                outcome="denied_unknown_action", params=params,
                reason=f"unclassifiable action {action!r} — fail-closed deny",
            )
            raise PolicyDenied(f"unknown/unclassifiable action {action!r} — fail-closed deny")

        try:
            _preflight_practice_pin(params)
            if spec.preflight is not None:
                spec.preflight(params)
        except PolicyDenied as exc:
            self._audit_fn(
                action=action, tier=spec.tier.value, actor=actor, allowed=False,
                outcome="denied_preflight", reason=str(exc), params=params,
            )
            raise
        except Exception as exc:  # noqa: BLE001 - never let a preflight bug bypass the audit trail
            self._audit_fn(
                action=action, tier=spec.tier.value, actor=actor, allowed=False,
                outcome="error_preflight", reason=repr(exc), params=params,
            )
            raise

        if spec.tier is ActionTier.ESCALATION:
            # Redundant runtime check, not just the __post_init__ guard: a
            # frozen dataclass's field can still be forced via
            # object.__setattr__ (a documented Python escape hatch bypassing
            # __setattr__ entirely). Nothing in this codebase does that, but
            # this branch refuses to trust field immutability alone — even a
            # tampered spec cannot reach spec.execute from here, because this
            # branch never calls it under any condition, checked or not.
            if spec.execute is not None:
                self._audit_fn(
                    action=action, tier=spec.tier.value, actor=actor, allowed=False,
                    outcome="denied_tampered_spec",
                    reason="escalation spec carries an execute callable at submit-time — refused",
                    params=params,
                )
                raise PolicyRegistrationError(
                    f"structural violation detected at submit-time: escalation action "
                    f"{action!r} carries an execute callable — refusing to proceed at all "
                    "(this should be structurally impossible; treat as tampering)"
                )
            proposal = Proposal(
                action=action, tier=spec.tier, params=params, rationale=spec.description,
                proposal_id=compute_proposal_id(action, params),
            )
            self._audit_fn(
                action=action, tier=spec.tier.value, actor=actor, allowed=False,
                outcome="proposed_for_operator", reason=spec.description, params=params,
            )
            return PolicyResult(action=action, tier=spec.tier, executed=False, proposal=proposal)

        try:
            outcome = spec.execute(**params) or {}
        except PolicyDenied as exc:
            self._audit_fn(
                action=action, tier=spec.tier.value, actor=actor, allowed=False,
                outcome="denied_by_action", reason=str(exc), params=params,
            )
            raise
        except Exception as exc:  # noqa: BLE001 — preflight passed, the effect itself failed
            self._audit_fn(
                action=action, tier=spec.tier.value, actor=actor, allowed=True,
                outcome="error", reason=repr(exc), params=params,
            )
            raise

        self._audit_fn(
            action=action, tier=spec.tier.value, actor=actor, allowed=True,
            outcome="executed", reason="preflight ok", params=params, result=outcome,
        )
        return PolicyResult(action=action, tier=spec.tier, executed=True, outcome=outcome)


# ---------------------------------------------------------------------------
# De-escalation actions — autonomous, structurally risk-decreasing only.
# ---------------------------------------------------------------------------


def _deescalate_halt_lane(lane: Optional[str] = None, state_path: Optional[Any] = None) -> Dict[str, Any]:
    """Halt a lane (or, if ``lane`` is omitted, the global switch — which
    halts every lane). There is no parameter that could unhalt: the only
    boolean this function ever passes to ``set_halted`` is the literal
    ``True`` baked into the call below. ``state_path`` defaults to the real
    ``.claude/state.json`` (via ``StateEngine``'s own default) — tests pass an
    explicit tmp path so they never touch production state."""
    from src.scanner.automation.state_engine import KNOWN_LANES, StateEngine

    if lane is not None and lane not in KNOWN_LANES:
        raise PolicyDenied(f"unknown lane {lane!r}; allowed: {sorted(KNOWN_LANES)}")
    StateEngine(state_path=state_path).set_halted(True, lane=lane)
    return {"result": "halted", "lane": lane or "global (all lanes)"}


def _deescalate_reduce_gross_leverage(new_value: float) -> Dict[str, Any]:
    """Lower the gross-leverage override. Refuses (PolicyDenied) unless the
    requested value is a STRICT decrease from the currently effective value —
    de-escalation must never be a no-op or an increase."""
    from dashboard.server import control_safety as cs
    from src.equity.oanda_trend import DEFAULT_GROSS_LEVERAGE

    current = cs.read_overrides().get("gross_leverage")
    if current is None:
        current = DEFAULT_GROSS_LEVERAGE
    validated = cs.validate_leverage(new_value)
    if not (validated < float(current)):
        raise PolicyDenied(
            f"reduce_gross_leverage refused: new={validated} is not strictly less than "
            f"current={current} — de-escalation must be risk-decreasing"
        )
    updated = cs.set_override("gross_leverage", validated)
    return {"result": "leverage_reduced", "previous": current, "gross_leverage": updated["gross_leverage"]}


# ---------------------------------------------------------------------------
# Escalation actions — descriptive only; NEVER carry an execute callable.
# ---------------------------------------------------------------------------

_ESCALATION_ACTIONS: Dict[str, ActionSpec] = {
    "unhalt_lane": ActionSpec(
        name="unhalt_lane", tier=ActionTier.ESCALATION,
        description=(
            "Unhalt trading for a lane (or globally). Requires the operator's "
            "unhalt-eligibility review (drawdown, gate freshness, model staleness) "
            "and an explicit typed action — never autonomous."
        ),
    ),
    "arm_live_gate": ActionSpec(
        name="arm_live_gate", tier=ActionTier.ESCALATION,
        description=(
            "Arm the live/order-placing path (LiveGate.arm / dashboard ARM). "
            "Requires the operator's typed 'LIVE'/ARM confirmation — never autonomous."
        ),
    ),
    "increase_gross_leverage": ActionSpec(
        name="increase_gross_leverage", tier=ActionTier.ESCALATION,
        description="Raise gross leverage above its current value — risk-increasing; operator approval required.",
    ),
    "promote_model": ActionSpec(
        name="promote_model", tier=ActionTier.ESCALATION,
        description="Promote a shadow/candidate model to champion — requires ship-gate pass + operator approval.",
    ),
    "enable_new_exposure": ActionSpec(
        name="enable_new_exposure", tier=ActionTier.ESCALATION,
        description="Enable a new instrument/lane/strategy exposure — requires operator approval.",
    ),
    "change_strategy_or_code": ActionSpec(
        name="change_strategy_or_code", tier=ActionTier.ESCALATION,
        description="Change trading strategy, thresholds, or hot-path code — requires operator approval.",
    ),
}

_DEESCALATION_ACTIONS: Dict[str, ActionSpec] = {
    "halt_lane": ActionSpec(
        name="halt_lane", tier=ActionTier.DEESCALATION,
        description="Halt a lane (or globally) — always risk-decreasing.",
        execute=_deescalate_halt_lane,
    ),
    "reduce_gross_leverage": ActionSpec(
        name="reduce_gross_leverage", tier=ActionTier.DEESCALATION,
        description="Lower gross leverage — refuses any value that is not a strict decrease.",
        execute=_deescalate_reduce_gross_leverage,
    ),
}

BUILTIN_ACTIONS: Dict[str, ActionSpec] = {**_ESCALATION_ACTIONS, **_DEESCALATION_ACTIONS}


def build_registry(*extra: Dict[str, ActionSpec]) -> Dict[str, ActionSpec]:
    """Merge BUILTIN_ACTIONS with any extra (e.g. tool) action groups.

    Fail-closed on a name collision rather than silently letting one group
    shadow another.
    """
    registry = dict(BUILTIN_ACTIONS)
    for group in extra:
        overlap = set(registry) & set(group)
        if overlap:
            raise PolicyRegistrationError(f"action name collision: {sorted(overlap)}")
        registry.update(group)
    return registry


def default_engine(*extra: Dict[str, ActionSpec]) -> PolicyEngine:
    from src.agent_runtime.audit import record as _audit_record

    return PolicyEngine(build_registry(*extra), audit_fn=_audit_record)
