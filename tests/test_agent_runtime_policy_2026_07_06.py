"""AXIOM Agent Runtime — policy engine tier enforcement (reproduce-then-resolve).

Proves, against real classes and real disk (tmp_path; no unittest.mock):
  (a) an ESCALATION action can NEVER be auto-executed — only proposed
  (b) DE-ESCALATION is autonomous AND structurally risk-decreasing-only
  (c) OPERATIONAL actions require preflight + are audited
  (d) unclassifiable actions fail closed (denied, never silently allowed)
  (e) no registered action in the default registry can trade/arm/unhalt
"""
from __future__ import annotations

import inspect
import json

import pytest

from dashboard.server import control_safety as cs
from src.agent_runtime import policy
from src.scanner.automation.state_engine import StateEngine


def _isolate_audit(tmp_path, monkeypatch):
    monkeypatch.setattr(cs, "AUDIT_PATH", tmp_path / "control_audit.jsonl")


def _read_audit_rows(path):
    if not path.exists():
        return []
    return [json.loads(ln) for ln in path.read_text().splitlines() if ln.strip()]


def _fresh_state(tmp_path, *, halted_lanes=None):
    state_path = tmp_path / "state.json"
    eng = StateEngine(state_path=state_path)
    eng.set_halted(False)  # start unhalted-global so we can exercise per-lane transitions
    if halted_lanes:
        for lane, val in halted_lanes.items():
            eng.set_halted(val, lane=lane)
    return state_path


# ---------------------------------------------------------------------------
# (a) escalation can never be auto-executed — only proposed
# ---------------------------------------------------------------------------


def test_actionspec_construction_rejects_execute_callable_for_escalation_tier():
    """Structural guarantee: an ESCALATION ActionSpec cannot even be built with
    an execute callable — this is enforced at construction time, not just at
    call time, so there is no code path that could smuggle one in later."""
    with pytest.raises(policy.PolicyRegistrationError):
        policy.ActionSpec(
            name="sneaky_unhalt",
            tier=policy.ActionTier.ESCALATION,
            description="should be rejected",
            execute=lambda **kw: {"result": "should never run"},
        )


def test_all_builtin_escalation_actions_carry_no_execute_callable():
    for name, spec in policy.BUILTIN_ACTIONS.items():
        if spec.tier is policy.ActionTier.ESCALATION:
            assert spec.execute is None, f"{name} is ESCALATION but has an execute callable"


def test_submit_refuses_a_tampered_escalation_spec_even_if_frozen_guard_is_bypassed(tmp_path, monkeypatch):
    """Defense in depth: a frozen dataclass's field can still be forced via
    ``object.__setattr__`` (a documented Python escape hatch). Prove
    ``PolicyEngine.submit`` refuses such a spec at submit-time too, rather
    than relying solely on the construction-time guard."""
    _isolate_audit(tmp_path, monkeypatch)
    _fresh_state(tmp_path)
    spec = policy.ActionSpec(name="tampered", tier=policy.ActionTier.ESCALATION, description="x")
    object.__setattr__(spec, "execute", lambda **kw: {"result": "should never run"})
    eng = policy.default_engine({"tampered": spec})

    with pytest.raises(policy.PolicyRegistrationError):
        eng.submit("tampered", {}, actor="test-agent")

    rows = _read_audit_rows(cs.AUDIT_PATH)
    assert rows[0]["outcome"] == "denied_tampered_spec"


def test_escalation_action_returns_proposal_and_never_executes(tmp_path, monkeypatch):
    _isolate_audit(tmp_path, monkeypatch)
    state_path = _fresh_state(tmp_path, halted_lanes={"equity": True})
    eng = policy.default_engine()

    result = eng.submit("unhalt_lane", {"lane": "equity"}, actor="test-agent")

    assert result.executed is False
    assert result.proposal is not None
    assert result.proposal.tier is policy.ActionTier.ESCALATION
    assert result.proposal.action == "unhalt_lane"

    # the lane must be UNCHANGED — the proposal touched nothing
    assert StateEngine(state_path=state_path).get_halted(lane="equity") is True


def test_escalation_arm_action_returns_proposal_and_never_arms(tmp_path, monkeypatch):
    _isolate_audit(tmp_path, monkeypatch)
    _fresh_state(tmp_path)
    eng = policy.default_engine()

    result = eng.submit("arm_live_gate", {}, actor="test-agent")

    assert result.executed is False
    assert result.proposal is not None
    assert result.proposal.tier is policy.ActionTier.ESCALATION


def test_escalation_action_is_audited_as_proposed_not_allowed(tmp_path, monkeypatch):
    _isolate_audit(tmp_path, monkeypatch)
    _fresh_state(tmp_path, halted_lanes={"equity": True})
    eng = policy.default_engine()

    eng.submit("unhalt_lane", {"lane": "equity"}, actor="test-agent")

    rows = _read_audit_rows(cs.AUDIT_PATH)
    assert len(rows) == 1
    assert rows[0]["outcome"] == "proposed_for_operator"
    assert rows[0]["allowed"] is False
    assert rows[0]["tier"] == "escalation"
    assert rows[0]["actor"] == "test-agent"


# ---------------------------------------------------------------------------
# (b) de-escalation is autonomous AND risk-decreasing-only
# ---------------------------------------------------------------------------


def test_halt_lane_deescalation_executes_autonomously(tmp_path, monkeypatch):
    _isolate_audit(tmp_path, monkeypatch)
    state_path = _fresh_state(tmp_path, halted_lanes={"equity": False})
    eng = policy.default_engine()

    result = eng.submit("halt_lane", {"lane": "equity", "state_path": state_path}, actor="test-agent")

    assert result.executed is True
    assert result.proposal is None
    assert StateEngine(state_path=state_path).get_halted(lane="equity") is True


def test_halt_lane_function_has_no_parameter_that_could_unhalt():
    """The de-escalation halt action's own signature makes an unhalt
    structurally impossible: there is no boolean/value parameter to flip —
    only ``lane`` is accepted, and the function body hardcodes ``True``."""
    sig = inspect.signature(policy._deescalate_halt_lane)
    assert set(sig.parameters) <= {"lane", "state_path"}
    source = inspect.getsource(policy._deescalate_halt_lane)
    assert "set_halted(True" in source
    assert "set_halted(False" not in source


def test_reduce_gross_leverage_allows_a_strict_decrease(tmp_path, monkeypatch):
    _isolate_audit(tmp_path, monkeypatch)
    _fresh_state(tmp_path)
    monkeypatch.setattr(cs, "OVERRIDES_PATH", tmp_path / "control_overrides.json")
    cs.set_override("gross_leverage", 3.0)
    eng = policy.default_engine()

    result = eng.submit("reduce_gross_leverage", {"new_value": 1.0}, actor="test-agent")

    assert result.executed is True
    assert cs.read_overrides()["gross_leverage"] == 1.0


def test_reduce_gross_leverage_refuses_an_increase(tmp_path, monkeypatch):
    _isolate_audit(tmp_path, monkeypatch)
    _fresh_state(tmp_path)
    monkeypatch.setattr(cs, "OVERRIDES_PATH", tmp_path / "control_overrides.json")
    cs.set_override("gross_leverage", 1.0)
    eng = policy.default_engine()

    with pytest.raises(policy.PolicyDenied):
        eng.submit("reduce_gross_leverage", {"new_value": 5.0}, actor="test-agent")

    # override must be UNCHANGED — a denied de-escalation must not mutate state
    assert cs.read_overrides()["gross_leverage"] == 1.0


def test_reduce_gross_leverage_refuses_a_no_op_equal_value(tmp_path, monkeypatch):
    _isolate_audit(tmp_path, monkeypatch)
    _fresh_state(tmp_path)
    monkeypatch.setattr(cs, "OVERRIDES_PATH", tmp_path / "control_overrides.json")
    cs.set_override("gross_leverage", 2.0)
    eng = policy.default_engine()

    with pytest.raises(policy.PolicyDenied):
        eng.submit("reduce_gross_leverage", {"new_value": 2.0}, actor="test-agent")


def test_deescalation_denial_is_audited(tmp_path, monkeypatch):
    _isolate_audit(tmp_path, monkeypatch)
    _fresh_state(tmp_path)
    monkeypatch.setattr(cs, "OVERRIDES_PATH", tmp_path / "control_overrides.json")
    cs.set_override("gross_leverage", 1.0)
    eng = policy.default_engine()

    with pytest.raises(policy.PolicyDenied):
        eng.submit("reduce_gross_leverage", {"new_value": 5.0}, actor="test-agent")

    rows = _read_audit_rows(cs.AUDIT_PATH)
    assert any(r["outcome"] == "denied_by_action" and r["allowed"] is False for r in rows)


# ---------------------------------------------------------------------------
# (c) operational actions require preflight + are audited
# ---------------------------------------------------------------------------


def test_operational_action_runs_preflight_executes_and_is_audited(tmp_path, monkeypatch):
    _isolate_audit(tmp_path, monkeypatch)
    _fresh_state(tmp_path)
    calls = []

    def _read_something(**kwargs):
        calls.append(kwargs)
        return {"ok": True}

    spec = policy.ActionSpec(
        name="read_something",
        tier=policy.ActionTier.OPERATIONAL,
        description="a trivial read",
        execute=_read_something,
    )
    eng = policy.default_engine({"read_something": spec})

    result = eng.submit("read_something", {}, actor="test-agent")

    assert result.executed is True
    assert result.outcome == {"ok": True}
    assert len(calls) == 1
    rows = _read_audit_rows(cs.AUDIT_PATH)
    assert rows[0]["outcome"] == "executed"
    assert rows[0]["tier"] == "operational"
    assert rows[0]["allowed"] is True


def test_operational_action_denied_when_practice_pin_violated(tmp_path, monkeypatch):
    _isolate_audit(tmp_path, monkeypatch)
    _fresh_state(tmp_path)
    monkeypatch.setattr(policy, "_current_oanda_environment", lambda: "live")
    called = []
    spec = policy.ActionSpec(
        name="read_something",
        tier=policy.ActionTier.OPERATIONAL,
        description="a trivial read",
        execute=lambda **kw: called.append(1) or {"ok": True},
    )
    eng = policy.default_engine({"read_something": spec})

    with pytest.raises(policy.PolicyDenied):
        eng.submit("read_something", {}, actor="test-agent")

    assert not called  # preflight must block BEFORE execute runs
    rows = _read_audit_rows(cs.AUDIT_PATH)
    assert rows[0]["outcome"] == "denied_preflight"


def test_operational_action_construction_requires_execute_callable():
    with pytest.raises(policy.PolicyRegistrationError):
        policy.ActionSpec(
            name="broken",
            tier=policy.ActionTier.OPERATIONAL,
            description="missing execute",
        )


# ---------------------------------------------------------------------------
# (d) fail-closed on unclassifiable actions
# ---------------------------------------------------------------------------


def test_unknown_action_is_denied_fail_closed(tmp_path, monkeypatch):
    _isolate_audit(tmp_path, monkeypatch)
    _fresh_state(tmp_path)
    eng = policy.default_engine()

    with pytest.raises(policy.PolicyDenied):
        eng.submit("totally_made_up_action_xyz", {}, actor="test-agent")

    rows = _read_audit_rows(cs.AUDIT_PATH)
    assert rows[0]["outcome"] == "denied_unknown_action"
    assert rows[0]["allowed"] is False


def test_classify_returns_none_for_unknown_action():
    eng = policy.default_engine()
    assert eng.classify("totally_made_up_action_xyz") is None


def test_build_registry_rejects_name_collisions():
    with pytest.raises(policy.PolicyRegistrationError):
        policy.build_registry({"halt_lane": policy.BUILTIN_ACTIONS["halt_lane"]})


# ---------------------------------------------------------------------------
# (e) no registered action can trade / arm / unhalt
# ---------------------------------------------------------------------------

_FORBIDDEN_SUBSTRINGS = (
    "set_halted(False",
    ".arm(",
    "LiveGate(",
    "submit_order",
    "create_order",
    "close_position",
    "execute_trade(",
)


def test_no_builtin_executable_action_source_contains_a_forbidden_call():
    for name, spec in policy.BUILTIN_ACTIONS.items():
        if spec.execute is None:
            continue
        source = inspect.getsource(spec.execute)
        for forbidden in _FORBIDDEN_SUBSTRINGS:
            assert forbidden not in source, f"{name}.execute contains forbidden call {forbidden!r}"


def test_every_known_action_has_exactly_one_of_the_three_tiers():
    for name, spec in policy.BUILTIN_ACTIONS.items():
        assert spec.tier in (
            policy.ActionTier.OPERATIONAL,
            policy.ActionTier.DEESCALATION,
            policy.ActionTier.ESCALATION,
        ), f"{name} has an unclassified tier"


def test_default_engine_practice_pin_preflight_runs_for_every_tier(tmp_path, monkeypatch):
    """Even a DEESCALATION action must pass the practice-pin preflight —
    de-escalation autonomy is not an exemption from the Hard NO checks."""
    _isolate_audit(tmp_path, monkeypatch)
    state_path = _fresh_state(tmp_path, halted_lanes={"equity": False})
    monkeypatch.setattr(policy, "_current_oanda_environment", lambda: "live")
    eng = policy.default_engine()

    with pytest.raises(policy.PolicyDenied):
        eng.submit("halt_lane", {"lane": "equity", "state_path": state_path}, actor="test-agent")
