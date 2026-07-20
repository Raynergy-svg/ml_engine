"""AXIOM Agent Runtime — resident reasoning loop (reproduce-then-resolve).

Proves, against real classes and real disk (tmp_path; no unittest.mock):
  (a) with agent_autonomy_enabled=False, OPERATIONAL/DEESCALATION proposed
      actions are logged as shadow intent but NEVER executed
  (b) ESCALATION proposed actions are NEVER auto-executed, even with the
      autonomy flag True — always routed to a Proposal
  (c) DEESCALATION stays structurally risk-decreasing (refusals still refuse)
  (d) the loop degrades gracefully when the claude CLI is unavailable —
      honest status, no crash
  (e) every cycle (and every action within it) is captured in the unified
      audit trail
"""
from __future__ import annotations

import json
import subprocess

import pytest

from dashboard.server import control_safety as cs
from src.agent_runtime import loop as agent_loop
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
    eng.set_halted(False)
    if halted_lanes:
        for lane, val in halted_lanes.items():
            eng.set_halted(val, lane=lane)
    return state_path


def _test_registry(side_effects):
    """A tiny fast registry: one OPERATIONAL, one DEESCALATION action, each of
    which appends to ``side_effects`` when (and only when) actually executed."""

    def _op_execute(**kwargs):
        side_effects.append(("op_executed", kwargs))
        return {"result": "op_done"}

    def _deescalate_execute(**kwargs):
        side_effects.append(("deescalate_executed", kwargs))
        return {"result": "deescalated"}

    def _broken_execute():
        # Simulates a real failure mode: the LLM's proposal used a param name
        # the action doesn't accept (e.g. "tiers" instead of "keys") -- calling
        # this with any kwarg raises TypeError before ever reaching the body.
        return {"should": "never reach here with an unexpected kwarg"}

    op_spec = policy.ActionSpec(
        name="test_op_action", tier=policy.ActionTier.OPERATIONAL,
        description="test operational action", execute=_op_execute,
    )
    broken_spec = policy.ActionSpec(
        name="test_broken_action", tier=policy.ActionTier.OPERATIONAL,
        description="test action whose execute() raises an unexpected exception",
        execute=_broken_execute,
    )
    deescalate_spec = policy.ActionSpec(
        name="test_deescalate_action", tier=policy.ActionTier.DEESCALATION,
        description="test de-escalation action", execute=_deescalate_execute,
    )
    observe_spec = policy.ActionSpec(
        name="test_observe_tool", tier=policy.ActionTier.OPERATIONAL,
        description="test read-only observe tool",
        execute=lambda **kw: {"observed": True},
    )
    return policy.build_registry({
        "test_op_action": op_spec,
        "test_broken_action": broken_spec,
        "test_deescalate_action": deescalate_spec,
        "test_observe_tool": observe_spec,
    })


def _audit_record_for(tmp_path):
    from src.agent_runtime.audit import record

    return record


def _diagnosis_stdout(proposed_actions, reasoning="test reasoning"):
    payload = {
        "reasoning": reasoning,
        "beliefs": "test beliefs",
        "proposed_actions": proposed_actions,
    }
    return f"<agent-loop>{json.dumps(payload)}</agent-loop>"


def _fake_subprocess_runner(stdout: str, returncode: int = 0):
    def _run(*args, **kwargs):
        return subprocess.CompletedProcess(args=args, returncode=returncode, stdout=stdout, stderr="")

    return _run


def _build_loop(tmp_path, monkeypatch, *, side_effects, stdout, autonomy_enabled, returncode=0):
    _isolate_audit(tmp_path, monkeypatch)
    _fresh_state(tmp_path)
    engine = policy.PolicyEngine(_test_registry(side_effects), audit_fn=_audit_record_for(tmp_path))
    autonomy_path = tmp_path / "loop_autonomy.json"
    if autonomy_enabled:
        agent_loop.set_autonomy_enabled(True, path=autonomy_path)
    resident = agent_loop.ResidentLoop(
        engine=engine,
        cycles_path=tmp_path / "loop_cycles.jsonl",
        autonomy_path=autonomy_path,
        observe_tool_names=("test_observe_tool",),
        shadow_lanes=(),
        cli_resolver=lambda _bin: "/usr/bin/fake-claude",
        subprocess_runner=_fake_subprocess_runner(stdout, returncode=returncode),
    )
    return resident


# ---------------------------------------------------------------------------
# (a) autonomy flag gates OPERATIONAL / DEESCALATION execution — default False
# ---------------------------------------------------------------------------


def test_flag_false_operational_action_is_shadow_logged_not_executed(tmp_path, monkeypatch):
    side_effects = []
    stdout = _diagnosis_stdout([{"action": "test_op_action", "params": {}, "rationale": "try it"}])
    resident = _build_loop(tmp_path, monkeypatch, side_effects=side_effects, stdout=stdout, autonomy_enabled=False)

    result = resident.run_cycle(actor="test-agent")

    assert side_effects == []  # the real execute callable never ran
    assert result.autonomy_enabled is False
    outcome = result.outcomes[0]
    assert outcome.action == "test_op_action"
    assert outcome.executed is False
    assert outcome.shadow is True

    rows = _read_audit_rows(cs.AUDIT_PATH)
    shadow_rows = [r for r in rows if r["outcome"] == "shadow_not_executed"]
    assert len(shadow_rows) == 1
    assert shadow_rows[0]["action"] == "test_op_action"
    assert shadow_rows[0]["allowed"] is False


def test_flag_false_deescalation_action_is_shadow_logged_not_executed(tmp_path, monkeypatch):
    side_effects = []
    stdout = _diagnosis_stdout([{"action": "test_deescalate_action", "params": {}, "rationale": "de-risk"}])
    resident = _build_loop(tmp_path, monkeypatch, side_effects=side_effects, stdout=stdout, autonomy_enabled=False)

    result = resident.run_cycle(actor="test-agent")

    assert side_effects == []
    outcome = result.outcomes[0]
    assert outcome.tier == "deescalation"
    assert outcome.executed is False
    assert outcome.shadow is True


def test_flag_true_operational_action_actually_executes(tmp_path, monkeypatch):
    side_effects = []
    stdout = _diagnosis_stdout([{"action": "test_op_action", "params": {}, "rationale": "try it"}])
    resident = _build_loop(tmp_path, monkeypatch, side_effects=side_effects, stdout=stdout, autonomy_enabled=True)

    result = resident.run_cycle(actor="test-agent")

    assert side_effects == [("op_executed", {})]
    outcome = result.outcomes[0]
    assert outcome.executed is True
    assert outcome.shadow is False
    rows = _read_audit_rows(cs.AUDIT_PATH)
    assert any(r["outcome"] == "executed" and r["action"] == "test_op_action" for r in rows)


def test_flag_true_deescalation_action_actually_executes(tmp_path, monkeypatch):
    side_effects = []
    stdout = _diagnosis_stdout([{"action": "test_deescalate_action", "params": {}, "rationale": "de-risk"}])
    resident = _build_loop(tmp_path, monkeypatch, side_effects=side_effects, stdout=stdout, autonomy_enabled=True)

    result = resident.run_cycle(actor="test-agent")

    assert side_effects == [("deescalate_executed", {})]
    assert result.outcomes[0].executed is True


def test_action_raising_unexpected_exception_is_denied_not_a_cycle_crash(tmp_path, monkeypatch):
    """Reproduces a real incident: the LLM's diagnosis proposed a real action
    with a wrong param name (a TypeError from execute()). The whole cycle must
    NOT crash -- that one action is denied, the cycle still completes and logs."""
    side_effects = []
    stdout = _diagnosis_stdout([
        {"action": "test_broken_action", "params": {"unexpected_param": "x"}, "rationale": "bad params"},
        {"action": "test_op_action", "params": {}, "rationale": "should still run"},
    ])
    resident = _build_loop(tmp_path, monkeypatch, side_effects=side_effects, stdout=stdout, autonomy_enabled=True)

    result = resident.run_cycle(actor="test-agent")  # must not raise

    broken_outcome, op_outcome = result.outcomes
    assert broken_outcome.action == "test_broken_action"
    assert broken_outcome.executed is False
    assert broken_outcome.denied is True
    assert "unexpected keyword argument" in broken_outcome.detail["reason"]

    # the NEXT proposed action in the same cycle still ran -- one bad action
    # doesn't take down the rest of the cycle either.
    assert op_outcome.action == "test_op_action"
    assert op_outcome.executed is True
    assert side_effects == [("op_executed", {})]

    rows = _read_audit_rows(cs.AUDIT_PATH)
    assert any(r["outcome"] == "error" and r["action"] == "test_broken_action" for r in rows)


def test_run_cycle_fails_closed_when_flag_file_has_garbage_bytes_at_read_time(tmp_path, monkeypatch):
    """Adversarial: prove fail-closed through the FULL run_cycle()/_act() path,
    not just is_autonomy_enabled() in isolation -- the flag file is corrupted
    at the moment the loop actually reads it, and a real proposed action must
    still be shadow-logged, never executed."""
    side_effects = []
    stdout = _diagnosis_stdout([{"action": "test_op_action", "params": {}, "rationale": "x"}])
    resident = _build_loop(tmp_path, monkeypatch, side_effects=side_effects, stdout=stdout, autonomy_enabled=False)
    resident.autonomy_path.write_bytes(b"\x00\x01\xff not-json-at-all")

    result = resident.run_cycle(actor="adversarial-test")

    assert result.autonomy_enabled is False
    assert side_effects == []
    assert result.outcomes[0].shadow is True
    assert result.outcomes[0].executed is False


def test_run_cycle_fails_closed_when_flag_file_is_deleted_at_read_time(tmp_path, monkeypatch):
    side_effects = []
    stdout = _diagnosis_stdout([{"action": "test_op_action", "params": {}, "rationale": "x"}])
    resident = _build_loop(tmp_path, monkeypatch, side_effects=side_effects, stdout=stdout, autonomy_enabled=False)
    assert not resident.autonomy_path.exists()

    result = resident.run_cycle(actor="adversarial-test")

    assert result.autonomy_enabled is False
    assert side_effects == []


def test_run_cycle_fails_closed_on_a_string_true_value_at_read_time(tmp_path, monkeypatch):
    side_effects = []
    stdout = _diagnosis_stdout([{"action": "test_op_action", "params": {}, "rationale": "x"}])
    resident = _build_loop(tmp_path, monkeypatch, side_effects=side_effects, stdout=stdout, autonomy_enabled=False)
    resident.autonomy_path.write_text(json.dumps({"agent_autonomy_enabled": "true"}), encoding="utf-8")

    result = resident.run_cycle(actor="adversarial-test")

    assert result.autonomy_enabled is False
    assert side_effects == []


def test_autonomy_defaults_false_when_config_file_absent(tmp_path):
    assert agent_loop.is_autonomy_enabled(tmp_path / "does_not_exist.json") is False


def test_autonomy_defaults_false_on_corrupt_config_file(tmp_path):
    bad = tmp_path / "loop_autonomy.json"
    bad.write_text("{not valid json", encoding="utf-8")
    assert agent_loop.is_autonomy_enabled(bad) is False


@pytest.mark.parametrize("payload", [
    {},
    {"agent_autonomy_enabled": "yes"},
    {"agent_autonomy_enabled": "true"},
    {"agent_autonomy_enabled": 1},
    {"agent_autonomy_enabled": None},
    {"agent_autonomy_enabled": []},
    {"other_key": True},
])
def test_autonomy_fails_closed_on_non_boolean_true_values(tmp_path, payload):
    """Only a literal JSON ``true`` under the exact key flips the flag on.
    Any other truthy-looking-but-wrong-typed content must still resolve to
    False -- this is the fail-closed contract the ACT gate depends on."""
    path = tmp_path / "loop_autonomy.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    assert agent_loop.is_autonomy_enabled(path) is False


def test_autonomy_fails_closed_when_file_is_a_json_array(tmp_path):
    path = tmp_path / "loop_autonomy.json"
    path.write_text(json.dumps([{"agent_autonomy_enabled": True}]), encoding="utf-8")
    assert agent_loop.is_autonomy_enabled(path) is False


def test_autonomy_fails_closed_on_garbage_bytes(tmp_path):
    path = tmp_path / "loop_autonomy.json"
    path.write_bytes(b"\x00\x01\xff\xfe not json at all")
    assert agent_loop.is_autonomy_enabled(path) is False


def test_autonomy_fails_closed_on_empty_file(tmp_path):
    path = tmp_path / "loop_autonomy.json"
    path.write_text("", encoding="utf-8")
    assert agent_loop.is_autonomy_enabled(path) is False


def test_set_autonomy_enabled_round_trips(tmp_path):
    path = tmp_path / "loop_autonomy.json"
    agent_loop.set_autonomy_enabled(True, path=path)
    assert agent_loop.is_autonomy_enabled(path) is True
    agent_loop.set_autonomy_enabled(False, path=path)
    assert agent_loop.is_autonomy_enabled(path) is False


# ---------------------------------------------------------------------------
# (b) escalation is ALWAYS proposal-only, even with autonomy flag True
# ---------------------------------------------------------------------------


def test_escalation_action_never_auto_executes_even_with_autonomy_flag_true(tmp_path, monkeypatch):
    _isolate_audit(tmp_path, monkeypatch)
    state_path = _fresh_state(tmp_path, halted_lanes={"equity": True})
    autonomy_path = tmp_path / "loop_autonomy.json"
    agent_loop.set_autonomy_enabled(True, path=autonomy_path)
    side_effects = []
    engine = policy.PolicyEngine(
        {**_test_registry(side_effects), **policy.BUILTIN_ACTIONS}, audit_fn=_audit_record_for(tmp_path),
    )
    stdout = _diagnosis_stdout([{"action": "unhalt_lane", "params": {"lane": "equity", "state_path": str(state_path)}, "rationale": "operator wants it unhalted"}])
    resident = agent_loop.ResidentLoop(
        engine=engine, cycles_path=tmp_path / "loop_cycles.jsonl", autonomy_path=autonomy_path,
        observe_tool_names=(), shadow_lanes=(),
        cli_resolver=lambda _bin: "/usr/bin/fake-claude",
        subprocess_runner=_fake_subprocess_runner(stdout),
    )

    result = resident.run_cycle(actor="test-agent")

    outcome = result.outcomes[0]
    assert outcome.action == "unhalt_lane"
    assert outcome.executed is False
    assert outcome.proposal is True
    assert result.verified is True
    # the lane must remain halted -- the proposal touched nothing
    assert StateEngine(state_path=state_path).get_halted(lane="equity") is True


def test_tampered_escalation_spec_is_denied_not_a_cycle_crash(tmp_path, monkeypatch):
    """Reproduces a defense-in-depth failure mode: if an ESCALATION spec's
    execute callable were ever forced onto it at runtime (bypassing
    __post_init__ via object.__setattr__ -- the documented Python escape hatch
    for frozen dataclasses that policy.ActionSpec's docstring warns about),
    PolicyEngine.submit() correctly refuses to run it and raises
    PolicyRegistrationError. That raise must not propagate past
    _act_escalation and crash the whole cycle -- the escalation stays BLOCKED
    either way; only the failure mode changes from a process kill to a logged
    deny."""
    _isolate_audit(tmp_path, monkeypatch)
    _fresh_state(tmp_path)
    autonomy_path = tmp_path / "loop_autonomy.json"
    agent_loop.set_autonomy_enabled(True, path=autonomy_path)

    tampered_spec = policy.ActionSpec(
        name="tampered_escalation", tier=policy.ActionTier.ESCALATION,
        description="escalation spec tampered with an execute callable at runtime",
    )
    object.__setattr__(tampered_spec, "execute", lambda **kw: {"should": "never run"})

    engine = policy.PolicyEngine({"tampered_escalation": tampered_spec}, audit_fn=_audit_record_for(tmp_path))
    stdout = _diagnosis_stdout([{"action": "tampered_escalation", "params": {}, "rationale": "should be blocked, not crash"}])
    resident = agent_loop.ResidentLoop(
        engine=engine, cycles_path=tmp_path / "loop_cycles.jsonl", autonomy_path=autonomy_path,
        observe_tool_names=(), shadow_lanes=(),
        cli_resolver=lambda _bin: "/usr/bin/fake-claude",
        subprocess_runner=_fake_subprocess_runner(stdout),
    )

    result = resident.run_cycle(actor="test-agent")  # must not raise

    outcome = result.outcomes[0]
    assert outcome.action == "tampered_escalation"
    assert outcome.tier == "escalation"
    assert outcome.executed is False
    assert outcome.proposal is False
    assert outcome.denied is True
    assert "PolicyRegistrationError" in outcome.detail["reason"]

    rows = _read_audit_rows(cs.AUDIT_PATH)
    assert any(r["outcome"] == "denied_tampered_spec" and r["action"] == "tampered_escalation" for r in rows)


def test_unknown_action_name_is_denied_fail_closed(tmp_path, monkeypatch):
    side_effects = []
    stdout = _diagnosis_stdout([{"action": "totally_bogus_action_xyz", "params": {}, "rationale": "??"}])
    resident = _build_loop(tmp_path, monkeypatch, side_effects=side_effects, stdout=stdout, autonomy_enabled=True)

    result = resident.run_cycle(actor="test-agent")

    outcome = result.outcomes[0]
    assert outcome.tier is None
    assert outcome.denied is True
    assert outcome.executed is False
    rows = _read_audit_rows(cs.AUDIT_PATH)
    assert any(r["outcome"] == "denied_unknown_action" for r in rows)


# ---------------------------------------------------------------------------
# (c) de-escalation stays risk-decreasing-only, even when autonomy is on
# ---------------------------------------------------------------------------


def test_deescalation_refusal_is_not_swallowed_as_a_silent_success(tmp_path, monkeypatch):
    _isolate_audit(tmp_path, monkeypatch)
    _fresh_state(tmp_path)
    monkeypatch.setattr(cs, "OVERRIDES_PATH", tmp_path / "control_overrides.json")
    cs.set_override("gross_leverage", 1.0)
    autonomy_path = tmp_path / "loop_autonomy.json"
    agent_loop.set_autonomy_enabled(True, path=autonomy_path)
    engine = policy.default_engine()
    stdout = _diagnosis_stdout([{"action": "reduce_gross_leverage", "params": {"new_value": 5.0}, "rationale": "oops, this is an increase"}])
    resident = agent_loop.ResidentLoop(
        engine=engine, cycles_path=tmp_path / "loop_cycles.jsonl", autonomy_path=autonomy_path,
        observe_tool_names=(), shadow_lanes=(),
        cli_resolver=lambda _bin: "/usr/bin/fake-claude",
        subprocess_runner=_fake_subprocess_runner(stdout),
    )

    result = resident.run_cycle(actor="test-agent")

    outcome = result.outcomes[0]
    assert outcome.executed is False
    assert outcome.denied is True
    assert cs.read_overrides()["gross_leverage"] == 1.0  # unchanged


# ---------------------------------------------------------------------------
# (d) graceful degradation when the claude CLI is unavailable
# ---------------------------------------------------------------------------


def test_cli_unavailable_degrades_honestly_without_crashing(tmp_path, monkeypatch):
    _isolate_audit(tmp_path, monkeypatch)
    _fresh_state(tmp_path)
    engine = policy.PolicyEngine(_test_registry([]), audit_fn=_audit_record_for(tmp_path))
    resident = agent_loop.ResidentLoop(
        engine=engine, cycles_path=tmp_path / "loop_cycles.jsonl", autonomy_path=tmp_path / "loop_autonomy.json",
        observe_tool_names=("test_observe_tool",), shadow_lanes=(),
        cli_resolver=lambda _bin: None,  # CLI not found
        subprocess_runner=lambda *a, **kw: pytest.fail("must not spawn a subprocess when CLI is unavailable"),
    )

    result = resident.run_cycle(actor="test-agent")

    assert result.cli_available is False
    assert result.diagnosis["available"] is False
    assert result.proposed_actions == []
    assert result.outcomes == []
    assert result.verified is True


def test_cli_timeout_degrades_honestly_without_crashing(tmp_path, monkeypatch):
    _isolate_audit(tmp_path, monkeypatch)
    _fresh_state(tmp_path)
    engine = policy.PolicyEngine(_test_registry([]), audit_fn=_audit_record_for(tmp_path))

    def _timeout_runner(*args, **kwargs):
        raise subprocess.TimeoutExpired(cmd="claude", timeout=1)

    resident = agent_loop.ResidentLoop(
        engine=engine, cycles_path=tmp_path / "loop_cycles.jsonl", autonomy_path=tmp_path / "loop_autonomy.json",
        observe_tool_names=(), shadow_lanes=(),
        cli_resolver=lambda _bin: "/usr/bin/fake-claude",
        subprocess_runner=_timeout_runner,
    )

    result = resident.run_cycle(actor="test-agent", timeout_seconds=1)

    assert result.cli_available is True
    assert result.diagnosis["available"] is True
    assert result.proposed_actions == []


def test_cli_nonzero_returncode_degrades_honestly(tmp_path, monkeypatch):
    side_effects = []
    resident = _build_loop(tmp_path, monkeypatch, side_effects=side_effects, stdout="garbage, no block", autonomy_enabled=True, returncode=1)

    result = resident.run_cycle(actor="test-agent")

    assert result.proposed_actions == []
    assert "exited 1" in result.diagnosis["reasoning"]


def test_malformed_diagnosis_block_degrades_to_no_actions(tmp_path, monkeypatch):
    side_effects = []
    resident = _build_loop(tmp_path, monkeypatch, side_effects=side_effects, stdout="<agent-loop>{not valid json</agent-loop>", autonomy_enabled=True)

    result = resident.run_cycle(actor="test-agent")

    assert result.proposed_actions == []


# ---------------------------------------------------------------------------
# (e) unified audit trail captures every cycle + every action
# ---------------------------------------------------------------------------


def test_every_cycle_writes_a_cycle_level_audit_entry(tmp_path, monkeypatch):
    side_effects = []
    stdout = _diagnosis_stdout([])
    resident = _build_loop(tmp_path, monkeypatch, side_effects=side_effects, stdout=stdout, autonomy_enabled=False)

    result = resident.run_cycle(actor="test-agent")

    rows = _read_audit_rows(cs.AUDIT_PATH)
    cycle_rows = [r for r in rows if r["action"] == "resident_loop_cycle"]
    assert len(cycle_rows) == 1
    assert cycle_rows[0]["params"]["cycle_id"] == result.cycle_id
    assert cycle_rows[0]["params"]["autonomy_enabled"] is False


def test_observe_step_reads_tools_through_the_policy_engine_and_is_audited(tmp_path, monkeypatch):
    side_effects = []
    stdout = _diagnosis_stdout([])
    resident = _build_loop(tmp_path, monkeypatch, side_effects=side_effects, stdout=stdout, autonomy_enabled=False)

    result = resident.run_cycle(actor="test-agent")

    assert result.observations["test_observe_tool"]["ok"] is True
    assert result.observations["test_observe_tool"]["data"] == {"observed": True}
    rows = _read_audit_rows(cs.AUDIT_PATH)
    assert any(r["action"] == "test_observe_tool" and r["outcome"] == "executed" for r in rows)


def test_cycle_is_appended_to_the_mind_window_ledger(tmp_path, monkeypatch):
    side_effects = []
    stdout = _diagnosis_stdout([{"action": "test_op_action", "params": {}, "rationale": "x"}])
    resident = _build_loop(tmp_path, monkeypatch, side_effects=side_effects, stdout=stdout, autonomy_enabled=False)

    result = resident.run_cycle(actor="test-agent")

    lines = (tmp_path / "loop_cycles.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    row = json.loads(lines[0])
    assert row["cycle_id"] == result.cycle_id
    assert row["outcomes"][0]["shadow"] is True


def test_read_mind_window_reports_pending_operator_proposals(tmp_path, monkeypatch):
    _isolate_audit(tmp_path, monkeypatch)
    state_path = _fresh_state(tmp_path, halted_lanes={"equity": True})
    autonomy_path = tmp_path / "loop_autonomy.json"
    engine = policy.default_engine()
    stdout = _diagnosis_stdout([{"action": "unhalt_lane", "params": {"lane": "equity", "state_path": str(state_path)}, "rationale": "please unhalt"}])
    cycles_path = tmp_path / "loop_cycles.jsonl"
    resident = agent_loop.ResidentLoop(
        engine=engine, cycles_path=cycles_path, autonomy_path=autonomy_path,
        observe_tool_names=(), shadow_lanes=(),
        cli_resolver=lambda _bin: "/usr/bin/fake-claude",
        subprocess_runner=_fake_subprocess_runner(stdout),
    )
    resident.run_cycle(actor="test-agent")

    snapshot = agent_loop.read_mind_window(cycles_path=cycles_path, autonomy_path=autonomy_path)

    assert snapshot["has_run"] is True
    assert snapshot["autonomy_enabled"] is False
    assert len(snapshot["pending_operator_proposals"]) == 1
    assert snapshot["pending_operator_proposals"][0]["action"] == "unhalt_lane"


def test_read_mind_window_honest_empty_state_before_first_cycle(tmp_path):
    snapshot = agent_loop.read_mind_window(
        cycles_path=tmp_path / "no_such_file.jsonl", autonomy_path=tmp_path / "no_such_flag.json",
    )
    assert snapshot["has_run"] is False
    assert snapshot["autonomy_enabled"] is False
    assert snapshot["recent_cycles"] == []
    assert snapshot["pending_operator_proposals"] == []


# ---------------------------------------------------------------------------
# Structural: the resident loop's default wiring only ever sees real
# read-only tools as OPERATIONAL, never as escalation/de-escalation.
# ---------------------------------------------------------------------------


def test_default_resident_loop_wires_all_registered_tools_as_operational(tmp_path, monkeypatch):
    _isolate_audit(tmp_path, monkeypatch)
    _fresh_state(tmp_path)
    resident = agent_loop.ResidentLoop()
    from src.agent_runtime.tools.registry import TOOLS

    for name in TOOLS:
        assert resident.engine.classify(name) is policy.ActionTier.OPERATIONAL


# ---------------------------------------------------------------------------
# DIAGNOSE failure observability (2026-07-20). The CLI prints its failure
# reason to stdout/stderr; the loop used to discard both on a non-zero exit,
# which made a 38-hour credential expiry look identical to any other crash --
# 1,446 of 2,187 recorded cycles carried only "claude exited 1". These tests
# pin the reason reaching the cycle record, and pin reasoning_ok as the honest
# liveness signal (cli_available deliberately stays True: the binary WAS
# found, and 8 consumers depend on that meaning).
# ---------------------------------------------------------------------------


def _runner_with_streams(stdout: str, stderr: str, returncode: int):
    def _run(*args, **kwargs):
        return subprocess.CompletedProcess(args=args, returncode=returncode, stdout=stdout, stderr=stderr)

    return _run


def _loop_with_runner(tmp_path, monkeypatch, runner):
    _isolate_audit(tmp_path, monkeypatch)
    _fresh_state(tmp_path)
    engine = policy.PolicyEngine(_test_registry([]), audit_fn=_audit_record_for(tmp_path))
    return agent_loop.ResidentLoop(
        engine=engine, cycles_path=tmp_path / "loop_cycles.jsonl",
        autonomy_path=tmp_path / "loop_autonomy.json",
        observe_tool_names=(), shadow_lanes=(),
        cli_resolver=lambda _bin: "/usr/bin/fake-claude",
        subprocess_runner=runner,
    )


def test_expired_credentials_are_classified_and_surfaced(tmp_path, monkeypatch):
    """The exact live failure: CLI exits 1 printing a 401 to stdout."""
    resident = _loop_with_runner(tmp_path, monkeypatch, _runner_with_streams(
        stdout="API Error: 401 OAuth access token has expired", stderr="", returncode=1,
    ))

    result = resident.run_cycle(actor="test-agent")

    assert result.reasoning_ok is False
    assert result.cli_available is True  # binary found -- unchanged meaning
    assert result.diagnosis["degraded"] is True
    assert result.diagnosis["failure_kind"] == "auth"
    assert result.diagnosis["returncode"] == 1
    assert "401" in result.diagnosis["failure_detail"]
    assert "re-authenticate" in result.diagnosis["reasoning"]


def test_non_auth_failure_classified_as_error_and_stderr_captured(tmp_path, monkeypatch):
    resident = _loop_with_runner(tmp_path, monkeypatch, _runner_with_streams(
        stdout="", stderr="Segmentation fault", returncode=139,
    ))

    result = resident.run_cycle(actor="test-agent")

    assert result.reasoning_ok is False
    assert result.diagnosis["failure_kind"] == "error"
    assert "Segmentation fault" in result.diagnosis["failure_detail"]


def test_failure_detail_is_bounded_so_it_cannot_bloat_the_cycle_ledger(tmp_path, monkeypatch):
    resident = _loop_with_runner(tmp_path, monkeypatch, _runner_with_streams(
        stdout="x" * 50_000, stderr="", returncode=1,
    ))

    result = resident.run_cycle(actor="test-agent")

    assert len(result.diagnosis["failure_detail"]) < 3000
    assert "truncated" in result.diagnosis["failure_detail"]


def test_healthy_cycle_reports_reasoning_ok_and_no_degraded_marker(tmp_path, monkeypatch):
    stdout = _diagnosis_stdout([])
    resident = _build_loop(tmp_path, monkeypatch, side_effects=[], stdout=stdout, autonomy_enabled=False)

    result = resident.run_cycle(actor="test-agent")

    assert result.reasoning_ok is True
    assert "degraded" not in result.diagnosis


def test_missing_cli_is_degraded_and_not_reasoning_ok(tmp_path, monkeypatch):
    _isolate_audit(tmp_path, monkeypatch)
    _fresh_state(tmp_path)
    engine = policy.PolicyEngine(_test_registry([]), audit_fn=_audit_record_for(tmp_path))
    resident = agent_loop.ResidentLoop(
        engine=engine, cycles_path=tmp_path / "loop_cycles.jsonl",
        autonomy_path=tmp_path / "loop_autonomy.json",
        observe_tool_names=(), shadow_lanes=(),
        cli_resolver=lambda _bin: None,
        subprocess_runner=lambda *a, **kw: pytest.fail("must not spawn when CLI missing"),
    )

    result = resident.run_cycle(actor="test-agent")

    assert result.reasoning_ok is False
    assert result.diagnosis["failure_kind"] == "cli_missing"


def test_reasoning_ok_is_persisted_to_the_cycle_ledger(tmp_path, monkeypatch):
    """A monitor must be able to alarm on a degraded streak from disk alone."""
    resident = _loop_with_runner(tmp_path, monkeypatch, _runner_with_streams(
        stdout="401 unauthorized", stderr="", returncode=1,
    ))

    resident.run_cycle(actor="test-agent")

    rows = [json.loads(line) for line in (tmp_path / "loop_cycles.jsonl").read_text().splitlines() if line.strip()]
    assert rows[-1]["reasoning_ok"] is False
    assert rows[-1]["diagnosis"]["failure_kind"] == "auth"


def test_captured_cli_output_is_redacted_before_it_reaches_disk(tmp_path, monkeypatch):
    """An auth failure is exactly when a tool may echo the rejected credential.

    loop_cycles.jsonl is append-only, so a leaked token there is unrecoverable.
    """
    leaky = (
        "Failed to authenticate. api_key=sk-abcdefghijklmnop1234 "
        "Authorization: Bearer eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiIxMjM0NTY3ODkwIn0.dozjgNryP4J3jVmNHl0w5N "
        "token: supersecretvalue123"
    )
    resident = _loop_with_runner(tmp_path, monkeypatch, _runner_with_streams(
        stdout=leaky, stderr="", returncode=1,
    ))

    result = resident.run_cycle(actor="test-agent")
    blob = json.dumps(result.diagnosis)

    assert "sk-abcdefghijklmnop1234" not in blob
    assert "supersecretvalue123" not in blob
    assert "eyJhbGciOiJIUzI1NiJ9" not in blob
    assert "[REDACTED]" in blob
    # ...and it is still legible enough to diagnose.
    assert "Failed to authenticate" in blob
    assert result.diagnosis["failure_kind"] == "auth"

    # The persisted ledger row must be clean too, not just the in-memory result.
    ledger = (tmp_path / "loop_cycles.jsonl").read_text()
    assert "sk-abcdefghijklmnop1234" not in ledger
    assert "supersecretvalue123" not in ledger


def test_redaction_runs_before_truncation_so_secrets_cannot_survive_the_cutoff(tmp_path, monkeypatch):
    resident = _loop_with_runner(tmp_path, monkeypatch, _runner_with_streams(
        stdout="x" * 40_000 + " sk-tailingsecretvalue999", stderr="", returncode=1,
    ))

    result = resident.run_cycle(actor="test-agent")

    assert "sk-tailingsecretvalue999" not in json.dumps(result.diagnosis)
