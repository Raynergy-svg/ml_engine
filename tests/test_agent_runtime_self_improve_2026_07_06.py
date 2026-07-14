"""AXIOM Agent Runtime — SELF_IMPROVE tier structural-boundary tests.

Proves, against real classes, real files, and a real (tmp) git repo — no
unittest.mock:
  (a) the structural denylist refuses every listed path, unconditionally
  (b) an unlisted path is refused too (fail-closed default-deny)
  (c) allowlisted paths pass the check cleanly
  (d) ESCALATION actions (arm/unhalt/size-up/promote/new-exposure/code-change)
      stay proposal-only even with autonomy_enabled=True and SELF_IMPROVE
      merged into the registry
  (e) the gated-edit lifecycle reverts (no commit) on a failing test/verify
      step, and commits (scoped, reversible) on a full pass
  (f) no self-improve execute function accepts a caller-supplied path/target
"""
from __future__ import annotations

import inspect
import json
import subprocess
import sys

import pytest

from src.agent_runtime import policy
from src.agent_runtime import self_improve as si


# ---------------------------------------------------------------------------
# (a)/(b)/(c) structural boundary
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("bad_path", si.STRUCTURAL_DENYLIST)
def test_every_denylisted_path_is_refused(bad_path):
    target = bad_path if not bad_path.endswith("/") else bad_path + "some_file.py"
    with pytest.raises(policy.PolicyDenied, match="structural denylist"):
        si._check_target_paths([target])


def test_unlisted_path_is_refused_fail_closed():
    with pytest.raises(policy.PolicyDenied, match="not on the explicit allowlist"):
        si._check_target_paths(["some/random/never/listed/file.py"])


@pytest.mark.parametrize("good_path", si.ALLOWLIST)
def test_allowlisted_paths_pass_the_check(good_path):
    si._check_target_paths([good_path])  # must not raise


def test_denylist_and_allowlist_do_not_overlap():
    for good in si.ALLOWLIST:
        for bad in si.STRUCTURAL_DENYLIST:
            assert not good.startswith(bad), f"{good!r} is shadowed by denylist entry {bad!r}"


def test_denylist_wins_even_if_a_path_were_also_allowlisted(monkeypatch):
    """Defense in depth: if ALLOWLIST were ever misconfigured to include a
    denylisted path, the denylist check (which runs first, unconditionally)
    must still refuse it -- there is no code path where allowlist membership
    overrides the denylist."""
    monkeypatch.setattr(si, "ALLOWLIST", si.ALLOWLIST + ("src/scanner/execution.py",))
    with pytest.raises(policy.PolicyDenied, match="structural denylist"):
        si._check_target_paths(["src/scanner/execution.py"])


# ---------------------------------------------------------------------------
# (d) ESCALATION stays proposal-only even with SELF_IMPROVE merged + autonomy ON
# ---------------------------------------------------------------------------

_ESCALATION_ACTION_NAMES = (
    "unhalt_lane",
    "arm_live_gate",
    "increase_gross_leverage",
    "promote_model",
    "enable_new_exposure",
    "change_strategy_or_code",
)


@pytest.mark.parametrize("action_name", _ESCALATION_ACTION_NAMES)
def test_escalation_actions_remain_proposal_only_with_self_improve_registered(action_name, tmp_path, monkeypatch):
    from dashboard.server import control_safety as cs

    monkeypatch.setattr(cs, "AUDIT_PATH", tmp_path / "control_audit.jsonl")
    engine = policy.default_engine(si.SELF_IMPROVE_ACTIONS)

    assert engine.classify(action_name) is policy.ActionTier.ESCALATION

    result = engine.submit(action_name, {}, actor="test-agent")

    assert result.executed is False
    assert result.proposal is not None
    assert result.proposal.tier is policy.ActionTier.ESCALATION


def test_self_improve_actions_are_correctly_tiered_and_carry_execute():
    for name, spec in si.SELF_IMPROVE_ACTIONS.items():
        assert spec.tier is policy.ActionTier.SELF_IMPROVE, name
        assert spec.execute is not None, name


def test_self_improve_actions_carry_a_params_hint_the_diagnosing_llm_will_see():
    """Reproduces a real incident: the diagnosing LLM proposed
    self_improve_purge_agent_weight_keys with param names it invented
    ("tiers", then "regimes") because the prompt only showed the bare action
    name -- execute()'s strict signature rejected the unexpected kwarg and the
    action was denied every cycle. params_hint is how loop.py's _build_prompt
    tells the LLM the real contract; every self-improve action must set one."""
    from src.agent_runtime import loop as agent_loop

    for name, spec in si.SELF_IMPROVE_ACTIONS.items():
        assert spec.params_hint, f"{name} has no params_hint -- the diagnosing LLM will guess"

    engine = policy.default_engine(si.SELF_IMPROVE_ACTIONS)
    resident = agent_loop.ResidentLoop.__new__(agent_loop.ResidentLoop)
    resident.engine = engine
    prompt = resident._build_prompt({})
    assert "keys" in prompt
    assert "lesson_id" in prompt
    assert "self_improve_purge_agent_weight_keys" in prompt


def test_every_self_improve_action_routes_through_the_gated_edit_lifecycle():
    """Convention alone doesn't stop a future action from skipping the gate --
    prove it structurally the same way policy.py's own test suite proves no
    builtin action contains a forbidden call: inspect the actual source. If a
    future SELF_IMPROVE action's execute function doesn't call
    ``_run_gated_edit(``, this test catches it immediately, before it ships."""
    for name, spec in si.SELF_IMPROVE_ACTIONS.items():
        source = inspect.getsource(spec.execute)
        assert "_run_gated_edit(" in source, (
            f"{name}.execute does not route through _run_gated_edit -- it would bypass "
            "the test+verify+commit-or-revert gate entirely"
        )


def test_self_improve_actions_do_not_collide_with_builtin_registry():
    registry = policy.build_registry(si.SELF_IMPROVE_ACTIONS)
    for name in si.SELF_IMPROVE_ACTIONS:
        assert registry[name].tier is policy.ActionTier.SELF_IMPROVE
    for name in policy.BUILTIN_ACTIONS:
        assert registry[name] is policy.BUILTIN_ACTIONS[name]


# ---------------------------------------------------------------------------
# (f) no execute function accepts a caller-supplied path/target
# ---------------------------------------------------------------------------


def test_purge_action_rejects_a_path_kwarg_it_was_never_given():
    with pytest.raises(TypeError):
        si._self_improve_purge_agent_weight_keys(path="src/scanner/execution.py")


def test_lessons_action_rejects_a_path_kwarg_it_was_never_given():
    with pytest.raises(TypeError):
        si._self_improve_add_lessons_recall_row(lesson_id="L-024", path="src/scanner/execution.py")


# ---------------------------------------------------------------------------
# (g) pure apply-function correctness (real tmp files, no mocks)
# ---------------------------------------------------------------------------


def test_purge_removes_orphan_key_and_preserves_canonical_schema(tmp_path, monkeypatch):
    monkeypatch.setattr(si, "_repo_root", lambda: tmp_path)
    weights_path = tmp_path / si._AGENT_WEIGHTS_PATH
    weights_path.parent.mkdir(parents=True)
    regime = {name: 1.0 for name in si._CANONICAL_AGENT_NAMES}
    regime["invalid_agent_xyz"] = 0.15
    data = {
        "NORMAL": dict(regime), "_global": dict(regime),
        "EXTREME": {n: 1.0 for n in si._CANONICAL_AGENT_NAMES},
        "HIGH": {n: 1.0 for n in si._CANONICAL_AGENT_NAMES},
        "LOW": {n: 1.0 for n in si._CANONICAL_AGENT_NAMES},
        "_meta": {"weight_history_invalid_agent_xyz": [0.1, 0.2], "weight_variance_invalid_agent_xyz": 0.01},
    }
    weights_path.write_text(json.dumps(data), encoding="utf-8")

    result = si._apply_purge_agent_weight_keys(["invalid_agent_xyz"])

    saved = json.loads(weights_path.read_text(encoding="utf-8"))
    assert "invalid_agent_xyz" not in saved["NORMAL"]
    assert "invalid_agent_xyz" not in saved["_global"]
    assert "weight_history_invalid_agent_xyz" not in saved["_meta"]
    assert set(saved["NORMAL"]) == si._CANONICAL_AGENT_NAMES
    assert "NORMAL" in result["purged"]


def test_purge_refuses_if_it_would_remove_a_canonical_agent(tmp_path, monkeypatch):
    monkeypatch.setattr(si, "_repo_root", lambda: tmp_path)
    weights_path = tmp_path / si._AGENT_WEIGHTS_PATH
    weights_path.parent.mkdir(parents=True)
    data = {"NORMAL": {n: 1.0 for n in si._CANONICAL_AGENT_NAMES}}
    weights_path.write_text(json.dumps(data), encoding="utf-8")

    with pytest.raises(policy.PolicyDenied, match="safety invariant violated"):
        si._apply_purge_agent_weight_keys(["trend"])  # trend is canonical, not an orphan


def test_lessons_recall_row_derives_trigger_from_disk_and_is_idempotent(tmp_path, monkeypatch):
    monkeypatch.setattr(si, "_repo_root", lambda: tmp_path)
    lessons_path = tmp_path / si._LESSONS_PATH
    lessons_path.parent.mkdir(parents=True)
    lessons_path.write_text(
        "# LESSONS\n\n"
        "## Recall-trigger index\n\n"
        "| If your task touches… | Recall |\n"
        "|---|---|\n"
        "| existing thing | **L-001** existing |\n\n"
        "---\n\n"
        "## L-024 — a fake lesson   [ACTIVE]\n"
        "- Trigger: doing the risky thing that should recall this lesson\n"
        "- Root cause: reasons\n"
        "- Rule: don't do the thing\n"
        "- Scope: everywhere\n"
        "- Source: a test\n",
        encoding="utf-8",
    )

    result = si._apply_add_lessons_recall_row("L-024")
    text = lessons_path.read_text(encoding="utf-8")
    assert "**L-024**" in text
    assert "doing the risky thing that should recall this lesson" in text
    assert result["lesson_id"] == "L-024"

    # idempotent -- calling again must not duplicate the row
    result2 = si._apply_add_lessons_recall_row("L-024")
    assert result2.get("already_present") is True
    assert text.count("**L-024**") == lessons_path.read_text(encoding="utf-8").count("**L-024**")


def test_lessons_recall_row_refuses_nonexistent_lesson(tmp_path, monkeypatch):
    monkeypatch.setattr(si, "_repo_root", lambda: tmp_path)
    lessons_path = tmp_path / si._LESSONS_PATH
    lessons_path.parent.mkdir(parents=True)
    lessons_path.write_text("# LESSONS\n\n## Recall-trigger index\n\n| x | y |\n|---|---|\n\n---\n", encoding="utf-8")

    with pytest.raises(policy.PolicyDenied, match="no entry in LESSONS.md"):
        si._apply_add_lessons_recall_row("L-999")


# ---------------------------------------------------------------------------
# (e) full gated lifecycle: revert-on-failure, commit-on-success (real git)
# ---------------------------------------------------------------------------


def _init_repo(tmp_path):
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=tmp_path, check=True)
    subprocess.run(["git", "config", "user.name", "Test"], cwd=tmp_path, check=True)
    (tmp_path / "README.md").write_text("seed\n", encoding="utf-8")
    subprocess.run(["git", "add", "README.md"], cwd=tmp_path, check=True)
    subprocess.run(["git", "commit", "-q", "-m", "seed"], cwd=tmp_path, check=True)


def _write_stand_in_gates(tmp_path, *, verify_ok=True, risk_ok=True):
    gate_dir = tmp_path / ".claude" / "loop"
    tools_dir = tmp_path / ".claude" / "tools"
    gate_dir.mkdir(parents=True, exist_ok=True)
    tools_dir.mkdir(parents=True, exist_ok=True)
    (gate_dir / "verify_gate.py").write_text(
        f"import sys\nsys.exit({0 if verify_ok else 2})\n", encoding="utf-8",
    )
    (tools_dir / "risk_monitor.sh").write_text(
        f"#!/usr/bin/env bash\nexit {0 if risk_ok else 2}\n", encoding="utf-8",
    )


def test_gated_edit_reverts_and_does_not_commit_when_test_command_fails(tmp_path, monkeypatch):
    monkeypatch.setattr(si, "_repo_root", lambda: tmp_path)
    monkeypatch.setattr(si, "ALLOWLIST", si.ALLOWLIST + ("data.json",))
    _init_repo(tmp_path)
    _write_stand_in_gates(tmp_path)
    target = tmp_path / "data.json"
    target.write_text('{"a": 1}\n', encoding="utf-8")

    log_before = subprocess.run(["git", "log", "--oneline"], cwd=tmp_path, capture_output=True, text=True).stdout

    def _apply():
        target.write_text('{"a": 2}\n', encoding="utf-8")
        return {"changed": True}

    with pytest.raises(policy.PolicyDenied, match="REVERTED"):
        si._run_gated_edit(
            action_name="test_action",
            target_paths=["data.json"],
            apply_fn=_apply,
            test_cmd=[sys.executable, "-c", "import sys; sys.exit(1)"],
        )

    assert target.read_text(encoding="utf-8") == '{"a": 1}\n'  # restored
    log_after = subprocess.run(["git", "log", "--oneline"], cwd=tmp_path, capture_output=True, text=True).stdout
    assert log_after == log_before  # no new commit


def test_gated_edit_already_present_is_a_clean_no_op_not_a_failed_commit(tmp_path, monkeypatch):
    """apply_fn signaling {"already_present": True} (no actual file change) must
    short-circuit before _git_commit -- committing a clean tree fails, which
    would previously trip _revert() and misreport an idempotent no-op action
    as a REVERTED failure."""
    monkeypatch.setattr(si, "_repo_root", lambda: tmp_path)
    monkeypatch.setattr(si, "ALLOWLIST", si.ALLOWLIST + ("data.json",))
    _init_repo(tmp_path)
    _write_stand_in_gates(tmp_path)
    target = tmp_path / "data.json"
    target.write_text('{"a": 1}\n', encoding="utf-8")
    subprocess.run(["git", "add", "data.json"], cwd=tmp_path, check=True)
    subprocess.run(["git", "commit", "-q", "-m", "seed data"], cwd=tmp_path, check=True)

    log_before = subprocess.run(
        ["git", "log", "--oneline"], cwd=tmp_path, capture_output=True, text=True
    ).stdout

    def _apply_no_op():
        return {"already_present": True, "lesson_id": "L-024"}

    result = si._run_gated_edit(
        action_name="test_action",
        target_paths=["data.json"],
        apply_fn=_apply_no_op,
        # A failing test_cmd proves this path never reaches test execution --
        # if it did, the test would raise PolicyDenied instead of returning.
        test_cmd=[sys.executable, "-c", "import sys; sys.exit(1)"],
    )

    assert result["committed"] is False
    assert result["reverted"] is False
    assert result["already_present"] is True
    assert target.read_text(encoding="utf-8") == '{"a": 1}\n'  # untouched
    log_after = subprocess.run(
        ["git", "log", "--oneline"], cwd=tmp_path, capture_output=True, text=True
    ).stdout
    assert log_after == log_before  # no new commit attempted


def test_gated_edit_commits_scoped_and_reversible_on_full_pass(tmp_path, monkeypatch):
    monkeypatch.setattr(si, "_repo_root", lambda: tmp_path)
    monkeypatch.setattr(si, "ALLOWLIST", si.ALLOWLIST + ("data.json",))
    _init_repo(tmp_path)
    _write_stand_in_gates(tmp_path)
    target = tmp_path / "data.json"
    target.write_text('{"a": 1}\n', encoding="utf-8")
    subprocess.run(["git", "add", "data.json"], cwd=tmp_path, check=True)
    subprocess.run(["git", "commit", "-q", "-m", "seed data"], cwd=tmp_path, check=True)

    # a concurrent, unrelated tracked file changes on disk but is NOT staged --
    # the commit must stay scoped to `data.json` and not sweep this in.
    (tmp_path / "README.md").write_text("seed\nconcurrent daemon write\n", encoding="utf-8")

    def _apply():
        target.write_text('{"a": 2}\n', encoding="utf-8")
        return {"changed": True}

    result = si._run_gated_edit(
        action_name="test_action",
        target_paths=["data.json"],
        apply_fn=_apply,
        test_cmd=[sys.executable, "-c", "pass"],
    )

    assert result["committed"] is True
    assert result["commit_sha"]
    assert target.read_text(encoding="utf-8") == '{"a": 2}\n'

    show = subprocess.run(
        ["git", "show", "--name-only", "--pretty=format:", "HEAD"],
        cwd=tmp_path, capture_output=True, text=True,
    ).stdout
    changed_files = [ln for ln in show.splitlines() if ln.strip()]
    assert changed_files == ["data.json"]  # scoped commit, README.md not swept in

    status = subprocess.run(["git", "status", "--porcelain"], cwd=tmp_path, capture_output=True, text=True).stdout
    assert "README.md" in status  # concurrent write is still there, uncommitted, untouched


def test_gated_edit_persists_designated_runtime_state_after_all_gates(tmp_path, monkeypatch):
    """A verified agent-weight correction is runtime state, not a source commit.

    This uses a real temporary Git repository and real stand-in safety scripts:
    it proves the designated runtime target survives the full test + gate
    lifecycle without becoming tracked or sweeping unrelated files into Git.
    """
    monkeypatch.setattr(si, "_repo_root", lambda: tmp_path)
    _init_repo(tmp_path)
    _write_stand_in_gates(tmp_path)
    target = tmp_path / "trained_data" / "models" / "agent_weights.json"
    target.parent.mkdir(parents=True)
    target.write_text('{"invalid_agent_xyz": 0.15}\n', encoding="utf-8")

    log_before = subprocess.run(
        ["git", "log", "--oneline"], cwd=tmp_path, capture_output=True, text=True,
    ).stdout

    def _apply():
        target.write_text('{"canonical": 1.0}\n', encoding="utf-8")
        return {"purged": {"_meta": ["weight_history_invalid_agent_xyz"]}}

    result = si._run_gated_edit(
        action_name="test_runtime_state_action",
        target_paths=["trained_data/models/agent_weights.json"],
        apply_fn=_apply,
        test_cmd=[sys.executable, "-c", "pass"],
    )

    assert result["committed"] is False
    assert result["persisted_runtime_state"] is True
    assert result["reverted"] is False
    assert result["tests_passed"] is True
    assert result["verify_gate_passed"] is True
    assert result["risk_monitor_green"] is True
    assert target.read_text(encoding="utf-8") == '{"canonical": 1.0}\n'

    log_after = subprocess.run(
        ["git", "log", "--oneline"], cwd=tmp_path, capture_output=True, text=True,
    ).stdout
    assert log_after == log_before
    tracked = subprocess.run(
        ["git", "ls-files", "--error-unmatch", "trained_data/models/agent_weights.json"],
        cwd=tmp_path, capture_output=True, text=True,
    )
    assert tracked.returncode != 0


def test_gated_edit_reverts_when_verify_gate_fails(tmp_path, monkeypatch):
    monkeypatch.setattr(si, "_repo_root", lambda: tmp_path)
    monkeypatch.setattr(si, "ALLOWLIST", si.ALLOWLIST + ("data.json",))
    _init_repo(tmp_path)
    _write_stand_in_gates(tmp_path, verify_ok=False)
    target = tmp_path / "data.json"
    target.write_text('{"a": 1}\n', encoding="utf-8")

    def _apply():
        target.write_text('{"a": 2}\n', encoding="utf-8")
        return {}

    with pytest.raises(policy.PolicyDenied, match="verify_gate.py FAIL"):
        si._run_gated_edit(
            action_name="test_action", target_paths=["data.json"],
            apply_fn=_apply, test_cmd=[sys.executable, "-c", "pass"],
        )
    assert target.read_text(encoding="utf-8") == '{"a": 1}\n'


def test_gated_edit_reverts_when_risk_monitor_alarms(tmp_path, monkeypatch):
    monkeypatch.setattr(si, "_repo_root", lambda: tmp_path)
    monkeypatch.setattr(si, "ALLOWLIST", si.ALLOWLIST + ("data.json",))
    _init_repo(tmp_path)
    _write_stand_in_gates(tmp_path, risk_ok=False)
    target = tmp_path / "data.json"
    target.write_text('{"a": 1}\n', encoding="utf-8")

    def _apply():
        target.write_text('{"a": 2}\n', encoding="utf-8")
        return {}

    with pytest.raises(policy.PolicyDenied, match="risk_monitor.sh ALARM"):
        si._run_gated_edit(
            action_name="test_action", target_paths=["data.json"],
            apply_fn=_apply, test_cmd=[sys.executable, "-c", "pass"],
        )
    assert target.read_text(encoding="utf-8") == '{"a": 1}\n'


def test_gated_edit_reverts_and_reraises_when_apply_fn_itself_denies(tmp_path, monkeypatch):
    monkeypatch.setattr(si, "_repo_root", lambda: tmp_path)
    monkeypatch.setattr(si, "ALLOWLIST", si.ALLOWLIST + ("data.json",))
    _init_repo(tmp_path)
    target = tmp_path / "data.json"
    target.write_text('{"a": 1}\n', encoding="utf-8")

    def _apply():
        target.write_text('{"a": 2}\n', encoding="utf-8")
        raise policy.PolicyDenied("safety invariant violated inside apply_fn")

    with pytest.raises(policy.PolicyDenied, match="REVERTED"):
        si._run_gated_edit(
            action_name="test_action", target_paths=["data.json"],
            apply_fn=_apply, test_cmd=[sys.executable, "-c", "pass"],
        )
    assert target.read_text(encoding="utf-8") == '{"a": 1}\n'
