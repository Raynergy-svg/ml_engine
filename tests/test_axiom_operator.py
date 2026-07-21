from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path
from unittest.mock import patch

from src.axiom_operator.policy import ActionPolicy
from src.axiom_operator.runner import AxiomOperator
from src.axiom_operator.session import read_operator_snapshot


def completed(stdout: str = "", stderr: str = "", returncode: int = 0) -> subprocess.CompletedProcess[str]:
    return subprocess.CompletedProcess(args=["claude"], returncode=returncode, stdout=stdout, stderr=stderr)


def test_policy_blocks_human_required_actions() -> None:
    policy = ActionPolicy()

    for action in ["unhalt", "start_loop", "set_gross_leverage", "flatten", "promote_model", "disable_safety_gate"]:
        decision = policy.evaluate(action)
        assert decision.allowed is False
        assert decision.requires_human is True


def test_policy_permits_safe_auto_actions() -> None:
    policy = ActionPolicy()

    assert policy.evaluate("diagnose").allowed is True
    assert policy.evaluate("recheck").allowed is True
    assert policy.evaluate("halt_lane", {"lane": "oanda_fx"}).allowed is True
    assert policy.evaluate("stop_loop", {"loop": "tier7"}).allowed is True


def test_policy_rejects_unknown_lane_and_loop() -> None:
    policy = ActionPolicy()

    assert policy.evaluate("halt_lane", {"lane": "live_account"}).allowed is False
    assert policy.evaluate("stop_loop", {"loop": "anything"}).allowed is False


def test_operator_refuses_api_key_by_default(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
    operator = AxiomOperator(
        project_root=tmp_path,
        session_path=tmp_path / "operator_session.json",
        decisions_path=tmp_path / "operator_decisions.jsonl",
        subprocess_runner=lambda **_: completed(),
    )

    result = operator.run_once()

    assert result.ok is False
    assert result.status == "held"
    assert result.session["api_key_refused"] is True
    assert "API billing" in (result.error or "")


def test_operator_writes_and_resumes_handoff(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    stdout = """
<axiom-operator>{
  "summary": "Trend lane quiet but Tier 7 is fine",
  "reasoning": "I checked the current incident context and only need a recheck.",
  "action": "recheck",
  "params": {},
  "tools_called": ["mcp__buddy__health_check"],
  "open_incidents": ["trend log quiet"],
  "blocked_reasons": [],
  "handoff_summary": "Trend log quiet; recheck next before touching controls.",
  "next_action": "recheck"
}</axiom-operator>
"""
    calls = []

    def runner(*args, **kwargs):
        calls.append(args[0])
        return completed(stdout=stdout)

    operator = AxiomOperator(
        project_root=tmp_path,
        session_path=tmp_path / "operator_session.json",
        decisions_path=tmp_path / "operator_decisions.jsonl",
        subprocess_runner=runner,
        effect_executor=lambda action, params, policy: {"applied": True, "result": "test_effect"},
    )
    with patch("shutil.which", return_value="/usr/local/bin/claude"):
        result = operator.run_once(observation={"incident": "trend quiet"})

    assert result.ok is True
    assert result.status == "acted"
    assert result.session["epoch"] == 1
    assert result.session["handoff_summary"] == "Trend log quiet; recheck next before touching controls."
    assert result.session["last_tools"] == ["mcp__buddy__health_check"]
    assert result.session["last_effect"] == {"applied": True, "result": "test_effect"}
    assert "--allowedTools" in calls[0]
    assert "--mcp-config" not in calls[0]  # tmp_path has no mcp.json

    snapshot = read_operator_snapshot(
        session_path=tmp_path / "operator_session.json",
        decisions_path=tmp_path / "operator_decisions.jsonl",
    )
    assert snapshot["has_run"] is True
    assert snapshot["last_decision"]["action"] == "recheck"


def test_operator_rolls_over_at_context_threshold(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    session_path = tmp_path / "operator_session.json"
    session_path.write_text(json.dumps({
        "session_id": "test",
        "epoch": 0,
        "context_budget_chars": 200,
        "rollover_threshold": 0.8,
        "handoff_summary": "seed",
    }), encoding="utf-8")
    stdout = "<axiom-operator>{\"action\":\"diagnose\",\"handoff_summary\":\"compact next epoch\"}</axiom-operator>" + ("x" * 300)
    operator = AxiomOperator(
        project_root=tmp_path,
        session_path=session_path,
        decisions_path=tmp_path / "operator_decisions.jsonl",
        subprocess_runner=lambda *_, **__: completed(stdout=stdout),
        effect_executor=lambda action, params, policy: {"applied": True, "result": "test_effect"},
    )

    with patch("shutil.which", return_value="/usr/local/bin/claude"):
        result = operator.run_once()

    assert result.status == "handoff"
    assert result.session["context_used_pct"] >= 0.8
    assert result.decision["rollover"] is True


def test_operator_failed_claude_is_held_and_non_trading(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    operator = AxiomOperator(
        project_root=tmp_path,
        session_path=tmp_path / "operator_session.json",
        decisions_path=tmp_path / "operator_decisions.jsonl",
        subprocess_runner=lambda *_, **__: completed(stdout="usage limit reached|1999999999", returncode=1),
    )

    with patch("shutil.which", return_value="/usr/local/bin/claude"):
        result = operator.run_once()

    assert result.ok is False
    assert result.status == "held"
    assert "claude exited 1" in (result.error or "")
    assert result.action == "observe"


def test_operator_applies_safe_effect_when_policy_allows(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    stdout = """
<axiom-operator>{
  "summary": "Trend lane needs containment",
  "reasoning": "Critical alert is active, so pause the OANDA lane.",
  "action": "halt_lane",
  "params": {"lane": "oanda_fx"},
  "tools_called": ["mcp__buddy__health_check"],
  "open_incidents": ["critical alert"],
  "blocked_reasons": [],
  "handoff_summary": "OANDA lane pause requested.",
  "next_action": "recheck"
}</axiom-operator>
"""
    seen = {}

    def effect(action, params, policy):
        seen["action"] = action
        seen["params"] = dict(params)
        seen["allowed"] = policy.allowed
        return {"applied": True, "result": "halted"}

    operator = AxiomOperator(
        project_root=tmp_path,
        session_path=tmp_path / "operator_session.json",
        decisions_path=tmp_path / "operator_decisions.jsonl",
        subprocess_runner=lambda *_, **__: completed(stdout=stdout),
        effect_executor=effect,
    )

    with patch("shutil.which", return_value="/usr/local/bin/claude"):
        result = operator.run_once()

    assert result.ok is True
    assert result.action == "halt_lane"
    assert seen == {"action": "halt_lane", "params": {"lane": "oanda_fx"}, "allowed": True}
    assert result.decision["effect"] == {"applied": True, "result": "halted"}


def test_operator_holds_when_safe_effect_not_applied(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    stdout = "<axiom-operator>{\"action\":\"halt_lane\",\"params\":{\"lane\":\"oanda_fx\"}}</axiom-operator>"
    operator = AxiomOperator(
        project_root=tmp_path,
        session_path=tmp_path / "operator_session.json",
        decisions_path=tmp_path / "operator_decisions.jsonl",
        subprocess_runner=lambda *_, **__: completed(stdout=stdout),
        effect_executor=lambda action, params, policy: {"applied": False, "reason": "control disabled"},
    )

    with patch("shutil.which", return_value="/usr/local/bin/claude"):
        result = operator.run_once()

    assert result.ok is False
    assert result.status == "held"  # The real-world effect did not land.
    assert "control disabled" in result.decision["blocked_reasons"]


def test_operator_holds_when_safe_effect_raises(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    stdout = "<axiom-operator>{\"action\":\"halt_lane\",\"params\":{\"lane\":\"oanda_fx\"}}</axiom-operator>"

    def effect(*_args, **_kwargs):
        raise RuntimeError("control router refused")

    operator = AxiomOperator(
        project_root=tmp_path,
        session_path=tmp_path / "operator_session.json",
        decisions_path=tmp_path / "operator_decisions.jsonl",
        subprocess_runner=lambda *_, **__: completed(stdout=stdout),
        effect_executor=effect,
    )

    with patch("shutil.which", return_value="/usr/local/bin/claude"):
        result = operator.run_once()

    assert result.ok is False
    assert result.status == "held"
    assert result.decision["effect"]["reason"] == "effect_error: RuntimeError"
    assert "effect_error: RuntimeError" in result.decision["blocked_reasons"]
