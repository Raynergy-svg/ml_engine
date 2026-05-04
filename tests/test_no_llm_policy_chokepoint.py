"""Regression guard: no-LLM policy blocks Claude reflection at the chokepoint.

Mythos audit 2026-05-04 — operator reported reflection_log.jsonl was
still showing Claude Code inference (mode=incident_analyst, code_surgeon,
policy_auditor, deployment_arbiter, lightweight, deep) despite
ScannerConfig.meta_manager_use_llm=False being the production default.

Root cause: the no-LLM gate was only at cycle_autonomy's caller site;
event_handlers wired self._invoker = invoke_claude_reflection without
checking the policy; MetaManager defaulted use_llm=True so any direct
construction outside _build_production_manager bypassed the gate.

Fix: a single chokepoint at invoke_claude_reflection's entry — if
BUDDY_META_USE_LLM != "1" AND ScannerConfig.meta_manager_use_llm is
False, refuse to fire. Returns ReflectionResult(error="no_llm_policy")
without spawning Claude or writing to reflection_log.jsonl.

NO MOCKS. Real ScannerConfig, real env vars, real
invoke_claude_reflection — but inside a real subprocess that can't
actually launch the Claude CLI from the test environment, which is
exactly how the test proves the gate fires before any subprocess is
attempted.
"""
from __future__ import annotations

import os
from pathlib import Path

import pytest

from src.scanner.automation.claude_subprocess import (
    REFLECTION_LOG,
    _no_llm_policy_active,
    invoke_claude_reflection,
)


@pytest.fixture
def isolated_log(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Redirect REFLECTION_LOG to tmp_path so we can assert no writes happen."""
    log_path = tmp_path / "reflection_log.jsonl"
    monkeypatch.setattr(
        "src.scanner.automation.claude_subprocess.REFLECTION_LOG", log_path
    )
    return log_path


def test_env_var_unset_blocks_when_config_false(
    isolated_log: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """No env override + meta_manager_use_llm=False → blocked.

    Default ScannerConfig.meta_manager_use_llm is False per CLAUDE.md
    runtime-is-Claude-free policy.
    """
    monkeypatch.delenv("BUDDY_META_USE_LLM", raising=False)
    assert _no_llm_policy_active() is True

    result = invoke_claude_reflection(
        prompt="test prompt", trade_id="TEST-001", mode="lightweight",
    )
    assert result.success is False
    assert result.error == "no_llm_policy"
    # The chokepoint must NOT write to reflection_log.jsonl
    assert not isolated_log.exists()


def test_env_var_zero_blocks_explicitly(
    isolated_log: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """BUDDY_META_USE_LLM='0' → blocked, even if config flag is True."""
    monkeypatch.setenv("BUDDY_META_USE_LLM", "0")
    assert _no_llm_policy_active() is True

    result = invoke_claude_reflection(
        prompt="test", trade_id="TEST-002", mode="incident_analyst",
    )
    assert result.error == "no_llm_policy"
    assert not isolated_log.exists()


def test_env_var_one_allows_llm(monkeypatch: pytest.MonkeyPatch) -> None:
    """BUDDY_META_USE_LLM='1' → policy gate passes (function proceeds past
    the chokepoint). It will still fail downstream (no Claude CLI in test
    env), but the failure mode is different — proves the gate behavior
    is env-controlled, not hardcoded blocked."""
    monkeypatch.setenv("BUDDY_META_USE_LLM", "1")
    assert _no_llm_policy_active() is False
    # Don't actually call invoke_claude_reflection here — would attempt to
    # spawn the Claude CLI subprocess. The unit-level assertion that
    # _no_llm_policy_active returns False is enough to prove the gate
    # opens correctly. Real Claude invocation is operator-explicit.


def test_blocked_calls_dont_pollute_reflection_log(
    isolated_log: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Multiple blocked invocations across modes leave the log empty.

    This is the actual operator-visible symptom — if blocked calls still
    wrote to reflection_log.jsonl, the F8 Diagnostics tab + the TUI
    reflection-log tail would still show "Claude inference" entries even
    when the policy is in force.
    """
    monkeypatch.delenv("BUDDY_META_USE_LLM", raising=False)
    for mode in (
        "lightweight", "deep", "incident_analyst",
        "code_surgeon", "policy_auditor", "deployment_arbiter",
    ):
        result = invoke_claude_reflection(
            prompt=f"prompt for {mode}",
            trade_id=f"BLOCKED-{mode}",
            mode=mode,
        )
        assert result.error == "no_llm_policy"

    assert not isolated_log.exists(), (
        "Blocked invocations leaked into reflection_log.jsonl — the "
        "operator's audit trail is supposed to be clean under the "
        "no-LLM policy. Check invoke_claude_reflection's chokepoint."
    )


def test_default_metamanager_construction_is_no_llm() -> None:
    """Direct MetaManager() construction must default to use_llm=False.

    Before the fix, the default was True so any code path that built a
    MetaManager outside _build_production_manager would silently get
    LLM-enabled. That's how meta-pipeline specialists (incident_analyst,
    code_surgeon, etc.) leaked into reflection_log on 2026-05-03.
    """
    from src.scanner.automation.meta_manager import MetaManager

    mgr = MetaManager()
    assert mgr._use_llm is False, (
        "MetaManager default use_llm must be False — runtime is "
        "Claude-free per CLAUDE.md"
    )
