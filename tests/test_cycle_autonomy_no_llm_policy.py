"""TDD coverage for cycle_autonomy's no-LLM policy guard.

Mythos audit 2026-04-28: with meta_manager_use_llm=False the operator has
opted out of Claude in the post-trade plane — orchestrator's deterministic
SelfHeal path is the only legitimate actuator. cycle_autonomy must NOT
fall through to invoke_claude_reflection when meta routing fails (or is
disabled), because that bypasses the operator's policy and burns Claude
budget on duplicate work the orchestrator already handles.

The reflection_log entry SELFHEAL-C1 (2026-04-28T23:30:33Z) hit Claude
API directly, got 529 Overloaded, produced no artifacts — the exact
dead-air this guard prevents.
"""
from __future__ import annotations

import pytest

from src.scanner.automation.cycle_autonomy import CycleAutonomyTriggers


def _make_triggers(monkeypatch, captured: list) -> CycleAutonomyTriggers:
    """Build a triggers instance with the Claude subprocess stubbed.

    captured records every call so a test can assert Claude was reached
    (failure case) or skipped (success case).
    """
    t = CycleAutonomyTriggers(brain_callback=lambda _msg: None)

    def _stub_claude(*args, **kwargs):
        captured.append(("claude_call", args, kwargs))
        class _Result:
            success = True
            cost_usd = 0.0
            hypothesis = ""
        return _Result()

    monkeypatch.setattr(
        "src.scanner.automation.claude_subprocess.invoke_claude_reflection",
        _stub_claude,
        raising=False,
    )
    return t


def test_no_llm_policy_blocks_claude_when_meta_disabled(monkeypatch):
    """When meta is OFF and use_llm policy is False, _invoke must NOT
    fall through to Claude. This is the reflection_log SELFHEAL-C1 case.
    """
    captured: list = []
    monkeypatch.delenv("BUDDY_META_MANAGER_ENABLED", raising=False)
    monkeypatch.setenv("BUDDY_META_USE_LLM", "0")

    t = _make_triggers(monkeypatch, captured)
    t._invoke(prompt="trigger", trade_id="self_heal", mode="deep", timeout=60)

    assert not any(c[0] == "claude_call" for c in captured), (
        f"Claude was invoked despite no-LLM policy. captured={captured}"
    )


def test_env_override_re_enables_claude_for_debugging(monkeypatch):
    """Operators must be able to flip the policy back on without restart
    via BUDDY_META_USE_LLM=1 — used when troubleshooting the meta-pipeline.
    """
    captured: list = []
    monkeypatch.delenv("BUDDY_META_MANAGER_ENABLED", raising=False)
    monkeypatch.setenv("BUDDY_META_USE_LLM", "1")

    t = _make_triggers(monkeypatch, captured)
    # Force budget+lock paths past stub-friendly defaults.
    monkeypatch.setattr(
        "src.scanner.automation.claude_subprocess.ReflectionBudget",
        lambda: type("B", (), {
            "allows": lambda self, mode: True,
            "record": lambda self, **kw: None,
        })(),
        raising=False,
    )
    monkeypatch.setattr(
        "src.scanner.automation.claude_subprocess.SingleFlightLock",
        lambda: type("L", (), {
            "acquire": lambda self: True,
            "release": lambda self: None,
        })(),
        raising=False,
    )
    t._invoke(prompt="trigger", trade_id="self_heal", mode="deep", timeout=60)

    assert any(c[0] == "claude_call" for c in captured), (
        f"Operator override (BUDDY_META_USE_LLM=1) failed to allow Claude. "
        f"captured={captured}"
    )


def test_meta_routed_short_circuits_before_policy_check(monkeypatch):
    """When meta IS enabled and routes the incident successfully, Claude
    must not be called regardless of the no-LLM policy — meta absorbed
    the signal. Ensures policy guard doesn't accidentally double-suppress.
    """
    captured: list = []
    monkeypatch.delenv("BUDDY_META_USE_LLM", raising=False)

    def _stub_route(_incident):
        captured.append(("route_incident", _incident))
        return True  # meta accepted

    def _stub_enabled():
        return True

    monkeypatch.setattr(
        "src.scanner.automation.meta_manager.is_enabled",
        _stub_enabled,
        raising=False,
    )
    monkeypatch.setattr(
        "src.scanner.automation.meta_manager.route_incident",
        _stub_route,
        raising=False,
    )

    t = _make_triggers(monkeypatch, captured)
    t._invoke(prompt="trigger", trade_id="self_heal", mode="deep", timeout=60)

    assert any(c[0] == "route_incident" for c in captured)
    assert not any(c[0] == "claude_call" for c in captured), (
        "Claude invoked despite successful meta routing"
    )


def test_policy_helper_resolves_env_then_dataclass_default():
    """_claude_reflection_allowed precedence: env var (1/0) > dataclass."""
    import os
    saved = os.environ.pop("BUDDY_META_USE_LLM", None)
    try:
        # Without env, dataclass default is False (committed to meta-layers).
        assert CycleAutonomyTriggers._claude_reflection_allowed() is False
        os.environ["BUDDY_META_USE_LLM"] = "1"
        assert CycleAutonomyTriggers._claude_reflection_allowed() is True
        os.environ["BUDDY_META_USE_LLM"] = "0"
        assert CycleAutonomyTriggers._claude_reflection_allowed() is False
    finally:
        os.environ.pop("BUDDY_META_USE_LLM", None)
        if saved is not None:
            os.environ["BUDDY_META_USE_LLM"] = saved
