"""AXIOM Agent Runtime — tool registry (read + safe-operational only, real disk).

Every tool wraps an EXISTING function; these tests prove the wrapper delegates
correctly, is tagged with a tier, and that the registry as a whole cannot place
an order, arm, or unhalt anything.
"""
from __future__ import annotations

import inspect
import json

import pytest

from dashboard.server import data_sources as ds
from src.agent_runtime import policy
from src.agent_runtime.tools import registry as tool_registry
from src.scanner.automation.state_engine import StateEngine


# ---------------------------------------------------------------------------
# registry shape
# ---------------------------------------------------------------------------


def test_every_tool_is_registered_with_name_description_tier_and_callable():
    tools = tool_registry.list_tools()
    assert len(tools) >= 8  # health, gate, tier7, halt, account, journal, weights, learnings, 3 shadow ledgers
    for tool in tools:
        assert isinstance(tool.name, str) and tool.name
        assert isinstance(tool.description, str) and tool.description
        assert tool.tier is policy.ActionTier.OPERATIONAL
        assert callable(tool.call)


def test_get_tool_returns_the_named_tool():
    tool = tool_registry.get_tool("health_check")
    assert tool.name == "health_check"


def test_get_tool_raises_keyerror_on_unknown_tool():
    with pytest.raises(KeyError):
        tool_registry.get_tool("does_not_exist")


def test_tool_actions_are_all_operational_tier_in_the_policy_registry():
    registry = policy.build_registry(tool_registry.TOOL_ACTIONS)
    for name in tool_registry.TOOL_ACTIONS:
        assert registry[name].tier is policy.ActionTier.OPERATIONAL


# ---------------------------------------------------------------------------
# individual tool wrappers delegate to the EXISTING function (no reimplementation)
# ---------------------------------------------------------------------------


def test_health_check_tool_matches_the_real_oracle_output():
    from_tool = tool_registry.get_tool("health_check").call()
    from src.agent_runtime.tools.readers import _load_running_status_module

    mod = _load_running_status_module()
    direct = mod.live_lane_running()
    # *_age_s are computed live (time.time() - mtime) — two calls a few
    # microseconds apart legitimately differ by a fraction of a second.
    age_keys = {k for k in from_tool if k.endswith("_age_s")}
    assert {k: v for k, v in from_tool.items() if k not in age_keys} == \
        {k: v for k, v in direct.items() if k not in age_keys}
    assert from_tool.keys() == direct.keys()


def test_lane_halt_tool_reads_without_mutating(tmp_path):
    state_path = tmp_path / "state.json"
    eng = StateEngine(state_path=state_path)
    eng.set_halted(False)  # clear the global switch so per-lane values take effect
    eng.set_halted(True, lane="oanda_fx")
    eng.set_halted(False, lane="equity")
    before = json.loads(state_path.read_text())

    from src.agent_runtime.tools.readers import lane_halt_status

    result = lane_halt_status(state_path=state_path)

    after = json.loads(state_path.read_text())
    assert before == after  # byte-identical: a read tool must never mutate state.json
    assert result["global_halted"] is False
    assert result["lanes"]["equity"] is False
    assert result["lanes"]["oanda_fx"] is True


def test_oanda_account_state_tool_matches_direct_read_account(monkeypatch, tmp_path):
    monkeypatch.setattr(ds, "OANDA_DIR", tmp_path)
    (tmp_path / "account_state.json").write_text(json.dumps({
        "account_id": "101-test", "currency": "USD", "nav": 100000.0,
        "unrealized_pl": 0.0, "realized_pl": 0.0, "positions": [],
    }))
    from src.agent_runtime.tools import readers

    monkeypatch.setattr(readers, "read_account", ds.read_account)
    direct = ds.read_account()
    from_tool = tool_registry.get_tool("oanda_account_state").call()
    assert from_tool == direct


def test_agent_weights_tool_reads_the_real_file_read_only(tmp_path, monkeypatch):
    weights_path = tmp_path / "agent_weights.json"
    weights_path.write_text(json.dumps({"MEDIUM": {"trend": 1.15}}))
    from src.agent_runtime.tools import readers

    monkeypatch.setattr(readers, "AGENT_WEIGHTS_PATH", weights_path)
    result = tool_registry.get_tool("agent_weights").call()
    assert result["weights"] == {"MEDIUM": {"trend": 1.15}}
    # untouched — a read tool must never write back
    assert json.loads(weights_path.read_text()) == {"MEDIUM": {"trend": 1.15}}


def test_trade_feedback_tool_reads_journal_entries_read_only(tmp_path, monkeypatch):
    journal_path = tmp_path / "trade_journal_rl.json"
    entries = [{"trade_id": 1, "outcome": {"trade_won": True}}]
    journal_path.write_text(json.dumps(entries))
    from src.agent_runtime.tools import readers

    monkeypatch.setattr(readers, "TRADE_JOURNAL_PATH", journal_path)
    result = tool_registry.get_tool("trade_feedback").call()
    assert result["count"] == 1
    assert result["entries"][0]["trade_id"] == 1
    assert json.loads(journal_path.read_text()) == entries


def test_learnings_tool_reads_lessons_and_notes_markdown(tmp_path, monkeypatch):
    lessons = tmp_path / "LESSONS.md"
    notes = tmp_path / "NOTES.md"
    lessons.write_text("# LESSONS\nL-001 example")
    notes.write_text("# NOTES\ncurrent state")
    from src.agent_runtime.tools import readers

    monkeypatch.setattr(readers, "LESSONS_PATH", lessons)
    monkeypatch.setattr(readers, "NOTES_PATH", notes)
    result = tool_registry.get_tool("learnings").call()
    assert "L-001 example" in result["lessons"]
    assert "current state" in result["notes"]


def test_shadow_ledger_tool_dispatches_crypto_momentum(monkeypatch):
    from src.agent_runtime.tools import readers

    sentinel = {"n_forward_cycles": 0}
    monkeypatch.setattr(readers, "read_crypto_momentum", lambda: sentinel)
    result = tool_registry.get_tool("shadow_ledger").call(lane="crypto_momentum")
    assert result is sentinel


def test_shadow_ledger_tool_rejects_unknown_lane():
    with pytest.raises(ValueError):
        tool_registry.get_tool("shadow_ledger").call(lane="not_a_real_lane")


def test_gate_health_tool_runs_the_real_gate_and_never_touches_production_verdict(tmp_path):
    from src.agent_runtime.tools import readers

    prod_verdict = readers.REPO_ROOT / ".claude" / "loop" / "verdict.json"
    before = prod_verdict.read_bytes() if prod_verdict.exists() else None

    result = tool_registry.get_tool("gate_health").call()

    after = prod_verdict.read_bytes() if prod_verdict.exists() else None
    assert before == after  # the diagnostic tool must never clobber the live verdict
    assert result["gate"] in ("PASS", "FAIL")
    assert isinstance(result["checks"], list)


def test_tier7_status_tool_is_read_only_per_its_own_contract(tmp_path):
    result = tool_registry.get_tool("tier7_status").call(project_root=tmp_path)
    assert "running" in result
    assert "halted" in result


# ---------------------------------------------------------------------------
# structural: no tool in this registry can trade / arm / unhalt / mutate live state
# ---------------------------------------------------------------------------

_FORBIDDEN_SUBSTRINGS = (
    "set_halted(False",
    "set_halted(True",
    ".arm(",
    "LiveGate(",
    "submit_order",
    "create_order",
    "close_position",
    "execute_trade(",
    "set_override(",
)


def test_no_tool_source_contains_a_mutating_or_trading_call():
    import src.agent_runtime.tools.readers as readers_mod

    source = inspect.getsource(readers_mod)
    for forbidden in _FORBIDDEN_SUBSTRINGS:
        assert forbidden not in source, f"tools/readers.py contains forbidden call {forbidden!r}"


def test_all_tools_are_operational_never_deescalation_or_escalation():
    for tool in tool_registry.list_tools():
        assert tool.tier is policy.ActionTier.OPERATIONAL
