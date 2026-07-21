"""AXIOM Agent Runtime — tool registry.

Every tool exposed here is READ or SAFE-OPERATIONAL only (all are reads in
this build): a name, a description, a tier, typed args, and a callable that
delegates to an EXISTING function. None mutates trading state; that
guarantee is checked structurally in
tests/test_agent_runtime_tools_2026_07_06.py (a source scan for forbidden
calls) and by ``all_operational`` below.

Tools double as ``policy.ActionSpec`` entries (``TOOL_ACTIONS``) so every tool
invocation — by a future agent loop — goes through the SAME preflight +
audit gateway as halt/de-risk/escalation actions (see policy.default_engine).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional

from src.agent_runtime.policy import ActionSpec, ActionTier
from src.agent_runtime.tools import readers


@dataclass(frozen=True)
class Tool:
    name: str
    description: str
    tier: ActionTier
    args: Dict[str, type]
    call: Callable[..., Dict[str, Any]]


TOOLS: Dict[str, Tool] = {
    "health_check": Tool(
        name="health_check",
        description="LIVE-lane running:YES/NO oracle (OANDA trend loop + Tier7 heartbeat).",
        tier=ActionTier.OPERATIONAL,
        args={},
        call=readers.health_check,
    ),
    "gate_health": Tool(
        name="gate_health",
        description="Re-run verify_gate's Hard-NO + integrity checks; returns PASS/FAIL + per-check detail.",
        tier=ActionTier.OPERATIONAL,
        args={},
        call=readers.gate_health,
    ),
    "tier7_status": Tool(
        name="tier7_status",
        description="Read-only Tier 7 / self-heal supervisor snapshot (budget, debounce, recent events).",
        tier=ActionTier.OPERATIONAL,
        args={"project_root": Optional[str]},
        call=readers.tier7_status,
    ),
    "lane_halt_status": Tool(
        name="lane_halt_status",
        description="Global + per-lane halt state (oanda_fx/equity/brain/crypto_momentum/track_b).",
        tier=ActionTier.OPERATIONAL,
        args={"state_path": Optional[str]},
        call=readers.lane_halt_status,
    ),
    "oanda_account_state": Tool(
        name="oanda_account_state",
        description="OANDA practice account snapshot: NAV, positions, margin, drawdown (read-only).",
        tier=ActionTier.OPERATIONAL,
        args={},
        call=readers.oanda_account_state,
    ),
    "trade_feedback": Tool(
        name="trade_feedback",
        description="Tail of closed-trade journal entries (outcomes, RL context) — read-only.",
        tier=ActionTier.OPERATIONAL,
        args={"limit": int},
        call=readers.trade_feedback,
    ),
    "agent_weights": Tool(
        name="agent_weights",
        description="Learned scanner-agent weight overrides by autonomy level — read-only.",
        tier=ActionTier.OPERATIONAL,
        args={},
        call=readers.agent_weights,
    ),
    "learnings": Tool(
        name="learnings",
        description="Raw LESSONS.md + NOTES.md text — permanent failure modes + live working memory.",
        tier=ActionTier.OPERATIONAL,
        args={"max_chars": int},
        call=readers.learnings,
    ),
    "shadow_ledger": Tool(
        name="shadow_ledger",
        description=(
            "Shadow-lane ledger for one of: " + ", ".join(readers.list_shadow_lanes()) +
            ". Fails closed on an unknown lane name."
        ),
        tier=ActionTier.OPERATIONAL,
        args={"lane": str},
        call=readers.shadow_ledger,
    ),
}


def list_tools() -> List[Tool]:
    return list(TOOLS.values())


def get_tool(name: str) -> Tool:
    try:
        return TOOLS[name]
    except KeyError:
        raise KeyError(f"unknown tool {name!r}; known tools: {sorted(TOOLS)}") from None


def all_operational() -> bool:
    return all(tool.tier is ActionTier.OPERATIONAL for tool in TOOLS.values())


TOOL_ACTIONS: Dict[str, ActionSpec] = {
    name: ActionSpec(name=name, tier=tool.tier, description=tool.description, execute=tool.call)
    for name, tool in TOOLS.items()
}
