"""Bounded side effects for AXIOM operator actions."""
from __future__ import annotations

import os
from typing import Any, Dict, Mapping

from src.axiom_operator.policy import PolicyDecision


def execute_safe_action(action: str, params: Mapping[str, Any], policy: PolicyDecision) -> Dict[str, Any]:
    """Apply only the small defensive action set approved by policy.

    This function deliberately routes through the dashboard control layer for
    halt/stop actions so the existing practice pin, parameter validation, and
    audit trail remain the load-bearing guardrails.
    """
    if not policy.allowed:
        return {"applied": False, "reason": policy.reason}

    if action in {"observe", "summarize", "none", ""}:
        return {"applied": False, "reason": "read-only action"}

    if action in {"diagnose", "recheck"}:
        return {"applied": True, "result": "observed", "detail": "no mutating control required"}

    if action == "write_learning":
        content = str(params.get("content") or "").strip()
        if not content:
            return {"applied": False, "reason": "write_learning missing content"}
        from src.mcp.buddy_server import write_learning

        return {"applied": True, "result": write_learning(content)}

    if action in {"halt_lane", "stop_loop"}:
        if os.environ.get("AXIOM_CONTROL_ENABLED", "").lower() not in {"1", "true", "yes"}:
            return {"applied": False, "reason": "AXIOM_CONTROL_ENABLED is not set"}
        from dashboard.server.control import _run

        if action == "halt_lane":
            result = _run("halt", {"lane": params.get("lane")}, actor="axiom_operator")
        else:
            result = _run("stop_loop", {"loop": params.get("loop")}, actor="axiom_operator")
        return {"applied": True, "result": result}

    return {"applied": False, "reason": f"unsupported safe action {action!r}"}
