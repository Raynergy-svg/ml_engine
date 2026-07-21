"""Policy layer for AXIOM's subscription-backed operator.

Claude can reason, but this module decides what a proposed action means. The
default stance is intentionally narrow: V1 auto-actions are operationally
defensive and never include unhalting, leverage increases, promotions, safety
disables, or broker-direct calls.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Optional


AUTO_ALLOWED_ACTIONS = frozenset({
    "diagnose",
    "recheck",
    "write_learning",
    "halt_lane",
    "stop_loop",
})

HUMAN_REQUIRED_ACTIONS = frozenset({
    "unhalt",
    "start_loop",
    "set_gross_leverage",
    "raise_leverage",
    "promote_model",
    "promote_strategy",
    "promote_code",
    "disable_safety_gate",
    "flatten",
})

READ_ONLY_ACTIONS = frozenset({
    "observe",
    "summarize",
    "none",
    "",
})


@dataclass(frozen=True)
class PolicyDecision:
    action: str
    allowed: bool
    tier: str
    requires_human: bool
    reason: str


class ActionPolicy:
    """Classify operator actions before any side effect is considered."""

    def evaluate(self, action: Optional[str], params: Optional[Mapping[str, Any]] = None) -> PolicyDecision:
        normalized = str(action or "none").strip().lower()
        params = params or {}

        if normalized in READ_ONLY_ACTIONS:
            return PolicyDecision(
                action=normalized or "none",
                allowed=True,
                tier="read_only",
                requires_human=False,
                reason="read-only observation",
            )

        if normalized in AUTO_ALLOWED_ACTIONS:
            if normalized == "halt_lane" and str(params.get("lane") or "").strip() not in {"oanda_fx", "equity", "brain"}:
                return PolicyDecision(
                    action=normalized,
                    allowed=False,
                    tier="safe_auto",
                    requires_human=True,
                    reason="halt_lane requires a known lane",
                )
            if normalized == "stop_loop" and str(params.get("loop") or "").strip() not in {"trend", "tier7"}:
                return PolicyDecision(
                    action=normalized,
                    allowed=False,
                    tier="safe_auto",
                    requires_human=True,
                    reason="stop_loop requires a whitelisted loop",
                )
            return PolicyDecision(
                action=normalized,
                allowed=True,
                tier="safe_auto",
                requires_human=False,
                reason="defensive operator action",
            )

        if normalized in HUMAN_REQUIRED_ACTIONS:
            return PolicyDecision(
                action=normalized,
                allowed=False,
                tier="human_required",
                requires_human=True,
                reason="operator approval required for trade-risk or promotion action",
            )

        return PolicyDecision(
            action=normalized,
            allowed=False,
            tier="unknown",
            requires_human=True,
            reason="unknown operator action",
        )
