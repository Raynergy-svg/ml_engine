"""Shared policy dataclasses for the Tier 7 action policy engine.

Defines the vocabulary of actions the system can take, how they're
classified (allow/deny/confirm), and the structure of policy decisions.
"""

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional


class ActionType(Enum):
    """All autonomous actions that can be policy-gated."""
    EXECUTE_TRADE = "execute_trade"
    CLOSE_TRADE_EARLY = "close_trade_early"
    TIGHTEN_STOP = "tighten_stop"
    TAKE_PARTIAL = "take_partial"
    SWITCH_PAIR_MODEL = "switch_pair_model"
    APPLY_THRESHOLD_ADJUSTMENT = "apply_threshold_adjustment"
    START_BACKGROUND_RETRAIN = "start_background_retrain"
    START_BACKGROUND_DEBUG = "start_background_debug"
    RESCAN_AFTER_CLOSE = "rescan_after_close"
    FORCE_TRADE_DEGRADED = "force_trade_degraded"


class PolicyDecisionType(Enum):
    """Outcome of a policy evaluation."""
    AUTO_ALLOW = "auto_allow"
    NEEDS_CONFIRMATION = "needs_confirmation"
    DENY = "deny"


@dataclass
class ActionRequest:
    """A request to perform an autonomous action."""
    action_type: ActionType
    source: str                         # e.g. "continuous_scanner", "orchestrator"
    context: Dict[str, Any] = field(default_factory=dict)
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


@dataclass
class PolicyDecision:
    """The result of evaluating an ActionRequest against policy rules."""
    decision_id: str = field(default_factory=lambda: str(uuid.uuid4())[:12])
    action_type: ActionType = ActionType.EXECUTE_TRADE
    decision: PolicyDecisionType = PolicyDecisionType.AUTO_ALLOW
    reasons: List[str] = field(default_factory=list)
    matched_rules: List[str] = field(default_factory=list)
    environment_snapshot: Dict[str, Any] = field(default_factory=dict)
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> Dict[str, Any]:
        return {
            "decision_id": self.decision_id,
            "action_type": self.action_type.value,
            "decision": self.decision.value,
            "reasons": self.reasons,
            "matched_rules": self.matched_rules,
            "environment_snapshot": self.environment_snapshot,
            "timestamp": self.timestamp,
        }


@dataclass
class PolicyRule:
    """A single human-readable policy rule."""
    name: str
    action_types: List[ActionType]
    decision: PolicyDecisionType
    condition: str                      # Human-readable condition description
    priority: int = 100                 # Lower = evaluated first
    enabled: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "action_types": [a.value for a in self.action_types],
            "decision": self.decision.value,
            "condition": self.condition,
            "priority": self.priority,
            "enabled": self.enabled,
        }
