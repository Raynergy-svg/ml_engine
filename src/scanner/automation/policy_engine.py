"""Tier 7 Policy Engine — human-readable action policy for autonomous trading decisions.

Classifies actions into allow/deny/needs_confirmation based on configurable
rules and runtime environment. Starts in LOGGING-ONLY mode: all actions are
auto-allowed but decisions are recorded for review before enforcement.

Inspired by donor autoMode.ts three-tier rule system (allow/soft_deny/environment).
"""

import json
import os
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional

import structlog

from src.scanner.automation.policy_types import (
    ActionType,
    ActionRequest,
    PolicyDecision,
    PolicyDecisionType,
    PolicyRule,
)
from src.scanner.automation.safe_json import safe_json_read, safe_json_write, safe_jsonl_append

logger = structlog.get_logger(__name__)

_DECISIONS_PATH = Path("trained_data/policy_decisions.jsonl")
_RULES_PATH = Path("trained_data/policy_rules.json")


def _default_rules() -> List[PolicyRule]:
    """Default policy rules — logging-only safe defaults."""
    return [
        # ── ALLOW rules ──
        PolicyRule(
            name="allow_tighten_stop_open_trade",
            action_types=[ActionType.TIGHTEN_STOP],
            decision=PolicyDecisionType.AUTO_ALLOW,
            condition="Trade is open and drawdown rules are not violated",
            priority=10,
        ),
        PolicyRule(
            name="allow_rescan_after_close",
            action_types=[ActionType.RESCAN_AFTER_CLOSE],
            decision=PolicyDecisionType.AUTO_ALLOW,
            condition="Always allowed — rescanning is read-only",
            priority=10,
        ),
        PolicyRule(
            name="allow_background_retrain_normal",
            action_types=[ActionType.START_BACKGROUND_RETRAIN],
            decision=PolicyDecisionType.AUTO_ALLOW,
            condition="Broker session is not degraded and queue health is normal",
            priority=20,
        ),
        PolicyRule(
            name="allow_threshold_adjustment_paper",
            action_types=[ActionType.APPLY_THRESHOLD_ADJUSTMENT],
            decision=PolicyDecisionType.AUTO_ALLOW,
            condition="Account mode is paper/practice",
            priority=20,
        ),
        PolicyRule(
            name="allow_trade_execution_paper",
            action_types=[ActionType.EXECUTE_TRADE],
            decision=PolicyDecisionType.AUTO_ALLOW,
            condition="Account mode is paper/practice",
            priority=20,
        ),
        # ── SOFT DENY rules ──
        PolicyRule(
            name="confirm_lower_thresholds_live",
            action_types=[ActionType.APPLY_THRESHOLD_ADJUSTMENT],
            decision=PolicyDecisionType.NEEDS_CONFIRMATION,
            condition="Account mode is live and adjustment lowers a gate threshold",
            priority=50,
        ),
        PolicyRule(
            name="confirm_model_switch_few_trades",
            action_types=[ActionType.SWITCH_PAIR_MODEL],
            decision=PolicyDecisionType.NEEDS_CONFIRMATION,
            condition="Fewer than 10 closed trades on the current model",
            priority=50,
        ),
        PolicyRule(
            name="confirm_retrain_degraded",
            action_types=[ActionType.START_BACKGROUND_RETRAIN],
            decision=PolicyDecisionType.NEEDS_CONFIRMATION,
            condition="Broker session is degraded",
            priority=50,
        ),
        # ── DENY rules ──
        PolicyRule(
            name="deny_trade_during_degraded",
            action_types=[ActionType.FORCE_TRADE_DEGRADED],
            decision=PolicyDecisionType.DENY,
            condition="Broker session is degraded — no new trades",
            priority=90,
        ),
    ]


class PolicyEngine:
    """Evaluates autonomous actions against configurable policy rules.

    Starts in logging-only mode (``enforcement=False``): all evaluations
    return ``AUTO_ALLOW`` but the *would-be* decision is recorded.
    Toggle ``enforcement=True`` to start enforcing decisions.
    """

    def __init__(
        self,
        rules_path: Optional[str] = None,
        enforcement: bool = False,
    ):
        self._rules: List[PolicyRule] = []
        self._enforcement = enforcement
        self._rules_path = Path(rules_path) if rules_path else _RULES_PATH
        self._lock = threading.Lock()
        self._environment_providers: List = []

        # Load rules
        self._load_rules()
        if not self._rules:
            self._rules = _default_rules()
            logger.info("policy_engine.loaded_defaults", count=len(self._rules))

    # ── Rule Management ────────────────────────────────────────────────

    def _load_rules(self) -> None:
        """Load rules from JSON file if it exists."""
        data = safe_json_read(self._rules_path, default=[])
        if not isinstance(data, list):
            return
        for rd in data:
            try:
                action_types = [ActionType(a) for a in rd.get("action_types", [])]
                decision = PolicyDecisionType(rd.get("decision", "auto_allow"))
                rule = PolicyRule(
                    name=rd.get("name", "unnamed"),
                    action_types=action_types,
                    decision=decision,
                    condition=rd.get("condition", ""),
                    priority=rd.get("priority", 100),
                    enabled=rd.get("enabled", True),
                )
                self._rules.append(rule)
            except (ValueError, KeyError) as e:
                logger.warning("policy_engine.rule_load_error", error=str(e), rule=rd)

    def save_rules(self) -> None:
        """Persist current rules to JSON."""
        data = [r.to_dict() for r in self._rules]
        safe_json_write(self._rules_path, data)

    # ── Evaluation ─────────────────────────────────────────────────────

    def evaluate(self, request: ActionRequest) -> PolicyDecision:
        """Evaluate an action request against policy rules.

        In logging-only mode (enforcement=False), always returns AUTO_ALLOW
        but records what the decision *would have been*.

        Args:
            request: The action to evaluate.

        Returns:
            PolicyDecision with the result.
        """
        env = self.get_environment()
        matched_rules: List[str] = []
        reasons: List[str] = []

        # Find matching rules (sorted by priority)
        with self._lock:
            sorted_rules = sorted(
                [r for r in self._rules if r.enabled],
                key=lambda r: r.priority,
            )

        result_decision = PolicyDecisionType.AUTO_ALLOW

        for rule in sorted_rules:
            if request.action_type not in rule.action_types:
                continue

            # Rule matches this action type
            matched_rules.append(rule.name)
            reasons.append(f"{rule.name}: {rule.condition}")

            # Most restrictive decision wins (DENY > NEEDS_CONFIRMATION > AUTO_ALLOW)
            if rule.decision == PolicyDecisionType.DENY:
                result_decision = PolicyDecisionType.DENY
            elif (
                rule.decision == PolicyDecisionType.NEEDS_CONFIRMATION
                and result_decision != PolicyDecisionType.DENY
            ):
                result_decision = PolicyDecisionType.NEEDS_CONFIRMATION

        decision = PolicyDecision(
            action_type=request.action_type,
            decision=result_decision,
            reasons=reasons,
            matched_rules=matched_rules,
            environment_snapshot=env,
        )

        # Log the decision
        self.log_decision(decision, request)

        # In logging-only mode, override to AUTO_ALLOW
        if not self._enforcement and result_decision != PolicyDecisionType.AUTO_ALLOW:
            logger.info(
                "policy_engine.logging_only",
                action=request.action_type.value,
                would_be=result_decision.value,
                matched=matched_rules,
            )
            decision.decision = PolicyDecisionType.AUTO_ALLOW
            decision.reasons.append("LOGGING_ONLY: overridden to auto_allow")

        return decision

    # ── Environment ────────────────────────────────────────────────────

    def get_environment(self) -> Dict[str, Any]:
        """Capture current runtime environment for policy context."""
        env: Dict[str, Any] = {
            "account_mode": os.getenv("OANDA_ENVIRONMENT", "practice"),
            "enforcement_enabled": self._enforcement,
        }

        # Collect from registered providers (control plane, etc.)
        for provider in self._environment_providers:
            try:
                env.update(provider())
            except Exception as e:
                logger.warning("policy_engine.provider_error", error=str(e))

        return env

    def register_environment_provider(self, provider) -> None:
        """Register a callable that returns a dict of environment facts."""
        self._environment_providers.append(provider)

    # ── Decision Log ───────────────────────────────────────────────────

    def log_decision(self, decision: PolicyDecision, request: ActionRequest) -> None:
        """Append decision to JSONL log for operator review."""
        record = {
            **decision.to_dict(),
            "request_source": request.source,
            "request_context": request.context,
        }
        safe_jsonl_append(_DECISIONS_PATH, record)

    def get_recent_decisions(self, n: int = 10) -> List[Dict[str, Any]]:
        """Return the N most recent policy decisions."""
        if not _DECISIONS_PATH.exists():
            return []
        try:
            lines = _DECISIONS_PATH.read_text().strip().split("\n")
            records = []
            for line in lines[-n:]:
                if line.strip():
                    try:
                        records.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
            return records
        except Exception:
            return []

    # ── Rule Linting ───────────────────────────────────────────────────

    def lint_rules(self) -> List[str]:
        """Validate rules for conflicts, gaps, and issues."""
        issues: List[str] = []

        # Check for duplicate names
        names = [r.name for r in self._rules]
        dupes = [n for n in set(names) if names.count(n) > 1]
        if dupes:
            issues.append(f"Duplicate rule names: {dupes}")

        # Check for action types with no rules
        covered = set()
        for rule in self._rules:
            for at in rule.action_types:
                covered.add(at)
        uncovered = set(ActionType) - covered
        if uncovered:
            issues.append(f"Action types with no rules: {[a.value for a in uncovered]}")

        # Check for conflicting decisions on same action type
        for at in ActionType:
            decisions = set()
            for rule in self._rules:
                if at in rule.action_types:
                    decisions.add(rule.decision)
            if PolicyDecisionType.AUTO_ALLOW in decisions and PolicyDecisionType.DENY in decisions:
                issues.append(
                    f"Conflicting rules for {at.value}: both ALLOW and DENY "
                    f"(most restrictive wins, but this may be unintentional)"
                )

        return issues

    # ── Status ─────────────────────────────────────────────────────────

    def get_status(self) -> Dict[str, Any]:
        """Return policy engine status for system_status reporting."""
        recent = self.get_recent_decisions(5)
        return {
            "enforcement_enabled": self._enforcement,
            "total_rules": len(self._rules),
            "enabled_rules": sum(1 for r in self._rules if r.enabled),
            "recent_decisions": len(recent),
            "lint_issues": self.lint_rules(),
        }

    @property
    def is_enforcing(self) -> bool:
        return self._enforcement

    def enable_enforcement(self) -> None:
        self._enforcement = True
        logger.warning("policy_engine.enforcement_enabled")

    def disable_enforcement(self) -> None:
        self._enforcement = False
        logger.info("policy_engine.enforcement_disabled")

    def shutdown(self) -> None:
        """Graceful shutdown — persist rules if modified."""
        pass  # Rules auto-persist on save_rules() call


# ── Singleton ──────────────────────────────────────────────────────────

_engine_instance: Optional[PolicyEngine] = None
_engine_lock = threading.Lock()


def get_policy_engine() -> PolicyEngine:
    """Return the module-level singleton PolicyEngine."""
    global _engine_instance
    if _engine_instance is None:
        with _engine_lock:
            if _engine_instance is None:
                _engine_instance = PolicyEngine()
    return _engine_instance


def reset_policy_engine() -> None:
    """Reset the singleton for testing."""
    global _engine_instance
    with _engine_lock:
        _engine_instance = None
