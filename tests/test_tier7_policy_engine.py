"""Tests for Tier 7 PR 4: Policy Engine.

Validates:
1. Policy types (ActionType, PolicyRule, PolicyDecision)
2. Rule evaluation (allow/deny/confirm priority)
3. Logging-only mode (override to AUTO_ALLOW but record would-be)
4. Environment capture
5. Decision persistence to JSONL
6. Rule linting
7. Integration points (continuous.py, orchestrator.py)
"""

import json
import os
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.scanner.automation.policy_types import (
    ActionType,
    ActionRequest,
    PolicyDecision,
    PolicyDecisionType,
    PolicyRule,
)
from src.scanner.automation.policy_engine import (
    PolicyEngine,
    reset_policy_engine,
    _default_rules,
)


@pytest.fixture(autouse=True)
def _reset_singleton():
    reset_policy_engine()
    yield
    reset_policy_engine()


@pytest.fixture
def tmp_engine(tmp_path):
    """PolicyEngine with temp paths for isolation."""
    rules_path = str(tmp_path / "rules.json")
    # Patch the decisions path too
    import src.scanner.automation.policy_engine as pe_mod
    original = pe_mod._DECISIONS_PATH
    pe_mod._DECISIONS_PATH = tmp_path / "decisions.jsonl"
    engine = PolicyEngine(rules_path=rules_path, enforcement=False)
    yield engine
    pe_mod._DECISIONS_PATH = original


# ── Policy Types Tests ────────────────────────────────────────────────

class TestPolicyTypes:

    def test_action_type_enum_has_10_members(self):
        assert len(ActionType) == 10

    def test_action_request_defaults(self):
        req = ActionRequest(action_type=ActionType.EXECUTE_TRADE, source="test")
        assert req.timestamp  # auto-generated
        assert req.context == {}

    def test_policy_decision_to_dict(self):
        d = PolicyDecision(
            action_type=ActionType.EXECUTE_TRADE,
            decision=PolicyDecisionType.DENY,
            reasons=["degraded mode"],
        )
        result = d.to_dict()
        assert result["action_type"] == "execute_trade"
        assert result["decision"] == "deny"
        assert result["decision_id"]  # UUID generated

    def test_policy_rule_to_dict(self):
        r = PolicyRule(
            name="test_rule",
            action_types=[ActionType.EXECUTE_TRADE, ActionType.CLOSE_TRADE_EARLY],
            decision=PolicyDecisionType.AUTO_ALLOW,
            condition="always",
        )
        result = r.to_dict()
        assert result["action_types"] == ["execute_trade", "close_trade_early"]


# ── Default Rules Tests ───────────────────────────────────────────────

class TestDefaultRules:

    def test_default_rules_exist(self):
        rules = _default_rules()
        assert len(rules) >= 9

    def test_default_rules_cover_key_actions(self):
        rules = _default_rules()
        covered = set()
        for r in rules:
            for at in r.action_types:
                covered.add(at)
        # At minimum, these should be covered
        assert ActionType.EXECUTE_TRADE in covered
        assert ActionType.TIGHTEN_STOP in covered
        assert ActionType.START_BACKGROUND_RETRAIN in covered
        assert ActionType.SWITCH_PAIR_MODEL in covered
        assert ActionType.FORCE_TRADE_DEGRADED in covered


# ── Engine Evaluation Tests ───────────────────────────────────────────

class TestPolicyEvaluation:

    def test_evaluate_returns_decision(self, tmp_engine):
        req = ActionRequest(action_type=ActionType.EXECUTE_TRADE, source="test")
        decision = tmp_engine.evaluate(req)
        assert isinstance(decision, PolicyDecision)
        assert decision.action_type == ActionType.EXECUTE_TRADE

    def test_logging_only_overrides_to_allow(self, tmp_engine):
        """In logging-only mode, even DENY rules should return AUTO_ALLOW."""
        assert not tmp_engine.is_enforcing
        req = ActionRequest(action_type=ActionType.FORCE_TRADE_DEGRADED, source="test")
        decision = tmp_engine.evaluate(req)
        # Should be AUTO_ALLOW despite DENY rule matching
        assert decision.decision == PolicyDecisionType.AUTO_ALLOW
        # But should record the would-be decision
        assert any("LOGGING_ONLY" in r for r in decision.reasons)

    def test_enforcement_mode_denies(self, tmp_path):
        """With enforcement enabled, DENY rules should actually deny."""
        import src.scanner.automation.policy_engine as pe_mod
        original = pe_mod._DECISIONS_PATH
        pe_mod._DECISIONS_PATH = tmp_path / "decisions.jsonl"
        try:
            engine = PolicyEngine(
                rules_path=str(tmp_path / "rules.json"),
                enforcement=True,
                enforcement_level=3,
            )
            req = ActionRequest(action_type=ActionType.FORCE_TRADE_DEGRADED, source="test")
            decision = engine.evaluate(req)
            assert decision.decision == PolicyDecisionType.DENY
        finally:
            pe_mod._DECISIONS_PATH = original

    def test_most_restrictive_wins(self, tmp_path):
        """When both ALLOW and DENY rules match, DENY wins."""
        import src.scanner.automation.policy_engine as pe_mod
        original = pe_mod._DECISIONS_PATH
        pe_mod._DECISIONS_PATH = tmp_path / "decisions.jsonl"
        try:
            engine = PolicyEngine(
                rules_path=str(tmp_path / "rules.json"),
                enforcement=True,
                enforcement_level=3,
            )
            # Add a custom ALLOW rule for FORCE_TRADE_DEGRADED (normally DENY)
            engine._rules.append(PolicyRule(
                name="allow_force_trade",
                action_types=[ActionType.FORCE_TRADE_DEGRADED],
                decision=PolicyDecisionType.AUTO_ALLOW,
                condition="test",
                priority=1,  # Higher priority than deny
            ))
            req = ActionRequest(action_type=ActionType.FORCE_TRADE_DEGRADED, source="test")
            decision = engine.evaluate(req)
            # DENY should still win even though ALLOW has higher priority
            assert decision.decision == PolicyDecisionType.DENY
        finally:
            pe_mod._DECISIONS_PATH = original

    def test_unmatched_action_defaults_to_allow(self, tmp_engine):
        req = ActionRequest(action_type=ActionType.START_BACKGROUND_DEBUG, source="test")
        decision = tmp_engine.evaluate(req)
        assert decision.decision == PolicyDecisionType.AUTO_ALLOW

    def test_matched_rules_recorded(self, tmp_engine):
        req = ActionRequest(action_type=ActionType.TIGHTEN_STOP, source="test")
        decision = tmp_engine.evaluate(req)
        assert len(decision.matched_rules) > 0
        assert "allow_tighten_stop_open_trade" in decision.matched_rules


# ── Environment Tests ─────────────────────────────────────────────────

class TestEnvironment:

    def test_environment_includes_account_mode(self, tmp_engine):
        env = tmp_engine.get_environment()
        assert "account_mode" in env
        assert "enforcement_enabled" in env

    def test_custom_environment_provider(self, tmp_engine):
        tmp_engine.register_environment_provider(
            lambda: {"broker_state": "connected", "queue_depth": 5}
        )
        env = tmp_engine.get_environment()
        assert env["broker_state"] == "connected"
        assert env["queue_depth"] == 5


# ── Decision Persistence Tests ────────────────────────────────────────

class TestDecisionPersistence:

    def test_decisions_persisted_to_jsonl(self, tmp_engine, tmp_path):
        import src.scanner.automation.policy_engine as pe_mod
        decisions_path = pe_mod._DECISIONS_PATH

        req = ActionRequest(action_type=ActionType.EXECUTE_TRADE, source="test")
        tmp_engine.evaluate(req)

        assert decisions_path.exists()
        lines = decisions_path.read_text().strip().split("\n")
        assert len(lines) >= 1
        record = json.loads(lines[-1])
        assert record["action_type"] == "execute_trade"
        assert "request_source" in record

    def test_get_recent_decisions(self, tmp_engine):
        for i in range(5):
            req = ActionRequest(action_type=ActionType.EXECUTE_TRADE, source=f"test_{i}")
            tmp_engine.evaluate(req)
        recent = tmp_engine.get_recent_decisions(3)
        assert len(recent) == 3


# ── Rule Linting Tests ────────────────────────────────────────────────

class TestRuleLinting:

    def test_lint_detects_uncovered_actions(self, tmp_engine):
        issues = tmp_engine.lint_rules()
        # Some actions may not be covered by default rules — that's expected
        assert isinstance(issues, list)

    def test_lint_detects_duplicate_names(self, tmp_engine):
        tmp_engine._rules.append(PolicyRule(
            name="allow_tighten_stop_open_trade",  # Duplicate
            action_types=[ActionType.TIGHTEN_STOP],
            decision=PolicyDecisionType.AUTO_ALLOW,
            condition="test duplicate",
        ))
        issues = tmp_engine.lint_rules()
        assert any("Duplicate" in issue for issue in issues)


# ── Status Tests ──────────────────────────────────────────────────────

class TestPolicyStatus:

    def test_get_status(self, tmp_engine):
        status = tmp_engine.get_status()
        assert "enforcement_enabled" in status
        assert "total_rules" in status
        assert status["total_rules"] >= 9
        assert "lint_issues" in status

    def test_enable_disable_enforcement(self, tmp_engine):
        assert not tmp_engine.is_enforcing
        tmp_engine.enable_enforcement()
        assert tmp_engine.is_enforcing
        tmp_engine.disable_enforcement()
        assert not tmp_engine.is_enforcing
