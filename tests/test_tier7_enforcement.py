"""Tests for Tier 7 PR 7: Controlled enforcement rollout.

Validates:
1. Enforcement levels (0-3) with per-action-type gating
2. Level 0 = logging-only (all actions auto-allowed)
3. Level 1 = safe actions enforced (tighten_stop, rescan, debug)
4. Level 2 = + config changes (model switch, threshold adjust, retrain)
5. Level 3 = full enforcement (including trade execution)
6. Transition between levels
"""

import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from src.scanner.automation.policy_types import (
    ActionType,
    ActionRequest,
    PolicyDecisionType,
    PolicyRule,
)
from src.scanner.automation.policy_engine import (
    PolicyEngine,
    reset_policy_engine,
)


@pytest.fixture(autouse=True)
def _reset():
    reset_policy_engine()
    yield
    reset_policy_engine()


@pytest.fixture
def engine(tmp_path):
    import src.scanner.automation.policy_engine as pe_mod
    original = pe_mod._DECISIONS_PATH
    pe_mod._DECISIONS_PATH = tmp_path / "decisions.jsonl"
    e = PolicyEngine(rules_path=str(tmp_path / "rules.json"), enforcement=False)
    yield e
    pe_mod._DECISIONS_PATH = original


# ── Enforcement Level Tests ───────────────────────────────────────────

class TestEnforcementLevels:

    def test_level_0_allows_everything(self, engine):
        """Level 0 = logging-only. Even DENY rules don't block."""
        engine.set_enforcement_level(0)
        req = ActionRequest(action_type=ActionType.FORCE_TRADE_DEGRADED, source="test")
        decision = engine.evaluate(req)
        assert decision.decision == PolicyDecisionType.AUTO_ALLOW

    def test_level_1_enforces_safe_actions(self, engine):
        """Level 1 enforces tighten_stop, rescan, debug."""
        engine.set_enforcement_level(1)

        # TIGHTEN_STOP is in level 1 — should be enforced (auto_allow by default rule)
        req = ActionRequest(action_type=ActionType.TIGHTEN_STOP, source="test")
        decision = engine.evaluate(req)
        assert decision.decision == PolicyDecisionType.AUTO_ALLOW  # Rule says allow

        # EXECUTE_TRADE is NOT in level 1 — even DENY rules won't block
        # (but there's no deny rule for execute_trade, just an allow for paper)
        req = ActionRequest(action_type=ActionType.FORCE_TRADE_DEGRADED, source="test")
        decision = engine.evaluate(req)
        assert decision.decision == PolicyDecisionType.AUTO_ALLOW  # Not enforced at level 1

    def test_level_2_enforces_config_changes(self, engine):
        """Level 2 adds model switch, threshold adjust, retrain."""
        engine.set_enforcement_level(2)

        # SWITCH_PAIR_MODEL is in level 2 — NEEDS_CONFIRMATION rule should enforce
        req = ActionRequest(action_type=ActionType.SWITCH_PAIR_MODEL, source="test")
        decision = engine.evaluate(req)
        # Default rule says NEEDS_CONFIRMATION for model switch with few trades
        assert decision.decision == PolicyDecisionType.NEEDS_CONFIRMATION

        # FORCE_TRADE_DEGRADED is NOT in level 2 — should still be allowed
        req2 = ActionRequest(action_type=ActionType.FORCE_TRADE_DEGRADED, source="test")
        decision2 = engine.evaluate(req2)
        assert decision2.decision == PolicyDecisionType.AUTO_ALLOW

    def test_level_3_enforces_everything(self, engine):
        """Level 3 = full enforcement including trade execution."""
        engine.set_enforcement_level(3)

        # FORCE_TRADE_DEGRADED has a DENY rule — should actually deny
        req = ActionRequest(action_type=ActionType.FORCE_TRADE_DEGRADED, source="test")
        decision = engine.evaluate(req)
        assert decision.decision == PolicyDecisionType.DENY

    def test_invalid_level_raises(self, engine):
        with pytest.raises(ValueError):
            engine.set_enforcement_level(5)

    def test_level_transition(self, engine):
        """Can move between levels."""
        engine.set_enforcement_level(1)
        assert engine.enforcement_level == 1
        assert engine.is_enforcing is True

        engine.set_enforcement_level(0)
        assert engine.enforcement_level == 0
        assert engine.is_enforcing is False

        engine.set_enforcement_level(3)
        assert engine.enforcement_level == 3
        assert engine.is_enforcing is True

    def test_disable_enforcement_sets_level_0(self, engine):
        engine.set_enforcement_level(3)
        engine.disable_enforcement()
        assert engine.enforcement_level == 0
        assert engine.is_enforcing is False

    def test_enable_enforcement_sets_level_3(self, engine):
        engine.enable_enforcement()
        assert engine.enforcement_level == 3
        assert engine.is_enforcing is True


# ── Status Reports Level Info ─────────────────────────────────────────

class TestEnforcementStatus:

    def test_status_includes_level(self, engine):
        engine.set_enforcement_level(2)
        status = engine.get_status()
        assert status["enforcement_level"] == 2
        assert "enforced_actions" in status
        assert "switch_pair_model" in status["enforced_actions"]

    def test_status_level_0_empty_actions(self, engine):
        engine.set_enforcement_level(0)
        status = engine.get_status()
        assert status["enforced_actions"] == []


# ── Logging-Only Records Would-Be Decision ────────────────────────────

class TestLoggingOnlyRecords:

    def test_logging_records_would_be_deny(self, engine):
        """At level 0, DENY rules log but don't block."""
        engine.set_enforcement_level(0)
        req = ActionRequest(action_type=ActionType.FORCE_TRADE_DEGRADED, source="test")
        decision = engine.evaluate(req)
        assert decision.decision == PolicyDecisionType.AUTO_ALLOW
        assert any("LOGGING_ONLY" in r for r in decision.reasons)
        assert any("level 0" in r for r in decision.reasons)

    def test_partial_enforcement_logs_unenforced(self, engine):
        """At level 1, unenforced action types still log but don't block."""
        engine.set_enforcement_level(1)
        req = ActionRequest(action_type=ActionType.FORCE_TRADE_DEGRADED, source="test")
        decision = engine.evaluate(req)
        assert decision.decision == PolicyDecisionType.AUTO_ALLOW
        assert any("LOGGING_ONLY" in r for r in decision.reasons)
