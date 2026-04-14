"""Tests for SupervisionPolicyEngine — policy stacking, expiration, adjustments."""

import time

import pytest

from src.scanner.automation.supervision_policy_engine import (
    AdaptivePolicy,
    PolicyType,
    SupervisionPolicyEngine,
)


def _make_policy(
    policy_type=PolicyType.GATE_TIGHTEN,
    parameter="confidence_threshold",
    adjustment=0.05,
    reason="test",
    ttl=3600,
    pair=None,
):
    """Helper to create a policy expiring `ttl` seconds from now."""
    return AdaptivePolicy(
        policy_type=policy_type,
        parameter=parameter,
        adjustment=adjustment,
        reason=reason,
        expires_at=time.time() + ttl,
        pair=pair,
    )


class TestSupervisonPolicyEngine:
    """Test suite for SupervisionPolicyEngine."""

    def test_add_policy_stores_policy(self):
        engine = SupervisionPolicyEngine()
        p = _make_policy()
        engine.add_policy(p)
        active = engine.get_active_policies()
        assert len(active) == 1
        assert active[0].policy_id == p.policy_id

    def test_get_active_policies_filters_expired(self):
        engine = SupervisionPolicyEngine()
        alive = _make_policy(ttl=3600)
        dead = _make_policy(ttl=-1)  # already expired
        engine.add_policy(alive)
        engine.add_policy(dead)
        active = engine.get_active_policies()
        assert len(active) == 1
        assert active[0].policy_id == alive.policy_id

    def test_get_effective_adjustment_sums_multiple(self):
        engine = SupervisionPolicyEngine()
        engine.add_policy(_make_policy(adjustment=0.05, pair="EUR_USD"))
        engine.add_policy(_make_policy(adjustment=0.03, pair="EUR_USD"))
        engine.add_policy(_make_policy(adjustment=0.10, pair="GBP_USD"))  # different pair
        total = engine.get_effective_adjustment("EUR_USD", "confidence_threshold")
        assert total == pytest.approx(0.08)

    def test_global_policies_apply_to_all_pairs(self):
        engine = SupervisionPolicyEngine()
        # Global policy (pair=None) should apply to any pair
        engine.add_policy(_make_policy(adjustment=0.02, pair=None))
        total_eur = engine.get_effective_adjustment("EUR_USD", "confidence_threshold")
        total_gbp = engine.get_effective_adjustment("GBP_USD", "confidence_threshold")
        assert total_eur == pytest.approx(0.02)
        assert total_gbp == pytest.approx(0.02)

    def test_expire_stale_removes_old_and_returns_count(self):
        engine = SupervisionPolicyEngine()
        engine.add_policy(_make_policy(ttl=-1))
        engine.add_policy(_make_policy(ttl=-1))
        engine.add_policy(_make_policy(ttl=3600))
        removed = engine.expire_stale()
        assert removed == 2
        assert len(engine.get_active_policies()) == 1

    def test_has_active_block_detects_entry_block(self):
        engine = SupervisionPolicyEngine()
        engine.add_policy(
            _make_policy(policy_type=PolicyType.ENTRY_BLOCK, pair="EUR_USD")
        )
        assert engine.has_active_block("EUR_USD") is True
        assert engine.has_active_block("GBP_USD") is False

    def test_has_active_block_detects_pair_cooldown(self):
        engine = SupervisionPolicyEngine()
        engine.add_policy(
            _make_policy(policy_type=PolicyType.PAIR_COOLDOWN, pair="USD_JPY")
        )
        assert engine.has_active_block("USD_JPY") is True
        assert engine.has_active_block("AUD_USD") is False

    def test_has_active_block_global_block_applies_to_any_pair(self):
        engine = SupervisionPolicyEngine()
        engine.add_policy(
            _make_policy(policy_type=PolicyType.ENTRY_BLOCK, pair=None)
        )
        assert engine.has_active_block("EUR_USD") is True
        assert engine.has_active_block("GBP_USD") is True

    def test_clear_removes_all_policies(self):
        engine = SupervisionPolicyEngine()
        for _ in range(5):
            engine.add_policy(_make_policy())
        engine.clear()
        assert len(engine.get_active_policies()) == 0

    def test_to_dict_serializes_correctly(self):
        engine = SupervisionPolicyEngine()
        engine.add_policy(_make_policy(pair="EUR_USD", ttl=3600))
        engine.add_policy(_make_policy(pair="GBP_USD", ttl=-1))  # expired
        result = engine.to_dict()
        assert "active_policies" in result
        assert "total_count" in result
        # Only the non-expired policy should appear in active_policies
        assert len(result["active_policies"]) == 1
        assert result["total_count"] == 2  # both still in internal list
        # Each policy dict should have string policy_type, not enum
        p_dict = result["active_policies"][0]
        assert isinstance(p_dict["policy_type"], str)
        assert p_dict["pair"] == "EUR_USD"

    def test_get_active_policies_filters_by_type(self):
        engine = SupervisionPolicyEngine()
        engine.add_policy(_make_policy(policy_type=PolicyType.GATE_TIGHTEN))
        engine.add_policy(_make_policy(policy_type=PolicyType.SIZING_ADJUST))
        active = engine.get_active_policies(policy_type=PolicyType.GATE_TIGHTEN)
        assert len(active) == 1
        assert active[0].policy_type == PolicyType.GATE_TIGHTEN
