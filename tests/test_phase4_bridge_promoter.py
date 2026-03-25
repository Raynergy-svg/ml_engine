"""Tests for Phase 4: Bridge Rules Engine + Aura Rule Promoter.

Covers:
- BridgeRulesEngine: rule creation, evaluation, expiration, persistence
- AuraRulePromoter: pattern scanning, promotion pipeline, dedup
- Integration: pattern → promoter → bridge rule → gate adjustment
"""

import json
import sys
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.aura.bridge.rules_engine import (
    RULE_TYPES,
    BridgeRule,
    BridgeRulesEngine,
)
from src.aura.patterns.base import (
    DetectedPattern,
    EvidenceItem,
    PatternDomain,
    PatternStatus,
    PatternTier,
)
from src.aura.patterns.rule_promoter import AuraRulePromoter


# --- Fixtures ---


@pytest.fixture
def tmp_dir():
    with tempfile.TemporaryDirectory() as d:
        yield Path(d)


@pytest.fixture
def engine(tmp_dir):
    return BridgeRulesEngine(rules_path=tmp_dir / "active_rules.json")


@pytest.fixture
def promoter(engine, tmp_dir):
    return AuraRulePromoter(
        rules_engine=engine,
        promotion_log_path=tmp_dir / "promotion_log.jsonl",
    )


def _make_pattern(
    pattern_id: str,
    description: str,
    domain: PatternDomain = PatternDomain.HUMAN,
    confidence: float = 0.7,
    observation_count: int = 3,
    status: PatternStatus = PatternStatus.RECURRING,
    suggested_rule: str = None,
) -> DetectedPattern:
    """Helper to create a promotable pattern."""
    p = DetectedPattern(
        pattern_id=pattern_id,
        tier=PatternTier.T2_WEEKLY,
        domain=domain,
        description=description,
        confidence=confidence,
        observation_count=observation_count,
        status=status,
        suggested_rule=suggested_rule,
    )
    # Add enough evidence to be promotable
    for i in range(observation_count):
        p.evidence.append(EvidenceItem(
            source_type="conversation",
            source_id=f"conv-{i}",
            timestamp=datetime.now(timezone.utc).isoformat(),
            summary=f"Evidence {i}",
        ))
    return p


# ============================================================
# Bridge Rules Engine Tests
# ============================================================


def test_create_rule_from_pattern(engine):
    """Create a bridge rule from a pattern description."""
    rule = engine.create_rule_from_pattern(
        pattern_id="test-pattern-001",
        pattern_description="Emotional drift detected over 3 weeks",
        pattern_domain="human",
        pattern_confidence=0.75,
        observation_count=4,
    )

    assert rule is not None
    assert rule.rule_type == "raise_min_confidence"
    assert rule.direction == "aura_to_buddy"
    assert rule.active is True
    assert "test-pattern-001" in rule.source_pattern_ids


def test_create_rule_explicit_type(engine):
    """Create rule with explicit type override."""
    rule = engine.create_rule_from_pattern(
        pattern_id="p-002",
        pattern_description="Something vague",
        pattern_domain="human",
        pattern_confidence=0.8,
        observation_count=3,
        suggested_rule_type="pause_trading",
    )

    assert rule is not None
    assert rule.rule_type == "pause_trading"
    assert rule.adjustment["value"] is False


def test_reinforce_existing_rule(engine):
    """Second pattern of same type reinforces existing rule."""
    engine.create_rule_from_pattern(
        pattern_id="p-001",
        pattern_description="Emotional drift first time",
        pattern_domain="human",
        pattern_confidence=0.6,
        observation_count=3,
    )

    rule = engine.create_rule_from_pattern(
        pattern_id="p-002",
        pattern_description="Emotional drift again",
        pattern_domain="human",
        pattern_confidence=0.7,
        observation_count=4,
    )

    assert rule is not None
    assert len(rule.source_pattern_ids) == 2
    assert rule.confidence > 0.6  # Should have been boosted


def test_get_buddy_gate_adjustments(engine):
    """Active aura→buddy rules produce gate adjustments."""
    engine.create_rule_from_pattern(
        pattern_id="p-stress",
        pattern_description="Stress accumulation over 4 weeks",
        pattern_domain="human",
        pattern_confidence=0.8,
        observation_count=5,
    )

    adjustments = engine.get_buddy_gate_adjustments()
    assert len(adjustments) > 0
    assert "max_risk_per_trade" in adjustments


def test_get_aura_gate_adjustments(engine):
    """Active buddy→aura rules produce gate adjustments."""
    engine.create_rule_from_pattern(
        pattern_id="p-streak",
        pattern_description="Losing streak 5 consecutive losses",
        pattern_domain="trading",
        pattern_confidence=0.85,
        observation_count=3,
    )

    adjustments = engine.get_aura_gate_adjustments()
    assert "conversation_mode" in adjustments
    assert adjustments["conversation_mode"] == "empathy_first"


def test_rule_expiration(engine):
    """Rules expire after TTL."""
    rule = engine.create_rule_from_pattern(
        pattern_id="p-exp",
        pattern_description="Emotional drift test",
        pattern_domain="human",
        pattern_confidence=0.7,
        observation_count=3,
    )

    # Manually set expires_at to past
    rule.expires_at = (datetime.now(timezone.utc) - timedelta(days=1)).isoformat()
    engine._save_rules()

    expired_count = engine.expire_stale_rules()
    assert expired_count == 1

    active = engine.get_active_rules()
    assert len(active) == 0


def test_rules_persistence(tmp_dir):
    """Rules persist across engine instances."""
    path = tmp_dir / "rules.json"

    e1 = BridgeRulesEngine(rules_path=path)
    e1.create_rule_from_pattern(
        pattern_id="p-persist",
        pattern_description="Override pattern increasing",
        pattern_domain="human",
        pattern_confidence=0.7,
        observation_count=3,
    )

    # New instance should load the saved rule
    e2 = BridgeRulesEngine(rules_path=path)
    assert len(e2.get_active_rules()) == 1


def test_rules_summary(engine):
    """Summary includes correct counts."""
    engine.create_rule_from_pattern(
        pattern_id="p-a2b",
        pattern_description="Emotional drift",
        pattern_domain="human",
        pattern_confidence=0.7,
        observation_count=3,
    )
    engine.create_rule_from_pattern(
        pattern_id="p-b2a",
        pattern_description="Losing streak pattern",
        pattern_domain="trading",
        pattern_confidence=0.8,
        observation_count=4,
    )

    summary = engine.get_rules_summary()
    assert summary["total_rules"] == 2
    assert summary["active_rules"] == 2
    assert summary["aura_to_buddy"] == 1
    assert summary["buddy_to_aura"] == 1


def test_no_rule_for_irrelevant_pattern(engine):
    """Pattern with no bridge-relevant keywords returns None."""
    rule = engine.create_rule_from_pattern(
        pattern_id="p-irrelevant",
        pattern_description="User prefers dark mode",
        pattern_domain="human",
        pattern_confidence=0.9,
        observation_count=5,
    )
    assert rule is None


# ============================================================
# Aura Rule Promoter Tests
# ============================================================


def test_promote_eligible_pattern(promoter, engine):
    """Eligible pattern with bridge relevance gets promoted."""
    pattern = _make_pattern(
        "pat-emotional-drift",
        "Emotional drift detected over 3 weeks — negative trend",
        domain=PatternDomain.HUMAN,
        confidence=0.75,
        observation_count=4,
    )

    promotions = promoter.scan_and_promote([pattern])
    assert len(promotions) == 1
    assert promotions[0]["pattern_id"] == "pat-emotional-drift"
    assert pattern.status == PatternStatus.PROMOTED


def test_skip_non_promotable_pattern(promoter):
    """Pattern below threshold is skipped."""
    pattern = _make_pattern(
        "pat-weak",
        "Emotional drift",
        confidence=0.3,  # Too low
        observation_count=1,  # Not enough
        status=PatternStatus.DETECTED,
    )

    promotions = promoter.scan_and_promote([pattern])
    assert len(promotions) == 0


def test_skip_non_bridge_relevant(promoter):
    """Pattern without bridge keywords is skipped."""
    pattern = _make_pattern(
        "pat-no-bridge",
        "User logs in at consistent times",  # No bridge keywords
        domain=PatternDomain.HUMAN,
    )

    promotions = promoter.scan_and_promote([pattern])
    assert len(promotions) == 0


def test_dedup_already_promoted(promoter):
    """Same pattern isn't promoted twice."""
    pattern = _make_pattern(
        "pat-dedup",
        "Stress accumulation pattern",
    )

    promotions_1 = promoter.scan_and_promote([pattern])
    assert len(promotions_1) == 1

    # Reset status to test dedup (promoter tracks IDs)
    pattern.status = PatternStatus.RECURRING
    promotions_2 = promoter.scan_and_promote([pattern])
    assert len(promotions_2) == 0  # Already promoted


def test_cross_engine_patterns_always_relevant(promoter):
    """Cross-engine domain patterns always have bridge relevance."""
    pattern = _make_pattern(
        "pat-cross",
        "Generic cross-engine observation",
        domain=PatternDomain.CROSS_ENGINE,
    )

    # Cross-engine patterns always qualify for bridge relevance
    assert promoter._has_bridge_relevance(pattern) is True


def test_promotion_log_persisted(promoter, tmp_dir):
    """Promotion events are logged to JSONL file."""
    pattern = _make_pattern(
        "pat-log-test",
        "Override trajectory increasing",
    )

    promoter.scan_and_promote([pattern])

    log_path = tmp_dir / "promotion_log.jsonl"
    assert log_path.exists()

    with open(log_path) as f:
        entries = [json.loads(line) for line in f if line.strip()]
    assert len(entries) == 1
    assert entries[0]["pattern_id"] == "pat-log-test"


def test_promotion_stats(promoter):
    """Stats track total promotions."""
    patterns = [
        _make_pattern("p1", "Emotional drift pattern"),
        _make_pattern("p2", "Stress accumulation detected"),
    ]

    promoter.scan_and_promote(patterns)
    stats = promoter.get_promotion_stats()
    assert stats["total_promoted"] == 2


# ============================================================
# Integration: Full Pipeline
# ============================================================


def test_full_pipeline_pattern_to_gate_adjustment(tmp_dir):
    """End-to-end: pattern → promoter → bridge rule → gate adjustment."""
    rules_path = tmp_dir / "rules.json"
    log_path = tmp_dir / "promotions.jsonl"

    engine = BridgeRulesEngine(rules_path=rules_path)
    promoter = AuraRulePromoter(rules_engine=engine, promotion_log_path=log_path)

    # Create a promotable "emotional drift" pattern
    pattern = _make_pattern(
        "pipeline-test",
        "Emotional drift over 4 weeks — negative sentiment trend",
        domain=PatternDomain.HUMAN,
        confidence=0.8,
        observation_count=5,
    )

    # Promote it
    promotions = promoter.scan_and_promote([pattern])
    assert len(promotions) == 1

    # Get gate adjustments
    adjustments = engine.get_buddy_gate_adjustments()
    assert "min_confidence_threshold" in adjustments
    assert adjustments["min_confidence_threshold"] > 0.55  # Should be raised

    # Verify no aura-side adjustments (this is aura→buddy)
    aura_adj = engine.get_aura_gate_adjustments()
    assert "min_confidence_threshold" not in aura_adj


def test_bridge_rule_dataclass():
    """BridgeRule serialization round-trip."""
    rule = BridgeRule(
        rule_id="test-001",
        rule_type="raise_min_confidence",
        direction="aura_to_buddy",
        description="Test rule",
        adjustment={"parameter": "min_confidence_threshold", "value": 0.65, "operator": "set"},
        source_pattern_ids=["p1", "p2"],
        source_count=5,
        confidence=0.8,
    )

    d = rule.to_dict()
    assert d["rule_id"] == "test-001"
    assert json.dumps(d)  # Must be JSON-serializable

    # Reconstruct
    rule2 = BridgeRule(**d)
    assert rule2.rule_type == "raise_min_confidence"
    assert rule2.confidence == 0.8


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
