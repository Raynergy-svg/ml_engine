"""Bridge Rules Engine — Cross-engine promoted rules that gate both Aura and Buddy.

PRD v2.2 §13 Phase 4: "Bridge rules" and "Aura rule promoter"

Bridge rules are promoted patterns that span both the human engine (Aura) and
the trading engine (Buddy). They modify runtime behavior on both sides:

  Aura → Buddy:
    - Emotional drift detected → raise min_confidence_threshold
    - Stress accumulation → reduce max_position_risk
    - Override pattern (losing) → tighten execution gates

  Buddy → Aura:
    - Losing streak → trigger empathy-first conversation mode
    - High drawdown → suppress readiness score boost
    - Exceptional performance → flag potential overconfidence

Rules are stored as JSON in .aura/bridge/active_rules.json and evaluated
by both engines before key decisions.

Zero-dependency — uses only Python stdlib and project types.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# --- Bridge Rule Types ---

RULE_TYPES = {
    # Aura → Buddy gates
    "raise_min_confidence": {
        "direction": "aura_to_buddy",
        "target": "scanner_config",
        "parameter": "min_confidence_threshold",
        "description": "Raise minimum confidence threshold when human state is compromised",
    },
    "reduce_position_risk": {
        "direction": "aura_to_buddy",
        "target": "risk_config",
        "parameter": "max_risk_per_trade",
        "description": "Reduce per-trade risk when stress is elevated",
    },
    "tighten_rr_ratio": {
        "direction": "aura_to_buddy",
        "target": "scanner_config",
        "parameter": "min_rr_ratio",
        "description": "Require higher R:R when override patterns suggest impulsivity",
    },
    "pause_trading": {
        "direction": "aura_to_buddy",
        "target": "execution",
        "parameter": "trading_enabled",
        "description": "Pause trading when critical emotional state detected",
    },
    # Buddy → Aura gates
    "empathy_mode": {
        "direction": "buddy_to_aura",
        "target": "companion_config",
        "parameter": "conversation_mode",
        "description": "Switch to empathy-first mode during losing streaks",
    },
    "suppress_readiness_boost": {
        "direction": "buddy_to_aura",
        "target": "readiness_config",
        "parameter": "max_readiness_boost",
        "description": "Cap readiness boost during drawdown to prevent false confidence",
    },
    "flag_overconfidence": {
        "direction": "buddy_to_aura",
        "target": "companion_config",
        "parameter": "overconfidence_alert",
        "description": "Alert companion when exceptional performance may breed overconfidence",
    },
}


@dataclass
class BridgeRule:
    """A cross-engine rule that gates behavior on both sides.

    Rules have a limited lifespan (TTL) and automatically expire if not
    reinforced by continued pattern observations.
    """

    rule_id: str
    rule_type: str                    # Key from RULE_TYPES
    direction: str                    # "aura_to_buddy" or "buddy_to_aura"
    description: str
    adjustment: Dict[str, Any]        # {"parameter": "X", "value": Y, "operator": "set|add|multiply"}
    source_pattern_ids: List[str] = field(default_factory=list)
    source_count: int = 0             # How many pattern observations triggered this
    confidence: float = 0.5
    active: bool = True
    created_at: str = ""
    expires_at: str = ""              # Rules expire after TTL
    last_evaluated: str = ""
    evaluation_count: int = 0
    triggered_count: int = 0          # How many times this rule actually modified behavior

    def __post_init__(self):
        now = datetime.now(timezone.utc)
        if not self.created_at:
            self.created_at = now.isoformat()
        if not self.expires_at:
            # Default TTL: 14 days
            self.expires_at = (now + timedelta(days=14)).isoformat()

    def is_expired(self, now: Optional[datetime] = None) -> bool:
        """Check if this rule has passed its expiration.

        US-206: Fail-safe — malformed or missing expires_at is treated as expired
        (not silently kept active). Logs a warning so corrupted rules are visible.
        """
        now = now or datetime.now(timezone.utc)
        if not self.expires_at:
            logger.warning(
                "BridgeRule %s has empty expires_at — treating as expired (fail-safe)",
                self.rule_id,
            )
            return True
        try:
            exp = datetime.fromisoformat(self.expires_at.replace("Z", "+00:00"))
            return now > exp
        except (ValueError, AttributeError) as e:
            logger.warning(
                "BridgeRule %s has malformed expires_at '%s': %s — treating as expired (fail-safe)",
                self.rule_id, self.expires_at, e,
            )
            return True

    def extend_ttl(self, days: int = 14) -> None:
        """Extend the rule's TTL (called when reinforcing pattern is observed)."""
        now = datetime.now(timezone.utc)
        self.expires_at = (now + timedelta(days=days)).isoformat()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "rule_id": self.rule_id,
            "rule_type": self.rule_type,
            "direction": self.direction,
            "description": self.description,
            "adjustment": self.adjustment,
            "source_pattern_ids": self.source_pattern_ids,
            "source_count": self.source_count,
            "confidence": round(self.confidence, 3),
            "active": self.active,
            "created_at": self.created_at,
            "expires_at": self.expires_at,
            "last_evaluated": self.last_evaluated,
            "evaluation_count": self.evaluation_count,
            "triggered_count": self.triggered_count,
        }


class BridgeRulesEngine:
    """Manages cross-engine bridge rules.

    Responsibilities:
    1. Create bridge rules from promoted patterns
    2. Evaluate active rules against current state
    3. Return gate modifications for both engines
    4. Expire stale rules and persist state

    Args:
        rules_path: Where to persist active rules
    """

    def __init__(self, rules_path: Optional[Path] = None):
        self._rules_path = rules_path or Path(".aura/bridge/active_rules.json")
        self._rules_path.parent.mkdir(parents=True, exist_ok=True)
        self._rules: List[BridgeRule] = self._load_rules()

    # --- Rule Creation ---

    def create_rule_from_pattern(
        self,
        pattern_id: str,
        pattern_description: str,
        pattern_domain: str,
        pattern_confidence: float,
        observation_count: int,
        suggested_rule_type: Optional[str] = None,
    ) -> Optional[BridgeRule]:
        """Create a bridge rule from a promoted pattern.

        Args:
            pattern_id: ID of the source pattern
            pattern_description: Human-readable description
            pattern_domain: "human", "trading", or "cross"
            pattern_confidence: Pattern confidence (0-1)
            observation_count: How many times pattern was observed
            suggested_rule_type: Explicit rule type (auto-detected if None)

        Returns:
            BridgeRule if created, None if pattern doesn't warrant a rule
        """
        # Determine rule type
        rule_type = suggested_rule_type or self._infer_rule_type(
            pattern_description, pattern_domain
        )
        if not rule_type or rule_type not in RULE_TYPES:
            logger.debug(
                f"No bridge rule type for pattern '{pattern_id}' "
                f"(domain={pattern_domain})"
            )
            return None

        # Check if a rule of this type already exists
        existing = self._find_rule_by_type(rule_type)
        if existing:
            # Reinforce existing rule
            existing.source_pattern_ids.append(pattern_id)
            existing.source_count += observation_count
            existing.confidence = min(1.0, existing.confidence + 0.05)
            existing.extend_ttl()
            self._save_rules()
            logger.info(
                f"Reinforced bridge rule '{existing.rule_id}' "
                f"(confidence={existing.confidence:.2f})"
            )
            return existing

        # Create new rule
        meta = RULE_TYPES[rule_type]
        adjustment = self._compute_adjustment(
            rule_type, pattern_confidence, observation_count
        )

        rule = BridgeRule(
            rule_id=f"bridge-{rule_type}-{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S')}",
            rule_type=rule_type,
            direction=meta["direction"],
            description=meta["description"],
            adjustment=adjustment,
            source_pattern_ids=[pattern_id],
            source_count=observation_count,
            confidence=pattern_confidence,
        )

        self._rules.append(rule)
        self._save_rules()
        logger.info(f"Created bridge rule: {rule.rule_id} ({rule_type})")
        return rule

    # --- Rule Evaluation ---

    def get_buddy_gate_adjustments(self) -> Dict[str, Any]:
        """Get all active adjustments that should modify Buddy's gates.

        Returns dict of parameter → adjusted_value for scanner/risk config.
        Called by the orchestrator before each scan cycle.
        """
        adjustments: Dict[str, Any] = {}
        now = datetime.now(timezone.utc)

        # US-210: Sort rules by operator precedence: multiply → add → set
        # This ensures deterministic evaluation: multiplicative adjustments first,
        # additive adjustments second, and explicit overrides (set) last.
        _OP_ORDER = {"multiply": 0, "add": 1, "set": 2}
        active_rules = []
        for rule in self._rules:
            if not rule.active or rule.direction != "aura_to_buddy":
                continue
            if rule.is_expired(now):
                rule.active = False
                continue
            active_rules.append(rule)

        active_rules.sort(
            key=lambda r: _OP_ORDER.get(r.adjustment.get("operator", "set"), 2)
        )

        for rule in active_rules:
            rule.last_evaluated = now.isoformat()
            rule.evaluation_count += 1

            param = rule.adjustment.get("parameter", "")
            value = rule.adjustment.get("value")
            operator = rule.adjustment.get("operator", "set")

            if operator == "set":
                adjustments[param] = value
            elif operator == "add" and param in adjustments:
                adjustments[param] = adjustments[param] + value
            elif operator == "add":
                adjustments[param] = value
            elif operator == "multiply" and param in adjustments:
                adjustments[param] = adjustments[param] * value

            rule.triggered_count += 1

        # US-210: Clamp final values to valid ranges
        _CLAMP_RANGES = {
            "min_confidence_threshold": (0.0, 1.0),
            "max_uncertainty_score": (0.0, 1.0),
            "weighted_vote_threshold": (0.0, 1.0),
            "position_size_multiplier": (0.3, 2.0),
            "risk_multiplier": (0.3, 2.0),
            "min_risk_reward_ratio": (0.5, 5.0),
        }
        for param, (lo, hi) in _CLAMP_RANGES.items():
            if param in adjustments and isinstance(adjustments[param], (int, float)):
                original = adjustments[param]
                adjustments[param] = max(lo, min(hi, adjustments[param]))
                if adjustments[param] != original:
                    logger.info(
                        "US-210: Clamped %s from %.3f to %.3f (range [%.1f, %.1f])",
                        param, original, adjustments[param], lo, hi,
                    )

        self._save_rules()
        return adjustments

    def get_aura_gate_adjustments(self) -> Dict[str, Any]:
        """Get all active adjustments that should modify Aura's behavior.

        Returns dict of parameter → adjusted_value for companion/readiness config.
        Called by the companion before generating responses.
        """
        adjustments: Dict[str, Any] = {}
        now = datetime.now(timezone.utc)

        for rule in self._rules:
            if not rule.active or rule.direction != "buddy_to_aura":
                continue
            if rule.is_expired(now):
                rule.active = False
                continue

            rule.last_evaluated = now.isoformat()
            rule.evaluation_count += 1

            param = rule.adjustment.get("parameter", "")
            value = rule.adjustment.get("value")
            adjustments[param] = value
            rule.triggered_count += 1

        self._save_rules()
        return adjustments

    # --- Rule Lifecycle ---

    def expire_stale_rules(self) -> int:
        """Expire rules past their TTL. Returns count of expired rules."""
        now = datetime.now(timezone.utc)
        expired = 0
        for rule in self._rules:
            if rule.active and rule.is_expired(now):
                rule.active = False
                expired += 1
                logger.info(f"Bridge rule expired: {rule.rule_id}")

        if expired:
            self._save_rules()
        return expired

    def get_active_rules(self) -> List[BridgeRule]:
        """Get all currently active (non-expired) bridge rules."""
        now = datetime.now(timezone.utc)
        return [r for r in self._rules if r.active and not r.is_expired(now)]

    def get_rules_summary(self) -> Dict[str, Any]:
        """Get a summary of all bridge rules for status display."""
        active = self.get_active_rules()
        expired = [r for r in self._rules if not r.active or r.is_expired()]

        return {
            "total_rules": len(self._rules),
            "active_rules": len(active),
            "expired_rules": len(expired),
            "aura_to_buddy": len([r for r in active if r.direction == "aura_to_buddy"]),
            "buddy_to_aura": len([r for r in active if r.direction == "buddy_to_aura"]),
            "rules": [r.to_dict() for r in active],
        }

    # --- Inference ---

    def _infer_rule_type(self, description: str, domain: str) -> Optional[str]:
        """Infer the appropriate bridge rule type from pattern description."""
        desc = description.lower()

        # Aura → Buddy rules (human state affects trading)
        # Check more specific phrases before broader keywords
        if domain in ("human", "cross"):
            if any(kw in desc for kw in ["stress accumulation", "burnout", "fatigue"]):
                return "reduce_position_risk"
            if any(kw in desc for kw in ["emotional drift", "emotion", "anxiety"]):
                return "raise_min_confidence"
            if any(kw in desc for kw in ["override", "impuls", "impatient"]):
                return "tighten_rr_ratio"
            if any(kw in desc for kw in ["crisis", "breakdown", "severe"]):
                return "pause_trading"

        # Buddy → Aura rules (trading outcomes affect human engine)
        if domain in ("trading", "cross"):
            if any(kw in desc for kw in ["losing streak", "consecutive loss"]):
                return "empathy_mode"
            if any(kw in desc for kw in ["drawdown", "max loss"]):
                return "suppress_readiness_boost"
            if any(kw in desc for kw in ["exceptional", "overperform", "hot streak"]):
                return "flag_overconfidence"

        return None

    def _compute_adjustment(
        self,
        rule_type: str,
        confidence: float,
        observation_count: int,
    ) -> Dict[str, Any]:
        """Compute the specific parameter adjustment for a rule type.

        Adjustments scale with confidence and observation count.
        Higher confidence + more observations = stronger adjustment.
        """
        # Scale factor: 0.5-1.0 based on confidence, boosted by observations
        scale = min(1.0, confidence * (1 + 0.1 * min(observation_count, 5)))

        adjustments = {
            "raise_min_confidence": {
                "parameter": "min_confidence_threshold",
                "value": round(0.55 + 0.15 * scale, 2),  # 0.55 → 0.70
                "operator": "set",
            },
            "reduce_position_risk": {
                "parameter": "max_risk_per_trade",
                "value": round(0.03 - 0.01 * scale, 3),  # 3% → 2%
                "operator": "set",
            },
            "tighten_rr_ratio": {
                "parameter": "min_rr_ratio",
                "value": round(1.2 + 0.5 * scale, 2),  # 1.2 → 1.7
                "operator": "set",
            },
            "pause_trading": {
                "parameter": "trading_enabled",
                "value": False,
                "operator": "set",
            },
            "empathy_mode": {
                "parameter": "conversation_mode",
                "value": "empathy_first",
                "operator": "set",
            },
            "suppress_readiness_boost": {
                "parameter": "max_readiness_boost",
                "value": round(0.05 - 0.03 * scale, 3),  # 5% → 2%
                "operator": "set",
            },
            "flag_overconfidence": {
                "parameter": "overconfidence_alert",
                "value": True,
                "operator": "set",
            },
        }

        return adjustments.get(rule_type, {
            "parameter": "unknown",
            "value": None,
            "operator": "set",
        })

    def _find_rule_by_type(self, rule_type: str) -> Optional[BridgeRule]:
        """Find an active rule of a given type."""
        for rule in self._rules:
            if rule.rule_type == rule_type and rule.active:
                return rule
        return None

    # --- Persistence ---

    def _load_rules(self) -> List[BridgeRule]:
        """Load rules from disk."""
        if not self._rules_path.exists():
            return []
        try:
            data = json.loads(self._rules_path.read_text())
            return [BridgeRule(**r) for r in data]
        except Exception as e:
            logger.warning(f"Failed to load bridge rules: {e}")
            return []

    def _save_rules(self) -> None:
        """Persist rules to disk."""
        try:
            self._rules_path.write_text(
                json.dumps([r.to_dict() for r in self._rules], indent=2)
            )
        except Exception as e:
            logger.warning(f"Failed to save bridge rules: {e}")
