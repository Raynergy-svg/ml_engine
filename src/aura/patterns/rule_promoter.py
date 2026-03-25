"""Aura Rule Promoter — Pattern → Bridge Rule pipeline.

PRD v2.2 §13 Phase 4: "Aura rule promoter"

Scans all active patterns across T1/T2/T3 tiers and promotes eligible
patterns into bridge rules via the BridgeRulesEngine.

Promotion criteria (from patterns/base.py):
  - observation_count >= 3
  - confidence >= 0.6
  - status in (RECURRING, PROMOTED)
  - user_rating is None or >= 2.5

The promoter runs after the pattern engine and handles:
1. Scanning all tiers for promotable patterns
2. Mapping pattern domains/descriptions to bridge rule types
3. Creating or reinforcing bridge rules
4. Logging promotion events for audit trail

Zero-dependency — uses only project types and Python stdlib.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.aura.patterns.base import DetectedPattern, PatternDomain, PatternStatus

logger = logging.getLogger(__name__)


class AuraRulePromoter:
    """Scans pattern engine output and promotes eligible patterns to bridge rules.

    Args:
        rules_engine: BridgeRulesEngine instance for creating rules
        promotion_log_path: Where to persist promotion audit trail
    """

    def __init__(
        self,
        rules_engine=None,
        promotion_log_path: Optional[Path] = None,
    ):
        self._rules_engine = rules_engine
        self._log_path = promotion_log_path or Path(".aura/promotion_log.jsonl")
        self._log_path.parent.mkdir(parents=True, exist_ok=True)
        self._promoted_ids: set = self._load_promoted_ids()

    def scan_and_promote(
        self,
        patterns: List[DetectedPattern],
    ) -> List[Dict[str, Any]]:
        """Scan patterns and promote eligible ones to bridge rules.

        Args:
            patterns: All active patterns from the pattern engine

        Returns:
            List of promotion events (dict with pattern_id, rule_id, etc.)
        """
        if not self._rules_engine:
            logger.debug("No rules engine configured — skipping promotion scan")
            return []

        promotions: List[Dict[str, Any]] = []

        for pattern in patterns:
            # Skip already-promoted patterns
            if pattern.pattern_id in self._promoted_ids:
                continue

            # Check promotion eligibility
            if not pattern.is_promotable():
                continue

            # Only promote patterns with cross-engine relevance
            if not self._has_bridge_relevance(pattern):
                continue

            # Attempt promotion
            rule = self._rules_engine.create_rule_from_pattern(
                pattern_id=pattern.pattern_id,
                pattern_description=pattern.description,
                pattern_domain=pattern.domain.value,
                pattern_confidence=pattern.confidence,
                observation_count=pattern.observation_count,
                suggested_rule_type=self._suggest_rule_type(pattern),
            )

            if rule:
                # Mark pattern as promoted
                pattern.status = PatternStatus.PROMOTED
                self._promoted_ids.add(pattern.pattern_id)

                event = {
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "pattern_id": pattern.pattern_id,
                    "rule_id": rule.rule_id,
                    "rule_type": rule.rule_type,
                    "direction": rule.direction,
                    "confidence": pattern.confidence,
                    "observation_count": pattern.observation_count,
                    "description": pattern.description[:100],
                }
                promotions.append(event)
                self._log_promotion(event)

                logger.info(
                    f"Promoted pattern '{pattern.pattern_id}' → "
                    f"bridge rule '{rule.rule_id}' ({rule.rule_type})"
                )

        return promotions

    def get_promotion_stats(self) -> Dict[str, Any]:
        """Get summary of promotion activity."""
        return {
            "total_promoted": len(self._promoted_ids),
            "promoted_ids": list(self._promoted_ids)[:20],
        }

    # --- Pattern Analysis ---

    def _has_bridge_relevance(self, pattern: DetectedPattern) -> bool:
        """Check if a pattern has cross-engine bridge relevance.

        Not all patterns should become bridge rules. Only patterns that
        imply a state change relevant to the other engine qualify.
        """
        # Cross-engine patterns always qualify
        if pattern.domain == PatternDomain.CROSS_ENGINE:
            return True

        # Human patterns that affect trading
        if pattern.domain == PatternDomain.HUMAN:
            desc = pattern.description.lower()
            human_bridge_keywords = [
                "emotional", "stress", "override", "impuls",
                "drift", "burnout", "crisis", "fatigue",
                "anxiety", "confidence", "avoidance",
            ]
            return any(kw in desc for kw in human_bridge_keywords)

        # Trading patterns that affect human state
        if pattern.domain == PatternDomain.TRADING:
            desc = pattern.description.lower()
            trading_bridge_keywords = [
                "streak", "drawdown", "consecutive",
                "overperform", "exceptional", "hot streak",
                "max loss", "blowup",
            ]
            return any(kw in desc for kw in trading_bridge_keywords)

        return False

    def _suggest_rule_type(self, pattern: DetectedPattern) -> Optional[str]:
        """Suggest a specific rule type based on pattern characteristics.

        Returns None to let the rules engine infer from description.
        Only returns explicit types when pattern metadata gives clear signal.
        """
        # Check for suggested_rule in pattern metadata
        if pattern.suggested_rule:
            rule_text = pattern.suggested_rule.lower()
            if "confidence" in rule_text:
                return "raise_min_confidence"
            if "risk" in rule_text or "position" in rule_text:
                return "reduce_position_risk"
            if "r:r" in rule_text or "ratio" in rule_text:
                return "tighten_rr_ratio"
            if "pause" in rule_text or "stop" in rule_text:
                return "pause_trading"

        # Check for T3 narrative arcs that map to specific rules
        desc = pattern.description.lower()
        if "emotional drift" in desc and pattern.confidence >= 0.7:
            return "raise_min_confidence"
        if "stress accumulation" in desc:
            return "reduce_position_risk"
        if "override trajectory" in desc and "increasing" in desc:
            return "tighten_rr_ratio"

        # Let the rules engine infer from description
        return None

    # --- Persistence ---

    def _log_promotion(self, event: Dict[str, Any]) -> None:
        """Append a promotion event to the audit log."""
        try:
            with open(self._log_path, "a") as f:
                f.write(json.dumps(event) + "\n")
        except Exception as e:
            logger.warning(f"Failed to log promotion event: {e}")

    def _load_promoted_ids(self) -> set:
        """Load previously promoted pattern IDs from log."""
        ids = set()
        if not self._log_path.exists():
            return ids
        try:
            with open(self._log_path) as f:
                for line in f:
                    line = line.strip()
                    if line:
                        entry = json.loads(line)
                        ids.add(entry.get("pattern_id", ""))
        except Exception:
            pass
        return ids
