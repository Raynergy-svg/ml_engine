"""Domain-agnostic recursive learning engine.

Extracted from Buddy's LearningEngine (src/scanner/automation/learning_engine.py).
Generalizes the observe → extract patterns → promote to rules pipeline so both
Buddy and Aura can instantiate it with different data types.

Buddy instantiation:
    data_stream = trade outcomes (trade_journal_rl.json)
    pattern_types = ["sl_too_tight", "high_consensus_wins", "disagreement_predicted_loss", ...]
    promotion_threshold = 3

Aura instantiation:
    data_stream = conversations + life events + override events
    pattern_types = ["emotional_override", "stress_correlation", "decision_avoidance", ...]
    promotion_threshold = 3
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Protocol

from src.scanner.automation.safe_json import safe_json_write

logger = logging.getLogger(__name__)


@dataclass
class LearningEntry:
    """A single extracted insight from an observation."""

    date: str
    category: str
    insight: str
    action: str
    source_id: str = ""
    domain: str = ""  # "market" or "human" or "bridge"
    confidence: float = 0.5
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "date": self.date,
            "category": self.category,
            "insight": self.insight,
            "action": self.action,
            "source_id": self.source_id,
            "domain": self.domain,
            "confidence": self.confidence,
            "metadata": self.metadata,
        }


class PatternExtractor(Protocol):
    """Protocol for domain-specific pattern extraction.

    Each domain (market, human) implements this to extract patterns
    from its specific data types.
    """

    def extract(self, observation: Dict[str, Any]) -> List[LearningEntry]:
        """Extract learning entries from a domain-specific observation."""
        ...


@dataclass
class PromotedRule:
    """A pattern observed enough times to become an active rule."""

    rule_text: str
    source_count: int
    promoted_date: str
    category: str
    domain: str
    source_insights: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "rule_text": self.rule_text,
            "source_count": self.source_count,
            "promoted_date": self.promoted_date,
            "category": self.category,
            "domain": self.domain,
            "source_insights": self.source_insights,
        }


class RecursiveLearner:
    """Domain-agnostic learning engine.

    Implements the core loop:
        observe(data) → extract_patterns(data) → count_occurrences()
        → promote_if_threshold_met() → persist()

    Args:
        domain: Name of the domain ("market", "human", "bridge")
        pattern_types: List of valid pattern category names
        promotion_threshold: How many times a pattern must appear before promotion
        learnings_path: Where to persist extracted learnings
        rules_path: Where to persist promoted rules
        extractors: Dict of category -> PatternExtractor for domain-specific logic
    """

    def __init__(
        self,
        domain: str,
        pattern_types: List[str],
        promotion_threshold: int = 3,
        learnings_path: Optional[Path] = None,
        rules_path: Optional[Path] = None,
        extractors: Optional[Dict[str, PatternExtractor]] = None,
    ):
        self.domain = domain
        self.pattern_types = pattern_types
        self.promotion_threshold = promotion_threshold
        self.learnings_path = learnings_path or Path(f".aura/learnings_{domain}.json")
        self.rules_path = rules_path or Path(f".aura/rules_{domain}.json")
        self.extractors = extractors or {}
        self._learnings: List[LearningEntry] = self._load_learnings()
        self._rules: List[PromotedRule] = self._load_rules()

    def observe(self, observation: Dict[str, Any]) -> List[LearningEntry]:
        """Process a new observation and extract learnings.

        Args:
            observation: Domain-specific observation data.

        Returns:
            List of newly extracted LearningEntry objects.
        """
        new_entries: List[LearningEntry] = []

        for category, extractor in self.extractors.items():
            try:
                entries = extractor.extract(observation)
                for entry in entries:
                    entry.domain = self.domain
                    if not entry.date:
                        entry.date = datetime.now(timezone.utc).strftime("%Y-%m-%d")
                new_entries.extend(entries)
            except Exception as e:
                logger.warning(f"Extractor '{category}' failed: {e}")

        if new_entries:
            self._learnings.extend(new_entries)
            self._persist_learnings()
            logger.info(
                f"[{self.domain}] Extracted {len(new_entries)} learnings from observation"
            )

        return new_entries

    def check_promotions(self) -> List[PromotedRule]:
        """Check if any patterns have been observed enough times for promotion.

        Groups learnings by (category, action) and promotes those exceeding
        the threshold.

        Returns:
            List of newly promoted rules.
        """
        from collections import Counter

        # Count occurrences of each (category, action) pair
        pattern_counts: Dict[str, int] = Counter()
        pattern_insights: Dict[str, List[str]] = {}

        for entry in self._learnings:
            key = f"{entry.category}::{entry.action}"
            pattern_counts[key] += 1
            if key not in pattern_insights:
                pattern_insights[key] = []
            pattern_insights[key].append(entry.insight)

        # Check existing rules to avoid re-promotion
        existing_rules = {r.rule_text for r in self._rules}

        new_rules: List[PromotedRule] = []
        now = datetime.now(timezone.utc).strftime("%Y-%m-%d")

        for key, count in pattern_counts.items():
            if count >= self.promotion_threshold:
                category, action = key.split("::", 1)
                if action not in existing_rules:
                    rule = PromotedRule(
                        rule_text=action,
                        source_count=count,
                        promoted_date=now,
                        category=category,
                        domain=self.domain,
                        source_insights=pattern_insights.get(key, [])[:5],
                    )
                    new_rules.append(rule)
                    self._rules.append(rule)

        if new_rules:
            self._persist_rules()
            logger.info(
                f"[{self.domain}] Promoted {len(new_rules)} patterns to rules"
            )

        return new_rules

    def get_active_rules(self) -> List[PromotedRule]:
        """Return all currently active promoted rules."""
        return list(self._rules)

    def get_learnings(self, category: Optional[str] = None) -> List[LearningEntry]:
        """Return learnings, optionally filtered by category."""
        if category:
            return [e for e in self._learnings if e.category == category]
        return list(self._learnings)

    # --- Persistence ---

    def _load_learnings(self) -> List[LearningEntry]:
        if not self.learnings_path.exists():
            return []
        try:
            data = json.loads(self.learnings_path.read_text())
            return [
                LearningEntry(**entry)
                for entry in data
                if isinstance(entry, dict)
            ]
        except Exception as e:
            logger.warning(f"Failed to load learnings from {self.learnings_path}: {e}")
            return []

    def _load_rules(self) -> List[PromotedRule]:
        if not self.rules_path.exists():
            return []
        try:
            data = json.loads(self.rules_path.read_text())
            return [
                PromotedRule(**rule)
                for rule in data
                if isinstance(rule, dict)
            ]
        except Exception as e:
            logger.warning(f"Failed to load rules from {self.rules_path}: {e}")
            return []

    def _persist_learnings(self) -> None:
        data = [e.to_dict() for e in self._learnings]
        if not safe_json_write(self.learnings_path, data):
            logger.error(f"safe_json_write failed for learnings: {self.learnings_path}")

    def _persist_rules(self) -> None:
        data = [r.to_dict() for r in self._rules]
        if not safe_json_write(self.rules_path, data):
            logger.error(f"safe_json_write failed for rules: {self.rules_path}")
