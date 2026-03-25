"""Override Pattern Extractor — bridges Aura override events into RecursiveLearner.

Implements the PatternExtractor protocol from src/recursive_intelligence/learner.py
for the "bridge" domain. This enables override events to flow through the
domain-agnostic learning pipeline (observe → extract → promote).

PRD v2.2 §7.4: "OVERRIDE_PATTERN type in learning engine"
  - Dual-engine logging: domain logs outcome, human engine logs emotional context
  - Override events from bridge/signals.py → RecursiveLearner → promoted rules

Usage:
    from src.recursive_intelligence.learner import RecursiveLearner
    from src.aura.patterns.override_extractor import OverridePatternExtractor

    learner = RecursiveLearner(
        domain="bridge",
        pattern_types=["override_win", "override_loss", "emotional_override",
                       "cognitive_override", "confidence_mismatch"],
        extractors={"override": OverridePatternExtractor()},
    )
    learner.observe(override_event.to_dict())
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, Dict, List

logger = logging.getLogger(__name__)

# Emotional states that indicate compromised decision-making
NEGATIVE_EMOTIONAL_STATES = frozenset({
    "anxious", "stressed", "frustrated", "angry", "fearful",
    "panic", "revenge", "impulsive", "overwhelmed", "desperate",
})

# Cognitive load levels that indicate degraded judgment
HIGH_COGNITIVE_LOAD = frozenset({
    "high", "overloaded", "exhausted", "overwhelmed", "saturated",
})


class OverridePatternExtractor:
    """Extracts learning entries from override events for the RecursiveLearner.

    Implements the PatternExtractor protocol:
        def extract(self, observation: Dict[str, Any]) -> List[LearningEntry]

    Each override event can yield multiple learning entries capturing different
    facets of the override decision — market context, emotional context, and
    their interaction.
    """

    def extract(self, observation: Dict[str, Any]) -> List["LearningEntry"]:
        """Extract learning entries from an override event observation.

        Args:
            observation: An OverrideEvent.to_dict() payload.

        Returns:
            List of LearningEntry objects for the recursive learner.
        """
        # Lazy import to avoid circular dependency
        from src.recursive_intelligence.learner import LearningEntry

        entries: List[LearningEntry] = []
        now = datetime.now(timezone.utc).strftime("%Y-%m-%d")

        outcome = observation.get("outcome")
        if not outcome:
            return entries  # Can't learn without resolved outcome

        pair = observation.get("pair", "UNKNOWN")
        override_type = observation.get("override_type", "unknown")
        pnl_pips = observation.get("pnl_pips", 0.0)
        emotional_state = (observation.get("emotional_state", "") or "").lower()
        cognitive_load = (observation.get("cognitive_load", "") or "").lower()
        confidence = observation.get("confidence_at_time", 0.0)
        weighted_vote = observation.get("weighted_vote_at_time", 0.0)
        regime = observation.get("regime", "NORMAL")
        ts = observation.get("timestamp", "")

        is_loss = outcome == "loss"
        is_win = outcome == "win"

        # --- Core override outcome ---
        if is_loss:
            entries.append(LearningEntry(
                date=now,
                category="override_loss",
                insight=(
                    f"{override_type} override on {pair} lost {abs(pnl_pips):.1f}p "
                    f"(conf={confidence:.0%}, regime={regime})"
                ),
                action=f"Track {override_type} override loss rate for gating",
                source_id=ts,
                domain="bridge",
                confidence=0.7,
                metadata={
                    "override_type": override_type,
                    "pair": pair,
                    "pnl_pips": pnl_pips,
                    "emotional_state": emotional_state,
                    "cognitive_load": cognitive_load,
                    "buddy_confidence": confidence,
                    "weighted_vote": weighted_vote,
                    "regime": regime,
                },
            ))

        if is_win:
            entries.append(LearningEntry(
                date=now,
                category="override_win",
                insight=(
                    f"{override_type} override on {pair} won {pnl_pips:.1f}p "
                    f"(conf={confidence:.0%}, regime={regime})"
                ),
                action=f"Note: trader intuition correct — review Buddy model for {pair}",
                source_id=ts,
                domain="bridge",
                confidence=0.6,
                metadata={
                    "override_type": override_type,
                    "pair": pair,
                    "pnl_pips": pnl_pips,
                    "buddy_confidence": confidence,
                },
            ))

        # --- Emotional state interaction ---
        is_negative_emotional = any(
            neg in emotional_state for neg in NEGATIVE_EMOTIONAL_STATES
        )
        if is_loss and is_negative_emotional:
            entries.append(LearningEntry(
                date=now,
                category="emotional_override",
                insight=(
                    f"Emotional override ({emotional_state}) on {pair} lost {abs(pnl_pips):.1f}p"
                ),
                action="NEVER override during negative emotional states — emotional overrides lose",
                source_id=ts,
                domain="bridge",
                confidence=0.8,
                metadata={
                    "emotional_state": emotional_state,
                    "override_type": override_type,
                    "pair": pair,
                },
            ))

        # --- Cognitive load interaction ---
        is_high_load = any(term in cognitive_load for term in HIGH_COGNITIVE_LOAD)
        if is_loss and is_high_load:
            entries.append(LearningEntry(
                date=now,
                category="cognitive_override",
                insight=(
                    f"High cognitive load ({cognitive_load}) override on {pair} "
                    f"lost {abs(pnl_pips):.1f}p"
                ),
                action="NEVER override when cognitive load is high — decision quality degrades",
                source_id=ts,
                domain="bridge",
                confidence=0.8,
                metadata={
                    "cognitive_load": cognitive_load,
                    "override_type": override_type,
                    "pair": pair,
                },
            ))

        # --- Confidence mismatch ---
        if is_loss and confidence >= 0.70:
            entries.append(LearningEntry(
                date=now,
                category="confidence_mismatch",
                insight=(
                    f"Override against high-conf signal ({confidence:.0%}) on {pair} "
                    f"lost {abs(pnl_pips):.1f}p"
                ),
                action="NEVER override high-confidence Buddy signals (>=70%) — they're usually right",
                source_id=ts,
                domain="bridge",
                confidence=0.85,
                metadata={
                    "buddy_confidence": confidence,
                    "weighted_vote": weighted_vote,
                    "pair": pair,
                },
            ))

        if is_win and confidence < 0.50:
            entries.append(LearningEntry(
                date=now,
                category="confidence_mismatch",
                insight=(
                    f"Override of low-conf signal ({confidence:.0%}) on {pair} "
                    f"won {pnl_pips:.1f}p — Buddy was weak"
                ),
                action=f"Review Buddy model for {pair} when confidence < 50%",
                source_id=ts,
                domain="bridge",
                confidence=0.6,
                metadata={
                    "buddy_confidence": confidence,
                    "pair": pair,
                },
            ))

        return entries
