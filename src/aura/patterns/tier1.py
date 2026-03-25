"""Tier 1 Pattern Engine — Daily frequency-based patterns.

Lightweight pattern detection that runs after every conversation or trade cycle.
Detects:
  - Emotional frequency patterns (e.g., "stressed 4 of last 7 conversations")
  - Override frequency patterns (e.g., "3 overrides in 48 hours")
  - Readiness trend patterns (e.g., "readiness declining 5 consecutive sessions")
  - Stressor recurrence (e.g., "career_decision mentioned in 80% of conversations")

T1 patterns are cheap to compute — they're counting and averaging over
recent data windows. They feed into the T2 cross-domain engine as inputs.
"""

from __future__ import annotations

import json
import logging
import uuid
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.aura.patterns.base import (
    DetectedPattern,
    EvidenceItem,
    PatternDomain,
    PatternStatus,
    PatternTier,
)

logger = logging.getLogger(__name__)

# --- Detection thresholds ---
STRESS_FREQUENCY_THRESHOLD = 0.6   # 60%+ conversations show stress
OVERRIDE_FREQUENCY_THRESHOLD = 3   # 3+ overrides in window
READINESS_DECLINE_STREAK = 3       # 3+ consecutive declines
STRESSOR_RECURRENCE_THRESHOLD = 0.5  # Same stressor in 50%+ of conversations


class Tier1FrequencyDetector:
    """Daily frequency-based pattern detection.

    Scans recent conversation history, readiness history, and override logs
    to detect recurring patterns based on simple frequency counts.

    Args:
        patterns_dir: Where to persist detected patterns
    """

    def __init__(self, patterns_dir: Optional[Path] = None):
        self.patterns_dir = patterns_dir or Path(".aura/patterns")
        self.patterns_dir.mkdir(parents=True, exist_ok=True)
        self._patterns_file = self.patterns_dir / "t1_patterns.json"
        self._active_patterns: Dict[str, DetectedPattern] = {}
        self._load_patterns()

    def _load_patterns(self) -> None:
        """Load persisted patterns from disk."""
        if not self._patterns_file.exists():
            return
        try:
            data = json.loads(self._patterns_file.read_text())
            for pid, pdata in data.items():
                evidence = [EvidenceItem(**e) for e in pdata.pop("evidence", [])]
                pdata["tier"] = PatternTier(pdata["tier"])
                pdata["domain"] = PatternDomain(pdata["domain"])
                pdata["status"] = PatternStatus(pdata["status"])
                self._active_patterns[pid] = DetectedPattern(
                    evidence=evidence, **pdata
                )
        except Exception as e:
            logger.warning(f"T1: Failed to load patterns: {e}")

    def _save_patterns(self) -> None:
        """Persist patterns to disk."""
        try:
            data = {
                pid: p.to_dict()
                for pid, p in self._active_patterns.items()
                if p.status not in (PatternStatus.ARCHIVED, PatternStatus.INVALIDATED)
            }
            self._patterns_file.write_text(json.dumps(data, indent=2, default=str))
        except Exception as e:
            logger.error(f"T1: Failed to save patterns: {e}")

    def detect(
        self,
        conversations: List[Dict[str, Any]],
        readiness_history: List[Dict[str, Any]],
        override_events: List[Dict[str, Any]],
    ) -> List[DetectedPattern]:
        """Run all T1 frequency detections.

        Args:
            conversations: Recent conversation records from self-model DB
            readiness_history: Recent readiness score entries
            override_events: Recent override events from bridge

        Returns:
            List of newly detected or updated patterns
        """
        new_patterns: List[DetectedPattern] = []

        new_patterns.extend(self._detect_emotional_frequency(conversations))
        new_patterns.extend(self._detect_stressor_recurrence(conversations))
        new_patterns.extend(self._detect_override_frequency(override_events))
        new_patterns.extend(self._detect_readiness_trends(readiness_history))

        self._save_patterns()

        if new_patterns:
            logger.info(f"T1: Detected {len(new_patterns)} patterns")

        return new_patterns

    def _detect_emotional_frequency(
        self, conversations: List[Dict[str, Any]]
    ) -> List[DetectedPattern]:
        """Detect if negative emotional states dominate recent conversations."""
        if len(conversations) < 3:
            return []

        negative_states = {"stressed", "anxious", "fatigued", "frustrated", "overwhelmed"}
        recent = conversations[:7]  # Last 7 conversations
        negative_count = sum(
            1 for c in recent
            if c.get("emotional_state", "").lower() in negative_states
        )
        ratio = negative_count / len(recent)

        if ratio < STRESS_FREQUENCY_THRESHOLD:
            return []

        pattern_key = "emotional_frequency_negative"
        dominant_emotion = Counter(
            c.get("emotional_state", "neutral") for c in recent
            if c.get("emotional_state", "").lower() in negative_states
        ).most_common(1)

        dominant = dominant_emotion[0][0] if dominant_emotion else "stressed"
        description = (
            f"Negative emotional state ({dominant}) in {negative_count}/{len(recent)} "
            f"recent conversations ({ratio:.0%})"
        )

        return self._upsert_pattern(
            pattern_key=pattern_key,
            domain=PatternDomain.HUMAN,
            description=description,
            evidence=EvidenceItem(
                source_type="conversation_aggregate",
                source_id=f"last_{len(recent)}_conversations",
                timestamp=datetime.now(timezone.utc).isoformat(),
                summary=f"{negative_count}/{len(recent)} conversations negative ({dominant})",
                data={
                    "negative_count": negative_count,
                    "total": len(recent),
                    "ratio": round(ratio, 3),
                    "dominant_emotion": dominant,
                },
            ),
            confidence=min(0.9, 0.5 + ratio * 0.4),
            suggested_rule=(
                f"When emotional frequency of '{dominant}' exceeds {STRESS_FREQUENCY_THRESHOLD:.0%} "
                f"in recent conversations, reduce readiness score by 15%"
            ),
        )

    def _detect_stressor_recurrence(
        self, conversations: List[Dict[str, Any]]
    ) -> List[DetectedPattern]:
        """Detect if specific stressors keep coming up across conversations."""
        if len(conversations) < 4:
            return []

        recent = conversations[:10]
        stressor_counts: Counter = Counter()

        for conv in recent:
            topics_raw = conv.get("topics", "[]")
            if isinstance(topics_raw, str):
                try:
                    topics = json.loads(topics_raw)
                except (json.JSONDecodeError, TypeError):
                    topics = []
            else:
                topics = topics_raw

            for topic in topics:
                stressor_counts[topic] += 1

        patterns = []
        for stressor, count in stressor_counts.items():
            ratio = count / len(recent)
            if ratio >= STRESSOR_RECURRENCE_THRESHOLD:
                pattern_key = f"stressor_recurrence_{stressor}"
                description = (
                    f"'{stressor}' appears in {count}/{len(recent)} recent "
                    f"conversations ({ratio:.0%}) — persistent stressor"
                )
                result = self._upsert_pattern(
                    pattern_key=pattern_key,
                    domain=PatternDomain.HUMAN,
                    description=description,
                    evidence=EvidenceItem(
                        source_type="stressor_frequency",
                        source_id=stressor,
                        timestamp=datetime.now(timezone.utc).isoformat(),
                        summary=f"'{stressor}' in {count}/{len(recent)} conversations",
                        data={"stressor": stressor, "count": count, "total": len(recent)},
                    ),
                    confidence=min(0.85, 0.5 + ratio * 0.35),
                )
                patterns.extend(result)

        return patterns

    def _detect_override_frequency(
        self, override_events: List[Dict[str, Any]]
    ) -> List[DetectedPattern]:
        """Detect clustering of override events."""
        if len(override_events) < OVERRIDE_FREQUENCY_THRESHOLD:
            return []

        recent = override_events[-10:]  # Last 10 overrides
        losing = [e for e in recent if e.get("outcome") == "loss"]
        losing_rate = len(losing) / max(len(recent), 1)

        if len(recent) < OVERRIDE_FREQUENCY_THRESHOLD:
            return []

        pattern_key = "override_frequency_high"
        description = (
            f"{len(recent)} override events recently, "
            f"{len(losing)} losses ({losing_rate:.0%} loss rate)"
        )

        return self._upsert_pattern(
            pattern_key=pattern_key,
            domain=PatternDomain.TRADING,
            description=description,
            evidence=EvidenceItem(
                source_type="override_aggregate",
                source_id=f"last_{len(recent)}_overrides",
                timestamp=datetime.now(timezone.utc).isoformat(),
                summary=f"{len(recent)} overrides, {losing_rate:.0%} loss rate",
                data={
                    "total_overrides": len(recent),
                    "losing_overrides": len(losing),
                    "loss_rate": round(losing_rate, 3),
                },
            ),
            confidence=min(0.9, 0.5 + losing_rate * 0.4),
            suggested_rule=(
                f"Override loss rate at {losing_rate:.0%} — increase consensus "
                f"threshold when readiness < 60"
            ),
        )

    def _detect_readiness_trends(
        self, readiness_history: List[Dict[str, Any]]
    ) -> List[DetectedPattern]:
        """Detect sustained readiness score decline."""
        if len(readiness_history) < READINESS_DECLINE_STREAK:
            return []

        # readiness_history is newest-first from the DB query
        recent = readiness_history[:10]
        scores = [r.get("score", 70) for r in recent]

        # Check for sustained decline (each score lower than previous)
        decline_streak = 0
        for i in range(len(scores) - 1):
            if scores[i] < scores[i + 1]:  # newer < older = decline
                decline_streak += 1
            else:
                break

        if decline_streak < READINESS_DECLINE_STREAK:
            return []

        drop_amount = scores[-1] - scores[0]  # Positive = decline (newest is lower)
        pattern_key = "readiness_declining_streak"
        description = (
            f"Readiness score declining for {decline_streak} consecutive sessions "
            f"(from {scores[-1]:.0f} to {scores[0]:.0f}, drop of {abs(drop_amount):.0f} points)"
        )

        return self._upsert_pattern(
            pattern_key=pattern_key,
            domain=PatternDomain.HUMAN,
            description=description,
            evidence=EvidenceItem(
                source_type="readiness_trend",
                source_id=f"streak_{decline_streak}",
                timestamp=datetime.now(timezone.utc).isoformat(),
                summary=f"{decline_streak}-session decline from {scores[-1]:.0f} to {scores[0]:.0f}",
                data={
                    "streak_length": decline_streak,
                    "start_score": scores[-1],
                    "end_score": scores[0],
                    "scores": [round(s, 1) for s in scores],
                },
            ),
            confidence=min(0.9, 0.5 + decline_streak * 0.1),
            suggested_rule=(
                f"Readiness declining {decline_streak} sessions — "
                f"trigger proactive check-in with user"
            ),
        )

    def _upsert_pattern(
        self,
        pattern_key: str,
        domain: PatternDomain,
        description: str,
        evidence: EvidenceItem,
        confidence: float = 0.5,
        suggested_rule: Optional[str] = None,
    ) -> List[DetectedPattern]:
        """Create or update a pattern. Returns list with the pattern if new/updated."""
        existing = self._active_patterns.get(pattern_key)

        if existing and existing.status not in (
            PatternStatus.ARCHIVED,
            PatternStatus.INVALIDATED,
        ):
            existing.add_evidence(evidence)
            existing.confidence = confidence
            existing.description = description
            if suggested_rule:
                existing.suggested_rule = suggested_rule
            logger.debug(
                f"T1: Updated pattern '{pattern_key}' "
                f"(obs={existing.observation_count}, conf={confidence:.2f})"
            )
            return [existing]

        pattern = DetectedPattern(
            pattern_id=pattern_key,
            tier=PatternTier.T1_DAILY,
            domain=domain,
            description=description,
            evidence=[evidence],
            observation_count=1,
            confidence=confidence,
            suggested_rule=suggested_rule,
        )
        self._active_patterns[pattern_key] = pattern
        logger.info(f"T1: New pattern detected: '{pattern_key}'")
        return [pattern]

    def get_active_patterns(self) -> List[DetectedPattern]:
        """Return all non-archived T1 patterns."""
        return [
            p for p in self._active_patterns.values()
            if p.status not in (PatternStatus.ARCHIVED, PatternStatus.INVALIDATED)
        ]

    def get_promotable_patterns(self) -> List[DetectedPattern]:
        """Return patterns ready for rule promotion."""
        return [p for p in self._active_patterns.values() if p.is_promotable()]
