"""Conversation processor — extracts emotional signals from user messages.

Phase 1: Keyword-based sentiment and stress detection.
Phase 2: Will use Phi-4 14B via MLX for deep understanding.

This module is the "ear" of Aura — it listens to what the user says and
extracts structured signals that feed into the readiness computation.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Set

logger = logging.getLogger(__name__)


# --- Keyword Dictionaries ---
# These are Phase 1 heuristics. Phase 2 replaces with LLM inference.

STRESS_KEYWORDS: Set[str] = {
    "stressed", "stress", "overwhelmed", "exhausted", "tired", "anxious",
    "worried", "frustrated", "angry", "can't sleep", "insomnia", "burnout",
    "deadline", "pressure", "overworked", "losing money", "lost money",
    "argument", "fight", "conflict", "breakup", "divorce",
}

POSITIVE_KEYWORDS: Set[str] = {
    "great", "amazing", "wonderful", "happy", "excited", "energized",
    "productive", "focused", "calm", "relaxed", "confident", "winning",
    "breakthrough", "inspired", "motivated", "grateful", "optimistic",
}

FATIGUE_KEYWORDS: Set[str] = {
    "tired", "exhausted", "didn't sleep", "couldn't sleep", "insomnia",
    "drained", "low energy", "fatigued", "burnt out", "running on empty",
}

TRADING_OVERRIDE_KEYWORDS: Set[str] = {
    "override", "ignored buddy", "took the trade anyway", "closed early",
    "moved my stop", "changed the sl", "changed the tp", "didn't listen",
    "went against", "manual trade", "gut feeling", "just felt like",
}

STRESSOR_KEYWORDS: Dict[str, str] = {
    "career": "career decision",
    "job": "career change",
    "quit": "career change",
    "promotion": "career decision",
    "interview": "job search",
    "relationship": "relationship stress",
    "partner": "relationship stress",
    "health": "health concern",
    "sick": "health concern",
    "money": "financial stress",
    "debt": "financial stress",
    "moving": "relocation",
    "baby": "new parent",
    "pregnant": "expecting child",
    "parent": "family responsibility",
}


@dataclass
class ConversationSignals:
    """Extracted signals from a conversation exchange."""

    emotional_state: str = "neutral"  # calm, anxious, stressed, energized, fatigued, etc.
    stress_keywords_found: List[str] = field(default_factory=list)
    positive_keywords_found: List[str] = field(default_factory=list)
    detected_stressors: List[str] = field(default_factory=list)
    fatigue_detected: bool = False
    override_mentioned: bool = False
    sentiment_score: float = 0.5  # 0=very negative, 1=very positive
    topics: List[str] = field(default_factory=list)
    confidence_trend: str = "stable"  # rising, falling, stable
    message_count: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "emotional_state": self.emotional_state,
            "stress_keywords": self.stress_keywords_found,
            "positive_keywords": self.positive_keywords_found,
            "detected_stressors": self.detected_stressors,
            "fatigue_detected": self.fatigue_detected,
            "override_mentioned": self.override_mentioned,
            "sentiment_score": round(self.sentiment_score, 3),
            "topics": self.topics,
            "confidence_trend": self.confidence_trend,
        }


class ConversationProcessor:
    """Processes conversation messages and extracts emotional/cognitive signals.

    Phase 1: keyword-based analysis.
    Phase 2: Phi-4 14B inference via MLX for deep understanding.
    """

    def __init__(self):
        self._session_messages: List[Dict[str, str]] = []
        self._cumulative_stress: float = 0.0
        self._cumulative_positive: float = 0.0
        self._previous_sentiment: float = 0.5

    def process_message(self, message: str, role: str = "user") -> ConversationSignals:
        """Process a single message and extract signals.

        Args:
            message: The message text
            role: "user" or "assistant"

        Returns:
            ConversationSignals with extracted emotional data
        """
        self._session_messages.append({"role": role, "content": message, "timestamp": datetime.now(timezone.utc).isoformat()})

        if role != "user":
            # Only analyze user messages for emotional signals
            return ConversationSignals(message_count=len(self._session_messages))

        message_lower = message.lower()

        # --- Keyword extraction ---
        stress_found = [kw for kw in STRESS_KEYWORDS if kw in message_lower]
        positive_found = [kw for kw in POSITIVE_KEYWORDS if kw in message_lower]
        fatigue_found = any(kw in message_lower for kw in FATIGUE_KEYWORDS)
        override_found = any(kw in message_lower for kw in TRADING_OVERRIDE_KEYWORDS)

        # Detect stressors — US-204: use word-boundary regex to prevent
        # false positives (e.g. "parent" matching inside "apartment")
        stressors = []
        for keyword, stressor in STRESSOR_KEYWORDS.items():
            if re.search(r'\b' + re.escape(keyword) + r'\b', message_lower) and stressor not in stressors:
                stressors.append(stressor)

        # --- Sentiment computation ---
        stress_score = len(stress_found) * 0.15
        positive_score = len(positive_found) * 0.12
        fatigue_penalty = 0.15 if fatigue_found else 0.0

        raw_sentiment = 0.5 + positive_score - stress_score - fatigue_penalty
        sentiment = max(0.0, min(1.0, raw_sentiment))

        # Update cumulative trackers
        self._cumulative_stress += stress_score
        self._cumulative_positive += positive_score

        # --- Determine emotional state ---
        if fatigue_found:
            emotional_state = "fatigued"
        elif stress_score > 0.3:
            emotional_state = "stressed"
        elif stress_score > 0.15:
            emotional_state = "anxious"
        elif positive_score > 0.3:
            emotional_state = "energized"
        elif positive_score > 0.15:
            emotional_state = "calm"
        else:
            emotional_state = "neutral"

        # --- Confidence trend ---
        if sentiment > self._previous_sentiment + 0.1:
            confidence_trend = "rising"
        elif sentiment < self._previous_sentiment - 0.1:
            confidence_trend = "falling"
        else:
            confidence_trend = "stable"
        self._previous_sentiment = sentiment

        # --- Build topics list ---
        topics = []
        if any(kw in message_lower for kw in ["trade", "trading", "buddy", "market", "forex", "fx"]):
            topics.append("trading")
        if any(kw in message_lower for kw in ["career", "job", "work", "promotion"]):
            topics.append("career")
        if any(kw in message_lower for kw in ["relationship", "partner", "family"]):
            topics.append("relationships")
        if any(kw in message_lower for kw in ["health", "sleep", "exercise", "sick"]):
            topics.append("health")
        if override_found:
            topics.append("trading_override")

        signals = ConversationSignals(
            emotional_state=emotional_state,
            stress_keywords_found=stress_found,
            positive_keywords_found=positive_found,
            detected_stressors=stressors,
            fatigue_detected=fatigue_found,
            override_mentioned=override_found,
            sentiment_score=sentiment,
            topics=topics,
            confidence_trend=confidence_trend,
            message_count=len(self._session_messages),
        )

        logger.debug(
            f"Conversation signals: state={emotional_state}, "
            f"sentiment={sentiment:.2f}, stressors={stressors}"
        )

        return signals

    def get_session_summary(self) -> Dict[str, Any]:
        """Get a summary of the current conversation session."""
        return {
            "message_count": len(self._session_messages),
            "cumulative_stress": round(self._cumulative_stress, 3),
            "cumulative_positive": round(self._cumulative_positive, 3),
            "net_sentiment": round(self._cumulative_positive - self._cumulative_stress, 3),
        }

    def reset_session(self) -> None:
        """Reset session state for a new conversation."""
        self._session_messages = []
        self._cumulative_stress = 0.0
        self._cumulative_positive = 0.0
        self._previous_sentiment = 0.5
