"""Trader Readiness Score — Aura's primary output signal to Buddy.

Computes a 0-100 score reflecting the trader's cognitive and emotional
fitness for trading decisions. This score feeds into Buddy's agent team
as the TraderReadinessAgent (#13).

Components (PRD v2.2 §5):
    - Emotional state (detected from conversations)
    - Decision pattern quality (recent override history)
    - Cognitive load (concurrent life stressors)
    - Stress level (conversation sentiment analysis)
    - Confidence trend (rising/falling/stable)

The score modulates Buddy's behavior:
    - 80-100: Full trading capacity
    - 60-79:  Reduced position sizes (-20%)
    - 40-59:  Significantly reduced (-40%), wider SL buffers
    - 20-39:  Minimum positions only
    - 0-19:   Block new trades entirely
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class ReadinessComponents:
    """Individual components that make up the readiness score."""

    emotional_state_score: float = 0.7     # 0-1, higher = calmer/more positive
    cognitive_load_score: float = 0.7      # 0-1, higher = less loaded
    override_discipline_score: float = 0.8 # 0-1, higher = fewer bad overrides
    stress_level_score: float = 0.7        # 0-1, higher = less stressed
    confidence_trend_score: float = 0.7    # 0-1, higher = more stable/rising
    engagement_score: float = 0.5          # 0-1, higher = more recently engaged with Aura

    def to_dict(self) -> Dict[str, float]:
        return {
            "emotional_state": self.emotional_state_score,
            "cognitive_load": self.cognitive_load_score,
            "override_discipline": self.override_discipline_score,
            "stress_level": self.stress_level_score,
            "confidence_trend": self.confidence_trend_score,
            "engagement": self.engagement_score,
        }


@dataclass
class ReadinessSignal:
    """The complete readiness signal sent from Aura to Buddy.

    This is the JSON contract defined in PRD v2.2 §5:
    Human→Domain signal interface.
    """

    readiness_score: float              # 0-100
    cognitive_load: str                 # "low" | "medium" | "high"
    active_stressors: List[str]         # Current life stressors
    override_loss_rate_7d: float        # 0-1, rate of losing overrides last 7 days
    emotional_state: str                # "calm" | "anxious" | "stressed" | "energized" | "fatigued"
    confidence_trend: str               # "rising" | "falling" | "stable"
    components: ReadinessComponents     # Detailed breakdown
    timestamp: str = ""
    conversation_count_7d: int = 0      # How many conversations in last 7 days

    def to_dict(self) -> Dict[str, Any]:
        return {
            "readiness_score": round(self.readiness_score, 1),
            "cognitive_load": self.cognitive_load,
            "active_stressors": self.active_stressors,
            "override_loss_rate_7d": round(self.override_loss_rate_7d, 3),
            "emotional_state": self.emotional_state,
            "confidence_trend": self.confidence_trend,
            "components": self.components.to_dict(),
            "timestamp": self.timestamp,
            "conversation_count_7d": self.conversation_count_7d,
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2, default=str)


# Component weights for the composite score
_COMPONENT_WEIGHTS = {
    "emotional_state": 0.25,
    "cognitive_load": 0.20,
    "override_discipline": 0.25,
    "stress_level": 0.15,
    "confidence_trend": 0.10,
    "engagement": 0.05,
}


class ReadinessComputer:
    """Computes the trader readiness score from self-model data.

    This is the core algorithm that transforms Aura's understanding of the
    user into an actionable signal for Buddy.

    Args:
        signal_path: Where to write the readiness signal JSON for Buddy to read
    """

    def __init__(self, signal_path: Optional[Path] = None):
        self.signal_path = signal_path or Path(".aura/bridge/readiness_signal.json")
        self.signal_path.parent.mkdir(parents=True, exist_ok=True)
        self._override_history: List[Dict[str, Any]] = []

    def compute(
        self,
        emotional_state: str = "calm",
        stress_keywords: List[str] = None,
        active_stressors: List[str] = None,
        recent_override_events: Optional[List[Dict[str, Any]]] = None,
        conversation_count_7d: int = 0,
        confidence_trend: str = "stable",
    ) -> ReadinessSignal:
        """Compute the readiness score from current state.

        Args:
            emotional_state: Current emotional state label
            stress_keywords: Keywords from recent conversations indicating stress
            active_stressors: Known life stressors (career decision, relationship, etc.)
            recent_override_events: Override events from last 7 days with outcomes
            conversation_count_7d: Number of Aura conversations in last 7 days
            confidence_trend: "rising", "falling", or "stable"

        Returns:
            ReadinessSignal with composite score and all components
        """
        stress_keywords = stress_keywords or []
        active_stressors = active_stressors or []
        recent_override_events = recent_override_events or []

        # --- Compute individual components ---

        # Emotional state score
        emotional_scores = {
            "calm": 0.9,
            "energized": 0.85,
            "neutral": 0.7,
            "anxious": 0.4,
            "stressed": 0.3,
            "fatigued": 0.35,
            "frustrated": 0.25,
            "overwhelmed": 0.15,
        }
        emotional_score = emotional_scores.get(emotional_state.lower(), 0.5)

        # Cognitive load score — based on number of active stressors
        stressor_count = len(active_stressors)
        if stressor_count == 0:
            cognitive_load_score = 0.9
            cognitive_load_label = "low"
        elif stressor_count <= 2:
            cognitive_load_score = 0.65
            cognitive_load_label = "medium"
        else:
            cognitive_load_score = 0.3
            cognitive_load_label = "high"

        # Stress keyword penalty
        stress_penalty = min(len(stress_keywords) * 0.05, 0.3)
        cognitive_load_score = max(0.1, cognitive_load_score - stress_penalty)

        # Override discipline score
        if recent_override_events:
            total_overrides = len(recent_override_events)
            losing_overrides = sum(
                1 for e in recent_override_events if not e.get("trade_won", True)
            )
            override_loss_rate = losing_overrides / max(total_overrides, 1)
            override_discipline = max(0.1, 1.0 - override_loss_rate)
        else:
            override_loss_rate = 0.0
            override_discipline = 0.8  # Default — no data, assume decent

        # Confidence trend score
        trend_scores = {"rising": 0.85, "stable": 0.7, "falling": 0.4}
        confidence_trend_score = trend_scores.get(confidence_trend, 0.5)

        # Engagement score — recent engagement with Aura is positive
        if conversation_count_7d >= 5:
            engagement_score = 0.9
        elif conversation_count_7d >= 3:
            engagement_score = 0.7
        elif conversation_count_7d >= 1:
            engagement_score = 0.5
        else:
            engagement_score = 0.2  # No recent conversations = lower readiness

        # Stress level (inverse of stress indicators)
        stress_indicator_count = len(stress_keywords) + stressor_count
        stress_level_score = max(0.1, 1.0 - (stress_indicator_count * 0.08))

        # --- Assemble components ---
        components = ReadinessComponents(
            emotional_state_score=emotional_score,
            cognitive_load_score=cognitive_load_score,
            override_discipline_score=override_discipline,
            stress_level_score=stress_level_score,
            confidence_trend_score=confidence_trend_score,
            engagement_score=engagement_score,
        )

        # --- Compute weighted composite score (0-100) ---
        raw_score = (
            emotional_score * _COMPONENT_WEIGHTS["emotional_state"]
            + cognitive_load_score * _COMPONENT_WEIGHTS["cognitive_load"]
            + override_discipline * _COMPONENT_WEIGHTS["override_discipline"]
            + stress_level_score * _COMPONENT_WEIGHTS["stress_level"]
            + confidence_trend_score * _COMPONENT_WEIGHTS["confidence_trend"]
            + engagement_score * _COMPONENT_WEIGHTS["engagement"]
        )
        readiness_score = max(0.0, min(100.0, raw_score * 100))

        # --- Build signal ---
        signal = ReadinessSignal(
            readiness_score=readiness_score,
            cognitive_load=cognitive_load_label,
            active_stressors=active_stressors,
            override_loss_rate_7d=override_loss_rate,
            emotional_state=emotional_state,
            confidence_trend=confidence_trend,
            components=components,
            timestamp=datetime.now(timezone.utc).isoformat(),
            conversation_count_7d=conversation_count_7d,
        )

        # --- Write signal to bridge file for Buddy to read ---
        self._write_signal(signal)

        logger.info(
            f"Readiness score: {readiness_score:.0f}/100 "
            f"(emotional={emotional_score:.2f}, cognitive={cognitive_load_score:.2f}, "
            f"override={override_discipline:.2f}, stress={stress_level_score:.2f})"
        )

        return signal

    def _write_signal(self, signal: ReadinessSignal) -> None:
        """Write the readiness signal to the bridge file.

        US-202: Uses FeedbackBridge._locked_write for concurrent access safety.
        """
        try:
            from src.aura.bridge.signals import FeedbackBridge
            FeedbackBridge._locked_write(self.signal_path, signal.to_json())
        except ImportError:
            # Fallback to direct write if bridge not available
            try:
                self.signal_path.write_text(signal.to_json())
            except Exception as e:
                logger.error(f"Failed to write readiness signal: {e}")
        except Exception as e:
            logger.error(f"Failed to write readiness signal: {e}")

    def read_latest_signal(self) -> Optional[ReadinessSignal]:
        """Read the latest readiness signal from disk.

        US-202: Uses FeedbackBridge._locked_read for concurrent access safety.
        """
        try:
            from src.aura.bridge.signals import FeedbackBridge
            raw = FeedbackBridge._locked_read(self.signal_path)
        except ImportError:
            if not self.signal_path.exists():
                return None
            raw = self.signal_path.read_text()
        if raw is None:
            return None
        try:
            data = json.loads(raw)
            components = ReadinessComponents(**data.pop("components", {}))
            return ReadinessSignal(components=components, **data)
        except Exception as e:
            logger.warning(f"Failed to read readiness signal: {e}")
            return None
