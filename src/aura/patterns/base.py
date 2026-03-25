"""Pattern engine base types — shared across all tiers.

Defines the data structures for detected patterns, evidence trails,
and the PatternDetector protocol that each tier implements.

PRD v2.2 §7.1 Pattern Engine:
  T1 (daily, lightweight): frequency, distribution
  T2 (weekly, moderate): cross-domain correlations
  T3 (monthly, deep): narrative arcs, prediction
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Protocol

logger = logging.getLogger(__name__)


class PatternTier(str, Enum):
    T1_DAILY = "t1_daily"       # Frequency-based, lightweight
    T2_WEEKLY = "t2_weekly"     # Cross-domain correlations
    T3_MONTHLY = "t3_monthly"   # Narrative arcs, prediction models


class PatternDomain(str, Enum):
    HUMAN = "human"             # Emotional/cognitive patterns
    TRADING = "trading"         # Market/execution patterns
    CROSS_ENGINE = "cross"      # Patterns spanning both domains


class PatternStatus(str, Enum):
    DETECTED = "detected"       # First observed
    RECURRING = "recurring"     # Seen 2+ times
    PROMOTED = "promoted"       # 3+ observations → active rule
    INVALIDATED = "invalidated" # User-rated as unhelpful or disproven
    ARCHIVED = "archived"       # Superseded or no longer relevant


@dataclass
class EvidenceItem:
    """A single piece of evidence supporting a pattern."""

    source_type: str          # "conversation", "trade", "override", "readiness"
    source_id: str            # ID of the source record
    timestamp: str            # When the evidence occurred
    summary: str              # Human-readable description
    data: Dict[str, Any] = field(default_factory=dict)  # Raw data

    def to_dict(self) -> Dict[str, Any]:
        return {
            "source_type": self.source_type,
            "source_id": self.source_id,
            "timestamp": self.timestamp,
            "summary": self.summary,
            "data": self.data,
        }


@dataclass
class DetectedPattern:
    """A pattern detected by any tier of the pattern engine.

    Patterns accumulate evidence over time. When observation_count >= 3,
    they're eligible for promotion to active rules.
    """

    pattern_id: str                         # Unique identifier
    tier: PatternTier                       # Which tier detected it
    domain: PatternDomain                   # Human, trading, or cross-engine
    description: str                        # Human-readable pattern description
    evidence: List[EvidenceItem] = field(default_factory=list)
    observation_count: int = 1              # How many times observed
    confidence: float = 0.5                 # 0-1, how confident we are
    status: PatternStatus = PatternStatus.DETECTED
    user_rating: Optional[float] = None     # User quality rating (0-5)
    created_at: str = ""
    updated_at: str = ""
    # Statistical backing (for T2 correlations)
    p_value: Optional[float] = None         # Statistical significance
    correlation_strength: Optional[float] = None  # -1 to 1
    sample_size: int = 0
    # Actionable output
    suggested_rule: Optional[str] = None    # What rule to promote if pattern holds
    suggested_config_change: Optional[Dict[str, Any]] = None

    def __post_init__(self):
        now = datetime.now(timezone.utc).isoformat()
        if not self.created_at:
            self.created_at = now
        self.updated_at = now

    def add_evidence(self, evidence: EvidenceItem) -> None:
        """Add a new piece of evidence to this pattern."""
        self.evidence.append(evidence)
        self.observation_count = len(self.evidence)
        self.updated_at = datetime.now(timezone.utc).isoformat()

        # Auto-transition status based on observation count
        if self.observation_count >= 3 and self.status == PatternStatus.RECURRING:
            self.status = PatternStatus.PROMOTED
        elif self.observation_count >= 2 and self.status == PatternStatus.DETECTED:
            self.status = PatternStatus.RECURRING

    def is_promotable(self) -> bool:
        """Check if this pattern is ready for rule promotion."""
        return (
            self.observation_count >= 3
            and self.confidence >= 0.6
            and self.status in (PatternStatus.RECURRING, PatternStatus.PROMOTED)
            and (self.user_rating is None or self.user_rating >= 2.5)
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "pattern_id": self.pattern_id,
            "tier": self.tier.value,
            "domain": self.domain.value,
            "description": self.description,
            "evidence": [e.to_dict() for e in self.evidence],
            "observation_count": self.observation_count,
            "confidence": round(self.confidence, 3),
            "status": self.status.value,
            "user_rating": self.user_rating,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "p_value": self.p_value,
            "correlation_strength": self.correlation_strength,
            "sample_size": self.sample_size,
            "suggested_rule": self.suggested_rule,
            "suggested_config_change": self.suggested_config_change,
        }


class PatternDetector(Protocol):
    """Protocol that each pattern tier must implement."""

    def detect(self, **kwargs) -> List[DetectedPattern]:
        """Run pattern detection and return any new/updated patterns."""
        ...

    def get_active_patterns(self) -> List[DetectedPattern]:
        """Return all currently active (non-archived) patterns."""
        ...
