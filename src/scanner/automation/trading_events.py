"""Strongly-typed trading event definitions for the event bus and queue system."""

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional


class TradingEventType(Enum):
    """All event types flowing through the trading event bus."""
    TRADE_OPENED = "trade_opened"
    TRADE_CLOSED = "trade_closed"
    TRADE_MODIFIED = "trade_modified"
    POLICY_BLOCKED = "policy_blocked"
    RETRAIN_REQUESTED = "retrain_requested"
    DEGRADED_MODE_ENTERED = "degraded_mode_entered"
    DEGRADED_MODE_EXITED = "degraded_mode_exited"
    QUEUE_FAILURE = "queue_failure"
    # Training lifecycle — closes the autonomy loop with tier-7 observability.
    TRAINING_PROPOSED = "training_proposed"       # retrain_agent → queue
    TRAINING_STARTED = "training_started"          # worker begins run_train_joint
    TRAINING_COMPLETED = "training_completed"      # validation + ensemble checks passed
    MODEL_PROMOTED = "model_promoted"              # :production alias moved
    MODEL_HELD = "model_held"                      # staged, policy said no
    MODEL_REGRESSION_DETECTED = "model_regression_detected"  # live-fire post-promo rollback trigger


@dataclass
class TradingEvent:
    """Base event envelope for all trading events."""
    event_id: str
    event_type: TradingEventType
    timestamp: str
    source: str
    session_id: str
    correlation_id: str
    payload: dict
    payload_version: int = 1


@dataclass
class TradeClosedPayload:
    """Payload for TRADE_CLOSED events."""
    trade_id: str
    pair: str
    direction: str
    entry_price: float
    exit_price: float
    realized_pl: float
    pnl_pips: float
    trade_won: bool
    exit_reason: str
    duration_minutes: float
    confidence: float
    regime: str
    agent_reasons: Dict[str, Any]
    model: str
    analysis_context: Dict[str, Any]
    close_time: str
    sl_pips: float
    tp_pips: float
    weighted_vote_score: Optional[float] = None
    uncertainty_score: Optional[float] = None
    model_disagreement: Optional[float] = None


@dataclass
class TradeOpenedPayload:
    """Payload for TRADE_OPENED events."""
    trade_id: str
    pair: str
    direction: str
    entry_price: float
    sl_pips: float
    tp_pips: float
    lots: float
    confidence: float
    regime: str
    model: str


@dataclass
class PolicyBlockedPayload:
    """Payload for POLICY_BLOCKED events."""
    action_type: str
    decision: str
    reasons: List[str]
    matched_rules: List[str]
    environment_snapshot: Dict[str, Any]


@dataclass
class TrainingCompletedPayload:
    """Payload for TRAINING_COMPLETED events.

    Published by scheduled_retrain.py (or retrain_agent) when:
      * run_train_joint() returned success, AND
      * validate_holdout_multi_timeframe passed on >= 1 timeframe, AND
      * verify_ensemble_refreshed confirmed all artifacts post-retrain.

    Consumed by the promotion handler, which logs the artifact to W&B,
    consults promotion_policy.should_promote, and emits MODEL_PROMOTED
    or MODEL_HELD. The round-trip is the closed autonomy loop.
    """
    pairs: List[str]
    trained_at: str
    model_dir: str
    accuracy: Optional[float]
    holdout_samples: Optional[int]
    timeframes_passed: Optional[int]
    timeframes_tested: Optional[int]
    max_component_age_days: Optional[float]
    ensemble_complete: Optional[bool]
    ensemble_stale_groups: List[str] = field(default_factory=list)
    per_timeframe: Dict[str, Any] = field(default_factory=dict)
    reason: str = "scheduled"  # "scheduled" | "drift" | "losing_streak" | "stale_models"


@dataclass
class ModelPromotedPayload:
    """Payload for MODEL_PROMOTED (or MODEL_HELD) events."""
    artifact_name: str
    artifact_version: str
    promoted: bool
    reason: str
    decision_diagnostics: Dict[str, Any]
    candidate_metrics: Dict[str, Any]
    production_metrics: Optional[Dict[str, Any]] = None


def create_trade_closed_event(outcome_data: dict, session_id: str) -> TradingEvent:
    """Factory: build a TradingEvent from raw trade outcome data.

    Args:
        outcome_data: Dict with trade outcome fields (trade_id, pair, etc.)
        session_id: Current session identifier.

    Returns:
        A fully populated TradingEvent with TRADE_CLOSED type.
    """
    event_id = str(uuid.uuid4())
    now = datetime.now(timezone.utc).isoformat()
    correlation_id = outcome_data.get("correlation_id", str(uuid.uuid4()))

    def _f(v, default=0.0):
        """Safely convert to float, handling None and non-numeric values."""
        if v is None:
            return default
        try:
            return float(v)
        except (TypeError, ValueError):
            return default

    payload = TradeClosedPayload(
        trade_id=outcome_data.get("trade_id", "") or "",
        pair=outcome_data.get("pair", "") or "",
        direction=outcome_data.get("direction", "") or "",
        entry_price=_f(outcome_data.get("entry_price")),
        exit_price=_f(outcome_data.get("exit_price")),
        realized_pl=_f(outcome_data.get("realized_pl")),
        pnl_pips=_f(outcome_data.get("pnl_pips")),
        trade_won=bool(outcome_data.get("trade_won", False)),
        exit_reason=outcome_data.get("exit_reason", "") or "",
        duration_minutes=_f(outcome_data.get("duration_minutes")),
        confidence=_f(outcome_data.get("confidence")),
        regime=outcome_data.get("regime", "") or "",
        agent_reasons=outcome_data.get("agent_reasons") or {},
        model=outcome_data.get("model", "") or "",
        analysis_context=outcome_data.get("analysis_context") or {},
        close_time=outcome_data.get("close_time") or now,
        sl_pips=_f(outcome_data.get("sl_pips")),
        tp_pips=_f(outcome_data.get("tp_pips")),
        weighted_vote_score=outcome_data.get("weighted_vote_score"),
        uncertainty_score=outcome_data.get("uncertainty_score"),
        model_disagreement=outcome_data.get("model_disagreement"),
    )

    return TradingEvent(
        event_id=event_id,
        event_type=TradingEventType.TRADE_CLOSED,
        timestamp=now,
        source="execution_manager",
        session_id=session_id,
        correlation_id=correlation_id,
        payload=payload.__dict__,
    )


def validate_event(event: TradingEvent) -> bool:
    """Validate that a TradingEvent has all required fields non-empty.

    Args:
        event: The TradingEvent to validate.

    Returns:
        True if all required fields are present and non-empty, False otherwise.
    """
    if not event.event_id:
        return False
    if not isinstance(event.event_type, TradingEventType):
        return False
    if not event.timestamp:
        return False
    if not event.source:
        return False
    if not event.session_id:
        return False
    if not event.correlation_id:
        return False
    if not isinstance(event.payload, dict):
        return False
    if event.payload_version < 1:
        return False
    return True
