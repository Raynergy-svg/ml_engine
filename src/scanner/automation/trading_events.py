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

    payload = TradeClosedPayload(
        trade_id=outcome_data.get("trade_id", ""),
        pair=outcome_data.get("pair", ""),
        direction=outcome_data.get("direction", ""),
        entry_price=float(outcome_data.get("entry_price", 0.0)),
        exit_price=float(outcome_data.get("exit_price", 0.0)),
        realized_pl=float(outcome_data.get("realized_pl", 0.0)),
        pnl_pips=float(outcome_data.get("pnl_pips", 0.0)),
        trade_won=bool(outcome_data.get("trade_won", False)),
        exit_reason=outcome_data.get("exit_reason", ""),
        duration_minutes=float(outcome_data.get("duration_minutes", 0.0)),
        confidence=float(outcome_data.get("confidence", 0.0)),
        regime=outcome_data.get("regime", ""),
        agent_reasons=outcome_data.get("agent_reasons", {}),
        model=outcome_data.get("model", ""),
        analysis_context=outcome_data.get("analysis_context", {}),
        close_time=outcome_data.get("close_time", now),
        sl_pips=float(outcome_data.get("sl_pips", 0.0)),
        tp_pips=float(outcome_data.get("tp_pips", 0.0)),
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
