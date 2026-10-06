"""Narrow adapter from LEAN live events to Axiom observations.

LEAN remains the owner of subscriptions, consolidation, scheduling, live
synchronization, and reconnect signaling. This module only validates the
provider envelope and maps it to Axiom's deterministic monitor contract.
There is deliberately no broker, portfolio, approval, or execution import.
"""

from __future__ import annotations

from collections.abc import Mapping
from datetime import datetime
from typing import Callable, Literal, Protocol

from pydantic import Field

from src.evidence.contracts import StrictContract

from .contracts import CandidateState, MarketObservation
from .monitor import AxiomMonitor


class LeanMarketEvent(StrictContract):
    """The minimal event envelope expected from a pinned LEAN sidecar."""

    source: Literal["LEAN"]
    event_id: str = Field(min_length=1)
    candidate_id: str = Field(min_length=1)
    instrument_id: str = Field(min_length=1)
    source_sequence: int = Field(ge=0)
    observed_at: datetime
    received_at: datetime
    connected: bool
    price_micros: int | None = Field(default=None, gt=0)
    volume: int | None = Field(default=None, ge=0)


class LeanEventSource(Protocol):
    """Small subscription shape implemented by the external LEAN runtime."""

    def subscribe_market_events(
        self,
        callback: Callable[[LeanMarketEvent], CandidateState],
    ) -> None:
        """Deliver live/consolidated LEAN events to the adapter callback."""


class LeanObservationAdapter:
    """Normalize LEAN data; never decide or submit an order."""

    def __init__(self, monitor: AxiomMonitor):
        self.monitor = monitor

    def attach(self, source: LeanEventSource) -> None:
        """Attach to an already-configured LEAN subscription boundary."""
        source.subscribe_market_events(self.on_market_event)

    def on_market_event(
        self,
        event: LeanMarketEvent | Mapping[str, object],
    ) -> CandidateState:
        """Convert one LEAN event into exactly one Axiom observation."""
        envelope = (
            event
            if isinstance(event, LeanMarketEvent)
            else LeanMarketEvent.model_validate(event)
        )
        observation = MarketObservation(
            candidate_id=envelope.candidate_id,
            instrument_id=envelope.instrument_id,
            event_id=envelope.event_id,
            source_sequence=envelope.source_sequence,
            observed_at=envelope.observed_at,
            received_at=envelope.received_at,
            connected=envelope.connected,
            price_micros=envelope.price_micros,
            volume=envelope.volume,
        )
        return self.monitor.observe(observation)


__all__ = [
    "LeanEventSource",
    "LeanMarketEvent",
    "LeanObservationAdapter",
]
