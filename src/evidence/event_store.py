"""Trusted-receipt ledger validation and deterministic state reconstruction."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Iterable

from .contracts import DispositionEvent, DispositionReceipt, DispositionState
from .signing import TrustStore, verify_disposition_timing, verify_envelope
from .transition_policy import (
    AuthorityRegistry,
    TransitionPolicyError,
    enforce_separation_of_duties,
    enforce_transition,
)


class EventChainError(ValueError):
    """Raised when an event sequence cannot be one valid linear history."""


@dataclass(frozen=True)
class ReconstructedDisposition:
    package_digest: str
    state: DispositionState
    head_event_digest: str
    sequence: int
    updated_at: datetime


def reconstruct_disposition(
    receipts: Iterable[DispositionReceipt],
    *,
    trust_store: TrustStore,
    authorities: AuthorityRegistry,
) -> ReconstructedDisposition:
    received = list(receipts)
    if not received:
        raise EventChainError("disposition ledger is empty")

    expected_package: str | None = None
    prior_event: DispositionEvent | None = None
    prior_digest: str | None = None
    seen_digests: set[str] = set()

    validated_events: list[DispositionEvent] = []
    for offset, receipt in enumerate(received):
        envelope = receipt.envelope
        try:
            payload = verify_envelope(envelope, DispositionEvent, trust_store)
            assert isinstance(payload, DispositionEvent)
            verify_disposition_timing(
                envelope,
                payload,
                received_at=receipt.received_at,
                trust_store=trust_store,
            )
            enforce_transition(payload, authorities, envelope.signature.key_id)
            validated_events.append(payload)
            enforce_separation_of_duties(validated_events)
        except (ValueError, TransitionPolicyError) as exc:
            raise EventChainError(f"invalid event at offset {offset}: {exc}") from exc

        if envelope.payload_digest in seen_digests:
            raise EventChainError(f"duplicate event digest at offset {offset}")
        seen_digests.add(envelope.payload_digest)

        if expected_package is None:
            expected_package = payload.package_digest
        elif payload.package_digest != expected_package:
            raise EventChainError("one ledger cannot contain multiple package digests")

        if payload.sequence != offset:
            raise EventChainError(
                f"non-monotonic sequence at offset {offset}: event sequence={payload.sequence}"
            )
        if payload.previous_event_digest != prior_digest:
            raise EventChainError(f"previous_event_digest mismatch at sequence {payload.sequence}")
        if prior_event is None:
            if payload.from_state is not None:
                raise EventChainError("genesis from_state must be null")
        else:
            if payload.from_state != prior_event.to_state:
                raise EventChainError(f"from_state mismatch at sequence {payload.sequence}")
            if payload.occurred_at < prior_event.occurred_at:
                raise EventChainError(f"occurred_at regressed at sequence {payload.sequence}")

        prior_event = payload
        prior_digest = envelope.payload_digest

    assert expected_package is not None and prior_event is not None and prior_digest is not None
    return ReconstructedDisposition(
        package_digest=expected_package,
        state=prior_event.to_state,
        head_event_digest=prior_digest,
        sequence=prior_event.sequence,
        updated_at=prior_event.occurred_at,
    )
