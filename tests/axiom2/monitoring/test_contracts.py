from datetime import datetime, timedelta, timezone

import pytest
from pydantic import ValidationError

from src.axiom2.monitoring.contracts import (
    CandidateRegistration,
    CandidateState,
    MarketObservation,
    ThresholdRule,
)


NOW = datetime(2026, 10, 6, 18, 0, tzinfo=timezone.utc)


def candidate() -> CandidateRegistration:
    return CandidateRegistration(
        candidate_id="candidate-spy-1",
        instrument_id="SPY",
        thesis_version="thesis-v1",
        thesis_digest="a" * 64,
        registered_at=NOW,
        expires_at=NOW + timedelta(hours=2),
        freshness_seconds=30,
        entry_rules=(
            ThresholdRule(
                metric="PRICE",
                operator="AT_OR_ABOVE",
                threshold=500_000_000,
                role="ENTRY",
            ),
            ThresholdRule(
                metric="VOLUME",
                operator="AT_OR_ABOVE",
                threshold=10_000,
                role="ENTRY",
            ),
        ),
        invalidation_rules=(
            ThresholdRule(
                metric="PRICE",
                operator="AT_OR_BELOW",
                threshold=495_000_000,
                role="INVALIDATION",
            ),
        ),
        exit_rules=(
            ThresholdRule(
                metric="PRICE",
                operator="AT_OR_BELOW",
                threshold=498_000_000,
                role="EXIT",
            ),
        ),
    )


def test_candidate_contract_is_immutable_and_contains_deterministic_rules():
    value = candidate()

    assert value.candidate_id == "candidate-spy-1"
    assert value.entry_rules[0].threshold == 500_000_000
    assert value.state == CandidateState.DISCOVERED

    with pytest.raises((TypeError, ValidationError)):
        value.candidate_id = "changed"


def test_market_observation_rejects_naive_timestamps():
    with pytest.raises(ValidationError):
        MarketObservation(
            candidate_id="candidate-spy-1",
            instrument_id="SPY",
            event_id="lean-event-1",
            source_sequence=1,
            observed_at=datetime(2026, 10, 6, 18, 0),
            received_at=NOW,
            connected=True,
            price_micros=500_000_001,
            volume=12_000,
        )


def test_rules_reject_invalid_metric_role_combinations():
    with pytest.raises(ValidationError):
        ThresholdRule(
            metric="PRICE",
            operator="AT_OR_ABOVE",
            threshold=1,
            role="NOT_A_ROLE",
        )

    with pytest.raises(ValidationError):
        CandidateRegistration(
            **candidate().model_dump(exclude={"state", "schema_version"}),
            entry_rules=(),
        )


def test_candidate_state_enum_includes_monitoring_lifecycle():
    assert tuple(CandidateState) == (
        CandidateState.DISCOVERED,
        CandidateState.WATCHING,
        CandidateState.WAITING,
        CandidateState.TRIGGERED,
        CandidateState.REVALIDATING,
        CandidateState.READY,
        CandidateState.INVALIDATED,
        CandidateState.EXPIRED,
        CandidateState.EXECUTED,
        CandidateState.MONITORING_POSITION,
        CandidateState.EXIT_TRIGGERED,
        CandidateState.CLOSED,
    )
