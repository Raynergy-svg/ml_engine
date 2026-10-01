"""Signer-present negatives cannot pass merely because signing is absent."""
from datetime import datetime, timedelta, timezone

import pytest

from src.axiom2.promotion.authority import (
    PromotionDecisionPayload, PromotionPackage, evaluate_promotion,
)
from src.evidence.signing import Ed25519Signer, TrustStore, verify_envelope

NOW = datetime(2026, 10, 1, 14, tzinfo=timezone.utc)


def package(**changes):
    values = dict(
        candidate_id="c" * 64, candidate_artifact_hash="a" * 64,
        frozen_artifact_hash="a" * 64, baseline_net_return=0.001,
        candidate_net_return=0.005, cost_evidence_id="cost-v1",
        experiment_count=4, registered_experiment_count=4,
        holdout_id="synthetic-only", holdout_status="CONSUMED",
        holdout_result_hash="b" * 64, holdout_candidate_id="c" * 64,
        min_excess_return=0.001, holdout_net_return=0.004,
        holdout_consumption_count=1,
    )
    values.update(changes)
    return PromotionPackage(**values)


@pytest.fixture
def authority():
    signer = Ed25519Signer.generate()
    trust = TrustStore()
    trust.add(signer.trusted_key(valid_from=NOW - timedelta(days=1)))
    return signer, trust


def signed_decision(authority, **changes):
    signer, trust = authority
    result = evaluate_promotion(package(**changes), signer=signer, created_at=NOW)
    assert result.envelope is not None
    payload = verify_envelope(result.envelope, PromotionDecisionPayload, trust)
    assert (payload.status, payload.decision_hash, payload.reasons) == (
        result.status, result.decision_hash, result.reasons,
    )
    assert "promotion decision requires signing authority" not in result.reasons
    return result


def test_signed_positive_control_and_repeatability(authority):
    first = signed_decision(authority)
    second = signed_decision(authority)
    assert first.status == "REJECTED"
    assert first.reasons == ("unverified scalar promotion package",)
    assert first.verified is False
    assert first.decision_hash == second.decision_hash


@pytest.mark.parametrize("field", [
    "baseline_net_return", "candidate_net_return",
    "holdout_net_return", "min_excess_return",
])
@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_every_nonfinite_metric_is_rejected_with_signer(authority, field, value):
    decision = signed_decision(authority, **{field: value})
    assert decision.status == "REJECTED"
    assert f"non-finite {field}" in decision.reasons


@pytest.mark.parametrize("changes,reason", [
    ({"baseline_net_return": None}, "missing baseline"),
    ({"cost_evidence_id": ""}, "missing costs"),
    ({"experiment_count": 3}, "experiment count mismatch"),
    ({"frozen_artifact_hash": "d" * 64}, "candidate mutated after freeze"),
    ({"holdout_consumption_count": 2}, "holdout not single-use consumed"),
    ({"holdout_status": "OPENED"}, "holdout not single-use consumed"),
    ({"holdout_candidate_id": "e" * 64}, "holdout candidate mismatch"),
    ({"candidate_artifact_hash": "bad"}, "invalid candidate_artifact_hash"),
    ({"holdout_result_hash": "bad"}, "invalid holdout_result_hash"),
    ({"candidate_net_return": 0.0015}, "development promotion rule failed"),
    ({"holdout_net_return": 0.0015}, "holdout promotion rule failed"),
])
def test_existing_rejections_are_exercised_with_signer(authority, changes, reason):
    result = signed_decision(authority, **changes)
    assert result.status == "REJECTED"
    assert reason in result.reasons


@pytest.mark.parametrize("value", [True, 4.0])
def test_attempt_counts_require_real_integers(authority, value):
    result = signed_decision(
        authority, experiment_count=value, registered_experiment_count=value,
    )
    assert result.status == "REJECTED"
    assert "experiment count mismatch" in result.reasons


@pytest.mark.parametrize("value", [True, 1.0])
def test_holdout_count_requires_a_real_integer(authority, value):
    result = signed_decision(authority, holdout_consumption_count=value)
    assert result.status == "REJECTED"
    assert "holdout not single-use consumed" in result.reasons


@pytest.mark.parametrize("field", ["experiment_count", "registered_experiment_count"])
def test_each_count_type_is_checked_independently(authority, field):
    result = signed_decision(authority, **{field: 4.0})
    assert result.status == "REJECTED"
    assert "experiment count mismatch" in result.reasons
