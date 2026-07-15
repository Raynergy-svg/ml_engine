from __future__ import annotations

import base64
from dataclasses import asdict
from datetime import date, datetime, timedelta, timezone

import pytest
from pydantic import ValidationError

from src.evidence.canonical import CanonicalizationError, canonical_bytes, canonical_json
from src.evidence.contracts import (
    ArtifactRef,
    AssetClass,
    AuthorityRole,
    CapabilityProfile,
    DatasetManifest,
    DispositionEvent,
    DispositionReceipt,
    DispositionState,
    EvaluationReport,
    EvidencePackage,
    GateResult,
    GateStatus,
    ImportCheck,
    ImportDecision,
    JobManifest,
    LocalImportVerdict,
    MetricValue,
    ModelTrainingSpec,
    PartitionRef,
    ResourceClass,
    SafetyAssertion,
    StrategyManifest,
    StrictContract,
)
from src.evidence.event_store import EventChainError, reconstruct_disposition
from src.evidence.hashing import content_digest
from src.evidence.importer import build_import_verdict
from src.evidence.signing import (
    Ed25519Signer,
    SignatureVerificationError,
    TrustStore,
    verify_envelope,
)
from src.evidence.transition_policy import AuthorityRegistry, TransitionPolicyError, enforce_transition

ZERO = "0" * 64
ONE = "1" * 64
NOW = datetime(2026, 7, 11, 16, 30, tzinfo=timezone.utc)


def dataset_manifest() -> DatasetManifest:
    partition = PartitionRef(
        partition_id="fx-daily-2026-07",
        uri="axiom-data/fx/daily/year=2026/month=07/part-000.parquet",
        digest=ONE,
        size_bytes=1024,
        row_count=80,
    )
    return DatasetManifest(
        dataset_id="fx-daily-2026-07-11",
        asset_class=AssetClass.FX,
        domain="daily-bars",
        provider="oanda",
        retrieved_at=NOW,
        coverage_start=date(2014, 1, 1),
        coverage_end=date(2026, 7, 10),
        instruments=("EUR_USD", "USD_JPY"),
        data_schema_version="fx-daily-v1",
        point_in_time_rules=("bar close must precede feature as-of time",),
        exclusions=("weekend synthetic bars",),
        license_metadata={"redistribution": False},
        partitions=(partition,),
    )


def test_canonical_serialization_golden_vector() -> None:
    value = {"z": [3, -0.0, True], "a": "café", "schema_version": "1.0.0"}
    expected = '{"a":"café","schema_version":"1.0.0","z":[3,0.0,true]}'
    assert canonical_json(value) == expected
    assert content_digest(value) == "e1ecf3dcd69dd741aa69fa5e460c4c7bdc7a0d4837ef755f25c75ba0f370cbd5"
    assert canonical_bytes(value) == expected.encode("utf-8")


def test_canonical_serialization_rejects_unstable_values() -> None:
    with pytest.raises(CanonicalizationError, match="non-finite"):
        canonical_json({"bad": float("nan")})
    with pytest.raises(CanonicalizationError, match="keys must be strings"):
        canonical_json({1: "bad"})


def test_contracts_are_frozen_strict_and_reject_unknown_fields() -> None:
    profile = CapabilityProfile(profile_id="remote-risk-target")
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        CapabilityProfile.model_validate({"profile_id": "remote", "may_trade": False})
    with pytest.raises(ValidationError, match="Input should be False"):
        CapabilityProfile(profile_id="unsafe", may_place_or_cancel_orders=True)  # type: ignore[arg-type]
    with pytest.raises(ValidationError, match="frozen"):
        profile.profile_id = "changed"  # type: ignore[misc]


def test_explicit_schema_migration_and_unknown_version_refusal() -> None:
    class ExampleContract(StrictContract):
        label: str
        count: int

    def migrate_090(payload: dict[str, object]) -> dict[str, object]:
        payload["label"] = payload.pop("legacy_label")
        return payload

    ExampleContract.register_migration("0.9.0", "1.0.0", migrate_090)
    migrated = ExampleContract.from_versioned_payload(
        {"schema_version": "0.9.0", "legacy_label": "risk", "count": 2}
    )
    assert migrated == ExampleContract(label="risk", count=2)
    with pytest.raises(ValueError, match="unsupported"):
        ExampleContract.from_versioned_payload({"schema_version": "9.0.0", "label": "x", "count": 1})


def test_signed_schema_migration_authenticates_original_payload_before_migrating() -> None:
    class ExampleContract(StrictContract):
        label: str
        count: int

    def migrate_090(payload: dict[str, object]) -> dict[str, object]:
        payload["label"] = payload.pop("legacy_label")
        return payload

    ExampleContract.register_migration("0.9.0", "1.0.0", migrate_090)
    signer = Ed25519Signer.generate()
    trust = TrustStore()
    trust.add(signer.trusted_key(valid_from=NOW - timedelta(seconds=1)))
    envelope = signer.sign_versioned_payload(
        {"schema_version": "0.9.0", "legacy_label": "risk", "count": 2},
        payload_type="ExampleContract",
        created_at=NOW,
    )
    assert verify_envelope(envelope, ExampleContract, trust) == ExampleContract(label="risk", count=2)


def test_nested_contract_values_are_deeply_immutable() -> None:
    class NestedContract(StrictContract):
        metrics: dict[str, object]

    contract = NestedContract(metrics={"summary": {"count": 1}, "samples": [1, 2]})
    digest = content_digest(contract)
    with pytest.raises(TypeError, match="immutable"):
        contract.metrics["summary"]["count"] = 2  # type: ignore[index]
    with pytest.raises(AttributeError):
        contract.metrics["samples"].append(3)  # type: ignore[union-attr]
    assert content_digest(contract) == digest


def test_all_eight_formal_contracts_validate_as_one_lineage() -> None:
    dataset = dataset_manifest()
    dataset_digest = content_digest(dataset)
    strategy = StrategyManifest(
        strategy_id="crypto-momentum-v1",
        lane_id="crypto_momentum",
        experiment_id="cm-2026-07",
        hypothesis="Cross-sectional momentum survives declared costs.",
        exploratory=False,
        frozen_at=NOW,
        universe=("BTC_USDT", "ETH_USDT"),
        causal_lag="1h",
        parameters={"lookback_days": 14},
        cost_model={"bps_per_turn": 10.0},
        evaluation_windows=("2021-01-01/2026-05-31",),
        search_space={"lookback_days": [14]},
        trial_budget=1,
        metrics=("deflated_sharpe", "max_drawdown"),
        promotion_criteria=("deflated_sharpe_probability >= 0.95",),
    )
    model_spec = ModelTrainingSpec(
        target="forward_realized_volatility",
        features=("atr_20", "realized_vol_20"),
        causal_lag="1d",
        search_space={"max_depth": [2, 3]},
        metrics=("qlike", "mae"),
        promotion_criteria=("qlike < incumbent_qlike",),
    )
    job = JobManifest(
        job_id="risk-target-fold-0",
        git_commit="a" * 40,
        container_digest="b" * 64,
        dataset_manifest_digests=(dataset_digest,),
        model_training_spec=model_spec,
        feature_pipeline_version="risk-target-v1",
        configuration_digest="c" * 64,
        random_seeds=(7,),
        assigned_unit="volatility-head/fold-0",
        resource_class=ResourceClass.CPU_SMALL,
        capability_profile_digest=content_digest(CapabilityProfile(profile_id="remote-risk")),
        trial_budget=2,
        expected_outputs=("model.pkl", "evaluation.json"),
        created_at=NOW,
    )
    report = EvaluationReport(
        report_id="risk-target-fold-0-report",
        job_manifest_digest=content_digest(job),
        evaluator_id="independent-verifier-1",
        independent=True,
        created_at=NOW,
        temporal_holdout="2025-01-01/2026-07-10",
        purge_observations=20,
        embargo_observations=5,
        trial_count=2,
        effective_sample_size=250.0,
        metrics={"qlike": MetricValue(value=0.82, tolerance=1e-9)},
        gates=(GateResult(gate_id="incumbent", status=GateStatus.PASS),),
        incumbent_comparison={"qlike_delta": -0.04},
        passed=True,
    )
    artifact = ArtifactRef(
        artifact_id="volatility-head",
        relative_path="artifacts/volatility.pkl",
        digest="d" * 64,
        size_bytes=512,
        media_type="application/octet-stream",
    )
    package = EvidencePackage(
        package_id="risk-target-2026-07-fold-0",
        lane_id="risk_target",
        producer_id="worker-1",
        created_at=NOW,
        job_manifest_digest=content_digest(job),
        dataset_manifest_digests=(dataset_digest,),
        strategy_manifest_digest=content_digest(strategy),
        evaluation_report_digests=(content_digest(report),),
        artifacts=(artifact,),
        safety_assertions=(SafetyAssertion(assertion_id="no-broker-credentials", passed=True),),
        checksums={artifact.relative_path: artifact.digest},
    )
    event = DispositionEvent(
        package_digest=content_digest(package),
        sequence=0,
        previous_event_digest=None,
        from_state=None,
        to_state=DispositionState.CREATED,
        authority=AuthorityRole.PRODUCER,
        actor_id="worker-1",
        signer_key_id="ed25519:test-key",
        occurred_at=NOW,
        reason="producer completed declared workload",
    )
    verdict = LocalImportVerdict(
        verdict_id="import-1",
        package_digest=event.package_digest,
        importer_id="local-importer",
        created_at=NOW,
        checks=(ImportCheck(check_id="digest", passed=True),),
        decision=ImportDecision.ACCEPT_FOR_QUARANTINE,
    )
    assert all(content_digest(item) for item in (dataset, strategy, job, report, package, event, verdict))


def test_one_byte_payload_change_invalidates_signed_object() -> None:
    signer = Ed25519Signer.generate()
    trust = TrustStore()
    trust.add(signer.trusted_key(valid_from=NOW - timedelta(seconds=1)))
    envelope = signer.sign(dataset_manifest(), created_at=NOW)
    assert verify_envelope(envelope, DatasetManifest, trust) == dataset_manifest()

    payload = dict(envelope.payload)
    payload["provider"] = "oandb"  # one UTF-8 byte differs from the signed value
    tampered = envelope.model_copy(update={"payload": payload})
    with pytest.raises(SignatureVerificationError, match="digest mismatch"):
        verify_envelope(tampered, DatasetManifest, trust)


def test_one_byte_signature_change_is_detected() -> None:
    signer = Ed25519Signer.generate()
    trust = TrustStore()
    trust.add(signer.trusted_key(valid_from=NOW - timedelta(seconds=1)))
    envelope = signer.sign(dataset_manifest(), created_at=NOW)
    raw = bytearray(base64.b64decode(envelope.signature.signature_b64))
    raw[0] ^= 0x01
    changed_signature = envelope.signature.model_copy(
        update={"signature_b64": base64.b64encode(bytes(raw)).decode("ascii")}
    )
    tampered = envelope.model_copy(update={"signature": changed_signature})
    with pytest.raises(SignatureVerificationError, match="signature verification failed"):
        verify_envelope(tampered, DatasetManifest, trust)


def test_signature_metadata_is_cryptographically_bound() -> None:
    signer = Ed25519Signer.generate()
    trust = TrustStore()
    trust.add(signer.trusted_key(valid_from=NOW - timedelta(days=1)))
    envelope = signer.sign(dataset_manifest(), created_at=NOW)
    changed_timestamp = envelope.signature.model_copy(update={"created_at": NOW - timedelta(seconds=1)})
    tampered = envelope.model_copy(update={"signature": changed_timestamp})
    with pytest.raises(SignatureVerificationError, match="signature verification failed"):
        verify_envelope(tampered, DatasetManifest, trust)


def test_key_rotation_overlap_and_revocation() -> None:
    old = Ed25519Signer.generate()
    new = Ed25519Signer.generate()
    trust = TrustStore()
    trust.add(old.trusted_key(valid_from=NOW - timedelta(days=30)))
    trust.add(new.trusted_key(valid_from=NOW - timedelta(minutes=5)))
    assert verify_envelope(old.sign(dataset_manifest(), created_at=NOW), DatasetManifest, trust)
    assert verify_envelope(new.sign(dataset_manifest(), created_at=NOW), DatasetManifest, trust)

    trust.revoke(old.key_id, NOW + timedelta(minutes=1))
    before_revocation = old.sign(dataset_manifest(), created_at=NOW)
    assert verify_envelope(before_revocation, DatasetManifest, trust)
    after_revocation = old.sign(dataset_manifest(), created_at=NOW + timedelta(minutes=2))
    with pytest.raises(SignatureVerificationError, match="trusted interval"):
        verify_envelope(after_revocation, DatasetManifest, trust)


def make_ledger() -> tuple[list, TrustStore, AuthorityRegistry]:
    signers = {
        AuthorityRole.PRODUCER: Ed25519Signer.generate(),
        AuthorityRole.LOCAL_IMPORTER: Ed25519Signer.generate(),
        AuthorityRole.INDEPENDENT_VERIFIER: Ed25519Signer.generate(),
        AuthorityRole.OPERATOR: Ed25519Signer.generate(),
        AuthorityRole.PROMOTION_SERVICE: Ed25519Signer.generate(),
        AuthorityRole.SAFETY_MONITOR: Ed25519Signer.generate(),
    }
    actors = {
        AuthorityRole.PRODUCER: "worker-1",
        AuthorityRole.LOCAL_IMPORTER: "local-importer",
        AuthorityRole.INDEPENDENT_VERIFIER: "independent-verifier",
        AuthorityRole.OPERATOR: "operator-primary",
        AuthorityRole.PROMOTION_SERVICE: "local-promotion-service",
        AuthorityRole.SAFETY_MONITOR: "tier-7",
    }
    trust = TrustStore()
    authorities = AuthorityRegistry()
    for role, signer in signers.items():
        trust.add(signer.trusted_key(valid_from=NOW - timedelta(seconds=1)))
        authorities.register(actor_id=actors[role], role=role, key_ids=(signer.key_id,))

    steps = [
        (None, DispositionState.CREATED, AuthorityRole.PRODUCER),
        (DispositionState.CREATED, DispositionState.RECEIVED, AuthorityRole.LOCAL_IMPORTER),
        (DispositionState.RECEIVED, DispositionState.HASH_VERIFIED, AuthorityRole.LOCAL_IMPORTER),
        (DispositionState.HASH_VERIFIED, DispositionState.POLICY_VERIFIED, AuthorityRole.LOCAL_IMPORTER),
        (DispositionState.POLICY_VERIFIED, DispositionState.METRIC_REPLAYED, AuthorityRole.INDEPENDENT_VERIFIER),
        (DispositionState.METRIC_REPLAYED, DispositionState.QUARANTINED, AuthorityRole.LOCAL_IMPORTER),
        (DispositionState.QUARANTINED, DispositionState.SHADOW, AuthorityRole.OPERATOR),
        (DispositionState.SHADOW, DispositionState.OPERATOR_APPROVED, AuthorityRole.OPERATOR),
        (DispositionState.OPERATOR_APPROVED, DispositionState.CHAMPION, AuthorityRole.PROMOTION_SERVICE),
        (DispositionState.CHAMPION, DispositionState.RETIRED, AuthorityRole.SAFETY_MONITOR),
    ]
    ledger = []
    previous_digest = None
    package_digest = "e" * 64
    for sequence, (from_state, to_state, role) in enumerate(steps):
        signer = signers[role]
        event = DispositionEvent(
            package_digest=package_digest,
            sequence=sequence,
            previous_event_digest=previous_digest,
            from_state=from_state,
            to_state=to_state,
            authority=role,
            actor_id=actors[role],
            signer_key_id=signer.key_id,
            occurred_at=NOW + timedelta(seconds=sequence),
            reason=f"advance to {to_state.value}",
        )
        envelope = signer.sign(event, created_at=event.occurred_at)
        ledger.append(envelope)
        previous_digest = envelope.payload_digest
    receipts = [
        DispositionReceipt(envelope=envelope, received_at=NOW + timedelta(minutes=1))
        for envelope in ledger
    ]
    return receipts, trust, authorities


def test_event_chain_reconstructs_current_state_from_ledger_only() -> None:
    ledger, trust, authorities = make_ledger()
    state = reconstruct_disposition(ledger, trust_store=trust, authorities=authorities)
    assert state.package_digest == "e" * 64
    assert state.state == DispositionState.RETIRED
    assert state.sequence == 9
    assert state.head_event_digest == ledger[-1].envelope.payload_digest

    # Reconstruction must not depend on the ledger's concrete iterable type.
    from_tuple = reconstruct_disposition(tuple(ledger), trust_store=trust, authorities=authorities)
    assert canonical_bytes(asdict(from_tuple)) == canonical_bytes(asdict(state))


def test_event_chain_rejects_broken_link_duplicate_and_fork() -> None:
    ledger, trust, authorities = make_ledger()
    broken_payload = dict(ledger[2].envelope.payload)
    broken_payload["previous_event_digest"] = ZERO
    broken_envelope = ledger[2].envelope.model_copy(update={"payload": broken_payload})
    broken = DispositionReceipt(envelope=broken_envelope, received_at=ledger[2].received_at)
    with pytest.raises(EventChainError, match="invalid event"):
        reconstruct_disposition([ledger[0], ledger[1], broken], trust_store=trust, authorities=authorities)

    with pytest.raises(EventChainError, match="duplicate event digest"):
        reconstruct_disposition(
            [ledger[0], ledger[1], ledger[1]], trust_store=trust, authorities=authorities
        )


def test_transition_authority_is_not_self_attested() -> None:
    producer = Ed25519Signer.generate()
    registry = AuthorityRegistry()
    registry.register(actor_id="worker-1", role=AuthorityRole.PRODUCER, key_ids=(producer.key_id,))
    forged = DispositionEvent(
        package_digest="e" * 64,
        sequence=1,
        previous_event_digest="f" * 64,
        from_state=DispositionState.SHADOW,
        to_state=DispositionState.OPERATOR_APPROVED,
        authority=AuthorityRole.OPERATOR,
        actor_id="worker-1",
        signer_key_id=producer.key_id,
        occurred_at=NOW,
        reason="worker attempted self-promotion",
    )
    with pytest.raises(TransitionPolicyError, match="not registered"):
        enforce_transition(forged, registry, producer.key_id)


def test_one_identity_cannot_self_promote_across_a_complete_ledger() -> None:
    signer = Ed25519Signer.generate()
    trust = TrustStore()
    trust.add(signer.trusted_key(valid_from=NOW - timedelta(seconds=1)))
    authorities = AuthorityRegistry()
    for role in (
        AuthorityRole.PRODUCER,
        AuthorityRole.LOCAL_IMPORTER,
        AuthorityRole.INDEPENDENT_VERIFIER,
        AuthorityRole.OPERATOR,
        AuthorityRole.PROMOTION_SERVICE,
    ):
        authorities.register(actor_id="same-process", role=role, key_ids=(signer.key_id,))
    steps = (
        (None, DispositionState.CREATED, AuthorityRole.PRODUCER),
        (DispositionState.CREATED, DispositionState.RECEIVED, AuthorityRole.LOCAL_IMPORTER),
        (DispositionState.RECEIVED, DispositionState.HASH_VERIFIED, AuthorityRole.LOCAL_IMPORTER),
        (DispositionState.HASH_VERIFIED, DispositionState.POLICY_VERIFIED, AuthorityRole.LOCAL_IMPORTER),
        (DispositionState.POLICY_VERIFIED, DispositionState.METRIC_REPLAYED, AuthorityRole.INDEPENDENT_VERIFIER),
        (DispositionState.METRIC_REPLAYED, DispositionState.QUARANTINED, AuthorityRole.LOCAL_IMPORTER),
        (DispositionState.QUARANTINED, DispositionState.SHADOW, AuthorityRole.OPERATOR),
        (DispositionState.SHADOW, DispositionState.OPERATOR_APPROVED, AuthorityRole.OPERATOR),
        (DispositionState.OPERATOR_APPROVED, DispositionState.CHAMPION, AuthorityRole.PROMOTION_SERVICE),
    )
    receipts = []
    previous = None
    for sequence, (from_state, to_state, role) in enumerate(steps):
        event = DispositionEvent(
            package_digest="e" * 64,
            sequence=sequence,
            previous_event_digest=previous,
            from_state=from_state,
            to_state=to_state,
            authority=role,
            actor_id="same-process",
            signer_key_id=signer.key_id,
            occurred_at=NOW + timedelta(seconds=sequence),
            reason="adversarial self-promotion",
        )
        envelope = signer.sign(event, created_at=event.occurred_at)
        receipts.append(DispositionReceipt(envelope=envelope, received_at=NOW + timedelta(minutes=1)))
        previous = envelope.payload_digest
    with pytest.raises(EventChainError, match="separation of duties"):
        reconstruct_disposition(receipts, trust_store=trust, authorities=authorities)


def test_backdated_revoked_key_cannot_append_a_future_event() -> None:
    signer = Ed25519Signer.generate()
    trust = TrustStore()
    trust.add(signer.trusted_key(valid_from=NOW - timedelta(days=1), revoked_at=NOW))
    authorities = AuthorityRegistry()
    authorities.register(actor_id="worker", role=AuthorityRole.PRODUCER, key_ids=(signer.key_id,))
    event = DispositionEvent(
        package_digest="e" * 64,
        sequence=0,
        previous_event_digest=None,
        from_state=None,
        to_state=DispositionState.CREATED,
        authority=AuthorityRole.PRODUCER,
        actor_id="worker",
        signer_key_id=signer.key_id,
        occurred_at=NOW + timedelta(days=1),
        reason="backdated signature attempt",
    )
    envelope = signer.sign(event, created_at=NOW - timedelta(seconds=1))
    receipt = DispositionReceipt(envelope=envelope, received_at=NOW + timedelta(days=1))
    with pytest.raises(EventChainError, match="does not match disposition occurred_at"):
        reconstruct_disposition((receipt,), trust_store=trust, authorities=authorities)

    backdated_event = event.model_copy(update={"occurred_at": NOW - timedelta(seconds=1)})
    backdated_envelope = signer.sign(backdated_event, created_at=backdated_event.occurred_at)
    backdated_receipt = DispositionReceipt(
        envelope=backdated_envelope,
        received_at=NOW + timedelta(seconds=1),
    )
    with pytest.raises(EventChainError, match="not trusted at ledger receipt"):
        reconstruct_disposition((backdated_receipt,), trust_store=trust, authorities=authorities)


def test_import_verdict_fails_closed() -> None:
    accepted = build_import_verdict(
        verdict_id="v1",
        package_digest="e" * 64,
        importer_id="local-importer",
        created_at=NOW,
        checks=(ImportCheck(check_id="hash", passed=True),),
    )
    assert accepted.decision == ImportDecision.ACCEPT_FOR_QUARANTINE

    rejected = build_import_verdict(
        verdict_id="v2",
        package_digest="e" * 64,
        importer_id="local-importer",
        created_at=NOW,
        checks=(ImportCheck(check_id="hash", passed=False, details="mismatch"),),
    )
    assert rejected.decision == ImportDecision.REJECT
    assert rejected.rejection_reason == "failed checks: hash"
