from __future__ import annotations

import os
import shutil
import threading
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from src.evidence.canonical import canonical_bytes
from src.evidence.contracts import (
    ArtifactRef,
    AuthorityRole,
    ChampionPointer,
    DispositionEvent,
    DispositionReceipt,
    DispositionState,
    EvidencePackage,
    ImportCheck,
    SafetyAssertion,
)
from src.evidence.hashing import content_digest, sha256_bytes
from src.evidence.importer import build_import_verdict
from src.evidence.signing import Ed25519Signer, TrustStore
from src.evidence.store import (
    ChampionPointerError,
    ConcurrentHeadError,
    EvidenceStore,
    EvidenceStoreError,
    ProjectionStaleError,
    StoreCorruptionError,
)
from src.evidence.transition_policy import AuthorityRegistry

NOW = datetime(2026, 7, 11, 17, 0, tzinfo=timezone.utc)


@dataclass
class StoreContext:
    store: EvidenceStore
    signers: dict[AuthorityRole, Ed25519Signer]
    actors: dict[AuthorityRole, str]
    package: EvidencePackage
    package_digest: str
    artifact_bytes: bytes


def build_store(tmp_path: Path, *, write_package: bool = True) -> StoreContext:
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
        trust.add(signer.trusted_key(valid_from=NOW - timedelta(days=1)))
        authorities.register(actor_id=actors[role], role=role, key_ids=(signer.key_id,))

    artifact_bytes = b"immutable-risk-target-model-v1"
    artifact = ArtifactRef(
        artifact_id="volatility-head",
        relative_path="artifacts/volatility.bin",
        digest=sha256_bytes(artifact_bytes),
        size_bytes=len(artifact_bytes),
        media_type="application/octet-stream",
    )
    package = EvidencePackage(
        package_id="risk-target-package-v1",
        lane_id="risk_target",
        producer_id=actors[AuthorityRole.PRODUCER],
        created_at=NOW,
        job_manifest_digest="1" * 64,
        dataset_manifest_digests=("2" * 64,),
        evaluation_report_digests=("3" * 64,),
        artifacts=(artifact,),
        safety_assertions=(SafetyAssertion(assertion_id="remote-no-authority", passed=True),),
        checksums={artifact.relative_path: artifact.digest},
    )
    store = EvidenceStore(
        tmp_path / "evidence",
        trust_store=trust,
        authorities=authorities,
        trusted_clock=lambda: NOW + timedelta(minutes=1),
    )
    envelope = signers[AuthorityRole.PRODUCER].sign(package, created_at=NOW)
    if write_package:
        store.write_package(envelope, {artifact.relative_path: artifact_bytes})
    return StoreContext(
        store=store,
        signers=signers,
        actors=actors,
        package=package,
        package_digest=envelope.payload_digest,
        artifact_bytes=artifact_bytes,
    )


def signed_event(
    context: StoreContext,
    *,
    sequence: int,
    previous_digest: str | None,
    from_state: DispositionState | None,
    to_state: DispositionState,
    role: AuthorityRole,
    reason: str | None = None,
    metadata: dict[str, str] | None = None,
):
    signer = context.signers[role]
    event = DispositionEvent(
        package_digest=context.package_digest,
        sequence=sequence,
        previous_event_digest=previous_digest,
        from_state=from_state,
        to_state=to_state,
        authority=role,
        actor_id=context.actors[role],
        signer_key_id=signer.key_id,
        occurred_at=NOW + timedelta(seconds=sequence),
        reason=reason or f"advance to {to_state.value}",
        metadata=metadata or {},
    )
    return signer.sign(event, created_at=event.occurred_at)


def write_accepted_verdict(context: StoreContext) -> str:
    verdict = build_import_verdict(
        verdict_id=f"accepted-{context.package.package_id}",
        package_digest=context.package_digest,
        importer_id=context.actors[AuthorityRole.LOCAL_IMPORTER],
        created_at=NOW,
        checks=(ImportCheck(check_id="local-policy", passed=True),),
    )
    signer = context.signers[AuthorityRole.LOCAL_IMPORTER]
    return context.store.write_verdict(signer.sign(verdict, created_at=NOW))


def advance_to_champion(context: StoreContext) -> str:
    verdict_digest = write_accepted_verdict(context)
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
    ]
    head = None
    for sequence, (from_state, to_state, role) in enumerate(steps):
        metadata = None
        if to_state == DispositionState.QUARANTINED:
            metadata = {"verdict_digest": verdict_digest}
        if to_state == DispositionState.CHAMPION:
            artifact = context.package.artifacts[0]
            metadata = {
                "artifact_digest": artifact.digest,
                "artifact_id": artifact.artifact_id,
                "lane_id": context.package.lane_id,
            }
        envelope = signed_event(
            context,
            sequence=sequence,
            previous_digest=head,
            from_state=from_state,
            to_state=to_state,
            role=role,
            metadata=metadata,
        )
        state = context.store.append_disposition(envelope, expected_head_digest=head)
        head = state.head_event_digest
    assert head is not None
    return head


def test_store_creates_required_layout_and_atomically_publishes_package(tmp_path: Path) -> None:
    context = build_store(tmp_path)
    for name in EvidenceStore.DIRECTORY_NAMES:
        assert (context.store.root / name).is_dir()
    package_dir = context.store.root / "packages" / context.package_digest
    assert package_dir.is_dir()
    assert context.store.load_package(context.package_digest) == context.package
    assert not list((context.store.root / "packages").glob(".*.tmp-*"))

    signer = context.signers[AuthorityRole.PRODUCER]
    envelope = signer.sign(context.package, created_at=NOW)
    assert context.store.write_package(
        envelope,
        {"artifacts/volatility.bin": context.artifact_bytes},
    ) == context.package_digest


def test_partial_package_failure_leaves_no_visible_or_temporary_package(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    context = build_store(tmp_path, write_package=False)
    signer = context.signers[AuthorityRole.PRODUCER]
    envelope = signer.sign(context.package, created_at=NOW)
    original = EvidenceStore._write_new_file.__func__

    def fail_after_envelope(cls, destination: Path, data: bytes) -> None:
        if destination.name == "volatility.bin":
            raise OSError("injected artifact write failure")
        original(cls, destination, data)

    monkeypatch.setattr(EvidenceStore, "_write_new_file", classmethod(fail_after_envelope))
    with pytest.raises(OSError, match="injected"):
        context.store.write_package(
            envelope,
            {"artifacts/volatility.bin": context.artifact_bytes},
        )
    assert not (context.store.root / "packages" / context.package_digest).exists()
    assert not list((context.store.root / "packages").glob(".*.tmp-*"))


def test_disposition_append_is_monotonic_linked_and_append_only(tmp_path: Path) -> None:
    context = build_store(tmp_path)
    created = signed_event(
        context,
        sequence=0,
        previous_digest=None,
        from_state=None,
        to_state=DispositionState.CREATED,
        role=AuthorityRole.PRODUCER,
    )
    state = context.store.append_disposition(created, expected_head_digest=None)
    received = signed_event(
        context,
        sequence=1,
        previous_digest=state.head_event_digest,
        from_state=DispositionState.CREATED,
        to_state=DispositionState.RECEIVED,
        role=AuthorityRole.LOCAL_IMPORTER,
    )
    state = context.store.append_disposition(received, expected_head_digest=state.head_event_digest)
    assert state.state == DispositionState.RECEIVED
    ledger = context.store.load_ledger(context.package_digest)
    assert [item.payload_digest for item in ledger] == [created.payload_digest, received.payload_digest]

    invalid = signed_event(
        context,
        sequence=2,
        previous_digest="0" * 64,
        from_state=DispositionState.RECEIVED,
        to_state=DispositionState.HASH_VERIFIED,
        role=AuthorityRole.LOCAL_IMPORTER,
    )
    with pytest.raises(EvidenceStoreError, match="previous_event_digest mismatch"):
        context.store.append_disposition(invalid, expected_head_digest=state.head_event_digest)
    assert len(context.store.load_ledger(context.package_digest)) == 2


def test_concurrent_head_compare_and_swap_allows_only_one_branch(tmp_path: Path) -> None:
    context = build_store(tmp_path)
    created = signed_event(
        context,
        sequence=0,
        previous_digest=None,
        from_state=None,
        to_state=DispositionState.CREATED,
        role=AuthorityRole.PRODUCER,
    )
    first = context.store.append_disposition(created, expected_head_digest=None)
    received = signed_event(
        context,
        sequence=1,
        previous_digest=first.head_event_digest,
        from_state=DispositionState.CREATED,
        to_state=DispositionState.RECEIVED,
        role=AuthorityRole.LOCAL_IMPORTER,
    )
    second = context.store.append_disposition(received, expected_head_digest=first.head_event_digest)
    expected_head = second.head_event_digest
    branches = [
        signed_event(
            context,
            sequence=2,
            previous_digest=expected_head,
            from_state=DispositionState.RECEIVED,
            to_state=DispositionState.HASH_VERIFIED,
            role=AuthorityRole.LOCAL_IMPORTER,
            reason="hash passed",
        ),
        signed_event(
            context,
            sequence=2,
            previous_digest=expected_head,
            from_state=DispositionState.RECEIVED,
            to_state=DispositionState.REJECTED,
            role=AuthorityRole.LOCAL_IMPORTER,
            reason="hash failed",
        ),
    ]
    barrier = threading.Barrier(2)
    outcomes: list[str] = []

    def append(branch) -> None:
        barrier.wait()
        try:
            context.store.append_disposition(branch, expected_head_digest=expected_head)
            outcomes.append("committed")
        except ConcurrentHeadError:
            outcomes.append("conflict")

    threads = [threading.Thread(target=append, args=(branch,)) for branch in branches]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=10)
    assert outcomes.count("committed") == 1
    assert outcomes.count("conflict") == 1
    assert len(context.store.load_ledger(context.package_digest)) == 3


def test_injected_fork_is_detected_during_rebuild(tmp_path: Path) -> None:
    context = build_store(tmp_path)
    created = signed_event(
        context,
        sequence=0,
        previous_digest=None,
        from_state=None,
        to_state=DispositionState.CREATED,
        role=AuthorityRole.PRODUCER,
    )
    state = context.store.append_disposition(created, expected_head_digest=None)
    received = signed_event(
        context,
        sequence=1,
        previous_digest=state.head_event_digest,
        from_state=DispositionState.CREATED,
        to_state=DispositionState.RECEIVED,
        role=AuthorityRole.LOCAL_IMPORTER,
        reason="branch one",
    )
    context.store.append_disposition(received, expected_head_digest=state.head_event_digest)
    fork = signed_event(
        context,
        sequence=1,
        previous_digest=state.head_event_digest,
        from_state=DispositionState.CREATED,
        to_state=DispositionState.RECEIVED,
        role=AuthorityRole.LOCAL_IMPORTER,
        reason="branch two",
    )
    path = (
        context.store.root
        / "dispositions"
        / context.package_digest
        / f"{1:020d}-{fork.payload_digest}.json"
    )
    path.write_bytes(
        canonical_bytes(
            DispositionReceipt(envelope=fork, received_at=NOW + timedelta(minutes=1))
        )
    )
    with pytest.raises(StoreCorruptionError, match="invalid disposition ledger"):
        context.store.rebuild_indexes()


def test_signed_import_verdict_is_immutable_and_authorized(tmp_path: Path) -> None:
    context = build_store(tmp_path)
    verdict = build_import_verdict(
        verdict_id="import-v1",
        package_digest=context.package_digest,
        importer_id=context.actors[AuthorityRole.LOCAL_IMPORTER],
        created_at=NOW,
        checks=(ImportCheck(check_id="hash", passed=True),),
    )
    envelope = context.signers[AuthorityRole.LOCAL_IMPORTER].sign(verdict, created_at=NOW)
    digest = context.store.write_verdict(envelope)
    path = context.store.root / "verdicts" / f"{digest}.json"
    assert path.read_bytes() == canonical_bytes(envelope)
    assert context.store.write_verdict(envelope) == digest


def test_orphan_import_verdict_is_rejected(tmp_path: Path) -> None:
    context = build_store(tmp_path)
    verdict = build_import_verdict(
        verdict_id="orphan",
        package_digest="f" * 64,
        importer_id=context.actors[AuthorityRole.LOCAL_IMPORTER],
        created_at=NOW,
        checks=(ImportCheck(check_id="hash", passed=True),),
    )
    envelope = context.signers[AuthorityRole.LOCAL_IMPORTER].sign(verdict, created_at=NOW)
    with pytest.raises(FileNotFoundError, match="unknown package"):
        context.store.write_verdict(envelope)


def test_quarantine_requires_a_durable_accepted_import_verdict(tmp_path: Path) -> None:
    context = build_store(tmp_path)
    steps = [
        (None, DispositionState.CREATED, AuthorityRole.PRODUCER),
        (DispositionState.CREATED, DispositionState.RECEIVED, AuthorityRole.LOCAL_IMPORTER),
        (DispositionState.RECEIVED, DispositionState.HASH_VERIFIED, AuthorityRole.LOCAL_IMPORTER),
        (DispositionState.HASH_VERIFIED, DispositionState.POLICY_VERIFIED, AuthorityRole.LOCAL_IMPORTER),
        (DispositionState.POLICY_VERIFIED, DispositionState.METRIC_REPLAYED, AuthorityRole.INDEPENDENT_VERIFIER),
    ]
    head = None
    for sequence, (from_state, to_state, role) in enumerate(steps):
        envelope = signed_event(
            context,
            sequence=sequence,
            previous_digest=head,
            from_state=from_state,
            to_state=to_state,
            role=role,
        )
        head = context.store.append_disposition(
            envelope, expected_head_digest=head
        ).head_event_digest
    quarantine = signed_event(
        context,
        sequence=5,
        previous_digest=head,
        from_state=DispositionState.METRIC_REPLAYED,
        to_state=DispositionState.QUARANTINED,
        role=AuthorityRole.LOCAL_IMPORTER,
        metadata={"verdict_digest": "f" * 64},
    )
    with pytest.raises(EvidenceStoreError, match="accepted local import verdict"):
        context.store.append_disposition(quarantine, expected_head_digest=head)
    assert len(context.store.load_ledger(context.package_digest)) == 5

    rejected = build_import_verdict(
        verdict_id="rejected-local-policy",
        package_digest=context.package_digest,
        importer_id=context.actors[AuthorityRole.LOCAL_IMPORTER],
        created_at=NOW,
        checks=(ImportCheck(check_id="local-policy", passed=False),),
    )
    signer = context.signers[AuthorityRole.LOCAL_IMPORTER]
    rejected_digest = context.store.write_verdict(signer.sign(rejected, created_at=NOW))
    rejected_quarantine = signed_event(
        context,
        sequence=5,
        previous_digest=head,
        from_state=DispositionState.METRIC_REPLAYED,
        to_state=DispositionState.QUARANTINED,
        role=AuthorityRole.LOCAL_IMPORTER,
        metadata={"verdict_digest": rejected_digest},
    )
    with pytest.raises(EvidenceStoreError, match="not accepted"):
        context.store.append_disposition(rejected_quarantine, expected_head_digest=head)

    accepted_digest = write_accepted_verdict(context)
    accepted_quarantine = signed_event(
        context,
        sequence=5,
        previous_digest=head,
        from_state=DispositionState.METRIC_REPLAYED,
        to_state=DispositionState.QUARANTINED,
        role=AuthorityRole.LOCAL_IMPORTER,
        metadata={"verdict_digest": accepted_digest},
    )
    state = context.store.append_disposition(accepted_quarantine, expected_head_digest=head)
    assert state.state == DispositionState.QUARANTINED


def test_package_retry_repairs_projection_after_committed_stale_result(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    context = build_store(tmp_path, write_package=False)
    signer = context.signers[AuthorityRole.PRODUCER]
    envelope = signer.sign(context.package, created_at=NOW)
    original = context.store._rebuild_indexes_unlocked
    calls = 0

    def fail_once() -> bytes:
        nonlocal calls
        calls += 1
        if calls == 1:
            raise OSError("injected projection failure")
        return original()

    monkeypatch.setattr(context.store, "_rebuild_indexes_unlocked", fail_once)
    with pytest.raises(ProjectionStaleError, match="package committed"):
        context.store.write_package(
            envelope, {"artifacts/volatility.bin": context.artifact_bytes}
        )
    assert context.store.load_package(context.package_digest) == context.package
    context.store.index_path.unlink(missing_ok=True)
    assert context.store.write_package(
        envelope, {"artifacts/volatility.bin": context.artifact_bytes}
    ) == context.package_digest
    assert context.store.index_path.exists()


def test_disposition_retry_repairs_projection_after_committed_stale_result(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    context = build_store(tmp_path)
    created = signed_event(
        context,
        sequence=0,
        previous_digest=None,
        from_state=None,
        to_state=DispositionState.CREATED,
        role=AuthorityRole.PRODUCER,
    )
    original = EvidenceStore._atomic_write_bytes.__func__
    failed = False

    def fail_index_once(cls, destination: Path, data: bytes, *, mode: int = 0o600) -> None:
        nonlocal failed
        if destination == context.store.index_path and not failed:
            failed = True
            raise OSError("injected disposition projection failure")
        original(cls, destination, data, mode=mode)

    monkeypatch.setattr(EvidenceStore, "_atomic_write_bytes", classmethod(fail_index_once))
    with pytest.raises(ProjectionStaleError, match="disposition committed"):
        context.store.append_disposition(created, expected_head_digest=None)
    assert len(context.store.load_ledger(context.package_digest)) == 1
    context.store.index_path.unlink(missing_ok=True)
    state = context.store.append_disposition(created, expected_head_digest=None)
    assert state.state == DispositionState.CREATED
    assert context.store.index_path.exists()


def test_second_same_lane_champion_is_rejected_before_publication(tmp_path: Path) -> None:
    first = build_store(tmp_path)
    advance_to_champion(first)

    second_artifact_bytes = b"immutable-risk-target-model-v2"
    second_artifact = ArtifactRef(
        artifact_id="volatility-head-v2",
        relative_path="artifacts/volatility-v2.bin",
        digest=sha256_bytes(second_artifact_bytes),
        size_bytes=len(second_artifact_bytes),
        media_type="application/octet-stream",
    )
    second_package = first.package.model_copy(
        update={
            "package_id": "risk-target-package-v2",
            "artifacts": (second_artifact,),
            "checksums": {second_artifact.relative_path: second_artifact.digest},
        }
    )
    producer = first.signers[AuthorityRole.PRODUCER]
    package_envelope = producer.sign(second_package, created_at=NOW)
    first.store.write_package(
        package_envelope, {second_artifact.relative_path: second_artifact_bytes}
    )
    second = StoreContext(
        store=first.store,
        signers=first.signers,
        actors=first.actors,
        package=second_package,
        package_digest=package_envelope.payload_digest,
        artifact_bytes=second_artifact_bytes,
    )
    verdict_digest = write_accepted_verdict(second)
    steps = [
        (None, DispositionState.CREATED, AuthorityRole.PRODUCER),
        (DispositionState.CREATED, DispositionState.RECEIVED, AuthorityRole.LOCAL_IMPORTER),
        (DispositionState.RECEIVED, DispositionState.HASH_VERIFIED, AuthorityRole.LOCAL_IMPORTER),
        (DispositionState.HASH_VERIFIED, DispositionState.POLICY_VERIFIED, AuthorityRole.LOCAL_IMPORTER),
        (DispositionState.POLICY_VERIFIED, DispositionState.METRIC_REPLAYED, AuthorityRole.INDEPENDENT_VERIFIER),
        (DispositionState.METRIC_REPLAYED, DispositionState.QUARANTINED, AuthorityRole.LOCAL_IMPORTER),
        (DispositionState.QUARANTINED, DispositionState.SHADOW, AuthorityRole.OPERATOR),
        (DispositionState.SHADOW, DispositionState.OPERATOR_APPROVED, AuthorityRole.OPERATOR),
    ]
    head = None
    for sequence, (from_state, to_state, role) in enumerate(steps):
        metadata = (
            {"verdict_digest": verdict_digest}
            if to_state == DispositionState.QUARANTINED
            else None
        )
        event = signed_event(
            second,
            sequence=sequence,
            previous_digest=head,
            from_state=from_state,
            to_state=to_state,
            role=role,
            metadata=metadata,
        )
        head = first.store.append_disposition(event, expected_head_digest=head).head_event_digest
    champion = signed_event(
        second,
        sequence=8,
        previous_digest=head,
        from_state=DispositionState.OPERATOR_APPROVED,
        to_state=DispositionState.CHAMPION,
        role=AuthorityRole.PROMOTION_SERVICE,
        metadata={
            "artifact_digest": second_artifact.digest,
            "artifact_id": second_artifact.artifact_id,
            "lane_id": second_package.lane_id,
        },
    )
    with pytest.raises(StoreCorruptionError, match="multiple champion packages"):
        first.store.append_disposition(champion, expected_head_digest=head)
    assert len(first.store.load_ledger(second.package_digest)) == 8
    first.store.rebuild_indexes()


def test_champion_pointer_resolves_exact_package_artifact_and_head(tmp_path: Path) -> None:
    context = build_store(tmp_path)
    head = advance_to_champion(context)
    artifact = context.package.artifacts[0]
    signer = context.signers[AuthorityRole.PROMOTION_SERVICE]
    pointer = ChampionPointer(
        pointer_id="risk-target-champion-v1",
        lane_id=context.package.lane_id,
        package_digest=context.package_digest,
        artifact_id=artifact.artifact_id,
        artifact_digest=artifact.digest,
        disposition_head_digest=head,
        actor_id=context.actors[AuthorityRole.PROMOTION_SERVICE],
        signer_key_id=signer.key_id,
        created_at=NOW + timedelta(minutes=1),
    )
    envelope = signer.sign(pointer, created_at=pointer.created_at)
    pointer_digest = context.store.write_champion_pointer(envelope, expected_pointer_digest=None)
    assert context.store.load_champion_pointer(context.package.lane_id) == pointer
    index = context.store.current_index_bytes().decode("utf-8")
    assert context.package_digest in index
    assert artifact.digest in index
    assert head in index

    wrong = pointer.model_copy(update={"artifact_digest": "f" * 64, "pointer_id": "wrong"})
    wrong_envelope = signer.sign(wrong, created_at=pointer.created_at)
    with pytest.raises(ChampionPointerError, match="does not resolve"):
        context.store.write_champion_pointer(
            wrong_envelope,
            expected_pointer_digest=pointer_digest,
        )


def test_destroyed_indexes_rebuild_to_byte_equivalent_state(tmp_path: Path) -> None:
    context = build_store(tmp_path)
    head = advance_to_champion(context)
    artifact = context.package.artifacts[0]
    signer = context.signers[AuthorityRole.PROMOTION_SERVICE]
    pointer = ChampionPointer(
        pointer_id="risk-target-champion-v1",
        lane_id=context.package.lane_id,
        package_digest=context.package_digest,
        artifact_id=artifact.artifact_id,
        artifact_digest=artifact.digest,
        disposition_head_digest=head,
        actor_id=context.actors[AuthorityRole.PROMOTION_SERVICE],
        signer_key_id=signer.key_id,
        created_at=NOW + timedelta(minutes=1),
    )
    context.store.write_champion_pointer(
        signer.sign(pointer, created_at=pointer.created_at),
        expected_pointer_digest=None,
    )
    before = context.store.current_index_bytes()
    shutil.rmtree(context.store.root / "indexes")
    rebuilt = context.store.rebuild_indexes()
    assert rebuilt == before
    assert context.store.current_index_bytes() == before
    assert not list((context.store.root / "indexes").glob(".*.tmp-*"))


def test_missing_authoritative_champion_pointer_is_not_advertised(tmp_path: Path) -> None:
    context = build_store(tmp_path)
    head = advance_to_champion(context)
    artifact = context.package.artifacts[0]
    signer = context.signers[AuthorityRole.PROMOTION_SERVICE]
    pointer = ChampionPointer(
        pointer_id="risk-target-champion-v1",
        lane_id=context.package.lane_id,
        package_digest=context.package_digest,
        artifact_id=artifact.artifact_id,
        artifact_digest=artifact.digest,
        disposition_head_digest=head,
        actor_id=context.actors[AuthorityRole.PROMOTION_SERVICE],
        signer_key_id=signer.key_id,
        created_at=NOW + timedelta(minutes=1),
    )
    context.store.write_champion_pointer(
        signer.sign(pointer, created_at=pointer.created_at), expected_pointer_digest=None
    )
    context.store._champion_path(context.package.lane_id).unlink()
    rebuilt = context.store.rebuild_indexes().decode("utf-8")
    assert '"champions":{}' in rebuilt


def test_removing_bound_import_verdict_breaks_promotion_replay(tmp_path: Path) -> None:
    context = build_store(tmp_path)
    advance_to_champion(context)
    verdicts = list((context.store.root / "verdicts").glob("*.json"))
    assert len(verdicts) == 1
    verdicts[0].unlink()
    with pytest.raises(EvidenceStoreError, match="accepted local import verdict"):
        context.store.rebuild_indexes()


def test_champion_pointer_retry_repairs_projection_after_commit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    context = build_store(tmp_path)
    head = advance_to_champion(context)
    artifact = context.package.artifacts[0]
    signer = context.signers[AuthorityRole.PROMOTION_SERVICE]
    pointer = ChampionPointer(
        pointer_id="risk-target-champion-v1",
        lane_id=context.package.lane_id,
        package_digest=context.package_digest,
        artifact_id=artifact.artifact_id,
        artifact_digest=artifact.digest,
        disposition_head_digest=head,
        actor_id=context.actors[AuthorityRole.PROMOTION_SERVICE],
        signer_key_id=signer.key_id,
        created_at=NOW + timedelta(minutes=1),
    )
    envelope = signer.sign(pointer, created_at=pointer.created_at)
    original = EvidenceStore._atomic_write_bytes.__func__
    failed = False

    def fail_index_once(cls, destination: Path, data: bytes, *, mode: int = 0o600) -> None:
        nonlocal failed
        if destination == context.store.index_path and not failed:
            failed = True
            raise OSError("injected pointer projection failure")
        original(cls, destination, data, mode=mode)

    monkeypatch.setattr(EvidenceStore, "_atomic_write_bytes", classmethod(fail_index_once))
    with pytest.raises(ProjectionStaleError, match="champion pointer committed"):
        context.store.write_champion_pointer(envelope, expected_pointer_digest=None)
    assert context.store.load_champion_pointer(context.package.lane_id) == pointer
    context.store.index_path.unlink(missing_ok=True)
    assert context.store.write_champion_pointer(
        envelope, expected_pointer_digest=None
    ) == envelope.payload_digest
    assert context.store.index_path.exists()


def test_first_ledger_creation_fsyncs_ledger_and_parent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    context = build_store(tmp_path)
    calls: list[Path] = []
    original = EvidenceStore._fsync_directory

    def record(path: Path) -> None:
        calls.append(path)
        original(path)

    monkeypatch.setattr(EvidenceStore, "_fsync_directory", staticmethod(record))
    created = signed_event(
        context,
        sequence=0,
        previous_digest=None,
        from_state=None,
        to_state=DispositionState.CREATED,
        role=AuthorityRole.PRODUCER,
    )
    context.store.append_disposition(created, expected_head_digest=None)
    ledger = context.store.root / "dispositions" / context.package_digest
    assert ledger in calls
    assert ledger.parent in calls


def test_failed_atomic_index_replace_preserves_previous_index(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    context = build_store(tmp_path)
    before = context.store.current_index_bytes()
    original_replace = os.replace

    def fail_index_replace(source, destination) -> None:
        if Path(destination) == context.store.index_path:
            raise OSError("injected index replace failure")
        original_replace(source, destination)

    monkeypatch.setattr(os, "replace", fail_index_replace)
    with pytest.raises(OSError, match="injected index"):
        context.store.rebuild_indexes()
    assert context.store.index_path.read_bytes() == before
    assert not list((context.store.root / "indexes").glob(".*.tmp-*"))


def test_artifact_corruption_blocks_index_reconstruction(tmp_path: Path) -> None:
    context = build_store(tmp_path)
    artifact_path = (
        context.store.root
        / "packages"
        / context.package_digest
        / context.package.artifacts[0].relative_path
    )
    artifact_path.write_bytes(b"tampered-risk-target-model-v1")
    shutil.rmtree(context.store.root / "indexes")
    with pytest.raises(StoreCorruptionError, match="stored artifact mismatch"):
        context.store.rebuild_indexes()
