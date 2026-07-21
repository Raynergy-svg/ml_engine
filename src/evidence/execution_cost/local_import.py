"""Local independent verification for execution-cost evidence."""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Mapping

from src.evidence.contracts import (
    AuthorityRole, CapabilityProfile, DatasetManifest, DispositionEvent,
    DispositionState, EvaluationReport, ImportCheck, JobManifest,
)
from src.evidence.hashing import content_digest, sha256_bytes
from src.evidence.importer import build_import_verdict
from src.evidence.signing import Ed25519Signer, TrustStore, verify_envelope
from src.evidence.store import EvidenceStore

from .evaluation import EvaluationParams, evaluate_partitions
from .manifests import METRICS
from .models import ImportOutcome, PackagedModel


def _check(check_id: str, passed: bool, details: str) -> ImportCheck:
    return ImportCheck(check_id=check_id, passed=passed, details=details)


def _authority_free(profile: CapabilityProfile) -> bool:
    return profile.network_endpoints == () and all(
        getattr(profile, field) is False
        for field in (
            "may_read_broker_credentials", "may_place_or_cancel_orders",
            "may_read_operator_keys", "may_change_halts", "may_change_live_gate",
            "may_write_champion_pointer", "may_modify_local_models", "may_approve_evidence",
        )
    )


def _preflight(
    head: PackagedModel, *, job: JobManifest, capability: CapabilityProfile,
    dataset: DatasetManifest, partitions: Mapping[str, bytes], report: EvaluationReport,
    params: EvaluationParams,
) -> tuple[ImportCheck, ...]:
    package = head.package
    artifact = package.artifacts[0]
    supplied = head.files.get(artifact.relative_path)
    artifact_ok = supplied is not None and len(supplied) == artifact.size_bytes and sha256_bytes(supplied) == artifact.digest
    declared = {ref.partition_id: ref for ref in dataset.partitions}
    dataset_ok = set(partitions) == set(declared) and all(
        len(data) == declared[name].size_bytes and sha256_bytes(data) == declared[name].digest
        for name, data in partitions.items()
    )
    lineage_ok = (
        head.package_envelope.payload_digest == content_digest(package)
        and head.package_digest == head.package_envelope.payload_digest
        and package.job_manifest_digest == content_digest(job)
        and tuple(package.dataset_manifest_digests) == tuple(job.dataset_manifest_digests)
        and package.evaluation_report_digests == (head.evaluation_report_envelope.payload_digest,)
        and report.job_manifest_digest == content_digest(job)
        and job.capability_profile_digest == content_digest(capability)
        and content_digest(dataset) in job.dataset_manifest_digests
    )
    metric_surface_ok = set(report.metrics) == set(METRICS)
    try:
        replay = evaluate_partitions(partitions, params)
    except Exception as exc:  # replay failure is evidence, not an importer crash
        return (
            _check("artifact_hash_verified", artifact_ok, "artifact bytes match the signed reference"),
            _check("dataset_hashes_verified", dataset_ok, "all immutable partition bytes match the signed manifest"),
            _check("lineage_verified", lineage_ok, "job, dataset, capability and report digests are bound"),
            _check("no_forbidden_capabilities", _authority_free(capability), "worker declares no network, execution or promotion authority"),
            _check("metric_surface_complete", metric_surface_ok, "report contains exactly the frozen metric surface"),
            _check("metric_replay_reproduces", False, f"independent replay error: {exc}"),
            _check("artifact_reproduces", False, "independent replay did not complete"),
            _check("gate_verdict_reproduces", False, "independent replay did not complete"),
            _check("candidate_evaluation_gate_passed", False, "independent replay did not complete"),
        )
    mismatches = []
    for name in METRICS:
        if name not in report.metrics or name not in replay.metrics:
            mismatches.append(f"{name}:absent")
            continue
        tolerance = report.metrics[name].tolerance
        bound = params.replay_tolerance if tolerance is None else tolerance
        if abs(report.metrics[name].value - float(replay.metrics[name])) > bound:
            mismatches.append(name)
    replay_gate = {gate.gate_id: gate.status for gate in replay.gates}
    gate_ok = report.passed == replay.passed and len(report.gates) == len(replay.gates) and all(
        replay_gate.get(gate.gate_id) == gate.status for gate in report.gates
    )
    return (
        _check("artifact_hash_verified", artifact_ok, "artifact bytes match the signed reference"),
        _check("dataset_hashes_verified", dataset_ok, "all immutable partition bytes match the signed manifest"),
        _check("lineage_verified", lineage_ok, "job, dataset, capability and report digests are bound"),
        _check("no_forbidden_capabilities", _authority_free(capability), "worker declares no network, execution or promotion authority"),
        _check("metric_surface_complete", metric_surface_ok, "report contains exactly the frozen metric surface"),
        _check("metric_replay_reproduces", not mismatches, "metrics reproduce" if not mismatches else f"mismatched metrics: {mismatches}"),
        _check("artifact_reproduces", supplied == replay.artifact_bytes, "JSON model artifact reproduces byte-for-byte"),
        _check("gate_verdict_reproduces", gate_ok, "independent replay reproduces all gates and verdict"),
        _check("candidate_evaluation_gate_passed", report.passed, "candidate clears its frozen chronological OOS gates"),
    )


class _Chain:
    def __init__(self, store: EvidenceStore, package_digest: str, created_at: datetime) -> None:
        self.store = store
        self.package_digest = package_digest
        self.created_at = created_at
        self.head: str | None = None
        self.state: DispositionState | None = None
        self.sequence = 0

    def created(self, envelope) -> None:
        state = self.store.append_disposition(envelope, expected_head_digest=None)
        self.head = state.head_event_digest
        self.state = DispositionState.CREATED
        self.sequence = 1

    def append(self, target: DispositionState, signer: Ed25519Signer, actor: str, role: AuthorityRole, reason: str, metadata: dict | None = None) -> None:
        when = self.created_at + timedelta(seconds=self.sequence)
        event = DispositionEvent(
            package_digest=self.package_digest, sequence=self.sequence,
            previous_event_digest=self.head, from_state=self.state, to_state=target,
            authority=role, actor_id=actor, signer_key_id=signer.key_id,
            occurred_at=when, reason=reason, metadata=metadata or {},
        )
        state = self.store.append_disposition(signer.sign(event, created_at=when), expected_head_digest=self.head)
        self.head = state.head_event_digest
        self.state = target
        self.sequence += 1


def import_model(
    store: EvidenceStore, head: PackagedModel, *, job_envelope,
    capability_profile_envelope, dataset_manifest_envelope,
    partitions: Mapping[str, bytes], trust_store: TrustStore,
    importer: Ed25519Signer, importer_id: str,
    verifier: Ed25519Signer, verifier_id: str,
    created_at: datetime, params: EvaluationParams | None = None,
) -> ImportOutcome:
    params = params or EvaluationParams()
    # Verify every supplied envelope and all replayable bytes before any durable
    # package/event write. Malformed or forged evidence cannot poison the store.
    job = verify_envelope(job_envelope, JobManifest, trust_store)
    capability = verify_envelope(capability_profile_envelope, CapabilityProfile, trust_store)
    dataset = verify_envelope(dataset_manifest_envelope, DatasetManifest, trust_store)
    report = verify_envelope(head.evaluation_report_envelope, EvaluationReport, trust_store)
    verify_envelope(head.package_envelope, type(head.package), trust_store)
    if not all(isinstance(value, expected) for value, expected in (
        (job, JobManifest), (capability, CapabilityProfile),
        (dataset, DatasetManifest), (report, EvaluationReport),
    )):
        raise ValueError("signed envelope payload type mismatch")
    checks = _preflight(head, job=job, capability=capability, dataset=dataset,
                        partitions=partitions, report=report, params=params)
    package_digest = store.write_package(head.package_envelope, dict(head.files))
    verdict = build_import_verdict(
        verdict_id="execution-cost-model-verdict", package_digest=package_digest,
        importer_id=importer_id, created_at=created_at, checks=checks,
    )
    verdict_digest = store.write_verdict(importer.sign(verdict, created_at=created_at))
    chain = _Chain(store, package_digest, created_at)
    chain.created(head.created_event_envelope)
    chain.append(DispositionState.RECEIVED, importer, importer_id, AuthorityRole.LOCAL_IMPORTER, "received execution-cost evidence")
    terminal = DispositionState.REJECTED
    reason = next((f"{check.check_id}: {check.details}" for check in checks if not check.passed), "verification failed")
    groups = (
        ({"artifact_hash_verified", "dataset_hashes_verified", "lineage_verified"}, DispositionState.HASH_VERIFIED, "artifact and lineage hashes verified"),
        ({"no_forbidden_capabilities", "metric_surface_complete"}, DispositionState.POLICY_VERIFIED, "capability and frozen metric policy verified"),
        ({"metric_replay_reproduces", "artifact_reproduces", "gate_verdict_reproduces"}, DispositionState.METRIC_REPLAYED, "independent replay reproduced model, metrics and gates"),
    )
    by_id = {check.check_id: check for check in checks}
    for ids, state, success_reason in groups:
        if not all(by_id[name].passed for name in ids):
            chain.append(DispositionState.REJECTED, importer, importer_id, AuthorityRole.LOCAL_IMPORTER, reason)
            break
        role = AuthorityRole.INDEPENDENT_VERIFIER if state == DispositionState.METRIC_REPLAYED else AuthorityRole.LOCAL_IMPORTER
        signer, actor = (verifier, verifier_id) if role == AuthorityRole.INDEPENDENT_VERIFIER else (importer, importer_id)
        chain.append(state, signer, actor, role, success_reason)
    else:
        if by_id["candidate_evaluation_gate_passed"].passed:
            terminal = DispositionState.QUARANTINED
            reason = "accepted for quarantine; no installation or promotion authority granted"
            chain.append(terminal, importer, importer_id, AuthorityRole.LOCAL_IMPORTER, reason, {"verdict_digest": verdict_digest})
        else:
            chain.append(DispositionState.REJECTED, importer, importer_id, AuthorityRole.LOCAL_IMPORTER, reason)
    if chain.state == DispositionState.REJECTED:
        terminal = DispositionState.REJECTED
    assert chain.head is not None
    return ImportOutcome(package_digest=package_digest, verdict=verdict, final_state=terminal,
                         reason=reason, disposition_head_digest=chain.head)


__all__ = ["import_model"]
