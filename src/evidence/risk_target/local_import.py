"""Local evidence authority for the risk-target slice.

The local importer receives a producer-signed package, writes it into the
immutable store, and drives the signed disposition chain:

    CREATED (producer)
    -> RECEIVED           (local importer)
    -> HASH_VERIFIED      (local importer)      : artifact + lineage hashes
    -> POLICY_VERIFIED    (local importer)      : capability + safety policy
    -> METRIC_REPLAYED    (independent verifier): reproduce metrics + verdict
    -> QUARANTINED        (local importer)       : accepted for quarantine
       or REJECTED        (local importer)       : any failed check / failed head

It reproduces the producer's metrics itself before accepting anything, so the
producing process is never the sole approver (roadmap §4.2). A candidate never
overwrites an incumbent: the chain stops at QUARANTINED; champion promotion is
a separate, operator-authorized transition this module never performs.

The verification path takes no registry/network dependency — an external
registry outage cannot stop a local verdict.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timedelta
from decimal import Decimal, ROUND_HALF_UP
from typing import Mapping

from src.evidence.canonical import canonical_bytes
from src.evidence.contracts import (
    AuthorityRole,
    CapabilityProfile,
    DatasetManifest,
    DispositionEvent,
    DispositionState,
    EvaluationReport,
    ImportCheck,
    JobManifest,
    LocalImportVerdict,
    SignedEnvelope,
)
from src.evidence.hashing import content_digest, sha256_bytes
from src.evidence.importer import build_import_verdict
from src.evidence.signing import Ed25519Signer, TrustStore, verify_envelope
from src.evidence.store import EvidenceStore

from .evaluation import EvaluationParams, evaluate_partitions
from .models import (
    CAPABILITY_PROFILE_PATH,
    DATASET_MANIFEST_PATH,
    EVALUATION_REPORT_PATH,
    JOB_MANIFEST_PATH,
    HeadImportOutcome,
    PackagedHead,
)

logger = logging.getLogger(__name__)

Evaluator = Callable[[Mapping[str, bytes], EvaluationParams], "tuple"]


@dataclass(frozen=True)
class _StageChecks:
    hash_checks: tuple[ImportCheck, ...]
    policy_checks: tuple[ImportCheck, ...]
    replay_checks: tuple[ImportCheck, ...]
    gate_check: ImportCheck

    def all(self) -> tuple[ImportCheck, ...]:
        return self.hash_checks + self.policy_checks + self.replay_checks + (self.gate_check,)


def _check(check_id: str, passed: bool, details: str) -> ImportCheck:
    return ImportCheck(check_id=check_id, passed=passed, details=details)


def _first_failure(checks: tuple[ImportCheck, ...], fallback: str) -> str:
    for check in checks:
        if not check.passed:
            return f"{check.check_id}: {check.details}"
    return fallback


def _finite_float_candidate(value: object) -> float:
    """Convert JSON numeric metadata without letting oversized ints abort import."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return math.nan
    try:
        return float(value)
    except (OverflowError, TypeError, ValueError):
        return math.nan


def _partitions_match_manifest(
    dataset_manifest: DatasetManifest, partitions: Mapping[str, bytes]
) -> bool:
    declared = {ref.partition_id: ref for ref in dataset_manifest.partitions}
    if set(partitions) != set(declared):
        return False
    for partition_id, data in partitions.items():
        ref = declared[partition_id]
        if len(data) != ref.size_bytes or sha256_bytes(data) != ref.digest:
            return False
    return True


def _hash_checks(
    head: PackagedHead,
    job: JobManifest,
    job_envelope: SignedEnvelope,
    report: EvaluationReport,
    capability_profile_envelope: SignedEnvelope,
    dataset_manifest_envelope: SignedEnvelope,
    dataset_manifest: DatasetManifest,
    partitions: Mapping[str, bytes],
) -> tuple[ImportCheck, ...]:
    package = head.package
    artifact = package.artifacts[0]
    artifact_ok = (
        sha256_bytes(head.files[artifact.relative_path]) == artifact.digest
        and head.package_envelope.payload_digest == content_digest(package)
    )
    # Bind the exact evaluation report to the signed package. verify_envelope
    # authenticates the report's signature, but without this the package's
    # declared evaluation_report_digests are dead on the import side — a
    # differently-but-validly-signed report (with swapped audit fields the
    # metric replay does not cover: holdout, embargo, effective_n, evaluator)
    # could pass unnoticed. Both are content_digest-derived, so the check is
    # exact. (Code review F1, 2026-07-13.)
    report_bound = (
        head.evaluation_report_envelope.payload_digest in package.evaluation_report_digests
    )
    signed_lineage = {
        JOB_MANIFEST_PATH: job_envelope,
        DATASET_MANIFEST_PATH: dataset_manifest_envelope,
        CAPABILITY_PROFILE_PATH: capability_profile_envelope,
        EVALUATION_REPORT_PATH: head.evaluation_report_envelope,
    }
    signed_lineage_durable = all(
        head.files.get(relative_path) == canonical_bytes(envelope)
        for relative_path, envelope in signed_lineage.items()
    )
    lineage_ok = (
        tuple(package.dataset_manifest_digests) == tuple(job.dataset_manifest_digests)
        and report.job_manifest_digest == package.job_manifest_digest
        and content_digest(job) == package.job_manifest_digest
        and content_digest(dataset_manifest) in package.dataset_manifest_digests
    )
    # Independently re-verify the declared partitions on the import side too
    # (the producer already checked them; symmetry closes the loop for an
    # importer that does not fully re-run training). (Code review F4.)
    partitions_ok = _partitions_match_manifest(dataset_manifest, partitions)
    return (
        _check("artifact_hashes_match", artifact_ok, "artifact bytes match declared digest"),
        _check("evaluation_report_bound_to_package", report_bound,
               "evaluation report digest is bound in the package"),
        _check("signed_lineage_objects_persisted", signed_lineage_durable,
               "signed job, dataset, capability and evaluation envelopes are immutable package members"),
        _check("dataset_lineage_matches_job", lineage_ok, "package binds the signed job's dataset lineage"),
        _check("dataset_partitions_match_manifest", partitions_ok,
               "supplied partitions match the signed dataset manifest hashes"),
    )


def _policy_checks(
    head: PackagedHead,
    job: JobManifest,
    capability_profile: CapabilityProfile,
    report: EvaluationReport,
) -> tuple[ImportCheck, ...]:
    package = head.package
    cap_ok = (
        content_digest(capability_profile) == job.capability_profile_digest
        and capability_profile.may_read_broker_credentials is False
        and capability_profile.may_place_or_cancel_orders is False
        and capability_profile.may_read_operator_keys is False
        and capability_profile.may_change_halts is False
        and capability_profile.may_change_live_gate is False
        and capability_profile.may_write_champion_pointer is False
        and capability_profile.may_modify_local_models is False
        and capability_profile.may_approve_evidence is False
        and capability_profile.network_endpoints == ()
    )
    safety_ok = bool(package.safety_assertions) and all(a.passed for a in package.safety_assertions)
    completeness_ok = (
        len(report.gates) >= 1 and report.trial_count >= 1 and report.effective_sample_size > 0
    )
    cost = report.cost
    usage = report.resource_usage
    amount = cost.get("amount")
    rate = cost.get("rate_per_hour")
    wall_seconds = usage.get("wall_seconds")
    numeric_amount = _finite_float_candidate(amount)
    numeric_rate = _finite_float_candidate(rate)
    numeric_wall_seconds = _finite_float_candidate(wall_seconds)
    head_count = usage.get("head_count")
    expected_amount = None
    if (
        math.isfinite(numeric_rate)
        and numeric_rate >= 0.0
        and math.isfinite(numeric_wall_seconds)
        and numeric_wall_seconds >= 0.0
    ):
        expected_amount = float(
            (
                Decimal(str(numeric_wall_seconds))
                * Decimal(str(numeric_rate))
                / Decimal("3600")
            ).quantize(Decimal("0.000001"), rounding=ROUND_HALF_UP)
        )
    expected_basis = (
        "local_unmetered_no_incremental_provider_charge"
        if numeric_rate == 0.0
        else "configured_local_cpu_hourly_rate"
    )
    cost_ok = (
        math.isfinite(numeric_amount)
        and numeric_amount >= 0.0
        and math.isfinite(numeric_rate)
        and numeric_rate >= 0.0
        and math.isfinite(numeric_wall_seconds)
        and numeric_wall_seconds >= 0.0
        and expected_amount is not None
        and numeric_amount == expected_amount
        and cost.get("currency") == "USD"
        and cost.get("basis") == expected_basis
        and cost.get("scope") == "producer_evaluation"
        and cost.get("allocation") == "equal_share_of_job_evaluator_wall_time"
        and usage.get("scope") == "producer_evaluation"
        and usage.get("allocation") == "equal_share_of_job_evaluator_wall_time"
        and isinstance(head_count, int)
        and not isinstance(head_count, bool)
        and head_count == len(job.expected_outputs)
    )
    return (
        _check("no_forbidden_capabilities", cap_ok, "worker capability profile grants no authority"),
        _check("safety_assertions_present_and_passed", safety_ok, "all package safety assertions passed"),
        _check("evaluation_completeness", completeness_ok, "report has gates, a trial count and a holdout"),
        _check(
            "evaluation_cost_accounted",
            cost_ok,
            "signed report records allocated producer compute and USD cost basis",
        ),
    )


def _replay_checks(
    head: PackagedHead,
    report: EvaluationReport,
    partitions: Mapping[str, bytes],
    params: EvaluationParams,
    evaluator: Evaluator,
) -> tuple[ImportCheck, ...]:
    """Independently reproduce the producer's metrics and gate verdict."""
    try:
        reproduced = {r.head_id: r for r in evaluator(partitions, params)}
    except Exception as exc:  # noqa: BLE001 - a replay failure is a verdict, not a crash
        return (_check("metric_replay_reproduces", False, f"replay error: {exc}"),)

    result = reproduced.get(head.head_id)
    if result is None:
        return (_check("metric_replay_reproduces", False, f"replay produced no {head.head_id} head"),)

    mismatches: list[str] = []
    for name, metric in report.metrics.items():
        if name not in result.metrics:
            mismatches.append(f"{name}:absent")
            continue
        tolerance = metric.tolerance if metric.tolerance is not None else params.replay_tolerance
        if abs(float(result.metrics[name]) - metric.value) > tolerance:
            mismatches.append(f"{name}:{result.metrics[name]}!={metric.value}")
    metric_ok = not mismatches

    reproduced_gate = {g.gate_id: g.status for g in result.gates}
    gate_ok = bool(result.passed) == bool(report.passed) and all(
        reproduced_gate.get(g.gate_id) == g.status for g in report.gates
    )

    return (
        _check("metric_replay_reproduces", metric_ok,
               "metrics reproduce within declared tolerance" if metric_ok else "; ".join(mismatches)),
        _check("gate_verdict_reproduces", gate_ok, "reproduced gate verdict matches the producer's"),
    )


class _Chain:
    """Drives one package's disposition chain, tracking head + current state."""

    def __init__(self, store: EvidenceStore, package_digest: str, base_time: datetime) -> None:
        self._store = store
        self._package_digest = package_digest
        self._base_time = base_time
        self.head_digest: str | None = None
        self.current_state: DispositionState | None = None
        self._sequence = 0

    def append(
        self,
        *,
        from_state: DispositionState | None,
        to_state: DispositionState,
        signer: Ed25519Signer,
        actor_id: str,
        role: AuthorityRole,
        reason: str,
        metadata: dict | None = None,
    ) -> None:
        occurred_at = self._base_time + timedelta(seconds=self._sequence)
        event = DispositionEvent(
            package_digest=self._package_digest,
            sequence=self._sequence,
            previous_event_digest=self.head_digest,
            from_state=from_state,
            to_state=to_state,
            authority=role,
            actor_id=actor_id,
            signer_key_id=signer.key_id,
            occurred_at=occurred_at,
            reason=reason,
            metadata=metadata or {},
        )
        state = self._store.append_disposition(
            signer.sign(event, created_at=occurred_at), expected_head_digest=self.head_digest
        )
        self.head_digest = state.head_event_digest
        self.current_state = to_state
        self._sequence += 1

    def append_created_from_producer(self, envelope) -> None:
        state = self._store.append_disposition(envelope, expected_head_digest=None)
        self.head_digest = state.head_event_digest
        self.current_state = DispositionState.CREATED
        self._sequence = 1


def import_head(
    store: EvidenceStore,
    head: PackagedHead,
    *,
    job_envelope: SignedEnvelope,
    capability_profile_envelope: SignedEnvelope,
    dataset_manifest_envelope: SignedEnvelope,
    partitions: Mapping[str, bytes],
    trust_store: TrustStore,
    importer: Ed25519Signer,
    importer_id: str,
    verifier: Ed25519Signer,
    verifier_id: str,
    created_at: datetime,
    params: EvaluationParams | None = None,
    evaluator: Evaluator = evaluate_partitions,
) -> HeadImportOutcome:
    """Import one head package to QUARANTINED or REJECTED.

    Writes the immutable package, runs staged verification (hash, policy,
    independent metric replay), builds the signed ``LocalImportVerdict``, and
    drives the disposition chain to its terminal state. Raises only if the
    package is structurally inadmissible (tampered bytes / bad signature) — a
    package that is admissible but fails a check is REJECTED, not raised.

    The capability profile and dataset manifest are reconstructed from their
    signed envelopes here (not trusted as caller-supplied objects), so the
    local authority verifies the producer's exact declared capabilities and
    data lineage rather than relying on the orchestrator to pass honest
    objects. (Security review C + code review F4, 2026-07-13.)
    """
    params = params or EvaluationParams()
    package_digest = store.write_package(head.package_envelope, dict(head.files))

    job = verify_envelope(job_envelope, JobManifest, trust_store)
    assert isinstance(job, JobManifest)
    report = verify_envelope(head.evaluation_report_envelope, EvaluationReport, trust_store)
    assert isinstance(report, EvaluationReport)
    capability_profile = verify_envelope(capability_profile_envelope, CapabilityProfile, trust_store)
    assert isinstance(capability_profile, CapabilityProfile)
    dataset_manifest = verify_envelope(dataset_manifest_envelope, DatasetManifest, trust_store)
    assert isinstance(dataset_manifest, DatasetManifest)

    checks = _StageChecks(
        hash_checks=_hash_checks(
            head,
            job,
            job_envelope,
            report,
            capability_profile_envelope,
            dataset_manifest_envelope,
            dataset_manifest,
            partitions,
        ),
        policy_checks=_policy_checks(head, job, capability_profile, report),
        replay_checks=_replay_checks(head, report, partitions, params, evaluator),
        gate_check=_check(
            "candidate_evaluation_gate_passed",
            bool(report.passed),
            "head cleared its own OOS admissibility gate",
        ),
    )

    verdict = build_import_verdict(
        verdict_id=f"{head.head_id}-verdict",
        package_digest=package_digest,
        importer_id=importer_id,
        created_at=created_at,
        checks=checks.all(),
    )
    verdict_digest = store.write_verdict(importer.sign(verdict, created_at=created_at))

    chain = _Chain(store, package_digest, created_at)
    chain.append_created_from_producer(head.created_event_envelope)
    chain.append(
        from_state=DispositionState.CREATED, to_state=DispositionState.RECEIVED,
        signer=importer, actor_id=importer_id, role=AuthorityRole.LOCAL_IMPORTER,
        reason="received remote evidence",
    )

    def reject(reason: str) -> HeadImportOutcome:
        chain.append(
            from_state=chain.current_state, to_state=DispositionState.REJECTED,
            signer=importer, actor_id=importer_id, role=AuthorityRole.LOCAL_IMPORTER,
            reason=reason,
        )
        return _outcome(head, package_digest, verdict, DispositionState.REJECTED, reason, chain)

    if not all(c.passed for c in checks.hash_checks):
        return reject(_first_failure(checks.hash_checks, "hash verification failed"))
    chain.append(
        from_state=DispositionState.RECEIVED, to_state=DispositionState.HASH_VERIFIED,
        signer=importer, actor_id=importer_id, role=AuthorityRole.LOCAL_IMPORTER,
        reason="artifact and lineage hashes verified",
    )

    if not all(c.passed for c in checks.policy_checks):
        return reject(_first_failure(checks.policy_checks, "policy verification failed"))
    chain.append(
        from_state=DispositionState.HASH_VERIFIED, to_state=DispositionState.POLICY_VERIFIED,
        signer=importer, actor_id=importer_id, role=AuthorityRole.LOCAL_IMPORTER,
        reason="capability and safety policy verified",
    )

    if not all(c.passed for c in checks.replay_checks):
        return reject(_first_failure(checks.replay_checks, "metric replay failed"))
    chain.append(
        from_state=DispositionState.POLICY_VERIFIED, to_state=DispositionState.METRIC_REPLAYED,
        signer=verifier, actor_id=verifier_id, role=AuthorityRole.INDEPENDENT_VERIFIER,
        reason="independent metric replay reproduced the producer's numbers",
    )

    if not checks.gate_check.passed:
        return reject("candidate head failed its own OOS admissibility gate")

    chain.append(
        from_state=DispositionState.METRIC_REPLAYED, to_state=DispositionState.QUARANTINED,
        signer=importer, actor_id=importer_id, role=AuthorityRole.LOCAL_IMPORTER,
        reason="accepted for quarantine", metadata={"verdict_digest": verdict_digest},
    )
    return _outcome(
        head, package_digest, verdict, DispositionState.QUARANTINED,
        "accepted for quarantine", chain,
    )


def _outcome(
    head: PackagedHead,
    package_digest: str,
    verdict: LocalImportVerdict,
    final_state: DispositionState,
    reason: str,
    chain: _Chain,
) -> HeadImportOutcome:
    assert chain.head_digest is not None
    return HeadImportOutcome(
        lane_id=head.lane_id,
        package_digest=package_digest,
        verdict=verdict,
        final_state=final_state,
        reason=reason,
        disposition_head_digest=chain.head_digest,
    )


__all__ = ["import_head", "HeadImportOutcome"]
