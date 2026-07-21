"""Local evidence authority for the hedge-evaluation slice.

The local importer receives a producer-signed strategy package, writes it into
the immutable store, and drives the signed disposition chain:

    CREATED (producer)
    -> RECEIVED           (local importer)
    -> HASH_VERIFIED      (local importer)      : artifact + lineage hashes
    -> POLICY_VERIFIED    (local importer)      : capability + safety policy
    -> METRIC_REPLAYED    (independent verifier): reproduce metrics + verdict
    -> QUARANTINED        (local importer)       : accepted for quarantine
       or REJECTED        (local importer)       : any failed check / failed head

It reproduces the producer's scorecard itself before accepting anything, so the
producing process is never the sole approver (roadmap §4.2). A candidate never
overwrites an incumbent: the chain stops at QUARANTINED; champion promotion is
a separate, operator-authorized transition this module never performs.

The verification path takes no registry/network dependency — an external
registry outage cannot stop a local verdict.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Mapping

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
    StrategyManifest,
)
from src.evidence.hashing import content_digest, sha256_bytes
from src.evidence.importer import build_import_verdict
from src.evidence.signing import Ed25519Signer, TrustStore, verify_envelope
from src.evidence.store import EvidenceStore

from .evaluation import EvaluationParams, evaluate_partitions
from .models import HedgeImportOutcome, PackagedHedgeHead

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
    head: PackagedHedgeHead,
    job: JobManifest,
    report: EvaluationReport,
    dataset_manifest: DatasetManifest,
    strategy_manifest: StrategyManifest,
    partitions: Mapping[str, bytes],
) -> tuple[ImportCheck, ...]:
    package = head.package
    artifact = package.artifacts[0]
    artifact_ok = (
        sha256_bytes(head.files[artifact.relative_path]) == artifact.digest
        and head.package_envelope.payload_digest == content_digest(package)
    )
    # Bind the exact evaluation report to the signed package (mirrors risk-target
    # code review F1): verify_envelope authenticates the report signature, but
    # this closes the gap where a differently-but-validly-signed report with
    # swapped audit fields could slip past.
    report_bound = (
        head.evaluation_report_envelope.payload_digest in package.evaluation_report_digests
    )
    lineage_ok = (
        tuple(package.dataset_manifest_digests) == tuple(job.dataset_manifest_digests)
        and report.job_manifest_digest == package.job_manifest_digest
        and content_digest(job) == package.job_manifest_digest
        and content_digest(dataset_manifest) in package.dataset_manifest_digests
    )
    # The strategy manifest is the hedge slice's workload spec; bind it exactly
    # (the risk-target slice binds a ModelTrainingSpec inside the job instead).
    strategy_ok = (
        package.strategy_manifest_digest == job.strategy_manifest_digest
        and package.strategy_manifest_digest == content_digest(strategy_manifest)
    )
    partitions_ok = _partitions_match_manifest(dataset_manifest, partitions)
    return (
        _check("artifact_hashes_match", artifact_ok, "artifact bytes match declared digest"),
        _check("evaluation_report_bound_to_package", report_bound,
               "evaluation report digest is bound in the package"),
        _check("dataset_lineage_matches_job", lineage_ok, "package binds the signed job's dataset lineage"),
        _check("strategy_manifest_bound_to_job", strategy_ok,
               "package binds the signed job's strategy manifest"),
        _check("dataset_partitions_match_manifest", partitions_ok,
               "supplied partitions match the signed dataset manifest hashes"),
    )


def _policy_checks(
    head: PackagedHedgeHead,
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
    return (
        _check("no_forbidden_capabilities", cap_ok, "worker capability profile grants no authority"),
        _check("safety_assertions_present_and_passed", safety_ok, "all package safety assertions passed"),
        _check("evaluation_completeness", completeness_ok, "report has gates, a trial count and a holdout"),
    )


def _replay_checks(
    head: PackagedHedgeHead,
    report: EvaluationReport,
    partitions: Mapping[str, bytes],
    params: EvaluationParams,
    evaluator: Evaluator,
) -> tuple[ImportCheck, ...]:
    """Independently reproduce the producer's scorecard metrics and gate verdict."""
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
        # Ignore the producer-declared tolerance on the import side — a local
        # authority must not let the producer widen its own reproduction bar
        # (MetricValue.tolerance has no upper bound). Use the local params bar.
        if abs(float(result.metrics[name]) - metric.value) > params.replay_tolerance:
            mismatches.append(f"{name}:{result.metrics[name]}!={metric.value}")
    metric_ok = not mismatches

    reproduced_gate = {g.gate_id: g.status for g in result.gates}
    gate_ok = bool(result.passed) == bool(report.passed) and all(
        reproduced_gate.get(g.gate_id) == g.status for g in report.gates
    )

    # The hedge artifact IS the deterministic Canonical-JSON scorecard we just
    # reproduced (unlike risk-target's opaque model blob), so bind the reproduced
    # bytes to the stored, signed artifact digest. Catches a producer that signs
    # an honest EvaluationReport but packages a doctored scorecard.json — the
    # metrics/gates reproduce, yet the artifact a human/dashboard reads would lie.
    artifact = head.package.artifacts[0]
    artifact_ok = sha256_bytes(result.artifact_bytes) == artifact.digest

    return (
        _check("metric_replay_reproduces", metric_ok,
               "metrics reproduce within declared tolerance" if metric_ok else "; ".join(mismatches)),
        _check("gate_verdict_reproduces", gate_ok, "reproduced gate verdict matches the producer's"),
        _check("artifact_reproduces", artifact_ok,
               "reproduced scorecard bytes match the stored artifact digest"),
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
    head: PackagedHedgeHead,
    *,
    job: JobManifest,
    capability_profile_envelope,
    dataset_manifest_envelope,
    strategy_manifest_envelope,
    partitions: Mapping[str, bytes],
    trust_store: TrustStore,
    importer: Ed25519Signer,
    importer_id: str,
    verifier: Ed25519Signer,
    verifier_id: str,
    created_at: datetime,
    params: EvaluationParams | None = None,
    evaluator: Evaluator = evaluate_partitions,
) -> HedgeImportOutcome:
    """Import one strategy head package to QUARANTINED or REJECTED.

    Writes the immutable package, runs staged verification (hash, policy,
    independent scorecard replay), builds the signed ``LocalImportVerdict``, and
    drives the disposition chain to its terminal state. Raises only if the
    package is structurally inadmissible (tampered bytes / bad signature) — a
    package that is admissible but fails a check is REJECTED, not raised.

    The capability profile, dataset manifest and strategy manifest are
    reconstructed from their signed envelopes here (not trusted as
    caller-supplied objects), so the local authority verifies the producer's
    exact declared capabilities, data lineage and workload spec rather than
    relying on the orchestrator to pass honest objects (mirrors risk-target
    security review C + code review F4).
    """
    params = params or EvaluationParams()
    package_digest = store.write_package(head.package_envelope, dict(head.files))

    report = verify_envelope(head.evaluation_report_envelope, EvaluationReport, trust_store)
    assert isinstance(report, EvaluationReport)
    capability_profile = verify_envelope(capability_profile_envelope, CapabilityProfile, trust_store)
    assert isinstance(capability_profile, CapabilityProfile)
    dataset_manifest = verify_envelope(dataset_manifest_envelope, DatasetManifest, trust_store)
    assert isinstance(dataset_manifest, DatasetManifest)
    strategy_manifest = verify_envelope(strategy_manifest_envelope, StrategyManifest, trust_store)
    assert isinstance(strategy_manifest, StrategyManifest)

    checks = _StageChecks(
        hash_checks=_hash_checks(head, job, report, dataset_manifest, strategy_manifest, partitions),
        policy_checks=_policy_checks(head, job, capability_profile, report),
        replay_checks=_replay_checks(head, report, partitions, params, evaluator),
        gate_check=_check(
            "candidate_evaluation_gate_passed",
            bool(report.passed),
            "strategy cleared its own after-cost admissibility gate",
        ),
    )

    verdict = build_import_verdict(
        verdict_id=f"hedge-{head.head_id}-verdict",
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
        reason="received remote hedge evidence",
    )

    def reject(reason: str) -> HedgeImportOutcome:
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
        reason="independent scorecard replay reproduced the producer's numbers",
    )

    if not checks.gate_check.passed:
        return reject("candidate strategy failed its own after-cost admissibility gate")

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
    head: PackagedHedgeHead,
    package_digest: str,
    verdict: LocalImportVerdict,
    final_state: DispositionState,
    reason: str,
    chain: _Chain,
) -> HedgeImportOutcome:
    assert chain.head_digest is not None
    return HedgeImportOutcome(
        lane_id=head.lane_id,
        strategy=head.head_id,
        package_digest=package_digest,
        verdict=verdict,
        final_state=final_state,
        reason=reason,
        disposition_head_digest=chain.head_digest,
    )


__all__ = ["import_head", "HedgeImportOutcome"]
