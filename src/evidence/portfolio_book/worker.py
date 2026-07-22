"""Producer-side portfolio-book worker with no live, execution or promotion authority."""

from __future__ import annotations

import logging
from collections.abc import Callable
from datetime import datetime
from typing import Mapping

from src.evidence.contracts import (
    ArtifactRef, AuthorityRole, CapabilityProfile, DatasetManifest,
    DispositionEvent, DispositionState, EvaluationReport, EvidencePackage,
    JobManifest, MetricValue, SafetyAssertion, SignedEnvelope, StrategyManifest,
)
from src.evidence.hashing import content_digest, sha256_bytes
from src.evidence.signing import Ed25519Signer, TrustStore, verify_envelope

from .evaluation import EvaluationParams, evaluate_partitions
from .lineage import LineageError, verify_source_lineage
from .manifests import params_from_dict
from .models import BookResult, HEAD_ID, LANE_ID, PackagedBook, WorkerOutput

logger = logging.getLogger(__name__)
Evaluator = Callable[[Mapping[str, bytes], EvaluationParams], BookResult]
RegistryPublish = Callable[[str, Mapping[str, float]], None]


class JobVerificationError(ValueError):
    pass


class DatasetHashError(ValueError):
    pass


def _verify_partitions(manifest: DatasetManifest, partitions: Mapping[str, bytes]) -> None:
    declared = {ref.partition_id: ref for ref in manifest.partitions}
    if set(partitions) != set(declared):
        raise DatasetHashError("supplied sleeve partitions do not match the signed dataset manifest")
    for sleeve_id, data in partitions.items():
        ref = declared[sleeve_id]
        if ref.size_bytes != len(data) or ref.digest != sha256_bytes(data):
            raise DatasetHashError(f"sleeve partition {sleeve_id!r} does not match its declared hash")


def _package(
    result: BookResult, *, job: JobManifest, job_digest: str, strategy: StrategyManifest,
    producer: Ed25519Signer, producer_id: str, created_at: datetime,
) -> PackagedBook:
    source_digests = tuple(sorted(str(v) for v in strategy.parameters["source_package_digests"].values()))
    report = EvaluationReport(
        report_id="portfolio-book-combined-book-eval",
        job_manifest_digest=job_digest,
        evaluator_id=producer_id,
        independent=False,
        created_at=created_at,
        temporal_holdout=result.temporal_holdout,
        purge_observations=0,
        embargo_observations=0,
        trial_count=job.trial_budget,
        effective_sample_size=result.effective_sample_size,
        metrics={name: MetricValue(value=float(value), tolerance=1e-9) for name, value in result.metrics.items()},
        gates=result.gates,
        incumbent_comparison=dict(result.incumbent_comparison),
        passed=result.passed,
    )
    report_envelope = producer.sign(report, created_at=created_at)
    path = "artifacts/portfolio_book.json"
    artifact = ArtifactRef(
        artifact_id="portfolio-book-combined-book", relative_path=path,
        digest=sha256_bytes(result.artifact_bytes), size_bytes=len(result.artifact_bytes),
        media_type="application/json",
    )
    package = EvidencePackage(
        package_id="portfolio-book-combined-book",
        lane_id=LANE_ID,
        producer_id=producer_id,
        created_at=created_at,
        job_manifest_digest=job_digest,
        dataset_manifest_digests=job.dataset_manifest_digests,
        strategy_manifest_digest=job.strategy_manifest_digest,
        evaluation_report_digests=(report_envelope.payload_digest,),
        derived_from_package_digests=source_digests,
        artifacts=(artifact,),
        safety_assertions=(
            SafetyAssertion(assertion_id="worker_no_broker_credentials", passed=True),
            SafetyAssertion(assertion_id="worker_cannot_place_or_cancel_orders", passed=True),
            SafetyAssertion(assertion_id="worker_cannot_promote_or_install_book", passed=True),
            SafetyAssertion(assertion_id="worker_cannot_flip_halts_or_live_gate", passed=True),
        ),
        checksums={path: artifact.digest},
    )
    package_envelope = producer.sign(package, created_at=created_at)
    package_digest = package_envelope.payload_digest
    event = DispositionEvent(
        package_digest=package_digest, sequence=0, previous_event_digest=None,
        from_state=None, to_state=DispositionState.CREATED,
        authority=AuthorityRole.PRODUCER, actor_id=producer_id,
        signer_key_id=producer.key_id, occurred_at=created_at,
        reason="authority-free worker produced combined-book allocation evidence",
        metadata={"head_id": HEAD_ID, "passed": result.passed, "n_sleeves": len(strategy.universe)},
    )
    return PackagedBook(
        package=package, package_digest=package_digest,
        package_envelope=package_envelope, files={path: result.artifact_bytes},
        evaluation_report_envelope=report_envelope,
        created_event_envelope=producer.sign(event, created_at=created_at),
    )


def run_worker(
    *, job_envelope: SignedEnvelope, dataset_manifest_envelope: SignedEnvelope,
    strategy_manifest_envelope: SignedEnvelope, capability_profile_envelope: SignedEnvelope,
    partitions: Mapping[str, bytes], producer: Ed25519Signer, producer_id: str,
    trust_store: TrustStore, created_at: datetime,
    evaluator: Evaluator = evaluate_partitions,
    registry_publish: RegistryPublish = lambda *_: None,
) -> WorkerOutput:
    try:
        job = verify_envelope(job_envelope, JobManifest, trust_store)
        dataset = verify_envelope(dataset_manifest_envelope, DatasetManifest, trust_store)
        strategy = verify_envelope(strategy_manifest_envelope, StrategyManifest, trust_store)
        capability = verify_envelope(capability_profile_envelope, CapabilityProfile, trust_store)
    except ValueError as exc:
        raise JobVerificationError(str(exc)) from exc
    if not all((
        isinstance(job, JobManifest), isinstance(dataset, DatasetManifest),
        isinstance(strategy, StrategyManifest), isinstance(capability, CapabilityProfile),
    )):
        raise JobVerificationError("signed envelope payload type mismatch")
    if content_digest(dataset) not in job.dataset_manifest_digests:
        raise JobVerificationError("job does not bind the supplied dataset manifest")
    if job.strategy_manifest_digest != content_digest(strategy):
        raise JobVerificationError("job does not bind the supplied strategy manifest")
    if job.capability_profile_digest != content_digest(capability):
        raise JobVerificationError("job does not bind the supplied capability profile")
    if strategy.lane_id != LANE_ID:
        raise JobVerificationError("strategy manifest is not a portfolio-book workload")
    if set(strategy.universe) != set(partitions):
        raise JobVerificationError("signed strategy universe does not match the supplied sleeve partitions")
    if tuple(job.expected_outputs) != (LANE_ID,):
        raise JobVerificationError("job expected outputs do not declare the portfolio-book lane")
    _verify_partitions(dataset, partitions)
    source_digests = {str(k): str(v) for k, v in strategy.parameters["source_package_digests"].items()}
    try:
        verify_source_lineage(partitions, source_digests)
    except LineageError as exc:
        raise JobVerificationError(str(exc)) from exc
    params = params_from_dict(strategy.parameters)
    result = evaluator(partitions, params)
    book = _package(result, job=job, job_digest=job_envelope.payload_digest, strategy=strategy,
                    producer=producer, producer_id=producer_id, created_at=created_at)
    try:
        registry_publish(LANE_ID, dict(result.metrics))
    except Exception as exc:  # fail-open mirror only
        logger.warning("portfolio-book registry publish failed (ignored): %s", exc)
    return WorkerOutput(
        job_envelope=job_envelope, dataset_manifest_envelope=dataset_manifest_envelope,
        strategy_manifest_envelope=strategy_manifest_envelope,
        capability_profile_envelope=capability_profile_envelope, book=book,
    )


__all__ = ["Evaluator", "RegistryPublish", "JobVerificationError", "DatasetHashError", "run_worker"]
