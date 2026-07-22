"""Producer-side crypto-carry evaluation worker (authority-free).

Given a signed ``JobManifest``, the matching signed ``DatasetManifest`` and
``StrategyManifest``, and the declared cell partition bytes, the worker:

1. verifies the job / dataset / strategy manifest signatures,
2. confirms the supplied partition bytes match the declared dataset hashes,
3. confirms the job binds this exact dataset and strategy manifest,
4. scores every venue/cost/regime cell and aggregates one head per carry id with
   the injected evaluator (which refuses an incomplete/duplicate cell set),
5. requires every declared carry_id head to be produced,
6. packages each carry_id into an immutable ``EvidencePackage`` with its own
   ``EvaluationReport`` and a producer-signed ``null -> CREATED`` event,
7. fail-open mirrors metrics to an external registry (never load-bearing).

It imports nothing that can reach a broker credential, an order, or the control
state; its only data input is bytes it is handed. The worker cannot promote,
quarantine, or approve anything — it only *creates* evidence.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from datetime import datetime
from typing import Mapping

from src.evidence.contracts import (
    ArtifactRef,
    AuthorityRole,
    CapabilityProfile,
    DatasetManifest,
    DispositionEvent,
    DispositionState,
    EvaluationReport,
    EvidencePackage,
    JobManifest,
    MetricValue,
    SafetyAssertion,
    SignedEnvelope,
    StrategyManifest,
)
from src.evidence.hashing import content_digest, sha256_bytes
from src.evidence.signing import Ed25519Signer, TrustStore, verify_envelope

from .evaluation import EvaluationParams, evaluate_partitions
from .models import CryptoCarryWorkerOutput, PackagedCarryHead, CarryHeadResult

logger = logging.getLogger(__name__)

Evaluator = Callable[..., "tuple[CarryHeadResult, ...]"]
RegistryPublish = Callable[[str, Mapping[str, float]], None]


class JobVerificationError(ValueError):
    """The job manifest signature or referenced lineage is invalid."""


class DatasetHashError(ValueError):
    """Supplied partition bytes do not match the declared dataset hashes."""


class MissingHeadError(ValueError):
    """A declared carry_id head was not produced (aggregation is incomplete)."""


def _noop_registry(experiment: str, metrics: Mapping[str, float]) -> None:
    del experiment, metrics


def _fail_open_registry(experiment: str, metrics: Mapping[str, float]) -> None:
    """Best-effort mirror; a registry outage must never stop evidence creation."""
    try:  # pragma: no cover - exercised via injection in tests
        from src.training.mlflow_mirror import mirror_training_session

        mirror_training_session(
            {"pair": "crypto_carry", "timestamp": experiment, "granularity": "cell",
             "warm_start": False, "duration_seconds": 0.0,
             "models_trained": ["carry_scorecard"], "hyperparams": {}, "metrics": dict(metrics)},
            experiment_name=experiment,
        )
    except Exception as exc:  # noqa: BLE001 - fail-open by contract
        logger.warning("crypto_carry registry mirror failed (ignored): %s", exc)


def _verify_partition_hashes(
    dataset_manifest: DatasetManifest, partitions: Mapping[str, bytes]
) -> None:
    declared = {ref.partition_id: ref for ref in dataset_manifest.partitions}
    if set(partitions) != set(declared):
        raise DatasetHashError(
            f"partition set mismatch: supplied={sorted(partitions)}, declared={sorted(declared)}"
        )
    for partition_id, data in partitions.items():
        ref = declared[partition_id]
        if len(data) != ref.size_bytes or sha256_bytes(data) != ref.digest:
            raise DatasetHashError(f"cell partition {partition_id!r} does not match its declared hash")


def _build_evaluation_report(
    head: CarryHeadResult,
    *,
    job_digest: str,
    producer_id: str,
    created_at: datetime,
) -> EvaluationReport:
    metrics = {
        name: MetricValue(value=float(value), tolerance=head.metric_tolerances.get(name))
        for name, value in head.metrics.items()
    }
    return EvaluationReport(
        report_id=f"crypto-carry-{head.campaign_id}-{head.carry_id}-eval",
        job_manifest_digest=job_digest,
        evaluator_id=producer_id,
        independent=False,
        created_at=created_at,
        temporal_holdout=head.temporal_holdout,
        purge_observations=0,
        embargo_observations=0,
        trial_count=1,
        effective_sample_size=float(head.n_cells),
        metrics=metrics,
        gates=head.gates,
        incumbent_comparison=dict(head.incumbent_comparison),
        passed=head.passed,
    )


def _package_head(
    head: CarryHeadResult,
    *,
    job: JobManifest,
    job_digest: str,
    producer: Ed25519Signer,
    producer_id: str,
    created_at: datetime,
) -> PackagedCarryHead:
    report = _build_evaluation_report(
        head, job_digest=job_digest, producer_id=producer_id, created_at=created_at
    )
    report_envelope = producer.sign(report, created_at=created_at)

    artifact_path = "artifacts/carry_scorecard.json"
    artifact = ArtifactRef(
        artifact_id=f"crypto-carry-{head.campaign_id}-{head.carry_id}-scorecard",
        relative_path=artifact_path,
        digest=sha256_bytes(head.artifact_bytes),
        size_bytes=len(head.artifact_bytes),
        media_type=head.media_type,
    )
    artifacts = [artifact]
    files = {artifact_path: head.artifact_bytes}
    checksums = {artifact_path: artifact.digest}
    if head.strategy_return_bytes:
        contract_path = "artifacts/strategy_returns.jsonl"
        contract_artifact = ArtifactRef(
            artifact_id=f"crypto-carry-{head.campaign_id}-{head.carry_id}-strategy-returns",
            relative_path=contract_path,
            digest=sha256_bytes(head.strategy_return_bytes),
            size_bytes=len(head.strategy_return_bytes),
            media_type="application/x-ndjson",
        )
        artifacts.append(contract_artifact)
        files[contract_path] = head.strategy_return_bytes
        checksums[contract_path] = contract_artifact.digest
    package = EvidencePackage(
        package_id=f"crypto-carry-{head.campaign_id}-{head.carry_id}",
        lane_id=head.lane_id,
        producer_id=producer_id,
        created_at=created_at,
        job_manifest_digest=job_digest,
        dataset_manifest_digests=job.dataset_manifest_digests,
        strategy_manifest_digest=job.strategy_manifest_digest,
        evaluation_report_digests=(report_envelope.payload_digest,),
        artifacts=tuple(artifacts),
        safety_assertions=(
            SafetyAssertion(assertion_id="remote_worker_no_broker_credentials", passed=True),
            SafetyAssertion(assertion_id="remote_worker_cannot_read_control_state", passed=True),
            SafetyAssertion(assertion_id="worker_cannot_promote_or_overwrite_incumbent", passed=True),
        ),
        checksums=checksums,
    )
    package_envelope = producer.sign(package, created_at=created_at)
    package_digest = package_envelope.payload_digest

    created_event = DispositionEvent(
        package_digest=package_digest,
        sequence=0,
        previous_event_digest=None,
        from_state=None,
        to_state=DispositionState.CREATED,
        authority=AuthorityRole.PRODUCER,
        actor_id=producer_id,
        signer_key_id=producer.key_id,
        occurred_at=created_at,
        reason=f"remote worker produced {head.carry_id} carry evidence for {head.campaign_id}",
        metadata={"campaign_id": head.campaign_id, "carry_id": head.carry_id,
                  "verdict": head.verdict, "passed": head.passed},
    )
    created_envelope = producer.sign(created_event, created_at=created_at)

    return PackagedCarryHead(
        head_id=head.head_id,
        lane_id=head.lane_id,
        campaign_id=head.campaign_id,
        carry_id=head.carry_id,
        package=package,
        package_digest=package_digest,
        package_envelope=package_envelope,
        files=files,
        evaluation_report_envelope=report_envelope,
        created_event_envelope=created_envelope,
        result_passed=head.passed,
    )


def run_worker(
    *,
    campaign_id: str,
    job_envelope: SignedEnvelope,
    dataset_manifest_envelope: SignedEnvelope,
    strategy_manifest_envelope: SignedEnvelope,
    capability_profile_envelope: SignedEnvelope,
    partitions: Mapping[str, bytes],
    producer: Ed25519Signer,
    producer_id: str,
    trust_store: TrustStore,
    created_at: datetime,
    params: EvaluationParams | None = None,
    evaluator: Evaluator = evaluate_partitions,
    registry_publish: RegistryPublish = _fail_open_registry,
) -> CryptoCarryWorkerOutput:
    """Execute one signed crypto-carry job and emit per-carry_id evidence."""
    params = params or EvaluationParams()

    try:
        job = verify_envelope(job_envelope, JobManifest, trust_store)
        dataset_manifest = verify_envelope(dataset_manifest_envelope, DatasetManifest, trust_store)
        strategy_manifest = verify_envelope(strategy_manifest_envelope, StrategyManifest, trust_store)
        capability_profile = verify_envelope(
            capability_profile_envelope, CapabilityProfile, trust_store
        )
    except ValueError as exc:
        raise JobVerificationError(str(exc)) from exc
    if not isinstance(job, JobManifest):
        raise JobVerificationError("job envelope did not decode to JobManifest")
    if not isinstance(dataset_manifest, DatasetManifest):
        raise JobVerificationError("dataset envelope did not decode to DatasetManifest")
    if not isinstance(strategy_manifest, StrategyManifest):
        raise JobVerificationError("strategy envelope did not decode to StrategyManifest")
    if not isinstance(capability_profile, CapabilityProfile):
        raise JobVerificationError("capability envelope did not decode to CapabilityProfile")

    dataset_digest = content_digest(dataset_manifest)
    if dataset_digest not in job.dataset_manifest_digests:
        raise JobVerificationError("job does not reference the supplied dataset manifest")
    if job.strategy_manifest_digest != content_digest(strategy_manifest):
        raise JobVerificationError("job does not reference the supplied strategy manifest")
    if job.capability_profile_digest != content_digest(capability_profile):
        raise JobVerificationError("job does not reference the supplied capability profile")
    if content_digest(job) != job_envelope.payload_digest:
        raise JobVerificationError("job manifest digest mismatch")

    _verify_partition_hashes(dataset_manifest, partitions)

    results = evaluator(partitions, params, campaign_id=campaign_id)
    produced: dict[str, CarryHeadResult] = {}
    for head in results:
        if head.lane_id in produced:
            raise MissingHeadError(f"duplicate carry_id lane {head.lane_id!r} produced")
        produced[head.lane_id] = head
    expected = set(job.expected_outputs)
    if set(produced) != expected:
        raise MissingHeadError(
            f"produced carry_id lanes {sorted(produced)} do not match declared outputs {sorted(expected)}"
        )

    job_digest = job_envelope.payload_digest
    packaged = tuple(
        _package_head(
            produced[lane_id],
            job=job,
            job_digest=job_digest,
            producer=producer,
            producer_id=producer_id,
            created_at=created_at,
        )
        for lane_id in sorted(produced)
    )

    # Registry mirroring is best-effort and never load-bearing: an external
    # registry outage must not stop evidence creation (roadmap §12). Wrap every
    # call so even an injected, raising registry cannot break the run.
    for head in packaged:
        try:
            registry_publish(head.lane_id, dict(produced[head.lane_id].metrics))
        except Exception as exc:  # noqa: BLE001 - fail-open by contract
            logger.warning("crypto_carry registry publish failed (ignored): %s", exc)

    # Sanity: every signed package is genuinely content-addressed — the envelope
    # digest equals the canonical digest of its own payload.
    for head in packaged:
        if head.package_envelope.payload_digest != content_digest(head.package):
            raise JobVerificationError(
                f"package {head.package.package_id!r} is not content-addressed correctly"
            )

    return CryptoCarryWorkerOutput(
        job_envelope=job_envelope,
        dataset_manifest_envelope=dataset_manifest_envelope,
        strategy_manifest_envelope=strategy_manifest_envelope,
        capability_profile_envelope=capability_profile_envelope,
        heads=packaged,
    )


__all__ = [
    "Evaluator",
    "RegistryPublish",
    "JobVerificationError",
    "DatasetHashError",
    "MissingHeadError",
    "run_worker",
]
