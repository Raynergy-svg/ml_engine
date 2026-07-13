"""Producer-side hedge-evaluation worker (authority-free).

Given a signed ``JobManifest``, the matching signed ``DatasetManifest`` and
``StrategyManifest``, and the declared per-strategy ledger partition bytes, the
worker:

1. verifies the job / dataset / strategy manifest signatures,
2. confirms the supplied partition bytes match the declared dataset hashes,
3. confirms the job binds this exact dataset and strategy manifest,
4. scores only the declared partitions with the injected evaluator,
5. requires every declared strategy head to be produced,
6. packages each strategy into an immutable ``EvidencePackage`` with its own
   ``EvaluationReport`` and a producer-signed ``null -> CREATED`` event,
7. fail-open mirrors metrics to an external registry (never load-bearing).

It imports nothing that can reach a broker credential, an order, or the
control state; its only data input is bytes it is handed. The worker cannot
promote, quarantine, or approve anything — it only *creates* evidence.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from datetime import datetime
from typing import Mapping

from src.evidence.contracts import (
    ArtifactRef,
    AuthorityRole,
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
from .models import HedgeHeadResult, HedgeWorkerOutput, PackagedHedgeHead

logger = logging.getLogger(__name__)

Evaluator = Callable[[Mapping[str, bytes], EvaluationParams], "tuple[HedgeHeadResult, ...]"]
RegistryPublish = Callable[[str, Mapping[str, float]], None]


class JobVerificationError(ValueError):
    """The job manifest signature or referenced lineage is invalid."""


class DatasetHashError(ValueError):
    """Supplied partition bytes do not match the declared dataset hashes."""


class MissingHeadError(ValueError):
    """A declared strategy head was not produced (aggregation is incomplete)."""


def _noop_registry(experiment: str, metrics: Mapping[str, float]) -> None:
    del experiment, metrics


def _fail_open_registry(experiment: str, metrics: Mapping[str, float]) -> None:
    """Best-effort mirror; a registry outage must never stop evidence creation."""
    try:  # pragma: no cover - exercised via injection in tests
        from src.training.mlflow_mirror import mirror_training_session

        mirror_training_session(
            {"pair": "hedge_eval", "timestamp": experiment, "granularity": "cycle",
             "warm_start": False, "duration_seconds": 0.0,
             "models_trained": ["hedge_scorecard"], "hyperparams": {}, "metrics": dict(metrics)},
            experiment_name=experiment,
        )
    except Exception as exc:  # noqa: BLE001 - fail-open by contract
        logger.warning("hedge_eval registry mirror failed (ignored): %s", exc)


def _verify_partition_hashes(
    dataset_manifest: DatasetManifest, partitions: Mapping[str, bytes]
) -> None:
    declared = {ref.partition_id: ref for ref in dataset_manifest.partitions}
    if set(partitions) != set(declared):
        raise DatasetHashError(
            f"partition set mismatch: supplied={sorted(partitions)}, "
            f"declared={sorted(declared)}"
        )
    for partition_id, data in partitions.items():
        ref = declared[partition_id]
        if len(data) != ref.size_bytes or sha256_bytes(data) != ref.digest:
            raise DatasetHashError(f"partition {partition_id!r} does not match its declared hash")


def _build_evaluation_report(
    head: HedgeHeadResult,
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
        report_id=f"hedge-{head.strategy}-eval",
        job_manifest_digest=job_digest,
        evaluator_id=producer_id,
        independent=False,
        created_at=created_at,
        temporal_holdout=head.temporal_holdout,
        purge_observations=0,
        embargo_observations=0,
        trial_count=1,
        effective_sample_size=float(head.n_cycles),
        metrics=metrics,
        gates=head.gates,
        incumbent_comparison=dict(head.incumbent_comparison),
        passed=head.passed,
    )


def _package_head(
    head: HedgeHeadResult,
    *,
    job: JobManifest,
    job_digest: str,
    producer: Ed25519Signer,
    producer_id: str,
    created_at: datetime,
) -> PackagedHedgeHead:
    report = _build_evaluation_report(
        head, job_digest=job_digest, producer_id=producer_id, created_at=created_at
    )
    report_envelope = producer.sign(report, created_at=created_at)

    artifact_path = "artifacts/scorecard.json"
    artifact = ArtifactRef(
        artifact_id=f"hedge-{head.strategy}-scorecard",
        relative_path=artifact_path,
        digest=sha256_bytes(head.artifact_bytes),
        size_bytes=len(head.artifact_bytes),
        media_type=head.media_type,
    )
    package = EvidencePackage(
        package_id=f"hedge-eval-{head.strategy}",
        lane_id=head.lane_id,
        producer_id=producer_id,
        created_at=created_at,
        job_manifest_digest=job_digest,
        dataset_manifest_digests=job.dataset_manifest_digests,
        strategy_manifest_digest=job.strategy_manifest_digest,
        evaluation_report_digests=(report_envelope.payload_digest,),
        artifacts=(artifact,),
        safety_assertions=(
            SafetyAssertion(assertion_id="remote_worker_no_broker_credentials", passed=True),
            SafetyAssertion(assertion_id="remote_worker_cannot_read_control_state", passed=True),
            SafetyAssertion(assertion_id="worker_cannot_promote_or_overwrite_incumbent", passed=True),
        ),
        checksums={artifact_path: artifact.digest},
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
        reason=f"remote worker produced {head.strategy} hedge evidence",
        metadata={"strategy": head.strategy, "verdict": head.verdict, "passed": head.passed},
    )
    created_envelope = producer.sign(created_event, created_at=created_at)

    return PackagedHedgeHead(
        head_id=head.head_id,
        lane_id=head.lane_id,
        package=package,
        package_digest=package_digest,
        package_envelope=package_envelope,
        files={artifact_path: head.artifact_bytes},
        evaluation_report_envelope=report_envelope,
        created_event_envelope=created_envelope,
        result_passed=head.passed,
    )


def run_worker(
    *,
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
) -> HedgeWorkerOutput:
    """Execute one signed hedge-evaluation job and emit per-strategy evidence."""
    params = params or EvaluationParams()

    try:
        job = verify_envelope(job_envelope, JobManifest, trust_store)
        dataset_manifest = verify_envelope(dataset_manifest_envelope, DatasetManifest, trust_store)
        strategy_manifest = verify_envelope(strategy_manifest_envelope, StrategyManifest, trust_store)
    except ValueError as exc:
        raise JobVerificationError(str(exc)) from exc
    assert isinstance(job, JobManifest)
    assert isinstance(dataset_manifest, DatasetManifest)
    assert isinstance(strategy_manifest, StrategyManifest)

    dataset_digest = content_digest(dataset_manifest)
    if dataset_digest not in job.dataset_manifest_digests:
        raise JobVerificationError("job does not reference the supplied dataset manifest")
    if job.strategy_manifest_digest != content_digest(strategy_manifest):
        raise JobVerificationError("job does not reference the supplied strategy manifest")
    if content_digest(job) != job_envelope.payload_digest:
        raise JobVerificationError("job manifest digest mismatch")

    _verify_partition_hashes(dataset_manifest, partitions)

    results = evaluator(partitions, params)
    produced = {head.head_id: head for head in results}
    expected = set(job.expected_outputs)
    if set(produced) != expected:
        raise MissingHeadError(
            f"produced strategies {sorted(produced)} do not match declared outputs {sorted(expected)}"
        )

    job_digest = job_envelope.payload_digest
    packaged = tuple(
        _package_head(
            produced[head_id],
            job=job,
            job_digest=job_digest,
            producer=producer,
            producer_id=producer_id,
            created_at=created_at,
        )
        for head_id in sorted(produced)
    )

    # Registry mirroring is best-effort and never load-bearing: an external
    # registry outage must not stop evidence creation (roadmap §12). Wrap every
    # call so even an injected, raising registry cannot break the run.
    for head in packaged:
        try:
            registry_publish(f"hedge_eval_{head.head_id}", dict(produced[head.head_id].metrics))
        except Exception as exc:  # noqa: BLE001 - fail-open by contract
            logger.warning("hedge_eval registry publish failed (ignored): %s", exc)

    # Sanity: every signed package is genuinely content-addressed — the envelope
    # digest equals the canonical digest of its own payload.
    for head in packaged:
        assert head.package_envelope.payload_digest == content_digest(head.package)

    return HedgeWorkerOutput(
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
