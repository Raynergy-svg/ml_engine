"""Authority-free Track B scoring and independent forward-evaluation workers."""

from __future__ import annotations

import json
from datetime import datetime
from typing import Mapping

from src.evidence.contracts import (
    ArtifactRef, AuthorityRole, CapabilityProfile, DatasetManifest, DispositionEvent,
    DispositionState, EvaluationReport, EvidencePackage, GateResult, GateStatus,
    JobManifest, MetricValue, SafetyAssertion, SignedEnvelope, StrategyManifest,
)
from src.evidence.hashing import content_digest, sha256_bytes
from src.evidence.signing import Ed25519Signer, TrustStore, verify_envelope

from .evaluation import EvaluationParams, evaluate_partitions
from .lineage import score_batch, validate_score_lineage
from .models import PackagedTrackBHead, TrackBHeadResult, TrackBWorkerOutput, lane_id


class JobVerificationError(ValueError):
    pass


class DatasetHashError(ValueError):
    pass


def _verify_inputs(
    *,
    job_envelope: SignedEnvelope,
    dataset_manifest_envelope: SignedEnvelope,
    strategy_manifest_envelope: SignedEnvelope,
    capability_profile_envelope: SignedEnvelope,
    partitions: Mapping[str, bytes],
    trust_store: TrustStore,
) -> tuple[JobManifest, DatasetManifest, StrategyManifest]:
    try:
        job = verify_envelope(job_envelope, JobManifest, trust_store)
        dataset = verify_envelope(dataset_manifest_envelope, DatasetManifest, trust_store)
        strategy = verify_envelope(strategy_manifest_envelope, StrategyManifest, trust_store)
        capability = verify_envelope(capability_profile_envelope, CapabilityProfile, trust_store)
    except ValueError as exc:
        raise JobVerificationError(str(exc)) from exc
    if not isinstance(job, JobManifest) or not isinstance(dataset, DatasetManifest):
        raise JobVerificationError("job or dataset envelope decoded to wrong type")
    if not isinstance(strategy, StrategyManifest) or not isinstance(capability, CapabilityProfile):
        raise JobVerificationError("strategy or capability envelope decoded to wrong type")
    if content_digest(dataset) not in job.dataset_manifest_digests:
        raise JobVerificationError("job does not bind supplied dataset manifest")
    if content_digest(strategy) != job.strategy_manifest_digest:
        raise JobVerificationError("job does not bind supplied strategy manifest")
    if content_digest(capability) != job.capability_profile_digest:
        raise JobVerificationError("job does not bind supplied capability profile")
    declared = {ref.partition_id: ref for ref in dataset.partitions}
    if set(partitions) != set(declared):
        raise DatasetHashError("partition set mismatch")
    for partition_id, data in partitions.items():
        ref = declared[partition_id]
        if len(data) != ref.size_bytes or sha256_bytes(data) != ref.digest:
            raise DatasetHashError(f"partition {partition_id!r} hash mismatch")
    return job, dataset, strategy


def _package_head(
    head: TrackBHeadResult,
    *,
    job: JobManifest,
    job_digest: str,
    producer: Ed25519Signer,
    producer_id: str,
    created_at: datetime,
) -> PackagedTrackBHead:
    report = EvaluationReport(
        report_id=f"track-b-{head.stage}-{head.campaign_id}-{head.unit_id}-eval",
        job_manifest_digest=job_digest,
        evaluator_id=producer_id,
        independent=False,
        created_at=created_at,
        temporal_holdout=head.temporal_holdout,
        purge_observations=0,
        embargo_observations=0,
        trial_count=1,
        effective_sample_size=max(1.0, head.metrics.get("n_scored", head.metrics.get("n_cells", 1.0))),
        metrics={name: MetricValue(value=value, tolerance=1e-12) for name, value in head.metrics.items()},
        gates=head.gates,
        incumbent_comparison={"has_incumbent": False, "stage": head.stage},
        passed=head.passed,
    )
    report_envelope = producer.sign(report, created_at=created_at)
    main_path = f"artifacts/{head.stage}_artifact.jsonl"
    files = {main_path: head.artifact_bytes, **dict(head.extra_artifacts)}
    artifacts = tuple(
        ArtifactRef(
            artifact_id=f"track-b-{head.stage}-{head.campaign_id}-{head.unit_id}-{index}",
            relative_path=path,
            digest=sha256_bytes(data),
            size_bytes=len(data),
            media_type="application/x-ndjson",
        )
        for index, (path, data) in enumerate(sorted(files.items()))
    )
    package = EvidencePackage(
        package_id=f"track-b-{head.stage}-{head.campaign_id}-{head.unit_id}",
        lane_id=head.lane_id,
        producer_id=producer_id,
        created_at=created_at,
        job_manifest_digest=job_digest,
        dataset_manifest_digests=job.dataset_manifest_digests,
        strategy_manifest_digest=job.strategy_manifest_digest,
        evaluation_report_digests=(report_envelope.payload_digest,),
        derived_from_package_digests=head.derived_from_package_digests,
        artifacts=artifacts,
        safety_assertions=(
            SafetyAssertion(assertion_id="remote_worker_no_broker_credentials", passed=True),
            SafetyAssertion(assertion_id="remote_worker_cannot_read_control_state", passed=True),
            SafetyAssertion(assertion_id="worker_cannot_promote_or_overwrite_incumbent", passed=True),
        ),
        checksums={artifact.relative_path: artifact.digest for artifact in artifacts},
    )
    package_envelope = producer.sign(package, created_at=created_at)
    event = DispositionEvent(
        package_digest=package_envelope.payload_digest,
        sequence=0,
        previous_event_digest=None,
        from_state=None,
        to_state=DispositionState.CREATED,
        authority=AuthorityRole.PRODUCER,
        actor_id=producer_id,
        signer_key_id=producer.key_id,
        occurred_at=created_at,
        reason=f"Track B {head.stage} worker produced {head.unit_id}",
        metadata={"stage": head.stage, "unit_id": head.unit_id, "passed": head.passed},
    )
    return PackagedTrackBHead(
        stage=head.stage,
        campaign_id=head.campaign_id,
        unit_id=head.unit_id,
        lane_id=head.lane_id,
        package=package,
        package_digest=package_envelope.payload_digest,
        package_envelope=package_envelope,
        files=files,
        evaluation_report_envelope=report_envelope,
        created_event_envelope=producer.sign(event, created_at=created_at),
        result_passed=head.passed,
    )


def run_scoring_worker(
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
) -> TrackBWorkerOutput:
    job, _, strategy = _verify_inputs(
        job_envelope=job_envelope,
        dataset_manifest_envelope=dataset_manifest_envelope,
        strategy_manifest_envelope=strategy_manifest_envelope,
        capability_profile_envelope=capability_profile_envelope,
        partitions=partitions,
        trust_store=trust_store,
    )
    expected_batches = strategy.parameters.get("expected_batches")
    if tuple(expected_batches or ()) != tuple(sorted(partitions)):
        raise JobVerificationError("signed scoring batches do not match dataset partitions")
    heads: list[TrackBHeadResult] = []
    for batch_id in sorted(partitions):
        artifact, queue, metrics = score_batch(batch_id, partitions[batch_id], batch_id=batch_id)
        for line in artifact.decode("utf-8").splitlines():
            record = json.loads(line)
            validate_score_lineage(record)
            if (
                record["model_cutoff"] != strategy.parameters["model_cutoff"]
                or record["scorer_model"] != strategy.parameters["scorer_model"]
                or record["scorer_version"] != strategy.parameters["scorer_version"]
                or record["prompt_digest"] != strategy.parameters["prompt_digest"]
            ):
                raise JobVerificationError("score lineage does not match signed scorer contract")
        no_manual = metrics["n_manual_review"] == 0.0
        gates = (
            GateResult(
                gate_id="full_score_lineage_valid",
                status=GateStatus.PASS,
                observed=int(metrics["n_scored"]),
                threshold="all scored records satisfy §J5",
                reason="every score is content-addressed and carries required lineage",
            ),
            GateResult(
                gate_id="manual_review_queue_empty",
                status=GateStatus.PASS if no_manual else GateStatus.FAIL,
                observed=int(metrics["n_manual_review"]),
                threshold=0,
                reason="malformed, unblinded, ungrounded or extreme scores require manual review",
            ),
        )
        unit_id = f"{batch_id}-{strategy.parameters['scorer_version']}"
        heads.append(
            TrackBHeadResult(
                stage="scoring", campaign_id=campaign_id, unit_id=unit_id,
                lane_id=lane_id("scoring", campaign_id, unit_id), metrics=metrics,
                passed=no_manual, gates=gates, artifact_bytes=artifact,
                extra_artifacts={"artifacts/manual_review_queue.jsonl": queue},
                temporal_holdout=f"model cutoff {strategy.parameters['model_cutoff']}",
            )
        )
    produced = {head.lane_id for head in heads}
    if produced != set(job.expected_outputs) or len(produced) != len(heads):
        raise JobVerificationError("scoring worker outputs do not match signed job")
    packaged = tuple(
        _package_head(
            head, job=job, job_digest=job_envelope.payload_digest,
            producer=producer, producer_id=producer_id, created_at=created_at,
        )
        for head in heads
    )
    return TrackBWorkerOutput(
        job_envelope=job_envelope,
        dataset_manifest_envelope=dataset_manifest_envelope,
        strategy_manifest_envelope=strategy_manifest_envelope,
        capability_profile_envelope=capability_profile_envelope,
        heads=packaged,
    )


def run_evaluation_worker(
    *,
    campaign_id: str,
    job_envelope: SignedEnvelope,
    dataset_manifest_envelope: SignedEnvelope,
    strategy_manifest_envelope: SignedEnvelope,
    capability_profile_envelope: SignedEnvelope,
    partitions: Mapping[str, bytes],
    scoring_package_digests: tuple[str, ...],
    producer: Ed25519Signer,
    producer_id: str,
    trust_store: TrustStore,
    created_at: datetime,
    params: EvaluationParams,
) -> TrackBWorkerOutput:
    job, _, strategy = _verify_inputs(
        job_envelope=job_envelope,
        dataset_manifest_envelope=dataset_manifest_envelope,
        strategy_manifest_envelope=strategy_manifest_envelope,
        capability_profile_envelope=capability_profile_envelope,
        partitions=partitions,
        trust_store=trust_store,
    )
    if tuple(strategy.parameters.get("expected_cells") or ()) != tuple(sorted(partitions)):
        raise JobVerificationError("signed evaluation cells do not match dataset partitions")
    if tuple(sorted(strategy.parameters.get("scoring_package_digests", []))) != tuple(sorted(scoring_package_digests)):
        raise JobVerificationError("scoring package derivation is not bound by strategy manifest")
    result = evaluate_partitions(partitions, params)
    gates = (
        GateResult(
            gate_id="adequate_forward_time",
            status=GateStatus.PASS if result["enough_forward_time"] else GateStatus.FAIL,
            observed=result["n_valid_forward_cells"], threshold=params.min_forward_cells,
            reason="Track B requires independent rebalance/fold observations, not merely more names",
        ),
        GateResult(
            gate_id="rank_ic_clears",
            status=GateStatus.PASS if result["signal_gate_passed"] else GateStatus.FAIL,
            observed=result["mean_rank_ic"], threshold=params.rank_ic_bar,
            reason="mean forward rank IC must clear the frozen bar",
        ),
        GateResult(
            gate_id="placebo_clean",
            status=GateStatus.PASS if result["placebo_gate_passed"] else GateStatus.FAIL,
            observed=result["mean_placebo_rank_ic"], threshold=params.placebo_abs_limit,
            reason="deterministic entity-score placebo must remain near zero",
        ),
    )
    metrics = {
        key: float(value)
        for key, value in result.items()
        if key in {"n_cells", "n_valid_forward_cells", "mean_rank_ic", "mean_placebo_rank_ic"}
        and value is not None
    }
    unit_id = "forward-rank-ic"
    head = TrackBHeadResult(
        stage="evaluation", campaign_id=campaign_id, unit_id=unit_id,
        lane_id=lane_id("evaluation", campaign_id, unit_id), metrics=metrics,
        passed=bool(result["passed"]), gates=gates,
        artifact_bytes=result["artifact_bytes"], extra_artifacts={},
        temporal_holdout=f"{result['n_valid_forward_cells']} valid rebalance/fold cells",
        derived_from_package_digests=tuple(sorted(scoring_package_digests)),
    )
    if {head.lane_id} != set(job.expected_outputs):
        raise JobVerificationError("evaluation output does not match signed job")
    packaged = _package_head(
        head, job=job, job_digest=job_envelope.payload_digest,
        producer=producer, producer_id=producer_id, created_at=created_at,
    )
    return TrackBWorkerOutput(
        job_envelope=job_envelope,
        dataset_manifest_envelope=dataset_manifest_envelope,
        strategy_manifest_envelope=strategy_manifest_envelope,
        capability_profile_envelope=capability_profile_envelope,
        heads=(packaged,),
    )


__all__ = [
    "JobVerificationError", "DatasetHashError", "run_scoring_worker", "run_evaluation_worker",
]
