"""Frozen manifests for the execution-cost model workload."""

from __future__ import annotations

from datetime import date, datetime
from typing import Mapping, Sequence

from src.evidence.contracts import (
    AssetClass, CapabilityProfile, DatasetManifest, JobManifest,
    ModelTrainingSpec, PartitionRef, ResourceClass,
)
from src.evidence.hashing import content_digest, sha256_bytes

from .evaluation import FEATURES
from .models import LANE_ID

CAPABILITY_PROFILE_ID = "axiom-execution-cost-worker"
METRICS = (
    "oos_mae_bps", "oos_baseline_mae_bps", "oos_rmse_bps",
    "oos_p95_underprediction_bps", "oos_baseline_p95_underprediction_bps",
    "train_rows", "test_rows",
)


def build_capability_profile() -> CapabilityProfile:
    return CapabilityProfile(
        profile_id=CAPABILITY_PROFILE_ID,
        read_scopes=("axiom-data/execution_cost/snapshots/**",),
        write_scopes=("job:{job_id}/outputs/**",),
        network_endpoints=(),
    )


def build_dataset_manifest(
    partitions: Mapping[str, bytes], *, dataset_id: str, retrieved_at: datetime,
    coverage_start: date, coverage_end: date,
    row_counts: Mapping[str, int] | None = None,
) -> DatasetManifest:
    if not partitions:
        raise ValueError("at least one fill/slippage partition is required")
    row_counts = row_counts or {}
    refs = tuple(
        PartitionRef(
            partition_id=partition_id,
            uri=f"axiom-data/execution_cost/snapshots/{dataset_id}/fills/{partition_id}.jsonl",
            digest=sha256_bytes(data), size_bytes=len(data),
            row_count=row_counts.get(partition_id),
        )
        for partition_id, data in sorted(partitions.items())
    )
    return DatasetManifest(
        dataset_id=dataset_id,
        asset_class=AssetClass.MULTI_ASSET,
        domain="execution_cost_fill_book",
        provider="canonical_forward_fill_capture",
        retrieved_at=retrieved_at,
        coverage_start=coverage_start,
        coverage_end=coverage_end,
        instruments=tuple(sorted(partitions)),
        data_schema_version="execution_cost_fill_context.v1",
        point_in_time_rules=(
            "features are captured no later than the order decision; realized cost is labeled only after fill",
            "training consumes immutable declared partitions and makes no live broker calls",
        ),
        partitions=refs,
    )


def build_training_spec(*, trial_budget: int = 1) -> ModelTrainingSpec:
    if trial_budget != 1:
        raise ValueError("execution-cost v1 is preregistered for exactly one deterministic trial")
    return ModelTrainingSpec(
        target="realized_execution_cost_bps",
        features=FEATURES,
        causal_lag="feature_as_of <= decision_at < fill_at; the realized-cost label is post-fill only",
        search_space={
            "model_family": ["ridge_linear"],
            "ridge_penalty": [1e-6],
            "trial_budget": trial_budget,
            "universe": "all instruments represented by the signed fill-book snapshot",
            "cost_model": "realized implementation shortfall in basis points after spread, fees and market impact",
            "evaluation_window": "strict chronological holdout selected by the signed configuration digest",
        },
        metrics=METRICS,
        promotion_criteria=(
            "chronological OOS MAE must beat the train-only median baseline",
            "chronological OOS p95 cost underprediction must be bounded and no worse than baseline",
            "accepted evidence remains quarantined; installation requires separate operator-authorized promotion",
        ),
    )


def build_job_manifest(
    *, job_id: str, dataset_manifest: DatasetManifest,
    capability_profile: CapabilityProfile, git_commit: str,
    container_digest: str, configuration_digest: str,
    feature_pipeline_version: str, created_at: datetime,
    random_seeds: Sequence[int] = (0,), trial_budget: int = 1,
) -> JobManifest:
    if trial_budget != 1:
        raise ValueError("execution-cost v1 is preregistered for exactly one deterministic trial")
    return JobManifest(
        job_id=job_id,
        git_commit=git_commit,
        container_digest=container_digest,
        dataset_manifest_digests=(content_digest(dataset_manifest),),
        model_training_spec=build_training_spec(trial_budget=trial_budget),
        feature_pipeline_version=feature_pipeline_version,
        configuration_digest=configuration_digest,
        random_seeds=tuple(random_seeds),
        assigned_unit="execution_cost/realized_cost_bps/chronological_fold_0",
        resource_class=ResourceClass.CPU_SMALL,
        capability_profile_digest=content_digest(capability_profile),
        trial_budget=trial_budget,
        expected_outputs=(LANE_ID,),
        created_at=created_at,
    )


__all__ = ["CAPABILITY_PROFILE_ID", "METRICS", "build_capability_profile", "build_dataset_manifest", "build_training_spec", "build_job_manifest"]
