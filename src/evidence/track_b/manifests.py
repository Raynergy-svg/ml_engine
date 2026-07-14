"""Signed manifest builders for Track B scoring and independent evaluation."""

from __future__ import annotations

from datetime import date, datetime
from typing import Mapping, Sequence

from src.evidence.contracts import (
    AssetClass, CapabilityProfile, DatasetManifest, JobManifest, PartitionRef,
    ResourceClass, StrategyManifest,
)
from src.evidence.hashing import content_digest, sha256_bytes

TRACK_B_CAPABILITY_PROFILE_ID = "axiom-track-b-worker"


def build_capability_profile() -> CapabilityProfile:
    return CapabilityProfile(
        profile_id=TRACK_B_CAPABILITY_PROFILE_ID,
        read_scopes=("axiom-data/track_b/snapshots/**",),
        write_scopes=("job:{job_id}/outputs/**",),
        network_endpoints=(),
    )


def build_dataset_manifest(
    partitions: Mapping[str, bytes],
    *,
    dataset_id: str,
    domain: str,
    retrieved_at: datetime,
    coverage_start: date,
    coverage_end: date,
) -> DatasetManifest:
    if not partitions:
        raise ValueError("Track B dataset requires at least one partition")
    refs = tuple(
        PartitionRef(
            partition_id=partition_id,
            uri=f"axiom-data/track_b/snapshots/{dataset_id}/{partition_id}.jsonl",
            digest=sha256_bytes(data),
            size_bytes=len(data),
            row_count=len([line for line in data.splitlines() if line.strip()]),
        )
        for partition_id, data in sorted(partitions.items())
    )
    return DatasetManifest(
        dataset_id=dataset_id,
        asset_class=(AssetClass.FILINGS if "filing" in domain else AssetClass.EQUITY),
        domain=domain,
        provider="sec_edgar_track_b",
        retrieved_at=retrieved_at,
        coverage_start=coverage_start,
        coverage_end=coverage_end,
        instruments=tuple(sorted(partitions)),
        data_schema_version=f"{domain}.v1",
        point_in_time_rules=(
            "filing_date <= as_of_date is enforced before scoring",
            "evaluation rows retain the exact scored-record digest and score-artifact digest",
        ),
        partitions=refs,
    )


def build_scoring_strategy_manifest(
    *,
    campaign_id: str,
    expected_batches: Sequence[str],
    scorer_model: str,
    scorer_version: str,
    prompt_digest: str,
    model_cutoff: str,
    frozen_at: datetime,
) -> StrategyManifest:
    if not expected_batches or len(expected_batches) != len(set(expected_batches)):
        raise ValueError("expected scoring batches must be non-empty and unique")
    return StrategyManifest(
        strategy_id="track-b-filing-scoring",
        lane_id="track_b_scoring",
        experiment_id=f"track-b-scoring-{campaign_id}",
        hypothesis="Entity-blinded PIT filings can be scored reproducibly with full §J5 lineage.",
        exploratory=True,
        frozen_at=frozen_at,
        universe=tuple(sorted(expected_batches)),
        causal_lag="filing_date <= as_of_date; scoring observes only entity-blinded filing text",
        parameters={
            "campaign_id": campaign_id,
            "expected_batches": sorted(expected_batches),
            "scorer_model": scorer_model,
            "scorer_version": scorer_version,
            "prompt_digest": prompt_digest,
            "model_cutoff": model_cutoff,
        },
        cost_model={"basis": "offline document-processing workload"},
        evaluation_windows=("one immutable filing batch per scorer version",),
        search_space={"note": "frozen deterministic scorer; no tuning"},
        trial_budget=1,
        metrics=("n_scored", "n_manual_review", "n_post_model_cutoff"),
        promotion_criteria=("all score lineage validates and manual-review queue is empty",),
    )


def build_evaluation_strategy_manifest(
    *,
    campaign_id: str,
    expected_cells: Sequence[str],
    scoring_package_digests: Sequence[str],
    min_forward_cells: int,
    rank_ic_bar: float,
    placebo_abs_limit: float,
    frozen_at: datetime,
) -> StrategyManifest:
    if not expected_cells or len(expected_cells) != len(set(expected_cells)):
        raise ValueError("expected evaluation cells must be non-empty and unique")
    if not scoring_package_digests or len(scoring_package_digests) != len(set(scoring_package_digests)):
        raise ValueError("scoring package digests must be non-empty and unique")
    return StrategyManifest(
        strategy_id="track-b-forward-evaluation",
        lane_id="track_b_evaluation",
        experiment_id=f"track-b-evaluation-{campaign_id}",
        hypothesis="PIT filing scores retain cross-sectional forward rank IC beyond a clean placebo.",
        exploratory=True,
        frozen_at=frozen_at,
        universe=tuple(sorted(expected_cells)),
        causal_lag="score as_of_date precedes rebalance; forward return begins after rebalance",
        parameters={
            "campaign_id": campaign_id,
            "expected_cells": sorted(expected_cells),
            "scoring_package_digests": sorted(scoring_package_digests),
            "min_forward_cells": min_forward_cells,
            "rank_ic_bar": rank_ic_bar,
            "placebo_abs_limit": placebo_abs_limit,
        },
        cost_model={"basis": "rank IC evaluates signal; portfolio costs remain in downstream book gate"},
        evaluation_windows=("score artifact × rebalance date × fold",),
        search_space={"note": "frozen rank-IC and deterministic placebo; no tuning"},
        trial_budget=1,
        metrics=("n_cells", "n_valid_forward_cells", "mean_rank_ic", "mean_placebo_rank_ic"),
        promotion_criteria=(
            "adequate forward cells, mean rank IC >= bar, and absolute placebo rank IC <= limit",
        ),
    )


def build_job_manifest(
    *,
    job_id: str,
    dataset_manifest: DatasetManifest,
    strategy_manifest: StrategyManifest,
    capability_profile: CapabilityProfile,
    git_commit: str,
    container_digest: str,
    configuration_digest: str,
    feature_pipeline_version: str,
    created_at: datetime,
    expected_outputs: Sequence[str],
    assigned_unit: str,
    resource_class: ResourceClass,
) -> JobManifest:
    if not expected_outputs or len(expected_outputs) != len(set(expected_outputs)):
        raise ValueError("expected outputs must be non-empty and unique")
    return JobManifest(
        job_id=job_id,
        git_commit=git_commit,
        container_digest=container_digest,
        dataset_manifest_digests=(content_digest(dataset_manifest),),
        strategy_manifest_digest=content_digest(strategy_manifest),
        feature_pipeline_version=feature_pipeline_version,
        configuration_digest=configuration_digest,
        random_seeds=(0,),
        assigned_unit=assigned_unit,
        resource_class=resource_class,
        capability_profile_digest=content_digest(capability_profile),
        trial_budget=1,
        expected_outputs=tuple(sorted(expected_outputs)),
        created_at=created_at,
    )


__all__ = [
    "build_capability_profile", "build_dataset_manifest",
    "build_scoring_strategy_manifest", "build_evaluation_strategy_manifest",
    "build_job_manifest",
]
