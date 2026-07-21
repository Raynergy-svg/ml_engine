"""Dataset / capability / strategy / job manifest construction for the equity
research evaluation slice.

Pure builders over the formal evidence contracts. No scoring, no I/O side
effects, no authority — a manifest only *declares* a run; it never executes one.

Like the hedge slice (and unlike the risk-target slice's model-training
workload), an equity-research evaluation is a *strategy/research* workload: it
scores an already-computed per-fold cross-section of research scores against
realized forward returns. It therefore fills the ``StrategyManifest`` half of the
``JobManifest`` contract. What is new here is the fold granularity: each dataset
partition is ONE fold (``{tier}::fold{i}``), and the strategy manifest declares
the exact expected fold set per universe tier so the deterministic aggregator can
refuse an incomplete or duplicated aggregation.
"""

from __future__ import annotations

from datetime import date, datetime
from typing import Mapping, Sequence

from src.evidence.contracts import (
    AssetClass,
    CapabilityProfile,
    DatasetManifest,
    JobManifest,
    PartitionRef,
    ResourceClass,
    SignedEnvelope,
    StrategyManifest,
)
from src.evidence.hashing import content_digest, sha256_bytes
from src.evidence.signing import Ed25519Signer

# Immutable safety facts for this workload. Remote compute may read approved
# research-score snapshots and write job-scoped results and nothing else
# (roadmap §4.4).
EQUITY_RESEARCH_CAPABILITY_PROFILE_ID = "axiom-equity-research-worker"

EQUITY_RESEARCH_STRATEGY_ID = "axiom-equity-research-cross-sectional-alpha"
EQUITY_RESEARCH_LANE_ID = "equity_research"

# The three universe tiers the roadmap requires reported SEPARATELY (J2). Each is
# an independent evidence head; the branch documents that the attractive curated
# result and the defensible wide result are materially different.
UNIVERSE_TIERS: tuple[str, ...] = ("curated", "wide", "full_history")

# Scorecard metric surface the report exposes for display + independent replay.
EQUITY_RESEARCH_METRIC_IDS: tuple[str, ...] = (
    "n_folds",
    "n_oos_folds",
    "breadth_mean_names",
    "pooled_oos_rank_ic",
    "pooled_is_rank_ic",
    "oos_ic_tstat",
    "mean_oos_q5_q1_spread",
)


def build_capability_profile(
    *,
    profile_id: str = EQUITY_RESEARCH_CAPABILITY_PROFILE_ID,
) -> CapabilityProfile:
    """The worker's declared capabilities.

    Read scope is limited to the approved research-score snapshot domain; write
    scope is the job's own output namespace; there are no network endpoints.
    Every authority flag is ``Literal[False]`` by the contract's own type, so a
    worker cannot even *express* broker, order, halt, live-gate, champion, or
    self-approval authority.
    """
    return CapabilityProfile(
        profile_id=profile_id,
        read_scopes=("axiom-data/equity_research/snapshots/**",),
        write_scopes=("job:{job_id}/outputs/**",),
        network_endpoints=(),
    )


def fold_partition_id(tier: str, fold_index: int) -> str:
    """Canonical partition id for one fold of one universe tier."""
    return f"{tier}::fold{fold_index}"


def build_research_fold_dataset_manifest(
    partitions: Mapping[str, bytes],
    *,
    dataset_id: str,
    retrieved_at: datetime,
    coverage_start: date,
    coverage_end: date,
    provider: str = "equity_research_pipeline",
    data_schema_version: str = "research_fold_cross_section.v1",
    point_in_time_rules: Sequence[str] = (
        "each fold row records a PIT research score (as_of <= fold start) and the "
        "realized forward return over the fold window; no forward look-ahead",
    ),
    row_counts: Mapping[str, int] | None = None,
) -> DatasetManifest:
    """Build a ``DatasetManifest`` where each partition is one fold.

    Each fold's JSONL bytes become one content-addressed ``PartitionRef``
    (``digest`` = SHA-256 of the exact bytes the worker will be handed). A single
    changed byte in any fold partition changes its digest and therefore the
    manifest digest — the hash rail the acceptance tests exercise.
    """
    if not partitions:
        raise ValueError("at least one fold partition is required")
    row_counts = row_counts or {}
    refs: list[PartitionRef] = []
    for fold_id in sorted(partitions):
        data = partitions[fold_id]
        refs.append(
            PartitionRef(
                partition_id=fold_id,
                uri=f"axiom-data/equity_research/snapshots/{dataset_id}/folds/{fold_id}.jsonl",
                digest=sha256_bytes(data),
                size_bytes=len(data),
                row_count=row_counts.get(fold_id),
            )
        )
    return DatasetManifest(
        dataset_id=dataset_id,
        asset_class=AssetClass.EQUITY,
        domain="equity_research_fold_cross_section",
        provider=provider,
        retrieved_at=retrieved_at,
        coverage_start=coverage_start,
        coverage_end=coverage_end,
        instruments=tuple(sorted(partitions)),
        data_schema_version=data_schema_version,
        point_in_time_rules=tuple(point_in_time_rules),
        partitions=tuple(refs),
    )


def build_research_strategy_manifest(
    hypothesis_id: str,
    expected_folds_by_tier: Mapping[str, Sequence[str]],
    *,
    frozen_at: datetime,
    min_oos_folds: int,
    ic_bar: float,
    ic_tstat_bar: float,
    strategy_id: str = EQUITY_RESEARCH_STRATEGY_ID,
    lane_id: str = EQUITY_RESEARCH_LANE_ID,
) -> StrategyManifest:
    """Freeze the cross-sectional research-alpha evaluation program.

    The declared ``expected_folds_by_tier`` is signed into the manifest and is
    the authority the aggregator refuses incomplete/duplicate aggregations
    against — an omitted fold cannot be silently scored as a complete tier.
    """
    tiers = sorted(expected_folds_by_tier)
    if not tiers:
        raise ValueError("at least one universe tier is required")
    for tier in tiers:
        if not expected_folds_by_tier[tier]:
            raise ValueError(f"tier {tier!r} declares no expected folds")
    declared = {tier: sorted(expected_folds_by_tier[tier]) for tier in tiers}
    return StrategyManifest(
        strategy_id=strategy_id,
        lane_id=lane_id,
        experiment_id=f"equity-research-{hypothesis_id}",
        hypothesis=(
            "A pre-registered cross-sectional research composite predicts forward "
            "returns; reported separately for the curated, wide survivorship-"
            "corrected, and full-history universes, since the attractive curated "
            "result and the defensible wide result are materially different."
        ),
        exploratory=True,
        frozen_at=frozen_at,
        universe=tuple(tiers),
        causal_lag=(
            "each fold scores PIT research composites (as_of <= fold start) against "
            "the realized forward return over the fold window; no look-ahead"
        ),
        parameters={
            "hypothesis_id": hypothesis_id,
            "expected_folds_by_tier": declared,
            "min_oos_folds_for_verdict": min_oos_folds,
            "ic_bar": ic_bar,
            "ic_tstat_bar": ic_tstat_bar,
            "composite": "frozen PRIMARY_WEIGHTS (src.equity.research.contracts)",
        },
        cost_model={
            "basis": "rank-IC and Q5-Q1 spread are gross signal-strength measures",
            "note": "portfolio-level net costs are evaluated downstream by the allocator/book gate",
        },
        evaluation_windows=("per-fold out-of-sample (post model-cutoff) cross-sections",),
        search_space={"note": "no search — deterministic rank-IC aggregation over declared folds"},
        trial_budget=1,
        metrics=EQUITY_RESEARCH_METRIC_IDS,
        promotion_criteria=(
            "tier admissible for quarantine iff its pooled out-of-sample rank-IC "
            "clears ic_bar AND the across-fold IC t-statistic clears ic_tstat_bar",
        ),
    )


def build_research_evaluation_job_manifest(
    *,
    job_id: str,
    dataset_manifest: DatasetManifest,
    strategy_manifest: StrategyManifest,
    capability_profile: CapabilityProfile,
    git_commit: str,
    container_digest: str,
    configuration_digest: str,
    scorecard_version: str,
    created_at: datetime,
    expected_tier_lanes: Sequence[str],
    assigned_unit: str = "equity_research/all_tiers/all_folds",
    resource_class: ResourceClass = ResourceClass.CPU_SMALL,
    trial_budget: int = 1,
) -> JobManifest:
    """Assemble the signed-job payload binding data, strategy spec, config, outputs.

    The rank-IC aggregation has no stochastic component, so ``random_seeds``
    carries the degenerate ``(0,)`` — the contract requires a seed, but the
    workload is fully deterministic float arithmetic over the declared folds.
    ``expected_outputs`` are the per-tier evidence lane ids (one head per tier).
    """
    if not expected_tier_lanes:
        raise ValueError("at least one expected tier lane output is required")
    return JobManifest(
        job_id=job_id,
        git_commit=git_commit,
        container_digest=container_digest,
        dataset_manifest_digests=(content_digest(dataset_manifest),),
        strategy_manifest_digest=content_digest(strategy_manifest),
        feature_pipeline_version=scorecard_version,
        configuration_digest=configuration_digest,
        random_seeds=(0,),
        assigned_unit=assigned_unit,
        resource_class=resource_class,
        capability_profile_digest=content_digest(capability_profile),
        trial_budget=trial_budget,
        expected_outputs=tuple(sorted(expected_tier_lanes)),
        created_at=created_at,
    )


def sign(signer: Ed25519Signer, payload, *, created_at: datetime) -> SignedEnvelope:
    """Sign one contract payload with a producer/authority key."""
    return signer.sign(payload, created_at=created_at)


__all__ = [
    "EQUITY_RESEARCH_CAPABILITY_PROFILE_ID",
    "EQUITY_RESEARCH_STRATEGY_ID",
    "EQUITY_RESEARCH_LANE_ID",
    "UNIVERSE_TIERS",
    "EQUITY_RESEARCH_METRIC_IDS",
    "build_capability_profile",
    "fold_partition_id",
    "build_research_fold_dataset_manifest",
    "build_research_strategy_manifest",
    "build_research_evaluation_job_manifest",
    "sign",
]
