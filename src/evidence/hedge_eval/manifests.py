"""Dataset / capability / strategy / job manifest construction for the hedge
evaluation slice.

Pure builders over the formal evidence contracts. No scoring, no I/O side
effects, no authority — a manifest only *declares* a run; it never executes
one.

Unlike the risk-target slice (a model-training workload described by a
``ModelTrainingSpec``), a hedge evaluation is a *strategy* workload: it scores
an already-logged raw-vs-hedged ledger. It therefore fills the other half of
the ``JobManifest`` contract — a signed ``StrategyManifest`` and its digest —
exercising the exactly-one-workload-spec branch the risk-target slice does not.
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

# Immutable safety facts for this workload, asserted into every job's
# capability profile. Remote compute may read approved ledger snapshots and
# write job-scoped results and nothing else (roadmap §4.4).
HEDGE_EVAL_CAPABILITY_PROFILE_ID = "axiom-hedge-eval-worker"

HEDGE_STRATEGY_ID = "axiom-hedge-overlay-evaluation"
HEDGE_LANE_ID = "hedge_evaluation"

# Scorecard metric surface the report exposes for display + independent replay.
HEDGE_METRIC_IDS: tuple[str, ...] = (
    "n_cycles",
    "resolved_raw_cycles",
    "raw_after_cost_expectancy",
    "raw_max_drawdown",
    "raw_sharpe_annualized",
    "hedged_net_after_cost_expectancy",
    "hedged_net_max_drawdown",
    "hedged_net_sharpe_annualized",
    "exposure_net_beta_mean",
)


def build_capability_profile(
    *,
    profile_id: str = HEDGE_EVAL_CAPABILITY_PROFILE_ID,
) -> CapabilityProfile:
    """The worker's declared capabilities.

    Read scope is limited to the approved hedge-ledger snapshot domain; write
    scope is the job's own output namespace; there are no network endpoints.
    Every authority flag is ``Literal[False]`` by the contract's own type, so a
    worker cannot even *express* broker, order, halt, live-gate, champion, or
    self-approval authority.
    """
    return CapabilityProfile(
        profile_id=profile_id,
        read_scopes=("axiom-data/hedge/snapshots/**",),
        write_scopes=("job:{job_id}/outputs/**",),
        network_endpoints=(),
    )


def build_hedge_ledger_dataset_manifest(
    partitions: Mapping[str, bytes],
    *,
    dataset_id: str,
    retrieved_at: datetime,
    coverage_start: date,
    coverage_end: date,
    provider: str = "hedged_shadow_lane",
    data_schema_version: str = "raw_vs_hedged_ledger.v1",
    point_in_time_rules: Sequence[str] = (
        "each ledger row records realized raw & hedged returns as-of its cycle; no forward look-ahead",
    ),
    row_counts: Mapping[str, int] | None = None,
) -> DatasetManifest:
    """Build a ``DatasetManifest`` over the per-strategy raw-vs-hedged ledger.

    Each strategy's JSONL bytes become one content-addressed ``PartitionRef``
    (``digest`` = SHA-256 of the exact bytes the worker will be handed). A
    single changed byte in any partition changes its digest and therefore the
    manifest digest — the hash rail the acceptance tests exercise.
    """
    if not partitions:
        raise ValueError("at least one strategy ledger partition is required")
    row_counts = row_counts or {}
    refs: list[PartitionRef] = []
    for strategy in sorted(partitions):
        data = partitions[strategy]
        refs.append(
            PartitionRef(
                partition_id=strategy,
                uri=f"axiom-data/hedge/snapshots/{dataset_id}/partitions/{strategy}.jsonl",
                digest=sha256_bytes(data),
                size_bytes=len(data),
                row_count=row_counts.get(strategy),
            )
        )
    return DatasetManifest(
        dataset_id=dataset_id,
        asset_class=AssetClass.MULTI_ASSET,
        domain="hedge_raw_vs_hedged_ledger",
        provider=provider,
        retrieved_at=retrieved_at,
        coverage_start=coverage_start,
        coverage_end=coverage_end,
        instruments=tuple(sorted(partitions)),
        data_schema_version=data_schema_version,
        point_in_time_rules=tuple(point_in_time_rules),
        partitions=tuple(refs),
    )


def build_hedge_strategy_manifest(
    strategies: Sequence[str],
    *,
    frozen_at: datetime,
    min_history: int,
    strategy_id: str = HEDGE_STRATEGY_ID,
    lane_id: str = HEDGE_LANE_ID,
) -> StrategyManifest:
    """Freeze the raw-vs-hedged alpha-vs-beta evaluation program."""
    if not strategies:
        raise ValueError("at least one strategy is required in the hedge evaluation universe")
    return StrategyManifest(
        strategy_id=strategy_id,
        lane_id=lane_id,
        experiment_id="hedge-raw-vs-hedged-alpha-vs-beta",
        hypothesis=(
            "For each shadow strategy, hedging its market/sector beta improves after-cost "
            "expectancy and cuts drawdown iff the residual is genuine alpha; a return that "
            "dies once hedged was beta, not alpha."
        ),
        exploratory=True,
        frozen_at=frozen_at,
        universe=tuple(sorted(strategies)),
        causal_lag=(
            "each cycle scores realized raw & hedged returns already logged; the ledger row "
            "is the as-of record, so there is no forward look-ahead"
        ),
        parameters={
            "min_history_for_verdict": min_history,
            "annualization": "median calendar-day cadence per lane",
        },
        cost_model={
            "basis": "after-cost net_return from the ledger",
            "cost_unmodeled_venues": "reported on a gross basis and can never clear the after-cost gate",
        },
        evaluation_windows=("full logged raw-vs-hedged ledger history",),
        search_space={"note": "no search — deterministic scorecard over logged cycles"},
        trial_budget=1,
        metrics=HEDGE_METRIC_IDS,
        promotion_criteria=(
            "hedge admissible for quarantine iff verdict == signal_real_but_noisy on an "
            "after-cost (cost-known) basis with resolved_raw_cycles >= min_history",
        ),
    )


def build_hedge_evaluation_job_manifest(
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
    expected_strategies: Sequence[str],
    assigned_unit: str = "hedge_eval/all_strategies/full_history",
    resource_class: ResourceClass = ResourceClass.CPU_SMALL,
    trial_budget: int = 1,
) -> JobManifest:
    """Assemble the signed-job payload binding data, strategy spec, config, outputs.

    The hedge scorecard has no stochastic component, so ``random_seeds`` carries
    the degenerate ``(0,)`` — the contract requires a seed, but the workload is
    fully deterministic float arithmetic over the logged rows.
    """
    if not expected_strategies:
        raise ValueError("at least one expected strategy output is required")
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
        expected_outputs=tuple(sorted(expected_strategies)),
        created_at=created_at,
    )


def sign(signer: Ed25519Signer, payload, *, created_at: datetime) -> SignedEnvelope:
    """Sign one contract payload with a producer/authority key."""
    return signer.sign(payload, created_at=created_at)


__all__ = [
    "HEDGE_EVAL_CAPABILITY_PROFILE_ID",
    "HEDGE_STRATEGY_ID",
    "HEDGE_LANE_ID",
    "HEDGE_METRIC_IDS",
    "build_capability_profile",
    "build_hedge_ledger_dataset_manifest",
    "build_hedge_strategy_manifest",
    "build_hedge_evaluation_job_manifest",
    "sign",
]
