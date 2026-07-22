"""Frozen manifests for the portfolio-book allocation workload.

The book's ``StrategyManifest.parameters`` binds every contributing sleeve_id
to the digest of the already-QUARANTINED upstream EvidencePackage it was
computed from (``source_package_digests``) — the same derivation-binding
pattern Track B uses for its evaluation stage (``scoring_package_digests``).
The local importer re-checks this binding and that every referenced source
package is independently QUARANTINED in the local store before it will accept
the combined-book package for its own quarantine.
"""

from __future__ import annotations

from datetime import date, datetime
from typing import Mapping, Sequence

from src.evidence.contracts import (
    AssetClass, CapabilityProfile, DatasetManifest, JobManifest,
    PartitionRef, ResourceClass, StrategyManifest,
)
from src.evidence.hashing import content_digest, sha256_bytes

from .evaluation import EvaluationParams, METRICS
from .models import LANE_ID

CAPABILITY_PROFILE_ID = "axiom-portfolio-book-worker"


def build_capability_profile() -> CapabilityProfile:
    return CapabilityProfile(
        profile_id=CAPABILITY_PROFILE_ID,
        read_scopes=("axiom-data/portfolio_book/sleeve_returns/**",),
        write_scopes=("job:{job_id}/outputs/**",),
        network_endpoints=(),
    )


def build_dataset_manifest(
    partitions: Mapping[str, bytes], *, dataset_id: str, retrieved_at: datetime,
    coverage_start: date, coverage_end: date,
    row_counts: Mapping[str, int] | None = None,
) -> DatasetManifest:
    if len(partitions) < 2:
        raise ValueError("a portfolio book requires at least 2 sleeve return partitions")
    row_counts = row_counts or {}
    refs = tuple(
        PartitionRef(
            partition_id=sleeve_id,
            uri=f"axiom-data/portfolio_book/sleeve_returns/{dataset_id}/{sleeve_id}.jsonl",
            digest=sha256_bytes(data), size_bytes=len(data),
            row_count=row_counts.get(sleeve_id),
        )
        for sleeve_id, data in sorted(partitions.items())
    )
    return DatasetManifest(
        dataset_id=dataset_id,
        asset_class=AssetClass.PORTFOLIO,
        domain="portfolio_book_sleeve_returns",
        provider="axiom_evidence_lanes",
        retrieved_at=retrieved_at,
        coverage_start=coverage_start,
        coverage_end=coverage_end,
        instruments=tuple(sorted(partitions)),
        data_schema_version="portfolio_book_sleeve_return.v1",
        point_in_time_rules=(
            "each sleeve return series is drawn only from its own already-QUARANTINED evidence package",
            "allocation weights are fit only on dates strictly before the frozen oos_start cutoff",
            "the book is scored only on dates at or after oos_start, using weights held static (no in-window rebalancing)",
        ),
        partitions=refs,
    )


def _params_to_dict(params: EvaluationParams) -> dict[str, object]:
    return {
        "oos_start": params.oos_start,
        "min_est_days": params.min_est_days,
        "min_oos_days": params.min_oos_days,
        "cash_reserve_frac": params.cash_reserve_frac,
        "max_allocation": params.max_allocation,
        "min_allocation": params.min_allocation,
        "min_sleeve_vol": params.min_sleeve_vol,
        "lane_capacity": dict(params.lane_capacity),
        "tail_corr_threshold": params.tail_corr_threshold,
        "tail_quantile": params.tail_quantile,
        "drawdown_budget": params.drawdown_budget,
        "max_book_drawdown": params.max_book_drawdown,
        "min_book_sharpe": params.min_book_sharpe,
        "erc_max_iterations": params.erc_max_iterations,
        "erc_tolerance": params.erc_tolerance,
        "capacity_max_rounds": params.capacity_max_rounds,
        "replay_tolerance": params.replay_tolerance,
    }


def params_from_dict(payload: Mapping[str, object]) -> EvaluationParams:
    return EvaluationParams(
        oos_start=str(payload["oos_start"]),
        min_est_days=int(payload["min_est_days"]),
        min_oos_days=int(payload["min_oos_days"]),
        cash_reserve_frac=float(payload["cash_reserve_frac"]),
        max_allocation=float(payload["max_allocation"]),
        min_allocation=float(payload["min_allocation"]),
        min_sleeve_vol=float(payload["min_sleeve_vol"]),
        lane_capacity={str(k): float(v) for k, v in dict(payload["lane_capacity"]).items()},
        tail_corr_threshold=float(payload["tail_corr_threshold"]),
        tail_quantile=float(payload["tail_quantile"]),
        drawdown_budget=float(payload["drawdown_budget"]),
        max_book_drawdown=float(payload["max_book_drawdown"]),
        min_book_sharpe=float(payload["min_book_sharpe"]),
        erc_max_iterations=int(payload["erc_max_iterations"]),
        erc_tolerance=float(payload["erc_tolerance"]),
        capacity_max_rounds=int(payload["capacity_max_rounds"]),
        replay_tolerance=float(payload["replay_tolerance"]),
    )


def build_strategy_manifest(
    *, strategy_id: str, experiment_id: str, sleeve_ids: Sequence[str],
    source_package_digests: Mapping[str, str], params: EvaluationParams, frozen_at: datetime,
) -> StrategyManifest:
    sleeve_ids = tuple(sorted(sleeve_ids))
    if set(source_package_digests) != set(sleeve_ids):
        raise ValueError("source_package_digests must have exactly one entry per sleeve_id")
    return StrategyManifest(
        strategy_id=strategy_id,
        lane_id=LANE_ID,
        experiment_id=experiment_id,
        hypothesis=(
            "a correlation-aware, capacity- and drawdown-budget-constrained combination of "
            "already-QUARANTINED sleeves (HRP across sleeves, pre-registered) improves the "
            "risk-adjusted OOS profile over any single sleeve without breaching per-lane capacity, "
            "the cash reserve, tail-correlation limits, or per-sleeve drawdown budgets"
        ),
        exploratory=False,
        frozen_at=frozen_at,
        universe=sleeve_ids,
        causal_lag="allocation weights are fit only on dates < oos_start; the book is scored only on dates >= oos_start",
        parameters={
            **_params_to_dict(params),
            "source_package_digests": dict(sorted(source_package_digests.items())),
        },
        cost_model={"look_through_turnover": "weighted mean of each sleeve's own reported turnover, no additional rebalancing cost modeled"},
        evaluation_windows=("estimation_window", "oos_window"),
        search_space={
            "allocation_methods_evaluated": ["inverse_volatility", "equal_risk_contribution", "correlation_penalized", "hrp_across_sleeves"],
            "final_policy": "hrp_across_sleeves",
            "trial_budget": 1,
        },
        trial_budget=1,
        metrics=METRICS,
        promotion_criteria=(
            "combined-book OOS Sharpe must clear the frozen floor",
            "combined-book OOS drawdown must not exceed the frozen ceiling",
            "no sleeve may exceed its declared lane capacity",
            "allocated weight must not exceed 1 minus the cash reserve",
            "tail-day correlation between surviving active sleeves must clear after exclusion",
            "each sleeve's weight times its estimation-window max drawdown must stay within its risk budget",
            "accepted evidence remains quarantined; installation requires separate operator-authorized promotion",
            "no lane, and no book, becomes capital-active from this gate alone",
        ),
    )


def build_job_manifest(
    *, job_id: str, dataset_manifest: DatasetManifest, strategy_manifest: StrategyManifest,
    capability_profile: CapabilityProfile, git_commit: str, container_digest: str,
    configuration_digest: str, feature_pipeline_version: str, created_at: datetime,
    random_seeds: Sequence[int] = (0,), trial_budget: int = 1,
) -> JobManifest:
    if trial_budget != 1:
        raise ValueError("portfolio-book v1 is preregistered for exactly one deterministic trial")
    return JobManifest(
        job_id=job_id,
        git_commit=git_commit,
        container_digest=container_digest,
        dataset_manifest_digests=(content_digest(dataset_manifest),),
        strategy_manifest_digest=content_digest(strategy_manifest),
        feature_pipeline_version=feature_pipeline_version,
        configuration_digest=configuration_digest,
        random_seeds=tuple(random_seeds),
        assigned_unit=f"portfolio_book/combined_book_oos/{strategy_manifest.strategy_id}",
        resource_class=ResourceClass.CPU_SMALL,
        capability_profile_digest=content_digest(capability_profile),
        trial_budget=trial_budget,
        expected_outputs=(LANE_ID,),
        created_at=created_at,
    )


__all__ = [
    "CAPABILITY_PROFILE_ID", "build_capability_profile", "build_dataset_manifest",
    "build_strategy_manifest", "build_job_manifest", "params_from_dict",
]
