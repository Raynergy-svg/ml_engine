"""Dataset / capability / strategy / job manifest construction for the crypto
momentum evaluation slice.

Pure builders over the formal evidence contracts. No scoring, no I/O side
effects, no authority — a manifest only *declares* a run; it never executes one.

Like the equity-research and hedge slices (and unlike the risk-target slice's
model-training workload), a crypto-momentum evaluation is a *strategy/research*
workload: it scores an already-computed per-cell forward-return series against
the frozen construction's admissibility bars. It therefore fills the
``StrategyManifest`` half of the ``JobManifest`` contract. The fold granularity
is ``construction × fold × stress``: each dataset partition is ONE cell
(``{construction}::{stress}::fold{i}``), and the strategy manifest declares the
exact expected cell set per construction so the deterministic aggregator can
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
# crypto shadow-ledger snapshots and write job-scoped results and nothing else
# (roadmap §4.4).
CRYPTO_MOMENTUM_CAPABILITY_PROFILE_ID = "axiom-crypto-momentum-worker"

CRYPTO_MOMENTUM_STRATEGY_ID = "axiom-crypto-xs-momentum-forward"
CRYPTO_MOMENTUM_LANE_ID = "crypto_momentum"

# Scorecard metric surface the report exposes for display + independent replay.
CRYPTO_MOMENTUM_METRIC_IDS: tuple[str, ...] = (
    "n_folds",
    "n_baseline_oos_folds",
    "pooled_oos_sharpe",
    "pooled_is_sharpe",
    "oos_sharpe_tstat",
    "pooled_oos_max_drawdown",
    "drop_one_oos_sharpe",
)


def build_capability_profile(
    *,
    profile_id: str = CRYPTO_MOMENTUM_CAPABILITY_PROFILE_ID,
) -> CapabilityProfile:
    """The worker's declared capabilities.

    Read scope is limited to the approved crypto-momentum shadow snapshot domain;
    write scope is the job's own output namespace; there are no network
    endpoints. Every authority flag is ``Literal[False]`` by the contract's own
    type, so a worker cannot even *express* broker, order, halt, live-gate,
    champion, or self-approval authority.
    """
    return CapabilityProfile(
        profile_id=profile_id,
        read_scopes=("axiom-data/crypto_momentum/snapshots/**",),
        write_scopes=("job:{job_id}/outputs/**",),
        network_endpoints=(),
    )


FORWARD_LEDGER_PREFIX = "forward_ledger::"
_RESERVED_ID_SEGMENT = "forward_ledger"  # Code Reviewer finding (crypto_carry sibling), 2026-07-22


def _reject_reserved_id(construction: str) -> None:
    """A construction literally equal to ``"forward_ledger"`` would make
    ``cell_partition_id(construction, ...)`` start with
    ``FORWARD_LEDGER_PREFIX``, misclassifying every one of its cells as
    ledger data in ``evaluate_partitions``. Fails loud (a ValueError from the
    ledger-set mismatch check), never silently, but is still worth refusing
    outright."""
    if construction == _RESERVED_ID_SEGMENT:
        raise ValueError(
            f"construction cannot be {_RESERVED_ID_SEGMENT!r} — reserved for forward-ledger partition ids"
        )


def cell_partition_id(construction: str, stress: str, fold_index: int) -> str:
    """Canonical partition id for one cell of one construction under one stress."""
    if not construction or not stress or "::" in construction or "::" in stress:
        raise ValueError(
            "construction and stress must be non-empty and cannot contain '::'"
        )
    if isinstance(fold_index, bool) or not isinstance(fold_index, int) or fold_index < 0:
        raise ValueError("fold_index must be a non-negative integer")
    _reject_reserved_id(construction)
    return f"{construction}::{stress}::fold{fold_index}"


def forward_ledger_partition_id(construction: str) -> str:
    """Canonical partition id for one construction's REAL forward-shadow ledger
    (roadmap §14 standardized return contract source) — reserved prefix keeps
    it unambiguously distinct from ``{construction}::{stress}::fold{i}`` cell
    ids everywhere partitions are classified (evaluation.py, local_import.py)."""
    if not construction or "::" in construction:
        raise ValueError("construction must be non-empty and cannot contain '::'")
    _reject_reserved_id(construction)
    return f"{FORWARD_LEDGER_PREFIX}{construction}"


def build_momentum_cell_dataset_manifest(
    partitions: Mapping[str, bytes],
    *,
    dataset_id: str,
    retrieved_at: datetime,
    coverage_start: date,
    coverage_end: date,
    provider: str = "crypto_momentum_shadow_ledger",
    data_schema_version: str = "crypto_momentum_cell_returns.v1",
    point_in_time_rules: Sequence[str] = (
        "each cell row is a realized forward net-return period; out-of-sample "
        "cells (in_sample=False) are post-activation shadow periods; no look-ahead",
    ),
    row_counts: Mapping[str, int] | None = None,
) -> DatasetManifest:
    """Build a ``DatasetManifest`` where each partition is one cell.

    Each cell's JSONL bytes become one content-addressed ``PartitionRef``
    (``digest`` = SHA-256 of the exact bytes the worker will be handed). A single
    changed byte in any cell partition changes its digest and therefore the
    manifest digest — the hash rail the acceptance tests exercise.
    """
    if not partitions:
        raise ValueError("at least one cell partition is required")
    row_counts = row_counts or {}
    refs: list[PartitionRef] = []
    for cell_id in sorted(partitions):
        data = partitions[cell_id]
        refs.append(
            PartitionRef(
                partition_id=cell_id,
                uri=f"axiom-data/crypto_momentum/snapshots/{dataset_id}/cells/{cell_id}.jsonl",
                digest=sha256_bytes(data),
                size_bytes=len(data),
                row_count=row_counts.get(cell_id),
            )
        )
    return DatasetManifest(
        dataset_id=dataset_id,
        asset_class=AssetClass.CRYPTO,
        domain="crypto_momentum_cell_returns",
        provider=provider,
        retrieved_at=retrieved_at,
        coverage_start=coverage_start,
        coverage_end=coverage_end,
        instruments=tuple(sorted(partitions)),
        data_schema_version=data_schema_version,
        point_in_time_rules=tuple(point_in_time_rules),
        partitions=tuple(refs),
    )


def build_momentum_strategy_manifest(
    campaign_id: str,
    expected_cells_by_construction: Mapping[str, Sequence[str]],
    *,
    frozen_at: datetime,
    min_oos_folds: int,
    sharpe_bar: float,
    sharpe_tstat_bar: float,
    max_drawdown_limit: float,
    stress_sharpe_floor: float,
    drop_one_sharpe_floor: float,
    strategy_id: str = CRYPTO_MOMENTUM_STRATEGY_ID,
    lane_id: str = CRYPTO_MOMENTUM_LANE_ID,
    forward_ledger_constructions: Sequence[str] = (),
) -> StrategyManifest:
    """Freeze the crypto XS-momentum forward-evaluation program.

    The declared ``expected_cells_by_construction`` is signed into the manifest
    and is the authority the aggregator refuses incomplete/duplicate aggregations
    against — an omitted cell cannot be silently scored as a complete
    construction. ``forward_ledger_constructions`` is the SAME kind of signed
    declaration for which constructions' packages must carry the roadmap §14
    return contract (built from their real forward-shadow ledger) — a
    construction can be silently missing its promised ledger data no more than
    it can be missing a declared cell.
    """
    constructions = sorted(expected_cells_by_construction)
    if not constructions:
        raise ValueError("at least one construction is required")
    for construction in constructions:
        _reject_reserved_id(construction)
        cells = expected_cells_by_construction[construction]
        if not cells:
            raise ValueError(f"construction {construction!r} declares no expected cells")
        if len(cells) != len(set(cells)):
            raise ValueError(f"construction {construction!r} declares duplicate expected cells")
    declared = {c: sorted(expected_cells_by_construction[c]) for c in constructions}
    ledger_constructions = sorted(set(forward_ledger_constructions))
    if len(ledger_constructions) != len(forward_ledger_constructions):
        raise ValueError("forward_ledger_constructions must not contain duplicates")
    unknown = set(ledger_constructions) - set(constructions)
    if unknown:
        raise ValueError(f"forward_ledger_constructions references undeclared constructions: {sorted(unknown)}")
    return StrategyManifest(
        strategy_id=strategy_id,
        lane_id=lane_id,
        experiment_id=f"crypto-momentum-{campaign_id}",
        hypothesis=(
            "A frozen cross-sectional crypto-momentum construction earns a "
            "statistically distinguishable, cost-and-funding-aware, drawdown-"
            "bounded, stress-surviving forward edge. The pre-registration's "
            "significance did not clear, so the lane stays shadow-only and an "
            "honest forward scorecard must not admit it."
        ),
        exploratory=True,
        frozen_at=frozen_at,
        universe=tuple(constructions),
        causal_lag=(
            "each cell scores a realized forward net-return series; out-of-sample "
            "cells are post-activation shadow periods; no look-ahead"
        ),
        parameters={
            "campaign_id": campaign_id,
            "expected_cells_by_construction": declared,
            "min_oos_folds_for_verdict": min_oos_folds,
            "sharpe_bar": sharpe_bar,
            "sharpe_tstat_bar": sharpe_tstat_bar,
            "max_drawdown_limit": max_drawdown_limit,
            "stress_sharpe_floor": stress_sharpe_floor,
            "drop_one_sharpe_floor": drop_one_sharpe_floor,
            "construction": "frozen H4 XS-momentum (src.crypto.momentum_shadow.construction_manifest)",
            "forward_ledger_constructions": ledger_constructions,
        },
        cost_model={
            "basis": "net Sharpe is computed on funding-aware, cost-deducted per-period returns",
            "note": "explicit costs are already inside each cell's net_return; no further deduction here",
        },
        evaluation_windows=("per-fold out-of-sample (post-activation) forward return series",),
        search_space={"note": "no search — deterministic Sharpe aggregation over declared cells"},
        trial_budget=1,
        metrics=CRYPTO_MOMENTUM_METRIC_IDS,
        promotion_criteria=(
            "construction admissible for quarantine iff its baseline pooled OOS "
            "Sharpe clears sharpe_bar with an across-fold t-stat clearing "
            "sharpe_tstat_bar, drawdown within max_drawdown_limit, drop-one-fold "
            "robustness holds, AND every adverse stress scenario survives",
        ),
    )


def build_momentum_evaluation_job_manifest(
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
    expected_construction_lanes: Sequence[str],
    assigned_unit: str = "crypto_momentum/all_constructions/all_cells",
    resource_class: ResourceClass = ResourceClass.CPU_SMALL,
    trial_budget: int = 1,
) -> JobManifest:
    """Assemble the signed-job payload binding data, strategy spec, config, outputs.

    The Sharpe aggregation has no stochastic component, so ``random_seeds``
    carries the degenerate ``(0,)`` — the contract requires a seed, but the
    workload is fully deterministic float arithmetic over the declared cells.
    ``expected_outputs`` are the per-construction evidence lane ids.
    """
    if not expected_construction_lanes:
        raise ValueError("at least one expected construction lane output is required")
    if len(expected_construction_lanes) != len(set(expected_construction_lanes)):
        raise ValueError("expected construction lane outputs must be unique")
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
        expected_outputs=tuple(sorted(expected_construction_lanes)),
        created_at=created_at,
    )


def sign(signer: Ed25519Signer, payload, *, created_at: datetime) -> SignedEnvelope:
    """Sign one contract payload with a producer/authority key."""
    return signer.sign(payload, created_at=created_at)


__all__ = [
    "CRYPTO_MOMENTUM_CAPABILITY_PROFILE_ID",
    "CRYPTO_MOMENTUM_STRATEGY_ID",
    "CRYPTO_MOMENTUM_LANE_ID",
    "CRYPTO_MOMENTUM_METRIC_IDS",
    "build_capability_profile",
    "cell_partition_id",
    "FORWARD_LEDGER_PREFIX",
    "forward_ledger_partition_id",
    "build_momentum_cell_dataset_manifest",
    "build_momentum_strategy_manifest",
    "build_momentum_evaluation_job_manifest",
    "sign",
]
