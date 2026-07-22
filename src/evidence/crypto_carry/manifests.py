"""Dataset / capability / strategy / job manifest carry for the crypto
carry evaluation slice.

Pure builders over the formal evidence contracts. No scoring, no I/O side
effects, no authority — a manifest only *declares* a run; it never executes one.

Like the equity-research and hedge slices (and unlike the risk-target slice's
model-training workload), a crypto-carry evaluation is a *strategy/research*
workload: it scores an already-computed per-cell forward-return series against
the frozen carry's admissibility bars. It therefore fills the
``StrategyManifest`` half of the ``JobManifest`` contract. The granularity is
``venue set × cost model × regime``: each dataset partition is ONE cell, and the strategy manifest declares the
exact expected cell set per carry so the deterministic aggregator can
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
CRYPTO_CARRY_CAPABILITY_PROFILE_ID = "axiom-crypto-carry-worker"

CRYPTO_CARRY_STRATEGY_ID = "axiom-crypto-cash-and-carry"
CRYPTO_CARRY_LANE_ID = "crypto_carry"

# Scorecard metric surface the report exposes for display + independent replay.
CRYPTO_CARRY_METRIC_IDS: tuple[str, ...] = (
    "n_cells",
    "minimum_cell_net_sharpe",
    "maximum_cell_drawdown",
    "maximum_margin_utilization",
    "maximum_tracking_error",
    "minimum_capacity_usd",
    "tail_evidence_gate_passed",
    "counterparty_evidence_gate_passed",
)


def build_capability_profile(
    *,
    profile_id: str = CRYPTO_CARRY_CAPABILITY_PROFILE_ID,
) -> CapabilityProfile:
    """The worker's declared capabilities.

    Read scope is limited to the approved crypto-carry shadow snapshot domain;
    write scope is the job's own output namespace; there are no network
    endpoints. Every authority flag is ``Literal[False]`` by the contract's own
    type, so a worker cannot even *express* broker, order, halt, live-gate,
    champion, or self-approval authority.
    """
    return CapabilityProfile(
        profile_id=profile_id,
        read_scopes=("axiom-data/crypto_carry/snapshots/**",),
        write_scopes=("job:{job_id}/outputs/**",),
        network_endpoints=(),
    )


FORWARD_LEDGER_PREFIX = "forward_ledger::"
_RESERVED_ID_SEGMENT = "forward_ledger"  # Code Reviewer finding, 2026-07-22: see below


def _reject_reserved_id(carry_id: str) -> None:
    """A carry_id literally equal to ``"forward_ledger"`` would make
    ``cell_partition_id(carry_id, ...)`` start with ``FORWARD_LEDGER_PREFIX``
    (e.g. ``"forward_ledger::binance::base::normal"``), misclassifying every
    one of that carry's cells as ledger data in ``evaluate_partitions``. Fails
    LOUD today (a ValueError from the ledger-set mismatch check, never a
    silent misread — confirmed by the Code Reviewer), but an unusable carry_id
    is still worth refusing outright rather than relying on a downstream
    check to catch it."""
    if carry_id == _RESERVED_ID_SEGMENT:
        raise ValueError(
            f"carry_id cannot be {_RESERVED_ID_SEGMENT!r} — reserved for forward-ledger partition ids"
        )


def cell_partition_id(
    carry_id: str, venue_set: str, cost_model: str, regime: str
) -> str:
    """Canonical id for one ``venue set × cost model × regime`` cell."""
    components = (carry_id, venue_set, cost_model, regime)
    if any(not value or "::" in value for value in components):
        raise ValueError("cell id components must be non-empty and cannot contain '::'")
    _reject_reserved_id(carry_id)
    return "::".join(components)


def forward_ledger_partition_id(carry_id: str) -> str:
    """Canonical partition id for one carry's REAL forward-shadow ledger
    (roadmap §14 standardized return contract source) — reserved prefix keeps
    it unambiguously distinct from ``{carry_id}::{venue_set}::{cost_model}::
    {regime}`` cell ids everywhere partitions are classified."""
    if not carry_id or "::" in carry_id:
        raise ValueError("carry_id must be non-empty and cannot contain '::'")
    _reject_reserved_id(carry_id)
    return f"{FORWARD_LEDGER_PREFIX}{carry_id}"


def build_carry_cell_dataset_manifest(
    partitions: Mapping[str, bytes],
    *,
    dataset_id: str,
    retrieved_at: datetime,
    coverage_start: date,
    coverage_end: date,
    provider: str = "crypto_carry_shadow_ledger",
    data_schema_version: str = "crypto_carry_cell_returns.v1",
    point_in_time_rules: Sequence[str] = (
        "each cell row is a realized net carry period for an immutable venue-set, "
        "cost-model and regime identity; no look-ahead",
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
                uri=f"axiom-data/crypto_carry/snapshots/{dataset_id}/cells/{cell_id}.jsonl",
                digest=sha256_bytes(data),
                size_bytes=len(data),
                row_count=row_counts.get(cell_id),
            )
        )
    return DatasetManifest(
        dataset_id=dataset_id,
        asset_class=AssetClass.CRYPTO,
        domain="crypto_carry_cell_returns",
        provider=provider,
        retrieved_at=retrieved_at,
        coverage_start=coverage_start,
        coverage_end=coverage_end,
        instruments=tuple(sorted(partitions)),
        data_schema_version=data_schema_version,
        point_in_time_rules=tuple(point_in_time_rules),
        partitions=tuple(refs),
    )


def build_carry_strategy_manifest(
    campaign_id: str,
    expected_cells_by_carry: Mapping[str, Sequence[str]],
    *,
    frozen_at: datetime,
    min_cells: int,
    net_sharpe_floor: float,
    max_drawdown_limit: float,
    max_margin_utilization: float,
    max_tracking_error: float,
    min_capacity_usd: float,
    strategy_id: str = CRYPTO_CARRY_STRATEGY_ID,
    lane_id: str = CRYPTO_CARRY_LANE_ID,
    forward_ledger_carries: Sequence[str] = (),
) -> StrategyManifest:
    """Freeze the crypto cash-and-carry evaluation program.

    The declared ``expected_cells_by_carry`` is signed into the manifest
    and is the authority the aggregator refuses incomplete/duplicate aggregations
    against — an omitted cell cannot be silently scored as a complete
    carry. ``forward_ledger_carries`` is the SAME kind of signed declaration
    for which carries' packages must carry the roadmap §14 return contract
    (built from their real forward-shadow ledger).
    """
    carries = sorted(expected_cells_by_carry)
    if not carries:
        raise ValueError("at least one carry is required")
    for carry in carries:
        _reject_reserved_id(carry)
        cells = expected_cells_by_carry[carry]
        if not cells:
            raise ValueError(f"carry {carry!r} declares no expected cells")
        if len(cells) != len(set(cells)):
            raise ValueError(f"carry {carry!r} declares duplicate expected cells")
    declared = {c: sorted(expected_cells_by_carry[c]) for c in carries}
    ledger_carries = sorted(set(forward_ledger_carries))
    if len(ledger_carries) != len(forward_ledger_carries):
        raise ValueError("forward_ledger_carries must not contain duplicates")
    unknown = set(ledger_carries) - set(carries)
    if unknown:
        raise ValueError(f"forward_ledger_carries references undeclared carries: {sorted(unknown)}")
    return StrategyManifest(
        strategy_id=strategy_id,
        lane_id=lane_id,
        experiment_id=f"crypto-carry-{campaign_id}",
        hypothesis=(
            "A delta-neutral spot/perpetual carry construction retains positive "
            "net funding capture after both legs' costs while satisfying margin, "
            "tracking, capacity, liquidation-tail, venue-failure and explicit "
            "counterparty-evidence gates. Sharpe alone is never sufficient."
        ),
        exploratory=True,
        frozen_at=frozen_at,
        universe=tuple(carries),
        causal_lag=(
            "each cell contains realized, point-in-time net returns for one venue "
            "set, cost model and market regime; no look-ahead"
        ),
        parameters={
            "campaign_id": campaign_id,
            "expected_cells_by_carry": declared,
            "min_cells": min_cells,
            "net_sharpe_floor": net_sharpe_floor,
            "max_drawdown_limit": max_drawdown_limit,
            "max_margin_utilization": max_margin_utilization,
            "max_tracking_error": max_tracking_error,
            "min_capacity_usd": min_capacity_usd,
            "required_tail_scenarios": [
                "basis_blowout", "liquidation", "venue_failure",
                "withdrawal_suspension", "stablecoin_depeg",
                "cross_venue_settlement",
            ],
            "forward_ledger_carries": ledger_carries,
        },
        cost_model={
            "basis": "net return includes fees, borrow/transfer costs and both spot/perp legs",
            "note": "gross funding capture and two-leg costs remain separately reported",
        },
        evaluation_windows=("per-regime realized carry return series",),
        search_space={"note": "no search — deterministic aggregation over declared cells"},
        trial_budget=1,
        metrics=CRYPTO_CARRY_METRIC_IDS,
        promotion_criteria=(
            "carry admissible only when every declared cell clears net return, "
            "drawdown, margin, tracking and capacity bars AND every mandatory "
            "tail scenario and venue counterparty classification is present",
        ),
    )


def build_carry_evaluation_job_manifest(
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
    expected_carry_lanes: Sequence[str],
    assigned_unit: str = "crypto_carry/all_carries/all_venue_cost_regime_cells",
    resource_class: ResourceClass = ResourceClass.CPU_SMALL,
    trial_budget: int = 1,
) -> JobManifest:
    """Assemble the signed-job payload binding data, strategy spec, config, outputs.

    The Sharpe aggregation has no stochastic component, so ``random_seeds``
    carries the degenerate ``(0,)`` — the contract requires a seed, but the
    workload is fully deterministic float arithmetic over the declared cells.
    ``expected_outputs`` are the per-carry evidence lane ids.
    """
    if not expected_carry_lanes:
        raise ValueError("at least one expected carry lane output is required")
    if len(expected_carry_lanes) != len(set(expected_carry_lanes)):
        raise ValueError("expected carry lane outputs must be unique")
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
        expected_outputs=tuple(sorted(expected_carry_lanes)),
        created_at=created_at,
    )


def sign(signer: Ed25519Signer, payload, *, created_at: datetime) -> SignedEnvelope:
    """Sign one contract payload with a producer/authority key."""
    return signer.sign(payload, created_at=created_at)


__all__ = [
    "CRYPTO_CARRY_CAPABILITY_PROFILE_ID",
    "CRYPTO_CARRY_STRATEGY_ID",
    "CRYPTO_CARRY_LANE_ID",
    "CRYPTO_CARRY_METRIC_IDS",
    "build_capability_profile",
    "cell_partition_id",
    "FORWARD_LEDGER_PREFIX",
    "forward_ledger_partition_id",
    "build_carry_cell_dataset_manifest",
    "build_carry_strategy_manifest",
    "build_carry_evaluation_job_manifest",
    "sign",
]
