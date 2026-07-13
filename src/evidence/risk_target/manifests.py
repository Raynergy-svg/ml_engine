"""Dataset / capability / job manifest construction for the risk-target slice.

Pure builders over the formal evidence contracts. No training, no I/O side
effects, no authority — a manifest only *declares* a run; it never executes
one.
"""

from __future__ import annotations

from datetime import date, datetime
from typing import Mapping, Sequence

from src.evidence.contracts import (
    AssetClass,
    CapabilityProfile,
    DatasetManifest,
    JobManifest,
    ModelTrainingSpec,
    PartitionRef,
    ResourceClass,
    SignedEnvelope,
)
from src.evidence.hashing import content_digest, sha256_bytes
from src.evidence.signing import Ed25519Signer
from src.training.risk_target_features import FEATURE_COLUMNS

from .models import DRAWDOWN_HEAD_ID, VOLATILITY_HEAD_ID

# Immutable safety facts for this workload, asserted into every job's
# capability profile. Remote compute may read approved snapshots and write
# job-scoped results and nothing else (roadmap §4.4).
RISK_TARGET_CAPABILITY_PROFILE_ID = "axiom-risk-target-worker"

VOLATILITY_TARGET = "forward_realized_volatility"
DRAWDOWN_TARGET = "forward_drawdown_state"

# Per-head expected outputs the local aggregator requires before a package is
# admissible. A missing head == a missing fold == a hard import failure.
EXPECTED_HEADS: tuple[str, ...] = (VOLATILITY_HEAD_ID, DRAWDOWN_HEAD_ID)


def build_capability_profile(
    *,
    dataset_domain: str = "fx",
    profile_id: str = RISK_TARGET_CAPABILITY_PROFILE_ID,
) -> CapabilityProfile:
    """The worker's declared capabilities.

    Read scope is limited to the approved FX snapshot domain; write scope is
    the job's own output namespace; there are no network endpoints. Every
    authority flag is ``Literal[False]`` by the contract's own type, so a
    worker cannot even *express* broker, order, halt, live-gate, champion, or
    self-approval authority.
    """
    return CapabilityProfile(
        profile_id=profile_id,
        read_scopes=(f"axiom-data/{dataset_domain}/snapshots/**",),
        write_scopes=("job:{job_id}/outputs/**",),
        network_endpoints=(),
    )


def build_fx_daily_dataset_manifest(
    partitions: Mapping[str, bytes],
    *,
    dataset_id: str,
    retrieved_at: datetime,
    coverage_start: date,
    coverage_end: date,
    provider: str = "cached_fx_daily_panel",
    data_schema_version: str = "fx_daily_ohlcv.v1",
    point_in_time_rules: Sequence[str] = ("bar close is the as-of instant; no intrabar look-ahead",),
    row_counts: Mapping[str, int] | None = None,
) -> DatasetManifest:
    """Build a ``DatasetManifest`` over the cached daily FX panel.

    Each instrument's CSV bytes become one content-addressed ``PartitionRef``
    (``digest`` = SHA-256 of the exact bytes the worker will be handed). A
    single changed byte in any partition changes its digest and therefore the
    manifest digest — the hash rail the acceptance tests exercise.
    """
    if not partitions:
        raise ValueError("at least one dataset partition is required")
    row_counts = row_counts or {}
    refs: list[PartitionRef] = []
    for instrument in sorted(partitions):
        data = partitions[instrument]
        refs.append(
            PartitionRef(
                partition_id=instrument,
                uri=f"axiom-data/fx/snapshots/{dataset_id}/partitions/{instrument}.csv",
                digest=sha256_bytes(data),
                size_bytes=len(data),
                row_count=row_counts.get(instrument),
            )
        )
    return DatasetManifest(
        dataset_id=dataset_id,
        asset_class=AssetClass.FX,
        domain="risk_target_fx_daily",
        provider=provider,
        retrieved_at=retrieved_at,
        coverage_start=coverage_start,
        coverage_end=coverage_end,
        instruments=tuple(sorted(partitions)),
        data_schema_version=data_schema_version,
        point_in_time_rules=tuple(point_in_time_rules),
        partitions=tuple(refs),
    )


def build_risk_target_training_spec() -> ModelTrainingSpec:
    """Freeze the dual-head risk-target training specification."""
    return ModelTrainingSpec(
        target=f"{VOLATILITY_TARGET}+{DRAWDOWN_TARGET}",
        features=tuple(FEATURE_COLUMNS) + ("pair",),
        causal_lag="features at bar t; targets over bars (t, t+H] with a 20-bar embargo",
        search_space={"note": "no search — frozen pre-registered configuration"},
        metrics=("qlike", "pinball_median", "r2", "mae", "auc", "brier"),
        promotion_criteria=(
            "forward_volatility: OOS QLIKE < naive-persistence QLIKE AND OOS R2 > 0",
            "drawdown_state: OOS AUC >= 0.55 AND OOS Brier < base-rate baseline",
        ),
    )


def build_risk_target_job_manifest(
    *,
    job_id: str,
    dataset_manifest: DatasetManifest,
    capability_profile: CapabilityProfile,
    git_commit: str,
    container_digest: str,
    configuration_digest: str,
    feature_pipeline_version: str,
    random_seeds: Sequence[int],
    created_at: datetime,
    assigned_unit: str = "risk_target/all_heads/fold_0",
    resource_class: ResourceClass = ResourceClass.CPU_SMALL,
    trial_budget: int = 1,
    expected_heads: Sequence[str] = EXPECTED_HEADS,
) -> JobManifest:
    """Assemble the signed-job payload binding data, code, config and outputs."""
    return JobManifest(
        job_id=job_id,
        git_commit=git_commit,
        container_digest=container_digest,
        dataset_manifest_digests=(content_digest(dataset_manifest),),
        model_training_spec=build_risk_target_training_spec(),
        feature_pipeline_version=feature_pipeline_version,
        configuration_digest=configuration_digest,
        random_seeds=tuple(random_seeds),
        assigned_unit=assigned_unit,
        resource_class=resource_class,
        capability_profile_digest=content_digest(capability_profile),
        trial_budget=trial_budget,
        expected_outputs=tuple(expected_heads),
        created_at=created_at,
    )


def sign(signer: Ed25519Signer, payload, *, created_at: datetime) -> SignedEnvelope:
    """Sign one contract payload with a producer/authority key."""
    return signer.sign(payload, created_at=created_at)


__all__ = [
    "RISK_TARGET_CAPABILITY_PROFILE_ID",
    "VOLATILITY_TARGET",
    "DRAWDOWN_TARGET",
    "EXPECTED_HEADS",
    "build_capability_profile",
    "build_fx_daily_dataset_manifest",
    "build_risk_target_training_spec",
    "build_risk_target_job_manifest",
    "sign",
]
