"""End-to-end signed portfolio-book evidence slice, stopping at quarantine.

Combines sleeve return-series partitions into one book, gates it, and stops at
QUARANTINED — the same terminal state every other AXIOM evidence slice reaches
through this authority-free path. No lane, and no book, becomes capital-active
from a standalone gate: promotion to CHAMPION requires the separate,
human-operated promotion service (Phase L), which this module does not touch.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Mapping

from src.evidence.canonical import canonical_bytes
from src.evidence.contracts import AuthorityRole, SignedEnvelope
from src.evidence.hashing import sha256_bytes
from src.evidence.signing import Ed25519Signer, TrustStore
from src.evidence.store import EvidenceStore
from src.evidence.transition_policy import AuthorityRegistry

from .evaluation import EvaluationParams
from .local_import import import_book
from .manifests import build_capability_profile, build_dataset_manifest, build_job_manifest, build_strategy_manifest
from .models import ImportOutcome, WorkerOutput
from .worker import run_worker

_ROLES = (AuthorityRole.PRODUCER, AuthorityRole.LOCAL_IMPORTER, AuthorityRole.INDEPENDENT_VERIFIER)
_ACTORS = {
    AuthorityRole.PRODUCER: "portfolio-book-remote-worker",
    AuthorityRole.LOCAL_IMPORTER: "axiom-local-importer",
    AuthorityRole.INDEPENDENT_VERIFIER: "axiom-independent-verifier",
}


@dataclass(frozen=True)
class SliceIdentities:
    signers: Mapping[AuthorityRole, Ed25519Signer]
    actors: Mapping[AuthorityRole, str]
    trust_store: TrustStore
    authorities: AuthorityRegistry


@dataclass(frozen=True)
class Produced:
    worker_output: WorkerOutput


@dataclass(frozen=True)
class SliceResult:
    job_envelope: SignedEnvelope
    dataset_manifest_envelope: SignedEnvelope
    strategy_manifest_envelope: SignedEnvelope
    worker_output: WorkerOutput
    outcome: ImportOutcome


def build_slice_identities(now: datetime) -> SliceIdentities:
    signers = {role: Ed25519Signer.generate() for role in _ROLES}
    trust = TrustStore()
    authorities = AuthorityRegistry()
    for role, signer in signers.items():
        trust.add(signer.trusted_key(valid_from=now - timedelta(days=1)))
        authorities.register(actor_id=_ACTORS[role], role=role, key_ids=(signer.key_id,))
    return SliceIdentities(signers=signers, actors=_ACTORS, trust_store=trust, authorities=authorities)


def build_evidence_store(root: str | Path, identities: SliceIdentities, *, clock_now: datetime) -> EvidenceStore:
    return EvidenceStore(root, trust_store=identities.trust_store,
                         authorities=identities.authorities, trusted_clock=lambda: clock_now)


def _config_digest(params: EvaluationParams, feature_version: str) -> str:
    return sha256_bytes(canonical_bytes({
        "oos_start": params.oos_start, "min_est_days": params.min_est_days, "min_oos_days": params.min_oos_days,
        "cash_reserve_frac": params.cash_reserve_frac, "max_allocation": params.max_allocation,
        "min_allocation": params.min_allocation, "min_sleeve_vol": params.min_sleeve_vol,
        "lane_capacity": dict(params.lane_capacity),
        "tail_corr_threshold": params.tail_corr_threshold, "tail_quantile": params.tail_quantile,
        "drawdown_budget": params.drawdown_budget, "max_book_drawdown": params.max_book_drawdown,
        "min_book_sharpe": params.min_book_sharpe, "feature_pipeline_version": feature_version,
    }))


def produce_worker_output(
    identities: SliceIdentities, partitions: Mapping[str, bytes], *,
    source_package_digests: Mapping[str, str], dataset_id: str,
    coverage_start: date, coverage_end: date, retrieved_at: datetime, created_at: datetime,
    git_commit: str, job_id: str = "portfolio-book-vertical-slice",
    strategy_id: str = "portfolio-book-hrp-v1", experiment_id: str = "phase-k-portfolio-allocator",
    feature_pipeline_version: str = "portfolio_book_features.v1",
    params: EvaluationParams | None = None,
    registry_publish=lambda *_: None,
) -> Produced:
    params = params or EvaluationParams()
    producer = identities.signers[AuthorityRole.PRODUCER]
    capability = build_capability_profile()
    capability_envelope = producer.sign(capability, created_at=created_at)
    dataset = build_dataset_manifest(
        partitions, dataset_id=dataset_id, retrieved_at=retrieved_at,
        coverage_start=coverage_start, coverage_end=coverage_end,
        row_counts={name: len(data.splitlines()) for name, data in partitions.items()},
    )
    dataset_envelope = producer.sign(dataset, created_at=created_at)
    strategy = build_strategy_manifest(
        strategy_id=strategy_id, experiment_id=experiment_id, sleeve_ids=sorted(partitions),
        source_package_digests=source_package_digests, params=params, frozen_at=created_at,
    )
    strategy_envelope = producer.sign(strategy, created_at=created_at)
    job = build_job_manifest(
        job_id=job_id, dataset_manifest=dataset, strategy_manifest=strategy, capability_profile=capability,
        git_commit=git_commit, container_digest=sha256_bytes(b"axiom-portfolio-book:local-inprocess:v1"),
        configuration_digest=_config_digest(params, feature_pipeline_version),
        feature_pipeline_version=feature_pipeline_version, created_at=created_at,
    )
    job_envelope = producer.sign(job, created_at=created_at)
    output = run_worker(
        job_envelope=job_envelope, dataset_manifest_envelope=dataset_envelope,
        strategy_manifest_envelope=strategy_envelope, capability_profile_envelope=capability_envelope,
        partitions=partitions, producer=producer, producer_id=identities.actors[AuthorityRole.PRODUCER],
        trust_store=identities.trust_store, created_at=created_at, registry_publish=registry_publish,
    )
    return Produced(worker_output=output)


def run_portfolio_book_evidence_slice(
    store: EvidenceStore, identities: SliceIdentities, partitions: Mapping[str, bytes], **kwargs,
) -> SliceResult:
    produced = produce_worker_output(identities, partitions, **kwargs)
    output = produced.worker_output
    outcome = import_book(
        store, output.book, job_envelope=output.job_envelope,
        capability_profile_envelope=output.capability_profile_envelope,
        dataset_manifest_envelope=output.dataset_manifest_envelope,
        strategy_manifest_envelope=output.strategy_manifest_envelope,
        partitions=partitions, trust_store=identities.trust_store,
        importer=identities.signers[AuthorityRole.LOCAL_IMPORTER],
        importer_id=identities.actors[AuthorityRole.LOCAL_IMPORTER],
        verifier=identities.signers[AuthorityRole.INDEPENDENT_VERIFIER],
        verifier_id=identities.actors[AuthorityRole.INDEPENDENT_VERIFIER],
        created_at=kwargs["created_at"],
    )
    return SliceResult(job_envelope=output.job_envelope, dataset_manifest_envelope=output.dataset_manifest_envelope,
                       strategy_manifest_envelope=output.strategy_manifest_envelope,
                       worker_output=output, outcome=outcome)


def utc(year: int, month: int, day: int, hour: int = 0) -> datetime:
    return datetime(year, month, day, hour, tzinfo=timezone.utc)


__all__ = [
    "SliceIdentities", "Produced", "SliceResult", "build_slice_identities", "build_evidence_store",
    "produce_worker_output", "run_portfolio_book_evidence_slice", "utc",
]
