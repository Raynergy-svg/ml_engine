"""End-to-end crypto-carry evidence vertical slice + read-only cockpit view.

``run_crypto_carry_evidence_slice`` executes the whole roadmap §12 flow for
one signed job over one campaign's venue-set/cost-model/regime cells: it builds
and signs the dataset / strategy / capability / job manifests, runs the
authority-free worker (which scores every cell and aggregates one head per
carry id), and imports every produced carry head into the local evidence
store, ending each at QUARANTINED or REJECTED. A candidate never overwrites an
incumbent — the slice never promotes a champion.

Carry programs are evaluated and dispositioned independently. Return, tail and
counterparty gates are separate and this module contains no promotion path
(roadmap J4).

``crypto_carry_evidence_view`` is a display-only reader over the store's own
committed index projection (no signing keys required), suitable for the dashboard.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Mapping

from src.evidence.canonical import canonical_bytes
from src.evidence.contracts import (
    AuthorityRole,
    CapabilityProfile,
    JobManifest,
    SignedEnvelope,
    StrategyManifest,
)
from src.evidence.hashing import sha256_bytes
from src.evidence.signing import Ed25519Signer, TrustStore
from src.evidence.store import EvidenceStore
from src.evidence.transition_policy import AuthorityRegistry

from .dashboard import crypto_carry_evidence_view
from .evaluation import EvaluationParams, evaluate_partitions
from .local_import import import_head
from .manifests import (
    build_capability_profile,
    build_carry_evaluation_job_manifest,
    build_carry_cell_dataset_manifest,
    build_carry_strategy_manifest,
)
from .models import CryptoCarryWorkerOutput, CarryImportOutcome, lane_id_for_carry
from .worker import _fail_open_registry, run_worker

# Scorecard contract version (bump on any change to the scored fields / verdict
# logic in src/crypto/carry_scorecard.py that would change numbers).
CRYPTO_CARRY_SCORECARD_VERSION = "crypto_carry_scorecard.v1"

# The producer, local importer and independent verifier must be distinct actors
# and keys — separation of duties is enforced cryptographically by the store.
_SLICE_ROLES = (
    AuthorityRole.PRODUCER,
    AuthorityRole.LOCAL_IMPORTER,
    AuthorityRole.INDEPENDENT_VERIFIER,
)
_DEFAULT_ACTORS = {
    AuthorityRole.PRODUCER: "crypto-carry-remote-worker",
    AuthorityRole.LOCAL_IMPORTER: "axiom-local-importer",
    AuthorityRole.INDEPENDENT_VERIFIER: "axiom-independent-verifier",
}


@dataclass(frozen=True)
class SliceIdentities:
    signers: Mapping[AuthorityRole, Ed25519Signer]
    actors: Mapping[AuthorityRole, str]
    trust_store: TrustStore
    authorities: AuthorityRegistry

    @property
    def producer(self) -> Ed25519Signer:
        return self.signers[AuthorityRole.PRODUCER]

    @property
    def importer(self) -> Ed25519Signer:
        return self.signers[AuthorityRole.LOCAL_IMPORTER]

    @property
    def verifier(self) -> Ed25519Signer:
        return self.signers[AuthorityRole.INDEPENDENT_VERIFIER]


def build_slice_identities(
    now: datetime,
    *,
    actors: Mapping[AuthorityRole, str] | None = None,
) -> SliceIdentities:
    """Generate three distinct signing identities and register their trust."""
    actors = {**_DEFAULT_ACTORS, **(actors or {})}
    signers = {role: Ed25519Signer.generate() for role in _SLICE_ROLES}
    trust = TrustStore()
    authorities = AuthorityRegistry()
    valid_from = now - timedelta(days=1)
    for role, signer in signers.items():
        trust.add(signer.trusted_key(valid_from=valid_from))
        authorities.register(actor_id=actors[role], role=role, key_ids=(signer.key_id,))
    return SliceIdentities(signers=signers, actors=actors, trust_store=trust, authorities=authorities)


def build_evidence_store(
    root: str | Path, identities: SliceIdentities, *, clock_now: datetime
) -> EvidenceStore:
    """Construct the store bound to the slice's trust and a fixed trusted clock."""
    return EvidenceStore(
        root,
        trust_store=identities.trust_store,
        authorities=identities.authorities,
        trusted_clock=lambda: clock_now,
    )


def _expected_cells_by_carry(
    partitions: Mapping[str, bytes],
) -> dict[str, tuple[str, ...]]:
    """Derive the declared expected cell set from partition ids."""
    grouped: dict[str, list[str]] = {}
    for cell_id in partitions:
        carry_id = cell_id.split("::", 1)[0]
        grouped.setdefault(carry_id, []).append(cell_id)
    return {carry_id: tuple(sorted(ids)) for carry_id, ids in grouped.items()}


@dataclass(frozen=True)
class Produced:
    """The producer half's output plus the local objects needed to import it."""

    campaign_id: str
    job: JobManifest
    strategy_manifest: StrategyManifest
    capability_profile: CapabilityProfile
    worker_output: CryptoCarryWorkerOutput


@dataclass(frozen=True)
class SliceResult:
    job_envelope: SignedEnvelope
    dataset_manifest_envelope: SignedEnvelope
    strategy_manifest_envelope: SignedEnvelope
    worker_output: CryptoCarryWorkerOutput
    outcomes: Mapping[str, CarryImportOutcome] = field(default_factory=dict)


def _config_digest(params: EvaluationParams, scorecard_version: str) -> str:
    payload = {
        "params": {
            "min_cells": params.min_cells,
            "net_sharpe_floor": params.net_sharpe_floor,
            "max_drawdown_limit": params.max_drawdown_limit,
            "max_margin_utilization": params.max_margin_utilization,
            "max_tracking_error": params.max_tracking_error,
            "min_capacity_usd": params.min_capacity_usd,
            "replay_tolerance": params.replay_tolerance,
        },
        "scorecard_version": scorecard_version,
    }
    return sha256_bytes(canonical_bytes(payload))


def _resolve_params(
    partitions: Mapping[str, bytes], params: EvaluationParams | None
) -> EvaluationParams:
    """Bind the declared expected-cell map into params if not already set."""
    base = params or EvaluationParams()
    if base.expected_cells_by_carry:
        return base
    from dataclasses import replace

    return replace(
        base,
        expected_cells_by_carry=_expected_cells_by_carry(partitions),
    )


def produce_worker_output(
    identities: SliceIdentities,
    partitions: Mapping[str, bytes],
    *,
    campaign_id: str,
    dataset_id: str,
    coverage_start: date,
    coverage_end: date,
    retrieved_at: datetime,
    created_at: datetime,
    git_commit: str,
    job_id: str = "crypto-carry-vertical-slice",
    params: EvaluationParams | None = None,
    evaluator=evaluate_partitions,
    registry_publish=_fail_open_registry,
) -> Produced:
    """Build+sign the manifests and run the authority-free worker (no import)."""
    params = _resolve_params(partitions, params)
    producer = identities.producer
    producer_id = identities.actors[AuthorityRole.PRODUCER]

    expected_cells = params.expected_cells_by_carry
    carry_lanes = sorted(
        lane_id_for_carry(campaign_id, carry_id)
        for carry_id in expected_cells
    )

    capability_profile = build_capability_profile()
    capability_envelope = producer.sign(capability_profile, created_at=created_at)

    dataset_manifest = build_carry_cell_dataset_manifest(
        partitions,
        dataset_id=dataset_id,
        retrieved_at=retrieved_at,
        coverage_start=coverage_start,
        coverage_end=coverage_end,
        row_counts={f: data.count(b"\n") for f, data in partitions.items()},
    )
    dataset_envelope = producer.sign(dataset_manifest, created_at=created_at)

    strategy_manifest = build_carry_strategy_manifest(
        campaign_id, expected_cells, frozen_at=created_at,
        min_cells=params.min_cells,
        net_sharpe_floor=params.net_sharpe_floor,
        max_drawdown_limit=params.max_drawdown_limit,
        max_margin_utilization=params.max_margin_utilization,
        max_tracking_error=params.max_tracking_error,
        min_capacity_usd=params.min_capacity_usd,
    )
    strategy_envelope = producer.sign(strategy_manifest, created_at=created_at)

    container_digest = sha256_bytes(b"axiom-crypto-carry:local-inprocess:v1")
    job = build_carry_evaluation_job_manifest(
        job_id=job_id,
        dataset_manifest=dataset_manifest,
        strategy_manifest=strategy_manifest,
        capability_profile=capability_profile,
        git_commit=git_commit,
        container_digest=container_digest,
        configuration_digest=_config_digest(params, CRYPTO_CARRY_SCORECARD_VERSION),
        scorecard_version=CRYPTO_CARRY_SCORECARD_VERSION,
        created_at=created_at,
        expected_carry_lanes=carry_lanes,
    )
    job_envelope = producer.sign(job, created_at=created_at)

    worker_output = run_worker(
        campaign_id=campaign_id,
        job_envelope=job_envelope,
        dataset_manifest_envelope=dataset_envelope,
        strategy_manifest_envelope=strategy_envelope,
        capability_profile_envelope=capability_envelope,
        partitions=partitions,
        producer=producer,
        producer_id=producer_id,
        trust_store=identities.trust_store,
        created_at=created_at,
        params=params,
        evaluator=evaluator,
        registry_publish=registry_publish,
    )
    return Produced(
        campaign_id=campaign_id, job=job, strategy_manifest=strategy_manifest,
        capability_profile=capability_profile, worker_output=worker_output,
    )


def import_worker_output(
    store: EvidenceStore,
    identities: SliceIdentities,
    produced: Produced,
    partitions: Mapping[str, bytes],
    *,
    created_at: datetime,
    params: EvaluationParams | None = None,
    evaluator=evaluate_partitions,
) -> dict[str, CarryImportOutcome]:
    """Import every produced carry head (QUARANTINED/REJECTED)."""
    params = _resolve_params(partitions, params)
    outcomes: dict[str, CarryImportOutcome] = {}
    for head in produced.worker_output.heads:
        outcomes[head.lane_id] = import_head(
            store,
            head,
            campaign_id=produced.campaign_id,
            job=produced.job,
            capability_profile_envelope=produced.worker_output.capability_profile_envelope,
            dataset_manifest_envelope=produced.worker_output.dataset_manifest_envelope,
            strategy_manifest_envelope=produced.worker_output.strategy_manifest_envelope,
            partitions=partitions,
            trust_store=identities.trust_store,
            importer=identities.importer,
            importer_id=identities.actors[AuthorityRole.LOCAL_IMPORTER],
            verifier=identities.verifier,
            verifier_id=identities.actors[AuthorityRole.INDEPENDENT_VERIFIER],
            created_at=created_at,
            params=params,
            evaluator=evaluator,
        )
    return outcomes


def run_crypto_carry_evidence_slice(
    store: EvidenceStore,
    identities: SliceIdentities,
    partitions: Mapping[str, bytes],
    *,
    campaign_id: str,
    dataset_id: str,
    coverage_start: date,
    coverage_end: date,
    retrieved_at: datetime,
    created_at: datetime,
    git_commit: str,
    job_id: str = "crypto-carry-vertical-slice",
    params: EvaluationParams | None = None,
    evaluator=evaluate_partitions,
    registry_publish=_fail_open_registry,
) -> SliceResult:
    """Run the full cells -> job -> worker -> import flow for one campaign."""
    params = _resolve_params(partitions, params)
    produced = produce_worker_output(
        identities, partitions,
        campaign_id=campaign_id, dataset_id=dataset_id,
        coverage_start=coverage_start, coverage_end=coverage_end,
        retrieved_at=retrieved_at, created_at=created_at, git_commit=git_commit,
        job_id=job_id, params=params, evaluator=evaluator, registry_publish=registry_publish,
    )
    outcomes = import_worker_output(
        store, identities, produced, partitions,
        created_at=created_at, params=params, evaluator=evaluator,
    )
    return SliceResult(
        job_envelope=produced.worker_output.job_envelope,
        dataset_manifest_envelope=produced.worker_output.dataset_manifest_envelope,
        strategy_manifest_envelope=produced.worker_output.strategy_manifest_envelope,
        worker_output=produced.worker_output,
        outcomes=outcomes,
    )


def utc(year: int, month: int, day: int, hour: int = 0, minute: int = 0) -> datetime:
    return datetime(year, month, day, hour, minute, tzinfo=timezone.utc)


__all__ = [
    "CRYPTO_CARRY_SCORECARD_VERSION",
    "SliceIdentities",
    "SliceResult",
    "Produced",
    "build_slice_identities",
    "build_evidence_store",
    "produce_worker_output",
    "import_worker_output",
    "run_crypto_carry_evidence_slice",
    "crypto_carry_evidence_view",
    "utc",
]
