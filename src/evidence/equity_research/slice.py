"""End-to-end equity-research evidence vertical slice + read-only cockpit view.

``run_equity_research_evidence_slice`` executes the whole roadmap §12 flow for
one signed job over one research hypothesis's per-fold cross-sections: it builds
and signs the dataset / strategy / capability / job manifests, runs the
authority-free worker (which scores every fold and aggregates one head per
universe tier), and imports every produced tier head into the local evidence
store, ending each at QUARANTINED or REJECTED. A candidate never overwrites an
incumbent — the slice never promotes a champion.

The three universe tiers (curated / wide survivorship-corrected / full_history)
are evaluated and dispositioned fully independently, so their verdicts are
reported separately (roadmap J2) — the attractive curated result and the
defensible wide result are preserved side by side even when they disagree.

``equity_research_evidence_view`` is a display-only reader over the store's own
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

from .dashboard import equity_research_evidence_view
from .evaluation import EvaluationParams, evaluate_partitions
from .local_import import import_head
from .manifests import (
    build_capability_profile,
    build_research_evaluation_job_manifest,
    build_research_fold_dataset_manifest,
    build_research_strategy_manifest,
)
from .models import EquityResearchWorkerOutput, TierImportOutcome, lane_id_for_tier
from .worker import _fail_open_registry, run_worker

# Scorecard contract version (bump on any change to the scored fields / verdict
# logic in src/equity/research/rank_ic_scorecard.py that would change numbers).
EQUITY_RESEARCH_SCORECARD_VERSION = "rank_ic_scorecard.v1"

# The producer, local importer and independent verifier must be distinct actors
# and keys — separation of duties is enforced cryptographically by the store.
_SLICE_ROLES = (
    AuthorityRole.PRODUCER,
    AuthorityRole.LOCAL_IMPORTER,
    AuthorityRole.INDEPENDENT_VERIFIER,
)
_DEFAULT_ACTORS = {
    AuthorityRole.PRODUCER: "equity-research-remote-worker",
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
    actors = dict(actors or _DEFAULT_ACTORS)
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


def _expected_folds_by_tier(partitions: Mapping[str, bytes]) -> dict[str, tuple[str, ...]]:
    """Derive the declared expected fold set per tier from the partition ids."""
    by_tier: dict[str, list[str]] = {}
    for fold_id in partitions:
        tier = fold_id.split("::", 1)[0]
        by_tier.setdefault(tier, []).append(fold_id)
    return {tier: tuple(sorted(ids)) for tier, ids in by_tier.items()}


@dataclass(frozen=True)
class Produced:
    """The producer half's output plus the local objects needed to import it."""

    hypothesis_id: str
    job: JobManifest
    strategy_manifest: StrategyManifest
    capability_profile: CapabilityProfile
    worker_output: EquityResearchWorkerOutput


@dataclass(frozen=True)
class SliceResult:
    job_envelope: SignedEnvelope
    dataset_manifest_envelope: SignedEnvelope
    strategy_manifest_envelope: SignedEnvelope
    worker_output: EquityResearchWorkerOutput
    outcomes: Mapping[str, TierImportOutcome] = field(default_factory=dict)


def _config_digest(params: EvaluationParams, scorecard_version: str) -> str:
    payload = {
        "params": {
            "min_oos_folds": params.min_oos_folds,
            "ic_bar": params.ic_bar,
            "ic_tstat_bar": params.ic_tstat_bar,
            "replay_tolerance": params.replay_tolerance,
        },
        "scorecard_version": scorecard_version,
    }
    return sha256_bytes(canonical_bytes(payload))


def _resolve_params(
    partitions: Mapping[str, bytes], params: EvaluationParams | None
) -> EvaluationParams:
    """Bind the declared expected-fold map into the params if not already set."""
    base = params or EvaluationParams()
    if base.expected_folds_by_tier:
        return base
    from dataclasses import replace

    return replace(base, expected_folds_by_tier=_expected_folds_by_tier(partitions))


def produce_worker_output(
    identities: SliceIdentities,
    partitions: Mapping[str, bytes],
    *,
    hypothesis_id: str,
    dataset_id: str,
    coverage_start: date,
    coverage_end: date,
    retrieved_at: datetime,
    created_at: datetime,
    git_commit: str,
    job_id: str = "equity-research-vertical-slice",
    params: EvaluationParams | None = None,
    evaluator=evaluate_partitions,
    registry_publish=_fail_open_registry,
) -> Produced:
    """Build+sign the manifests and run the authority-free worker (no import)."""
    params = _resolve_params(partitions, params)
    producer = identities.producer
    producer_id = identities.actors[AuthorityRole.PRODUCER]

    expected_folds = params.expected_folds_by_tier
    tier_lanes = sorted(lane_id_for_tier(hypothesis_id, tier) for tier in expected_folds)

    capability_profile = build_capability_profile()
    capability_envelope = producer.sign(capability_profile, created_at=created_at)

    dataset_manifest = build_research_fold_dataset_manifest(
        partitions,
        dataset_id=dataset_id,
        retrieved_at=retrieved_at,
        coverage_start=coverage_start,
        coverage_end=coverage_end,
        row_counts={f: data.count(b"\n") for f, data in partitions.items()},
    )
    dataset_envelope = producer.sign(dataset_manifest, created_at=created_at)

    strategy_manifest = build_research_strategy_manifest(
        hypothesis_id, expected_folds, frozen_at=created_at,
        min_oos_folds=params.min_oos_folds, ic_bar=params.ic_bar, ic_tstat_bar=params.ic_tstat_bar,
    )
    strategy_envelope = producer.sign(strategy_manifest, created_at=created_at)

    container_digest = sha256_bytes(b"axiom-equity-research:local-inprocess:v1")
    job = build_research_evaluation_job_manifest(
        job_id=job_id,
        dataset_manifest=dataset_manifest,
        strategy_manifest=strategy_manifest,
        capability_profile=capability_profile,
        git_commit=git_commit,
        container_digest=container_digest,
        configuration_digest=_config_digest(params, EQUITY_RESEARCH_SCORECARD_VERSION),
        scorecard_version=EQUITY_RESEARCH_SCORECARD_VERSION,
        created_at=created_at,
        expected_tier_lanes=tier_lanes,
    )
    job_envelope = producer.sign(job, created_at=created_at)

    worker_output = run_worker(
        hypothesis_id=hypothesis_id,
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
        hypothesis_id=hypothesis_id, job=job, strategy_manifest=strategy_manifest,
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
) -> dict[str, TierImportOutcome]:
    """Import every produced tier head (QUARANTINED/REJECTED)."""
    params = _resolve_params(partitions, params)
    outcomes: dict[str, TierImportOutcome] = {}
    for head in produced.worker_output.heads:
        outcomes[head.lane_id] = import_head(
            store,
            head,
            hypothesis_id=produced.hypothesis_id,
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


def run_equity_research_evidence_slice(
    store: EvidenceStore,
    identities: SliceIdentities,
    partitions: Mapping[str, bytes],
    *,
    hypothesis_id: str,
    dataset_id: str,
    coverage_start: date,
    coverage_end: date,
    retrieved_at: datetime,
    created_at: datetime,
    git_commit: str,
    job_id: str = "equity-research-vertical-slice",
    params: EvaluationParams | None = None,
    evaluator=evaluate_partitions,
    registry_publish=_fail_open_registry,
) -> SliceResult:
    """Run the full folds -> job -> worker -> import flow for one hypothesis."""
    params = _resolve_params(partitions, params)
    produced = produce_worker_output(
        identities, partitions,
        hypothesis_id=hypothesis_id, dataset_id=dataset_id,
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
    "EQUITY_RESEARCH_SCORECARD_VERSION",
    "SliceIdentities",
    "SliceResult",
    "Produced",
    "build_slice_identities",
    "build_evidence_store",
    "produce_worker_output",
    "import_worker_output",
    "run_equity_research_evidence_slice",
    "equity_research_evidence_view",
    "utc",
]
