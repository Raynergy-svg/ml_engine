"""End-to-end risk-target evidence vertical slice + read-only cockpit view.

``run_risk_target_evidence_slice`` executes the whole roadmap §12 flow for one
signed job: it builds and signs the dataset / capability / job manifests, runs
the authority-free worker, and imports every produced head into the local
evidence store, ending each at QUARANTINED or REJECTED. A candidate never
overwrites an incumbent — the slice never promotes a champion.

``risk_target_evidence_view`` is a display-only reader over the store's own
committed index projection (no signing keys required), suitable for the
dashboard evidence cockpit.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Mapping

from src.evidence.canonical import canonical_bytes
from src.evidence.contracts import AuthorityRole, SignedEnvelope
from src.evidence.hashing import sha256_bytes
from src.evidence.signing import Ed25519Signer, TrustStore
from src.evidence.store import EvidenceStore
from src.evidence.transition_policy import AuthorityRegistry

from .dashboard import risk_target_evidence_view
from .evaluation import EvaluationParams, evaluate_partitions
from .local_import import import_head
from .manifests import (
    build_capability_profile,
    build_fx_daily_dataset_manifest,
    build_risk_target_job_manifest,
)
from src.evidence.contracts import CapabilityProfile, JobManifest

from .models import HeadImportOutcome, WorkerOutput
from .worker import _fail_open_registry, run_worker

# The producer, local importer and independent verifier must be distinct
# actors and keys — separation of duties is enforced cryptographically by the
# store. This slice needs exactly these three (champion/operator authority is
# out of scope: the slice never promotes).
_SLICE_ROLES = (
    AuthorityRole.PRODUCER,
    AuthorityRole.LOCAL_IMPORTER,
    AuthorityRole.INDEPENDENT_VERIFIER,
)
_DEFAULT_ACTORS = {
    AuthorityRole.PRODUCER: "risk-target-remote-worker",
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


@dataclass(frozen=True)
class Produced:
    """The producer half's output plus the local objects needed to import it."""

    job: JobManifest
    capability_profile: CapabilityProfile
    worker_output: WorkerOutput


@dataclass(frozen=True)
class SliceResult:
    job_envelope: SignedEnvelope
    dataset_manifest_envelope: SignedEnvelope
    worker_output: WorkerOutput
    outcomes: Mapping[str, HeadImportOutcome] = field(default_factory=dict)


def _config_digest(params: EvaluationParams, feature_version: str) -> str:
    payload = {
        "params": {
            "horizon_bars": params.horizon_bars,
            "oos_start": params.oos_start,
            "val_frac": params.val_frac,
            "stressed_quantile": params.stressed_quantile,
            "n_estimators": params.n_estimators,
            "max_depth": params.max_depth,
            "learning_rate": params.learning_rate,
            "early_stopping_rounds": params.early_stopping_rounds,
            "seed": params.seed,
        },
        "feature_pipeline_version": feature_version,
    }
    return sha256_bytes(canonical_bytes(payload))


def produce_worker_output(
    identities: SliceIdentities,
    partitions: Mapping[str, bytes],
    *,
    dataset_id: str,
    coverage_start: date,
    coverage_end: date,
    retrieved_at: datetime,
    created_at: datetime,
    git_commit: str,
    job_id: str = "risk-target-vertical-slice",
    params: EvaluationParams | None = None,
    evaluator=evaluate_partitions,
    registry_publish=_fail_open_registry,
    feature_pipeline_version: str | None = None,
) -> Produced:
    """Build+sign the manifests and run the authority-free worker (no import)."""
    params = params or EvaluationParams()
    if feature_pipeline_version is None:
        from src.training.risk_target_features import RISK_TARGET_FEATURE_PIPELINE_VERSION

        feature_pipeline_version = RISK_TARGET_FEATURE_PIPELINE_VERSION

    producer = identities.producer
    producer_id = identities.actors[AuthorityRole.PRODUCER]

    capability_profile = build_capability_profile()
    capability_envelope = producer.sign(capability_profile, created_at=created_at)

    dataset_manifest = build_fx_daily_dataset_manifest(
        partitions,
        dataset_id=dataset_id,
        retrieved_at=retrieved_at,
        coverage_start=coverage_start,
        coverage_end=coverage_end,
        row_counts={inst: data.count(b"\n") for inst, data in partitions.items()},
    )
    dataset_envelope = producer.sign(dataset_manifest, created_at=created_at)

    container_digest = sha256_bytes(b"axiom-risk-target:local-inprocess:v1")
    job = build_risk_target_job_manifest(
        job_id=job_id,
        dataset_manifest=dataset_manifest,
        capability_profile=capability_profile,
        git_commit=git_commit,
        container_digest=container_digest,
        configuration_digest=_config_digest(params, feature_pipeline_version),
        feature_pipeline_version=feature_pipeline_version,
        random_seeds=(params.seed,),
        created_at=created_at,
    )
    job_envelope = producer.sign(job, created_at=created_at)

    worker_output = run_worker(
        job_envelope=job_envelope,
        dataset_manifest_envelope=dataset_envelope,
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
    return Produced(job=job, capability_profile=capability_profile, worker_output=worker_output)


def import_worker_output(
    store: EvidenceStore,
    identities: SliceIdentities,
    produced: Produced,
    partitions: Mapping[str, bytes],
    *,
    created_at: datetime,
    params: EvaluationParams | None = None,
    evaluator=evaluate_partitions,
) -> dict[str, HeadImportOutcome]:
    """Import every produced head into the local store (QUARANTINED/REJECTED)."""
    params = params or EvaluationParams()
    outcomes: dict[str, HeadImportOutcome] = {}
    for head in produced.worker_output.heads:
        outcomes[head.lane_id] = import_head(
            store,
            head,
            job=produced.job,
            capability_profile_envelope=produced.worker_output.capability_profile_envelope,
            dataset_manifest_envelope=produced.worker_output.dataset_manifest_envelope,
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


def run_risk_target_evidence_slice(
    store: EvidenceStore,
    identities: SliceIdentities,
    partitions: Mapping[str, bytes],
    *,
    dataset_id: str,
    coverage_start: date,
    coverage_end: date,
    retrieved_at: datetime,
    created_at: datetime,
    git_commit: str,
    job_id: str = "risk-target-vertical-slice",
    params: EvaluationParams | None = None,
    evaluator=evaluate_partitions,
    registry_publish=_fail_open_registry,
    feature_pipeline_version: str | None = None,
) -> SliceResult:
    """Run the full dataset -> job -> worker -> import flow for one job."""
    params = params or EvaluationParams()
    produced = produce_worker_output(
        identities, partitions,
        dataset_id=dataset_id, coverage_start=coverage_start, coverage_end=coverage_end,
        retrieved_at=retrieved_at, created_at=created_at, git_commit=git_commit,
        job_id=job_id, params=params, evaluator=evaluator,
        registry_publish=registry_publish, feature_pipeline_version=feature_pipeline_version,
    )
    outcomes = import_worker_output(
        store, identities, produced, partitions,
        created_at=created_at, params=params, evaluator=evaluator,
    )
    return SliceResult(
        job_envelope=produced.worker_output.job_envelope,
        dataset_manifest_envelope=produced.worker_output.dataset_manifest_envelope,
        worker_output=produced.worker_output,
        outcomes=outcomes,
    )


def utc(year: int, month: int, day: int, hour: int = 0, minute: int = 0) -> datetime:
    return datetime(year, month, day, hour, minute, tzinfo=timezone.utc)


__all__ = [
    "SliceIdentities",
    "SliceResult",
    "Produced",
    "build_slice_identities",
    "build_evidence_store",
    "produce_worker_output",
    "import_worker_output",
    "run_risk_target_evidence_slice",
    "risk_target_evidence_view",
    "utc",
]
