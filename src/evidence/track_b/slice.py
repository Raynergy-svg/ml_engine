"""End-to-end builders for separate Track B scoring and evaluation jobs."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Mapping

from src.evidence.canonical import canonical_bytes
from src.evidence.contracts import AuthorityRole, ResourceClass
from src.evidence.hashing import sha256_bytes
from src.evidence.signing import Ed25519Signer, TrustStore
from src.evidence.store import EvidenceStore
from src.evidence.transition_policy import AuthorityRegistry

from .evaluation import EvaluationParams
from .lineage import PROMPT_DIGEST, SCORER_MODEL, SCORER_VERSION
from .local_import import import_head
from .manifests import (
    build_capability_profile, build_dataset_manifest,
    build_evaluation_strategy_manifest, build_job_manifest,
    build_scoring_strategy_manifest,
)
from .models import TrackBImportOutcome, TrackBWorkerOutput, lane_id
from .worker import run_evaluation_worker, run_scoring_worker

TRACK_B_PIPELINE_VERSION = "track_b_evidence.v1"
_ROLES = (
    AuthorityRole.PRODUCER,
    AuthorityRole.LOCAL_IMPORTER,
    AuthorityRole.INDEPENDENT_VERIFIER,
)
_ACTORS = {
    AuthorityRole.PRODUCER: "track-b-remote-worker",
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


@dataclass(frozen=True)
class Produced:
    job: object
    worker_output: TrackBWorkerOutput


@dataclass(frozen=True)
class SliceResult:
    produced: Produced
    outcomes: Mapping[str, TrackBImportOutcome]


def build_slice_identities(
    now: datetime, *, actors: Mapping[AuthorityRole, str] | None = None
) -> SliceIdentities:
    actor_map = {**_ACTORS, **(actors or {})}
    signers = {role: Ed25519Signer.generate() for role in _ROLES}
    trust = TrustStore()
    authorities = AuthorityRegistry()
    for role, signer in signers.items():
        trust.add(signer.trusted_key(valid_from=now - timedelta(days=1)))
        authorities.register(actor_id=actor_map[role], role=role, key_ids=(signer.key_id,))
    return SliceIdentities(signers, actor_map, trust, authorities)


def build_evidence_store(
    root: str | Path, identities: SliceIdentities, *, clock_now: datetime
) -> EvidenceStore:
    return EvidenceStore(
        root, trust_store=identities.trust_store, authorities=identities.authorities,
        trusted_clock=lambda: clock_now,
    )


def _config(payload: object) -> str:
    return sha256_bytes(canonical_bytes(payload))


def produce_scoring(
    identities: SliceIdentities,
    partitions: Mapping[str, bytes],
    *,
    campaign_id: str,
    dataset_id: str,
    model_cutoff: str,
    coverage_start: date,
    coverage_end: date,
    retrieved_at: datetime,
    created_at: datetime,
    git_commit: str,
) -> Produced:
    producer = identities.producer
    capability = build_capability_profile()
    dataset = build_dataset_manifest(
        partitions, dataset_id=dataset_id, domain="track_b_raw_filing_batches",
        retrieved_at=retrieved_at, coverage_start=coverage_start, coverage_end=coverage_end,
    )
    strategy = build_scoring_strategy_manifest(
        campaign_id=campaign_id, expected_batches=tuple(sorted(partitions)),
        scorer_model=SCORER_MODEL, scorer_version=SCORER_VERSION,
        prompt_digest=PROMPT_DIGEST, model_cutoff=model_cutoff, frozen_at=created_at,
    )
    outputs = [
        lane_id("scoring", campaign_id, f"{batch}-{SCORER_VERSION}")
        for batch in sorted(partitions)
    ]
    job = build_job_manifest(
        job_id=f"track-b-scoring-{campaign_id}", dataset_manifest=dataset,
        strategy_manifest=strategy, capability_profile=capability,
        git_commit=git_commit, container_digest=sha256_bytes(b"axiom-track-b-scoring:v1"),
        configuration_digest=_config(strategy.parameters),
        feature_pipeline_version=TRACK_B_PIPELINE_VERSION, created_at=created_at,
        expected_outputs=outputs, assigned_unit="filing_batch/scorer_version",
        resource_class=ResourceClass.DOCUMENT_PROCESSING,
    )
    envelopes = (
        producer.sign(job, created_at=created_at),
        producer.sign(dataset, created_at=created_at),
        producer.sign(strategy, created_at=created_at),
        producer.sign(capability, created_at=created_at),
    )
    output = run_scoring_worker(
        campaign_id=campaign_id, job_envelope=envelopes[0],
        dataset_manifest_envelope=envelopes[1], strategy_manifest_envelope=envelopes[2],
        capability_profile_envelope=envelopes[3], partitions=partitions,
        producer=producer, producer_id=identities.actors[AuthorityRole.PRODUCER],
        trust_store=identities.trust_store, created_at=created_at,
    )
    return Produced(job=job, worker_output=output)


def produce_evaluation(
    identities: SliceIdentities,
    partitions: Mapping[str, bytes],
    *,
    campaign_id: str,
    dataset_id: str,
    scoring_package_digests: tuple[str, ...],
    coverage_start: date,
    coverage_end: date,
    retrieved_at: datetime,
    created_at: datetime,
    git_commit: str,
    params: EvaluationParams | None = None,
) -> Produced:
    params = params or EvaluationParams(expected_cells=tuple(sorted(partitions)))
    if not params.expected_cells:
        from dataclasses import replace
        params = replace(params, expected_cells=tuple(sorted(partitions)))
    producer = identities.producer
    capability = build_capability_profile()
    dataset = build_dataset_manifest(
        partitions, dataset_id=dataset_id, domain="track_b_forward_evaluation_cells",
        retrieved_at=retrieved_at, coverage_start=coverage_start, coverage_end=coverage_end,
    )
    strategy = build_evaluation_strategy_manifest(
        campaign_id=campaign_id, expected_cells=params.expected_cells,
        scoring_package_digests=scoring_package_digests,
        min_forward_cells=params.min_forward_cells, rank_ic_bar=params.rank_ic_bar,
        placebo_abs_limit=params.placebo_abs_limit, frozen_at=created_at,
    )
    output_lane = lane_id("evaluation", campaign_id, "forward-rank-ic")
    job = build_job_manifest(
        job_id=f"track-b-evaluation-{campaign_id}", dataset_manifest=dataset,
        strategy_manifest=strategy, capability_profile=capability,
        git_commit=git_commit, container_digest=sha256_bytes(b"axiom-track-b-harness:v1"),
        configuration_digest=_config(strategy.parameters),
        feature_pipeline_version=TRACK_B_PIPELINE_VERSION, created_at=created_at,
        expected_outputs=(output_lane,), assigned_unit="score_artifact/rebalance_date/fold",
        resource_class=ResourceClass.CPU_SMALL,
    )
    envelopes = (
        producer.sign(job, created_at=created_at),
        producer.sign(dataset, created_at=created_at),
        producer.sign(strategy, created_at=created_at),
        producer.sign(capability, created_at=created_at),
    )
    output = run_evaluation_worker(
        campaign_id=campaign_id, job_envelope=envelopes[0],
        dataset_manifest_envelope=envelopes[1], strategy_manifest_envelope=envelopes[2],
        capability_profile_envelope=envelopes[3], partitions=partitions,
        scoring_package_digests=scoring_package_digests,
        producer=producer, producer_id=identities.actors[AuthorityRole.PRODUCER],
        trust_store=identities.trust_store, created_at=created_at, params=params,
    )
    return Produced(job=job, worker_output=output)


def import_produced(
    store: EvidenceStore,
    identities: SliceIdentities,
    produced: Produced,
    partitions: Mapping[str, bytes],
    *,
    created_at: datetime,
) -> dict[str, TrackBImportOutcome]:
    outcomes: dict[str, TrackBImportOutcome] = {}
    for head in produced.worker_output.heads:
        outcomes[head.lane_id] = import_head(
            store, head, job=produced.job,
            capability_profile_envelope=produced.worker_output.capability_profile_envelope,
            dataset_manifest_envelope=produced.worker_output.dataset_manifest_envelope,
            strategy_manifest_envelope=produced.worker_output.strategy_manifest_envelope,
            partitions=partitions, trust_store=identities.trust_store,
            importer=identities.importer,
            importer_id=identities.actors[AuthorityRole.LOCAL_IMPORTER],
            verifier=identities.verifier,
            verifier_id=identities.actors[AuthorityRole.INDEPENDENT_VERIFIER],
            created_at=created_at,
        )
    return outcomes


def utc(year: int, month: int, day: int, hour: int = 0) -> datetime:
    return datetime(year, month, day, hour, tzinfo=timezone.utc)


__all__ = [
    "SliceIdentities", "Produced", "SliceResult", "build_slice_identities",
    "build_evidence_store", "produce_scoring", "produce_evaluation",
    "import_produced", "utc",
]
