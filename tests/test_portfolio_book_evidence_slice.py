"""Phase K acceptance tests — portfolio allocator / combined-book evidence slice.

NO MOCKS: every test runs the actual evidence store, real Ed25519 signing, the
real disposition state machine (``src.evidence.transition_policy``), and the
REAL deterministic allocator (``src.evidence.portfolio_book.evaluation``) over
real JSONL sleeve-return rows — never a test-double.

Because the allocator is deterministic pure float arithmetic, the default
evaluator is exercised directly: the worker and the importer's independent
replay obtain identical results from identical partition bytes, exactly the
reproducibility contract the store enforces.

The book's derivation-binding invariant ("no lane, and no book, becomes
capital-active from a standalone gate") requires every contributing sleeve to
already be an independently QUARANTINED package in the SAME local store, AND
every row of its return-series partition to carry that package's digest
(``src.evidence.portfolio_book.lineage``) so the binding is cryptographically
tied to the hash-verified partition bytes, not just asserted in the signed
StrategyManifest. Since staging five full upstream evidence slices per test
would be prohibitively heavy, ``_stage_source`` builds a minimal REAL
EvidencePackage and walks it through the REAL disposition chain (CREATED ->
... -> QUARANTINED, including a real accepted LocalImportVerdict) to stand in
for an already-quarantined sleeve. This is real contract objects and real
store machinery — a thin fixture, not a mock of the system under test.
"""

from __future__ import annotations

import datetime as dt
import json
import random
from datetime import date

import pytest

from src.evidence.contracts import (
    ArtifactRef, AuthorityRole, DispositionEvent, DispositionState,
    EvidencePackage, ImportCheck, ImportDecision, LocalImportVerdict, SafetyAssertion,
)
from src.evidence.hashing import sha256_bytes
from src.evidence.portfolio_book.evaluation import EvaluationParams
from src.evidence.portfolio_book.local_import import import_book
from src.evidence.portfolio_book.slice import (
    build_evidence_store, build_slice_identities, produce_worker_output,
    run_portfolio_book_evidence_slice, utc,
)
from src.evidence.portfolio_book.worker import DatasetHashError, JobVerificationError, run_worker

NOW = utc(2026, 7, 22, 12)
SLICE_KW = dict(
    dataset_id="portfolio-book-test", coverage_start=date(2025, 1, 1),
    coverage_end=date(2026, 3, 1), retrieved_at=NOW, created_at=NOW, git_commit="0" * 40,
)
DEFAULT_PARAMS = EvaluationParams(oos_start="2025-11-01", min_est_days=150, min_oos_days=20)
N_DAYS = 300
SLEEVE_SPECS = {  # sleeve_id -> (mu, sigma, seed)
    "risk_target": (0.0006, 0.006, 1),
    "hedge_eval": (0.0004, 0.004, 2),
    "equity_research": (0.0005, 0.008, 3),
}


def _make_partition(mu: float, sigma: float, n_days: int, *, seed: int, source_package_digest: str, start: str = "2025-01-01") -> bytes:
    rng = random.Random(seed)
    day = date.fromisoformat(start)
    rows = []
    count = 0
    while count < n_days:
        if day.weekday() < 5:
            rows.append({
                "date": day.isoformat(), "net_return": rng.gauss(mu, sigma),
                "gross_exposure": 1.0, "turnover": 0.05, "source_package_digest": source_package_digest,
            })
            count += 1
        day += dt.timedelta(days=1)
    return ("\n".join(json.dumps(r) for r in rows) + "\n").encode("utf-8")


def _stage_source(store, identities, sleeve_id: str, now, *, final_state=DispositionState.QUARANTINED) -> str:
    """Stage a minimal REAL EvidencePackage for ``sleeve_id`` and walk it
    through the REAL disposition chain up to (and optionally stopping before)
    QUARANTINED. Returns the package digest."""
    producer = identities.signers[AuthorityRole.PRODUCER]
    importer = identities.signers[AuthorityRole.LOCAL_IMPORTER]
    verifier = identities.signers[AuthorityRole.INDEPENDENT_VERIFIER]
    artifact_bytes = json.dumps({"sleeve": sleeve_id}).encode()
    path = f"artifacts/{sleeve_id}.json"
    artifact = ArtifactRef(
        artifact_id=f"{sleeve_id}-src", relative_path=path, digest=sha256_bytes(artifact_bytes),
        size_bytes=len(artifact_bytes), media_type="application/json",
    )
    package = EvidencePackage(
        package_id=f"source-{sleeve_id}", lane_id=sleeve_id,
        producer_id=identities.actors[AuthorityRole.PRODUCER], created_at=now,
        job_manifest_digest="0" * 64, dataset_manifest_digests=("1" * 64,),
        evaluation_report_digests=("2" * 64,), artifacts=(artifact,),
        safety_assertions=(SafetyAssertion(assertion_id="seed_ok", passed=True),),
        checksums={path: artifact.digest},
    )
    digest = store.write_package(producer.sign(package, created_at=now), {path: artifact_bytes})
    verdict = LocalImportVerdict(
        verdict_id=f"{sleeve_id}-verdict", package_digest=digest,
        importer_id=identities.actors[AuthorityRole.LOCAL_IMPORTER], created_at=now,
        checks=(ImportCheck(check_id="seed_ok", passed=True),), decision=ImportDecision.ACCEPT_FOR_QUARANTINE,
    )
    verdict_digest = store.write_verdict(importer.sign(verdict, created_at=now))
    full_chain = [
        (None, DispositionState.CREATED, producer, AuthorityRole.PRODUCER, {}),
        (DispositionState.CREATED, DispositionState.RECEIVED, importer, AuthorityRole.LOCAL_IMPORTER, {}),
        (DispositionState.RECEIVED, DispositionState.HASH_VERIFIED, importer, AuthorityRole.LOCAL_IMPORTER, {}),
        (DispositionState.HASH_VERIFIED, DispositionState.POLICY_VERIFIED, importer, AuthorityRole.LOCAL_IMPORTER, {}),
        (DispositionState.POLICY_VERIFIED, DispositionState.METRIC_REPLAYED, verifier, AuthorityRole.INDEPENDENT_VERIFIER, {}),
        (DispositionState.METRIC_REPLAYED, DispositionState.QUARANTINED, importer, AuthorityRole.LOCAL_IMPORTER, {"verdict_digest": verdict_digest}),
    ]
    head = None
    for sequence, (frm, to, signer, role, meta) in enumerate(full_chain):
        event = DispositionEvent(
            package_digest=digest, sequence=sequence, previous_event_digest=head, from_state=frm, to_state=to,
            authority=role, actor_id=identities.actors[role], signer_key_id=signer.key_id, occurred_at=now,
            reason="seed", metadata=meta,
        )
        state = store.append_disposition(signer.sign(event, created_at=now), expected_head_digest=head)
        head = state.head_event_digest
        if to == final_state:
            break
    return digest


def _stage_sources_and_partitions(store, identities, sleeve_ids, now, *, quarantine: bool = True):
    """Stage a real quarantined (or not) source package per sleeve, then build
    partitions whose rows embed that sleeve's own staged digest — the ordering
    that ``lineage.verify_source_lineage`` requires."""
    final = DispositionState.QUARANTINED if quarantine else DispositionState.CREATED
    source_digests = {s: _stage_source(store, identities, s, now, final_state=final) for s in sleeve_ids}
    partitions = {
        s: _make_partition(*SLEEVE_SPECS[s][:2], N_DAYS, seed=SLEEVE_SPECS[s][2], source_package_digest=source_digests[s])
        for s in sleeve_ids
    }
    return partitions, source_digests


def _fresh(tmp_path):
    identities = build_slice_identities(NOW)
    store = build_evidence_store(tmp_path / "evidence", identities, clock_now=NOW)
    return identities, store


def test_full_lifecycle_reaches_quarantine(tmp_path):
    identities, store = _fresh(tmp_path)
    partitions, source_digests = _stage_sources_and_partitions(store, identities, SLEEVE_SPECS, NOW)
    result = run_portfolio_book_evidence_slice(
        store, identities, partitions, source_package_digests=source_digests,
        params=DEFAULT_PARAMS, **SLICE_KW,
    )
    assert result.outcome.final_state == DispositionState.QUARANTINED
    assert result.worker_output.book.package.lane_id == "portfolio_book"
    assert result.worker_output.book.package.derived_from_package_digests == tuple(sorted(source_digests.values()))


def test_replay_reproduces_metrics_and_gate(tmp_path):
    identities, store = _fresh(tmp_path)
    partitions, source_digests = _stage_sources_and_partitions(store, identities, SLEEVE_SPECS, NOW)
    result = run_portfolio_book_evidence_slice(
        store, identities, partitions, source_package_digests=source_digests,
        params=DEFAULT_PARAMS, **SLICE_KW,
    )
    verdict = result.outcome.verdict
    assert verdict.decision == ImportDecision.ACCEPT_FOR_QUARANTINE
    assert all(check.passed for check in verdict.checks)


def test_source_package_not_quarantined_is_rejected(tmp_path):
    identities, store = _fresh(tmp_path)
    partitions, source_digests = _stage_sources_and_partitions(store, identities, SLEEVE_SPECS, NOW, quarantine=False)
    result = run_portfolio_book_evidence_slice(
        store, identities, partitions, source_package_digests=source_digests,
        params=DEFAULT_PARAMS, **SLICE_KW,
    )
    assert result.outcome.final_state == DispositionState.REJECTED
    failed = {check.check_id for check in result.outcome.verdict.checks if not check.passed}
    assert "source_packages_quarantined" in failed


def test_missing_source_package_is_rejected(tmp_path):
    identities, store = _fresh(tmp_path)
    partitions, source_digests = _stage_sources_and_partitions(store, identities, SLEEVE_SPECS, NOW)
    fake_digest = "f" * 64  # never written to this store
    source_digests["equity_research"] = fake_digest
    mu, sigma, seed = SLEEVE_SPECS["equity_research"]
    partitions["equity_research"] = _make_partition(mu, sigma, N_DAYS, seed=seed, source_package_digest=fake_digest)
    result = run_portfolio_book_evidence_slice(
        store, identities, partitions, source_package_digests=source_digests,
        params=DEFAULT_PARAMS, **SLICE_KW,
    )
    assert result.outcome.final_state == DispositionState.REJECTED
    failed = {check.check_id for check in result.outcome.verdict.checks if not check.passed}
    assert "source_packages_quarantined" in failed


def test_forged_source_digest_binding_is_refused_at_the_worker(tmp_path):
    """A caller cannot claim an unrelated-but-genuinely-quarantined package as
    a sleeve's source without also controlling that sleeve's partition bytes:
    the row-embedded digest must match the signed strategy's claimed digest.
    The reference worker refuses to even sign a package over this mismatch
    (defense before a package is ever produced, not merely at import time)."""
    identities, store = _fresh(tmp_path)
    partitions, source_digests = _stage_sources_and_partitions(store, identities, SLEEVE_SPECS, NOW)
    # Swap in a digest for a DIFFERENT, still-quarantined sleeve — legitimate
    # package, wrong sleeve — without touching the partition rows.
    forged = dict(source_digests)
    forged["equity_research"] = source_digests["hedge_eval"]
    with pytest.raises(JobVerificationError, match="source_package_digest"):
        run_portfolio_book_evidence_slice(
            store, identities, partitions, source_package_digests=forged,
            params=DEFAULT_PARAMS, **SLICE_KW,
        )


def test_forged_source_digest_binding_is_rejected_even_if_a_worker_skips_the_check(tmp_path):
    """Defense in depth: the LOCAL IMPORTER (the actual trust boundary — the
    worker is a remote, less-trusted producer) must independently catch the
    same forged binding even if a non-reference worker implementation never
    called ``verify_source_lineage`` itself. Constructs the signed package by
    hand, bypassing ``run_worker``'s own guard, to prove the importer-side
    check is load-bearing on its own."""
    identities, store = _fresh(tmp_path)
    partitions, source_digests = _stage_sources_and_partitions(store, identities, SLEEVE_SPECS, NOW)
    from src.evidence.portfolio_book.manifests import (
        build_capability_profile, build_dataset_manifest, build_job_manifest, build_strategy_manifest,
    )
    from src.evidence.portfolio_book.worker import _package as package_book

    forged = dict(source_digests)
    forged["equity_research"] = source_digests["hedge_eval"]  # legitimate package, wrong sleeve
    producer = identities.signers[AuthorityRole.PRODUCER]
    capability = build_capability_profile()
    capability_envelope = producer.sign(capability, created_at=NOW)
    dataset = build_dataset_manifest(
        partitions, dataset_id=SLICE_KW["dataset_id"], retrieved_at=NOW,
        coverage_start=SLICE_KW["coverage_start"], coverage_end=SLICE_KW["coverage_end"],
    )
    dataset_envelope = producer.sign(dataset, created_at=NOW)
    strategy = build_strategy_manifest(
        strategy_id="forged-strategy", experiment_id="forged-experiment",
        sleeve_ids=sorted(partitions), source_package_digests=forged, params=DEFAULT_PARAMS, frozen_at=NOW,
    )
    strategy_envelope = producer.sign(strategy, created_at=NOW)
    job = build_job_manifest(
        job_id="forged-job", dataset_manifest=dataset, strategy_manifest=strategy, capability_profile=capability,
        git_commit=SLICE_KW["git_commit"], container_digest=sha256_bytes(b"forged"),
        configuration_digest=sha256_bytes(b"forged-config"), feature_pipeline_version="portfolio_book_features.v1",
        created_at=NOW,
    )
    job_envelope = producer.sign(job, created_at=NOW)
    from src.evidence.portfolio_book.evaluation import evaluate_partitions
    result = evaluate_partitions(partitions, DEFAULT_PARAMS)
    book = package_book(result, job=job, job_digest=job_envelope.payload_digest, strategy=strategy,
                        producer=producer, producer_id=identities.actors[AuthorityRole.PRODUCER], created_at=NOW)
    outcome = import_book(
        store, book, job_envelope=job_envelope, capability_profile_envelope=capability_envelope,
        dataset_manifest_envelope=dataset_envelope, strategy_manifest_envelope=strategy_envelope,
        partitions=partitions, trust_store=identities.trust_store,
        importer=identities.signers[AuthorityRole.LOCAL_IMPORTER], importer_id=identities.actors[AuthorityRole.LOCAL_IMPORTER],
        verifier=identities.signers[AuthorityRole.INDEPENDENT_VERIFIER], verifier_id=identities.actors[AuthorityRole.INDEPENDENT_VERIFIER],
        created_at=NOW,
    )
    assert outcome.final_state == DispositionState.REJECTED
    failed = {check.check_id for check in outcome.verdict.checks if not check.passed}
    assert "source_lineage_bound_in_partitions" in failed


def test_worker_rejects_sleeve_universe_mismatch(tmp_path):
    identities, store = _fresh(tmp_path)
    partitions, source_digests = _stage_sources_and_partitions(store, identities, SLEEVE_SPECS, NOW)
    produced = produce_worker_output(
        identities, partitions, source_package_digests=source_digests, params=DEFAULT_PARAMS, **SLICE_KW,
    )
    # Tamper the supplied partitions after signing so the universe no longer matches.
    tampered = dict(partitions)
    del tampered["equity_research"]
    with pytest.raises((JobVerificationError, DatasetHashError)):
        run_worker(
            job_envelope=produced.worker_output.job_envelope,
            dataset_manifest_envelope=produced.worker_output.dataset_manifest_envelope,
            strategy_manifest_envelope=produced.worker_output.strategy_manifest_envelope,
            capability_profile_envelope=produced.worker_output.capability_profile_envelope,
            partitions=tampered, producer=identities.signers[AuthorityRole.PRODUCER],
            producer_id=identities.actors[AuthorityRole.PRODUCER], trust_store=identities.trust_store,
            created_at=NOW,
        )


def test_changed_sleeve_partition_after_signing_causes_hash_failure(tmp_path):
    identities, store = _fresh(tmp_path)
    partitions, source_digests = _stage_sources_and_partitions(store, identities, SLEEVE_SPECS, NOW)
    produced = produce_worker_output(
        identities, partitions, source_package_digests=source_digests, params=DEFAULT_PARAMS, **SLICE_KW,
    )
    tampered = dict(partitions)
    tampered["risk_target"] = _make_partition(
        0.05, 0.006, 300, seed=99, source_package_digest=source_digests["risk_target"],
    )  # different bytes, same key set and same claimed digest
    outcome = import_book(
        store, produced.worker_output.book, job_envelope=produced.worker_output.job_envelope,
        capability_profile_envelope=produced.worker_output.capability_profile_envelope,
        dataset_manifest_envelope=produced.worker_output.dataset_manifest_envelope,
        strategy_manifest_envelope=produced.worker_output.strategy_manifest_envelope,
        partitions=tampered, trust_store=identities.trust_store,
        importer=identities.signers[AuthorityRole.LOCAL_IMPORTER], importer_id=identities.actors[AuthorityRole.LOCAL_IMPORTER],
        verifier=identities.signers[AuthorityRole.INDEPENDENT_VERIFIER], verifier_id=identities.actors[AuthorityRole.INDEPENDENT_VERIFIER],
        created_at=NOW,
    )
    assert outcome.final_state == DispositionState.REJECTED
    failed = {check.check_id for check in outcome.verdict.checks if not check.passed}
    assert "dataset_hashes_verified" in failed


def test_book_can_never_reach_operator_approved_or_champion(tmp_path):
    identities, store = _fresh(tmp_path)
    partitions, source_digests = _stage_sources_and_partitions(store, identities, SLEEVE_SPECS, NOW)
    result = run_portfolio_book_evidence_slice(
        store, identities, partitions, source_package_digests=source_digests,
        params=DEFAULT_PARAMS, **SLICE_KW,
    )
    assert result.outcome.final_state == DispositionState.QUARANTINED
    with pytest.raises(Exception):
        store.load_champion_pointer("portfolio_book")


def test_insufficient_sleeves_is_a_worker_error(tmp_path):
    identities, store = _fresh(tmp_path)
    digest = "a" * 64
    single = {"risk_target": _make_partition(0.0006, 0.006, 300, seed=1, source_package_digest=digest)}
    source_digests = {"risk_target": digest}
    with pytest.raises(Exception):
        run_portfolio_book_evidence_slice(
            store, identities, single, source_package_digests=source_digests,
            params=DEFAULT_PARAMS, **SLICE_KW,
        )
