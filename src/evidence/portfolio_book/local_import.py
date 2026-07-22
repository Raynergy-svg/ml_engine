"""Local independent verification for portfolio-book combined-book evidence."""

from __future__ import annotations

import json
from datetime import datetime, timedelta
from typing import Mapping

from src.evidence.contracts import (
    AuthorityRole, CapabilityProfile, DatasetManifest, DispositionEvent,
    DispositionState, EvaluationReport, ImportCheck, JobManifest, StrategyManifest,
)
from src.evidence.hashing import content_digest, sha256_bytes
from src.evidence.importer import build_import_verdict
from src.evidence.signing import Ed25519Signer, TrustStore, verify_envelope
from src.evidence.store import EvidenceStore

from .evaluation import evaluate_partitions, METRICS
from .lineage import LineageError, verify_source_lineage
from .manifests import params_from_dict
from .models import ImportOutcome, PackagedBook

# The path convention a source lane's own evidence package must use if it wants
# portfolio_book to be able to verify its sleeve's return data byte-for-byte
# against that lane's own signed artifact (roadmap §14's standardized
# strategy-return/exposure contract). Only src/evidence/hedge_eval/ emits this
# today (retrofitted 2026-07-22); a source lane without it is not penalized —
# the check below degrades to existence+quarantine-only for that sleeve and
# says so explicitly in its details, rather than silently claiming a strength
# of binding that does not exist.
STRATEGY_RETURN_CONTRACT_PATH = "artifacts/strategy_returns.jsonl"


def _check(check_id: str, passed: bool, details: str) -> ImportCheck:
    return ImportCheck(check_id=check_id, passed=passed, details=details)


def _authority_free(profile: CapabilityProfile) -> bool:
    return profile.network_endpoints == () and all(
        getattr(profile, field) is False
        for field in (
            "may_read_broker_credentials", "may_place_or_cancel_orders",
            "may_read_operator_keys", "may_change_halts", "may_change_live_gate",
            "may_write_champion_pointer", "may_modify_local_models", "may_approve_evidence",
        )
    )


def _source_packages_quarantined(store: EvidenceStore, source_digests: Mapping[str, str]) -> tuple[ImportCheck, ...]:
    """No lane, and no book, becomes capital-active from a standalone gate: every
    sleeve this book claims to combine must already be an independently
    QUARANTINED package in THIS local store before the book itself is eligible
    for quarantine. Mirrors Track B's ``_evaluation_derivation_checks``."""
    try:
        index = json.loads(store.current_index_bytes())
    except Exception as exc:  # noqa: BLE001
        return (_check("source_packages_quarantined", False, f"cannot load local index: {exc}"),)
    packages_index = index.get("packages", {}) if isinstance(index, dict) else {}
    missing = []
    not_quarantined = []
    for sleeve_id, digest in sorted(source_digests.items()):
        info = packages_index.get(digest)
        if info is None:
            missing.append(sleeve_id)
        elif info.get("state") != "QUARANTINED":
            not_quarantined.append(sleeve_id)
    ok = not missing and not not_quarantined
    details = "all source sleeve packages are locally QUARANTINED"
    if not ok:
        details = f"missing={missing} not_quarantined={not_quarantined}"
    return (_check("source_packages_quarantined", ok, details),)


_CONTRACT_COMPARISON_FIELDS = ("net_return", "gross_exposure", "turnover")
_CONTRACT_COMPARISON_TOLERANCE = 1e-9  # mirrors replay_tolerance elsewhere in this module


def _contract_field_matches(declared: object, supplied: object) -> bool:
    if declared is None or supplied is None:
        return declared == supplied
    if isinstance(declared, (int, float)) and isinstance(supplied, (int, float)):
        return abs(float(declared) - float(supplied)) <= _CONTRACT_COMPARISON_TOLERANCE
    return declared == supplied


def _parse_contract_rows_by_date(data: bytes) -> dict[str, dict]:
    rows: dict[str, dict] = {}
    for line in data.decode("utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        rows[row["date"]] = row
    return rows


def _source_content_bound(
    store: EvidenceStore, source_digests: Mapping[str, str], partitions: Mapping[str, bytes],
) -> tuple[ImportCheck, ...]:
    """Where a source lane publishes its OWN standardized strategy-return
    artifact (``STRATEGY_RETURN_CONTRACT_PATH``), require the sleeve partition
    supplied to this book to agree with it on every declared trading date's
    ``net_return``/``gross_exposure``/``turnover`` — proof the return VALUES
    were actually computed by that lane, not merely that a legitimate-but-
    unrelated digest was cited (the gap ``source_packages_quarantined`` alone
    left open). Compared by parsed row content rather than raw bytes: this
    book's own partition schema additionally embeds a
    ``source_package_digest`` lineage field (``lineage.py``) the source lane's
    contract does not and should not know about, so byte-identity would always
    fail for a schema difference that isn't itself evidence of tampering. A
    source lane that does not yet publish this artifact is not penalized: this
    degrades to a no-op for that sleeve, reported honestly in the details
    rather than silently claimed as content-verified."""
    mismatched: list[str] = []
    unverifiable: list[str] = []
    for sleeve_id, digest in sorted(source_digests.items()):
        try:
            package = store.load_package(digest)
        except Exception:  # noqa: BLE001 - existence/quarantine already checked separately
            unverifiable.append(sleeve_id)
            continue
        has_contract = any(a.relative_path == STRATEGY_RETURN_CONTRACT_PATH for a in package.artifacts)
        if not has_contract:
            unverifiable.append(sleeve_id)
            continue
        try:
            declared_rows = _parse_contract_rows_by_date(store.load_package_file(digest, STRATEGY_RETURN_CONTRACT_PATH))
            supplied_rows = _parse_contract_rows_by_date(partitions[sleeve_id])
        except Exception:  # noqa: BLE001
            mismatched.append(sleeve_id)
            continue
        if set(declared_rows) != set(supplied_rows):
            mismatched.append(sleeve_id)
            continue
        for date, declared_row in declared_rows.items():
            supplied_row = supplied_rows[date]
            if any(
                not _contract_field_matches(declared_row.get(field), supplied_row.get(field))
                for field in _CONTRACT_COMPARISON_FIELDS
            ):
                mismatched.append(sleeve_id)
                break
    ok = not mismatched
    if mismatched:
        details = f"sleeve partition content does not match the source lane's own published return contract: {mismatched}"
    elif unverifiable:
        details = (
            f"content-bound for sleeves whose source lane publishes {STRATEGY_RETURN_CONTRACT_PATH}; "
            f"existence+quarantine only (no published contract to bind against) for: {unverifiable}"
        )
    else:
        details = "every sleeve partition matches its source lane's own published return contract byte-for-byte"
    return (_check("source_content_bound", ok, details),)


def _preflight(
    head: PackagedBook, *, job: JobManifest, capability: CapabilityProfile,
    dataset: DatasetManifest, strategy: StrategyManifest, partitions: Mapping[str, bytes],
    report: EvaluationReport, store: EvidenceStore,
) -> tuple[ImportCheck, ...]:
    package = head.package
    artifact = package.artifacts[0]
    supplied = head.files.get(artifact.relative_path)
    artifact_ok = supplied is not None and len(supplied) == artifact.size_bytes and sha256_bytes(supplied) == artifact.digest
    declared = {ref.partition_id: ref for ref in dataset.partitions}
    dataset_ok = set(partitions) == set(declared) and all(
        len(data) == declared[name].size_bytes and sha256_bytes(data) == declared[name].digest
        for name, data in partitions.items()
    )
    lineage_ok = (
        head.package_envelope.payload_digest == content_digest(package)
        and head.package_digest == head.package_envelope.payload_digest
        and package.job_manifest_digest == content_digest(job)
        and tuple(package.dataset_manifest_digests) == tuple(job.dataset_manifest_digests)
        and package.strategy_manifest_digest == content_digest(strategy)
        and package.strategy_manifest_digest == job.strategy_manifest_digest
        and package.evaluation_report_digests == (head.evaluation_report_envelope.payload_digest,)
        and report.job_manifest_digest == content_digest(job)
        and job.capability_profile_digest == content_digest(capability)
        and content_digest(dataset) in job.dataset_manifest_digests
        and strategy.lane_id == package.lane_id
        and set(strategy.universe) == set(partitions)
    )
    metric_surface_ok = set(report.metrics) == set(METRICS)
    source_digests = {str(k): str(v) for k, v in strategy.parameters["source_package_digests"].items()}
    derivation_bound = tuple(sorted(source_digests.values())) == tuple(sorted(package.derived_from_package_digests))
    quarantine_checks = _source_packages_quarantined(store, source_digests)
    content_bound_checks = _source_content_bound(store, source_digests, partitions)
    try:
        verify_source_lineage(partitions, source_digests)
        source_lineage_ok = True
    except LineageError:
        source_lineage_ok = False
    try:
        params = params_from_dict(strategy.parameters)
        replay = evaluate_partitions(partitions, params)
    except Exception as exc:  # replay failure is evidence, not an importer crash
        return (
            _check("artifact_hash_verified", artifact_ok, "artifact bytes match the signed reference"),
            _check("dataset_hashes_verified", dataset_ok, "all sleeve partition bytes match the signed manifest"),
            _check("lineage_verified", lineage_ok, "job, dataset, strategy, capability and report digests are bound"),
            _check("no_forbidden_capabilities", _authority_free(capability), "worker declares no network, execution or promotion authority"),
            _check("metric_surface_complete", metric_surface_ok, "report contains exactly the frozen metric surface"),
            _check("source_package_derivation_bound", derivation_bound, "declared source_package_digests match the package's derived-from set"),
            _check("source_lineage_bound_in_partitions", source_lineage_ok, "every sleeve row carries its bound source_package_digest"),
            *quarantine_checks,
            *content_bound_checks,
            _check("metric_replay_reproduces", False, f"independent replay error: {exc}"),
            _check("artifact_reproduces", False, "independent replay did not complete"),
            _check("gate_verdict_reproduces", False, "independent replay did not complete"),
            _check("candidate_evaluation_gate_passed", False, "independent replay did not complete"),
        )
    mismatches = []
    for name in METRICS:
        if name not in report.metrics or name not in replay.metrics:
            mismatches.append(f"{name}:absent")
            continue
        tolerance = report.metrics[name].tolerance
        bound = params.replay_tolerance if tolerance is None else tolerance
        if abs(report.metrics[name].value - float(replay.metrics[name])) > bound:
            mismatches.append(name)
    replay_gate = {gate.gate_id: gate.status for gate in replay.gates}
    gate_ok = report.passed == replay.passed and len(report.gates) == len(replay.gates) and all(
        replay_gate.get(gate.gate_id) == gate.status for gate in report.gates
    )
    return (
        _check("artifact_hash_verified", artifact_ok, "artifact bytes match the signed reference"),
        _check("dataset_hashes_verified", dataset_ok, "all sleeve partition bytes match the signed manifest"),
        _check("lineage_verified", lineage_ok, "job, dataset, strategy, capability and report digests are bound"),
        _check("no_forbidden_capabilities", _authority_free(capability), "worker declares no network, execution or promotion authority"),
        _check("metric_surface_complete", metric_surface_ok, "report contains exactly the frozen metric surface"),
        _check("source_package_derivation_bound", derivation_bound, "declared source_package_digests match the package's derived-from set"),
        _check("source_lineage_bound_in_partitions", source_lineage_ok, "every sleeve row carries its bound source_package_digest"),
        *quarantine_checks,
        *content_bound_checks,
        _check("metric_replay_reproduces", not mismatches, "metrics reproduce" if not mismatches else f"mismatched metrics: {mismatches}"),
        _check("artifact_reproduces", supplied == replay.artifact_bytes, "JSON book artifact reproduces byte-for-byte"),
        _check("gate_verdict_reproduces", gate_ok, "independent replay reproduces all gates and verdict"),
        _check("candidate_evaluation_gate_passed", report.passed, "combined book clears its frozen OOS/capacity/tail-correlation/drawdown-budget gates"),
    )


class _Chain:
    def __init__(self, store: EvidenceStore, package_digest: str, created_at: datetime) -> None:
        self.store = store
        self.package_digest = package_digest
        self.created_at = created_at
        self.head: str | None = None
        self.state: DispositionState | None = None
        self.sequence = 0

    def created(self, envelope) -> None:
        state = self.store.append_disposition(envelope, expected_head_digest=None)
        self.head = state.head_event_digest
        self.state = DispositionState.CREATED
        self.sequence = 1

    def append(self, target: DispositionState, signer: Ed25519Signer, actor: str, role: AuthorityRole, reason: str, metadata: dict | None = None) -> None:
        when = self.created_at + timedelta(seconds=self.sequence)
        event = DispositionEvent(
            package_digest=self.package_digest, sequence=self.sequence,
            previous_event_digest=self.head, from_state=self.state, to_state=target,
            authority=role, actor_id=actor, signer_key_id=signer.key_id, occurred_at=when,
            reason=reason, metadata=metadata or {},
        )
        state = self.store.append_disposition(signer.sign(event, created_at=when), expected_head_digest=self.head)
        self.head = state.head_event_digest
        self.state = target
        self.sequence += 1


def import_book(
    store: EvidenceStore, head: PackagedBook, *, job_envelope, capability_profile_envelope,
    dataset_manifest_envelope, strategy_manifest_envelope, partitions: Mapping[str, bytes],
    trust_store: TrustStore, importer: Ed25519Signer, importer_id: str,
    verifier: Ed25519Signer, verifier_id: str, created_at: datetime,
) -> ImportOutcome:
    # Verify every supplied envelope and all replayable bytes before any durable
    # package/event write. Malformed or forged evidence cannot poison the store.
    job = verify_envelope(job_envelope, JobManifest, trust_store)
    capability = verify_envelope(capability_profile_envelope, CapabilityProfile, trust_store)
    dataset = verify_envelope(dataset_manifest_envelope, DatasetManifest, trust_store)
    strategy = verify_envelope(strategy_manifest_envelope, StrategyManifest, trust_store)
    report = verify_envelope(head.evaluation_report_envelope, EvaluationReport, trust_store)
    verify_envelope(head.package_envelope, type(head.package), trust_store)
    if not all(isinstance(value, expected) for value, expected in (
        (job, JobManifest), (capability, CapabilityProfile), (dataset, DatasetManifest),
        (strategy, StrategyManifest), (report, EvaluationReport),
    )):
        raise ValueError("signed envelope payload type mismatch")
    checks = _preflight(head, job=job, capability=capability, dataset=dataset, strategy=strategy,
                        partitions=partitions, report=report, store=store)
    package_digest = store.write_package(head.package_envelope, dict(head.files))
    verdict = build_import_verdict(
        verdict_id="portfolio-book-combined-book-verdict", package_digest=package_digest,
        importer_id=importer_id, created_at=created_at, checks=checks,
    )
    verdict_digest = store.write_verdict(importer.sign(verdict, created_at=created_at))
    chain = _Chain(store, package_digest, created_at)
    chain.created(head.created_event_envelope)
    chain.append(DispositionState.RECEIVED, importer, importer_id, AuthorityRole.LOCAL_IMPORTER, "received portfolio-book evidence")
    terminal = DispositionState.REJECTED
    reason = next((f"{check.check_id}: {check.details}" for check in checks if not check.passed), "verification failed")
    groups = (
        ({"artifact_hash_verified", "dataset_hashes_verified", "lineage_verified"}, DispositionState.HASH_VERIFIED, "artifact and lineage hashes verified"),
        ({"no_forbidden_capabilities", "metric_surface_complete", "source_package_derivation_bound", "source_lineage_bound_in_partitions", "source_packages_quarantined", "source_content_bound"}, DispositionState.POLICY_VERIFIED, "capability, metric and source-derivation policy verified"),
        ({"metric_replay_reproduces", "artifact_reproduces", "gate_verdict_reproduces"}, DispositionState.METRIC_REPLAYED, "independent replay reproduced book, metrics and gates"),
    )
    by_id = {check.check_id: check for check in checks}
    for ids, state, success_reason in groups:
        if not all(by_id[name].passed for name in ids):
            chain.append(DispositionState.REJECTED, importer, importer_id, AuthorityRole.LOCAL_IMPORTER, reason)
            break
        role = AuthorityRole.INDEPENDENT_VERIFIER if state == DispositionState.METRIC_REPLAYED else AuthorityRole.LOCAL_IMPORTER
        signer, actor = (verifier, verifier_id) if role == AuthorityRole.INDEPENDENT_VERIFIER else (importer, importer_id)
        chain.append(state, signer, actor, role, success_reason)
    else:
        if by_id["candidate_evaluation_gate_passed"].passed:
            terminal = DispositionState.QUARANTINED
            reason = "accepted for quarantine; no installation, capital-activation or promotion authority granted"
            chain.append(terminal, importer, importer_id, AuthorityRole.LOCAL_IMPORTER, reason, {"verdict_digest": verdict_digest})
        else:
            chain.append(DispositionState.REJECTED, importer, importer_id, AuthorityRole.LOCAL_IMPORTER, reason)
    if chain.state == DispositionState.REJECTED:
        terminal = DispositionState.REJECTED
    assert chain.head is not None
    return ImportOutcome(package_digest=package_digest, verdict=verdict, final_state=terminal,
                         reason=reason, disposition_head_digest=chain.head)


__all__ = ["import_book"]
