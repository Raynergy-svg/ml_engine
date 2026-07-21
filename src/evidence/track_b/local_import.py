"""Local authority and independent replay for Track B scoring/evaluation packages."""

from __future__ import annotations

import json
from datetime import datetime, timedelta
from typing import Mapping

from src.evidence.contracts import (
    AuthorityRole, CapabilityProfile, DatasetManifest, DispositionEvent,
    DispositionState, EvaluationReport, EvidencePackage, GateStatus, ImportCheck,
    JobManifest, StrategyManifest,
)
from src.evidence.hashing import content_digest, sha256_bytes
from src.evidence.importer import build_import_verdict
from src.evidence.signing import Ed25519Signer, TrustStore, verify_envelope
from src.evidence.store import EvidenceStore

from .evaluation import EvaluationParams, evaluate_partitions
from .lineage import score_batch, validate_score_lineage
from .models import PackagedTrackBHead, TrackBImportOutcome


def _check(check_id: str, passed: bool, details: str) -> ImportCheck:
    return ImportCheck(check_id=check_id, passed=passed, details=details)


def _partition_hashes_match(dataset: DatasetManifest, partitions: Mapping[str, bytes]) -> bool:
    declared = {ref.partition_id: ref for ref in dataset.partitions}
    return set(partitions) == set(declared) and all(
        len(data) == declared[key].size_bytes and sha256_bytes(data) == declared[key].digest
        for key, data in partitions.items()
    )


def _main_artifact(head: PackagedTrackBHead) -> bytes:
    suffix = f"artifacts/{head.stage}_artifact.jsonl"
    if suffix not in head.files:
        raise ValueError(f"package missing {suffix}")
    return head.files[suffix]


def _evaluation_derivation_checks(
    store: EvidenceStore,
    head: PackagedTrackBHead,
    partitions: Mapping[str, bytes],
) -> tuple[ImportCheck, ...]:
    if head.stage != "evaluation":
        return ()
    try:
        index = json.loads(store.current_index_bytes())
        packages = [store.load_package(digest) for digest in head.package.derived_from_package_digests]
    except Exception as exc:  # noqa: BLE001
        return (_check("scoring_packages_quarantined", False, f"cannot load derivation: {exc}"),)
    states_ok = all(
        index.get("packages", {}).get(digest, {}).get("state") == "QUARANTINED"
        for digest in head.package.derived_from_package_digests
    )
    artifact_digests = {
        artifact.digest
        for package in packages
        for artifact in package.artifacts
        if artifact.relative_path == "artifacts/scoring_artifact.jsonl"
    }
    consumed_digests = {cell_id.split("::", 1)[0] for cell_id in partitions}
    artifacts_bound = bool(consumed_digests) and consumed_digests <= artifact_digests
    return (
        _check(
            "scoring_packages_quarantined", states_ok,
            "all derived scoring packages are locally QUARANTINED",
        ),
        _check(
            "score_artifacts_bound_to_packages", artifacts_bound,
            "every consumed score-artifact digest belongs to a derived scoring package",
        ),
    )


def _replay_scoring(
    head: PackagedTrackBHead,
    partitions: Mapping[str, bytes],
    strategy: StrategyManifest,
    report: EvaluationReport,
) -> tuple[ImportCheck, ...]:
    version = strategy.parameters.get("scorer_version")
    matched = [batch for batch in sorted(partitions) if head.unit_id == f"{batch}-{version}"]
    if len(matched) != 1:
        return (_check("scoring_replay", False, "head does not resolve to exactly one signed batch"),)
    batch = matched[0]
    try:
        artifact, queue, metrics = score_batch(batch, partitions[batch], batch_id=batch)
        for line in artifact.decode("utf-8").splitlines():
            validate_score_lineage(json.loads(line))
        metric_ok = set(report.metrics) == set(metrics) and all(
            abs(metrics[name] - report.metrics[name].value) <= 1e-12
            for name in metrics
        )
        artifact_ok = artifact == _main_artifact(head)
        queue_ok = queue == head.files.get("artifacts/manual_review_queue.jsonl")
        expected_status = {
            "full_score_lineage_valid": GateStatus.PASS,
            "manual_review_queue_empty": (
                GateStatus.PASS if metrics["n_manual_review"] == 0.0 else GateStatus.FAIL
            ),
        }
        report_status = {gate.gate_id: gate.status for gate in report.gates}
        gate_ok = (
            report_status == expected_status
            and report.passed == (metrics["n_manual_review"] == 0.0)
        )
    except Exception as exc:  # noqa: BLE001
        return (_check("scoring_replay", False, f"replay error: {exc}"),)
    return (
        _check("metric_replay_reproduces", metric_ok, "scoring metrics reproduce"),
        _check("artifact_reproduces", artifact_ok, "score artifact reproduces byte-for-byte"),
        _check("manual_review_queue_reproduces", queue_ok, "manual-review queue reproduces"),
        _check("gate_verdict_reproduces", gate_ok, "scoring gate reproduces"),
    )


def _replay_evaluation(
    head: PackagedTrackBHead,
    partitions: Mapping[str, bytes],
    strategy: StrategyManifest,
    report: EvaluationReport,
) -> tuple[ImportCheck, ...]:
    parameters = strategy.parameters
    try:
        params = EvaluationParams(
            expected_cells=tuple(str(value) for value in parameters["expected_cells"]),
            min_forward_cells=int(parameters["min_forward_cells"]),
            rank_ic_bar=float(parameters["rank_ic_bar"]),
            placebo_abs_limit=float(parameters["placebo_abs_limit"]),
        )
        result = evaluate_partitions(partitions, params)
        expected_metrics = {
            key: float(value)
            for key, value in result.items()
            if key in {"n_cells", "n_valid_forward_cells", "mean_rank_ic", "mean_placebo_rank_ic"}
            and value is not None
        }
        metric_ok = set(report.metrics) == set(expected_metrics) and all(
            abs(expected_metrics[name] - report.metrics[name].value) <= 1e-12
            for name in expected_metrics
        )
        artifact_ok = result["artifact_bytes"] == _main_artifact(head)
        expected_status = {
            "adequate_forward_time": (
                GateStatus.PASS if result["enough_forward_time"] else GateStatus.FAIL
            ),
            "rank_ic_clears": (
                GateStatus.PASS if result["signal_gate_passed"] else GateStatus.FAIL
            ),
            "placebo_clean": (
                GateStatus.PASS if result["placebo_gate_passed"] else GateStatus.FAIL
            ),
        }
        gate_ok = (
            {gate.gate_id: gate.status for gate in report.gates} == expected_status
            and bool(result["passed"]) == bool(report.passed)
        )
        derived_expected = tuple(
            sorted(str(value) for value in parameters["scoring_package_digests"])
        )
        derived_ok = tuple(sorted(head.package.derived_from_package_digests)) == derived_expected
    except Exception as exc:  # noqa: BLE001
        return (_check("evaluation_replay", False, f"replay error: {exc}"),)
    return (
        _check("metric_replay_reproduces", metric_ok, "forward metrics reproduce"),
        _check("artifact_reproduces", artifact_ok, "evaluation artifact reproduces"),
        _check("gate_verdict_reproduces", gate_ok, "forward gate reproduces"),
        _check("scoring_package_derivation_bound", derived_ok, "scoring package derivation matches signed strategy"),
    )


class _Chain:
    def __init__(self, store: EvidenceStore, digest: str, now: datetime) -> None:
        self.store = store
        self.digest = digest
        self.now = now
        self.head: str | None = None
        self.state: DispositionState | None = None
        self.sequence = 0

    def created(self, envelope) -> None:
        state = self.store.append_disposition(envelope, expected_head_digest=None)
        self.head = state.head_event_digest
        self.state = DispositionState.CREATED
        self.sequence = 1

    def append(
        self,
        to_state: DispositionState,
        *,
        signer: Ed25519Signer,
        actor: str,
        role: AuthorityRole,
        reason: str,
        metadata: dict | None = None,
    ) -> None:
        occurred_at = self.now + timedelta(seconds=self.sequence)
        event = DispositionEvent(
            package_digest=self.digest,
            sequence=self.sequence,
            previous_event_digest=self.head,
            from_state=self.state,
            to_state=to_state,
            authority=role,
            actor_id=actor,
            signer_key_id=signer.key_id,
            occurred_at=occurred_at,
            reason=reason,
            metadata=metadata or {},
        )
        state = self.store.append_disposition(
            signer.sign(event, created_at=occurred_at), expected_head_digest=self.head
        )
        self.head = state.head_event_digest
        self.state = to_state
        self.sequence += 1


def import_head(
    store: EvidenceStore,
    head: PackagedTrackBHead,
    *,
    job: JobManifest,
    capability_profile_envelope,
    dataset_manifest_envelope,
    strategy_manifest_envelope,
    partitions: Mapping[str, bytes],
    trust_store: TrustStore,
    importer: Ed25519Signer,
    importer_id: str,
    verifier: Ed25519Signer,
    verifier_id: str,
    created_at: datetime,
) -> TrackBImportOutcome:
    package = verify_envelope(head.package_envelope, EvidencePackage, trust_store)
    report = verify_envelope(head.evaluation_report_envelope, EvaluationReport, trust_store)
    capability = verify_envelope(capability_profile_envelope, CapabilityProfile, trust_store)
    dataset = verify_envelope(dataset_manifest_envelope, DatasetManifest, trust_store)
    strategy = verify_envelope(strategy_manifest_envelope, StrategyManifest, trust_store)
    created_event = verify_envelope(
        head.created_event_envelope, DispositionEvent, trust_store
    )
    if not all((isinstance(package, EvidencePackage), isinstance(report, EvaluationReport),
                isinstance(capability, CapabilityProfile), isinstance(dataset, DatasetManifest),
                isinstance(strategy, StrategyManifest),
                isinstance(created_event, DispositionEvent))):
        raise ValueError("Track B envelope type mismatch")
    if package != head.package or head.package_envelope.payload_digest != content_digest(package):
        raise ValueError("Track B packaged head does not match its signed package envelope")
    if set(head.files) != set(package.checksums) or not all(
        sha256_bytes(head.files[path]) == digest for path, digest in package.checksums.items()
    ):
        raise ValueError("Track B package files do not match signed checksums")
    if (
        created_event.package_digest != head.package_envelope.payload_digest
        or created_event.sequence != 0
        or created_event.to_state != DispositionState.CREATED
    ):
        raise ValueError("Track B CREATED event does not match the signed package")
    # Persist only after every supplied envelope and byte set has passed minimal
    # structural verification. A bad companion envelope must not leave an
    # un-dispositioned durable package behind.
    package_digest = store.write_package(head.package_envelope, dict(head.files))
    artifact_hashes = all(
        path in head.files and sha256_bytes(head.files[path]) == digest
        for path, digest in head.package.checksums.items()
    )
    report_bound = report.job_manifest_digest == head.package.job_manifest_digest and (
        head.evaluation_report_envelope.payload_digest in head.package.evaluation_report_digests
    )
    lineage = (
        content_digest(job) == head.package.job_manifest_digest
        and tuple(job.dataset_manifest_digests) == tuple(head.package.dataset_manifest_digests)
        and content_digest(dataset) in job.dataset_manifest_digests
        and job.strategy_manifest_digest == content_digest(strategy)
        and content_digest(strategy) == head.package.strategy_manifest_digest
    )
    capability_ok = (
        content_digest(capability) == job.capability_profile_digest
        and capability.network_endpoints == ()
        and capability.may_read_broker_credentials is False
        and capability.may_place_or_cancel_orders is False
        and capability.may_read_operator_keys is False
        and capability.may_change_halts is False
        and capability.may_change_live_gate is False
        and capability.may_write_champion_pointer is False
        and capability.may_modify_local_models is False
        and capability.may_approve_evidence is False
    )
    hash_checks = (
        _check("artifact_hashes_match", artifact_hashes, "all package artifacts match checksums"),
        _check("evaluation_report_bound", report_bound, "report is bound to package/job"),
        _check("manifest_lineage_bound", lineage, "package binds job, dataset and strategy"),
        _check("dataset_partitions_match", _partition_hashes_match(dataset, partitions), "input bytes match signed dataset"),
    )
    policy_checks = (
        _check("no_forbidden_capabilities", capability_ok, "worker has no execution or approval authority"),
    ) + _evaluation_derivation_checks(store, head, partitions)
    replay_checks = (
        _replay_scoring(head, partitions, strategy, report)
        if head.stage == "scoring"
        else _replay_evaluation(head, partitions, strategy, report)
    )
    gate_check = _check("candidate_evaluation_gate_passed", bool(report.passed), "stage gate passed")
    checks = hash_checks + policy_checks + replay_checks + (gate_check,)
    verdict = build_import_verdict(
        verdict_id=f"track-b-{head.stage}-{head.campaign_id}-{head.unit_id}-verdict",
        package_digest=package_digest,
        importer_id=importer_id,
        created_at=created_at,
        checks=checks,
    )
    verdict_digest = store.write_verdict(importer.sign(verdict, created_at=created_at))
    chain = _Chain(store, package_digest, created_at)
    chain.created(head.created_event_envelope)
    chain.append(
        DispositionState.RECEIVED, signer=importer, actor=importer_id,
        role=AuthorityRole.LOCAL_IMPORTER, reason="received Track B evidence",
    )
    failed_hash = next((check for check in hash_checks if not check.passed), None)
    failed_policy = next((check for check in policy_checks if not check.passed), None)
    failed_replay = next((check for check in replay_checks if not check.passed), None)
    failure = failed_hash or failed_policy or failed_replay
    if failed_hash is None:
        chain.append(
            DispositionState.HASH_VERIFIED, signer=importer, actor=importer_id,
            role=AuthorityRole.LOCAL_IMPORTER, reason="hash and lineage checks passed",
        )
    if failure is None:
        chain.append(
            DispositionState.POLICY_VERIFIED, signer=importer, actor=importer_id,
            role=AuthorityRole.LOCAL_IMPORTER, reason="capability policy passed",
        )
        chain.append(
            DispositionState.METRIC_REPLAYED, signer=verifier, actor=verifier_id,
            role=AuthorityRole.INDEPENDENT_VERIFIER, reason="independent Track B replay reproduced",
        )
    reason = f"{failure.check_id}: {failure.details}" if failure else "stage evaluation gate failed"
    if failure is not None or not gate_check.passed:
        chain.append(
            DispositionState.REJECTED, signer=importer, actor=importer_id,
            role=AuthorityRole.LOCAL_IMPORTER, reason=reason,
        )
        final = DispositionState.REJECTED
    else:
        chain.append(
            DispositionState.QUARANTINED, signer=importer, actor=importer_id,
            role=AuthorityRole.LOCAL_IMPORTER, reason="accepted for quarantine",
            metadata={"verdict_digest": verdict_digest},
        )
        final = DispositionState.QUARANTINED
        reason = "accepted for quarantine"
    if chain.head is None:
        raise RuntimeError("Track B disposition chain has no durable head")
    return TrackBImportOutcome(
        lane_id=head.lane_id,
        package_digest=package_digest,
        verdict=verdict,
        final_state=final,
        reason=reason,
        disposition_head_digest=chain.head,
    )


__all__ = ["import_head"]
