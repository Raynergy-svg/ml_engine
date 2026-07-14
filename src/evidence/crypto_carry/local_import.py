"""Local evidence authority for the crypto-carry evaluation slice.

The local importer receives a producer-signed carry package, writes it into the
immutable store, and drives the signed disposition chain:

    CREATED (producer)
    -> RECEIVED           (local importer)
    -> HASH_VERIFIED      (local importer)      : artifact + lineage hashes
    -> POLICY_VERIFIED    (local importer)      : capability + safety policy
    -> METRIC_REPLAYED    (independent verifier): reproduce metrics + verdict
    -> QUARANTINED        (local importer)       : accepted for quarantine
       or REJECTED        (local importer)       : any failed check / failed head

It re-runs the deterministic carry aggregation itself before accepting
anything, so the producing process is never the sole approver (roadmap §4.2). A
candidate never overwrites an incumbent: the chain stops at QUARANTINED; champion
promotion is a separate, operator-authorized transition this module never
performs. The verification path takes no registry/network dependency — an
external registry outage cannot stop a local verdict.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass, replace
from datetime import datetime, timedelta
from typing import Mapping

from src.evidence.contracts import (
    AuthorityRole,
    CapabilityProfile,
    DatasetManifest,
    DispositionEvent,
    DispositionState,
    EvaluationReport,
    ImportCheck,
    JobManifest,
    LocalImportVerdict,
    StrategyManifest,
)
from src.evidence.hashing import content_digest, sha256_bytes
from src.evidence.importer import build_import_verdict
from src.evidence.signing import Ed25519Signer, TrustStore, verify_envelope
from src.evidence.store import EvidenceStore

from .evaluation import EvaluationParams, evaluate_partitions
from .models import PackagedCarryHead, CarryImportOutcome

logger = logging.getLogger(__name__)

Evaluator = Callable[..., "tuple"]


@dataclass(frozen=True)
class _StageChecks:
    hash_checks: tuple[ImportCheck, ...]
    policy_checks: tuple[ImportCheck, ...]
    replay_checks: tuple[ImportCheck, ...]
    gate_check: ImportCheck

    def all(self) -> tuple[ImportCheck, ...]:
        return self.hash_checks + self.policy_checks + self.replay_checks + (self.gate_check,)


def _check(check_id: str, passed: bool, details: str) -> ImportCheck:
    return ImportCheck(check_id=check_id, passed=passed, details=details)


def _first_failure(checks: tuple[ImportCheck, ...], fallback: str) -> str:
    for check in checks:
        if not check.passed:
            return f"{check.check_id}: {check.details}"
    return fallback


def _declared_cells_by_carry(
    strategy_manifest: StrategyManifest,
) -> dict[str, tuple[str, ...]]:
    """The signed expected-cell-set-per-carry declared in the strategy manifest."""
    raw = strategy_manifest.parameters.get("expected_cells_by_carry") or {}
    if not isinstance(raw, dict):
        return {}
    out: dict[str, tuple[str, ...]] = {}
    for carry, cells in raw.items():
        if isinstance(cells, (list, tuple)):
            out[str(carry)] = tuple(sorted(str(f) for f in cells))
    return out


def _declared_bars(strategy_manifest: StrategyManifest) -> dict[str, float]:
    """The signed admissibility bars declared in the strategy manifest.

    Returned as a partial kwargs dict for :func:`dataclasses.replace` over the
    replay params, so the independent verifier reproduces the producer's verdict
    against the SAME signed contract rather than against local defaults."""
    params = strategy_manifest.parameters
    out: dict[str, float] = {}
    min_cells = params.get("min_cells")
    if isinstance(min_cells, int) and not isinstance(min_cells, bool):
        out["min_cells"] = min_cells
    for src_key, dst_key in (
        ("net_sharpe_floor", "net_sharpe_floor"),
        ("max_drawdown_limit", "max_drawdown_limit"),
        ("max_margin_utilization", "max_margin_utilization"),
        ("max_tracking_error", "max_tracking_error"),
        ("min_capacity_usd", "min_capacity_usd"),
    ):
        val = params.get(src_key)
        if isinstance(val, (int, float)) and not isinstance(val, bool):
            out[dst_key] = float(val)
    return out


def _dataset_cells_by_carry(
    dataset_manifest: DatasetManifest,
) -> dict[str, tuple[str, ...]]:
    """The cell-set-per-carry implied by the signed dataset manifest partitions."""
    grouped: dict[str, set[str]] = defaultdict(set)
    for ref in dataset_manifest.partitions:
        carry = ref.partition_id.split("::", 1)[0]
        grouped[carry].add(ref.partition_id)
    return {carry: tuple(sorted(ids)) for carry, ids in grouped.items()}


def _partitions_match_manifest(
    dataset_manifest: DatasetManifest, partitions: Mapping[str, bytes]
) -> bool:
    declared = {ref.partition_id: ref for ref in dataset_manifest.partitions}
    if set(partitions) != set(declared):
        return False
    for partition_id, data in partitions.items():
        ref = declared[partition_id]
        if len(data) != ref.size_bytes or sha256_bytes(data) != ref.digest:
            return False
    return True


def _hash_checks(
    head: PackagedCarryHead,
    job: JobManifest,
    report: EvaluationReport,
    dataset_manifest: DatasetManifest,
    strategy_manifest: StrategyManifest,
    partitions: Mapping[str, bytes],
) -> tuple[ImportCheck, ...]:
    package = head.package
    artifact = package.artifacts[0]
    artifact_ok = (
        sha256_bytes(head.files[artifact.relative_path]) == artifact.digest
        and head.package_envelope.payload_digest == content_digest(package)
    )
    # Bind the exact evaluation report to the signed package (mirrors risk-target
    # code review F1 / hedge slice): verify_envelope authenticates the report
    # signature, but this closes the gap where a differently-but-validly-signed
    # report with swapped audit fields could slip past.
    report_bound = (
        head.evaluation_report_envelope.payload_digest in package.evaluation_report_digests
    )
    lineage_ok = (
        tuple(package.dataset_manifest_digests) == tuple(job.dataset_manifest_digests)
        and report.job_manifest_digest == package.job_manifest_digest
        and content_digest(job) == package.job_manifest_digest
        and content_digest(dataset_manifest) in package.dataset_manifest_digests
    )
    strategy_ok = (
        package.strategy_manifest_digest == job.strategy_manifest_digest
        and package.strategy_manifest_digest == content_digest(strategy_manifest)
    )
    partitions_ok = _partitions_match_manifest(dataset_manifest, partitions)
    # The signed strategy manifest's declared expected fold set must match the
    # signed dataset manifest's partitions per carry — otherwise a producer could
    # sign a strategy manifest declaring cells the dataset never carried (the
    # "missing fold" guarantee the aggregator enforces would then be silently
    # weaker than the signed declaration claims).
    declared_cells = _declared_cells_by_carry(strategy_manifest)
    strategy_cells_ok = (
        bool(declared_cells)
        and declared_cells == _dataset_cells_by_carry(dataset_manifest)
    )
    return (
        _check("artifact_hashes_match", artifact_ok, "artifact bytes match declared digest"),
        _check("evaluation_report_bound_to_package", report_bound,
               "evaluation report digest is bound in the package"),
        _check("dataset_lineage_matches_job", lineage_ok, "package binds the signed job's dataset lineage"),
        _check("strategy_manifest_bound_to_job", strategy_ok,
               "package binds the signed job's strategy manifest"),
        _check("dataset_partitions_match_manifest", partitions_ok,
               "supplied cell partitions match the signed dataset manifest hashes"),
        _check("strategy_declared_cells_match_dataset", strategy_cells_ok,
               "strategy manifest's declared cell set per carry matches the dataset manifest"),
    )


def _policy_checks(
    head: PackagedCarryHead,
    job: JobManifest,
    capability_profile: CapabilityProfile,
    report: EvaluationReport,
) -> tuple[ImportCheck, ...]:
    package = head.package
    cap_ok = (
        content_digest(capability_profile) == job.capability_profile_digest
        and capability_profile.may_read_broker_credentials is False
        and capability_profile.may_place_or_cancel_orders is False
        and capability_profile.may_read_operator_keys is False
        and capability_profile.may_change_halts is False
        and capability_profile.may_change_live_gate is False
        and capability_profile.may_write_champion_pointer is False
        and capability_profile.may_modify_local_models is False
        and capability_profile.may_approve_evidence is False
        and capability_profile.network_endpoints == ()
    )
    safety_ok = bool(package.safety_assertions) and all(a.passed for a in package.safety_assertions)
    completeness_ok = (
        len(report.gates) >= 1 and report.trial_count >= 1 and report.effective_sample_size > 0
    )
    return (
        _check("no_forbidden_capabilities", cap_ok, "worker capability profile grants no authority"),
        _check("safety_assertions_present_and_passed", safety_ok, "all package safety assertions passed"),
        _check("evaluation_completeness", completeness_ok, "report has gates, a trial count and a holdout"),
    )


def _replay_checks(
    head: PackagedCarryHead,
    report: EvaluationReport,
    partitions: Mapping[str, bytes],
    params: EvaluationParams,
    evaluator: Evaluator,
    campaign_id: str,
) -> tuple[ImportCheck, ...]:
    """Independently reproduce the producer's carry aggregation, metrics and verdict."""
    try:
        reproduced = {r.lane_id: r for r in evaluator(partitions, params, campaign_id=campaign_id)}
    except Exception as exc:  # noqa: BLE001 - a replay failure is a verdict, not a crash
        return (_check("metric_replay_reproduces", False, f"replay error: {exc}"),)

    result = reproduced.get(head.lane_id)
    if result is None:
        return (_check("metric_replay_reproduces", False, f"replay produced no {head.lane_id} head"),)

    mismatches: list[str] = []
    for name, metric in report.metrics.items():
        if name not in result.metrics:
            mismatches.append(f"{name}:absent")
            continue
        # Ignore the producer-declared tolerance on the import side — a local
        # authority must not let the producer widen its own reproduction bar.
        if abs(float(result.metrics[name]) - metric.value) > params.replay_tolerance:
            mismatches.append(f"{name}:{result.metrics[name]}!={metric.value}")
    metric_ok = not mismatches

    reproduced_gate = {g.gate_id: g.status for g in result.gates}
    gate_ok = bool(result.passed) == bool(report.passed) and all(
        reproduced_gate.get(g.gate_id) == g.status for g in report.gates
    )

    # The artifact IS the deterministic Canonical-JSON aggregated scorecard we
    # just reproduced, so bind the reproduced bytes to the stored, signed artifact
    # digest. Catches a producer that signs an honest EvaluationReport but
    # packages a doctored carry_scorecard.json.
    artifact = head.package.artifacts[0]
    artifact_ok = sha256_bytes(result.artifact_bytes) == artifact.digest

    return (
        _check("metric_replay_reproduces", metric_ok,
               "metrics reproduce within declared tolerance" if metric_ok else "; ".join(mismatches)),
        _check("gate_verdict_reproduces", gate_ok, "reproduced gate verdict matches the producer's"),
        _check("artifact_reproduces", artifact_ok,
               "reproduced scorecard bytes match the stored artifact digest"),
    )


class _Chain:
    """Drives one package's disposition chain, tracking head + current state."""

    def __init__(self, store: EvidenceStore, package_digest: str, base_time: datetime) -> None:
        self._store = store
        self._package_digest = package_digest
        self._base_time = base_time
        self.head_digest: str | None = None
        self.current_state: DispositionState | None = None
        self._sequence = 0

    def append(
        self,
        *,
        from_state: DispositionState | None,
        to_state: DispositionState,
        signer: Ed25519Signer,
        actor_id: str,
        role: AuthorityRole,
        reason: str,
        metadata: dict | None = None,
    ) -> None:
        occurred_at = self._base_time + timedelta(seconds=self._sequence)
        event = DispositionEvent(
            package_digest=self._package_digest,
            sequence=self._sequence,
            previous_event_digest=self.head_digest,
            from_state=from_state,
            to_state=to_state,
            authority=role,
            actor_id=actor_id,
            signer_key_id=signer.key_id,
            occurred_at=occurred_at,
            reason=reason,
            metadata=metadata or {},
        )
        state = self._store.append_disposition(
            signer.sign(event, created_at=occurred_at), expected_head_digest=self.head_digest
        )
        self.head_digest = state.head_event_digest
        self.current_state = to_state
        self._sequence += 1

    def append_created_from_producer(self, envelope) -> None:
        state = self._store.append_disposition(envelope, expected_head_digest=None)
        self.head_digest = state.head_event_digest
        self.current_state = DispositionState.CREATED
        self._sequence = 1


def import_head(
    store: EvidenceStore,
    head: PackagedCarryHead,
    *,
    campaign_id: str,
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
    params: EvaluationParams | None = None,
    evaluator: Evaluator = evaluate_partitions,
) -> CarryImportOutcome:
    """Import one carry head package to QUARANTINED or REJECTED.

    Writes the immutable package, runs staged verification (hash, policy,
    independent carry aggregation replay), builds the signed
    ``LocalImportVerdict``, and drives the disposition chain to its terminal
    state. Raises only if the package is structurally inadmissible (tampered
    bytes / bad signature) — a package that is admissible but fails a check is
    REJECTED, not raised.

    The capability profile, dataset manifest and strategy manifest are
    reconstructed from their signed envelopes here (not trusted as
    caller-supplied objects), so the local authority verifies the producer's
    exact declared capabilities, data lineage and workload spec.
    """
    params = params or EvaluationParams()
    package_digest = store.write_package(head.package_envelope, dict(head.files))

    report = verify_envelope(head.evaluation_report_envelope, EvaluationReport, trust_store)
    if not isinstance(report, EvaluationReport):
        raise ValueError("report envelope did not decode to EvaluationReport")
    capability_profile = verify_envelope(capability_profile_envelope, CapabilityProfile, trust_store)
    if not isinstance(capability_profile, CapabilityProfile):
        raise ValueError("capability envelope did not decode to CapabilityProfile")
    dataset_manifest = verify_envelope(dataset_manifest_envelope, DatasetManifest, trust_store)
    if not isinstance(dataset_manifest, DatasetManifest):
        raise ValueError("dataset envelope did not decode to DatasetManifest")
    strategy_manifest = verify_envelope(strategy_manifest_envelope, StrategyManifest, trust_store)
    if not isinstance(strategy_manifest, StrategyManifest):
        raise ValueError("strategy envelope did not decode to StrategyManifest")

    # Reproduce against the SIGNED strategy manifest's declared cell set (not a
    # set derived from the supplied partitions), so an omitted cell the signed
    # declaration required makes the replay fail rather than silently pass.
    declared_cells = _declared_cells_by_carry(strategy_manifest)
    replay_overrides = _declared_bars(strategy_manifest)
    if declared_cells:
        replay_overrides["expected_cells_by_carry"] = declared_cells
    replay_params = replace(params, **replay_overrides) if replay_overrides else params

    checks = _StageChecks(
        hash_checks=_hash_checks(head, job, report, dataset_manifest, strategy_manifest, partitions),
        policy_checks=_policy_checks(head, job, capability_profile, report),
        replay_checks=_replay_checks(head, report, partitions, replay_params, evaluator, campaign_id),
        gate_check=_check(
            "candidate_evaluation_gate_passed",
            bool(report.passed),
            "carry cleared independent return, tail and counterparty gates",
        ),
    )

    verdict = build_import_verdict(
        verdict_id=f"crypto-carry-{head.campaign_id}-{head.carry_id}-verdict",
        package_digest=package_digest,
        importer_id=importer_id,
        created_at=created_at,
        checks=checks.all(),
    )
    verdict_digest = store.write_verdict(importer.sign(verdict, created_at=created_at))

    chain = _Chain(store, package_digest, created_at)
    chain.append_created_from_producer(head.created_event_envelope)
    chain.append(
        from_state=DispositionState.CREATED, to_state=DispositionState.RECEIVED,
        signer=importer, actor_id=importer_id, role=AuthorityRole.LOCAL_IMPORTER,
        reason="received remote crypto-carry evidence",
    )

    def reject(reason: str) -> CarryImportOutcome:
        chain.append(
            from_state=chain.current_state, to_state=DispositionState.REJECTED,
            signer=importer, actor_id=importer_id, role=AuthorityRole.LOCAL_IMPORTER,
            reason=reason,
        )
        return _outcome(head, package_digest, verdict, DispositionState.REJECTED, reason, chain)

    if not all(c.passed for c in checks.hash_checks):
        return reject(_first_failure(checks.hash_checks, "hash verification failed"))
    chain.append(
        from_state=DispositionState.RECEIVED, to_state=DispositionState.HASH_VERIFIED,
        signer=importer, actor_id=importer_id, role=AuthorityRole.LOCAL_IMPORTER,
        reason="artifact and lineage hashes verified",
    )

    if not all(c.passed for c in checks.policy_checks):
        return reject(_first_failure(checks.policy_checks, "policy verification failed"))
    chain.append(
        from_state=DispositionState.HASH_VERIFIED, to_state=DispositionState.POLICY_VERIFIED,
        signer=importer, actor_id=importer_id, role=AuthorityRole.LOCAL_IMPORTER,
        reason="capability and safety policy verified",
    )

    if not all(c.passed for c in checks.replay_checks):
        return reject(_first_failure(checks.replay_checks, "metric replay failed"))
    chain.append(
        from_state=DispositionState.POLICY_VERIFIED, to_state=DispositionState.METRIC_REPLAYED,
        signer=verifier, actor_id=verifier_id, role=AuthorityRole.INDEPENDENT_VERIFIER,
        reason="independent carry aggregation replay reproduced the producer's numbers",
    )

    if not checks.gate_check.passed:
        return reject("candidate carry failed return, tail or counterparty admissibility")

    chain.append(
        from_state=DispositionState.METRIC_REPLAYED, to_state=DispositionState.QUARANTINED,
        signer=importer, actor_id=importer_id, role=AuthorityRole.LOCAL_IMPORTER,
        reason="accepted for quarantine", metadata={"verdict_digest": verdict_digest},
    )
    return _outcome(
        head, package_digest, verdict, DispositionState.QUARANTINED,
        "accepted for quarantine", chain,
    )


def _outcome(
    head: PackagedCarryHead,
    package_digest: str,
    verdict: LocalImportVerdict,
    final_state: DispositionState,
    reason: str,
    chain: _Chain,
) -> CarryImportOutcome:
    if chain.head_digest is None:
        raise RuntimeError("disposition chain ended without a durable head digest")
    return CarryImportOutcome(
        lane_id=head.lane_id,
        campaign_id=head.campaign_id,
        carry_id=head.carry_id,
        package_digest=package_digest,
        verdict=verdict,
        final_state=final_state,
        reason=reason,
        disposition_head_digest=chain.head_digest,
    )


__all__ = ["import_head", "CarryImportOutcome"]
