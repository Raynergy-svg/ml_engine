"""Phase L local promotion service — QUARANTINED evidence to approved champion.

Composes the existing :class:`EvidenceStore` primitives into the roadmap §15
flow (steps 1–11) without weakening any rail:

1.  Resolve the package's current disposition head from its signed ledger.
2.  Verify package + artifact digests from disk (the store re-hashes every
    member on read; this module re-hashes the model artifact again itself).
3.  REQUIRE explicit operator approval (:class:`OperatorApproval`) — no
    approval, no promotion, hard error. There is no auto-approve path.
4.  Confirm lane and feature-pipeline compatibility: the package's signed
    ``JobManifest.feature_pipeline_version`` must equal what the consumer
    expects, and the signed ``EvaluationReport`` must bind the same job.
5.  Copy the candidate artifact into a versioned local artifact store
    (``<root>/champion_artifacts/<lane>/<package_digest>/...``).
6.  Load smoke test — actually load the model artifact bytes.
7.  Dry inference — run one real predict and assert finite output.
8.  Atomically write the champion pointer via
    :meth:`EvidenceStore.write_champion_pointer` with exact hashes.
    (The store requires the ledger to already be at CHAMPION, so step 9's
    disposition events are appended *before* the pointer swap.)
9.  Append the promotion disposition events THROUGH the transition policy:
    QUARANTINED → SHADOW → OPERATOR_APPROVED (operator authority) →
    CHAMPION (promotion-service authority). The policy is never bypassed or
    relaxed; an incumbent champion in the lane is first retired
    (CHAMPION → RETIRED, operator authority) because the store refuses two
    concurrent champions per lane.
10. Append a JSONL promotion event (``<root>/promotions/promotions.jsonl``).
11. Preserve the prior champion pointer envelope + artifact copy in the
    promotion record for rollback.

SHIP GATE (CLAUDE.md Hard NO #3): before any ledger mutation this service
reads EVERY pre-registered signed ``EvaluationReport`` the package declares
and REFUSES to promote any head with a FAIL gate result, a ``passed=False``
verdict, or missing/unreadable gate evidence (a declared-but-absent report
digest refuses fail-closed). A passing report never rescues a failing one —
neither across packages (each head is its own package) nor within one.

READ-ONLY PREFLIGHT: module-level :func:`preflight` runs the identical
verification pipeline (:func:`_verify_candidate`) with zero writes — no
ledger append, no pointer, no artifact copy, no log — for dry-run/status use.

ROLLBACK: :meth:`PromotionService.rollback` restores the preserved prior
pointer atomically when the transition policy permits it (the prior package is
still CHAMPION — e.g. an artifact re-point within one package). When the prior
package was retired during promotion, the policy has no RETIRED → CHAMPION
edge, so exact pointer restoration is structurally forbidden; rollback then
fails CLOSED by retiring the promoted champion so the lane serves *no*
champion (consumers abstain) and records the structural refusal. The policy is
reported, never edited.
"""

from __future__ import annotations

import fcntl
import json
import math
import os
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path, PurePosixPath
from typing import Any
from uuid import uuid4

from pydantic import ValidationError

from .contracts import (
    ArtifactRef,
    AuthorityRole,
    ChampionPointer,
    DispositionEvent,
    DispositionState,
    EvaluationReport,
    EvidencePackage,
    GateStatus,
    JobManifest,
    SignedEnvelope,
)
from .hashing import sha256_bytes
from .signing import Ed25519Signer, SignatureVerificationError, TrustStore, verify_envelope
from .store import (
    ChampionPointerError,
    EvidenceStore,
    EvidenceStoreError,
    StoreCorruptionError,
)
from .transition_policy import AuthorityRegistry

PROMOTION_LOG_SCHEMA_VERSION = "1.0.0"

_PROMOTABLE_STATES = frozenset(
    {
        DispositionState.QUARANTINED,
        DispositionState.SHADOW,
        DispositionState.OPERATOR_APPROVED,
        DispositionState.CHAMPION,
    }
)


class PromotionError(RuntimeError):
    """Base error for local promotion-service operations."""


class PromotionRefusedError(PromotionError):
    """A promotion or rollback was refused; nothing was promoted."""

    def __init__(self, reason_code: str, detail: str) -> None:
        self.reason_code = reason_code
        self.detail = detail
        super().__init__(f"{reason_code}: {detail}")


@dataclass(frozen=True)
class OperatorApproval:
    """Explicit, recorded operator approval. Required for every promotion.

    ``reason`` is free text and must be non-empty — it is written into every
    disposition event and the promotion log. Constructing this object is a
    deliberate operator act; the service never fabricates one.
    """

    reason: str


@dataclass(frozen=True)
class InferenceProbe:
    """Real load + dry-inference callables for the candidate artifact.

    ``load`` receives the digest-verified artifact bytes and must return a
    usable model object (step 6). ``predict`` receives that object and must
    return numeric output for one held sample (step 7); the service asserts
    the output is non-empty and finite. Both are real calls — never stubs.
    """

    load: Callable[[bytes], object]
    predict: Callable[[object], object]


@dataclass(frozen=True)
class LedgerView:
    """Verified, decoded view of one package's disposition ledger."""

    events: tuple[DispositionEvent, ...]
    head_digest: str
    state: DispositionState
    last_occurred_at: datetime
    next_sequence: int


# --------------------------------------------------------------------------- #
# Read-only verification pipeline (shared by promote() and preflight())        #
# --------------------------------------------------------------------------- #


def _resolve_ledger_from_store(store: EvidenceStore, package_digest: str) -> LedgerView | None:
    try:
        envelopes = store.load_ledger(package_digest)
    except (FileNotFoundError, StoreCorruptionError, EvidenceStoreError) as exc:
        raise PromotionRefusedError("ledger_verification_failed", str(exc)) from exc
    if not envelopes:
        return None
    events: list[DispositionEvent] = []
    for envelope in envelopes:
        try:
            payload = verify_envelope(envelope, DispositionEvent, store.trust_store)
        except (SignatureVerificationError, ValueError) as exc:
            raise PromotionRefusedError("ledger_verification_failed", str(exc)) from exc
        assert isinstance(payload, DispositionEvent)
        events.append(payload)
    last = events[-1]
    return LedgerView(
        events=tuple(events),
        head_digest=envelopes[-1].payload_digest,
        state=last.to_state,
        last_occurred_at=last.occurred_at,
        next_sequence=last.sequence + 1,
    )


def _select_artifact(package: EvidencePackage, artifact_id: str | None) -> ArtifactRef:
    artifacts = {ref.artifact_id: ref for ref in package.artifacts}
    if artifact_id is not None:
        ref = artifacts.get(artifact_id)
        if ref is None:
            raise PromotionRefusedError(
                "artifact_not_found",
                f"package declares no artifact {artifact_id!r}; has {sorted(artifacts)}",
            )
        return ref
    if len(package.artifacts) != 1:
        raise PromotionRefusedError(
            "artifact_ambiguous",
            f"package declares {len(package.artifacts)} artifacts; pass artifact_id explicitly",
        )
    return package.artifacts[0]


def _collect_signed_envelopes(
    store: EvidenceStore,
    package_digest: str,
    package: EvidencePackage,
    refusal_code: str,
) -> list[SignedEnvelope]:
    """Return every package member that parses as a SignedEnvelope."""
    envelopes: list[SignedEnvelope] = []
    for ref in package.logs + package.artifacts:
        try:
            data = store.load_package_file(package_digest, ref.relative_path)
        except (FileNotFoundError, StoreCorruptionError, EvidenceStoreError) as exc:
            raise PromotionRefusedError(refusal_code, str(exc)) from exc
        try:
            envelopes.append(SignedEnvelope.model_validate_json(data, strict=True))
        except ValidationError:
            continue
    return envelopes


def _load_evaluation_reports(
    store: EvidenceStore, package_digest: str, package: EvidencePackage
) -> tuple[EvaluationReport, ...]:
    """Load and verify EVERY pre-registered evaluation report — fail closed.

    Every digest in ``package.evaluation_report_digests`` must resolve to a
    verifiable signed ``EvaluationReport`` member. A declared-but-missing or
    unreadable report refuses; the ship gate is later enforced on ALL of them,
    so a passing report can never rescue a failing one.
    """
    declared = frozenset(package.evaluation_report_digests)
    found: dict[str, EvaluationReport] = {}
    for envelope in _collect_signed_envelopes(
        store, package_digest, package, "gate_results_missing"
    ):
        if envelope.payload_digest not in declared:
            continue
        if envelope.payload_type != EvaluationReport.__name__:
            continue
        try:
            payload = verify_envelope(envelope, EvaluationReport, store.trust_store)
        except (SignatureVerificationError, ValueError) as exc:
            raise PromotionRefusedError(
                "gate_results_missing",
                f"pre-registered EvaluationReport failed verification: {exc}",
            ) from exc
        assert isinstance(payload, EvaluationReport)
        found[envelope.payload_digest] = payload
    missing = declared - set(found)
    if missing:
        raise PromotionRefusedError(
            "gate_results_missing",
            f"package {package_digest} declares evaluation report(s) it does not carry "
            f"verifiably: {sorted(missing)} — refusing fail-closed",
        )
    return tuple(found[digest] for digest in sorted(found))


def _load_job_manifest(
    store: EvidenceStore, package_digest: str, package: EvidencePackage
) -> JobManifest:
    for envelope in _collect_signed_envelopes(
        store, package_digest, package, "job_manifest_missing"
    ):
        if envelope.payload_digest != package.job_manifest_digest:
            continue
        if envelope.payload_type != JobManifest.__name__:
            continue
        try:
            payload = verify_envelope(envelope, JobManifest, store.trust_store)
        except (SignatureVerificationError, ValueError) as exc:
            raise PromotionRefusedError(
                "job_manifest_missing", f"JobManifest failed verification: {exc}"
            ) from exc
        assert isinstance(payload, JobManifest)
        return payload
    raise PromotionRefusedError(
        "job_manifest_missing",
        f"package {package_digest} carries no verifiable signed JobManifest "
        "(feature-pipeline contract); refusing fail-closed",
    )


def _enforce_ship_gate(report: EvaluationReport) -> None:
    """Hard NO #3 — refuse any head whose pre-registered gate result is FAIL."""
    failed = [gate.gate_id for gate in report.gates if gate.status == GateStatus.FAIL]
    if failed:
        raise PromotionRefusedError(
            "ship_gate_failed",
            f"pre-registered gate(s) FAILED: {', '.join(sorted(failed))} — "
            "a failing head is never promoted (Hard NO #3)",
        )
    if not report.passed:
        raise PromotionRefusedError(
            "ship_gate_failed", "evaluation report verdict is passed=False"
        )
    if not any(gate.status == GateStatus.PASS for gate in report.gates):
        raise PromotionRefusedError(
            "ship_gate_failed",
            "no gate registered a PASS result; refusing fail-closed",
        )


def _finite_outputs(raw: object) -> list[float]:
    import numpy as np

    try:
        array = np.asarray(raw, dtype=float).ravel()
    except (TypeError, ValueError) as exc:
        raise PromotionRefusedError(
            "dry_inference_failed", f"dry inference output is not numeric: {exc}"
        ) from exc
    if array.size == 0:
        raise PromotionRefusedError("dry_inference_failed", "dry inference produced no output")
    values = [float(value) for value in array.tolist()]
    if not all(math.isfinite(value) for value in values):
        raise PromotionRefusedError(
            "dry_inference_failed", f"dry inference produced non-finite output: {values}"
        )
    return values


@dataclass(frozen=True)
class _CandidateChecks:
    """Everything the read-only verification pipeline established."""

    package: EvidencePackage
    artifact: ArtifactRef
    artifact_bytes: bytes
    view: LedgerView
    reports: tuple[EvaluationReport, ...]
    job: JobManifest
    outputs: tuple[float, ...]


def _verify_candidate(
    store: EvidenceStore,
    package_digest: str,
    *,
    probe: InferenceProbe,
    expected_feature_pipeline_version: str,
    expected_lane_id: str | None,
    artifact_id: str | None,
) -> _CandidateChecks:
    """Run every promotion check that requires NO store mutation.

    Steps 1, 2, 4, 6, 7 of the Phase L flow plus the Hard-NO-#3 ship gate.
    Raises :class:`PromotionRefusedError` on the first failed check; on return
    the candidate is verified promotable (only the mutations remain).
    """
    # Step 2 — verify package + artifact digests from disk (re-hash).
    try:
        package = store.load_package(package_digest)
    except (FileNotFoundError, StoreCorruptionError, EvidenceStoreError) as exc:
        raise PromotionRefusedError("package_verification_failed", str(exc)) from exc
    if expected_lane_id is not None and package.lane_id != expected_lane_id:
        raise PromotionRefusedError(
            "lane_mismatch",
            f"package lane {package.lane_id!r} != expected {expected_lane_id!r}",
        )
    artifact = _select_artifact(package, artifact_id)
    try:
        artifact_bytes = store.load_package_file(package_digest, artifact.relative_path)
    except (FileNotFoundError, StoreCorruptionError, EvidenceStoreError) as exc:
        raise PromotionRefusedError("artifact_verification_failed", str(exc)) from exc
    if sha256_bytes(artifact_bytes) != artifact.digest:
        raise PromotionRefusedError(
            "artifact_digest_mismatch",
            f"re-hash of {artifact.relative_path!r} does not match its declared digest",
        )

    # Step 1 — resolve the current disposition head for the package.
    view = _resolve_ledger_from_store(store, package_digest)
    if view is None:
        raise PromotionRefusedError(
            "no_disposition_ledger", f"package {package_digest} has no disposition ledger"
        )
    if view.state not in _PROMOTABLE_STATES:
        raise PromotionRefusedError(
            "disposition_not_promotable",
            f"package {package_digest} is {view.state.value}; "
            "only QUARANTINED/SHADOW/OPERATOR_APPROVED (or CHAMPION re-point) may promote",
        )

    # Hard NO #3 — ship gate from EVERY pre-registered report in the package.
    reports = _load_evaluation_reports(store, package_digest, package)
    for report in reports:
        _enforce_ship_gate(report)

    # Step 4 — lane + feature-pipeline compatibility.
    job = _load_job_manifest(store, package_digest, package)
    for report in reports:
        if report.job_manifest_digest != package.job_manifest_digest:
            raise PromotionRefusedError(
                "evaluation_report_job_mismatch",
                "signed evaluation report binds a different job manifest than the package",
            )
    if job.feature_pipeline_version != expected_feature_pipeline_version:
        raise PromotionRefusedError(
            "feature_pipeline_mismatch",
            f"artifact was built with feature_pipeline_version="
            f"{job.feature_pipeline_version!r}; consumer expects "
            f"{expected_feature_pipeline_version!r}",
        )

    # Step 6 — load smoke test (real load of the digest-verified bytes).
    try:
        model = probe.load(artifact_bytes)
    except Exception as exc:  # noqa: BLE001 — any load failure is a refusal, not a crash
        raise PromotionRefusedError("load_smoke_failed", f"artifact failed to load: {exc}") from exc

    # Step 7 — dry inference with finite-output assertion (real predict).
    try:
        raw_outputs = probe.predict(model)
    except Exception as exc:  # noqa: BLE001 — any inference failure is a refusal
        raise PromotionRefusedError("dry_inference_failed", f"dry inference raised: {exc}") from exc
    outputs = tuple(_finite_outputs(raw_outputs))

    return _CandidateChecks(
        package=package,
        artifact=artifact,
        artifact_bytes=artifact_bytes,
        view=view,
        reports=reports,
        job=job,
        outputs=outputs,
    )


@dataclass(frozen=True)
class PreflightReport:
    """Read-only verdict: the package WOULD be promotable. Nothing was written."""

    schema_version: str
    action: str
    lane_id: str
    package_digest: str
    artifact_id: str
    artifact_digest: str
    disposition_state: str
    disposition_head_digest: str
    feature_pipeline_version: str
    gate_ids: tuple[str, ...]
    dry_inference_outputs: tuple[float, ...]
    promotable: bool

    def as_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["gate_ids"] = list(self.gate_ids)
        payload["dry_inference_outputs"] = list(self.dry_inference_outputs)
        return payload


def preflight(
    store: EvidenceStore,
    package_digest: str,
    *,
    probe: InferenceProbe,
    expected_feature_pipeline_version: str,
    expected_lane_id: str | None = None,
    artifact_id: str | None = None,
) -> PreflightReport:
    """READ-ONLY dry run of every promotion check — never mutates the store.

    Requires no operator approval and no signing keys because it cannot
    promote: no ledger append, no pointer write, no artifact copy, no JSONL
    log entry. Raises :class:`PromotionRefusedError` on any check the real
    promotion would refuse; returns a :class:`PreflightReport` when the
    package would be promotable.
    """
    checks = _verify_candidate(
        store,
        package_digest,
        probe=probe,
        expected_feature_pipeline_version=expected_feature_pipeline_version,
        expected_lane_id=expected_lane_id,
        artifact_id=artifact_id,
    )
    return PreflightReport(
        schema_version=PROMOTION_LOG_SCHEMA_VERSION,
        action="preflight",
        lane_id=checks.package.lane_id,
        package_digest=package_digest,
        artifact_id=checks.artifact.artifact_id,
        artifact_digest=checks.artifact.digest,
        disposition_state=checks.view.state.value,
        disposition_head_digest=checks.view.head_digest,
        feature_pipeline_version=checks.job.feature_pipeline_version,
        gate_ids=tuple(
            gate.gate_id for report in checks.reports for gate in report.gates
        ),
        dry_inference_outputs=checks.outputs,
        promotable=True,
    )


@dataclass(frozen=True)
class PromotionRecord:
    """Durable record of one completed promotion (JSONL step 10 + rollback input)."""

    schema_version: str
    action: str
    lane_id: str
    package_digest: str
    artifact_id: str
    artifact_digest: str
    pointer_digest: str
    disposition_head_digest: str
    prior_pointer_digest: str | None
    prior_pointer_envelope: dict[str, Any] | None
    retired_package_digest: str | None
    versioned_artifact_path: str
    operator_actor_id: str
    operator_reason: str
    promotion_actor_id: str
    promoted_at: str
    feature_pipeline_version: str
    gate_ids: tuple[str, ...] = ()
    dry_inference_outputs: tuple[float, ...] = ()

    def as_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["gate_ids"] = list(self.gate_ids)
        payload["dry_inference_outputs"] = list(self.dry_inference_outputs)
        return payload

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "PromotionRecord":
        data = dict(payload)
        data["gate_ids"] = tuple(data.get("gate_ids", ()))
        data["dry_inference_outputs"] = tuple(
            float(value) for value in data.get("dry_inference_outputs", ())
        )
        known = {f.name for f in cls.__dataclass_fields__.values()}  # type: ignore[attr-defined]
        return cls(**{key: value for key, value in data.items() if key in known})


@dataclass(frozen=True)
class RollbackRecord:
    """Durable record of one rollback attempt (always logged)."""

    schema_version: str
    action: str
    lane_id: str
    rolled_back_pointer_digest: str
    pointer_restored: bool
    restored_pointer_digest: str | None
    retired_package_digest: str | None
    structural_note: str | None
    operator_actor_id: str
    operator_reason: str
    rolled_back_at: str

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


class PromotionService:
    """Local Phase L promotion service over one :class:`EvidenceStore`.

    The operator and promotion-service signers must be DISTINCT identities
    (actor + key) from each other and from the producer / importer / verifier
    already present in the ledger — ``enforce_separation_of_duties`` inside
    the store rejects any overlap. Registration of both identities in the
    store's :class:`AuthorityRegistry` and :class:`TrustStore` is the caller's
    responsibility; unregistered signers fail closed at append time.
    """

    def __init__(
        self,
        store: EvidenceStore,
        *,
        operator_signer: Ed25519Signer,
        operator_actor_id: str,
        promotion_signer: Ed25519Signer,
        promotion_actor_id: str,
        artifact_root: str | Path | None = None,
        log_path: str | Path | None = None,
    ) -> None:
        if not operator_actor_id or not promotion_actor_id:
            raise ValueError("operator_actor_id and promotion_actor_id are required")
        if operator_actor_id == promotion_actor_id:
            raise ValueError(
                "operator and promotion-service actors must be distinct (separation of duties)"
            )
        if operator_signer.key_id == promotion_signer.key_id:
            raise ValueError(
                "operator and promotion-service signing keys must be distinct (separation of duties)"
            )
        self._store = store
        self._operator_signer = operator_signer
        self._operator_actor_id = operator_actor_id
        self._promotion_signer = promotion_signer
        self._promotion_actor_id = promotion_actor_id
        self._artifact_root = Path(artifact_root) if artifact_root else store.root / "champion_artifacts"
        self._log_path = Path(log_path) if log_path else store.root / "promotions" / "promotions.jsonl"

    # ------------------------------------------------------------------ #
    # public API                                                          #
    # ------------------------------------------------------------------ #

    def promote(
        self,
        package_digest: str,
        *,
        approval: OperatorApproval | None,
        probe: InferenceProbe,
        expected_feature_pipeline_version: str,
        expected_lane_id: str | None = None,
        artifact_id: str | None = None,
        now: datetime | None = None,
    ) -> PromotionRecord:
        """Promote one quarantined evidence package to lane champion.

        Refuses (raising :class:`PromotionRefusedError`) on: missing approval,
        any digest failure, a non-promotable disposition state, a FAIL ship
        gate, missing gate evidence, lane or feature-pipeline mismatch, or a
        failed load/dry-inference smoke. All verification happens BEFORE any
        ledger mutation.
        """
        # Step 3 — explicit operator approval, first and fail-closed.
        self._require_approval(approval)
        assert approval is not None
        now = self._utc_now(now)

        # Steps 1, 2, 4, 6, 7 + ship gate — the shared READ-ONLY pipeline.
        # Nothing below mutates the store until every check has passed.
        checks = _verify_candidate(
            self._store,
            package_digest,
            probe=probe,
            expected_feature_pipeline_version=expected_feature_pipeline_version,
            expected_lane_id=expected_lane_id,
            artifact_id=artifact_id,
        )
        package = checks.package
        artifact = checks.artifact
        job = checks.job
        outputs = checks.outputs

        # Step 5 — copy the verified candidate into the versioned artifact store.
        artifact_path = self._write_versioned_artifact(
            package, package_digest, artifact, checks.artifact_bytes
        )

        # Steps 1/11 bookkeeping — capture the prior champion for CAS + rollback.
        prior_envelope, expected_pointer_digest = self._current_pointer(package.lane_id)
        prior_pointer: ChampionPointer | None = None
        if prior_envelope is not None:
            decoded = ChampionPointer.from_versioned_payload(prior_envelope.payload)
            assert isinstance(decoded, ChampionPointer)
            prior_pointer = decoded

        retired_package_digest: str | None = None
        same_package_repoint = (
            prior_pointer is not None and prior_pointer.package_digest == package_digest
        )
        try:
            # Step 9 (policy-ordered before step 8) — drive the ledger through
            # the transition policy, retiring an incumbent champion first.
            if prior_pointer is not None and not same_package_repoint:
                self._retire_champion(
                    prior_pointer.package_digest,
                    reason=(
                        f"superseded by operator-approved promotion of {package_digest}: "
                        f"{approval.reason}"
                    ),
                    base_time=now,
                )
                retired_package_digest = prior_pointer.package_digest
            # A pointer that selects another (now retired) package — or a
            # stale pointer left by an earlier rollback — must be removed
            # before this package's CHAMPION event can commit: the store's
            # index projection validates the pointer against the champion
            # package. CAS-guarded; the lane serves no champion in between
            # (fail-closed), never a wrong one.
            pointer_cas = expected_pointer_digest
            if expected_pointer_digest is not None and not same_package_repoint:
                self._store.clear_champion_pointer(
                    package.lane_id, expected_pointer_digest=expected_pointer_digest
                )
                pointer_cas = None
            head_digest = self._drive_to_champion(
                package_digest, artifact_id=artifact.artifact_id,
                artifact_digest=artifact.digest, lane_id=package.lane_id,
                approval=approval, base_time=now,
            )

            # Step 8 — atomic champion-pointer swap with exact hashes + CAS.
            pointer = ChampionPointer(
                pointer_id=f"{package.lane_id}:{package_digest[:16]}:{uuid4().hex[:12]}",
                lane_id=package.lane_id,
                package_digest=package_digest,
                artifact_id=artifact.artifact_id,
                artifact_digest=artifact.digest,
                disposition_head_digest=head_digest,
                actor_id=self._promotion_actor_id,
                signer_key_id=self._promotion_signer.key_id,
                created_at=now,
            )
            pointer_envelope = self._promotion_signer.sign(pointer, created_at=now)
            pointer_digest = self._store.write_champion_pointer(
                pointer_envelope, expected_pointer_digest=pointer_cas
            )
        except PromotionError:
            raise
        except (EvidenceStoreError, SignatureVerificationError, ValidationError) as exc:
            self._append_log(
                {
                    "schema_version": PROMOTION_LOG_SCHEMA_VERSION,
                    "action": "promote_failed",
                    "lane_id": package.lane_id,
                    "package_digest": package_digest,
                    "error": str(exc),
                    "occurred_at": now.isoformat(),
                }
            )
            raise PromotionRefusedError("promotion_mutation_failed", str(exc)) from exc

        record = PromotionRecord(
            schema_version=PROMOTION_LOG_SCHEMA_VERSION,
            action="promote",
            lane_id=package.lane_id,
            package_digest=package_digest,
            artifact_id=artifact.artifact_id,
            artifact_digest=artifact.digest,
            pointer_digest=pointer_digest,
            disposition_head_digest=head_digest,
            prior_pointer_digest=expected_pointer_digest,
            prior_pointer_envelope=(
                prior_envelope.model_dump(mode="json", round_trip=True, warnings=False)
                if prior_envelope
                else None
            ),
            retired_package_digest=retired_package_digest,
            versioned_artifact_path=str(artifact_path),
            operator_actor_id=self._operator_actor_id,
            operator_reason=approval.reason,
            promotion_actor_id=self._promotion_actor_id,
            promoted_at=now.isoformat(),
            feature_pipeline_version=job.feature_pipeline_version,
            gate_ids=tuple(
                gate.gate_id for report in checks.reports for gate in report.gates
            ),
            dry_inference_outputs=tuple(outputs),
        )
        # Step 10 — durable JSONL promotion event (dashboard/Tier 7 can tail it).
        self._append_log(record.as_dict())
        return record

    def rollback(
        self,
        record: PromotionRecord,
        *,
        approval: OperatorApproval | None,
        now: datetime | None = None,
    ) -> RollbackRecord:
        """Roll back one promotion, preserving every transition-policy rail.

        If the prior champion package is still CHAMPION (pointer-only change,
        e.g. an artifact re-point within the same package), the preserved
        prior pointer is restored atomically via the store's CAS write.

        If the prior package was RETIRED during promotion, the transition
        policy has no RETIRED → CHAMPION edge — exact restoration is
        structurally forbidden and this method will NOT weaken the policy to
        do it. Instead it fails closed: the promoted champion is retired
        (CHAMPION → RETIRED, operator authority), leaving the lane with no
        loadable champion, and the structural refusal is recorded.
        """
        self._require_approval(approval)
        assert approval is not None
        now = self._utc_now(now)
        lane_id = record.lane_id

        current_digest = self._store.peek_champion_pointer_digest(lane_id)
        if current_digest is None:
            raise PromotionRefusedError(
                "rollback_no_pointer", f"lane {lane_id!r} has no champion pointer to roll back"
            )
        if current_digest != record.pointer_digest:
            raise PromotionRefusedError(
                "rollback_pointer_moved",
                f"lane {lane_id!r} pointer is {current_digest}, not the promoted "
                f"{record.pointer_digest}; refusing to roll back a pointer this record did not write",
            )
        if record.prior_pointer_envelope is None:
            raise PromotionRefusedError(
                "rollback_nothing_to_restore",
                "promotion record preserved no prior champion pointer",
            )
        try:
            prior_envelope = SignedEnvelope.from_versioned_payload(record.prior_pointer_envelope)
            assert isinstance(prior_envelope, SignedEnvelope)
            prior_pointer = ChampionPointer.from_versioned_payload(prior_envelope.payload)
            assert isinstance(prior_pointer, ChampionPointer)
        except (ValidationError, ValueError) as exc:
            raise PromotionRefusedError(
                "rollback_prior_pointer_invalid", f"preserved prior pointer is unreadable: {exc}"
            ) from exc

        prior_view = self._resolve_ledger(prior_pointer.package_digest)
        if prior_view is not None and prior_view.state == DispositionState.CHAMPION:
            try:
                restored_digest = self._store.write_champion_pointer(
                    prior_envelope, expected_pointer_digest=current_digest
                )
            except (EvidenceStoreError, SignatureVerificationError) as exc:
                raise PromotionRefusedError("rollback_restore_failed", str(exc)) from exc
            result = RollbackRecord(
                schema_version=PROMOTION_LOG_SCHEMA_VERSION,
                action="rollback",
                lane_id=lane_id,
                rolled_back_pointer_digest=record.pointer_digest,
                pointer_restored=True,
                restored_pointer_digest=restored_digest,
                retired_package_digest=None,
                structural_note=None,
                operator_actor_id=self._operator_actor_id,
                operator_reason=approval.reason,
                rolled_back_at=now.isoformat(),
            )
        else:
            prior_state = prior_view.state.value if prior_view else "absent"
            promoted_view = self._resolve_ledger(record.package_digest)
            if promoted_view is None or promoted_view.state != DispositionState.CHAMPION:
                raise PromotionRefusedError(
                    "rollback_champion_missing",
                    f"promoted package {record.package_digest} is not CHAMPION; nothing to retire",
                )
            self._retire_champion(
                record.package_digest,
                reason=f"operator rollback of promotion {record.pointer_digest}: {approval.reason}",
                base_time=now,
            )
            result = RollbackRecord(
                schema_version=PROMOTION_LOG_SCHEMA_VERSION,
                action="rollback",
                lane_id=lane_id,
                rolled_back_pointer_digest=record.pointer_digest,
                pointer_restored=False,
                restored_pointer_digest=None,
                retired_package_digest=record.package_digest,
                structural_note=(
                    "transition policy has no RETIRED->CHAMPION edge; prior champion "
                    f"package is {prior_state} and cannot be re-pointed. Failed closed: "
                    "promoted champion retired, lane now serves no champion. The prior "
                    "artifact remains preserved in the versioned artifact store for a "
                    "legitimate re-promotion cycle."
                ),
                operator_actor_id=self._operator_actor_id,
                operator_reason=approval.reason,
                rolled_back_at=now.isoformat(),
            )
        self._append_log(result.as_dict())
        return result

    # ------------------------------------------------------------------ #
    # verification helpers                                                #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _require_approval(approval: OperatorApproval | None) -> None:
        if approval is None:
            raise PromotionRefusedError(
                "operator_approval_missing",
                "promotion requires an explicit OperatorApproval; there is no auto-approve path",
            )
        if not isinstance(approval.reason, str) or not approval.reason.strip():
            raise PromotionRefusedError(
                "operator_approval_missing", "operator approval requires a non-empty reason"
            )

    @staticmethod
    def _utc_now(now: datetime | None) -> datetime:
        if now is None:
            return datetime.now(timezone.utc)
        if now.tzinfo is None or now.utcoffset() is None:
            raise ValueError("now must be timezone-aware")
        return now.astimezone(timezone.utc)

    def _resolve_ledger(self, package_digest: str) -> LedgerView | None:
        return _resolve_ledger_from_store(self._store, package_digest)

    # ------------------------------------------------------------------ #
    # versioned artifact store (step 5)                                   #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _safe_join(base: Path, relative_path: str) -> Path:
        normalized = PurePosixPath(relative_path.replace("\\", "/"))
        if normalized.is_absolute() or ".." in normalized.parts or not normalized.parts:
            raise PromotionRefusedError(
                "artifact_verification_failed", f"unsafe artifact path: {relative_path!r}"
            )
        return base.joinpath(*normalized.parts)

    def _write_versioned_artifact(
        self, package: EvidencePackage, package_digest: str, artifact, artifact_bytes: bytes
    ) -> Path:
        lane_component = "".join(
            ch if ch.isalnum() or ch in "._-" else "_" for ch in package.lane_id
        )
        version_dir = self._artifact_root / lane_component / package_digest
        destination = self._safe_join(version_dir, artifact.relative_path)
        if destination.exists():
            existing = destination.read_bytes()
            if sha256_bytes(existing) != artifact.digest:
                raise PromotionRefusedError(
                    "artifact_store_conflict",
                    f"versioned artifact copy at {destination} does not match the "
                    "package digest; refusing to overwrite",
                )
            return destination
        destination.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        temporary = destination.parent / f".{destination.name}.tmp-{uuid4().hex}"
        descriptor = os.open(temporary, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        try:
            with os.fdopen(descriptor, "wb") as handle:
                handle.write(artifact_bytes)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, destination)
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise
        return destination

    # ------------------------------------------------------------------ #
    # disposition drive (step 9) — always THROUGH the transition policy   #
    # ------------------------------------------------------------------ #

    def _append_event(
        self,
        *,
        package_digest: str,
        sequence: int,
        previous_digest: str,
        from_state: DispositionState,
        to_state: DispositionState,
        signer: Ed25519Signer,
        actor_id: str,
        role: AuthorityRole,
        reason: str,
        occurred_at: datetime,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        event = DispositionEvent(
            package_digest=package_digest,
            sequence=sequence,
            previous_event_digest=previous_digest,
            from_state=from_state,
            to_state=to_state,
            authority=role,
            actor_id=actor_id,
            signer_key_id=signer.key_id,
            occurred_at=occurred_at,
            reason=reason,
            metadata=metadata or {},
        )
        envelope = signer.sign(event, created_at=occurred_at)
        state = self._store.append_disposition(envelope, expected_head_digest=previous_digest)
        return state.head_event_digest

    def _drive_to_champion(
        self,
        package_digest: str,
        *,
        artifact_id: str,
        artifact_digest: str,
        lane_id: str,
        approval: OperatorApproval,
        base_time: datetime,
    ) -> str:
        view = self._resolve_ledger(package_digest)
        assert view is not None
        if view.state == DispositionState.CHAMPION:
            return view.head_digest

        operator = (self._operator_signer, self._operator_actor_id, AuthorityRole.OPERATOR)
        promoter = (
            self._promotion_signer,
            self._promotion_actor_id,
            AuthorityRole.PROMOTION_SERVICE,
        )
        remaining: dict[DispositionState, list] = {
            DispositionState.QUARANTINED: [
                (DispositionState.SHADOW, operator),
                (DispositionState.OPERATOR_APPROVED, operator),
                (DispositionState.CHAMPION, promoter),
            ],
            DispositionState.SHADOW: [
                (DispositionState.OPERATOR_APPROVED, operator),
                (DispositionState.CHAMPION, promoter),
            ],
            DispositionState.OPERATOR_APPROVED: [
                (DispositionState.CHAMPION, promoter),
            ],
        }
        steps = remaining[view.state]
        occurred = max(base_time, view.last_occurred_at + timedelta(seconds=1))
        head = view.head_digest
        sequence = view.next_sequence
        from_state = view.state
        for offset, (to_state, (signer, actor_id, role)) in enumerate(steps):
            metadata: dict[str, Any] = {"operator_reason": approval.reason}
            if to_state == DispositionState.CHAMPION:
                metadata.update(
                    {
                        "lane_id": lane_id,
                        "artifact_id": artifact_id,
                        "artifact_digest": artifact_digest,
                    }
                )
            head = self._append_event(
                package_digest=package_digest,
                sequence=sequence + offset,
                previous_digest=head,
                from_state=from_state,
                to_state=to_state,
                signer=signer,
                actor_id=actor_id,
                role=role,
                reason=f"operator-approved promotion: {approval.reason}",
                occurred_at=occurred + timedelta(seconds=offset),
                metadata=metadata,
            )
            from_state = to_state
        return head

    def _retire_champion(self, package_digest: str, *, reason: str, base_time: datetime) -> str:
        view = self._resolve_ledger(package_digest)
        if view is None or view.state != DispositionState.CHAMPION:
            state = view.state.value if view else "absent"
            raise PromotionRefusedError(
                "retire_target_not_champion",
                f"package {package_digest} is {state}; only a CHAMPION can be retired here",
            )
        occurred = max(base_time, view.last_occurred_at + timedelta(seconds=1))
        return self._append_event(
            package_digest=package_digest,
            sequence=view.next_sequence,
            previous_digest=view.head_digest,
            from_state=DispositionState.CHAMPION,
            to_state=DispositionState.RETIRED,
            signer=self._operator_signer,
            actor_id=self._operator_actor_id,
            role=AuthorityRole.OPERATOR,
            reason=reason,
            occurred_at=occurred,
        )

    # ------------------------------------------------------------------ #
    # promotion log (step 10)                                             #
    # ------------------------------------------------------------------ #

    @property
    def log_path(self) -> Path:
        return self._log_path

    def _append_log(self, entry: Mapping[str, Any]) -> None:
        self._log_path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        line = (json.dumps(dict(entry), sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")
        descriptor = os.open(self._log_path, os.O_CREAT | os.O_WRONLY | os.O_APPEND, 0o600)
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX)
            os.write(descriptor, line)
            os.fsync(descriptor)
        finally:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
            os.close(descriptor)

    def load_promotion_records(self, lane_id: str | None = None) -> tuple[PromotionRecord, ...]:
        """Read completed promotion records back from the JSONL log."""
        if not self._log_path.exists():
            return ()
        records: list[PromotionRecord] = []
        for line in self._log_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue
            if payload.get("action") != "promote":
                continue
            if lane_id is not None and payload.get("lane_id") != lane_id:
                continue
            try:
                records.append(PromotionRecord.from_dict(payload))
            except TypeError:
                continue
        return tuple(records)

    # ------------------------------------------------------------------ #
    # pointer bookkeeping                                                 #
    # ------------------------------------------------------------------ #

    def _current_pointer(self, lane_id: str) -> tuple[SignedEnvelope | None, str | None]:
        """Return (validated prior envelope or None, raw digest for CAS)."""
        try:
            envelope = self._store.load_champion_pointer_envelope(lane_id)
            return envelope, envelope.payload_digest
        except FileNotFoundError:
            return None, None
        except ChampionPointerError:
            # Stale pointer (e.g. its package was retired by a rollback):
            # unusable as a prior champion, but its raw digest is still the
            # CAS expectation for the atomic replacement write.
            return None, self._store.peek_champion_pointer_digest(lane_id)


# --------------------------------------------------------------------------- #
# Persistent promotion-side identities (operator + promotion service)          #
# --------------------------------------------------------------------------- #

_PROMOTION_ROLES = (AuthorityRole.OPERATOR, AuthorityRole.PROMOTION_SERVICE)
_DEFAULT_PROMOTION_ACTORS = {
    AuthorityRole.OPERATOR: "axiom-local-operator",
    AuthorityRole.PROMOTION_SERVICE: "axiom-local-promotion-service",
}
PROMOTION_IDENTITY_METADATA_FILENAME = "promotion_identities.json"


class PromotionIdentityError(RuntimeError):
    """Persisted promotion identity material is missing or inconsistent."""


@dataclass(frozen=True)
class PromotionIdentities:
    """Operator + promotion-service signing identities for the local CLI."""

    signers: Mapping[AuthorityRole, Ed25519Signer]
    actors: Mapping[AuthorityRole, str]
    valid_from: Mapping[AuthorityRole, datetime] = field(default_factory=dict)

    @property
    def operator(self) -> Ed25519Signer:
        return self.signers[AuthorityRole.OPERATOR]

    @property
    def promotion(self) -> Ed25519Signer:
        return self.signers[AuthorityRole.PROMOTION_SERVICE]

    def register(self, trust_store: TrustStore, authorities: AuthorityRegistry) -> None:
        for role in _PROMOTION_ROLES:
            signer = self.signers[role]
            trust_store.add(signer.trusted_key(valid_from=self.valid_from[role]))
            authorities.register(
                actor_id=self.actors[role], role=role, key_ids=(signer.key_id,)
            )


def load_or_create_promotion_identities(
    key_dir: str | Path,
    *,
    now: datetime,
    actors: Mapping[AuthorityRole, str] | None = None,
) -> PromotionIdentities:
    """Load (or mint once) persistent operator + promotion-service keys.

    Mirrors ``risk_target.persistent_identity``: raw 32-byte Ed25519 keys,
    mode 0600, ``O_CREAT|O_EXCL`` (never overwritten), plus a metadata file
    binding role -> actor_id, key_id and the trust interval's ``valid_from``
    so envelopes signed by an earlier run keep verifying.
    """
    if now.tzinfo is None or now.utcoffset() is None:
        raise ValueError("now must be timezone-aware")
    key_dir = Path(key_dir)
    key_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
    metadata_path = key_dir / PROMOTION_IDENTITY_METADATA_FILENAME
    actors = dict(actors or _DEFAULT_PROMOTION_ACTORS)

    if metadata_path.exists():
        return _load_existing_promotion_identities(key_dir, metadata_path)

    signers = {role: Ed25519Signer.generate() for role in _PROMOTION_ROLES}
    valid_from = (now - timedelta(days=1)).astimezone(timezone.utc)
    for role, signer in signers.items():
        _write_raw_private_key(key_dir / f"{role.value}.key", signer)
    metadata = {
        "schema_version": PROMOTION_LOG_SCHEMA_VERSION,
        "created_at": now.astimezone(timezone.utc).isoformat(),
        "roles": {
            role.value: {
                "actor_id": actors[role],
                "key_id": signers[role].key_id,
                "valid_from": valid_from.isoformat(),
            }
            for role in _PROMOTION_ROLES
        },
    }
    temporary = metadata_path.parent / f".{metadata_path.name}.tmp"
    temporary.write_text(json.dumps(metadata, indent=2, sort_keys=True))
    os.replace(temporary, metadata_path)
    return PromotionIdentities(
        signers=signers,
        actors=actors,
        valid_from={role: valid_from for role in _PROMOTION_ROLES},
    )


def _write_raw_private_key(path: Path, signer: Ed25519Signer) -> None:
    descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(signer.private_bytes())
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException:
        path.unlink(missing_ok=True)
        raise


def _load_existing_promotion_identities(
    key_dir: Path, metadata_path: Path
) -> PromotionIdentities:
    import stat

    try:
        metadata = json.loads(metadata_path.read_text())
        role_entries = metadata["roles"]
    except (OSError, json.JSONDecodeError, KeyError, TypeError) as exc:
        raise PromotionIdentityError(
            f"unreadable promotion identity metadata {metadata_path}: {exc}"
        ) from exc

    signers: dict[AuthorityRole, Ed25519Signer] = {}
    actors: dict[AuthorityRole, str] = {}
    valid_from_by_role: dict[AuthorityRole, datetime] = {}
    for role in _PROMOTION_ROLES:
        entry = role_entries.get(role.value)
        if not isinstance(entry, dict):
            raise PromotionIdentityError(
                f"promotion identity metadata has no entry for role {role.value!r}"
            )
        key_path = key_dir / f"{role.value}.key"
        if not key_path.exists():
            raise PromotionIdentityError(f"missing private key file: {key_path}")
        mode = stat.S_IMODE(key_path.stat().st_mode)
        if mode & 0o077:
            raise PromotionIdentityError(
                f"private key {key_path} is group/world accessible (mode {oct(mode)})"
            )
        raw = key_path.read_bytes()
        if len(raw) != 32:
            raise PromotionIdentityError(f"private key {key_path} is not raw 32-byte Ed25519")
        signer = Ed25519Signer.from_private_bytes(raw)
        if signer.key_id != entry.get("key_id"):
            raise PromotionIdentityError(
                f"key file {key_path} derives key_id {signer.key_id}, metadata records "
                f"{entry.get('key_id')!r} — refusing a swapped key"
            )
        try:
            valid_from = datetime.fromisoformat(entry["valid_from"])
        except (KeyError, TypeError, ValueError) as exc:
            raise PromotionIdentityError(
                f"invalid valid_from for role {role.value!r}"
            ) from exc
        if valid_from.tzinfo is None or valid_from.utcoffset() is None:
            raise PromotionIdentityError(
                f"valid_from for role {role.value!r} must be timezone-aware"
            )
        actor_id = entry.get("actor_id")
        if not isinstance(actor_id, str) or not actor_id:
            raise PromotionIdentityError(f"invalid actor_id for role {role.value!r}")
        signers[role] = signer
        actors[role] = actor_id
        valid_from_by_role[role] = valid_from
    return PromotionIdentities(signers=signers, actors=actors, valid_from=valid_from_by_role)


__all__ = [
    "InferenceProbe",
    "LedgerView",
    "OperatorApproval",
    "PreflightReport",
    "PromotionError",
    "PromotionIdentities",
    "PromotionIdentityError",
    "PromotionRecord",
    "PromotionRefusedError",
    "PromotionService",
    "RollbackRecord",
    "load_or_create_promotion_identities",
    "preflight",
]
