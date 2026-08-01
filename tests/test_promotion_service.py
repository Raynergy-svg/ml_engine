"""Phase L local promotion service — real, no-mock acceptance tests.

Every test runs the real :class:`EvidenceStore` on ``tmp_path``, real Ed25519
signing through the real contracts, the real transition policy, and a real
tiny sklearn artifact for the load-smoke / dry-inference steps. No
``unittest.mock``, no ``MagicMock``, no ``patch``.

Covered (per the Phase L mission):
- happy path writes a champion pointer + preserves the prior champion
- ship gate (Hard NO #3): a FAIL-gate head is refused; missing gate evidence
  refuses fail-closed; a REJECTED head (drawdown) is refused
- digest mismatch refuses
- missing operator approval refuses (no auto-approve path)
- feature-pipeline mismatch refuses
- transition-policy violations refuse (unregistered operator key)
- rollback restores the prior pointer where the policy permits it, and fails
  CLOSED (retire, no champion) where the policy structurally forbids
  restoration (no RETIRED -> CHAMPION edge)
"""

from __future__ import annotations

import json
import pickle
from datetime import date, timedelta

import numpy as np
import pytest
from sklearn.linear_model import LinearRegression

from src.evidence.canonical import canonical_bytes
from src.evidence.contracts import (
    ArtifactRef,
    AuthorityRole,
    DispositionEvent,
    DispositionState,
    EvaluationReport,
    EvidencePackage,
    GateResult,
    GateStatus,
    ImportCheck,
    JobManifest,
    MetricValue,
    ModelTrainingSpec,
    ResourceClass,
    SafetyAssertion,
)
from src.evidence.hashing import sha256_bytes
from src.evidence.importer import build_import_verdict
from src.evidence.promotion import (
    InferenceProbe,
    OperatorApproval,
    PromotionRecord,
    PromotionRefusedError,
    PromotionService,
    load_or_create_promotion_identities,
    preflight,
)
from src.evidence.risk_target.models import (
    DRAWDOWN_HEAD_ID,
    VOLATILITY_HEAD_ID,
    HeadResult,
)
from src.evidence.signing import Ed25519Signer, verify_envelope
from src.evidence.store import ChampionPointerError
from src.evidence.risk_target.slice import (
    build_evidence_store,
    build_slice_identities,
    run_risk_target_evidence_slice,
    utc,
)

NOW = utc(2026, 7, 13, 12, 0)
FPV = "fpv-test-1.0"
N_FEATURES = 3
SLICE_KW = dict(
    dataset_id="fx-daily-test",
    coverage_start=date(2019, 1, 1),
    coverage_end=date(2026, 1, 1),
    retrieved_at=NOW,
    created_at=NOW,
    git_commit="0" * 40,
    feature_pipeline_version=FPV,
)

OPERATOR_ACTOR = "test-operator"
PROMOTION_ACTOR = "test-promotion-service"


def _fitted_model_bytes(seed: int) -> bytes:
    rng = np.random.default_rng(seed)
    features = rng.normal(size=(30, N_FEATURES))
    target = features @ np.array([0.5, -0.2, 0.1]) + 1.0
    model = LinearRegression().fit(features, target)
    return pickle.dumps(model)


MODEL_BYTES = _fitted_model_bytes(7)
PROBE = InferenceProbe(
    load=pickle.loads,
    predict=lambda model: model.predict(np.zeros((1, N_FEATURES))),
)


def _noop_registry(experiment, metrics):
    return None


def _dummy_partitions():
    return {
        "EUR_USD": b"EUR_USD synthetic risk-target partition bytes\n",
        "USD_JPY": b"USD_JPY synthetic risk-target partition bytes\n",
    }


def _head(head_id: str, lane_id: str, passed: bool, score: float) -> HeadResult:
    status = GateStatus.PASS if passed else GateStatus.FAIL
    gates = (
        GateResult(gate_id=f"{head_id}_admissibility", status=status, observed=score,
                   threshold=0.1, reason="synthetic admissibility gate"),
    )
    return HeadResult(
        head_id=head_id, lane_id=lane_id, target=head_id,
        metrics={"score": score, "secondary": score * 2.0},
        metric_tolerances={"score": 1e-9, "secondary": 1e-9},
        gates=gates, passed=passed,
        model_bytes=MODEL_BYTES,
        media_type="application/octet-stream",
        trial_count=1, effective_sample_size=120.0, temporal_holdout="OOS bucket",
        purge_observations=20, embargo_observations=20,
    )


def _sklearn_evaluator(vol_passed: bool = True, dd_passed: bool = False, score: float = 0.05):
    def evaluator(partitions, params):
        return (
            _head(VOLATILITY_HEAD_ID, "risk_target_vol", vol_passed, score),
            _head(DRAWDOWN_HEAD_ID, "risk_target_drawdown", dd_passed, score),
        )
    return evaluator


def _identities_with_promotion_roles():
    identities = build_slice_identities(NOW)
    operator = Ed25519Signer.generate()
    promotion = Ed25519Signer.generate()
    valid_from = NOW - timedelta(days=1)
    identities.trust_store.add(operator.trusted_key(valid_from=valid_from))
    identities.trust_store.add(promotion.trusted_key(valid_from=valid_from))
    identities.authorities.register(
        actor_id=OPERATOR_ACTOR, role=AuthorityRole.OPERATOR, key_ids=(operator.key_id,)
    )
    identities.authorities.register(
        actor_id=PROMOTION_ACTOR,
        role=AuthorityRole.PROMOTION_SERVICE,
        key_ids=(promotion.key_id,),
    )
    return identities, operator, promotion


def _service(store, operator, promotion) -> PromotionService:
    return PromotionService(
        store,
        operator_signer=operator,
        operator_actor_id=OPERATOR_ACTOR,
        promotion_signer=promotion,
        promotion_actor_id=PROMOTION_ACTOR,
    )


def _slice_store(tmp_path, score: float = 0.05):
    identities, operator, promotion = _identities_with_promotion_roles()
    store = build_evidence_store(tmp_path / "evidence", identities, clock_now=NOW)
    result = run_risk_target_evidence_slice(
        store, identities, _dummy_partitions(),
        evaluator=_sklearn_evaluator(score=score),
        registry_publish=_noop_registry, **SLICE_KW,
    )
    return identities, store, result, _service(store, operator, promotion)


def _rerun_slice(store, identities, score: float):
    return run_risk_target_evidence_slice(
        store, identities, _dummy_partitions(),
        evaluator=_sklearn_evaluator(score=score),
        registry_publish=_noop_registry, **SLICE_KW,
    )


def _approval(reason: str = "operator validated forward-vol head on holdout"):
    return OperatorApproval(reason=reason)


def _promote_kwargs(**overrides):
    kwargs = dict(
        approval=_approval(),
        probe=PROBE,
        expected_feature_pipeline_version=FPV,
        now=NOW + timedelta(seconds=10),
    )
    kwargs.update(overrides)
    return kwargs


def _ledger_states(store, identities, package_digest):
    events = [
        verify_envelope(env, DispositionEvent, identities.trust_store)
        for env in store.load_ledger(package_digest)
    ]
    return [event.to_state for event in events], events


# --------------------------------------------------------------------------- #
# Happy path                                                                   #
# --------------------------------------------------------------------------- #

def test_happy_path_promotes_quarantined_vol_head(tmp_path):
    identities, store, result, service = _slice_store(tmp_path)
    vol = result.outcomes["risk_target_vol"]
    assert vol.final_state == DispositionState.QUARANTINED

    record = service.promote(vol.package_digest, **_promote_kwargs())

    # Champion pointer resolves to the exact package + artifact hashes.
    pointer = store.load_champion_pointer("risk_target_vol")
    package = store.load_package(vol.package_digest)
    artifact = package.artifacts[0]
    assert pointer.package_digest == vol.package_digest
    assert pointer.artifact_digest == artifact.digest
    assert record.pointer_digest and record.artifact_digest == artifact.digest

    # The ledger was driven THROUGH the transition policy, never around it.
    states, events = _ledger_states(store, identities, vol.package_digest)
    assert states[-3:] == [
        DispositionState.SHADOW,
        DispositionState.OPERATOR_APPROVED,
        DispositionState.CHAMPION,
    ]
    assert events[-3].authority == AuthorityRole.OPERATOR
    assert events[-2].authority == AuthorityRole.OPERATOR
    assert events[-1].authority == AuthorityRole.PROMOTION_SERVICE
    assert _approval().reason in events[-1].reason

    # Step 5 — versioned artifact copy exists and re-hashes to the digest.
    copied = tmp_path / "evidence" / "champion_artifacts" / "risk_target_vol" / vol.package_digest
    files = [path for path in copied.rglob("*") if path.is_file()]
    assert len(files) == 1
    assert sha256_bytes(files[0].read_bytes()) == artifact.digest

    # Step 7 — dry inference really ran on the pickled sklearn model.
    assert record.dry_inference_outputs
    assert all(np.isfinite(record.dry_inference_outputs))

    # Step 10 — the JSONL promotion event landed and round-trips.
    logged = service.load_promotion_records("risk_target_vol")
    assert len(logged) == 1
    assert logged[0].package_digest == vol.package_digest


def test_second_promotion_preserves_prior_champion_for_rollback(tmp_path):
    identities, store, result_a, service = _slice_store(tmp_path)
    package_a = result_a.outcomes["risk_target_vol"].package_digest
    record_a = service.promote(package_a, **_promote_kwargs())

    result_b = _rerun_slice(store, identities, score=0.06)
    package_b = result_b.outcomes["risk_target_vol"].package_digest
    assert package_b != package_a
    record_b = service.promote(
        package_b, **_promote_kwargs(now=NOW + timedelta(seconds=18))
    )

    # Pointer now selects B; the record preserves A's pointer + retirement.
    assert store.load_champion_pointer("risk_target_vol").package_digest == package_b
    assert record_b.prior_pointer_digest == record_a.pointer_digest
    assert record_b.prior_pointer_envelope is not None
    assert record_b.retired_package_digest == package_a

    # The incumbent was retired through the policy, not deleted: its ledger,
    # package bytes, and versioned artifact copy all remain verifiable.
    states_a, _ = _ledger_states(store, identities, package_a)
    assert states_a[-1] == DispositionState.RETIRED
    assert store.load_package(package_a) is not None
    prior_copy = (
        tmp_path / "evidence" / "champion_artifacts" / "risk_target_vol" / package_a
    )
    assert any(path.is_file() for path in prior_copy.rglob("*"))


# --------------------------------------------------------------------------- #
# Operator approval (step 3)                                                   #
# --------------------------------------------------------------------------- #

def test_missing_operator_approval_refuses_and_mutates_nothing(tmp_path):
    identities, store, result, service = _slice_store(tmp_path)
    vol = result.outcomes["risk_target_vol"]
    before, _ = _ledger_states(store, identities, vol.package_digest)

    with pytest.raises(PromotionRefusedError, match="operator_approval_missing"):
        service.promote(vol.package_digest, **_promote_kwargs(approval=None))
    with pytest.raises(PromotionRefusedError, match="non-empty reason"):
        service.promote(
            vol.package_digest, **_promote_kwargs(approval=OperatorApproval(reason="   "))
        )

    after, _ = _ledger_states(store, identities, vol.package_digest)
    assert after == before
    with pytest.raises(FileNotFoundError):
        store.load_champion_pointer("risk_target_vol")


# --------------------------------------------------------------------------- #
# Read-only preflight (dry-run)                                                #
# --------------------------------------------------------------------------- #

def _store_tree_snapshot(root):
    """Map every file under the store root to (mtime_ns, content digest)."""
    return {
        str(path.relative_to(root)): (path.stat().st_mtime_ns, sha256_bytes(path.read_bytes()))
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def test_preflight_reports_promotable_and_writes_nothing(tmp_path):
    identities, store, result, _ = _slice_store(tmp_path)
    vol = result.outcomes["risk_target_vol"]
    before = _store_tree_snapshot(store.root)

    report = preflight(
        store, vol.package_digest,
        probe=PROBE, expected_feature_pipeline_version=FPV,
        expected_lane_id="risk_target_vol",
    )

    assert report.promotable is True
    assert report.disposition_state == DispositionState.QUARANTINED.value
    assert report.package_digest == vol.package_digest
    assert report.gate_ids and report.dry_inference_outputs
    assert all(np.isfinite(report.dry_inference_outputs))

    # ZERO writes: every file byte-identical with unchanged mtimes, champions
    # still empty, no artifact copy, no promotion log, no ledger growth.
    assert _store_tree_snapshot(store.root) == before
    assert list((store.root / "champions").iterdir()) == []
    assert not (store.root / "champion_artifacts").exists()
    assert not (store.root / "promotions").exists()


def test_preflight_refuses_rejected_head_without_writing(tmp_path):
    identities, store, result, _ = _slice_store(tmp_path)
    dd = result.outcomes["risk_target_drawdown"]
    before = _store_tree_snapshot(store.root)
    with pytest.raises(PromotionRefusedError, match="disposition_not_promotable"):
        preflight(
            store, dd.package_digest,
            probe=PROBE, expected_feature_pipeline_version=FPV,
        )
    assert _store_tree_snapshot(store.root) == before


# --------------------------------------------------------------------------- #
# Ship gate — Hard NO #3                                                       #
# --------------------------------------------------------------------------- #

def test_rejected_drawdown_head_is_refused(tmp_path):
    identities, store, result, service = _slice_store(tmp_path)
    dd = result.outcomes["risk_target_drawdown"]
    assert dd.final_state == DispositionState.REJECTED
    with pytest.raises(PromotionRefusedError, match="disposition_not_promotable"):
        service.promote(dd.package_digest, **_promote_kwargs())
    with pytest.raises(FileNotFoundError):
        store.load_champion_pointer("risk_target_drawdown")


def test_failed_gate_head_is_refused_even_when_quarantined(tmp_path):
    # Defense in depth: even if an importer wrongly quarantined a FAIL-gate
    # head, the promotion service re-reads the pre-registered EvaluationReport
    # itself and refuses. A passing head elsewhere never rescues it.
    context = _manual_package(tmp_path, gate_status=GateStatus.FAIL)
    _drive_to_quarantined(context)
    with pytest.raises(PromotionRefusedError, match="ship_gate_failed"):
        context["service"].promote(context["package_digest"], **_promote_kwargs())
    states, _ = _ledger_states(
        context["store"], context["identities"], context["package_digest"]
    )
    assert states[-1] == DispositionState.QUARANTINED
    with pytest.raises(FileNotFoundError):
        context["store"].load_champion_pointer(context["package"].lane_id)


def test_missing_gate_results_refuse_fail_closed(tmp_path):
    # The package declares an evaluation report digest but carries no readable
    # signed report member — promotion must refuse, never assume PASS.
    context = _manual_package(tmp_path, include_report=False)
    _drive_to_quarantined(context)
    with pytest.raises(PromotionRefusedError, match="gate_results_missing"):
        context["service"].promote(context["package_digest"], **_promote_kwargs())


def test_passing_report_never_rescues_failing_report(tmp_path):
    # One package carrying TWO pre-registered reports (one PASS, one FAIL):
    # the ship gate is enforced on every declared report, so the passing one
    # can never rescue the failing one.
    context = _manual_package(tmp_path, extra_fail_report=True)
    _drive_to_quarantined(context)
    with pytest.raises(PromotionRefusedError, match="ship_gate_failed"):
        context["service"].promote(context["package_digest"], **_promote_kwargs())
    with pytest.raises(FileNotFoundError):
        context["store"].load_champion_pointer(context["package"].lane_id)


# --------------------------------------------------------------------------- #
# Digest + contract verification (steps 2 and 4)                               #
# --------------------------------------------------------------------------- #

def test_tampered_artifact_bytes_refuse(tmp_path):
    identities, store, result, service = _slice_store(tmp_path)
    vol = result.outcomes["risk_target_vol"]
    package = store.load_package(vol.package_digest)
    artifact = package.artifacts[0]
    on_disk = store.root / "packages" / vol.package_digest / artifact.relative_path
    mutated = bytearray(on_disk.read_bytes())
    mutated[0] ^= 0xFF
    on_disk.write_bytes(bytes(mutated))

    with pytest.raises(PromotionRefusedError, match="package_verification_failed"):
        service.promote(vol.package_digest, **_promote_kwargs())
    with pytest.raises(FileNotFoundError):
        store.load_champion_pointer("risk_target_vol")


def test_feature_pipeline_mismatch_refuses(tmp_path):
    identities, store, result, service = _slice_store(tmp_path)
    vol = result.outcomes["risk_target_vol"]
    before, _ = _ledger_states(store, identities, vol.package_digest)
    with pytest.raises(PromotionRefusedError, match="feature_pipeline_mismatch"):
        service.promote(
            vol.package_digest,
            **_promote_kwargs(expected_feature_pipeline_version="some-other-version"),
        )
    after, _ = _ledger_states(store, identities, vol.package_digest)
    assert after == before


def test_lane_mismatch_refuses(tmp_path):
    _, _, result, service = _slice_store(tmp_path)
    vol = result.outcomes["risk_target_vol"]
    with pytest.raises(PromotionRefusedError, match="lane_mismatch"):
        service.promote(
            vol.package_digest, **_promote_kwargs(expected_lane_id="some_other_lane")
        )


# --------------------------------------------------------------------------- #
# Transition-policy violations                                                 #
# --------------------------------------------------------------------------- #

def test_unregistered_operator_key_refuses_through_the_policy(tmp_path):
    identities, store, result, _ = _slice_store(tmp_path)
    vol = result.outcomes["risk_target_vol"]
    # A rogue service with signing keys the AuthorityRegistry never registered:
    # the store's transition policy must refuse the SHADOW append.
    rogue_operator = Ed25519Signer.generate()
    rogue_promotion = Ed25519Signer.generate()
    identities.trust_store.add(rogue_operator.trusted_key(valid_from=NOW - timedelta(days=1)))
    identities.trust_store.add(rogue_promotion.trusted_key(valid_from=NOW - timedelta(days=1)))
    rogue = PromotionService(
        store,
        operator_signer=rogue_operator,
        operator_actor_id="rogue-operator",
        promotion_signer=rogue_promotion,
        promotion_actor_id="rogue-promotion",
    )
    before, _ = _ledger_states(store, identities, vol.package_digest)
    with pytest.raises(PromotionRefusedError, match="promotion_mutation_failed"):
        rogue.promote(vol.package_digest, **_promote_kwargs())
    after, _ = _ledger_states(store, identities, vol.package_digest)
    assert after == before
    with pytest.raises(FileNotFoundError):
        store.load_champion_pointer("risk_target_vol")


# --------------------------------------------------------------------------- #
# Rollback (step 11)                                                           #
# --------------------------------------------------------------------------- #

def test_rollback_restores_prior_pointer_when_policy_permits(tmp_path):
    # Same-package artifact re-point: the prior champion package is still
    # CHAMPION, so the transition policy permits exact pointer restoration.
    context = _manual_package(tmp_path, artifact_count=2)
    _drive_to_quarantined(context)
    service = context["service"]
    store = context["store"]
    lane = context["package"].lane_id

    record_a = service.promote(
        context["package_digest"], **_promote_kwargs(artifact_id="model-a")
    )
    assert store.load_champion_pointer(lane).artifact_id == "model-a"
    record_b = service.promote(
        context["package_digest"],
        **_promote_kwargs(artifact_id="model-b", now=NOW + timedelta(seconds=16)),
    )
    assert store.load_champion_pointer(lane).artifact_id == "model-b"
    assert record_b.prior_pointer_digest == record_a.pointer_digest

    rollback = service.rollback(
        record_b,
        approval=_approval("model-b dry metrics regressed; restore model-a"),
        now=NOW + timedelta(seconds=20),
    )
    assert rollback.pointer_restored is True
    assert rollback.restored_pointer_digest == record_a.pointer_digest
    restored = store.load_champion_pointer(lane)
    assert restored.artifact_id == "model-a"
    assert restored.package_digest == context["package_digest"]


def test_rollback_across_retired_champion_fails_closed(tmp_path):
    # Cross-package rollback: the prior champion was RETIRED during promotion
    # and the transition policy has no RETIRED -> CHAMPION edge. Rollback must
    # NOT weaken the policy; it retires the promoted champion so the lane
    # serves no champion at all (consumers abstain).
    identities, store, result_a, service = _slice_store(tmp_path)
    package_a = result_a.outcomes["risk_target_vol"].package_digest
    service.promote(package_a, **_promote_kwargs())
    result_b = _rerun_slice(store, identities, score=0.06)
    package_b = result_b.outcomes["risk_target_vol"].package_digest
    record_b = service.promote(
        package_b, **_promote_kwargs(now=NOW + timedelta(seconds=18))
    )

    rollback = service.rollback(
        record_b,
        approval=_approval("forward monitoring breach on package B"),
        now=NOW + timedelta(seconds=24),
    )
    assert rollback.pointer_restored is False
    assert rollback.retired_package_digest == package_b
    assert "RETIRED->CHAMPION" in (rollback.structural_note or "")
    states_b, _ = _ledger_states(store, identities, package_b)
    assert states_b[-1] == DispositionState.RETIRED
    # Fail-closed: the stale pointer no longer resolves — no champion served.
    with pytest.raises(ChampionPointerError):
        store.load_champion_pointer("risk_target_vol")
    # The prior artifact remains preserved for a legitimate re-promotion.
    prior_copy = (
        tmp_path / "evidence" / "champion_artifacts" / "risk_target_vol" / package_a
    )
    assert any(path.is_file() for path in prior_copy.rglob("*"))


def test_rollback_refuses_when_pointer_moved_or_unapproved(tmp_path):
    context = _manual_package(tmp_path, artifact_count=2)
    _drive_to_quarantined(context)
    service = context["service"]
    record = service.promote(
        context["package_digest"], **_promote_kwargs(artifact_id="model-a")
    )
    with pytest.raises(PromotionRefusedError, match="operator_approval_missing"):
        service.rollback(record, approval=None, now=NOW + timedelta(seconds=20))
    tampered = PromotionRecord.from_dict(
        {**record.as_dict(), "pointer_digest": "f" * 64}
    )
    with pytest.raises(PromotionRefusedError, match="rollback_pointer_moved"):
        service.rollback(tampered, approval=_approval(), now=NOW + timedelta(seconds=20))
    with pytest.raises(PromotionRefusedError, match="rollback_nothing_to_restore"):
        service.rollback(record, approval=_approval(), now=NOW + timedelta(seconds=20))


# --------------------------------------------------------------------------- #
# Persistent promotion identities (CLI support)                                #
# --------------------------------------------------------------------------- #

def test_promotion_identities_persist_and_reload(tmp_path):
    key_dir = tmp_path / "keys"
    first = load_or_create_promotion_identities(key_dir, now=NOW)
    second = load_or_create_promotion_identities(key_dir, now=NOW + timedelta(days=3))
    assert first.operator.key_id == second.operator.key_id
    assert first.promotion.key_id == second.promotion.key_id
    assert first.operator.key_id != first.promotion.key_id
    metadata = json.loads((key_dir / "promotion_identities.json").read_text())
    assert set(metadata["roles"]) == {"operator", "promotion_service"}


# --------------------------------------------------------------------------- #
# Manual package construction (real contracts, real signing)                   #
# --------------------------------------------------------------------------- #

def _manual_package(
    tmp_path,
    *,
    gate_status: GateStatus = GateStatus.PASS,
    include_report: bool = True,
    artifact_count: int = 1,
    extra_fail_report: bool = False,
):
    """Build one real signed package + store, decoupled from the slice.

    Used to exercise the promotion service against ledger/report shapes the
    slice can never produce (FAIL-gate-yet-quarantined, missing report member,
    multi-artifact packages).
    """
    identities, operator, promotion = _identities_with_promotion_roles()
    store = build_evidence_store(tmp_path / "evidence", identities, clock_now=NOW)
    producer = identities.producer
    producer_id = identities.actors[AuthorityRole.PRODUCER]

    job = JobManifest(
        job_id="manual-job",
        git_commit="0" * 40,
        container_digest=sha256_bytes(b"manual-container"),
        dataset_manifest_digests=(sha256_bytes(b"manual-dataset"),),
        model_training_spec=ModelTrainingSpec(
            target="forward_volatility",
            features=("f1", "f2", "f3"),
            causal_lag="t+1",
            search_space={},
            metrics=("score",),
            promotion_criteria=("score<=0.1",),
        ),
        feature_pipeline_version=FPV,
        configuration_digest=sha256_bytes(b"manual-config"),
        random_seeds=(7,),
        assigned_unit="manual-head",
        resource_class=ResourceClass.CPU_SMALL,
        capability_profile_digest=sha256_bytes(b"manual-capability"),
        trial_budget=1,
        expected_outputs=("forward_volatility",),
        created_at=NOW,
    )
    job_envelope = producer.sign(job, created_at=NOW)
    passed = gate_status != GateStatus.FAIL
    report = EvaluationReport(
        report_id="manual-report",
        job_manifest_digest=job_envelope.payload_digest,
        evaluator_id=producer_id,
        independent=False,
        created_at=NOW,
        temporal_holdout="OOS bucket",
        purge_observations=20,
        embargo_observations=20,
        trial_count=1,
        effective_sample_size=120.0,
        metrics={"score": MetricValue(value=0.05)},
        gates=(
            GateResult(
                gate_id="vol_admissibility", status=gate_status,
                observed=0.05, threshold=0.1, reason="manual gate",
            ),
        ),
        incumbent_comparison={},
        passed=passed,
    )
    report_envelope = producer.sign(report, created_at=NOW)
    fail_report_envelope = None
    if extra_fail_report:
        fail_report = report.model_copy(
            update={
                "report_id": "manual-report-fail",
                "gates": (
                    GateResult(
                        gate_id="vol_stability", status=GateStatus.FAIL,
                        observed=0.9, threshold=0.1, reason="manual failing gate",
                    ),
                ),
                "passed": False,
            }
        )
        fail_report_envelope = producer.sign(fail_report, created_at=NOW)

    artifact_ids = ["model-a", "model-b"][:artifact_count]
    artifact_bytes = {
        artifact_id: _fitted_model_bytes(11 + index)
        for index, artifact_id in enumerate(artifact_ids)
    }
    artifacts = tuple(
        ArtifactRef(
            artifact_id=artifact_id,
            relative_path=f"artifacts/{artifact_id}.pkl",
            digest=sha256_bytes(data),
            size_bytes=len(data),
            media_type="application/octet-stream",
        )
        for artifact_id, data in artifact_bytes.items()
    )
    logs = [
        ArtifactRef(
            artifact_id="job-manifest",
            relative_path="manifests/job.json",
            digest=sha256_bytes(canonical_bytes(job_envelope)),
            size_bytes=len(canonical_bytes(job_envelope)),
            media_type="application/vnd.axiom.signed-envelope+json",
        )
    ]
    files = {"manifests/job.json": canonical_bytes(job_envelope)}
    if include_report:
        report_bytes = canonical_bytes(report_envelope)
        logs.append(
            ArtifactRef(
                artifact_id="evaluation-report",
                relative_path="evaluations/report.json",
                digest=sha256_bytes(report_bytes),
                size_bytes=len(report_bytes),
                media_type="application/vnd.axiom.signed-envelope+json",
            )
        )
        files["evaluations/report.json"] = report_bytes
    if fail_report_envelope is not None:
        fail_bytes = canonical_bytes(fail_report_envelope)
        logs.append(
            ArtifactRef(
                artifact_id="evaluation-report-fail",
                relative_path="evaluations/report_fail.json",
                digest=sha256_bytes(fail_bytes),
                size_bytes=len(fail_bytes),
                media_type="application/vnd.axiom.signed-envelope+json",
            )
        )
        files["evaluations/report_fail.json"] = fail_bytes
    for ref in artifacts:
        files[ref.relative_path] = artifact_bytes[ref.artifact_id]

    checksums = {ref.relative_path: ref.digest for ref in artifacts}
    checksums.update({ref.relative_path: ref.digest for ref in logs})
    package = EvidencePackage(
        package_id="manual-package",
        lane_id="risk_target_manual",
        producer_id=producer_id,
        created_at=NOW,
        job_manifest_digest=job_envelope.payload_digest,
        dataset_manifest_digests=(sha256_bytes(b"manual-dataset"),),
        evaluation_report_digests=(
            (report_envelope.payload_digest, fail_report_envelope.payload_digest)
            if fail_report_envelope is not None
            else (report_envelope.payload_digest,)
        ),
        artifacts=artifacts,
        logs=tuple(logs),
        safety_assertions=(SafetyAssertion(assertion_id="no-authority", passed=True),),
        checksums=checksums,
    )
    package_envelope = producer.sign(package, created_at=NOW)
    package_digest = store.write_package(package_envelope, files)
    return {
        "identities": identities,
        "store": store,
        "package": package,
        "package_digest": package_digest,
        "service": _service(store, operator, promotion),
    }


def _drive_to_quarantined(context) -> None:
    identities = context["identities"]
    store = context["store"]
    package_digest = context["package_digest"]
    importer = identities.importer
    importer_id = identities.actors[AuthorityRole.LOCAL_IMPORTER]
    verdict = build_import_verdict(
        verdict_id="manual-verdict",
        package_digest=package_digest,
        importer_id=importer_id,
        created_at=NOW,
        checks=(ImportCheck(check_id="hash-verified", passed=True),),
    )
    verdict_digest = store.write_verdict(importer.sign(verdict, created_at=NOW))

    steps = [
        (None, DispositionState.CREATED, AuthorityRole.PRODUCER),
        (DispositionState.CREATED, DispositionState.RECEIVED, AuthorityRole.LOCAL_IMPORTER),
        (DispositionState.RECEIVED, DispositionState.HASH_VERIFIED, AuthorityRole.LOCAL_IMPORTER),
        (DispositionState.HASH_VERIFIED, DispositionState.POLICY_VERIFIED, AuthorityRole.LOCAL_IMPORTER),
        (
            DispositionState.POLICY_VERIFIED,
            DispositionState.METRIC_REPLAYED,
            AuthorityRole.INDEPENDENT_VERIFIER,
        ),
        (DispositionState.METRIC_REPLAYED, DispositionState.QUARANTINED, AuthorityRole.LOCAL_IMPORTER),
    ]
    head = None
    for sequence, (from_state, to_state, role) in enumerate(steps):
        signer = identities.signers[role]
        metadata = (
            {"verdict_digest": verdict_digest}
            if to_state == DispositionState.QUARANTINED
            else {}
        )
        event = DispositionEvent(
            package_digest=package_digest,
            sequence=sequence,
            previous_event_digest=head,
            from_state=from_state,
            to_state=to_state,
            authority=role,
            actor_id=identities.actors[role],
            signer_key_id=signer.key_id,
            occurred_at=NOW + timedelta(seconds=sequence),
            reason=f"advance to {to_state.value}",
            metadata=metadata,
        )
        envelope = signer.sign(event, created_at=event.occurred_at)
        head = store.append_disposition(envelope, expected_head_digest=head).head_event_digest
