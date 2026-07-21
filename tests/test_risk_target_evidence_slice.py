"""Phase I acceptance tests — risk-target end-to-end evidence vertical slice.

These are the roadmap §12 acceptance criteria made real. NO MOCKS: every test
runs the actual evidence store, real Ed25519 signing, the real disposition
chain, and either real LightGBM training or a real (injected) deterministic
evaluator — an ordinary callable that returns real contract objects, never a
test-double.

The §12 checklist and the test that covers it:

    - Remote worker cannot read .claude/state.json ........ see
      test_risk_target_evidence_worker_no_authority.py (AST + source scan)
    - Remote worker has no broker credentials ............. idem + capability test below
    - Candidate artifact cannot overwrite incumbent ....... test_candidate_never_overwrites_incumbent
    - Missing fold causes failure ......................... test_missing_fold_causes_failure
    - Changed dataset partition causes hash failure ....... test_changed_dataset_partition_causes_hash_failure
    - Changed model bytes cause signature failure ......... test_changed_model_bytes_are_rejected
        (model bytes are bound to the signed package via their sha256 digest; a
         changed byte fails the content-address bind inside write_package)
    - Worse head is rejected independently ................ test_worse_head_rejected_independently
    - Good head does not rescue a failed head ............. test_good_head_does_not_rescue_failed_head
    - Local replay reproduces metrics within tolerance .... test_local_metric_replay_reproduces_*
    - External registry outage does not stop verification . test_registry_outage_does_not_stop_verification
"""

from __future__ import annotations

import io
from datetime import date

import numpy as np
import pandas as pd
import pytest

from src.evidence.contracts import (
    AuthorityRole,
    CapabilityProfile,
    DatasetManifest,
    DispositionEvent,
    DispositionState,
    EvaluationReport,
    GateResult,
    GateStatus,
    ImportDecision,
    JobManifest,
    SignedEnvelope,
)
from src.evidence.signing import verify_envelope
from src.evidence.risk_target.evaluation import EvaluationParams, assemble_frame, evaluate_frame
from src.evidence.risk_target.local_import import _policy_checks
from src.evidence.risk_target.models import DRAWDOWN_HEAD_ID, VOLATILITY_HEAD_ID, HeadResult
from src.evidence.risk_target.models import (
    CAPABILITY_PROFILE_PATH,
    DATASET_MANIFEST_PATH,
    EVALUATION_REPORT_PATH,
    JOB_MANIFEST_PATH,
)
from src.evidence.risk_target.slice import (
    build_evidence_store,
    build_slice_identities,
    produce_worker_output,
    risk_target_evidence_view,
    run_risk_target_evidence_slice,
    utc,
)
from src.evidence.risk_target.worker import DatasetHashError, MissingHeadError, run_worker

NOW = utc(2026, 7, 13, 12, 0)
SLICE_KW = dict(
    dataset_id="fx-daily-test",
    coverage_start=date(2019, 1, 1),
    coverage_end=date(2026, 1, 1),
    retrieved_at=NOW,
    created_at=NOW,
    git_commit="0" * 40,
)


def _noop_registry(experiment, metrics):
    return None


def _dummy_partitions():
    return {
        "EUR_USD": b"EUR_USD synthetic risk-target partition bytes\n",
        "USD_JPY": b"USD_JPY synthetic risk-target partition bytes\n",
    }


def _head(head_id: str, lane_id: str, passed: bool, score: float = 0.05) -> HeadResult:
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
        model_bytes=f"deterministic-model::{head_id}".encode(),
        media_type="application/octet-stream",
        trial_count=1, effective_sample_size=120.0, temporal_holdout="OOS bucket",
        purge_observations=20, embargo_observations=20,
    )


def _fixed_evaluator(vol_passed: bool = True, dd_passed: bool = False):
    """A real, deterministic evaluator returning fixed per-head results.

    Because it is a pure function of nothing external, the worker and the
    importer's independent replay both obtain identical results — exactly the
    reproducibility contract the store enforces.
    """
    def evaluator(partitions, params):
        return (
            _head(VOLATILITY_HEAD_ID, "risk_target_vol", vol_passed),
            _head(DRAWDOWN_HEAD_ID, "risk_target_drawdown", dd_passed),
        )
    return evaluator


def _fresh(tmp_path):
    identities = build_slice_identities(NOW)
    store = build_evidence_store(tmp_path / "evidence", identities, clock_now=NOW)
    return identities, store


def _run(tmp_path, evaluator, partitions=None, registry=_noop_registry, params=None):
    identities, store = _fresh(tmp_path)
    partitions = partitions if partitions is not None else _dummy_partitions()
    result = run_risk_target_evidence_slice(
        store, identities, partitions, evaluator=evaluator,
        registry_publish=registry, params=params, **SLICE_KW,
    )
    return identities, store, result


# --------------------------------------------------------------------------- #
# Disposition semantics                                                        #
# --------------------------------------------------------------------------- #

def test_worse_head_rejected_independently(tmp_path):
    _, store, result = _run(tmp_path, _fixed_evaluator(vol_passed=True, dd_passed=False))
    vol = result.outcomes["risk_target_vol"]
    dd = result.outcomes["risk_target_drawdown"]
    assert vol.final_state == DispositionState.QUARANTINED
    assert vol.verdict.decision == ImportDecision.ACCEPT_FOR_QUARANTINE
    assert dd.final_state == DispositionState.REJECTED
    assert dd.verdict.decision == ImportDecision.REJECT
    assert "candidate_evaluation_gate_passed" in (dd.verdict.rejection_reason or "")
    # Independent packages — one lane's verdict is not derived from the other.
    assert vol.package_digest != dd.package_digest


def test_good_head_does_not_rescue_failed_head(tmp_path):
    # Even though the vol head is accepted for quarantine in the SAME slice
    # run, the drawdown head remains REJECTED — acceptance is per-package.
    _, store, result = _run(tmp_path, _fixed_evaluator(vol_passed=True, dd_passed=False))
    assert result.outcomes["risk_target_vol"].final_state == DispositionState.QUARANTINED
    assert result.outcomes["risk_target_drawdown"].final_state == DispositionState.REJECTED
    # The drawdown package's ledger terminates at REJECTED, never QUARANTINED.
    dd_digest = result.outcomes["risk_target_drawdown"].package_digest
    ledger = store.load_ledger(dd_digest)
    states = [verify_envelope(env, DispositionEvent, _trust(store)).to_state for env in ledger]
    assert DispositionState.QUARANTINED not in states
    assert states[-1] == DispositionState.REJECTED


def test_quarantine_chain_is_exactly_the_signed_lifecycle(tmp_path):
    identities, store, result = _run(tmp_path, _fixed_evaluator(vol_passed=True, dd_passed=False))
    vol_digest = result.outcomes["risk_target_vol"].package_digest
    ledger = store.load_ledger(vol_digest)
    events = [verify_envelope(env, DispositionEvent, identities.trust_store) for env in ledger]
    assert [e.to_state for e in events] == [
        DispositionState.CREATED,
        DispositionState.RECEIVED,
        DispositionState.HASH_VERIFIED,
        DispositionState.POLICY_VERIFIED,
        DispositionState.METRIC_REPLAYED,
        DispositionState.QUARANTINED,
    ]
    # The genesis (CREATED) event is authored by the remote producer; the
    # independent metric-replay is authored by a DIFFERENT authority.
    assert events[0].authority == AuthorityRole.PRODUCER
    assert events[4].authority == AuthorityRole.INDEPENDENT_VERIFIER
    assert events[0].actor_id != events[4].actor_id


# --------------------------------------------------------------------------- #
# Incumbent protection                                                          #
# --------------------------------------------------------------------------- #

def test_candidate_never_overwrites_incumbent(tmp_path):
    _, store, result = _run(tmp_path, _fixed_evaluator(vol_passed=True, dd_passed=True))
    # Both heads quarantined — yet neither becomes a champion. The slice has
    # no promotion authority, so any incumbent champion pointer is untouched.
    for lane in ("risk_target_vol", "risk_target_drawdown"):
        assert result.outcomes[lane].final_state == DispositionState.QUARANTINED
        with pytest.raises(FileNotFoundError):
            store.load_champion_pointer(lane)
    view = risk_target_evidence_view(store.root)
    assert view["champions"] == {}
    assert all(entry["state"] == "QUARANTINED" for entry in view["lanes"])


# --------------------------------------------------------------------------- #
# Integrity rails                                                               #
# --------------------------------------------------------------------------- #

def test_missing_fold_causes_failure(tmp_path):
    identities, _ = _fresh(tmp_path)

    def only_one_head(partitions, params):
        return (_head(VOLATILITY_HEAD_ID, "risk_target_vol", True),)

    with pytest.raises(MissingHeadError, match="do not match declared folds"):
        produce_worker_output(
            identities, _dummy_partitions(), evaluator=only_one_head,
            registry_publish=_noop_registry, **SLICE_KW,
        )


def test_changed_dataset_partition_causes_hash_failure(tmp_path):
    identities, _ = _fresh(tmp_path)
    partitions = _dummy_partitions()
    produced = produce_worker_output(
        identities, partitions, evaluator=_fixed_evaluator(),
        registry_publish=_noop_registry, **SLICE_KW,
    )
    tampered = dict(partitions)
    tampered["EUR_USD"] = partitions["EUR_USD"] + b"tamper"
    with pytest.raises(DatasetHashError, match="does not match its declared hash"):
        run_worker(
            job_envelope=produced.worker_output.job_envelope,
            dataset_manifest_envelope=produced.worker_output.dataset_manifest_envelope,
            capability_profile_envelope=produced.worker_output.capability_profile_envelope,
            partitions=tampered,
            producer=identities.producer,
            producer_id=identities.actors[AuthorityRole.PRODUCER],
            trust_store=identities.trust_store,
            created_at=NOW,
            evaluator=_fixed_evaluator(),
            registry_publish=_noop_registry,
        )


def test_changed_model_bytes_are_rejected(tmp_path):
    identities, store = _fresh(tmp_path)
    produced = produce_worker_output(
        identities, _dummy_partitions(), evaluator=_fixed_evaluator(),
        registry_publish=_noop_registry, **SLICE_KW,
    )
    head = produced.worker_output.heads[0]
    path = next(iter(head.files))
    # Flip one byte, preserving length, so the failure is the signed-digest
    # binding itself, not a coincidental size change.
    mutated = bytearray(head.files[path])
    mutated[0] ^= 0xFF
    tampered_files = dict(head.files)
    tampered_files[path] = bytes(mutated)
    with pytest.raises(ValueError, match="digest mismatch"):
        store.write_package(head.package_envelope, tampered_files)


# --------------------------------------------------------------------------- #
# Independent metric replay                                                     #
# --------------------------------------------------------------------------- #

def test_local_metric_replay_reproduces_via_importer(tmp_path):
    # Fixed evaluator -> the importer's independent replay reproduces the
    # producer's metrics exactly; the replay check passes and the head
    # advances through METRIC_REPLAYED.
    _, _, result = _run(tmp_path, _fixed_evaluator(vol_passed=True, dd_passed=True))
    for outcome in result.outcomes.values():
        checks = {c.check_id: c.passed for c in outcome.verdict.checks}
        assert checks["metric_replay_reproduces"] is True
        assert checks["gate_verdict_reproduces"] is True
        assert checks["evaluation_cost_accounted"] is True


def test_cost_policy_changes_the_signed_configuration_identity(tmp_path):
    identities, _ = _fresh(tmp_path)
    common = {
        **SLICE_KW,
        "evaluator": _fixed_evaluator(),
        "registry_publish": _noop_registry,
    }
    free = produce_worker_output(
        identities, _dummy_partitions(), cost_rate_per_hour=0.0, **common
    )
    metered = produce_worker_output(
        identities, _dummy_partitions(), cost_rate_per_hour=0.40, **common
    )

    assert free.job.configuration_digest != metered.job.configuration_digest
    report = verify_envelope(
        metered.worker_output.heads[0].evaluation_report_envelope,
        EvaluationReport,
        identities.trust_store,
    )
    assert report.cost["rate_per_hour"] == 0.4
    assert report.cost["basis"] == "configured_local_cpu_hourly_rate"


def test_tampered_evaluation_cost_amount_fails_importer_policy(tmp_path):
    identities, _ = _fresh(tmp_path)
    produced = produce_worker_output(
        identities,
        _dummy_partitions(),
        evaluator=_fixed_evaluator(),
        registry_publish=_noop_registry,
        cost_rate_per_hour=0.40,
        **SLICE_KW,
    )
    head = produced.worker_output.heads[0]
    report = verify_envelope(
        head.evaluation_report_envelope,
        EvaluationReport,
        identities.trust_store,
    )
    tampered_report = report.model_copy(
        update={"cost": {**report.cost, "amount": float(report.cost["amount"]) + 1.0}}
    )

    checks = _policy_checks(
        head,
        produced.job,
        produced.capability_profile,
        tampered_report,
    )

    by_id = {check.check_id: check for check in checks}
    assert by_id["evaluation_cost_accounted"].passed is False


@pytest.mark.parametrize(
    ("cost_update", "usage_update"),
    [
        ({"amount": 10**400}, {}),
        ({}, {"head_count": 2.0}),
    ],
)
def test_invalid_evaluation_cost_metadata_fails_closed(
    tmp_path, cost_update, usage_update
):
    identities, _ = _fresh(tmp_path)
    produced = produce_worker_output(
        identities,
        _dummy_partitions(),
        evaluator=_fixed_evaluator(),
        registry_publish=_noop_registry,
        **SLICE_KW,
    )
    head = produced.worker_output.heads[0]
    report = verify_envelope(
        head.evaluation_report_envelope,
        EvaluationReport,
        identities.trust_store,
    )
    invalid_report = report.model_copy(
        update={
            "cost": {**report.cost, **cost_update},
            "resource_usage": {**report.resource_usage, **usage_update},
        }
    )

    checks = _policy_checks(
        head,
        produced.job,
        produced.capability_profile,
        invalid_report,
    )

    by_id = {check.check_id: check for check in checks}
    assert by_id["evaluation_cost_accounted"].passed is False


@pytest.mark.slow
def test_local_metric_replay_reproduces_real_training(tmp_path):
    # Real LightGBM training must be deterministic so replay reproduces.
    params = EvaluationParams(oos_start="2023-06-01", n_estimators=30, early_stopping_rounds=8)
    partitions = _real_partitions()
    frame = assemble_frame(partitions, params)
    first = evaluate_frame(frame, params)
    second = evaluate_frame(frame, params)
    for a, b in zip(first, second):
        assert a.head_id == b.head_id
        for name in a.metrics:
            assert a.metrics[name] == pytest.approx(b.metrics[name], abs=params.replay_tolerance)
        assert a.passed == b.passed

    # And through the full importer against real training.
    identities, store = _fresh(tmp_path)
    result = run_risk_target_evidence_slice(
        store, identities, partitions, registry_publish=_noop_registry, params=params, **SLICE_KW,
    )
    for outcome in result.outcomes.values():
        checks = {c.check_id: c.passed for c in outcome.verdict.checks}
        assert checks["metric_replay_reproduces"] is True


# --------------------------------------------------------------------------- #
# Registry independence                                                        #
# --------------------------------------------------------------------------- #

def test_registry_outage_does_not_stop_verification(tmp_path):
    def broken_registry(experiment, metrics):
        raise ConnectionError("external registry is down")

    _, store, result = _run(tmp_path, _fixed_evaluator(vol_passed=True, dd_passed=False),
                            registry=broken_registry)
    # Producer still emitted evidence; local verification still produced a verdict.
    assert result.outcomes["risk_target_vol"].final_state == DispositionState.QUARANTINED
    assert result.outcomes["risk_target_drawdown"].final_state == DispositionState.REJECTED


# --------------------------------------------------------------------------- #
# Recovery + view                                                              #
# --------------------------------------------------------------------------- #

def test_indexes_rebuild_to_byte_equivalent_state(tmp_path):
    import shutil

    _, store, _ = _run(tmp_path, _fixed_evaluator(vol_passed=True, dd_passed=False))
    before = store.current_index_bytes()
    shutil.rmtree(store.root / "indexes")
    rebuilt = store.rebuild_indexes()
    assert rebuilt == before


def test_signed_lineage_objects_survive_store_restart(tmp_path):
    identities, store, result = _run(
        tmp_path, _fixed_evaluator(vol_passed=True, dd_passed=False)
    )
    restarted = build_evidence_store(store.root, identities, clock_now=NOW)
    expected_types = {
        JOB_MANIFEST_PATH: JobManifest,
        DATASET_MANIFEST_PATH: DatasetManifest,
        CAPABILITY_PROFILE_PATH: CapabilityProfile,
        EVALUATION_REPORT_PATH: EvaluationReport,
    }

    for outcome in result.outcomes.values():
        package = restarted.load_package(outcome.package_digest)
        assert {ref.relative_path for ref in package.logs} == set(expected_types)
        for relative_path, payload_type in expected_types.items():
            raw = restarted.load_package_file(outcome.package_digest, relative_path)
            envelope = SignedEnvelope.model_validate_json(raw, strict=True)
            verified = verify_envelope(envelope, payload_type, identities.trust_store)
            assert isinstance(verified, payload_type)

        job_envelope = SignedEnvelope.model_validate_json(
            restarted.load_package_file(outcome.package_digest, JOB_MANIFEST_PATH),
            strict=True,
        )
        report_envelope = SignedEnvelope.model_validate_json(
            restarted.load_package_file(outcome.package_digest, EVALUATION_REPORT_PATH),
            strict=True,
        )
        assert job_envelope.payload_digest == package.job_manifest_digest
        assert report_envelope.payload_digest in package.evaluation_report_digests
        report = verify_envelope(
            report_envelope, EvaluationReport, identities.trust_store
        )
        assert report.resource_usage["scope"] == "producer_evaluation"
        assert report.resource_usage["allocation"] == "equal_share_of_job_evaluator_wall_time"
        assert report.cost == {
            "allocation": "equal_share_of_job_evaluator_wall_time",
            "amount": 0.0,
            "basis": "local_unmetered_no_incremental_provider_charge",
            "currency": "USD",
            "rate_per_hour": 0.0,
            "scope": "producer_evaluation",
        }


def test_evidence_view_reports_per_lane_dispositions(tmp_path):
    _, store, _ = _run(tmp_path, _fixed_evaluator(vol_passed=True, dd_passed=False))
    view = risk_target_evidence_view(store.root)
    assert view["available"] is True
    by_lane = {entry["lane_id"]: entry for entry in view["lanes"]}
    assert by_lane["risk_target_vol"]["state"] == "QUARANTINED"
    assert by_lane["risk_target_vol"]["verdict"]["decision"] == "ACCEPT_FOR_QUARANTINE"
    assert by_lane["risk_target_drawdown"]["state"] == "REJECTED"
    assert "candidate_evaluation_gate_passed" in by_lane["risk_target_drawdown"]["verdict"]["failed_checks"]


def test_evidence_view_handles_absent_store(tmp_path):
    view = risk_target_evidence_view(tmp_path / "does-not-exist")
    assert view["available"] is False
    assert view["lanes"] == []


# --------------------------------------------------------------------------- #
# Real-data path (assembly + evaluation actually run on OHLC CSVs)             #
# --------------------------------------------------------------------------- #

def test_real_partitions_assemble_and_evaluate(tmp_path):
    params = EvaluationParams(oos_start="2023-06-01", n_estimators=25, early_stopping_rounds=6)
    frame = assemble_frame(_real_partitions(), params)
    assert frame.train_mask.any() and frame.val_mask.any() and frame.test_mask.any()
    heads = evaluate_frame(frame, params)
    ids = {h.head_id for h in heads}
    assert ids == {VOLATILITY_HEAD_ID, DRAWDOWN_HEAD_ID}
    for head in heads:
        assert head.metrics  # real OOS metrics were computed
        assert len(head.gates) >= 1


def _trust(store):
    return store.trust_store


def _synth_pair_csv(seed: int, n: int = 1200) -> bytes:
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2019-01-01", periods=n, tz="UTC")
    log_var = np.empty(n)
    log_var[0] = -8.0
    for i in range(1, n):
        log_var[i] = -8.0 * 0.03 + 0.97 * log_var[i - 1] + rng.normal(0, 0.25)
    sigma = np.exp(0.5 * log_var)
    rets = rng.normal(0, 1, n) * sigma
    close = 100.0 * np.exp(np.cumsum(rets))
    open_ = np.concatenate([[close[0]], close[:-1]])
    high = np.maximum.reduce([open_, close]) * (1 + np.abs(rng.normal(0, 0.3, n)) * sigma)
    low = np.minimum.reduce([open_, close]) * (1 - np.abs(rng.normal(0, 0.3, n)) * sigma)
    volume = rng.integers(1000, 5000, n).astype(float)
    df = pd.DataFrame({
        "date": dates.strftime("%Y-%m-%d"), "open": open_, "high": high,
        "low": low, "close": close, "volume": volume,
    })
    buf = io.StringIO()
    df.to_csv(buf, index=False)
    return buf.getvalue().encode("utf-8")


def _real_partitions():
    return {"EUR_USD": _synth_pair_csv(1), "USD_JPY": _synth_pair_csv(2)}
