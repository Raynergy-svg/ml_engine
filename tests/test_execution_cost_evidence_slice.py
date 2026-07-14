from __future__ import annotations

import json
from datetime import date, timedelta

import pytest

from src.evidence.contracts import DispositionState, GateStatus
from src.evidence.execution_cost.dashboard import execution_cost_evidence_view
from src.evidence.execution_cost.evaluation import EvaluationParams, evaluate_partitions
from src.evidence.execution_cost.manifests import build_training_spec
from src.evidence.execution_cost.slice import (
    build_evidence_store, build_slice_identities, produce_worker_output,
    run_execution_cost_evidence_slice, utc,
)
from src.evidence.execution_cost.worker import DatasetHashError, run_worker

NOW = utc(2026, 7, 14, 15)
PARAMS = EvaluationParams(oos_start="2026-01-01T00:00:00+00:00")


def _partitions(*, shock: bool = False) -> dict[str, bytes]:
    rows = []
    start = utc(2025, 11, 1)
    for index in range(45):
        decision = start + timedelta(days=index * 2)
        spread = 0.5 + (index % 5) * 0.2
        volatility = 0.2 + (index % 7) * 0.1
        participation = 0.01 + (index % 4) * 0.02
        notional = 10_000.0 + index * 2_000.0
        imbalance = ((index % 5) - 2) / 5
        cost = 0.25 + 0.8 * spread + 0.5 * volatility + 8.0 * participation + 0.3 * abs(imbalance)
        if shock and decision.year == 2026:
            cost += 10.0 + (index % 3) * 5.0
        rows.append({
            "order_id": f"order-{index:03d}", "instrument": "EUR_USD",
            "feature_as_of": (decision - timedelta(seconds=1)).isoformat(),
            "decision_at": decision.isoformat(),
            "fill_at": (decision + timedelta(seconds=1)).isoformat(),
            "spread_bps": spread, "volatility_1m": volatility,
            "participation_rate": participation, "order_notional_usd": notional,
            "book_imbalance": imbalance, "realized_cost_bps": cost,
        })
    data = b"".join(json.dumps(row, sort_keys=True, separators=(",", ":")).encode() + b"\n" for row in rows)
    return {"fx-fills-2025q4-2026q1": data}


def _kwargs():
    return dict(
        dataset_id="forward-fills-2026q1", coverage_start=date(2025, 11, 1),
        coverage_end=date(2026, 1, 28), retrieved_at=NOW, created_at=NOW,
        git_commit="4" * 40, params=PARAMS, registry_publish=lambda *_: None,
    )


def _fresh(tmp_path):
    identities = build_slice_identities(NOW)
    store = build_evidence_store(tmp_path / "evidence", identities, clock_now=NOW)
    return identities, store


def test_model_training_spec_freezes_cost_contract():
    spec = build_training_spec()
    assert spec.target == "realized_execution_cost_bps"
    assert "spread_bps" in spec.features
    assert "decision_at < fill_at" in spec.causal_lag
    assert tuple(spec.search_space["model_family"]) == ("ridge_linear",)
    assert "implementation shortfall" in spec.search_space["cost_model"]
    with pytest.raises(ValueError, match="exactly one deterministic trial"):
        build_training_spec(trial_budget=2)


def test_deterministic_model_beats_naive_and_artifact_reproduces():
    first = evaluate_partitions(_partitions(), PARAMS)
    second = evaluate_partitions(_partitions(), PARAMS)
    assert first.artifact_bytes == second.artifact_bytes
    assert first.metrics == second.metrics
    assert first.passed is True
    assert first.metrics["oos_mae_bps"] < first.metrics["oos_baseline_mae_bps"]


def test_tail_cost_shock_is_preserved_as_honest_negative():
    result = evaluate_partitions(_partitions(shock=True), PARAMS)
    assert result.passed is False
    statuses = {gate.gate_id: gate.status for gate in result.gates}
    assert statuses["tail_underprediction_bounded"] == GateStatus.FAIL


def test_evaluator_refuses_future_features_and_duplicate_orders():
    parts = _partitions()
    rows = [json.loads(line) for line in next(iter(parts.values())).decode().splitlines()]
    rows[0]["feature_as_of"] = rows[0]["fill_at"]
    bad = {"bad": b"".join(json.dumps(row).encode() + b"\n" for row in rows)}
    with pytest.raises(ValueError, match="causal"):
        evaluate_partitions(bad, PARAMS)
    duplicate = dict(parts)
    duplicate["copy"] = next(iter(parts.values()))
    with pytest.raises(ValueError, match="duplicate order_id"):
        evaluate_partitions(duplicate, PARAMS)


def test_end_to_end_clear_model_stops_at_quarantine(tmp_path):
    identities, store = _fresh(tmp_path)
    result = run_execution_cost_evidence_slice(store, identities, _partitions(), **_kwargs())
    assert result.outcome.final_state == DispositionState.QUARANTINED
    checks = {check.check_id: check.passed for check in result.outcome.verdict.checks}
    assert checks["metric_replay_reproduces"] is True
    assert checks["artifact_reproduces"] is True
    view = execution_cost_evidence_view(store.root)
    assert view["lanes"][0]["state"] == "QUARANTINED"
    assert view["champions"] == {}


def test_end_to_end_negative_is_signed_and_rejected(tmp_path):
    identities, store = _fresh(tmp_path)
    result = run_execution_cost_evidence_slice(store, identities, _partitions(shock=True), **_kwargs())
    assert result.outcome.final_state == DispositionState.REJECTED
    checks = {check.check_id: check.passed for check in result.outcome.verdict.checks}
    assert checks["metric_replay_reproduces"] is True
    assert checks["candidate_evaluation_gate_passed"] is False


def test_worker_refuses_changed_partition_bytes(tmp_path):
    identities, _ = _fresh(tmp_path)
    parts = _partitions()
    produced = produce_worker_output(identities, parts, **_kwargs()).worker_output
    changed = dict(parts)
    changed[next(iter(changed))] += b"{}\n"
    with pytest.raises(DatasetHashError, match="declared hash"):
        run_worker(
            job_envelope=produced.job_envelope,
            dataset_manifest_envelope=produced.dataset_manifest_envelope,
            capability_profile_envelope=produced.capability_profile_envelope,
            partitions=changed,
            producer=identities.signers[next(role for role in identities.signers if role.value == "producer")],
            producer_id="execution-cost-remote-worker", trust_store=identities.trust_store,
            created_at=NOW, params=PARAMS,
        )


def test_registry_outage_is_not_load_bearing(tmp_path):
    identities, _ = _fresh(tmp_path)
    produced = produce_worker_output(
        identities, _partitions(), **{**_kwargs(), "registry_publish": lambda *_: (_ for _ in ()).throw(RuntimeError("offline"))}
    )
    assert produced.worker_output.model.package.lane_id == "execution_cost_model"


def test_dashboard_reader_fails_closed_on_structurally_invalid_json(tmp_path):
    root = tmp_path / "evidence"
    (root / "indexes").mkdir(parents=True)
    (root / "indexes" / "current.json").write_text('[]')
    view = execution_cost_evidence_view(root)
    assert view["available"] is False
    assert "invalid structure" in view["reason"]
