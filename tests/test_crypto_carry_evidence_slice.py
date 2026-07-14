from __future__ import annotations

import json
from datetime import date

import pytest

from src.crypto.carry_scorecard import (
    REQUIRED_TAIL_SCENARIOS,
    VERDICT_CLEARS,
    VERDICT_MISSING_COUNTERPARTY,
    VERDICT_MISSING_TAIL,
    aggregate_carry_scorecard,
    compute_cell_scorecard,
)
from src.evidence.contracts import AuthorityRole, DispositionState, GateStatus
from src.evidence.crypto_carry.dashboard import crypto_carry_evidence_view
from src.evidence.crypto_carry.evaluation import EvaluationParams, evaluate_partitions
from src.evidence.crypto_carry.manifests import cell_partition_id
from src.evidence.crypto_carry.models import lane_id_for_carry
from src.evidence.crypto_carry.slice import (
    build_evidence_store,
    build_slice_identities,
    produce_worker_output,
    run_crypto_carry_evidence_slice,
    utc,
)
from src.evidence.crypto_carry.worker import DatasetHashError, run_worker

NOW = utc(2026, 7, 14, 14)
CAMPAIGN = "carry-shadow-2026q3"
CARRY_ID = "spot-perp-delta-neutral"
VENUE_SET = "binance+coinbase"
LANE = lane_id_for_carry(CAMPAIGN, CARRY_ID)


def _jsonl(rows: list[dict]) -> bytes:
    return b"".join(
        json.dumps(row, sort_keys=True, separators=(",", ":")).encode() + b"\n"
        for row in rows
    )


def _tail(*, complete: bool) -> dict[str, dict]:
    evidence = {
        scenario: {
            "evidence_digest": (f"{index + 1:x}" * 64)[:64],
            "loss_fraction": 0.10 + index * 0.01,
            "source": f"stress-harness:{scenario}",
        }
        for index, scenario in enumerate(REQUIRED_TAIL_SCENARIOS)
    }
    if not complete:
        evidence.pop("venue_failure")
    return evidence


def _rows(
    cell_id: str,
    cost_model: str,
    regime: str,
    *,
    tail_complete: bool,
    counterparty_complete: bool = True,
) -> list[dict]:
    classifications = {
        "binance": {
            "classification": "tier_2",
            "evidence_digest": "a" * 64,
            "as_of": "2026-07-01",
        },
        "coinbase": {
            "classification": "tier_1",
            "evidence_digest": "b" * 64,
            "as_of": "2026-07-01",
        },
    }
    if not counterparty_complete:
        classifications.pop("coinbase")
    return [
        {
            "cell_id": cell_id,
            "carry_id": CARRY_ID,
            "venue_set": VENUE_SET,
            "cost_model": cost_model,
            "regime": regime,
            "period_index": period,
            "net_return": 0.002 + (0.001 if period % 2 else -0.001),
            "gross_funding_capture": 0.003,
            "two_leg_cost": 0.001,
            "spot_perp_mark_pnl": (0.001 if period % 2 else -0.001),
            "tracking_error": 0.01,
            "margin_utilization": 0.25,
            "capacity_usd": 100_000.0,
            "counterparty_classifications": classifications,
            "tail_evidence": _tail(complete=tail_complete),
        }
        for period in range(20)
    ]


def _partitions(*, tail_complete: bool, counterparty_complete: bool = True):
    output: dict[str, bytes] = {}
    for cost_model, regime in (
        ("base", "normal"),
        ("fees_2x", "high_vol"),
        ("borrow_stress", "depeg"),
    ):
        cell_id = cell_partition_id(CARRY_ID, VENUE_SET, cost_model, regime)
        output[cell_id] = _jsonl(
            _rows(
                cell_id,
                cost_model,
                regime,
                tail_complete=tail_complete,
                counterparty_complete=counterparty_complete,
            )
        )
    return output


def _params(parts):
    return EvaluationParams(expected_cells_by_carry={CARRY_ID: tuple(sorted(parts))})


def _fresh(tmp_path):
    identities = build_slice_identities(NOW)
    store = build_evidence_store(tmp_path / "evidence", identities, clock_now=NOW)
    return identities, store


def _kwargs():
    return dict(
        campaign_id=CAMPAIGN,
        dataset_id="carry-cells-2026q3",
        coverage_start=date(2026, 4, 1),
        coverage_end=date(2026, 7, 1),
        retrieved_at=NOW,
        created_at=NOW,
        git_commit="2" * 40,
        registry_publish=lambda *_: None,
    )


def _cards(parts):
    return [
        compute_cell_scorecard(
            cell_id, [json.loads(line) for line in data.decode().splitlines()]
        )
        for cell_id, data in sorted(parts.items())
    ]


def test_high_sharpe_cannot_pass_without_tail_evidence():
    parts = _partitions(tail_complete=False)
    head = evaluate_partitions(parts, _params(parts), campaign_id=CAMPAIGN)[0]
    assert head.metrics["minimum_cell_net_sharpe"] > 1.0
    assert head.verdict == VERDICT_MISSING_TAIL
    assert head.passed is False
    gates = {gate.gate_id: gate.status for gate in head.gates}
    assert gates["cost_adjusted_return_and_capacity"] == GateStatus.PASS
    assert gates["mandatory_tail_evidence"] == GateStatus.FAIL


def test_high_sharpe_cannot_pass_without_counterparty_evidence():
    parts = _partitions(tail_complete=True, counterparty_complete=False)
    head = evaluate_partitions(parts, _params(parts), campaign_id=CAMPAIGN)[0]
    assert head.verdict == VERDICT_MISSING_COUNTERPARTY
    assert head.passed is False


def test_complete_return_tail_and_counterparty_evidence_can_clear():
    parts = _partitions(tail_complete=True)
    head = evaluate_partitions(parts, _params(parts), campaign_id=CAMPAIGN)[0]
    assert head.verdict == VERDICT_CLEARS
    assert head.passed is True


def test_aggregator_refuses_missing_and_duplicate_cells():
    parts = _partitions(tail_complete=True)
    cards = _cards(parts)
    expected = sorted(parts)
    with pytest.raises(ValueError, match="cell set mismatch"):
        aggregate_carry_scorecard(CARRY_ID, cards[:-1], expected)
    with pytest.raises(ValueError, match="duplicate cells"):
        aggregate_carry_scorecard(CARRY_ID, cards + [cards[0]], expected)


def test_end_to_end_missing_tail_is_signed_and_rejected(tmp_path):
    identities, store = _fresh(tmp_path)
    parts = _partitions(tail_complete=False)
    result = run_crypto_carry_evidence_slice(store, identities, parts, **_kwargs())
    outcome = result.outcomes[LANE]
    assert outcome.final_state == DispositionState.REJECTED
    checks = {check.check_id: check.passed for check in outcome.verdict.checks}
    assert checks["metric_replay_reproduces"] is True
    assert checks["artifact_reproduces"] is True
    assert checks["candidate_evaluation_gate_passed"] is False


def test_end_to_end_complete_evidence_stops_at_quarantine(tmp_path):
    identities, store = _fresh(tmp_path)
    parts = _partitions(tail_complete=True)
    result = run_crypto_carry_evidence_slice(store, identities, parts, **_kwargs())
    assert result.outcomes[LANE].final_state == DispositionState.QUARANTINED
    assert crypto_carry_evidence_view(store.root)["champions"] == {}


def test_worker_refuses_changed_cell_bytes(tmp_path):
    identities, _ = _fresh(tmp_path)
    parts = _partitions(tail_complete=True)
    produced = produce_worker_output(identities, parts, **_kwargs())
    tampered = dict(parts)
    cell_id = sorted(tampered)[0]
    tampered[cell_id] += b"{}\n"
    with pytest.raises(DatasetHashError, match="declared hash"):
        run_worker(
            campaign_id=CAMPAIGN,
            job_envelope=produced.worker_output.job_envelope,
            dataset_manifest_envelope=produced.worker_output.dataset_manifest_envelope,
            strategy_manifest_envelope=produced.worker_output.strategy_manifest_envelope,
            capability_profile_envelope=produced.worker_output.capability_profile_envelope,
            partitions=tampered,
            producer=identities.producer,
            producer_id=identities.actors[AuthorityRole.PRODUCER],
            trust_store=identities.trust_store,
            created_at=NOW,
            registry_publish=lambda *_: None,
        )


def test_reader_groups_only_crypto_carry_lanes(tmp_path):
    identities, store = _fresh(tmp_path)
    run_crypto_carry_evidence_slice(
        store, identities, _partitions(tail_complete=False), **_kwargs()
    )
    view = crypto_carry_evidence_view(store.root)
    assert view["available"] is True
    assert view["carry_results"][CAMPAIGN][CARRY_ID]["state"] == "REJECTED"


def test_scorecard_is_deterministic():
    parts = _partitions(tail_complete=True)
    first = evaluate_partitions(parts, _params(parts), campaign_id=CAMPAIGN)[0]
    second = evaluate_partitions(parts, _params(parts), campaign_id=CAMPAIGN)[0]
    assert first.metrics == second.metrics
    assert first.artifact_bytes == second.artifact_bytes


def test_boolean_tail_claims_are_not_evidence_records():
    parts = _partitions(tail_complete=True)
    rewritten: dict[str, bytes] = {}
    for cell_id, data in parts.items():
        rows = [json.loads(line) for line in data.decode().splitlines()]
        for row in rows:
            row["tail_evidence"] = {
                scenario: True for scenario in REQUIRED_TAIL_SCENARIOS
            }
        rewritten[cell_id] = _jsonl(rows)
    head = evaluate_partitions(rewritten, _params(rewritten), campaign_id=CAMPAIGN)[0]
    assert head.verdict == VERDICT_MISSING_TAIL


def test_net_return_must_reconcile_to_gross_mark_and_two_leg_cost():
    parts = _partitions(tail_complete=True)
    cell_id = sorted(parts)[0]
    rows = [json.loads(line) for line in parts[cell_id].decode().splitlines()]
    rows[0]["net_return"] += 0.01
    with pytest.raises(ValueError, match="net_return must equal"):
        compute_cell_scorecard(cell_id, rows)
