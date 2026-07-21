from __future__ import annotations

import json
from datetime import date

import pytest

from src.crypto.momentum_scorecard import (
    VERDICT_SHADOW_INSUFFICIENT_SIGNIFICANCE,
    aggregate_construction_scorecard,
    compute_fold_scorecard,
)
from src.evidence.contracts import AuthorityRole, DispositionState, GateStatus
from src.evidence.crypto_momentum.dashboard import crypto_momentum_evidence_view
from src.evidence.crypto_momentum.evaluation import EvaluationParams, evaluate_partitions
from src.evidence.crypto_momentum.manifests import cell_partition_id
from src.evidence.crypto_momentum.models import lane_id_for_construction
from src.evidence.crypto_momentum.slice import (
    build_evidence_store,
    build_slice_identities,
    produce_worker_output,
    run_crypto_momentum_evidence_slice,
    utc,
)
from src.evidence.crypto_momentum.worker import DatasetHashError, run_worker

NOW = utc(2026, 7, 14, 12)
CAMPAIGN = "frozen-xs-v1"
CONSTRUCTION = "mom14_q5_weekly"
LANE = lane_id_for_construction(CAMPAIGN, CONSTRUCTION)


def _jsonl(rows: list[dict]) -> bytes:
    return b"".join(
        json.dumps(row, sort_keys=True, separators=(",", ":")).encode() + b"\n"
        for row in rows
    )


def _rows(cell_id: str, fold: int, stress: str, mean: float) -> list[dict]:
    # Alternating dispersion keeps Sharpe defined; the small/unstable means make
    # the baseline result positive but honestly insignificant across folds.
    return [
        {
            "fold_id": cell_id,
            "construction": CONSTRUCTION,
            "stress": stress,
            "fold_index": fold,
            "in_sample": False,
            "period_index": index,
            "net_return": mean + (0.01 if index % 2 else -0.01),
            "turnover": 0.2,
            "funding_return": -0.00002,
        }
        for index in range(20)
    ]


def _partitions() -> dict[str, bytes]:
    output: dict[str, bytes] = {}
    means = (0.0010, 0.0004, -0.0002)
    for stress in ("baseline", "funding_spike"):
        for fold, mean in enumerate(means):
            cell_id = cell_partition_id(CONSTRUCTION, stress, fold)
            stressed = mean - 0.0003 if stress != "baseline" else mean
            output[cell_id] = _jsonl(_rows(cell_id, fold, stress, stressed))
    return output


def _params(parts: dict[str, bytes]) -> EvaluationParams:
    return EvaluationParams(
        expected_cells_by_construction={CONSTRUCTION: tuple(sorted(parts))}
    )


def _fresh(tmp_path):
    identities = build_slice_identities(NOW)
    store = build_evidence_store(tmp_path / "evidence", identities, clock_now=NOW)
    return identities, store


def _slice_kwargs():
    return dict(
        campaign_id=CAMPAIGN,
        dataset_id="crypto-momentum-forward-2026-07",
        coverage_start=date(2026, 4, 1),
        coverage_end=date(2026, 7, 1),
        retrieved_at=NOW,
        created_at=NOW,
        git_commit="1" * 40,
        registry_publish=lambda *_: None,
    )


def test_complete_insignificant_result_is_preserved_as_honest_negative():
    parts = _partitions()
    head = evaluate_partitions(parts, _params(parts), campaign_id=CAMPAIGN)[0]
    assert head.verdict == VERDICT_SHADOW_INSUFFICIENT_SIGNIFICANCE
    assert head.passed is False
    assert {gate.gate_id: gate.status for gate in head.gates}["significance_clears"] == GateStatus.FAIL
    assert head.artifact_bytes


def test_aggregator_refuses_missing_and_duplicate_cells():
    parts = _partitions()
    cards = [
        compute_fold_scorecard(cell_id, json.loads("[" + data.decode().replace("\n", ",").rstrip(",") + "]"))
        for cell_id, data in sorted(parts.items())
    ]
    expected = sorted(parts)
    with pytest.raises(ValueError, match="fold set mismatch"):
        aggregate_construction_scorecard(CONSTRUCTION, cards[:-1], expected)
    with pytest.raises(ValueError, match="duplicate fold scorecards"):
        aggregate_construction_scorecard(CONSTRUCTION, cards + [cards[0]], expected)


def test_short_catastrophic_fold_still_counts_toward_drawdown():
    parts = _partitions()
    cards = [
        compute_fold_scorecard(
            cell_id,
            [json.loads(line) for line in data.decode().splitlines()],
        )
        for cell_id, data in sorted(parts.items())
    ]
    cell_id = cell_partition_id(CONSTRUCTION, "baseline", 99)
    catastrophic = _rows(cell_id, 99, "baseline", 0.0)[:2]
    catastrophic[0]["net_return"] = -0.90
    catastrophic_card = compute_fold_scorecard(cell_id, catastrophic)
    assert catastrophic_card["sharpe"] is None
    aggregate = aggregate_construction_scorecard(
        CONSTRUCTION, cards + [catastrophic_card], sorted(parts) + [cell_id]
    )
    assert aggregate["pooled_oos_max_drawdown"] >= 0.90


def test_end_to_end_shadow_negative_is_signed_and_rejected(tmp_path):
    identities, store = _fresh(tmp_path)
    result = run_crypto_momentum_evidence_slice(
        store, identities, _partitions(), **_slice_kwargs()
    )
    outcome = result.outcomes[LANE]
    assert outcome.final_state == DispositionState.REJECTED
    checks = {check.check_id: check.passed for check in outcome.verdict.checks}
    assert checks["metric_replay_reproduces"] is True
    assert checks["artifact_reproduces"] is True
    assert checks["candidate_evaluation_gate_passed"] is False


def test_worker_refuses_changed_cell_bytes(tmp_path):
    identities, _ = _fresh(tmp_path)
    parts = _partitions()
    produced = produce_worker_output(identities, parts, **_slice_kwargs())
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


def test_read_only_view_groups_construction_results(tmp_path):
    identities, store = _fresh(tmp_path)
    run_crypto_momentum_evidence_slice(store, identities, _partitions(), **_slice_kwargs())
    view = crypto_momentum_evidence_view(store.root)
    assert view["available"] is True
    assert view["construction_results"][CAMPAIGN][CONSTRUCTION]["state"] == "REJECTED"
    assert view["champions"] == {}


def test_evaluator_is_byte_deterministic():
    parts = _partitions()
    first = evaluate_partitions(parts, _params(parts), campaign_id=CAMPAIGN)[0]
    second = evaluate_partitions(parts, _params(parts), campaign_id=CAMPAIGN)[0]
    assert first.metrics == second.metrics
    assert first.gates == second.gates
    assert first.artifact_bytes == second.artifact_bytes


def test_invalid_utf8_cell_fails_closed_with_cell_context():
    cell_id = cell_partition_id(CONSTRUCTION, "baseline", 0)
    with pytest.raises(ValueError, match=cell_id.replace("[", "\\[").replace("]", "\\]")):
        evaluate_partitions(
            {cell_id: b"\xff"},
            EvaluationParams(expected_cells_by_construction={CONSTRUCTION: (cell_id,)}),
            campaign_id=CAMPAIGN,
        )


def test_identifiers_refuse_ambiguous_delimiters():
    with pytest.raises(ValueError, match="cannot contain '__'"):
        lane_id_for_construction("campaign__other", CONSTRUCTION)
    with pytest.raises(ValueError, match="cannot contain '::'"):
        cell_partition_id("construction::other", "baseline", 0)
    with pytest.raises(ValueError, match="non-negative integer"):
        cell_partition_id(CONSTRUCTION, "baseline", -1)


def test_partial_actor_override_retains_other_role_defaults():
    from src.evidence.contracts import AuthorityRole

    identities = build_slice_identities(
        NOW, actors={AuthorityRole.PRODUCER: "custom-momentum-producer"}
    )
    assert identities.actors[AuthorityRole.PRODUCER] == "custom-momentum-producer"
    assert identities.actors[AuthorityRole.LOCAL_IMPORTER] == "axiom-local-importer"
