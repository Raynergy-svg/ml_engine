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
    import_worker_output,
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


def _ledger_rows(n: int = 10) -> list[dict]:
    return [
        {
            "asof_date": f"2026-06-{i + 1:02d}", "today_net_return": 0.001 * (1 if i % 2 else -1),
            "gross_leverage": 0.4 + 0.01 * i, "today_turnover": 0.05 + 0.001 * i,
        }
        for i in range(n)
    ]


def _ledger_bytes(rows: list[dict]) -> bytes:
    return b"".join(json.dumps(r, sort_keys=True).encode() + b"\n" for r in rows)


def test_forward_ledger_contract_is_present_and_well_formed(tmp_path):
    """Roadmap §14 standardized strategy-return/exposure contract, retrofitted
    additively onto crypto_momentum: sourced from the construction's REAL
    forward-shadow ledger (src.crypto.momentum_shadow.record_shadow_cycle),
    with all four fields populated (no null turnover — this ledger always
    records it, unlike hedge_eval's)."""
    from src.evidence.crypto_momentum.manifests import forward_ledger_partition_id

    parts = _partitions()
    rows = _ledger_rows()
    parts_with_ledger = {**parts, forward_ledger_partition_id(CONSTRUCTION): _ledger_bytes(rows)}
    identities, store = _fresh(tmp_path)
    result = run_crypto_momentum_evidence_slice(store, identities, parts_with_ledger, **_slice_kwargs())
    package_digest = result.outcomes[LANE].package_digest
    payload = store.load_package_file(package_digest, "artifacts/strategy_returns.jsonl")
    contract_rows = [json.loads(line) for line in payload.decode().splitlines() if line.strip()]
    assert len(contract_rows) == len(rows)
    dates = [r["date"] for r in contract_rows]
    assert dates == sorted(dates)
    for contract_row, source_row in zip(contract_rows, rows):
        assert set(contract_row) == {"date", "net_return", "gross_exposure", "turnover"}
        assert contract_row["net_return"] == pytest.approx(source_row["today_net_return"])
        assert contract_row["gross_exposure"] == pytest.approx(source_row["gross_leverage"])
        assert contract_row["turnover"] == pytest.approx(source_row["today_turnover"])


def test_construction_without_forward_ledger_is_unaffected(tmp_path):
    """Purely additive: a construction with NO forward-ledger partition gets
    exactly the same single-artifact package as before this retrofit."""
    identities, store = _fresh(tmp_path)
    result = run_crypto_momentum_evidence_slice(store, identities, _partitions(), **_slice_kwargs())
    package_digest = result.outcomes[LANE].package_digest
    package = store.load_package(package_digest)
    assert [a.relative_path for a in package.artifacts] == ["artifacts/construction_scorecard.json"]


def test_import_rejects_doctored_forward_ledger_contract(tmp_path):
    import dataclasses

    from src.evidence.crypto_momentum.evaluation import evaluate_partitions as _eval
    from src.evidence.crypto_momentum.manifests import forward_ledger_partition_id

    parts = _partitions()
    parts_with_ledger = {**parts, forward_ledger_partition_id(CONSTRUCTION): _ledger_bytes(_ledger_rows())}

    def tamper_evaluator(partitions, params, *, campaign_id):
        heads = _eval(partitions, params, campaign_id=campaign_id)
        return tuple(
            dataclasses.replace(h, strategy_return_bytes=h.strategy_return_bytes + b"\n")
            if h.strategy_return_bytes else h
            for h in heads
        )

    identities, store = _fresh(tmp_path)
    produced = produce_worker_output(
        identities, parts_with_ledger, evaluator=tamper_evaluator, **_slice_kwargs()
    )
    outcome = import_worker_output(
        store, identities, produced, parts_with_ledger, created_at=NOW,
    )[LANE]
    assert outcome.final_state == DispositionState.REJECTED
    assert "strategy_return_contract_reproduces" in (outcome.reason or "")


def test_import_rejects_doctored_scorecard_when_ledger_artifact_also_present(tmp_path):
    """Independent Code Reviewer finding (2026-07-22): the only doctored-
    artifact test tampered the SECOND (ledger) artifact while the first
    (scorecard) was valid — this covers the reverse, catching a regression
    like 'only check the first/last artifact in the list' that the other
    test alone would miss."""
    import dataclasses

    from src.evidence.crypto_momentum.evaluation import evaluate_partitions as _eval
    from src.evidence.crypto_momentum.manifests import forward_ledger_partition_id

    parts = _partitions()
    parts_with_ledger = {**parts, forward_ledger_partition_id(CONSTRUCTION): _ledger_bytes(_ledger_rows())}

    def tamper_evaluator(partitions, params, *, campaign_id):
        heads = _eval(partitions, params, campaign_id=campaign_id)
        return tuple(dataclasses.replace(h, artifact_bytes=h.artifact_bytes + b" ") for h in heads)

    identities, store = _fresh(tmp_path)
    produced = produce_worker_output(
        identities, parts_with_ledger, evaluator=tamper_evaluator, **_slice_kwargs()
    )
    outcome = import_worker_output(
        store, identities, produced, parts_with_ledger, created_at=NOW,
    )[LANE]
    assert outcome.final_state == DispositionState.REJECTED
    assert "artifact_reproduces" in (outcome.reason or "")


def test_missing_declared_forward_ledger_is_refused_at_the_worker(tmp_path):
    """A signed declaration that a construction carries forward-ledger data
    cannot be satisfied by simply not supplying it — the deterministic
    aggregator refuses the mismatch at the worker, the same fail-closed
    discipline the cell aggregator already applies to missing folds."""
    from dataclasses import replace as dc_replace

    parts = _partitions()
    identities, _ = _fresh(tmp_path)
    params = dc_replace(_params(parts), expected_ledger_constructions=(CONSTRUCTION,))
    with pytest.raises(ValueError, match="do not match"):
        produce_worker_output(identities, parts, params=params, **_slice_kwargs())


def test_reserved_id_forward_ledger_is_rejected():
    """Code Reviewer finding (crypto_carry sibling, 2026-07-22): a
    construction literally equal to 'forward_ledger' would make its cell
    partition ids start with FORWARD_LEDGER_PREFIX, misclassifying every
    cell as ledger data. Refused outright."""
    from src.evidence.crypto_momentum.manifests import build_momentum_strategy_manifest, forward_ledger_partition_id

    with pytest.raises(ValueError, match="reserved for forward-ledger"):
        cell_partition_id("forward_ledger", "baseline", 0)
    with pytest.raises(ValueError, match="reserved for forward-ledger"):
        forward_ledger_partition_id("forward_ledger")
    with pytest.raises(ValueError, match="reserved for forward-ledger"):
        build_momentum_strategy_manifest(
            CAMPAIGN, {"forward_ledger": ("x",)}, frozen_at=NOW,
            min_oos_folds=1, sharpe_bar=0.0, sharpe_tstat_bar=0.0,
            max_drawdown_limit=1.0, stress_sharpe_floor=0.0, drop_one_sharpe_floor=0.0,
        )


def test_evaluate_partitions_rejects_duplicate_expected_ledger_constructions():
    """Symmetry fix (Code Reviewer, 2026-07-22): evaluate_partitions used to
    silently dedupe a duplicated expected_ledger_constructions declaration."""
    from dataclasses import replace as dc_replace

    parts = _partitions()
    params = dc_replace(_params(parts), expected_ledger_constructions=(CONSTRUCTION, CONSTRUCTION))
    with pytest.raises(ValueError, match="duplicates"):
        evaluate_partitions(parts, params, campaign_id=CAMPAIGN)


def test_partial_actor_override_retains_other_role_defaults():
    from src.evidence.contracts import AuthorityRole

    identities = build_slice_identities(
        NOW, actors={AuthorityRole.PRODUCER: "custom-momentum-producer"}
    )
    assert identities.actors[AuthorityRole.PRODUCER] == "custom-momentum-producer"
    assert identities.actors[AuthorityRole.LOCAL_IMPORTER] == "axiom-local-importer"
