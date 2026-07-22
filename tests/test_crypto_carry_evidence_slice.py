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


def _ledger_rows(n: int = 10) -> list[dict]:
    return [
        {
            "asof_date": f"2026-06-{i + 1:02d}", "today_net_return": 0.0008 * (1 if i % 2 else -1),
            "gross_leverage": 0.3 + 0.01 * i, "today_turnover": 0.04 + 0.001 * i,
        }
        for i in range(n)
    ]


def _ledger_bytes(rows: list[dict]) -> bytes:
    return b"".join(json.dumps(r, sort_keys=True).encode() + b"\n" for r in rows)


def test_forward_ledger_contract_is_present_and_well_formed(tmp_path):
    """Roadmap §14 standardized strategy-return/exposure contract, retrofitted
    additively onto crypto_carry: sourced from the carry's REAL forward-shadow
    ledger (src.crypto.carry_shadow.record_shadow_cycle)."""
    from src.evidence.crypto_carry.manifests import forward_ledger_partition_id

    parts = _partitions(tail_complete=True)
    rows = _ledger_rows()
    parts_with_ledger = {**parts, forward_ledger_partition_id(CARRY_ID): _ledger_bytes(rows)}
    identities, store = _fresh(tmp_path)
    result = run_crypto_carry_evidence_slice(store, identities, parts_with_ledger, **_kwargs())
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


def test_carry_without_forward_ledger_is_unaffected(tmp_path):
    """Purely additive: a carry with NO forward-ledger partition gets exactly
    the same single-artifact package as before this retrofit."""
    identities, store = _fresh(tmp_path)
    result = run_crypto_carry_evidence_slice(store, identities, _partitions(tail_complete=True), **_kwargs())
    package_digest = result.outcomes[LANE].package_digest
    package = store.load_package(package_digest)
    assert [a.relative_path for a in package.artifacts] == ["artifacts/carry_scorecard.json"]


def test_import_rejects_doctored_forward_ledger_contract(tmp_path):
    import dataclasses

    from src.evidence.crypto_carry.evaluation import evaluate_partitions as _eval
    from src.evidence.crypto_carry.manifests import forward_ledger_partition_id
    from src.evidence.crypto_carry.slice import import_worker_output

    parts = _partitions(tail_complete=True)
    parts_with_ledger = {**parts, forward_ledger_partition_id(CARRY_ID): _ledger_bytes(_ledger_rows())}

    def tamper_evaluator(partitions, params, *, campaign_id):
        heads = _eval(partitions, params, campaign_id=campaign_id)
        return tuple(
            dataclasses.replace(h, strategy_return_bytes=h.strategy_return_bytes + b"\n")
            if h.strategy_return_bytes else h
            for h in heads
        )

    identities, store = _fresh(tmp_path)
    produced = produce_worker_output(identities, parts_with_ledger, evaluator=tamper_evaluator, **_kwargs())
    outcome = import_worker_output(store, identities, produced, parts_with_ledger, created_at=NOW)[LANE]
    assert outcome.final_state == DispositionState.REJECTED
    assert "strategy_return_contract_reproduces" in (outcome.reason or "")


def test_import_rejects_doctored_scorecard_when_ledger_artifact_also_present(tmp_path):
    """Covers the reverse of the test above: the first (scorecard) artifact
    tampered while the second (ledger) is valid — catches a regression like
    'only check the first/last artifact in the list'."""
    import dataclasses

    from src.evidence.crypto_carry.evaluation import evaluate_partitions as _eval
    from src.evidence.crypto_carry.manifests import forward_ledger_partition_id
    from src.evidence.crypto_carry.slice import import_worker_output

    parts = _partitions(tail_complete=True)
    parts_with_ledger = {**parts, forward_ledger_partition_id(CARRY_ID): _ledger_bytes(_ledger_rows())}

    def tamper_evaluator(partitions, params, *, campaign_id):
        heads = _eval(partitions, params, campaign_id=campaign_id)
        return tuple(dataclasses.replace(h, artifact_bytes=h.artifact_bytes + b" ") for h in heads)

    identities, store = _fresh(tmp_path)
    produced = produce_worker_output(identities, parts_with_ledger, evaluator=tamper_evaluator, **_kwargs())
    outcome = import_worker_output(store, identities, produced, parts_with_ledger, created_at=NOW)[LANE]
    assert outcome.final_state == DispositionState.REJECTED
    assert "artifact_reproduces" in (outcome.reason or "")


def test_reserved_id_forward_ledger_is_rejected():
    """Code Reviewer finding (2026-07-22): a carry_id literally equal to
    'forward_ledger' would make its cell partition ids start with
    FORWARD_LEDGER_PREFIX, misclassifying every cell as ledger data. Refused
    outright rather than relying on the downstream mismatch check to catch it."""
    from src.evidence.crypto_carry.manifests import build_carry_strategy_manifest, forward_ledger_partition_id

    with pytest.raises(ValueError, match="reserved for forward-ledger"):
        cell_partition_id("forward_ledger", VENUE_SET, "base", "normal")
    with pytest.raises(ValueError, match="reserved for forward-ledger"):
        forward_ledger_partition_id("forward_ledger")
    with pytest.raises(ValueError, match="reserved for forward-ledger"):
        build_carry_strategy_manifest(
            CAMPAIGN, {"forward_ledger": ("x",)}, frozen_at=NOW,
            min_cells=1, net_sharpe_floor=0.0, max_drawdown_limit=1.0,
            max_margin_utilization=1.0, max_tracking_error=1.0, min_capacity_usd=0.0,
        )


def test_evaluate_partitions_rejects_duplicate_expected_ledger_carries():
    """Symmetry fix (Code Reviewer, 2026-07-22): evaluate_partitions used to
    silently dedupe a duplicated expected_ledger_carries declaration via
    set(...); build_carry_strategy_manifest already refused duplicates, so
    a caller constructing EvaluationParams directly should get the same bar."""
    from dataclasses import replace as dc_replace

    parts = _partitions(tail_complete=True)
    params = dc_replace(_params(parts), expected_ledger_carries=(CARRY_ID, CARRY_ID))
    with pytest.raises(ValueError, match="duplicates"):
        evaluate_partitions(parts, params, campaign_id=CAMPAIGN)


def test_swapped_strategy_manifest_with_different_ledger_declaration_is_rejected(tmp_path):
    """A validly-signed but DIFFERENT strategy manifest (declaring no
    forward-ledger carries, vs. the real one produced alongside the package)
    is refused. Attempting to isolate JUST strategy_declared_ledger_carries_
    match_dataset here surfaced an honest finding worth recording rather than
    papering over: because the ledger declaration is embedded in the
    manifest's own canonical content, ANY change to it also changes the
    manifest's digest — so this always trips strategy_manifest_bound_to_job
    FIRST (checked earlier in the tuple). strategy_declared_ledger_carries_
    match_dataset can therefore never be the FIRST failing check while the
    manifest-binding checks both pass, given this binding architecture — it
    remains real, cheap defense-in-depth (documents intent, survives a
    hypothetical future refactor of the binding checks), not dead code, but
    it is not independently reachable via a manifest swap the way this test
    initially assumed."""
    from dataclasses import replace as dc_replace

    from src.evidence.crypto_carry.local_import import import_head
    from src.evidence.crypto_carry.manifests import build_carry_strategy_manifest, forward_ledger_partition_id

    parts = _partitions(tail_complete=True)
    parts_with_ledger = {**parts, forward_ledger_partition_id(CARRY_ID): _ledger_bytes(_ledger_rows())}
    identities, store = _fresh(tmp_path)
    params = dc_replace(_params(parts), expected_ledger_carries=(CARRY_ID,))
    produced = produce_worker_output(identities, parts_with_ledger, params=params, **_kwargs())
    head = produced.worker_output.heads[0]

    honest_params = _params(parts)  # expected_ledger_carries=() default
    liar_manifest = build_carry_strategy_manifest(
        CAMPAIGN, honest_params.expected_cells_by_carry, frozen_at=NOW,
        min_cells=honest_params.min_cells, net_sharpe_floor=honest_params.net_sharpe_floor,
        max_drawdown_limit=honest_params.max_drawdown_limit,
        max_margin_utilization=honest_params.max_margin_utilization,
        max_tracking_error=honest_params.max_tracking_error,
        min_capacity_usd=honest_params.min_capacity_usd,
        forward_ledger_carries=(),
    )
    liar_envelope = identities.producer.sign(liar_manifest, created_at=NOW)

    outcome = import_head(
        store, head, campaign_id=CAMPAIGN, job=produced.job,
        capability_profile_envelope=produced.worker_output.capability_profile_envelope,
        dataset_manifest_envelope=produced.worker_output.dataset_manifest_envelope,
        strategy_manifest_envelope=liar_envelope,
        partitions=parts_with_ledger, trust_store=identities.trust_store,
        importer=identities.importer, importer_id=identities.actors[AuthorityRole.LOCAL_IMPORTER],
        verifier=identities.verifier, verifier_id=identities.actors[AuthorityRole.INDEPENDENT_VERIFIER],
        created_at=NOW,
    )
    assert outcome.final_state == DispositionState.REJECTED
    assert "strategy_manifest_bound_to_job" in (outcome.reason or "")


def test_missing_declared_forward_ledger_is_refused_at_the_worker(tmp_path):
    """A signed declaration that a carry carries forward-ledger data cannot
    be satisfied by simply not supplying it — the deterministic aggregator
    refuses the mismatch at the worker."""
    from dataclasses import replace as dc_replace

    parts = _partitions(tail_complete=True)
    identities, _ = _fresh(tmp_path)
    params = dc_replace(_params(parts), expected_ledger_carries=(CARRY_ID,))
    with pytest.raises(ValueError, match="do not match"):
        produce_worker_output(identities, parts, params=params, **_kwargs())
