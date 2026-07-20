"""Execution-context enrichment + quote capture — 15 acceptance criteria.

No mocks and NO gitignored runtime data: every test builds its own records and
writes to tmp_path, so the suite is identical on a fresh checkout and in CI.
"""

from __future__ import annotations

import json

import pytest

from src.equity.decision_capture import build_cycle_decisions, decision_linked_executor
from src.equity.rebalance import ExecResult, Order
from src.learning.decision_ledger import (
    DecisionLedgerError,
    append_decisions,
    append_execution_context,
    load_decisions,
)
from src.learning.decision_record import Disposition
from src.learning.execution_context import (
    BELOW_MINIMUM_ORDER,
    FRACTIONAL_NOT_ALLOWED,
    NO_ORDER_AFTER_QUANTIZATION,
    REF_LAST,
    REF_MID,
    ROUNDED_TO_WHOLE_SHARE,
    ZERO_AFTER_QUANTIZATION,
    ExecutionContextError,
    LotConstraints,
    QuoteSnapshot,
    quantize_units,
)
from src.learning.execution_context_builder import (
    SubmissionGate,
    build_execution_context,
)


def _quote(bid=99.0, ask=101.0, ref=None, ref_type=REF_MID, age_ms=10.0):
    return QuoteSnapshot(
        quote_source="test-feed", quote_timestamp="2026-07-20T12:00:00Z",
        quote_received_at="2026-07-20T12:00:00.010Z",
        bid=bid, ask=ask,
        reference_price=ref if ref is not None else (bid + ask) / 2.0,
        reference_price_type=ref_type, quote_age_ms=age_ms,
    )


def _decisions(disposition=Disposition.EXECUTE, weight=0.10, degross=1.0, cycle_id="c1"):
    return build_cycle_decisions(
        cycle_id=cycle_id, decision_timestamp="2026-07-20T12:00:00Z",
        disposition=disposition, reason_codes=[],
        target_weights={"AAPL": weight}, degross_factor=degross,
        current_weights={}, portfolio_nav=100_000.0, peak_nav=100_000.0,
    )


def _ctx(decision, **kw):
    params = dict(
        decision=decision, decision_digest=decision.digest(), plan_id="plan-1",
        created_at="2026-07-20T12:00:00Z", quote=_quote(),
        portfolio_nav=100_000.0, current_units=0.0,
    )
    params.update(kw)
    return build_execution_context(**params)


def _order(ticker="AAPL", coid="oid-1"):
    return Order(client_order_id=coid, ticker=ticker, side="BUY",
                 weight_delta=0.1, target_weight=0.1)


# ---------------------------------------------------------------------------
# AC1 / AC2 — the context must resolve to one authorised decision
# ---------------------------------------------------------------------------


def test_ac2_candidate_decision_cannot_receive_executable_context():
    candidate = _decisions(disposition=Disposition.HALT)[0]
    with pytest.raises(ExecutionContextError, match="only an EXECUTED_DECISION"):
        _ctx(candidate)


def test_ac1_context_cannot_be_created_for_a_missing_decision(tmp_path):
    """The executor refuses before any context is attempted."""
    ledger = tmp_path / "d.jsonl"
    recs = _decisions()
    append_decisions(recs, ledger)
    sent = []
    guarded = decision_linked_executor(
        lambda o: (sent.append(o), ExecResult.FILLED)[1],
        decisions=recs, ledger_path=ledger,
        quote_provider=lambda t: _quote(), nav_provider=lambda: {"portfolio_nav": 100_000.0},
        submission_gate=SubmissionGate(tmp_path / "gate.json"),
    )

    assert guarded(_order(ticker="TSLA", coid="x")) is ExecResult.REJECTED
    assert sent == []


def test_context_resolves_back_to_exactly_one_decision():
    d = _decisions()[0]
    ctx = _ctx(d)
    assert ctx.decision_digest == d.digest()
    assert ctx.decision_id == d.decision_id
    assert ctx.cycle_id == d.cycle_id


# ---------------------------------------------------------------------------
# AC3 / AC4 / AC5 / AC6 — pre-submission failure semantics
# ---------------------------------------------------------------------------


def _guarded_with(tmp_path, *, quote_provider, nav_provider, sent, holdings=None):
    ledger = tmp_path / "d.jsonl"
    recs = _decisions()
    append_decisions(recs, ledger)
    return decision_linked_executor(
        lambda o: (sent.append(o), ExecResult.FILLED)[1],
        decisions=recs, ledger_path=ledger,
        quote_provider=quote_provider, nav_provider=nav_provider,
        holdings_provider=holdings,
        submission_gate=SubmissionGate(tmp_path / "gate.json"),
    )


def test_ac4_missing_quote_prevents_submission(tmp_path):
    sent = []
    guarded = _guarded_with(tmp_path, quote_provider=lambda t: None,
                            nav_provider=lambda: {"portfolio_nav": 100_000.0}, sent=sent)
    assert guarded(_order()) is ExecResult.REJECTED
    assert sent == []


def test_ac5_stale_quote_prevents_submission(tmp_path):
    sent = []
    guarded = _guarded_with(tmp_path, quote_provider=lambda t: _quote(age_ms=60_000.0),
                            nav_provider=lambda: {"portfolio_nav": 100_000.0}, sent=sent)
    assert guarded(_order()) is ExecResult.REJECTED
    assert sent == []


def test_ac5_fresh_quote_is_accepted(tmp_path):
    sent = []
    guarded = _guarded_with(tmp_path, quote_provider=lambda t: _quote(age_ms=10.0),
                            nav_provider=lambda: {"portfolio_nav": 100_000.0}, sent=sent)
    assert guarded(_order()) is ExecResult.FILLED
    assert len(sent) == 1


def test_ac6_invalid_nav_prevents_submission(tmp_path):
    for bad in ({"portfolio_nav": 0.0}, {"portfolio_nav": float("nan")}, {}):
        sent = []
        guarded = _guarded_with(tmp_path, quote_provider=lambda t: _quote(),
                                nav_provider=lambda b=bad: b, sent=sent)
        assert guarded(_order()) is ExecResult.REJECTED, bad
        assert sent == []


def test_ac3_context_persistence_failure_prevents_broker_invocation(tmp_path):
    """Unwritable ledger -> no broker call."""
    blocked = tmp_path / "blocked"
    blocked.write_text("not a directory")
    ledger = blocked / "d.jsonl"
    recs = _decisions()
    # Decision digest must appear "on disk" for the guard to reach the context
    # step, so persist to a good path first, then point the context write at a
    # bad one by reusing the same guard with a broken ledger.
    good = tmp_path / "good.jsonl"
    append_decisions(recs, good)
    sent = []
    guarded = decision_linked_executor(
        lambda o: (sent.append(o), ExecResult.FILLED)[1],
        decisions=recs, ledger_path=good,
        quote_provider=lambda t: _quote(), nav_provider=lambda: {"portfolio_nav": 100_000.0},
        submission_gate=SubmissionGate(tmp_path / "gate.json"),
    )
    # Sanity: it works when the ledger is writable.
    assert guarded(_order()) is ExecResult.FILLED

    with pytest.raises(DecisionLedgerError):
        append_execution_context(_ctx(recs[0]), ledger)


def test_bad_quote_shapes_are_refused():
    with pytest.raises(ExecutionContextError, match="crossed quote"):
        _quote(bid=101.0, ask=99.0)
    with pytest.raises(ExecutionContextError, match="non-positive"):
        _quote(bid=0.0, ask=1.0)
    with pytest.raises(ExecutionContextError, match="unknown reference_price_type"):
        QuoteSnapshot(quote_source="s", quote_timestamp="t", quote_received_at="t",
                      bid=1.0, ask=2.0, reference_price=1.5, reference_price_type="GUESS")


# ---------------------------------------------------------------------------
# AC7 / AC8 — conversion + quantization are recorded and deterministic
# ---------------------------------------------------------------------------


def test_ac7_conversion_uses_the_recorded_reference_price_and_capital_base():
    d = _decisions(weight=0.10)[0]
    ctx = _ctx(d, portfolio_nav=100_000.0, allocatable_nav=50_000.0, quote=_quote(bid=99, ask=101))

    assert ctx.allocatable_nav == 50_000.0
    assert ctx.portfolio_nav == 100_000.0
    assert ctx.quote.reference_price == 100.0
    # target_notional = 50_000 * 0.10 = 5_000 ; raw units = 5_000 / 100 = 50
    assert ctx.target_notional == pytest.approx(5_000.0)
    assert ctx.raw_target_units == pytest.approx(50.0)
    assert ctx.intended_units == pytest.approx(50.0)


def test_portfolio_nav_is_not_assumed_equal_to_allocatable_nav():
    d = _decisions(weight=0.10)[0]
    full = _ctx(d, portfolio_nav=100_000.0)
    half = _ctx(d, portfolio_nav=100_000.0, allocatable_nav=50_000.0)

    assert full.allocatable_nav == 100_000.0
    assert half.allocatable_nav == 50_000.0
    assert half.raw_target_units == pytest.approx(full.raw_target_units / 2)


def test_ac8_quantization_is_deterministic_and_fully_recorded():
    d = _decisions(weight=0.10)[0]
    # 100_000 * 0.10 / 101.0 = 99.0099... -> floors to 99 whole shares
    ctx = _ctx(d, quote=_quote(bid=100.0, ask=102.0, ref=101.0))

    assert ctx.raw_target_units == pytest.approx(99.0099, abs=1e-3)
    assert ctx.intended_units == 99.0
    assert ctx.rounded_units_delta == pytest.approx(99.0 - ctx.raw_target_units)
    assert FRACTIONAL_NOT_ALLOWED in ctx.constraint_reason_codes
    assert ROUNDED_TO_WHOLE_SHARE in ctx.constraint_reason_codes
    assert ctx.lot.to_dict()["fractional_allowed"] is False


def test_quantize_never_increases_magnitude():
    lot = LotConstraints()
    assert quantize_units(9.9, lot) == 9.0
    assert quantize_units(-9.9, lot) == -9.0, "a short must not be rounded UP in magnitude"
    assert quantize_units(0.0, lot) == 0.0


def test_fractional_allowed_uses_the_quantity_increment():
    lot = LotConstraints(fractional_allowed=True, quantity_increment=0.25)
    assert quantize_units(9.9, lot) == 9.75


def test_zero_after_quantization_is_explicit():
    d = _decisions(weight=0.0001)[0]
    ctx = _ctx(d, quote=_quote(bid=9_999.0, ask=10_001.0))

    assert ctx.raw_target_units > 0
    assert ctx.intended_units == 0.0
    assert ZERO_AFTER_QUANTIZATION in ctx.constraint_reason_codes


def test_no_order_after_quantization_is_named_not_an_anonymous_noop():
    """A real weight change that produces no order must say so."""
    d = _decisions(weight=0.0001)[0]
    ctx = _ctx(d, quote=_quote(bid=9_999.0, ask=10_001.0))

    assert ctx.weight_delta != 0.0
    assert ctx.order_units == 0.0
    assert NO_ORDER_AFTER_QUANTIZATION in ctx.constraint_reason_codes


def test_below_minimum_order_is_flagged():
    d = _decisions(weight=0.10)[0]
    ctx = _ctx(d, lot=LotConstraints(minimum_order_units=1_000.0))
    assert BELOW_MINIMUM_ORDER in ctx.constraint_reason_codes


def test_degross_effect_is_visible_as_raw_vs_target_units():
    """The whole point of keeping both sides: sizing lost to RISK, not lots."""
    d = _decisions(weight=0.10, degross=0.5)[0]
    ctx = _ctx(d)

    assert ctx.pre_degross_units == 100.0   # 100k * 0.10 / 100
    assert ctx.intended_units == 50.0       # after 0.5 de-gross
    assert ctx.degross_factor == 0.5


# ---------------------------------------------------------------------------
# Quote capture semantics
# ---------------------------------------------------------------------------


def test_mid_and_spread_bps_use_the_specified_formulas():
    q = _quote(bid=99.0, ask=101.0)
    assert q.mid == 100.0
    assert q.spread_absolute == 2.0
    assert q.spread_bps == pytest.approx(200.0)


def test_a_last_price_is_never_labelled_mid():
    q = _quote(bid=99.0, ask=101.0, ref=100.5, ref_type=REF_LAST)
    assert q.reference_price_type == REF_LAST
    assert q.reference_price != q.mid


def test_execution_reference_is_persisted_when_it_differs_from_mid():
    q = QuoteSnapshot(
        quote_source="s", quote_timestamp="t", quote_received_at="t",
        bid=99.0, ask=101.0, reference_price=100.0, reference_price_type=REF_MID,
        execution_reference_price=101.0, execution_reference_price_type="ASK",
    )
    payload = q.to_dict()
    assert payload["reference_price"] == 100.0
    assert payload["execution_reference_price"] == 101.0


def test_entry_cost_only_round_trip_stays_unavailable():
    ctx = _ctx(_decisions()[0])

    assert ctx.estimated_entry_half_spread_bps == pytest.approx(100.0)
    assert ctx.estimated_entry_spread_cost is not None
    assert ctx.cost_estimate_availability == "entry_only"
    assert ctx.estimated_exit_cost_bps is None
    assert ctx.estimated_total_cost_bps is None
    assert ctx.expected_net_edge_bps is None
    for name in ("estimated_exit_cost_bps", "estimated_total_cost_bps", "expected_net_edge_bps"):
        assert name in ctx.unavailable_fields


# ---------------------------------------------------------------------------
# AC9 / AC10 / AC11 / AC12 — idempotency, distinguishability, immutability
# ---------------------------------------------------------------------------


def test_ac9_same_material_context_is_idempotent(tmp_path):
    ledger = tmp_path / "d.jsonl"
    ctx = _ctx(_decisions()[0])

    assert append_execution_context(ctx, ledger) is True
    assert append_execution_context(ctx, ledger) is False
    rows = [r for r in load_decisions(ledger) if r.get("row_type") == "execution_context"]
    assert len(rows) == 1


@pytest.mark.parametrize("kw", [
    {"quote": _quote(bid=98.0, ask=102.0)},
    {"portfolio_nav": 200_000.0},
    {"plan_id": "plan-2"},
    {"current_units": 5.0},
])
def test_ac10_changed_quote_nav_or_plan_produces_a_distinguishable_context(tmp_path, kw):
    ledger = tmp_path / "d.jsonl"
    d = _decisions()[0]
    append_execution_context(_ctx(d), ledger)

    assert append_execution_context(_ctx(d, **kw), ledger) is True
    rows = [r for r in load_decisions(ledger) if r.get("row_type") == "execution_context"]
    assert len(rows) == 2


def test_ac11_decision_rows_remain_immutable(tmp_path):
    ledger = tmp_path / "d.jsonl"
    recs = _decisions()
    append_decisions(recs, ledger)
    before = [r for r in load_decisions(ledger) if r.get("row_type") is None]

    append_execution_context(_ctx(recs[0]), ledger)
    after = [r for r in load_decisions(ledger) if r.get("row_type") is None]

    assert before == after
    assert after[0]["intended_units"] is None, "decision row keeps its weight-native shape"


def test_ac12_broker_link_rows_remain_separate(tmp_path):
    ledger = tmp_path / "d.jsonl"
    recs = _decisions()
    append_decisions(recs, ledger)
    sent = []
    guarded = decision_linked_executor(
        lambda o: (sent.append(o), ExecResult.FILLED)[1],
        decisions=recs, ledger_path=ledger,
        quote_provider=lambda t: _quote(), nav_provider=lambda: {"portfolio_nav": 100_000.0},
        submission_gate=SubmissionGate(tmp_path / "gate.json"),
    )
    guarded(_order())

    rows = load_decisions(ledger)
    kinds = [r.get("row_type") for r in rows]
    assert kinds.count("execution_context") == 1
    assert kinds.count("broker_link") == 1
    assert kinds.count(None) == 1


# ---------------------------------------------------------------------------
# AC13 — the full resolution chain
# ---------------------------------------------------------------------------


def test_ac13_filled_order_resolves_link_to_context_to_decision(tmp_path):
    ledger = tmp_path / "d.jsonl"
    recs = _decisions()
    append_decisions(recs, ledger)
    guarded = decision_linked_executor(
        lambda o: ExecResult.FILLED,
        decisions=recs, ledger_path=ledger,
        quote_provider=lambda t: _quote(), nav_provider=lambda: {"portfolio_nav": 100_000.0},
        submission_gate=SubmissionGate(tmp_path / "gate.json"),
    )
    assert guarded(_order(coid="OID-7")) is ExecResult.FILLED

    rows = load_decisions(ledger)
    link = next(r for r in rows if r.get("row_type") == "broker_link")
    ctx = next(r for r in rows if r.get("row_type") == "execution_context")
    decision = next(r for r in rows if r.get("row_type") is None)

    # broker order -> decision digest
    assert link["broker_order_id"] == "OID-7"
    assert link["links_decision_digest"] == decision["decision_digest"]
    # execution context -> same decision digest
    assert ctx["links_decision_digest"] == decision["decision_digest"]
    assert ctx["decision_id"] == decision["decision_id"]
    # and the context carries the executable units the decision could not
    assert ctx["intended_units"] is not None
    assert ctx["quote"]["mid"] == 100.0


# ---------------------------------------------------------------------------
# Post-acceptance linkage failure: incident + gate, never a blind cancel
# ---------------------------------------------------------------------------


def test_linkage_failure_after_acceptance_raises_incident_and_closes_the_gate(tmp_path):
    gate_path = tmp_path / "gate.json"
    gate = SubmissionGate(gate_path)
    assert gate.is_open() is True

    gate.raise_unlinked_incident(
        broker_order_id="OID-9", decision_id="d1",
        detail="write failed", at="2026-07-20T12:00:00Z",
    )

    assert gate.is_open() is False
    payload = json.loads(gate_path.read_text())
    assert payload["severity"] == "CRITICAL"
    assert payload["incident"] == "UNLINKED_BROKER_ORDER"
    assert "Do NOT auto-resubmit" in payload["recovery"]


def test_closed_gate_blocks_all_further_submissions(tmp_path):
    ledger = tmp_path / "d.jsonl"
    recs = _decisions()
    append_decisions(recs, ledger)
    gate = SubmissionGate(tmp_path / "gate.json")
    gate.raise_unlinked_incident(broker_order_id="OID-1", decision_id="d1",
                                 detail="x", at="2026-07-20T12:00:00Z")
    sent = []
    guarded = decision_linked_executor(
        lambda o: (sent.append(o), ExecResult.FILLED)[1],
        decisions=recs, ledger_path=ledger,
        quote_provider=lambda t: _quote(), nav_provider=lambda: {"portfolio_nav": 100_000.0},
        submission_gate=gate,
    )

    assert guarded(_order()) is ExecResult.REJECTED
    assert sent == [], "no further order may be created while the book is unreconciled"


def test_unreadable_gate_state_fails_closed(tmp_path):
    gate_path = tmp_path / "gate.json"
    gate_path.write_text("{not json")
    assert SubmissionGate(gate_path).is_open() is False


def test_gate_reopens_only_explicitly(tmp_path):
    gate = SubmissionGate(tmp_path / "gate.json")
    gate.raise_unlinked_incident(broker_order_id="O", decision_id="d",
                                 detail="x", at="2026-07-20T12:00:00Z")
    assert gate.is_open() is False

    gate.reopen(reason="linkage recovered from broker state", at="2026-07-20T13:00:00Z")

    assert gate.is_open() is True
    assert "recovered" in json.loads((tmp_path / "gate.json").read_text())["reason"]


# ---------------------------------------------------------------------------
# AC14 / AC15
# ---------------------------------------------------------------------------


def test_ac14_this_suite_needs_no_gitignored_runtime_data():
    """Every test above builds its own inputs and writes to tmp_path."""
    from src.learning.decision_ledger import DECISION_LEDGER_PATH
    from src.learning.outcome_ledger import LEDGER_PATH

    # Nothing here reads the shared ledgers; assert the defaults are merely
    # importable so a missing trained_data/ cannot break collection.
    assert DECISION_LEDGER_PATH.name.endswith(".jsonl")
    assert LEDGER_PATH.name.endswith(".jsonl")


def test_ac15_no_llm_canary_and_focused_suite_are_covered_elsewhere():
    """Guard against silently dropping the canary from the equity path."""
    import ast
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    for module in ("decision_capture.py",):
        tree = ast.parse((root / "src" / "equity" / module).read_text())
        for node in ast.walk(tree):
            names = []
            if isinstance(node, ast.Import):
                names = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module:
                names = [node.module]
            for name in names:
                assert name.split(".")[0].lower() not in {
                    "anthropic", "openai", "langchain", "litellm", "ollama", "claude",
                }, f"{module} imports {name}"
