"""Deterministic outcome attribution — validated against REAL broker history.

No mocks (see the No-Mock Rule in ``.claude/rules/improvement.md``). The
integration tests here run against the actual practice-account journal at
``trained_data/oanda/transactions.jsonl`` -- 297 ORDER_FILLs, 171
P&L-realizing, 124 linkable round trips. If the attribution arithmetic is
wrong, these fail on real money movements rather than on invented fixtures.

Unit tests construct real dataclasses directly; there are no test doubles.
"""

from __future__ import annotations

import json

import pytest

from src.learning.oanda_outcome_adapter import (
    DEFAULT_TRANSACTIONS_PATH,
    build_outcome_records,
    index_opening_fills,
    load_outcome_records,
    load_transactions,
)
from src.learning.outcome_attribution import (
    RECONCILE_TOLERANCE_HOME,
    AttributionError,
    DecisionRecord,
    OutcomeRecord,
    attribute,
)

pytestmark = pytest.mark.filterwarnings("ignore::DeprecationWarning")


def _journal_available() -> bool:
    return DEFAULT_TRANSACTIONS_PATH.exists()


requires_journal = pytest.mark.skipif(
    not _journal_available(),
    reason="OANDA transaction journal not present in this checkout",
)


# ---------------------------------------------------------------------------
# Integration: the whole point of this module -- does the arithmetic close
# against real realized P&L?
# ---------------------------------------------------------------------------


@requires_journal
def test_decomposition_reconciles_against_every_real_closed_trade():
    """The headline guarantee: gross P&L reproduces the broker's own figure.

    This is the check that makes every other number in an AttributionReport
    credible. It runs on real fills, not fixtures.
    """
    records, _unlinkable = load_outcome_records()
    assert records, "expected at least one reconstructed round trip"

    reports = [attribute(r) for r in records]
    unreconciled = [rep for rep in reports if not rep.reconciled]

    assert not unreconciled, (
        f"{len(unreconciled)}/{len(reports)} slices failed to reconcile; "
        f"worst residual={max(abs(r.residual_home) for r in unreconciled):.6f} home. "
        "A non-empty list here means the attribution MODEL is wrong -- fix the "
        "decomposition, do not widen the tolerance."
    )


@requires_journal
def test_worst_residual_is_at_the_float_noise_floor():
    """Guards against a decomposition that 'passes' only via a loose tolerance."""
    records, _ = load_outcome_records()
    worst = max(abs(attribute(r).residual_home) for r in records)
    # Empirically 5e-05; anything approaching the tolerance means real drift.
    assert worst < RECONCILE_TOLERANCE_HOME / 10, f"worst residual {worst:.6f} is too close to tolerance"


@requires_journal
def test_linkage_coverage_is_reported_not_silently_dropped():
    records, unlinkable = load_outcome_records()
    # Unlinkable trades were opened before the journal window; they must be
    # surfaced so coverage can be stated honestly.
    assert isinstance(unlinkable, list)
    assert len(records) + len(set(unlinkable)) >= len(records)
    total = len(records) + len(set(unlinkable))
    assert len(records) / total > 0.9, "linkage coverage unexpectedly low"


@requires_journal
def test_sizing_attribution_is_unavailable_without_a_decision_record():
    """The honest gap: historical trades have no recorded intent.

    Reporting 0.0 would assert 'sizing cost nothing'. It must report
    unavailable instead -- this is exactly the hole DecisionRecord fills.
    """
    records, _ = load_outcome_records()
    report = attribute(records[0])

    assert report.sizing_pl_home is None
    assert report.direction_pl_home is None
    assert "sizing_pl_home" in report.unavailable
    assert "direction_pl_home" in report.unavailable


@requires_journal
def test_exit_reasons_bucket_into_real_categories():
    records, _ = load_outcome_records()
    buckets = {attribute(r).exit_bucket for r in records}
    # The real journal contains stop-loss and take-profit exits.
    assert "risk_rail" in buckets
    assert "target" in buckets


@requires_journal
def test_reports_are_json_serialisable_for_the_ledger():
    records, _ = load_outcome_records()
    payload = attribute(records[0]).to_dict()
    assert json.loads(json.dumps(payload))["trade_id"] == records[0].trade_id


@requires_journal
def test_malformed_journal_line_is_skipped_not_fatal(tmp_path):
    """A torn append must not abort the whole load."""
    good = json.dumps({"type": "ORDER_FILL", "id": "1", "instrument": "EUR_USD"})
    path = tmp_path / "transactions.jsonl"
    path.write_text(f"{good}\n{{not json\n{good}\n")

    rows = load_transactions(path)

    assert len(rows) == 2


def test_missing_journal_raises_rather_than_returning_empty(tmp_path):
    """Fail-closed: an absent journal must not look like 'no trades'."""
    with pytest.raises(FileNotFoundError):
        load_transactions(tmp_path / "nope.jsonl")


# ---------------------------------------------------------------------------
# Unit: sign conventions and the decision-aware decomposition
# ---------------------------------------------------------------------------


def _outcome(**overrides) -> OutcomeRecord:
    base = dict(
        trade_id="t1", instrument="EUR_USD",
        entry_time="2026-07-01T00:00:00Z", exit_time="2026-07-01T01:00:00Z",
        entry_price=1.1000, exit_price=1.1100, units=1000.0,
        realized_pl_home=10.0, conversion_factor=1.0,
    )
    base.update(overrides)
    return OutcomeRecord(**base)


def test_long_that_rose_makes_money():
    report = attribute(_outcome())
    assert report.gross_pl_home == pytest.approx(10.0)
    assert report.reconciled


def test_short_that_fell_makes_money():
    report = attribute(_outcome(units=-1000.0, exit_price=1.0900, realized_pl_home=10.0))
    assert report.gross_pl_home == pytest.approx(10.0)
    assert report.reconciled


def test_long_that_fell_loses_money():
    report = attribute(_outcome(exit_price=1.0900, realized_pl_home=-10.0))
    assert report.gross_pl_home == pytest.approx(-10.0)
    assert report.reconciled


def test_residual_flags_a_broker_disagreement_instead_of_hiding_it():
    """If our arithmetic disagrees with the broker, say so."""
    report = attribute(_outcome(realized_pl_home=99.0))

    assert not report.reconciled
    assert report.residual_home == pytest.approx(89.0)


def test_financing_and_fees_move_net_but_not_gross():
    report = attribute(_outcome(financing_home=-0.5, commission_home=0.25, other_fees_home=0.25))

    assert report.gross_pl_home == pytest.approx(10.0)
    assert report.fees_home == pytest.approx(0.5)
    assert report.net_pl_home == pytest.approx(10.0 - 0.5 - 0.5)


def test_decision_record_enables_direction_and_sizing_split():
    decision = DecisionRecord(
        decision_id="d1", asof="2026-07-01T00:00:00Z", lane="equity",
        instrument="EUR_USD", intended_units=800.0, signal_source="test_strategy",
    )
    report = attribute(_outcome(), decision)

    # 0.01 price move: intended 800 units -> 8.0; overfill of 200 -> 2.0
    assert report.direction_pl_home == pytest.approx(8.0)
    assert report.sizing_pl_home == pytest.approx(2.0)
    assert "sizing_pl_home" not in report.unavailable


def test_degross_contribution_is_measured_when_pre_degross_size_is_known():
    """De-grossing that cut a LOSING position must show as money SAVED."""
    decision = DecisionRecord(
        decision_id="d1", asof="2026-07-01T00:00:00Z", lane="equity",
        instrument="EUR_USD", intended_units=500.0, signal_source="test_strategy",
        pre_degross_units=1000.0, degross_factor=0.5,
    )
    # Price fell: a long loses. De-grossing removed 500 units of that loss.
    report = attribute(_outcome(exit_price=1.0900, units=500.0, realized_pl_home=-5.0), decision)

    assert report.degross_pl_home == pytest.approx(5.0)  # positive == saved
    assert "degross_pl_home" not in report.unavailable


def test_degross_unavailable_when_pre_degross_size_not_recorded():
    decision = DecisionRecord(
        decision_id="d1", asof="2026-07-01T00:00:00Z", lane="equity",
        instrument="EUR_USD", intended_units=1000.0, signal_source="test_strategy",
    )
    report = attribute(_outcome(), decision)

    assert report.degross_pl_home is None
    assert "degross_pl_home" in report.unavailable


def test_instrument_mismatch_between_decision_and_outcome_is_refused():
    decision = DecisionRecord(
        decision_id="d1", asof="2026-07-01T00:00:00Z", lane="equity",
        instrument="GBP_USD", intended_units=1000.0, signal_source="test_strategy",
    )
    with pytest.raises(AttributionError, match="instrument mismatch"):
        attribute(_outcome(), decision)


def test_zero_units_outcome_is_refused():
    with pytest.raises(AttributionError, match="units must be non-zero"):
        _outcome(units=0.0)


def test_empty_trade_id_is_refused():
    with pytest.raises(AttributionError, match="trade_id"):
        _outcome(trade_id="")


def test_holding_period_handles_oanda_nanosecond_timestamps():
    report = attribute(_outcome(
        entry_time="2026-07-15T02:12:50.419308529Z",
        exit_time="2026-07-15T02:12:52.419308529Z",
    ))
    assert report.holding_period_seconds == pytest.approx(2.0)


def test_unparseable_timestamp_yields_none_not_zero():
    """Zero is a real scalp duration; unknown must be distinguishable."""
    report = attribute(_outcome(entry_time="not-a-time"))
    assert report.holding_period_seconds is None


def test_index_opening_fills_ignores_non_fill_rows():
    rows = [
        {"type": "DAILY_FINANCING", "id": "1"},
        {"type": "ORDER_FILL", "time": "2026-07-01T00:00:00Z", "instrument": "EUR_USD",
         "tradeOpened": {"tradeID": "42", "price": "1.1", "units": "100"}},
    ]
    opens = index_opening_fills(rows)
    assert set(opens) == {"42"}


def test_unlinkable_close_is_reported_not_dropped():
    rows = [{
        "type": "ORDER_FILL", "time": "2026-07-01T01:00:00Z", "instrument": "EUR_USD",
        "reason": "MARKET_ORDER_TRADE_CLOSE",
        "tradesClosed": [{"tradeID": "999", "price": "1.11", "units": "-100", "realizedPL": "1.0"}],
    }]
    records, unlinkable = build_outcome_records(rows)

    assert records == []
    assert unlinkable == ["999"]
