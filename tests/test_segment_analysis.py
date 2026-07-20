"""Segment expectancy + the out-of-sample gate.

No mocks. Real ledger rows built from real dataclasses; the integration test
runs against the committed outcome ledger.
"""

from __future__ import annotations

import pytest

from src.learning.outcome_attribution import OutcomeRecord, attribute
from src.learning.outcome_ledger import LEDGER_PATH, build_row, load_ledger
from src.learning.segment_analysis import (
    EDGE_CONSUMED_BY_COST,
    NO_EDGE_AT_MID,
    PROFITABLE_AFTER_COST,
    segment_report,
    segment_stats,
    temporal_split,
    validate_segment,
)


def _row(trade_id, *, pl, spread=0.0, instrument="EUR_USD", entry="2026-07-01T00:00:00Z"):
    """A row whose realized P&L is exactly `pl` and half-spread exactly `spread`."""
    # Choose an exit price that makes gross_pl_home == pl for units=1000.
    outcome = OutcomeRecord(
        trade_id=trade_id, instrument=instrument,
        entry_time=entry, exit_time="2026-07-05T00:00:00Z",
        entry_price=1.1000, exit_price=1.1000 + pl / 1000.0,
        units=1000.0, realized_pl_home=pl, conversion_factor=1.0,
        execution_cost_home=spread,
    )
    return build_row(outcome, attribute(outcome))


# ---------------------------------------------------------------------------
# The cost model -- the easiest thing to get wrong
# ---------------------------------------------------------------------------


def test_at_mid_adds_spread_back_rather_than_subtracting_it():
    """realized is ALREADY net of spread; at_mid is the zero-cost counterfactual."""
    stats = segment_stats([_row("t1", pl=-10.0, spread=25.0)])

    assert stats["realized_home"] == pytest.approx(-10.0)
    assert stats["at_mid_home"] == pytest.approx(15.0)


def test_edge_consumed_by_cost_is_distinguished_from_no_edge():
    """Two different diagnoses needing opposite fixes."""
    cost_problem = segment_stats([_row("t1", pl=-10.0, spread=25.0)])
    signal_problem = segment_stats([_row("t2", pl=-10.0, spread=2.0)])

    assert cost_problem["verdict"] == EDGE_CONSUMED_BY_COST
    assert signal_problem["verdict"] == NO_EDGE_AT_MID


def test_profitable_after_cost_verdict():
    assert segment_stats([_row("t1", pl=50.0, spread=5.0)])["verdict"] == PROFITABLE_AFTER_COST


def test_spread_share_is_none_when_there_is_no_edge_to_eat():
    """A percentage of a negative edge is meaningless -- must not report a number."""
    assert segment_stats([_row("t1", pl=-10.0, spread=2.0)])["spread_share_of_edge"] is None


def test_spread_share_computed_when_edge_is_positive():
    stats = segment_stats([_row("t1", pl=-10.0, spread=20.0)])
    assert stats["spread_share_of_edge"] == pytest.approx(2.0)


def test_empty_segment_is_insufficient_not_zero():
    assert segment_stats([])["n"] == 0
    assert segment_stats([])["verdict"] == "INSUFFICIENT"


# ---------------------------------------------------------------------------
# The out-of-sample gate -- the reason this module exists
# ---------------------------------------------------------------------------


def test_split_is_chronological_earlier_half_first():
    rows = [_row(f"t{i}", pl=1.0, entry=f"2026-07-{i+1:02d}T00:00:00Z") for i in range(6)]
    first, second = temporal_split(list(reversed(rows)))

    assert first[0]["outcome"]["entry_time"] < second[0]["outcome"]["entry_time"]
    assert len(first) == len(second) == 3


def test_sign_flip_across_halves_is_rejected_not_actionable():
    """The exact failure this module was built to prevent."""
    early = [_row(f"e{i}", pl=-50.0, spread=1.0, entry=f"2026-04-{i+1:02d}T00:00:00Z") for i in range(4)]
    late = [_row(f"l{i}", pl=+50.0, spread=1.0, entry=f"2026-07-{i+1:02d}T00:00:00Z") for i in range(4)]

    result = validate_segment(early + late)

    assert result["adequate_samples"] is True
    assert result["sign_agrees"] is False
    assert result["actionable"] is False
    assert "unstable" in result["reason"]
    assert result["stability"] == 0.0


def test_consistent_sign_across_halves_is_actionable():
    rows = [_row(f"t{i}", pl=+40.0, spread=1.0, entry=f"2026-0{4 + i // 4}-{i % 4 + 1:02d}T00:00:00Z")
            for i in range(8)]

    result = validate_segment(rows)

    assert result["sign_agrees"] is True
    assert result["actionable"] is True
    assert "validated" in result["reason"]


def test_thin_segment_is_not_actionable_even_if_signs_agree():
    """Two profitable trades is not evidence."""
    rows = [_row("t1", pl=100.0, spread=1.0, entry="2026-04-01T00:00:00Z"),
            _row("t2", pl=100.0, spread=1.0, entry="2026-07-01T00:00:00Z")]

    result = validate_segment(rows)

    assert result["adequate_samples"] is False
    assert result["actionable"] is False
    assert "need n>=" in result["reason"]


def test_report_only_promotes_validated_segments():
    winner = [_row(f"w{i}", pl=+40.0, spread=1.0, instrument="USD_JPY",
                   entry=f"2026-0{4 + i // 4}-{i % 4 + 1:02d}T00:00:00Z") for i in range(8)]
    flipper = [_row(f"a{i}", pl=-60.0, spread=1.0, instrument="USD_CHF",
                    entry=f"2026-04-{i + 1:02d}T00:00:00Z") for i in range(4)]
    flipper += [_row(f"b{i}", pl=+60.0, spread=1.0, instrument="USD_CHF",
                     entry=f"2026-07-{i + 1:02d}T00:00:00Z") for i in range(4)]

    report = segment_report(winner + flipper)
    actionable = {s["label"] for s in report["actionable_segments"] if s["dimension"] == "instrument"}

    assert "USD_JPY" in actionable
    assert "USD_CHF" not in actionable, "a sign-flipping segment must never be promoted"


def test_report_still_exposes_raw_stats_for_rejected_segments():
    """Rejected != hidden. The numbers stay readable, just not actionable."""
    flipper = [_row(f"a{i}", pl=-60.0, spread=1.0, entry=f"2026-04-{i + 1:02d}T00:00:00Z") for i in range(4)]
    flipper += [_row(f"b{i}", pl=+60.0, spread=1.0, entry=f"2026-07-{i + 1:02d}T00:00:00Z") for i in range(4)]

    report = segment_report(flipper)
    entry = report["dimensions"]["instrument"]["EUR_USD"]

    assert entry["actionable"] is False
    assert entry["overall"]["n"] == 8


def test_malformed_rows_are_skipped_not_fatal():
    report = segment_report([_row("t1", pl=10.0), {"junk": 1}])
    assert report["overall"]["n"] == 1


# ---------------------------------------------------------------------------
# Integration against the real committed ledger
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not LEDGER_PATH.exists(), reason="outcome ledger not present")
def test_real_ledger_report_is_internally_consistent():
    rows = load_ledger()
    report = segment_report(rows)
    overall = report["overall"]

    assert overall["n"] == len(rows)
    assert overall["at_mid_home"] == pytest.approx(
        overall["realized_home"] + overall["spread_home"]
    )
    # Every promoted segment must genuinely carry the actionable flag.
    for seg in report["actionable_segments"]:
        dim = report["dimensions"][seg["dimension"]][seg["label"]]
        assert dim["actionable"] is True
        assert dim["sign_agrees"] is True


# ---------------------------------------------------------------------------
# Cut-point stability: a single midpoint split can be passed by boundary luck
# ---------------------------------------------------------------------------


def test_verdict_must_survive_every_cut_point_not_just_the_midpoint():
    """The real USD_CHF failure mode, reproduced.

    Constructed so the exact midpoint agrees but neighbouring cuts do not; the
    segment must still be refused.
    """
    # 6 losers, then 1 big winner, then 6 losers -> midpoint agrees (both
    # halves negative) but cuts either side of the winner disagree.
    rows = []
    for i in range(6):
        rows.append(_row(f"a{i}", pl=-30.0, spread=1.0, entry=f"2026-04-{i + 1:02d}T00:00:00Z"))
    rows.append(_row("spike", pl=+400.0, spread=1.0, entry="2026-05-01T00:00:00Z"))
    for i in range(6):
        rows.append(_row(f"b{i}", pl=-30.0, spread=1.0, entry=f"2026-06-{i + 1:02d}T00:00:00Z"))

    result = validate_segment(rows)

    assert result["cuts_tested"] > 1, "must test more than the midpoint"
    assert result["stability"] < 1.0
    assert result["actionable"] is False
    assert "unstable" in result["reason"]


def test_stability_is_reported_so_a_near_miss_is_visible():
    rows = [_row(f"a{i}", pl=-30.0, spread=1.0, entry=f"2026-04-{i + 1:02d}T00:00:00Z") for i in range(6)]
    rows.append(_row("spike", pl=+400.0, spread=1.0, entry="2026-05-01T00:00:00Z"))
    rows += [_row(f"b{i}", pl=-30.0, spread=1.0, entry=f"2026-06-{i + 1:02d}T00:00:00Z") for i in range(6)]

    result = validate_segment(rows)

    assert 0.0 <= result["stability"] <= 1.0


def test_genuinely_stable_segment_scores_full_stability():
    rows = [_row(f"t{i}", pl=+40.0, spread=1.0, entry=f"2026-0{4 + i // 5}-{i % 5 + 1:02d}T00:00:00Z")
            for i in range(10)]

    result = validate_segment(rows)

    assert result["stability"] == 1.0
    assert result["actionable"] is True


def test_real_usd_chf_is_refused_by_the_stability_gate():
    """Regression: this segment passed the naive midpoint gate and must not."""
    from src.learning.outcome_ledger import LEDGER_PATH, load_ledger

    if not LEDGER_PATH.exists():
        pytest.skip("outcome ledger not present")
    chf = [r for r in load_ledger() if r["outcome"]["instrument"] == "USD_CHF"]
    if len(chf) < 10:
        pytest.skip("insufficient USD_CHF history in this checkout")

    result = validate_segment(chf)

    assert result["actionable"] is False, (
        "USD_CHF's at-mid edge flips sign depending on the cut point; it must "
        "never be promoted as a validated segment"
    )
