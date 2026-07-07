"""AXIOM Hedge Layer Phase 2 — portfolio exposure engine (SHADOW/ANALYSIS-ONLY).

Reproduce-then-resolve headline case: the motivating failure from the FX
incident (-$816 day) — "Long USD/CAD + USD/CHF + USD/JPY looked like 3
trades but was one USD macro bet." The engine must net these to reveal a
single large USD bucket, not three independent-looking tickets.

NO MOCKS per CLAUDE.md No-Mock Rule — real config JSON loaded from disk via
build_exposure_report()'s default arguments; pure-function inputs otherwise.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.hedge.portfolio_exposure import (  # noqa: E402
    ExposurePosition,
    build_exposure_report,
    net_beta_exposure,
    net_correlation_bucket_exposure,
    net_currency_exposure,
    net_sector_exposure,
)


# --------------------------------------------------------------------- #
# THE headline case                                                     #
# --------------------------------------------------------------------- #
def test_usd_pileup_nets_to_one_large_usd_bet():
    """3 instruments, all long, all base=USD -> one concentrated USD bucket,
    not 3 independent 1R bets."""
    positions = [
        ExposurePosition("fx", "USD_CAD", "long", 1.0),
        ExposurePosition("fx", "USD_CHF", "long", 1.0),
        ExposurePosition("fx", "USD_JPY", "long", 1.0),
    ]
    report = build_exposure_report(positions)

    assert report.net_currency_exposure["USD"] == pytest.approx(3.0)
    assert report.net_currency_exposure["CAD"] == pytest.approx(-1.0)
    assert report.net_currency_exposure["CHF"] == pytest.approx(-1.0)
    assert report.net_currency_exposure["JPY"] == pytest.approx(-1.0)
    assert not report.fail_closed

    assert report.concentration_warnings, "must flag the USD concentration"
    usd_warning = next(w for w in report.concentration_warnings if "USD" in w)
    assert "long" in usd_warning
    assert "3.00" in usd_warning
    assert "3 instruments" in usd_warning

    narrative = report.narrative()
    assert any("concentration" in line for line in narrative)


def test_usd_pileup_correlation_bucket_view_matches_currency_view():
    positions = [
        ExposurePosition("fx", "USD_CAD", "long", 1.0),
        ExposurePosition("fx", "USD_CHF", "long", 1.0),
        ExposurePosition("fx", "USD_JPY", "long", 1.0),
    ]
    report = build_exposure_report(positions)
    assert report.net_correlation_bucket_exposure["fx_currency:USD"] == pytest.approx(3.0)


# --------------------------------------------------------------------- #
# Offsetting positions must NOT be flagged as concentrated               #
# --------------------------------------------------------------------- #
def test_offsetting_fx_positions_net_down_and_no_false_concentration():
    positions = [
        ExposurePosition("fx", "EUR_USD", "long", 1.0),   # +EUR / -USD
        ExposurePosition("fx", "GBP_USD", "short", 1.0),   # -GBP / +USD
    ]
    result = net_currency_exposure(positions)
    assert result.net["USD"] == pytest.approx(0.0)
    assert result.net["EUR"] == pytest.approx(1.0)
    assert result.net["GBP"] == pytest.approx(-1.0)

    report = build_exposure_report(positions)
    assert not report.concentration_warnings, (
        "two independent 50/50 currency bets must not read as one concentrated bet")


def test_single_instrument_bucket_not_flagged_even_at_100pct_share():
    """bias_detector's own min-instrument guard, reused: a single ticket
    isn't a 'disguised' concentration no matter its share."""
    positions = [ExposurePosition("fx", "AUD_USD", "long", 5.0)]
    report = build_exposure_report(positions)
    assert report.net_currency_exposure["AUD"] == pytest.approx(5.0)
    assert not report.concentration_warnings


# --------------------------------------------------------------------- #
# Equity sector / beta / correlation-bucket netting                     #
# --------------------------------------------------------------------- #
def test_net_sector_exposure_aggregates_equity_positions_by_sector():
    positions = [
        ExposurePosition("equity", "AAPL", "long", 2.0),
        ExposurePosition("equity", "MSFT", "long", 3.0),
        ExposurePosition("equity", "JPM", "short", 1.0),
    ]
    result = net_sector_exposure(positions)
    assert result.net["Technology"] == pytest.approx(5.0)
    assert result.net["Financials"] == pytest.approx(-1.0)


def test_net_beta_exposure_sums_beta_weighted_signed_risk():
    positions = [
        ExposurePosition("equity", "AAPL", "long", 2.0),   # beta 1.25 -> +2.5
        ExposurePosition("equity", "JPM", "short", 1.0),    # beta 1.10 -> -1.10
    ]
    result = net_beta_exposure(positions)
    assert result.net_beta_home == pytest.approx(1.40)


def test_net_beta_exposure_empty_book_is_zero_not_none():
    result = net_beta_exposure([])
    assert result.net_beta_home == 0.0


def test_net_correlation_bucket_exposure_unifies_fx_and_equity_clusters():
    positions = [
        ExposurePosition("fx", "USD_CAD", "long", 1.0),
        ExposurePosition("equity", "AAPL", "long", 2.0),     # Technology -> growth_cyclical
        ExposurePosition("equity", "AMZN", "long", 1.0),     # Consumer Discretionary -> growth_cyclical
    ]
    result = net_correlation_bucket_exposure(positions)
    assert result.net["fx_currency:USD"] == pytest.approx(1.0)
    assert result.net["equity_cluster:growth_cyclical"] == pytest.approx(3.0)
    # two distinct tickers (AAPL, AMZN) feed the same cluster
    assert set(result.bucket_instruments["equity_cluster:growth_cyclical"]) == {"AAPL", "AMZN"}


# --------------------------------------------------------------------- #
# Fail-closed contract at the report level                               #
# --------------------------------------------------------------------- #
def test_unknown_instrument_marks_report_fail_closed_not_silently_zero():
    positions = [
        ExposurePosition("fx", "USD_CAD", "long", 1.0),        # resolvable
        ExposurePosition("fx", "NOT_A_REAL_PAIR", "long", 1.0),  # not
    ]
    report = build_exposure_report(positions)
    assert report.fail_closed
    assert "NOT_A_REAL_PAIR" in report.skipped_positions
    # the resolvable leg is still reported, transparently, alongside the failure
    assert report.net_currency_exposure["USD"] == pytest.approx(1.0)
    assert any("FAIL-CLOSED" in line for line in report.narrative())


def test_unknown_asset_class_fails_closed():
    positions = [ExposurePosition("commodity_future", "CL1", "long", 1.0)]
    report = build_exposure_report(positions)
    assert report.fail_closed
    assert "CL1" in report.skipped_positions


def test_empty_book_returns_zero_exposure_not_fail_closed():
    report = build_exposure_report([])
    assert not report.fail_closed
    assert report.net_currency_exposure == {}
    assert report.net_sector_exposure == {}
    assert report.net_beta_exposure == 0.0
    assert report.position_count == 0
    assert "empty book" in report.narrative()[0]


def test_candidate_position_included_in_netting():
    open_book = [
        ExposurePosition("fx", "USD_CAD", "long", 1.0),
        ExposurePosition("fx", "USD_CHF", "long", 1.0),
    ]
    candidate = ExposurePosition("fx", "USD_JPY", "long", 1.0, source="candidate")
    without_candidate = build_exposure_report(open_book)
    with_candidate = build_exposure_report(open_book, candidate=candidate)

    assert without_candidate.net_currency_exposure["USD"] == pytest.approx(2.0)
    assert with_candidate.net_currency_exposure["USD"] == pytest.approx(3.0)
    assert with_candidate.position_count == 3
    assert without_candidate.position_count == 2
    # the candidate widens the USD concentration warning to name a 3rd instrument
    assert "USD_JPY" not in " ".join(without_candidate.concentration_warnings)
    assert "USD_JPY" in " ".join(with_candidate.concentration_warnings)
