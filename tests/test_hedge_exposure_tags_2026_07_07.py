"""AXIOM Hedge Layer Phase 1 — exposure tagging (SHADOW/ANALYSIS-ONLY).

NO MOCKS per CLAUDE.md No-Mock Rule. Real config JSON loaded from disk for
the "loads for real" tests; an explicit in-memory map is passed for the
fail-closed edge cases so each test is self-contained about what it's
proving.
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.hedge.exposure_tags import (  # noqa: E402
    CORRELATION_BUCKET_MAP_PATH,
    CURRENCY_BUCKET_MAP_PATH,
    SECTOR_BUCKET_MAP_PATH,
    load_bucket_map,
    tag_equity_exposure,
    tag_fx_exposure,
)

REAL_CURRENCY_MAP = load_bucket_map(CURRENCY_BUCKET_MAP_PATH)
REAL_SECTOR_MAP = load_bucket_map(SECTOR_BUCKET_MAP_PATH)


# --------------------------------------------------------------------- #
# Config maps load for real off disk                                    #
# --------------------------------------------------------------------- #
def test_currency_bucket_map_loads_and_covers_expected_currencies():
    expected = {"USD", "EUR", "GBP", "JPY", "CHF", "CAD", "AUD", "NZD", "XAU"}
    assert expected.issubset(REAL_CURRENCY_MAP.keys())
    assert "_meta" not in REAL_CURRENCY_MAP  # stripped, not a real currency entry


def test_sector_bucket_map_loads_and_correlation_map_shares_sector_strings():
    correlation_map = load_bucket_map(CORRELATION_BUCKET_MAP_PATH)
    sectors_in_use = {entry["sector"] for entry in REAL_SECTOR_MAP.values()}
    assert sectors_in_use.issubset(correlation_map.keys()), (
        "every sector referenced by sector_bucket_map.json must have a cluster "
        "in correlation_bucket_map.json, else net_correlation_bucket_exposure "
        "silently fails closed on real tickers")


def test_load_bucket_map_missing_file_returns_empty_not_crash():
    result = load_bucket_map(Path("/nonexistent/path/does_not_exist.json"))
    assert result == {}


# --------------------------------------------------------------------- #
# FX tagging — reuses trend_risk_gates.currency_legs                    #
# --------------------------------------------------------------------- #
def test_tag_fx_exposure_long_usd_cad_signed_legs():
    tag = tag_fx_exposure("USD_CAD", "long", 1.0, REAL_CURRENCY_MAP)
    assert not tag.fail_closed
    by_ccy = {leg.currency: leg for leg in tag.legs}
    assert by_ccy["USD"].direction == "long"
    assert by_ccy["USD"].signed_home == 1.0
    assert "safe_haven" in by_ccy["USD"].tags
    assert by_ccy["CAD"].direction == "short"
    assert by_ccy["CAD"].signed_home == -1.0
    assert "commodity_linked" in by_ccy["CAD"].tags


def test_tag_fx_exposure_short_flips_both_legs():
    long_tag = tag_fx_exposure("EUR_USD", "long", 1.0, REAL_CURRENCY_MAP)
    short_tag = tag_fx_exposure("EUR_USD", "short", 1.0, REAL_CURRENCY_MAP)
    long_by_ccy = {leg.currency: leg.signed_home for leg in long_tag.legs}
    short_by_ccy = {leg.currency: leg.signed_home for leg in short_tag.legs}
    assert long_by_ccy["EUR"] == 1.0 and long_by_ccy["USD"] == -1.0
    assert short_by_ccy["EUR"] == -1.0 and short_by_ccy["USD"] == 1.0


def test_tag_fx_exposure_unresolvable_instrument_fails_closed():
    tag = tag_fx_exposure("NOT_A_PAIR_X_Y", "long", 1.0, REAL_CURRENCY_MAP)
    assert tag.fail_closed
    assert tag.legs == ()
    assert any("unresolvable_instrument" in r for r in tag.reasons)


def test_tag_fx_exposure_unknown_currency_fails_closed_not_zero():
    partial_map = {"USD": {"tags": ["safe_haven"]}}  # CAD deliberately missing
    tag = tag_fx_exposure("USD_CAD", "long", 1.0, partial_map)
    assert tag.fail_closed
    assert tag.legs == ()  # NOT a partial USD-only leg — whole tag refuses
    assert any("unknown_currency:CAD" in r for r in tag.reasons)


def test_tag_fx_exposure_invalid_direction_fails_closed():
    tag = tag_fx_exposure("USD_CAD", "sideways", 1.0, REAL_CURRENCY_MAP)
    assert tag.fail_closed
    assert any("invalid_direction" in r for r in tag.reasons)


def test_tag_fx_exposure_negative_risk_fails_closed():
    tag = tag_fx_exposure("USD_CAD", "long", -5.0, REAL_CURRENCY_MAP)
    assert tag.fail_closed
    assert any("negative_risk_home" in r for r in tag.reasons)


def test_tag_fx_exposure_nan_risk_fails_closed_not_silently_zero():
    # float("nan") < 0 is False, so a naive negative-only guard lets NaN
    # through and it flows into signed_home arithmetic downstream.
    tag = tag_fx_exposure("USD_CAD", "long", float("nan"), REAL_CURRENCY_MAP)
    assert tag.fail_closed
    assert tag.legs == ()
    assert any("non_finite_risk_home" in r for r in tag.reasons)


def test_tag_fx_exposure_infinite_risk_fails_closed_not_silently_zero():
    tag = tag_fx_exposure("USD_CAD", "long", float("inf"), REAL_CURRENCY_MAP)
    assert tag.fail_closed
    assert tag.legs == ()
    assert any("non_finite_risk_home" in r for r in tag.reasons)


# --------------------------------------------------------------------- #
# Equity tagging                                                        #
# --------------------------------------------------------------------- #
def test_tag_equity_exposure_known_ticker_resolves_sector_beta_factor():
    tag = tag_equity_exposure("AAPL", "long", 2.0, REAL_SECTOR_MAP)
    assert not tag.fail_closed
    assert tag.sector == "Technology"
    assert tag.market_beta == 1.25
    assert "growth" in tag.factor_tags
    assert tag.benchmark_hedge == "XLK"
    assert tag.signed_home == 2.0
    assert tag.signed_beta_home == 2.5


def test_tag_equity_exposure_short_flips_signed_home_and_beta():
    tag = tag_equity_exposure("AAPL", "short", 2.0, REAL_SECTOR_MAP)
    assert tag.signed_home == -2.0
    assert tag.signed_beta_home == -2.5


def test_tag_equity_exposure_unknown_ticker_fails_closed_not_zero():
    tag = tag_equity_exposure("ZZZZ_NOT_REAL", "long", 1.0, REAL_SECTOR_MAP)
    assert tag.fail_closed
    assert tag.sector is None
    assert tag.market_beta is None
    assert any("unknown_ticker:ZZZZ_NOT_REAL" in r for r in tag.reasons)


def test_tag_equity_exposure_invalid_direction_fails_closed():
    tag = tag_equity_exposure("AAPL", "flat", 1.0, REAL_SECTOR_MAP)
    assert tag.fail_closed


def test_tag_equity_exposure_nan_risk_fails_closed_not_silently_zero():
    tag = tag_equity_exposure("AAPL", "long", float("nan"), REAL_SECTOR_MAP)
    assert tag.fail_closed
    assert any("non_finite_risk_home" in r for r in tag.reasons)


def test_tag_equity_exposure_infinite_risk_fails_closed_not_silently_zero():
    tag = tag_equity_exposure("AAPL", "long", float("-inf"), REAL_SECTOR_MAP)
    assert tag.fail_closed
    assert any("non_finite_risk_home" in r for r in tag.reasons)
