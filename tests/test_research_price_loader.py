"""Tests for src.equity.research.price_loader.

Two tiers, per the No-Mock rule (docs/incidents.md "No-Mock catastrophe"):

* OFFLINE: pure JSON->series parsing (``parse_chart_payload``) against synthetic
  chart dicts, including a null-adjclose entry that must be dropped in lockstep.
  No network, no mocks — just a real dict fed to a pure function.
* INTEGRATION: a REAL network call to the Yahoo chart API (Yahoo is reachable
  through the agent proxy CA bundle). Marked ``@pytest.mark.integration`` and
  SKIPPED CLEANLY (``pytest.skip``) if Yahoo is unreachable / non-200, so the
  suite never hangs or fails on transient outage.
"""

from __future__ import annotations

import os

import pandas as pd
import pytest

from src.equity.research.price_loader import (
    PriceLoaderError,
    load_price_panel,
    parse_chart_payload,
)


# --------------------------------------------------------------------------- #
# OFFLINE: pure parse                                                          #
# --------------------------------------------------------------------------- #

def _synthetic_payload(timestamps, adjcloses):
    return {
        "chart": {
            "result": [
                {
                    "timestamp": timestamps,
                    "indicators": {"adjclose": [{"adjclose": adjcloses}]},
                }
            ],
            "error": None,
        }
    }


def test_parse_drops_null_adjclose_in_lockstep():
    # 1672531200 = 2023-01-01 00:00 UTC, +1d steps. Middle bar is null -> drop.
    ts = [1672531200, 1672617600, 1672704000]
    adj = [100.0, None, 102.5]
    dates, closes = parse_chart_payload(_synthetic_payload(ts, adj))

    assert len(dates) == len(closes) == 2  # null bar dropped
    assert closes == [100.0, 102.5]
    # Timestamps floored to UTC midnight, and the null date is gone.
    assert all(isinstance(d, pd.Timestamp) for d in dates)
    assert all(d.tzinfo is not None for d in dates)
    assert all(d == d.normalize() for d in dates)
    assert dates[0] == pd.Timestamp("2023-01-01", tz="UTC")
    assert dates[1] == pd.Timestamp("2023-01-03", tz="UTC")


def test_parse_floors_intraday_epoch_to_trading_date():
    # Yahoo returns market-open intraday epochs, not midnight.
    intraday = 1672756200  # 2023-01-03 14:30 UTC (NYSE open)
    dates, closes = parse_chart_payload(_synthetic_payload([intraday], [123.45]))
    assert dates == [pd.Timestamp("2023-01-03", tz="UTC")]
    assert closes == [123.45]


@pytest.mark.parametrize(
    "payload",
    [
        {},
        {"chart": None},
        {"chart": {"result": None}},
        {"chart": {"result": []}},
        {"chart": {"result": [{"timestamp": None, "indicators": {}}]}},
        {"chart": {"result": [{"timestamp": [1672531200],
                               "indicators": {"adjclose": [{"adjclose": None}]}}]}},
        {"chart": {"result": [{"timestamp": [1672531200],
                               "indicators": {"adjclose": [{"adjclose": [None]}]}}]}},
    ],
)
def test_parse_returns_empty_on_unusable_payloads(payload):
    dates, closes = parse_chart_payload(payload)
    assert dates == []
    assert closes == []


def test_parse_handles_non_dict():
    assert parse_chart_payload(None) == ([], [])
    assert parse_chart_payload("garbage") == ([], [])


# --------------------------------------------------------------------------- #
# OFFLINE: caller-fault input validation                                       #
# --------------------------------------------------------------------------- #

def test_load_rejects_empty_tickers():
    with pytest.raises(PriceLoaderError):
        load_price_panel([], "2023-01-01", "2024-01-01")


def test_load_rejects_inverted_window():
    with pytest.raises(PriceLoaderError):
        load_price_panel(["AAPL"], "2024-01-01", "2023-01-01")


def test_load_rejects_bad_iso_date():
    with pytest.raises(PriceLoaderError):
        load_price_panel(["AAPL"], "not-a-date", "2024-01-01")


# --------------------------------------------------------------------------- #
# INTEGRATION: real network                                                    #
# --------------------------------------------------------------------------- #

def _yahoo_reachable() -> bool:
    """Probe the chart endpoint; True only on a 200 with a parseable body."""
    import requests

    ca = os.environ.get("REQUESTS_CA_BUNDLE") or "/root/.ccr/ca-bundle.crt"
    url = "https://query1.finance.yahoo.com/v8/finance/chart/AAPL"
    params = {"period1": 1672531200, "period2": 1675209600, "interval": "1d"}
    try:
        r = requests.get(
            url,
            params=params,
            headers={"User-Agent": "Mozilla/5.0"},
            timeout=(5, 30),
            verify=ca if os.path.exists(ca) else True,
        )
    except requests.exceptions.RequestException:
        return False
    if r.status_code != 200:
        return False
    dates, _ = parse_chart_payload(r.json())
    return len(dates) > 0


@pytest.mark.integration
def test_load_price_panel_real_network():
    if not _yahoo_reachable():
        pytest.skip("Yahoo chart API unreachable / non-200 — skipping live test")

    panel = load_price_panel(["AAPL", "MSFT"], "2023-01-01", "2024-01-01")

    assert isinstance(panel, pd.DataFrame)
    assert list(panel.columns) == ["AAPL", "MSFT"]
    assert len(panel) > 200  # ~250 trading days in a year

    # UTC, monotonic-increasing, midnight-normalised index.
    assert isinstance(panel.index, pd.DatetimeIndex)
    assert str(panel.index.tz) == "UTC"
    assert panel.index.is_monotonic_increasing
    assert (panel.index == panel.index.normalize()).all()

    # PIT: never a row dated after `end`.
    assert panel.index.max() <= pd.Timestamp("2024-01-01", tz="UTC")

    # Adjusted closes are positive where present.
    aapl = panel["AAPL"].dropna()
    assert len(aapl) > 200
    assert (aapl > 0).all()
    assert (panel["MSFT"].dropna() > 0).all()


@pytest.mark.integration
def test_load_price_panel_drops_bad_ticker_keeps_good():
    if not _yahoo_reachable():
        pytest.skip("Yahoo chart API unreachable / non-200 — skipping live test")

    # A clearly invalid symbol must be dropped, not crash the panel.
    panel = load_price_panel(
        ["AAPL", "ZZZ_NOT_A_REAL_TICKER_XYZ"], "2023-06-01", "2023-07-01"
    )
    assert "AAPL" in panel.columns
    assert "ZZZ_NOT_A_REAL_TICKER_XYZ" not in panel.columns
    assert len(panel) > 5
