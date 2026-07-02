"""VALUE + ACCRUALS factor extension tests (offline; no mocks; L-018 discipline).

Covers, per docs/experiment-sec-edgar-value-accruals-2026-07-02.md's test
commitment:
  (a) new XBRL concept merging (CFO/Assets/SharesOutstanding, incl. the dei
      namespace read + the empirically-discovered filed-date join, not end-keyed)
  (b) accruals ratio computation
  (c) value panel causal fundamentals x price join + its PIT validator catching
      an injected lookahead
  (d) evaluate_book_wide's new return_full param not breaking existing callers

Network-touching EDGAR fetch tests are marked @pytest.mark.integration and
skip cleanly without network — never mocked (hard repo rule).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.equity.edgar_fundamentals import (
    _shares_outstanding_by_filing,
    quality_records_from_facts,
)
from src.equity.pit_quality_eval import evaluate_book_wide
from src.equity.quality_data import build_quality_panel, validate_panel_pit
from src.equity.universe import UniverseRules, UniverseSnapshot, compute_universe_hash
from src.equity.value_data import build_value_panel, panel_coverage, validate_value_panel_pit


# --------------------------------------------------------------------------- #
# (a) New XBRL concept merging: CFO / Assets / SharesOutstanding, incl. dei.
# --------------------------------------------------------------------------- #
def _synthetic_facts() -> dict:
    """A minimal, realistically-shaped companyfacts payload for ONE fiscal year.

    Mirrors the empirically-observed EDGAR shape (2026-07-02, verified live
    against AAPL/MSFT): the dei cover-page shares fact shares the SAME
    ``filed``/``accn`` as the us-gaap facts in the same 10-K, but its own
    ``end`` date is NOT the fiscal year-end.
    """
    return {
        "facts": {
            "us-gaap": {
                "Revenues": {"units": {"USD": [
                    {"end": "2020-12-31", "start": "2020-01-01", "val": 1_000_000_000.0,
                     "filed": "2021-02-15", "form": "10-K", "fp": "FY"},
                ]}},
                "CostOfRevenue": {"units": {"USD": [
                    {"end": "2020-12-31", "start": "2020-01-01", "val": 600_000_000.0,
                     "filed": "2021-02-15", "form": "10-K", "fp": "FY"},
                ]}},
                "NetIncomeLoss": {"units": {"USD": [
                    {"end": "2020-12-31", "start": "2020-01-01", "val": 150_000_000.0,
                     "filed": "2021-02-15", "form": "10-K", "fp": "FY"},
                ]}},
                "Liabilities": {"units": {"USD": [
                    {"end": "2020-12-31", "val": 400_000_000.0,
                     "filed": "2021-02-15", "form": "10-K", "fp": "FY"},
                ]}},
                "StockholdersEquity": {"units": {"USD": [
                    {"end": "2020-12-31", "val": 900_000_000.0,
                     "filed": "2021-02-15", "form": "10-K", "fp": "FY"},
                ]}},
                "NetCashProvidedByUsedInOperatingActivities": {"units": {"USD": [
                    {"end": "2020-12-31", "start": "2020-01-01", "val": 200_000_000.0,
                     "filed": "2021-02-15", "form": "10-K", "fp": "FY"},
                ]}},
                "Assets": {"units": {"USD": [
                    {"end": "2020-12-31", "val": 1_300_000_000.0,
                     "filed": "2021-02-15", "form": "10-K", "fp": "FY"},
                ]}},
            },
            "dei": {
                "EntityCommonStockSharesOutstanding": {"units": {"shares": [
                    # SAME filed date as the us-gaap facts above (co-filed in the
                    # same 10-K accession) but a DIFFERENT `end` (cover-page date,
                    # ~6 weeks after FY-end) -- the empirically-discovered shape.
                    {"end": "2021-02-01", "val": 100_000_000.0,
                     "filed": "2021-02-15", "form": "10-K", "fp": "FY"},
                ]}},
            },
        }
    }


def test_dei_shares_outstanding_keyed_by_filed_not_end():
    """The dei cover-page fact must be discoverable via its OWN filed date,
    not by matching `end` against the us-gaap concepts (which would silently
    drop it, since the cover-page `end` != fiscal year-end)."""
    facts = _synthetic_facts()
    dei = facts["facts"]["dei"]
    usgaap = facts["facts"]["us-gaap"]
    shares = _shares_outstanding_by_filing(dei, usgaap)
    assert "2021-02-15" in shares
    filed, val = shares["2021-02-15"]
    assert filed == "2021-02-15"
    assert val == 100_000_000.0


def test_shares_outstanding_falls_back_to_usgaap_when_dei_absent():
    facts = _synthetic_facts()
    del facts["facts"]["dei"]["EntityCommonStockSharesOutstanding"]
    facts["facts"]["us-gaap"]["CommonStockSharesOutstanding"] = {"units": {"shares": [
        {"end": "2020-12-31", "val": 95_000_000.0, "filed": "2021-02-15", "form": "10-K", "fp": "FY"},
    ]}}
    shares = _shares_outstanding_by_filing(facts["facts"]["dei"], facts["facts"]["us-gaap"])
    assert shares["2021-02-15"][1] == 95_000_000.0


def test_quality_records_include_cfo_assets_shares_and_raw_exports():
    facts = _synthetic_facts()
    recs = quality_records_from_facts(facts)
    assert len(recs) == 1
    r = recs[0]
    assert r["report_period"] == "2020-12-31"
    assert r["equity"] == 900_000_000.0
    assert r["net_income"] == 150_000_000.0
    assert r["shares_outstanding"] == 100_000_000.0
    # Existing fields still present (additive, not replaced).
    assert "gross_margin" in r
    assert "debt_to_equity" in r


# --------------------------------------------------------------------------- #
# (b) Accruals ratio computation.
# --------------------------------------------------------------------------- #
def test_accruals_ratio_matches_sloan_formula():
    facts = _synthetic_facts()
    recs = quality_records_from_facts(facts)
    r = recs[0]
    # accruals = (NetIncome - CFO) / Assets = (150M - 200M) / 1300M
    expected = (150_000_000.0 - 200_000_000.0) / 1_300_000_000.0
    assert r["accruals"] == pytest.approx(expected)


def test_accruals_missing_when_input_absent():
    facts = _synthetic_facts()
    del facts["facts"]["us-gaap"]["Assets"]
    recs = quality_records_from_facts(facts)
    assert "accruals" not in recs[0]  # never fabricated/zero-filled


def test_build_quality_panel_reused_unmodified_for_accruals():
    """The pre-reg's central reuse claim: build_quality_panel(components=[("accruals",
    -1.0)]) works with ZERO changes to quality_data.py for the new field."""
    dump = {
        "AAA": [{"report_period": "2020-12-31", "filed": "2021-02-15", "accruals": -0.10}],
        "BBB": [{"report_period": "2020-12-31", "filed": "2021-02-15", "accruals": 0.15}],
    }
    idx = pd.DatetimeIndex(pd.bdate_range("2020-01-01", "2021-06-30", tz="UTC")).normalize()
    accruals_components = [("accruals", -1.0)]
    panel = build_quality_panel(dump, idx, ["AAA", "BBB"], components=accruals_components)
    # independent no-lookahead re-derivation, must not raise
    validate_panel_pit(panel, dump, components=accruals_components)
    last = panel.iloc[-1]
    # AAA has LOWER accruals -> sign=-1.0 -> HIGHER (better) score than BBB.
    assert last["AAA"] > last["BBB"]


# --------------------------------------------------------------------------- #
# (c) Value panel causal join + its independent PIT validator.
# --------------------------------------------------------------------------- #
def _price_panel():
    idx = pd.DatetimeIndex(pd.bdate_range("2020-01-01", "2021-06-30", tz="UTC")).normalize()
    rng = np.random.default_rng(3)
    df = pd.DataFrame(index=idx)
    df["AAA"] = 50 * np.cumprod(1 + rng.normal(0.0003, 0.01, len(idx)))
    df["BBB"] = 50 * np.cumprod(1 + rng.normal(0.0003, 0.01, len(idx)))
    return df


def _value_dump():
    return {
        "AAA": [{"report_period": "2020-12-31", "filed": "2021-02-15",
                 "equity": 900_000_000.0, "net_income": 150_000_000.0,
                 "shares_outstanding": 100_000_000.0}],
        "BBB": [{"report_period": "2020-12-31", "filed": "2021-02-15",
                 "equity": 50_000_000.0, "net_income": 5_000_000.0,
                 "shares_outstanding": 100_000_000.0}],
    }


def test_value_panel_nan_before_filing_available():
    dump = _value_dump()
    prices = _price_panel()
    panel = build_value_panel(dump, prices, prices.index, ["AAA", "BBB"])
    cutoff = pd.Timestamp("2021-02-15", tz="UTC")
    assert panel.loc[panel.index < cutoff].isna().all().all()
    assert panel.loc[panel.index >= cutoff].notna().all().all()


def test_value_panel_market_cap_join_is_causal_and_ordering_correct():
    # AAA: equity=900M, NI=150M, shares=100M -> at price P, book_to_market and
    # earnings_yield are both HIGHER for AAA than BBB (BBB: equity=50M, NI=5M,
    # shares=100M) whenever prices are similar -> AAA should score higher (cheaper).
    dump = _value_dump()
    prices = _price_panel()
    panel = build_value_panel(dump, prices, prices.index, ["AAA", "BBB"])
    last = panel.iloc[-1]
    assert last["AAA"] > last["BBB"]


def test_value_panel_validator_passes_on_clean_panel():
    dump = _value_dump()
    prices = _price_panel()
    panel = build_value_panel(dump, prices, prices.index, ["AAA", "BBB"])
    validate_value_panel_pit(panel, dump)  # must not raise


def test_value_panel_validator_catches_planted_lookahead():
    dump = _value_dump()
    prices = _price_panel()
    panel = build_value_panel(dump, prices, prices.index, ["AAA", "BBB"])
    # Inject a score before the filing was public (2021-02-15).
    panel.loc[pd.Timestamp("2020-01-02", tz="UTC"), "AAA"] = 1.0
    with pytest.raises(ValueError, match="PIT violation"):
        validate_value_panel_pit(panel, dump)


def test_value_panel_coverage_reflects_active_cells():
    dump = _value_dump()
    prices = _price_panel()
    members = ["AAA", "BBB"]
    panel = build_value_panel(dump, prices, prices.index, members)
    membership = pd.DataFrame(True, index=prices.index, columns=members)
    cov = panel_coverage(panel, membership)
    assert 0.0 < cov < 1.0  # partially covered: pre-filing window is NaN


def test_value_panel_market_cap_non_positive_is_nan_not_fabricated():
    dump = _value_dump()
    prices = _price_panel()
    prices2 = prices.copy()
    prices2["AAA"] = 0.0  # non-positive price -> non-positive market cap -> NaN, not inf/garbage
    panel = build_value_panel(dump, prices2, prices2.index, ["AAA", "BBB"])
    assert panel["AAA"].isna().all()


# --------------------------------------------------------------------------- #
# (d) evaluate_book_wide's new return_full param — no regression for existing
# callers (default False keeps output shape identical to before this change).
# --------------------------------------------------------------------------- #
def _staggered_prices(n=300):
    idx = pd.DatetimeIndex(pd.bdate_range("2018-01-01", periods=n, tz="UTC")).normalize()
    rng = np.random.default_rng(11)
    df = pd.DataFrame(index=idx)
    df["FULL"] = 100 * np.cumprod(1 + rng.normal(0.0004, 0.01, n))
    df["OTHER"] = 100 * np.cumprod(1 + rng.normal(0.0003, 0.011, n))
    return df


def _snap(prices):
    members = list(prices.columns)
    membership = pd.DataFrame(prices.notna().values, index=prices.index, columns=members)
    return UniverseSnapshot(
        membership=membership, rules=UniverseRules(top_n=len(members)),
        built_at="2026-07-02T00:00:00+00:00", pipeline_version="test",
        universe_hash=compute_universe_hash(membership),
    )


def _ew_weights(snapshot, prices):
    active = prices.notna()
    n = active.sum(axis=1).replace(0, np.nan)
    return active.astype(float).div(n, axis=0).fillna(0.0)


def test_evaluate_book_wide_default_unchanged_shape():
    px = _staggered_prices()
    snap = _snap(px)
    res = evaluate_book_wide(snap, px, _ew_weights(snap, px))
    assert "_returns" not in res  # default False -> no new key, backward-compatible
    assert set(res.keys()) == {"net_sharpe", "max_dd", "positive_years", "total_years", "gate_pass"}


def test_evaluate_book_wide_return_full_attaches_returns_series():
    px = _staggered_prices()
    snap = _snap(px)
    res = evaluate_book_wide(snap, px, _ew_weights(snap, px), return_full=True)
    assert "_returns" in res
    assert isinstance(res["_returns"], pd.Series)
    assert len(res["_returns"]) > 0


# --------------------------------------------------------------------------- #
# Integration (live EDGAR network) — skip cleanly without network, no mocks.
# --------------------------------------------------------------------------- #
@pytest.mark.integration
def test_live_edgar_dei_shares_outstanding_for_aapl():
    """Live confirmation of the empirical finding this module's fix is based on."""
    requests = pytest.importorskip("requests")
    try:
        sess = requests.Session()
        resp = sess.get(
            "https://data.sec.gov/api/xbrl/companyfacts/CIK0000320193.json",
            headers={"User-Agent": "ml_engine-research dcertan84@gmail.com"},
            timeout=(5, 30),
        )
        resp.raise_for_status()
        facts = resp.json()
    except Exception as exc:  # pragma: no cover - network-dependent
        pytest.skip(f"EDGAR network unavailable: {exc}")
    recs = quality_records_from_facts(facts)
    assert len(recs) > 5
    with_shares = [r for r in recs if r.get("shares_outstanding") is not None]
    assert len(with_shares) > 5  # dei join actually resolves for a real large filer
