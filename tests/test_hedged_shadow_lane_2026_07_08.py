"""AXIOM Hedge Layer Phase 4 — raw-vs-hedged shadow lane (SHADOW/ANALYSIS-ONLY).

Reproduce-then-resolve tests, mirroring Phases 1-3's own discipline:

* a PURE-BETA book (both names move identically to the "market") ->
  hedging collapses gross expectancy toward/through zero -- the return
  doesn't survive losing its market exposure.
* a GENUINELY-RELATIVE (near market-neutral) book -> the market hedge's
  risk_home collapses to ~0, so the hedge overlay barely touches the raw
  return -- hedged tracks raw, i.e. "the signal holds".
* an UNKNOWN-COST equity hedge (SPY/sector-ETF, no OANDA fill history) ->
  GROSS overlay P&L is always computed and shown; NET is honestly None,
  never defaulted to gross or zero.

NO MOCKS per CLAUDE.md No-Mock Rule: real config JSON via default-argument
disk loads (``load_bucket_map``), real ``ExposurePosition``/
``ExposureReport``/hedge-report machinery (Phases 1-3, not forked), real
``estimate_execution_cost`` (which genuinely has no OANDA fills for SPY/
sector ETFs -- that is the real model's real behavior, not a fake). Only
the PRICE PANEL is a small in-memory ``pandas.DataFrame`` built directly by
the test (deterministic, no network, no parquet fixture needed) -- this is
real data the test authored, not a mock/stub of any production object; the
``book=`` injection point on ``run_cycle_for_strategy`` exists for exactly
this purpose (see its docstring).
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.hedge import hedged_shadow_lane as hsl  # noqa: E402
from src.hedge.exposure_tags import (  # noqa: E402
    CURRENCY_BUCKET_MAP_PATH,
    SECTOR_BUCKET_MAP_PATH,
    load_bucket_map,
)

REAL_SECTOR_MAP = load_bucket_map(SECTOR_BUCKET_MAP_PATH)
REAL_CURRENCY_MAP = load_bucket_map(CURRENCY_BUCKET_MAP_PATH)


def _steady_market_panel(daily_return: float = 0.01, n_days: int = 8) -> pd.DataFrame:
    """AAPL, MSFT and NVDA all move IDENTICALLY every day -- an idealized
    pure-beta universe. All three are already resolvable in the real sector
    map (Technology, benchmark_hedge=XLK); three names clears
    ``MIN_SECTOR_PROXY_TICKERS`` so the market proxy (curated-universe
    average, see ``market_proxy_return``) actually resolves. Only AAPL+MSFT
    are held in the book -- NVDA exists purely to give the market-proxy
    average enough curated names to average over."""
    dates = pd.date_range("2026-01-01", periods=n_days, freq="D", tz="UTC")
    prices = [100.0]
    for _ in range(n_days - 1):
        prices.append(prices[-1] * (1.0 + daily_return))
    return pd.DataFrame({"AAPL": prices, "MSFT": prices, "NVDA": prices}, index=dates)


def _run_history(strategy: str, weights: dict, panel: pd.DataFrame) -> list:
    rows = []
    dates = list(panel.index)
    for i in range(len(dates) - 1):
        asof = str(dates[i].date())
        book = hsl.BookSnapshot(
            strategy=strategy, asset_class="equity", asof_date=asof,
            weights=weights, raw_net_return=None, raw_gross_return=None, raw_cost=None,
            return_source="must_compute", meta={})
        row = hsl.run_cycle_for_strategy(
            strategy, price_panel=panel, sector_bucket_map=REAL_SECTOR_MAP,
            currency_bucket_map=REAL_CURRENCY_MAP, persist=False, book=book)
        assert row is not None
        rows.append(row)
    return rows


# --------------------------------------------------------------------- #
# Case 1: pure beta -- hedge should remove (most of) the raw return      #
# --------------------------------------------------------------------- #
def test_pure_beta_book_hedge_collapses_gross_expectancy():
    panel = _steady_market_panel(daily_return=0.01)
    weights = {"AAPL": 0.5, "MSFT": 0.5}  # 100% long, fully mapped -> book fully resolves
    rows = _run_history("equity_harvester", weights, panel)

    raw_returns = [r["raw"]["net_return"] for r in rows]
    hedged_gross = [r["hedged"]["gross_return"] for r in rows]

    assert all(r == pytest.approx(0.01) for r in raw_returns), "raw book IS the market by construction"
    assert not any(r["exposure"]["fail_closed"] for r in rows), "AAPL+MSFT are both in the real sector map"
    assert all(r["hedge"]["status"] == "applied" for r in rows)
    assert all(r["hedge"]["applied_proposal"]["style"] == "market" for r in rows)

    # The market hedge is beta-weighted (net_beta_home), not price-weighted,
    # so it won't cancel a raw price return to the exact cent -- but it must
    # remove the overwhelming majority of it.
    for raw, hedged in zip(raw_returns, hedged_gross):
        assert abs(hedged) < abs(raw) * 0.5, (
            f"hedge should remove most of the pure-beta return: raw={raw} hedged_gross={hedged}")


def test_pure_beta_book_cost_is_honestly_unknown_never_zero_filled():
    panel = _steady_market_panel(daily_return=0.01)
    weights = {"AAPL": 0.5, "MSFT": 0.5}
    rows = _run_history("equity_harvester", weights, panel)
    for row in rows:
        assert row["hedge"]["overlay_cost_known"] is False, (
            "no OANDA fills exist for SPY -- cost model must report unknown, not fabricate a value")
        assert row["hedge"]["overlay_cost_dollars"] is None
        assert row["hedged"]["net_return"] is None, "net must stay None when cost is unknown"
        assert row["hedged"]["gross_return"] is not None, "gross must still be computed and shown"
        assert row["hedged"]["return_basis"] == "gross_cost_unknown"


# --------------------------------------------------------------------- #
# Case 2: genuinely relative (near market-neutral) -- hedged tracks raw  #
# --------------------------------------------------------------------- #
def test_market_neutral_book_hedge_is_near_noop_signal_holds():
    dates = pd.date_range("2026-01-01", periods=8, freq="D", tz="UTC")
    aapl = [100.0]
    msft = [100.0]
    # nvda not held in the book -- only present so the curated market-proxy
    # universe (MIN_SECTOR_PROXY_TICKERS=3) resolves
    nvda = [100.0]
    daily_moves = [0.02, -0.01, 0.015, -0.03, 0.01, -0.005, 0.02]
    for m in daily_moves:
        aapl.append(aapl[-1] * (1 + m))
        msft.append(msft[-1] * (1 + m * 0.6))  # correlated but not identical
        nvda.append(nvda[-1] * (1 + m * 1.4))
    panel = pd.DataFrame({"AAPL": aapl, "MSFT": msft, "NVDA": nvda}, index=dates)

    # AAPL beta=1.25, MSFT beta=1.10 (real sector map values) -- size the
    # short leg to null the book's net BETA-weighted exposure, not its
    # dollar exposure. This is the "genuinely relative" construction: the
    # book expresses a view on AAPL-vs-MSFT, not a market direction bet.
    weights = {"AAPL": 0.5, "MSFT": -0.5 * (1.25 / 1.10)}
    rows = _run_history("track_b", weights, panel)

    for row in rows:
        assert abs(row["exposure"]["net_beta_exposure"]) < 1e-6, "book must be beta-neutral by construction"
        raw = row["raw"]["net_return"]
        hedged = row["hedged"]["gross_return"]
        assert hedged == pytest.approx(raw, abs=1e-6), (
            f"near-zero net beta -> hedge overlay must be negligible: raw={raw} hedged_gross={hedged}")


# --------------------------------------------------------------------- #
# Fail-closed / degrade paths                                           #
# --------------------------------------------------------------------- #
def test_missing_book_returns_none_never_a_fabricated_row(tmp_path):
    result = hsl.load_equity_harvester_book(state_path=tmp_path / "does_not_exist.json")
    assert result is None
    result2 = hsl.load_crypto_momentum_book(ledger_path=tmp_path / "does_not_exist.jsonl")
    assert result2 is None
    result3 = hsl.load_track_b_book(ledger_path=tmp_path / "does_not_exist.jsonl")
    assert result3 is None


def test_crypto_asset_class_is_unsupported_and_fails_closed_not_fabricated():
    panel = pd.DataFrame()  # crypto never touches the equity price panel
    book = hsl.BookSnapshot(
        strategy="crypto_momentum", asset_class="crypto", asof_date="2026-01-01",
        weights={"BTCUSDT": 0.3, "ETHUSDT": -0.3},
        raw_net_return=0.004, raw_gross_return=0.005, raw_cost=0.001,
        return_source="own_ledger", meta={})
    row = hsl.run_cycle_for_strategy(
        "crypto_momentum", price_panel=panel, sector_bucket_map=REAL_SECTOR_MAP,
        currency_bucket_map=REAL_CURRENCY_MAP, persist=False, book=book)
    assert row is not None
    assert row["hedge"]["status"] == "unsupported_asset_class"
    assert row["hedge"]["applied_proposal"] is None
    assert row["hedged"]["net_return"] == pytest.approx(0.004), "no hedge applied -> hedged == raw exactly"
    assert row["runtime_allowed"] is False
    assert row["paper_only"] is True
    assert row["human_review_required"] is True


def test_stale_price_panel_reports_insufficient_data_not_zero():
    """asof_date past the panel's last bar -> no forward mark exists. Must
    report None, never silently treat "no data" as "flat/zero return"."""
    panel = _steady_market_panel(daily_return=0.01, n_days=3)
    last_date = str(panel.index[-1].date())
    book = hsl.BookSnapshot(
        strategy="equity_harvester", asset_class="equity", asof_date=last_date,
        weights={"AAPL": 0.5, "MSFT": 0.5}, raw_net_return=None, raw_gross_return=None,
        raw_cost=None, return_source="must_compute", meta={})
    row = hsl.run_cycle_for_strategy(
        "equity_harvester", price_panel=panel, sector_bucket_map=REAL_SECTOR_MAP,
        currency_bucket_map=REAL_CURRENCY_MAP, persist=False, book=book)
    assert row["raw"]["net_return"] is None
    assert any(n.startswith("insufficient_forward_price_data") for n in row["raw"]["notes"])
    assert row["hedged"]["gross_return"] is None
    assert row["hedged"]["return_basis"] == "unresolved"


def test_partial_ticker_coverage_fails_the_whole_basket_return_closed():
    """One resolvable ticker + one unresolvable ticker must NOT return a
    number computed only over the resolved leg -- a weighted sum over a
    subset is mathematically identical to zero-filling the missing leg's
    contribution, which is exactly the silent zero-fill this project's
    rules forbid. The whole basket return must fail closed (None), same
    all-or-nothing contract as ``build_exposure_report``."""
    panel = _steady_market_panel(daily_return=0.02, n_days=3)  # only has AAPL/MSFT/NVDA
    weights = {"AAPL": 0.5, "GOOGL": 0.5}  # GOOGL has no price data in this panel
    ret, notes = hsl.basket_forward_return(panel, weights, str(panel.index[0].date()))
    assert ret is None, "must fail closed, not silently return the AAPL-only weighted contribution"
    assert any(n.startswith("insufficient_forward_price_data") for n in notes)
    assert any("GOOGL" in n for n in notes)


# --------------------------------------------------------------------- #
# Safety triplet + reuse contract                                       #
# --------------------------------------------------------------------- #
def test_safety_triplet_is_fixed_never_overridable():
    assert hsl.RUNTIME_ALLOWED is False
    assert hsl.PAPER_ONLY is True
    assert hsl.HUMAN_REVIEW_REQUIRED is True
    # No parameter on run_cycle_for_strategy / run_all can flip these --
    # verified structurally: the triplet is written from the module
    # constants directly in the row dict, never from a function arg.
    import inspect
    sig = inspect.signature(hsl.run_cycle_for_strategy)
    assert "runtime_allowed" not in sig.parameters
    assert "paper_only" not in sig.parameters
    assert "human_review_required" not in sig.parameters


def test_positions_from_weights_skips_zero_weight_and_signs_direction():
    positions = hsl.positions_from_weights({"AAPL": 0.5, "MSFT": -0.3, "FLAT": 0.0}, "equity", notional=1000.0)
    by_inst = {p.instrument: p for p in positions}
    assert "FLAT" not in by_inst
    assert by_inst["AAPL"].direction == "long"
    assert by_inst["AAPL"].risk_home == pytest.approx(500.0)
    assert by_inst["MSFT"].direction == "short"
    assert by_inst["MSFT"].risk_home == pytest.approx(300.0)


def test_ledger_paths_are_scoped_to_trained_data_hedge():
    assert hsl.RAW_VS_HEDGED_LEDGER_PATH.parent == hsl.HEDGE_LEDGER_DIR
    assert hsl.HEDGE_DECISION_LOG_PATH.parent == hsl.HEDGE_LEDGER_DIR
    assert "trained_data/hedge" in str(hsl.RAW_VS_HEDGED_LEDGER_PATH)


def test_no_execution_or_broker_import_anywhere_in_module():
    """Grep-level guarantee, mirrors Phase 3's own verification pattern:
    this module must never import a broker/execution/halt-control path."""
    source = Path(hsl.__file__).read_text(encoding="utf-8")
    forbidden = ("import src.brokers", "from src.brokers", "execution.py",
                 "place_market_order", "place_limit_order", "StateEngine",
                 "set_halted", "LiveGate.arm")
    for token in forbidden:
        assert token not in source, f"forbidden execution/control token found: {token}"
