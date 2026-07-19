"""Strategy-owned forward lanes (2026-07-18 readiness report, step 2).

No mocks: synthetic price/funding/eligibility panels (real pandas), real
frozen harness functions, real JSONL ledgers on real disk (tmp_path).

Locks in:
  1. crypto_ts_trend (H5) net series is BIT-FOR-BIT the pre-registered
     ``ts_trend_backtest`` — the lane can never drift from the frozen spec;
  2. H5 book semantics: sign-of-30d-trend direction, gross normalization,
     leverage cap, ledger record + duplicate-asof guard, H5 manifest (a
     DIFFERENT construction from the H4 crypto_momentum lane);
  3. multi_asset_trend net series is BIT-FOR-BIT ``combined_portfolio`` at
     the pre-registered target_vol=0.10; held-state matches the frozen
     single-asset rule; ledger roundtrip compounds forward-only returns;
  4. both lanes are registered in the hedge/portfolio registry and their
     loaders read real ledger rows (missing ledger -> None, fail-closed).
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from src.crypto import ts_trend_shadow as ts
from src.equity import multi_asset_trend_lane as lane
from src.equity.multi_asset_trend import single_asset_trend_returns


# ── synthetic panels ────────────────────────────────────────────────────────

def _crypto_panels(n_coins=12, n_days=140, seed=7):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2026-01-01", periods=n_days, freq="D", tz="UTC")
    cols = [f"C{i}USDT" for i in range(n_coins)]
    # half steady uptrends, half steady downtrends + noise
    drifts = np.where(np.arange(n_coins) % 2 == 0, 0.004, -0.004)
    rets = rng.normal(0, 0.01, size=(n_days, n_coins)) + drifts
    close = pd.DataFrame(100 * np.exp(np.cumsum(rets, axis=0)), index=idx, columns=cols)
    funding = pd.DataFrame(0.0, index=idx, columns=cols)
    eligible = pd.DataFrame(True, index=idx, columns=cols)
    return close, funding, eligible, cols


def _equity_panel(n_days=420, seed=3):
    idx = pd.date_range("2024-06-01", periods=n_days, freq="B")
    rng = np.random.default_rng(seed)
    up = 100 * np.exp(np.cumsum(rng.normal(0.0012, 0.008, n_days)))    # trending up
    up2 = 50 * np.exp(np.cumsum(rng.normal(0.0010, 0.010, n_days)))    # trending up
    down = 80 * np.exp(np.cumsum(rng.normal(-0.0015, 0.009, n_days)))  # trending down
    return pd.DataFrame({"UPA": up, "UPB": up2, "DWN": down}, index=idx)


# ── 1-2. crypto H5 lane ─────────────────────────────────────────────────────

def test_h5_net_series_matches_frozen_harness():
    import experiment_crypto_round2 as r2
    close, funding, eligible, cols = _crypto_panels()
    out, _book = ts._compute_book_and_returns(close, funding, eligible, cols)
    frozen = r2.ts_trend_backtest(close, funding, eligible, cols,
                                  cost_bps=ts.COST_BPS, vol_target=True)
    pd.testing.assert_series_equal(out["net"], frozen["net"], check_exact=True)
    pd.testing.assert_series_equal(out["cost"], frozen["cost"], check_exact=True)


def test_h5_book_direction_normalization_and_cap():
    close, funding, eligible, cols = _crypto_panels()
    _out, book = ts._compute_book_and_returns(close, funding, eligible, cols)
    row = book.iloc[-1]
    ups = [c for i, c in enumerate(cols) if i % 2 == 0]
    downs = [c for i, c in enumerate(cols) if i % 2 == 1]
    assert all(row[c] > 0 for c in ups), "uptrending coins must be held long"
    assert all(row[c] < 0 for c in downs), "downtrending coins must be held short"
    # unlevered gross = 1.0 -> levered gross <= MAX_LEV on every date
    assert (book.abs().sum(axis=1) <= ts.MAX_LEV + 1e-9).all()


def test_h5_manifest_is_the_frozen_h5_not_h4():
    m = ts.construction_manifest()
    assert m["signal"] == "ts_trend_sign_30d"
    assert m["rebalance_days"] == 7 and m["cost_bps"] == 10.0
    assert m["vol_target_ann"] == 0.10 and m["max_leverage"] == 3.0
    assert m["dollar_neutral"] is False
    assert "h5" in m["source_doc"].lower()
    from src.crypto.momentum_shadow import construction_manifest as h4_manifest
    assert m["signal"] != h4_manifest()["signal"], "H5 must never overwrite/alias H4"


def test_h5_ledger_record_and_duplicate_guard(tmp_path):
    close, funding, eligible, cols = _crypto_panels()
    out, book = ts._compute_book_and_returns(close, funding, eligible, cols)
    row = book.iloc[-1]
    res = ts.ShadowCycleResult(
        asof_date="2026-05-20", universe_size=len(cols),
        longs={c: float(w) for c, w in row.items() if w > 0},
        shorts={c: float(w) for c, w in row.items() if w < 0},
        gross_leverage=float(row.abs().sum()),
        today_net_return=float(out["net"].iloc[-1]),
        today_price_return=float(out["price"].iloc[-1]),
        today_carry_return=float(out["carry"].iloc[-1]),
        today_cost=float(out["cost"].iloc[-1]),
        today_turnover=float(out["turnover"].iloc[-1]),
        construction=ts.construction_manifest(),
    )
    ledger = tmp_path / "shadow_ts_trend_ledger.jsonl"
    first = ts.record_shadow_cycle(res, ledger_path=ledger, cycle_ts_iso="2026-05-20T00:00:00+00:00")
    assert first is not None and first["forward_cycle_seq"] == 1
    assert first["construction"]["signal"] == "ts_trend_sign_30d"
    # duplicate asof -> honest no-op, ledger unchanged
    assert ts.record_shadow_cycle(res, ledger_path=ledger,
                                  cycle_ts_iso="2026-05-21T00:00:00+00:00") is None
    assert len(ledger.read_text().strip().splitlines()) == 1
    summary = ts.forward_oos_summary(ledger_path=ledger)
    assert summary["n_cycles"] == 1 and summary["forward_sharpe_annualized"] is None


# ── 3. multi-asset trend lane ───────────────────────────────────────────────

def test_trend_lane_net_matches_frozen_portfolio_and_book_is_causal():
    prices = _equity_panel()
    cycle = lane.compute_lane_cycle(prices)
    frozen_net = lane.FROZEN_PORTFOLIO(prices, target_vol=lane.TARGET_VOL)
    assert cycle.today_net_return == float(frozen_net.iloc[-1]), \
        "lane mark must be bit-for-bit the pre-registered portfolio's latest bar"
    assert cycle.asof_date == str(frozen_net.index[-1].date())
    # long-or-flat: no shorts; downtrending asset must NOT be in the book
    assert all(w > 0 for w in cycle.longs.values())
    assert "DWN" not in cycle.longs
    assert cycle.gross_leverage <= lane.MAX_LEV + 1e-9
    assert cycle.universe_size == 3


def test_trend_lane_held_state_matches_frozen_single_asset_rule():
    prices = _equity_panel()
    for col in prices.columns:
        held = lane._held_state(prices[col])
        assert held is not None
        # reconstruct the frozen net stream from the held state and compare
        px = prices[col]
        applied = held.shift(1).fillna(0.0)
        gross = applied * px.pct_change()
        turnover = applied.diff().abs().fillna(applied.abs())
        net = (gross - turnover * (lane.COST_BPS_PER_SIDE * 1e-4)).dropna()
        frozen = single_asset_trend_returns(px)
        pd.testing.assert_series_equal(net, frozen, check_exact=True)


def test_trend_lane_manifest_freezes_the_preregistered_spec():
    m = lane.construction_manifest()
    assert m["signal"] == "price_gt_sma200_long_or_flat"
    assert m["step_days"] == 21 and m["cost_bps_per_side"] == 2.0
    assert m["vol_target_ann"] == 0.10, "pre-registered 0.10, NOT the code default 0.12"
    assert m["max_leverage"] == 3.0 and m["universe_size"] == 37
    assert m["universe_sha256"] == lane.universe_sha256()
    assert len(m["universe_sha256"]) == 64


def test_trend_lane_ledger_roundtrip_and_duplicate_guard(tmp_path):
    prices = _equity_panel()
    cycle = lane.compute_lane_cycle(prices)
    ledger = tmp_path / "shadow_multi_asset_trend_ledger.jsonl"
    row = lane.record_lane_cycle(cycle, ledger_path=ledger,
                                 cycle_ts_iso="2026-07-18T00:00:00+00:00")
    assert row is not None and row["forward_cycle_seq"] == 1
    assert row["n_shorts"] == 0 and row["orders_placed"] == 0 and row["broker"] is None
    assert row["cumulative_shadow_return"] == pytest.approx(cycle.today_net_return)
    assert lane.record_lane_cycle(cycle, ledger_path=ledger,
                                  cycle_ts_iso="2026-07-19T00:00:00+00:00") is None
    persisted = [json.loads(ln) for ln in ledger.read_text().splitlines() if ln.strip()]
    assert len(persisted) == 1
    assert persisted[0]["construction"]["universe_sha256"] == lane.universe_sha256()
    s = lane.forward_summary(ledger_path=ledger)
    assert s["n_cycles"] == 1 and s["forward_sharpe_annualized"] is None


# ── 4. registry integration ─────────────────────────────────────────────────

def test_registry_covers_both_new_lanes(tmp_path):
    from src.hedge import hedged_shadow_lane as hsl
    assert "crypto_ts_trend" in hsl.STRATEGIES and "crypto_ts_trend" in hsl._LOADERS
    assert "multi_asset_trend" in hsl.STRATEGIES and "multi_asset_trend" in hsl._LOADERS

    # missing ledgers -> fail-closed None, never a fabricated book
    assert hsl.load_crypto_ts_trend_book(ledger_path=tmp_path / "missing.jsonl") is None
    assert hsl.load_multi_asset_trend_book(ledger_path=tmp_path / "missing.jsonl") is None

    ts_ledger = tmp_path / "ts.jsonl"
    ts_ledger.write_text(json.dumps({
        "asof_date": "2026-05-20", "today_net_return": 0.004, "today_cost": 0.0002,
        "book": {"longs": {"BTCUSDT": 0.5}, "shorts": {"ADAUSDT": -0.25}},
        "gross_leverage": 0.75, "construction": {"signal": "ts_trend_sign_30d"},
    }) + "\n")
    book = hsl.load_crypto_ts_trend_book(ledger_path=ts_ledger)
    assert book is not None and book.asset_class == "crypto"
    assert book.strategy == "crypto_ts_trend" and book.return_source == "own_ledger"
    assert book.weights == {"BTCUSDT": 0.5, "ADAUSDT": -0.25}
    assert book.raw_net_return == pytest.approx(0.004)

    tr_ledger = tmp_path / "trend.jsonl"
    tr_ledger.write_text(json.dumps({
        "asof_date": "2026-07-17", "today_net_return": 0.001,
        "book": {"longs": {"SPY": 0.2, "GLD": 0.1}, "shorts": {}},
        "gross_leverage": 0.3, "overlay_leverage": 1.1,
        "construction": {"universe_sha256": lane.universe_sha256()},
    }) + "\n")
    book2 = hsl.load_multi_asset_trend_book(ledger_path=tr_ledger)
    assert book2 is not None and book2.asset_class == "multi_asset"
    assert book2.weights == {"SPY": 0.2, "GLD": 0.1}
    assert book2.meta["overlay_leverage"] == 1.1


def test_promotion_report_universally_covers_seven_lanes(tmp_path):
    from src.hedge.hedged_shadow_lane import STRATEGIES
    from src.hedge.portfolio_promotion import VERDICT_SHADOW, build_portfolio_promotion_report
    report = build_portfolio_promotion_report(
        ledger_path=tmp_path / "empty_ledger.jsonl",
        scorecard_path=tmp_path / "no_scorecard.json",
        out_path=tmp_path / "report.json",
        allocations_path=tmp_path / "no_allocations.json")
    assert set(report["verdicts"]) == set(STRATEGIES)
    assert len(STRATEGIES) == 7
    for s in STRATEGIES:
        assert report["verdicts"][s]["verdict"] == VERDICT_SHADOW
