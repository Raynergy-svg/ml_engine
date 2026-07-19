"""Strategy-owned forward lanes — evidence-safe recording (2026-07-19 review).

No mocks: synthetic price/funding/eligibility panels (real pandas), real
frozen harness functions, real JSONL ledgers on real disk (tmp_path).

Covers the review's required tests:
  1  capture twice on stale data -> lane ledgers AND hedge ledger unchanged;
  2  empty ledger -> activation baseline (no realized return, cum 0);
  3  one later bar -> exactly the first forward return;
  4  three later bars in one refresh -> all three, chronological;
  5  cumulative == compounding the complete unseen slice;
  6  rows align applied_book with the realized period (period fields + the
     applied book reproduces the row's price P&L);
  7  multi-asset applied vs next-target split across held-state (rebalance)
     boundaries with the overlay scalar factored;
  8  37-asset manifest locks the correct Round-3 metrics (not the 21-asset
     experiment's);
  9  scorecard history cannot increase when snapshot identity is unchanged;
  10 readiness cadence counts weeks/rebalances, never raw daily bars.
Plus the frozen-construction regression locks from rev 1 (H5 bit-for-bit vs
ts_trend_backtest; lane net bit-for-bit vs combined_portfolio; held-state vs
single_asset_trend_returns) and registry coverage.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from src.crypto import ts_trend_shadow as ts
from src.equity import multi_asset_trend_lane as lane
from src.equity.multi_asset_trend import single_asset_trend_returns
from src.evidence.forward_ledger import cadence_counts, forward_rows, read_rows


# ── synthetic panels ────────────────────────────────────────────────────────

def _crypto_panels(n_coins=12, n_days=140, seed=7):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2026-01-01", periods=n_days, freq="D", tz="UTC")
    cols = [f"C{i}USDT" for i in range(n_coins)]
    drifts = np.where(np.arange(n_coins) % 2 == 0, 0.004, -0.004)
    rets = rng.normal(0, 0.01, size=(n_days, n_coins)) + drifts
    close = pd.DataFrame(100 * np.exp(np.cumsum(rets, axis=0)), index=idx, columns=cols)
    funding = pd.DataFrame(0.0, index=idx, columns=cols)
    eligible = pd.DataFrame(True, index=idx, columns=cols)
    return close, funding, eligible, cols


def _equity_panel(n_days=420, seed=3):
    idx = pd.date_range("2024-06-01", periods=n_days, freq="B")
    rng = np.random.default_rng(seed)
    up = 100 * np.exp(np.cumsum(rng.normal(0.0012, 0.008, n_days)))
    up2 = 50 * np.exp(np.cumsum(rng.normal(0.0010, 0.010, n_days)))
    down = 80 * np.exp(np.cumsum(rng.normal(-0.0015, 0.009, n_days)))
    return pd.DataFrame({"UPA": up, "UPB": up2, "DWN": down}, index=idx)


def _ts_capture(panels, ledger, ts_iso="2026-07-19T00:00:00+00:00"):
    return ts.capture_forward(ledger_path=ledger, cycle_ts_iso=ts_iso, panels=panels)


# ── frozen-construction regression locks (rev 1, retained) ─────────────────

def test_h5_net_series_matches_frozen_harness():
    import experiment_crypto_round2 as r2
    close, funding, eligible, cols = _crypto_panels()
    out, _book = ts._compute_book_and_returns(close, funding, eligible, cols)
    frozen = r2.ts_trend_backtest(close, funding, eligible, cols,
                                  cost_bps=ts.COST_BPS, vol_target=True)
    pd.testing.assert_series_equal(out["net"], frozen["net"], check_exact=True)


def test_trend_lane_net_matches_frozen_portfolio():
    prices = _equity_panel()
    frames = lane.compute_frames(prices)
    frozen_net = lane.FROZEN_PORTFOLIO(prices, target_vol=lane.TARGET_VOL)
    pd.testing.assert_series_equal(frames.net, frozen_net, check_exact=True)


def test_trend_lane_held_state_matches_frozen_single_asset_rule():
    prices = _equity_panel()
    for col in prices.columns:
        held = lane._held_state(prices[col])
        assert held is not None
        px = prices[col]
        applied = held.shift(1).fillna(0.0)
        gross = applied * px.pct_change()
        turnover = applied.diff().abs().fillna(applied.abs())
        net = (gross - turnover * (lane.COST_BPS_PER_SIDE * 1e-4)).dropna()
        pd.testing.assert_series_equal(net, single_asset_trend_returns(px), check_exact=True)


# ── 2: activation baseline ──────────────────────────────────────────────────

def test_empty_ledger_creates_activation_baseline_not_a_realized_return(tmp_path):
    panels = _crypto_panels()
    ledger = tmp_path / "ts.jsonl"
    appended = _ts_capture(panels, ledger)
    assert len(appended) == 1
    row = appended[0]
    assert row["kind"] == "activation"
    assert row["today_net_return"] is None, \
        "the latest bar's return was already known at activation — never counted"
    assert row["applied_book"] is None
    assert row["cumulative_shadow_return"] == 0.0
    assert row["forward_cycle_seq"] == 0
    assert row["book"]["longs"] or row["book"]["shorts"]   # baseline book snapshot
    assert forward_rows(read_rows(ledger)) == []


# ── 1: stale re-runs are no-ops everywhere ─────────────────────────────────

def test_capture_twice_on_stale_data_changes_nothing(tmp_path):
    panels = _crypto_panels()
    ledger = tmp_path / "ts.jsonl"
    _ts_capture(panels, ledger)                       # activation
    before = ledger.read_bytes()
    assert _ts_capture(panels, ledger) == []          # same data -> no-op
    assert _ts_capture(panels, ledger) == []          # and again
    assert ledger.read_bytes() == before

    prices = _equity_panel()
    tledger = tmp_path / "trend.jsonl"
    lane.capture_forward(prices=prices, ledger_path=tledger,
                         cycle_ts_iso="2026-07-19T00:00:00+00:00")
    tbefore = tledger.read_bytes()
    assert lane.capture_forward(prices=prices, ledger_path=tledger,
                                cycle_ts_iso="2026-07-20T00:00:00+00:00") == []
    assert tledger.read_bytes() == tbefore


def test_hedge_lane_refuses_duplicate_snapshot_identity(tmp_path):
    from src.hedge import hedged_shadow_lane as hsl
    book = hsl.BookSnapshot(
        strategy="equity_harvester", asset_class="equity", asof_date="2026-06-01",
        weights={"AAA": 0.5}, raw_net_return=None, raw_gross_return=None,
        raw_cost=None, return_source="must_compute")
    idx = pd.to_datetime(["2026-05-30", "2026-06-01", "2026-06-02"]).tz_localize("UTC")
    panel = pd.DataFrame({"AAA": [99.0, 100.0, 102.0]}, index=idx)
    kw = dict(book=book, price_panel=panel, sector_bucket_map={}, currency_bucket_map={},
              ledger_path=tmp_path / "l.jsonl", decision_log_path=tmp_path / "d.jsonl")
    first = hsl.run_cycle_for_strategy("equity_harvester", **kw)
    assert first is not None and first["book_identity_sha256"]
    ledger_before = (tmp_path / "l.jsonl").read_bytes()
    # scheduler re-run against the unchanged snapshot: REFUSED, ledger untouched
    assert hsl.run_cycle_for_strategy("equity_harvester", **kw) is None
    assert (tmp_path / "l.jsonl").read_bytes() == ledger_before
    # a genuinely different book on the same date IS a new snapshot
    book2 = hsl.BookSnapshot(
        strategy="equity_harvester", asset_class="equity", asof_date="2026-06-01",
        weights={"AAA": 0.7}, raw_net_return=None, raw_gross_return=None,
        raw_cost=None, return_source="must_compute")
    assert hsl.run_cycle_for_strategy("equity_harvester", **{**kw, "book": book2}) is not None


# ── 3+4+5: deterministic backfill of every unseen bar ───────────────────────

def test_backfill_appends_exactly_the_unseen_bars_in_order(tmp_path):
    close, funding, eligible, cols = _crypto_panels()
    ledger = tmp_path / "ts.jsonl"
    # activate on the truncated panel (3 bars withheld)
    _ts_capture((close.iloc[:-3], funding.iloc[:-3], eligible.iloc[:-3], cols), ledger)
    frozen, _ = ts._compute_book_and_returns(close, funding, eligible, cols)

    # one later bar available -> exactly the first forward return
    one = _ts_capture((close.iloc[:-2], funding.iloc[:-2], eligible.iloc[:-2], cols), ledger)
    assert len(one) == 1 and one[0]["kind"] == "forward"
    d0 = str(close.index[-3].date())
    assert one[0]["asof_date"] == d0
    assert one[0]["today_net_return"] == pytest.approx(float(frozen["net"].iloc[-3]), abs=0)
    assert one[0]["forward_cycle_seq"] == 1

    # scheduler "fails" for the next two bars, then one refresh sees both:
    # BOTH are backfilled, chronologically — never silently lost
    rest = _ts_capture((close, funding, eligible, cols), ledger)
    assert [r["asof_date"] for r in rest] == [str(close.index[-2].date()),
                                              str(close.index[-1].date())]
    assert [r["forward_cycle_seq"] for r in rest] == [2, 3]

    # cumulative == compounding the complete unseen slice, exactly
    expected = float((1.0 + frozen["net"].iloc[-3:]).prod() - 1.0)
    assert rest[-1]["cumulative_shadow_return"] == pytest.approx(expected, rel=0, abs=1e-15)


# ── 6: period/book alignment (crypto) ───────────────────────────────────────

def test_forward_row_applied_book_reproduces_the_periods_price_pnl(tmp_path):
    close, funding, eligible, cols = _crypto_panels()
    ledger = tmp_path / "ts.jsonl"
    _ts_capture((close.iloc[:-4], funding.iloc[:-4], eligible.iloc[:-4], cols), ledger)
    appended = _ts_capture((close, funding, eligible, cols), ledger)
    assert len(appended) == 4
    px_ret = close.pct_change()
    for row in appended:
        assert row["return_period_end"] == row["asof_date"]
        assert row["decision_asof"] == row["return_period_start"]
        applied = {**row["applied_book"]["longs"], **row["applied_book"]["shorts"]}
        d = pd.Timestamp(row["asof_date"], tz="UTC")
        reproduced = sum(w * float(px_ret.loc[d, c]) for c, w in applied.items())
        assert reproduced == pytest.approx(row["today_price_return"], abs=1e-12), \
            "applied_book must be EXACTLY the book that earned this row's return"


# ── 7: multi-asset applied vs next-target across boundaries ────────────────

def _equity_panel_with_flip(n_days=420, seed=3):
    """One asset trends up then reverses hard in the tail, so its held state
    flips OFF at a monthly step boundary inside the captured window."""
    p = _equity_panel(n_days, seed)
    rng = np.random.default_rng(99)
    flip = np.concatenate([
        np.cumsum(rng.normal(0.0020, 0.004, n_days - 120)),          # strong up
        np.cumsum(rng.normal(-0.0100, 0.004, 120)) + np.cumsum(      # hard down
            rng.normal(0.0020, 0.004, n_days - 120))[-1],
    ])
    p["FLP"] = 60 * np.exp(flip)
    return p


def test_trend_lane_applied_vs_next_target_split_and_scalar_factor(tmp_path):
    prices = _equity_panel_with_flip()
    frames = lane.compute_frames(prices)
    ledger = tmp_path / "trend.jsonl"
    n_tail = 45   # spans the tail step boundary where FLP's held state flips off
    lane.capture_forward(prices=prices.iloc[:-n_tail], ledger_path=ledger,
                         cycle_ts_iso="2026-07-19T00:00:00+00:00")
    appended = lane.capture_forward(prices=prices, ledger_path=ledger,
                                    cycle_ts_iso="2026-07-19T01:00:00+00:00")
    assert len(appended) == n_tail
    dates = [str(t.date()) for t in frames.net.index]
    pos = {d: i for i, d in enumerate(dates)}
    saw_boundary_divergence = False
    for row in appended:
        i = pos[row["asof_date"]]
        lev = float(frames.scalar.iloc[i])
        assert row["overlay_leverage"] == pytest.approx(lev)
        hrp_row = frames.hrp.iloc[i]
        for a in frames.held.columns:
            expect_applied = float(hrp_row.get(a, 0.0)) * float(frames.held.iloc[i - 1][a]) * lev
            expect_next = float(hrp_row.get(a, 0.0)) * float(frames.held.iloc[i][a]) * lev
            got_applied = row["applied_book"]["longs"].get(a, 0.0)
            got_next = row["book"]["longs"].get(a, 0.0)
            assert got_applied == pytest.approx(expect_applied, abs=1e-12), (a, row["asof_date"])
            assert got_next == pytest.approx(expect_next, abs=1e-12), (a, row["asof_date"])
        if row["applied_book"]["longs"] != row["book"]["longs"]:
            saw_boundary_divergence = True
    # the 30-bar window crosses a monthly-step boundary where the held state
    # (and/or overlay scalar) moved: the two books MUST diverge there
    assert saw_boundary_divergence, \
        "expected at least one bar where applied and next-target books differ"


# ── 8: manifest provenance ──────────────────────────────────────────────────

def test_manifest_locks_the_37_asset_round3_metrics_not_the_21_asset_run():
    m = lane.construction_manifest()
    pre = m["pre_registered_37_asset"]
    assert pre == {"full_sharpe": 0.665, "full_max_dd": 0.167,
                   "full_dsr_n20": 0.987, "full_bootstrap_p": 0.0,
                   "oos_sharpe": 0.788, "oos_max_dd": 0.162,
                   "oos_dsr_n20": 0.843, "oos_bootstrap_p": 0.002,
                   "clears_gate": False}
    assert "clears_gate=FALSE" in m["gate_verdict"]
    assert "0.843" in m["gate_verdict"]
    # the 21-asset numbers are background context, never this lane's verdict
    assert m["background_21_asset_context"]["full_sharpe"] == 0.744
    assert "0.744" not in m["gate_verdict"] and "0.829" not in m["gate_verdict"]
    assert m["vol_target_ann"] == 0.10 and m["universe_size"] == 37


def test_h5_manifest_is_the_frozen_h5_not_h4():
    m = ts.construction_manifest()
    assert m["signal"] == "ts_trend_sign_30d" and m["rebalance_days"] == 7
    from src.crypto.momentum_shadow import construction_manifest as h4_manifest
    assert m["signal"] != h4_manifest()["signal"]


# ── 9: scorecard cannot double-count an unchanged snapshot ─────────────────

def test_scorecard_history_cannot_increase_on_duplicate_identity(tmp_path):
    from src.hedge.hedge_scorecard import compute_strategy_scorecard, read_ledger_rows
    row = {
        "strategy": "candidate", "asof_date": "2026-06-01", "notional": 100000.0,
        "weights": {"AAA": 0.5}, "raw": {"net_return": 0.01},
        "hedge": {"status": "applied", "overlay_cost_known": True,
                  "overlay_cost_dollars": 10.0, "applied_proposal": {"legs": []}},
        "hedged": {"net_return": 0.008, "gross_return": None, "return_basis": "net"},
        "exposure": {"fail_closed": False},
    }
    ledger = tmp_path / "raw_vs_hedged_ledger.jsonl"
    with open(ledger, "w", encoding="utf-8") as fh:
        for _ in range(7):                       # 7 scheduler runs, one observation
            fh.write(json.dumps(row) + "\n")
    rows = read_ledger_rows(ledger)
    assert len(rows) == 1, "7 identical snapshots must collapse to ONE observation"
    card = compute_strategy_scorecard(rows)
    assert card["n_cycles"] == 1


# ── 10: cadence accounting ──────────────────────────────────────────────────

def test_cadence_counts_explicit_rebalance_fields_never_book_inequality():
    # 2026-07-19 scheduler-safety audit finding 4: rebalances come from the
    # frozen rule's EXPLICIT fields. The books below DRIFT every bar (vol
    # scaling) — inference from book inequality would count 10 rebalances;
    # the explicit fields say 2 signal events across 2 holding periods + the
    # activation-period remainder.
    rows = [{"kind": "activation", "asof_date": "2026-06-30",
             "book": {"longs": {"A": 1.0}}, "today_net_return": None}]
    for i, d in enumerate(pd.date_range("2026-07-01", periods=10, freq="D")):
        rows.append({"kind": "forward", "asof_date": str(d.date()),
                     # book weights drift EVERY bar (scalar movement)
                     "book": {"longs": {"A": 1.0 + i * 0.01}},
                     "rebalance_event": i in (4, 8),      # frozen-rule events
                     "rebalance_id": 0 if i < 4 else (1 if i < 8 else 2),
                     "holding_period_id": 0 if i < 4 else (1 if i < 8 else 2),
                     "today_net_return": 0.001})
    c = cadence_counts(rows)
    assert c["n_return_bars"] == 10
    assert c["n_calendar_weeks"] == 2            # Jul 1-10 2026 spans ISO weeks 27+28
    assert c["n_completed_rebalances"] == 2, "explicit events, NOT 10 daily book changes"
    assert c["n_independent_holding_periods"] == 3
    assert "EXPLICIT rebalance fields" in c["note"]
    # legacy rows without the fields -> None, never inferred
    legacy = [{"kind": "forward", "asof_date": "2026-07-01",
               "book": {"longs": {"A": 1.0}}, "today_net_return": 0.001}]
    lc = cadence_counts(legacy)
    assert lc["n_completed_rebalances"] is None and lc["n_independent_holding_periods"] is None


def test_forward_summary_counts_only_forward_rows(tmp_path):
    panels = _crypto_panels()
    close, funding, eligible, cols = panels
    ledger = tmp_path / "ts.jsonl"
    _ts_capture((close.iloc[:-2], funding.iloc[:-2], eligible.iloc[:-2], cols), ledger)
    _ts_capture(panels, ledger)
    s = ts.forward_summary(ledger_path=ledger)
    assert s["activated"] is True
    assert s["n_cycles"] == 2                    # activation excluded
    assert s["cadence"]["n_return_bars"] == 2
    assert "WEEKLY" in s["evidence_target"]


# ── registry integration (rev 1, retained + new-schema loaders) ────────────

def test_registry_covers_both_new_lanes(tmp_path):
    from src.hedge import hedged_shadow_lane as hsl
    assert "crypto_ts_trend" in hsl.STRATEGIES and "multi_asset_trend" in hsl.STRATEGIES

    assert hsl.load_crypto_ts_trend_book(ledger_path=tmp_path / "missing.jsonl") is None

    # the registry loads the STANDING TARGET book ("book"), not applied_book
    ts_ledger = tmp_path / "ts.jsonl"
    ts_ledger.write_text(json.dumps({
        "kind": "forward", "asof_date": "2026-05-20", "today_net_return": 0.004,
        "book": {"longs": {"BTCUSDT": 0.5}, "shorts": {"ADAUSDT": -0.25}},
        "applied_book": {"longs": {"BTCUSDT": 0.9}, "shorts": {}},
        "gross_leverage": 0.75, "construction": {"signal": "ts_trend_sign_30d"},
    }) + "\n")
    book = hsl.load_crypto_ts_trend_book(ledger_path=ts_ledger)
    assert book is not None
    assert book.weights == {"BTCUSDT": 0.5, "ADAUSDT": -0.25}, \
        "hedge registry must consume the next-bar standing target"


def test_promotion_report_universally_covers_seven_lanes(tmp_path):
    from src.hedge.hedged_shadow_lane import STRATEGIES
    from src.hedge.portfolio_promotion import VERDICT_SHADOW, build_portfolio_promotion_report
    report = build_portfolio_promotion_report(
        ledger_path=tmp_path / "empty.jsonl", scorecard_path=tmp_path / "none.json",
        out_path=tmp_path / "report.json", allocations_path=tmp_path / "no_alloc.json")
    assert set(report["verdicts"]) == set(STRATEGIES) and len(STRATEGIES) == 7
    for s in STRATEGIES:
        assert report["verdicts"][s]["verdict"] == VERDICT_SHADOW
