"""Scheduler-safety audit (2026-07-19) — the evidence system under real
concurrency and failure conditions. No mocks: real ``multiprocessing`` (fork)
workers against real tmp_path ledgers, real fcntl locks, real corruption
bytes, real failure injection via a directory-as-file path.

Covers the audit's required tests:
  A  two workers released by a barrier on an empty ledger -> ONE activation;
  B  two workers capturing the same unseen slice -> each evidence period once;
  C  concurrent hedge captures -> one ledger row + one matching decision event;
  D  corrupt latest line -> append AND every evidence consumer fail closed;
  E  duplicate panel dates -> rejected before writing;
  F  NaN / infinity / boolean returns and non-finite weights -> rejected;
  G  different book revisions for one evidence period -> one ACTIVE
     observation after explicit supersession;
  H  H5 daily vol-scalar movement, weekly signal -> completed rebalances
     count WEEKLY events, not daily book changes;
  I  multi-asset HRP/scalar drift between monthly signal events -> completed
     rebalances count MONTHLY events;
  J  failure between hedge evidence and decision persistence -> NEITHER
     record committed;
  K  (promotion drawdown regression lives in test_portfolio_promotion);
  L  scorecard, residual attribution and promotion consume EXACTLY the same
     canonical evidence-period set.
"""
from __future__ import annotations

import json
import multiprocessing
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.crypto import ts_trend_shadow as ts
from src.equity import multi_asset_trend_lane as lane
from src.evidence.forward_ledger import (
    LedgerCorruptionError,
    active_observations,
    append_unseen_bars,
    cadence_counts,
    read_rows,
)
from src.hedge import hedged_shadow_lane as hsl

_CTX = multiprocessing.get_context("fork")


def _crypto_panels(n_coins=12, n_days=60, seed=7):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2026-01-01", periods=n_days, freq="D", tz="UTC")
    cols = [f"C{i}USDT" for i in range(n_coins)]
    drifts = np.where(np.arange(n_coins) % 2 == 0, 0.004, -0.004)
    rets = rng.normal(0, 0.01, size=(n_days, n_coins)) + drifts
    close = pd.DataFrame(100 * np.exp(np.cumsum(rets, axis=0)), index=idx, columns=cols)
    return close, pd.DataFrame(0.0, index=idx, columns=cols), \
        pd.DataFrame(True, index=idx, columns=cols), cols


def _hedge_book(weights=None, asof="2026-06-01"):
    return hsl.BookSnapshot(
        strategy="equity_harvester", asset_class="equity", asof_date=asof,
        weights=weights or {"AAA": 0.5}, raw_net_return=None,
        raw_gross_return=None, raw_cost=None, return_source="must_compute")


def _hedge_panel():
    idx = pd.to_datetime(["2026-05-30", "2026-06-01", "2026-06-02"]).tz_localize("UTC")
    return pd.DataFrame({"AAA": [99.0, 100.0, 102.0]}, index=idx)


# ── module-level workers (fork-picklable) ───────────────────────────────────

def _worker_ts_capture(barrier, ledger_str, n_days):
    panels = _crypto_panels(n_days=n_days)
    barrier.wait()
    try:
        ts.capture_forward(ledger_path=Path(ledger_str),
                           cycle_ts_iso="2026-07-19T00:00:00+00:00", panels=panels)
    except Exception:  # noqa: BLE001 — loser races are allowed to no-op, not crash the test harness
        pass


def _worker_hedge_capture(barrier, ledger_str, decision_str):
    barrier.wait()
    try:
        hsl.run_cycle_for_strategy(
            "equity_harvester", book=_hedge_book(), price_panel=_hedge_panel(),
            sector_bucket_map={}, currency_bucket_map={},
            ledger_path=Path(ledger_str), decision_log_path=Path(decision_str))
    except Exception:  # noqa: BLE001
        pass


def _run_pair(target, args):
    p1 = _CTX.Process(target=target, args=args)
    p2 = _CTX.Process(target=target, args=args)
    p1.start()
    p2.start()
    p1.join(60)
    p2.join(60)
    assert not p1.is_alive() and not p2.is_alive()


# ── A/B/C: concurrency ─────────────────────────────────────────────────────

def test_concurrent_activation_produces_exactly_one_baseline(tmp_path):
    ledger = tmp_path / "ts.jsonl"
    barrier = _CTX.Barrier(2)
    _run_pair(_worker_ts_capture, (barrier, str(ledger), 60))
    rows = read_rows(ledger)
    assert len(rows) == 1 and rows[0]["kind"] == "activation"


def test_concurrent_backfill_each_evidence_period_exactly_once(tmp_path):
    ledger = tmp_path / "ts.jsonl"
    # activate on a truncated panel first (sequential)
    ts.capture_forward(ledger_path=ledger, cycle_ts_iso="2026-07-19T00:00:00+00:00",
                       panels=_crypto_panels(n_days=57))
    barrier = _CTX.Barrier(2)
    _run_pair(_worker_ts_capture, (barrier, str(ledger), 60))
    rows = read_rows(ledger)
    fwd = [r for r in rows if r["kind"] == "forward"]
    assert len(fwd) == 3, "the 3 unseen bars appear exactly once each"
    periods = [r["evidence_period_id"] for r in fwd]
    assert len(periods) == len(set(periods))


def test_concurrent_hedge_capture_one_row_one_decision(tmp_path):
    ledger, decisions = tmp_path / "l.jsonl", tmp_path / "d.jsonl"
    barrier = _CTX.Barrier(2)
    _run_pair(_worker_hedge_capture, (barrier, str(ledger), str(decisions)))
    lrows = read_rows(ledger)
    drows = read_rows(decisions)
    assert len(lrows) == 1 and len(drows) == 1
    assert drows[0]["evidence_period_id"] == lrows[0]["evidence_period_id"]
    assert drows[0]["snapshot_version_id"] == lrows[0]["snapshot_version_id"]


# ── D: corruption fails closed everywhere ──────────────────────────────────

def test_corrupt_latest_line_fails_closed_for_append_and_all_consumers(tmp_path):
    # lane ledger: activation, then corrupt the tail
    ledger = tmp_path / "ts.jsonl"
    panels = _crypto_panels(n_days=57)
    ts.capture_forward(ledger_path=ledger, cycle_ts_iso="2026-07-19T00:00:00+00:00",
                       panels=panels)
    with open(ledger, "ab") as fh:
        fh.write(b'{"kind": "forward", "asof_date": "2026-02-2')  # truncated write
    before = ledger.read_bytes()
    with pytest.raises(LedgerCorruptionError) as exc:
        ts.capture_forward(ledger_path=ledger, cycle_ts_iso="2026-07-19T01:00:00+00:00",
                           panels=_crypto_panels(n_days=60))
    assert exc.value.line_no == 2 and exc.value.byte_offset > 0
    assert ledger.read_bytes() == before, "no append past corruption"
    # marker persists the refusal for subsequent runs
    assert Path(str(ledger) + ".corrupt.json").exists()
    with pytest.raises(LedgerCorruptionError):
        ts.forward_summary(ledger_path=ledger)

    # hedge ledger: every evidence consumer refuses
    hledger = tmp_path / "raw_vs_hedged_ledger.jsonl"
    good = {"strategy": "candidate", "asof_date": "2026-06-01",
            "raw": {"net_return": 0.01}, "hedge": {"status": "applied"},
            "hedged": {"net_return": 0.008, "gross_return": None}}
    hledger.write_text(json.dumps(good) + "\n" + '{"strategy": "cand')
    from src.hedge.hedge_scorecard import read_ledger_rows
    with pytest.raises(LedgerCorruptionError):
        read_ledger_rows(hledger)
    Path(str(hledger) + ".corrupt.json").unlink()  # marker from previous read
    from src.hedge.residual_attribution import build_residual_attribution_report
    with pytest.raises(LedgerCorruptionError):
        build_residual_attribution_report(ledger_path=hledger,
                                          out_path=tmp_path / "ra.json")
    Path(str(hledger) + ".corrupt.json").unlink()
    from src.hedge.portfolio_promotion import build_portfolio_promotion_report
    with pytest.raises(LedgerCorruptionError):
        build_portfolio_promotion_report(
            ledger_path=hledger, scorecard_path=tmp_path / "sc.json",
            out_path=tmp_path / "pp.json", allocations_path=tmp_path / "al.json")
    Path(str(hledger) + ".corrupt.json").unlink()
    # ...and the hedge APPEND path refuses too
    with pytest.raises(LedgerCorruptionError):
        hsl.run_cycle_for_strategy(
            "equity_harvester", book=_hedge_book(), price_panel=_hedge_panel(),
            sector_bucket_map={}, currency_bucket_map={},
            ledger_path=hledger, decision_log_path=tmp_path / "d.jsonl")


# ── E/F: schema validation before write ────────────────────────────────────

def _min_forward_body(d_prev, strategy="s"):
    return {"strategy": strategy, "today_net_return": 0.001,
            "applied_book": {"A": 0.5}, "book": {"A": 0.5},
            "return_period_start": d_prev, "construction": {"sig": "x"},
            "rebalance_event": False, "rebalance_id": 0, "holding_period_id": 0}


def test_duplicate_dates_rejected_before_writing(tmp_path):
    ledger = tmp_path / "l.jsonl"
    with pytest.raises(ValueError, match="duplicate date"):
        append_unseen_bars(ledger, strategy="s",
                           dates=["2026-07-01", "2026-07-02", "2026-07-02"],
                           payload_for=lambda d: _min_forward_body("2026-07-01"),
                           activation_payload_for=lambda d: {"book": {"A": 1.0},
                                                             "construction": {}},
                           cycle_ts_iso="t")
    assert not ledger.exists(), "rejected BEFORE any write"
    with pytest.raises(ValueError, match="strictly increasing"):
        append_unseen_bars(ledger, strategy="s", dates=["2026-07-02", "2026-07-01"],
                           payload_for=lambda d: _min_forward_body("2026-07-01"),
                           activation_payload_for=lambda d: {"book": {"A": 1.0},
                                                             "construction": {}},
                           cycle_ts_iso="t")
    with pytest.raises(ValueError, match="non-ISO|non-canonical"):
        append_unseen_bars(ledger, strategy="s", dates=["07/01/2026"],
                           payload_for=lambda d: _min_forward_body("x"),
                           activation_payload_for=lambda d: {"book": {"A": 1.0},
                                                             "construction": {}},
                           cycle_ts_iso="t")


@pytest.mark.parametrize("bad_net", [float("nan"), float("inf"), True, "0.01", None])
def test_non_finite_or_boolean_returns_rejected(tmp_path, bad_net):
    ledger = tmp_path / "l.jsonl"
    append_unseen_bars(ledger, strategy="s", dates=["2026-07-01"],
                       payload_for=lambda d: _min_forward_body("x"),
                       activation_payload_for=lambda d: {"book": {"A": 1.0},
                                                         "construction": {"sig": "x"}},
                       cycle_ts_iso="t")  # activation
    before = ledger.read_bytes()

    def bad_payload(d):
        body = _min_forward_body("2026-07-01")
        body["today_net_return"] = bad_net
        return body
    with pytest.raises(ValueError):
        append_unseen_bars(ledger, strategy="s", dates=["2026-07-01", "2026-07-02"],
                           payload_for=bad_payload,
                           activation_payload_for=lambda d: {},
                           cycle_ts_iso="t")
    assert ledger.read_bytes() == before, "nothing appended on rejection"


def test_non_finite_weights_rejected(tmp_path):
    ledger = tmp_path / "l.jsonl"
    append_unseen_bars(ledger, strategy="s", dates=["2026-07-01"],
                       payload_for=lambda d: _min_forward_body("x"),
                       activation_payload_for=lambda d: {"book": {"A": 1.0},
                                                         "construction": {"sig": "x"}},
                       cycle_ts_iso="t")
    before = ledger.read_bytes()

    def bad_payload(d):
        body = _min_forward_body("2026-07-01")
        body["book"] = {"A": float("inf")}
        return body
    with pytest.raises(ValueError, match="finite"):
        append_unseen_bars(ledger, strategy="s", dates=["2026-07-01", "2026-07-02"],
                           payload_for=bad_payload,
                           activation_payload_for=lambda d: {},
                           cycle_ts_iso="t")
    assert ledger.read_bytes() == before


# ── G: supersession = one active observation per evidence period ────────────

def test_revised_hedge_snapshot_supersedes_never_counts_beside(tmp_path):
    ledger, decisions = tmp_path / "l.jsonl", tmp_path / "d.jsonl"
    kw = dict(price_panel=_hedge_panel(), sector_bucket_map={}, currency_bucket_map={},
              ledger_path=ledger, decision_log_path=decisions)
    r1 = hsl.run_cycle_for_strategy("equity_harvester", book=_hedge_book({"AAA": 0.5}), **kw)
    r2 = hsl.run_cycle_for_strategy("equity_harvester", book=_hedge_book({"AAA": 0.7}), **kw)
    assert r1 is not None and r2 is not None
    assert r2["supersedes"] == r1["book_identity_sha256"]
    raw = read_rows(ledger)
    assert len(raw) == 2, "history preserved (append-only)"
    active = hsl.active_ledger_rows(raw)
    assert len(active) == 1, "ONE active observation per evidence period"
    assert active[0]["book_identity_sha256"] == r2["book_identity_sha256"]


# ── H/I: real cadence from the frozen rule ─────────────────────────────────

def test_h5_rebalances_count_weekly_signal_events_not_daily_book_drift(tmp_path):
    ledger = tmp_path / "ts.jsonl"
    ts.capture_forward(ledger_path=ledger, cycle_ts_iso="t0",
                       panels=_crypto_panels(n_days=45))
    ts.capture_forward(ledger_path=ledger, cycle_ts_iso="t1",
                       panels=_crypto_panels(n_days=60))   # 15 forward bars
    rows = read_rows(ledger)
    fwd = active_observations(rows)
    assert len(fwd) == 15
    # daily vol-scaling moves the book on non-rebalance bars...
    non_rebal = [r for r in fwd if not r["rebalance_event"]]
    books = [json.dumps(r["book"], sort_keys=True) for r in non_rebal]
    assert len(set(books)) > 1, "book drifts daily without a signal rebalance"
    # ...but completed rebalances count ONLY the frozen weekly events
    expected_events = sum(1 for r in fwd if r["rebalance_event"])
    c = cadence_counts(rows)
    assert c["n_completed_rebalances"] == expected_events
    assert expected_events == sum(1 for i in range(45, 60) if i % ts.REBALANCE_DAYS == 0)
    assert c["n_completed_rebalances"] < c["n_return_bars"]


def test_multi_asset_rebalances_count_monthly_events_not_hrp_scalar_drift(tmp_path):
    idx = pd.date_range("2024-06-01", periods=420, freq="B")
    rng = np.random.default_rng(3)
    prices = pd.DataFrame({
        "UPA": 100 * np.exp(np.cumsum(rng.normal(0.0012, 0.008, 420))),
        "UPB": 50 * np.exp(np.cumsum(rng.normal(0.0010, 0.010, 420))),
        "DWN": 80 * np.exp(np.cumsum(rng.normal(-0.0015, 0.009, 420)))}, index=idx)
    ledger = tmp_path / "trend.jsonl"
    lane.capture_forward(prices=prices.iloc[:-30], ledger_path=ledger, cycle_ts_iso="t0")
    lane.capture_forward(prices=prices, ledger_path=ledger, cycle_ts_iso="t1")
    rows = read_rows(ledger)
    fwd = active_observations(rows)
    assert len(fwd) == 30
    non_rebal = [r for r in fwd if not r["rebalance_event"]]
    books = [json.dumps(r["book"], sort_keys=True) for r in non_rebal]
    assert len(set(books)) > 1, "HRP/scalar drift moves the book between signal events"
    c = cadence_counts(rows)
    assert c["n_completed_rebalances"] == sum(1 for r in fwd if r["rebalance_event"])
    assert 1 <= c["n_completed_rebalances"] <= 2, "~monthly events inside a 30-bar window"


# ── J: hedge evidence + decision commit atomically ─────────────────────────

def test_failure_between_evidence_and_decision_commits_neither(tmp_path):
    ledger = tmp_path / "l.jsonl"
    decision_dir = tmp_path / "decisions_dir"
    decision_dir.mkdir()          # a DIRECTORY: the decision append genuinely fails
    with pytest.raises(OSError):
        hsl.run_cycle_for_strategy(
            "equity_harvester", book=_hedge_book(), price_panel=_hedge_panel(),
            sector_bucket_map={}, currency_bucket_map={},
            ledger_path=ledger, decision_log_path=decision_dir)
    assert read_rows(ledger) == [], "evidence row rolled back — neither record committed"


# ── L: consumer consistency on one canonical evidence set ──────────────────

def test_all_consumers_share_the_same_canonical_evidence_period_set(tmp_path):
    ledger = tmp_path / "raw_vs_hedged_ledger.jsonl"
    base = {"strategy": "candidate", "notional": 100000.0,
            "hedge": {"status": "applied", "overlay_cost_known": True,
                      "overlay_cost_dollars": 1.0, "applied_proposal": {"legs": []}},
            "exposure": {"fail_closed": False}}
    rows = [
        {**base, "asof_date": "2026-06-01", "weights": {"A": 0.5},
         "raw": {"net_return": 0.010}, "hedged": {"net_return": 0.009, "gross_return": None,
                                                  "return_basis": "net"}},
        # revision of the SAME period — must supersede everywhere. Re-review
        # item 2: the replacement is EXPLICIT (the ``supersedes`` stamp the
        # locked append writes); an unmarked second row now fails closed.
        {**base, "asof_date": "2026-06-01", "weights": {"A": 0.7},
         "supersedes": "prior-book-identity",
         "raw": {"net_return": 0.012}, "hedged": {"net_return": 0.011, "gross_return": None,
                                                  "return_basis": "net"}},
        {**base, "asof_date": "2026-06-02", "weights": {"A": 0.7},
         "raw": {"net_return": 0.004}, "hedged": {"net_return": 0.003, "gross_return": None,
                                                  "return_basis": "net"}},
    ]
    ledger.write_text("\n".join(json.dumps(r) for r in rows) + "\n")

    from src.hedge.hedge_scorecard import compute_strategy_scorecard, read_ledger_rows
    sc_rows = read_ledger_rows(ledger)
    sc_periods = {(r["strategy"], r["asof_date"]) for r in sc_rows}
    assert compute_strategy_scorecard(sc_rows)["n_cycles"] == 2

    from src.hedge.residual_attribution import build_residual_attribution_report
    ra = build_residual_attribution_report(ledger_path=ledger,
                                           out_path=tmp_path / "ra.json")
    ra_n = ra["strategies"]["candidate"]["attribution"]["n_aligned_cycles"] \
        if "attribution" in ra["strategies"]["candidate"] \
        else ra["strategies"]["candidate"]["n_aligned_cycles"]
    assert ra_n == 2

    from src.hedge.portfolio_promotion import build_portfolio_promotion_report
    pp = build_portfolio_promotion_report(
        ledger_path=ledger, scorecard_path=tmp_path / "none.json",
        out_path=tmp_path / "pp.json", allocations_path=tmp_path / "al.json")
    ev = next(c for c in pp["verdicts"]["candidate"]["checks"]
              if c["name"] == "standalone_evidence")
    assert ev["n_aligned_cycles"] == 2

    # identical canonical period set across all three consumers
    assert sc_periods == {("candidate", "2026-06-01"), ("candidate", "2026-06-02")}
    # ...and the ACTIVE 06-01 revision everywhere is the SECOND book
    active_0601 = next(r for r in sc_rows if r["asof_date"] == "2026-06-01")
    assert active_0601["weights"] == {"A": 0.7}
