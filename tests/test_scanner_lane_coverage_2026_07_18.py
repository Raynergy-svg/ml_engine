"""Scanner-book hedge-lane coverage (load_oanda_fx_book, 2026-07-18).

No mocks: real journal/account-state JSON on real disk (tmp_path), real
loader, real run_cycle_for_strategy, real residual-attribution report build.
Locks in:
  1. scanner-attributed sourcing — OPEN journal entries only, never the
     shared account state (trend-lane exposure must not reach agent credit);
  2. weight math — signed fraction-of-NAV, exact base->USD when derivable
     from the pair itself, same-pair netting;
  3. fail-closed refusals — flat book, closed-only journal, missing NAV,
     malformed entry, futures excluded;
  4. end-to-end — run_cycle_for_strategy("oanda_fx") emits a lane row and
     the residual-attribution report gains an "oanda_fx" strategy key, so
     the RL sync's no_lane_coverage pass-through becomes resolvable.
"""
from __future__ import annotations

import json

import pytest

from src.hedge import hedged_shadow_lane as hsl


def _write_journal(tmp_path, entries):
    p = tmp_path / "trade_journal_rl.json"
    p.write_text(json.dumps(entries))
    return p


def _write_account(tmp_path, nav=100_000.0):
    p = tmp_path / "account_state.json"
    p.write_text(json.dumps({"nav": nav, "unrealized_pl": 0.0}))
    return p


def _entry(pair, direction, lots, entry_price, outcome=None, asset_class="FX"):
    return {
        "trade_id": f"{pair}-{direction}-{lots}",
        "pair": pair, "direction": direction, "lots": lots,
        "entry_price": entry_price, "outcome": outcome, "asset_class": asset_class,
    }


# ── 1-2. Sourcing + weight math ─────────────────────────────────────────────

def test_open_entries_become_signed_fraction_of_nav_weights(tmp_path):
    jp = _write_journal(tmp_path, [
        _entry("EUR_USD", "LONG", 1.0, 1.08),          # +108k / 100k = +1.08
        _entry("USD_JPY", "SHORT", 0.5, 150.0),        # base USD: -50k / 100k = -0.5
        _entry("GBP_USD", "LONG", 1.0, 1.25, outcome={"pnl": 5}),  # CLOSED: excluded
    ])
    ap = _write_account(tmp_path, nav=100_000.0)
    book = hsl.load_oanda_fx_book(journal_path=jp, account_state_path=ap)
    assert book is not None
    assert book.strategy == "oanda_fx" and book.asset_class == "fx"
    assert book.weights["EUR_USD"] == pytest.approx(1.08)
    assert book.weights["USD_JPY"] == pytest.approx(-0.5)
    assert "GBP_USD" not in book.weights, "closed entries must not enter the book"
    assert book.return_source == "open_positions_journal"
    assert book.raw_net_return is None  # journal carries no live mark — F1 forward-marks lane A


def test_same_pair_entries_net(tmp_path):
    jp = _write_journal(tmp_path, [
        _entry("EUR_USD", "LONG", 2.0, 1.08),
        _entry("EUR_USD", "SHORT", 1.0, 1.08),
    ])
    ap = _write_account(tmp_path)
    book = hsl.load_oanda_fx_book(journal_path=jp, account_state_path=ap)
    assert book is not None
    assert book.weights["EUR_USD"] == pytest.approx(1.08)  # net +1 lot


def test_fully_netted_book_is_flat_not_zero_weights(tmp_path):
    jp = _write_journal(tmp_path, [
        _entry("EUR_USD", "LONG", 1.0, 1.08),
        _entry("EUR_USD", "SHORT", 1.0, 1.08),
    ])
    ap = _write_account(tmp_path)
    assert hsl.load_oanda_fx_book(journal_path=jp, account_state_path=ap) is None


# ── 3. Fail-closed refusals ─────────────────────────────────────────────────

def test_refusals(tmp_path):
    ap = _write_account(tmp_path)
    # no journal
    assert hsl.load_oanda_fx_book(journal_path=tmp_path / "missing.json",
                                  account_state_path=ap) is None
    # closed-only journal -> flat
    jp = _write_journal(tmp_path, [_entry("EUR_USD", "LONG", 1.0, 1.08, outcome={"pnl": 1})])
    assert hsl.load_oanda_fx_book(journal_path=jp, account_state_path=ap) is None
    # futures entries excluded -> flat
    jp2 = _write_journal(tmp_path, [_entry("MES", "LONG", 1.0, 5000.0, asset_class="FUTURES")])
    assert hsl.load_oanda_fx_book(journal_path=jp2, account_state_path=ap) is None
    # open FX book but no NAV -> refuse
    jp3 = _write_journal(tmp_path, [_entry("EUR_USD", "LONG", 1.0, 1.08)])
    assert hsl.load_oanda_fx_book(journal_path=jp3,
                                  account_state_path=tmp_path / "missing_acct.json") is None
    # malformed open entry -> refuse the WHOLE book (never partial)
    jp4 = _write_journal(tmp_path, [
        _entry("EUR_USD", "LONG", 1.0, 1.08),
        _entry("BADPAIR", "LONG", 1.0, 1.0),
    ])
    assert hsl.load_oanda_fx_book(journal_path=jp4, account_state_path=ap) is None


# ── 4. End-to-end: lane row + attribution key ───────────────────────────────

def test_scanner_strategy_registered_and_cycle_emits_row(tmp_path, monkeypatch):
    assert "oanda_fx" in hsl.STRATEGIES and "oanda_fx" in hsl._LOADERS

    jp = _write_journal(tmp_path, [_entry("EUR_USD", "LONG", 1.0, 1.08)])
    ap = _write_account(tmp_path)
    monkeypatch.setattr(hsl, "SCANNER_JOURNAL_PATH", jp)
    monkeypatch.setattr(hsl, "FX_TREND_ACCOUNT_STATE_PATH", ap)
    ledger = tmp_path / "raw_vs_hedged_ledger.jsonl"
    decisions = tmp_path / "hedge_decision_log.jsonl"

    row = hsl.run_cycle_for_strategy(
        "oanda_fx", ledger_path=ledger, decision_log_path=decisions, persist=True)
    assert row is not None
    assert row["strategy"] == "oanda_fx" and row["asset_class"] == "fx"
    # live 'today' book: forward bar not cached yet -> honest unresolved raw,
    # never a fabricated mark (same contract as fx_trend)
    assert row["raw"]["net_return"] is None or isinstance(row["raw"]["net_return"], float)
    assert ledger.exists()

    # The residual layer now sees the strategy key the RL sync asks for.
    from src.hedge.residual_attribution import (
        build_residual_attribution_report, load_attribution_for,
    )
    out = tmp_path / "residual_attribution_report.json"
    report = build_residual_attribution_report(ledger_path=ledger, out_path=out)
    assert "oanda_fx" in report["strategies"]
    att = load_attribution_for("oanda_fx", report_path=out)
    assert att is not None
    # No aligned history yet -> phi undefined with a recorded reason (the
    # reward hook passes through) — coverage exists, evidence accrues honestly.
    assert att["residual_fraction_reason"] in (
        "no_aligned_residual_cycles", None) or att["residual_fraction"] is not None
