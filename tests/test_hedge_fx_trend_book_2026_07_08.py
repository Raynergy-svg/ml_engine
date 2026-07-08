"""No-mock tests for the FX trend-lane book loader in
src/hedge/hedged_shadow_lane.py — real tmp_path account_state.json + real
parquet tick fixtures + the real committed bucket maps. Verifies the live FX
book flows through the raw-vs-hedged shadow machinery (exposure decomposition
+ FX hedge proposal), and that the hedged overlay honestly degrades to
unresolved (no FX forward-price source wired). No unittest.mock anywhere."""
from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import pytest

from src.hedge.hedged_shadow_lane import (
    STRATEGIES,
    load_fx_trend_book,
    run_cycle_for_strategy,
)

NOW = datetime.now(timezone.utc)


def _write_account_state(path: Path, positions, *, nav=100000.0, unrealized_pl=0.0) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({
        "account_id": "101-000-0000000-001", "currency": "USD", "nav": nav,
        "margin_used": 5000.0, "margin_available": nav - 5000.0,
        "open_trade_count": len(positions), "unrealized_pl": unrealized_pl,
        "realized_pl": 0.0, "positions": positions,
    }), encoding="utf-8")


def _write_ticks(root: Path, pair: str, mid: float, *, ts: datetime = None) -> None:
    ts = ts or NOW
    f = root / pair / f"{ts:%Y}" / f"{ts:%m}" / f"{ts:%d}.parquet"
    f.parent.mkdir(parents=True, exist_ok=True)
    idx = pd.DatetimeIndex([ts - timedelta(seconds=30), ts], name="time")
    pd.DataFrame({"instrument": pair, "bid": mid - 0.0001, "ask": mid + 0.0001,
                  "mid": [mid, mid], "bid_liq": 1e6, "ask_liq": 1e6,
                  "status": "tradeable", "source": "test"}, index=idx).to_parquet(f)


@pytest.fixture
def env(tmp_path):
    state = tmp_path / "oanda" / "account_state.json"
    ticks = tmp_path / "ticks"
    return state, ticks


def _load(state, ticks):
    return load_fx_trend_book(account_state_path=state, ticks_root=ticks)


def test_fx_trend_registered():
    assert "fx_trend" in STRATEGIES
    from src.hedge.hedged_shadow_lane import _LOADERS
    assert _LOADERS["fx_trend"] is load_fx_trend_book


def test_usd_pileup_decomposed_and_hedge_proposed(env):
    """The roadmap §2.3 case: 3 trades that are secretly one big net-USD bet."""
    state, ticks = env
    _write_account_state(state, [
        {"instrument": "USD_JPY", "net_units": 100000.0, "stop_loss": 160.0},
        {"instrument": "USD_CHF", "net_units": 80000.0, "stop_loss": 0.88},
        {"instrument": "EUR_JPY", "net_units": 50000.0, "stop_loss": 172.0},
    ], nav=100000.0, unrealized_pl=-30.0)
    _write_ticks(ticks, "USD_JPY", 162.5)
    _write_ticks(ticks, "USD_CHF", 0.90)
    _write_ticks(ticks, "EUR_JPY", 175.0)
    _write_ticks(ticks, "EUR_USD", 1.08)   # supplies EUR->USD for EUR_JPY notional

    book = _load(state, ticks)
    assert book is not None
    assert book.asset_class == "fx" and book.strategy == "fx_trend"
    assert book.return_source == "own_ledger"
    # weights = signed home-notional / NAV. USD_JPY long 100k units, base USD
    # -> notional 100000, weight +1.0. EUR_JPY long 50k EUR -> 54000 USD, +0.54.
    assert book.weights["USD_JPY"] == pytest.approx(1.0)
    assert book.weights["USD_CHF"] == pytest.approx(0.8)
    assert book.weights["EUR_JPY"] == pytest.approx(0.54)
    assert book.raw_net_return == pytest.approx(-30.0 / 100000.0)

    row = run_cycle_for_strategy("fx_trend", book=book, persist=False)
    net = row["exposure"]["net_currency_exposure"]
    # Scaled to SHADOW_NOTIONAL (100k) but relative dominance preserved: net
    # USD long, net JPY short — the "one USD bet wearing three outfits".
    assert net["USD"] > 0 and net["JPY"] < 0
    assert row["hedge"]["status"] in ("applied", "no_valid_hedge")
    # Hedged overlay is honestly unresolved for FX (no forward-price source).
    assert row["hedged"]["return_basis"] == "unresolved"
    assert row["hedged"]["net_return"] is None
    assert book.meta["hedged_overlay"] == "unresolved_no_fx_forward_price_source"


def test_short_position_gives_negative_weight(env):
    state, ticks = env
    _write_account_state(state, [
        {"instrument": "EUR_USD", "net_units": -50000.0, "stop_loss": 1.10},
    ], nav=100000.0)
    _write_ticks(ticks, "EUR_USD", 1.08)
    book = _load(state, ticks)
    assert book is not None
    # short 50k EUR -> 54000 USD notional, negative weight.
    assert book.weights["EUR_USD"] == pytest.approx(-0.54)


def test_flat_book_returns_none(env):
    state, ticks = env
    _write_account_state(state, [], nav=100000.0)
    assert _load(state, ticks) is None


def test_missing_state_returns_none(env):
    state, ticks = env
    assert _load(state, ticks) is None


def test_stale_state_returns_none(env):
    import os
    state, ticks = env
    _write_account_state(state, [
        {"instrument": "USD_JPY", "net_units": 100000.0, "stop_loss": 160.0},
    ])
    _write_ticks(ticks, "USD_JPY", 162.5)
    old = (NOW - timedelta(hours=3)).timestamp()
    os.utime(state, (old, old))
    assert _load(state, ticks) is None


def test_zero_nav_returns_none(env):
    state, ticks = env
    _write_account_state(state, [
        {"instrument": "USD_JPY", "net_units": 100000.0, "stop_loss": 160.0},
    ], nav=0.0)
    _write_ticks(ticks, "USD_JPY", 162.5)
    assert _load(state, ticks) is None


def test_unresolvable_position_skipped_not_fabricated(env):
    state, ticks = env
    _write_account_state(state, [
        {"instrument": "USD_JPY", "net_units": 100000.0, "stop_loss": 160.0},
        {"instrument": "EUR_GBP", "net_units": 10000.0, "stop_loss": 0.84},
    ], nav=100000.0)
    _write_ticks(ticks, "USD_JPY", 162.5)  # no EUR/GBP rate -> EUR_GBP unresolved
    book = _load(state, ticks)
    assert book is not None
    assert "USD_JPY" in book.weights
    assert "EUR_GBP" not in book.weights          # never fabricated
    assert book.meta["skipped_instruments"] == ["EUR_GBP"]
