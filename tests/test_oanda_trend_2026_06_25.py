"""OANDA practice TREND runner — pure-logic tests (no network, no mocks).

Verifies the candle->panel parse (complete-bars-only), the trend signal (long-or-
flat, causal, reuses the validated trend_sleeve rule), and unit sizing. Real
functions against real synthetic OANDA candle dicts; the broker/network is NOT
exercised here (that needs a live practice token, surfaced as the blocker).
"""
from __future__ import annotations

import numpy as np

from src.equity.oanda_trend import (
    base_to_home_rate,
    candles_to_close_panel,
    rebalance_delta,
    repair_missing_trade_brackets,
    run_oanda_trend_cycle,
    target_units,
    trend_targets,
)
from src.scanner.automation.state_engine import StateEngine


def test_compute_atr_from_ohlc_candles():
    from src.equity.oanda_trend import compute_atr
    # 20 complete candles, constant 1.0-wide range -> ATR == 1.0; incomplete excluded.
    candles = {"X": {"candles": (
        [{"complete": True, "mid": {"h": "11", "l": "10", "c": "10.5"}} for _ in range(20)]
        + [{"complete": False, "mid": {"h": "99", "l": "0", "c": "50"}}]
    )}}
    atr = compute_atr(candles, period=14)
    assert abs(atr["X"] - 1.0) < 1e-9


def test_bracket_prices_long_sl_and_tp_on_by_default():
    from src.equity.oanda_trend import bracket_prices
    sl, tp = bracket_prices(100.0, 2.0, sl_mult=2.0, tp_mult=4.0)
    assert abs(sl - 96.0) < 1e-9 and abs(tp - 108.0) < 1e-9
    assert bracket_prices(100.0, None) == (None, None)             # no ATR -> no fabricated stop
    assert bracket_prices(100.0, 2.0, enable_sl=False)[0] is None


def test_margin_guard_clamps_under_stress():
    from src.equity.oanda_trend import margin_scale, margin_projection
    px = {"EUR_USD": 1.10, "USD_JPY": 150.0}
    # base-USD pair USD_JPY: rate 1.0; EUR_USD: rate 1.10. margin_rate 0.04.
    want = {"USD_JPY": 1_000_000, "EUR_USD": 1_000_000}
    nav = 100_000.0
    proj = margin_projection(want, px, margin_rate=0.04)
    assert proj > 0.5 * nav                                        # this book breaches a 50% cap
    scaled, factor = margin_scale(want, px, nav, max_margin_util=0.50, margin_rate=0.04)
    assert factor < 1.0                                            # GUARD FIRED
    # post-scale projected margin is at/below the cap
    assert margin_projection(scaled, px, margin_rate=0.04) <= 0.50 * nav + 1
    # a small book within the cap is untouched
    small = {"USD_JPY": 100}
    _, f2 = margin_scale(small, px, nav, max_margin_util=0.50, margin_rate=0.04)
    assert f2 == 1.0


def test_clamp_leverage_bounds():
    from src.equity.oanda_trend import DEFAULT_GROSS_LEVERAGE, MAX_GROSS_LEVERAGE, clamp_leverage
    assert clamp_leverage(3.0) == 3.0
    assert clamp_leverage(0) == 0.0                          # control 0x means flat/no exposure
    assert clamp_leverage(-5) == DEFAULT_GROSS_LEVERAGE
    assert clamp_leverage(999) == MAX_GROSS_LEVERAGE         # capped at margin-call guard
    assert clamp_leverage(10) == 10.0
    assert clamp_leverage(float("nan")) == DEFAULT_GROSS_LEVERAGE   # non-finite -> default
    assert clamp_leverage(float("inf")) == DEFAULT_GROSS_LEVERAGE   # non-finite -> default
    assert clamp_leverage("bad") == DEFAULT_GROSS_LEVERAGE          # unparseable -> default


def test_nav_drawdown_auto_halt(tmp_path):
    from src.equity.oanda_trend import nav_drawdown_breached
    p = tmp_path / "peak_nav.json"
    assert nav_drawdown_breached(100000.0, p) is None      # first call sets peak, no dd
    assert nav_drawdown_breached(95000.0, p) is None        # 5% dd < 20%, peak holds
    dd = nav_drawdown_breached(79000.0, p)                  # 21% from peak 100k -> breach
    assert dd is not None and dd > 0.20
    # peak ratchets UP to a new high, then drawdown is measured from the new peak
    assert nav_drawdown_breached(120000.0, p) is None        # new peak 120k
    assert nav_drawdown_breached(100000.0, p) is None        # 16.7% from 120k < 20%
    assert nav_drawdown_breached(95000.0, p) is not None     # 20.8% from 120k -> breach


def test_rebalance_band_suppresses_micro_churn_but_allows_flips():
    # NAV-drift micro-delta within the band -> no trade (no churn)
    assert rebalance_delta(12769, 12770) == 0
    assert rebalance_delta(12769, 12740) == 0          # 29 units < 2% of 12769
    # real moves clear the band
    assert rebalance_delta(12769, 0) == 12769          # open from flat
    assert rebalance_delta(0, 12769) == -12769         # flip to flat (close)
    assert rebalance_delta(12769, 12000) == 769        # 769 > 2% -> rebalance


def _candles(closes, *, complete=True, start="2020-01-01"):
    import pandas as pd
    idx = pd.bdate_range(start, periods=len(closes))
    return {"candles": [
        {"time": t.isoformat() + "Z", "complete": complete, "mid": {"c": f"{c:.5f}"}}
        for t, c in zip(idx, closes)
    ]}


def _ohlc_candles(closes, *, start="2020-01-01"):
    import pandas as pd
    idx = pd.bdate_range(start, periods=len(closes))
    return {"candles": [
        {
            "time": t.isoformat() + "Z",
            "complete": True,
            "mid": {"h": f"{c + 0.001:.5f}", "l": f"{c - 0.001:.5f}", "c": f"{c:.5f}"},
        }
        for t, c in zip(idx, closes)
    ]}


class _Config:
    oanda_environment = "practice"


class _FakeClient:
    def __init__(self, candles, trades=None):
        self.candles = candles
        self.trades = trades or []
        self.orders = []
        self.bracket_repairs = []

    def get_candles(self, instrument, **_kwargs):
        return self.candles[instrument]

    def get_account_summary(self):
        return {"account": {"NAV": "100000"}}

    def get_open_positions(self):
        return {"positions": []}

    def get_trades(self, *, state="OPEN"):
        assert state == "OPEN"
        return {"trades": self.trades}

    def create_market_order(self, **kwargs):
        self.orders.append(kwargs)
        return {"orderCreateTransaction": {"id": str(len(self.orders))}}

    def set_trade_dependent_orders(self, **kwargs):
        self.bracket_repairs.append(kwargs)
        return {"relatedTransactionIDs": ["1"]}


def test_repair_missing_trade_brackets_adds_tp_without_replacing_sl():
    client = _FakeClient({}, trades=[{
        "id": "42",
        "instrument": "EUR_USD",
        "currentUnits": "1000",
        "stopLossOrder": {"price": "1.09000"},
    }])

    repaired = repair_missing_trade_brackets(
        client,
        last_px={"EUR_USD": 1.1000},
        atr={"EUR_USD": 0.0020},
    )

    assert repaired == 1
    repair = client.bracket_repairs[0]
    assert repair["trade_id"] == "42"
    assert repair["stop_loss_price"] is None
    assert repair["take_profit_price"] > 1.1000
    assert repair["take_profit_price"] >= 1.1000 + 1.2 * (1.1000 - 1.0900)


def test_repair_missing_trade_brackets_fails_closed_without_atr():
    client = _FakeClient({}, trades=[{
        "id": "42",
        "instrument": "EUR_USD",
        "currentUnits": "1000",
        "stopLossOrder": {"price": "1.09000"},
    }])

    import pytest
    with pytest.raises(ValueError):
        repair_missing_trade_brackets(client, last_px={"EUR_USD": 1.1000}, atr={})
    assert client.bracket_repairs == []


def test_trend_cycle_sends_sl_and_tp_on_new_long(tmp_path):
    StateEngine(tmp_path / ".claude" / "state.json").set_halted(False)
    closes = list(np.linspace(1.05, 1.20, 160))
    client = _FakeClient({"EUR_USD": _ohlc_candles(closes)})

    result = run_oanda_trend_cycle(
        client=client,
        config=_Config(),
        instruments=["EUR_USD"],
        project_root=tmp_path,
        sma_window=20,
        candle_count=160,
        gross_leverage=0.01,
    )

    assert result.orders_placed == 1
    order = client.orders[0]
    assert order["units"] > 0
    # Brackets are now sent as fill-anchored DISTANCES (sl_mult*ATR / tp_mult*ATR),
    # not absolute prices — so R:R is a constant 2.0 regardless of fill-vs-close drift
    # (the 2026-06-30 stale-anchor fix). No absolute price is sent on the new long.
    assert order.get("stop_loss_price") is None
    assert order.get("take_profit_price") is None
    sl_d = order["stop_loss_distance"]
    tp_d = order["take_profit_distance"]
    assert sl_d is not None and tp_d is not None
    assert sl_d > 0 and tp_d > 0
    assert abs(tp_d / sl_d - 2.0) < 1e-9      # tp_mult/sl_mult = 4/2
    assert tp_d / sl_d >= 1.2                  # R:R invariant holds by construction


def test_trend_cycle_refuses_new_long_when_required_brackets_missing(tmp_path):
    StateEngine(tmp_path / ".claude" / "state.json").set_halted(False)
    closes = list(np.linspace(1.05, 1.20, 80))
    client = _FakeClient({"EUR_USD": _candles(closes)})

    result = run_oanda_trend_cycle(
        client=client,
        config=_Config(),
        instruments=["EUR_USD"],
        project_root=tmp_path,
        sma_window=20,
        candle_count=80,
        gross_leverage=0.01,
    )

    assert result.ran is True
    assert result.orders_placed == 0
    assert client.orders == []


def test_panel_drops_incomplete_candles():
    up = list(np.linspace(100, 120, 50))
    resp = _candles(up)
    resp["candles"][-1]["complete"] = False  # forming bar must be excluded
    panel = candles_to_close_panel({"EUR_USD": resp})
    assert panel.shape[0] == 49
    assert "EUR_USD" in panel.columns
    assert abs(panel["EUR_USD"].iloc[0] - 100.0) < 1e-6


def test_trend_long_when_uptrend_flat_when_downtrend():
    n = 260
    up = list(100 * np.cumprod(1 + np.full(n, 0.002)))      # steady uptrend -> ON
    down = list(200 * np.cumprod(1 + np.full(n, -0.002)))   # steady downtrend -> FLAT
    panel = candles_to_close_panel({
        "UP": _candles(up), "DOWN": _candles(down)})
    tg = trend_targets(panel, sma_window=100)
    assert tg["UP"] > 0      # above its rising MA -> held
    assert tg["DOWN"] == 0   # below its falling MA -> cash


def test_base_to_home_rate_per_base_currency():
    px = {"EUR_USD": 1.10, "USD_JPY": 150.0, "GBP_USD": 1.27, "USD_CAD": 1.36}
    assert base_to_home_rate("USD_JPY", px) == 1.0          # base USD -> 1.0
    assert base_to_home_rate("USD_CAD", px) == 1.0          # base USD -> 1.0
    assert abs(base_to_home_rate("EUR_USD", px) - 1.10) < 1e-9   # direct BASE_USD
    assert abs(base_to_home_rate("GBP_JPY", px) - 1.27) < 1e-9   # base GBP via GBP_USD cross
    # base with only an inverse HOME_BASE leg (base=CAD -> 1/USD_CAD)
    assert abs(base_to_home_rate("CAD_JPY", px) - 1.0 / 1.36) < 1e-9
    assert base_to_home_rate("ZZZ_JPY", px) is None         # no rate -> None (refuse)


def test_target_units_equal_home_notional_and_correct_base_semantics():
    # The bug being fixed: USD_JPY (base USD) must NOT be divided by ~150.
    targets = {"EUR_USD": 0.5, "USD_JPY": 0.5, "GBP_USD": 0.0}
    px = {"EUR_USD": 1.10, "USD_JPY": 150.0, "GBP_USD": 1.27}
    units = target_units(targets, nav=10000.0, last_prices=px, gross_leverage=0.5)
    assert units["GBP_USD"] == 0
    # each held name targets equal home (USD) notional = 0.5*10000*0.5 = 2500
    usd_eur = units["EUR_USD"] * base_to_home_rate("EUR_USD", px)
    usd_jpy = units["USD_JPY"] * base_to_home_rate("USD_JPY", px)
    assert abs(usd_eur - 2500) < 5 and abs(usd_jpy - 2500) < 5   # consistent, not 150x off
    assert units["USD_JPY"] > 2000           # base-USD: ~2500 units (NOT ~17 as before)


def test_target_units_zero_leverage_flattens_on_signals():
    targets = {"EUR_USD": 0.5, "USD_JPY": 0.5}
    px = {"EUR_USD": 1.10, "USD_JPY": 150.0}
    assert target_units(targets, nav=10000.0, last_prices=px, gross_leverage=0.0) == {
        "EUR_USD": 0,
        "USD_JPY": 0,
    }


def test_empty_panel_yields_no_targets():
    assert trend_targets(candles_to_close_panel({})) == {}
