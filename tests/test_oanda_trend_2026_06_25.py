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
    target_units,
    trend_targets,
)


def test_clamp_leverage_bounds():
    from src.equity.oanda_trend import DEFAULT_GROSS_LEVERAGE, MAX_GROSS_LEVERAGE, clamp_leverage
    assert clamp_leverage(3.0) == 3.0
    assert clamp_leverage(0) == DEFAULT_GROSS_LEVERAGE       # invalid -> default
    assert clamp_leverage(-5) == DEFAULT_GROSS_LEVERAGE
    assert clamp_leverage(999) == MAX_GROSS_LEVERAGE         # capped at margin-call guard
    assert clamp_leverage(10) == 10.0


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


def test_empty_panel_yields_no_targets():
    assert trend_targets(candles_to_close_panel({})) == {}
