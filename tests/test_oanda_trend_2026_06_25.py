"""OANDA practice TREND runner — pure-logic tests (no network, no mocks).

Verifies the candle->panel parse (complete-bars-only), the trend signal (long-or-
flat, causal, reuses the validated trend_sleeve rule), and unit sizing. Real
functions against real synthetic OANDA candle dicts; the broker/network is NOT
exercised here (that needs a live practice token, surfaced as the blocker).
"""
from __future__ import annotations

import numpy as np

from src.equity.oanda_trend import (
    candles_to_close_panel,
    target_units,
    trend_targets,
)


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


def test_target_units_long_or_flat_and_conservative():
    targets = {"EUR_USD": 0.5, "USD_JPY": 0.5, "GBP_USD": 0.0}
    units = target_units(targets, nav=10000.0, last_prices={
        "EUR_USD": 1.10, "USD_JPY": 150.0, "GBP_USD": 1.25}, gross_leverage=0.5)
    assert units["GBP_USD"] == 0          # flat name -> 0 units
    assert units["EUR_USD"] > 0           # on name -> long
    # gross notional <= leverage * NAV (conservative): sum(|units|*price) bound
    gross = abs(units["EUR_USD"]) * 1.10 + abs(units["USD_JPY"]) * 150.0
    assert gross <= 0.5 * 10000.0 + 200.0  # ~<= leverage*NAV (+rounding slack)


def test_empty_panel_yields_no_targets():
    assert trend_targets(candles_to_close_panel({})) == {}
