"""Sleeve A — multi-asset trend / managed-futures (long-or-flat; offline; running:NO).

An ORTHOGONAL return stream to the long equal-weight equity book: time-series
momentum across a broad multi-asset ETF set. Each asset is HELD (equal weight
among the "on" set) when its price is above its long SMA, else CASH — long-only,
no shorting, no leverage. The long-or-flat switch means the sleeve goes to cash
in sustained drawdowns, so it is weakly/negatively correlated to long equity
exactly when that matters (the crisis-diversifier property).

PRE-REGISTERED params (canonical Faber GTAA / TSMOM; fixed BEFORE seeing results,
anti-p-hacking): ETF set = SPY/EFA/TLT/IEF/GLD/DBC/VNQ; rule = price > 200-day SMA
(both `shift(1)` -> strictly causal); monthly rebalance (21d); cost 2bps;
execution_lag>=1. Not tuned to the sample.

SEPARATE sleeve module; baseline NOT modified. NON-hot-path; price-only (free
yfinance); nothing runs in a process or trades.
"""
from __future__ import annotations

import pandas as pd

from src.equity.backtest import DEFAULT_COST_BPS, DEFAULT_EXECUTION_LAG, run_portfolio_backtest

TREND_ETFS = ("SPY", "EFA", "TLT", "IEF", "GLD", "DBC", "VNQ")
DEFAULT_SMA = 200       # ~10-month SMA — canonical GTAA, untuned
DEFAULT_STEP = 21       # monthly rebalance


def _normalize(prices: pd.DataFrame) -> pd.DataFrame:
    px = prices.copy()
    if not isinstance(px.index, pd.DatetimeIndex):
        px.index = pd.DatetimeIndex(px.index)
    if px.index.tz is None:
        px.index = px.index.tz_localize("UTC")
    else:
        px.index = px.index.tz_convert("UTC")
    px.index = px.index.normalize()
    return px


def trend_sleeve_weights(
    prices: pd.DataFrame,
    *,
    sma_window: int = DEFAULT_SMA,
    step: int = DEFAULT_STEP,
):
    """Causal long-or-flat trend weights: equal-weight the "on" assets, else cash.

    ``on[t] = price[t-1] > SMA(prices through t-1)`` — uses only data <= t-1.
    Targets are set on rebalance dates and held until the next one.
    """
    px = _normalize(prices)
    cols = [c for c in TREND_ETFS if c in px.columns] or list(px.columns)
    px = px[cols]
    sma = px.rolling(int(sma_window), min_periods=max(20, int(sma_window) // 2)).mean().shift(1)
    on = (px.shift(1) > sma).fillna(False)  # uptrend as of t-1
    idx = px.index
    targets = pd.DataFrame(0.0, index=idx, columns=cols)
    last = pd.Series(0.0, index=cols)
    for i in range(len(idx)):
        if i % int(step) == 0:
            row_on = on.iloc[i]
            n = int(row_on.sum())
            w = pd.Series(0.0, index=cols)
            if n > 0:
                w[row_on.values] = 1.0 / n
            last = w
        targets.iloc[i] = last.values
    return targets


def trend_sleeve_returns(
    prices: pd.DataFrame,
    *,
    sma_window: int = DEFAULT_SMA,
    step: int = DEFAULT_STEP,
    cost_bps: float = DEFAULT_COST_BPS,
    execution_lag: int = DEFAULT_EXECUTION_LAG,
) -> pd.Series:
    """Net-of-cost daily return series for the trend sleeve (for the combiner).

    Trims to the window where every ETF has data (ETFs have different inception
    dates — e.g. DBC ~2006-02); internal gaps are forward-filled, leading
    pre-inception rows are dropped (the backtest refuses NaN, by design).
    """
    px = _normalize(prices)
    cols = [c for c in TREND_ETFS if c in px.columns] or list(px.columns)
    px = px[cols].ffill()
    px = px.loc[px.notna().all(axis=1)]  # window where ALL ETFs exist
    w = trend_sleeve_weights(px, sma_window=sma_window, step=step)
    px = px.reindex(index=w.index, columns=w.columns)
    result = run_portfolio_backtest(
        w,
        px,
        cost_bps=cost_bps,
        slippage_bps_per_pct_adv=0.0,
        adv=None,
        execution_lag=execution_lag,
    )
    return result.returns
