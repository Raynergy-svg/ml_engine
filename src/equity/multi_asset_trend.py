"""Diversified multi-asset TREND / managed-futures portfolio — pre-registered (offline; running:NO).

Implements exactly the construction fixed in
``docs/experiment-multi-asset-trend-2026-06-29.md``: the ONE canonical trend rule
(price > 200d SMA, long-or-flat, shift(1) causal, monthly) applied UNIFORMLY to every
asset; per-asset net-return streams combined by the HRP-across-sleeves combiner; a
10% vol-target overlay. Judged at the PORTFOLIO level on the ship gate, out-of-sample.

No per-asset tuning, no market cherry-picking. OFFLINE evaluation; NOT wired to
execution. If a real edge appears it is REPORTED, never auto-promoted/traded.
"""
from __future__ import annotations

import logging
from typing import Dict

import numpy as np
import pandas as pd

from src.equity.backtest import overlay, stats
from src.equity.ship_gate import DEFAULT_DD_HARD, DEFAULT_DD_SOFT, DEFAULT_TARGET_VOL
from src.factor.ship_gate import evaluate_gate

logger = logging.getLogger(__name__)

SMA_WINDOW = 200
STEP = 21
COST_BPS_PER_SIDE = 2.0


def single_asset_trend_returns(price: pd.Series, *, sma_window: int = SMA_WINDOW,
                               step: int = STEP, cost_bps: float = COST_BPS_PER_SIDE) -> pd.Series:
    """Long-or-flat single-asset trend net return (price>SMA -> long, else cash).

    Strictly causal: on[t] = price[t-1] > SMA(<=t-1); weight applied with a 1-bar lag;
    monthly-held. Net of ``cost_bps`` per side on turnover."""
    px = pd.Series(price).dropna().astype(float)
    if len(px) < sma_window + step:
        return pd.Series(dtype=float)
    sma = px.rolling(sma_window, min_periods=max(20, sma_window // 2)).mean().shift(1)
    on = (px.shift(1) > sma).fillna(False)
    w = pd.Series(0.0, index=px.index)
    last = 0.0
    for i in range(len(px)):
        if i % step == 0:
            last = 1.0 if bool(on.iloc[i]) else 0.0
        w.iloc[i] = last
    applied = w.shift(1).fillna(0.0)             # execution lag >= 1
    gross = applied * px.pct_change()
    turnover = applied.diff().abs().fillna(applied.abs())
    cost = turnover * (float(cost_bps) * 1e-4)
    return (gross - cost).dropna()


def trend_streams(prices: pd.DataFrame, **kw) -> pd.DataFrame:
    """Per-asset single-asset trend net-return streams (date×asset)."""
    cols = {}
    for c in prices.columns:
        s = single_asset_trend_returns(prices[c], **kw)
        if not s.empty:
            cols[c] = s
    if not cols:
        return pd.DataFrame()
    return pd.DataFrame(cols).sort_index()


def combined_portfolio(prices: pd.DataFrame, *, target_vol: float = DEFAULT_TARGET_VOL,
                       **kw) -> pd.Series:
    """HRP-combine the per-asset trend streams, then 10% vol-target overlay -> net return.

    Combine = ``sleeve_combiner.combine_sleeves`` (HRP across the return matrix). Overlay
    is the causal vol-target + drawdown circuit-breaker (``backtest.overlay``)."""
    from src.equity.sleeve_combiner import combine_sleeves

    streams = trend_streams(prices, **kw)
    if streams.empty or streams.shape[1] < 2:
        return pd.Series(dtype=float)
    combined, _w = combine_sleeves(streams)
    combined = combined.dropna()
    if combined.empty:
        return combined
    scalar = overlay(combined.fillna(0.0), target_vol=target_vol,
                     dd_soft=DEFAULT_DD_SOFT, dd_hard=DEFAULT_DD_HARD, max_lev=3.0)
    net = (combined * scalar.reindex(combined.index).fillna(0.0)).dropna()
    return net


def gate_eval(net: pd.Series) -> Dict[str, object]:
    """Ship-gate verdict + stats for a net-return series."""
    s = stats(pd.Series(net).dropna())
    per_year = s.get("per_year", {}) or {}
    pos = sum(1 for v in per_year.values() if float(v) > 0)
    tot = len(per_year)
    net_sharpe = float(s.get("net_sharpe", 0.0))
    max_dd = float(s.get("max_dd", 1.0))
    verdict = evaluate_gate({"net_sharpe": net_sharpe, "positive_years": pos,
                             "total_years": tot, "max_drawdown": max_dd, "walk_forward": True})
    return {"net_sharpe": round(net_sharpe, 3), "max_dd": round(max_dd, 3),
            "positive_years": int(pos), "total_years": int(tot), "gate_pass": bool(verdict.passed)}


def ew_buy_hold(prices: pd.DataFrame) -> pd.Series:
    """Equal-weight buy-and-hold of all available assets (baseline)."""
    return prices.pct_change().mean(axis=1, skipna=True).dropna()


def sixty_forty(prices: pd.DataFrame, *, eq: str = "SPY", bond: str = "TLT") -> pd.Series:
    """60/40 SPY/TLT baseline (if both present)."""
    if eq not in prices.columns or bond not in prices.columns:
        return pd.Series(dtype=float)
    r = prices[[eq, bond]].pct_change()
    return (0.6 * r[eq] + 0.4 * r[bond]).dropna()


def per_asset_sharpes(prices: pd.DataFrame, **kw) -> Dict[str, float]:
    """Per-asset single-asset trend Sharpe — TRANSPARENCY ONLY (decision is the portfolio)."""
    out = {}
    for c in prices.columns:
        s = single_asset_trend_returns(prices[c], **kw)
        if len(s) > 60 and s.std() > 0:
            out[c] = round(float(s.mean() / s.std() * np.sqrt(252)), 2)
    return out
