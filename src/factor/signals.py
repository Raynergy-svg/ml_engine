"""Factor signals — all causal, all in [-1, 1] per (date, pair).

US-003 Trend (TSMOM): self-sufficient from price alone.
US-002 Carry: cross-sectional rank of policy-rate differentials (needs a rate
              panel; ``carry_signal`` is a pure function of that panel).
US-004 Value: BIS REER deviation z-score (staged; pure function of a REER panel).

Causality contract: the signal at date T is a function of data with index <= T
only. Combined with the backtester's 1-day execution lag, no future information
can reach a position. ``tests/test_factor_signals.py`` proves window-invariance
in the style of ``tests/test_feature_window_invariance.py``.
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np
import pandas as pd

# Trading-day lookbacks for 3m / 6m / 12m.
TSMOM_LOOKBACKS = (63, 126, 252)


def tsmom_signal(
    close: pd.DataFrame,
    lookbacks: tuple = TSMOM_LOOKBACKS,
) -> pd.DataFrame:
    """Time-series momentum: equal-weighted mean of sign(return) over lookbacks.

    Parameters
    ----------
    close : date×pair frame of close prices.

    Returns a date×pair frame in [-1, 1]; rows with insufficient history (before
    the longest lookback) are NaN and must be excluded by the caller.
    """
    if not isinstance(close, pd.DataFrame):
        raise TypeError("tsmom_signal expects a date×pair DataFrame")
    parts = []
    for lb in lookbacks:
        ret = close / close.shift(lb) - 1.0
        parts.append(np.sign(ret))
    sig = sum(parts) / float(len(lookbacks))
    # Mask any column-row where the longest lookback isn't available yet.
    sig = sig.where(close.shift(max(lookbacks)).notna())
    return sig


def carry_signal(rate_diff: pd.DataFrame) -> pd.DataFrame:
    """Cross-sectional carry from policy-rate differentials (annualized, %).

    ``rate_diff[pair]`` is the *base-minus-quote* policy rate so a positive value
    means the long-the-pair position earns positive carry. Signal is the
    cross-sectional rank scaled to [-1, 1] (rank, not raw level, per US-002).
    """
    if rate_diff is None or rate_diff.empty:
        raise ValueError("carry_signal requires a non-empty rate_diff panel")
    ranked = rate_diff.rank(axis=1, method="average")
    n = ranked.count(axis=1)
    # Map rank 1..n -> [-1, 1]; centered, robust to n changing over time.
    centered = ranked.sub((n + 1) / 2.0, axis=0)
    scaled = centered.div((n - 1) / 2.0, axis=0)
    return scaled.where(rate_diff.notna())


def value_signal(reer: pd.DataFrame, window: int = 1260) -> pd.DataFrame:
    """Value: long cheap / short rich via REER deviation z-score (US-004).

    ``reer[pair]`` is the pair's real effective exchange rate proxy. A currency
    trading *below* its 5y (≈1260 trading day) mean is cheap -> long. We negate
    the z-score so cheap (negative deviation) maps to a positive (long) signal,
    then clip to [-1, 1].
    """
    if reer is None or reer.empty:
        raise ValueError("value_signal requires a non-empty reer panel")
    mean = reer.rolling(window, min_periods=window // 2).mean()
    std = reer.rolling(window, min_periods=window // 2).std()
    z = (reer - mean) / std
    return (-z).clip(-1.0, 1.0).where(std.notna())


def combine_signals(
    signals: List[pd.DataFrame],
    names: Optional[List[str]] = None,
) -> pd.DataFrame:
    """Equal-weighted average of available factor signals per (date, pair).

    NaNs (a factor not yet warmed up, or missing data for a pair) are skipped so
    a pair scored by 1 factor isn't penalized vs. one scored by 3. Dates/pairs
    are aligned on the union; the result is NaN only where *no* factor has a value.
    """
    if not signals:
        raise ValueError("combine_signals requires at least one signal frame")
    aligned = [s.reindex(index=signals[0].index.union(_union_index(signals)))
               for s in signals]
    cols = sorted(set().union(*[set(s.columns) for s in aligned]))
    aligned = [s.reindex(columns=cols) for s in aligned]
    stack = np.stack([s.to_numpy(dtype=float) for s in aligned], axis=0)
    with np.errstate(invalid="ignore"):
        combined = np.nanmean(stack, axis=0)
    out = pd.DataFrame(combined, index=aligned[0].index, columns=cols)
    return out


def _union_index(signals: List[pd.DataFrame]) -> pd.Index:
    idx = signals[0].index
    for s in signals[1:]:
        idx = idx.union(s.index)
    return idx
