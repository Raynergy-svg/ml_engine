"""Variant 2 — long-only LOW-VOLATILITY tilt (price-only; offline; running:NO).

Tilts the long equal-weight book toward low realized-vol names by inverse-vol
weighting WITHIN the active membership set (no shorting, no leverage — this is
the long-only low-vol tilt, NOT the leveraged/short BAB construction). Strictly
causal: realized vol uses a trailing window through ``t-1`` (`shift(1)`), so the
weight at ``t`` cannot peek at ``t``'s return. Warmup rows (insufficient vol
history) fall back to equal weight.

SEPARATE variant module; baseline NOT modified. NON-hot-path.
"""
from __future__ import annotations

import numpy as np

from src.equity.variant_eval import aligned_inputs

DEFAULT_VOL_LOOKBACK = 63  # ~3 months — canonical, untuned


def lowvol_weights(snapshot, prices, *, vol_lookback: int = DEFAULT_VOL_LOOKBACK):
    """Causal inverse-realized-vol weights, long-only, rows sum to 1 over active."""
    ew, active, rets = aligned_inputs(snapshot, prices)
    min_periods = max(5, int(vol_lookback) // 2)
    rvol = rets.rolling(int(vol_lookback), min_periods=min_periods).std().shift(1)
    inv = (1.0 / rvol).where(active & rvol.gt(0))  # NaN where inactive / no vol
    raw = inv.where(np.isfinite(inv)).fillna(0.0)
    row_sum = raw.sum(axis=1)
    w = raw.div(row_sum.replace(0.0, np.nan), axis=0)
    # Warmup / all-zero rows -> equal weight (still long-only, sums to 1 over active)
    fallback = row_sum <= 0
    w = w.fillna(0.0)
    if fallback.any():
        w.loc[fallback] = ew.loc[fallback].values
    return w
