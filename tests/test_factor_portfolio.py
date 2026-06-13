"""No-mock property tests for portfolio construction + guards (US-005)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.factor.portfolio import (
    target_weights,
    apply_no_trade_band,
    enforce_guards,
    weekly_rebalance_mask,
    GROSS_CAP,
    PER_PAIR_CAP,
)


def _scores_returns(n=300, seed=1):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2014-01-01", periods=n, tz="UTC")
    pairs = ["EUR_USD", "USD_JPY", "GBP_USD", "AUD_USD"]
    score = pd.DataFrame(rng.uniform(-1, 1, (n, len(pairs))), index=idx, columns=pairs)
    returns = pd.DataFrame(
        rng.normal(0, 0.006, (n, len(pairs))), index=idx, columns=pairs
    )
    return score, returns


def test_target_weights_respect_caps():
    score, returns = _scores_returns()
    w = target_weights(score, returns)
    gross = w.abs().sum(axis=1)
    assert (gross <= GROSS_CAP + 1e-6).all()
    # Per-position absolute ceiling = per_pair_cap × gross_cap.
    assert (w.abs() <= PER_PAIR_CAP * GROSS_CAP + 1e-6).to_numpy().all()


def test_higher_vol_target_raises_gross():
    score, returns = _scores_returns()
    lo = target_weights(score, returns, vol_target=0.05).abs().sum(axis=1).mean()
    hi = target_weights(score, returns, vol_target=0.20).abs().sum(axis=1).mean()
    assert hi > lo


def test_enforce_guards_clamps_overleveraged_book():
    idx = pd.bdate_range("2014-01-01", periods=3, tz="UTC")
    cols = ["A", "B", "C"]
    # Deliberately breach both caps.
    over = pd.DataFrame([[3.0, 3.0, 3.0]] * 3, index=idx, columns=cols)
    clamped = enforce_guards(over)
    gross = clamped.abs().sum(axis=1)
    assert (gross <= GROSS_CAP + 1e-9).all()
    assert (clamped.abs() <= PER_PAIR_CAP * GROSS_CAP + 1e-9).to_numpy().all()


def test_no_trade_band_cuts_turnover():
    score, returns = _scores_returns(seed=7)
    raw = target_weights(score, returns)
    rb = weekly_rebalance_mask(score.index)
    banded = apply_no_trade_band(raw, band=0.05, rebalance_mask=rb)
    turn_raw = raw.diff().abs().sum(axis=1).sum()
    turn_banded = banded.diff().abs().sum(axis=1).sum()
    assert turn_banded < turn_raw
