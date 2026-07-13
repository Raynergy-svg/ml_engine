"""Regime-stratified reporting without lane-specific verdict drift."""

from __future__ import annotations

import pandas as pd

from .backtest import summarize_returns


def stratify_returns(returns: pd.Series, regimes: pd.Series) -> dict[str, dict[str, object]]:
    aligned_returns, aligned_regimes = returns.align(regimes, join="inner")
    if aligned_returns.empty or aligned_regimes.isna().any():
        raise ValueError("returns and regimes must have complete overlapping observations")
    return {
        str(regime): summarize_returns(aligned_returns[aligned_regimes == regime])
        for regime in sorted(aligned_regimes.unique(), key=str)
        if bool((aligned_regimes == regime).any())
    }
