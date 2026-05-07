"""Real-disk tests for the volatility_regime heuristic bridge.

NO MOCKS — uses real OHLCV from market_data/ and the real heuristic.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.scanner.heuristics.volatility_regime import (
    DEFAULT_LOOKBACK_BARS,
    VOL_REGIME_EXTREME,
    VOL_REGIME_HIGH,
    VOL_REGIME_LOW,
    VOL_REGIME_NORMAL,
    compute_volatility_regime,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
FIXTURE = REPO_ROOT / "market_data" / "oanda_EUR_USD_H1_live_20260103_222658.csv"


def _atr_pct_14(df: pd.DataFrame) -> pd.Series:
    """Compute ATR-14 percentage exactly as the runtime helper does.

    Mirrors src/core/modular_data_loaders.py::_compute_atr_pct_fast logic
    enough for the heuristic input: rolling-mean of true-range / close.
    """
    high = df["high"].to_numpy(dtype=np.float64)
    low = df["low"].to_numpy(dtype=np.float64)
    close = df["close"].to_numpy(dtype=np.float64)
    prev_close = np.roll(close, 1)
    prev_close[0] = close[0]
    tr = np.maximum.reduce([
        high - low,
        np.abs(high - prev_close),
        np.abs(low - prev_close),
    ])
    atr = pd.Series(tr).rolling(window=14, min_periods=1).mean().to_numpy()
    atr_pct = atr / np.maximum(close, 1e-10)
    return pd.Series(atr_pct, index=df.index, name="atr_pct_14")


@pytest.fixture(scope="module")
def fx_df() -> pd.DataFrame:
    df = pd.read_csv(FIXTURE).tail(500).reset_index(drop=True)
    return df


def test_shape_matches_input(fx_df: pd.DataFrame) -> None:
    atr_pct = _atr_pct_14(fx_df)
    regimes = compute_volatility_regime(atr_pct)
    assert len(regimes) == len(atr_pct)
    assert regimes.index.equals(atr_pct.index)


def test_values_in_valid_range(fx_df: pd.DataFrame) -> None:
    atr_pct = _atr_pct_14(fx_df)
    regimes = compute_volatility_regime(atr_pct)
    valid = {VOL_REGIME_LOW, VOL_REGIME_NORMAL, VOL_REGIME_HIGH, VOL_REGIME_EXTREME}
    assert set(regimes.unique()).issubset(valid)


def test_all_four_classes_present_on_500_bars(fx_df: pd.DataFrame) -> None:
    atr_pct = _atr_pct_14(fx_df)
    regimes = compute_volatility_regime(atr_pct)
    # Drop the warmup region (first lookback_bars rows are forced NORMAL)
    post_warmup = regimes.iloc[DEFAULT_LOOKBACK_BARS:]
    classes = set(post_warmup.unique())
    # Real H1 EUR_USD over 500 bars is volatile enough that the percentile
    # cuts at 0.25/0.60/0.85 generate all four classes.
    assert {VOL_REGIME_LOW, VOL_REGIME_NORMAL, VOL_REGIME_HIGH, VOL_REGIME_EXTREME}.issubset(
        classes
    ), f"Missing classes; got {sorted(classes)}"


def test_warmup_region_is_normal(fx_df: pd.DataFrame) -> None:
    atr_pct = _atr_pct_14(fx_df)
    regimes = compute_volatility_regime(atr_pct, lookback_bars=DEFAULT_LOOKBACK_BARS)
    # Rows 0..lookback_bars-1 should all equal NORMAL
    warmup = regimes.iloc[:DEFAULT_LOOKBACK_BARS]
    assert (warmup == VOL_REGIME_NORMAL).all()


def test_handles_nan_inf(fx_df: pd.DataFrame) -> None:
    atr_pct = _atr_pct_14(fx_df).copy()
    atr_pct.iloc[10] = np.nan
    atr_pct.iloc[20] = np.inf
    atr_pct.iloc[30] = -np.inf
    regimes = compute_volatility_regime(atr_pct)
    # No exceptions; values still in valid range
    valid = {VOL_REGIME_LOW, VOL_REGIME_NORMAL, VOL_REGIME_HIGH, VOL_REGIME_EXTREME}
    assert set(regimes.unique()).issubset(valid)


def test_invalid_input_raises() -> None:
    with pytest.raises(TypeError):
        compute_volatility_regime(np.array([1.0, 2.0, 3.0]))  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        compute_volatility_regime(pd.Series([0.01, 0.02]), lookback_bars=0)
    with pytest.raises(ValueError):
        compute_volatility_regime(pd.Series([0.01, 0.02]), thresholds=(0.5,))  # type: ignore[arg-type]
