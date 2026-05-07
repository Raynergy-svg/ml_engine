"""Real-disk tests for the momentum heuristic bridge.

NO MOCKS — uses real OHLCV from market_data/ + real numpy/pandas.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.scanner.heuristics.momentum import (
    DEFAULT_ACCELERATION_LAG,
    DEFAULT_MOMENTUM_WINDOW,
    compute_acceleration,
    compute_momentum_score,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
FIXTURE = REPO_ROOT / "market_data" / "oanda_EUR_USD_H1_live_20260103_222658.csv"


@pytest.fixture(scope="module")
def returns_series() -> pd.Series:
    df = pd.read_csv(FIXTURE).tail(2000).reset_index(drop=True)
    close = df["close"].astype(np.float64)
    returns = close.pct_change().fillna(0.0)
    returns.name = "returns"
    return returns


def test_momentum_score_shape_and_range(returns_series: pd.Series) -> None:
    score = compute_momentum_score(returns_series)
    assert isinstance(score, pd.Series)
    assert len(score) == len(returns_series)
    assert score.index.equals(returns_series.index)
    valid = score.dropna()
    assert (valid >= 0.0).all() and (valid <= 1.0).all()


def test_momentum_score_nan_until_window_warmup(returns_series: pd.Series) -> None:
    score = compute_momentum_score(returns_series, window=DEFAULT_MOMENTUM_WINDOW)
    # First (window-1) rows should be NaN (rolling-mean min_periods=window).
    assert score.iloc[: DEFAULT_MOMENTUM_WINDOW - 1].isna().all()


def test_momentum_score_with_explicit_norm_factor(returns_series: pd.Series) -> None:
    # Explicit norm_factor should clip at 1.0 and not depend on rolling lookback.
    score = compute_momentum_score(returns_series, norm_factor=0.001)
    valid = score.dropna()
    assert (valid >= 0.0).all() and (valid <= 1.0).all()


def test_momentum_score_invalid_inputs() -> None:
    with pytest.raises(TypeError, match="must be pd.Series"):
        compute_momentum_score(np.array([1.0, 2.0]))  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="window must be"):
        compute_momentum_score(pd.Series([0.01, 0.02]), window=0)
    with pytest.raises(ValueError, match="norm_factor must be"):
        compute_momentum_score(pd.Series([0.01, 0.02]), norm_factor=-1.0)


def test_acceleration_binary_with_nan_warmup(returns_series: pd.Series) -> None:
    score = compute_momentum_score(returns_series)
    accel = compute_acceleration(score, lag=DEFAULT_ACCELERATION_LAG)
    assert isinstance(accel, pd.Series)
    assert len(accel) == len(score)
    # Values should be in {0.0, 1.0, NaN}
    nonnan = accel.dropna()
    assert nonnan.isin([0.0, 1.0]).all()
    # Where momentum_score is NaN, acceleration must be NaN too
    nan_mask_score = score.isna()
    assert accel[nan_mask_score].isna().all()


def test_acceleration_first_lag_rows_nan() -> None:
    # Synthetic: pure ascending series, first `lag` rows must be NaN
    score = pd.Series(np.linspace(0.1, 0.9, 100), name="momentum_score")
    accel = compute_acceleration(score, lag=5)
    # First 5 rows: shifted is NaN, so accel must be NaN
    assert accel.iloc[:5].isna().all()
    # Ascending series → after warmup, all 1.0
    assert (accel.iloc[5:] == 1.0).all()


def test_acceleration_invalid_inputs() -> None:
    with pytest.raises(TypeError, match="must be pd.Series"):
        compute_acceleration(np.array([0.1, 0.2]))  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="lag must be"):
        compute_acceleration(pd.Series([0.1, 0.2]), lag=0)


def test_constant_returns_acceleration_all_zero() -> None:
    # If momentum_score is flat, acceleration should be 0 everywhere (after warmup).
    score = pd.Series(np.full(50, 0.3), name="momentum_score")
    accel = compute_acceleration(score, lag=5)
    # After warmup, all comparisons are 0.3 > 0.3 = False = 0
    assert (accel.iloc[5:] == 0.0).all()
