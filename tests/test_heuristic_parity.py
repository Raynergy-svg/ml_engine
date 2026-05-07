"""Parity tests proving heuristic bridges are functionally equivalent
to the leaky training-time formulas they replace.

The point: the leak audit at docs/label_leak_audit_2026-05-05.md proved
the leaky labels are deterministic functions of features in X. The ML
heads were approximating these formulas. This test asserts that the
heuristic bridge IS the formula — i.e., switching from "model that
recovers the formula" to "the formula explicitly" loses no functional
capacity.

NO MOCKS — real numpy/pandas; in-test re-implementation of the leaky
formula compared to the heuristic bridge output. Mean absolute deviation
must be < 5% (the audit's empirical threshold; the model's recovery is
typically within 0.5%, but small numerical differences from the bridge's
NaN handling / vectorization are tolerated).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.scanner.heuristics.momentum import (
    compute_acceleration,
    compute_momentum_score,
)
from src.scanner.heuristics.streak_prob import compute_streak_prob
from src.scanner.heuristics.volatility_regime import (
    DEFAULT_LOOKBACK_BARS,
    compute_volatility_regime,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
FIXTURE = REPO_ROOT / "market_data" / "oanda_EUR_USD_H1_live_20260103_222658.csv"

PARITY_TOLERANCE = 0.05  # 5% mean-absolute-deviation cap


@pytest.fixture(scope="module")
def fx_df() -> pd.DataFrame:
    return pd.read_csv(FIXTURE).tail(2000).reset_index(drop=True)


def _atr_pct_14(df: pd.DataFrame) -> pd.Series:
    """Reproduce src/core/modular_data_loaders.py::_compute_atr_pct_fast."""
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
    return pd.Series(atr / np.maximum(close, 1e-10), index=df.index, name="atr_pct_14")


def _legacy_volatility_regime(atr_pct_14: pd.Series, lookback: int = 100,
                              thresholds: tuple = (0.25, 0.60, 0.85)) -> pd.Series:
    """In-test replica of compute_volatility_regime in modular_data_loaders.py:612-685."""
    n = len(atr_pct_14)
    arr = np.nan_to_num(atr_pct_14.to_numpy(dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0)
    regimes = np.full(n, 1, dtype=np.int32)  # NORMAL default
    for i in range(lookback, n):
        window = arr[i - lookback:i]
        if np.std(window) < 1e-10:
            continue
        rank = np.sum(window <= arr[i]) / lookback
        if rank < thresholds[0]:
            regimes[i] = 0
        elif rank < thresholds[1]:
            regimes[i] = 1
        elif rank < thresholds[2]:
            regimes[i] = 2
        else:
            regimes[i] = 3
    return pd.Series(regimes, index=atr_pct_14.index)


def _legacy_momentum_score(returns: np.ndarray, window: int = 10,
                           norm_factor: float = 1.0) -> np.ndarray:
    """In-test replica of the leaky momentum_score formula
    at src/core/modular_data_loaders.py:3018-3030."""
    n = len(returns)
    raw = np.zeros(n, dtype=np.float64)
    for i in range(window, n):
        raw[i] = np.mean(np.abs(returns[i - window:i]))
    return np.minimum(raw / max(norm_factor, 1e-10), 1.0)


def _legacy_streak_prob(v10: np.ndarray, v20: np.ndarray) -> np.ndarray:
    """In-test replica of the leaky streak_prob at modular_data_loaders.py:3239-3245."""
    n = len(v10)
    out = np.zeros(n, dtype=np.float64)
    for i in range(n):
        if abs(v20[i]) > 1e-10:
            ratio = v10[i] / v20[i]
            out[i] = np.clip((ratio - 0.8) / 0.4, 0.0, 1.0)
        else:
            out[i] = 0.5
    return out


# ---------------------------------------------------------------------
# Parity tests
# ---------------------------------------------------------------------


def test_volatility_regime_parity_post_warmup(fx_df: pd.DataFrame) -> None:
    atr = _atr_pct_14(fx_df)
    heuristic = compute_volatility_regime(atr)
    legacy = _legacy_volatility_regime(atr)

    # Compare post-warmup region (first lookback rows are NORMAL by design)
    post_warmup_h = heuristic.iloc[DEFAULT_LOOKBACK_BARS:]
    post_warmup_l = legacy.iloc[DEFAULT_LOOKBACK_BARS:]
    # Class-by-class agreement should be effectively 100% (same algorithm)
    agreement = (post_warmup_h.to_numpy() == post_warmup_l.to_numpy()).mean()
    assert agreement >= 0.99, f"Class agreement {agreement:.4f} below 99% threshold"


def test_momentum_score_parity(fx_df: pd.DataFrame) -> None:
    close = fx_df["close"].astype(np.float64).to_numpy()
    returns = np.zeros_like(close)
    returns[1:] = (close[1:] - close[:-1]) / close[:-1]

    # Use a fixed norm_factor for parity (the rolling default differs from
    # the legacy single-percentile-on-train-slice; that's expected and
    # tested separately in test_heuristic_momentum.py).
    norm_factor = 0.001
    legacy = _legacy_momentum_score(returns, window=10, norm_factor=norm_factor)

    heuristic = compute_momentum_score(
        pd.Series(returns), window=10, norm_factor=norm_factor
    )

    # Skip warmup (first window rows): legacy returns 0 (raw_momentum=0/norm_factor=0),
    # heuristic returns NaN (rolling-mean min_periods=window). Both correct, just
    # different conventions for "no-data".
    h_post = heuristic.iloc[10:].to_numpy()
    l_post = legacy[10:]
    mad = np.mean(np.abs(h_post - l_post))
    assert mad < PARITY_TOLERANCE, f"MAD {mad:.4f} exceeds 5% parity tolerance"


def test_acceleration_parity(fx_df: pd.DataFrame) -> None:
    close = fx_df["close"].astype(np.float64).to_numpy()
    returns = np.zeros_like(close)
    returns[1:] = (close[1:] - close[:-1]) / close[:-1]

    # Build momentum_score via the heuristic, then compute acceleration both ways
    score = compute_momentum_score(pd.Series(returns), window=10, norm_factor=0.001)
    heuristic_accel = compute_acceleration(score, lag=5)

    # Legacy formula: 1[score[i] > score[i-5]] (where defined)
    legacy_accel = np.full(len(score), np.nan, dtype=np.float64)
    score_arr = score.to_numpy()
    for i in range(5, len(score_arr)):
        if not np.isnan(score_arr[i]) and not np.isnan(score_arr[i - 5]):
            legacy_accel[i] = 1.0 if score_arr[i] > score_arr[i - 5] else 0.0

    # Class agreement on rows where both are defined
    h_arr = heuristic_accel.to_numpy()
    valid_mask = ~np.isnan(h_arr) & ~np.isnan(legacy_accel)
    if valid_mask.sum() == 0:
        pytest.skip("no overlapping non-NaN rows")
    agreement = (h_arr[valid_mask] == legacy_accel[valid_mask]).mean()
    assert agreement == 1.0, f"Acceleration parity {agreement:.4f} != 100%"


def test_streak_prob_parity(fx_df: pd.DataFrame) -> None:
    # Build vol_10 and vol_20 from returns (mirror the existing pipeline)
    close = fx_df["close"].astype(np.float64).to_numpy()
    returns = np.zeros_like(close)
    returns[1:] = np.diff(close) / np.maximum(close[:-1], 1e-10)
    n = len(returns)
    v10 = np.array([np.std(returns[max(0, i - 9):i + 1]) if i >= 1 else 0.01 for i in range(n)])
    v20 = np.array([np.std(returns[max(0, i - 19):i + 1]) if i >= 1 else 0.01 for i in range(n)])

    heuristic = compute_streak_prob(pd.Series(v10), pd.Series(v20)).to_numpy()
    legacy = _legacy_streak_prob(v10, v20)
    mad = np.mean(np.abs(heuristic - legacy))
    assert mad < PARITY_TOLERANCE, f"MAD {mad:.4f} exceeds 5% parity tolerance"
