"""BRIDGE — replaces ML head [momentum_score + acceleration] until:
  (a) Track C resolves direction head and proves ML stack produces signal
  (b) Trade journal reaches >=5K trades (sample sufficiency)
  (c) Feature set includes cross-pair / microstructure / calendar signals

Until then: deterministic formula. Same expression the leak was silently
approximating; now explicit and explainable. Promote to ML once (a)+(b)+(c) hold.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

DEFAULT_MOMENTUM_WINDOW = 10
DEFAULT_ACCELERATION_LAG = 5
DEFAULT_NORM_LOOKBACK = 1000
DEFAULT_NORM_MIN_PERIODS = 100


def compute_momentum_score(
    returns: pd.Series,
    window: int = DEFAULT_MOMENTUM_WINDOW,
    norm_factor: Optional[float] = None,
    lookback_for_norm: int = DEFAULT_NORM_LOOKBACK,
) -> pd.Series:
    """Rolling-mean absolute return, normalized to [0, 1].

    Mirrors the leaky training-time formula at
    ``src/core/modular_data_loaders.py:2986-3068`` (``load_xgboost_data``):

      raw_momentum[i] = mean(|returns[i-window:i]|)
      momentum_score[i] = min(raw_momentum[i] / norm_factor, 1.0)

    The training-time pipeline derived ``norm_factor`` from the train-slice
    median absolute return divided by 0.3 (so median maps to 0.3 after
    normalization). This bridge mirrors that scaling for parity.

    Args:
        returns: Series of bar-to-bar return values (signed). NaN is
            permitted; rolling-mean handles them.
        window: Trailing-window size for the absolute-return mean.
        norm_factor: Explicit normalization divisor. When None, computed
            online as a rolling ``lookback_for_norm``-bar 50th-percentile
            of ``|returns|`` divided by 0.3. Pass explicitly for strict
            inference parity with a saved training norm_factor.
        lookback_for_norm: Window size for the rolling normalization when
            ``norm_factor`` is None. Default 1000 bars (~6 weeks of H1).

    Returns:
        Series of float momentum scores in [0, 1] (NaN until ``window``
        rows accumulated; NaN until ``DEFAULT_NORM_MIN_PERIODS`` rows
        accumulated when ``norm_factor`` is None).
    """
    if not isinstance(returns, pd.Series):
        raise TypeError(f"returns must be pd.Series, got {type(returns)}")
    if window < 1:
        raise ValueError(f"window must be >= 1, got {window}")

    abs_returns = returns.abs()
    raw_momentum = abs_returns.rolling(window=window, min_periods=window).mean()

    if norm_factor is None:
        rolling_p50 = abs_returns.rolling(
            window=lookback_for_norm,
            min_periods=DEFAULT_NORM_MIN_PERIODS,
        ).quantile(0.5)
        # Scale so median absolute return maps to 0.3 (mirror train code line 3057)
        norm_series = (rolling_p50 / 0.3).where(rolling_p50 > 0, other=0.001)
        score = (raw_momentum / norm_series).clip(upper=1.0)
    else:
        if norm_factor <= 0:
            raise ValueError(f"norm_factor must be > 0, got {norm_factor}")
        score = (raw_momentum / norm_factor).clip(upper=1.0)

    return score.rename("momentum_score")


def compute_acceleration(
    momentum_score: pd.Series,
    lag: int = DEFAULT_ACCELERATION_LAG,
) -> pd.Series:
    """Binary indicator of momentum increase over ``lag`` bars.

    Mirrors the leaky training-time formula at
    ``src/core/modular_data_loaders.py:3032-3036``:

      acceleration[i] = 1 if momentum_score[i] > momentum_score[i-lag] else 0

    The leak audit at ``docs/label_leak_audit_2026-05-05.md`` notes the
    label is fully feature-derivable: it uses no future data, just two
    backward-looking momentum_score values. Replacing the model with
    this expression is a no-op in expected information.

    Args:
        momentum_score: Series produced by ``compute_momentum_score``.
        lag: Bars to look back for the comparison. Default 5.

    Returns:
        Series of float values in {0.0, 1.0, NaN}. NaN for the first
        ``lag`` rows (insufficient lookback) or wherever input
        ``momentum_score`` is NaN.
    """
    if not isinstance(momentum_score, pd.Series):
        raise TypeError(f"momentum_score must be pd.Series, got {type(momentum_score)}")
    if lag < 1:
        raise ValueError(f"lag must be >= 1, got {lag}")

    shifted = momentum_score.shift(lag)
    is_valid = shifted.notna() & momentum_score.notna()
    accel_values = np.where(
        is_valid,
        (momentum_score > shifted).astype(float),
        np.nan,
    )
    return pd.Series(accel_values, index=momentum_score.index, name="acceleration")
