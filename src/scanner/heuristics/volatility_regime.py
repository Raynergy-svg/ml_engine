"""BRIDGE — replaces ML head [volatility_regime] until:
  (a) Track C resolves direction head and proves ML stack produces signal
  (b) Trade journal reaches >=5K trades (sample sufficiency)
  (c) Feature set includes cross-pair / microstructure / calendar signals

Until then: deterministic formula. Same expression the leak was silently
approximating; now explicit and explainable. Promote to ML once (a)+(b)+(c) hold.
"""
from __future__ import annotations

from typing import Tuple

import numpy as np
import pandas as pd

# Volatility regime constants (mirror src/core/modular_data_loaders.py:590-593)
VOL_REGIME_LOW = 0
VOL_REGIME_NORMAL = 1
VOL_REGIME_HIGH = 2
VOL_REGIME_EXTREME = 3

DEFAULT_LOOKBACK_BARS = 100
DEFAULT_THRESHOLDS: Tuple[float, float, float] = (0.25, 0.60, 0.85)


def compute_volatility_regime(
    atr_pct_14: pd.Series,
    lookback_bars: int = DEFAULT_LOOKBACK_BARS,
    thresholds: Tuple[float, float, float] = DEFAULT_THRESHOLDS,
) -> pd.Series:
    """Bin ATR percentile-rank into LOW/NORMAL/HIGH/EXTREME regimes.

    For each row i, computes the percentile rank of ``atr_pct_14[i]`` against
    the trailing ``lookback_bars`` window of ``atr_pct_14`` and bins the rank
    using ``thresholds``.

    This mirrors the leaky training-time formula at
    ``src/core/modular_data_loaders.py:612-685`` (``compute_volatility_regime``).
    The TCN model trained on those labels was approximating this exact
    expression — the audit at ``docs/label_leak_audit_2026-05-05.md:38-81``
    proved the label is closed-form derivable from a feature
    (``atr_pct_14``) the model could already see. Switching the runtime to
    this heuristic loses no signal capacity (parity within numerical
    precision; see ``tests/test_heuristic_parity.py``).

    Args:
        atr_pct_14: Series of ATR percentage values (non-negative). NaN/inf
            are coerced to 0.0 to match the leaky-formula implementation.
        lookback_bars: Trailing-window size for the percentile rank.
        thresholds: 3-tuple of percentile cuts. Values strictly less than
            ``thresholds[0]`` map to LOW; ``[thresholds[0], thresholds[1])``
            map to NORMAL; ``[thresholds[1], thresholds[2])`` map to HIGH;
            ``>= thresholds[2]`` map to EXTREME.

    Returns:
        Series of int regime labels in ``{0, 1, 2, 3}``, indexed identically
        to ``atr_pct_14``. The first ``lookback_bars`` rows are filled with
        ``VOL_REGIME_NORMAL`` (insufficient lookback) — the same default the
        training-time formula uses.
    """
    if not isinstance(atr_pct_14, pd.Series):
        raise TypeError(f"atr_pct_14 must be pd.Series, got {type(atr_pct_14)}")
    if lookback_bars < 1:
        raise ValueError(f"lookback_bars must be >= 1, got {lookback_bars}")
    if len(thresholds) != 3:
        raise ValueError(f"thresholds must have 3 entries, got {len(thresholds)}")

    n = len(atr_pct_14)
    # Coerce NaN/inf -> 0.0 (mirrors np.nan_to_num at modular_data_loaders.py:645)
    atr = np.nan_to_num(atr_pct_14.to_numpy(dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0)

    # Default = NORMAL (matches modular_data_loaders.py:648)
    regimes = np.full(n, VOL_REGIME_NORMAL, dtype=np.int32)

    t_low, t_normal, t_high = thresholds[0], thresholds[1], thresholds[2]

    for i in range(lookback_bars, n):
        window = atr[i - lookback_bars:i]
        current = atr[i]

        # Skip if window has no variance (matches modular_data_loaders.py:659)
        if np.std(window) < 1e-10:
            continue

        # Percentile rank: fraction of window <= current
        pct_rank = float(np.sum(window <= current)) / float(lookback_bars)

        if pct_rank < t_low:
            regimes[i] = VOL_REGIME_LOW
        elif pct_rank < t_normal:
            regimes[i] = VOL_REGIME_NORMAL
        elif pct_rank < t_high:
            regimes[i] = VOL_REGIME_HIGH
        else:
            regimes[i] = VOL_REGIME_EXTREME

    return pd.Series(regimes, index=atr_pct_14.index, name="volatility_regime")
