"""BRIDGE — replaces ML head [streak_prob] until:
  (a) Track C resolves direction head and proves ML stack produces signal
  (b) Trade journal reaches >=5K trades (sample sufficiency)
  (c) Feature set includes cross-pair / microstructure / calendar signals

Until then: deterministic formula. Same expression the leak was silently
approximating; now explicit and explainable. Promote to ML once (a)+(b)+(c) hold.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# Match the divide-by-zero guard at src/core/modular_data_loaders.py:3242
_VOL_ZERO_TOL = 1e-10
_FALLBACK_PROB = 0.5


def compute_streak_prob(
    volatility_10: pd.Series,
    volatility_20: pd.Series,
) -> pd.Series:
    """Loss-streak probability proxy from short/long volatility ratio.

    Mirrors the leaky training-time formula at
    ``src/core/modular_data_loaders.py:3239-3245``:

      streak_prob[i] = clip((volatility_10[i] / volatility_20[i] - 0.8) / 0.4, 0, 1)

    The leak audit at ``docs/label_leak_audit_2026-05-05.md`` confirmed
    this label is a deterministic linear-and-clipped expression of two
    columns that are also features in X. The LightGBM model was recovering
    that arithmetic with 150 trees of depth 6 — pure formula re-encoding,
    no learned signal.

    The runtime fallback ``_FALLBACK_PROB = 0.5`` matches the original
    implementation's behavior when ``volatility_20`` is near zero (a
    degenerate case where the ratio is undefined).

    Args:
        volatility_10: Series of 10-bar volatility values (non-negative).
        volatility_20: Series of 20-bar volatility values (non-negative).
            Must share index with ``volatility_10``.

    Returns:
        Series of float streak probabilities in [0, 1], indexed identically
        to inputs. Returns ``_FALLBACK_PROB`` (0.5) where ``volatility_20``
        is ~0 to match the leaky formula's div-by-zero guard.
    """
    if not isinstance(volatility_10, pd.Series):
        raise TypeError(f"volatility_10 must be pd.Series, got {type(volatility_10)}")
    if not isinstance(volatility_20, pd.Series):
        raise TypeError(f"volatility_20 must be pd.Series, got {type(volatility_20)}")
    if len(volatility_10) != len(volatility_20):
        raise ValueError(
            f"volatility_10 and volatility_20 must have equal length: "
            f"{len(volatility_10)} vs {len(volatility_20)}"
        )

    v10 = volatility_10.to_numpy(dtype=np.float64)
    v20 = volatility_20.to_numpy(dtype=np.float64)

    # Compute ratio with a div-by-zero guard; result is masked back to fallback below.
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(np.abs(v20) > _VOL_ZERO_TOL, v10 / v20, np.nan)
        prob = np.clip((ratio - 0.8) / 0.4, 0.0, 1.0)

    # Replace NaN (from div-by-zero or NaN inputs) with fallback
    prob = np.where(np.isnan(prob), _FALLBACK_PROB, prob)

    return pd.Series(prob, index=volatility_10.index, name="streak_prob", dtype=np.float64)
