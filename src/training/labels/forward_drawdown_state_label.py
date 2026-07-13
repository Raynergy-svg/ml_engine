"""Forward-realized drawdown-state labels (calm vs. stressed).

Greenfield target — a repo-wide search (2026-07-08, risk-target ML redirect
task) found no prior "drawdown_state"/"calm"/"stressed" label definition
anywhere in this codebase. This is a genuinely different statistic from the
existing forward-volatility-regime label
(`realized_volatility_regime_label.py`): volatility is a *dispersion*
measure (stddev of returns); drawdown is *path-dependent* (the worst
peak-to-trough decline actually realized within the window), so a quiet-
dispersion window can still contain a sharp single-direction drawdown, and
a high-dispersion (choppy, mean-reverting) window can have a shallow max
drawdown. Both are useful, distinct risk signals.

Same leak-prevention contract and train-only-cut-fitting methodology as the
volatility-regime label, for consistency:

    For row `i`, the label depends ONLY on `close[i+1 .. i+1+H]` — never on
    row `i` or earlier. The forward window starts at `i+1`, so no row-`i`
    feature can deterministically reconstruct the label from information
    available at prediction time.

Formula: within the forward window, track a running peak (starting from the
window's own first price, i.e. the peak resets at the start of each row's
own forward window — this is the standard "worst drawdown experienced
during the next H bars" definition, not a drawdown relative to some earlier
all-time high) and take the maximum peak-to-trough fractional decline
observed anywhere in the window:

    running_peak_t = max(close[i+1 .. t])          for t in [i+1, i+1+H)
    drawdown_t     = (running_peak_t - close_t) / running_peak_t
    max_dd[i]      = max(drawdown_t for t in [i+1, i+1+H))

`max_dd[i]` is then binarized: 1 ("stressed") if it exceeds the
`stressed_quantile` percentile of `max_dd` **fit on the train slice only**
(mirrors `_fit_cuts` in the vol-regime label), else 0 ("calm").
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Sentinel value for rows lacking a valid forward window (last
# `drawdown_horizon_bars` rows). Caller drops these before training.
DRAWDOWN_NAN_SENTINEL: int = -1

_DEFAULT_STRESSED_QUANTILE = 0.75
_EPS = 1e-12


def _resolve_close(df: pd.DataFrame) -> Optional[np.ndarray]:
    if "close" not in df.columns:
        return None
    return df["close"].astype(np.float64).values


def _compute_forward_max_drawdown(
    close: np.ndarray,
    drawdown_horizon_bars: int,
) -> np.ndarray:
    """Per-row forward max peak-to-trough decline (running peak resets at i+1).

    Rows with insufficient forward bars (`i + drawdown_horizon_bars >= len`)
    receive `np.nan`. Same tail-NaN convention as
    `_compute_forward_realized_vol` in the volatility-regime label module.
    """
    n = len(close)
    out = np.full(n, np.nan, dtype=np.float64)
    if n == 0:
        return out
    if drawdown_horizon_bars < 1:
        logger.warning(
            "_compute_forward_max_drawdown: drawdown_horizon_bars=%d too small (need >=1); all NaN",
            drawdown_horizon_bars,
        )
        return out

    safe_close = np.where(close > 0, close, np.nan)
    last_valid_i = n - drawdown_horizon_bars - 1
    if last_valid_i < 0:
        return out

    for i in range(last_valid_i + 1):
        start = i + 1
        end = i + 1 + drawdown_horizon_bars  # exclusive
        if end > n:
            break
        window = safe_close[start:end]
        if not np.all(np.isfinite(window)):
            continue
        running_peak = np.maximum.accumulate(window)
        # Guard against a zero/negative running peak (shouldn't happen for
        # FX close prices but be defensive rather than divide-by-zero).
        safe_peak = np.where(running_peak > _EPS, running_peak, np.nan)
        drawdowns = (running_peak - window) / safe_peak
        if not np.all(np.isfinite(drawdowns)):
            continue
        out[i] = float(np.max(drawdowns))

    return out


def _fit_stressed_cut(
    max_dd: np.ndarray,
    train_idx: Optional[Sequence[int]],
    stressed_quantile: float,
) -> Tuple[float, int]:
    """Fit the single stressed/calm cut point on the train slice (or all)."""
    if train_idx is not None:
        sample = max_dd[np.asarray(train_idx, dtype=int)]
    else:
        sample = max_dd
    finite = sample[np.isfinite(sample)]
    if finite.size == 0:
        logger.warning("_fit_stressed_cut: no finite max_dd samples; cut=0.0")
        return 0.0, 0
    cut = float(np.quantile(finite, stressed_quantile))
    return cut, int(finite.size)


def _class_balance(labels: np.ndarray) -> Dict[str, float]:
    valid = labels[labels >= 0]
    if valid.size == 0:
        return {"calm": 0.0, "stressed": 0.0}
    return {
        "calm": float(np.mean(valid == 0)),
        "stressed": float(np.mean(valid == 1)),
    }


def compute_forward_drawdown_state_labels(
    df: pd.DataFrame,
    drawdown_horizon_bars: int = 20,
    train_idx: Optional[Sequence[int]] = None,
    stressed_quantile: float = _DEFAULT_STRESSED_QUANTILE,
    val_idx: Optional[Sequence[int]] = None,
    test_idx: Optional[Sequence[int]] = None,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Compute forward-realized binary drawdown-state labels.

    Args:
        df: DataFrame with at least a `close` column.
        drawdown_horizon_bars: forward window length in bars (>= 1).
        train_idx: row indices used to fit the stressed/calm cut point (the
            train-slice max_dd distribution). If None, fits on all non-NaN
            rows. Recommended: pass the temporal-split train indices to
            avoid cross-split leakage into the cut point.
        stressed_quantile: percentile of train-slice max_dd above which a
            row is labeled "stressed" (default 0.75 — top quartile).
        val_idx, test_idx: optional, only used to populate per-split
            class-balance metadata.

    Returns:
        labels: np.ndarray of dtype int8, values in {0=calm, 1=stressed,
            -1=DRAWDOWN_NAN_SENTINEL for rows with insufficient forward
            window}. Caller MUST drop -1 rows before training.
        metadata: dict with `cut_value`, `n_train_used_for_cut`,
            `n_nan_dropped`, `class_balance` (all/train/val/test),
            `drawdown_horizon_bars`, `stressed_quantile`, `formula`.
    """
    if not (0.0 < stressed_quantile < 1.0):
        raise ValueError(
            f"compute_forward_drawdown_state_labels: stressed_quantile must be in (0,1), got {stressed_quantile}"
        )
    if drawdown_horizon_bars < 1:
        raise ValueError(
            f"compute_forward_drawdown_state_labels: drawdown_horizon_bars must be >= 1, got {drawdown_horizon_bars}"
        )

    n = len(df)
    metadata: Dict[str, Any] = {
        "drawdown_horizon_bars": int(drawdown_horizon_bars),
        "stressed_quantile": float(stressed_quantile),
        "formula": "max_running_peak_to_trough_decline_in_forward_window",
        "cut_value": 0.0,
        "n_train_used_for_cut": 0,
        "n_nan_dropped": 0,
        "class_balance": {"all": {}, "train": {}, "val": {}, "test": {}},
        "n_total": int(n),
    }

    if n == 0:
        return np.zeros(0, dtype=np.int8), metadata

    close = _resolve_close(df)
    if close is None:
        logger.warning(
            "compute_forward_drawdown_state_labels: missing 'close' column; all-sentinel labels",
        )
        labels = np.full(n, DRAWDOWN_NAN_SENTINEL, dtype=np.int8)
        metadata["n_nan_dropped"] = int(n)
        return labels, metadata

    if n <= drawdown_horizon_bars:
        logger.warning(
            "compute_forward_drawdown_state_labels: len(df)=%d <= horizon=%d; all-sentinel",
            n, drawdown_horizon_bars,
        )
        labels = np.full(n, DRAWDOWN_NAN_SENTINEL, dtype=np.int8)
        metadata["n_nan_dropped"] = int(n)
        return labels, metadata

    max_dd = _compute_forward_max_drawdown(close, drawdown_horizon_bars)

    cut, n_used = _fit_stressed_cut(max_dd, train_idx, stressed_quantile)
    metadata["cut_value"] = cut
    metadata["n_train_used_for_cut"] = n_used

    labels = np.full(n, DRAWDOWN_NAN_SENTINEL, dtype=np.int8)
    finite_mask = np.isfinite(max_dd)
    labels[finite_mask] = (max_dd[finite_mask] > cut).astype(np.int8)
    n_dropped = int(np.sum(~finite_mask))
    metadata["n_nan_dropped"] = n_dropped

    metadata["class_balance"]["all"] = _class_balance(labels)
    if train_idx is not None:
        metadata["class_balance"]["train"] = _class_balance(labels[np.asarray(train_idx, dtype=int)])
    if val_idx is not None:
        metadata["class_balance"]["val"] = _class_balance(labels[np.asarray(val_idx, dtype=int)])
    if test_idx is not None:
        metadata["class_balance"]["test"] = _class_balance(labels[np.asarray(test_idx, dtype=int)])

    logger.info(
        "compute_forward_drawdown_state_labels: n=%d horizon=%d cut=%.6f n_train_for_cut=%d "
        "n_nan_dropped=%d class_balance_all=%s",
        n, drawdown_horizon_bars, cut, n_used, n_dropped,
        {k: round(v, 3) for k, v in metadata["class_balance"]["all"].items()},
    )

    return labels, metadata


__all__ = [
    "compute_forward_drawdown_state_labels",
    "DRAWDOWN_NAN_SENTINEL",
]
