"""Realized-outcome volatility regime labels.

Replaces the closed-form rolling-percentile volatility regime formula at
`src/core/modular_data_loaders.py:612-682` (`compute_volatility_regime`)
which leaked into the TCN feature matrix because:

- The label was a percentile rank of `atr_pct_14[i]` against the prior
  `atr_pct_14[i-100:i]` window.
- The TCN's X-feature list at `modular_data_loaders.py:3580-3595` includes
  `atr_pct_14`, `atr_pct_5/10/20`, `volatility_5/10/20`, `bb_width_norm`,
  `high_low_range` — i.e., the exact inputs that the rolling-percentile
  formula consumes.
- A TCN with 60-bar receptive field can therefore reconstruct the
  percentile-cut function almost exactly. The audit measured >>random
  accuracy as a result.

Fix path (option (a) from the audit / spec section 4.2): generate
forward-realized-volatility labels. For each row `i`, compute realized
volatility over the next `vol_horizon_bars` bars (using forward OHLC
only) and bin into 4 classes. The label depends ONLY on rows
`i+1 .. i+vol_horizon_bars`, never on row `i` features. The TCN may
still find weak forward-vol signal in `volume_ratio_*` or `bb_width_*`
but it cannot reconstruct the label from the features.

Cut-fitting basis: percentile cuts (corresponding to [0.25, 0.60, 0.85])
are fit on the TRAINING-SLICE realized_vol distribution only and applied
to all splits. This mirrors the train-only normalization pattern at
`modular_data_loaders.py:3050-3058` (the momentum loader).

Returns:
    labels (np.ndarray, dtype=int8, shape `(n,)`): integer regime in
        {0, 1, 2, 3} for rows with valid forward window, sentinel `-1`
        for the trailing rows whose forward window runs off the end.
        Caller MUST drop `-1` rows before training.
    metadata (dict): cut_values, n_train_used_for_cuts, n_nan_dropped,
        class_balance per split, vol_horizon_bars, formula,
        leak_fix_version.

Leak-prevention guarantee:
    For row `i`, the label is a function of `close[i+1 .. i+1+H]` only.
    No row-`i` feature column can deterministically reproduce the label.
    A leak-detection probe (in tests) trains a tiny LGBMClassifier on
    (X = same TCN feature columns, y = these labels) and asserts macro-F1
    < 0.6 on synthetic random-walk data — if the threshold is exceeded,
    a residual leak is present.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Default cut points (quantiles) used when fitting the 4-class binning.
# Match the legacy `compute_volatility_regime` thresholds for compatibility
# with the TCN trainer's `class_names = [QUIET, STABLE, ACTIVE, EXTREME]`.
_DEFAULT_CUT_QUANTILES: Tuple[float, float, float] = (0.25, 0.60, 0.85)

# Sentinel value for rows lacking a valid forward window (last
# `vol_horizon_bars` rows). Caller drops these before training.
NAN_SENTINEL: int = -1

# Small epsilon to keep the (stddev_log_returns / mean_close) formula
# well-defined when a window has zero close-mean (impossible for FX) or
# zero variance (very rare flat patches).
_EPS = 1e-12


def _resolve_close(df: pd.DataFrame) -> Optional[np.ndarray]:
    """Return close prices as a float64 numpy array, or None if absent."""
    if "close" not in df.columns:
        return None
    return df["close"].astype(np.float64).values


def _compute_forward_realized_vol(
    close: np.ndarray,
    vol_horizon_bars: int,
) -> np.ndarray:
    """Per-row forward realized volatility.

    For each row `i`:
        log_returns = diff(log(close[i+1 .. i+1+vol_horizon_bars]))
        realized_vol[i] = stddev(log_returns) / max(mean(close_window), eps)

    Rows with insufficient forward bars (`i + vol_horizon_bars >= len`)
    receive `np.nan`.

    The forward window starts at `i+1` so the label for row `i` is
    strictly a function of bars after `i` — no leak from row `i` itself.

    Returns:
        np.ndarray of shape `(len(close),)`, dtype float64.
    """
    n = len(close)
    out = np.full(n, np.nan, dtype=np.float64)
    if n == 0:
        return out

    # Need at least 2 forward bars to compute a stddev of log-returns.
    if vol_horizon_bars < 2:
        logger.warning(
            "_compute_forward_realized_vol: vol_horizon_bars=%d too small (need >=2); all NaN",
            vol_horizon_bars,
        )
        return out

    # Pre-compute log close once; the forward window slice picks from this.
    # Guard against non-positive close (shouldn't happen for FX but be safe).
    safe_close = np.where(close > 0, close, np.nan)
    log_close = np.log(safe_close)

    last_valid_i = n - vol_horizon_bars - 1
    if last_valid_i < 0:
        return out

    for i in range(last_valid_i + 1):
        start = i + 1
        end = i + 1 + vol_horizon_bars  # exclusive
        if end > n:
            break
        window_close = close[start:end]
        window_log = log_close[start:end]
        if not np.all(np.isfinite(window_log)):
            continue
        # log-returns within the forward window: end - start - 1 entries.
        log_rets = np.diff(window_log)
        if log_rets.size < 1:
            continue
        sd = float(np.std(log_rets, ddof=0))
        mean_c = float(np.mean(window_close))
        if not np.isfinite(sd) or not np.isfinite(mean_c):
            continue
        denom = max(abs(mean_c), _EPS)
        out[i] = sd / denom

    return out


def _fit_cuts(
    realized_vol: np.ndarray,
    train_idx: Optional[Sequence[int]],
    cut_quantiles: Sequence[float],
) -> Tuple[np.ndarray, int]:
    """Fit `len(cut_quantiles)` percentile cuts on the train slice (or all).

    Mirrors the train-only normalization pattern at
    `modular_data_loaders.py:3050-3058`.

    Args:
        realized_vol: per-row forward realized vol (with NaNs at the tail).
        train_idx: row indices to use for fitting. If None, fit on all
            non-NaN realized_vol rows.
        cut_quantiles: e.g., (0.25, 0.60, 0.85).

    Returns:
        cut_values: np.ndarray of length `len(cut_quantiles)`.
        n_used: number of finite samples used for fitting.
    """
    if train_idx is not None:
        sample = realized_vol[np.asarray(train_idx, dtype=int)]
    else:
        sample = realized_vol
    finite = sample[np.isfinite(sample)]
    if finite.size == 0:
        # Pathological: no valid forward windows. Fall back to zeros.
        # Caller will see all-NaN labels and drop the head.
        logger.warning("_fit_cuts: no finite realized_vol samples; cuts=zeros")
        return np.zeros(len(cut_quantiles), dtype=np.float64), 0
    cuts = np.quantile(finite, cut_quantiles).astype(np.float64)
    return cuts, int(finite.size)


def _bin_with_cuts(
    realized_vol: np.ndarray,
    cuts: np.ndarray,
) -> np.ndarray:
    """Bin per-row realized vol into 4 integer classes [0..3].

    Rows with NaN realized_vol (insufficient forward window) are tagged
    with the `NAN_SENTINEL` (-1) so the caller can drop them.
    """
    n = realized_vol.size
    labels = np.full(n, NAN_SENTINEL, dtype=np.int8)
    finite_mask = np.isfinite(realized_vol)
    # np.searchsorted with `side='right'` gives 0..len(cuts) inclusive,
    # which yields exactly 4 bins for 3 cuts.
    if cuts.size != 3:
        # Defensive: the public API hard-codes 3 cuts.
        raise ValueError(f"_bin_with_cuts expected 3 cuts, got {cuts.size}")
    bins = np.searchsorted(cuts, realized_vol[finite_mask], side="right")
    # Clip to valid class range (in case of ties at the upper edge).
    bins = np.clip(bins, 0, 3).astype(np.int8)
    labels[finite_mask] = bins
    return labels


def _class_balance(
    labels: np.ndarray,
    n_classes: int,
) -> Dict[str, float]:
    """Return per-class fraction (excluding sentinel rows)."""
    valid = labels[labels >= 0]
    out: Dict[str, float] = {}
    if valid.size == 0:
        for k in range(n_classes):
            out[str(k)] = 0.0
        return out
    for k in range(n_classes):
        out[str(k)] = float(np.mean(valid == k))
    return out


def compute_forward_realized_volatility(
    df: pd.DataFrame,
    vol_horizon_bars: int,
    *,
    annualization_factor: Optional[float] = None,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Continuous-value counterpart to `compute_realized_volatility_regime_labels`.

    Returns the raw forward-realized-volatility value per row instead of a
    binned regime class — for a regression target (e.g. QLIKE/pinball-scored
    vol forecasting) rather than a 4-class classification target. Reuses the
    exact same leak-free forward-window formula (`_compute_forward_realized_vol`):
    for row `i`, the value depends ONLY on `close[i+1 .. i+1+vol_horizon_bars]`,
    never on row `i` or earlier — same leak-prevention guarantee as the binned
    version (see module docstring).

    Args:
        df: DataFrame with at least a `close` column.
        vol_horizon_bars: forward window length in bars (must be >= 2).
        annualization_factor: if provided, multiplies the raw (per-bar) stddev
            by this factor (e.g. sqrt(252) for daily bars) before returning.
            None leaves the value in raw per-bar units.

    Returns:
        values: np.ndarray of dtype float64, shape (len(df),). NaN for rows
            with insufficient forward window (the trailing `vol_horizon_bars`
            rows) — caller MUST drop NaN rows before training (unlike the
            binned version, there is no integer sentinel; NaN is the sentinel
            here since this is a continuous target).
        metadata: dict with `vol_horizon_bars`, `n_total`, `n_nan_dropped`,
            `annualization_factor`, `formula`, `leak_fix_version`.
    """
    if vol_horizon_bars < 2:
        raise ValueError(
            f"compute_forward_realized_volatility: vol_horizon_bars must be >= 2, got {vol_horizon_bars}"
        )
    n = len(df)
    metadata: Dict[str, Any] = {
        "vol_horizon_bars": int(vol_horizon_bars),
        "annualization_factor": annualization_factor,
        "formula": "stddev_log_returns_over_mean_close" + (
            "_annualized" if annualization_factor else ""
        ),
        "leak_fix_version": "2026-05-06",
        "n_total": int(n),
        "n_nan_dropped": 0,
    }
    if n == 0:
        return np.zeros(0, dtype=np.float64), metadata

    close = _resolve_close(df)
    if close is None:
        logger.warning(
            "compute_forward_realized_volatility: missing 'close' column; returning all-NaN",
        )
        metadata["n_nan_dropped"] = int(n)
        return np.full(n, np.nan, dtype=np.float64), metadata

    values = _compute_forward_realized_vol(close, vol_horizon_bars)
    if annualization_factor:
        values = values * float(annualization_factor)
    metadata["n_nan_dropped"] = int(np.sum(~np.isfinite(values)))
    return values, metadata


def compute_realized_volatility_regime_labels(
    df: pd.DataFrame,
    vol_horizon_bars: int = 24,
    n_classes: int = 4,
    train_idx: Optional[Sequence[int]] = None,
    cut_quantiles: Sequence[float] = _DEFAULT_CUT_QUANTILES,
    val_idx: Optional[Sequence[int]] = None,
    test_idx: Optional[Sequence[int]] = None,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Compute forward-realized 4-class volatility regime labels.

    For each row `i`:
        forward window = close[i+1 .. i+1+vol_horizon_bars]
        log_returns    = diff(log(forward window))
        realized_vol[i] = stddev(log_returns) / mean(forward window)

    Then bin realized_vol[] into `n_classes` classes using percentile cuts
    fit on `realized_vol[train_idx]` (or on all non-NaN values if
    `train_idx` is None).

    The last `vol_horizon_bars` rows have insufficient forward window and
    receive the `NAN_SENTINEL` (-1) value. Caller MUST drop these before
    training.

    Args:
        df: DataFrame with at least a `close` column. Other columns are
            ignored.
        vol_horizon_bars: how many forward bars to scan (operator-confirmed
            default 24, matching the confidence head's lookahead). Must be
            >= 2 for the stddev to be defined.
        n_classes: must be 4 (matches TCN trainer's class_names list).
        train_idx: row indices to use for fitting the percentile cuts.
            If None, cuts are fit on all non-NaN realized_vol rows.
            Recommended: pass the temporal-split train slice indices to
            avoid cross-split leakage from val/test into the cut points.
        cut_quantiles: 3 quantile points (must yield n_classes-1 cuts).
            Default (0.25, 0.60, 0.85) matches the legacy
            `VolatilityRegimeConfig.thresholds`.
        val_idx, test_idx: optional, only used to populate per-split
            class-balance metadata.

    Returns:
        labels: np.ndarray of dtype int8 with values in {0, 1, 2, 3, -1}
            where -1 is the NaN sentinel. Shape (len(df),).
        metadata: dict containing:
            - cut_values: list of 3 floats (the fit percentile cuts)
            - n_train_used_for_cuts: int
            - n_nan_dropped: int (count of -1 sentinel rows)
            - class_balance: {"all": {...}, "train": {...}, "val": {...},
                              "test": {...}}
            - vol_horizon_bars: int
            - n_classes: int
            - formula: str
            - leak_fix_version: str
    """
    if n_classes != 4:
        raise ValueError(
            f"compute_realized_volatility_regime_labels: n_classes must be 4 "
            f"(matches TCN trainer's 4-class head), got {n_classes}"
        )
    if len(cut_quantiles) != n_classes - 1:
        raise ValueError(
            f"compute_realized_volatility_regime_labels: cut_quantiles must "
            f"have length n_classes-1={n_classes - 1}, got {len(cut_quantiles)}"
        )
    if vol_horizon_bars < 2:
        raise ValueError(
            f"compute_realized_volatility_regime_labels: vol_horizon_bars must "
            f"be >= 2, got {vol_horizon_bars}"
        )

    n = len(df)
    metadata: Dict[str, Any] = {
        "vol_horizon_bars": int(vol_horizon_bars),
        "n_classes": int(n_classes),
        "formula": "stddev_log_returns_over_mean_close",
        "leak_fix_version": "2026-05-06",
        "cut_values": [],
        "n_train_used_for_cuts": 0,
        "n_nan_dropped": 0,
        "class_balance": {"all": {}, "train": {}, "val": {}, "test": {}},
        "n_total": int(n),
    }

    if n == 0:
        labels = np.zeros(0, dtype=np.int8)
        return labels, metadata

    close = _resolve_close(df)
    if close is None:
        logger.warning(
            "compute_realized_volatility_regime_labels: missing 'close' column; "
            "returning all-NaN labels (caller should drop the head)",
        )
        labels = np.full(n, NAN_SENTINEL, dtype=np.int8)
        metadata["n_nan_dropped"] = int(n)
        return labels, metadata

    # Edge case: not enough rows to compute any forward window.
    if n <= vol_horizon_bars:
        logger.warning(
            "compute_realized_volatility_regime_labels: len(df)=%d <= "
            "vol_horizon_bars=%d; all rows will be NaN-sentinel",
            n,
            vol_horizon_bars,
        )
        labels = np.full(n, NAN_SENTINEL, dtype=np.int8)
        metadata["n_nan_dropped"] = int(n)
        return labels, metadata

    # 1) Forward realized vol (NaN at tail).
    realized_vol = _compute_forward_realized_vol(close, vol_horizon_bars)

    # 2) Fit cuts on train slice (or all non-NaN if train_idx not provided).
    cuts, n_used = _fit_cuts(realized_vol, train_idx, cut_quantiles)
    metadata["cut_values"] = [float(c) for c in cuts]
    metadata["n_train_used_for_cuts"] = int(n_used)

    # 3) Bin → integer labels with NAN_SENTINEL at the tail.
    labels = _bin_with_cuts(realized_vol, cuts)
    n_dropped = int(np.sum(labels == NAN_SENTINEL))
    metadata["n_nan_dropped"] = n_dropped

    # 4) Per-split class balance (caller may pass val_idx, test_idx).
    metadata["class_balance"]["all"] = _class_balance(labels, n_classes)
    if train_idx is not None:
        metadata["class_balance"]["train"] = _class_balance(
            labels[np.asarray(train_idx, dtype=int)], n_classes
        )
    if val_idx is not None:
        metadata["class_balance"]["val"] = _class_balance(
            labels[np.asarray(val_idx, dtype=int)], n_classes
        )
    if test_idx is not None:
        metadata["class_balance"]["test"] = _class_balance(
            labels[np.asarray(test_idx, dtype=int)], n_classes
        )

    logger.info(
        "compute_realized_volatility_regime_labels: n=%d horizon=%d cuts=%s "
        "n_train_for_cuts=%d n_nan_dropped=%d class_balance_all=%s",
        n,
        vol_horizon_bars,
        ["%.6e" % c for c in cuts],
        n_used,
        n_dropped,
        {k: round(v, 3) for k, v in metadata["class_balance"]["all"].items()},
    )

    return labels, metadata


__all__ = [
    "compute_realized_volatility_regime_labels",
    "compute_forward_realized_volatility",
    "NAN_SENTINEL",
]
