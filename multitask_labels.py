"""Label utilities for unified multi-head training.

These defaults are designed to work with the project's existing OHLCV CSVs.
They intentionally avoid adding new dependencies.

Label definitions (defaults):
- price: next-step scaled target from `prepare_sequences` (caller provides)
- trend: next-step return (close[t] / close[t-1] - 1)
- risk: realized volatility proxy (rolling std of returns over `risk_window`)
- state: volatility regime class (quantile bins over risk)

Important: These are *defaults*. For real trading, you may want more explicit,
product-specific definitions.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class MultitaskTargets:
    trend: np.ndarray  # (n,)
    direction: np.ndarray  # (n,)
    risk: np.ndarray  # (n,)
    state: np.ndarray  # (n,) int64 classes


def _safe_close_series(df: pd.DataFrame, close_col: str) -> np.ndarray:
    if close_col not in df.columns:
        raise ValueError(f"Missing close column: {close_col}")
    close = pd.to_numeric(df[close_col], errors="coerce").to_numpy(dtype=float)
    if np.isnan(close).any():
        close = np.nan_to_num(close, nan=np.nanmedian(close))
    return close


def build_multitask_targets(
    df: pd.DataFrame,
    *,
    sequence_length: int,
    target_shift: int,
    close_col: str = "close",
    risk_window: int = 14,
    state_classes: int = 3,
    train_split_fraction: Optional[float] = None,
) -> MultitaskTargets:
    """Build aligned targets for trend/risk/state.

    Returns arrays aligned to the sequences produced by `prepare_sequences`:
    index i corresponds to target time t = i + sequence_length + target_shift - 1.
    
    Args:
        df: DataFrame with price data
        sequence_length: Length of input sequences
        target_shift: How many steps ahead to predict
        close_col: Name of close price column
        risk_window: Window for rolling volatility calculation
        state_classes: Number of volatility regime classes
        train_split_fraction: If provided, compute risk quantiles only on first N%
                             of data to prevent data leakage. E.g., 0.8 means use
                             first 80% for quantile computation.
    
    Note:
        When train_split_fraction is provided, risk quantile edges are computed
        ONLY on the training portion to prevent leaking validation distribution
        into training. This is critical for proper time-series cross-validation.
    """
    if df is None or df.empty:
        raise ValueError("df is None or empty")
    if sequence_length <= 0 or target_shift <= 0:
        raise ValueError("sequence_length and target_shift must be > 0")
    if risk_window <= 1:
        raise ValueError("risk_window must be > 1")
    if state_classes < 2:
        raise ValueError("state_classes must be >= 2")

    close = _safe_close_series(df, close_col)

    # Simple returns; first return undefined -> 0.
    returns = np.zeros_like(close, dtype=float)
    returns[1:] = (close[1:] / np.clip(close[:-1], 1e-12, None)) - 1.0

    # Volatility proxy: rolling std of returns.
    risk = np.zeros_like(returns, dtype=float)
    for i in range(len(returns)):
        start = max(0, i - risk_window + 1)
        window = returns[start : i + 1]
        risk[i] = float(np.std(window))

    # Vol regime bins based on quantiles.
    # FIX: Compute quantiles only on training data to prevent leakage
    if train_split_fraction is not None:
        train_end_idx = int(len(risk) * train_split_fraction)
        risk_for_quantiles = risk[:train_end_idx]
    else:
        risk_for_quantiles = risk  # Legacy behavior: use all data
    
    qs = np.linspace(0.0, 1.0, state_classes + 1)
    edges = np.quantile(risk_for_quantiles, qs)
    edges = np.unique(edges)
    if len(edges) < 3:
        # Degenerate case: all same risk -> single bin
        state = np.zeros_like(risk, dtype=np.int64)
    else:
        # np.digitize returns 1..k; convert to 0..k-1
        # Apply quantile edges (computed on train) to ALL data
        state = np.digitize(risk, edges[1:-1], right=False).astype(np.int64)

    # Align to sequence targets.
    n = len(close) - sequence_length - target_shift + 1
    if n <= 0:
        raise ValueError("Not enough rows for sequence_length/target_shift")

    # Each sequence i ends at base index t0 = i + sequence_length - 1
    # and predicts at horizon t1 = t0 + target_shift.
    base_indices = np.arange(n, dtype=int) + sequence_length - 1
    target_indices = base_indices + target_shift

    # =========================================================================
    # TREND-AGNOSTIC LABELING (2025 Best Practice)
    # Use LOG RETURNS instead of price ratios for stationarity
    # Add DEAD-ZONE threshold to filter noise from spreads/random movements
    # =========================================================================
    
    # Log returns are stationary and trend-neutral (mean/variance don't drift)
    # This removes the bias where model learns "buy in downtrend" patterns
    horizon_log_return = np.log(
        close[target_indices] / np.clip(close[base_indices], 1e-12, None)
    )
    trend_aligned = horizon_log_return  # Use log returns for trend target too
    
    # Dynamic dead-zone threshold: 0.5× rolling standard deviation of returns
    # This filters out noise from spreads (0.5-2 pips) and random walk movements
    # Samples in dead-zone get direction=-1 (HOLD) and should be filtered/downweighted
    rolling_std = np.zeros_like(returns, dtype=float)
    std_window = min(100, len(returns) // 4)  # Adaptive window for small datasets
    for i in range(len(returns)):
        start = max(0, i - std_window + 1)
        window = returns[start : i + 1]
        rolling_std[i] = float(np.std(window)) if len(window) > 1 else 0.001
    
    # Use mean rolling std as threshold (0.5× std dev = half a standard deviation)
    # This is approximately where random noise ends and true signals begin
    threshold = np.mean(rolling_std[rolling_std > 0]) * 0.5
    threshold = max(threshold, 0.0001)  # Minimum threshold to avoid div-by-zero edge cases
    
    # Three-class labeling: 1=UP, 0=DOWN, -1=HOLD (dead-zone)
    # Dead-zone samples should be filtered or given lower sample weights during training
    direction_aligned = np.where(
        horizon_log_return > threshold, 1.0,  # Clear UP signal
        np.where(
            horizon_log_return < -threshold, 0.0,  # Clear DOWN signal
            -1.0  # Dead-zone: ambiguous, filter or downweight
        )
    ).astype(np.float32)

    # Risk/state remain aligned to the prediction time t1 (they are targets).
    risk_aligned = risk[target_indices]
    state_aligned = state[target_indices]

    # Normalize risk to [0,1] for sigmoid head compatibility.
    # FIX: Use training data statistics for normalization to prevent leakage
    if train_split_fraction is not None:
        # Compute normalization stats from training portion only
        train_n = int(n * train_split_fraction)
        train_risk = risk_aligned[:train_n]
        r_min = float(np.min(train_risk))
        r_max = float(np.max(train_risk))
    else:
        r_min = float(np.min(risk_aligned))
        r_max = float(np.max(risk_aligned))
    
    if r_max > r_min:
        risk_norm = (risk_aligned - r_min) / (r_max - r_min)
        # Clip to [0, 1] in case validation data has values outside training range
        risk_norm = np.clip(risk_norm, 0.0, 1.0)
    else:
        risk_norm = np.zeros_like(risk_aligned, dtype=float)

    return MultitaskTargets(
        trend=trend_aligned.astype(np.float32),
        direction=direction_aligned.astype(np.float32),
        risk=risk_norm.astype(np.float32),
        state=state_aligned.astype(np.int64),
    )


def split_time_series(
    n: int, 
    *, 
    val_fraction: float = 0.2,
    purge_gap: int = 0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Split time series data into train and validation sets.
    
    Args:
        n: Total number of samples
        val_fraction: Fraction of data for validation (default 0.2 = 20%)
        purge_gap: Number of samples to skip between train and val to prevent
                   temporal leakage. This creates a "buffer zone" where no
                   training or validation samples exist.
    
    Returns:
        Tuple of (train_indices, val_indices)
    
    Example with purge_gap=10:
        Data: [0, 1, 2, ..., 79, 80, 81, ..., 89, 90, 91, ..., 99]
                   TRAIN          PURGE GAP       VALIDATION
        
        train_idx: [0, 1, ..., 79]
        val_idx: [90, 91, ..., 99]
        (samples 80-89 are excluded to prevent leakage)
    
    Note:
        The purge gap is critical for time-series cross-validation because
        features computed with rolling windows can "see" into the future
        if there's no gap between train and validation.
    """
    if n <= 1:
        raise ValueError("n must be > 1")
    if not (0.0 < val_fraction < 1.0):
        raise ValueError("val_fraction must be in (0,1)")
    if purge_gap < 0:
        raise ValueError("purge_gap must be >= 0")

    # Calculate split point
    split = int(round(n * (1.0 - val_fraction)))
    split = max(1, min(n - 1 - purge_gap, split))
    
    # Train indices: from 0 to split
    train_idx = np.arange(0, split, dtype=int)
    
    # Validation indices: start after purge gap
    val_start = split + purge_gap
    if val_start >= n:
        # If purge gap is too large, reduce it
        val_start = split + max(0, (n - split) // 2)
    
    val_idx = np.arange(val_start, n, dtype=int)
    
    return train_idx, val_idx
