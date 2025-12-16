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
from typing import Tuple

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
) -> MultitaskTargets:
    """Build aligned targets for trend/risk/state.

    Returns arrays aligned to the sequences produced by `prepare_sequences`:
    index i corresponds to target time t = i + sequence_length + target_shift - 1.
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
    qs = np.linspace(0.0, 1.0, state_classes + 1)
    edges = np.quantile(risk, qs)
    edges = np.unique(edges)
    if len(edges) < 3:
        # Degenerate case: all same risk -> single bin
        state = np.zeros_like(risk, dtype=np.int64)
    else:
        # np.digitize returns 1..k; convert to 0..k-1
        state = np.digitize(risk, edges[1:-1], right=False).astype(np.int64)

    # Align to sequence targets.
    n = len(close) - sequence_length - target_shift + 1
    if n <= 0:
        raise ValueError("Not enough rows for sequence_length/target_shift")

    # Each sequence i ends at base index t0 = i + sequence_length - 1
    # and predicts at horizon t1 = t0 + target_shift.
    base_indices = np.arange(n, dtype=int) + sequence_length - 1
    target_indices = base_indices + target_shift

    # Trend/direction are defined over the prediction horizon (t0 -> t1),
    # not the one-step return at t1. This makes the direction head learnable
    # for the same horizon as the price target.
    horizon_return = (close[target_indices] / np.clip(close[base_indices], 1e-12, None)) - 1.0
    trend_aligned = horizon_return
    direction_aligned = (horizon_return > 0).astype(np.float32)

    # Risk/state remain aligned to the prediction time t1 (they are targets).
    risk_aligned = risk[target_indices]
    state_aligned = state[target_indices]

    # Normalize risk to [0,1] for sigmoid head compatibility.
    r_min = float(np.min(risk_aligned))
    r_max = float(np.max(risk_aligned))
    if r_max > r_min:
        risk_norm = (risk_aligned - r_min) / (r_max - r_min)
    else:
        risk_norm = np.zeros_like(risk_aligned, dtype=float)

    return MultitaskTargets(
        trend=trend_aligned.astype(np.float32),
        direction=direction_aligned.astype(np.float32),
        risk=risk_norm.astype(np.float32),
        state=state_aligned.astype(np.int64),
    )


def split_time_series(
    n: int, *, val_fraction: float = 0.2
) -> Tuple[np.ndarray, np.ndarray]:
    if n <= 1:
        raise ValueError("n must be > 1")
    if not (0.0 < val_fraction < 1.0):
        raise ValueError("val_fraction must be in (0,1)")

    split = int(round(n * (1.0 - val_fraction)))
    split = max(1, min(n - 1, split))
    train_idx = np.arange(0, split, dtype=int)
    val_idx = np.arange(split, n, dtype=int)
    return train_idx, val_idx
