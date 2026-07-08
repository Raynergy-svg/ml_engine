"""Feature pipeline for the risk-target model family (forward volatility +
forward drawdown-state).

Deliberately DECOUPLED from `src/core/modular_data_loaders.py`'s
`compute_normalized_features()`/`FEATURE_PIPELINE_VERSION` (the FX
direction/TCN pipeline) — this is a new model family with its own targets,
own universe (daily bars, not intraday), and its own version contract
(`RISK_TARGET_FEATURE_PIPELINE_VERSION`). Coupling to the direction
pipeline would mean an unrelated direction-pipeline change (news fusion,
OBV fix, etc.) could force a spurious version bump / quarantine on this
model, or vice versa — see the risk-target infra survey (2026-07-08) for
the reasoning.

Every feature is a TRAILING (backward-looking) rolling statistic computed
at row `i` from rows `<= i` only. None are cumulative-from-dataset-start —
that anchored-window failure mode (L-001: `obv`/`cum_returns`/`atr_log`/
`vol_log` leaking the row's position within the analysis window into the
feature value) is exactly what produced the false 56-70% direction
accuracy this repo was burned by. `tests/test_risk_target_features.py`
carries a window-invariance canary mirroring
`tests/test_feature_window_invariance.py`.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

RISK_TARGET_FEATURE_PIPELINE_VERSION = "2026-07-08-v1"

# Trading-day annualization factor for daily-bar realized vol.
_ANNUALIZATION_FACTOR = float(np.sqrt(252.0))

FEATURE_COLUMNS: List[str] = [
    "realized_vol_5",
    "realized_vol_10",
    "realized_vol_20",
    "realized_vol_60",
    "atr_14",
    "hl_range_pct_14",
    "return_mean_5",
    "return_mean_20",
    "vol_of_vol_20",
    "volume_zscore_20",
    "day_of_week",
]
# `pair` is added by the caller (pooling step) as a separate categorical
# column — not computed here since a single-pair frame has no pair axis.
CATEGORICAL_FEATURE_COLUMNS: List[str] = ["day_of_week", "pair"]


def compute_risk_target_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute the trailing risk-target feature set for one instrument.

    Args:
        df: DataFrame with columns `date, open, high, low, close, volume`,
            sorted ascending by date, single instrument (no `pair` mixing —
            rolling windows must not blend two instruments' histories).

    Returns:
        A copy of `df` with the `FEATURE_COLUMNS` appended. Leading rows
        within the longest warm-up window (60 bars) have NaN feature
        values (insufficient trailing history) — caller drops these before
        training, same convention as the label warm-up trim in
        `modular_data_loaders.py` (`valid_start = lookback_bars`).
    """
    required = {"date", "open", "high", "low", "close", "volume"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"compute_risk_target_features: missing required columns {missing}")

    out = df.copy().reset_index(drop=True)
    close = out["close"].astype(np.float64)
    high = out["high"].astype(np.float64)
    low = out["low"].astype(np.float64)
    volume = out["volume"].astype(np.float64)

    log_ret = np.log(close / close.shift(1))

    # Rolling realized vol at multiple trailing windows (annualized).
    for window in (5, 10, 20, 60):
        out[f"realized_vol_{window}"] = (
            log_ret.rolling(window=window, min_periods=window).std(ddof=0) * _ANNUALIZATION_FACTOR
        )

    # True range (Wilder), rolling mean, normalized by close — a trailing,
    # non-cumulative ATR proxy.
    prev_close = close.shift(1)
    true_range = pd.concat(
        [
            high - low,
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    out["atr_14"] = (true_range.rolling(window=14, min_periods=14).mean()) / close

    out["hl_range_pct_14"] = ((high - low) / close).rolling(window=14, min_periods=14).mean()

    out["return_mean_5"] = log_ret.rolling(window=5, min_periods=5).mean()
    out["return_mean_20"] = log_ret.rolling(window=20, min_periods=20).mean()

    # Vol-of-vol: dispersion of the short-window realized vol itself.
    out["vol_of_vol_20"] = out["realized_vol_5"].rolling(window=20, min_periods=20).std(ddof=0)

    # Rolling z-score of volume vs its own trailing 60-day mean/std.
    vol_roll_mean = volume.rolling(window=60, min_periods=60).mean()
    vol_roll_std = volume.rolling(window=60, min_periods=60).std(ddof=0)
    out["volume_zscore_20"] = (volume - vol_roll_mean) / vol_roll_std.replace(0.0, np.nan)

    out["day_of_week"] = pd.to_datetime(out["date"], utc=True, errors="coerce").dt.dayofweek.fillna(0).astype(np.int32)

    return out


def build_pooled_feature_frame(
    per_pair_frames: Dict[str, pd.DataFrame],
) -> Tuple[pd.DataFrame, List[str]]:
    """Compute features per-pair (rolling windows never cross a pair
    boundary), then concatenate into one pooled frame with a `pair`
    categorical column.

    Args:
        per_pair_frames: mapping instrument -> raw OHLCV DataFrame (each
            already sorted ascending by date).

    Returns:
        (pooled_df, feature_columns) where pooled_df has one row per
        (pair, date) with all `FEATURE_COLUMNS` + `pair` + `close` (needed
        downstream by the label generators) + `date`, sorted by
        `(date, pair)` so a calendar-date-based walk-forward split can be
        applied uniformly across the pooled universe. NaN warm-up rows
        (per pair, first 60 trailing-window bars) are NOT dropped here —
        the caller drops them per-pair before pooling the label indices,
        since each pair's warm-up region differs in absolute row count
        but not in calendar meaning.
    """
    computed = []
    for pair, raw in per_pair_frames.items():
        feat = compute_risk_target_features(raw)
        feat["pair"] = pair
        computed.append(feat)

    pooled = pd.concat(computed, ignore_index=True)
    pooled["date"] = pd.to_datetime(pooled["date"], utc=True, errors="coerce")
    pooled = pooled.sort_values(["date", "pair"]).reset_index(drop=True)
    return pooled, list(FEATURE_COLUMNS)


__all__ = [
    "RISK_TARGET_FEATURE_PIPELINE_VERSION",
    "FEATURE_COLUMNS",
    "CATEGORICAL_FEATURE_COLUMNS",
    "compute_risk_target_features",
    "build_pooled_feature_frame",
]
