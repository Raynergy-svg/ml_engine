"""Realized-outcome confidence labels.

Replaces the closed-form synthetic confidence formula at
`src/core/modular_data_loaders.py:3335-3426` (which leaked into the
feature matrix because the formula's five inputs were also features).

Two label sources, blended row-by-row:

1. **Real labels** — from closed trades in the trade journal. A row at
   timestamp `t` for instrument `I` gets a real label of `1.0` if a
   journal trade entered within +/-1 bar of `t` won (TP hit, positive PnL),
   `0.0` if it lost.

2. **Pseudo labels** — triple-barrier simulation. For every row without a
   real label, simulate a hypothetical long+short trade entered at
   row `t`'s close price with SL/TP set at ±k * ATR (using the same
   multipliers the runtime gates use). Walk forward `lookahead_bars`
   bars. The row's pseudo-label is `1.0` if EITHER hypothetical direction
   would have hit TP first, `0.0` if both hit SL, NaN if the walk ran out
   of bars. (We pick the better of the two directions because the loader
   doesn't know which direction the agent would have signaled at this row.)

Returns:
    labels (np.ndarray of float, shape `(n,)`): 0.0 or 1.0 for labeled
        rows, NaN for rows with no usable label. Caller MUST filter NaN
        before training.
    metadata (dict): `n_real`, `n_pseudo`, `n_unavailable`,
        `class_balance_real`, `class_balance_pseudo`.

Leak-prevention guarantee:
    For real-labeled rows, the label is a function of the journal
    `outcome.trade_won` field — it does NOT read any feature column at
    row `t`. For pseudo-labeled rows, the label is a function of forward
    OHLC bars (rows t+1 .. t+lookahead) plus the row-t ATR for the SL/TP
    distance. Crucially, the label CANNOT be predicted from the row-t
    feature values alone in the way the old closed-form formula could.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Literal, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Tolerance for matching journal trade timestamps to dataframe rows.
# Journal timestamps are signal-time (bar-close); we accept ±1 bar of slack.
_JOURNAL_MATCH_BARS = 1


def _build_journal_index(
    journal: List[Dict[str, Any]],
    instrument: str,
) -> Dict[pd.Timestamp, bool]:
    """Build a {bar_timestamp -> trade_won} dict for the given instrument.

    Timestamps are floored to the bar (we use minute precision; the H1
    granularity is preserved by the journal).
    """
    index: Dict[pd.Timestamp, bool] = {}
    inst_norm = instrument.replace("/", "_").upper()
    for t in journal:
        if t.get("pair") != inst_norm:
            continue
        ts_raw = t.get("timestamp")
        if not ts_raw:
            continue
        try:
            ts = pd.Timestamp(ts_raw)
        except (ValueError, TypeError):
            continue
        # Floor to hour for H1 (most common runtime granularity); journal
        # timestamps may have microsecond precision.
        ts = ts.tz_convert("UTC") if ts.tzinfo is not None else ts.tz_localize("UTC")
        # Use floor("h") to align with H1 bars
        ts_floor = ts.floor("h")
        won = t.get("trade_won")
        if won is None:
            continue
        # First write wins (live journal is loaded before archives in the
        # caller; later duplicates are already filtered out).
        if ts_floor not in index:
            index[ts_floor] = bool(won)
    return index


def _normalize_df_index(df: pd.DataFrame) -> pd.DatetimeIndex:
    """Get a tz-aware UTC DatetimeIndex from df, hour-floored."""
    if isinstance(df.index, pd.DatetimeIndex):
        idx = df.index
    elif "time" in df.columns:
        idx = pd.DatetimeIndex(pd.to_datetime(df["time"], utc=True))
    elif "timestamp" in df.columns:
        idx = pd.DatetimeIndex(pd.to_datetime(df["timestamp"], utc=True))
    else:
        # No timestamps; return synthetic ones (real labels won't match,
        # only pseudo path will be available).
        return pd.DatetimeIndex(pd.date_range("1970-01-01", periods=len(df), freq="h"))
    if idx.tz is None:
        idx = idx.tz_localize("UTC")
    else:
        idx = idx.tz_convert("UTC")
    return idx.floor("h")


def _resolve_atr_column(df: pd.DataFrame) -> Optional[np.ndarray]:
    """Return per-row ATR in price units (absolute, not pct)."""
    if "atr" in df.columns:
        return df["atr"].astype(float).values
    if "atr_14" in df.columns:
        return df["atr_14"].astype(float).values
    if "atr_pct_14" in df.columns and "close" in df.columns:
        # atr_pct_14 is fractional (atr/close)
        return (df["atr_pct_14"].astype(float) * df["close"].astype(float)).values
    return None


def _triple_barrier_pseudo(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    atr: np.ndarray,
    sl_atr_mult: float,
    tp_atr_mult: float,
    lookahead_bars: int,
) -> np.ndarray:
    """Compute pseudo-labels via dual-direction triple-barrier.

    For each row i, simulate:
      - LONG: SL = close[i] - sl_mult * atr[i],  TP = close[i] + tp_mult * atr[i]
      - SHORT: SL = close[i] + sl_mult * atr[i], TP = close[i] - tp_mult * atr[i]

    Walk forward bars i+1 .. i+lookahead, checking each bar's [low, high]
    against barriers. Outcome:
      - 1.0 if EITHER direction's TP hits before its SL within window
      - 0.0 if both directions' SL hits before TP (or both time out at SL-side losers)
      - NaN if neither direction resolved within window AND we ran out of forward bars

    Why dual-direction: the loader doesn't know which direction a future
    agent would have voted at row i. We label the row with "would a
    hypothetically-correct trade at this bar have been profitable?" which
    is captured by `max(long_win, short_win)`. This is a noisy proxy but
    it's *not* a closed-form function of the row-i features (it depends
    on forward bar OHLC).
    """
    n = len(close)
    out = np.full(n, np.nan, dtype=np.float32)

    if n == 0 or atr is None:
        return out

    for i in range(n):
        if not np.isfinite(atr[i]) or atr[i] <= 0:
            continue
        max_j = min(i + lookahead_bars, n - 1)
        if max_j <= i:
            continue
        c0 = close[i]
        a = atr[i]
        long_sl = c0 - sl_atr_mult * a
        long_tp = c0 + tp_atr_mult * a
        short_sl = c0 + sl_atr_mult * a
        short_tp = c0 - tp_atr_mult * a

        long_outcome: Optional[int] = None
        short_outcome: Optional[int] = None

        for j in range(i + 1, max_j + 1):
            hj = high[j]
            lj = low[j]
            if long_outcome is None:
                # Bar can hit both barriers; conservative rule: SL wins
                # when both touched in same bar (mirrors live behavior).
                if lj <= long_sl:
                    long_outcome = 0
                elif hj >= long_tp:
                    long_outcome = 1
            if short_outcome is None:
                if hj >= short_sl:
                    short_outcome = 0
                elif lj <= short_tp:
                    short_outcome = 1
            if long_outcome is not None and short_outcome is not None:
                break

        # Combine: if either direction won → 1; if both lost → 0; otherwise NaN
        if long_outcome == 1 or short_outcome == 1:
            out[i] = 1.0
        elif long_outcome == 0 and short_outcome == 0:
            out[i] = 0.0
        # else: leave as NaN (timed out without resolution)

    return out


def compute_realized_confidence_labels(
    df: pd.DataFrame,
    instrument: str,
    journal: Optional[List[Dict[str, Any]]] = None,
    lookahead_bars: int = 24,
    sl_atr_mult: float = 1.0,
    tp_atr_mult: float = 2.5,
    label_mode: Literal["real", "pseudo", "blend"] = "blend",
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Compute realized-confidence labels in [0, 1] for each row of df.

    Args:
        df: DataFrame with at least open/high/low/close columns and a
            time-like index or 'time'/'timestamp' column. ATR is read from
            'atr', 'atr_14', or computed from 'atr_pct_14' * 'close'.
        instrument: e.g. 'EUR_USD'.
        journal: list of normalized trade dicts (from
            `journal_loader.load_all_journal_trades`). If None, only pseudo
            path is available.
        lookahead_bars: how many forward bars to scan for triple-barrier.
        sl_atr_mult: SL distance in ATR units (matches runtime gate
            `atr_sl_multiplier`).
        tp_atr_mult: TP distance in ATR units (matches runtime gate
            `atr_tp_multiplier`).
        label_mode:
            - 'real': only journal-real labels (NaN elsewhere)
            - 'pseudo': only triple-barrier (NaN at edges)
            - 'blend' (default): real labels override pseudo

    Returns:
        labels: float array of shape (n,) with values in {0.0, 1.0, NaN}.
        metadata: dict with n_real, n_pseudo, n_unavailable,
            class_balance_real, class_balance_pseudo.
    """
    n = len(df)
    out = np.full(n, np.nan, dtype=np.float32)
    is_real = np.zeros(n, dtype=bool)

    metadata: Dict[str, Any] = {
        "n_total": int(n),
        "n_real": 0,
        "n_pseudo": 0,
        "n_unavailable": 0,
        "class_balance_real": None,
        "class_balance_pseudo": None,
        "lookahead_bars": int(lookahead_bars),
        "sl_atr_mult": float(sl_atr_mult),
        "tp_atr_mult": float(tp_atr_mult),
        "label_mode": label_mode,
        "instrument": instrument,
    }

    if n == 0:
        return out, metadata

    # --- Real labels (journal join) ---
    if label_mode in ("real", "blend") and journal:
        idx = _normalize_df_index(df)
        j_index = _build_journal_index(journal, instrument)
        for i, ts in enumerate(idx):
            if ts in j_index:
                out[i] = 1.0 if j_index[ts] else 0.0
                is_real[i] = True
        metadata["n_real"] = int(is_real.sum())
        if metadata["n_real"] > 0:
            real_vals = out[is_real]
            metadata["class_balance_real"] = float(np.mean(real_vals == 1.0))

    # --- Pseudo labels (triple-barrier) ---
    if label_mode in ("pseudo", "blend"):
        for col in ("high", "low", "close"):
            if col not in df.columns:
                logger.warning(
                    "compute_realized_confidence_labels: missing %s; skipping pseudo-label path",
                    col,
                )
                metadata["n_unavailable"] = int(np.isnan(out).sum())
                return out, metadata

        atr = _resolve_atr_column(df)
        if atr is None:
            logger.warning(
                "compute_realized_confidence_labels: no ATR column; skipping pseudo-label path",
            )
            metadata["n_unavailable"] = int(np.isnan(out).sum())
            return out, metadata

        high = df["high"].astype(float).values
        low = df["low"].astype(float).values
        close = df["close"].astype(float).values

        pseudo = _triple_barrier_pseudo(
            high, low, close, atr,
            sl_atr_mult=sl_atr_mult,
            tp_atr_mult=tp_atr_mult,
            lookahead_bars=lookahead_bars,
        )
        # Apply pseudo only where real is missing (real overrides)
        if label_mode == "pseudo":
            out = pseudo
            is_real[:] = False
        else:
            mask = (~is_real) & ~np.isnan(pseudo)
            out[mask] = pseudo[mask]

        pseudo_filled = (~is_real) & ~np.isnan(out)
        metadata["n_pseudo"] = int(pseudo_filled.sum())
        if metadata["n_pseudo"] > 0:
            metadata["class_balance_pseudo"] = float(
                np.mean(out[pseudo_filled] == 1.0)
            )

    metadata["n_unavailable"] = int(np.isnan(out).sum())
    return out, metadata
