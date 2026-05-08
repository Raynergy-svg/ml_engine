"""Time-alignment of sparse news events to dense price bars.

Algorithm overview (full discussion in design doc §4):

    Inputs:
        events           : List[NewsEvent]            (sparse, asynchronous)
        bar_timestamps   : List[datetime]             (dense, regular M15 bars)
        embeddings       : np.ndarray (n_events, D)   (parallel to events)
        lookback_window_hours : int (default 24)

    For each bar at t_b:
        1. Collect events with t_e in [t_b - lookback_window_hours, t_b)
           (open upper bound — strict < t_b prevents lookahead leak).
        2. Time-decay weight each event: w_e = exp(-(t_b - t_e) / tau)
           where tau = lookback_window_hours / 3.
        3. Multiply by relevance_score: w_e *= event.relevance_score
        4. Mean-pool event embeddings weighted by w_e:
              bar_embedding[t_b] = sum_e(w_e * embed_e) / max(sum_e(w_e), 1e-6)
        5. Emit a side ``event_class_count`` vector R^8 over
              {NFP, CPI, GDP, FOMC, ECB, BoE, BoJ, OTHER}
           with weighted counts in the same window.
        6. No-event bar: bar_embedding = zeros(D), counts = zeros(8).

    Output:
        np.ndarray of shape (len(bar_timestamps), embedding_dim + 8).

    Lookahead-bias guard:
        - Events MUST have timezone-aware timestamps (NewsEvent.__post_init__
          already enforces this).
        - Bar timestamps must also be tz-aware UTC (caller's responsibility;
          this function validates).
        - The window is closed-open: t_b - lookback <= t_e < t_b. The strict
          upper bound is non-negotiable; relaxing it leaks the current bar's
          news into the model's input for that bar.

    Walk-forward validation guard:
        - PCA compression (768 -> 32) lives downstream in Phase 3 alongside the
          existing price-feature scaler. PCA fits on TRAIN fold ONLY, then
          applies frozen to val/test. align_news_to_bars itself is fold-
          agnostic; the caller (compute_normalized_features modification) is
          responsible for fold isolation.

    Insertion point in Phase 3:
        src/core/modular_data_loaders.py:compute_normalized_features
        — add optional ``news_features_df`` arg; if present, join to df by
        timestamp index after the existing 186-feature compute. The dynamic
        feature-selection block at line ~1772 will correlation-prune
        redundant news columns automatically.

Contract change (Stage 4-A, 2026-05-08):
    The Phase-1 stub signature was (events, bar_timestamps, lookback_window_hours).
    Phase-3 implementation adds ``embeddings: np.ndarray`` as the third
    positional arg between bar_timestamps and lookback_window_hours. Rationale:
    embeddings are produced externally (FinBERTEmbedder.embed) so the alignment
    function stays embedder-agnostic and unit-testable without loading a 440MB
    transformer. The Phase-1 docstring already flagged this as the planned
    resolution (see "embedding is held externally"); Stage 4-A locks it.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import List

import numpy as np

from src.data.news.source import NewsEvent

# Locked category set per design doc §4 step 5.
# Order is load-bearing — column index in the side count vector maps 1:1.
# Categories not in this set (e.g. "BoC", "RBA", "RBNZ", "SNB" from source.py
# docstring) collapse into OTHER. If we ever expand the set, the trainer's
# news-feature column count changes and old artifacts become incompatible.
_EVENT_CATEGORIES: tuple[str, ...] = (
    "NFP",
    "CPI",
    "GDP",
    "FOMC",
    "ECB",
    "BoE",
    "BoJ",
    "OTHER",
)
_CATEGORY_INDEX = {name: i for i, name in enumerate(_EVENT_CATEGORIES)}
_OTHER_INDEX = _CATEGORY_INDEX["OTHER"]
_N_CATEGORIES = len(_EVENT_CATEGORIES)  # 8


def align_news_to_bars(
    events: List[NewsEvent],
    bar_timestamps: List[datetime],
    embeddings: np.ndarray,
    lookback_window_hours: int = 24,
) -> np.ndarray:
    """Align sparse news events to a regular bar timeline.

    Args:
        events: List of NewsEvent. May be empty (function returns all-zero
            features). Each event carries a tz-aware UTC timestamp, category,
            and relevance_score; the embedder produces the parallel
            ``embeddings`` array.
        bar_timestamps: List of timezone-aware UTC datetimes for the price
            bars to align to. Typically ``df.index.to_pydatetime().tolist()``.
            Empty list returns shape (0, D + 8).
        embeddings: np.ndarray of shape (len(events), embedding_dim). Parallel
            to ``events`` (events[i] <-> embeddings[i]). Must be finite (no
            NaN/Inf). For empty events, pass any 2D array with shape (0, D)
            where D is the contracted embedding dim (FinBERT: 768).
        lookback_window_hours: How many hours of history each bar can "see".
            Default 24h captures multi-session position-building around macro
            events. tau = lookback_window_hours / 3, so events 8h+ stale
            (at default) carry <5% weight.

    Returns:
        np.ndarray of shape (len(bar_timestamps), embedding_dim + 8) with
        dtype float32. Columns 0..D-1 are the weighted-mean embedding;
        columns D..D+7 are the per-category weighted counts in the locked
        order [NFP, CPI, GDP, FOMC, ECB, BoE, BoJ, OTHER].

    Raises:
        ValueError: if any bar timestamp is timezone-naive, if events length
            mismatches embeddings rows, if embeddings contains NaN/Inf, if
            lookback_window_hours <= 0, or if embeddings is not 2D.
    """
    # ---------- input validation (lookahead-bias + shape guards) ----------
    if lookback_window_hours <= 0:
        raise ValueError(
            f"lookback_window_hours must be positive, got {lookback_window_hours}"
        )

    if not isinstance(embeddings, np.ndarray):
        raise ValueError(
            f"embeddings must be np.ndarray, got {type(embeddings).__name__}"
        )
    if embeddings.ndim != 2:
        raise ValueError(
            f"embeddings must be 2D (n_events, embedding_dim), got shape {embeddings.shape}"
        )
    if embeddings.shape[0] != len(events):
        raise ValueError(
            f"embeddings rows ({embeddings.shape[0]}) must match len(events) "
            f"({len(events)})"
        )
    if embeddings.size > 0 and not np.all(np.isfinite(embeddings)):
        raise ValueError(
            "embeddings contains NaN or Inf — embedder produced invalid output. "
            "Surface the bad event upstream rather than silently zero-filling."
        )

    embedding_dim = embeddings.shape[1]
    n_bars = len(bar_timestamps)

    # Validate bar timestamps are tz-aware UTC. Naive timestamps make event-
    # bar comparisons ambiguous (Python raises TypeError mid-loop, but the
    # bias risk is the real reason — naive comparison with a UTC event would
    # silently treat naive-as-UTC and could miss DST shifts in caller code).
    for i, ts in enumerate(bar_timestamps):
        if not isinstance(ts, datetime):
            raise ValueError(
                f"bar_timestamps[{i}] must be datetime, got {type(ts).__name__}"
            )
        if ts.tzinfo is None or ts.tzinfo.utcoffset(ts) is None:
            raise ValueError(
                f"bar_timestamps[{i}] is timezone-naive ({ts!r}). All bar "
                "timestamps must be tz-aware UTC to avoid lookahead-bias from "
                "naive-vs-aware comparisons against tz-aware NewsEvent.timestamp."
            )

    # ---------- empty-events fast path ----------
    out = np.zeros((n_bars, embedding_dim + _N_CATEGORIES), dtype=np.float32)
    if not events or n_bars == 0:
        return out

    # ---------- pre-compute event arrays for vectorized filtering ----------
    # Sort events ascending by timestamp once; then per-bar binary-search the
    # window. For the typical case (sparse events, dense bars) this is O((n_e
    # + n_b) log n_e) rather than O(n_e * n_b).
    event_ts = np.array(
        [e.timestamp.astimezone(timezone.utc) for e in events],
        dtype=object,
    )
    sort_idx = np.argsort([e.timestamp for e in events])
    event_ts_sorted = event_ts[sort_idx]
    embeddings_sorted = embeddings[sort_idx].astype(np.float32, copy=False)
    relevance_sorted = np.array(
        [float(events[i].relevance_score) for i in sort_idx],
        dtype=np.float32,
    )
    # Map each event's category to a column in the side-count vector;
    # unknown categories collapse to OTHER.
    cat_idx_sorted = np.array(
        [_CATEGORY_INDEX.get(events[i].category, _OTHER_INDEX) for i in sort_idx],
        dtype=np.int64,
    )

    tau_seconds = (lookback_window_hours / 3.0) * 3600.0
    lookback_delta = timedelta(hours=lookback_window_hours)

    # ---------- per-bar weighted aggregate ----------
    for b_idx, t_b in enumerate(bar_timestamps):
        t_b_utc = t_b.astimezone(timezone.utc)
        window_start = t_b_utc - lookback_delta

        # Filter: window_start <= t_e < t_b (strict upper). Manual loop over
        # event_ts_sorted using np.searchsorted on the *Python* datetime array
        # is awkward because numpy compares object arrays element-wise; cheaper
        # to do the linear filter, since events are sparse (~1-10 in any 24h
        # window).
        mask = np.array(
            [(window_start <= t_e) and (t_e < t_b_utc) for t_e in event_ts_sorted],
            dtype=bool,
        )
        if not mask.any():
            continue

        # Compute decay weights for the in-window events.
        # delta_seconds = (t_b - t_e).total_seconds() — always positive given strict <
        deltas = np.array(
            [(t_b_utc - event_ts_sorted[i]).total_seconds() for i in np.where(mask)[0]],
            dtype=np.float64,
        )
        decay = np.exp(-deltas / tau_seconds).astype(np.float32)
        weights = decay * relevance_sorted[mask]

        weight_sum = float(weights.sum())
        denom = max(weight_sum, 1e-6)

        # Weighted mean of embeddings.
        emb_block = embeddings_sorted[mask]
        weighted_emb = (weights[:, None] * emb_block).sum(axis=0) / denom
        out[b_idx, :embedding_dim] = weighted_emb

        # Weighted per-category counts (NOT divided by denom — these are
        # "weighted event presence", not a mean).
        cat_block = cat_idx_sorted[mask]
        for col, w in zip(cat_block, weights):
            out[b_idx, embedding_dim + int(col)] += float(w)

    return out
