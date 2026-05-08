"""Time-alignment of sparse news events to dense price bars (Phase 3 stub).

Algorithm overview (full discussion in design doc §4):

    Inputs:
        events           : List[NewsEvent]            (sparse, asynchronous)
        bar_timestamps   : List[datetime]             (dense, regular M15 bars)
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
"""

from __future__ import annotations

from datetime import datetime
from typing import List

import numpy as np

from src.data.news.source import NewsEvent


def align_news_to_bars(
    events: List[NewsEvent],
    bar_timestamps: List[datetime],
    lookback_window_hours: int = 24,
) -> np.ndarray:
    """Align sparse news events to a regular bar timeline. (Phase 3 stub.)

    Args:
        events: List of NewsEvent. May be empty (function returns all-zero
            features). Must already be embedded — events carry text, but the
            embedding is held externally; in Phase 3 the call site will pass
            a parallel ``embeddings`` array of shape (len(events), D). The
            Phase-1 stub keeps this signature minimal until that detail is
            settled (see design doc §4 / §9).
        bar_timestamps: List of timezone-aware UTC datetimes for the price
            bars to align to. Typically ``df.index.to_pydatetime().tolist()``.
        lookback_window_hours: How many hours of history each bar can "see".
            Default 24h captures multi-session position-building around macro
            events. Phase 3 may add a multi-window variant ([4h, 24h]).

    Returns:
        np.ndarray of shape (len(bar_timestamps), embedding_dim + 8) where
        embedding_dim is the embedder's contracted dimensionality (FinBERT:
        768) and 8 is the event-class one-hot count vector.

    Raises:
        NotImplementedError: until Phase 3.
        ValueError (Phase 3+): if any bar timestamp is timezone-naive.

    Notes:
        - Phase 1 ships the function signature + algorithm doc to lock the
          contract. Phase 3 implementation is < 60 lines (event filtering +
          decay weighting + numpy mean-pool); the doc is the load-bearing
          artifact, not the code.
    """
    raise NotImplementedError(
        "align_news_to_bars is a Phase 3 implementation. "
        "See docs/superpowers/plans/2026-05-08-news-macro-signal-design.md "
        "§4 (Time-alignment problem) for the full algorithm + lookahead-bias "
        "guards. The function signature is locked; Phase 3 only adds the body."
    )
