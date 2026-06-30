"""Variant 1 — QUALITY / profitability tilt MECHANISM (data-injected; running:NO).

Tilts the long equal-weight book toward high-quality names. This module
FABRICATES NO quality data — it REQUIRES a real point-in-time quality-score panel
(date×ticker, known at each date t) supplied by the caller.

DATA-HONESTY NOTE (L-018): clean point-in-time fundamentals (ROE / leverage /
earnings-stability) are NOT freely available for this 38-member universe — the
financial-datasets API returns only the latest fiscal-year snapshot, and yfinance
is current-snapshot only. Applying a single current ROE across 2006-2026 would be
lookahead + survivorship bias — a fabricated "quality" result. So the bake-off
reports the quality arm as NOT_EVALUATED rather than producing a fake number; this
mechanism exists (and is tested with a synthetic score panel) so it runs the moment
a real PIT quality panel is sourced. The caller MUST ensure ``quality_panel`` is
point-in-time (no future info); this module adds no lookahead of its own.

SEPARATE variant module; baseline NOT modified. NON-hot-path.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.equity.variant_eval import aligned_inputs

DEFAULT_TILT = 1.0  # cross-sectional z-score tilt strength — canonical, untuned


def quality_tilt_weights(snapshot, prices, quality_panel: pd.DataFrame, *, tilt: float = DEFAULT_TILT):
    """Tilt EW toward high quality scores (causal in the panel), long-only, rows sum to 1.

    weight ∝ ew × exp(tilt · z(quality)) within the active set, renormalized. A row
    with no usable scores falls back to equal weight.
    """
    ew, active, _ = aligned_inputs(snapshot, prices)
    q = quality_panel.reindex(index=ew.index, columns=ew.columns)
    qa = q.where(active)
    mu = qa.mean(axis=1)
    sd = qa.std(axis=1)
    z = qa.sub(mu, axis=0).div(sd.replace(0.0, np.nan), axis=0).fillna(0.0)
    raw = (ew * np.exp(float(tilt) * z)).where(active, 0.0)
    row_sum = raw.sum(axis=1)
    w = raw.div(row_sum.replace(0.0, np.nan), axis=0)
    fallback = row_sum <= 0
    w = w.fillna(0.0)
    if fallback.any():
        w.loc[fallback] = ew.loc[fallback].values
    return w
