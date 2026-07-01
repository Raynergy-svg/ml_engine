"""Variant 3 — Hierarchical Risk Parity construction (price-only; offline; running:NO).

Replaces equal-weight construction with HRP (López de Prado 2016): cluster names
by return-correlation distance, quasi-diagonalize, then recursive inverse-variance
bisection. Long-only (HRP weights are non-negative), no leverage. Strictly causal:
at each rebalance date the covariance uses a trailing window through ``t-1``; the
target weights are held until the next rebalance.

Implemented DIRECTLY with scipy (already a repo dependency) rather than adding
``riskfolio-lib`` — avoids a new third-party dependency and its supply-chain /
license surface, and keeps the algorithm auditable in-repo. SEPARATE variant
module; baseline NOT modified. NON-hot-path.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import squareform

from src.equity.variant_eval import aligned_inputs

DEFAULT_LOOKBACK = 252  # ~12 months covariance window — canonical, untuned
DEFAULT_STEP = 21       # monthly rebalance


def _quasi_diag(link: np.ndarray) -> list:
    """Standard HRP quasi-diagonalization: leaf order that puts similar names
    adjacent (López de Prado)."""
    link = link.astype(int)
    sort_ix = pd.Series([link[-1, 0], link[-1, 1]])
    num_items = link[-1, 3]
    while sort_ix.max() >= num_items:
        sort_ix.index = range(0, sort_ix.shape[0] * 2, 2)
        df0 = sort_ix[sort_ix >= num_items]
        i = df0.index
        j = df0.values - num_items
        sort_ix[i] = link[j, 0]
        df0 = pd.Series(link[j, 1], index=i + 1)
        sort_ix = pd.concat([sort_ix, df0]).sort_index()
        sort_ix.index = range(sort_ix.shape[0])
    return sort_ix.tolist()


def _cluster_var(cov: pd.DataFrame, items: list) -> float:
    sub = cov.loc[items, items].values
    ivp = 1.0 / np.diag(sub)
    ivp /= ivp.sum()
    return float(ivp @ sub @ ivp)


def _recursive_bisection(cov: pd.DataFrame, order: list) -> pd.Series:
    w = pd.Series(1.0, index=order)
    clusters = [order]
    while clusters:
        clusters = [
            c[j:k]
            for c in clusters
            for j, k in ((0, len(c) // 2), (len(c) // 2, len(c)))
            if len(c) > 1
        ]
        for i in range(0, len(clusters), 2):
            c0, c1 = clusters[i], clusters[i + 1]
            v0, v1 = _cluster_var(cov, c0), _cluster_var(cov, c1)
            alpha = 1.0 - v0 / (v0 + v1)
            w[c0] *= alpha
            w[c1] *= 1.0 - alpha
    return w


def _hrp_from_returns(window: pd.DataFrame) -> pd.Series:
    """HRP weights over the tickers in ``window`` (a clean returns DataFrame)."""
    cov = window.cov()
    corr = window.corr()
    dist = np.sqrt(np.clip((1.0 - corr) / 2.0, 0.0, None))
    condensed = squareform(dist.values, checks=False)
    link = linkage(condensed, method="single")
    order_idx = _quasi_diag(link)
    order = [corr.index[i] for i in order_idx]
    w = _recursive_bisection(cov, order)
    total = float(w.sum())
    if total <= 0:
        return pd.Series(1.0 / len(window.columns), index=window.columns)
    return (w / total).reindex(window.columns).fillna(0.0)


def hrp_weights(
    snapshot,
    prices,
    *,
    lookback: int = DEFAULT_LOOKBACK,
    step: int = DEFAULT_STEP,
    min_names: int = 2,
):
    """Causal HRP target-weight panel, long-only, rows sum to 1 over active."""
    ew, active, rets = aligned_inputs(snapshot, prices)
    cols = list(ew.columns)
    idx = ew.index
    targets = pd.DataFrame(0.0, index=idx, columns=cols)
    last: pd.Series | None = None
    for i, t in enumerate(idx):
        if i > 0 and (last is None or i % int(step) == 0):
            window = rets.iloc[max(0, i - int(lookback)): i]  # rows < i -> through t-1
            act = [c for c in cols if bool(active.iloc[i][c])]
            clean = [c for c in act if window[c].notna().sum() >= max(min_names, lookback // 2)]
            if len(clean) >= int(min_names):
                w = _hrp_from_returns(window[clean].dropna(axis=0, how="any"))
                full = pd.Series(0.0, index=cols)
                full.loc[w.index] = w.values
                last = full
        if last is not None:
            targets.iloc[i] = last.values
    # Per-date: mask to currently-active + renormalize; warmup rows -> equal weight.
    masked = targets.where(active, 0.0)
    row_sum = masked.sum(axis=1)
    w = masked.div(row_sum.replace(0.0, np.nan), axis=0)
    fallback = row_sum <= 0
    w = w.fillna(0.0)
    if fallback.any():
        w.loc[fallback] = ew.loc[fallback].values
    return w
