"""OANDA position-book CONTRARIAN sentiment signal — pre-registered test (offline; running:NO).

Implements exactly the signal pre-registered in
``docs/experiment-oanda-sentiment-2026-06-29.md``: fade retail crowding. NL = fraction
of open positions LONG (from the position book); contrarian signal S = -(NL - 0.5).
Strictly causal: S_i,t uses only the book at t; the realized return is from book-time t
to t+1 (the same 20-min snapshot grid, each book carries its own ``price``).

This is a SEPARATE strategy/overlay — it does NOT modify the live trend lane and is
NOT wired into execution. OFFLINE evaluation only. The order/position book was always
flagged DATA-ONLY; this is the deferred pre-registered test of whether it adds edge.
"""
from __future__ import annotations

import math
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

COST_BPS_PER_SIDE = 1.5   # pre-registered realistic FX-major spread cost (per side)


def book_long_fraction(book_resp: dict) -> Optional[float]:
    """Fraction of positions that are LONG from a position-book response (None if empty).

    NL = Σ longCountPercent / (Σ longCountPercent + Σ shortCountPercent). The per-bucket
    percentages sum to ~100 across both sides, so this is the crowd's net long share."""
    pb = (book_resp or {}).get("positionBook") or {}
    buckets = pb.get("buckets") or []
    long_sum = short_sum = 0.0
    for b in buckets:
        try:
            long_sum += float(b.get("longCountPercent", 0) or 0)
            short_sum += float(b.get("shortCountPercent", 0) or 0)
        except (TypeError, ValueError):
            continue
    total = long_sum + short_sum
    if total <= 0:
        return None
    return long_sum / total


def contrarian_signal(nl: float) -> float:
    """Pre-registered contrarian signal: S = -(NL - 0.5). Crowded-long -> negative (fade)."""
    return -(float(nl) - 0.5)


def book_price(book_resp: dict) -> Optional[float]:
    pb = (book_resp or {}).get("positionBook") or {}
    try:
        return float(pb.get("price"))
    except (TypeError, ValueError):
        return None


def build_panels(books_by_inst: Dict[str, list]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """{inst: [book_resp,...]} -> (signal_df, fwd_return_df), date×inst, strictly causal.

    signal[t] = contrarian_signal(NL at book t); fwd_return[t] = price[t+1]/price[t]-1
    (last row NaN). Both indexed by the book snapshot time."""
    sig_cols: Dict[str, pd.Series] = {}
    ret_cols: Dict[str, pd.Series] = {}
    for inst, books in (books_by_inst or {}).items():
        rows = []
        for bk in books or []:
            nl = book_long_fraction(bk)
            px = book_price(bk)
            t = ((bk or {}).get("positionBook") or {}).get("time")
            if nl is None or px is None or not t:
                continue
            rows.append((pd.Timestamp(t), contrarian_signal(nl), px))
        if not rows:
            continue
        df = pd.DataFrame(rows, columns=["t", "s", "px"]).drop_duplicates("t").sort_values("t").set_index("t")
        sig_cols[inst] = df["s"]
        ret_cols[inst] = df["px"].shift(-1) / df["px"] - 1.0   # causal: t -> t+1
    if not sig_cols:
        return pd.DataFrame(), pd.DataFrame()
    sig = pd.DataFrame(sig_cols).sort_index()
    ret = pd.DataFrame(ret_cols).reindex_like(sig)
    return sig, ret


def pooled_ic(signal_df: pd.DataFrame, return_df: pd.DataFrame) -> Dict[str, float]:
    """Pooled Pearson IC = corr(S, r) over all (inst, t) + naive t-stat + n.

    NOTE (verifier caveat): pooled N overstates independence — FX majors are cross-
    correlated (esp. USD pairs), so the naive t is optimistic. The portfolio Sharpe and
    the per-instrument breakdown are the economically honest reads."""
    s = signal_df.values.flatten()
    r = return_df.values.flatten()
    m = np.isfinite(s) & np.isfinite(r)
    s, r = s[m], r[m]
    n = int(s.size)
    if n < 10 or s.std() == 0 or r.std() == 0:
        return {"ic": 0.0, "t_stat": 0.0, "n": n}
    ic = float(np.corrcoef(s, r)[0, 1])
    t = ic * math.sqrt(max(n - 2, 1)) / math.sqrt(max(1 - ic * ic, 1e-12))
    return {"ic": round(ic, 5), "t_stat": round(t, 3), "n": n}


def per_instrument_ic(signal_df: pd.DataFrame, return_df: pd.DataFrame) -> Dict[str, float]:
    """Time-series IC per instrument (corr over time) — robust to cross-sectional corr."""
    out: Dict[str, float] = {}
    for inst in signal_df.columns:
        s = signal_df[inst]
        r = return_df[inst]
        m = s.notna() & r.notna()
        if m.sum() >= 10 and s[m].std() > 0 and r[m].std() > 0:
            out[inst] = round(float(np.corrcoef(s[m], r[m])[0, 1]), 4)
    return out


def contrarian_portfolio(signal_df: pd.DataFrame, return_df: pd.DataFrame, *,
                         cost_bps_per_side: float = COST_BPS_PER_SIDE) -> pd.Series:
    """Daily-rebalanced EW contrarian portfolio net return series (gross leverage 1.0).

    Weights w_i,t = S_i,t / Σ|S_j,t| (gross exposure 1). Net return =
    Σ w_i,t·r_i,t − turnover·cost. Strictly uses S at t against r (t→t+1); no lookahead."""
    sig = signal_df.copy()
    gross = sig.abs().sum(axis=1).replace(0.0, np.nan)
    w = sig.div(gross, axis=0).fillna(0.0)
    gross_ret = (w * return_df.fillna(0.0)).sum(axis=1)
    turnover = w.diff().abs().sum(axis=1).fillna(w.abs().sum(axis=1))
    cost = turnover * (2.0 * float(cost_bps_per_side) * 1e-4)   # per-side bps -> fraction
    net = (gross_ret - cost).dropna()
    return net
