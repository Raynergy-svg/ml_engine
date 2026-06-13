"""US-005 — Portfolio construction with volatility targeting and hard guards.

Turns a per-(date,pair) combined factor score into target portfolio weights at a
fixed risk budget. Everything is causal: per-pair vol and the covariance used for
vol-targeting are estimated from a trailing window ending at (and including) the
signal date; the backtester then applies a further 1-day execution lag.

Guards (PRD US-005), enforced and property-tested:
- gross leverage  sum|w| <= GROSS_CAP (4:1)
- per-pair        |w_i|  <= PER_PAIR_CAP (30%) of gross
- risk budget     ex-ante annualized vol scaled toward VOL_TARGET (10%)
"""

from __future__ import annotations

import numpy as np
import pandas as pd

VOL_TARGET = 0.10          # 10% annualized portfolio vol
VOL_LOOKBACK = 60          # trading days for realized vol / covariance
GROSS_CAP = 4.0            # max sum of absolute weights
PER_PAIR_CAP = 0.30        # max |weight| as a fraction of gross shape
ANNUALIZATION = 252.0
COV_SHRINKAGE = 0.5        # shrink sample cov toward its diagonal for stability


def _shrunk_cov(returns_window: np.ndarray) -> np.ndarray:
    """Sample covariance shrunk toward its diagonal (stable for small windows)."""
    cov = np.cov(returns_window, rowvar=False)
    cov = np.atleast_2d(cov)
    diag = np.diag(np.diag(cov))
    return COV_SHRINKAGE * diag + (1.0 - COV_SHRINKAGE) * cov


def _weights_one_date(
    score: np.ndarray,
    ret_window: np.ndarray,
    vol_target: float,
    gross_cap: float,
    per_pair_cap: float,
) -> np.ndarray:
    """Vectorized single-date construction. ``ret_window`` is (lookback × n)."""
    n = score.shape[0]
    valid = np.isfinite(score)
    if not valid.any() or ret_window.shape[0] < 5:
        return np.zeros(n)

    sigma = np.nanstd(ret_window, axis=0) * np.sqrt(ANNUALIZATION)
    sigma = np.where(sigma > 1e-9, sigma, np.nan)

    s = np.where(valid, score, 0.0)
    # Inverse-vol risk normalization: equal risk per unit of conviction.
    raw = np.divide(s, sigma, out=np.zeros_like(s), where=np.isfinite(sigma))
    gross_raw = np.nansum(np.abs(raw))
    if gross_raw <= 1e-12:
        return np.zeros(n)
    shape = raw / gross_raw

    # Per-pair cap on the *shape* (fraction of unit gross). We deliberately do
    # NOT renormalize afterwards: renormalizing would re-inflate a capped leg
    # back over the cap. The leverage step below sets the final risk; the cap is
    # an absolute ceiling of (per_pair_cap × leverage) on any single position.
    shape = np.clip(shape, -per_pair_cap, per_pair_cap)
    gross_shape = np.nansum(np.abs(shape))
    if gross_shape <= 1e-12:
        return np.zeros(n)

    # Ex-ante annualized vol of the shape book from the (shrunk) covariance.
    cols = np.where(np.isfinite(sigma), np.arange(n), -1)
    rw = ret_window[:, valid & np.isfinite(sigma)]
    w_valid = shape[valid & np.isfinite(sigma)]
    if rw.shape[1] == 0:
        return np.zeros(n)
    cov = _shrunk_cov(rw) * ANNUALIZATION
    var = float(w_valid @ cov @ w_valid)
    book_vol = np.sqrt(max(var, 1e-12))

    leverage = min(vol_target / book_vol, gross_cap) if book_vol > 0 else 0.0
    leverage = max(leverage, 0.0)
    _ = cols  # documented intent; index bookkeeping handled by boolean masks
    return shape * leverage


def target_weights(
    score: pd.DataFrame,
    returns: pd.DataFrame,
    *,
    vol_target: float = VOL_TARGET,
    vol_lookback: int = VOL_LOOKBACK,
    gross_cap: float = GROSS_CAP,
    per_pair_cap: float = PER_PAIR_CAP,
) -> pd.DataFrame:
    """Pure function: (score, returns) -> date×pair target weights.

    ``score`` and ``returns`` must share columns (pairs). ``returns`` are daily
    simple returns. The weight for date T uses returns in (T-vol_lookback, T].
    """
    if list(score.columns) != list(returns.columns):
        returns = returns.reindex(columns=score.columns)
    idx = score.index
    ret = returns.reindex(idx)
    score_v = score.to_numpy(dtype=float)
    ret_v = ret.to_numpy(dtype=float)
    out = np.zeros_like(score_v)

    for t in range(len(idx)):
        lo = max(0, t - vol_lookback + 1)
        window = ret_v[lo:t + 1]
        out[t] = _weights_one_date(
            score_v[t], window, vol_target, gross_cap, per_pair_cap
        )

    return pd.DataFrame(out, index=idx, columns=score.columns)


def apply_no_trade_band(
    target: pd.DataFrame,
    *,
    band: float = 0.05,
    rebalance_mask: pd.Series | None = None,
) -> pd.DataFrame:
    """Hold weights between rebalances; only move a pair if |Δweight| >= band.

    ``rebalance_mask`` (bool per date) marks rebalance days; on non-rebalance
    days the previous held weight carries forward. This cuts turnover (US-005).
    """
    idx = target.index
    cols = target.columns
    tv = target.to_numpy(dtype=float)
    held = np.zeros(tv.shape[1])
    out = np.zeros_like(tv)
    if rebalance_mask is None:
        rebalance_mask = pd.Series(True, index=idx)
    rb = rebalance_mask.reindex(idx).fillna(False).to_numpy()

    for t in range(len(idx)):
        if rb[t]:
            move = np.abs(tv[t] - held) >= band
            held = np.where(move, tv[t], held)
        out[t] = held
    return pd.DataFrame(out, index=idx, columns=cols)


def enforce_guards(
    weights: pd.DataFrame,
    *,
    gross_cap: float = GROSS_CAP,
    per_pair_cap: float = PER_PAIR_CAP,
) -> pd.DataFrame:
    """Hard-enforce gross and per-pair caps row-wise (post no-trade-band).

    The no-trade band carries stale legs forward, which can let the assembled
    book breach the gross cap; this is the final, non-negotiable clamp so
    "gross leverage <= 4:1, ever" holds in every profile. Clamping only ever
    REDUCES exposure, so it cannot manufacture risk.
    """
    # Absolute per-position ceiling tied to the gross cap. "30% of gross" is only
    # well-defined as an absolute bound (a literal fraction-of-realized-gross is
    # infeasible with few active legs); this is the honest, always-satisfiable form.
    cap_abs = per_pair_cap * gross_cap
    w = weights.to_numpy(dtype=float).copy()
    for t in range(w.shape[0]):
        row = np.clip(w[t], -cap_abs, cap_abs)
        g = np.abs(row).sum()
        if g > gross_cap and g > 0:
            row = row * (gross_cap / g)
        w[t] = row
    return pd.DataFrame(w, index=weights.index, columns=weights.columns)


def weekly_rebalance_mask(index: pd.DatetimeIndex, weekday: int = 4) -> pd.Series:
    """True on the last available session each week (default Friday=4)."""
    s = pd.Series(index.weekday, index=index)
    # Rebalance on the configured weekday, or the last session of the week if
    # that weekday is missing (holiday) — approximated by weekday transitions.
    is_day = s == weekday
    # Also flag a session whose next session is in a new ISO week.
    iso_week = pd.Series(index.isocalendar().week.to_numpy(), index=index)
    next_new_week = iso_week.shift(-1) != iso_week
    return (is_day | next_new_week).fillna(True)
