"""Track B research-alpha backtest harness (prereg §3–§4).

Turns a list of per-``(ticker, as_of)`` :class:`~src.equity.research.contracts.ResearchScore`
artifacts (produced OFFLINE — by a deterministic baseline scorer or by the
offline LLM research pass, read back from a versioned artifact) into a
cross-sectional long-only quintile book, and evaluates that book across the four
pre-registered lookahead-control arms (``full`` / ``pre_cutoff`` / ``post_cutoff``
/ ``placebo``), each against the mechanical ship gate
(:func:`src.factor.ship_gate.evaluate_gate`).

This module is LLM-FREE and NETWORK-FREE by design. The scores it consumes
already exist as data; the harness never calls a model. Everything here is pure
pandas/numpy and is deterministic — in particular the ``placebo`` permutation is
a fixed roll keyed by the sorted-ticker index (NO RNG), so two runs on the same
inputs produce byte-identical books.

What is reused vs implemented
-----------------------------

* **Reused** (canonical, so the research book cannot diverge from the deployable
  harvester's math):

  - :func:`src.equity.backtest.overlay` — the causal 10%-vol-target + drawdown
    de-gross circuit breaker. It does ``shift(1)`` on rolling vol and on the
    base book's drawdown internally, so the scalar at ``t`` never peeks at ``t``.
  - :func:`src.equity.backtest.run_portfolio_backtest` — the causal portfolio
    backtest (``execution_lag=1`` ⇒ ``shift(1)`` on the weight panel) with the
    per-side ``cost_bps`` turnover cost model.
  - :func:`src.factor.ship_gate.evaluate_gate` — the mechanical gate.

* **Implemented here** (the research-specific seam): per-rebalance latest-score
  gathering, the cross-sectional composite + Q5 (and secondary Q5−Q1) selection,
  the daily weight-panel construction, the overlay application onto the book, the
  arm splitting, and the deterministic placebo permutation.

Lookahead discipline
---------------------

Positions for any date ``t`` are derived ONLY from scores with ``as_of <= t`` and
prices ``<= t``; the returns are earned on ``t+1`` via the backtest's
``execution_lag=1``. A test asserts that shuffling FUTURE prices does not change
past positions.
"""

from __future__ import annotations

import logging
import math
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
from scipy.stats import norm

from src.equity.backtest import (
    BacktestError,
    overlay,
    run_portfolio_backtest,
    stats,
)
from src.equity.research.contracts import (
    ALL_ARMS,
    ARM_FULL,
    ARM_PLACEBO,
    ARM_POST_CUTOFF,
    ARM_PRE_CUTOFF,
    BONFERRONI_ALPHA,
    N_TRIALS,
    PRIMARY_WEIGHTS,
    ResearchScore,
)
from src.factor.ship_gate import evaluate_gate, verdict_to_dict
from src.research.trial_budget import bonferroni_alpha, resolve_trial_budget

logger = logging.getLogger(__name__)

# Overlay knobs mirror the deployable harvester defaults (§3.4 "identical to the
# deployable harvester"). vol_target is passed explicitly per the prereg; the
# drawdown de-gross band uses the harvester's validated soft/hard floors.
OVERLAY_DD_SOFT = 0.10
OVERLAY_DD_HARD = 0.20
OVERLAY_MAX_LEV = 1.0

# Quintile cut (§3.3): Q5 = top 20% by composite. Frozen.
QUINTILE = 5

# Minimum cross-section to form a quintile book at a rebalance. Below this we
# skip the rebalance (logged) rather than form a degenerate 1-name "quintile".
MIN_NAMES_FOR_QUINTILE = QUINTILE


class ResearchHarnessError(RuntimeError):
    """Raised when harness inputs are mis-shaped or internally inconsistent."""


# ---------------------------------------------------------------------------
# Input normalisation
# ---------------------------------------------------------------------------


def _normalize_prices(prices: pd.DataFrame) -> pd.DataFrame:
    """Coerce ``prices`` to a sorted, UTC, de-duplicated date x ticker frame."""
    if not isinstance(prices, pd.DataFrame) or prices.empty:
        raise ResearchHarnessError("prices must be a non-empty DataFrame")
    df = prices.copy()
    if not isinstance(df.index, pd.DatetimeIndex):
        try:
            df.index = pd.DatetimeIndex(df.index)
        except (TypeError, ValueError) as exc:
            raise ResearchHarnessError(
                f"prices index is not date-like: {exc}"
            ) from exc
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    else:
        df.index = df.index.tz_convert("UTC")
    df = df.sort_index()
    df = df[~df.index.duplicated(keep="last")]
    df.columns = [str(c) for c in df.columns]
    return df


def _to_utc_ts(iso_date: str) -> pd.Timestamp:
    """Parse an ISO date string to a UTC-normalised midnight timestamp."""
    ts = pd.Timestamp(iso_date)
    if ts.tz is None:
        ts = ts.tz_localize("UTC")
    else:
        ts = ts.tz_convert("UTC")
    return ts.normalize()


def _validate_scores(scores: Sequence[ResearchScore]) -> List[ResearchScore]:
    """Validate score ranges (prereg §3.1) and return them as a list.

    A score that is out of its pre-registered range is a contract violation, not
    something to silently clip — fail loud (mirrors the inference-contract rule).
    """
    if not isinstance(scores, (list, tuple)) or len(scores) == 0:
        raise ResearchHarnessError("scores must be a non-empty list")
    out: List[ResearchScore] = []
    for s in scores:
        if not isinstance(s, ResearchScore):
            raise ResearchHarnessError(
                f"every score must be a ResearchScore, got {type(s)!r}"
            )
        s.validate()
        out.append(s)
    return out


# ---------------------------------------------------------------------------
# Rebalance schedule + per-rebalance cross-section
# ---------------------------------------------------------------------------


def _rebalance_dates(
    price_index: pd.DatetimeIndex,
    step_days: int,
) -> List[pd.Timestamp]:
    """Pick rebalance bars every ``step_days`` trading rows of the price index.

    The first rebalance is the first available bar; the last realisable bar is
    excluded as a rebalance because its return cannot be earned (execution_lag=1
    needs a following bar).
    """
    if step_days < 1:
        raise ResearchHarnessError(f"rebalance_step_days must be >= 1, got {step_days}")
    n = len(price_index)
    if n < 2:
        raise ResearchHarnessError("price index needs >= 2 bars")
    # Stop before the final bar so each rebalance has at least one forward bar.
    idx_positions = range(0, n - 1, step_days)
    return [price_index[i] for i in idx_positions]


def _latest_scores_asof(
    scores_by_ticker: Dict[str, List[ResearchScore]],
    asof: pd.Timestamp,
) -> Dict[str, ResearchScore]:
    """Latest score per ticker with ``as_of <= asof`` (PIT — no lookahead)."""
    latest: Dict[str, ResearchScore] = {}
    for ticker, ticker_scores in scores_by_ticker.items():
        chosen: Optional[ResearchScore] = None
        chosen_ts: Optional[pd.Timestamp] = None
        for s in ticker_scores:
            s_ts = _to_utc_ts(s.as_of)
            if s_ts <= asof and (chosen_ts is None or s_ts >= chosen_ts):
                chosen = s
                chosen_ts = s_ts
        if chosen is not None:
            latest[ticker] = chosen
    return latest


def _presort_scores_by_ticker(
    scores_by_ticker: Dict[str, List[ResearchScore]],
) -> Dict[str, tuple]:
    """Pre-parse + sort each ticker's scores by ``as_of`` ascending, once.

    Feeds :func:`_latest_scores_asof_presorted` so ``_build_weight_panel``'s
    per-rebalance loop does an O(log n) bisect instead of re-parsing every ISO
    date string and linear-scanning on every rebalance (see
    :func:`_latest_scores_asof` docstring — that O(n) path is fine for Track
    B's sparse per-filing scores but is intractable for a DENSE per-day score
    panel, e.g. an alt-data z-score signal with one score per ticker per day).
    """
    presorted: Dict[str, tuple] = {}
    for ticker, ticker_scores in scores_by_ticker.items():
        if not ticker_scores:
            continue
        parsed = sorted(
            ((_to_utc_ts(s.as_of), s) for s in ticker_scores), key=lambda x: x[0]
        )
        ts_list, s_list = zip(*parsed)
        presorted[ticker] = (list(ts_list), list(s_list))
    return presorted


def _latest_scores_asof_presorted(
    presorted: Dict[str, tuple],
    asof: pd.Timestamp,
) -> Dict[str, ResearchScore]:
    """Latest score per ticker with ``as_of <= asof``, via bisect on pre-sorted lists.

    Byte-for-byte equivalent to :func:`_latest_scores_asof` given the same
    inputs (both pick the score with the maximal ``as_of <= asof``); this is
    purely a performance path for callers with large/dense score sets.
    """
    from bisect import bisect_right

    latest: Dict[str, ResearchScore] = {}
    for ticker, (ts_list, s_list) in presorted.items():
        idx = bisect_right(ts_list, asof) - 1
        if idx >= 0:
            latest[ticker] = s_list[idx]
    return latest


def _bit_reversal_order(n: int) -> List[int]:
    """A deterministic, RNG-free permutation of range(n) that scatters order.

    Sorts positions by the bit-reversal of their index (LSB-first). Adjacent
    positions land far apart and the result is NOT a contiguity-preserving
    rotation, so when applied to a rank-ordered value vector it genuinely
    destroys the ranking (a roll/rotation does not — it keeps a contiguous block
    on top, which would leak signal through the placebo). Fully reproducible.
    """
    if n <= 1:
        return list(range(n))
    width = max(1, (n - 1).bit_length())

    def _rev(x: int) -> int:
        r = 0
        for _ in range(width):
            r = (r << 1) | (x & 1)
            x >>= 1
        return r

    # Stable sort positions by (bit-reversed value, position) -> deterministic.
    return sorted(range(n), key=lambda i: (_rev(i), i))


def _derangement_order(n: int) -> List[int]:
    """A deterministic, RNG-free DERANGEMENT of range(n) that scatters order.

    Starts from the decorrelating bit-reversal scatter (:func:`_bit_reversal_order`)
    and repairs its fixed points so that ``order[i] != i`` for every ``i``. The
    bit-reversal alone fixes index 0 and clusters fixed points near powers of two
    (e.g. n=64 → 8 fixed points); a fixed point would let that ticker keep its OWN
    real composite in the placebo, leaking signal into the experiment's primary
    falsification control (reviewers HIGH, 2026-06-30). The repair below makes the
    near-zero placebo behaviour a PROPERTY of the permutation, not luck of ``n``.
    """
    if n < 2:
        return list(range(n))
    order = _bit_reversal_order(n)
    fixed = [i for i in range(n) if order[i] == i]
    if len(fixed) == 1:
        # Swap the lone fixed value with its neighbour's. The neighbour's value
        # != i (it's a permutation), so neither position is fixed afterward.
        i = fixed[0]
        j = (i + 1) % n
        order[i], order[j] = order[j], order[i]
    elif len(fixed) >= 2:
        # Rotate the fixed positions' values among themselves by one: position
        # f_k receives f_{k+1}'s value (!= f_k), erasing every fixed point while
        # staying a permutation (only these positions are touched).
        vals = [order[f] for f in fixed]
        rotated = vals[1:] + vals[:1]
        for f, v in zip(fixed, rotated):
            order[f] = v
    return order


def _placebo_permute(
    composites: Dict[str, float],
) -> Dict[str, float]:
    """Deterministically permute composite VALUES across tickers (no RNG).

    Destroys the score<->ticker link while preserving the marginal distribution
    of composite values for the rebalance. Uses a fixed-point-free DERANGEMENT
    (see :func:`_derangement_order`) keyed by the sorted-ticker index, so it is
    reproducible AND order-decorrelating, and — crucially — NO ticker ever keeps
    its own composite (which would leak real signal into the placebo top
    quintile). Returns a ticker -> permuted-composite dict.
    """
    tickers = sorted(composites)
    values = [composites[t] for t in tickers]
    n = len(values)
    if n < 2:
        # Single name: a permutation is a no-op; the placebo arm will be ~empty
        # signal anyway. Return as-is rather than inventing a value.
        return dict(composites)
    order = _derangement_order(n)
    # ticker[i] (sorted) receives the value at scattered position order[i].
    return {tickers[i]: values[order[i]] for i in range(n)}


def _quintile_long_only_weights(
    composites: Dict[str, float],
) -> Dict[str, float]:
    """Equal-weight the top quintile (Q5) by composite; weights sum to ~1.

    Whole-name, long-only. Ties are broken by ticker name for determinism.
    """
    if len(composites) < MIN_NAMES_FOR_QUINTILE:
        return {}
    # Deterministic ordering: primary key composite (desc), tiebreak ticker.
    ranked = sorted(composites.items(), key=lambda kv: (-kv[1], kv[0]))
    q5_n = max(1, len(ranked) // QUINTILE)
    top = ranked[:q5_n]
    w = 1.0 / len(top)
    return {ticker: w for ticker, _ in top}


def _quintile_long_short_weights(
    composites: Dict[str, float],
) -> Dict[str, float]:
    """Q5 long minus Q1 short, dollar-neutral (each leg sums to 0.5 gross).

    Secondary book — reported, never the ship candidate (§3.3).
    """
    if len(composites) < 2 * MIN_NAMES_FOR_QUINTILE:
        return {}
    ranked = sorted(composites.items(), key=lambda kv: (-kv[1], kv[0]))
    q_n = max(1, len(ranked) // QUINTILE)
    top = ranked[:q_n]
    bot = ranked[-q_n:]
    long_w = 0.5 / len(top)
    short_w = 0.5 / len(bot)
    weights: Dict[str, float] = {t: long_w for t, _ in top}
    for t, _ in bot:
        weights[t] = weights.get(t, 0.0) - short_w
    return weights


# ---------------------------------------------------------------------------
# Weight-panel construction (daily, held between rebalances)
# ---------------------------------------------------------------------------


def _build_weight_panel(
    scores_by_ticker: Dict[str, List[ResearchScore]],
    prices: pd.DataFrame,
    rebalance_dates: Sequence[pd.Timestamp],
    weights: Dict[str, float],
    *,
    long_only: bool,
    placebo: bool,
) -> pd.DataFrame:
    """Build a daily date x ticker target-weight panel from the score book.

    For each rebalance date we gather the latest PIT score per ticker, composite
    them, optionally permute (placebo), select the Q5 (long-only) or Q5−Q1 (L/S)
    book, and STAMP that book onto the rebalance row. The panel is then
    forward-filled between rebalances so the book is held until the next
    rebalance. The backtest applies ``shift(1)`` (execution_lag=1) on top of
    this, so the held weights are realised one bar late — causal.
    """
    all_tickers = sorted(prices.columns)
    panel = pd.DataFrame(
        0.0, index=prices.index, columns=all_tickers, dtype=float
    )
    # Mark which rows are rebalance rows so we can ffill ONLY from them.
    stamped_rows: List[pd.Timestamp] = []

    # Pre-parse/sort once (O(n log n)) so the per-rebalance lookup below is an
    # O(log n) bisect rather than re-parsing every score's ISO date on every
    # rebalance (O(n) per lookup) — see _presort_scores_by_ticker docstring.
    presorted = _presort_scores_by_ticker(scores_by_ticker)

    for rb in rebalance_dates:
        latest = _latest_scores_asof_presorted(presorted, rb)
        if not latest:
            continue
        composites = {
            t: s.composite(weights) for t, s in latest.items()
            if t in panel.columns
        }
        # Require the name to be priced on the rebalance bar (tradeable set).
        composites = {
            t: c for t, c in composites.items()
            if t in prices.columns and pd.notna(prices.at[rb, t])
        }
        if not composites:
            continue
        if placebo:
            composites = _placebo_permute(composites)
        if long_only:
            book = _quintile_long_only_weights(composites)
        else:
            book = _quintile_long_short_weights(composites)
        if not book:
            continue
        # Zero the row first (a fresh book), then stamp.
        panel.loc[rb, :] = 0.0
        for ticker, wt in book.items():
            panel.at[rb, ticker] = wt
        stamped_rows.append(rb)

    if not stamped_rows:
        raise ResearchHarnessError(
            "no rebalance produced a non-empty book — check scores vs prices "
            "coverage (need >= 5 priced, scored names on a rebalance bar)"
        )

    # Forward-fill the book between rebalances: rows that are not a stamped
    # rebalance inherit the most recent stamped book. We do this by masking
    # non-stamped rows to NaN, then ffill, then fill the leading gap with 0.
    stamped_mask = panel.index.isin(stamped_rows)
    held = panel.where(
        pd.Series(stamped_mask, index=panel.index), other=np.nan
    )
    held = held.ffill().fillna(0.0)
    return held


def _apply_overlay(
    weight_panel: pd.DataFrame,
    prices: pd.DataFrame,
    *,
    vol_target: float,
) -> pd.DataFrame:
    """Scale the book by the causal vol-target + DD de-gross overlay.

    The base-book return that drives the overlay is the equal-weight return of
    the held names (the gross book return at unit exposure). ``overlay`` does
    ``shift(1)`` internally so the scalar at ``t`` uses only info ``<= t-1`` —
    causal. The scaled weights are then handed to ``run_portfolio_backtest``
    which applies a further ``shift(1)`` execution lag.
    """
    asset_ret = prices.pct_change()
    # Gross book return at each bar from the (pre-lag) held weights. This is the
    # book whose realised vol we target. NaN asset returns (first row) -> 0.
    base_ret = (weight_panel * asset_ret).sum(axis=1, skipna=True).fillna(0.0)
    scalar = overlay(
        base_ret,
        target_vol=vol_target,
        dd_soft=OVERLAY_DD_SOFT,
        dd_hard=OVERLAY_DD_HARD,
        max_lev=OVERLAY_MAX_LEV,
    )
    scalar = scalar.reindex(weight_panel.index).fillna(0.0)
    return weight_panel.mul(scalar, axis=0)


# ---------------------------------------------------------------------------
# Per-arm evaluation
# ---------------------------------------------------------------------------


def _years_span(returns: pd.Series) -> float:
    """Approximate the return series span in calendar years."""
    ret = returns.dropna()
    if len(ret) < 2:
        return 0.0
    delta_days = (ret.index.max() - ret.index.min()).days
    return delta_days / 365.25


def _build_report(returns: pd.Series) -> Dict[str, object]:
    """Build the dict :func:`evaluate_gate` expects, computed for REAL.

    ``walk_forward`` is True iff the realised window is at least the gate's
    ``MIN_TOTAL_YEARS`` long (a genuine multi-year out-of-sample span); a short
    arm (e.g. the post-cutoff arm) reports its real, shorter span and will fail
    the ``history_length`` / ``walk_forward`` criteria honestly rather than
    being faked into a pass.
    """
    summ = stats(returns)
    net_sharpe = float(summ.get("net_sharpe", 0.0))
    max_dd = float(summ.get("max_dd", 1.0))
    per_year = summ.get("per_year", {}) or {}
    positive_years = int(sum(1 for v in per_year.values() if v > 0))
    total_years = int(len(per_year))
    span_years = _years_span(returns)
    # walk_forward: real multi-year span, not a synthetic flag.
    walk_forward = span_years >= 10.0 and total_years >= 10
    return {
        "net_sharpe": net_sharpe,
        "positive_years": positive_years,
        "total_years": total_years,
        "max_drawdown": max_dd,
        "walk_forward": walk_forward,
        "span_years": round(span_years, 3),
        "per_year": per_year,
    }


def _effective_n(
    weight_panel: pd.DataFrame,
    realised_index: pd.Index,
) -> Optional[float]:
    """Average held-name count on bars where the book is non-empty.

    A correlation participation ratio needs a cross-section of per-name return
    streams; for a long-only whole-name book the held-name count is the honest,
    directly-computable effective breadth. Returns None if the book never holds.
    """
    held = weight_panel.reindex(index=realised_index).fillna(0.0)
    counts = (held.abs() > 1e-12).sum(axis=1)
    active = counts[counts > 0]
    if active.empty:
        return None
    return float(active.mean())


# ---------------------------------------------------------------------------
# §4.5 — Deflated Sharpe Ratio (DSR-OOS, N=22) + Bonferroni block-bootstrap.
# ---------------------------------------------------------------------------

# Minimum realised bars below which neither statistic is meaningful. Below this
# floor we report None with an explicit reason rather than a noise-driven number
# (mirrors the inference-contract "abstain, never fabricate" rule).
_MIN_BARS_FOR_SIGNIFICANCE = 10

# Fixed block-bootstrap knobs. A block length of 21 (one rebalance cycle) keeps
# the autocorrelation structure the causal overlay induces; a fixed RNG seed
# makes the p-value reproducible byte-for-bit across runs on the same input,
# matching the module's "no RNG" determinism discipline in spirit — the
# resampling scheme itself needs randomness, but the SEED is frozen so two runs
# on the same returns produce the identical bootstrap distribution.
_BOOTSTRAP_BLOCK_SIZE = 21
_BOOTSTRAP_N_REPS = 5000
_BOOTSTRAP_SEED = 20260630  # frozen with the prereg date; never tuned per-result

_EULER_MASCHERONI = 0.5772156649015329


def _deflated_sharpe_ratio(
    returns: pd.Series, n_trials: int
) -> Optional[Dict[str, float]]:
    """Bailey & Lopez de Prado (2014) DSR, adjusted for ``n_trials`` multiple testing.

    Works in PER-PERIOD (daily, un-annualised) Sharpe units throughout — DSR is a
    probability (in [0, 1]), not itself annualised. Returns ``None`` when there
    are too few bars to estimate skew/kurtosis/variance meaningfully (honest
    abstention, never a noise-driven number).

    ``sr0`` is the expected maximum Sharpe ratio across ``n_trials`` independent
    trials under a true-zero-Sharpe null (the multiple-testing benchmark the
    observed Sharpe must clear); ``dsr`` is the probability the observed Sharpe
    exceeds that benchmark once skew/kurtosis/sample-size are accounted for.
    The trial-to-trial Sharpe variance (``sr_std``) is approximated by this
    trial's own PSR denominator — the standard practical substitution used when
    the actual empirical Sharpe distribution of the other N-1 trials is not
    available (see e.g. mlfinlab's deflated-Sharpe implementation).
    """
    ret = returns.dropna()
    t = len(ret)
    if t < _MIN_BARS_FOR_SIGNIFICANCE:
        return None
    std = float(ret.std(ddof=1))
    if std <= 1e-12:
        return None
    sr = float(ret.mean() / std)
    skew = float(ret.skew())
    kurt = float(ret.kurtosis()) + 3.0  # pandas kurtosis() is EXCESS; formula wants Pearson's
    if pd.isna(skew) or pd.isna(kurt):
        return None
    variance_term = 1.0 - skew * sr + (kurt - 1.0) / 4.0 * sr**2
    if variance_term <= 0:
        return None
    sr_std = math.sqrt(variance_term / (t - 1))
    if sr_std <= 1e-12:
        return None
    sr0 = sr_std * (
        (1.0 - _EULER_MASCHERONI) * norm.ppf(1.0 - 1.0 / n_trials)
        + _EULER_MASCHERONI * norm.ppf(1.0 - 1.0 / (n_trials * math.e))
    )
    dsr = float(norm.cdf((sr - sr0) / sr_std))
    return {
        "sr_hat_per_period": sr,
        "sr0_benchmark_per_period": float(sr0),
        "sr_std_per_period": sr_std,
        "skew": skew,
        "kurtosis_pearson": kurt,
        "n_obs": t,
        "dsr": dsr,
    }


def _circular_block_bootstrap_sharpe_pvalue(
    returns: pd.Series,
    *,
    block_size: int = _BOOTSTRAP_BLOCK_SIZE,
    n_reps: int = _BOOTSTRAP_N_REPS,
    seed: int = _BOOTSTRAP_SEED,
) -> Optional[Dict[str, float]]:
    """One-sided block-bootstrap p-value that the true (per-period) Sharpe <= 0.

    Circular block bootstrap: resamples overlapping blocks of ``block_size``
    consecutive returns (wrapping at the series end) to preserve the
    autocorrelation the causal overlay induces, unlike an iid bootstrap. The RNG
    is seeded so the same input returns always yield the identical p-value.
    Returns ``None`` below :data:`_MIN_BARS_FOR_SIGNIFICANCE` bars.
    """
    ret = returns.dropna().to_numpy()
    t = len(ret)
    if t < _MIN_BARS_FOR_SIGNIFICANCE:
        return None
    block = max(1, min(block_size, t))
    n_blocks = math.ceil(t / block)
    rng = np.random.default_rng(seed)
    starts = rng.integers(0, t, size=(n_reps, n_blocks))
    boot_sharpes = np.empty(n_reps, dtype=float)
    for i in range(n_reps):
        pieces = [
            ret[np.arange(s, s + block) % t] for s in starts[i]
        ]
        sample = np.concatenate(pieces)[:t]
        sd = sample.std(ddof=1)
        boot_sharpes[i] = (sample.mean() / sd) if sd > 1e-12 else 0.0
    p_le_zero = float(np.mean(boot_sharpes <= 0.0))
    return {
        "p_oos_sharpe_le_zero": p_le_zero,
        "block_size": block,
        "n_bootstrap_reps": n_reps,
        "seed": seed,
        "boot_sharpe_mean": float(np.mean(boot_sharpes)),
        "boot_sharpe_std": float(np.std(boot_sharpes, ddof=1)),
    }


def _dsr_oos_n22(
    arm_returns: pd.Series,
    n_trials: int = N_TRIALS,
    *,
    frozen_replay: bool = False,
) -> Dict[str, object]:
    """Prereg §4.5 criterion: DSR-OOS(N=22) >= 0.95 AND Bonferroni p-OOS < alpha.

    Combines :func:`_deflated_sharpe_ratio` and
    :func:`_circular_block_bootstrap_sharpe_pvalue` into the single dict the
    harness reports per arm. ``passes_significance`` is ``None`` (not False)
    when either component could not be computed (too few bars) — an uncomputed
    criterion is never silently treated as failed OR passed.

    ``n_trials`` is validated against the campaign register
    (:mod:`src.research.trial_budget`) and the Bonferroni alpha is DERIVED from
    it. Previously the alpha was the module constant regardless of the
    ``n_trials`` argument, so calling with a different budget deflated the DSR
    at one budget while testing the bootstrap p at another. At the default
    (``N_TRIALS`` = 24) the derived alpha is identical to the old constant, so
    no recorded verdict moves. ``frozen_replay=True`` permits a historical
    budget for re-deriving an archived artifact exactly as recorded.
    """
    applied_trials = resolve_trial_budget(
        n_trials,
        context="equity.research.harness._dsr_oos_n22",
        frozen_replay=frozen_replay,
    )
    alpha = bonferroni_alpha(applied_trials)
    dsr_block = _deflated_sharpe_ratio(arm_returns, applied_trials)
    boot_block = _circular_block_bootstrap_sharpe_pvalue(arm_returns)
    passes: Optional[bool] = None
    if dsr_block is not None and boot_block is not None:
        passes = bool(
            dsr_block["dsr"] >= 0.95
            and boot_block["p_oos_sharpe_le_zero"] < alpha
        )
    return {
        "dsr": dsr_block["dsr"] if dsr_block else None,
        "bonferroni_alpha": alpha,
        "p_oos_bootstrap": boot_block["p_oos_sharpe_le_zero"] if boot_block else None,
        "passes_significance": passes,
        "n_trials": applied_trials,
        "budget_source": "src.research.trial_budget.TRIAL_LEDGER",
        "budget_is_authoritative": applied_trials == N_TRIALS,
        "detail": {"dsr": dsr_block, "bootstrap": boot_block},
        "insufficient_data": dsr_block is None or boot_block is None,
    }


def _run_one_arm(
    scores_by_ticker: Dict[str, List[ResearchScore]],
    prices: pd.DataFrame,
    rebalance_dates: Sequence[pd.Timestamp],
    weights: Dict[str, float],
    *,
    vol_target: float,
    cost_bps: float,
    placebo: bool,
    long_only: bool = True,
    window: Optional[tuple] = None,
) -> Dict[str, object]:
    """Build the book for one arm, run the causal backtest, evaluate the gate.

    ``window`` is an optional ``(lo, hi)`` pair of UTC timestamps (either may be
    ``None``). When set, the REALISED RETURN SERIES is sliced to the window
    BEFORE computing stats / the gate report — so the pre-cutoff arm earns only
    pre-cutoff returns and the post-cutoff arm only post-cutoff returns, even
    though both build their weight panel over the shared price index. The full
    weight panel is still built so positions held across the window boundary are
    causally correct; only the evaluation window is restricted.
    """
    panel = _build_weight_panel(
        scores_by_ticker,
        prices,
        rebalance_dates,
        weights,
        long_only=long_only,
        placebo=placebo,
    )
    managed = _apply_overlay(panel, prices, vol_target=vol_target)
    bt = run_portfolio_backtest(
        managed,
        prices,
        cost_bps=cost_bps,
        execution_lag=1,
    )
    arm_returns = bt.returns
    if window is not None:
        lo, hi = window
        if lo is not None:
            arm_returns = arm_returns[arm_returns.index >= lo]
        if hi is not None:
            arm_returns = arm_returns[arm_returns.index < hi]
        if arm_returns.empty:
            raise ResearchHarnessError(
                f"arm window {window} produced an empty realised return series"
            )
    report = _build_report(arm_returns)
    verdict = evaluate_gate(report)
    eff_n = _effective_n(managed, arm_returns.index)
    # Equity curve recomputed over the ARM window so the summary matches the
    # arm's own realised returns (not the full-sample curve).
    arm_equity = (1.0 + arm_returns).cumprod()
    arm_turnover = bt.turnover.reindex(arm_returns.index)
    arm_costs = bt.costs.reindex(arm_returns.index)
    equity_summary = {
        "final_equity": float(arm_equity.iloc[-1])
        if len(arm_equity)
        else 1.0,
        "n_bars": int(len(arm_returns)),
        "avg_turnover": float(arm_turnover.mean())
        if len(arm_turnover)
        else 0.0,
        "avg_cost": float(arm_costs.mean()) if len(arm_costs) else 0.0,
        "span": (
            f"{arm_returns.index.min().date()}->"
            f"{arm_returns.index.max().date()}"
        ),
    }
    dsr_result = _dsr_oos_n22(arm_returns)
    return {
        "report": report,
        "gate": verdict_to_dict(verdict),
        "gate_passed": bool(verdict.passed),
        "equity_curve_summary": equity_summary,
        "effective_n": eff_n,
        # §4.5 significance extension: DSR-OOS(N=22) + Bonferroni block-bootstrap
        # p-OOS, computed for REAL (see _dsr_oos_n22). `passes_significance` is
        # None (not False) when too few bars exist to estimate it — an
        # uncomputed criterion is never silently treated as passed or failed.
        "dsr_oos_n22": dsr_result,
        "n_trials": N_TRIALS,
        "bonferroni_note": (
            f"multiple-testing budget N_TRIALS={N_TRIALS}, alpha="
            f"{BONFERRONI_ALPHA:.5f}; dsr_oos_n22.passes_significance reflects "
            "criterion 5 (§4.5) and is SEPARATE from gate_passed (criteria 1-4); "
            "both must hold for a full clear."
        ),
        "is_clean_arm": False,  # set by caller for the post-cutoff arm
    }


def run_research_backtest(
    scores: List[ResearchScore],
    prices: pd.DataFrame,
    model_cutoff: str,
    *,
    weights: Dict[str, float] = PRIMARY_WEIGHTS,
    rebalance_step_days: int = 21,
    vol_target: float = 0.10,
    cost_bps: float = 2.0,
    blinding_audit_clean: Optional[bool] = None,
) -> dict:
    """Evaluate the research book across the four pre-registered arms.

    Parameters
    ----------
    scores:
        List of :class:`ResearchScore` artifacts (one per ``(ticker, as_of)``),
        produced OFFLINE. The harness never calls an LLM.
    prices:
        Date-indexed (UTC) adjusted-close panel; columns are tickers.
    model_cutoff:
        ISO date — the research model's training cutoff. Splits the pre/post
        arms: ``pre_cutoff`` = rebalances strictly before it, ``post_cutoff`` =
        rebalances on/after it (the only clean / uncontaminated arm).
    weights:
        Composite weights (default the frozen ``PRIMARY_WEIGHTS``).
    rebalance_step_days:
        Trading-bar cadence between rebalances (default 21 ≈ monthly).
    vol_target:
        Annualised vol target for the causal overlay (default 0.10).
    cost_bps:
        Per-side cost in basis points on turnover (default 2.0).

    Returns
    -------
    dict
        ``{arm: {report, gate, gate_passed, equity_curve_summary, effective_n,
        dsr_oos_n22, ...} for arm in ALL_ARMS}`` plus a top-level ``summary``
        and the run parameters.
    """
    scores = _validate_scores(scores)
    prices = _normalize_prices(prices)
    cutoff_ts = _to_utc_ts(model_cutoff)

    # Group scores by ticker once (consumed by every arm).
    scores_by_ticker: Dict[str, List[ResearchScore]] = {}
    for s in scores:
        scores_by_ticker.setdefault(str(s.ticker), []).append(s)

    all_rb = _rebalance_dates(prices.index, rebalance_step_days)
    pre_rb = [d for d in all_rb if d < cutoff_ts]
    post_rb = [d for d in all_rb if d >= cutoff_ts]

    arm_rebalances = {
        ARM_FULL: all_rb,
        ARM_PRE_CUTOFF: pre_rb,
        ARM_POST_CUTOFF: post_rb,
        ARM_PLACEBO: all_rb,  # same dates as FULL; the SCORES are permuted
    }
    arm_placebo = {
        ARM_FULL: False,
        ARM_PRE_CUTOFF: False,
        ARM_POST_CUTOFF: False,
        ARM_PLACEBO: True,
    }
    # Per-arm realised-return window. Pre/post earn returns only in their span;
    # full/placebo span the whole sample. (lo inclusive, hi exclusive.)
    arm_window = {
        ARM_FULL: None,
        ARM_PRE_CUTOFF: (None, cutoff_ts),
        ARM_POST_CUTOFF: (cutoff_ts, None),
        ARM_PLACEBO: None,
    }

    results: Dict[str, object] = {}
    for arm in ALL_ARMS:
        rb_dates = arm_rebalances[arm]
        if not rb_dates:
            results[arm] = {
                "report": None,
                "gate": None,
                "gate_passed": False,
                "equity_curve_summary": None,
                "effective_n": None,
                "dsr_oos_n22": None,
                "n_trials": N_TRIALS,
                "skipped": True,
                "skip_reason": (
                    f"no rebalance dates in arm '{arm}' "
                    f"(cutoff={model_cutoff})"
                ),
            }
            continue
        try:
            arm_result = _run_one_arm(
                scores_by_ticker,
                prices,
                rb_dates,
                weights,
                vol_target=vol_target,
                cost_bps=cost_bps,
                placebo=arm_placebo[arm],
                window=arm_window[arm],
            )
        except (ResearchHarnessError, BacktestError) as exc:
            logger.warning("arm %s could not be evaluated: %s", arm, exc)
            arm_result = {
                "report": None,
                "gate": None,
                "gate_passed": False,
                "equity_curve_summary": None,
                "effective_n": None,
                "dsr_oos_n22": None,
                "n_trials": N_TRIALS,
                "skipped": True,
                "skip_reason": str(exc),
            }
        if arm == ARM_POST_CUTOFF and not arm_result.get("skipped"):
            arm_result["is_clean_arm"] = True
        results[arm] = arm_result

    summary = _build_summary(results, blinding_audit_clean=blinding_audit_clean)
    return {
        **results,
        "summary": summary,
        "params": {
            "model_cutoff": model_cutoff,
            "weights": dict(weights),
            "rebalance_step_days": rebalance_step_days,
            "vol_target": vol_target,
            "cost_bps": cost_bps,
            "n_scores": len(scores),
            "n_tickers": len(scores_by_ticker),
        },
    }


def _build_summary(
    results: Dict[str, object],
    *,
    blinding_audit_clean: Optional[bool] = None,
) -> Dict[str, object]:
    """Top-level cross-arm summary + the frozen control decision rule (§1).

    The prereg decision rule: reportable as a real edge ONLY IF the post-cutoff
    OOS arm is directionally consistent with full AND the placebo is ≈0
    (|Sharpe| < 0.15). We expose the booleans; the gate verdict per arm is also
    returned so a verifier can re-derive this.

    ``blinding_audit_clean`` wires prereg §8's deferred item 5 (the human/LLM
    re-identification audit result) into the binding verdict. It defaults to
    ``None`` — preserving the original fail-closed behaviour (REAL unreachable)
    for any caller that does not supply it — and must be produced by an actual
    audit run OUTSIDE this pure/deterministic module (this harness never reads
    filing text or makes an identity judgement itself).
    """
    def _sharpe(arm: str) -> Optional[float]:
        a = results.get(arm)
        if not isinstance(a, dict) or a.get("report") is None:
            return None
        return float(a["report"].get("net_sharpe", 0.0))

    full_s = _sharpe(ARM_FULL)
    post_s = _sharpe(ARM_POST_CUTOFF)
    placebo_s = _sharpe(ARM_PLACEBO)

    placebo_clean = placebo_s is not None and abs(placebo_s) < 0.15
    # §1.2 "no in-sample->OOS sign flip" is only meaningful when the full arm
    # actually shows a positive edge. A both-negative case is NOT "consistent"
    # (there is no edge to confirm) — the old `full_s<=0 or post_s>0` form
    # mislabelled it True. Require: full shows an edge AND post-cutoff keeps the
    # same sign without collapsing (post must stay positive).
    post_consistent = (
        full_s is not None
        and post_s is not None
        and full_s > 0           # there is a full-sample edge to confirm
        and post_s > 0           # it survives into the clean window (no sign flip)
    )
    arm_gate_passed = {
        arm: bool(results[arm].get("gate_passed", False))
        if isinstance(results.get(arm), dict)
        else False
        for arm in ALL_ARMS
    }

    # §4.7 BINDING verdict — controls OVERRIDE any arm gate pass. The blinding
    # audit (§1.1) is a separate human/LLM verifier step not computed here; it
    # is an INPUT the caller supplies (see ``blinding_audit_clean`` param) — an
    # uncomputed control (None, the default) is treated as not-yet-satisfied,
    # never as passed (fail-closed).
    controls_satisfied = (
        placebo_clean
        and post_consistent
        and blinding_audit_clean is True
    )
    # §4.5 significance extension (DSR-OOS(N=22) + Bonferroni p-OOS), computed
    # for the FULL arm. None (uncomputed / too few bars) counts as not-satisfied
    # — same fail-closed discipline as every other control here.
    full_result = results.get(ARM_FULL)
    full_dsr = (
        full_result.get("dsr_oos_n22") if isinstance(full_result, dict) else None
    )
    full_significance_passes = bool(
        isinstance(full_dsr, dict) and full_dsr.get("passes_significance") is True
    )
    full_gate = arm_gate_passed.get(ARM_FULL, False)
    full_clears_criteria_1_5 = full_gate and full_significance_passes
    if controls_satisfied and full_clears_criteria_1_5:
        overall_verdict = "REAL"
    elif full_gate and not (controls_satisfied and full_significance_passes):
        # The dangerous case the §1 apparatus exists to catch: a contaminated
        # full-sample arm clears criteria 1-4 but the §1 controls and/or the
        # §4.5 significance extension do not confirm it.
        overall_verdict = "LOOKAHEAD_CONTAMINATED"
    else:
        overall_verdict = "INSUFFICIENT"  # no full-arm gate pass to adjudicate

    return {
        "arm_net_sharpe": {
            ARM_FULL: full_s,
            ARM_PRE_CUTOFF: _sharpe(ARM_PRE_CUTOFF),
            ARM_POST_CUTOFF: post_s,
            ARM_PLACEBO: placebo_s,
        },
        "arm_gate_passed": arm_gate_passed,
        "placebo_is_clean": bool(placebo_clean),
        "post_cutoff_consistent_with_full": bool(post_consistent),
        "blinding_audit_clean": blinding_audit_clean,
        "full_significance_passes": full_significance_passes,
        # THE single field a consumer must read. Fuses gate + §1 controls + the
        # §4.5 significance extension; a contaminated full-arm pass can NEVER
        # read as REAL here.
        "overall_verdict": overall_verdict,
        "controls_note": (
            "overall_verdict is BINDING (§4.7): REAL requires full-arm gate pass "
            "(criteria 1-4) AND dsr_oos_n22.passes_significance on the full arm "
            "(criterion 5) AND placebo ~0 (|Sharpe|<0.15) AND post-cutoff "
            "confirms the edge (full>0 and post>0) AND a clean blinding audit "
            "(§1.1 — caller-supplied; None/unset is treated as not satisfied). "
            "Read overall_verdict, NOT any single arm's gate_passed."
        ),
    }


__all__ = [
    "ResearchHarnessError",
    "run_research_backtest",
]
