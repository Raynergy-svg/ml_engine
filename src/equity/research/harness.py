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
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

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
    N_TRIALS,
    PRIMARY_WEIGHTS,
    ResearchScore,
)
from src.factor.ship_gate import evaluate_gate, verdict_to_dict

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


def _placebo_permute(
    composites: Dict[str, float],
) -> Dict[str, float]:
    """Deterministically permute composite VALUES across tickers (no RNG).

    Destroys the score<->ticker link while preserving the marginal distribution
    of composite values for the rebalance. The permutation is a fixed
    bit-reversal scatter (see :func:`_bit_reversal_order`) keyed by the
    sorted-ticker index, so it is reproducible AND order-decorrelating: unlike a
    roll/rotation it does not keep the top-composite names contiguous, so a
    monotone score ramp does not leak through into the placebo's top quintile.
    Returns a ticker -> permuted-composite dict.
    """
    tickers = sorted(composites)
    values = [composites[t] for t in tickers]
    n = len(values)
    if n < 2:
        # Single name: a permutation is a no-op; the placebo arm will be ~empty
        # signal anyway. Return as-is rather than inventing a value.
        return dict(composites)
    order = _bit_reversal_order(n)
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

    for rb in rebalance_dates:
        latest = _latest_scores_asof(scores_by_ticker, rb)
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
    return {
        "report": report,
        "gate": verdict_to_dict(verdict),
        "gate_passed": bool(verdict.passed),
        "equity_curve_summary": equity_summary,
        "effective_n": eff_n,
        # DSR-OOS(N=22) is a deliberate, clearly-labelled TODO — a full deflated
        # Sharpe / Bonferroni block-bootstrap OOS test is out of scope for this
        # first build. We report net Sharpe, maxDD and positive-years for REAL
        # (above) and the multiple-testing budget, but do NOT fake a DSR number.
        "dsr_oos_n22": None,  # TODO(prereg §4.5): block-bootstrap DSR-OOS(N=22)
        "n_trials": N_TRIALS,
        "bonferroni_note": (
            f"multiple-testing budget N_TRIALS={N_TRIALS}; DSR-OOS / Bonferroni "
            "p-OOS not yet computed (TODO §4.5) — gate verdict above reflects "
            "criteria 1-4 only, not the significance extension §4.5"
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

    summary = _build_summary(results)
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


def _build_summary(results: Dict[str, object]) -> Dict[str, object]:
    """Top-level cross-arm summary + the frozen control decision rule (§1).

    The prereg decision rule: reportable as a real edge ONLY IF the post-cutoff
    OOS arm is directionally consistent with full AND the placebo is ≈0
    (|Sharpe| < 0.15). We expose the booleans; the gate verdict per arm is also
    returned so a verifier can re-derive this.
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
    post_consistent = (
        full_s is not None
        and post_s is not None
        and (full_s <= 0 or post_s > 0)  # no in-sample->OOS sign flip
    )
    return {
        "arm_net_sharpe": {
            ARM_FULL: full_s,
            ARM_PRE_CUTOFF: _sharpe(ARM_PRE_CUTOFF),
            ARM_POST_CUTOFF: post_s,
            ARM_PLACEBO: placebo_s,
        },
        "arm_gate_passed": {
            arm: bool(results[arm].get("gate_passed", False))
            if isinstance(results.get(arm), dict)
            else False
            for arm in ALL_ARMS
        },
        "placebo_is_clean": bool(placebo_clean),
        "post_cutoff_consistent_with_full": bool(post_consistent),
        "controls_note": (
            "reportable as real edge ONLY IF post-cutoff consistent with full "
            "AND placebo ~0 (|Sharpe|<0.15); entity-blinding audit is a "
            "separate verifier step not computed here (§1.1)"
        ),
    }


__all__ = [
    "ResearchHarnessError",
    "run_research_backtest",
]
