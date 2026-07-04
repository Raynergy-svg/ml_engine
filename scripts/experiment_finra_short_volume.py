#!/usr/bin/env python3
"""FINRA daily short-volume experiment — H1 (abnormal pressure) + H2 (Delta-pressure).

Pre-registered in ``docs/prereg-finra-short-volume-2026-07-04.md`` (FROZEN
BEFORE this script produced any result — do not edit hypotheses, thresholds,
signal formula, rebalance cadence, or the multiple-testing budget after seeing
numbers from this script; a later edit to those constants is dredging and must
be called out explicitly, not silently made).

Signals (frozen, prereg SS "Signal construction"):

    SVR_t = ShortVolume_t / TotalVolume_t                      (per ticker)
    z_t   = (SVR_t - mean(SVR, trailing 60)) / std(SVR, trailing 60)
            computed PER-TICKER against only that ticker's own trailing
            history (no cross-sectional peeking, no future data).

    H1 (primary):   quintile long-short on z_t.
                    Long bottom quintile (low abnormal short pressure),
                    short top quintile (high abnormal short pressure) ->
                    hypothesis: heavy-short-pressure names underperform.
    H2 (secondary): Delta z_t = z_t - z_{t-5} (5-session change).
                    Quintile long-short on Delta z_t, same long/short sense
                    (rapid INCREASE in pressure predicts negative fwd return).

Rebalance cadence — DELIBERATE DIFFERENCE, documented per the pre-reg's
explicit instruction not to silently reuse H1's cadence for H2:

    H1 uses rebalance_step_days=5 (weekly) per the pre-reg's explicit
    instruction ("Weekly rebalance cadence").
    H2's hypothesis is a 1-3 SESSION forward horizon (a much faster,
    liquidity-driven reversal), not a weekly one. Rebalancing H1's cadence
    (every 5 sessions) for a signal whose edge is hypothesised to decay
    within 1-3 sessions would materially understate/misrepresent H2's own
    pre-registered horizon: most of H2's return would occur AFTER the
    position was already replaced by the next rebalance's DIFFERENT z-score
    regime, diluting whatever true short-horizon effect exists (and biasing
    towards ~0 in a way that would misleadingly read as "no edge" even if a
    faster-cadence version had one). rebalance_step_days=1 (every session)
    is the honest mapping of "1-3 session forward horizon" onto the harness's
    step-based rebalance-date picker — the position is refreshed every session
    so the realised holding period for any given entry is bounded by the very
    next day's rebalance, matching the pre-reg's own 1-3 session framing far
    more closely than a 5-day step would.

Adapter pattern (the entire point of this script): package each (ticker, day)
z-score into a ``ResearchScore`` (``fundamental_quality = clip(-z_t, -1, 1)``,
sign-flipped so higher composite = predicted-better = long side, since HIGH
abnormal short pressure should predict WORSE returns), then hand the scores to
the FROZEN Track B harness machinery
(``_rebalance_dates``, ``_quintile_long_short_weights``, ``_build_weight_panel``,
``_apply_overlay``, ``_deflated_sharpe_ratio``, ``_circular_block_bootstrap_sharpe_pvalue``,
``_dsr_oos_n22``, ``src.factor.ship_gate.evaluate_gate``) completely unmodified.

Usage:
    python scripts/experiment_finra_short_volume.py
    python scripts/experiment_finra_short_volume.py --refresh   # re-fetch FINRA files first
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src.equity.backtest import run_portfolio_backtest  # noqa: E402
from src.equity.research.contracts import ResearchScore  # noqa: E402
from src.equity.research.finra_short_volume_loader import (  # noqa: E402
    load_short_volume_ratio_panel,
)
from src.equity.research.harness import (  # noqa: E402
    _apply_overlay,
    _build_report,
    _build_weight_panel,
    _dsr_oos_n22,  # wraps _deflated_sharpe_ratio + _circular_block_bootstrap_sharpe_pvalue
    _effective_n,
    _normalize_prices,
    _rebalance_dates,
    MIN_NAMES_FOR_QUINTILE,
    ResearchHarnessError,
)
from src.equity.backtest import BacktestError  # noqa: E402
from src.factor.ship_gate import evaluate_gate, verdict_to_dict  # noqa: E402

PRICE_PANEL_PATH = REPO_ROOT / "market_data" / "equity" / "sp500_prices.parquet"
SAMPLE_START = "2019-01-02"
SAMPLE_END = "2026-06-24"

# Signal constants (frozen, prereg SS "Signal construction").
Z_TRAILING_WINDOW = 60          # sessions
MIN_TRAILING_HISTORY = 60       # sessions, needed for a defined z-score
H2_DELTA_LAG = 5                # sessions, Delta z_t = z_t - z_{t-5}

# Rebalance cadence: H1 weekly (frozen, pre-reg explicit); H2 faster, mapped to
# the 1-3 session horizon (see module docstring for the reasoning).
H1_REBALANCE_STEP_DAYS = 5
H2_REBALANCE_STEP_DAYS = 1

# Reused unmodified from the harness / backtest (identical to Track B).
VOL_TARGET = 0.10
COST_BPS = None  # None -> use src.equity.backtest.DEFAULT_COST_BPS
# Frozen budget bump (contracts.py N_TRIALS==24 too, but called explicitly per
# the pre-reg's instruction not to rely on the module-level default silently).
N_TRIALS_EXPLICIT = 24

# Composite weights: only fundamental_quality carries the signal (frozen).
SIGNAL_WEIGHTS: Dict[str, float] = {
    "fundamental_quality": 1.0,
    "accounting_red_flags": 0.0,
    "forward_outlook": 0.0,
}


def _load_price_panel() -> pd.DataFrame:
    """Load the cached S&P price panel, restricted to tickers with COMPLETE
    coverage across the full sample window.

    ``sp500_prices.parquet`` is a raw 684-ticker cache spanning 2012-2026; it
    naturally has entry/exit gaps (recent IPOs like COIN/ABNB/DASH, and
    delistings/M&A/renames like TWX/EMC/old-DD/FB). ``run_portfolio_backtest``
    correctly REFUSES to silently forward-fill a NaN price
    (``src/equity/backtest.py`` ``_align_weights_prices``) rather than risk
    booking a trade against a fabricated stale price on a day our cache simply
    lacks coverage. The safe fix is therefore to DROP any ticker with even one
    NaN in-window, not to paper over the gap — this is a deliberate,
    documented survivorship restriction (see the pre-reg addendum / final
    report): the tested cross-section is "names with continuous price history
    2019-2026", which excludes names that IPO'd or delisted mid-window. This
    mirrors the existing equity-harvester's documented GFC-window survivorship
    caveat rather than introducing a new, undisclosed one.
    """
    prices = pd.read_parquet(PRICE_PANEL_PATH)
    prices = _normalize_prices(prices)
    lo = pd.Timestamp(SAMPLE_START, tz="UTC")
    hi = pd.Timestamp(SAMPLE_END, tz="UTC")
    prices = prices.loc[(prices.index >= lo) & (prices.index <= hi)]
    n_before = prices.shape[1]
    complete_cols = prices.columns[~prices.isna().any()]
    dropped = sorted(set(prices.columns) - set(complete_cols))
    prices = prices[complete_cols]
    print(
        f"[finra_sv] price panel survivorship restriction: {n_before} -> "
        f"{prices.shape[1]} tickers with complete {SAMPLE_START}->{SAMPLE_END} "
        f"coverage (dropped {len(dropped)}, mostly mid-window IPOs/delistings): "
        f"{dropped}"
    )
    return prices


def _compute_z_panel(svr: pd.DataFrame) -> pd.DataFrame:
    """Per-ticker trailing z-score of SVR, using ONLY that ticker's own history.

    z_t = (SVR_t - trailing_60_mean) / trailing_60_std, computed independently
    per column (ticker) — no cross-sectional peeking. ``min_periods`` equals
    the full window so a name's z-score is NaN (undefined) until it has 60
    trailing sessions of its OWN short-volume history, matching the pre-reg's
    "at least 60 trailing sessions" cross-section-eligibility rule.
    """
    trailing_mean = svr.rolling(Z_TRAILING_WINDOW, min_periods=Z_TRAILING_WINDOW).mean()
    trailing_std = svr.rolling(Z_TRAILING_WINDOW, min_periods=Z_TRAILING_WINDOW).std(ddof=0)
    z = (svr - trailing_mean) / trailing_std
    z = z.replace([np.inf, -np.inf], np.nan)
    return z


def _scores_from_composite_panel(
    composite_panel: pd.DataFrame, rationale: str
) -> List[ResearchScore]:
    """Turn a date x ticker composite-signal panel into ResearchScore rows.

    ``fundamental_quality`` carries the (sign-flipped, clipped) signal;
    ``accounting_red_flags``/``forward_outlook`` are 0 (unused), ``conviction``
    = 1.0 (uniform), matching the pre-reg's adapter spec exactly.
    """
    scores: List[ResearchScore] = []
    stacked = composite_panel.stack().dropna()
    for (as_of_ts, ticker), val in stacked.items():
        scores.append(
            ResearchScore(
                ticker=str(ticker),
                as_of=as_of_ts.date().isoformat(),
                fundamental_quality=float(np.clip(val, -1.0, 1.0)),
                accounting_red_flags=0.0,
                forward_outlook=0.0,
                conviction=1.0,
                rationale=rationale,
                spans_used=[],
            )
        )
    return scores


def _cross_section_stats(
    composite_panel: pd.DataFrame,
    prices: pd.DataFrame,
    rebalance_dates: List[pd.Timestamp],
) -> Dict[str, object]:
    """Min/median/max cross-section size actually achieved across rebalances.

    Mirrors the harness's own eligibility rule (priced on the day + a valid
    score that day). Reads directly off the dense composite panel's not-NaN
    mask intersected with the price panel's not-NaN mask — O(rebalances x
    tickers), vectorized — rather than re-deriving eligibility by scanning the
    per-ticker ResearchScore list (which is O(rebalances x tickers x
    scores_per_ticker) and, for a dense daily panel, intractable at this
    sample's scale; the earlier version of this helper hit exactly that wall).
    """
    common_cols = [c for c in composite_panel.columns if c in prices.columns]
    score_ok = composite_panel[common_cols].notna()
    price_ok = prices[common_cols].notna()
    eligible_mask = (score_ok & price_ok).reindex(rebalance_dates)
    sizes_series = eligible_mask.sum(axis=1)
    sizes = sizes_series.tolist()
    skipped_below_min = int((sizes_series < MIN_NAMES_FOR_QUINTILE).sum())
    if not sizes:
        return {"min": None, "median": None, "max": None, "n_rebalances": 0,
                "skipped_below_min_names": 0}
    return {
        "min": int(np.min(sizes)),
        "median": float(np.median(sizes)),
        "max": int(np.max(sizes)),
        "n_rebalances": len(sizes),
        "skipped_below_min_names": skipped_below_min,
    }


def run_hypothesis(
    label: str,
    composite_panel: pd.DataFrame,
    prices: pd.DataFrame,
    rebalance_step_days: int,
) -> Dict[str, object]:
    """Run one pre-registered hypothesis end-to-end through the frozen harness."""
    scores = _scores_from_composite_panel(composite_panel, rationale=f"finra_short_volume_{label}")
    if not scores:
        return {"label": label, "error": "no scores produced (empty composite panel)"}

    scores_by_ticker: Dict[str, List[ResearchScore]] = {}
    for s in scores:
        scores_by_ticker.setdefault(s.ticker, []).append(s)

    rb_dates = _rebalance_dates(prices.index, rebalance_step_days)
    cs_stats = _cross_section_stats(composite_panel, prices, rb_dates)

    try:
        panel = _build_weight_panel(
            scores_by_ticker,
            prices,
            rb_dates,
            SIGNAL_WEIGHTS,
            long_only=False,  # pre-reg: quintile LONG-SHORT (Q5 long / Q1 short)
            placebo=False,
        )
    except ResearchHarnessError as exc:
        return {"label": label, "error": str(exc), "cross_section": cs_stats}

    managed = _apply_overlay(panel, prices, vol_target=VOL_TARGET)
    cost_bps_kwargs = {} if COST_BPS is None else {"cost_bps": COST_BPS}
    try:
        bt = run_portfolio_backtest(managed, prices, execution_lag=1, **cost_bps_kwargs)
    except BacktestError as exc:
        return {"label": label, "error": str(exc), "cross_section": cs_stats}

    arm_returns = bt.returns
    report = _build_report(arm_returns)
    verdict = evaluate_gate(report)
    dsr_result = _dsr_oos_n22(arm_returns, n_trials=N_TRIALS_EXPLICIT)
    eff_n = _effective_n(managed, arm_returns.index)

    passes_bar = bool(
        isinstance(dsr_result.get("dsr"), float)
        and dsr_result["dsr"] >= 0.95
        and dsr_result.get("p_oos_bootstrap") is not None
        and dsr_result["p_oos_bootstrap"] < dsr_result["bonferroni_alpha"]
    )

    return {
        "label": label,
        "rebalance_step_days": rebalance_step_days,
        "n_rebalances": len(rb_dates),
        "cross_section": cs_stats,
        "span": f"{arm_returns.index.min().date()}->{arm_returns.index.max().date()}"
        if len(arm_returns) else None,
        "report": report,
        "gate": verdict_to_dict(verdict),
        "gate_passed": bool(verdict.passed),
        "dsr_oos_n_trials_24": dsr_result,
        "passes_significance_bar": passes_bar,
        "effective_n": eff_n,
        "n_scores": len(scores),
        "n_tickers_scored": len(scores_by_ticker),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--refresh", action="store_true",
        help="re-run scripts/fetch_finra_short_volume.py before scoring",
    )
    args = ap.parse_args()

    if args.refresh:
        from scripts.fetch_finra_short_volume import run as fetch_run
        fetch_run(SAMPLE_START, SAMPLE_END, None)

    prices = _load_price_panel()
    print(f"[finra_sv] price panel: {prices.shape[1]} tickers x {prices.shape[0]} days "
          f"{prices.index[0].date()} -> {prices.index[-1].date()}")

    svr = load_short_volume_ratio_panel(prices.columns, dates=prices.index)
    svr = svr.reindex(index=prices.index)
    print(f"[finra_sv] short-volume-ratio panel: {svr.shape[1]} tickers x {svr.shape[0]} days, "
          f"overall NaN rate {svr.isna().mean().mean():.3f}")

    z = _compute_z_panel(svr)

    # H1: quintile long-short on z_t directly.
    h1_composite = (-z).clip(-1.0, 1.0)
    h1 = run_hypothesis("h1", h1_composite, prices, H1_REBALANCE_STEP_DAYS)

    # H2: Delta z_t = z_t - z_{t-5}; same sign convention (rapid increase in
    # pressure -> predicted negative return -> short side; sign-flip for the
    # composite so higher composite = long).
    delta_z = z - z.shift(H2_DELTA_LAG)
    h2_composite = (-delta_z).clip(-1.0, 1.0)
    h2 = run_hypothesis("h2", h2_composite, prices, H2_REBALANCE_STEP_DAYS)

    print("\n=== FINRA SHORT-VOLUME PRE-REGISTERED VERDICT ===")
    for res in (h1, h2):
        label = res["label"]
        if "error" in res:
            print(f"[{label}] ERROR: {res['error']}")
            continue
        rep = res["report"]
        dsr = res["dsr_oos_n_trials_24"]
        print(f"[{label}] rebalance_step_days={res['rebalance_step_days']} "
              f"n_rebalances={res['n_rebalances']} span={res['span']}")
        print(f"  cross_section: min={res['cross_section']['min']} "
              f"median={res['cross_section']['median']} max={res['cross_section']['max']} "
              f"(skipped_below_min={res['cross_section']['skipped_below_min_names']})")
        print(f"  net_sharpe={rep['net_sharpe']:.3f} max_drawdown={rep['max_drawdown']:.3f} "
              f"positive_years={rep['positive_years']}/{rep['total_years']}")
        print(f"  dsr={dsr.get('dsr')} p_oos_bootstrap={dsr.get('p_oos_bootstrap')} "
              f"bonferroni_alpha={dsr.get('bonferroni_alpha')}")
        print(f"  passes_significance_bar={res['passes_significance_bar']} "
              f"gate_passed(mechanical)={res['gate_passed']} "
              f"gate_summary={res['gate']['summary']}")

    out = REPO_ROOT / "trained_data" / "backtests" / (
        f"finra_short_volume_{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}.json"
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "prereg": "docs/prereg-finra-short-volume-2026-07-04.md",
        "sample_window": f"{SAMPLE_START}->{SAMPLE_END}",
        "price_panel_span": f"{prices.index[0].date()}->{prices.index[-1].date()}",
        "h1": h1,
        "h2": h2,
    }
    tmp = out.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str))
    tmp.rename(out)
    print(f"\n[finra_sv] wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
