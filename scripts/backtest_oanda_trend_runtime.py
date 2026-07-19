#!/usr/bin/env python
"""SIGNAL-ONLY backtest of the OANDA trend lane's target rule (readiness step 3).

SCOPE CORRECTION (2026-07-19 audit): this is a SIGNAL-layer validation, NOT a
full runtime reproduction. What is verbatim-exact is the TARGET-WEIGHT rule
(``oanda_trend.trend_targets`` -> ``trend_sleeve_weights``: equal-weight the
on-set, long-or-flat, daily step, double shift(1)). What the actual runtime
does AFTER those targets — ATR-based risk-normalized unit sizing (gross
leverage is only a ceiling), drawdown halts, margin limits, one-position
gate, currency-bucket limits, cost gates, stop-loss/take-profit brackets and
position management — is NOT modeled here. A negative here rejects the
equal-weight SMA SIGNAL specification on this universe; it does NOT prove
the risk-gated practice runtime is expectancy-negative.

Background: the 2026-07-18 readiness report found the practice runner and the validated
research construction disagree on the trend window: the runtime default is
SMA100 (``src/equity/oanda_trend.py:DEFAULT_SMA``) while the pre-registered
multi-asset trend work froze SMA200 (``multi_asset_trend.SMA_WINDOW`` /
``trend_sleeve.DEFAULT_SMA``). This script isolates that window mismatch at
the SIGNAL layer, offline, on the repo's cached OANDA daily panels. (The
runtime's OWN risk machinery is modeled separately by
``backtest_oanda_trend_atr_runtime.py``.)

WHAT IS EXACT (verbatim reuse, nothing re-derived) — the SIGNAL layer only:
  * Target rule = ``trend_sleeve_weights(close.ffill(), sma_window=W,
    step=1)`` — the PRECISE call ``oanda_trend.trend_targets`` makes (daily
    step, equal-weight on-set, long-or-flat, double shift(1) causal).
    Sizing is NOT this — the runtime sizes with ATR risk-normalized units.
  * Universe = the runner's ``CANDIDATE_INSTRUMENTS`` intersected with the
    cached daily panels (``market_data/factor/{PAIR}_D.csv``). The cache
    covers the 10 FX candidates; metals/index/commodity CFDs are NOT cached
    and are therefore NOT modeled — stated in the artifact, never imputed.
  * Gross leverage = the runner's DEFAULT_GROSS_LEVERAGE (3.0), reported
    alongside unlevered (Sharpe is leverage-invariant; maxDD is not).

TWO PRE-SPECIFIED ARMS, ONE VARIABLE (no sweep, no tuning):
  * ARM runtime_sma100 — the rule exactly as the practice runner ships.
  * ARM research_sma200 — the SAME machinery with the pre-registered
    research window. This isolates the window mismatch and nothing else.

COST MODEL (stated assumption, three fixed stress points, no fitting):
  turnover x {0.0, 1.0, 2.0} bps per side. OANDA practice fills at real
  spread; ~1bp/side approximates majors' typical all-in cost, 2bp is the
  stress case, 0bp bounds the frictionless ceiling.

WHAT IS NOT MODELED (honest limits): ATR risk-normalized sizing, brackets,
order-level risk gates (one-position gate, exposure fail-closed refusals),
intraday fill timing, financing/swap. This backtest validates the TARGET
SIGNAL specification only — nothing more.

ACCOUNTING (2026-07-19 re-review items 5+6): costs are charged on the
turnover of the APPLIED weights (``applied.diff()`` — the same series that
earns the returns), so a target change decided at bar t-1 pays its cost on
bar t, exactly when the position actually changes. Max drawdown is the
compounded equity-curve drawdown with the peak starting at INITIAL CAPITAL
(1.0) — the same implementation the promotion gate uses.

Usage:
    python scripts/backtest_oanda_trend_runtime.py
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
for _p in (str(REPO_ROOT), str(REPO_ROOT / "scripts")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import experiment_crypto_xs_signals as _sig                     # noqa: E402  frozen DSR/bootstrap fns
from run_oanda_trend import CANDIDATE_INSTRUMENTS               # noqa: E402  the runner's own universe
from src.hedge.portfolio_promotion import _max_drawdown         # noqa: E402  compounded, initial-capital peak
from src.equity.oanda_trend import DEFAULT_GROSS_LEVERAGE, DEFAULT_SMA  # noqa: E402
from src.equity.multi_asset_trend import SMA_WINDOW as RESEARCH_SMA     # noqa: E402
from src.equity.trend_sleeve import trend_sleeve_weights        # noqa: E402  the exact runtime call

ANN = 252.0
FACTOR_DIR = REPO_ROOT / "market_data" / "factor"
OUT_DIR = REPO_ROOT / "trained_data" / "backtests"
COST_BPS_GRID = (0.0, 1.0, 2.0)
N_TRIALS = 2  # exactly the two pre-specified arms; no other config examined


def load_cached_universe() -> pd.DataFrame:
    """Close panel for the runner's candidates that have cached daily data."""
    cols: Dict[str, pd.Series] = {}
    missing: List[str] = []
    for inst in CANDIDATE_INSTRUMENTS:
        path = FACTOR_DIR / f"{inst}_D.csv"
        if not path.exists():
            missing.append(inst)
            continue
        df = pd.read_csv(path, parse_dates=["date"], index_col="date")
        if "close" in df.columns and not df.empty:
            cols[inst] = df["close"]
    if not cols:
        raise SystemExit("no cached candidate instruments — nothing to backtest")
    panel = pd.DataFrame(cols).sort_index()
    if panel.index.tz is None:
        panel.index = panel.index.tz_localize("UTC")
    panel.attrs["uncovered_candidates"] = missing
    return panel


def backtest_arm(panel: pd.DataFrame, sma_window: int,
                 gross_leverage: float, *, with_series: bool = False):
    """One arm: the exact runtime TARGET rule at ``sma_window`` (signal-only).
    ``with_series=True`` additionally returns the unlevered gross-of-cost
    daily series so tests can assert causality bar-by-bar."""
    targets = trend_sleeve_weights(panel.ffill(), sma_window=sma_window, step=1)
    applied = targets.shift(1).fillna(0.0)            # position enters the NEXT bar
    rets = panel.pct_change().reindex(columns=applied.columns).fillna(0.0)
    base = (applied * rets).sum(axis=1)               # unlevered, gross of costs
    # 2026-07-19 re-review item 5: turnover from the APPLIED weights — the
    # series that earns the returns. The previous unshifted-target turnover
    # charged each cost one bar BEFORE the position actually changed,
    # misaligning costs with returns.
    turnover = applied.diff().abs().sum(axis=1).fillna(0.0)

    # Warmup: drop the span before the signal can exist at all.
    start = panel.index[min(len(panel) - 1, sma_window + 1)]
    base, turnover = base.loc[start:], turnover.loc[start:]

    out: Dict[str, object] = {
        "sma_window": sma_window,
        "n_days": int(len(base)),
        "span": [str(base.index[0].date()), str(base.index[-1].date())],
        "avg_annual_turnover": float(turnover.mean() * ANN),
        "cost_arms": {},
    }
    for cost_bps in COST_BPS_GRID:
        net1 = base - turnover * (cost_bps / 1e4)     # unlevered net
        net = net1 * gross_leverage
        yearly = net.groupby(net.index.year).sum()
        sh = float(np.sqrt(ANN) * net1.mean() / net1.std(ddof=1)) if net1.std(ddof=1) > 0 else 0.0
        dsr, _ = _sig.deflated_sr(net1, n_trials=N_TRIALS)
        out["cost_arms"][f"{cost_bps:g}bps"] = {
            "net_sharpe": round(sh, 4),                       # leverage-invariant
            "ann_return_unlevered": round(float(net1.mean() * ANN), 5),
            "ann_return_levered": round(float(net.mean() * ANN), 5),
            "ann_vol_levered": round(float(net.std(ddof=1) * np.sqrt(ANN)), 5),
            # item 6: compounded drawdown, peak starts at INITIAL capital —
            # _sig.max_drawdown omitted drawdown from the starting equity.
            "max_dd_unlevered": round(_max_drawdown(net1.tolist()), 5),
            "max_dd_levered": round(_max_drawdown(net.tolist()), 5),
            "deflated_sr_prob": None if np.isnan(dsr) else round(dsr, 4),
            "block_bootstrap_p": round(_sig.block_bootstrap_p(net1), 5),
            "positive_years": int((yearly > 0).sum()),
            "total_years": int(len(yearly)),
        }
    if with_series:
        return out, base
    return out


def main() -> int:
    panel = load_cached_universe()
    lev = float(DEFAULT_GROSS_LEVERAGE)
    result = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "purpose": ("SIGNAL-ONLY validation of the OANDA trend lane's target rule; "
                    "isolates the SMA100(runtime) vs SMA200(research) window mismatch. "
                    "NOT a runtime reproduction: ATR risk-normalized sizing, brackets "
                    "and order-level gates are NOT modeled — a negative here rejects "
                    "the equal-weight SMA SIGNAL spec, not the risk-gated runtime "
                    "(readiness report 2026-07-18 step 3; scope corrected 2026-07-19)"),
        "scope": "signal_only",
        "construction": "trend_sleeve_weights(close.ffill(), sma_window=W, step=1) — verbatim runtime call",
        "universe_cached": sorted(panel.columns),
        "universe_uncovered": panel.attrs["uncovered_candidates"],
        "gross_leverage": lev,
        "cost_model": "turnover x bps/side, fixed grid — stated assumption, not fitted",
        "n_trials_declared": N_TRIALS,
        "arms": {
            "runtime_sma100": backtest_arm(panel, int(DEFAULT_SMA), lev),
            "research_sma200": backtest_arm(panel, int(RESEARCH_SMA), lev),
        },
        "not_modeled": ["ATR risk-normalized unit sizing (runtime's actual sizer; "
                        "gross leverage is only a ceiling there)",
                        "stop-loss/take-profit brackets + position management",
                        "drawdown halts, margin limits, one-position gate,",
                        "currency-bucket limits, cost gates",
                        "order-level fill timing", "financing/swap",
                        "metals+CFD candidates (uncached)"],
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_path = OUT_DIR / f"oanda_trend_runtime_{ts}.json"
    out_path.write_text(json.dumps(result, indent=2, sort_keys=True))
    print(json.dumps(result["arms"], indent=1, sort_keys=True))
    print(f"\nwritten: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
