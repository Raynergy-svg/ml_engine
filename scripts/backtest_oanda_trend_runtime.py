#!/usr/bin/env python
"""Runtime-EXACT backtest of the OANDA practice trend lane (readiness step 3).

The 2026-07-18 readiness report found the practice runner and the validated
research construction disagree on the trend window: the runtime default is
SMA100 (``src/equity/oanda_trend.py:DEFAULT_SMA``) while the pre-registered
multi-asset trend work froze SMA200 (``multi_asset_trend.SMA_WINDOW`` /
``trend_sleeve.DEFAULT_SMA``). "Practice results cannot be assumed to
validate the research strategy. Backtest the exact live rule ... before
judging its practice P&L." This script does exactly that, offline, on the
repo's cached OANDA daily panels.

WHAT IS EXACT (verbatim reuse, nothing re-derived):
  * Signal + sizing = ``trend_sleeve_weights(close.ffill(), sma_window=W,
    step=1)`` — the PRECISE call ``oanda_trend.trend_targets`` makes (daily
    step, equal-weight on-set, long-or-flat, double shift(1) causal).
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

WHAT IS NOT MODELED (honest limits): order-level risk gates
(one-position gate, exposure fail-closed refusals), intraday fill timing,
financing/swap. Those bind occasionally at runtime; this backtest answers
the SIGNAL+SIZING question the report asked, not fill microstructure.

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
                 gross_leverage: float) -> Dict[str, object]:
    """One arm: the exact runtime construction at ``sma_window``."""
    targets = trend_sleeve_weights(panel.ffill(), sma_window=sma_window, step=1)
    applied = targets.shift(1).fillna(0.0)            # position enters the NEXT bar
    rets = panel.pct_change().reindex(columns=applied.columns).fillna(0.0)
    base = (applied * rets).sum(axis=1)               # unlevered, gross of costs
    turnover = (targets - targets.shift(1)).abs().sum(axis=1).fillna(0.0)

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
            "max_dd_unlevered": round(abs(_sig.max_drawdown(net1)), 5),
            "max_dd_levered": round(abs(_sig.max_drawdown(net)), 5),
            "deflated_sr_prob": None if np.isnan(dsr) else round(dsr, 4),
            "block_bootstrap_p": round(_sig.block_bootstrap_p(net1), 5),
            "positive_years": int((yearly > 0).sum()),
            "total_years": int(len(yearly)),
        }
    return out


def main() -> int:
    panel = load_cached_universe()
    lev = float(DEFAULT_GROSS_LEVERAGE)
    result = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "purpose": ("runtime-exact validation of the OANDA practice trend lane; "
                    "isolates the SMA100(runtime) vs SMA200(research) window mismatch "
                    "(readiness report 2026-07-18, step 3)"),
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
        "not_modeled": ["order-level risk gates", "intraday fill timing",
                        "financing/swap", "metals+CFD candidates (uncached)"],
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
