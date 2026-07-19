#!/usr/bin/env python
"""MODELED-RUNTIME backtest of the OANDA ATR-risk-sized trend lane (step 3, arm 3).

Closes the fork the 2026-07-19 audit left open: the signal-only backtest
rejected the equal-weight SMA signal spec but left "the actual
ATR-risk-sized, gate-constrained runtime ... unproven until it is modeled or
forward-tested." This models it, on the same cached daily OHLC panels,
driving the RUNTIME'S OWN functions verbatim wherever they are pure:

  * targets: ``trend_sleeve_weights(close, sma_window=100, step=1)`` — the
    exact ``oanda_trend.trend_targets`` call (on/off only; sizing below).
  * sizing: ``trend_risk_gates.risk_normalized_units`` (1% NAV risk per new
    position over the ATR(14)x2 stop distance, quote->USD converted by the
    runtime's own ``quote_to_home_rate``).
  * caps: the runtime's own ``leverage_cap_scale`` (3x NAV gross ceiling)
    then ``oanda_trend.margin_scale`` (50% NAV margin at 4% margin rate).
  * rule 1: one open ticket per instrument, no pyramiding.
  * rule 2: ``bucket_cap_gate`` + ``accumulate_bucket_risk`` — max 2R per
    currency-direction bucket, same-cycle accumulation included.
  * brackets: SL = entry - 2xATR(14), TP = entry + 4xATR(14)
    (``bracket_prices`` defaults), checked intra-bar on daily H/L with the
    repo's SL-FIRST convention; entries/exits fill at the NEXT bar's open.
  * drawdown rail: no NEW entries while NAV drawdown from peak >= 20%
    (``DEFAULT_DD_HARD``); existing positions keep their brackets and
    signal exits — the runtime's "stop adding risk" semantics.

ONE pre-specified configuration — the runtime's shipped defaults; no sweep.
Cumulative trial count for this OANDA validation campaign: N_TRIALS=3 (the
two signal arms already examined + this modeled-runtime arm; the
multiple-testing budget is never reset per the readiness protocol).

STATED APPROXIMATIONS (named, not hidden): daily bars (the runtime loops
hourly on daily candles — decisions once per completed daily bar here);
fills at next daily open with no slippage beyond the {0,1,2} bp/side cost
grid on traded notional; ATR at decision time prices the bracket (the
runtime re-prices at placement); no financing/swap; winner-management /
bracket-repair sub-rules not modeled; intra-bar SL-first is an assumption
where H/L straddle both brackets. NOT modeled from the runtime: transient
NAV-glitch refusals, tradable-instrument filtering (cache = the 10 FX
candidates; metals/CFDs uncached, stated).

Usage:
    python scripts/backtest_oanda_trend_atr_runtime.py
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
for _p in (str(REPO_ROOT), str(REPO_ROOT / "scripts")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import experiment_crypto_xs_signals as _sig                     # noqa: E402  frozen significance fns
from backtest_oanda_trend_runtime import load_cached_universe   # noqa: E402  same cached panel loader
from src.equity.oanda_trend import (                            # noqa: E402  runtime constants + rails
    DEFAULT_ATR_PERIOD,
    DEFAULT_ATR_SL_MULT,
    DEFAULT_ATR_TP_MULT,
    DEFAULT_DD_HARD,
    DEFAULT_GROSS_LEVERAGE,
    DEFAULT_SMA,
    margin_scale,
)
from src.equity.trend_risk_gates import (                       # noqa: E402  runtime sizing + gates
    DEFAULT_MAX_BUCKET_RISK_R,
    DEFAULT_RISK_PCT,
    accumulate_bucket_risk,
    bucket_cap_gate,
    compute_bucket_risk,
    leverage_cap_scale,
    quote_to_home_rate,
    risk_normalized_units,
)
from src.equity.trend_sleeve import trend_sleeve_weights        # noqa: E402  exact target rule

ANN = 252.0
OUT_DIR = REPO_ROOT / "trained_data" / "backtests"
COST_BPS_GRID = (0.0, 1.0, 2.0)
N_TRIALS = 3          # cumulative for the OANDA validation campaign — never reset
START_NAV = 100_000.0


def _ohlc_panels() -> Dict[str, pd.DataFrame]:
    from run_oanda_trend import CANDIDATE_INSTRUMENTS
    out: Dict[str, pd.DataFrame] = {}
    for inst in CANDIDATE_INSTRUMENTS:
        path = REPO_ROOT / "market_data" / "factor" / f"{inst}_D.csv"
        if not path.exists():
            continue
        df = pd.read_csv(path, parse_dates=["date"], index_col="date")
        if {"open", "high", "low", "close"} <= set(df.columns) and not df.empty:
            out[inst] = df[["open", "high", "low", "close"]]
    return out


def _atr_panel(ohlc: Dict[str, pd.DataFrame], close: pd.DataFrame) -> pd.DataFrame:
    """ATR(period) per instrument — same TR definition as ``compute_atr``
    (hi-lo vs |hi-prev_close| vs |lo-prev_close|), simple mean over the
    window, shifted nowhere (decision at t uses bars <= t)."""
    cols = {}
    for inst, df in ohlc.items():
        hi, lo, cl = df["high"], df["low"], df["close"]
        prev = cl.shift(1)
        tr = pd.concat([hi - lo, (hi - prev).abs(), (lo - prev).abs()], axis=1).max(axis=1)
        cols[inst] = tr.rolling(DEFAULT_ATR_PERIOD, min_periods=DEFAULT_ATR_PERIOD).mean()
    return pd.DataFrame(cols).reindex(close.index)


class _Position:
    __slots__ = ("units", "entry", "sl", "tp")

    def __init__(self, units: int, entry: float, sl: float, tp: float):
        self.units, self.entry, self.sl, self.tp = units, entry, sl, tp


def simulate(cost_bps: float, *, ohlc: Dict[str, pd.DataFrame],
             close: pd.DataFrame, atr: pd.DataFrame,
             targets: pd.DataFrame) -> Dict[str, object]:
    dates = list(close.index)
    nav, peak_nav = START_NAV, START_NAV
    positions: Dict[str, _Position] = {}
    navs: List[float] = []
    stats = {"entries": 0, "sl_exits": 0, "tp_exits": 0, "signal_exits": 0,
             "dd_halt_days": 0, "bucket_blocks": 0, "lev_scale_days": 0,
             "margin_scale_days": 0}
    pending_entries: Dict[str, int] = {}
    pending_exits: List[str] = []

    for i, d in enumerate(dates):
        # ---- execute yesterday's decisions at TODAY'S open ----------------
        last_open = {inst: float(ohlc[inst]["open"].loc[d])
                     for inst in ohlc if d in ohlc[inst].index}
        for inst in pending_exits:
            pos = positions.pop(inst, None)
            if pos is None or inst not in last_open:
                continue
            px = last_open[inst]
            q2h = quote_to_home_rate(inst, last_open) or 0.0
            nav += pos.units * (px - pos.entry) * q2h
            nav -= abs(pos.units) * px * q2h * (cost_bps / 1e4)
            stats["signal_exits"] += 1
        pending_exits = []
        for inst, units in pending_entries.items():
            if inst in positions or inst not in last_open or units <= 0:
                continue
            entry = last_open[inst]
            a = float(atr.loc[dates[i - 1], inst]) if i > 0 else float("nan")
            if not np.isfinite(a) or a <= 0:
                continue
            q2h = quote_to_home_rate(inst, last_open) or 0.0
            nav -= abs(units) * entry * q2h * (cost_bps / 1e4)
            positions[inst] = _Position(
                units, entry,
                sl=entry - DEFAULT_ATR_SL_MULT * a,
                tp=entry + DEFAULT_ATR_TP_MULT * a)
            stats["entries"] += 1
        pending_entries = {}

        # ---- intra-bar bracket exits (SL-FIRST when both straddled) --------
        for inst in list(positions):
            if d not in ohlc[inst].index:
                continue
            bar = ohlc[inst].loc[d]
            pos = positions[inst]
            exit_px: Optional[float] = None
            if float(bar["low"]) <= pos.sl:
                exit_px = pos.sl
                stats["sl_exits"] += 1
            elif float(bar["high"]) >= pos.tp:
                exit_px = pos.tp
                stats["tp_exits"] += 1
            if exit_px is not None:
                q2h = quote_to_home_rate(inst, last_open) or 0.0
                nav += pos.units * (exit_px - pos.entry) * q2h
                nav -= abs(pos.units) * exit_px * q2h * (cost_bps / 1e4)
                del positions[inst]

        # ---- mark NAV to today's close ------------------------------------
        last_close = {inst: float(ohlc[inst]["close"].loc[d])
                      for inst in ohlc if d in ohlc[inst].index}
        mtm = nav
        for inst, pos in positions.items():
            if inst in last_close:
                q2h = quote_to_home_rate(inst, last_close) or 0.0
                mtm += pos.units * (last_close[inst] - pos.entry) * q2h
        navs.append(mtm)
        peak_nav = max(peak_nav, mtm)

        # ---- decide for TOMORROW (runtime cycle after the complete bar) ----
        if i == len(dates) - 1:
            break
        row = targets.loc[d]
        on = {inst for inst in targets.columns if float(row.get(inst, 0.0)) > 0}
        pending_exits = [inst for inst in positions if inst not in on]

        dd_halted = peak_nav > 0 and (peak_nav - mtm) / peak_nav >= DEFAULT_DD_HARD
        if dd_halted:
            stats["dd_halt_days"] += 1
            continue                      # no NEW risk; exits/brackets still live

        risk_unit_home = mtm * DEFAULT_RISK_PCT
        want: Dict[str, int] = {}
        for inst in sorted(on):
            if inst in positions:         # rule 1: one ticket, no pyramiding
                continue
            a = float(atr.loc[d, inst]) if inst in atr.columns else float("nan")
            if not np.isfinite(a) or a <= 0:
                continue
            want[inst] = risk_normalized_units(
                instrument=inst, r_distance_quote=DEFAULT_ATR_SL_MULT * a,
                nav=mtm, risk_pct=DEFAULT_RISK_PCT, last_prices=last_close)
        # existing book counts toward the gross-leverage/margin ceilings
        held_units = {inst: p.units for inst, p in positions.items()}
        combined = {**held_units, **want}
        combined, lev_f = leverage_cap_scale(combined, last_close, mtm,
                                             gross_leverage=DEFAULT_GROSS_LEVERAGE)
        combined, mar_f = margin_scale(combined, last_close, mtm)
        if lev_f < 1.0:
            stats["lev_scale_days"] += 1
        if mar_f < 1.0:
            stats["margin_scale_days"] += 1
        want = {inst: u for inst, u in combined.items()
                if inst in want and u > 0}

        # rule 2: currency-bucket caps with same-cycle accumulation
        open_trades = [{"instrument": inst, "currentUnits": p.units, "id": inst}
                       for inst, p in positions.items()]
        r_dist_open = {inst: DEFAULT_ATR_SL_MULT * float(atr.loc[d, inst])
                       for inst in positions
                       if inst in atr.columns and np.isfinite(atr.loc[d, inst])}
        bucket_risk, _insts = compute_bucket_risk(
            open_trades, r_dist_open, last_close,
            r_distance_fallback_by_instrument=r_dist_open)
        for inst in sorted(want):
            r_dist = DEFAULT_ATR_SL_MULT * float(atr.loc[d, inst])
            allowed, _why = bucket_cap_gate(
                instrument=inst, candidate_units=want[inst],
                r_distance_quote=r_dist, last_prices=last_close,
                bucket_risk_home=bucket_risk, risk_unit_home=risk_unit_home,
                max_bucket_risk_r=DEFAULT_MAX_BUCKET_RISK_R)
            if not allowed:
                stats["bucket_blocks"] += 1
                continue
            q2h = quote_to_home_rate(inst, last_close) or 0.0
            accumulate_bucket_risk(bucket_risk, inst, want[inst] * r_dist * q2h)
            pending_entries[inst] = want[inst]

    nav_series = pd.Series(navs, index=dates[:len(navs)])
    rets = nav_series.pct_change().dropna()
    yearly = rets.groupby(rets.index.year).sum()
    eq = nav_series / START_NAV
    max_dd = float(abs((eq / eq.cummax() - 1.0).min()))
    sh = float(np.sqrt(ANN) * rets.mean() / rets.std(ddof=1)) if rets.std(ddof=1) > 0 else 0.0
    dsr, _ = _sig.deflated_sr(rets, n_trials=N_TRIALS)
    return {
        "net_sharpe": round(sh, 4),
        "ann_return": round(float(rets.mean() * ANN), 5),
        "ann_vol": round(float(rets.std(ddof=1) * np.sqrt(ANN)), 5),
        "max_drawdown": round(max_dd, 5),
        "final_nav_over_start": round(float(nav_series.iloc[-1] / START_NAV), 4),
        "deflated_sr_prob": None if np.isnan(dsr) else round(float(dsr), 4),
        "block_bootstrap_p": round(float(_sig.block_bootstrap_p(rets)), 5),
        "positive_years": int((yearly > 0).sum()),
        "total_years": int(len(yearly)),
        "n_days": int(len(rets)),
        "span": [str(rets.index[0].date()), str(rets.index[-1].date())],
        "activity": stats,
    }


def main() -> int:
    ohlc = _ohlc_panels()
    if not ohlc:
        raise SystemExit("no cached OHLC candidates — nothing to model")
    close = load_cached_universe()
    close = close[[c for c in close.columns if c in ohlc]]
    targets = trend_sleeve_weights(close.ffill(), sma_window=int(DEFAULT_SMA), step=1)
    atr = _atr_panel(ohlc, close)

    result = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "purpose": ("MODELED-RUNTIME validation: the ACTUAL ATR-risk-sized, "
                    "gate-constrained OANDA trend runtime (1% NAV risk / 2xATR stop, "
                    "4xATR TP, one ticket per instrument, 2R bucket caps, 3x gross "
                    "ceiling, 50% margin rail, 20% dd halt) driven through the "
                    "runtime's OWN sizing/cap functions on cached daily OHLC. "
                    "Closes the fork the signal-only verdict left open."),
        "scope": "modeled_runtime_daily",
        "runtime_defaults": {
            "sma_window": int(DEFAULT_SMA), "atr_period": DEFAULT_ATR_PERIOD,
            "atr_sl_mult": DEFAULT_ATR_SL_MULT, "atr_tp_mult": DEFAULT_ATR_TP_MULT,
            "risk_pct": DEFAULT_RISK_PCT, "gross_leverage": DEFAULT_GROSS_LEVERAGE,
            "max_bucket_risk_r": DEFAULT_MAX_BUCKET_RISK_R, "dd_hard": DEFAULT_DD_HARD,
        },
        "universe_cached": sorted(ohlc),
        "n_trials_cumulative": N_TRIALS,
        "cost_model": "turnover x bps/side on traded notional, fixed grid",
        "cost_arms": {},
        "approximations": ["daily bars (runtime loops on daily candles)",
                           "next-open fills, SL-first intra-bar",
                           "ATR at decision prices the bracket",
                           "no financing/swap; no winner-management sub-rules",
                           "metals/CFD candidates uncached (FX-only)"],
    }
    for cost in COST_BPS_GRID:
        result["cost_arms"][f"{cost:g}bps"] = simulate(
            cost, ohlc=ohlc, close=close, atr=atr, targets=targets)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_path = OUT_DIR / f"oanda_trend_atr_runtime_{ts}.json"
    out_path.write_text(json.dumps(result, indent=2, sort_keys=True))
    print(json.dumps(result["cost_arms"], indent=1, sort_keys=True))
    print(f"\nwritten: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
