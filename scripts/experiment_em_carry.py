#!/usr/bin/env python3
"""EM-carry universe — the last genuinely-new lever in the edge search.

The majors factor book is dead post-QE (carry +0.09 noise, trend negative — see
project_pre2014_factor_verdict). The one untested extension is EM carry, where
rate differentials are far larger (MXN/BRL/ZAR/INR yields >> USD). The thesis:
a long-high-carry / short-low-carry book across EM+G10 might pay where majors-only
doesn't. The well-documented counter-thesis: EM carry has catastrophic negative
skew (the "carry crash"), so high gross Sharpe can still fail on drawdown.

Honest construction (reuses src.factor signal/portfolio/backtest/gate unchanged):
  - Prices: FRED daily H.10 (1995+). EM are local-per-USD = USD_XXX convention
        USD_MXN DEXMXUS | USD_BRL DEXBZUS | USD_ZAR DEXSFUS | USD_INR DEXINUS | USD_KRW DEXKOUS
    G10 reuse the same DEX series as experiment_pre2014_factor.
  - Rates: ONE consistent series type for the whole universe — IRSTCI01 call money
    (G10 + all 5 EM all present on FRED), monthly, lagged 1 month, ffill to daily.
    (The existing carry code uses IR3TIB which only covers G10; using one type
    across the universe is required for an honest cross-sectional rank.)
  - EM costs: real EM spreads are ~10-50x a major's. The fixed-pip cost model
    can't express that, so EM spreads are set to target ~15 bps via representative
    price; a cost-sensitivity line (0/15/30 bps) bounds the net. The DECISIVE read
    is GROSS Sharpe + max drawdown + worst year — EM carry dies on the crash, not
    the spread.

Universes: G10 (7) | EM (5 vs USD) | GLOBAL (12). Books: carry, carry+trend.
Windows: full | 2014-2026 | 2020-2026.

Usage:  python scripts/experiment_em_carry.py [--refresh]
Research/evaluation only — no trading.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from io import StringIO
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src.factor import FACTOR_PIPELINE_VERSION  # noqa: E402
from src.factor.signals import tsmom_signal, carry_signal, combine_signals  # noqa: E402
from src.factor.portfolio import (  # noqa: E402
    target_weights, apply_no_trade_band, enforce_guards, weekly_rebalance_mask,
)
from src.factor.backtest import (  # noqa: E402
    run_backtest, DEFAULT_SPREADS_PIPS, _sharpe,
)
from src.factor.ship_gate import evaluate_gate, verdict_to_dict  # noqa: E402

# pair -> FRED daily FX series (orientation already matches OANDA pair convention).
FX_IDS = {
    "EUR_USD": "DEXUSEU", "GBP_USD": "DEXUSUK", "AUD_USD": "DEXUSAL",
    "NZD_USD": "DEXUSNZ", "USD_JPY": "DEXJPUS", "USD_CAD": "DEXCAUS",
    "USD_CHF": "DEXSZUS",
    "USD_MXN": "DEXMXUS", "USD_BRL": "DEXBZUS", "USD_ZAR": "DEXSFUS",
    "USD_INR": "DEXINUS", "USD_KRW": "DEXKOUS",
}
# currency -> IRSTCI01 call-money rate (one consistent series type across universe).
RATE_IDS = {
    "USD": "IRSTCI01USM156N", "EUR": "IRSTCI01EZM156N", "JPY": "IRSTCI01JPM156N",
    "GBP": "IRSTCI01GBM156N", "AUD": "IRSTCI01AUM156N", "NZD": "IRSTCI01NZM156N",
    "CAD": "IRSTCI01CAM156N", "CHF": "IRSTCI01CHM156N", "MXN": "IRSTCI01MXM156N",
    "BRL": "IRSTCI01BRM156N", "ZAR": "IRSTCI01ZAM156N", "INR": "IRSTCI01INM156N",
    "KRW": "IRSTCI01KRM156N",
}
G10 = ["EUR_USD", "USD_JPY", "GBP_USD", "AUD_USD", "NZD_USD", "USD_CAD", "USD_CHF"]
EM = ["USD_MXN", "USD_BRL", "USD_ZAR", "USD_INR", "USD_KRW"]
GLOBAL = G10 + EM
EM_TARGET_SPREAD_BPS = 15.0  # realistic EM full spread for the net estimate
CACHE = REPO_ROOT / "market_data" / "factor" / "fred_daily"
RATE_CACHE = REPO_ROOT / "market_data" / "factor" / "fred_irstci"


def _fetch_fred(fid: str) -> pd.Series:
    import requests
    last = None
    for _ in range(3):
        try:
            r = requests.get(f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={fid}",
                             timeout=25)
            if r.status_code == 200 and len(r.text) > 100:
                df = pd.read_csv(StringIO(r.text))
                df.columns = ["date", "v"]
                df["date"] = pd.to_datetime(df["date"], utc=True)
                df["v"] = pd.to_numeric(df["v"], errors="coerce")
                return df.dropna().set_index("date")["v"].sort_index()
        except Exception as exc:
            last = exc
            time.sleep(1)
    raise RuntimeError(f"FRED fetch failed for {fid}: {last}")


def _cached(fid: str, cache_dir: Path, refresh: bool) -> pd.Series:
    cache_dir.mkdir(parents=True, exist_ok=True)
    p = cache_dir / f"{fid}.csv"
    if p.exists() and not refresh:
        s = pd.read_csv(p, index_col=0, parse_dates=True)["v"]
        s.index = pd.DatetimeIndex(s.index)
        if s.index.tz is None:
            s.index = s.index.tz_localize("UTC")
        return s
    s = _fetch_fred(fid)
    tmp = p.with_suffix(".csv.tmp")
    s.to_frame("v").to_csv(tmp)
    tmp.rename(p)
    return s


def load_prices(pairs, refresh: bool) -> pd.DataFrame:
    series = {p: _cached(FX_IDS[p], CACHE, refresh).rename(p) for p in pairs}
    close = pd.concat(series.values(), axis=1, join="inner").sort_index()
    return close.dropna(how="any")


def build_rate_diff(daily_index, pairs, refresh: bool) -> pd.DataFrame:
    """date×pair base-minus-quote call-money differential (%), lagged 1 month."""
    ccys = sorted({c for p in pairs for c in p.split("_")})
    rates = {}
    for c in ccys:
        s = _cached(RATE_IDS[c], RATE_CACHE, refresh).shift(1)  # publication lag
        rates[c] = s
    out = {}
    for p in pairs:
        base, quote = p.split("_")
        diff_m = (rates[base] - rates[quote]).dropna()
        daily = diff_m.reindex(diff_m.index.union(daily_index)).sort_index()
        out[p] = daily.ffill().reindex(daily_index).rename(p)
    return pd.concat(out.values(), axis=1)


def em_spreads(close: pd.DataFrame) -> dict:
    """Spreads dict: G10 defaults + EM set to ~EM_TARGET_SPREAD_BPS via median price."""
    spreads = dict(DEFAULT_SPREADS_PIPS)
    for p in EM:
        if p in close.columns:
            repr_px = float(close[p].median())
            # full_spread_frac = pips*1e-4/price ; solve pips for target bps
            spreads[p] = (EM_TARGET_SPREAD_BPS / 1e4) * repr_px / 1e-4
    return spreads


def manual_net_sharpe(gross_daily: pd.Series, held_w: pd.DataFrame,
                      close: pd.DataFrame, full_spread_bps: float) -> float:
    """Transparent cost model independent of the pip kludge: charge full_spread_bps
    of notional on each pair's daily weight change. Bounds EM net robustly."""
    dw = held_w.shift(1).fillna(0.0).diff().abs().fillna(held_w.shift(1).abs())
    cost = dw.sum(axis=1) * (full_spread_bps / 1e4)
    net = (gross_daily - cost).to_numpy()
    return _sharpe(net)


def run_case(close_full, rd_full, pairs, book, start, end) -> dict:
    s = pd.Timestamp(start, tz="UTC"); e = pd.Timestamp(end, tz="UTC")
    cols = [p for p in pairs if p in close_full.columns]
    close = close_full[cols].loc[(close_full.index >= s) & (close_full.index <= e)].dropna(how="any")
    if len(close) < 252:
        return {"universe": _uname(pairs), "book": book, "window": f"{start[:4]}-{end[:4]}",
                "n_days": len(close), "skip": "insufficient history"}
    rd = rd_full[cols].reindex(close.index)
    returns = close.pct_change()
    accrual = rd / 100.0 / 252.0
    sigs = []
    if book in ("carry+trend", "carry"):
        sigs.append(carry_signal(rd))
    if book in ("carry+trend", "trend"):
        sigs.append(tsmom_signal(close))
    combined = combine_signals(sigs)
    raw_w = target_weights(combined, returns)
    rb = weekly_rebalance_mask(close.index)
    held_w = enforce_guards(apply_no_trade_band(raw_w, band=0.05, rebalance_mask=rb))

    result = run_backtest(held_w, close, carry_accrual=accrual, spreads_pips=em_spreads(close))
    # gross daily for the transparent cost sensitivity
    w_eff = held_w.shift(1).fillna(0.0)
    gross_daily = (w_eff * returns.reindex(columns=close.columns)).sum(axis=1) \
        + (w_eff * accrual.reindex(columns=close.columns).fillna(0.0)).sum(axis=1)
    gross_daily = gross_daily.fillna(0.0)
    net0 = _sharpe(gross_daily.to_numpy())
    net15 = manual_net_sharpe(gross_daily, held_w, close, 15.0)
    net30 = manual_net_sharpe(gross_daily, held_w, close, 30.0)
    worst_year = min(result.per_year_returns.values()) if result.per_year_returns else 0.0
    report = json.loads(json.dumps(result.__dict__, default=lambda o: o))
    verdict = evaluate_gate(report)
    return {
        "universe": _uname(pairs), "book": book, "window": f"{start[:4]}-{end[:4]}",
        "span": f"{result.start}->{result.end}", "n_days": result.n_days,
        "gross_sharpe": round(result.gross_sharpe, 3),
        "net_sharpe": round(result.net_sharpe, 3),
        "net_sh_0bps": round(net0, 3), "net_sh_15bps": round(net15, 3),
        "net_sh_30bps": round(net30, 3),
        "max_drawdown": round(result.max_drawdown, 3),
        "worst_year": round(worst_year, 4),
        "positive_years": f"{result.positive_years}/{result.total_years}",
        "annual_turnover": round(result.annual_turnover, 1),
        "gate_pass": verdict_to_dict(verdict).get("passed", False),
        "per_year_returns": result.per_year_returns,
    }


def _uname(pairs) -> str:
    if pairs is G10:
        return "G10"
    if pairs is EM:
        return "EM"
    return "GLOBAL"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--refresh", action="store_true")
    args = ap.parse_args()
    print(f"[em_carry] pipeline={FACTOR_PIPELINE_VERSION} source=FRED rate=IRSTCI01-callmoney")
    close_full = load_prices(GLOBAL, args.refresh)
    rd_full = build_rate_diff(close_full.index, GLOBAL, args.refresh)
    print(f"[em_carry] panel: {close_full.shape[1]} pairs × {close_full.shape[0]} days "
          f"{close_full.index[0].date()}->{close_full.index[-1].date()}")
    print(f"[em_carry] EM spreads (pips, ~{EM_TARGET_SPREAD_BPS:.0f}bps): "
          + ", ".join(f"{p}={em_spreads(close_full)[p]:.0f}" for p in EM))

    cases = []
    for pairs in (EM, GLOBAL, G10):
        for book in ("carry", "carry+trend"):
            for start, end in (("1996-01-01", "2026-12-31"),
                               ("2014-01-01", "2026-12-31"),
                               ("2020-01-01", "2026-12-31")):
                cases.append(run_case(close_full, rd_full, pairs, book, start, end))

    print("\n=== EM / GLOBAL CARRY (FRED daily, IRSTCI01 rates) ===")
    hdr = (f"{'univ':7s} {'book':12s} {'window':10s} {'gross':>6s} {'net':>6s} "
           f"{'n0bps':>6s} {'n15':>6s} {'n30':>6s} {'maxDD':>6s} {'worstYr':>8s} {'+yrs':>6s} gate")
    print(hdr); print("-" * len(hdr))
    for c in cases:
        if c.get("skip"):
            print(f"{c['universe']:7s} {c['book']:12s} {c['window']:10s}  -- {c['skip']}")
            continue
        print(f"{c['universe']:7s} {c['book']:12s} {c['window']:10s} "
              f"{c['gross_sharpe']:>6.2f} {c['net_sharpe']:>6.2f} {c['net_sh_0bps']:>6.2f} "
              f"{c['net_sh_15bps']:>6.2f} {c['net_sh_30bps']:>6.2f} {c['max_drawdown']:>6.1%} "
              f"{c['worst_year']:>+8.1%} {c['positive_years']:>6s} "
              f"{'PASS' if c['gate_pass'] else 'FAIL'}")
    print("\nGate: net Sharpe>=0.40 AND maxDD<=25%. EM-carry read: even if GROSS clears "
          "0.4, a maxDD>40% or a -30%+ worst year = the carry crash → not deployable. "
          "n15/n30 = net Sharpe at 15/30bps full EM spread (transparent cost sensitivity).")

    out = REPO_ROOT / "trained_data" / "backtests" / (
        f"em_carry_{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".json.tmp")
    tmp.write_text(json.dumps({"pipeline": FACTOR_PIPELINE_VERSION, "rate": "IRSTCI01",
                               "cases": cases}, indent=2, sort_keys=True))
    tmp.rename(out)
    print(f"\n[em_carry] wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
