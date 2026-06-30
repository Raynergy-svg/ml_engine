#!/usr/bin/env python3
"""Pre-2014 daily FX factor backtest — the one untested lever with NEW evidence.

The 2026-06-12 factor verdict (carry+trend, G10, 2014-2026) was a robust FAIL:
gross Sharpe ~=0 even at zero cost. But 2014-2026 is the QE-suppressed era where
carry and momentum are *known* to be weak. The honest open question (per
project_factor_pivot memory + the ceiling rule's "materially different setup")
is whether the SAME factor book shows edge in the pre-2014 / pre-GFC regime, when
rate differentials were wide and carry historically paid.

This reuses the EXACT src.factor signal/portfolio/backtest/carry/gate code — only
the PRICE SOURCE changes from OANDA (~2014+) to FRED daily H.10 series (1999+),
which map 1:1 onto the OANDA pair convention (no inversion):

    DEXUSEU/UK/AL/NZ = USD per unit  -> EUR_USD GBP_USD AUD_USD NZD_USD
    DEXJPUS/CAUS/SZUS = unit per USD  -> USD_JPY USD_CAD USD_CHF

Carry rates come from the existing build_rate_diff_panel (FRED IR3TIB monthly,
back to the 1990s). Majors-only (7 pairs) — every one is a direct FRED series.

Runs three windows for an apples-to-apples read:
    1999-2014  pre-QE / pre-GFC  (the untested window — the real test)
    1999-2026  full history
    2014-2026  control — should reproduce the known ~0 gross Sharpe FAIL

No trading is performed; research/evaluation only.

Usage:  python scripts/experiment_pre2014_factor.py [--refresh]
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

import pandas as pd  # noqa: E402

from src.factor import PAIRS, FACTOR_PIPELINE_VERSION  # noqa: E402
from src.factor.rates import build_rate_diff_panel  # noqa: E402
from src.factor.signals import tsmom_signal, carry_signal, combine_signals  # noqa: E402
from src.factor.portfolio import (  # noqa: E402
    target_weights,
    apply_no_trade_band,
    enforce_guards,
    weekly_rebalance_mask,
)
from src.factor.backtest import run_backtest  # noqa: E402
from src.factor.ship_gate import evaluate_gate, verdict_to_dict  # noqa: E402

# FRED daily H.10 spot series → OANDA pair (orientation already matches).
FRED_PX_IDS = {
    "EUR_USD": "DEXUSEU", "GBP_USD": "DEXUSUK", "AUD_USD": "DEXUSAL",
    "NZD_USD": "DEXUSNZ", "USD_JPY": "DEXJPUS", "USD_CAD": "DEXCAUS",
    "USD_CHF": "DEXSZUS",
}
PX_CACHE = REPO_ROOT / "market_data" / "factor" / "fred_daily"
WINDOWS = [("1999-01-01", "2014-01-01"),  # the untested pre-QE window
           ("2014-01-01", "2026-12-31"),  # control: reproduce the known FAIL
           ("2020-01-01", "2026-12-31")]  # recent / forward-relevant regime
BOOKS = ["carry+trend", "trend", "carry"]  # decompose which leg carries the edge


def fetch_fred_daily(pair: str, fid: str, refresh: bool) -> pd.Series:
    """Daily spot for one pair from FRED, tz-aware UTC index (matches rates panel)."""
    import requests
    PX_CACHE.mkdir(parents=True, exist_ok=True)
    cache = PX_CACHE / f"{pair}_{fid}.csv"
    if cache.exists() and not refresh:
        s = pd.read_csv(cache, index_col=0, parse_dates=True)["px"]
        s.index = pd.DatetimeIndex(s.index)
        if s.index.tz is None:
            s.index = s.index.tz_localize("UTC")
        return s.rename(pair)
    url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={fid}"
    last_exc = None
    for _ in range(3):
        try:
            r = requests.get(url, timeout=20)
            if r.status_code == 200 and len(r.text) > 100:
                df = pd.read_csv(StringIO(r.text))
                df.columns = ["date", "px"]
                df["date"] = pd.to_datetime(df["date"], utc=True)
                df["px"] = pd.to_numeric(df["px"], errors="coerce")
                s = df.dropna().set_index("date")["px"].sort_index().rename(pair)
                tmp = cache.with_suffix(".csv.tmp")
                s.to_frame("px").to_csv(tmp)
                tmp.rename(cache)
                return s
        except Exception as exc:  # network flake -> retry
            last_exc = exc
            time.sleep(1)
    raise RuntimeError(f"{pair}: FRED daily fetch failed for {fid}: {last_exc}")


def load_fred_close(refresh: bool) -> pd.DataFrame:
    """date×pair close panel from FRED, inner-joined on the common calendar."""
    series = {p: fetch_fred_daily(p, fid, refresh) for p, fid in FRED_PX_IDS.items()}
    close = pd.concat(series.values(), axis=1, join="inner").sort_index()
    return close.dropna(how="any")


def run_window(close_full: pd.DataFrame, start: str, end: str, refresh: bool,
               book: str = "carry+trend") -> dict:
    """book in {'carry+trend','trend','carry'} — decompose which leg carries the edge."""
    s = pd.Timestamp(start, tz="UTC")
    e = pd.Timestamp(end, tz="UTC")
    close = close_full.loc[(close_full.index >= s) & (close_full.index <= e)]
    returns = close.pct_change()

    rd = build_rate_diff_panel(close.index, pairs=list(close.columns), refresh=refresh)
    accrual = rd / 100.0 / 252.0
    sigs = []
    if book in ("carry+trend", "carry"):
        sigs.append(carry_signal(rd))
    if book in ("carry+trend", "trend"):
        sigs.append(tsmom_signal(close))
    combined = combine_signals(sigs)
    # carry accrual is only earned if we actually hold carry-driven positions; for a
    # pure trend book the held weights still earn whatever carry their positions carry.
    raw_w = target_weights(combined, returns)
    rb = weekly_rebalance_mask(close.index)
    held_w = enforce_guards(apply_no_trade_band(raw_w, band=0.05, rebalance_mask=rb))

    result = run_backtest(held_w, close, carry_accrual=accrual)
    report = json.loads(json.dumps(result.__dict__, default=lambda o: o))
    verdict = evaluate_gate(report)
    return {
        "window": f"{start[:4]}-{end[:4]}",
        "book": book,
        "span": f"{result.start} -> {result.end}",
        "n_days": result.n_days,
        "total_years": result.total_years,
        "gross_sharpe": round(result.gross_sharpe, 3),
        "net_sharpe": round(result.net_sharpe, 3),
        "net_cagr": round(result.net_cagr, 4),
        "max_drawdown": round(result.max_drawdown, 3),
        "positive_years": f"{result.positive_years}/{result.total_years}",
        "annual_turnover": round(result.annual_turnover, 1),
        "annual_cost_drag": round(result.annual_cost_drag, 4),
        "per_year_returns": result.per_year_returns,
        "gate_pass": verdict_to_dict(verdict).get("passed", False),
        "gate_summary": verdict.summary,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--refresh", action="store_true", help="force re-fetch from FRED")
    args = ap.parse_args()

    print(f"[pre2014] pipeline={FACTOR_PIPELINE_VERSION} source=FRED-daily "
          f"universe=majors ({len(PAIRS)})")
    close_full = load_fred_close(args.refresh)
    print(f"[pre2014] FRED panel: {close_full.shape[1]} pairs × {close_full.shape[0]} "
          f"days {close_full.index[0].date()} -> {close_full.index[-1].date()}")

    rows = [run_window(close_full, s, e, args.refresh, book)
            for s, e in WINDOWS for book in BOOKS]

    print("\n=== FACTOR DECOMPOSITION (majors, FRED daily) — net Sharpe by book × window ===")
    hdr = (f"{'window':10s} {'book':12s} {'span':25s} {'gross_Sh':>9s} "
           f"{'net_Sh':>7s} {'CAGR':>7s} {'maxDD':>6s} {'+yrs':>6s} {'turn':>6s} gate")
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        print(f"{r['window']:10s} {r['book']:12s} {r['span']:25s} "
              f"{r['gross_sharpe']:>9.3f} {r['net_sharpe']:>7.3f} "
              f"{r['net_cagr']:>+7.1%} {r['max_drawdown']:>6.1%} "
              f"{r['positive_years']:>6s} {r['annual_turnover']:>6.1f} "
              f"{'PASS' if r['gate_pass'] else 'FAIL'}")
    print("\nGate needs net Sharpe >= 0.40 AND maxDD <= 25%. Read: which leg (trend vs "
          "carry) survives into the recent 2020-2026 regime? Trend is price-only and "
          "more regime-robust; carry is the decayed crowded anomaly. A leg that is "
          "positive 1999-2014 AND 2020-2026 is the only forward-deployable candidate.")

    out = REPO_ROOT / "trained_data" / "backtests" / (
        f"pre2014_factor_{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {"pipeline": FACTOR_PIPELINE_VERSION, "source": "FRED-daily",
               "universe": PAIRS, "windows": rows,
               "panel_span": f"{close_full.index[0].date()}->{close_full.index[-1].date()}"}
    tmp = out.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True))
    tmp.rename(out)
    print(f"\n[pre2014] wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
