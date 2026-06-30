#!/usr/bin/env python3
"""PEAD event study — the one fundamental-ALPHA test the free tier supports.

Post-earnings-announcement drift: stocks that beat earnings keep drifting up, misses
keep drifting down, for weeks after the announcement. Most replicated anomaly in finance.
It's an EVENT study — statistical power comes from the NUMBER of earnings events, not the
calendar span — so a 1-year window across a large universe can give a genuine (if single-
regime) read, where a value/quality time-series backtest cannot.

Data: financialdatasets.ai (free tier, 2025-06-18+). Surprise = EPS actual-vs-estimate
(`signals` EPS_BEAT/MISS surprise_pct). Point-in-time clean: ENTER at the close of the
first trading day STRICTLY AFTER filing_date (no look-ahead on the announcement). Forward
ABNORMAL return = stock return minus equal-weight universe return over the same window
(strips market beta).

HONEST LIMIT: ~1 year = one regime (2025-2026). Suggestive, not validated. Confirming PEAD
needs multi-year history = the Pro tier. Usage: python scripts/experiment_pead.py
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import requests  # noqa: E402

for line in (REPO_ROOT / ".env.local").read_text().splitlines():
    if line.startswith("FINANCIAL_DATASETS_API_KEY="):
        os.environ["FINANCIAL_DATASETS_API_KEY"] = line.split("=", 1)[1].strip()
H = {"X-API-KEY": os.environ["FINANCIAL_DATASETS_API_KEY"]}
B = "https://api.financialdatasets.ai"
CACHE = REPO_ROOT / "market_data" / "equity"
START, END = "2025-06-18", "2026-06-01"
HORIZONS = [5, 21, 42]   # trading-day drift windows

# ~80 liquid large caps across sectors (current names; 1-yr window => survivorship immaterial)
UNIVERSE = ("AAPL MSFT NVDA AMZN GOOGL META AVGO TSLA AMD ORCL CRM ADBE CSCO INTC QCOM TXN "
            "IBM NOW INTU AMAT MU JPM BAC WFC GS MS C AXP BLK SCHW V MA PYPL "
            "UNH JNJ LLY PFE MRK ABBV TMO ABT DHR BMY AMGN CVS "
            "WMT PG KO PEP COST MCD NKE SBUX TGT LOW HD DIS "
            "XOM CVX COP SLB EOG BA CAT GE HON UNP LMT RTX MMM "
            "T VZ CMCSA NEE DUK SO LIN").split()


def _get(path, **params):
    for _ in range(3):
        try:
            r = requests.get(f"{B}{path}", headers=H, params=params, timeout=25)
            if r.status_code == 200:
                return r.json()
            if r.status_code in (402, 403, 429):
                time.sleep(1.5)
        except Exception:
            time.sleep(1.0)
    return {}


def prices(ticker) -> pd.Series:
    CACHE.mkdir(parents=True, exist_ok=True)
    f = CACHE / f"{ticker}_px.csv"
    if f.exists():
        s = pd.read_csv(f, index_col=0, parse_dates=True)["close"]
        return s
    j = _get("/prices/", ticker=ticker, interval="day", interval_multiplier=1,
             start_date=START, end_date=END)
    rows = j.get("prices", [])
    if not rows:
        return pd.Series(dtype=float)
    df = pd.DataFrame(rows)
    df["t"] = pd.to_datetime(df["time"]).dt.tz_localize(None).dt.normalize()
    s = df.set_index("t")["close"].astype(float).sort_index()
    s.to_frame("close").to_csv(f)
    return s


def surprises(ticker):
    """List of (filing_date, surprise_pct) for EPS beats/misses."""
    out = []
    for e in _get("/earnings/", ticker=ticker, limit=8).get("earnings", []):
        fd = e.get("filing_date")
        for sig in (e.get("signals") or []):
            d = sig.get("details") or {}
            if d.get("metric") == "earnings_per_share" or "surprise_pct" in d or sig.get("metric") == "earnings_per_share":
                sp = d.get("surprise_pct")
                if sp is not None and fd:
                    out.append((pd.Timestamp(fd).normalize(), float(sp)))
                break
    return out


def main() -> int:
    t0 = time.time()
    px = {}
    for i, tk in enumerate(UNIVERSE):
        s = prices(tk)
        if len(s) > 150:
            px[tk] = s
        if i % 20 == 0:
            time.sleep(0.3)
    print(f"[pead] prices loaded for {len(px)}/{len(UNIVERSE)} tickers")
    panel = pd.DataFrame(px).sort_index()
    mkt = panel.pct_change()                      # equal-weight universe daily returns
    cal = panel.index

    events = []
    for tk in px:
        for fdate, sp in surprises(tk):
            # entry = first trading day strictly AFTER the filing date (no look-ahead)
            after = cal[cal > fdate]
            if len(after) == 0:
                continue
            e_idx = panel.index.get_loc(after[0])
            for hz in HORIZONS:
                if e_idx + hz >= len(cal):
                    continue
                p0, p1 = panel[tk].iloc[e_idx], panel[tk].iloc[e_idx + hz]
                if not (p0 > 0 and p1 > 0):
                    continue
                stock_ret = p1 / p0 - 1
                bench_ret = mkt.iloc[e_idx + 1: e_idx + hz + 1].mean(axis=1).add(1).prod() - 1
                events.append({"ticker": tk, "surprise": sp, "hz": hz,
                               "abn_ret": stock_ret - bench_ret})
    df = pd.DataFrame(events)
    print(f"[pead] {len(df)//len(HORIZONS)} earnings events × {len(HORIZONS)} horizons\n")

    print("=== PEAD — abnormal drift by EPS-surprise quintile (point-in-time, 1yr regime) ===")
    out = {"n_events_per_hz": int(len(df) / len(HORIZONS)), "horizons": {}}
    for hz in HORIZONS:
        d = df[df.hz == hz].copy()
        if len(d) < 25:
            continue
        d["q"] = pd.qcut(d["surprise"].rank(method="first"), 5, labels=[1, 2, 3, 4, 5])
        by = d.groupby("q", observed=True)["abn_ret"].mean()
        ls = float(by.get(5, np.nan) - by.get(1, np.nan))   # top-minus-bottom surprise
        # crude IR: long-short mean / std across events, annualized by overlapping horizon
        topbot = d[d.q.isin([1, 5])].copy()
        topbot["sgn"] = np.where(topbot.q == 5, 1.0, -1.0) * topbot["abn_ret"]
        ir = float(topbot["sgn"].mean() / (topbot["sgn"].std() + 1e-12) * np.sqrt(252 / hz))
        print(f" +{hz:2d}d  Q1(miss)={by.get(1,float('nan')):+.3%}  Q3={by.get(3,float('nan')):+.3%}  "
              f"Q5(beat)={by.get(5,float('nan')):+.3%}  | long-short={ls:+.3%}  IR≈{ir:+.2f}")
        out["horizons"][hz] = {"by_quintile": {int(k): round(float(v), 5) for k, v in by.items()},
                               "long_short_abn": round(ls, 5), "approx_IR": round(ir, 3),
                               "n": int(len(d))}
    print("\nREAD: monotonic Q1<...<Q5 and positive long-short = PEAD present. IR is a CRUDE "
          "single-regime estimate (overlapping windows, 1yr) — suggestive only. A real verdict "
          "needs multi-year history (Pro tier) + non-overlapping/walk-forward design.")
    p = REPO_ROOT / "trained_data" / "backtests" / (
        f"pead_{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}.json")
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(".json.tmp"); tmp.write_text(json.dumps(out, indent=2, sort_keys=True))
    tmp.rename(p)
    print(f"\nwrote {p}  ({round(time.time()-t0,1)}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
