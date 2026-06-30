#!/usr/bin/env python3
"""PEAD on FREE point-in-time SEC data — the last distinct ALPHA lever.

Post-earnings-announcement drift is a BEHAVIORAL anomaly (underreaction to earnings news),
distinct from value/quality valuation factors and historically the most robust. Now testable
FREE: reuses cached SEC income (diluted_eps, point-in-time via filing_date) from
experiment_fundamental_factor.py. Surprise = SUE (seasonal random walk): (EPS_q - EPS_{q-4})
standardized by the trailing std of that difference. Entry = first trading day AFTER filing_date.

Two reads: (1) event study — abnormal drift by SUE quintile; (2) tradeable long-high/short-low
SUE portfolio (hold 21d post-filing), cost-aware, ship gate. yfinance daily prices.
Usage: python scripts/experiment_pead_sec.py
"""
from __future__ import annotations

import warnings; warnings.filterwarnings("ignore")
import json, sys, time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

SEC = REPO_ROOT / "market_data" / "equity" / "sec"
ANN, COST, HOLD = 252, 0.0010, 21
GATE_SHARPE, GATE_MAXDD = 0.40, 0.25


def sue_events():
    """List of (ticker, filing_date, SUE) from cached SEC income+filings (point-in-time)."""
    ev = []
    for inc_f in sorted(SEC.glob("*_inc.parquet")):
        tk = inc_f.name.replace("_inc.parquet", "")
        fil_f = SEC / f"{tk}_fil.parquet"
        if not fil_f.exists():
            continue
        I = pd.read_parquet(inc_f); F = pd.read_parquet(fil_f)
        epc = "diluted_eps" if "diluted_eps" in I.columns else ("basic_eps" if "basic_eps" in I.columns else None)
        if epc is None or "period_ending" not in I.columns:
            continue
        I = I[["period_ending", epc]].copy(); I["pe"] = pd.to_datetime(I["period_ending"])
        I = I.dropna(subset=[epc]).drop_duplicates("pe").sort_values("pe")
        if len(I) < 12:
            continue
        eps = I.set_index("pe")[epc].astype(float)
        diff = eps - eps.shift(4)                       # seasonal random walk surprise
        sue = diff / diff.rolling(8).std()
        F = F.copy(); F["report_date"] = pd.to_datetime(F["report_date"]); F["filing_date"] = pd.to_datetime(F["filing_date"])
        fmap = F.dropna(subset=["report_date", "filing_date"]).groupby("report_date")["filing_date"].min()
        for pe, s in sue.dropna().items():
            fd = fmap.get(pe, pe + pd.Timedelta(days=50))
            ev.append((tk, pd.Timestamp(fd), float(s)))
    return ev


def main():
    t0 = time.time()
    ev = sue_events()
    tickers = sorted({e[0] for e in ev})
    print(f"[pead-sec] {len(ev)} SUE events across {len(tickers)} tickers")
    import yfinance as yf
    px = yf.download(tickers, start="2010-06-01", end="2026-06-01", interval="1d", progress=False, auto_adjust=True)["Close"]
    px = px.dropna(axis=1, how="all"); cal = px.index
    mkt = px.pct_change().mean(axis=1)                  # equal-weight universe daily return

    rows = []
    for tk, fd, sue in ev:
        if tk not in px.columns:
            continue
        after = cal[cal > fd]
        if len(after) == 0:
            continue
        i = px.index.get_loc(after[0])
        for hz in (21, 42, 63):
            if i + hz >= len(cal):
                continue
            p0, p1 = px[tk].iloc[i], px[tk].iloc[i + hz]
            if not (p0 > 0 and p1 > 0):
                continue
            bench = mkt.iloc[i + 1:i + hz + 1].add(1).prod() - 1
            rows.append({"tk": tk, "sue": sue, "hz": hz, "abn": (p1 / p0 - 1) - bench, "entry": cal[i]})
    d = pd.DataFrame(rows)
    print(f"[pead-sec] {len(d)//3} usable events × 3 horizons\n")

    print("=== PEAD event study — abnormal drift by SUE quintile (point-in-time SEC) ===")
    out = {"n_events": int(len(d) / 3), "n_tickers": len(tickers), "horizons": {}}
    for hz in (21, 42, 63):
        dd = d[d.hz == hz].copy()
        if len(dd) < 50:
            continue
        dd["q"] = pd.qcut(dd["sue"].rank(method="first"), 5, labels=[1, 2, 3, 4, 5])
        by = dd.groupby("q", observed=True)["abn"].mean()
        ls = float(by.get(5, np.nan) - by.get(1, np.nan))
        tb = dd[dd.q.isin([1, 5])].copy(); tb["v"] = np.where(tb.q == 5, 1, -1) * tb["abn"]
        ir = float(tb["v"].mean() / (tb["v"].std() + 1e-12) * np.sqrt(ANN / hz))
        print(f" +{hz}d  Q1(miss)={by.get(1,float('nan')):+.3%}  Q3={by.get(3,float('nan')):+.3%}  "
              f"Q5(beat)={by.get(5,float('nan')):+.3%}  | L/S={ls:+.3%}  IR≈{ir:+.2f}")
        out["horizons"][hz] = {"by_q": {int(k): round(float(v), 5) for k, v in by.items()},
                               "long_short_abn": round(ls, 5), "approx_IR": round(ir, 3), "n": int(len(dd))}

    # Tradeable: daily long top-SUE-quintile / short bottom, active for HOLD days post-filing, vol-target.
    thr_hi, thr_lo = np.quantile([e[2] for e in ev], [0.8, 0.2])
    longs = pd.DataFrame(0.0, index=cal, columns=px.columns)
    shorts = pd.DataFrame(0.0, index=cal, columns=px.columns)
    for tk, fd, sue in ev:
        if tk not in px.columns:
            continue
        after = cal[cal > fd]
        if len(after) == 0:
            continue
        i = px.index.get_loc(after[0]); win = slice(i, min(i + HOLD, len(cal)))
        if sue >= thr_hi: longs.iloc[win, longs.columns.get_loc(tk)] = 1.0
        elif sue <= thr_lo: shorts.iloc[win, shorts.columns.get_loc(tk)] = 1.0
    lw = longs.div(longs.sum(axis=1).replace(0, np.nan), axis=0).fillna(0.0)
    sw = shorts.div(shorts.sum(axis=1).replace(0, np.nan), axis=0).fillna(0.0)
    w = (lw - sw)
    ret = px.pct_change()
    port = (w.shift(1).fillna(0.0) * ret).sum(axis=1)
    rv = port.rolling(21).std().mul(np.sqrt(ANN)).shift(1).replace(0, np.nan)
    w2 = w.mul((0.10 / rv).clip(upper=3.0).fillna(1.0), axis=0)
    turn = w2.diff().abs().sum(axis=1).fillna(0.0)
    pnl = ((w2.shift(1).fillna(0.0) * ret).sum(axis=1) - turn * COST).dropna()
    eq = (1 + pnl).cumprod(); dd_ = float(((eq.cummax() - eq) / eq.cummax()).max())
    sh = float(pnl.mean() / (pnl.std() + 1e-12) * np.sqrt(ANN))
    gate = sh >= GATE_SHARPE and dd_ <= GATE_MAXDD
    print(f"\n=== Tradeable PEAD L/S (long top-SUE / short bottom, hold {HOLD}d, vol-targeted) ===")
    print(f" net Sharpe {sh:.3f}  maxDD {dd_:.1%}  CAGR {float(eq.iloc[-1]**(ANN/len(pnl))-1):+.1%}  "
          f"-> {'PASS' if gate else 'FAIL'}  (gate Sh>={GATE_SHARPE} & DD<={GATE_MAXDD:.0%})")
    out["tradeable"] = {"net_sharpe": round(sh, 3), "max_dd": round(dd_, 3), "gate_pass": bool(gate)}
    print("\nREAD: monotonic Q1<Q5 + positive L/S abn + tradeable Sharpe>=0.4 = real PEAD alpha. "
          "This is the LAST distinct alpha lever; behavioral, not valuation.")
    p = REPO_ROOT / "trained_data" / "backtests" / f"pead_sec_{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}.json"
    p.parent.mkdir(parents=True, exist_ok=True); tmp = p.with_suffix(".json.tmp"); tmp.write_text(json.dumps(out, indent=2, sort_keys=True)); tmp.rename(p)
    print(f"\nwrote {p} ({round(time.time()-t0,1)}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
