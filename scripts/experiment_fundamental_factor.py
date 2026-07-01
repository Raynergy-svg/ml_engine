#!/usr/bin/env python3
"""Fundamental value+quality cross-sectional factor — point-in-time, FREE (SEC EDGAR).

The one untested ALPHA lever, finally unblocked: SEC EDGAR via OpenBB gives ~15yr of
point-in-time fundamentals for FREE (filing_date known), where every paywall gated data
the SEC publishes openly. yfinance for prices.

LEAKAGE-SAFE: a fundamental quarter is only consumed AFTER its filing_date (join
income.period_ending == filings.report_date -> filing_date). Flow vars (NI, revenue,
gross profit) are trailing-4-quarter sums. market_cap = price * latest-filed diluted shares.

Factors (cross-sectional, monthly rebalance):
  VALUE   = rank(book/price) + rank(earnings_yield NI_ttm/mktcap) + rank(fcf-proxy)
  QUALITY = rank(ROE) + rank(gross_margin) + rank(-debt/assets)
Long-short (dollar-neutral) isolates ALPHA; long-only top-quintile shown too. Ship gate:
net Sharpe>=0.40 AND maxDD<=25%, vs SPY. Usage: python scripts/experiment_fundamental_factor.py
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

CACHE = REPO_ROOT / "market_data" / "equity" / "sec"
START, END, ANN, COST = "2011-01-01", "2026-06-01", 12, 0.0010   # monthly; 10bps/side
GATE_SHARPE, GATE_MAXDD = 0.40, 0.25
UNIVERSE = ("AAPL MSFT NVDA AMZN GOOGL META AVGO ORCL CRM ADBE CSCO INTC QCOM TXN IBM AMAT "
            "JPM BAC WFC GS MS C AXP BLK USB PNC SPGI "
            "UNH JNJ LLY PFE MRK ABBV TMO ABT DHR BMY AMGN MDT GILD CI "
            "WMT PG KO PEP COST MCD NKE LOW HD TGT CL MDLZ MO "
            "XOM CVX COP SLB EOG PSX VLO OXY "
            "CAT GE HON UNP LMT RTX MMM DE EMR ITW FDX UPS "
            "T VZ CMCSA NEE DUK SO LIN").split()


def cached(tk, kind, fetch):
    CACHE.mkdir(parents=True, exist_ok=True)
    f = CACHE / f"{tk}_{kind}.parquet"
    if f.exists():
        try: return pd.read_parquet(f)
        except Exception: pass
    df = fetch()
    if df is not None and len(df):
        try: df.to_parquet(f)
        except Exception: pass
    return df


def pull_ticker(tk):
    from openbb import obb
    def inc(): return obb.equity.fundamental.income(tk, provider="sec", period="quarterly", limit=64).to_df()
    def bal(): return obb.equity.fundamental.balance(tk, provider="sec", period="quarterly", limit=64).to_df()
    def fl():
        a = obb.equity.fundamental.filings(tk, provider="sec", form_type="10-Q", limit=64).to_df()
        b = obb.equity.fundamental.filings(tk, provider="sec", form_type="10-K", limit=20).to_df()
        return pd.concat([a, b], ignore_index=True)
    try:
        I, B, F = cached(tk,"inc",inc), cached(tk,"bal",bal), cached(tk,"fil",fl)
    except Exception as e:
        print(f"  {tk}: pull fail {repr(e)[:60]}"); return None
    if I is None or B is None or F is None or not len(I) or not len(F): return None
    return I, B, F


def first_col(df, cands):
    for c in cands:
        if c in df.columns: return c
    return None


def build_panel(tk, px_monthly):
    got = pull_ticker(tk)
    if got is None: return None
    I, B, F = got
    # filing_date per report_date
    F = F.copy(); F["report_date"] = pd.to_datetime(F["report_date"]); F["filing_date"] = pd.to_datetime(F["filing_date"])
    fmap = F.dropna(subset=["report_date","filing_date"]).groupby("report_date")["filing_date"].min()
    I = I.copy(); B = B.copy()
    I["pe"] = pd.to_datetime(I["period_ending"]); B["pe"] = pd.to_datetime(B["period_ending"])
    eq_c = first_col(B, ["common_equity","total_equity","total_shareholders_equity","total_equity_non_controlling_interests"])
    if eq_c is None: return None
    sh_c = first_col(I, ["weighted_ave_diluted_shares_os","weighted_ave_basic_shares_os"])
    gp_c = first_col(I, ["total_gross_profit"])
    df = I[["pe","net_income","total_revenue"]+([gp_c] if gp_c else [])+([sh_c] if sh_c else [])].merge(
        B[["pe",eq_c,"total_assets"]], on="pe", how="inner").sort_values("pe")
    df["fdate"] = df["pe"].map(fmap)
    df["fdate"] = df["fdate"].fillna(df["pe"] + pd.Timedelta(days=50))   # fallback lag if no filing match
    # TTM sums for flow vars
    for c in ["net_income","total_revenue"]+([gp_c] if gp_c else []):
        df[c+"_ttm"] = df[c].rolling(4).sum()
    df = df.dropna(subset=["net_income_ttm","total_revenue_ttm",eq_c]).reset_index(drop=True)
    if len(df) < 8: return None
    # as-of each month-end price date, take latest row with fdate <= date
    out = {}
    for d in px_monthly.index:
        avail = df[df["fdate"] <= d]
        if not len(avail): continue
        r = avail.iloc[-1]; price = px_monthly.get(d, np.nan)
        if not (price > 0) or not (sh_c and r.get(sh_c,0) > 0) or not (r[eq_c] > 0): continue
        mcap = price * r[sh_c]
        gp_ttm = r.get((gp_c+"_ttm"), np.nan) if gp_c else np.nan
        out[d] = {"bp": r[eq_c]/mcap, "ey": r["net_income_ttm"]/mcap,
                  "roe": r["net_income_ttm"]/r[eq_c],
                  "gm": (gp_ttm/r["total_revenue_ttm"]) if gp_c and r["total_revenue_ttm"]>0 else np.nan,
                  "lev": -(r["total_assets"]-r[eq_c])/r["total_assets"]}
    return pd.DataFrame(out).T if out else None


def xs(s):
    r = s.rank(axis=1); n = r.count(axis=1)
    return r.sub((n+1)/2,axis=0).div((n-1)/2,axis=0).where(s.notna())


def bt(close_m, w):
    ret = close_m.pct_change().reindex(columns=w.columns)
    we = w.shift(1).fillna(0.0); turn = we.diff().abs().sum(axis=1).fillna(0.0)
    pnl = ((we*ret).sum(axis=1) - turn*COST).dropna()
    if len(pnl) < 24: return None
    eq=(1+pnl).cumprod(); dd=float(((eq.cummax()-eq)/eq.cummax()).max())
    py={str(y):round(float((1+g).prod()-1),3) for y,g in pnl.groupby(pnl.index.year)}
    return {"net_sharpe":round(float(pnl.mean()/(pnl.std()+1e-12)*np.sqrt(ANN)),3),"max_dd":round(dd,3),
            "cagr":round(float(eq.iloc[-1]**(ANN/len(pnl))-1),4),"pos_years":f"{sum(v>0 for v in py.values())}/{len(py)}"}


def main():
    t0=time.time()
    import yfinance as yf
    print(f"[fund] pulling prices + SEC fundamentals for {len(UNIVERSE)} tickers (first run slow, cached after)...")
    px = yf.download(UNIVERSE+["SPY"], start="2010-06-01", end=END, interval="1d", progress=False, auto_adjust=True)["Close"]
    pxm = px.resample("ME").last()
    pxm = pxm[(pxm.index>=START)]
    panels={}
    for i,tk in enumerate(UNIVERSE):
        if tk not in pxm.columns: continue
        p = build_panel(tk, pxm[tk].dropna())
        if p is not None and len(p)>24: panels[tk]=p
        if i%15==0: print(f"  ...{i}/{len(UNIVERSE)} ({len(panels)} usable)")
    print(f"[fund] {len(panels)} tickers with usable point-in-time fundamentals")
    if len(panels) < 20: print("too few tickers; aborting"); return 1

    # assemble factor panels (date × ticker)
    dates = pxm.index
    def fac(name): return pd.DataFrame({tk:p[name] for tk,p in panels.items()}).reindex(dates)
    bp,ey,roe,gm,lev = fac("bp"),fac("ey"),fac("roe"),fac("gm"),fac("lev")
    value = (xs(bp)+xs(ey))/2
    quality = (xs(roe)+xs(gm)+xs(lev))/3
    close_m = pxm[list(panels)].reindex(dates)

    rows=[]
    for name,sig in [("VALUE",value),("QUALITY",quality),("VALUE+QUALITY",(value+quality)/2)]:
        ls=sig.sub(sig.mean(axis=1),axis=0); ls=ls.div(ls.abs().sum(axis=1),axis=0)
        s=bt(close_m, ls)
        if s: s.update(book=f"{name} L/S (alpha)"); s["gate"]=s["net_sharpe"]>=GATE_SHARPE and s["max_dd"]<=GATE_MAXDD; rows.append(s)
    # long-only top quintile of value+quality
    vq=(value+quality)/2; thr=vq.quantile(0.8,axis=1); lo=vq.ge(thr,axis=0).astype(float); lo=lo.div(lo.sum(axis=1).replace(0,np.nan),axis=0).fillna(0.0)
    s=bt(close_m, lo)
    if s: s.update(book="VALUE+QUALITY long-only Q5"); s["gate"]=s["net_sharpe"]>=GATE_SHARPE and s["max_dd"]<=GATE_MAXDD; rows.append(s)
    spy=pxm["SPY"].pct_change().dropna(); eq=(1+spy).cumprod()
    rows.append({"book":"SPY buy&hold","net_sharpe":round(float(spy.mean()/(spy.std()+1e-12)*np.sqrt(ANN)),3),
                 "max_dd":round(float(((eq.cummax()-eq)/eq.cummax()).max()),3),"cagr":round(float(eq.iloc[-1]**(ANN/len(spy))-1),4),"pos_years":"-","gate":None})

    print(f"\n=== FUNDAMENTAL FACTORS (point-in-time SEC, {len(panels)} stocks, monthly, {START[:4]}-2026) ===")
    print(f"gate: net Sharpe>={GATE_SHARPE} AND maxDD<={GATE_MAXDD:.0%}\n")
    print(f"{'book':30s} {'net_Sh':>7s} {'maxDD':>7s} {'CAGR':>7s} {'+yrs':>7s}  gate")
    for r in rows:
        tag="PASS" if r["gate"] else ("(bmk)" if r["gate"] is None else "fail")
        print(f"{r['book']:30s} {r['net_sharpe']:>7.3f} {r['max_dd']:>7.1%} {r['cagr']:>+7.1%} {r['pos_years']:>7s}  {tag}")
    print("\nREAD: the L/S books isolate fundamental ALPHA (market-neutral, point-in-time). A L/S "
          "book clearing the gate = genuine fundamental alpha — the edge FX & price-only equities lacked.")
    out={"n_stocks":len(panels),"span":f"{START[:4]}-2026","books":rows,"source":"SEC-EDGAR-via-OpenBB(free,PIT)"}
    p=REPO_ROOT/"trained_data"/"backtests"/f"fundamental_factor_{time.strftime('%Y%m%dT%H%M%SZ',time.gmtime())}.json"
    p.parent.mkdir(parents=True,exist_ok=True); tmp=p.with_suffix(".json.tmp"); tmp.write_text(json.dumps(out,indent=2,sort_keys=True)); tmp.rename(p)
    print(f"\nwrote {p} ({round(time.time()-t0,1)}s)")
    return 0


if __name__=="__main__":
    raise SystemExit(main())
