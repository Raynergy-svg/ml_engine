#!/usr/bin/env python3
"""Large-universe cross-sectional equity factor test — the PROPER free version.

The earlier equity test used only 16 ETFs — too thin a cross-section for stock-picking
factors. This runs the canonical test: ~120 single stocks, 2010-2026, PRICE-ONLY
cross-sectional factors (momentum 12-1, low-vol, short-term reversal), long-short
dollar-neutral AND long-only, vol-targeted, cost-aware, causal — through the same ship
gate (net Sharpe>=0.40 AND maxDD<=25%). Free yfinance data, multi-year, no API credits.

CAVEAT: universe = CURRENT large caps → survivorship bias (optimistic, esp. long-only;
less so for the dollar-neutral long-short spread). A clean test needs point-in-time
constituents. This is the best free multi-year read. Usage: python scripts/experiment_equity_xs_factor.py
"""
from __future__ import annotations

import json, sys, time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

START, END, ANN, COST = "2010-01-01", "2026-06-01", 252, 0.0005
GATE_SHARPE, GATE_MAXDD = 0.40, 0.25
UNIVERSE = ("AAPL MSFT NVDA AMZN GOOGL META AVGO TSLA AMD ORCL CRM ADBE CSCO INTC QCOM TXN IBM "
            "INTU AMAT MU ACN NOW JPM BAC WFC GS MS C AXP BLK SCHW V MA PYPL SPGI CB PGR USB PNC "
            "UNH JNJ LLY PFE MRK ABBV TMO ABT DHR BMY AMGN CVS MDT ISRG GILD VRTX CI ELV "
            "WMT PG KO PEP COST MCD NKE SBUX TGT LOW HD DIS CL KMB MDLZ MO PM EL "
            "XOM CVX COP SLB EOG PSX MPC VLO OXY WMB "
            "BA CAT GE HON UNP LMT RTX MMM DE EMR ETN ITW CSX NSC FDX UPS "
            "T VZ CMCSA TMUS NEE DUK SO D AEP EXC LIN APD SHW FCX NEM").split()


def load():
    import yfinance as yf
    df = yf.download(UNIVERSE, start=START, end=END, interval="1d", progress=False, auto_adjust=True)["Close"]
    return df.dropna(axis=1, thresh=int(0.7*len(df)))


def xs_rank(s):
    r = s.rank(axis=1); n = r.count(axis=1)
    return r.sub((n+1)/2.0, axis=0).div((n-1)/2.0, axis=0).where(s.notna())


def bt(close, w, cost=COST):
    ret = close.pct_change().reindex(columns=w.columns)
    we = w.shift(1).fillna(0.0)
    turn = we.diff().abs().sum(axis=1).fillna(0.0)
    pnl = ((we*ret).sum(axis=1) - turn*cost).dropna()
    eq = (1+pnl).cumprod(); dd = float(((eq.cummax()-eq)/eq.cummax()).max())
    py = {str(y): round(float((1+g).prod()-1),4) for y,g in pnl.groupby(pnl.index.year)}
    return {"net_sharpe": round(float(pnl.mean()/(pnl.std()+1e-12)*np.sqrt(ANN)),3),
            "max_dd": round(dd,3), "cagr": round(float(eq.iloc[-1]**(ANN/len(pnl))-1),4),
            "pos_years": f"{sum(v>0 for v in py.values())}/{len(py)}"}


def vt(raw, close, tv=0.10):
    port = (raw.shift(1)*close.pct_change()).sum(axis=1)
    rv = port.rolling(21).std().mul(np.sqrt(ANN)).shift(1).replace(0,np.nan)
    return raw.mul((tv/rv).clip(upper=3.0).fillna(1.0), axis=0)


def main():
    t0=time.time(); close=load()
    print(f"[xs-equity] universe {close.shape[1]} stocks × {close.shape[0]} days "
          f"{close.index.min().date()}->{close.index.max().date()}")
    ret=close.pct_change()
    mom = xs_rank(close.shift(21)/close.shift(252)-1)          # 12-1 momentum
    lowvol = xs_rank(-ret.rolling(63).std())                    # low-vol
    rev = xs_rank(-(close/close.shift(5)-1))                    # 1-week reversal
    warm = close.shift(252).notna()
    books = {"momentum": mom, "low_vol": lowvol, "reversal": rev,
             "combo(mom+lowvol+rev)": ((mom+lowvol+rev)/3.0)}
    rows=[]
    for name,sig in books.items():
        sig=sig.where(warm)
        ls=sig.sub(sig.mean(axis=1),axis=0); ls=ls.div(ls.abs().sum(axis=1),axis=0)
        s=bt(close, vt(ls,close)); s["book"]=f"{name} L/S"; s["gate"]=s["net_sharpe"]>=GATE_SHARPE and s["max_dd"]<=GATE_MAXDD
        rows.append(s)
    # long-only top-quintile of the combo
    cb=books["combo(mom+lowvol+rev)"].where(warm)
    thr=cb.quantile(0.8,axis=1); lo=cb.ge(thr,axis=0).astype(float); lo=lo.div(lo.sum(axis=1).replace(0,np.nan),axis=0).fillna(0.0)
    s=bt(close, vt(lo,close)); s["book"]="combo long-only Q5"; s["gate"]=s["net_sharpe"]>=GATE_SHARPE and s["max_dd"]<=GATE_MAXDD
    rows.append(s)
    # SPY bmk
    import yfinance as yf
    spy=yf.download("SPY",start=START,end=END,progress=False,auto_adjust=True)["Close"]
    spy=(spy.iloc[:,0] if hasattr(spy,'columns') else spy).pct_change().dropna()
    eq=(1+spy).cumprod()
    rows.append({"book":"SPY buy&hold","net_sharpe":round(float(spy.mean()/(spy.std()+1e-12)*np.sqrt(ANN)),3),
                 "max_dd":round(float(((eq.cummax()-eq)/eq.cummax()).max()),3),
                 "cagr":round(float(eq.iloc[-1]**(ANN/len(spy))-1),4),"pos_years":"-","gate":None})

    print(f"\n=== CROSS-SECTIONAL EQUITY FACTORS — {close.shape[1]} stocks, {START[:4]}-2026 ===")
    print(f"gate: net Sharpe>={GATE_SHARPE} AND maxDD<={GATE_MAXDD:.0%}  (cost {COST*1e4:.0f}bps)")
    print(f"\n{'book':26s} {'net_Sh':>7s} {'maxDD':>7s} {'CAGR':>7s} {'+yrs':>7s}  gate")
    for r in rows:
        tag = "PASS" if r["gate"] else ("(bmk)" if r["gate"] is None else "fail")
        print(f"{r['book']:26s} {r['net_sharpe']:>7.3f} {r['max_dd']:>7.1%} {r['cagr']:>+7.1%} {r['pos_years']:>7s}  {tag}")
    print("\nREAD: the L/S books isolate stock-picking ALPHA (market-neutral). If a L/S book clears "
          "the gate, that's genuine cross-sectional alpha — the thing FX never had. Survivorship "
          "bias inflates long-only more than L/S. Fundamental factors (value/quality/PEAD) need paid data.")
    out={"universe_n":close.shape[1],"span":f"{close.index.min().date()}->{close.index.max().date()}","books":rows}
    p=REPO_ROOT/"trained_data"/"backtests"/f"equity_xs_factor_{time.strftime('%Y%m%dT%H%M%SZ',time.gmtime())}.json"
    p.parent.mkdir(parents=True,exist_ok=True); tmp=p.with_suffix(".json.tmp"); tmp.write_text(json.dumps(out,indent=2,sort_keys=True)); tmp.rename(p)
    print(f"\nwrote {p} ({round(time.time()-t0,1)}s)")
    return 0


if __name__=="__main__":
    raise SystemExit(main())
