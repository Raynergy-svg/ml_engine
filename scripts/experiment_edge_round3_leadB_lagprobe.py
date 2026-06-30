import sys; sys.path.insert(0,'.')
import numpy as np, pandas as pd
from pathlib import Path
px = pd.read_parquet('market_data/multi_asset/panel_round3.parquet')
have=[c for c in px.columns if px[c].notna().sum()>250]; px=px[have]
ANN=252.0; OOS=0.35
def trend_stream(price, lag, sma=200, step=21, cost=2.0):
    p=pd.Series(price).dropna().astype(float)
    if len(p)<sma+step: return pd.Series(dtype=float)
    s=p.rolling(sma,min_periods=max(20,sma//2)).mean().shift(lag)   # SMA lag
    on=(p.shift(lag)>s).fillna(False)                                # price lag = lag
    w=pd.Series(0.0,index=p.index); last=0.0
    for i in range(len(p)):
        if i%step==0: last=1.0 if bool(on.iloc[i]) else 0.0
        w.iloc[i]=last
    applied=w.shift(1).fillna(0.0)                                   # +1 execution lag always
    g=applied*p.pct_change(); t=applied.diff().abs().fillna(applied.abs())
    return (g - t*cost*1e-4).dropna()
def combined_sharpe(lag):
    streams={c:trend_stream(px[c],lag) for c in px.columns}
    df=pd.DataFrame({k:v for k,v in streams.items() if not v.empty}).sort_index()
    port=df.mean(axis=1).dropna()           # equal-weight combine (causality probe; HRP not needed here)
    # 10% vol target, lagged realized vol
    rv=port.rolling(60,min_periods=20).std().shift(1)
    lev=(0.10/np.sqrt(ANN)/rv).clip(upper=3.0).fillna(0.0)
    net=(port*lev).dropna()
    idx=net.index; oos=net.loc[idx[int(len(idx)*(1-OOS)):]]
    sh=lambda x: float(np.sqrt(ANN)*x.mean()/x.std(ddof=1)) if x.std(ddof=1)>0 else 0.0
    return sh(net), sh(oos)
print("lag  full_Sharpe  OOS_Sharpe   (lag=0 look-ahead, lag=1 as-run, lag=2/3 extra)")
for L in (0,1,2,3):
    f,o=combined_sharpe(L)
    tag={0:'LOOK-AHEAD',1:'AS-RUN',2:'extra',3:'extra'}[L]
    print(f"  {L}   {f:+.3f}      {o:+.3f}     {tag}")
