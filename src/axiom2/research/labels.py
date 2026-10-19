"""Forward benchmark-adjusted cross-sectional labels."""
import pandas as pd

def build_forward_rank_labels(prices: pd.Series, benchmark: pd.Series, *, horizon: int) -> pd.DataFrame:
    if horizon<=0: raise ValueError("horizon must be positive")
    if not isinstance(prices.index,pd.MultiIndex) or prices.index.names!=["timestamp","instrument_id"]: raise ValueError("prices require timestamp/instrument_id index")
    dates=pd.DatetimeIndex(prices.index.get_level_values("timestamp").unique()).sort_values()
    b=benchmark.copy(); b.index=pd.to_datetime(b.index,utc=True)
    if not b.index.equals(dates): raise ValueError("benchmark alignment must exactly match price dates")
    frame=prices.rename("price").unstack("instrument_id").reindex(dates); forward=frame.shift(-horizon)/frame-1; bf=b.shift(-horizon)/b-1
    excess=forward.sub(bf,axis=0); rank=excess.rank(axis=1,pct=True,method="average")
    out=pd.DataFrame({"forward_return":forward.stack(future_stack=True),"forward_excess_return":excess.stack(future_stack=True),"cross_sectional_rank":rank.stack(future_stack=True)})
    out.index.names=["timestamp","instrument_id"]; return out.reindex(prices.index)
