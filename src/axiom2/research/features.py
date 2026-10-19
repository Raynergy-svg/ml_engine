"""Minimal point-in-time Phase-1 equity feature factory."""
from dataclasses import dataclass
from datetime import datetime
import numpy as np
import pandas as pd
from src.axiom2.contracts.research_proposal import PHASE1_FEATURE_FAMILIES
from src.axiom2.data.universe import build_universe_as_of

@dataclass(frozen=True)
class FeatureFrame:
    frame: pd.DataFrame
    families: tuple[str, ...]

_BASE_REQUIRED={"timestamp","instrument_id","close","available_to_axiom_time","feature_cutoff"}
_FAMILY_COLUMNS={
    "price_return":("return_1","return_5"),
    "volume_liquidity":("volume_ratio_5",),
    "cross_sectional_relative_strength":("relative_strength_1","sector_relative_1"),
    "market_sector_regime":("benchmark_return_1","market_regime_5"),
}

def _as_utc_series(frame, column, *, name):
    values=pd.to_datetime(frame[column],utc=True,errors="coerce")
    if values.isna().any():
        raise ValueError(f"{name} availability metadata is required")
    return values

def _reference_input(value, row_cutoffs, *, name, sector=False):
    if not isinstance(value,pd.DataFrame) or "available_to_axiom_time" not in value.columns:
        raise ValueError(f"{name} availability metadata is required")
    item=value.copy()
    item.index=pd.to_datetime(item.index,utc=True)
    if item.index.has_duplicates:
        raise ValueError(f"{name} timestamps must be unique")
    item["available_to_axiom_time"]=_as_utc_series(item,"available_to_axiom_time",name=name)
    item=item[item.index.isin(row_cutoffs)].sort_index()
    for timestamp,row in item.iterrows():
        if row["available_to_axiom_time"]>row_cutoffs[timestamp]:
            raise ValueError(f"{name} availability is after row decision cutoff")
    values=item.drop(columns=["available_to_axiom_time"])
    if not sector:
        if list(values.columns)!=["value"]:
            raise ValueError("benchmark must contain exactly value plus availability")
        return values["value"]
    return values

def build_phase1_features(
    ohlcv: pd.DataFrame, benchmark: pd.DataFrame, sectors: pd.DataFrame, manifest,
    *, cutoff: datetime|pd.Timestamp, families: tuple[str,...]|None=None
) -> FeatureFrame:
    selected=tuple(sorted(PHASE1_FEATURE_FAMILIES)) if families is None else tuple(families)
    if not selected or len(set(selected))!=len(selected) or any(x not in PHASE1_FEATURE_FAMILIES for x in selected):
        raise ValueError("quarantined or unknown feature family")
    required=set(_BASE_REQUIRED)
    if "volume_liquidity" in selected:
        required.add("volume")
    if "cross_sectional_relative_strength" in selected:
        required.add("sector")
    if not isinstance(ohlcv,pd.DataFrame) or not required.issubset(ohlcv.columns):
        raise ValueError("OHLCV input missing required selected-family point-in-time fields including feature_cutoff")
    cut=pd.Timestamp(cutoff)
    if cut.tzinfo is None:
        raise ValueError("cutoff must be timezone-aware")
    cut=cut.tz_convert("UTC")
    x=ohlcv.copy()
    x["timestamp"]=pd.to_datetime(x["timestamp"],utc=True,errors="coerce")
    x["available_to_axiom_time"]=_as_utc_series(x,"available_to_axiom_time",name="source")
    x["feature_cutoff"]=pd.to_datetime(x["feature_cutoff"],utc=True,errors="coerce")
    if x["timestamp"].isna().any() or x["feature_cutoff"].isna().any():
        raise ValueError("timestamp and feature_cutoff must be known")
    x=x[x["timestamp"]<=cut].sort_values(["instrument_id","timestamp"])
    if (x["feature_cutoff"]>cut).any():
        raise ValueError("feature_cutoff is after requested cutoff")
    if (x["available_to_axiom_time"]>x["feature_cutoff"]).any():
        raise ValueError("source availability is after row decision cutoff")
    if (x["feature_cutoff"]<x["timestamp"]).any():
        raise ValueError("feature_cutoff cannot precede source timestamp")
    cutoff_counts=x.groupby("timestamp")["feature_cutoff"].nunique(dropna=False)
    if (cutoff_counts!=1).any():
        raise ValueError("feature_cutoff must be identical across instruments at a timestamp")
    row_cutoffs=x.groupby("timestamp")["feature_cutoff"].first().to_dict()
    x=x.loc[[row.instrument_id in build_universe_as_of(row.timestamp.to_pydatetime(),manifest) for row in x.itertuples()]].copy()
    g=x.groupby("instrument_id",sort=False)
    needs_returns=bool({"price_return","cross_sectional_relative_strength"} & set(selected))
    if needs_returns:
        x["return_1"]=g["close"].pct_change(fill_method=None)
    if "price_return" in selected:
        x["return_5"]=g["close"].pct_change(5,fill_method=None)
    if "volume_liquidity" in selected:
        x["volume_ratio_5"]=x["volume"]/g["volume"].transform(lambda s:s.rolling(5,min_periods=2).mean())-1
    needs_benchmark=bool({"cross_sectional_relative_strength","market_sector_regime"} & set(selected))
    if needs_benchmark:
        b=_reference_input(benchmark,row_cutoffs,name="benchmark")
        br=b.pct_change(fill_method=None)
    if "cross_sectional_relative_strength" in selected:
        sec=_reference_input(sectors,row_cutoffs,name="sector",sector=True)
        sr=sec.pct_change(fill_method=None)
        x["benchmark_return_1"]=x["timestamp"].map(br)
        x["relative_strength_1"]=x["return_1"]-x["benchmark_return_1"]
        x["sector_return_1"]=[sr.at[t,s] if t in sr.index and s in sr.columns else np.nan for t,s in zip(x["timestamp"],x["sector"])]
        x["sector_relative_1"]=x["return_1"]-x["sector_return_1"]
    if "market_sector_regime" in selected:
        x["benchmark_return_1"]=x["timestamp"].map(br)
        x["market_regime_5"]=x["timestamp"].map(b.pct_change(5,fill_method=None))
    cols=[column for family in selected for column in _FAMILY_COLUMNS[family]]
    return FeatureFrame(
        x.set_index(["timestamp","instrument_id"])[cols]
        .replace([np.inf,-np.inf],np.nan).sort_index(),
        selected,
    )
