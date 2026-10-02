from datetime import datetime, timezone
import pandas as pd
import pytest
from src.axiom2.data.universe import UniverseManifest, UniverseMembership

def manifest():
    return UniverseManifest(universe_id="u",source="fixture",evidence_id="e",schema_version="1",membership_basis="sp500",coverage_start=datetime(2025,1,1,tzinfo=timezone.utc),coverage_end=datetime(2025,1,10,tzinfo=timezone.utc),memberships=(UniverseMembership(instrument_id="A",asset_class="equity",member_from=datetime(2025,1,1,tzinfo=timezone.utc),member_until=datetime(2025,1,10,tzinfo=timezone.utc)),UniverseMembership(instrument_id="B",asset_class="equity",member_from=datetime(2025,1,1,tzinfo=timezone.utc),member_until=datetime(2025,1,10,tzinfo=timezone.utc))))

def bars(extra=False):
    dates=pd.date_range("2025-01-01",periods=7 if extra else 6,tz="UTC")
    rows=[]
    for i,d in enumerate(dates):
        for j,s in enumerate(("A","B")): rows.append({"timestamp":d,"instrument_id":s,"close":100+i*(j+1),"volume":1000+10*i+100*j,"available_to_axiom_time":d+pd.Timedelta(hours=1),"feature_cutoff":d+pd.Timedelta(hours=2),"sector":"tech"})
    return pd.DataFrame(rows)

def test_phase1_features_are_approved_cutoff_safe_and_future_invariant():
    from src.axiom2.research.features import build_phase1_features
    cutoff=pd.Timestamp("2025-01-06T23:00:00Z"); idx=pd.date_range("2025-01-01",periods=7,tz="UTC")
    bench=pd.DataFrame({"value":[100,101,102,103,104,105,999],"available_to_axiom_time":idx+pd.Timedelta(hours=1)},index=idx)
    sector=pd.DataFrame({"tech":[100,100.5,101,101.5,102,102.5,999],"available_to_axiom_time":idx+pd.Timedelta(hours=1)},index=idx)
    a=build_phase1_features(bars(False),bench.iloc[:6],sector.iloc[:6],manifest(),cutoff=cutoff)
    b=build_phase1_features(bars(True),bench,sector,manifest(),cutoff=cutoff)
    assert a.frame.equals(b.frame)
    assert set(a.families)=={"price_return","volume_liquidity","cross_sectional_relative_strength","market_sector_regime"}
    assert a.frame.index.get_level_values("timestamp").max()<=cutoff
    assert not a.frame.isin([float("inf"),float("-inf")]).any().any()

def test_rejects_late_availability_and_quarantined_family():
    from src.axiom2.research.features import build_phase1_features
    x=bars(); x.loc[x.index[-1],"available_to_axiom_time"]=pd.Timestamp("2025-02-01",tz="UTC")
    bench=pd.Series(range(6),index=pd.date_range("2025-01-01",periods=6,tz="UTC")); sector=pd.DataFrame({"tech":range(6)},index=bench.index)
    with pytest.raises(ValueError,match="availability"): build_phase1_features(x,bench,sector,manifest(),cutoff=pd.Timestamp("2025-01-06T23:00Z"))
    with pytest.raises(ValueError,match="quarantined"): build_phase1_features(bars(),bench,sector,manifest(),cutoff=pd.Timestamp("2025-01-06T23:00Z"),families=("fundamentals",))


def test_benchmark_and_sector_inputs_require_explicit_cutoff_safe_availability():
    from src.axiom2.research.features import build_phase1_features
    idx=pd.date_range("2025-01-01",periods=6,tz="UTC")
    bench=pd.DataFrame({"value":[100,101,102,103,104,105],"available_to_axiom_time":idx+pd.Timedelta(hours=1)},index=idx)
    sector=pd.DataFrame({"tech":[100,101,102,103,104,105],"available_to_axiom_time":idx+pd.Timedelta(hours=1)},index=idx)
    cutoff=pd.Timestamp("2025-01-06T23:00Z")
    result=build_phase1_features(bars(),bench,sector,manifest(),cutoff=cutoff)
    assert not result.frame.empty
    late=bench.copy(); late.loc[idx[-1],"available_to_axiom_time"]=pd.Timestamp("2025-02-01",tz="UTC")
    with pytest.raises(ValueError,match="benchmark availability"):
        build_phase1_features(bars(),late,sector,manifest(),cutoff=cutoff)
    missing=pd.Series(range(6),index=idx)
    with pytest.raises(ValueError,match="benchmark availability"):
        build_phase1_features(bars(),missing,sector,manifest(),cutoff=cutoff)

def _with_decision_cutoffs(frame):
    out=frame.copy()
    out["feature_cutoff"]=pd.to_datetime(out["timestamp"],utc=True)+pd.Timedelta(hours=2)
    return out

def test_feature_rows_require_explicit_decision_cutoff():
    from src.axiom2.research.features import build_phase1_features
    idx=pd.date_range("2025-01-01",periods=6,tz="UTC")
    bench=pd.DataFrame({"value":range(6),"available_to_axiom_time":idx+pd.Timedelta(hours=1)},index=idx)
    sector=pd.DataFrame({"tech":range(6),"available_to_axiom_time":idx+pd.Timedelta(hours=1)},index=idx)
    with pytest.raises(ValueError,match="feature_cutoff"):
        build_phase1_features(bars().drop(columns=["feature_cutoff"]),bench,sector,manifest(),cutoff=pd.Timestamp("2025-01-06T23:00Z"))

def test_unknown_or_late_row_availability_rejects_before_feature_construction():
    from src.axiom2.research.features import build_phase1_features
    idx=pd.date_range("2025-01-01",periods=6,tz="UTC")
    bench=pd.DataFrame({"value":range(6),"available_to_axiom_time":idx+pd.Timedelta(hours=1)},index=idx)
    sector=pd.DataFrame({"tech":range(6),"available_to_axiom_time":idx+pd.Timedelta(hours=1)},index=idx)
    x=_with_decision_cutoffs(bars())
    x.loc[x["timestamp"]==idx[1],"available_to_axiom_time"]=pd.NaT
    with pytest.raises(ValueError,match="availability"):
        build_phase1_features(x,bench,sector,manifest(),cutoff=pd.Timestamp("2025-01-06T23:00Z"))
    x=_with_decision_cutoffs(bars())
    x.loc[x["timestamp"]==idx[1],"available_to_axiom_time"]=idx[-1]
    with pytest.raises(ValueError,match="row decision cutoff"):
        build_phase1_features(x,bench,sector,manifest(),cutoff=pd.Timestamp("2025-01-06T23:00Z"))

def test_reference_availability_is_checked_against_each_historical_decision_cutoff():
    from src.axiom2.research.features import build_phase1_features
    idx=pd.date_range("2025-01-01",periods=6,tz="UTC")
    x=_with_decision_cutoffs(bars())
    bench=pd.DataFrame({"value":range(6),"available_to_axiom_time":idx+pd.Timedelta(hours=1)},index=idx)
    sector=pd.DataFrame({"tech":range(6),"available_to_axiom_time":idx+pd.Timedelta(hours=1)},index=idx)
    bench.loc[idx[1],"available_to_axiom_time"]=idx[-1]
    with pytest.raises(ValueError,match="benchmark availability.*row decision cutoff"):
        build_phase1_features(x,bench,sector,manifest(),cutoff=pd.Timestamp("2025-01-06T23:00Z"))
    bench.loc[idx[1],"available_to_axiom_time"]=pd.NaT
    with pytest.raises(ValueError,match="benchmark availability"):
        build_phase1_features(x,bench,sector,manifest(),cutoff=pd.Timestamp("2025-01-06T23:00Z"))

@pytest.mark.parametrize("families,columns",[
    (("price_return",),{"return_1","return_5"}),
    (("volume_liquidity",),{"volume_ratio_5"}),
    (("cross_sectional_relative_strength",),{"relative_strength_1","sector_relative_1"}),
    (("market_sector_regime",),{"benchmark_return_1","market_regime_5"}),
])
def test_selected_families_are_the_exact_emitted_feature_contract(families,columns):
    from src.axiom2.research.features import build_phase1_features
    idx=pd.date_range("2025-01-01",periods=6,tz="UTC")
    bench=pd.DataFrame({"value":range(100,106),"available_to_axiom_time":idx+pd.Timedelta(hours=1)},index=idx)
    sector=pd.DataFrame({"tech":range(100,106),"available_to_axiom_time":idx+pd.Timedelta(hours=1)},index=idx)
    result=build_phase1_features(_with_decision_cutoffs(bars()),bench,sector,manifest(),cutoff=pd.Timestamp("2025-01-06T23:00Z"),families=families)
    assert result.families==families
    assert set(result.frame.columns)==columns


def test_price_return_family_does_not_require_unselected_reference_or_volume_inputs():
    from src.axiom2.research.features import build_phase1_features
    minimal=bars().drop(columns=["volume","sector"])
    result=build_phase1_features(
        minimal,None,None,manifest(),cutoff=pd.Timestamp("2025-01-06T23:00Z"),
        families=("price_return",),
    )
    assert set(result.frame.columns)=={"return_1","return_5"}
