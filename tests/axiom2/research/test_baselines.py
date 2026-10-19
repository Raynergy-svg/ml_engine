import pandas as pd
import pytest

def test_naive_momentum_baseline_subtracts_explicit_costs():
    from src.axiom2.research.baselines import evaluate_momentum_baseline, CostAssumption
    idx=pd.MultiIndex.from_tuples([(pd.Timestamp("2025-01-01",tz="UTC"),"A"),(pd.Timestamp("2025-01-01",tz="UTC"),"B")],names=["timestamp","instrument_id"])
    f=pd.DataFrame({"return_5":[.2,.1]},index=idx); labels=pd.DataFrame({"forward_excess_return":[.03,.01],"forward_return":[.03,.01]},index=idx)
    c=CostAssumption(evidence_id="cost-v1",round_trip_bps=20)
    result=evaluate_momentum_baseline(f,labels,costs=c,top_n=1)
    assert result.gross_return==pytest.approx(.03); assert result.net_return==pytest.approx(.029); assert result.cost_evidence_id=="cost-v1"


def test_turnover_cost_tracks_actual_holdings_changes():
    from src.axiom2.research.baselines import evaluate_momentum_baseline, CostAssumption
    dates=pd.to_datetime(["2025-01-01","2025-01-02"],utc=True)
    idx=pd.MultiIndex.from_product([dates,["A","B"]],names=["timestamp","instrument_id"])
    labels=pd.DataFrame({"forward_excess_return":[.03,.01,.04,.02],"forward_return":[.03,.01,.04,.02]},index=idx)
    costs=CostAssumption(evidence_id="cost-v1",round_trip_bps=20)
    same=pd.DataFrame({"return_5":[.2,.1,.3,.1]},index=idx)
    changed=pd.DataFrame({"return_5":[.2,.1,.1,.3]},index=idx)
    a=evaluate_momentum_baseline(same,labels,costs=costs,top_n=1)
    b=evaluate_momentum_baseline(changed,labels,costs=costs,top_n=1)
    assert a.turnover==pytest.approx(.25)
    assert b.turnover==pytest.approx(.75)
    assert a.net_return==pytest.approx(a.gross_return-.0005)
    assert b.net_return==pytest.approx(b.gross_return-.0015)
