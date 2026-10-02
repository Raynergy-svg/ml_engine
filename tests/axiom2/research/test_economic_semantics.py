import math
from dataclasses import replace
import pandas as pd
import pytest

def test_cost_assumption_rejects_nonfinite_values():
    from src.axiom2.research.baselines import CostAssumption
    for value in (float("nan"),float("inf"),float("-inf")):
        with pytest.raises(ValueError,match="finite"):
            CostAssumption(evidence_id="cost",round_trip_bps=value)

def test_nonoverlap_policy_cannot_rebalance_before_holding_horizon():
    from src.axiom2.research.ranker import PortfolioPolicy
    with pytest.raises(ValueError,match="overlap"):
        PortfolioPolicy(top_k=5,holding_horizon=5,rebalance_every=1,overlap=False)

def test_drawdown_includes_initial_wealth():
    from src.axiom2.research.ranker import _max_drawdown
    assert _max_drawdown([-.10,0.0,0.0]) == pytest.approx(-.10)

def test_cross_sectional_ic_is_averaged_by_session_not_pooled():
    from src.axiom2.research.ranker import _mean_cross_sectional_ic
    dates=pd.to_datetime(["2025-01-01","2025-01-02"],utc=True)
    idx=pd.MultiIndex.from_product([dates,["A","B"]],names=["timestamp","instrument_id"])
    scored=pd.DataFrame({"_pred":[1,2,100,101],"cross_sectional_rank":[0,1,0,1]},index=idx)
    assert _mean_cross_sectional_ic(scored)==pytest.approx(1.0)

def test_equal_weight_turnover_accounts_for_weight_drift():
    from src.axiom2.research.ranker import _turnover_to_equal, _post_return_weights
    current=("A","B")
    assert _turnover_to_equal({},current)==pytest.approx(.5)
    drifted=_post_return_weights(current,[.2,-.2])
    assert drifted["A"]==pytest.approx(.6)
    assert drifted["B"]==pytest.approx(.4)
    assert _turnover_to_equal(drifted,current)==pytest.approx(.1)
def test_global_rebalance_dates_do_not_reset_at_fold_boundaries():
    from src.axiom2.research.ranker import _global_rebalance_dates
    from src.axiom2.research.splits import WalkForwardSplit
    dates=pd.date_range("2025-01-01",periods=20,tz="UTC")
    splits=(
        WalkForwardSplit(train=tuple(dates[:4]),test=tuple(dates[5:9])),
        WalkForwardSplit(train=tuple(dates[:9]),test=tuple(dates[10:14])),
    )
    assert _global_rebalance_dates(dates,splits,5)==(dates[5],dates[10])

def test_baseline_ignores_rows_outside_candidate_evaluation_dates():
    from src.axiom2.research.baselines import CostAssumption,evaluate_momentum_baseline
    dates=pd.date_range("2025-01-01",periods=3,tz="UTC")
    idx=pd.MultiIndex.from_product([dates,["A","B"]],names=["timestamp","instrument_id"])
    features=pd.DataFrame({"return_5":[.2,.1,.3,.1,.4,.1]},index=idx)
    labels=pd.DataFrame({
        "forward_excess_return":[.03,.01,.04,.02,999.,999.],
        "forward_return":[.03,.01,.04,.02,999.,999.],
    },index=idx)
    costs=CostAssumption(evidence_id="c",round_trip_bps=0)
    result=evaluate_momentum_baseline(features,labels,costs=costs,top_n=1,evaluation_dates=(dates[0],dates[1]))
    changed=labels.copy()
    changed.loc[(dates[2],slice(None)),"forward_excess_return"]=-999.
    other=evaluate_momentum_baseline(features,changed,costs=costs,top_n=1,evaluation_dates=(dates[0],dates[1]))
    assert result==other

def test_baseline_turnover_includes_drift_when_raw_forward_return_is_available():
    from src.axiom2.research.baselines import CostAssumption,evaluate_momentum_baseline
    dates=pd.date_range("2025-01-01",periods=2,tz="UTC")
    idx=pd.MultiIndex.from_product([dates,["A","B"]],names=["timestamp","instrument_id"])
    features=pd.DataFrame({"return_5":[.2,.1,.2,.1]},index=idx)
    labels=pd.DataFrame({
        "forward_excess_return":[.02,.01,.02,.01],
        "forward_return":[.2,-.2,.0,.0],
    },index=idx)
    result=evaluate_momentum_baseline(features,labels,costs=CostAssumption("c",0),top_n=2)
    assert result.turnover==pytest.approx((.5+.1)/2)

def test_campaign_split_admission_rejects_overlap_and_unpurged_labels():
    from src.axiom2.research.ranker import _validate_campaign_splits
    from src.axiom2.research.splits import WalkForwardSplit
    dates=pd.date_range("2025-01-01",periods=10,tz="UTC")
    with pytest.raises(ValueError,match="overlap"):
        _validate_campaign_splits(dates,(WalkForwardSplit(tuple(dates[:5]),tuple(dates[4:7])),),horizon=2)
    with pytest.raises(ValueError,match="purge"):
        _validate_campaign_splits(dates,(WalkForwardSplit(tuple(dates[:4]),tuple(dates[4:7])),),horizon=2)
    _validate_campaign_splits(dates,(WalkForwardSplit(tuple(dates[:3]),tuple(dates[5:8])),),horizon=2)

def test_top_k_selection_rejects_incomplete_cohort():
    from src.axiom2.research.ranker import _select_complete_top_k
    frame=pd.DataFrame({"_pred":[3,2,1]},index=["A","B","C"])
    with pytest.raises(ValueError,match="top-k"):
        _select_complete_top_k(frame,5)


def test_baseline_rejects_missing_raw_forward_return_needed_for_weight_drift():
    from src.axiom2.research.baselines import CostAssumption,evaluate_momentum_baseline
    dates=pd.date_range("2025-01-01",periods=2,tz="UTC")
    idx=pd.MultiIndex.from_product([dates,["A","B"]],names=["timestamp","instrument_id"])
    features=pd.DataFrame({"return_5":[.2,.1,.2,.1]},index=idx)
    labels=pd.DataFrame({"forward_excess_return":[.02,.01,.02,.01]},index=idx)
    with pytest.raises(ValueError,match="forward_return"):
        evaluate_momentum_baseline(features,labels,costs=CostAssumption("c",0),top_n=2)
