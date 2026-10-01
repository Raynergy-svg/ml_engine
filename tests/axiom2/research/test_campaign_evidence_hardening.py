from dataclasses import replace
import hashlib
import math
import pandas as pd
import pytest

from tests.evidence.equity_research._support import build_context
from tests.axiom2.research.test_ranker_campaign import campaign_proposal, fixture_frames

def _bound_proposal(experiment_id, features, labels, splits, *, space=({"num_leaves":7,"learning_rate":.05},), extra=()):
    from src.axiom2.contracts.research_proposal import DataDependency
    from src.axiom2.research.ranker import (
        PortfolioPolicy, campaign_manifest_digest, portfolio_policy_digest,
        search_space_digest,
    )
    policy=PortfolioPolicy(top_k=1,holding_horizon=1,rebalance_every=1)
    base=campaign_proposal(experiment_id,space)
    digest=campaign_manifest_digest(
        proposal_id=base.experiment_id, features=features, labels=labels, splits=splits,
        feature_schema_id="features-v1", label_schema_id="labels-v1",
        cost_bps=10, search_space=space, portfolio_policy=policy,
    )
    dep=DataDependency(dataset_id=digest,source="axiom2-campaign-manifest",schema_version="1")
    return replace(base,data_dependencies=base.data_dependencies+(dep,)+tuple(extra))

def _splits(dates):
    from src.axiom2.research.splits import purged_walk_forward_splits
    return purged_walk_forward_splits(dates,train_size=8,test_size=3,horizon=2,embargo=1)

def test_phase1_rejects_adapter_implementation_impersonation(tmp_path):
    from src.axiom2.research.ranker import LightGBMRankerAdapter, run_registered_campaign
    class Impostor(LightGBMRankerAdapter):
        pass
    ctx=build_context(tmp_path/"store"); f,y,dates=fixture_frames(); splits=_splits(dates)
    p=_bound_proposal("adapter-identity",f,y,splits); ctx.registry.register_experiment(p)
    with pytest.raises(ValueError,match="adapter implementation"):
        run_registered_campaign(p,f,y,splits,registry=ctx.registry,feature_schema_id="features-v1",label_schema_id="labels-v1",expected_feature_schema_id="features-v1",expected_label_schema_id="labels-v1",cost_bps=10,adapter=Impostor())
def test_adapter_fit_and_prediction_receive_features_separately_from_targets(tmp_path, monkeypatch):
    import lightgbm
    from src.axiom2.research.ranker import LightGBMRankerAdapter, run_registered_campaign
    ctx=build_context(tmp_path/"store"); f,y,dates=fixture_frames(); splits=_splits(dates)
    p=_bound_proposal("feature-only-adapter",f,y,splits); ctx.registry.register_experiment(p)
    seen=[]
    class Booster:
        def model_to_string(self): return "feature-only-model"
    class FakeRanker:
        booster_=Booster()
        def __init__(self,**kwargs): pass
        def fit(self,x,target,group):
            seen.append(("fit",tuple(x.columns),getattr(target,"name",None)))
            assert "forward_excess_return" not in x and "cross_sectional_rank" not in x
            return self
        def predict(self,x):
            seen.append(("predict",tuple(x.columns),None))
            assert "forward_excess_return" not in x and "cross_sectional_rank" not in x
            return x["return_1"].to_numpy()
    monkeypatch.setattr(lightgbm,"LGBMRanker",FakeRanker)
    result=run_registered_campaign(p,f,y,splits,registry=ctx.registry,feature_schema_id="features-v1",label_schema_id="labels-v1",expected_feature_schema_id="features-v1",expected_label_schema_id="labels-v1",cost_bps=10,adapter=LightGBMRankerAdapter())
    assert result.envelope is not None and any(kind=="predict" for kind,_,_ in seen)

def test_changed_campaign_inputs_require_a_new_registered_manifest(tmp_path):
    from src.axiom2.research.ranker import run_registered_campaign
    ctx=build_context(tmp_path/"store"); f,y,dates=fixture_frames(); splits=_splits(dates)
    p=_bound_proposal("manifest-bound",f,y,splits); ctx.registry.register_experiment(p)
    changed=f.copy(); changed.iloc[0,0]+=123
    with pytest.raises(ValueError,match="campaign manifest"):
        run_registered_campaign(p,changed,y,splits,registry=ctx.registry,feature_schema_id="features-v1",label_schema_id="labels-v1",expected_feature_schema_id="features-v1",expected_label_schema_id="labels-v1",cost_bps=10)

def test_failed_registered_experiment_cannot_emit_campaign_evidence(tmp_path):
    from src.axiom2.research.ranker import run_registered_campaign
    ctx=build_context(tmp_path/"store"); f,y,dates=fixture_frames(); splits=_splits(dates)
    p=_bound_proposal("failed-campaign",f,y,splits); ctx.registry.register_experiment(p); ctx.registry.record_failure(p.experiment_id,"failed")
    with pytest.raises(ValueError,match="active"):
        run_registered_campaign(p,f,y,splits,registry=ctx.registry,feature_schema_id="features-v1",label_schema_id="labels-v1",expected_feature_schema_id="features-v1",expected_label_schema_id="labels-v1",cost_bps=10)
def test_each_configuration_has_durable_started_and_terminal_evidence(tmp_path):
    from src.axiom2.research.ranker import run_registered_campaign
    space=({"num_leaves":1,"learning_rate":.05},{"num_leaves":3,"learning_rate":.05})
    ctx=build_context(tmp_path/"store"); f,y,dates=fixture_frames(); splits=_splits(dates)
    p=_bound_proposal("trial-ledger",f,y,splits,space=space); ctx.registry.register_experiment(p)
    result=run_registered_campaign(p,f,y,splits,registry=ctx.registry,feature_schema_id="features-v1",label_schema_id="labels-v1",expected_feature_schema_id="features-v1",expected_label_schema_id="labels-v1",cost_bps=10,search_space=space)
    trials=tuple(ctx.registry.snapshot().trials.values())
    assert len(trials)==2
    assert [trial.status for trial in trials]==["FAILED","SUCCEEDED"]
    assert result.experiment_count==2

def test_candidate_freeze_resolves_signed_campaign_and_actual_model_bytes(tmp_path):
    from src.axiom2.contracts.research_proposal import DataDependency
    from src.axiom2.research.ranker import (
        LightGBMRankerAdapter, candidate_policy_digest, development_gate_digest,
        freeze_development_candidate, run_registered_campaign,
    )
    ctx=build_context(tmp_path/"store"); f,y,dates=fixture_frames(); splits=_splits(dates)
    extra=(
        DataDependency(dataset_id=candidate_policy_digest("last_fold"),source="axiom2-candidate-policy",schema_version="1"),
        DataDependency(dataset_id=development_gate_digest(min_rank_ic=-1,min_net_return=-1),source="axiom2-development-gate",schema_version="1"),
    )
    p=_bound_proposal("verified-freeze",f,y,splits,extra=extra); ctx.registry.register_experiment(p)
    campaign=run_registered_campaign(p,f,y,splits,registry=ctx.registry,feature_schema_id="features-v1",label_schema_id="labels-v1",expected_feature_schema_id="features-v1",expected_label_schema_id="labels-v1",cost_bps=10)
    with pytest.raises(ValueError,match="model artifact hash"):
        freeze_development_candidate(p,replace(campaign,model_artifact_bytes=b"tampered"),registry=ctx.registry,policy="last_fold",min_rank_ic=-1,min_net_return=-1)
    with pytest.raises(ValueError,match="finite"):
        freeze_development_candidate(p,replace(campaign,mean_rank_correlation=float("nan")),registry=ctx.registry,policy="last_fold",min_rank_ic=-1,min_net_return=-1)
    with pytest.raises(ValueError,match="signed"):
        freeze_development_candidate(p,replace(campaign,envelope=None),registry=ctx.registry,policy="last_fold",min_rank_ic=-1,min_net_return=-1)
    frozen=freeze_development_candidate(p,campaign,registry=ctx.registry,policy="last_fold",min_rank_ic=-1,min_net_return=-1)
    assert frozen.candidate_id in ctx.registry.snapshot().candidates
    assert ctx.registry.snapshot().candidates[frozen.candidate_id].artifact_hash==campaign.model_artifact_hash

def test_signed_campaign_binds_selected_successful_trial(tmp_path):
    from src.axiom2.research.ranker import run_registered_campaign
    ctx=build_context(tmp_path/"store"); f,y,dates=fixture_frames(); splits=_splits(dates)
    p=_bound_proposal("selected-trial",f,y,splits); ctx.registry.register_experiment(p)
    campaign=run_registered_campaign(p,f,y,splits,registry=ctx.registry,feature_schema_id="features-v1",label_schema_id="labels-v1",expected_feature_schema_id="features-v1",expected_label_schema_id="labels-v1",cost_bps=10)
    trial=ctx.registry.snapshot().trials[campaign.selected_trial_id]
    assert trial.status=="SUCCEEDED"
    assert trial.result_digest==campaign.selected_trial_result_digest

def test_exact_adapter_instance_method_override_is_rejected(tmp_path):
    from src.axiom2.research.ranker import LightGBMRankerAdapter, run_registered_campaign
    ctx=build_context(tmp_path/"store"); f,y,dates=fixture_frames(); splits=_splits(dates)
    p=_bound_proposal("adapter-method-override",f,y,splits); ctx.registry.register_experiment(p)
    adapter=LightGBMRankerAdapter()
    adapter.fit=lambda *args: object()
    with pytest.raises(ValueError,match="adapter implementation"):
        run_registered_campaign(p,f,y,splits,registry=ctx.registry,feature_schema_id="features-v1",label_schema_id="labels-v1",expected_feature_schema_id="features-v1",expected_label_schema_id="labels-v1",cost_bps=10,adapter=adapter)
