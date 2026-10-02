from dataclasses import replace
import pandas as pd
import pytest
from tests.evidence.equity_research._support import build_context, proposal



def campaign_proposal(experiment_id, space=({"num_leaves":7,"learning_rate":.05},)):
    from src.axiom2.research.ranker import search_space_digest
    from src.axiom2.contracts.research_proposal import DataDependency
    base=proposal(experiment_id)
    return replace(base,data_dependencies=base.data_dependencies+(DataDependency(dataset_id=search_space_digest(space),source="axiom2-hyperparameter-search",schema_version="1"),))

def fixture_frames():
    dates=pd.date_range("2025-01-01",periods=18,tz="UTC"); idx=pd.MultiIndex.from_product([dates,["A","B","C"]],names=["timestamp","instrument_id"])
    f=pd.DataFrame({"return_1":[((i%3)-1)*.01 for i in range(len(idx))],"return_5":[((i%5)-2)*.02 for i in range(len(idx))],"volume_ratio_5":[(i%4)*.1 for i in range(len(idx))],"relative_strength_1":[((i%3)-1)*.015 for i in range(len(idx))]},index=idx)
    returns=[((i%3)-1)*.02 for i in range(len(idx))]
    y=pd.DataFrame({"forward_return":returns,"forward_excess_return":returns,"cross_sectional_rank":[(i%3+1)/3 for i in range(len(idx))]},index=idx)
    return f,y,dates

def bound_campaign_proposal(experiment_id,features,labels,splits,space=({"num_leaves":7,"learning_rate":.05},),extra=(),manifest_space=None):
    from src.axiom2.contracts.research_proposal import DataDependency
    from src.axiom2.research.ranker import PortfolioPolicy,campaign_manifest_digest
    base=campaign_proposal(experiment_id,space)
    policy=PortfolioPolicy(top_k=1,holding_horizon=1,rebalance_every=1)
    digest=campaign_manifest_digest(proposal_id=base.experiment_id,features=features,labels=labels,splits=splits,feature_schema_id="features-v1",label_schema_id="labels-v1",cost_bps=10,search_space=space if manifest_space is None else manifest_space,portfolio_policy=policy)
    dependency=DataDependency(dataset_id=digest,source="axiom2-campaign-manifest",schema_version="1")
    return replace(base,data_dependencies=base.data_dependencies+(dependency,)+tuple(extra))

def test_campaign_requires_registration_schema_and_never_opens_holdout(tmp_path):
    from src.axiom2.research.ranker import run_registered_campaign
    from src.axiom2.research.splits import purged_walk_forward_splits
    ctx=build_context(tmp_path/"store"); f,y,dates=fixture_frames(); splits=purged_walk_forward_splits(dates,train_size=8,test_size=3,horizon=2,embargo=1); p=bound_campaign_proposal("campaign-1",f,y,splits)
    with pytest.raises(ValueError,match="registered"): run_registered_campaign(p,f,y,splits,registry=ctx.registry,feature_schema_id="features-v1",label_schema_id="labels-v1",expected_feature_schema_id="features-v1",expected_label_schema_id="labels-v1",cost_bps=10)
    ctx.registry.register_experiment(p)
    with pytest.raises(ValueError,match="schema"): run_registered_campaign(p,f,y,splits,registry=ctx.registry,feature_schema_id="bad",label_schema_id="labels-v1",expected_feature_schema_id="features-v1",expected_label_schema_id="labels-v1",cost_bps=10)
    assert ctx.registry.snapshot().holdouts=={}

def test_campaign_is_deterministic_reports_baseline_search_count_and_artifact(tmp_path):
    from src.axiom2.research.ranker import run_registered_campaign
    from src.axiom2.research.splits import purged_walk_forward_splits
    ctx=build_context(tmp_path/"store"); space=({"num_leaves":3,"learning_rate":.05},{"num_leaves":5,"learning_rate":.05}); f,y,dates=fixture_frames(); splits=purged_walk_forward_splits(dates,train_size=8,test_size=3,horizon=2,embargo=1); p=bound_campaign_proposal("campaign-2",f,y,splits,space); ctx.registry.register_experiment(p)
    kwargs=dict(registry=ctx.registry,feature_schema_id="features-v1",label_schema_id="labels-v1",expected_feature_schema_id="features-v1",expected_label_schema_id="labels-v1",cost_bps=10,search_space=space)
    a=run_registered_campaign(p,f,y,splits,**kwargs); b=run_registered_campaign(p,f,y,splits,**kwargs)
    assert a==b; assert a.experiment_count==2; assert a.baseline_net_return is not None; assert len(a.artifact_hash)==64; assert a.model_type=="lightgbm"; assert a.holdout_accessed is False


def test_campaign_result_is_signed_by_registered_research_authority(tmp_path):
    from src.axiom2.research.ranker import run_registered_campaign
    from src.axiom2.research.splits import purged_walk_forward_splits
    from src.evidence.signing import verify_envelope
    from src.axiom2.research.ranker import CampaignPayload
    ctx=build_context(tmp_path/"store"); f,y,dates=fixture_frames(); splits=purged_walk_forward_splits(dates,train_size=8,test_size=3,horizon=2,embargo=1); p=bound_campaign_proposal("campaign-signed",f,y,splits); ctx.registry.register_experiment(p)
    result=run_registered_campaign(p,f,y,splits,registry=ctx.registry,feature_schema_id="features-v1",label_schema_id="labels-v1",expected_feature_schema_id="features-v1",expected_label_schema_id="labels-v1",cost_bps=10)
    payload=verify_envelope(result.envelope,CampaignPayload,ctx.store.trust_store)
    assert payload.artifact_hash==result.artifact_hash


def test_campaign_entrypoint_exists_and_does_not_import_holdout_authority():
    from pathlib import Path
    path=Path("scripts/axiom2_run_phase1_campaign.py")
    assert path.is_file()
    text=path.read_text()
    assert "HoldoutAuthority" not in text
    assert "open_holdout" not in text


def test_registered_search_digest_must_match_exact_hyperparameter_space(tmp_path):
    from src.axiom2.research.ranker import run_registered_campaign, search_space_digest
    from src.axiom2.research.splits import purged_walk_forward_splits
    from src.axiom2.contracts.research_proposal import DataDependency
    ctx=build_context(tmp_path/"store"); space=({"num_leaves":3,"learning_rate":.05},); bad=({"num_leaves":5,"learning_rate":.05},)
    f,y,dates=fixture_frames(); splits=purged_walk_forward_splits(dates,train_size=8,test_size=3,horizon=2,embargo=1)
    p=bound_campaign_proposal("campaign-search-bound",f,y,splits,space,manifest_space=bad)
    ctx.registry.register_experiment(p)
    with pytest.raises(ValueError,match="preregistered search"):
        run_registered_campaign(p,f,y,splits,registry=ctx.registry,feature_schema_id="features-v1",label_schema_id="labels-v1",expected_feature_schema_id="features-v1",expected_label_schema_id="labels-v1",cost_bps=10,search_space=bad)


def test_campaign_binds_exact_inputs_and_serialized_model_artifact(tmp_path):
    from src.axiom2.research.ranker import run_registered_campaign, search_space_digest, frame_digest, split_manifest_digest
    from src.axiom2.research.splits import purged_walk_forward_splits
    space=({"num_leaves":3,"learning_rate":.05},)
    ctx=build_context(tmp_path/"store"); f,y,dates=fixture_frames(); splits=purged_walk_forward_splits(dates,train_size=8,test_size=3,horizon=2,embargo=1); p=bound_campaign_proposal("campaign-input-bound",f,y,splits,space); ctx.registry.register_experiment(p)
    r=run_registered_campaign(p,f,y,splits,registry=ctx.registry,feature_schema_id="features-v1",label_schema_id="labels-v1",expected_feature_schema_id="features-v1",expected_label_schema_id="labels-v1",cost_bps=10,search_space=space)
    assert r.feature_digest==frame_digest(f)
    assert r.label_digest==frame_digest(y)
    assert r.split_digest==split_manifest_digest(splits)
    assert len(r.model_artifact_hash)==64
    assert r.model_artifact_bytes

def test_campaign_runner_accepts_model_adapter_contract_not_lightgbm_infrastructure():
    from src.axiom2.research.ranker import ModelResearchAdapter, LightGBMRankerAdapter
    assert isinstance(LightGBMRankerAdapter(),ModelResearchAdapter)
    assert LightGBMRankerAdapter().model_family=="lightgbm"


def test_phase1_rejects_other_model_adapter_even_if_interface_compliant(tmp_path):
    from src.axiom2.research.ranker import ModelResearchAdapter, run_registered_campaign
    from src.axiom2.research.splits import purged_walk_forward_splits
    class Other(ModelResearchAdapter):
        model_family="other"
        def fit(self,*args): return object()
        def predict_scores(self,*args): return []
        def serialize_artifact(self,*args): return b"x"
    ctx=build_context(tmp_path/"store"); f,y,dates=fixture_frames(); splits=purged_walk_forward_splits(dates,train_size=8,test_size=3,horizon=2,embargo=1); p=bound_campaign_proposal("campaign-other",f,y,splits); ctx.registry.register_experiment(p)
    with pytest.raises(ValueError,match="adapter implementation"):
        run_registered_campaign(p,f,y,splits,registry=ctx.registry,feature_schema_id="features-v1",label_schema_id="labels-v1",expected_feature_schema_id="features-v1",expected_label_schema_id="labels-v1",cost_bps=10,adapter=Other())


def test_full_development_retrain_is_distinct_reproducible_candidate_policy(tmp_path):
    from src.axiom2.research.ranker import run_registered_campaign, retrain_full_development_candidate
    from src.axiom2.research.splits import purged_walk_forward_splits
    space=({"num_leaves":3,"learning_rate":.05},)
    from src.axiom2.contracts.research_proposal import DataDependency
    from src.axiom2.research.ranker import candidate_policy_digest, development_gate_digest
    ctx=build_context(tmp_path/"store"); f,y,dates=fixture_frames(); splits=purged_walk_forward_splits(dates,train_size=8,test_size=3,horizon=2,embargo=1)
    extra=(
        DataDependency(dataset_id=candidate_policy_digest("full_development"),source="axiom2-candidate-policy",schema_version="1"),
        DataDependency(dataset_id=development_gate_digest(min_rank_ic=-1,min_net_return=-1),source="axiom2-development-gate",schema_version="1"),
    )
    p=bound_campaign_proposal("campaign-retrain",f,y,splits,space,extra=extra); ctx.registry.register_experiment(p)
    campaign=run_registered_campaign(p,f,y,splits,registry=ctx.registry,feature_schema_id="features-v1",label_schema_id="labels-v1",expected_feature_schema_id="features-v1",expected_label_schema_id="labels-v1",cost_bps=10,search_space=space)
    kwargs=dict(registry=ctx.registry,policy="full_development",min_rank_ic=-1,min_net_return=-1)
    a=retrain_full_development_candidate(p,f,y,campaign,**kwargs)
    b=retrain_full_development_candidate(p,f,y,campaign,**kwargs)
    assert a.model_artifact_hash==b.model_artifact_hash
    assert a.training_rows==len(f.join(y).dropna())
    assert a.search_space_digest==campaign.search_space_digest
    assert a.experiment_count==campaign.experiment_count


def test_candidate_policy_and_development_gate_must_be_preregistered(tmp_path):
    from src.axiom2.research.ranker import candidate_policy_digest, development_gate_digest, freeze_development_candidate, run_registered_campaign
    from src.axiom2.contracts.research_proposal import DataDependency
    space=({"num_leaves":3,"learning_rate":.05},)
    ctx=build_context(tmp_path/"store"); f,y,dates=fixture_frames()
    from src.axiom2.research.splits import purged_walk_forward_splits
    splits=purged_walk_forward_splits(dates,train_size=8,test_size=3,horizon=2,embargo=1)
    extra=(DataDependency(dataset_id=candidate_policy_digest("last_fold"),source="axiom2-candidate-policy",schema_version="1"),DataDependency(dataset_id=development_gate_digest(min_rank_ic=0.01,min_net_return=0.0),source="axiom2-development-gate",schema_version="1"))
    p=bound_campaign_proposal("campaign-gated",f,y,splits,space,extra=extra); ctx.registry.register_experiment(p)
    c=run_registered_campaign(p,f,y,splits,registry=ctx.registry,feature_schema_id="features-v1",label_schema_id="labels-v1",expected_feature_schema_id="features-v1",expected_label_schema_id="labels-v1",cost_bps=10,search_space=space)
    with pytest.raises(ValueError,match="development gate"):
        freeze_development_candidate(p,c,registry=ctx.registry,policy="last_fold",min_rank_ic=0.99,min_net_return=0.99)


def test_lightgbm_relevance_is_bounded_for_large_cross_sections():
    from src.axiom2.research.ranker import LightGBMRankerAdapter
    idx=pd.MultiIndex.from_product([[pd.Timestamp("2025-01-01",tz="UTC")],[f"S{i}" for i in range(40)]],names=["timestamp","instrument_id"])
    train=pd.DataFrame({"x":range(40),"cross_sectional_rank":[(i+1)/40 for i in range(40)]},index=idx)
    model=LightGBMRankerAdapter().fit(train[["x"]],train["cross_sectional_rank"],{"num_leaves":3,"learning_rate":.05})
    assert model is not None


def test_development_gate_can_reject_unacceptable_drawdown(tmp_path):
    from src.axiom2.research.ranker import development_gate_digest
    assert development_gate_digest(min_rank_ic=.01,min_net_return=0,max_drawdown=-.25) != development_gate_digest(min_rank_ic=.01,min_net_return=0,max_drawdown=-.50)


def test_portfolio_policy_is_preregistered_and_nonoverlapping():
    from src.axiom2.research.ranker import PortfolioPolicy, portfolio_policy_digest
    p=PortfolioPolicy(top_k=5,holding_horizon=5,rebalance_every=5,weighting="equal",overlap=False)
    assert len(portfolio_policy_digest(p))==64
    assert portfolio_policy_digest(p)!=portfolio_policy_digest(replace(p,top_k=10))

def test_development_gate_binds_fold_stability():
    from src.axiom2.research.ranker import development_gate_digest
    assert development_gate_digest(min_rank_ic=.01,min_net_return=0,max_drawdown=-.15,min_positive_fold_fraction=.6)!=development_gate_digest(min_rank_ic=.01,min_net_return=0,max_drawdown=-.15,min_positive_fold_fraction=.8)
