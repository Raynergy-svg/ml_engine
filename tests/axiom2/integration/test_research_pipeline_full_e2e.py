from dataclasses import replace
from datetime import timedelta
import hashlib

import pandas as pd
import pytest

from tests.evidence.equity_research._support import NOW, build_context
from src.evidence.canonical import canonical_bytes
from src.evidence.contracts import AuthorityRole
from src.evidence.signing import Ed25519Signer

def _synthetic_market():
    from src.axiom2.data.universe import UniverseManifest, UniverseMembership
    dates=pd.date_range("2025-01-01",periods=45,tz="UTC")
    symbols=("A","B","C","D","E","F")
    drifts=(.006,.005,.004,.003,.002,-.002)
    rows=[]; prices={}
    for symbol,drift in zip(symbols,drifts):
        values=[]
        for i,date in enumerate(dates):
            close=100.0*((1.0+drift)**i); values.append(close)
            rows.append({
                "timestamp":date,"instrument_id":symbol,"close":close,
                "volume":2_000_000+10_000*symbols.index(symbol),
                "available_to_axiom_time":date+pd.Timedelta(hours=1),
                "feature_cutoff":date+pd.Timedelta(hours=2),"sector":"fixture",
            })
        prices[symbol]=values
    ohlcv=pd.DataFrame(rows)
    benchmark=pd.DataFrame({
        "value":[100.0]*len(dates),
        "available_to_axiom_time":dates+pd.Timedelta(hours=1),
    },index=dates)
    sectors=pd.DataFrame({
        "fixture":[100.0]*len(dates),
        "available_to_axiom_time":dates+pd.Timedelta(hours=1),
    },index=dates)
    memberships=tuple(
        UniverseMembership(
            instrument_id=symbol,asset_class="equity",
            member_from=dates[0].to_pydatetime(),
            member_until=(dates[-1]+pd.Timedelta(days=2)).to_pydatetime(),
        ) for symbol in symbols
    )
    manifest=UniverseManifest(
        universe_id="synthetic-pit-six",source="fixture",evidence_id="synthetic-membership",
        schema_version="1",membership_basis="sp500",
        coverage_start=dates[0].to_pydatetime(),
        coverage_end=(dates[-1]+pd.Timedelta(days=2)).to_pydatetime(),
        memberships=memberships,
    )
    price_frame=pd.DataFrame(prices,index=dates)
    price_series=price_frame.stack(); price_series.index.names=["timestamp","instrument_id"]
    return dates,ohlcv,benchmark,sectors,manifest,price_series
def _execute_pipeline(tmp_path):
    from src.axiom2.contracts.research_proposal import (
        DataDependency, PHASE1_FEATURE_FAMILIES, ResearchProposal,
    )
    from src.axiom2.promotion.authority import (
        PromotionRule, evaluate_verified_promotion, promotion_rule_digest,
    )
    from src.axiom2.research.baselines import CostAssumption, evaluate_momentum_baseline
    from src.axiom2.research.features import build_phase1_features
    from src.axiom2.research.labels import build_forward_rank_labels
    from src.axiom2.research.ranker import (
        PortfolioPolicy, campaign_manifest_digest, candidate_policy_digest,
        development_gate_digest, freeze_development_candidate,
        portfolio_policy_digest, run_registered_campaign, search_space_digest,
    )
    from src.axiom2.research.splits import purged_walk_forward_splits

    dates,ohlcv,benchmark,sectors,manifest,prices=_synthetic_market()
    features=build_phase1_features(
        ohlcv,benchmark,sectors,manifest,
        cutoff=(dates[-1]+pd.Timedelta(hours=3)),
        families=tuple(sorted(PHASE1_FEATURE_FAMILIES)),
    ).frame
    labels=build_forward_rank_labels(prices,benchmark["value"],horizon=5)
    complete=features.join(
        labels[["forward_return","forward_excess_return","cross_sectional_rank"]],how="inner"
    ).dropna()
    eligible_dates=pd.DatetimeIndex(
        complete.index.get_level_values("timestamp").unique()
    ).sort_values()
    splits=purged_walk_forward_splits(
        eligible_dates,train_size=15,test_size=5,horizon=5,embargo=5
    )
    assert len(splits)>=2
    policy=PortfolioPolicy(
        top_k=5,holding_horizon=5,rebalance_every=5,weighting="equal",overlap=False
    )
    space=({"num_leaves":3,"learning_rate":.05},)
    rule=PromotionRule(min_excess_return=0.0)
    campaign_digest=campaign_manifest_digest(
        proposal_id="full-e2e",features=features,labels=labels,splits=splits,
        feature_schema_id="features-e2e-v1",label_schema_id="labels-e2e-v1",
        cost_bps=10,search_space=space,portfolio_policy=policy,
    )
    development_gate=development_gate_digest(
        min_rank_ic=-1.0,min_net_return=-1.0,max_drawdown=-1.0,
        min_positive_fold_fraction=0.0,
    )
    proposal=ResearchProposal(
        experiment_id="full-e2e",
        hypothesis="synthetic PIT pipeline proof only",
        universe_id=manifest.universe_id,
        feature_families=tuple(sorted(PHASE1_FEATURE_FAMILIES)),
        label_id="forward-five-session-rank",
        holding_horizon=5,
        benchmark_id="synthetic-flat-benchmark",
        cost_model_id="synthetic-10bps",
        model_id="lightgbm-ranker",
        baseline_ids=("momentum",),
        primary_metrics=("rank_ic","net_return"),
        promotion_rule_id="synthetic-zero-excess-proof",
        code_commit="a"*40,
        data_dependencies=(
            DataDependency(dataset_id="f"*64,source="synthetic-pit-fixture",schema_version="1"),
            DataDependency(dataset_id=search_space_digest(space),source="axiom2-hyperparameter-search",schema_version="1"),
            DataDependency(dataset_id=portfolio_policy_digest(policy),source="axiom2-portfolio-policy",schema_version="1"),
            DataDependency(dataset_id=candidate_policy_digest("last_fold"),source="axiom2-candidate-policy",schema_version="1"),
            DataDependency(dataset_id=development_gate,source="axiom2-development-gate",schema_version="1"),
            DataDependency(dataset_id=campaign_digest,source="axiom2-campaign-manifest",schema_version="1"),
            DataDependency(dataset_id=promotion_rule_digest(rule),source="axiom2-promotion-rule",schema_version="1"),
        ),
    )
    ctx=build_context(tmp_path/"store")
    promotion_signer=Ed25519Signer.generate()
    ctx.store.trust_store.add(promotion_signer.trusted_key(valid_from=NOW-timedelta(days=1)))
    ctx.store.authorities.register(
        actor_id="promotion_service",role=AuthorityRole.PROMOTION_SERVICE,
        key_ids=(promotion_signer.key_id,),
    )
    ctx.registry.register_experiment(proposal)
    ctx.operator.register_holdout(
        "synthetic-sacrificial",
        hashlib.sha256(b"synthetic-sacrificial-dataset").hexdigest(),
        NOW+timedelta(days=1),NOW+timedelta(days=2),
    )
    baseline=evaluate_momentum_baseline(
        features,labels,costs=CostAssumption(proposal.cost_model_id,10),top_n=5
    )
    campaign=run_registered_campaign(
        proposal,features,labels,splits,registry=ctx.registry,
        feature_schema_id="features-e2e-v1",label_schema_id="labels-e2e-v1",
        expected_feature_schema_id="features-e2e-v1",expected_label_schema_id="labels-e2e-v1",
        cost_bps=10,search_space=space,portfolio_policy=policy,
    )
    frozen=freeze_development_candidate(
        proposal,campaign,registry=ctx.registry,policy="last_fold",
        min_rank_ic=-1.0,min_net_return=-1.0,max_drawdown=-1.0,
        min_positive_fold_fraction=0.0,
    )
    assert frozen.candidate_id is not None
    ctx.evaluator.open_holdout(frozen.candidate_id,"synthetic-sacrificial")
    result_bytes=canonical_bytes({
        "candidate_id":frozen.candidate_id,
        "holdout_id":"synthetic-sacrificial",
        "net_return":float(campaign.baseline_net_return+.05),
        "cost_evidence_id":proposal.cost_model_id,
        "schema_version":1,
    })
    ctx.evaluator.consume_holdout(
        frozen.candidate_id,"synthetic-sacrificial",
        hashlib.sha256(result_bytes).hexdigest(),
    )
    decision=evaluate_verified_promotion(
        proposal=proposal,campaign=campaign,candidate_id=frozen.candidate_id,
        holdout_id="synthetic-sacrificial",holdout_result_bytes=result_bytes,
        registry=ctx.registry,promotion_rule=rule,signer=promotion_signer,
        actor_id="promotion_service",created_at=NOW,
    )
    return {
        "features":features,"labels":labels,"splits":splits,"baseline":baseline,
        "campaign":campaign,"frozen":frozen,"decision":decision,"ctx":ctx,
    }

def test_full_research_pipeline_reaches_verified_promotion_without_genuine_holdout(tmp_path):
    result=_execute_pipeline(tmp_path)
    assert not result["features"].empty and not result["labels"].empty
    assert len(result["splits"])>=2
    assert result["campaign"].holdout_accessed is False
    assert result["frozen"].candidate_id in result["ctx"].registry.snapshot().candidates
    assert result["decision"].verified is True
    assert result["decision"].status=="PROMOTED"
    assert result["ctx"].registry.snapshot().holdouts["synthetic-sacrificial"].status=="CONSUMED"

@pytest.mark.parametrize("module_name,function_name",[
    ("src.axiom2.research.features","build_phase1_features"),
    ("src.axiom2.research.labels","build_forward_rank_labels"),
    ("src.axiom2.research.splits","purged_walk_forward_splits"),
    ("src.axiom2.research.baselines","evaluate_momentum_baseline"),
    ("src.axiom2.research.ranker","run_registered_campaign"),
    ("src.axiom2.research.ranker","freeze_development_candidate"),
    ("src.axiom2.promotion.authority","evaluate_verified_promotion"),
])
def test_full_pipeline_proof_depends_on_every_required_stage(tmp_path,monkeypatch,module_name,function_name):
    import importlib
    module=importlib.import_module(module_name)
    def poisoned(*args,**kwargs):
        raise RuntimeError(f"POISONED_REQUIRED_STAGE:{function_name}")
    monkeypatch.setattr(module,function_name,poisoned)
    with pytest.raises(RuntimeError,match=f"POISONED_REQUIRED_STAGE:{function_name}"):
        _execute_pipeline(tmp_path)
