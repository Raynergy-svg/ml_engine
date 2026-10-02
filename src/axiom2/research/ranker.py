"""Model-agnostic preregistered campaign; LightGBM is the only Phase-1 adapter."""
from abc import ABC, abstractmethod
from dataclasses import dataclass
import hashlib, json
import numpy as np
import pandas as pd
from src.axiom2.contracts.research_proposal import validate_research_proposal
from src.axiom2.research.baselines import CostAssumption, evaluate_momentum_baseline
from src.evidence.canonical import canonical_bytes
from src.evidence.contracts import AuthorityRole, StrictContract, SignedEnvelope
from src.evidence.signing import verify_envelope

class ModelResearchAdapter(ABC):
    model_family: str
    schema_version: str="1"
    @abstractmethod
    def fit(self, features: pd.DataFrame, target_rank: pd.Series, params: dict): ...
    @abstractmethod
    def predict_scores(self, model, features: pd.DataFrame): ...
    @abstractmethod
    def serialize_artifact(self, model) -> bytes: ...

LIGHTGBM_ADAPTER_IMPLEMENTATION_ID=hashlib.sha256(canonical_bytes({
    "model_family":"lightgbm","adapter_contract":"rank-features-target-separated","schema_version":1,
})).hexdigest()

class LightGBMRankerAdapter(ModelResearchAdapter):
    model_family="lightgbm"
    implementation_id=LIGHTGBM_ADAPTER_IMPLEMENTATION_ID
    def fit(self,features,target_rank,params):
        from lightgbm import LGBMRanker
        features=features.sort_index()
        target_rank=target_rank.reindex(features.index)
        groups=features.groupby(level="timestamp",sort=True).size().to_list()
        relevance=(target_rank.clip(0,1)*30).round().astype(int)
        model=LGBMRanker(objective="lambdarank",random_state=0,n_estimators=20,verbosity=-1,n_jobs=1,**params)
        model.fit(features,relevance,group=groups); return model
    def predict_scores(self,model,features): return model.predict(features)
    def serialize_artifact(self,model): return model.booster_.model_to_string().encode("utf-8")

_AUTHORIZED_LIGHTGBM_METHODS={
    "fit":LightGBMRankerAdapter.fit,
    "predict_scores":LightGBMRankerAdapter.predict_scores,
    "serialize_artifact":LightGBMRankerAdapter.serialize_artifact,
}

def _validate_phase1_adapter(adapter):
    if type(adapter) is not LightGBMRankerAdapter or adapter.model_family!="lightgbm":
        raise ValueError("Phase-1 adapter implementation is not authorized")
    if getattr(adapter,"__dict__",{}):
        raise ValueError("Phase-1 adapter implementation has instance overrides")
    for name,expected in _AUTHORIZED_LIGHTGBM_METHODS.items():
        bound=getattr(adapter,name)
        if getattr(bound,"__func__",None) is not expected:
            raise ValueError("Phase-1 adapter implementation method identity changed")
    if adapter.implementation_id!=LIGHTGBM_ADAPTER_IMPLEMENTATION_ID:
        raise ValueError("Phase-1 adapter implementation identity changed")


@dataclass(frozen=True)
class PortfolioPolicy:
    top_k: int
    holding_horizon: int
    rebalance_every: int
    weighting: str = "equal"
    overlap: bool = False
    def __post_init__(self):
        if min(self.top_k,self.holding_horizon,self.rebalance_every)<=0 or self.weighting!="equal" or self.overlap:
            raise ValueError("unsupported Phase-1 portfolio policy")
        if not self.overlap and self.rebalance_every < self.holding_horizon:
            raise ValueError("non-overlap policy cannot rebalance before holding horizon")

def portfolio_policy_digest(policy):
    return hashlib.sha256(canonical_bytes({"top_k":policy.top_k,"holding_horizon":policy.holding_horizon,"rebalance_every":policy.rebalance_every,"weighting":policy.weighting,"overlap":policy.overlap,"schema_version":1})).hexdigest()

class CampaignPayload(StrictContract):
    artifact_hash: str
    model_artifact_hash: str
    proposal_id: str
    model_family: str
    search_space_digest: str
    feature_digest: str
    label_digest: str
    split_digest: str
    campaign_manifest_digest: str
    selected_trial_id: str
    selected_trial_result_digest: str
    best_params_digest: str
    experiment_count: int
    mean_rank_correlation: float
    portfolio_net_return: float
    baseline_net_return: float
    turnover: float
    max_drawdown: float
    positive_fold_fraction: float

@dataclass(frozen=True)
class CampaignResult:
    model_type: str; experiment_count: int; mean_rank_correlation: float; portfolio_net_return: float; baseline_net_return: float; turnover: float; max_drawdown: float
    regime_slices: tuple[tuple[str,float],...]; artifact_hash: str; search_space_digest: str; feature_digest: str; label_digest: str; split_digest: str; campaign_manifest_digest: str; selected_trial_id: str; selected_trial_result_digest: str; model_artifact_hash: str; model_artifact_bytes: bytes; best_params: tuple[tuple[str, object], ...]; positive_fold_fraction: float
    holdout_accessed: bool=False; envelope: SignedEnvelope|None=None

def search_space_digest(search_space): return hashlib.sha256(canonical_bytes(tuple(dict(x) for x in search_space))).hexdigest()
def frame_digest(frame):
    normalized=frame.sort_index().sort_index(axis=1); return hashlib.sha256(normalized.to_csv(index=True,date_format="%Y-%m-%dT%H:%M:%S.%f%z").encode()).hexdigest()
def split_manifest_digest(splits):
    payload=[{"train":[x.isoformat() for x in s.train],"test":[x.isoformat() for x in s.test]} for s in splits]; return hashlib.sha256(canonical_bytes(payload)).hexdigest()

def campaign_manifest_digest(*,proposal_id,features,labels,splits,feature_schema_id,label_schema_id,cost_bps,search_space,portfolio_policy):
    return hashlib.sha256(canonical_bytes({
        "proposal_id":proposal_id,
        "feature_digest":frame_digest(features),"label_digest":frame_digest(labels),
        "split_digest":split_manifest_digest(splits),"feature_schema_id":feature_schema_id,
        "label_schema_id":label_schema_id,"cost_bps":float(cost_bps),
        "search_space_digest":search_space_digest(search_space),
        "portfolio_policy_digest":portfolio_policy_digest(portfolio_policy),
        "adapter_implementation_id":LIGHTGBM_ADAPTER_IMPLEMENTATION_ID,
        "schema_version":1,
    })).hexdigest()

def _score(pred,truth):
    if len(pred)<2:return 0.0
    a=pd.Series(pred).rank().to_numpy(); b=pd.Series(truth).rank().to_numpy()
    if np.ptp(a)==0 or np.ptp(b)==0:return 0.0
    value=np.corrcoef(a,b)[0,1]; return 0.0 if np.isnan(value) else float(value)

def _mean_cross_sectional_ic(scored):
    values=[_score(group["_pred"].to_numpy(),group["cross_sectional_rank"].to_numpy()) for _,group in scored.groupby(level="timestamp",sort=True)]
    return float(np.mean(values)) if values else 0.0

def _post_return_weights(ids,returns):
    values={identifier:(1.0+float(ret))/len(ids) for identifier,ret in zip(ids,returns)}
    total=sum(values.values())
    if total<=0:return {}
    return {identifier:value/total for identifier,value in values.items()}

def _turnover_to_equal(previous_weights,current_ids):
    current=tuple(current_ids); target={identifier:1.0/len(current) for identifier in current}
    keys=set(previous_weights)|set(target)
    return .5*sum(abs(target.get(key,0.0)-previous_weights.get(key,0.0)) for key in keys)

def _global_rebalance_dates(all_dates,splits,interval):
    ordered=tuple(pd.DatetimeIndex(all_dates).sort_values().unique())
    position={date:i for i,date in enumerate(ordered)}
    tests=sorted({date for split in splits for date in split.test})
    if not tests:return ()
    anchor=position[tests[0]]
    return tuple(date for date in tests if (position[date]-anchor)%interval==0)

def _validate_campaign_splits(all_dates,splits,*,horizon):
    ordered=tuple(pd.DatetimeIndex(all_dates).sort_values().unique()); position={date:i for i,date in enumerate(ordered)}
    seen_test=set()
    for split in splits:
        if not split.train or not split.test: raise ValueError("campaign split cannot be empty")
        train=set(split.train); test=set(split.test)
        if train & test: raise ValueError("campaign train/test overlap")
        if seen_test & test: raise ValueError("campaign test folds overlap")
        if any(date not in position for date in train|test): raise ValueError("campaign split date missing from inputs")
        last=max(train); first=min(test)
        if position[last] >= position[first]-horizon:
            raise ValueError("campaign split purge is insufficient for label horizon")
        seen_test.update(test)

def _select_complete_top_k(group,k):
    if len(group)<k: raise ValueError("campaign top-k cohort is incomplete")
    return group.nlargest(k,"_pred")

def _max_drawdown(returns):
    wealth=np.concatenate(([1.0],np.cumprod(1.0+np.asarray(returns,dtype=float))))
    peak=np.maximum.accumulate(wealth)
    return float(np.min(wealth/peak-1.0))

def run_registered_campaign(proposal,features,labels,splits,*,registry,feature_schema_id,label_schema_id,expected_feature_schema_id,expected_label_schema_id,cost_bps,search_space=({"num_leaves":7,"learning_rate":.05},),adapter=None,portfolio_policy=None):
    validate_research_proposal(proposal)
    registered=registry.snapshot().experiments.get(proposal.experiment_id)
    if registered is None or registered.proposal!=proposal or registered.failure_reason is not None:
        raise ValueError("proposal must remain registered, active, and exact before campaign")
    if feature_schema_id!=expected_feature_schema_id or label_schema_id!=expected_label_schema_id:
        raise ValueError("feature/label schema mismatch")
    if isinstance(cost_bps,bool) or not isinstance(cost_bps,(int,float)) or not np.isfinite(float(cost_bps)) or cost_bps<0 or not search_space:
        raise ValueError("cost/search space must be explicit and finite")
    portfolio_policy=portfolio_policy or PortfolioPolicy(top_k=1,holding_horizon=1,rebalance_every=1)
    if portfolio_policy.holding_horizon!=proposal.holding_horizon and portfolio_policy.top_k!=1:
        raise ValueError("portfolio holding horizon differs from preregistered proposal")
    pdig=portfolio_policy_digest(portfolio_policy)
    if not any(d.source=="axiom2-portfolio-policy" and d.dataset_id==pdig and d.schema_version=="1" for d in proposal.data_dependencies):
        if portfolio_policy.top_k!=1 or portfolio_policy.holding_horizon!=1 or portfolio_policy.rebalance_every!=1:
            raise ValueError("portfolio policy was not preregistered")
    adapter=adapter or LightGBMRankerAdapter()
    _validate_phase1_adapter(adapter)
    search=tuple(dict(x) for x in search_space)
    search_digest=search_space_digest(search)
    campaign_digest=campaign_manifest_digest(
        proposal_id=proposal.experiment_id,features=features,labels=labels,splits=splits,
        feature_schema_id=feature_schema_id,label_schema_id=label_schema_id,cost_bps=cost_bps,
        search_space=search,portfolio_policy=portfolio_policy,
    )
    if not any(d.source=="axiom2-campaign-manifest" and d.dataset_id==campaign_digest and d.schema_version=="1" for d in proposal.data_dependencies):
        raise ValueError("exact campaign manifest was not preregistered")
    preregistered={d.dataset_id for d in proposal.data_dependencies if d.source=="axiom2-hyperparameter-search" and d.schema_version=="1"}
    if search_digest not in preregistered:
        raise ValueError("exact hyperparameter space was not preregistered search evidence")
    required_labels={"forward_return","forward_excess_return","cross_sectional_rank"}
    if not required_labels.issubset(labels.columns):
        raise ValueError("campaign labels require forward_return, forward_excess_return and cross_sectional_rank")
    fd,ld,sd=frame_digest(features),frame_digest(labels),split_manifest_digest(splits)
    joined=features.join(labels[sorted(required_labels)],how="inner").dropna()
    feature_cols=list(features.columns)
    all_dates=pd.DatetimeIndex(joined.index.get_level_values("timestamp").unique()).sort_values()
    _validate_campaign_splits(all_dates,splits,horizon=portfolio_policy.holding_horizon)
    rebalance_dates=frozenset(_global_rebalance_dates(all_dates,splits,portfolio_policy.rebalance_every))
    candidates=[]
    for config_index,params in enumerate(search):
        config_digest=hashlib.sha256(canonical_bytes({"params":params,"schema_version":1})).hexdigest()
        trial=registry.record_trial_started(proposal.experiment_id,campaign_digest,config_index,config_digest)
        if trial.status=="FAILED":
            continue
        scores=[]; rets=[]; turns=[]; prior_weights={}; curve=[]; fold_means=[]; final_model=None; used_dates=[]
        try:
            for split in splits:
                train=joined[joined.index.get_level_values("timestamp").isin(split.train)].sort_index()
                test=joined[joined.index.get_level_values("timestamp").isin(split.test)].sort_index()
                if train.empty or test.empty: continue
                train_features=train[feature_cols].copy()
                train_target=train["cross_sectional_rank"].copy()
                test_features=test[feature_cols].copy()
                final_model=adapter.fit(train_features,train_target,params)
                pred=adapter.predict_scores(final_model,test_features)
                scored=test.assign(_pred=pred)
                fold_score=_mean_cross_sectional_ic(scored)
                scores.append(fold_score); fold_means.append(fold_score)
                for timestamp,g in scored.groupby(level="timestamp",sort=True):
                    if timestamp not in rebalance_dates: continue
                    pick=_select_complete_top_k(g,portfolio_policy.top_k)
                    ids=tuple(pick.index.get_level_values("instrument_id"))
                    turn=_turnover_to_equal(prior_weights,ids)
                    ret=float(pick["forward_excess_return"].mean())-(float(cost_bps)/10000.)*turn
                    rets.append(ret); turns.append(turn); curve.append(ret); used_dates.append(timestamp)
                    prior_weights=_post_return_weights(ids,pick["forward_return"].to_numpy())
            if final_model is None or not rets:
                raise ValueError("configuration produced no evaluable trained portfolio")
            config_model_bytes=adapter.serialize_artifact(final_model)
            config_model_hash=hashlib.sha256(config_model_bytes).hexdigest()
            score=float(np.mean(scores)) if scores else -1.
            positive=float(np.mean(np.asarray(fold_means)>0)) if fold_means else 0.0
            result_digest=hashlib.sha256(canonical_bytes({
                "campaign_digest":campaign_digest,"config_digest":config_digest,
                "score":score,"returns":tuple(rets),"turnover":tuple(turns),
                "model_artifact_hash":config_model_hash,"positive_fold_fraction":positive,
                "schema_version":1,
            })).hexdigest()
            registry.record_trial_succeeded(trial.trial_id,result_digest)
            candidates.append((score,params,rets,turns,curve,final_model,positive,tuple(used_dates),config_model_bytes,config_model_hash,trial.trial_id,result_digest))
        except Exception as exc:
            message=str(exc).strip().replace("\x00","")[:220]
            registry.record_trial_failed(trial.trial_id,f"{type(exc).__name__}: {message}")
            continue
    if not candidates:
        raise ValueError("campaign produced no successful registered configurations")
    score,params,rets,turns,curve,model,positive_fold_fraction,used_dates,model_bytes,model_hash,selected_trial_id,selected_trial_result_digest=max(candidates,key=lambda x:(x[0],json.dumps(x[1],sort_keys=True)))
    net=float(np.mean(rets)); turnover=float(np.mean(turns)); dd=_max_drawdown(curve)
    baseline_result=evaluate_momentum_baseline(features,labels,costs=CostAssumption(proposal.cost_model_id,float(cost_bps)),top_n=portfolio_policy.top_k,evaluation_dates=used_dates)
    baseline=baseline_result.net_return
    regimes=(("all",net),)
    best_params_digest=hashlib.sha256(canonical_bytes(params)).hexdigest()
    summary={"proposal":proposal.experiment_id,"model_family":adapter.model_family,"search":search,"best":params,"score":score,"net":net,"baseline":baseline,"turnover":turnover,"max_drawdown":dd,"regimes":regimes,"feature_digest":fd,"label_digest":ld,"split_digest":sd,"campaign_manifest_digest":campaign_digest,"model_artifact_hash":model_hash}
    artifact=hashlib.sha256(canonical_bytes(summary)).hexdigest()
    payload=CampaignPayload(
        artifact_hash=artifact,model_artifact_hash=model_hash,proposal_id=proposal.experiment_id,
        model_family=adapter.model_family,search_space_digest=search_digest,feature_digest=fd,
        label_digest=ld,split_digest=sd,campaign_manifest_digest=campaign_digest,
        selected_trial_id=selected_trial_id,selected_trial_result_digest=selected_trial_result_digest,
        best_params_digest=best_params_digest,experiment_count=len(search),mean_rank_correlation=score,
        portfolio_net_return=net,baseline_net_return=baseline,turnover=turnover,max_drawdown=dd,
        positive_fold_fraction=positive_fold_fraction,
    )
    envelope=registry.signer.sign(payload,created_at=registry.store._trusted_clock())
    return CampaignResult(adapter.model_family,len(search),score,net,baseline,turnover,dd,regimes,artifact,search_digest,fd,ld,sd,campaign_digest,selected_trial_id,selected_trial_result_digest,model_hash,model_bytes,tuple(sorted(params.items())),positive_fold_fraction,False,envelope)


@dataclass(frozen=True)
class FrozenResearchCandidate:
    model_family: str
    model_artifact_hash: str
    model_artifact_bytes: bytes
    training_rows: int
    feature_digest: str
    label_digest: str
    search_space_digest: str
    experiment_count: int
    candidate_id: str | None = None

def candidate_policy_digest(policy):
    if policy not in {"last_fold","full_development"}:
        raise ValueError("unsupported candidate policy")
    return hashlib.sha256(canonical_bytes({"policy":policy,"schema_version":1})).hexdigest()

def development_gate_digest(*,min_rank_ic,min_net_return,max_drawdown=-1.0,min_positive_fold_fraction=0.0):
    values=(min_rank_ic,min_net_return,max_drawdown,min_positive_fold_fraction)
    if any(isinstance(value,bool) or not isinstance(value,(int,float)) or not np.isfinite(float(value)) for value in values):
        raise ValueError("development gate thresholds must be finite numbers")
    return hashlib.sha256(canonical_bytes({
        "min_rank_ic":float(min_rank_ic),"min_net_return":float(min_net_return),
        "max_drawdown":float(max_drawdown),"min_positive_fold_fraction":float(min_positive_fold_fraction),
        "schema_version":1,
    })).hexdigest()

def _require_candidate_gate(proposal,campaign,*,policy,min_rank_ic,min_net_return,max_drawdown,min_positive_fold_fraction):
    deps={(d.source,d.dataset_id,d.schema_version) for d in proposal.data_dependencies}
    pdigest=candidate_policy_digest(policy)
    gdigest=development_gate_digest(
        min_rank_ic=min_rank_ic,min_net_return=min_net_return,max_drawdown=max_drawdown,
        min_positive_fold_fraction=min_positive_fold_fraction,
    )
    if ("axiom2-candidate-policy",pdigest,"1") not in deps:
        raise ValueError("candidate construction policy was not preregistered")
    if ("axiom2-development-gate",gdigest,"1") not in deps:
        raise ValueError("development gate was not preregistered")
    if (
        campaign.mean_rank_correlation < min_rank_ic
        or campaign.portfolio_net_return < min_net_return
        or campaign.max_drawdown < max_drawdown
        or campaign.positive_fold_fraction < min_positive_fold_fraction
    ):
        raise ValueError("development gate failed; candidate cannot be frozen")

def _verify_campaign_evidence(proposal,campaign,*,registry):
    registered=registry.snapshot().experiments.get(proposal.experiment_id)
    if registered is None or registered.proposal!=proposal or registered.failure_reason is not None:
        raise ValueError("candidate requires an active exact registered proposal")
    if campaign.envelope is None:
        raise ValueError("candidate requires signed campaign evidence")
    payload=verify_envelope(campaign.envelope,CampaignPayload,registry.store.trust_store)
    registry.store.authorities.authorize_identity(
        actor_id=registry.actor_id,role=AuthorityRole.LOCAL_IMPORTER,
        key_id=campaign.envelope.signature.key_id,
    )
    metrics=(
        campaign.mean_rank_correlation,campaign.portfolio_net_return,campaign.baseline_net_return,
        campaign.turnover,campaign.max_drawdown,campaign.positive_fold_fraction,
    )
    if any(not np.isfinite(float(value)) for value in metrics):
        raise ValueError("campaign metrics must be finite")
    expected={
        "artifact_hash":campaign.artifact_hash,
        "model_artifact_hash":campaign.model_artifact_hash,
        "proposal_id":proposal.experiment_id,
        "model_family":campaign.model_type,
        "search_space_digest":campaign.search_space_digest,
        "feature_digest":campaign.feature_digest,
        "label_digest":campaign.label_digest,
        "split_digest":campaign.split_digest,
        "campaign_manifest_digest":campaign.campaign_manifest_digest,
        "selected_trial_id":campaign.selected_trial_id,
        "selected_trial_result_digest":campaign.selected_trial_result_digest,
        "best_params_digest":hashlib.sha256(canonical_bytes(dict(campaign.best_params))).hexdigest(),
        "experiment_count":campaign.experiment_count,
        "mean_rank_correlation":campaign.mean_rank_correlation,
        "portfolio_net_return":campaign.portfolio_net_return,
        "baseline_net_return":campaign.baseline_net_return,
        "turnover":campaign.turnover,
        "max_drawdown":campaign.max_drawdown,
        "positive_fold_fraction":campaign.positive_fold_fraction,
    }
    for name,value in expected.items():
        if getattr(payload,name)!=value:
            raise ValueError(f"signed campaign evidence mismatch: {name}")
    if hashlib.sha256(campaign.model_artifact_bytes).hexdigest()!=campaign.model_artifact_hash:
        raise ValueError("model artifact hash does not match actual bytes")
    if campaign.holdout_accessed:
        raise ValueError("development campaign must not access sealed holdout")
    state=registry.snapshot()
    trials=tuple(
        trial for trial in state.trials.values()
        if trial.experiment_id==proposal.experiment_id and trial.campaign_digest==campaign.campaign_manifest_digest
    )
    if len(trials)!=campaign.experiment_count or any(trial.status=="STARTED" for trial in trials):
        raise ValueError("campaign trial evidence is incomplete")
    if not any(trial.status=="SUCCEEDED" for trial in trials):
        raise ValueError("campaign has no successful registered trial")
    selected=state.trials.get(campaign.selected_trial_id)
    if selected is None or selected.status!="SUCCEEDED" or selected.result_digest!=campaign.selected_trial_result_digest:
        raise ValueError("selected campaign trial evidence does not resolve")
    return payload

def retrain_full_development_candidate(
    proposal,features,labels,campaign,*,registry,policy,min_rank_ic,min_net_return,
    max_drawdown=-1.0,min_positive_fold_fraction=0.0,adapter=None,
):
    _verify_campaign_evidence(proposal,campaign,registry=registry)
    _require_candidate_gate(
        proposal,campaign,policy=policy,min_rank_ic=min_rank_ic,min_net_return=min_net_return,
        max_drawdown=max_drawdown,min_positive_fold_fraction=min_positive_fold_fraction,
    )
    if policy!="full_development":
        raise ValueError("full development retrain requires full_development candidate policy")
    adapter=adapter or LightGBMRankerAdapter()
    if type(adapter) is not LightGBMRankerAdapter or adapter.model_family!=campaign.model_type:
        raise ValueError("candidate model adapter differs from authorized campaign adapter")
    if frame_digest(features)!=campaign.feature_digest or frame_digest(labels)!=campaign.label_digest:
        raise ValueError("development inputs changed after campaign")
    joined=features.join(labels[["cross_sectional_rank"]],how="inner").dropna().sort_index()
    feature_cols=list(features.columns)
    model=adapter.fit(joined[feature_cols],joined["cross_sectional_rank"],dict(campaign.best_params))
    artifact=adapter.serialize_artifact(model)
    digest=hashlib.sha256(artifact).hexdigest()
    registered_candidate=registry.freeze_candidate(proposal.experiment_id,digest,proposal.code_commit)
    return FrozenResearchCandidate(
        adapter.model_family,digest,artifact,len(joined),campaign.feature_digest,campaign.label_digest,
        campaign.search_space_digest,campaign.experiment_count,registered_candidate.candidate_id,
    )

def freeze_development_candidate(
    proposal,campaign,*,registry,policy,min_rank_ic,min_net_return,max_drawdown=-1.0,
    min_positive_fold_fraction=0.0,
):
    _verify_campaign_evidence(proposal,campaign,registry=registry)
    _require_candidate_gate(
        proposal,campaign,policy=policy,min_rank_ic=min_rank_ic,min_net_return=min_net_return,
        max_drawdown=max_drawdown,min_positive_fold_fraction=min_positive_fold_fraction,
    )
    if policy!="last_fold":
        raise ValueError("full_development requires explicit retrain inputs")
    registered_candidate=registry.freeze_candidate(
        proposal.experiment_id,campaign.model_artifact_hash,proposal.code_commit
    )
    return FrozenResearchCandidate(
        campaign.model_type,campaign.model_artifact_hash,campaign.model_artifact_bytes,0,
        campaign.feature_digest,campaign.label_digest,campaign.search_space_digest,
        campaign.experiment_count,registered_candidate.candidate_id,
    )
