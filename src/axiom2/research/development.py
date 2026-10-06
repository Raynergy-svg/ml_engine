"""Finite Phase-1 development composition; authority is injected by the service.

A signed audit is a trusted verifier's source review, never automatic proof that
historical timestamps are true. Exact evidence bytes are checked before use.
There is no holdout reader, evaluator, candidate freeze or broker capability.
"""
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import platform

import numpy as np
import pandas as pd
from pydantic import JsonValue

from src.axiom2.contracts.research_proposal import DataDependency, ResearchProposal
from src.axiom2.data.universe import UniverseManifest, UniverseMembership, build_universe_as_of
from src.axiom2.research.features import build_phase1_features
from src.axiom2.research.labels import build_forward_rank_labels
from src.axiom2.research.splits import purged_walk_forward_splits, WalkForwardSplit
from src.axiom2.research.ranker import (
    CampaignDiagnosticsPayload, CampaignPayload, PortfolioPolicy, campaign_manifest_digest, candidate_policy_digest,
    development_gate_digest, frame_digest, portfolio_policy_digest,
    search_space_digest, split_manifest_digest, run_registered_campaign,
    _max_drawdown, _verify_campaign_evidence,
)
from src.evidence.canonical import canonical_bytes
from src.evidence.contracts import AuthorityRole, SignedEnvelope, StrictContract
from src.evidence.signing import verify_envelope

PROTOCOL='phase1-qualified-family-ablation-v1'
WINDOW=('2022-01-03','2024-06-28')
P='price_return'
ARMS={'P':(P,), 'PV':(P,'volume_liquidity'),
      'PR':(P,'cross_sectional_relative_strength'),
      'PM':(P,'market_sector_regime'),
      'FULL':(P,'volume_liquidity','cross_sectional_relative_strength','market_sector_regime')}
SEARCH=({'num_leaves':7,'learning_rate':.05},)
POLICY=PortfolioPolicy(top_k=5,holding_horizon=5,rebalance_every=5)
GATE=dict(min_rank_ic=.01,min_net_return=0.,max_drawdown=-.15,min_positive_fold_fraction=.6)
EVIDENCE=('timestamps','revisions','membership','sectors','corporate_actions','calendar','liquidity','identity')
MAX_BYTES=64*1024*1024


def digest(value):
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


class SourceAudit(StrictContract):
    actor_id: str
    classification: str
    coverage_start: str
    coverage_end: str
    panel_digest: str
    evidence_digests: dict[str,str]
    calendar_id: str
    adjustment_id: str
    availability_basis: str
    membership_basis: str
    liquidity_policy: str
    terminal_policy: str
    audit_explanation: str


class ComparisonReport(StrictContract):
    protocol: str
    plan_digest: str
    manifest_evidence: str
    arm_order: tuple[str,...]
    classification: str
    code_commit: str
    runtime: dict[str,str]
    historical_attempt_count: int
    historical_head: str | None
    arms: dict[str,JsonValue]
    paired: dict[str,JsonValue]
    fold_events: tuple[str,...]
    primary_hypothesis_pass: bool=False
    execution_enabled: bool=False
    holdout_accessed: bool=False


class RunManifest(StrictContract):
    body: dict[str,JsonValue]


class AdmissionAttempt(StrictContract):
    snapshot_id: str
    status: str
    reason: str


class FoldEvent(StrictContract):
    plan_digest: str
    arm: str
    fold: int
    status: str
    previous: str | None
    body: dict[str,JsonValue]


@dataclass
class DevelopmentPlan:
    features: dict
    labels: pd.DataFrame
    label_ends: pd.Series
    splits: tuple
    audit: SourceAudit
    source_digest: str
    exclusions: tuple
    manifest: dict
    plan_digest: str


def _read(root, name):
    path=root/name
    if root.is_symlink() or path.is_symlink() or not path.is_file():
        raise ValueError('allowlisted input must be a regular nonsymlink file')
    if path.stat().st_size>MAX_BYTES:
        raise ValueError('input exceeds bounded size')
    data=path.read_bytes()
    if len(data)>MAX_BYTES:raise ValueError('input exceeds bounded size')
    return data


def _fields(value, names, stage):
    if type(value) is not dict or set(value)!=set(names):
        raise ValueError(stage+' fields are missing or unknown')


def _time(value, stage):
    if type(value) is not str:raise ValueError(stage+' availability/time is unknown')
    t=pd.Timestamp(value)
    if pd.isna(t) or t.tzinfo is None:raise ValueError(stage+' availability/time must be aware')
    return t.tz_convert('UTC')


def _positive(value, stage, zero=False):
    if type(value) not in (float,int) or not np.isfinite(value) or value<(0 if zero else np.finfo(float).tiny):
        raise ValueError(stage+' must be finite and positive')
    return float(value)


def admit_development_bundle(root, *, store, allow_synthetic=False):
    """Service-only allowlisted bundle loader. Authenticate classification first.

    Empirical auditors must inspect publication/delivery records, source revision
    semantics, calendar completeness and PIT membership/identity/liquidity/sector
    and terminal treatment. The loader binds that independent review to bytes;
    neither caller hashes nor ingestion time can replace it.
    """
    root=Path(root)
    env=SignedEnvelope.model_validate_json(_read(root,'admission.json'))
    audit=verify_envelope(env,SourceAudit,store.trust_store)
    store.authorities.authorize_identity(actor_id=audit.actor_id,role=AuthorityRole.INDEPENDENT_VERIFIER,key_id=env.signature.key_id)
    allowed={'development'}|({'synthetic-development'} if allow_synthetic else set())
    if audit.classification not in allowed:raise ValueError('dataset classification is not admitted development')
    if (audit.coverage_start,audit.coverage_end)!=WINDOW:raise ValueError('unauthorized development window')
    if audit.availability_basis!='historical-publication-and-delivery' or audit.membership_basis!='decision-time-sp500-snapshots':
        raise ValueError('historical availability and eligibility evidence unresolved')
    if audit.terminal_policy!='resolved-opening-total-return-proxy':raise ValueError('terminal outcome policy unresolved')
    if not all((audit.calendar_id,audit.adjustment_id,audit.liquidity_policy,audit.audit_explanation)):
        raise ValueError('source audit is incomplete')
    if set(audit.evidence_digests)!=set(EVIDENCE):raise ValueError('source evidence coverage incomplete')
    for name in EVIDENCE:
        data=_read(root,name+'.txt')
        if not data.strip() or hashlib.sha256(data).hexdigest()!=audit.evidence_digests[name]:
            raise ValueError('source evidence digest mismatch')
    raw=_read(root,'panel.json')
    if hashlib.sha256(raw).hexdigest()!=audit.panel_digest:raise ValueError('panel digest mismatch')
    panel=json.loads(raw)
    _fields(panel,('sessions','bars','references','eligibility'),'panel')
    sessions=panel['sessions']
    if not sessions:raise ValueError('calendar missing')
    opens=[]; closes=[]; decisions=[]
    for session in sessions:
        _fields(session,('open','close','decision'),'calendar')
        op=_time(session['open'],'opening'); cl=_time(session['close'],'close'); de=_time(session['decision'],'decision')
        local=op.tz_convert('America/New_York')
        if not de<op<cl or local.strftime('%H:%M:%S')!='09:30:00' or de!=op-pd.Timedelta(minutes=5):
            raise ValueError('entry must follow frozen pre-open decision cutoff')
        if not WINDOW[0]<=local.strftime('%Y-%m-%d')<=WINDOW[1]:raise ValueError('calendar outside development window')
        if opens and not opens[-1]<op:raise ValueError('calendar duplicates or out of order')
        opens.append(op);closes.append(cl);decisions.append(de)
    closes=pd.DatetimeIndex(closes); decisions=pd.DatetimeIndex(decisions)
    if opens[0].tz_convert('America/New_York').strftime('%Y-%m-%d')!=WINDOW[0] or opens[-1].tz_convert('America/New_York').strftime('%Y-%m-%d')!=WINDOW[1]:
        raise ValueError('calendar coverage incomplete')
    rows=[]
    for bar in panel['bars']:
        _fields(bar,('timestamp','instrument_id','open','close','volume','available','sector','adjustment','outcome'),'bar')
        t=_time(bar['timestamp'],'bar'); avail=_time(bar['available'],'bar')
        if t not in closes:raise ValueError('paired bar calendar mismatch')
        i=closes.get_loc(t)
        if avail<t or (i+1<len(closes) and avail>decisions[i+1]):raise ValueError('bar availability is late or precedes event')
        if bar['adjustment']!=audit.adjustment_id:raise ValueError('adjustment mismatch')
        if bar['outcome']!='resolved':raise ValueError('unresolved terminal outcome')
        if not isinstance(bar['instrument_id'],str) or not bar['instrument_id'] or not isinstance(bar['sector'],str) or not bar['sector']:
            raise ValueError('identity/sector fields missing')
        rows.append(dict(timestamp=t,instrument_id=bar['instrument_id'],open=_positive(bar['open'],'opening'),close=_positive(bar['close'],'close'),volume=_positive(bar['volume'],'volume',zero=True),sector=bar['sector'],available_to_axiom_time=avail,feature_cutoff=decisions[i+1] if i+1<len(closes) else avail))
    bars=pd.DataFrame(rows).sort_values(['timestamp','instrument_id'])
    if bars.empty or bars.duplicated(['timestamp','instrument_id']).any():raise ValueError('bar coverage missing or duplicate')
    refs=[];sec_rows=[]
    for ref in panel['references']:
        _fields(ref,('timestamp','available','benchmark','benchmark_open','sectors','adjustment'),'reference')
        t=_time(ref['timestamp'],'reference');av=_time(ref['available'],'reference')
        if t not in closes:raise ValueError('paired reference calendar mismatch')
        i=closes.get_loc(t)
        if av<t or (i+1<len(closes) and av>decisions[i+1]):raise ValueError('reference availability is late')
        if ref['adjustment']!=audit.adjustment_id:raise ValueError('reference adjustment mismatch')
        if type(ref['sectors']) is not dict or not ref['sectors']:raise ValueError('sector fields missing')
        refs.append(dict(timestamp=t,value=_positive(ref['benchmark'],'benchmark'),opening=_positive(ref['benchmark_open'],'benchmark opening'),available_to_axiom_time=av))
        sec_rows.append(dict(timestamp=t,**{k:_positive(v,'sector') for k,v in ref['sectors'].items()},available_to_axiom_time=av))
    benchmark=pd.DataFrame(refs).set_index('timestamp').sort_index()
    sectors=pd.DataFrame(sec_rows).set_index('timestamp').sort_index()
    if not benchmark.index.equals(closes) or not sectors.index.equals(closes) or sectors.isna().any().any():
        raise ValueError('paired reference dates/sector coverage incomplete')
    if not set(bars.sector).issubset(sectors.columns):raise ValueError('missing input family sector coverage')
    pit={};exclusions=[]
    for snapshot in panel['eligibility']:
        _fields(snapshot,('decision','available','members'),'eligibility')
        de=_time(snapshot['decision'],'eligibility');av=_time(snapshot['available'],'eligibility')
        if de not in decisions or de in pit or av>de:raise ValueError('eligibility availability/calendar invalid')
        members={}
        for member in snapshot['members']:
            _fields(member,('instrument_id','sector','eligible','reason'),'eligibility member')
            if type(member['eligible']) is not bool or not member['reason'] or member['instrument_id'] in members:
                raise ValueError('eligibility member invalid')
            members[member['instrument_id']]=member
            if not member['eligible']:exclusions.append((de.isoformat(),member['instrument_id'],member['reason']))
        pit[de]=members
    if set(pit)!=set(decisions):raise ValueError('eligibility decision coverage incomplete')
    # Build raw historical transformations on the full source coverage, then apply
    # contemporaneous PIT membership; no whole-window survivor intersection.
    ids=tuple(sorted(bars.instrument_id.unique()))
    history_manifest=UniverseManifest(universe_id='source-history',source='admitted-history',evidence_id=audit.panel_digest,schema_version='1',membership_basis='sp500',coverage_start=closes[0].to_pydatetime(),coverage_end=(closes[-1]+pd.Timedelta(days=1)).to_pydatetime(),memberships=tuple(UniverseMembership(instrument_id=s,asset_class='equity',member_from=closes[0].to_pydatetime(),member_until=(closes[-1]+pd.Timedelta(days=1)).to_pydatetime()) for s in ids))
    all_features={arm:build_phase1_features(bars,benchmark[['value','available_to_axiom_time']],sectors,history_manifest,cutoff=bars.feature_cutoff.max(),families=families).frame for arm,families in ARMS.items()}
    # Existing label helper uses five actual calendar sessions on opening proxies.
    opening_series=bars.set_index(['timestamp','instrument_id'])['open']
    labels=build_forward_rank_labels(opening_series,benchmark['opening'],horizon=5)
    label_rows=[];feature_rows={arm:[] for arm in ARMS};keys=[];label_ends={}
    for i in range(6,len(closes)-5):
        decision=decisions[i];prior=closes[i-1]
        members=pit[decision]
        # A per-decision UniverseManifest uses no future removal information.
        universe=UniverseManifest(universe_id='pit-sp500',source='audited-decision-snapshot',evidence_id=audit.panel_digest,schema_version='1',membership_basis='sp500',coverage_start=decision.to_pydatetime(),coverage_end=(decision+pd.Timedelta(microseconds=1)).to_pydatetime(),memberships=tuple(UniverseMembership(instrument_id=s,asset_class='equity',member_from=decision.to_pydatetime(),member_until=(decision+pd.Timedelta(microseconds=1)).to_pydatetime()) for s,m in members.items() if m['eligible']))
        eligible=build_universe_as_of(decision.to_pydatetime(),universe)
        available_ids=set(bars.loc[bars.timestamp==prior,'instrument_id'])
        if not set(members).issuperset(available_ids):raise ValueError('PIT membership/coverage inventory incomplete')
        if len(eligible)<POLICY.top_k:raise ValueError('incomplete eligible cohort')
        for s in eligible:
            key=(closes[i],s);fkey=(prior,s)
            if key not in labels.index or labels.loc[key].isna().any():raise ValueError('incomplete/delisted outcome cohort')
            history=bars[(bars.instrument_id==s)&bars.timestamp.isin(closes[i-6:i])]
            if len(history)!=6 or history.iloc[-1].sector!=members[s]['sector']:
                raise ValueError('eligible history/sector coverage incomplete')
            for arm,features in all_features.items():
                if fkey not in features.index or features.loc[fkey].isna().any():raise ValueError('missing input family in common cohort')
                feature_rows[arm].append(features.loc[fkey].to_numpy())
            keys.append((decision,s));label_rows.append(labels.loc[key].to_numpy())
        label_ends[decision]=opens[i+5]
    index=pd.MultiIndex.from_tuples(keys,names=['timestamp','instrument_id'])
    features={arm:pd.DataFrame(feature_rows[arm],index=index,columns=all_features[arm].columns) for arm in ARMS}
    bound_labels=pd.DataFrame(label_rows,index=index,columns=labels.columns)
    bound_labels['cross_sectional_rank']=bound_labels['forward_excess_return'].groupby(level='timestamp').rank(pct=True,method='average')
    ends=pd.Series(label_ends).sort_index()
    if ends.empty:raise ValueError('no complete development cohorts')
    initial=purged_walk_forward_splits(ends.index,train_size=258,test_size=63,horizon=6,embargo=5)
    splits=tuple(WalkForwardSplit(tuple(t for t in split.train if ends[t]<min(split.test))[-252:],split.test) for split in initial)
    if any(len(split.train)!=252 for split in splits):raise ValueError('fixed 252-session training coverage incomplete')
    if len(splits)!=5:raise ValueError('fixed five-fold development calendar incomplete')
    exclusions=tuple(exclusions)+(('warmup','first six source sessions','fixed pre-outcome rule'),('terminal','last five decisions','label exits outside authorized window'))
    manifest=dict(protocol=PROTOCOL,source=digest(audit),features={a:frame_digest(f) for a,f in features.items()},labels=frame_digest(bound_labels),label_ends=frame_digest(ends.to_frame('exit')),splits=split_manifest_digest(splits),arms=ARMS,search=SEARCH,policy=portfolio_policy_digest(POLICY),gate=development_gate_digest(**GATE),costs=(10,25,50),candidate_policy=candidate_policy_digest('last_fold'),bootstrap=dict(replicates=9999,seed=0,block=4,lower_quantile=.025),exclusions=exclusions)
    plan=DevelopmentPlan(features,bound_labels,ends,splits,audit,digest(audit),exclusions,manifest,digest(manifest))
    validate_paired_plan(plan)
    return plan


def validate_paired_plan(plan):
    if digest(plan.audit)!=plan.source_digest or plan.source_digest!=plan.manifest['source']:raise ValueError('altered source audit')
    if tuple(plan.features)!=tuple(ARMS) or digest(plan.manifest)!=plan.plan_digest:raise ValueError('altered paired manifest')
    common=plan.features['P'].index
    if not common.equals(plan.labels.index):raise ValueError('paired labels mismatch')
    for arm,features in plan.features.items():
        if not features.index.equals(common) or frame_digest(features)!=plan.manifest['features'][arm] or not np.isfinite(features.to_numpy()).all():raise ValueError('paired dates or altered features')
    if frame_digest(plan.labels)!=plan.manifest['labels'] or frame_digest(plan.label_ends.to_frame('exit'))!=plan.manifest['label_ends']:raise ValueError('altered label manifest')
    if split_manifest_digest(plan.splits)!=plan.manifest['splits']:raise ValueError('altered split manifest/purge')
    for split in plan.splits:
        if not split.train or not split.test or set(split.train)&set(split.test) or not set(split.train+split.test).issubset(plan.label_ends.index):raise ValueError('split calendar invalid')
        if max(plan.label_ends.loc[list(split.train)])>=min(split.test):raise ValueError('actual label-end purge overlap')


def paired_bounds(values, folds):
    """Fold-stratified paired circular block bootstrap; diagnostic only."""
    values=np.asarray(values,dtype=float);folds=np.asarray(folds)
    if len(values)!=len(folds) or len(values)<4 or not np.isfinite(values).all():raise ValueError('paired diagnostic coverage incomplete')
    rng=np.random.default_rng(0); sums=np.zeros(9999)
    for fold in np.unique(folds):
        x=values[folds==fold]; n=len(x)
        if n<4:raise ValueError('fold has fewer than four paired rebalances')
        starts=rng.integers(0,n,size=(9999,(n+3)//4))
        indices=((starts[:,:,None]+np.arange(4))%n).reshape(9999,-1)[:,:n]
        sums+=x[indices].sum(axis=1)
    means=sums/len(values)
    return dict(mean=float(values.mean()),lower_97_5=float(np.quantile(means,.025)),replicates=9999,seed=0,block_rebalances=4,fold_boundaries_retained=True,limitation='Finite development diagnostic; no source-leakage repair or profit guarantee.')


def _runtime():
    import lightgbm
    import sklearn
    return dict(python=platform.python_version(),numpy=np.__version__,pandas=pd.__version__,lightgbm=lightgbm.__version__,sklearn=sklearn.__version__)


def _persist(registry, envelope):
    data=canonical_bytes(envelope); key=hashlib.sha256(data).hexdigest()
    path=registry.store.root/'development'/ (key+'.json')
    path.parent.mkdir(parents=True,exist_ok=True)
    registry.store._atomic_create_bytes(path,data)
    return key


def _signed_load(registry,key,contract):
    if len(key)!=64 or any(c not in '0123456789abcdef' for c in key):raise ValueError('invalid evidence digest')
    raw=(registry.store.root/'development'/(key+'.json')).read_bytes()
    if hashlib.sha256(raw).hexdigest()!=key:raise ValueError('report digest mismatch')
    env=SignedEnvelope.model_validate_json(raw)
    payload=verify_envelope(env,contract,registry.store.trust_store)
    registry.store.authorities.authorize_identity(actor_id=registry.actor_id,role=AuthorityRole.LOCAL_IMPORTER,key_id=env.signature.key_id)
    return payload


def run_development_comparison(plan, *, registry, code_commit):
    """Register all five exact manifests before the first fold; never refit IDs.

    The service injects the established registry; no key/trust/authority creation.
    A claim survives crashes. Recovery may inspect evidence, never silently rerun.
    """
    validate_paired_plan(plan)
    historical=registry.snapshot()
    prefix=PROTOCOL+'-'+plan.plan_digest[:16]
    if any((prefix+'-'+arm) in historical.experiments for arm in ARMS):raise ValueError('comparison already registered; no refit permitted')
    runtime=_runtime()
    from lightgbm import LGBMRanker
    effective=LGBMRanker(objective='lambdarank',random_state=0,n_estimators=20,verbosity=-1,n_jobs=1,**SEARCH[0]).get_params()
    calibration=calibrate_paired_diagnostic()
    implementation=_implementation_digest()
    manifest=dict(plan.manifest,code_commit=code_commit,runtime=runtime,effective_params=effective,calibration=calibration,implementation=implementation,historical_experiments={k:v.registration_digest for k,v in historical.experiments.items()},historical_trials={k:v.status for k,v in historical.trials.items()},historical_head=historical.head_digest,historical_attempt_count=historical.attempt_count)
    run_digest=digest(manifest)
    claim=registry.store.root/'development'/('claim-'+plan.plan_digest+'.json')
    claim.parent.mkdir(parents=True,exist_ok=True)
    if claim.exists():raise ValueError('comparison already claimed; inspect retained evidence')
    manifest_key=_persist(registry,registry.signer.sign(RunManifest(body=json.loads(canonical_bytes(manifest))),created_at=registry.store._trusted_clock()))
    registry.store._atomic_create_bytes(claim,canonical_bytes(dict(manifest_evidence=manifest_key)))
    proposals={}
    for arm,families in ARMS.items():
        identifier=prefix+'-'+arm;schema=digest(dict(families=families,columns=list(plan.features[arm].columns)))
        campaign=campaign_manifest_digest(proposal_id=identifier,features=plan.features[arm],labels=plan.labels,splits=plan.splits,feature_schema_id=schema,label_schema_id='next-open-five-exchange-sessions-v1',cost_bps=10,search_space=SEARCH,portfolio_policy=POLICY)
        dependencies=(('axiom2-admitted-development',plan.source_digest),('axiom2-development-protocol',run_digest),('axiom2-hyperparameter-search',search_space_digest(SEARCH)),('axiom2-portfolio-policy',portfolio_policy_digest(POLICY)),('axiom2-candidate-policy',candidate_policy_digest('last_fold')),('axiom2-development-gate',development_gate_digest(**GATE)),('axiom2-campaign-manifest',campaign))
        proposal=ResearchProposal(experiment_id=identifier,hypothesis='FULL improves on paired P and momentum; secondary arms diagnostic only',universe_id='pit-sp500',feature_families=families,label_id='next-open-five-exchange-sessions-v1',holding_horizon=5,benchmark_id=plan.audit.panel_digest,cost_model_id='development-comparability-10bps',model_id='lightgbm-ranker',baseline_ids=('momentum',),primary_metrics=('rank_ic','net_excess_return','positive_fold_fraction','drawdown'),promotion_rule_id='development-only-no-promotion',code_commit=code_commit,data_dependencies=tuple(DataDependency(source=s,dataset_id=d,schema_version='1') for s,d in dependencies))
        registry.register_experiment(proposal);proposals[arm]=(proposal,schema)
    arms={};events=[]; books={}; stopped=False
    for arm,(proposal,schema) in proposals.items():
        if stopped:
            arms[arm]=dict(status='NOT_RUN',experiment_id=proposal.experiment_id,reason='Earlier arm failed; finite protocol stopped')
            continue
        def record(status,fold,body):
            body=dict(body)
            artifact=body.pop('artifact_bytes',None)
            if artifact is not None:
                ad=hashlib.sha256(artifact).hexdigest();ap=registry.store.root/'development'/(ad+'.model')
                registry.store._atomic_create_bytes(ap,artifact);body['artifact_digest']=ad
            event=FoldEvent(plan_digest=plan.plan_digest,arm=arm,fold=fold,status=status,previous=events[-1] if events else None,body=body)
            events.append(_persist(registry,registry.signer.sign(event,created_at=registry.store._trusted_clock())))
        try:
            result=run_registered_campaign(proposal,plan.features[arm],plan.labels,plan.splits,registry=registry,feature_schema_id=schema,label_schema_id=proposal.label_id,expected_feature_schema_id=schema,expected_label_schema_id=proposal.label_id,cost_bps=10,search_space=SEARCH,portfolio_policy=POLICY,fold_recorder=record)
            _verify_campaign_evidence(proposal,result,registry=registry)
            ic,book,fold_ic,artifacts=result.diagnostics
            books[arm]=book
            stress={}
            for bps in (10,25,50):
                raw=[r[2]-bps/10000*r[4] for r in book];excess=[r[3]-bps/10000*r[4] for r in book];baseline=[r[6]-bps/10000*r[7] for r in book]
                stress[str(bps)]=dict(net_excess_return=float(np.mean(excess)),momentum_net_excess_return=float(np.mean(baseline)),raw_wealth=[1.]+np.cumprod(1.+np.asarray(raw)).tolist(),excess_wealth=[1.]+np.cumprod(1.+np.asarray(excess)).tolist(),raw_drawdown=_max_drawdown(raw),excess_drawdown=_max_drawdown(excess))
            checks=dict(ic=result.mean_rank_correlation>=GATE['min_rank_ic'],net=result.portfolio_net_return>=GATE['min_net_return'],drawdown=result.max_drawdown>=GATE['max_drawdown'],stability=result.positive_fold_fraction>=GATE['min_positive_fold_fraction'])
            campaign_key=_persist(registry,result.envelope)
            diagnostic_key=_persist(registry,result.diagnostics_envelope)
            arms[arm]=dict(status='SUCCEEDED',experiment_id=proposal.experiment_id,campaign_evidence=campaign_key,diagnostic_evidence=diagnostic_key,trial_id=result.selected_trial_id,trial_result_digest=result.selected_trial_result_digest,rank_ic=result.mean_rank_correlation,fold_ic=fold_ic,session_ic=ic,book=book,fold_artifacts=artifacts,positive_fold_fraction=result.positive_fold_fraction,turnover=result.turnover,concentration=.2,stress_bps=stress,gate=checks,gate_pass=all(checks.values()),gate_digest=development_gate_digest(**GATE),schema=list(plan.features[arm].columns),exclusions=plan.exclusions)
        except Exception as exc:
            reason=type(exc).__name__+': '+str(exc)[:220]
            registry.record_failure(proposal.experiment_id,reason)
            arms[arm]=dict(status='FAILED',experiment_id=proposal.experiment_id,reason=reason)
            stopped=True
    paired={}
    if not stopped:
        full=books['FULL'];price=books['P']
        if [(r[0],r[1]) for r in full]!=[(r[0],r[1]) for r in price]:raise ValueError('paired prediction dates mismatch')
        paired['FULL_minus_P']=paired_bounds([(f[3]-.001*f[4])-(p[3]-.001*p[4]) for f,p in zip(full,price)],[r[0] for r in full])
        paired['FULL_minus_momentum']=paired_bounds([(f[3]-.001*f[4])-(f[6]-.001*f[7]) for f in full],[r[0] for r in full])
    report=ComparisonReport(protocol=PROTOCOL,plan_digest=plan.plan_digest,manifest_evidence=manifest_key,arm_order=tuple(ARMS),classification=plan.audit.classification,code_commit=code_commit,runtime=runtime,historical_attempt_count=historical.attempt_count,historical_head=historical.head_digest,arms=json.loads(canonical_bytes(arms)),paired=paired,fold_events=tuple(events),primary_hypothesis_pass=bool(not stopped and arms['FULL']['gate_pass'] and all(x['lower_97_5']>0 for x in paired.values())))
    return _persist(registry,registry.signer.sign(report,created_at=registry.store._trusted_clock()))


def _report_number(value, label):
    if type(value) not in (int,float) or isinstance(value,bool) or not np.isfinite(float(value)):
        raise ValueError(label+' must be finite')
    return float(value)


def _same_report_number(left,right):
    try:
        values=(float(left),float(right))
    except (TypeError,ValueError):
        return False
    return all(np.isfinite(value) for value in values) and bool(np.isclose(*values,rtol=0.0,atol=1e-12))


def _derive_arm_evidence(arm):
    session_ic=arm.get('session_ic')
    if not isinstance(session_ic,(list,tuple)) or len(session_ic)!=315:
        raise ValueError('derived IC coverage mismatch')
    by_fold={};seen=set()
    for row in session_ic:
        if not isinstance(row,(list,tuple)) or len(row)!=3:
            raise ValueError('derived IC row malformed')
        fold,timestamp,value=row
        if type(fold) is not int or fold not in range(5) or type(timestamp) is not str or not timestamp:
            raise ValueError('derived IC row malformed')
        if (fold,timestamp) in seen:
            raise ValueError('derived IC dates duplicated')
        seen.add((fold,timestamp))
        by_fold.setdefault(fold,[]).append(_report_number(value,'session IC'))
    if set(by_fold)!=set(range(5)):
        raise ValueError('derived IC fold coverage mismatch')
    fold_ic=tuple(float(np.mean(by_fold[index])) for index in range(5))
    reported_fold_ic=arm.get('fold_ic')
    if not isinstance(reported_fold_ic,(list,tuple)) or tuple(reported_fold_ic)!=fold_ic:
        raise ValueError('derived fold IC mismatch')
    rank_ic=float(np.mean(fold_ic))
    positive=float(np.mean(np.asarray(fold_ic)>0))
    if not all((_same_report_number(arm.get('rank_ic'),rank_ic),_same_report_number(arm.get('positive_fold_fraction'),positive))):
        raise ValueError('derived IC conclusion mismatch')

    book=arm.get('book')
    if not isinstance(book,(list,tuple)) or not book:
        raise ValueError('derived portfolio evidence missing')
    for row in book:
        if not isinstance(row,(list,tuple)) or len(row)!=9:
            raise ValueError('derived portfolio row malformed')
        fold,timestamp,raw_return,excess_return,turnover,baseline_return,baseline_excess,baseline_turn,ids=row
        if type(fold) is not int or fold not in range(5) or type(timestamp) is not str or not timestamp:
            raise ValueError('derived portfolio row malformed')
        for value,label in (
            (raw_return,'raw return'),(excess_return,'excess return'),
            (turnover,'turnover'),(baseline_return,'baseline return'),
            (baseline_excess,'baseline excess'),(baseline_turn,'baseline turnover'),
        ):
            _report_number(value,label)
        if not isinstance(ids,(list,tuple)) or not ids or any(type(identifier) is not str or not identifier for identifier in ids):
            raise ValueError('derived portfolio identities malformed')
    turnover=float(np.mean([float(row[4]) for row in book]))
    if arm.get('turnover')!=turnover:
        raise ValueError('derived turnover mismatch')
    stress={}
    for bps in (10,25,50):
        raw=[float(row[2])-bps/10000*float(row[4]) for row in book]
        excess=[float(row[3])-bps/10000*float(row[4]) for row in book]
        baseline=[float(row[6])-bps/10000*float(row[7]) for row in book]
        stress[str(bps)]=dict(
            net_excess_return=float(np.mean(excess)),
            momentum_net_excess_return=float(np.mean(baseline)),
            raw_wealth=[1.]+np.cumprod(1.+np.asarray(raw)).tolist(),
            excess_wealth=[1.]+np.cumprod(1.+np.asarray(excess)).tolist(),
            raw_drawdown=_max_drawdown(raw),
            excess_drawdown=_max_drawdown(excess),
        )
    if canonical_bytes(arm.get('stress_bps'))!=canonical_bytes(stress):
        raise ValueError('cost stress mismatch')
    expected_gate=dict(
        ic=rank_ic>=GATE['min_rank_ic'],
        net=stress['10']['net_excess_return']>=GATE['min_net_return'],
        drawdown=stress['10']['excess_drawdown']>=GATE['max_drawdown'],
        stability=positive>=GATE['min_positive_fold_fraction'],
    )
    if arm.get('gate')!=expected_gate or arm.get('gate_pass')!=all(expected_gate.values()):
        raise ValueError('derived gate mismatch')
    if arm.get('gate_digest')!=development_gate_digest(**GATE):
        raise ValueError('development gate identity mismatch')
    return rank_ic,fold_ic,positive,turnover,stress


def verify_comparison_report(key, *, registry):
    report=_signed_load(registry,key,ComparisonReport)
    if report.protocol!=PROTOCOL or report.execution_enabled or report.holdout_accessed or set(report.arms)!=set(ARMS) or report.arm_order!=tuple(ARMS):raise ValueError('invalid development comparison')
    manifest=_signed_load(registry,report.manifest_evidence,RunManifest).body
    if digest({k:v for k,v in manifest.items() if k not in ("code_commit","runtime","historical_head","historical_attempt_count","effective_params","calibration","implementation","historical_experiments","historical_trials")})!=report.plan_digest:raise ValueError("manifest evidence mismatch")
    if manifest["historical_head"]!=report.historical_head or manifest["historical_attempt_count"]!=report.historical_attempt_count:raise ValueError("ignored historical attempt context")
    if manifest.get("gate")!=development_gate_digest(**GATE) or tuple(manifest.get("costs",()))!=(10,25,50):raise ValueError('development protocol identity mismatch')
    if manifest.get("bootstrap")!=dict(replicates=9999,seed=0,block=4,lower_quantile=.025):raise ValueError('development bootstrap identity mismatch')
    state=registry.snapshot()
    if state.attempt_count<report.historical_attempt_count+5:raise ValueError('ignored historical attempt')
    for identifier,registration in manifest['historical_experiments'].items():
        if identifier not in state.experiments or state.experiments[identifier].registration_digest!=registration:raise ValueError('ignored historical attempt')
    for identifier,status in manifest['historical_trials'].items():
        if identifier not in state.trials or state.trials[identifier].status!=status:raise ValueError('ignored historical trial')
    previous=None; fold_statuses={};fold_artifacts={}
    for event_key in report.fold_events:
        event=_signed_load(registry,event_key,FoldEvent)
        if event.plan_digest!=report.plan_digest or event.previous!=previous:raise ValueError('fold event chain mismatch')
        slot=(event.arm,event.fold)
        statuses=fold_statuses.setdefault(slot,[]);statuses.append(event.status)
        if event.status=='SUCCEEDED':
            if event.body['effective_params']!=manifest['effective_params']:raise ValueError('effective fit parameters changed')
            fold_artifacts[slot]=event.body['artifact_digest']
        if statuses not in (['STARTED'],['STARTED','SUCCEEDED'],['STARTED','FAILED']):raise ValueError('fold attempt sequence invalid')
        if 'artifact_digest' in event.body:
            ad=event.body['artifact_digest'];raw=(registry.store.root/'development'/(ad+'.model')).read_bytes()
            if hashlib.sha256(raw).hexdigest()!=ad:raise ValueError('fold artifact mismatch')
        previous=event_key
    for arm_name,arm in report.arms.items():
        exp=state.experiments.get(arm['experiment_id'])
        if exp is None:raise ValueError('ignored registered attempt')
        if arm['status']=='SUCCEEDED':
            campaign=_signed_load(registry,arm['campaign_evidence'],CampaignPayload)
            signed_diagnostic=_signed_load(registry,arm['diagnostic_evidence'],CampaignDiagnosticsPayload)
            if signed_diagnostic.campaign_manifest_digest!=campaign.campaign_manifest_digest or signed_diagnostic.selected_trial_id!=campaign.selected_trial_id or signed_diagnostic.selected_trial_result_digest!=campaign.selected_trial_result_digest:raise ValueError('diagnostic campaign identity mismatch')
            diagnostic=(arm['session_ic'],arm['book'],arm['fold_ic'],arm['fold_artifacts'])
            for fold,artifact in arm['fold_artifacts']:
                if fold_statuses.get((arm_name,fold))!=['STARTED','SUCCEEDED'] or fold_artifacts.get((arm_name,fold))!=artifact:raise ValueError('ignored fold attempt')
            if len(arm['fold_artifacts'])!=5:raise ValueError('incomplete fold attempts')
            if digest(diagnostic)!=signed_diagnostic.diagnostics_digest:raise ValueError('campaign diagnostic evidence mismatch')
            if campaign.proposal_id!=arm['experiment_id'] or campaign.selected_trial_id!=arm['trial_id']:raise ValueError('campaign trial identity mismatch')
            relevant=[t for t in state.trials.values() if t.experiment_id==arm['experiment_id']]
            if len(relevant)!=1:raise ValueError('ignored or extra trial attempt')
            rank_ic,fold_ic,positive,turnover,stress=_derive_arm_evidence(arm)
            if not all((_same_report_number(campaign.mean_rank_correlation,rank_ic),
                _same_report_number(campaign.portfolio_net_return,stress['10']['net_excess_return']),
                _same_report_number(campaign.baseline_net_return,stress['10']['momentum_net_excess_return']),
                _same_report_number(campaign.turnover,turnover),
                _same_report_number(campaign.max_drawdown,stress['10']['excess_drawdown']),
                _same_report_number(campaign.positive_fold_fraction,positive))):
                raise ValueError('campaign derived metric mismatch')
            if arm['rank_ic']!=campaign.mean_rank_correlation or arm['positive_fold_fraction']!=campaign.positive_fold_fraction or arm['turnover']!=campaign.turnover:raise ValueError('campaign metric mismatch')
            trial=state.trials.get(campaign.selected_trial_id)
            if (
                trial is None
                or trial.status!='SUCCEEDED'
                or trial.experiment_id!=arm['experiment_id']
                or trial.campaign_digest!=campaign.campaign_manifest_digest
                or trial.result_digest!=campaign.selected_trial_result_digest
                or arm['trial_result_digest']!=campaign.selected_trial_result_digest
            ):raise ValueError('selected trial result digest mismatch')
        elif arm['status']=='FAILED' and exp.failure_reason!=arm['reason']:raise ValueError('failed attempt missing')
    all_succeeded=all(report.arms[name]['status']=='SUCCEEDED' for name in ARMS)
    expected_paired={}
    expected_primary=False
    if all_succeeded:
        full=report.arms['FULL']['book'];price=report.arms['P']['book']
        if [(row[0],row[1]) for row in full]!=[(row[0],row[1]) for row in price]:raise ValueError('paired evidence mismatch')
        expected_paired={
            'FULL_minus_P':paired_bounds(
                [float(f[3])-.001*float(f[4])-float(p[3])+.001*float(p[4]) for f,p in zip(full,price)],
                [row[0] for row in full],
            ),
            'FULL_minus_momentum':paired_bounds(
                [float(f[3])-.001*float(f[4])-float(f[6])+.001*float(f[7]) for f in full],
                [row[0] for row in full],
            ),
        }
        expected_primary=bool(report.arms['FULL']['gate_pass'] and all(item['lower_97_5']>0 for item in expected_paired.values()))
    if report.paired!=expected_paired:raise ValueError('paired bound mismatch')
    if report.primary_hypothesis_pass!=expected_primary:raise ValueError('primary hypothesis mismatch')
    return report

def handle_development_request(request, *, registry, bundles, code_commit, allow_synthetic=False):
    """Trusted service composition. Only configured snapshot IDs are resolvable.

    `bundles` and authority context come from service configuration, never the
    request. No import path, credential, proposal, file path or holdout ID input.
    Every failed admission is retained separately from registered fitted trials.
    """
    _fields(request,('action','snapshot_id'),'request')
    action=request['action'];identifier=request['snapshot_id']
    if type(identifier) is not str or identifier not in bundles:raise ValueError('snapshot is outside service development allowlist')
    if action not in ('preflight','run'):raise ValueError('unsupported bounded development action')
    try:
        plan=admit_development_bundle(bundles[identifier],store=registry.store,allow_synthetic=allow_synthetic)
    except Exception as exc:
        event=AdmissionAttempt(snapshot_id=identifier,status='REJECTED',reason=type(exc).__name__+': '+str(exc)[:220])
        _persist(registry,registry.signer.sign(event,created_at=registry.store._trusted_clock()))
        raise
    if action=='preflight':
        return dict(status='ADMITTED',plan_digest=plan.plan_digest,classification=plan.audit.classification,arms=tuple(ARMS),folds=len(plan.splits),execution_enabled=False,holdout_accessed=False)
    key=run_development_comparison(plan,registry=registry,code_commit=code_commit)
    report=verify_comparison_report(key,registry=registry)
    return dict(status='COMPLETE' if all(a['status']=='SUCCEEDED' for a in report.arms.values()) else 'STOPPED',report_digest=key,report=report.model_dump(mode='json'),execution_enabled=False,holdout_accessed=False)



def calibrate_paired_diagnostic():
    rng=np.random.default_rng(11);folds=[i//12 for i in range(60)]
    rejected=sum(paired_bounds(rng.normal(0,.02,60),folds)['lower_97_5']>0 for _ in range(40))
    if rejected>4 or paired_bounds([0.]*60,folds)['lower_97_5']!=0 or paired_bounds([.01]*60,folds)['lower_97_5']<=0:
        raise ValueError('synthetic diagnostic calibration failed')
    return dict(null_seed=11,null_trials=40,null_rejections=rejected,max_rejections=4,zero_and_positive_controls=True,limitations='Finite synthetic IID-null calibration; not universal coverage under arbitrary dependence.')


def _implementation_digest():
    root=Path(__file__).resolve().parents[3]
    policy=json.loads((root/'config/axiom2/research_boundary.json').read_bytes())
    for name,expected in policy['source_sha256'].items():
        if hashlib.sha256((root/name).read_bytes()).hexdigest()!=expected:
            raise ValueError('research implementation differs from audited source inventory')
    return digest(policy['source_sha256'])



def serve_development_stdio(reader, writer, *, registry, bundles, code_commit, allow_synthetic=False):
    """One bounded request on a pre-established service pipe; no authority setup."""
    line=reader.readline(65537)
    if not line or len(line)>65536 or not line.endswith('\n'):
        raise ValueError('bounded service request is incomplete')
    try:
        response=handle_development_request(json.loads(line),registry=registry,bundles=bundles,code_commit=code_commit,allow_synthetic=allow_synthetic)
    except Exception as exc:
        response=dict(status='REJECTED',reason=type(exc).__name__+': '+str(exc)[:220],execution_enabled=False,holdout_accessed=False)
    writer.write(json.dumps(response,allow_nan=False,separators=(',',':'))+'\n')
    writer.flush()
