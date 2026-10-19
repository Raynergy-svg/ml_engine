"""Read-only resolution of immutable artifacts and frozen offline session pointers.

Opaque model/build bytes are hashed, never loaded or executed. Only previously
consumed result bytes are accepted; this module cannot open a sealed dataset.
Service classification is authenticated metadata, not independent proof of
source truth. No live shadow, provider or operational qualification is issued.
"""
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
import hashlib,json,math,os,stat,csv,io
from decimal import Decimal,ROUND_FLOOR
from typing import Literal
from pydantic import Field
from src.evidence.contracts import StrictContract,SignedEnvelope
from src.evidence.canonical import canonical_bytes
from src.evidence.hashing import content_digest
from src.evidence.signing import verify_envelope
from src.evidence.store import EvidenceStore,StoreCorruptionError
from src.evidence.equity_research.experiment_registry import reconstruct_research
from src.axiom2.execution.resolution import _record
from src.axiom2.portfolio.contracts import RiskPolicy
from src.axiom2.portfolio.authority import risk_policy_digest

class ArtifactBundle(StrictContract):
    candidate_id:str
    registry_head_digest:str
    proposal_digest:str
    model_digest:str
    campaign_envelope_digest:str
    campaign_manifest_digest:str
    campaign_portfolio_digest:str
    campaign_summary_digest:str
    selected_trial_result_digest:str
    search_space_digest:str
    best_params_digest:str
    holdout_id:str
    holdout_result_digest:str
    classification_digest:str
    promotion_rule_digest:str
    promotion_digest:str
    calibration_digest:str
    risk_policy_digest:str
    build_digest:str

# Version-1 wire schema of research CampaignPayload. Kept here so the execution
# artifact never imports ranker/training code. Contract parity is tested.
class CampaignPayload(StrictContract):
    artifact_hash:str
    model_artifact_hash:str
    proposal_id:str
    model_family:str
    search_space_digest:str
    feature_digest:str
    label_digest:str
    split_digest:str
    campaign_manifest_digest:str
    selected_trial_id:str
    selected_trial_result_digest:str
    best_params_digest:str
    experiment_count:int
    mean_rank_correlation:float
    portfolio_net_return:float
    baseline_net_return:float
    turnover:float
    max_drawdown:float
    positive_fold_fraction:float

class PromotionDecisionPayload(StrictContract):
    status:str
    candidate_id:str
    decision_hash:str
    reasons:tuple[str,...]
    verified:bool=False
    evidence_digest:str|None=None

class HoldoutClassification(StrictContract):
    actor_id:str
    candidate_id:str
    holdout_id:str
    dataset_digest:str
    result_digest:str
    consumption_digest:str
    source_kind:Literal['SYNTHETIC','DEVELOPMENT','GENUINE_HOLDOUT']

class CalibrationArtifact(StrictContract):
    actor_id:str
    candidate_id:str
    model_digest:str
    promotion_digest:str
    campaign_manifest_digest:str
    cost_evidence_id:str
    source_kind:Literal['SYNTHETIC','DEVELOPMENT','LIVE_READONLY']
    evidence_digest:str
    expected_net_edge_bps:int=Field(gt=0)

class BuildArtifact(StrictContract):
    actor_id:str
    code_commit:str
    artifact_digest:str
    profile:Literal['execution-shadow-disabled']='execution-shadow-disabled'
    execution_enabled:Literal[False]=False
    capital_authorized:Literal[False]=False

class SessionArtifactPointer(StrictContract):
    actor_id:str
    account_alias:str
    session_id:str
    session_open:datetime
    session_close:datetime
    bundle_digest:str
    candidate_id:str
    model_digest:str
    promotion_digest:str
    calibration_digest:str
    risk_policy_digest:str
    build_digest:str
    source_kind:Literal['SYNTHETIC','DEVELOPMENT','LIVE_SHADOW']

class ArtifactResolution(StrictContract):
    candidate_id:str
    candidate_digest:str
    model_digest:str
    promotion_digest:str
    promotion_decision_hash:str
    calibration_digest:str
    calibration_payload_digest:str
    risk_policy_digest:str
    build_digest:str
    bundle_digest:str
    pointer_digest:str
    source_kind:Literal['SYNTHETIC','DEVELOPMENT','LIVE_SHADOW']
    holdout_classification:Literal['SYNTHETIC','DEVELOPMENT','GENUINE_HOLDOUT']
    campaign_portfolio_digest:str
    expected_net_edge_bps:int
    artifacts_verified:Literal[True]=True
    live_shadow_qualified:Literal[False]=False
    operational_isolation_verified:Literal[False]=False
    provider_semantics_verified:Literal[False]=False
    execution_enabled:Literal[False]=False
    capital_authorized:Literal[False]=False

class FrozenArtifactSession(StrictContract):
    pointer_envelope:SignedEnvelope
    received_at:datetime
    resolution:ArtifactResolution

class ArtifactSource:
    """Trusted configured content-addressed root, with no arbitrary path API."""
    def __init__(self,root):
        self.root=Path(root)
        if self.root.is_symlink() or not self.root.is_dir():raise ValueError('explicit artifact directory required')
    def read(self,digest):
        if type(digest) is not str or len(digest)!=64 or any(c not in '0123456789abcdef' for c in digest):raise ValueError('exact artifact SHA256 required')
        directory=os.open(self.root,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW)
        try:
            descriptor=os.open(digest+'.bin',os.O_RDONLY|os.O_NOFOLLOW|os.O_NONBLOCK,dir_fd=directory)
            with os.fdopen(descriptor,'rb') as handle:
                details=os.fstat(handle.fileno())
                if not stat.S_ISREG(details.st_mode) or details.st_size>32*1024*1024:raise ValueError('invalid bounded artifact')
                raw=handle.read(32*1024*1024+1)
                if len(raw)>32*1024*1024:raise ValueError('artifact exceeds size bound')
        finally:os.close(directory)
        if hashlib.sha256(raw).hexdigest()!=digest:raise ValueError('artifact bytes differ from digest')
        return raw
    def json(self,digest):
        raw=self.read(digest);value=json.loads(raw)
        if canonical_bytes(value)!=raw:raise ValueError('noncanonical artifact JSON')
        return value

class ArtifactResolver:
    ROLES=frozenset(('RESEARCH','PROMOTION','CLASSIFICATION','CALIBRATION','BUILD','POINTER'))
    def __init__(self,store,*,source,signer,role_trust_stores,actor_bindings,expected_feature_schema_id,expected_label_schema_id):
        if type(store) is not EvidenceStore or type(source) is not ArtifactSource or set(role_trust_stores)!=self.ROLES:raise TypeError('exact store/source and artifact service trust roles required')
        self.store=store;self.source=source;self.signer=signer;self.roles=dict(role_trust_stores);self.actors=dict(actor_bindings)
        self.feature_schema=expected_feature_schema_id;self.label_schema=expected_label_schema_id

    def _verify(self,envelope,contract,role,now,*,actor=None):
        envelope=SignedEnvelope.model_validate_json(canonical_bytes(envelope),strict=True)
        value=verify_envelope(envelope,contract,self.roles[role]);key=envelope.signature.key_id
        self.roles[role].require_trusted_at_receipt(key,now)
        if envelope.signature.created_at>now:raise ValueError('artifact signed in the future')
        expected=actor if actor is not None else getattr(value,'actor_id',self.actors.get((role,key)))
        if expected is None or self.actors.get((role,key))!=expected:raise ValueError('artifact service actor/key role mismatch')
        return value,envelope

    def _signed(self,digest,contract,role,now,*,actor=None):
        return self._verify(SignedEnvelope.model_validate_json(self.source.read(digest),strict=True),contract,role,now,actor=actor)

    def _resolve(self,pointer_envelope,now):
        pointer,pointer_envelope=self._verify(pointer_envelope,SessionArtifactPointer,'POINTER',now)
        if not pointer.session_open<=now<pointer.session_close:raise ValueError('artifact pointer session inactive')
        bundle=ArtifactBundle.model_validate_json(self.source.read(pointer.bundle_digest),strict=True)
        for name in ('candidate_id','model_digest','promotion_digest','calibration_digest','risk_policy_digest','build_digest'):
            if getattr(pointer,name)!=getattr(bundle,name):raise ValueError('pointer differs from exact artifact bundle')
        state=reconstruct_research(self.store._load_research_ledger_unlocked(),trust_store=self.store.trust_store,authorities=self.store.authorities)
        if state.head_digest!=bundle.registry_head_digest:raise ValueError('frozen registry head changed')
        candidate=state.candidates.get(bundle.candidate_id)
        if candidate is None or candidate.artifact_hash!=bundle.model_digest:raise ValueError('frozen candidate/model mismatch')
        self.store.trust_store.require_trusted_at_receipt(candidate.signer_key_id,now)
        registered=state.experiments.get(candidate.experiment_id)
        if registered is None or registered.failure_reason is not None or registered.registration_digest!=candidate.registration_digest or registered.proposal.code_commit!=candidate.code_commit:raise ValueError('candidate registration mismatch')
        proposal=registered.proposal
        if self.source.read(bundle.proposal_digest)!=canonical_bytes(asdict(proposal)):raise ValueError('actual proposal bytes mismatch')
        self.source.read(bundle.model_digest)
        campaign,_=self._signed(bundle.campaign_envelope_digest,CampaignPayload,'RESEARCH',now,actor=candidate.actor_id)
        if (campaign.proposal_id,campaign.model_artifact_hash,campaign.campaign_manifest_digest,campaign.artifact_hash,campaign.selected_trial_result_digest,campaign.best_params_digest,campaign.search_space_digest)!=(proposal.experiment_id,bundle.model_digest,bundle.campaign_manifest_digest,bundle.campaign_summary_digest,bundle.selected_trial_result_digest,bundle.best_params_digest,bundle.search_space_digest):raise ValueError('campaign artifact bindings mismatch')
        manifest,params,trial_result=self._campaign_artifacts(bundle,campaign,proposal,state)
        deps={(dep.source,dep.dataset_id,dep.schema_version) for dep in proposal.data_dependencies}
        for source,digest in [('axiom2-campaign-manifest',bundle.campaign_manifest_digest),('axiom2-hyperparameter-search',bundle.search_space_digest),('axiom2-promotion-rule',bundle.promotion_rule_digest)]:
            if (source,digest,'1') not in deps:raise ValueError('artifact dependency not preregistered')
        trials=[trial for trial in state.trials.values() if trial.experiment_id==proposal.experiment_id and trial.campaign_digest==bundle.campaign_manifest_digest]
        selected=state.trials.get(campaign.selected_trial_id)
        if not trials or len(trials)!=campaign.experiment_count or any(trial.status=='STARTED' for trial in trials) or selected is None or selected.status!='SUCCEEDED' or selected.result_digest!=bundle.selected_trial_result_digest or selected not in trials:raise ValueError('actual trial history incomplete or conflicting')
        holdout=state.holdouts.get(bundle.holdout_id)
        if holdout is None or holdout.status!='CONSUMED' or holdout.candidate_id!=candidate.candidate_id or holdout.result_hash!=bundle.holdout_result_digest or holdout.attempt_count!=state.attempt_count:raise ValueError('previously consumed candidate holdout result required')
        result=self.source.json(bundle.holdout_result_digest)
        if type(result) is not dict or set(result)!={'candidate_id','holdout_id','net_return','cost_evidence_id','schema_version'} or type(result['schema_version']) is not int or result['schema_version']!=1 or (result['candidate_id'],result['holdout_id'],result['cost_evidence_id'])!=(candidate.candidate_id,holdout.holdout_id,proposal.cost_model_id) or type(result['net_return']) not in (int,float) or not math.isfinite(result['net_return']):raise ValueError('actual consumed result schema/binding mismatch')
        classification,_=self._signed(bundle.classification_digest,HoldoutClassification,'CLASSIFICATION',now)
        if (classification.candidate_id,classification.holdout_id,classification.dataset_digest,classification.result_digest,classification.consumption_digest)!=(candidate.candidate_id,holdout.holdout_id,holdout.dataset_digest,bundle.holdout_result_digest,holdout.consumption_digest):raise ValueError('holdout classification binding mismatch')
        fixture_dependency=any('fixture' in dep.source.lower() or 'synthetic' in dep.source.lower() for dep in proposal.data_dependencies)
        if pointer.source_kind=='LIVE_SHADOW' and (classification.source_kind!='GENUINE_HOLDOUT' or fixture_dependency):raise ValueError('synthetic/development evidence cannot satisfy genuine holdout provenance')
        if pointer.source_kind!='LIVE_SHADOW' and classification.source_kind=='GENUINE_HOLDOUT':raise ValueError('mixed fixture/genuine evidence denied')
        rule=self.source.json(bundle.promotion_rule_digest)
        if type(rule) is not dict or set(rule)!={'min_excess_return','schema_version'} or type(rule['schema_version']) is not int or rule['schema_version']!=1 or type(rule['min_excess_return']) not in (int,float) or not math.isfinite(rule['min_excess_return']) or rule['min_excess_return']<0:raise ValueError('promotion rule schema invalid')
        if campaign.portfolio_net_return-campaign.baseline_net_return<rule['min_excess_return'] or result['net_return']-campaign.baseline_net_return<rule['min_excess_return']:raise ValueError('actual promotion thresholds failed')
        promotion,promotion_envelope=self._signed(bundle.promotion_digest,PromotionDecisionPayload,'PROMOTION',now,actor=self.actors.get(('PROMOTION',SignedEnvelope.model_validate_json(self.source.read(bundle.promotion_digest),strict=True).signature.key_id)))
        evidence=content_digest(dict(candidate_id=candidate.candidate_id,candidate_freeze_sequence=candidate.freeze_sequence,campaign_envelope_digest=SignedEnvelope.model_validate_json(self.source.read(bundle.campaign_envelope_digest),strict=True).payload_digest,campaign_manifest_digest=bundle.campaign_manifest_digest,selected_trial_id=campaign.selected_trial_id,selected_trial_result_digest=bundle.selected_trial_result_digest,holdout_consumption_digest=holdout.consumption_digest,holdout_result_hash=bundle.holdout_result_digest,promotion_rule_digest=bundle.promotion_rule_digest,research_head_digest=state.head_digest,research_event_count=state.event_count,schema_version=1))
        package=dict(candidate_id=candidate.candidate_id,candidate_artifact_hash=bundle.model_digest,frozen_artifact_hash=candidate.artifact_hash,baseline_net_return=campaign.baseline_net_return,candidate_net_return=campaign.portfolio_net_return,cost_evidence_id=proposal.cost_model_id,experiment_count=campaign.experiment_count,registered_experiment_count=len(trials),holdout_id=holdout.holdout_id,holdout_status='CONSUMED',holdout_result_hash=bundle.holdout_result_digest,holdout_candidate_id=candidate.candidate_id,min_excess_return=float(rule['min_excess_return']),holdout_net_return=float(result['net_return']),holdout_consumption_count=1)
        decision_hash=content_digest(dict(status='PROMOTED',candidate_id=candidate.candidate_id,reasons=(),package=package,verified=True,evidence_digest=evidence))
        if promotion.status!='PROMOTED' or not promotion.verified or promotion.reasons or promotion.candidate_id!=candidate.candidate_id or promotion.evidence_digest!=evidence or promotion.decision_hash!=decision_hash:raise ValueError('promotion summary does not resolve actual evidence')
        calibration,calibration_envelope=self._signed(bundle.calibration_digest,CalibrationArtifact,'CALIBRATION',now)
        if (calibration.candidate_id,calibration.model_digest,calibration.promotion_digest,calibration.campaign_manifest_digest,calibration.cost_evidence_id)!=(candidate.candidate_id,bundle.model_digest,bundle.promotion_digest,bundle.campaign_manifest_digest,proposal.cost_model_id):raise ValueError('calibration model/promotion/cost mismatch')
        economic=self.source.json(calibration.evidence_digest)
        self._economic_artifact(economic,calibration,campaign,result,manifest)
        expected_class={'SYNTHETIC':'SYNTHETIC','DEVELOPMENT':'DEVELOPMENT','LIVE_SHADOW':'GENUINE_HOLDOUT'}[pointer.source_kind]
        expected_calibration={'SYNTHETIC':'SYNTHETIC','DEVELOPMENT':'DEVELOPMENT','LIVE_SHADOW':'LIVE_READONLY'}[pointer.source_kind]
        if classification.source_kind!=expected_class or calibration.source_kind!=expected_calibration:raise ValueError('mixed artifact source classification')
        policy=_record(RiskPolicy,self.source.json(bundle.risk_policy_digest))
        if risk_policy_digest(policy)!=bundle.risk_policy_digest:raise ValueError('actual risk policy mismatch')
        build,_=self._signed(bundle.build_digest,BuildArtifact,'BUILD',now)
        self.source.read(build.artifact_digest)
        if build.code_commit!=candidate.code_commit:raise ValueError('disabled build candidate code mismatch')
        return ArtifactResolution(candidate_id=candidate.candidate_id,candidate_digest=content_digest(asdict(candidate)),model_digest=bundle.model_digest,promotion_digest=bundle.promotion_digest,promotion_decision_hash=decision_hash,calibration_digest=bundle.calibration_digest,calibration_payload_digest=calibration_envelope.payload_digest,risk_policy_digest=bundle.risk_policy_digest,build_digest=bundle.build_digest,bundle_digest=pointer.bundle_digest,pointer_digest=pointer_envelope.payload_digest,source_kind=pointer.source_kind,holdout_classification=classification.source_kind,campaign_portfolio_digest=bundle.campaign_portfolio_digest,expected_net_edge_bps=calibration.expected_net_edge_bps)

    def _campaign_artifacts(self,bundle,campaign,proposal,state):
        manifest=self.source.json(bundle.campaign_manifest_digest)
        keys={'proposal_id','feature_digest','label_digest','split_digest','feature_schema_id','label_schema_id','cost_bps','search_space_digest','portfolio_policy_digest','adapter_implementation_id','schema_version'}
        adapter=content_digest(dict(model_family='lightgbm',adapter_contract='rank-features-target-separated',schema_version=1))
        expected=dict(proposal_id=proposal.experiment_id,feature_digest=campaign.feature_digest,label_digest=campaign.label_digest,split_digest=campaign.split_digest,feature_schema_id=self.feature_schema,label_schema_id=self.label_schema,search_space_digest=bundle.search_space_digest,portfolio_policy_digest=bundle.campaign_portfolio_digest,adapter_implementation_id=adapter,schema_version=1)
        if type(manifest) is not dict or set(manifest)!=keys or any(canonical_bytes(manifest[key])!=canonical_bytes(value) for key,value in expected.items()) or type(manifest['schema_version']) is not int or type(manifest['cost_bps']) not in (int,float) or not math.isfinite(manifest['cost_bps']) or manifest['cost_bps']<0 or campaign.model_family!='lightgbm':raise ValueError('campaign manifest schema/adapter/binding mismatch')
        frames=[]
        for digest in (campaign.feature_digest,campaign.label_digest):
            rows=list(csv.DictReader(io.StringIO(self.source.read(digest).decode('utf-8'))))
            if not rows or not {'timestamp','instrument_id'}<=set(rows[0]):raise ValueError('campaign frame date-index facts unavailable')
            frame={}
            for row in rows:
                key=(datetime.fromisoformat(row['timestamp'].replace('Z','+00:00')),row['instrument_id'])
                if key in frame or not key[1]:raise ValueError('campaign frame index duplicate/empty')
                frame[key]=row
            frames.append(frame)
        labels={'forward_return','forward_excess_return','cross_sectional_rank'}
        if not labels<=set(next(iter(frames[1].values()))):raise ValueError('campaign label schema unavailable')
        dates=set()
        for key in frames[0].keys()&frames[1].keys():
            values=[value for name,value in frames[0][key].items() if name not in ('timestamp','instrument_id')]+[frames[1][key][name] for name in labels]
            if all(value and math.isfinite(float(value)) for value in values):dates.add(key[0])
        positions={date:index for index,date in enumerate(sorted(dates))}
        splits=self.source.json(campaign.split_digest)
        if type(splits) is not list or not splits:raise ValueError('campaign split manifest required')
        seen=set()
        for split in splits:
            if type(split) is not dict or set(split)!={'train','test'} or not split['train'] or not split['test'] or type(split['train']) is not list or type(split['test']) is not list:raise ValueError('campaign split schema')
            train=[datetime.fromisoformat(value.replace('Z','+00:00')) for value in split['train']];test=[datetime.fromisoformat(value.replace('Z','+00:00')) for value in split['test']]
            if len(set(train))!=len(train) or len(set(test))!=len(test) or set(train)&set(test) or max(train)>=min(test) or seen&set(test):raise ValueError('campaign split overlap/order')
            if any(date not in positions for date in train+test) or positions[max(train)]>=positions[min(test)]-5:raise ValueError('campaign split purge insufficient or input date missing')
            seen.update(test)
        policy=self.source.json(bundle.campaign_portfolio_digest)
        expected_policy=dict(top_k=5,holding_horizon=5,rebalance_every=5,weighting='equal',overlap=False,schema_version=1)
        if canonical_bytes(policy)!=canonical_bytes(expected_policy) or ('axiom2-portfolio-policy',bundle.campaign_portfolio_digest,'1') not in {(dep.source,dep.dataset_id,dep.schema_version) for dep in proposal.data_dependencies}:raise ValueError('campaign portfolio policy unsupported or unregistered')
        search=self.source.json(bundle.search_space_digest);params=self.source.json(bundle.best_params_digest)
        if type(search) is not list or not search or any(type(row) is not dict for row in search) or type(params) is not dict:raise ValueError('actual parameter/search schema')
        trials=[trial for trial in state.trials.values() if trial.experiment_id==proposal.experiment_id and trial.campaign_digest==bundle.campaign_manifest_digest]
        if len(trials)!=len(search):raise ValueError('trial/search cardinality mismatch')
        for trial in trials:
            if type(trial.config_index) is not int or not 0<=trial.config_index<len(search) or trial.config_digest!=content_digest(dict(params=search[trial.config_index],schema_version=1)):raise ValueError('registered trial configuration differs from search')
        selected=state.trials.get(campaign.selected_trial_id)
        if selected is None or selected.config_digest!=content_digest(dict(params=params,schema_version=1)) or search[selected.config_index]!=params:raise ValueError('selected parameters differ from registered trial')
        result=self.source.json(bundle.selected_trial_result_digest)
        if type(result) is not dict or set(result)!={'campaign_digest','config_digest','score','returns','turnover','model_artifact_hash','positive_fold_fraction','schema_version'} or type(result['schema_version']) is not int or result['schema_version']!=1 or (result['campaign_digest'],result['config_digest'],result['model_artifact_hash'])!=(bundle.campaign_manifest_digest,selected.config_digest,bundle.model_digest):raise ValueError('trial result schema/binding')
        for name in ('returns','turnover'):
            if type(result[name]) is not list or not result[name] or any(type(value) not in (int,float) or not math.isfinite(value) for value in result[name]):raise ValueError('trial economic series invalid')
        if len(result['returns'])!=len(result['turnover']) or any(value<0 for value in result['turnover']):raise ValueError('trial economic series bounds')
        comparisons=[(result['score'],campaign.mean_rank_correlation),(result['positive_fold_fraction'],campaign.positive_fold_fraction),(math.fsum(result['returns'])/len(result['returns']),campaign.portfolio_net_return),(math.fsum(result['turnover'])/len(result['turnover']),campaign.turnover)]
        if any(type(left) not in (int,float) or not math.isfinite(left) or not math.isclose(left,right,rel_tol=0,abs_tol=1e-12) for left,right in comparisons) or not -1<=campaign.mean_rank_correlation<=1 or not 0<=campaign.positive_fold_fraction<=1 or not -1<=campaign.max_drawdown<=0 or campaign.turnover<0:raise ValueError('trial/campaign economic metrics mismatch')
        summary=self.source.json(bundle.campaign_summary_digest)
        wealth=peak=1.0;drawdown=0.0
        for value in result['returns']:
            wealth*=1.0+value;peak=max(peak,wealth)
            if not math.isfinite(wealth) or peak<=0:raise ValueError('trial return curve invalid')
            drawdown=min(drawdown,wealth/peak-1.0)
        if not math.isclose(campaign.max_drawdown,drawdown,rel_tol=0,abs_tol=1e-12):raise ValueError('campaign drawdown differs from actual return curve')
        expected_summary=dict(proposal=proposal.experiment_id,model_family='lightgbm',search=search,best=params,score=campaign.mean_rank_correlation,net=campaign.portfolio_net_return,baseline=campaign.baseline_net_return,turnover=campaign.turnover,max_drawdown=campaign.max_drawdown,regimes=(('all',campaign.portfolio_net_return),),feature_digest=campaign.feature_digest,label_digest=campaign.label_digest,split_digest=campaign.split_digest,campaign_manifest_digest=bundle.campaign_manifest_digest,model_artifact_hash=bundle.model_digest)
        if canonical_bytes(summary)!=canonical_bytes(expected_summary):raise ValueError('campaign summary schema/content mismatch')
        return manifest,params,result

    @staticmethod
    def _economic_artifact(economic,calibration,campaign,holdout,manifest):
        expected=dict(schema_version=1,method='after-cost-excess-min-bps-v1',candidate_id=calibration.candidate_id,model_digest=calibration.model_digest,promotion_digest=calibration.promotion_digest,campaign_manifest_digest=calibration.campaign_manifest_digest,cost_evidence_id=calibration.cost_evidence_id,source_kind=calibration.source_kind,development_net_return=campaign.portfolio_net_return,holdout_net_return=holdout['net_return'],baseline_net_return=campaign.baseline_net_return,cost_bps=manifest['cost_bps'],expected_net_edge_bps=calibration.expected_net_edge_bps)
        if type(economic) is not dict or set(economic)!=set(expected)|{'buffer_bps'} or any(canonical_bytes(economic[key])!=canonical_bytes(value) for key,value in expected.items()) or type(economic['buffer_bps']) is not int or economic['buffer_bps']<0:raise ValueError('economic calibration schema/binding mismatch')
        edge=(min(Decimal(str(campaign.portfolio_net_return)),Decimal(str(holdout['net_return'])))-Decimal(str(campaign.baseline_net_return)))*10000
        conservative=int(edge.to_integral_value(rounding=ROUND_FLOOR))-economic['buffer_bps']
        if conservative!=calibration.expected_net_edge_bps or conservative<=0:raise ValueError('economic calibration after-cost edge mismatch')

    def _directory(self):
        path=self.store.root/'offline-artifact-sessions'
        if path.is_symlink() or path.exists() and not path.is_dir():raise StoreCorruptionError('invalid artifact session directory')
        return path

    def _frozen(self,path):
        if path.is_symlink() or not path.is_file():raise StoreCorruptionError('invalid artifact session path')
        raw=path.read_bytes();signed=SignedEnvelope.model_validate_json(raw,strict=True)
        if canonical_bytes(signed)!=raw:raise StoreCorruptionError('noncanonical frozen session')
        record=verify_envelope(signed,FrozenArtifactSession,self.store.trust_store)
        self.store.trust_store.require_trusted_at_receipt(signed.signature.key_id,record.received_at)
        if signed.signature.created_at!=record.received_at:raise StoreCorruptionError('frozen session receipt time mismatch')
        pointer=SessionArtifactPointer.from_versioned_payload(record.pointer_envelope.payload)
        identity=content_digest(dict(account_alias=pointer.account_alias,session_id=pointer.session_id))
        if path.name!=identity+'.json':raise StoreCorruptionError('frozen session address mismatch')
        return record,pointer

    def _commit_time(self,pointer_envelope,minimum):
        pointer=SessionArtifactPointer.from_versioned_payload(pointer_envelope.payload)
        bundle=ArtifactBundle.model_validate_json(self.source.read(pointer.bundle_digest),strict=True)
        keys=[('POINTER',pointer_envelope.signature.key_id)]
        for digest,role in ((bundle.campaign_envelope_digest,'RESEARCH'),(bundle.promotion_digest,'PROMOTION'),(bundle.classification_digest,'CLASSIFICATION'),(bundle.calibration_digest,'CALIBRATION'),(bundle.build_digest,'BUILD')):
            signed=SignedEnvelope.model_validate_json(self.source.read(digest),strict=True);keys.append((role,signed.signature.key_id))
        state=reconstruct_research(self.store._load_research_ledger_unlocked(),trust_store=self.store.trust_store,authorities=self.store.authorities)
        candidate=state.candidates.get(bundle.candidate_id)
        if candidate is None or state.head_digest!=bundle.registry_head_digest:raise ValueError('registry changed before artifact commit')
        # All filesystem reads precede this fresh clock and in-memory checks.
        now=self.store._trusted_clock()
        if now<minimum or not pointer.session_open<=now<pointer.session_close:raise ValueError('artifact pointer session inactive or clock regressed')
        for role,key in keys:self.roles[role].require_trusted_at_receipt(key,now)
        self.store.trust_store.require_trusted_at_receipt(candidate.signer_key_id,now)
        self.store.trust_store.require_trusted_at_receipt(self.signer.key_id,now)
        return now

    def _verify_unlocked(self,pointer_envelope,resolution,now,*,return_time=False):
        current=self._resolve(pointer_envelope,now)
        if current!=resolution:raise ValueError('resolved artifact identities changed')
        pointer=SessionArtifactPointer.from_versioned_payload(pointer_envelope.payload)
        identity=content_digest(dict(account_alias=pointer.account_alias,session_id=pointer.session_id))
        record,_=self._frozen(self._directory()/(identity+'.json'))
        if now<record.received_at or record.pointer_envelope!=pointer_envelope or record.resolution!=resolution:
            raise ValueError('frozen session changed or clock regressed')
        commit_now=self._commit_time(pointer_envelope,now)
        return (current,commit_now) if return_time else current

    def resolve(self,pointer_envelope):
        pointer_envelope=SignedEnvelope.model_validate_json(canonical_bytes(pointer_envelope),strict=True)
        with self.store._locked():
            now=self.store._trusted_clock();resolution=self._resolve(pointer_envelope,now)
            final_now=self.store._trusted_clock()
            if final_now<now:raise ValueError('artifact resolution clock regressed')
            # Recheck all current trust/session/source bindings after expensive reads.
            if self._resolve(pointer_envelope,final_now)!=resolution:raise ValueError('artifact resolution changed')
            now=final_now;pointer=SessionArtifactPointer.from_versioned_payload(pointer_envelope.payload)
            identity=content_digest(dict(account_alias=pointer.account_alias,session_id=pointer.session_id));directory=self._directory();path=directory/(identity+'.json')
            if path.exists() or path.is_symlink():return self._verify_unlocked(pointer_envelope,resolution,now)
            for prior_path in sorted(directory.iterdir()) if directory.exists() else ():
                prior,prior_pointer=self._frozen(prior_path)
                if now<prior.received_at:raise ValueError('artifact session clock regressed')
                if pointer.account_alias==prior_pointer.account_alias and max(pointer.session_open,prior_pointer.session_open)<min(pointer.session_close,prior_pointer.session_close):
                    raise ValueError('account artifact sessions overlap')
            now=self._commit_time(pointer_envelope,now)
            record=FrozenArtifactSession(pointer_envelope=pointer_envelope,received_at=now,resolution=resolution);signed=self.signer.sign(record,created_at=now)
            if not directory.exists():directory.mkdir(mode=0o700);self.store._fsync_directory(directory);self.store._fsync_directory(self.store.root)
            self.store._atomic_create_bytes(path,canonical_bytes(signed));return resolution
