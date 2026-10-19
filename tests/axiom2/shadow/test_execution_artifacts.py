"""Resolve actual fixture artifacts; never open genuine holdout data."""
from dataclasses import asdict,replace
from datetime import timedelta
import hashlib,json
import pytest
from src.evidence.canonical import canonical_bytes
from src.evidence.signing import Ed25519Signer,TrustStore
from src.evidence.contracts import AuthorityRole
from tests.evidence.equity_research._support import build_context,proposal,NOW


def fixture(tmp_path,semantic=None):
    from src.axiom2.execution.artifacts import (ArtifactSource,ArtifactResolver,ArtifactBundle,
        HoldoutClassification,CalibrationArtifact,BuildArtifact,SessionArtifactPointer)
    from src.axiom2.promotion.authority import PromotionRule,promotion_rule_digest,evaluate_verified_promotion
    from src.axiom2.research.ranker import CampaignPayload,CampaignResult
    from src.axiom2.contracts.research_proposal import DataDependency
    from tests.axiom2.shadow.test_execution_resolution import packet
    from src.axiom2.execution.resolution import reconstruct_risk_packet
    from src.axiom2.portfolio.authority import risk_policy_digest
    ctx=build_context(tmp_path/'store');root=tmp_path/'artifacts';root.mkdir()
    def put(value):
        raw=value if type(value) is bytes else canonical_bytes(value);digest=hashlib.sha256(raw).hexdigest();(root/(digest+'.bin')).write_bytes(raw);return digest
    model=b'opaque disabled fixture model';model_digest=put(model)
    dates=['2025-01-%02dT00:00:00+00:00'%day for day in range(1,8)]
    features=put(('timestamp,instrument_id,momentum\n'+''.join(date+',A,1\n' for date in dates)).encode())
    labels=put(('timestamp,instrument_id,forward_return,forward_excess_return,cross_sectional_rank\n'+''.join(date+',A,.02,.02,.5\n' for date in dates)).encode())
    split_digest=put([{'train':['2025-01-01T00:00:00Z'],'test':['2025-01-%02dT00:00:00Z'%(2 if semantic=='purge' else 7)]}])
    drawdown=-.1 if semantic=='drawdown' else 0.
    portfolio=put({'top_k':5,'holding_horizon':5,'rebalance_every':5,'weighting':'equal','overlap':False,'schema_version':1})
    search=put(({},));params=put({});config_digest=put({'params':{},'schema_version':1})
    adapter=hashlib.sha256(canonical_bytes({'model_family':'lightgbm','adapter_contract':'rank-features-target-separated','schema_version':1})).hexdigest()
    manifest=put({'proposal_id':'artifact-fixture','feature_digest':features,'label_digest':labels,'split_digest':split_digest,'feature_schema_id':'features-v1','label_schema_id':'labels-v1','cost_bps':10.,'search_space_digest':search,'portfolio_policy_digest':portfolio,'adapter_implementation_id':adapter,'schema_version':1})
    trial_result=put({'campaign_digest':manifest,'config_digest':config_digest,'score':.1,'returns':(.02,),'turnover':(.2,),'model_artifact_hash':model_digest,'positive_fold_fraction':1.,'schema_version':1})
    summary=put({'proposal':'artifact-fixture','model_family':'lightgbm','search':({},),'best':{},'score':.1,'net':.02,'baseline':.005,'turnover':.2,'max_drawdown':drawdown,'regimes':(('all',.02),),'feature_digest':features,'label_digest':labels,'split_digest':split_digest,'campaign_manifest_digest':manifest,'model_artifact_hash':model_digest})
    rule=PromotionRule(min_excess_return=.001);rule_digest=put({'min_excess_return':.001,'schema_version':1});assert rule_digest==promotion_rule_digest(rule)
    p=proposal('artifact-fixture');p=replace(p,data_dependencies=p.data_dependencies+tuple(DataDependency(dataset_id=d,source=s,schema_version='1') for d,s in [(manifest,'axiom2-campaign-manifest'),(search,'axiom2-hyperparameter-search'),(rule_digest,'axiom2-promotion-rule'),(portfolio,'axiom2-portfolio-policy')]))
    ctx.registry.register_experiment(p);proposal_digest=put(asdict(p))
    dataset_digest=hashlib.sha256(b'sacrificial fixture only').hexdigest()
    ctx.operator.register_holdout('sacrificial',dataset_digest,NOW+timedelta(days=1),NOW+timedelta(days=2))
    trial=ctx.registry.record_trial_started(p.experiment_id,manifest,0,config_digest);ctx.registry.record_trial_succeeded(trial.trial_id,trial_result)
    payload=CampaignPayload(artifact_hash=summary,model_artifact_hash=model_digest,proposal_id=p.experiment_id,model_family='lightgbm',search_space_digest=search,feature_digest=features,label_digest=labels,split_digest=split_digest,campaign_manifest_digest=manifest,selected_trial_id=trial.trial_id,selected_trial_result_digest=trial_result,best_params_digest=params,experiment_count=1,mean_rank_correlation=.1,portfolio_net_return=.02,baseline_net_return=.005,turnover=.2,max_drawdown=drawdown,positive_fold_fraction=1.)
    campaign_envelope=ctx.registry.signer.sign(payload,created_at=NOW)
    campaign=CampaignResult('lightgbm',1,.1,.02,.005,.2,drawdown,(('all',.02),),summary,search,features,labels,split_digest,manifest,trial.trial_id,trial_result,model_digest,model,(),1.,False,campaign_envelope)
    candidate=ctx.registry.freeze_candidate(p.experiment_id,model_digest,p.code_commit);ctx.evaluator.open_holdout(candidate.candidate_id,'sacrificial')
    result={'candidate_id':candidate.candidate_id,'holdout_id':'sacrificial','net_return':.02,'cost_evidence_id':p.cost_model_id,'schema_version':1};result_digest=put(result)
    ctx.evaluator.consume_holdout(candidate.candidate_id,'sacrificial',result_digest)
    signers={};trusts={};actors={}
    for role in ('PROMOTION','CLASSIFICATION','CALIBRATION','BUILD','POINTER'):
        signer=Ed25519Signer.generate();signers[role]=signer;trust=TrustStore();trust.add(signer.trusted_key(valid_from=NOW-timedelta(days=1)));trusts[role]=trust;actors[(role,signer.key_id)]=role.lower()
    research=ctx.registry.signer;trusts['RESEARCH']=ctx.store.trust_store;actors[('RESEARCH',research.key_id)]='local_importer'
    ctx.store.trust_store.add(signers['PROMOTION'].trusted_key(valid_from=NOW-timedelta(days=1)));ctx.store.authorities.register(actor_id='promotion',role=AuthorityRole.PROMOTION_SERVICE,key_ids=(signers['PROMOTION'].key_id,))
    promotion=evaluate_verified_promotion(proposal=p,campaign=campaign,candidate_id=candidate.candidate_id,holdout_id='sacrificial',holdout_result_bytes=canonical_bytes(result),registry=ctx.registry,promotion_rule=rule,signer=signers['PROMOTION'],actor_id='promotion',created_at=NOW)
    promotion_digest=put(promotion.envelope)
    holdout=ctx.registry.snapshot().holdouts['sacrificial']
    classification=HoldoutClassification(actor_id='classification',candidate_id=candidate.candidate_id,holdout_id='sacrificial',dataset_digest=dataset_digest,result_digest=result_digest,consumption_digest=holdout.consumption_digest,source_kind='SYNTHETIC')
    classification_digest=put(signers['CLASSIFICATION'].sign(classification,created_at=NOW))
    economic=put({'schema_version':1,'method':'after-cost-excess-min-bps-v1','candidate_id':candidate.candidate_id,'model_digest':model_digest,'promotion_digest':promotion_digest,'campaign_manifest_digest':manifest,'cost_evidence_id':p.cost_model_id,'source_kind':'SYNTHETIC','development_net_return':.02,'holdout_net_return':.02,'baseline_net_return':.005,'cost_bps':10.,'buffer_bps':51 if semantic=='economic' else 50,'expected_net_edge_bps':100})
    calibration=CalibrationArtifact(actor_id='calibration',candidate_id=candidate.candidate_id,model_digest=model_digest,promotion_digest=promotion_digest,campaign_manifest_digest=manifest,cost_evidence_id=p.cost_model_id,source_kind='SYNTHETIC',evidence_digest=economic,expected_net_edge_bps=100)
    calibration_digest=put(signers['CALIBRATION'].sign(calibration,created_at=NOW))
    policy,_,_=reconstruct_risk_packet(packet());policy_digest=put(asdict(policy));assert policy_digest==risk_policy_digest(policy)
    build_bytes=put(b'opaque disabled build fixture')
    build_digest=put(signers['BUILD'].sign(BuildArtifact(actor_id='build',code_commit=p.code_commit,artifact_digest=build_bytes),created_at=NOW))
    bundle=ArtifactBundle(candidate_id=candidate.candidate_id,registry_head_digest=ctx.registry.snapshot().head_digest,proposal_digest=proposal_digest,model_digest=model_digest,campaign_envelope_digest=put(campaign_envelope),campaign_manifest_digest=manifest,campaign_summary_digest=summary,campaign_portfolio_digest=portfolio,selected_trial_result_digest=trial_result,search_space_digest=search,best_params_digest=params,holdout_id='sacrificial',holdout_result_digest=result_digest,classification_digest=classification_digest,promotion_rule_digest=rule_digest,promotion_digest=promotion_digest,calibration_digest=calibration_digest,risk_policy_digest=policy_digest,build_digest=build_digest)
    bundle_digest=put(bundle)
    pointer=SessionArtifactPointer(actor_id='pointer',account_alias='shadow-account',session_id='synthetic-session-one',session_open=NOW,session_close=NOW+timedelta(hours=6),bundle_digest=bundle_digest,candidate_id=candidate.candidate_id,model_digest=model_digest,promotion_digest=promotion_digest,calibration_digest=calibration_digest,risk_policy_digest=policy_digest,build_digest=build_digest,source_kind='SYNTHETIC')
    # Bind fixture session to the same modeled risk request interval.
    _,risk_request,_=reconstruct_risk_packet(packet());shift=NOW-risk_request.as_of
    pointer=pointer.model_copy(update={'session_open':risk_request.session_open+shift,'session_close':risk_request.session_close+shift})
    signed_pointer=signers['POINTER'].sign(pointer,created_at=NOW)
    resolver=ArtifactResolver(ctx.store,source=ArtifactSource(root),signer=research,role_trust_stores=trusts,actor_bindings=actors,expected_feature_schema_id='features-v1',expected_label_schema_id='labels-v1')
    return resolver,ctx,root,put,bundle,signers,trusts,signed_pointer


def test_actual_artifacts_and_frozen_registry_resolve_without_capital(tmp_path):
    resolver,ctx,root,put,bundle,signers,trusts,pointer=fixture(tmp_path)
    result=resolver.resolve(pointer)
    assert result.model_digest==bundle.model_digest and result.source_kind=='SYNTHETIC'
    assert result.artifacts_verified is True and result.capital_authorized is False and result.execution_enabled is False
    assert resolver.resolve(pointer)==result


@pytest.mark.parametrize('change',['model','proposal','campaign','risk-policy','missing-holdout','wrong-role','retired-key','changed-pointer'])
def test_forged_or_unresolved_artifacts_cannot_pass(tmp_path,change):
    resolver,ctx,root,put,bundle,signers,trusts,pointer=fixture(tmp_path)
    if change in ('model','proposal','campaign','risk-policy','missing-holdout'):
        digest={'model':bundle.model_digest,'proposal':bundle.proposal_digest,'campaign':bundle.campaign_envelope_digest,'risk-policy':bundle.risk_policy_digest,'missing-holdout':bundle.holdout_result_digest}[change]
        path=root/(digest+'.bin')
        if change=='missing-holdout':path.unlink()
        else:path.write_bytes(b'changed')
    if change=='wrong-role':
        from src.axiom2.execution.artifacts import SessionArtifactPointer
        pointer=signers['BUILD'].sign(SessionArtifactPointer.from_versioned_payload(pointer.payload),created_at=NOW)
    if change=='retired-key':trusts['POINTER'].revoke(signers['POINTER'].key_id,revoked_at=NOW)
    if change=='changed-pointer':
        from src.axiom2.execution.artifacts import SessionArtifactPointer
        resolver.resolve(pointer);record=SessionArtifactPointer.from_versioned_payload(pointer.payload)
        pointer=signers['POINTER'].sign(record.model_copy(update={'model_digest':'f'*64}),created_at=NOW)
    with pytest.raises((ValueError,OSError)):resolver.resolve(pointer)


def test_alias_cannot_freeze_overlapping_sessions_under_new_name(tmp_path):
    from src.axiom2.execution.artifacts import SessionArtifactPointer
    resolver,ctx,root,put,bundle,signers,trusts,pointer=fixture(tmp_path)
    resolver.resolve(pointer)
    value=SessionArtifactPointer.from_versioned_payload(pointer.payload).model_copy(update={'session_id':'renamed-session'})
    renamed=signers['POINTER'].sign(value,created_at=NOW)
    with pytest.raises(ValueError,match='overlap'):resolver.resolve(renamed)


def test_session_expiry_during_resolution_cannot_freeze(tmp_path,monkeypatch):
    resolver,ctx,root,put,bundle,signers,trusts,pointer=fixture(tmp_path)
    original=resolver._resolve
    def resolve_and_expire(envelope,now):
        result=original(envelope,now)
        ctx.store._trusted_clock=lambda:NOW+timedelta(hours=7)
        return result
    monkeypatch.setattr(resolver,'_resolve',resolve_and_expire)
    with pytest.raises(ValueError,match='inactive'):resolver.resolve(pointer)
    assert not (ctx.store.root/'offline-artifact-sessions').exists()


def test_execution_wire_contracts_match_producers():
    from src.axiom2.execution.artifacts import CampaignPayload as WireCampaign,PromotionDecisionPayload as WirePromotion
    from src.axiom2.research.ranker import CampaignPayload
    from src.axiom2.promotion.authority import PromotionDecisionPayload
    for wire,producer in ((WireCampaign,CampaignPayload),(WirePromotion,PromotionDecisionPayload)):
        assert wire.model_json_schema()==producer.model_json_schema()


def test_content_addressed_fifo_is_rejected_without_blocking(tmp_path):
    import os
    from src.axiom2.execution.artifacts import ArtifactSource
    os.mkfifo(tmp_path/('a'*64+'.bin'))
    with pytest.raises(ValueError,match='bounded'):ArtifactSource(tmp_path).read('a'*64)


def resolved_ingress(tmp_path,change='none'):
    import runpy
    from pathlib import Path
    from src.axiom2.execution.lifecycle import ExecutionLifecycle,LifecycleEvent
    from src.axiom2.execution.authorization import AuthorizationJournal
    from src.axiom2.execution.journal import ExecutionJournal
    from src.axiom2.execution.fencing import OfflineExecutionFence
    from src.evidence.execution_shadow import AuthorityBindingEvent
    from src.axiom2.execution.resolution import reconstruct_risk_packet,SignedRiskPacket
    from src.axiom2.portfolio.authority import evaluate_portfolio
    from src.axiom2.portfolio.intents import build_order_intents,sign_risk_intent_receipt
    from src.evidence.hashing import content_digest
    from tests.axiom2.shadow.test_execution_resolution import packet
    from tests.axiom2.shadow.test_operator_authorization import fixture as auth_fixture
    resolver,ctx,root,put,bundle,signers,trusts,pointer=fixture(tmp_path/'artifacts')
    resolved=resolver.resolve(pointer)
    producer=Ed25519Signer.generate();observer=Ed25519Signer.generate();roles={};actors={}
    for role,key in [('GATEWAY',producer),('OBSERVER',observer)]:
        trust=TrustStore();trust.add(key.trusted_key(valid_from=NOW-timedelta(days=1)));roles[role]=trust;actors[(role,key.key_id)]=role.lower()
    kernel=ExecutionLifecycle(ctx.store,signer=ctx.registry.signer,role_trust_stores=roles,actor_bindings=actors)
    policy,request,_=reconstruct_risk_packet(packet());shift=NOW-request.as_of
    request=replace(request,as_of=NOW,session_open=request.session_open+shift,session_close=request.session_close+shift,
        signal_at=request.signal_at+shift,account_alias='shadow-account',snapshot=replace(request.snapshot,observed_at=request.snapshot.observed_at+shift),
        candidates=tuple(replace(c,quoted_at=c.quoted_at+shift,metadata_at=c.metadata_at+shift,expected_net_edge_bps=resolved.expected_net_edge_bps) for c in request.candidates),
        model_artifact_hash=resolved.model_digest,promotion_decision_hash=resolved.promotion_decision_hash,
        economic_evidence_digest=resolved.calibration_payload_digest,campaign_portfolio_digest=resolved.campaign_portfolio_digest)
    if change=='source':request=replace(request,source_kind='DEVELOPMENT')
    if change=='session':request=replace(request,session_close=request.session_close+timedelta(seconds=1))
    decision=evaluate_portfolio(request,policy)
    helper=runpy.run_path(str(Path(__file__).parents[1]/'portfolio'/'test_order_intents.py'))
    account,caps=helper['refs'](request);intents=build_order_intents(decision,request,account,caps,mode='SHADOW');order=intents[0]
    operator,gateway,operator_trust,_,scope,declaration,auth_store=auth_fixture(tmp_path/'auth')
    scope={**scope,'account_alias':account.account_alias,'intent_digest':content_digest(asdict(order)),
        'capability_digest':caps.capability_digest,'risk_policy_digest':resolved.risk_policy_digest,'model_digest':resolved.model_digest,
        'candidate_digest':resolved.candidate_digest,'build_digest':resolved.build_digest}
    operator_trust=TrustStore();operator_trust.add(operator.trusted_key(valid_from=NOW-timedelta(days=1)))
    declaration=declaration.model_copy(update={**scope,'not_before':NOW-timedelta(seconds=1),'expires_at':NOW+timedelta(minutes=1),'maximum_order_quantity':order.quantity,'maximum_capital_cents':10**8})
    auth=operator.sign(declaration,created_at=NOW)
    authorization=AuthorizationJournal(ctx.store,signer=ctx.registry.signer,operator_trust_store=operator_trust)
    risk=Ed25519Signer.generate();risk_trust=TrustStore();risk_trust.add(risk.trusted_key(valid_from=NOW-timedelta(days=1)))
    signed_packet=risk.sign(SignedRiskPacket(actor_id='risk',occurred_at=NOW,**json.loads(canonical_bytes(dict(policy=asdict(policy),request=asdict(request),decision=asdict(decision),account=asdict(account),capabilities=asdict(caps))))),created_at=NOW)
    risk_receipt=sign_risk_intent_receipt(decision,intents,account,caps,signer=risk,created_at=NOW)
    journal=ExecutionJournal(ctx.store,signer=ctx.registry.signer,actor_id='offline-fence',operator_trust_store=operator_trust,receipt_trust_store=ctx.store.trust_store)
    binding=AuthorityBindingEvent(sequence=0,previous_digest=None,action='REGISTER',actor_id=journal.actor_id,key_id=journal.signer.key_id,occurred_at=NOW)
    journal.configure(operator.sign(binding,created_at=NOW));fence=OfflineExecutionFence(journal,realm='shadow:artifact');token=fence.acquire('offline-one',ttl_seconds=30)
    event=LifecycleEvent(sequence=0,previous_digest=None,order_identity=order.intent_id,intent_digest=scope['intent_digest'],authorization_digest=auth.payload_digest,account_alias=order.account_alias,maximum_quantity=order.quantity,filled_quantity=0,state='PROPOSED',actor_id='gateway',role='GATEWAY',occurred_at=NOW)
    arguments=dict(artifact_resolver=resolver,artifact_pointer_envelope=pointer,
        fence=fence,fence_token=token,intent=order,intents=intents,risk_envelope=risk_receipt,risk_trust_store=risk_trust,
        risk_packet_envelope=signed_packet,risk_actor_bindings={risk.key_id:'risk'},account=account,capabilities=caps,
        authorization_envelope=auth,authorization_journal=authorization,expected_scope=scope,capital_cents=10**8,expected_head=None)
    return kernel,producer,observer,producer.sign(event,created_at=NOW),arguments,request


@pytest.mark.parametrize('change',['none','source','session'])
def test_resolved_artifacts_reach_fenced_durable_admission(tmp_path,change):
    kernel,producer,observer,envelope,arguments,request=resolved_ingress(tmp_path,change)
    pointer=arguments['artifact_pointer_envelope']
    resolved=arguments['artifact_resolver'].resolve(pointer)
    if change!='none':
        with pytest.raises(ValueError,match='classification/session'):kernel.reserve_resolved_proposal(envelope,**arguments)
        assert kernel.replay()==()
        return
    receipt=kernel.reserve_resolved_proposal(envelope,**arguments)
    assert receipt.artifact_pointer_envelope==pointer and receipt.fence_reservation is not None
    assert kernel.replay()[0].artifact_resolution['bundle_digest']==resolved.bundle_digest
    assert receipt.execution_enabled is False and receipt.capital_authorized is False


def test_expiry_during_second_resolution_is_rejected(tmp_path,monkeypatch):
    resolver,ctx,root,put,bundle,signers,trusts,pointer=fixture(tmp_path)
    original=resolver._resolve;calls=0
    def expire_on_second(envelope,now):
        nonlocal calls
        result=original(envelope,now);calls+=1
        if calls==2:ctx.store._trusted_clock=lambda:NOW+timedelta(hours=7)
        return result
    monkeypatch.setattr(resolver,'_resolve',expire_on_second)
    with pytest.raises(ValueError,match='inactive'):resolver.resolve(pointer)
    assert not (ctx.store.root/'offline-artifact-sessions').exists()


@pytest.mark.parametrize('semantic',['drawdown','purge','economic'])
def test_consistently_hashed_and_signed_false_semantics_are_rejected(tmp_path,semantic):
    resolver,ctx,root,put,bundle,signers,trusts,pointer=fixture(tmp_path,semantic)
    with pytest.raises(ValueError):resolver.resolve(pointer)
