"""Synthetic controls only; no market data or genuine holdout reads."""
import copy
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.evidence.canonical import canonical_bytes
from src.evidence.contracts import AuthorityRole
from tests.evidence.equity_research._support import build_context, NOW


def fixture_bundle(root, ctx, poison=None):
    from src.axiom2.research.development import SourceAudit
    dates=pd.bdate_range('2022-01-03','2024-06-28',tz='America/New_York')
    sessions=[]; bars=[]; references=[]; eligibility=[]
    for i,date in enumerate(dates):
        opening=date+pd.Timedelta(hours=9,minutes=30)
        close=date+pd.Timedelta(hours=16)
        sessions.append(dict(open=opening.isoformat(),close=close.isoformat(),decision=(opening-pd.Timedelta(minutes=5)).isoformat()))
        references.append(dict(timestamp=close.isoformat(),available=(close+pd.Timedelta(minutes=5)).isoformat(),benchmark=100*(1.0002**i),benchmark_open=100*(1.0002**i),sectors={'fixture':100*(1.0001**i)},adjustment='fixture-v1'))
        eligibility.append(dict(decision=(opening-pd.Timedelta(minutes=5)).isoformat(),available=(date+pd.Timedelta(hours=8)).isoformat(),members=[dict(instrument_id=s,sector='fixture',eligible=True,reason='pit-liquid') for s in 'ABCDEF']))
        for j,s in enumerate('ABCDEF'):
            price=100*((1+.001*(j+1))**i)*(1+.0005*np.sin(i+j))
            bars.append(dict(timestamp=close.isoformat(),instrument_id=s,open=price,close=price*1.0001,volume=2_000_000+i*500+j*1000,available=(close+pd.Timedelta(minutes=5)).isoformat(),sector='fixture',adjustment='fixture-v1',outcome='resolved'))
    panel=dict(sessions=sessions,bars=bars,references=references,eligibility=eligibility)
    if poison: poison(panel)
    root.mkdir(parents=True)
    evidence={name:('synthetic evidence '+name).encode() for name in ('timestamps','revisions','membership','sectors','corporate_actions','calendar','liquidity','identity')}
    raw=canonical_bytes(panel); (root/'panel.json').write_bytes(raw)
    for name,data in evidence.items(): (root/(name+'.txt')).write_bytes(data)
    audit=SourceAudit(actor_id='independent_verifier',classification='synthetic-development',coverage_start='2022-01-03',coverage_end='2024-06-28',panel_digest=hashlib.sha256(raw).hexdigest(),evidence_digests={name:hashlib.sha256(data).hexdigest() for name,data in evidence.items()},calendar_id='synthetic-weekdays',adjustment_id='fixture-v1',availability_basis='historical-publication-and-delivery',membership_basis='decision-time-sp500-snapshots',liquidity_policy='synthetic-pit-liquid',terminal_policy='resolved-opening-total-return-proxy',audit_explanation='Synthetic positive control; never empirical certification.')
    envelope=ctx.signers[AuthorityRole.INDEPENDENT_VERIFIER].sign(audit,created_at=NOW)
    (root/'admission.json').write_bytes(canonical_bytes(envelope))
    return root


def admit(root, ctx):
    from src.axiom2.research.development import admit_development_bundle
    return admit_development_bundle(root,store=ctx.store,allow_synthetic=True)


def test_positive_control_builds_same_five_arms_and_actual_end_purges(tmp_path):
    ctx=build_context(tmp_path/'store'); root=fixture_bundle(tmp_path/'data',ctx)
    plan=admit(root,ctx)
    assert tuple(plan.features)==('P','PV','PR','PM','FULL')
    assert len(plan.splits)==5
    assert all(len(split.train)==252 for split in plan.splits)
    assert all(x.index.equals(plan.features['P'].index) for x in plan.features.values())
    for split in plan.splits:
        assert max(plan.label_ends.loc[list(split.train)]) < min(split.test)
    assert plan.features['FULL'].shape[1]==7
    assert plan.labels.notna().all().all()
    assert not ctx.registry.snapshot().experiments


@pytest.mark.parametrize('poison,reason',[
    (lambda p:p['bars'][30].update(available=None),'availability'),
    (lambda p:p['bars'][30].update(available='2024-06-28T23:00:00Z'),'availability'),
    (lambda p:p['eligibility'][10].update(member_until='2024-06-28'),'eligibility'),
    (lambda p:p['sessions'][10].update(decision=p['sessions'][10]['open']),'entry'),
    (lambda p:p['bars'][90].update(outcome='unresolved-delisting'),'outcome'),
    (lambda p:p['bars'][90].pop('volume'),'fields'),
    (lambda p:p['references'].pop(10),'paired'),
    (lambda p:p['bars'][90].update(adjustment='future-v2'),'adjustment'),
    (lambda p:p['bars'][90].update(open=None),'opening'),
    (lambda p:p['eligibility'][10]['members'].pop(),'coverage'),
])
def test_poison_aborts_before_registration(tmp_path,poison,reason):
    ctx=build_context(tmp_path/'store'); root=fixture_bundle(tmp_path/'data',ctx,poison)
    with pytest.raises(ValueError,match=reason): admit(root,ctx)
    assert not ctx.registry.snapshot().experiments


def test_altered_bytes_fail_even_with_valid_signed_audit(tmp_path):
    ctx=build_context(tmp_path/'store'); root=fixture_bundle(tmp_path/'data',ctx)
    with (root/'panel.json').open('ab') as f:f.write(b' ')
    with pytest.raises(ValueError,match='digest'): admit(root,ctx)


def test_holdout_classification_denied_before_reading_panel(tmp_path,monkeypatch):
    from src.axiom2.research.development import SourceAudit
    from src.evidence.contracts import SignedEnvelope
    from src.evidence.signing import verify_envelope
    ctx=build_context(tmp_path/'store'); root=fixture_bundle(tmp_path/'data',ctx)
    env=SignedEnvelope.model_validate_json((root/'admission.json').read_bytes())
    audit=verify_envelope(env,SourceAudit,ctx.store.trust_store)
    data=audit.model_dump(); data['classification']='holdout'
    # Even an authorized auditor cannot admit a holdout classification.
    env=ctx.signers[AuthorityRole.INDEPENDENT_VERIFIER].sign(SourceAudit(**data),created_at=NOW)
    (root/'admission.json').write_bytes(canonical_bytes(env))
    original=Path.read_bytes
    def guard(path):
        if path.name=='panel.json':pytest.fail('holdout panel was opened')
        return original(path)
    monkeypatch.setattr(Path,'read_bytes',guard)
    with pytest.raises(ValueError,match='classification'): admit(root,ctx)


def test_untrusted_audit_cannot_self_certify(tmp_path):
    from src.evidence.signing import Ed25519Signer
    from src.axiom2.research.development import SourceAudit
    from src.evidence.contracts import SignedEnvelope
    from src.evidence.signing import verify_envelope
    ctx=build_context(tmp_path/'store'); root=fixture_bundle(tmp_path/'data',ctx)
    audit=verify_envelope(SignedEnvelope.model_validate_json((root/'admission.json').read_bytes()),SourceAudit,ctx.store.trust_store)
    (root/'admission.json').write_bytes(canonical_bytes(Ed25519Signer.generate().sign(audit,created_at=NOW)))
    with pytest.raises(Exception):admit(root,ctx)


def test_actual_label_overlap_rejected(tmp_path):
    from src.axiom2.research.development import validate_paired_plan
    from src.axiom2.research.splits import WalkForwardSplit
    ctx=build_context(tmp_path/'store'); plan=admit(fixture_bundle(tmp_path/'data',ctx),ctx)
    split=plan.splits[0]
    plan.splits=(WalkForwardSplit(split.train+(split.test[0]-pd.Timedelta(days=1),),split.test),)+plan.splits[1:]
    with pytest.raises(ValueError,match='purge|split'):validate_paired_plan(plan)


def test_mismatched_arm_dates_rejected(tmp_path):
    from src.axiom2.research.development import validate_paired_plan
    ctx=build_context(tmp_path/'store'); plan=admit(fixture_bundle(tmp_path/'data',ctx),ctx)
    plan.features['PV']=plan.features['PV'].iloc[1:]
    with pytest.raises(ValueError,match='paired|altered'):validate_paired_plan(plan)


def test_finite_five_arm_campaign_retains_signed_reports_and_history(tmp_path):
    from src.axiom2.research.development import run_development_comparison, verify_comparison_report
    from tests.evidence.equity_research._support import proposal
    ctx=build_context(tmp_path/'store')
    ctx.registry.register_experiment(proposal('old-failed'));ctx.registry.record_failure('old-failed','historical failure retained')
    plan=admit(fixture_bundle(tmp_path/'data',ctx),ctx)
    report=run_development_comparison(plan,registry=ctx.registry,code_commit='a'*40)
    verified=verify_comparison_report(report,registry=ctx.registry)
    assert verified.arm_order==('P','PV','PR','PM','FULL')
    assert verified.historical_attempt_count==1
    assert len(ctx.registry.snapshot().trials)==5
    assert ctx.registry.snapshot().experiments['old-failed'].failure_reason
    for arm in verified.arms.values():
        assert arm['status']=='SUCCEEDED'
        assert set(arm['stress_bps'])=={'10','25','50'}
        assert len(arm['fold_ic'])==5
        assert arm['gate_digest']=='3117d82f92a2ba707fd4d155af4a6391e364cbcc555dc17fa8841caae161eb0b'
    assert not ctx.registry.snapshot().holdouts
    assert not ctx.registry.snapshot().candidates
    # Same campaign cannot silently fit the already-successful trials again.
    with pytest.raises(ValueError,match='already'):run_development_comparison(plan,registry=ctx.registry,code_commit='a'*40)
    # A missing trial must invalidate the report, not reduce the stated search count.
    from dataclasses import replace
    snapshot=ctx.registry.snapshot(); original=ctx.registry.snapshot
    ctx.registry.snapshot=lambda:replace(snapshot,trials={})
    with pytest.raises(ValueError,match='trial'):verify_comparison_report(report,registry=ctx.registry)
    ctx.registry.snapshot=original


def test_campaign_failure_retained_and_remaining_arms_not_fitted(tmp_path,monkeypatch):
    import src.axiom2.research.ranker as ranker
    from src.axiom2.research.development import run_development_comparison,verify_comparison_report
    ctx=build_context(tmp_path/'store');plan=admit(fixture_bundle(tmp_path/'data',ctx),ctx)
    def poison(*args,**kwargs):raise RuntimeError('synthetic fold fit poison')
    monkeypatch.setattr(ranker.LightGBMRankerAdapter,'fit',poison)
    monkeypatch.setitem(ranker._AUTHORIZED_LIGHTGBM_METHODS,'fit',poison)
    report=verify_comparison_report(run_development_comparison(plan,registry=ctx.registry,code_commit='a'*40),registry=ctx.registry)
    assert report.arms['P']['status']=='FAILED'
    assert all(report.arms[x]['status']=='NOT_RUN' for x in ('PV','PR','PM','FULL'))
    assert len(ctx.registry.snapshot().trials)==1
    assert next(iter(ctx.registry.snapshot().trials.values())).status=='FAILED'


def test_bootstrap_positive_and_null_controls():
    from src.axiom2.research.development import paired_bounds
    assert paired_bounds([.01]*60,[i//12 for i in range(60)])['lower_97_5']==pytest.approx(.01)
    assert paired_bounds([0.]*60,[i//12 for i in range(60)])['lower_97_5']==0
    rng=np.random.default_rng(11); rejections=0
    for _ in range(40):
        series=rng.normal(0,.02,60)
        rejections+=paired_bounds(series,[i//12 for i in range(60)])['lower_97_5']>0
    assert rejections<=4


def test_service_allowlist_and_synthetic_default_deny(tmp_path):
    from src.axiom2.research.development import handle_development_request
    ctx=build_context(tmp_path/'store');root=fixture_bundle(tmp_path/'data',ctx)
    with pytest.raises(ValueError,match='allowlist'):
        handle_development_request({'action':'run','snapshot_id':'../../holdout'},registry=ctx.registry,bundles={'fixture':root},code_commit='a'*40)
    with pytest.raises(ValueError,match='classification'):
        handle_development_request({'action':'preflight','snapshot_id':'fixture'},registry=ctx.registry,bundles={'fixture':root},code_commit='a'*40)
    answer=handle_development_request({'action':'preflight','snapshot_id':'fixture'},registry=ctx.registry,bundles={'fixture':root},code_commit='a'*40,allow_synthetic=True)
    assert answer['status']=='ADMITTED'
    assert answer['classification']=='synthetic-development'
    assert not ctx.registry.snapshot().experiments


def test_every_pipeline_stage_is_required(tmp_path,monkeypatch):
    import src.axiom2.research.development as development
    ctx=build_context(tmp_path/'store');root=fixture_bundle(tmp_path/'data',ctx)
    def poison(*args,**kwargs):raise RuntimeError('POISONED_REQUIRED_STAGE')
    monkeypatch.setattr(development,'build_phase1_features',poison)
    with pytest.raises(RuntimeError,match='POISONED_REQUIRED_STAGE'):admit(root,ctx)


def test_null_bootstrap_is_not_an_empirical_promotion(tmp_path):
    from src.axiom2.research.development import paired_bounds
    assert paired_bounds([0.]*60,[i//12 for i in range(60)])['lower_97_5']==0


def test_report_verifier_rejects_altered_diagnostics(tmp_path):
    from src.axiom2.research.development import run_development_comparison,verify_comparison_report,ComparisonReport,_persist
    from src.evidence.contracts import SignedEnvelope
    from src.evidence.signing import verify_envelope
    ctx=build_context(tmp_path/'store');plan=admit(fixture_bundle(tmp_path/'data',ctx),ctx)
    key=run_development_comparison(plan,registry=ctx.registry,code_commit='a'*40)
    path=ctx.store.root/'development'/(key+'.json')
    report=verify_envelope(SignedEnvelope.model_validate_json(path.read_bytes()),ComparisonReport,ctx.store.trust_store)
    data=json.loads(report.model_dump_json());data['arms']['P']['book'][0][2]=9.
    changed=_persist(ctx.registry,ctx.registry.signer.sign(ComparisonReport.model_validate_json(json.dumps(data)),created_at=NOW))
    with pytest.raises(ValueError,match='diagnostic'):verify_comparison_report(changed,registry=ctx.registry)


def test_cli_delegates_only_snapshot_and_action_to_service(tmp_path):
    import subprocess
    import sys
    child=subprocess.Popen([sys.executable,'scripts/axiom2_run_phase1_campaign.py','--service-stdio','--snapshot-id','qualified-dev','--action','preflight'],stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True)
    output,error=child.communicate('{"status":"ADMITTED","execution_enabled":false,"holdout_accessed":false}\n',timeout=10)
    assert child.returncode==0,error
    request,response=output.splitlines()
    assert json.loads(request)=={'action':'preflight','snapshot_id':'qualified-dev'}
    assert json.loads(response)['status']=='ADMITTED'


@pytest.mark.parametrize('name',('build_phase1_features','build_forward_rank_labels','purged_walk_forward_splits'))
def test_admission_depends_on_existing_kernel_stages(tmp_path,monkeypatch,name):
    import src.axiom2.research.development as development
    ctx=build_context(tmp_path/'store');root=fixture_bundle(tmp_path/'data',ctx)
    def poison(*args,**kwargs):raise RuntimeError('POISONED_REQUIRED_STAGE:'+name)
    monkeypatch.setattr(development,name,poison)
    with pytest.raises(RuntimeError,match='POISONED_REQUIRED_STAGE'):admit(root,ctx)
    assert not ctx.registry.snapshot().experiments


def test_future_removal_does_not_erase_earlier_pit_cohort(tmp_path):
    ctx=build_context(tmp_path/'store')
    def removal(panel):
        for snapshot in panel['eligibility'][350:]:
            snapshot['members'][-1].update(eligible=False,reason='pit-removed')
    plan=admit(fixture_bundle(tmp_path/'data',ctx,removal),ctx)
    early=plan.features['FULL'].loc[plan.splits[0].test[0]]
    assert 'F' in early.index
    late=plan.features['FULL'].loc[plan.splits[-1].test[-1]]
    assert 'F' not in late.index
    assert any(x[1]=='F' for x in plan.exclusions)


def test_changed_source_audit_is_rejected(tmp_path):
    from src.axiom2.research.development import validate_paired_plan
    ctx=build_context(tmp_path/'store');plan=admit(fixture_bundle(tmp_path/'data',ctx),ctx)
    plan.audit=plan.audit.model_copy(update={'classification':'development'})
    with pytest.raises(ValueError,match='audit'):validate_paired_plan(plan)


def test_labels_rank_only_decision_eligible_names(tmp_path):
    ctx=build_context(tmp_path/'store')
    def removal(panel):
        for snapshot in panel['eligibility'][350:]:snapshot['members'][-1].update(eligible=False,reason='pit-removed')
    plan=admit(fixture_bundle(tmp_path/'data',ctx,removal),ctx)
    ranks=plan.labels.loc[plan.splits[-1].test[-1],'cross_sectional_rank']
    assert sorted(ranks)==pytest.approx([.2,.4,.6,.8,1.])


def test_service_stdio_rejects_unknown_snapshot_without_opening_data(tmp_path):
    from io import StringIO
    from src.axiom2.research.development import serve_development_stdio
    ctx=build_context(tmp_path/'store');reader=StringIO('{"action":"run","snapshot_id":"holdout"}\n');writer=StringIO()
    serve_development_stdio(reader,writer,registry=ctx.registry,bundles={},code_commit='a'*40)
    response=json.loads(writer.getvalue())
    assert response['status']=='REJECTED'
    assert not response['execution_enabled'] and not response['holdout_accessed']
    assert not ctx.registry.snapshot().experiments


def _fresh_signed_comparison_report(tmp_path):
    from src.axiom2.research.development import run_development_comparison, ComparisonReport, _persist
    from src.evidence.contracts import SignedEnvelope
    from src.evidence.signing import verify_envelope

    ctx=build_context(tmp_path/'store')
    plan=admit(fixture_bundle(tmp_path/'data',ctx),ctx)
    key=run_development_comparison(plan,registry=ctx.registry,code_commit='a'*40)
    raw=(ctx.store.root/'development'/(key+'.json')).read_bytes()
    report=verify_envelope(
        SignedEnvelope.model_validate_json(raw),ComparisonReport,ctx.store.trust_store
    )
    return ctx, report


def _persist_signed_comparison_report(ctx, data):
    from src.axiom2.research.development import ComparisonReport, _persist

    return _persist(
        ctx.registry,
        ctx.registry.signer.sign(
            ComparisonReport.model_validate(data),created_at=NOW
        ),
    )


def test_report_verifier_recomputes_signed_derived_conclusions(tmp_path):
    ctx, report=_fresh_signed_comparison_report(tmp_path)
    mutations=(
        ('gate', lambda data: data['arms']['P']['gate'].__setitem__('ic',False)),
        ('stress', lambda data: data['arms']['P']['stress_bps']['25'].__setitem__('net_excess_return',999.0)),
        ('paired', lambda data: data['paired']['FULL_minus_P'].__setitem__('lower_97_5',999.0)),
        ('primary', lambda data: data.__setitem__('primary_hypothesis_pass',not data['primary_hypothesis_pass'])),
    )
    for name, mutate in mutations:
        data=json.loads(report.model_dump_json())
        mutate(data)
        key=_persist_signed_comparison_report(ctx,data)
        with pytest.raises(ValueError,match=name):
            from src.axiom2.research.development import verify_comparison_report
            verify_comparison_report(key,registry=ctx.registry)


def test_report_verifier_binds_campaign_trial_result_to_registry(tmp_path):
    from src.axiom2.research.development import (
        CampaignDiagnosticsPayload,CampaignPayload,ComparisonReport,_persist,
        run_development_comparison,verify_comparison_report,
    )
    from src.evidence.contracts import SignedEnvelope
    from src.evidence.signing import verify_envelope

    ctx=build_context(tmp_path/'store')
    plan=admit(fixture_bundle(tmp_path/'data',ctx),ctx)
    key=run_development_comparison(plan,registry=ctx.registry,code_commit='a'*40)
    report_path=ctx.store.root/'development'/(key+'.json')
    report=verify_envelope(
        SignedEnvelope.model_validate_json(report_path.read_bytes()),
        ComparisonReport,ctx.store.trust_store
    )
    arm=report.arms['P']

    campaign_raw=(ctx.store.root/'development'/(arm['campaign_evidence']+'.json')).read_bytes()
    campaign=verify_envelope(
        SignedEnvelope.model_validate_json(campaign_raw),
        CampaignPayload,ctx.store.trust_store
    )
    diagnostic_raw=(ctx.store.root/'development'/(arm['diagnostic_evidence']+'.json')).read_bytes()
    diagnostic=verify_envelope(
        SignedEnvelope.model_validate_json(diagnostic_raw),
        CampaignDiagnosticsPayload,ctx.store.trust_store
    )
    bad_result_digest='f'*64
    campaign_data=campaign.model_dump(mode='json')
    campaign_data['selected_trial_result_digest']=bad_result_digest
    diagnostic_data=diagnostic.model_dump(mode='json')
    diagnostic_data['selected_trial_result_digest']=bad_result_digest
    campaign_key=_persist(
        ctx.registry,
        ctx.registry.signer.sign(
            CampaignPayload.model_validate(campaign_data),created_at=NOW
        ),
    )
    diagnostic_key=_persist(
        ctx.registry,
        ctx.registry.signer.sign(
            CampaignDiagnosticsPayload.model_validate(diagnostic_data),created_at=NOW
        ),
    )
    report_data=json.loads(report.model_dump_json())
    report_data['arms']['P']['campaign_evidence']=campaign_key
    report_data['arms']['P']['diagnostic_evidence']=diagnostic_key
    inconsistent_key=_persist_signed_comparison_report(ctx,report_data)

    with pytest.raises(ValueError,match='trial'):
        verify_comparison_report(inconsistent_key,registry=ctx.registry)
