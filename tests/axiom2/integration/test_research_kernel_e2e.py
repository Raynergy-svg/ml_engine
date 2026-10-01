from datetime import datetime,timezone,timedelta
import hashlib
import pytest

def test_research_kernel_e2e_and_single_use_holdout(tmp_path):
    from tests.evidence.equity_research._support import build_context,proposal
    from src.evidence.contracts import AuthorityRole
    from src.evidence.equity_research.holdout import HoldoutAuthority
    from src.axiom2.promotion.authority import PromotionPackage,evaluate_promotion
    ctx=build_context(tmp_path/"store"); p=proposal("e2e"); ctx.registry.register_experiment(p)
    holdout=ctx.operator; hd=hashlib.sha256(b"hidden").hexdigest(); holdout.register_holdout("hidden",hd,datetime(2025,2,1,tzinfo=timezone.utc),datetime(2025,3,1,tzinfo=timezone.utc))
    artifact=hashlib.sha256(b"candidate").hexdigest(); c=ctx.registry.freeze_candidate(p.experiment_id,artifact,p.code_commit)
    holdout=ctx.evaluator; opened=holdout.open_holdout(c.candidate_id,"hidden"); result_hash=hashlib.sha256(b"positive-result").hexdigest(); consumed=holdout.consume_holdout(c.candidate_id,"hidden",result_hash)
    with pytest.raises(ValueError): holdout.open_holdout(c.candidate_id,"hidden")
    pkg=PromotionPackage(candidate_id=c.candidate_id,candidate_artifact_hash=artifact,frozen_artifact_hash=artifact,baseline_net_return=.001,candidate_net_return=.006,cost_evidence_id="cost",experiment_count=1,registered_experiment_count=1,holdout_id="hidden",holdout_status=consumed.status,holdout_result_hash=result_hash,holdout_candidate_id=c.candidate_id,min_excess_return=.001,holdout_net_return=.005,holdout_consumption_count=1)
    d=evaluate_promotion(pkg,signer=ctx.signers[AuthorityRole.INDEPENDENT_VERIFIER],created_at=datetime.now(timezone.utc)); assert d.status=="REJECTED" and d.envelope is not None and d.verified is False

def test_future_data_and_unknown_membership_fail_closed():
    from src.axiom2.data.temporal import TemporalRecord,assert_trainable
    from src.axiom2.data.universe import UniverseManifest,build_universe_as_of
    now=datetime(2025,1,1,tzinfo=timezone.utc); r=TemporalRecord(event_time=now,published_time=None,available_to_axiom_time=now+timedelta(days=2),ingested_time=now+timedelta(days=3),source="x",revision="1",schema_version="1")
    with pytest.raises(ValueError): assert_trainable(r,now+timedelta(days=1))
    m=UniverseManifest(universe_id="u",source="x",evidence_id="e",schema_version="1",membership_basis="sp500",coverage_start=now,coverage_end=now+timedelta(days=1),memberships=())
    with pytest.raises(ValueError): build_universe_as_of(now+timedelta(days=2),m)
