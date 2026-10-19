from dataclasses import replace
from datetime import timedelta
import hashlib

import pytest

from tests.evidence.equity_research._support import NOW, build_context, proposal
from src.evidence.canonical import canonical_bytes
from src.evidence.contracts import AuthorityRole
from src.evidence.signing import Ed25519Signer, verify_envelope

def verified_fixture(tmp_path, *, threshold=.001):
    from src.axiom2.contracts.research_proposal import DataDependency
    from src.axiom2.promotion.authority import PromotionRule, promotion_rule_digest
    from src.axiom2.research.ranker import CampaignPayload, CampaignResult
    ctx=build_context(tmp_path/"store")
    promo=Ed25519Signer.generate()
    ctx.store.trust_store.add(promo.trusted_key(valid_from=NOW-timedelta(days=1)))
    ctx.store.authorities.register(
        actor_id="promotion_service",role=AuthorityRole.PROMOTION_SERVICE,key_ids=(promo.key_id,)
    )
    rule=PromotionRule(min_excess_return=threshold)
    campaign_digest="c"*64; search_digest="d"*64
    base=proposal("verified-promotion")
    deps=base.data_dependencies+(
        DataDependency(dataset_id=campaign_digest,source="axiom2-campaign-manifest",schema_version="1"),
        DataDependency(dataset_id=search_digest,source="axiom2-hyperparameter-search",schema_version="1"),
        DataDependency(dataset_id=promotion_rule_digest(rule),source="axiom2-promotion-rule",schema_version="1"),
    )
    p=replace(base,data_dependencies=deps)
    ctx.registry.register_experiment(p)
    dataset_hash=hashlib.sha256(b"sacrificial-hidden-fixture").hexdigest()
    ctx.operator.register_holdout("sacrificial",dataset_hash,NOW+timedelta(days=1),NOW+timedelta(days=2))
    config_digest=hashlib.sha256(b"config").hexdigest()
    trial=ctx.registry.record_trial_started(p.experiment_id,campaign_digest,0,config_digest)
    trial_result=hashlib.sha256(b"registered-trial-result").hexdigest()
    ctx.registry.record_trial_succeeded(trial.trial_id,trial_result)
    model_bytes=b"verified-model-artifact"
    model_hash=hashlib.sha256(model_bytes).hexdigest()
    artifact_hash=hashlib.sha256(b"campaign-summary").hexdigest()
    best_params=()
    best_params_digest=hashlib.sha256(canonical_bytes({})).hexdigest()
    payload=CampaignPayload(
        artifact_hash=artifact_hash,model_artifact_hash=model_hash,proposal_id=p.experiment_id,
        model_family="lightgbm",search_space_digest=search_digest,feature_digest="1"*64,
        label_digest="2"*64,split_digest="3"*64,campaign_manifest_digest=campaign_digest,
        selected_trial_id=trial.trial_id,selected_trial_result_digest=trial_result,
        best_params_digest=best_params_digest,experiment_count=1,mean_rank_correlation=.10,
        portfolio_net_return=.020,baseline_net_return=.005,turnover=.20,max_drawdown=-.10,
        positive_fold_fraction=1.0,
    )
    envelope=ctx.registry.signer.sign(payload,created_at=NOW)
    campaign=CampaignResult(
        model_type="lightgbm",experiment_count=1,mean_rank_correlation=.10,
        portfolio_net_return=.020,baseline_net_return=.005,turnover=.20,max_drawdown=-.10,
        regime_slices=(("all",.020),),artifact_hash=artifact_hash,search_space_digest=search_digest,
        feature_digest="1"*64,label_digest="2"*64,split_digest="3"*64,
        campaign_manifest_digest=campaign_digest,selected_trial_id=trial.trial_id,
        selected_trial_result_digest=trial_result,model_artifact_hash=model_hash,
        model_artifact_bytes=model_bytes,best_params=best_params,positive_fold_fraction=1.0,
        holdout_accessed=False,envelope=envelope,
    )
    candidate=ctx.registry.freeze_candidate(p.experiment_id,model_hash,p.code_commit)
    ctx.evaluator.open_holdout(candidate.candidate_id,"sacrificial")
    result_bytes=canonical_bytes({
        "candidate_id":candidate.candidate_id,"holdout_id":"sacrificial",
        "net_return":.020,"cost_evidence_id":p.cost_model_id,"schema_version":1,
    })
    ctx.evaluator.consume_holdout(
        candidate.candidate_id,"sacrificial",hashlib.sha256(result_bytes).hexdigest()
    )
    return ctx,p,campaign,candidate,rule,promo,result_bytes

def test_verified_promotion_resolves_complete_evidence_and_role(tmp_path):
    from src.axiom2.promotion.authority import (
        PromotionDecisionPayload,evaluate_verified_promotion,
    )
    ctx,p,campaign,candidate,rule,promo,result_bytes=verified_fixture(tmp_path)
    decision=evaluate_verified_promotion(
        proposal=p,campaign=campaign,candidate_id=candidate.candidate_id,
        holdout_id="sacrificial",holdout_result_bytes=result_bytes,registry=ctx.registry,
        promotion_rule=rule,signer=promo,actor_id="promotion_service",created_at=NOW,
    )
    assert decision.status=="PROMOTED"
    assert decision.verified is True and decision.evidence_digest
    payload=verify_envelope(decision.envelope,PromotionDecisionPayload,ctx.store.trust_store)
    assert payload.decision_hash==decision.decision_hash
    assert payload.evidence_digest==decision.evidence_digest

def test_scalar_promotion_evaluator_is_never_verified_authority():
    from src.axiom2.promotion.authority import PromotionPackage,evaluate_promotion
    signer=Ed25519Signer.generate()
    decision=evaluate_promotion(PromotionPackage(
        candidate_id="c"*64,candidate_artifact_hash="a"*64,frozen_artifact_hash="a"*64,
        baseline_net_return=.001,candidate_net_return=.01,cost_evidence_id="c",
        experiment_count=1,registered_experiment_count=1,holdout_id="h",
        holdout_status="CONSUMED",holdout_result_hash="b"*64,holdout_candidate_id="c"*64,
        min_excess_return=.001,holdout_net_return=.01,holdout_consumption_count=1,
    ),signer=signer,created_at=NOW)
    assert decision.status=="REJECTED" and decision.verified is False
    assert "unverified scalar promotion package" in decision.reasons
def test_verified_promotion_rejects_wrong_signer_role(tmp_path):
    from src.axiom2.promotion.authority import evaluate_verified_promotion
    ctx,p,campaign,candidate,rule,_,result_bytes=verified_fixture(tmp_path)
    with pytest.raises(ValueError,match="promotion_service"):
        evaluate_verified_promotion(
            proposal=p,campaign=campaign,candidate_id=candidate.candidate_id,
            holdout_id="sacrificial",holdout_result_bytes=result_bytes,registry=ctx.registry,
            promotion_rule=rule,signer=ctx.signers[AuthorityRole.PRODUCER],
            actor_id="producer",created_at=NOW,
        )

def test_verified_promotion_rejects_unregistered_rule_or_fabricated_candidate(tmp_path):
    from src.axiom2.promotion.authority import PromotionRule,evaluate_verified_promotion
    ctx,p,campaign,candidate,rule,promo,result_bytes=verified_fixture(tmp_path)
    with pytest.raises(ValueError,match="promotion rule"):
        evaluate_verified_promotion(
            proposal=p,campaign=campaign,candidate_id=candidate.candidate_id,
            holdout_id="sacrificial",holdout_result_bytes=result_bytes,registry=ctx.registry,
            promotion_rule=PromotionRule(min_excess_return=.002),signer=promo,
            actor_id="promotion_service",created_at=NOW,
        )
    with pytest.raises(ValueError,match="frozen candidate"):
        evaluate_verified_promotion(
            proposal=p,campaign=campaign,candidate_id="f"*64,holdout_id="sacrificial",
            holdout_result_bytes=result_bytes,registry=ctx.registry,promotion_rule=rule,
            signer=promo,actor_id="promotion_service",created_at=NOW,
        )

def test_verified_promotion_rejects_result_hash_or_attempt_history_drift(tmp_path):
    from src.axiom2.promotion.authority import evaluate_verified_promotion
    ctx,p,campaign,candidate,rule,promo,result_bytes=verified_fixture(tmp_path)
    with pytest.raises(ValueError,match="holdout result hash"):
        evaluate_verified_promotion(
            proposal=p,campaign=campaign,candidate_id=candidate.candidate_id,
            holdout_id="sacrificial",holdout_result_bytes=result_bytes+b"x",registry=ctx.registry,
            promotion_rule=rule,signer=promo,actor_id="promotion_service",created_at=NOW,
        )
    ctx.registry.register_experiment(proposal("post-holdout-attempt"))
    with pytest.raises(ValueError,match="attempt history"):
        evaluate_verified_promotion(
            proposal=p,campaign=campaign,candidate_id=candidate.candidate_id,
            holdout_id="sacrificial",holdout_result_bytes=result_bytes,registry=ctx.registry,
            promotion_rule=rule,signer=promo,actor_id="promotion_service",created_at=NOW,
        )

def test_promotion_rule_cannot_encode_negative_edge():
    from src.axiom2.promotion.authority import PromotionRule
    with pytest.raises(ValueError,match="nonnegative"):
        PromotionRule(min_excess_return=-1.0)


def test_scalar_compatibility_evaluator_can_never_emit_promoted_status():
    from src.axiom2.promotion.authority import PromotionPackage,evaluate_promotion
    from src.evidence.signing import Ed25519Signer
    package=PromotionPackage(
        candidate_id="c"*64,candidate_artifact_hash="a"*64,frozen_artifact_hash="a"*64,
        baseline_net_return=.001,candidate_net_return=.01,cost_evidence_id="cost",
        experiment_count=1,registered_experiment_count=1,holdout_id="fabricated",
        holdout_status="CONSUMED",holdout_result_hash="b"*64,holdout_candidate_id="c"*64,
        min_excess_return=0.0,holdout_net_return=.02,holdout_consumption_count=1,
    )
    decision=evaluate_promotion(package,signer=Ed25519Signer.generate())
    assert decision.status=="REJECTED"
    assert decision.verified is False
    assert "unverified scalar promotion package" in decision.reasons
