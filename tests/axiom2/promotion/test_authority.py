from dataclasses import replace
import pytest

def package(**kw):
    from src.axiom2.promotion.authority import PromotionPackage
    base=dict(candidate_id='c1',candidate_artifact_hash='a'*64,frozen_artifact_hash='a'*64,baseline_net_return=.001,candidate_net_return=.005,cost_evidence_id='cost-v1',experiment_count=4,registered_experiment_count=4,holdout_id='h1',holdout_status='CONSUMED',holdout_result_hash='b'*64,holdout_candidate_id='c1',min_excess_return=.001,holdout_net_return=.004,holdout_consumption_count=1)
    base.update(kw); return PromotionPackage(**base)

def test_rejects_missing_baseline_cost_accounting_mutation_reuse_and_rule_failure():
    from src.axiom2.promotion.authority import evaluate_promotion
    bad=[package(baseline_net_return=None),package(cost_evidence_id=''),package(experiment_count=3),package(frozen_artifact_hash='c'*64),package(holdout_consumption_count=2),package(holdout_net_return=.0015)]
    for p in bad:
        d=evaluate_promotion(p); assert d.status=='REJECTED'; assert d.reasons

def test_complete_package_promotes_deterministically_and_is_signed(tmp_path):
    from datetime import datetime,timezone,timedelta
    from src.axiom2.promotion.authority import evaluate_promotion,PromotionDecisionPayload
    from src.evidence.signing import Ed25519Signer,TrustStore,verify_envelope
    s=Ed25519Signer.generate(); now=datetime.now(timezone.utc); trust=TrustStore(); trust.add(s.trusted_key(valid_from=now-timedelta(days=1)))
    a=evaluate_promotion(package(),signer=s,created_at=now); b=evaluate_promotion(package(),signer=s,created_at=now)
    assert a.status=='REJECTED' and a.decision_hash==b.decision_hash
    assert "unverified scalar promotion package" in a.reasons and a.verified is False
    assert verify_envelope(a.envelope,PromotionDecisionPayload,trust).status=='REJECTED'


@pytest.mark.parametrize("changes",[
    {"candidate_net_return":float("nan")},
    {"holdout_net_return":float("inf")},
    {"candidate_artifact_hash":"bad"},
    {"holdout_result_hash":"bad"},
    {"experiment_count":0,"registered_experiment_count":0},
    {"holdout_candidate_id":"other"},
])
def test_adversarial_malformed_packages_fail_closed(changes):
    from src.axiom2.promotion.authority import evaluate_promotion
    assert evaluate_promotion(package(**changes)).status=="REJECTED"

def test_promoted_decision_requires_signer():
    from src.axiom2.promotion.authority import evaluate_promotion
    d=evaluate_promotion(package())
    assert d.status=="REJECTED"
