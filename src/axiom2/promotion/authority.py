"""Deterministic promotion evaluation plus evidence-resolving promotion authority."""
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import math

from src.evidence.canonical import canonical_bytes
from src.evidence.contracts import AuthorityRole, StrictContract, SignedEnvelope

@dataclass(frozen=True)
class PromotionPackage:
    candidate_id: str
    candidate_artifact_hash: str
    frozen_artifact_hash: str
    baseline_net_return: float | None
    candidate_net_return: float
    cost_evidence_id: str
    experiment_count: int
    registered_experiment_count: int
    holdout_id: str
    holdout_status: str
    holdout_result_hash: str
    holdout_candidate_id: str
    min_excess_return: float
    holdout_net_return: float
    holdout_consumption_count: int

@dataclass(frozen=True)
class PromotionRule:
    min_excess_return: float
    schema_version: int = 1

    def __post_init__(self):
        if type(self.schema_version) is not int or self.schema_version != 1:
            raise ValueError("promotion rule schema_version must be 1")
        if isinstance(self.min_excess_return, bool) or not isinstance(self.min_excess_return, (int, float)):
            raise TypeError("min_excess_return must be numeric")
        if not math.isfinite(float(self.min_excess_return)):
            raise ValueError("min_excess_return must be finite")
        if self.min_excess_return < 0:
            raise ValueError("min_excess_return must be nonnegative")

class PromotionDecisionPayload(StrictContract):
    status: str
    candidate_id: str
    decision_hash: str
    reasons: tuple[str, ...]
    verified: bool = False
    evidence_digest: str | None = None

@dataclass(frozen=True)
class PromotionDecision:
    status: str
    candidate_id: str
    reasons: tuple[str, ...]
    decision_hash: str
    envelope: SignedEnvelope | None = None
    verified: bool = False
    evidence_digest: str | None = None

def promotion_rule_digest(rule: PromotionRule) -> str:
    if type(rule) is not PromotionRule:
        raise TypeError("rule must be PromotionRule")
    rule.__post_init__()
    return hashlib.sha256(canonical_bytes({
        "min_excess_return": float(rule.min_excess_return),
        "schema_version": rule.schema_version,
    })).hexdigest()

def _valid_digest(value):
    return type(value) is str and len(value) == 64 and all(c in "0123456789abcdef" for c in value)

def _promotion_reasons(p, *, require_signer=False):
    reasons = []
    if require_signer:
        reasons.append("promotion decision requires signing authority")
    for name in ("candidate_net_return", "holdout_net_return", "min_excess_return"):
        value = getattr(p, name)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
            reasons.append(f"non-finite {name}")
    for name in ("candidate_artifact_hash", "frozen_artifact_hash", "holdout_result_hash"):
        if not _valid_digest(getattr(p, name)):
            reasons.append(f"invalid {name}")
    if p.baseline_net_return is not None and (
        isinstance(p.baseline_net_return, bool)
        or not isinstance(p.baseline_net_return, (int, float))
        or not math.isfinite(float(p.baseline_net_return))
    ):
        reasons.append("non-finite baseline_net_return")
    if p.baseline_net_return is None:
        reasons.append("missing baseline")
    if not p.cost_evidence_id:
        reasons.append("missing costs")
    if (
        type(p.experiment_count) is not int
        or type(p.registered_experiment_count) is not int
        or p.experiment_count != p.registered_experiment_count
        or p.experiment_count <= 0
    ):
        reasons.append("experiment count mismatch")
    if p.candidate_artifact_hash != p.frozen_artifact_hash:
        reasons.append("candidate mutated after freeze")
    if p.holdout_status != "CONSUMED" or type(p.holdout_consumption_count) is not int or p.holdout_consumption_count != 1:
        reasons.append("holdout not single-use consumed")
    if p.holdout_candidate_id != p.candidate_id:
        reasons.append("holdout candidate mismatch")
    if not p.holdout_result_hash:
        reasons.append("missing holdout result")
    baseline = p.baseline_net_return if p.baseline_net_return is not None else 0.0
    if (
        isinstance(p.candidate_net_return, (int, float))
        and not isinstance(p.candidate_net_return, bool)
        and isinstance(p.min_excess_return, (int, float))
        and not isinstance(p.min_excess_return, bool)
        and math.isfinite(float(p.candidate_net_return))
        and math.isfinite(float(p.min_excess_return))
        and p.candidate_net_return - baseline < p.min_excess_return
    ):
        reasons.append("development promotion rule failed")
    if (
        isinstance(p.holdout_net_return, (int, float))
        and not isinstance(p.holdout_net_return, bool)
        and isinstance(p.min_excess_return, (int, float))
        and not isinstance(p.min_excess_return, bool)
        and math.isfinite(float(p.holdout_net_return))
        and math.isfinite(float(p.min_excess_return))
        and p.holdout_net_return - baseline < p.min_excess_return
    ):
        reasons.append("holdout promotion rule failed")
    return tuple(reasons)

def _safe_package(p):
    return {
        key: (value if not isinstance(value, float) or math.isfinite(value) else str(value))
        for key, value in p.__dict__.items()
    }

def _decision_hash(p, status, reasons, *, verified, evidence_digest):
    body = {
        "status": status,
        "candidate_id": p.candidate_id,
        "reasons": tuple(reasons),
        "package": _safe_package(p),
        "verified": verified,
        "evidence_digest": evidence_digest,
    }
    return hashlib.sha256(canonical_bytes(body)).hexdigest()

def evaluate_promotion(p, *, signer=None, created_at=None):
    """Compatibility evaluator for scalar claims; it can never promote."""
    reasons = list(_promotion_reasons(p, require_signer=signer is None))
    reasons.append("unverified scalar promotion package")
    reasons = tuple(reasons)
    status = "REJECTED"
    digest = _decision_hash(p, status, reasons, verified=False, evidence_digest=None)
    envelope = None
    if signer is not None:
        payload = PromotionDecisionPayload(
            status=status,
            candidate_id=p.candidate_id,
            decision_hash=digest,
            reasons=reasons,
            verified=False,
            evidence_digest=None,
        )
        envelope = signer.sign(payload, created_at=created_at or datetime.now(timezone.utc))
    return PromotionDecision(status, p.candidate_id, reasons, digest, envelope, False, None)

def _parse_holdout_result(raw, *, candidate_id, holdout_id, cost_evidence_id):
    if type(raw) is not bytes:
        raise TypeError("holdout_result_bytes must be exact bytes")
    try:
        data = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("holdout result bytes are not canonical JSON") from exc
    expected = {"candidate_id", "holdout_id", "net_return", "cost_evidence_id", "schema_version"}
    if type(data) is not dict or set(data) != expected or canonical_bytes(data) != raw:
        raise ValueError("holdout result must use the exact canonical schema")
    if data["candidate_id"] != candidate_id or data["holdout_id"] != holdout_id:
        raise ValueError("holdout result identity mismatch")
    if data["cost_evidence_id"] != cost_evidence_id:
        raise ValueError("holdout result cost evidence mismatch")
    if type(data["schema_version"]) is not int or data["schema_version"] != 1:
        raise ValueError("holdout result schema_version must be 1")
    value = data["net_return"]
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        raise ValueError("holdout result net_return must be finite")
    return float(value)

def evaluate_verified_promotion(
    *, proposal, campaign, candidate_id, holdout_id, holdout_result_bytes,
    registry, promotion_rule, signer, actor_id, created_at=None,
):
    """Resolve evidence and authority before signing a verified promotion decision."""
    from src.axiom2.research.ranker import _verify_campaign_evidence

    rule_digest = promotion_rule_digest(promotion_rule)
    if not any(
        dep.source == "axiom2-promotion-rule"
        and dep.dataset_id == rule_digest
        and dep.schema_version == "1"
        for dep in proposal.data_dependencies
    ):
        raise ValueError("promotion rule was not preregistered")
    _verify_campaign_evidence(proposal, campaign, registry=registry)
    state = registry.snapshot()
    candidate = state.candidates.get(candidate_id)
    if candidate is None or candidate.experiment_id != proposal.experiment_id:
        raise ValueError("frozen candidate does not resolve to the proposal")
    if candidate.artifact_hash != campaign.model_artifact_hash:
        raise ValueError("frozen candidate artifact differs from verified campaign")
    holdout = state.holdouts.get(holdout_id)
    if holdout is None or holdout.status != "CONSUMED":
        raise ValueError("holdout must be durably consumed")
    if holdout.candidate_id != candidate_id or holdout.result_hash is None:
        raise ValueError("consumed holdout does not belong to frozen candidate")
    if holdout.attempt_count != state.attempt_count:
        raise ValueError("research attempt history changed after holdout opening")
    result_hash = hashlib.sha256(holdout_result_bytes).hexdigest()
    if result_hash != holdout.result_hash:
        raise ValueError("holdout result hash does not match consumed evidence")
    holdout_net_return = _parse_holdout_result(
        holdout_result_bytes,
        candidate_id=candidate_id,
        holdout_id=holdout_id,
        cost_evidence_id=proposal.cost_model_id,
    )
    trials = tuple(
        trial for trial in state.trials.values()
        if trial.experiment_id == proposal.experiment_id
        and trial.campaign_digest == campaign.campaign_manifest_digest
    )
    package = PromotionPackage(
        candidate_id=candidate_id,
        candidate_artifact_hash=campaign.model_artifact_hash,
        frozen_artifact_hash=candidate.artifact_hash,
        baseline_net_return=campaign.baseline_net_return,
        candidate_net_return=campaign.portfolio_net_return,
        cost_evidence_id=proposal.cost_model_id,
        experiment_count=campaign.experiment_count,
        registered_experiment_count=len(trials),
        holdout_id=holdout_id,
        holdout_status=holdout.status,
        holdout_result_hash=holdout.result_hash,
        holdout_candidate_id=holdout.candidate_id,
        min_excess_return=float(promotion_rule.min_excess_return),
        holdout_net_return=holdout_net_return,
        holdout_consumption_count=1,
    )
    when = created_at or registry.store._trusted_clock()
    registry.store.trust_store.require_trusted_at_receipt(signer.key_id, when)
    registry.store.authorities.authorize_identity(
        actor_id=actor_id,
        role=AuthorityRole.PROMOTION_SERVICE,
        key_id=signer.key_id,
    )
    evidence_digest = hashlib.sha256(canonical_bytes({
        "candidate_id": candidate_id,
        "candidate_freeze_sequence": candidate.freeze_sequence,
        "campaign_envelope_digest": campaign.envelope.payload_digest,
        "campaign_manifest_digest": campaign.campaign_manifest_digest,
        "selected_trial_id": campaign.selected_trial_id,
        "selected_trial_result_digest": campaign.selected_trial_result_digest,
        "holdout_consumption_digest": holdout.consumption_digest,
        "holdout_result_hash": holdout.result_hash,
        "promotion_rule_digest": rule_digest,
        "research_head_digest": state.head_digest,
        "research_event_count": state.event_count,
        "schema_version": 1,
    })).hexdigest()
    reasons = _promotion_reasons(package, require_signer=False)
    status = "REJECTED" if reasons else "PROMOTED"
    decision_hash = _decision_hash(
        package,
        status,
        reasons,
        verified=True,
        evidence_digest=evidence_digest,
    )
    payload = PromotionDecisionPayload(
        status=status,
        candidate_id=candidate_id,
        decision_hash=decision_hash,
        reasons=reasons,
        verified=True,
        evidence_digest=evidence_digest,
    )
    envelope = signer.sign(payload, created_at=when)
    return PromotionDecision(
        status,
        candidate_id,
        reasons,
        decision_hash,
        envelope,
        True,
        evidence_digest,
    )
