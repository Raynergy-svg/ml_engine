"""Shared dataclasses for the meta-cybernetic change pipeline.

A `ChangePackage` is the durable artifact that walks the 9 stages of
`MetaManager`. Every stage either enriches it, blocks it, or rolls it back.

Persistence layout:
    .claude/meta/changes/<change_id>.json    — current state of one package
    .claude/meta/changes.jsonl               — append-only ledger (memory layer)
    .claude/meta/scorecards.jsonl            — append-only eval results
"""

from __future__ import annotations

import hashlib
import json
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional


def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _new_change_id() -> str:
    return uuid.uuid4().hex[:12]


class ChangeKind(str, Enum):
    CONFIG = "config"
    CODE = "code"
    MODEL = "model"
    DISABLE = "disable"


class ChangeStage(str, Enum):
    INTAKE = "intake"
    DIAGNOSING = "diagnosing"
    PROPOSING = "proposing"
    EVALUATING = "evaluating"
    POLICY_CHECK = "policy_check"
    AWAITING_APPROVAL = "awaiting_approval"
    APPROVED = "approved"
    REJECTED = "rejected"
    DEPLOYED_SHADOW = "deployed_shadow"
    DEPLOYED_CANARY = "deployed_canary"
    DEPLOYED_LIVE = "deployed_live"
    POST_DEPLOY_REVIEW = "post_deploy_review"
    CLOSED = "closed"
    ABORTED = "aborted"


class DeployStage(str, Enum):
    SHADOW = "shadow"
    CANARY = "canary"
    LIVE = "live"


class Severity(str, Enum):
    LOW = "low"
    MED = "med"
    HIGH = "high"


@dataclass
class Diagnosis:
    """Output of the Incident Analyst specialist."""

    root_cause_hypothesis: str = ""
    affected_modules: List[str] = field(default_factory=list)
    severity: Severity = Severity.LOW
    proposed_intervention_kind: ChangeKind = ChangeKind.CONFIG
    raw_output: str = ""


@dataclass
class Proposal:
    """Output of stage 2 — the candidate change itself."""

    diff: str = ""
    changed_files: List[str] = field(default_factory=list)
    config_delta: Dict[str, Any] = field(default_factory=dict)
    branch: Optional[str] = None
    worktree_path: Optional[str] = None


@dataclass
class SubScore:
    """One axis of the four-layer scorecard."""

    name: str
    passed: bool
    score: float = 0.0
    details: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None


@dataclass
class Scorecard:
    """Output of stage 3."""

    software_correctness: Optional[SubScore] = None
    trading_quality: Optional[SubScore] = None
    governance_quality: Optional[SubScore] = None
    regime_robustness: Optional[SubScore] = None
    timestamp: str = field(default_factory=_utcnow_iso)

    def all_passed(self) -> bool:
        for sub in (
            self.software_correctness,
            self.trading_quality,
            self.governance_quality,
            self.regime_robustness,
        ):
            if sub is None or not sub.passed:
                return False
        return True

    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "software_correctness": asdict(self.software_correctness) if self.software_correctness else None,
            "trading_quality": asdict(self.trading_quality) if self.trading_quality else None,
            "governance_quality": asdict(self.governance_quality) if self.governance_quality else None,
            "regime_robustness": asdict(self.regime_robustness) if self.regime_robustness else None,
            "all_passed": self.all_passed(),
        }


@dataclass
class ClauseResult:
    """Result of evaluating one constitution clause."""

    clause_id: str
    name: str
    passed: bool
    detail: str = ""


@dataclass
class ConstitutionAttestation:
    """Output of stage 4 (machine half)."""

    passed: bool = True
    clauses: List[ClauseResult] = field(default_factory=list)
    auditor_specialist_agrees: Optional[bool] = None
    auditor_raw_output: str = ""
    timestamp: str = field(default_factory=_utcnow_iso)

    def failed_clauses(self) -> List[ClauseResult]:
        return [c for c in self.clauses if not c.passed]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "passed": self.passed,
            "clauses": [asdict(c) for c in self.clauses],
            "auditor_specialist_agrees": self.auditor_specialist_agrees,
            "timestamp": self.timestamp,
        }


@dataclass
class DeploymentRecord:
    """Per-stage deploy record (shadow/canary/live)."""

    stage: DeployStage
    deployed_at: str = field(default_factory=_utcnow_iso)
    # Anchor cycle so drain() can enforce shadow_cycles/canary_trades soak
    # windows. 0 is a sentinel meaning "unknown" — drain treats unknown as
    # "soak window already elapsed" to preserve backward compat with packages
    # serialized before this field existed.
    deployed_at_cycle: int = 0
    closed_trade_count_at_deploy: int = 0
    regime_at_deploy: Optional[str] = None
    rolled_back_at: Optional[str] = None
    rollback_reason: Optional[str] = None


@dataclass
class PostDeployReview:
    """Stage 9 output — comparison of predicted vs actual."""

    stage: DeployStage
    reviewed_at: str = field(default_factory=_utcnow_iso)
    predicted: Dict[str, Any] = field(default_factory=dict)
    actual: Dict[str, Any] = field(default_factory=dict)
    delta: Dict[str, Any] = field(default_factory=dict)
    passed: bool = True
    notes: str = ""


@dataclass
class ChangePackage:
    """The single durable artifact that walks the pipeline."""

    change_id: str = field(default_factory=_new_change_id)
    created_at: str = field(default_factory=_utcnow_iso)
    updated_at: str = field(default_factory=_utcnow_iso)
    stage: ChangeStage = ChangeStage.INTAKE
    kind: ChangeKind = ChangeKind.CONFIG
    incident: Dict[str, Any] = field(default_factory=dict)
    diagnosis: Optional[Diagnosis] = None
    proposal: Optional[Proposal] = None
    scorecard: Optional[Scorecard] = None
    constitution: Optional[ConstitutionAttestation] = None
    deploy_target: DeployStage = DeployStage.SHADOW
    deployments: List[DeploymentRecord] = field(default_factory=list)
    post_deploy_reviews: List[PostDeployReview] = field(default_factory=list)
    rollback_plan: str = ""
    dossier_summary: str = ""
    rejection_reason: Optional[str] = None
    history: List[Dict[str, Any]] = field(default_factory=list)

    def touch(self, note: str = "") -> None:
        self.updated_at = _utcnow_iso()
        if note:
            self.history.append({"at": self.updated_at, "stage": self.stage.value, "note": note})

    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {
            "change_id": self.change_id,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "stage": self.stage.value,
            "kind": self.kind.value,
            "incident": self.incident,
            "diagnosis": asdict(self.diagnosis) if self.diagnosis else None,
            "proposal": asdict(self.proposal) if self.proposal else None,
            "scorecard": self.scorecard.to_dict() if self.scorecard else None,
            "constitution": self.constitution.to_dict() if self.constitution else None,
            "deploy_target": self.deploy_target.value,
            "deployments": [
                {
                    "stage": dep.stage.value,
                    "deployed_at": dep.deployed_at,
                    "deployed_at_cycle": dep.deployed_at_cycle,
                    "closed_trade_count_at_deploy": dep.closed_trade_count_at_deploy,
                    "regime_at_deploy": dep.regime_at_deploy,
                    "rolled_back_at": dep.rolled_back_at,
                    "rollback_reason": dep.rollback_reason,
                }
                for dep in self.deployments
            ],
            "post_deploy_reviews": [
                {
                    "stage": pdr.stage.value,
                    "reviewed_at": pdr.reviewed_at,
                    "predicted": pdr.predicted,
                    "actual": pdr.actual,
                    "delta": pdr.delta,
                    "passed": pdr.passed,
                    "notes": pdr.notes,
                }
                for pdr in self.post_deploy_reviews
            ],
            "rollback_plan": self.rollback_plan,
            "dossier_summary": self.dossier_summary,
            "rejection_reason": self.rejection_reason,
            "history": self.history,
        }
        if self.diagnosis is not None:
            d["diagnosis"]["severity"] = self.diagnosis.severity.value
            d["diagnosis"]["proposed_intervention_kind"] = self.diagnosis.proposed_intervention_kind.value
        return d

    def dedup_hash(self) -> str:
        """Stable identity hash for duplicate-proposal suppression.

        Collides when (incident.kind, sorted(config_delta items)) is identical.
        Excludes timestamps, summaries, and incident metadata so two genuinely
        identical proposals from different incidents collapse to one package.

        Nested dict values are serialized with json.dumps(sort_keys=True) so
        that semantically equal deltas constructed in different key insertion
        order hash equal — preserving the G8 anti-dup intent. ``default=str``
        keeps non-JSON-native values (Decimal, Enum, datetime) hashable.
        """
        kind = str((self.incident or {}).get("kind", ""))
        delta = (self.proposal.config_delta if self.proposal else {}) or {}
        canonical = json.dumps(
            sorted(
                (k, json.dumps(v, sort_keys=True, default=str))
                for k, v in delta.items()
            ),
            sort_keys=True,
            default=str,
        )
        material = f"{kind}::{canonical}".encode("utf-8")
        # SHA-256 used for non-cryptographic dedup keying (not a signature);
        # truncated to 16 hex chars (64 bits) — ample for proposal-set dedup.
        return hashlib.sha256(material).hexdigest()[:16]

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> "ChangePackage":
        diag_d = data.get("diagnosis")
        diag = None
        if diag_d:
            diag = Diagnosis(
                root_cause_hypothesis=diag_d.get("root_cause_hypothesis", ""),
                affected_modules=list(diag_d.get("affected_modules", [])),
                severity=Severity(diag_d.get("severity", "low")),
                proposed_intervention_kind=ChangeKind(diag_d.get("proposed_intervention_kind", "config")),
                raw_output=diag_d.get("raw_output", ""),
            )
        prop_d = data.get("proposal")
        prop = Proposal(**prop_d) if prop_d else None
        score_d = data.get("scorecard")
        score = None
        if score_d:
            def _sub(k: str) -> Optional[SubScore]:
                v = score_d.get(k)
                return SubScore(**v) if v else None
            score = Scorecard(
                software_correctness=_sub("software_correctness"),
                trading_quality=_sub("trading_quality"),
                governance_quality=_sub("governance_quality"),
                regime_robustness=_sub("regime_robustness"),
                timestamp=score_d.get("timestamp", _utcnow_iso()),
            )
        const_d = data.get("constitution")
        const = None
        if const_d:
            const = ConstitutionAttestation(
                passed=const_d.get("passed", True),
                clauses=[ClauseResult(**c) for c in const_d.get("clauses", [])],
                auditor_specialist_agrees=const_d.get("auditor_specialist_agrees"),
                auditor_raw_output=const_d.get("auditor_raw_output", ""),
                timestamp=const_d.get("timestamp", _utcnow_iso()),
            )
        deployments = [
            DeploymentRecord(
                stage=DeployStage(d["stage"]),
                deployed_at=d.get("deployed_at", _utcnow_iso()),
                deployed_at_cycle=int(d.get("deployed_at_cycle", 0) or 0),
                closed_trade_count_at_deploy=int(d.get("closed_trade_count_at_deploy", 0) or 0),
                regime_at_deploy=d.get("regime_at_deploy"),
                rolled_back_at=d.get("rolled_back_at"),
                rollback_reason=d.get("rollback_reason"),
            )
            for d in data.get("deployments", [])
        ]
        reviews = [
            PostDeployReview(
                stage=DeployStage(r["stage"]),
                reviewed_at=r.get("reviewed_at", _utcnow_iso()),
                predicted=r.get("predicted", {}),
                actual=r.get("actual", {}),
                delta=r.get("delta", {}),
                passed=r.get("passed", True),
                notes=r.get("notes", ""),
            )
            for r in data.get("post_deploy_reviews", [])
        ]
        return ChangePackage(
            change_id=data["change_id"],
            created_at=data.get("created_at", _utcnow_iso()),
            updated_at=data.get("updated_at", _utcnow_iso()),
            stage=ChangeStage(data.get("stage", "intake")),
            kind=ChangeKind(data.get("kind", "config")),
            incident=data.get("incident", {}),
            diagnosis=diag,
            proposal=prop,
            scorecard=score,
            constitution=const,
            deploy_target=DeployStage(data.get("deploy_target", "shadow")),
            deployments=deployments,
            post_deploy_reviews=reviews,
            rollback_plan=data.get("rollback_plan", ""),
            dossier_summary=data.get("dossier_summary", ""),
            rejection_reason=data.get("rejection_reason"),
            history=list(data.get("history", [])),
        )
