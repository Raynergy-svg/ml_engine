"""MetaManager — the single coordinator for the meta-cybernetic change pipeline.

Owns one `ChangePackage` per change, walks it through the 9 stages defined in
`/root/.claude/plans/system-reminder-you-re-running-in-streamed-waterfall.md`,
and refuses to merge anything that hasn't survived all of them.

Architectural rule: this module *coordinates*. It does not re-implement
diagnosis, simulation, policy, deploy, or critique — those live in
`change_eval`, `constitution`, `approval_queue`, `staged_deployer`,
`post_deploy_critic`. Every external dependency is constructor-injected so
the manager is unit-testable without spawning Claude or hitting OANDA.

Public surface:
    intake(incident)           — opens a new package and starts the pipeline
    drive_pending(...)         — moves all in-flight packages forward one step
    drain()                    — alias used by the orchestrator dispatch table
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from src.scanner.automation.constitution import Constitution
from src.scanner.automation.approval_queue import ApprovalQueue
from src.scanner.automation.change_eval import ChangeEvalHarness
from src.scanner.automation.meta_types import (
    ChangeKind,
    ChangePackage,
    ChangeStage,
    DeployStage,
    Diagnosis,
    Proposal,
    Severity,
)
from src.scanner.automation.post_deploy_critic import PostDeployCritic
from src.scanner.automation.safe_json import safe_json_read, safe_json_write
from src.scanner.automation.staged_deployer import StagedDeployer

logger = logging.getLogger(__name__)


_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
_CHANGES_DIR = _PROJECT_ROOT / ".claude" / "meta" / "changes"
_LEDGER_PATH = _PROJECT_ROOT / ".claude" / "meta" / "changes.jsonl"


# Specialist invocation surface — callable taking (mode, prompt) and returning raw output.
SpecialistInvoker = Callable[[str, str], str]


def _default_specialist_invoker(mode: str, prompt: str) -> str:
    """Default: call the existing reflection harness; tolerate failure silently.

    Returns the stdout (or empty string on failure). The MetaManager never
    blocks on specialist output — they enrich, they don't gate (the
    Constitution gates).
    """
    try:
        from src.scanner.automation.claude_subprocess import invoke_claude_reflection
        result = invoke_claude_reflection(
            prompt=prompt,
            trade_id=mode,
            mode=mode,
            timeout_seconds=120,
        )
        return getattr(result, "result_block", "") or ""
    except Exception as e:
        logger.warning("meta_manager.specialist_invoke_failed mode=%s err=%s", mode, e)
        return ""


class MetaManager:
    """Coordinator for the 9-stage change pipeline."""

    def __init__(
        self,
        *,
        config: Any = None,
        eval_harness: Optional[ChangeEvalHarness] = None,
        constitution: Optional[Constitution] = None,
        approval_queue: Optional[ApprovalQueue] = None,
        staged_deployer: Optional[StagedDeployer] = None,
        post_deploy_critic: Optional[PostDeployCritic] = None,
        specialist_invoker: Optional[SpecialistInvoker] = None,
        changes_dir: Optional[Path] = None,
        ledger_path: Optional[Path] = None,
        max_concurrent: int = 1,
        episodic_memory: Optional[Any] = None,
    ):
        self._config = config
        self._eval = eval_harness
        self._const = constitution or Constitution()
        self._queue = approval_queue or ApprovalQueue()
        self._deploy = staged_deployer
        self._critic = post_deploy_critic or PostDeployCritic()
        self._invoke = specialist_invoker or _default_specialist_invoker
        self._changes_dir = Path(changes_dir or _CHANGES_DIR)
        self._ledger = Path(ledger_path or _LEDGER_PATH)
        self._max_concurrent = max(1, int(max_concurrent))
        self._episodic_memory = episodic_memory
        self._changes_dir.mkdir(parents=True, exist_ok=True)
        logger.info(
            "meta_manager.initialized changes_dir=%s ledger=%s",
            self._changes_dir, self._ledger,
        )

    # ---------- Stage 0-1: intake ----------

    def intake(self, incident: Dict[str, Any]) -> ChangePackage:
        """Open a new package, run Incident Analyst, persist."""
        if self._concurrent_count() >= self._max_concurrent:
            logger.info(
                "meta_manager.intake_throttled active=%s max=%s — incident dropped",
                self._concurrent_count(), self._max_concurrent,
            )
            pkg = ChangePackage(incident=dict(incident))
            pkg.stage = ChangeStage.ABORTED
            pkg.touch("throttled_by_max_concurrent")
            self._persist(pkg)
            self._append_ledger(pkg, "throttled")
            return pkg

        pkg = ChangePackage(incident=dict(incident))
        pkg.touch("intake")
        pkg.stage = ChangeStage.DIAGNOSING
        self._persist(pkg)
        logger.info(
            "meta_manager.intake change_id=%s kind=%s",
            pkg.change_id, incident.get("kind", "?"),
        )

        diag_raw = self._invoke("incident_analyst", _build_analyst_prompt(pkg))
        pkg.diagnosis = _parse_diagnosis(diag_raw, incident)
        pkg.kind = pkg.diagnosis.proposed_intervention_kind
        pkg.touch("diagnosis_complete")
        self._persist(pkg)
        return pkg

    # ---------- Stage 2: proposal ----------

    def propose(self, pkg: ChangePackage, proposer: Optional[Callable[[ChangePackage], Proposal]] = None) -> ChangePackage:
        pkg.stage = ChangeStage.PROPOSING
        try:
            if proposer is not None:
                pkg.proposal = proposer(pkg)
            else:
                pkg.proposal = self._default_proposer(pkg)
        except Exception as e:
            logger.warning("meta_manager.propose_failed change_id=%s err=%s", pkg.change_id, e)
            pkg.stage = ChangeStage.ABORTED
            pkg.touch(f"propose_error: {e}")
            self._persist(pkg)
            self._append_ledger(pkg, "propose_error")
            return pkg
        pkg.touch("proposal_built")
        self._persist(pkg)
        return pkg

    # ---------- Stage 3: eval ----------

    def evaluate(self, pkg: ChangePackage) -> ChangePackage:
        if self._eval is None:
            logger.info("meta_manager.evaluate_skipped no_harness change_id=%s", pkg.change_id)
            pkg.stage = ChangeStage.POLICY_CHECK
            self._persist(pkg)
            return pkg
        pkg.stage = ChangeStage.EVALUATING
        try:
            pkg.scorecard = self._eval.score(pkg)
        except Exception as e:
            logger.warning("meta_manager.evaluate_failed change_id=%s err=%s", pkg.change_id, e)
            pkg.stage = ChangeStage.ABORTED
            pkg.touch(f"eval_error: {e}")
            self._append_ledger(pkg, "eval_error")
            self._persist(pkg)
            return pkg
        pkg.touch(f"scorecard all_passed={pkg.scorecard.all_passed()}")
        self._persist(pkg)
        return pkg

    # ---------- Stage 4: constitution + policy auditor ----------

    def check_constitution(self, pkg: ChangePackage) -> ChangePackage:
        pkg.stage = ChangeStage.POLICY_CHECK
        attestation = self._const.check(pkg)
        auditor_raw = self._invoke("policy_auditor", _build_auditor_prompt(pkg, attestation))
        agrees = _parse_auditor_agreement(auditor_raw)
        attestation.auditor_specialist_agrees = agrees
        attestation.auditor_raw_output = auditor_raw[:4000]
        pkg.constitution = attestation

        if not attestation.passed:
            pkg.stage = ChangeStage.REJECTED
            pkg.rejection_reason = (
                "constitution_violation: "
                + ", ".join(c.clause_id for c in attestation.failed_clauses())
            )
            pkg.touch(pkg.rejection_reason)
            self._persist(pkg)
            self._append_ledger(pkg, "constitution_blocked")
            return pkg

        if agrees is False:
            pkg.stage = ChangeStage.REJECTED
            pkg.rejection_reason = "policy_auditor_disagreement_hard_stop"
            pkg.touch(pkg.rejection_reason)
            self._persist(pkg)
            self._append_ledger(pkg, "auditor_block")
            return pkg

        pkg.touch("constitution_passed")
        self._persist(pkg)
        return pkg

    # ---------- Stage 5: human approval ----------

    def request_approval(self, pkg: ChangePackage) -> ChangePackage:
        if not pkg.dossier_summary:
            arbiter_raw = self._invoke("deployment_arbiter", _build_arbiter_prompt(pkg))
            pkg.dossier_summary = _extract_dossier_summary(arbiter_raw, pkg)
        if not pkg.rollback_plan:
            pkg.rollback_plan = _default_rollback_plan(pkg)
        self._queue.enqueue(pkg)
        self._persist(pkg)
        return pkg

    # ---------- Stage 6-9: drain approved/rejected, deploy, review ----------

    def drain(self, current_cycle: int = 0) -> Dict[str, int]:
        """Process every approved/rejected package and every in-flight deploy.

        Called once per orchestrator cycle by the dispatch table. Returns
        a small dict of counters for observability.
        """
        counters = {
            "approved_advanced": 0,
            "rejected_finalized": 0,
            "post_deploy_reviews": 0,
            "promotions": 0,
            "rollbacks": 0,
        }

        just_advanced: set = set()
        for pkg in self._queue.drain_approved():
            counters["approved_advanced"] += 1
            if self._deploy is None:
                logger.warning(
                    "meta_manager.drain_no_deployer change_id=%s — skipped",
                    pkg.change_id,
                )
                self._persist(pkg)
                continue
            self._deploy.advance(pkg, current_cycle=current_cycle)
            just_advanced.add(pkg.change_id)
            self._persist(pkg)
            self._append_ledger(pkg, f"deployed_{pkg.deploy_target.value}")

        for pkg in self._queue.drain_rejected():
            counters["rejected_finalized"] += 1
            self._persist(pkg)
            self._append_ledger(pkg, "rejected_by_human")

        for pkg in self._iter_in_flight():
            if self._deploy is None:
                continue
            if pkg.change_id in just_advanced:
                continue
            review = self._critic.review(pkg, pkg.deploy_target)
            counters["post_deploy_reviews"] += 1
            self._persist(pkg)
            if not review.passed:
                self._deploy.rollback(pkg, reason=f"post_deploy_miss_{pkg.deploy_target.value}")
                counters["rollbacks"] += 1
                self._persist(pkg)
                self._append_ledger(pkg, "post_deploy_rollback")
                continue
            if pkg.deploy_target == DeployStage.LIVE:
                pkg.stage = ChangeStage.CLOSED
                pkg.touch("live_review_passed_closing")
                self._persist(pkg)
                self._append_ledger(pkg, "live_closed")
                continue
            self._deploy.promote_next(pkg, current_cycle=current_cycle)
            counters["promotions"] += 1
            self._persist(pkg)
            self._append_ledger(pkg, f"promoted_to_{pkg.deploy_target.value}")
        return counters

    # ---------- helpers ----------

    def list_packages(self) -> List[ChangePackage]:
        out: List[ChangePackage] = []
        for path in sorted(self._changes_dir.glob("*.json")):
            data = safe_json_read(path, default={})
            if data:
                try:
                    out.append(ChangePackage.from_dict(data))
                except Exception as e:
                    logger.warning("meta_manager.bad_package path=%s err=%s", path, e)
        return out

    def get(self, change_id: str) -> Optional[ChangePackage]:
        path = self._changes_dir / f"{change_id}.json"
        if not path.exists():
            return None
        data = safe_json_read(path, default={})
        if not data:
            return None
        return ChangePackage.from_dict(data)

    def _persist(self, pkg: ChangePackage) -> None:
        path = self._changes_dir / f"{pkg.change_id}.json"
        safe_json_write(path, pkg.to_dict())

    def _append_ledger(self, pkg: ChangePackage, event: str) -> None:
        try:
            self._ledger.parent.mkdir(parents=True, exist_ok=True)
            record = {
                "change_id": pkg.change_id,
                "stage": pkg.stage.value,
                "event": event,
                "kind": pkg.kind.value,
                "deploy_target": pkg.deploy_target.value,
                "updated_at": pkg.updated_at,
            }
            with open(self._ledger, "a", encoding="utf-8") as f:
                f.write(json.dumps(record, default=str) + "\n")
        except Exception as e:
            logger.warning("meta_manager.ledger_append_failed err=%s", e)

    def _concurrent_count(self) -> int:
        terminal = {ChangeStage.CLOSED, ChangeStage.REJECTED, ChangeStage.ABORTED}
        return sum(1 for pkg in self.list_packages() if pkg.stage not in terminal)

    def _iter_in_flight(self) -> List[ChangePackage]:
        deploy_stages = {
            ChangeStage.DEPLOYED_SHADOW,
            ChangeStage.DEPLOYED_CANARY,
            ChangeStage.DEPLOYED_LIVE,
        }
        return [pkg for pkg in self.list_packages() if pkg.stage in deploy_stages]

    def _default_proposer(self, pkg: ChangePackage) -> Proposal:
        """Build a Proposal from the diagnosis when no explicit proposer is given.

        For `kind=config`, the diagnosis must include a `proposed_config_delta`
        field in the incident payload (set by the upstream caller) — otherwise
        we return an empty proposal and let evaluation skip.
        """
        delta = (pkg.incident.get("proposed_config_delta") or {}) if pkg.incident else {}
        return Proposal(
            diff=json.dumps(delta, indent=2, sort_keys=True),
            changed_files=list(pkg.incident.get("changed_files", [])) if pkg.incident else [],
            config_delta=delta,
        )


# ---------- public routing helper used by upstream triggers ----------

def is_enabled() -> bool:
    """Cheap check for upstream call sites.

    Reads the env var (set by main.py / orchestrator on startup) and falls
    back to importing ScannerConfig only if the env var is absent. The env
    var path lets cycle_autonomy and prd_agent_chain skip the import cost
    on the hot path when the meta manager is off.
    """
    import os
    val = os.environ.get("BUDDY_META_MANAGER_ENABLED", "").lower()
    if val in ("1", "true", "yes", "on"):
        return True
    if val in ("0", "false", "no", "off"):
        return False
    # Fall back to dataclass default (False) — never import the heavy config
    # singleton here; if you want to override, set the env var.
    return False


def route_incident(incident: Dict[str, Any]) -> bool:
    """Route an incident through the meta-pipeline if enabled.

    Returns True when the meta manager swallowed the signal (the caller
    should NOT also spawn Ralph / reflection / direct config writes).
    Returns False when the legacy path should run.
    """
    if not is_enabled():
        return False
    try:
        mgr = MetaManager()
        mgr.intake(incident)
        return True
    except Exception as e:
        logger.warning("meta_manager.route_incident_failed err=%s — falling back to legacy", e)
        return False


# ---------- prompt builders (kept short — full text lives in the .md personality files) ----------

def _build_analyst_prompt(pkg: ChangePackage) -> str:
    return (
        "ROLE: incident_analyst (see .claude/agents/specialized-incident-analyst.md)\n"
        f"CHANGE_ID: {pkg.change_id}\n"
        f"INCIDENT: {json.dumps(pkg.incident, default=str)[:4000]}\n"
        "Return a YAML result block with: root_cause_hypothesis, affected_modules, "
        "severity (low/med/high), proposed_intervention_kind (config/code/model/disable)."
    )


def _build_auditor_prompt(pkg: ChangePackage, attestation: Any) -> str:
    return (
        "ROLE: policy_auditor (see .claude/agents/specialized-policy-auditor.md)\n"
        f"CHANGE_ID: {pkg.change_id}\n"
        f"DIFF: {pkg.proposal.diff[:4000] if pkg.proposal else ''}\n"
        f"MACHINE_ATTESTATION: passed={attestation.passed} clauses="
        f"{[c.clause_id for c in attestation.failed_clauses()]}\n"
        "Re-read the diff and the seed constitution. Reply with a YAML block "
        "containing exactly one key: agree (true/false), plus a one-line reason."
    )


def _build_arbiter_prompt(pkg: ChangePackage) -> str:
    score_summary = pkg.scorecard.to_dict() if pkg.scorecard else {}
    return (
        "ROLE: deployment_arbiter (see .claude/agents/specialized-deployment-arbiter.md)\n"
        f"CHANGE_ID: {pkg.change_id}\n"
        f"DIAGNOSIS: {pkg.diagnosis.root_cause_hypothesis if pkg.diagnosis else ''}\n"
        f"SCORECARD: {json.dumps(score_summary, default=str)[:3000]}\n"
        "Write one paragraph (<= 120 words) summarizing the change for a human "
        "reviewer. Be concrete: what is the predicted benefit, the worst-case "
        "regression, and the recommended deploy_target (shadow/canary/live)."
    )


# ---------- structured-output parsers (best-effort, never crash) ----------

def _parse_diagnosis(raw: str, incident: Dict[str, Any]) -> Diagnosis:
    """Best-effort parse of the analyst's YAML/JSON result block."""
    diag = Diagnosis()
    if raw:
        try:
            data = json.loads(raw) if raw.strip().startswith("{") else _yaml_to_dict(raw)
            diag.root_cause_hypothesis = str(data.get("root_cause_hypothesis", ""))[:400]
            diag.affected_modules = list(data.get("affected_modules", []))[:20]
            sev = str(data.get("severity", "low")).lower()
            diag.severity = Severity(sev) if sev in ("low", "med", "high") else Severity.LOW
            kind = str(data.get("proposed_intervention_kind", "config")).lower()
            diag.proposed_intervention_kind = (
                ChangeKind(kind) if kind in ("config", "code", "model", "disable") else ChangeKind.CONFIG
            )
            diag.raw_output = raw[:2000]
        except Exception as e:
            diag.raw_output = f"parse_error: {e}\n{raw[:1000]}"
    if not diag.affected_modules and incident.get("affected_modules"):
        diag.affected_modules = list(incident["affected_modules"])[:20]
    if incident.get("kind") in {"config", "code", "model", "disable"}:
        diag.proposed_intervention_kind = ChangeKind(incident["kind"])
    return diag


def _parse_auditor_agreement(raw: str) -> Optional[bool]:
    if not raw:
        return None
    lowered = raw.lower()
    if "agree: true" in lowered or '"agree": true' in lowered:
        return True
    if "agree: false" in lowered or '"agree": false' in lowered:
        return False
    return None


def _extract_dossier_summary(raw: str, pkg: ChangePackage) -> str:
    if raw:
        return raw.strip()[:1200]
    return f"Change {pkg.change_id}: kind={pkg.kind.value}, target={pkg.deploy_target.value}"


def _default_rollback_plan(pkg: ChangePackage) -> str:
    if pkg.kind == ChangeKind.CONFIG:
        return (
            f"Revert via ConfigAdjuster.revert_by_id(config, source_substring='{pkg.change_id}'). "
            "For shadow stage, the parallel namespace is dropped automatically by StagedDeployer.rollback."
        )
    return (
        "Roll back the worktree branch and re-run the orchestrator dispatch table. "
        "If live trades are open, the drawdown guardian retains its absolute caps."
    )


def _yaml_to_dict(text: str) -> Dict[str, Any]:
    """Tiny inline YAML parser — only handles the flat 'key: value' subset we expect.

    The reflection harness asks specialists to return YAML inside a result block;
    full PyYAML is overkill (and not always installed in CI). We tolerate JSON too.
    """
    out: Dict[str, Any] = {}
    for line in text.splitlines():
        if ":" not in line:
            continue
        k, _, v = line.partition(":")
        key = k.strip()
        val = v.strip()
        if not key or not val or key.startswith("#"):
            continue
        if val.startswith("[") and val.endswith("]"):
            inner = val[1:-1].strip()
            out[key] = [s.strip().strip('"\'') for s in inner.split(",") if s.strip()]
        elif val.lower() in ("true", "false"):
            out[key] = val.lower() == "true"
        else:
            out[key] = val.strip('"\'')
    return out
