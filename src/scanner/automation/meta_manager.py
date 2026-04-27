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
import re
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


_FENCED_BLOCK_RE = re.compile(
    r"```(?:yaml|yml|json)?\s*\n(.*?)\n```",
    re.DOTALL | re.IGNORECASE,
)


def _extract_specialist_output(stdout: str) -> str:
    """Pull the structured specialist response out of raw Claude CLI stdout.

    Meta specialists return fenced ```yaml/```json blocks per their persona
    instructions. The reflection harness's `_parse_result_block` only
    recognizes <reflection-result> XML tags — wrong format for this layer.
    We extract the FIRST fenced block; if none found, return stdout verbatim
    so the downstream parsers (_parse_diagnosis, _parse_auditor_agreement)
    can still do best-effort YAML parsing.
    """
    if not stdout:
        return ""
    m = _FENCED_BLOCK_RE.search(stdout)
    if m:
        return m.group(1).strip()
    return stdout.strip()


def _default_specialist_invoker(mode: str, prompt: str) -> str:
    """Default: call the existing reflection harness; tolerate failure silently.

    Returns the structured specialist output (fenced block contents) or empty
    string on failure. The MetaManager never blocks on specialist output —
    they enrich, they don't gate (the Constitution gates).

    Performance note (2026-04-26): timeout 360s. The headless `claude --print`
    subprocess loads 4 MCP servers on every invocation; cold-start alone is
    ~70s. With actual specialist thinking, ~150s typical.

    Format note (2026-04-26): we read `result.stdout` directly and extract
    the fenced ```yaml/```json block. The reflection harness's built-in
    `result.result_block` only recognizes <reflection-result> XML tags
    (a different format from what meta specialists emit).
    """
    try:
        from src.scanner.automation.claude_subprocess import invoke_claude_reflection
        result = invoke_claude_reflection(
            prompt=prompt,
            trade_id=mode,
            mode=mode,
            timeout_seconds=360,
        )
        stdout = getattr(result, "stdout", "") or ""
        return _extract_specialist_output(stdout)
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

            # Enforce soak-time gate before advancing to the next stage.
            # Shadow stage waits `shadow_cycles` cycles; canary stage waits
            # `canary_trades` cycles (cycle-as-proxy until drain receives a
            # real trade counter — TODO: thread trade_count through drain).
            # `deployed_at_cycle == 0` means an old package serialized before
            # the field existed — treated as "soak elapsed" for back-compat.
            current_dep = next(
                (d for d in reversed(pkg.deployments)
                 if d.stage == pkg.deploy_target and d.rolled_back_at is None),
                None,
            )
            if current_dep is not None and current_dep.deployed_at_cycle > 0:
                elapsed = current_cycle - current_dep.deployed_at_cycle
                required = (
                    self._deploy.shadow_cycles if pkg.deploy_target == DeployStage.SHADOW
                    else self._deploy.canary_trades
                )
                if elapsed < required:
                    logger.info(
                        "meta_manager.soak_window_active change_id=%s stage=%s "
                        "elapsed=%d required=%d — promotion deferred",
                        pkg.change_id, pkg.deploy_target.value, elapsed, required,
                    )
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

        Resolution order:
          1. If incident already carries `proposed_config_delta` (precomputed
             upstream by self_heal etc.), validate keys and use it.
          2. For `kind=config` with no precomputed delta, spawn the Code
             Surgeon LLM specialist with the diagnosis as input — returns
             a YAML block of {config_delta, rationale}.
          3. For other kinds (model/code/disable), return an empty proposal
             (those proposers are TODO; constitution will pass-through).

        All deltas are validated against ScannerConfig dataclass field names
        before being returned. Orphan keys are logged and dropped to prevent
        the silent dead-write failure mode promoted to a rule on 2026-04-16.
        """
        incident_delta = (pkg.incident.get("proposed_config_delta") or {}) if pkg.incident else {}
        rationale: Optional[str] = None

        if incident_delta:
            valid, dropped = _validate_config_delta(incident_delta)
            if dropped:
                logger.warning(
                    "meta_manager.proposer.dropped_orphan_keys change_id=%s dropped=%s",
                    pkg.change_id, dropped,
                )
            return Proposal(
                diff=json.dumps(valid, indent=2, sort_keys=True),
                changed_files=list(pkg.incident.get("changed_files", [])) if pkg.incident else [],
                config_delta=valid,
            )

        if pkg.kind == ChangeKind.CONFIG:
            surgeon_raw = self._invoke("code_surgeon", _build_surgeon_prompt(pkg))
            parsed = _yaml_to_dict(surgeon_raw) if surgeon_raw else {}
            raw_delta = parsed.get("config_delta", {}) if isinstance(parsed, dict) else {}
            if not isinstance(raw_delta, dict):
                raw_delta = {}
            valid, dropped = _validate_config_delta(raw_delta)
            if dropped:
                logger.warning(
                    "meta_manager.surgeon.dropped_orphan_keys change_id=%s dropped=%s",
                    pkg.change_id, dropped,
                )
            rationale = parsed.get("rationale") if isinstance(parsed, dict) else None
            logger.info(
                "meta_manager.surgeon_proposed change_id=%s keys=%s rationale=%s",
                pkg.change_id, list(valid.keys()), (rationale or "")[:120],
            )
            return Proposal(
                diff=json.dumps(valid, indent=2, sort_keys=True),
                changed_files=[],
                config_delta=valid,
            )

        # Other intervention kinds — proposers not yet implemented.
        return Proposal(diff="{}", changed_files=[], config_delta={})

    # ---------- top-level: drive an incident through Stages 1-5 ----------

    def process(self, incident: Dict[str, Any]) -> ChangePackage:
        """Drive a single incident from intake to awaiting_approval (or terminal).

        Runs Stage 1 (intake+diagnosis) → Stage 2 (propose) → Stage 3 (evaluate)
        → Stage 4 (constitution+auditor) → Stage 5 (enqueue for approval).

        Returns the package at whichever terminal stage it reached:
          - AWAITING_APPROVAL: package is in the human approval queue
          - REJECTED: constitution or auditor blocked it
          - ABORTED: an exception killed the pipeline (errors logged)

        Used by `route_incident` and the smoke driver. Individual stage
        methods remain available for tests and partial replays.
        """
        pkg = self.intake(incident)
        if pkg.stage in (ChangeStage.ABORTED, ChangeStage.REJECTED):
            return pkg
        pkg = self.propose(pkg)
        if pkg.stage in (ChangeStage.ABORTED, ChangeStage.REJECTED):
            return pkg
        pkg = self.evaluate(pkg)
        if pkg.stage in (ChangeStage.ABORTED, ChangeStage.REJECTED):
            return pkg
        pkg = self.check_constitution(pkg)
        if pkg.stage == ChangeStage.REJECTED:
            return pkg
        pkg = self.request_approval(pkg)
        return pkg


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

    Drives the full Stage 1-5 pipeline: intake → propose → evaluate →
    constitution → enqueue for approval. Returns True when the meta manager
    swallowed the signal (caller should NOT also spawn Ralph / reflection /
    direct config writes). Returns False when the legacy path should run.

    Note: this blocks for the duration of all LLM calls (3 specialists ×
    ~150s ≈ 7-8 min worst case). Callers must not be in the trade hot
    path. Today both call sites (cycle_autonomy, performance_prd_generator)
    run from the orchestrator's meta-cycle dispatcher, satisfying the
    "Claude-free runtime hot path" rule from CLAUDE.md.
    """
    if not is_enabled():
        return False
    try:
        mgr = MetaManager()
        mgr.process(incident)
        return True
    except Exception as e:
        logger.warning("meta_manager.route_incident_failed err=%s — falling back to legacy", e)
        return False


# ---------- Code Surgeon prompt + config-key validation ----------

_VALID_CONFIG_KEYS_CACHE: Optional[set] = None


def _valid_config_keys() -> set:
    """Return the set of legal ScannerConfig dataclass field names.

    Cached after first computation. Used to filter Code Surgeon-proposed
    deltas before they reach the constitution gate — enforces the
    "validate keys against ScannerConfig BEFORE writing" rule promoted
    from cycle 3 self-heal dead-letter analysis (2026-04-16).
    """
    global _VALID_CONFIG_KEYS_CACHE
    if _VALID_CONFIG_KEYS_CACHE is not None:
        return _VALID_CONFIG_KEYS_CACHE
    try:
        import dataclasses
        from src.scanner.config import ScannerConfig
        _VALID_CONFIG_KEYS_CACHE = {f.name for f in dataclasses.fields(ScannerConfig)}
    except Exception as e:
        logger.warning("meta_manager.config_keys_load_failed err=%s", e)
        _VALID_CONFIG_KEYS_CACHE = set()
    return _VALID_CONFIG_KEYS_CACHE


def _validate_config_delta(delta: Dict[str, Any]) -> tuple:
    """Filter a delta to keys that exist on ScannerConfig.

    Returns (valid_delta, dropped_keys). The empty-set case happens when
    ScannerConfig couldn't be imported — we conservatively pass everything
    through (rather than block all changes) and let the constitution gate
    catch real violations.
    """
    valid_keys = _valid_config_keys()
    if not valid_keys:
        return dict(delta or {}), []
    valid: Dict[str, Any] = {}
    dropped: List[str] = []
    for k, v in (delta or {}).items():
        if k in valid_keys:
            valid[k] = v
        else:
            dropped.append(k)
    return valid, dropped


def _build_surgeon_prompt(pkg: ChangePackage) -> str:
    """Build the Code Surgeon prompt — proposes a config_delta from diagnosis.

    Constraint surfaced in the prompt: keys must match ScannerConfig field
    names. Surgeon should return small, reversible deltas; Constitution will
    block widening of risk parameters or freshness-floor reductions.
    """
    diag_text = pkg.diagnosis.root_cause_hypothesis if pkg.diagnosis else ""
    sev = pkg.diagnosis.severity.value if pkg.diagnosis else "low"
    affected = pkg.diagnosis.affected_modules if pkg.diagnosis else []
    incident_signal = json.dumps(pkg.incident.get("signal", {}), default=str)[:1200] if pkg.incident else "{}"
    return (
        "ROLE: code_surgeon (proposes a minimal, reversible config delta to "
        "address the diagnosed root cause)\n"
        f"CHANGE_ID: {pkg.change_id}\n"
        f"DIAGNOSIS: {diag_text[:1500]}\n"
        f"SEVERITY: {sev}\n"
        f"AFFECTED_MODULES: {affected}\n"
        f"INCIDENT_SIGNAL: {incident_signal}\n"
        "Constraints:\n"
        "  - Keys MUST match ScannerConfig dataclass field names exactly "
        "(orphan keys are dropped before constitution check).\n"
        "  - Prefer small, reversible deltas. Avoid widening risk parameters; "
        "the Constitution will block risk-widening changes.\n"
        "  - If no clear config-only fix exists, return an empty config_delta "
        "with rationale stating why a code/model intervention is needed.\n"
        "Reply with EXACTLY one ```yaml fenced block containing two keys:\n"
        "  config_delta: <flat mapping of field → new value>\n"
        "  rationale: <one-line explanation>\n"
    )


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
    """Parse YAML/JSON specialist output into a flat dict.

    Prefers pyyaml (handles block scalars `|`/`>`, lists, nested mappings)
    so multi-line `root_cause_hypothesis: |` reaches us as full text rather
    than the literal `|`. Falls back to a tiny hand-rolled key:value parser
    when pyyaml is unavailable or chokes on something pathological.
    """
    text = (text or "").strip()
    if not text:
        return {}
    try:
        import yaml  # type: ignore
        loaded = yaml.safe_load(text)
        if isinstance(loaded, dict):
            return loaded
    except Exception:
        pass
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
