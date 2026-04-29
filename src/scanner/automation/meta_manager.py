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
        use_llm: bool = True,
        deterministic_surgeon: Optional[Any] = None,
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
        # Mythos audit 2026-04-28 — Deterministic Surgeon (Phase 1).
        # When use_llm=False the LLM specialists return "" so Code
        # Surgeon proposals stay empty. The deterministic surgeon
        # translates incident["diag"]["recommended_actions"] into a
        # real config_delta so the meta-pipeline produces actionable
        # proposals instead of black-hole packages. use_llm=True
        # preserves the legacy LLM-only behavior unchanged for tests
        # that don't pass this flag.
        self._use_llm = bool(use_llm)
        self._det_surgeon = deterministic_surgeon
        if not self._use_llm and self._det_surgeon is None:
            try:
                from src.scanner.automation.deterministic_surgeon import (
                    DeterministicSurgeon,
                )
                self._det_surgeon = DeterministicSurgeon()
            except Exception as e:
                logger.warning(
                    "meta_manager.det_surgeon_init_failed err=%s — "
                    "use_llm=False packages will fall through to empty proposals",
                    e,
                )
        self._changes_dir.mkdir(parents=True, exist_ok=True)
        logger.info(
            "meta_manager.initialized changes_dir=%s ledger=%s use_llm=%s "
            "surgeon=%s",
            self._changes_dir, self._ledger, self._use_llm,
            "llm" if self._use_llm else (
                "deterministic" if self._det_surgeon is not None else "none"
            ),
        )

    # ---------- Stage 0-1: intake ----------

    def intake(self, incident: Dict[str, Any]) -> ChangePackage:
        """Open a new package, run Incident Analyst, persist.

        Pre-processing gate order (in execution sequence):
          1. G8 dedup — drop incidents whose (kind, proposed_config_delta)
             hash matches an in-flight package
          2. G3 episodic memory — when injected, query_similar(setup_features)
             attaches historical_outcomes for downstream constitution check
          3. concurrency throttle — drop if max_concurrent active packages
          4. analyst invocation — proceed with diagnosis

        Each pre-stage gate is fail-open: errors log warnings and let the
        package proceed (rejecting on a corrupt state file would be worse
        than admitting a duplicate or running without history).
        """
        # G8: dedup check before any expensive processing.
        incident_dict = dict(incident or {})
        incident_delta = incident_dict.get("proposed_config_delta") or {}
        candidate = ChangePackage(
            incident=incident_dict,
            proposal=Proposal(config_delta=dict(incident_delta)) if incident_delta else None,
        )
        candidate_hash = candidate.dedup_hash()
        if self._is_duplicate_in_flight(candidate_hash):
            logger.info(
                "meta_manager.intake_dup_dropped change_id=%s hash=%s",
                candidate.change_id, candidate_hash,
            )
            candidate.stage = ChangeStage.ABORTED
            candidate.rejection_reason = "duplicate_proposal"
            candidate.touch("duplicate_proposal_dropped")
            # Drop the stub proposal — downstream consumers should not see a
            # synthetic Proposal that the surgeon never produced.
            candidate.proposal = None
            self._persist(candidate)
            self._append_ledger(candidate, "duplicate_dropped")
            return candidate

        # G3: episodic memory query — attach historical_outcomes for downstream use.
        if self._episodic_memory is not None:
            try:
                features = incident_dict.get("setup_features") or {}
                if features:
                    outcomes = self._episodic_memory.query_similar(
                        pair=features.get("pair", ""),
                        direction=features.get("direction", ""),
                        regime=features.get("regime", ""),
                        session=features.get("session", ""),
                        news_risk_score=float(features.get("news_risk_score", 0.0)),
                        uncertainty_score=float(features.get("uncertainty_score", 0.0)),
                    )
                    # Schema bridge: EpisodicMemory.query_similar returns
                    # `similar_count`, but Constitution._eval_historical_loss_rate
                    # (G7) reads `matched_episodes`. Without this normalisation
                    # the historical_loss_rate veto would *always* hit the
                    # `insufficient_history` abstain branch in production —
                    # the two halves never met before Phase 1 wired them
                    # end-to-end (Task 11 smoke surfaced the drift). Mirror
                    # additively so any other reader expecting `similar_count`
                    # keeps working.
                    if (
                        isinstance(outcomes, dict)
                        and "similar_count" in outcomes
                        and "matched_episodes" not in outcomes
                    ):
                        outcomes = {
                            **outcomes,
                            "matched_episodes": outcomes["similar_count"],
                        }
                    incident_dict["historical_outcomes"] = outcomes
            except Exception as e:
                logger.warning("meta_manager.intake_episodic_query_failed err=%s", e)
                incident_dict["historical_outcomes"] = None

        if self._concurrent_count() >= self._max_concurrent:
            logger.info(
                "meta_manager.intake_throttled active=%s max=%s — incident dropped",
                self._concurrent_count(), self._max_concurrent,
            )
            pkg = ChangePackage(incident=incident_dict)
            pkg.stage = ChangeStage.ABORTED
            pkg.touch("throttled_by_max_concurrent")
            self._persist(pkg)
            self._append_ledger(pkg, "throttled")
            return pkg

        pkg = ChangePackage(incident=incident_dict)
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

    def _is_duplicate_in_flight(self, dedup_hash: str) -> bool:
        """Scan changes_dir for an active package matching this dedup_hash.

        Skips terminal stages (closed/aborted/rejected). Failure-tolerant —
        on any read error, returns False (fails open) and logs a warning so
        we never reject a legitimate incident due to a corrupt JSON file in
        the changes directory.

        Both sides of the comparison hash from incident.proposed_config_delta
        so candidates and persisted packages use the same data source — the
        persisted package may still be at intake/diagnosing stage with no
        proposal attached, but the incident-level delta is set immediately.
        """
        try:
            for path in self._changes_dir.glob("*.json"):
                blob = safe_json_read(path, default={})
                if not isinstance(blob, dict):
                    continue
                stage = str(blob.get("stage", ""))
                if stage in ("closed", "aborted", "rejected"):
                    continue
                other_incident = blob.get("incident", {}) or {}
                other_delta = other_incident.get("proposed_config_delta") or {}
                other_pkg = ChangePackage(
                    incident=other_incident,
                    proposal=Proposal(config_delta=dict(other_delta)) if other_delta else None,
                )
                if other_pkg.dedup_hash() == dedup_hash:
                    return True
        except Exception as e:
            logger.warning("meta_manager.dedup_scan_failed err=%s", e)
        return False

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

    # Mythos audit 2026-04-29 — packages stuck in non-terminal stages
    # past this age are treated as orphaned (the producer crashed or
    # the pipeline lost track of them). Without this timeout, a single
    # stale package blocks ALL new incidents from routing through the
    # meta-pipeline because _concurrent_count >= max_concurrent.
    # Production case: be14953921fb stuck at policy_check for 22h
    # blocked every per-cycle diagnostic the orchestrator tried to
    # route. 2 hours is conservative — a real package fully
    # transitions in seconds without the LLM, minutes with one.
    _STALE_PACKAGE_AGE_S: float = 2.0 * 3600.0

    def _concurrent_count(self) -> int:
        """Count packages currently in flight (non-terminal stage AND not orphaned).

        Orphan filter: a package with updated_at older than
        ``_STALE_PACKAGE_AGE_S`` is treated as not in-flight regardless
        of stage. The stale package itself is NOT modified — operator
        can still inspect it via the F2 inbox or via revert_by_id —
        but it stops blocking the throttle.
        """
        terminal = {ChangeStage.CLOSED, ChangeStage.REJECTED, ChangeStage.ABORTED}
        from datetime import datetime, timezone
        now = datetime.now(timezone.utc)
        count = 0
        for pkg in self.list_packages():
            if pkg.stage in terminal:
                continue
            try:
                ts = (
                    datetime.fromisoformat(str(pkg.updated_at).replace("Z", "+00:00"))
                    if pkg.updated_at else None
                )
            except (ValueError, TypeError):
                ts = None
            if ts is not None:
                if ts.tzinfo is None:
                    ts = ts.replace(tzinfo=timezone.utc)
                age_s = (now - ts).total_seconds()
                if age_s > self._STALE_PACKAGE_AGE_S:
                    logger.info(
                        "meta_manager.stale_package_skipped change_id=%s "
                        "stage=%s age_h=%.1f — not counted toward throttle",
                        pkg.change_id, pkg.stage.value if hasattr(pkg.stage, 'value') else pkg.stage,
                        age_s / 3600.0,
                    )
                    continue
            count += 1
        return count

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
            # Deterministic surgeon path (Phase 1) — activates only when
            # use_llm=False AND a surgeon is wired AND the incident
            # carries diagnostic recommended_actions. Falls through to
            # the LLM Surgeon path otherwise so default behavior is
            # unchanged for existing tests.
            if (
                not self._use_llm
                and self._det_surgeon is not None
                and isinstance(pkg.incident, dict)
                and isinstance(pkg.incident.get("diag"), dict)
            ):
                det_delta, det_dropped = self._det_surgeon.translate(pkg.incident)
                if det_dropped:
                    logger.warning(
                        "meta_manager.det_surgeon.dropped change_id=%s actions=%s",
                        pkg.change_id, det_dropped,
                    )
                # Re-validate against ScannerConfig as defense-in-depth —
                # the surgeon already filters orphans but a corrupt
                # incident.diag could in principle slip through.
                valid, validator_dropped = _validate_config_delta(det_delta)
                if validator_dropped:
                    logger.warning(
                        "meta_manager.det_surgeon.validator_dropped "
                        "change_id=%s dropped=%s",
                        pkg.change_id, validator_dropped,
                    )
                if valid:
                    logger.info(
                        "meta_manager.det_surgeon_proposed change_id=%s keys=%s",
                        pkg.change_id, list(valid.keys()),
                    )
                    return Proposal(
                        diff=json.dumps(valid, indent=2, sort_keys=True),
                        changed_files=[],
                        config_delta=valid,
                    )
                # No deterministic delta produced (all actions deferred
                # or dropped) — fall through to the LLM path so an
                # operator-supplied invoker can still try.

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


# ---------- Production singleton (Phase 1 wiring) ----------

_PRODUCTION_MGR: Optional["MetaManager"] = None
_PRODUCTION_REGIME_PROBE: Optional[Any] = None


def _build_production_manager() -> "MetaManager":
    """Lazy-construct the live MetaManager with Phase 1 dependencies wired.

    Phase 1 closed the cybernetic feedback edges (G3 episodic-memory query
    on intake, G7 historical_loss_rate constitution clause, regime tagging
    on DeploymentRecord). All three require dependencies that must be
    injected at construction time:

    - episodic_memory: real EpisodicMemory pointed at the canonical
      trained_data/episodic_memory.json. The agent team writes outcomes
      to the same file, so the meta pipeline naturally sees real history.
    - regime_detector: a thin _LiveRegimeProbe whose .current() reads
      the most recent regime persisted to disk by the agent team. (No
      orchestrator-side wiring path exists in production — `cycle_autonomy`
      → `route_incident` is the only hot path, so the probe must
      self-fetch rather than be pushed updates.)

    Construction is gated on is_enabled() so importing this module is free
    when meta is disabled. Built lazily on first use and cached.
    """
    global _PRODUCTION_REGIME_PROBE

    # Episodic memory: shared backing file with the agent team
    episodic_memory: Optional[Any] = None
    try:
        from src.scanner.automation.episodic_memory import EpisodicMemory
        episodic_memory = EpisodicMemory()
    except Exception as e:
        logger.warning(
            "meta_manager.production_episodic_init_failed err=%s — "
            "G3 intake query and G7 historical_loss_rate veto will be no-ops",
            e,
        )

    # Regime probe: self-fetching from the dominant per-scan regime that
    # the agent team writes to a state file. Decoupled from any push path.
    if _PRODUCTION_REGIME_PROBE is None:
        _PRODUCTION_REGIME_PROBE = _LiveRegimeProbe()

    # Build the StagedDeployer with the regime probe wired.
    staged_deployer: Optional[Any] = None
    cfg: Optional[Any] = None
    try:
        from src.scanner.automation.staged_deployer import StagedDeployer
        from src.scanner.automation.adjustment_approver import AdjustmentApprover
        from src.scanner.config import ScannerConfig
        cfg = ScannerConfig()
        staged_deployer = StagedDeployer(
            config=cfg,
            adjustment_approver=AdjustmentApprover(),
            shadow_cycles=getattr(cfg, "staged_deploy_shadow_cycles", 20),
            canary_trades=getattr(cfg, "staged_deploy_canary_trades", 10),
            regime_detector=_PRODUCTION_REGIME_PROBE,
        )
    except Exception as e:
        logger.warning(
            "meta_manager.production_deployer_init_failed err=%s — "
            "regime_at_deploy will not be captured", e,
        )

    # The Meta pipeline is the automation that REPLACES LLM specialists in
    # the middle — Constitution + Scorecard + revert_by_id gate
    # deterministically. LLM specialists are an optional enrichment, not
    # the actual decision path. Default to the no-op invoker so the
    # production loop runs Claude-free (per CLAUDE.md trading invariant
    # "Buddy's runtime is Claude-free; LLM is for planning, post-mortems,
    # and brainstorming only"). Operators can flip meta_manager_use_llm=True
    # in the smart profile when they want specialists to enrich the dossier
    # for human review (post-hoc, not in the gate path).
    use_llm = False
    if cfg is not None:
        use_llm = bool(getattr(cfg, "meta_manager_use_llm", False))
    invoker: Optional[SpecialistInvoker] = None  # None → _default_specialist_invoker (Claude CLI)
    if not use_llm:
        invoker = lambda mode, prompt: ""  # noqa: E731

    return MetaManager(
        episodic_memory=episodic_memory,
        staged_deployer=staged_deployer,
        specialist_invoker=invoker,
        use_llm=use_llm,
    )


class _LiveRegimeProbe:
    """Self-fetching regime probe for the production MetaManager singleton.

    The orchestrator-side wiring (in orchestrator.py) pushes regime updates
    on every scan; that path doesn't apply here because the production live
    loop (cycle_autonomy → route_incident) bypasses the orchestrator. This
    probe instead pulls from a regime state file that other Buddy components
    already maintain. Falls through to None when no source is available —
    Phase-1 Task-7 contract is best-effort (regime tagging is observability,
    not gating).
    """

    def current(self) -> Optional[str]:
        # Order of preference:
        #   1. trained_data/microstructure_regime.json (`regime` key if present)
        #   2. trained_data/ewma_correlation_state.json (`regime`)
        #   3. None — caller falls back gracefully
        try:
            from pathlib import Path
            from src.scanner.automation.safe_json import safe_json_read
            root = Path(__file__).resolve().parents[3] / "trained_data"
            for fname in ("microstructure_regime.json", "ewma_correlation_state.json"):
                blob = safe_json_read(root / fname, default={})
                if isinstance(blob, dict):
                    val = blob.get("regime") or blob.get("dominant_regime") or blob.get("current_regime")
                    if val:
                        return str(val).upper()
        except Exception:
            pass
        return None


def route_incident(incident: Dict[str, Any]) -> bool:
    """Route an incident through the meta-pipeline if enabled.

    Drives the full Stage 1-5 pipeline: intake → propose → evaluate →
    constitution → enqueue for approval. Returns True when the meta manager
    swallowed the signal (caller should NOT also spawn Ralph / reflection /
    direct config writes). Returns False when the legacy path should run.

    Phase-1 wiring: the MetaManager singleton is lazy-built with full
    dependencies (episodic_memory, regime_detector via _build_production_manager)
    so the G3/G7 closed-loop edges actually circulate live data, not just
    the synthetic fixtures the unit tests use.

    Note: this blocks for the duration of all LLM calls (3 specialists ×
    ~150s ≈ 7-8 min worst case). Callers must not be in the trade hot
    path. Today both call sites (cycle_autonomy, performance_prd_generator)
    run from the orchestrator's meta-cycle dispatcher, satisfying the
    "Claude-free runtime hot path" rule from CLAUDE.md.
    """
    global _PRODUCTION_MGR
    if not is_enabled():
        return False
    try:
        if _PRODUCTION_MGR is None:
            _PRODUCTION_MGR = _build_production_manager()
            # Determine the cognition mode from the actual invoker we wired.
            # The no-op lambda returns "" — this is the deterministic-automation
            # mode (default per CLAUDE.md). The Claude-CLI default invoker is
            # opt-in via meta_manager_use_llm=True for dossier enrichment only.
            _probe_invoker = _PRODUCTION_MGR._invoke
            try:
                _is_noop = _probe_invoker("__probe__", "") == ""
            except Exception:
                _is_noop = False
            logger.info(
                "meta_manager.production_singleton_initialized "
                "mode=%s episodic_memory=%s regime_probe=%s deployer=%s",
                "AUTOMATION" if _is_noop else "LLM_ENRICHED",
                "wired" if _PRODUCTION_MGR._episodic_memory is not None else "DISABLED",
                "wired" if _PRODUCTION_REGIME_PROBE is not None else "DISABLED",
                "wired" if _PRODUCTION_MGR._deploy is not None else "DISABLED",
            )
        _PRODUCTION_MGR.process(incident)
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

    The prompt enumerates VALID_CONFIG_KEYS so the surgeon LLM is grounded
    in the actual config surface — reduces orphan-key hallucinations
    upstream of `_validate_config_delta` (which still drops them at parse
    time as a safety net).
    """
    diag_text = pkg.diagnosis.root_cause_hypothesis if pkg.diagnosis else ""
    sev = pkg.diagnosis.severity.value if pkg.diagnosis else "low"
    affected = pkg.diagnosis.affected_modules if pkg.diagnosis else []
    incident_signal = json.dumps(pkg.incident.get("signal", {}), default=str)[:1200] if pkg.incident else "{}"
    valid_keys_list = sorted(_valid_config_keys())
    if valid_keys_list:
        valid_keys_block = "\n  ".join(valid_keys_list)
    else:
        valid_keys_block = (
            "(unable to introspect ScannerConfig — surgeon may emit any "
            "plausible field; orphan keys will be dropped at parse time)"
        )
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
        "VALID_CONFIG_KEYS (use ONLY keys from this list):\n"
        f"  {valid_keys_block}\n"
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
