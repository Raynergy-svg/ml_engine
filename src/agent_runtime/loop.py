"""AXIOM Agent Runtime — resident reasoning loop.

OBSERVE -> DIAGNOSE -> PROPOSE -> ACT -> VERIFY -> LOG.

Claude reasons and sequences; it is NEVER in the trade hot path. This module
runs on its own schedule (cron/daemon/manual trigger, not per-scan/per-trade),
gathers real state through the read-only tool registry, asks the resolved
``claude`` CLI to reason over it, and can only ever mutate state through
``src.agent_runtime.policy.PolicyEngine`` — the SAME structural guarantees
from policy.py apply here unchanged:

  - ESCALATION is always proposal-only, regardless of any flag. This loop
    never gives an ESCALATION action a different treatment based on
    ``agent_autonomy_enabled`` -- the tier's structural guarantee (no execute
    callable can ever exist on an ESCALATION ActionSpec) makes that the only
    possible behavior, not a design choice this module could get wrong.
  - OPERATIONAL / DEESCALATION / SELF_IMPROVE actions PROPOSED by the LLM
    additionally require ``agent_autonomy_enabled=True`` (an operator-set
    flag, persisted to disk, DEFAULT FALSE) before this loop will actually
    call ``PolicyEngine.submit()`` with intent to execute. With the flag
    False, the loop still OBSERVEs (the read-only tools always run -- they
    gather the context the LLM reasons over, and are harmless by
    construction) and still PROPOSEs, but ACT logs a "shadow" audit line
    ("would execute ...") instead of calling submit(). SELF_IMPROVE actions
    are additionally bounded by ``src.agent_runtime.self_improve``'s own
    structural allow/deny check and test+verify+commit-or-revert gate --
    autonomy=True only lets the loop ATTEMPT one of a fixed menu of narrow,
    pre-coded edits; it does not widen what those edits are allowed to touch.

Degrades honestly if the claude CLI is unavailable: OBSERVE still runs,
DIAGNOSE returns an ``available: False`` stub with no proposed actions, and
the cycle completes and logs normally -- never a crash, never a fabricated
diagnosis.
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import tempfile
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from src.agent_runtime import audit as art_audit
from src.agent_runtime.policy import ActionTier, PolicyDenied, PolicyEngine, default_engine
from src.agent_runtime.self_improve import SELF_IMPROVE_ACTIONS
from src.agent_runtime.tools import readers
from src.agent_runtime.tools.registry import TOOL_ACTIONS, TOOLS
from src.axiom_operator.session import append_jsonl, tail_jsonl, utc_now
from src.utils.claude_cli import augmented_path_env, resolve_claude_cli

REPO_ROOT = Path(__file__).resolve().parents[2]
AXIOM_DIR = REPO_ROOT / "trained_data" / "axiom"
CYCLES_PATH = AXIOM_DIR / "loop_cycles.jsonl"
AUTONOMY_CONFIG_PATH = AXIOM_DIR / "loop_autonomy.json"

DIAGNOSIS_BLOCK_RE = re.compile(r"<agent-loop>(.*?)</agent-loop>", re.DOTALL | re.IGNORECASE)

_DEFAULT_OBSERVE_TOOLS: Tuple[str, ...] = tuple(name for name in TOOLS if name != "shadow_ledger")
_DEFAULT_SHADOW_LANES: Tuple[str, ...] = tuple(readers.list_shadow_lanes())

SubprocessRunner = Callable[..., "subprocess.CompletedProcess[str]"]


# ---------------------------------------------------------------------------
# Autonomy flag — operator-set, persisted, DEFAULT FALSE (shadow mode).
# ---------------------------------------------------------------------------


def is_autonomy_enabled(path: Path = AUTONOMY_CONFIG_PATH) -> bool:
    """True only if the operator has explicitly written ``{"agent_autonomy_enabled": true}``
    to ``path``. Missing file, corrupt JSON, or any other content => False.
    This is the fail-closed default: a resident loop that has never been
    configured must run in shadow (observe+propose-only) mode."""
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False
    if not isinstance(payload, dict):
        return False
    # Strict `is True` -- not `bool(...)` truthy coercion. A string "yes"/"true"
    # or a non-zero int in this field must NOT flip the flag on: only a literal
    # JSON boolean true does. This is the fail-closed contract the ACT gate
    # depends on for every OPERATIONAL/DEESCALATION action.
    return payload.get("agent_autonomy_enabled", False) is True


def set_autonomy_enabled(value: bool, path: Path = AUTONOMY_CONFIG_PATH) -> Dict[str, Any]:
    """Atomically persist the operator's autonomy choice (tmp + os.replace)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"agent_autonomy_enabled": bool(value), "updated_at": utc_now()}
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2, sort_keys=True)
            fh.write("\n")
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, path)
    finally:
        try:
            if os.path.exists(tmp):
                os.unlink(tmp)
        except OSError:
            pass
    return payload


# ---------------------------------------------------------------------------
# Cycle data shapes
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ProposedAction:
    action: str
    params: Dict[str, Any]
    rationale: str
    tier: Optional[str]  # None => unclassifiable / unknown action name


@dataclass(frozen=True)
class ActionOutcome:
    action: str
    tier: Optional[str]
    executed: bool
    shadow: bool
    proposal: bool
    denied: bool
    detail: Dict[str, Any] = field(default_factory=dict)
    proposal_id: Optional[str] = None
    """Set only when ``proposal`` is True -- the stable id (policy.compute_proposal_id)
    the Activity panel's Accept/Deny endpoints (dashboard/server/axiom_proposals.py)
    use to reference this specific ESCALATION proposal."""


@dataclass(frozen=True)
class CycleResult:
    cycle_id: str
    ts: str
    actor: str
    cli_available: bool
    autonomy_enabled: bool
    observations: Dict[str, Any]
    diagnosis: Dict[str, Any]
    proposed_actions: List[ProposedAction]
    outcomes: List[ActionOutcome]
    verified: bool
    verify_detail: Dict[str, Any]


class ResidentLoop:
    """One OBSERVE->DIAGNOSE->PROPOSE->ACT->VERIFY->LOG cycle per call to
    ``run_cycle()``. Scheduling (cron, daemon tick, dashboard-triggered
    button) lives OUTSIDE this class -- it has no internal timer."""

    def __init__(
        self,
        *,
        project_root: Path = REPO_ROOT,
        engine: Optional[PolicyEngine] = None,
        cycles_path: Path = CYCLES_PATH,
        autonomy_path: Path = AUTONOMY_CONFIG_PATH,
        observe_tool_names: Tuple[str, ...] = _DEFAULT_OBSERVE_TOOLS,
        shadow_lanes: Tuple[str, ...] = _DEFAULT_SHADOW_LANES,
        claude_bin: str = "claude",
        subprocess_runner: Optional[SubprocessRunner] = None,
        cli_resolver: Callable[[str], Optional[str]] = resolve_claude_cli,
    ) -> None:
        self.project_root = Path(project_root)
        self.engine = engine or default_engine(TOOL_ACTIONS, SELF_IMPROVE_ACTIONS)
        self.cycles_path = Path(cycles_path)
        self.autonomy_path = Path(autonomy_path)
        self.observe_tool_names = tuple(observe_tool_names)
        self.shadow_lanes = tuple(shadow_lanes)
        self.claude_bin = claude_bin
        self.subprocess_runner = subprocess_runner or subprocess.run
        self.cli_resolver = cli_resolver

    def autonomy_enabled(self) -> bool:
        return is_autonomy_enabled(self.autonomy_path)

    def run_cycle(self, *, actor: str = "agent_runtime_loop", timeout_seconds: int = 90) -> CycleResult:
        cycle_id = f"cycle-{uuid.uuid4().hex[:12]}"
        observations = self._observe(actor=actor)

        cli_path = self.cli_resolver(self.claude_bin)
        cli_available = bool(cli_path)
        diagnosis = self._diagnose(observations, cli_path=cli_path, timeout_seconds=timeout_seconds)

        proposed = self._propose(diagnosis)
        autonomy = self.autonomy_enabled()
        outcomes = self._act(proposed, actor=actor, autonomy_enabled=autonomy)
        try:
            verified, verify_detail = self._verify(outcomes)
        except Exception as exc:  # noqa: BLE001 - VERIFY must never suppress LOG of already-acted-on outcomes
            verified, verify_detail = False, {"verify_error": repr(exc)}

        result = CycleResult(
            cycle_id=cycle_id, ts=utc_now(), actor=actor, cli_available=cli_available,
            autonomy_enabled=autonomy, observations=observations, diagnosis=diagnosis,
            proposed_actions=proposed, outcomes=outcomes, verified=verified, verify_detail=verify_detail,
        )
        self._log(result)
        return result

    # -- OBSERVE --------------------------------------------------------

    def _observe(self, *, actor: str) -> Dict[str, Any]:
        """Call every read-only tool through the SAME policy gateway every
        other action uses (preflighted + audited). These are OPERATIONAL
        tier and harmless by construction (see tools/readers.py), so this
        step runs unconditionally -- it is the loop gathering its own
        context, not an action the LLM decided to take."""
        out: Dict[str, Any] = {}
        for name in self.observe_tool_names:
            params = self._default_tool_params(name)
            try:
                result = self.engine.submit(name, params, actor=actor)
                out[name] = {"ok": True, "data": result.outcome}
            except Exception as exc:  # noqa: BLE001 - one bad reader must not blank the whole cycle
                out[name] = {"ok": False, "error": repr(exc)}
        for lane in self.shadow_lanes:
            key = f"shadow_ledger:{lane}"
            try:
                result = self.engine.submit("shadow_ledger", {"lane": lane}, actor=actor)
                out[key] = {"ok": True, "data": result.outcome}
            except Exception as exc:  # noqa: BLE001
                out[key] = {"ok": False, "error": repr(exc)}
        return out

    def _default_tool_params(self, name: str) -> Dict[str, Any]:
        tool = TOOLS.get(name)
        if tool is None:
            return {}
        params: Dict[str, Any] = {}
        if "limit" in tool.args:
            params["limit"] = 20
        if "max_chars" in tool.args:
            params["max_chars"] = 4000
        return params

    # -- DIAGNOSE ---------------------------------------------------------

    def _diagnose(self, observations: Dict[str, Any], *, cli_path: Optional[str], timeout_seconds: int) -> Dict[str, Any]:
        if not cli_path:
            return {
                "available": False,
                "reasoning": "claude CLI not found on PATH -- degraded cycle: OBSERVE ran, no diagnosis, no actions proposed.",
                "beliefs": "unknown (no reasoning step available this cycle)",
                "proposed_actions": [],
            }

        prompt = self._build_prompt(observations)
        command = [
            cli_path, "--print", "--no-session-persistence", "--permission-mode", "dontAsk",
            "--disallowedTools", "Bash,Edit,Write,NotebookEdit,WebFetch,WebSearch",
            "--name", "AXIOM Resident Loop",
        ]
        child_env = os.environ.copy()
        child_env["PATH"] = augmented_path_env(child_env)

        prompt_file: Optional[str] = None
        try:
            with tempfile.NamedTemporaryFile("w", suffix=".md", prefix="agent_runtime_loop_", delete=False, dir="/tmp") as fh:
                fh.write(prompt)
                prompt_file = fh.name
            # Prompt content includes observation data (journal/learnings text,
            # not secrets, but still not for other local users) -- restrict to
            # owner-only before it's read back for the subprocess's stdin.
            os.chmod(prompt_file, 0o600)
            with open(prompt_file, "r", encoding="utf-8") as stdin_fh:
                proc = self.subprocess_runner(
                    command, stdin=stdin_fh, capture_output=True, text=True,
                    timeout=timeout_seconds, cwd=str(self.project_root), env=child_env,
                )
        except subprocess.TimeoutExpired:
            return {
                "available": True,
                "reasoning": f"claude timed out after {timeout_seconds}s -- degraded cycle, no actions proposed.",
                "beliefs": "unknown", "proposed_actions": [],
            }
        except OSError as exc:
            return {
                "available": True,
                "reasoning": f"claude subprocess failed: {exc} -- degraded cycle, no actions proposed.",
                "beliefs": "unknown", "proposed_actions": [],
            }
        finally:
            if prompt_file:
                try:
                    os.unlink(prompt_file)
                except OSError:
                    pass

        if proc.returncode != 0:
            return {
                "available": True,
                "reasoning": f"claude exited {proc.returncode} -- degraded cycle, no actions proposed.",
                "beliefs": "unknown", "proposed_actions": [],
            }

        parsed = self._parse_stdout(proc.stdout or "")
        parsed["available"] = True
        return parsed

    def _build_prompt(self, observations: Dict[str, Any]) -> str:
        registry = self.engine._registry  # noqa: SLF001 -- read-only introspection, same process
        known_actions = sorted(registry)
        param_hints = [
            f"  {name}: {spec.params_hint}"
            for name, spec in sorted(registry.items())
            if spec.params_hint
        ]
        hints_block = (
            "\n\nExact parameter contracts for actions that take params (using any OTHER "
            "kwarg name is REJECTED -- the action is denied, not partially applied):\n"
            + "\n".join(param_hints)
            if param_hints else ""
        )
        return (
            "You are the AXIOM resident reasoning loop for a PRACTICE-only trading bot. "
            "You reason and sequence; you are NEVER in the trade hot path and cannot execute "
            "anything directly. Every action you propose is independently tier-classified and "
            "gated by a policy engine -- ESCALATION actions (unhalt, arm, leverage increase, "
            "promotion, strategy/code change) are ALWAYS routed to the operator, never auto-run.\n\n"
            "Observed state (read-only tool output) is below. Return exactly one block:\n"
            "<agent-loop>{\n"
            '  "beliefs": "what you believe is true about system state right now",\n'
            '  "reasoning": "one or two sentences of reasoning",\n'
            '  "proposed_actions": [{"action": "<name>", "params": {}, "rationale": "..."}]\n'
            "}</agent-loop>\n\n"
            f"Known action names: {known_actions}"
            f"{hints_block}\n\n"
            f"Observations:\n{json.dumps(observations, indent=2, sort_keys=True, default=str)}\n"
        )

    def _parse_stdout(self, stdout: str) -> Dict[str, Any]:
        match = DIAGNOSIS_BLOCK_RE.search(stdout or "")
        if not match:
            return {"reasoning": "no structured diagnosis block found in claude output", "beliefs": "unknown", "proposed_actions": []}
        raw = match.group(1).strip()
        try:
            payload = json.loads(raw)
        except ValueError:
            return {"reasoning": "diagnosis block was not valid JSON", "beliefs": "unknown", "proposed_actions": []}
        if not isinstance(payload, dict):
            return {"reasoning": "diagnosis block was not a JSON object", "beliefs": "unknown", "proposed_actions": []}
        payload.setdefault("reasoning", "")
        payload.setdefault("beliefs", "")
        payload.setdefault("proposed_actions", [])
        return payload

    # -- PROPOSE ------------------------------------------------------------

    def _propose(self, diagnosis: Dict[str, Any]) -> List[ProposedAction]:
        raw = diagnosis.get("proposed_actions")
        if not isinstance(raw, list):
            return []
        proposed: List[ProposedAction] = []
        for item in raw[:20]:
            if not isinstance(item, dict):
                continue
            action = str(item.get("action") or "").strip()
            if not action:
                continue
            params = item.get("params") if isinstance(item.get("params"), dict) else {}
            rationale = str(item.get("rationale") or "")[:500]
            tier = self.engine.classify(action)
            proposed.append(ProposedAction(
                action=action, params=dict(params), rationale=rationale,
                tier=tier.value if tier is not None else None,
            ))
        return proposed

    # -- ACT ------------------------------------------------------------

    def _act(self, proposed: List[ProposedAction], *, actor: str, autonomy_enabled: bool) -> List[ActionOutcome]:
        outcomes: List[ActionOutcome] = []
        for item in proposed:
            if item.tier is None:
                outcomes.append(self._act_unknown(item, actor=actor))
                continue
            if item.tier == ActionTier.ESCALATION.value:
                outcomes.append(self._act_escalation(item, actor=actor))
                continue
            # OPERATIONAL / DEESCALATION / SELF_IMPROVE -- gated by the autonomy flag.
            # (SELF_IMPROVE's own module adds a second, independent allow/deny +
            # test/verify/commit-or-revert gate on top of this -- see self_improve.py.)
            if not autonomy_enabled:
                outcomes.append(self._act_shadow(item, actor=actor))
                continue
            outcomes.append(self._act_execute(item, actor=actor))
        return outcomes

    def _act_unknown(self, item: ProposedAction, *, actor: str) -> ActionOutcome:
        try:
            self.engine.submit(item.action, item.params, actor=actor)  # fail-closed deny, self-audits
        except PolicyDenied as exc:
            return ActionOutcome(
                action=item.action, tier=None, executed=False, shadow=False,
                proposal=False, denied=True, detail={"reason": str(exc)},
            )
        # Should be unreachable: submit() always raises PolicyDenied for an
        # unregistered action name. Treat a non-raise as a denial too (fail-closed).
        return ActionOutcome(
            action=item.action, tier=None, executed=False, shadow=False,
            proposal=False, denied=True, detail={"reason": "unknown action"},
        )

    def _act_escalation(self, item: ProposedAction, *, actor: str) -> ActionOutcome:
        try:
            result = self.engine.submit(item.action, item.params, actor=actor)
        except PolicyDenied as exc:
            return ActionOutcome(
                action=item.action, tier=item.tier, executed=False, shadow=False,
                proposal=False, denied=True, detail={"reason": str(exc)},
            )
        except Exception as exc:  # noqa: BLE001 -- an ESCALATION attempt is ALWAYS
            # refused (submit() never gives this tier an execute path -- see
            # policy.PolicyEngine.submit's ESCALATION branch). But the act of
            # BLOCKING it can itself raise (e.g. PolicyRegistrationError from the
            # tampered-spec defense-in-depth check, or any other error while
            # building/auditing the denial) -- that must degrade to a logged
            # deny, never propagate and kill the whole cycle. The escalation is
            # refused either way; only the failure mode changes.
            return ActionOutcome(
                action=item.action, tier=item.tier, executed=False, shadow=False,
                proposal=False, denied=True, detail={"reason": f"escalation blocked with error: {exc!r}"},
            )
        proposal_detail: Dict[str, Any] = {}
        proposal_id: Optional[str] = None
        if result.proposal is not None:
            proposal_detail = {
                "action": result.proposal.action, "rationale": result.proposal.rationale,
                "requires": result.proposal.requires, "params": result.proposal.params,
            }
            proposal_id = result.proposal.proposal_id
        return ActionOutcome(
            action=item.action, tier=item.tier, executed=result.executed, shadow=False,
            proposal=True, denied=False, detail=proposal_detail, proposal_id=proposal_id,
        )

    def _act_shadow(self, item: ProposedAction, *, actor: str) -> ActionOutcome:
        art_audit.record(
            action=item.action, tier=item.tier, actor=actor, allowed=False,
            outcome="shadow_not_executed",
            reason="agent_autonomy_enabled is False -- resident loop is in shadow mode; logging intent only",
            params=item.params,
        )
        return ActionOutcome(
            action=item.action, tier=item.tier, executed=False, shadow=True,
            proposal=False, denied=False, detail={"reason": "agent_autonomy_enabled=False"},
        )

    def _act_execute(self, item: ProposedAction, *, actor: str) -> ActionOutcome:
        try:
            result = self.engine.submit(item.action, item.params, actor=actor)
        except PolicyDenied as exc:
            return ActionOutcome(
                action=item.action, tier=item.tier, executed=False, shadow=False,
                proposal=False, denied=True, detail={"reason": str(exc)},
            )
        except Exception as exc:  # noqa: BLE001 -- an action's execute() raising an
            # unexpected error (e.g. the LLM's proposal used a wrong param name) must
            # deny that ONE action, not crash the whole cycle. submit() already
            # audited this as outcome="error" before re-raising -- nothing is lost.
            return ActionOutcome(
                action=item.action, tier=item.tier, executed=False, shadow=False,
                proposal=False, denied=True, detail={"reason": f"action execution error: {exc!r}"},
            )
        return ActionOutcome(
            action=item.action, tier=item.tier, executed=result.executed, shadow=False,
            proposal=False, denied=False, detail=result.outcome,
        )

    # -- VERIFY ------------------------------------------------------------

    def _verify(self, outcomes: List[ActionOutcome]) -> Tuple[bool, Dict[str, Any]]:
        from src.scanner.config import ScannerConfig

        env = getattr(ScannerConfig(), "oanda_environment", "practice")
        practice_ok = env == "practice"
        no_escalation_executed = all(
            not (o.tier == ActionTier.ESCALATION.value and o.executed) for o in outcomes
        )
        ok = practice_ok and no_escalation_executed
        detail = {
            "practice_pin_ok": practice_ok, "oanda_environment": env,
            "no_escalation_executed": no_escalation_executed,
        }
        if not ok:
            art_audit.record(
                action="resident_loop_verify", tier=None, actor="agent_runtime_loop",
                allowed=False, outcome="verify_failed", reason=json.dumps(detail), params={},
            )
        return ok, detail

    # -- LOG ------------------------------------------------------------

    def _log(self, result: CycleResult) -> None:
        executed_count = sum(1 for o in result.outcomes if o.executed)
        shadow_count = sum(1 for o in result.outcomes if o.shadow)
        proposal_count = sum(1 for o in result.outcomes if o.proposal)
        denied_count = sum(1 for o in result.outcomes if o.denied)

        art_audit.record_cycle(
            cycle_id=result.cycle_id, actor=result.actor, cli_available=result.cli_available,
            autonomy_enabled=result.autonomy_enabled,
            reasoning_summary=str(result.diagnosis.get("reasoning") or ""),
            proposed_count=len(result.proposed_actions), executed_count=executed_count,
            shadow_count=shadow_count, proposal_count=proposal_count, denied_count=denied_count,
            verified=result.verified,
        )
        append_jsonl(self.cycles_path, {
            "cycle_id": result.cycle_id, "ts": result.ts, "actor": result.actor,
            "cli_available": result.cli_available, "autonomy_enabled": result.autonomy_enabled,
            "observations": result.observations, "diagnosis": result.diagnosis,
            "proposed_actions": [vars(pa) for pa in result.proposed_actions],
            "outcomes": [vars(o) for o in result.outcomes],
            "verified": result.verified, "verify_detail": result.verify_detail,
        })


# ---------------------------------------------------------------------------
# Dashboard read accessor — the "mind-window" panel's data source.
# ---------------------------------------------------------------------------


def read_mind_window(
    *,
    cycles_path: Path = CYCLES_PATH,
    autonomy_path: Path = AUTONOMY_CONFIG_PATH,
    limit: int = 20,
) -> Dict[str, Any]:
    """Read-only snapshot for the AXIOM "mind-window" activity panel: recent
    cycles (noticed/believed/did-or-would-do) + any pending operator
    proposals (ESCALATION actions awaiting explicit approval). Honest empty
    state before the first cycle has ever run -- never fabricated.

    Each proposal is joined with its disposition (accepted/denied, if any --
    see src.agent_runtime.proposal_store), deduped by proposal_id so the SAME
    unresolved proposal re-surfacing across cycles shows once (most recent
    occurrence), not once per cycle. Already-decided proposals are split into
    ``resolved_proposals`` so accepting/denying one doesn't just hide it --
    the operator can still see what they decided and why."""
    from src.agent_runtime import proposal_store

    cycles = tail_jsonl(Path(cycles_path), limit)
    by_id: Dict[str, Dict[str, Any]] = {}
    for cycle in cycles:  # newest-first (tail_jsonl order) -- first write per id wins
        for outcome in cycle.get("outcomes", []) or []:
            if not outcome.get("proposal"):
                continue
            pid = outcome.get("proposal_id")
            key = pid or f"{cycle.get('cycle_id')}:{outcome.get('action')}"
            if key in by_id:
                continue
            by_id[key] = {"cycle_id": cycle.get("cycle_id"), "ts": cycle.get("ts"), **outcome}

    dispositions = proposal_store.list_dispositions()
    pending: List[Dict[str, Any]] = []
    resolved: List[Dict[str, Any]] = []
    for key, entry in by_id.items():
        disposition = dispositions.get(entry.get("proposal_id") or "")
        entry = {**entry, "disposition": disposition}
        (resolved if disposition else pending).append(entry)

    return {
        "has_run": bool(cycles),
        "autonomy_enabled": is_autonomy_enabled(Path(autonomy_path)),
        "recent_cycles": cycles,
        "last_cycle": cycles[0] if cycles else None,
        "pending_operator_proposals": pending,
        "resolved_proposals": resolved,
        "source": {
            "cycles": "trained_data/axiom/loop_cycles.jsonl",
            "autonomy_flag": "trained_data/axiom/loop_autonomy.json",
            "dispositions": "trained_data/axiom/proposal_dispositions.json",
        },
    }
