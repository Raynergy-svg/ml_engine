"""Claude CLI epoch runner for AXIOM's logical operator session."""
from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional

from src.axiom_operator.policy import ActionPolicy, PolicyDecision
from src.axiom_operator.effects import execute_safe_action
from src.axiom_operator.session import (
    DECISIONS_PATH,
    REPO_ROOT,
    SESSION_PATH,
    append_jsonl,
    load_session,
    save_session,
    utc_now,
)


SAFE_BUDDY_TOOLS = (
    "mcp__buddy__health_check",
    "mcp__buddy__get_system_status",
    "mcp__buddy__get_trade_feedback_log",
    "mcp__buddy__get_system_decisions",
    "mcp__buddy__get_agent_weights",
    "mcp__buddy__get_gate_health",
    "mcp__buddy__get_learnings",
    "mcp__buddy__trigger_diagnostics",
)

OPERATOR_BLOCK_RE = re.compile(r"<axiom-operator>(.*?)</axiom-operator>", re.DOTALL | re.IGNORECASE)
RATE_LIMIT_RE = re.compile(r"(usage\s+limit|rate\s*limit|credit\s+balance|hit\s+your\s+limit)", re.IGNORECASE)
RATE_LIMIT_EPOCH_RE = re.compile(r"\|\s*(\d{9,11})\b")


SubprocessRunner = Callable[..., subprocess.CompletedProcess[str]]


@dataclass(frozen=True)
class OperatorRunResult:
    ok: bool
    status: str
    epoch: int
    action: str
    policy: PolicyDecision
    session: Dict[str, Any]
    decision: Dict[str, Any]
    error: Optional[str] = None
    stdout_len: int = 0
    command: List[str] = field(default_factory=list)


class AxiomOperator:
    """Run one bounded Claude reasoning epoch and persist the result."""

    def __init__(
        self,
        *,
        project_root: Path = REPO_ROOT,
        session_path: Path = SESSION_PATH,
        decisions_path: Path = DECISIONS_PATH,
        claude_bin: str = "claude",
        subprocess_runner: Optional[SubprocessRunner] = None,
        policy: Optional[ActionPolicy] = None,
        effect_executor: Callable[[str, Mapping[str, Any], PolicyDecision], Dict[str, Any]] = execute_safe_action,
    ) -> None:
        self.project_root = Path(project_root)
        self.session_path = Path(session_path)
        self.decisions_path = Path(decisions_path)
        self.claude_bin = claude_bin
        self.subprocess_runner = subprocess_runner or subprocess.run
        self.policy = policy or ActionPolicy()
        self.effect_executor = effect_executor

    def run_once(
        self,
        *,
        observation: Optional[Mapping[str, Any]] = None,
        timeout_seconds: int = 120,
        allow_api_key: Optional[bool] = None,
    ) -> OperatorRunResult:
        session = load_session(self.session_path)
        session["status"] = "running"
        session["last_started_at"] = utc_now()
        session["api_key_refused"] = False
        save_session(session, self.session_path)

        if allow_api_key is None:
            allow_api_key = os.environ.get("AXIOM_OPERATOR_ALLOW_API_KEY", "").lower() in {"1", "true", "yes"}

        if os.environ.get("ANTHROPIC_API_KEY") and not allow_api_key:
            return self._held(
                session,
                action="none",
                reason="ANTHROPIC_API_KEY is set; refusing to avoid API billing instead of subscription auth",
                api_key_refused=True,
            )

        cli_path = shutil.which(self.claude_bin)
        if not cli_path:
            return self._held(session, action="none", reason="claude CLI not found on PATH", cli_available=False)
        session["cli_available"] = True

        prompt = self._build_prompt(session, observation or {})
        command = self._command(cli_path)
        started = time.time()
        stdout = ""
        stderr = ""
        returncode = -1
        prompt_file: Optional[str] = None
        try:
            with tempfile.NamedTemporaryFile("w", suffix=".md", prefix="axiom_operator_", delete=False, dir="/tmp") as fh:
                fh.write(prompt)
                prompt_file = fh.name
            with open(prompt_file, "r", encoding="utf-8") as stdin_fh:
                proc = self.subprocess_runner(
                    command,
                    stdin=stdin_fh,
                    capture_output=True,
                    text=True,
                    timeout=timeout_seconds,
                    cwd=str(self.project_root),
                    env=os.environ.copy(),
                )
            stdout = proc.stdout or ""
            stderr = proc.stderr or ""
            returncode = int(proc.returncode)
        except subprocess.TimeoutExpired:
            return self._held(session, action="none", reason=f"claude timed out after {timeout_seconds}s", cli_available=True)
        except OSError as exc:
            return self._held(session, action="none", reason=f"claude subprocess failed: {exc}", cli_available=True)
        finally:
            if prompt_file:
                try:
                    os.unlink(prompt_file)
                except OSError:
                    pass

        parsed = self._parse_stdout(stdout)
        action = str(parsed.get("action") or "observe").strip().lower()
        params = parsed.get("params") if isinstance(parsed.get("params"), dict) else {}
        policy_decision = self.policy.evaluate(action, params)
        context_used = len(prompt) + len(stdout) + len(stderr)
        budget = max(1, int(session.get("context_budget_chars") or 120000))
        used_pct = min(1.0, context_used / budget)
        threshold = float(session.get("rollover_threshold") or 0.8)
        rollover = used_pct >= threshold

        status = "held" if returncode != 0 or not policy_decision.allowed else "acted"
        if policy_decision.requires_human:
            status = "waiting"
        if rollover and status == "acted":
            status = "handoff"

        blocked_reasons = list(parsed.get("blocked_reasons") or [])
        if returncode != 0:
            blocked_reasons.append(self._stderr_reason(returncode, stdout, stderr))
        if not policy_decision.allowed:
            blocked_reasons.append(policy_decision.reason)
        reset_epoch = self._detect_ratelimit(stdout, stderr)
        if reset_epoch is not None:
            blocked_reasons.append(f"claude usage limit until {int(reset_epoch)}")
            status = "held"

        effect_result: Dict[str, Any] = {"applied": False, "reason": "not attempted"}
        if returncode == 0 and policy_decision.allowed:
            try:
                effect_result = self.effect_executor(action, params, policy_decision)
            except Exception as exc:  # noqa: BLE001 - effect boundary must return a held state, not a raw API 500
                effect_result = {
                    "applied": False,
                    "reason": f"effect_error: {type(exc).__name__}",
                    "error": str(exc),
                }
            if effect_result.get("applied") is False and action not in {"observe", "summarize", "none", ""}:
                blocked_reasons.append(str(effect_result.get("reason") or "safe effect not applied"))
                status = "held"

        epoch = int(session.get("epoch") or 0) + 1
        handoff = str(parsed.get("handoff_summary") or parsed.get("summary") or self._summarize_stdout(stdout))
        session.update({
            "status": status,
            "epoch": epoch,
            "context_used_chars": context_used,
            "context_used_pct": round(used_pct, 4),
            "handoff_summary": handoff[:2000],
            "open_incidents": self._list_of_strings(parsed.get("open_incidents")),
            "last_tools": self._list_of_strings(parsed.get("tools_called")),
            "blocked_reasons": blocked_reasons[-8:],
            "next_action": str(parsed.get("next_action") or ("wait_for_operator" if policy_decision.requires_human else "observe")),
            "last_action": action,
            "last_reasoning": str(parsed.get("reasoning") or parsed.get("summary") or self._summarize_stdout(stdout))[:2000],
            "last_error": "; ".join(blocked_reasons[-3:]) if blocked_reasons else None,
            "last_effect": effect_result,
            "last_completed_at": utc_now(),
            "cli_available": True,
            "mcp_config": "mcp.json" if (self.project_root / "mcp.json").exists() else None,
        })
        save_session(session, self.session_path)

        decision = {
            "ts": utc_now(),
            "epoch": epoch,
            "status": status,
            "action": action,
            "params": params,
            "policy": {
                "allowed": policy_decision.allowed,
                "tier": policy_decision.tier,
                "requires_human": policy_decision.requires_human,
                "reason": policy_decision.reason,
            },
            "tools_called": session["last_tools"],
            "blocked_reasons": blocked_reasons,
            "effect": effect_result,
            "context_used_chars": context_used,
            "context_used_pct": round(used_pct, 4),
            "rollover": rollover,
            "duration_seconds": round(time.time() - started, 2),
            "returncode": returncode,
            "handoff_summary": session["handoff_summary"],
            "next_action": session["next_action"],
        }
        append_jsonl(self.decisions_path, decision)
        ok = returncode == 0 and policy_decision.allowed and status not in {"held", "waiting"}
        return OperatorRunResult(
            ok=ok,
            status=status,
            epoch=epoch,
            action=action,
            policy=policy_decision,
            session=session,
            decision=decision,
            error=session["last_error"],
            stdout_len=len(stdout),
            command=command,
        )

    def _held(
        self,
        session: Dict[str, Any],
        *,
        action: str,
        reason: str,
        api_key_refused: bool = False,
        cli_available: Optional[bool] = None,
    ) -> OperatorRunResult:
        policy_decision = self.policy.evaluate(action)
        epoch = int(session.get("epoch") or 0) + 1
        session.update({
            "status": "held",
            "epoch": epoch,
            "blocked_reasons": [reason],
            "next_action": "wait_for_operator",
            "last_action": action,
            "last_reasoning": reason,
            "last_error": reason,
            "last_completed_at": utc_now(),
            "api_key_refused": api_key_refused,
        })
        if cli_available is not None:
            session["cli_available"] = cli_available
        save_session(session, self.session_path)
        decision = {
            "ts": utc_now(),
            "epoch": epoch,
            "status": "held",
            "action": action,
            "policy": {
                "allowed": False,
                "tier": "blocked",
                "requires_human": True,
                "reason": reason,
            },
            "tools_called": [],
            "blocked_reasons": [reason],
            "context_used_chars": 0,
            "context_used_pct": 0.0,
            "rollover": False,
            "handoff_summary": session.get("handoff_summary"),
            "next_action": "wait_for_operator",
        }
        append_jsonl(self.decisions_path, decision)
        return OperatorRunResult(
            ok=False,
            status="held",
            epoch=epoch,
            action=action,
            policy=policy_decision,
            session=session,
            decision=decision,
            error=reason,
        )

    def _command(self, cli_path: str) -> List[str]:
        cmd = [
            cli_path,
            "--print",
            "--no-session-persistence",
            "--permission-mode",
            "dontAsk",
            "--allowedTools",
            ",".join(SAFE_BUDDY_TOOLS),
            "--disallowedTools",
            "Bash,Edit,Write,NotebookEdit,WebFetch,WebSearch",
            "--name",
            "AXIOM Operator",
        ]
        mcp_config = self.project_root / "mcp.json"
        if mcp_config.exists():
            cmd.extend(["--mcp-config", str(mcp_config), "--strict-mcp-config"])
        return cmd

    def _build_prompt(self, session: Mapping[str, Any], observation: Mapping[str, Any]) -> str:
        payload = {
            "logical_session": {
                "session_id": session.get("session_id"),
                "epoch": session.get("epoch"),
                "handoff_summary": session.get("handoff_summary"),
                "open_incidents": session.get("open_incidents") or [],
                "last_tools": session.get("last_tools") or [],
                "blocked_reasons": session.get("blocked_reasons") or [],
                "next_action": session.get("next_action"),
            },
            "observation": observation,
            "allowed_actions": sorted(["diagnose", "recheck", "write_learning", "halt_lane", "stop_loop", "observe"]),
            "human_required_actions": sorted(["unhalt", "start_loop", "set_gross_leverage", "flatten", "promote_model", "promote_strategy", "promote_code", "disable_safety_gate"]),
        }
        return (
            "You are AXIOM Operator, a subscription-backed Claude Code reasoning epoch for a PRACTICE trading bot.\n"
            "Hard rules: never call broker code directly; never unhalt, start loops, raise leverage, flatten, promote models/code, or disable safety gates. "
            "Use only allowed Buddy MCP read/diagnostic tools if needed. Return concise structured output only.\n\n"
            "Output exactly one block:\n"
            "<axiom-operator>{\n"
            '  "summary": "short state",\n'
            '  "reasoning": "one or two sentences",\n'
            '  "action": "observe|diagnose|recheck|write_learning|halt_lane|stop_loop",\n'
            '  "params": {},\n'
            '  "tools_called": [],\n'
            '  "open_incidents": [],\n'
            '  "blocked_reasons": [],\n'
            '  "handoff_summary": "compact handoff for the next epoch",\n'
            '  "next_action": "next safe step or wait_for_operator"\n'
            "}</axiom-operator>\n\n"
            f"State JSON:\n{json.dumps(payload, indent=2, sort_keys=True, default=str)}\n"
        )

    def _parse_stdout(self, stdout: str) -> Dict[str, Any]:
        match = OPERATOR_BLOCK_RE.search(stdout or "")
        if not match:
            return {"summary": self._summarize_stdout(stdout), "action": "observe", "params": {}, "tools_called": []}
        raw = match.group(1).strip()
        try:
            payload = json.loads(raw)
        except ValueError:
            return {"summary": self._summarize_stdout(raw), "action": "observe", "params": {}, "tools_called": []}
        return payload if isinstance(payload, dict) else {"summary": self._summarize_stdout(stdout), "action": "observe"}

    def _summarize_stdout(self, stdout: str) -> str:
        text = " ".join((stdout or "").split())
        return (text[:400] if text else "No structured operator output.")

    def _list_of_strings(self, value: Any) -> List[str]:
        if not isinstance(value, list):
            return []
        return [str(item)[:200] for item in value[:12]]

    def _stderr_reason(self, returncode: int, stdout: str, stderr: str) -> str:
        out = " ".join((stdout or "").split())[-240:]
        err = " ".join((stderr or "").split())[-240:]
        parts = [f"claude exited {returncode}"]
        if out:
            parts.append(f"stdout={out}")
        if err:
            parts.append(f"stderr={err}")
        return " | ".join(parts)

    def _detect_ratelimit(self, stdout: str, stderr: str) -> Optional[float]:
        combined = f"{stdout or ''}\n{stderr or ''}"
        if not RATE_LIMIT_RE.search(combined):
            return None
        match = RATE_LIMIT_EPOCH_RE.search(combined)
        if not match:
            return time.time() + 3600
        try:
            return float(match.group(1))
        except ValueError:
            return time.time() + 3600
