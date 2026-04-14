"""Cycle-level autonomy triggers — fire Claude on scheduled events, not just
trade close.

The original ClaudeReflectionHandler only fires when
ExecutionManager.sync_closed_trades_rl finds a freshly closed trade. In
practice this means Claude stays dormant for long stretches — the user
sees "no autonomy, brainless script" whenever the market is quiet or
OANDA hasn't closed any trades.

This module adds three complementary triggers that run from the scan
cycle itself:

  1. PeriodicReflection — fires every N cycles regardless of trades
     ("what did buddy observe in the last hour?")
  2. SelfHealReflection — fires when PostTradeDiagnostics returns
     DEGRADED or CRITICAL ("something is off — Claude, diagnose + propose")
  3. RejectionReflection — fires when scan found setups but all were
     rejected ("why aren't we trading?")

All three:
  - Reuse invoke_claude_reflection (budget, lock, JSONL logging)
  - Honor the same daily budget cap + single-flight lock
  - Write to logs/reflection_log.jsonl so the TUI panel sees them
  - Can be disabled via env vars (BUDDY_DISABLE_PERIODIC_REFLECTION=1, etc.)
"""
from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import structlog

logger = structlog.get_logger(__name__)


# How often the periodic reflection fires (in scan cycles). With the
# default 5-min scan interval, every 12 cycles = once per hour.
DEFAULT_PERIODIC_EVERY_N_CYCLES = 12

# Consecutive-rejection threshold: N back-to-back cycles with zero
# executions despite having tradeable setups → fire rejection reflection.
DEFAULT_REJECTION_STREAK = 3


class CycleAutonomyTriggers:
    """Shared state for cycle-level Claude triggers.

    One instance lives on the EmbeddedScanner (or any scan loop) and is
    called once per cycle with the scan result + execution count. Each
    trigger independently decides whether to spawn Claude.
    """

    def __init__(
        self,
        brain_callback,  # Callable[[str], None] — TUI brain log emitter
        periodic_every_n: int = DEFAULT_PERIODIC_EVERY_N_CYCLES,
        rejection_streak_threshold: int = DEFAULT_REJECTION_STREAK,
    ):
        self._brain = brain_callback
        self._periodic_every_n = periodic_every_n
        self._rejection_streak_threshold = rejection_streak_threshold
        self._rejection_streak = 0
        self._last_periodic_cycle = -1
        # Track last DEGRADED/CRITICAL diagnosis so we don't spam Claude
        # every cycle on the same unresolved issue.
        self._last_heal_signature: Optional[str] = None

        # Feature flags (env-controlled for quick kill-switch)
        self._enable_periodic = os.environ.get(
            "BUDDY_DISABLE_PERIODIC_REFLECTION", ""
        ).lower() not in ("1", "true", "yes")
        self._enable_self_heal = os.environ.get(
            "BUDDY_DISABLE_SELF_HEAL_REFLECTION", ""
        ).lower() not in ("1", "true", "yes")
        self._enable_rejection = os.environ.get(
            "BUDDY_DISABLE_REJECTION_REFLECTION", ""
        ).lower() not in ("1", "true", "yes")

    # ── Trigger logic (called once per cycle) ──────────────────────────

    def on_cycle_complete(
        self,
        scan_count: int,
        scan_result: Any,
        trades_executed: int,
        tradeable_count: int,
    ) -> None:
        """Decide whether any of the three trigger types should fire.

        Called at the END of run_one_cycle, AFTER post-scan automation.
        """
        # Update rejection streak tracker
        if tradeable_count > 0 and trades_executed == 0:
            self._rejection_streak += 1
        elif trades_executed > 0:
            self._rejection_streak = 0
        # (If tradeable_count == 0, streak unchanged — no rejection, just dry scan)

        # Decide priority. Only one reflection per cycle — self-heal wins,
        # then rejection, then periodic. The single-flight lock inside
        # invoke_claude_reflection enforces this at the process level too.
        if self._enable_self_heal and self._should_fire_self_heal():
            self._fire_self_heal(scan_count)
            return

        if self._enable_rejection and self._should_fire_rejection():
            self._fire_rejection(scan_count, tradeable_count)
            return

        if self._enable_periodic and self._should_fire_periodic(scan_count):
            self._fire_periodic(scan_count, scan_result, trades_executed)

    # ── Decision predicates ────────────────────────────────────────────

    def _should_fire_periodic(self, scan_count: int) -> bool:
        if scan_count <= 0 or self._periodic_every_n <= 0:
            return False
        if scan_count == self._last_periodic_cycle:
            return False
        return scan_count % self._periodic_every_n == 0

    def _should_fire_rejection(self) -> bool:
        return self._rejection_streak >= self._rejection_streak_threshold

    def _should_fire_self_heal(self) -> bool:
        """Ask PostTradeDiagnostics for a health snapshot. Fire Claude only
        when status is DEGRADED or CRITICAL AND the signature changed (so we
        don't keep spawning Claude on the same unresolved issue).
        """
        try:
            from src.scanner.feedback.diagnostics import PostTradeDiagnostics

            diag = PostTradeDiagnostics().run()
        except Exception as e:
            logger.debug("self_heal_check_failed", error=str(e))
            return False

        status = str(diag.get("status", "HEALTHY")).upper()
        if status == "HEALTHY":
            # Reset signature so next degradation is allowed to fire again
            self._last_heal_signature = None
            return False

        # Signature = sorted list of (check, severity) so same issues
        # don't re-trigger until resolution or new issues appear.
        issues = diag.get("issues", []) or []
        sig_parts = sorted(
            f"{i.get('check', '?')}:{i.get('severity', '?')}" for i in issues
        )
        signature = "|".join(sig_parts)

        if signature == self._last_heal_signature:
            return False
        self._last_heal_signature = signature
        self._pending_diag = diag  # stash for prompt-building
        return True

    # ── Spawn helpers ──────────────────────────────────────────────────

    def _invoke(self, prompt: str, trade_id: str, mode: str, timeout: int) -> None:
        """Spawn Claude via the shared invoke_claude_reflection path.

        All constraints (single-flight lock, daily budget, logging) are
        enforced by invoke_claude_reflection itself. We just construct the
        prompt and call it.
        """
        try:
            from src.scanner.automation.claude_subprocess import (
                invoke_claude_reflection,
                ReflectionBudget,
                SingleFlightLock,
            )
        except ImportError as e:
            logger.warning("claude_subprocess_missing", error=str(e))
            return

        # Budget guard — same daily cap as the trade-close reflection
        budget = ReflectionBudget()
        if not budget.allows(mode):
            self._brain(f"[dim]  autonomy trigger skipped — budget exhausted ({mode})[/]")
            return

        # Single-flight — if another reflection is running (including the
        # trade-close one), skip rather than queue.
        lock = SingleFlightLock()
        if not lock.acquire():
            self._brain("[dim]  autonomy trigger skipped — reflection already in flight[/]")
            return

        try:
            result = invoke_claude_reflection(
                prompt=prompt,
                trade_id=trade_id,
                mode=mode,
                timeout_seconds=timeout,
                cwd=Path.cwd(),
            )
            try:
                budget.record(cost_usd=float(result.cost_usd or 0.0), mode=mode)
            except Exception:
                pass

            if result.success:
                hyp = (result.hypothesis or "")[:80]
                icon = "◆" if mode == "deep" else "▸"
                self._brain(
                    f"[magenta]  {icon} autonomy reflection complete — {hyp}[/]"
                )
            else:
                self._brain(f"[red]  ✗ autonomy reflection failed: {result.error}[/]")
        finally:
            lock.release()

    def _fire_periodic(
        self, scan_count: int, scan_result: Any, trades_executed: int
    ) -> None:
        """Every N cycles, ask Claude for a quick situational review."""
        self._last_periodic_cycle = scan_count
        tradeable_count = len(getattr(scan_result, "tradeable", []) or [])
        scanned_count = len(getattr(scan_result, "analyses", []) or [])
        prompt = _build_periodic_prompt(
            scan_count=scan_count,
            scanned_count=scanned_count,
            tradeable_count=tradeable_count,
            trades_executed=trades_executed,
        )
        trade_id = f"PERIODIC-C{scan_count}"
        self._brain(f"[cyan]  ▸ Periodic reflection firing (cycle #{scan_count})...[/]")
        # 180s — real Claude needs tool-call + analysis time for a periodic
        # review. 90s was too aggressive; saw timeouts even in calm states.
        self._invoke(prompt, trade_id, mode="lightweight", timeout=180)

    def _fire_rejection(self, scan_count: int, tradeable_count: int) -> None:
        """All tradeable setups rejected for N cycles in a row — diagnose."""
        streak = self._rejection_streak
        self._rejection_streak = 0  # Reset so we don't spam

        prompt = _build_rejection_prompt(
            scan_count=scan_count,
            streak=streak,
            tradeable_count=tradeable_count,
        )
        trade_id = f"REJECTION-C{scan_count}"
        self._brain(
            f"[yellow]  ▸ Rejection reflection firing "
            f"({streak} consecutive no-trade cycles)...[/]"
        )
        self._invoke(prompt, trade_id, mode="lightweight", timeout=240)

    def _fire_self_heal(self, scan_count: int) -> None:
        """PostTradeDiagnostics says DEGRADED/CRITICAL — spawn deep Claude."""
        diag = getattr(self, "_pending_diag", {}) or {}
        status = str(diag.get("status", "DEGRADED")).upper()
        prompt = _build_self_heal_prompt(scan_count=scan_count, diag=diag)
        trade_id = f"SELFHEAL-C{scan_count}"
        self._brain(
            f"[red]  ▸ Self-heal reflection firing — status={status}[/]"
        )
        self._invoke(prompt, trade_id, mode="deep", timeout=300)


# ── Prompt builders ────────────────────────────────────────────────────


def _build_periodic_prompt(
    scan_count: int,
    scanned_count: int,
    tradeable_count: int,
    trades_executed: int,
) -> str:
    ts = datetime.now(timezone.utc).isoformat()
    # Tight prompt — lightweight mode should be fast. NO tool calls, NO
    # multi-file reads, just ONE learning entry based on the current state
    # summary. Deep analysis is the job of the deep-mode self-heal path.
    return f"""You are Buddy's periodic reflection agent. A scan cycle completed.
Do a FAST situational check and write ONE learning entry. Do NOT use
MCP tools. Do NOT read many files. Just append one line.

STATE:
  timestamp: {ts}
  cycle: #{scan_count}
  scanned: {scanned_count} pairs
  tradeable: {tradeable_count}
  executed: {trades_executed}

ACTION (required):
  1. Append ONE line to .claude/learnings.md in this exact format:
     `- [YYYY-MM-DD] **PATTERN/<snake_case_key>**: <one-sentence observation>`
     If nothing interesting happened, write:
     `- [YYYY-MM-DD] **PATTERN/quiet_cycle_N{scan_count}**: scanned N pairs, M tradeable, K executed`

  2. End your response with EXACTLY:
     <reflection-result>
     artifacts_written:
       - .claude/learnings.md
     cost_usd: 0.01
     hypothesis: "<one-sentence takeaway>"
     </reflection-result>
     <promise>REFLECTION_COMPLETE</promise>

Keep total response under 300 tokens. No analysis prose. Just the append
and the result block.
"""


def _build_rejection_prompt(
    scan_count: int, streak: int, tradeable_count: int
) -> str:
    ts = datetime.now(timezone.utc).isoformat()
    return f"""Use the trade-reflection skill to diagnose REJECTION STREAK.

TRIGGER: rejection_streak
TIMESTAMP: {ts}
CYCLE: #{scan_count}
STREAK: {streak} consecutive cycles with tradeable setups found but ZERO executed

Something is blocking trades even though gates are passing. Common causes:
  - Correlation filter rejecting every signal
  - Policy engine in needs_confirmation mode
  - Portfolio optimizer in cold-start/observe mode for all pairs
  - Uncertainty penalty over-applied

Read the last scan's gate details, correlation state, and any policy
decisions logged in trained_data/policy_decisions.jsonl. Figure out which
filter is the bottleneck and write a learning entry identifying it. If
the cause is a config value that's too strict, propose an adjustment via
.claude/config_adjustments.json.

End with <reflection-result>...</reflection-result> and <promise>.
"""


def _build_self_heal_prompt(scan_count: int, diag: Dict[str, Any]) -> str:
    ts = datetime.now(timezone.utc).isoformat()
    status = diag.get("status", "DEGRADED")
    issues = diag.get("issues", []) or []
    actions = diag.get("recommended_actions", []) or []
    issue_lines = "\n".join(
        f"  - [{i.get('severity', '?')}] {i.get('check', '?')}: {i.get('detail', '')}"
        for i in issues
    )
    action_lines = "\n".join(f"  - {a}" for a in actions)
    return f"""Use the trade-reflection skill in DEEP mode — self-heal triggered.

TRIGGER: self_heal ({status})
TIMESTAMP: {ts}
CYCLE: #{scan_count}

DIAGNOSTIC REPORT:
  status: {status}
  issues:
{issue_lines or '    (none listed)'}
  recommended_actions:
{action_lines or '    (none listed)'}

This is a health degradation event. Your job:
  1. Review each issue against recent journal entries and gate health
  2. Write a learning entry explaining the ROOT CAUSE (not just the symptom)
  3. If safe, propose a config adjustment OR a proposed_weights delta to
     correct the imbalance
  4. If the issue is structural (code/config defect), write a rule draft
     describing the pattern so future occurrences get caught

Use MCP tools aggressively: get_agent_weights, get_gate_health,
get_learnings, get_closed_trades. Don't guess — ground every claim
in actual data.

End with <reflection-result>...</reflection-result> and <promise>.
"""


__all__ = [
    "CycleAutonomyTriggers",
    "DEFAULT_PERIODIC_EVERY_N_CYCLES",
    "DEFAULT_REJECTION_STREAK",
]
