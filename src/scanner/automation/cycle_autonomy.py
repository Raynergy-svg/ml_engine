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

# Consecutive-loss threshold: N back-to-back closed losing trades →
# fire deep losing-streak reflection. Critical: the user has trades
# executing, but they all lose. Different from REJECTION (where trades
# are blocked before execution).
DEFAULT_LOSING_STREAK = 3


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
        losing_streak_threshold: int = DEFAULT_LOSING_STREAK,
    ):
        self._brain = brain_callback
        self._periodic_every_n = periodic_every_n
        self._rejection_streak_threshold = rejection_streak_threshold
        self._losing_streak_threshold = losing_streak_threshold
        self._rejection_streak = 0
        self._last_periodic_cycle = -1
        # Track last DEGRADED/CRITICAL diagnosis so we don't spam Claude
        # every cycle on the same unresolved issue.
        self._last_heal_signature: Optional[str] = None
        # Track journal hash for losing-streak detection (changes when new
        # trade closes); avoid recomputing/re-firing on every cycle.
        self._last_journal_hash: Optional[str] = None
        self._last_losing_fire_signature: Optional[str] = None

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
        self._enable_losing = os.environ.get(
            "BUDDY_DISABLE_LOSING_STREAK_REFLECTION", ""
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
        # Autonomous retrainer: poll any in-flight retrain, then check if a
        # new one should spawn (only when freshness is CRITICAL by default).
        # Bypasses the broken Orchestrator → DriftRemediator chain that
        # never instantiated in production.
        try:
            from src.scanner.automation import autonomous_trainer as _at
            from src.scanner.automation.model_freshness import get_model_freshness

            _at.poll_completion(self._brain)
            if not _at.is_retrain_running():
                freshness = get_model_freshness()
                _at.maybe_spawn_autonomous_retrain(
                    freshness=freshness,
                    brain_callback=self._brain,
                )
        except Exception as _ar_err:
            logger.debug("autonomous_trainer cycle hook error: %s", str(_ar_err))

        # Update rejection streak tracker
        if tradeable_count > 0 and trades_executed == 0:
            self._rejection_streak += 1
        elif trades_executed > 0:
            self._rejection_streak = 0
        # (If tradeable_count == 0, streak unchanged — no rejection, just dry scan)

        # Decide priority. Only one reflection per cycle:
        #   self-heal > losing-streak > rejection > periodic
        # Self-heal wins because diagnostics aggregate everything; losing
        # streak is highest user-visible pain (real money losing); rejection
        # is medium; periodic is the steady-state heartbeat.
        if self._enable_self_heal and self._should_fire_self_heal():
            self._fire_self_heal(scan_count)
            return

        if self._enable_losing and self._should_fire_losing_streak():
            self._fire_losing_streak(scan_count)
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

    def _should_fire_losing_streak(self) -> bool:
        """Read trade_journal_rl.json for the last N closed trades. If the
        most recent N are all losses, fire deep reflection.

        Critical: this is the user's "I'm losing 6+ in a row" pain point.
        Different from rejection: trades ARE executing, but they all lose.
        Cause is usually stale models or regime shift the agents missed.
        """
        try:
            from pathlib import Path as _Path
            journal_path = _Path("trained_data/trade_journal_rl.json")
            if not journal_path.exists():
                return False
            data = json.loads(journal_path.read_text())
        except (json.JSONDecodeError, OSError) as e:
            logger.debug("losing_streak.read_failed", error=str(e))
            return False

        if not isinstance(data, list):
            return False

        # Look at closed trades only (outcome populated), most recent first
        closed = [e for e in data if isinstance(e, dict) and e.get("outcome")]
        if len(closed) < self._losing_streak_threshold:
            return False

        # Sort by close_time / timestamp descending so we get the latest N.
        # Note: trade_journal_rl.json schema has `outcome` as a STRING
        # ("win"/"loss") — NOT a dict. close_time lives at top level.
        # The earlier `entry.get("outcome", {}).get("close_time")` call
        # crashed on every closed-trade entry (real-world journal had
        # 16/17 string outcomes).
        def _ts(entry: Dict[str, Any]) -> str:
            return str(
                entry.get("close_time")
                or entry.get("timestamp")
                or ""
            )
        closed.sort(key=_ts, reverse=True)
        recent = closed[: self._losing_streak_threshold]

        def _is_loss(entry: Dict[str, Any]) -> bool:
            # Schema-aware: outcome may be a STRING ('win'/'loss'/'manual_close')
            # or a dict (legacy format with trade_won). Fall back to pnl_pips
            # at the top level (also legacy / external pipeline fixtures).
            outcome = entry.get("outcome")
            if isinstance(outcome, str):
                return outcome.strip().lower() == "loss"
            if isinstance(outcome, dict):
                if "trade_won" in outcome:
                    return not bool(outcome.get("trade_won"))
                try:
                    return float(outcome.get("pnl_pips", 0)) < 0
                except (TypeError, ValueError):
                    return False
            # Final fallback: pnl_pips at top level (test fixtures use this).
            try:
                return float(entry.get("pnl_pips", 0)) < 0
            except (TypeError, ValueError):
                return False

        if not all(_is_loss(e) for e in recent):
            return False

        # Dedup: don't re-fire on the same streak signature. Signature is
        # the trade IDs of the streak — once new trades close (winning or
        # losing) the signature changes.
        signature = "|".join(str(e.get("trade_id", "?")) for e in recent)
        if signature == self._last_losing_fire_signature:
            return False
        self._last_losing_fire_signature = signature
        # Stash for prompt
        self._pending_losing_streak = recent
        return True

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

        When the meta-cybernetic change pipeline is active, the autonomy
        trigger is rerouted into MetaManager.intake so the Claude spawn
        happens inside the Incident Analyst stage (with the rest of the
        9-stage pipeline downstream of it) rather than as a one-shot.
        """
        try:
            from src.scanner.automation.meta_manager import is_enabled as _meta_enabled, route_incident
            if _meta_enabled():
                routed = route_incident({
                    "kind": trade_id,  # one of "self_heal", "rejection", "losing_streak", etc.
                    "mode": mode,
                    "trigger_prompt_preview": prompt[:1500],
                    "diag": getattr(self, "_pending_diag", None),
                })
                if routed:
                    self._brain(f"[cyan]  ◆ autonomy trigger routed to meta-pipeline ({trade_id})[/]")
                    return
        except Exception as _e:
            logger.debug("cycle_autonomy.meta_route_failed err=%s", _e)

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

    def _fire_losing_streak(self, scan_count: int) -> None:
        """Last N closed trades all losses → spawn deep Claude reflection
        with full context (recent trades + model freshness + agent weights)."""
        recent = getattr(self, "_pending_losing_streak", []) or []
        prompt = _build_losing_streak_prompt(
            scan_count=scan_count, recent_losses=recent,
        )
        trade_id = f"LOSING-C{scan_count}"
        self._brain(
            f"[red]  ▸ LOSING STREAK reflection firing — last {len(recent)} trades all lost[/]"
        )
        # Deep mode: full MCP access, 7min timeout (CLAUDE.md Refinement Protocol
        # eats ~60-90s before the actual reflection starts)
        self._invoke(prompt, trade_id, mode="deep", timeout=420)

    def _fire_self_heal(self, scan_count: int) -> None:
        """PostTradeDiagnostics says DEGRADED/CRITICAL — spawn deep Claude."""
        diag = getattr(self, "_pending_diag", {}) or {}
        status = str(diag.get("status", "DEGRADED")).upper()
        prompt = _build_self_heal_prompt(scan_count=scan_count, diag=diag)
        trade_id = f"SELFHEAL-C{scan_count}"
        self._brain(
            f"[red]  ▸ Self-heal reflection firing — status={status}[/]"
        )
        self._invoke(prompt, trade_id, mode="deep", timeout=420)


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
    try:
        from src.scanner.automation.model_freshness import (
            format_freshness_for_prompt,
            get_model_freshness,
        )
        freshness_block = format_freshness_for_prompt(get_model_freshness())
    except Exception:
        freshness_block = "MODEL_FRESHNESS: (lookup failed)"

    return f"""Use the trade-reflection skill in DEEP mode — self-heal triggered.

TRIGGER: self_heal ({status})
TIMESTAMP: {ts}
CYCLE: #{scan_count}

{freshness_block}

DIAGNOSTIC REPORT:
  status: {status}
  issues:
{issue_lines or '    (none listed)'}
  recommended_actions:
{action_lines or '    (none listed)'}

This is a health degradation event. Your job:
  1. Check MODEL_FRESHNESS first — if any group is STALE (>14d) or
     CRITICAL (>30d), that may be the root cause. Recommend retraining
     in the learning entry.
  2. Review each issue against recent journal entries and gate health
  3. Write a learning entry explaining the ROOT CAUSE (not just the symptom)
  4. If safe, propose a config adjustment OR a proposed_weights delta to
     correct the imbalance
  5. If the issue is structural (code/config defect), write a rule draft
     describing the pattern so future occurrences get caught

Use MCP tools aggressively: get_agent_weights, get_gate_health,
get_learnings, get_closed_trades. Don't guess — ground every claim
in actual data.

End with <reflection-result>...</reflection-result> and <promise>.
"""


def _build_losing_streak_prompt(
    scan_count: int, recent_losses: List[Dict[str, Any]]
) -> str:
    """Deep-mode prompt for the highest-priority autonomous trigger.
    Includes model freshness front-and-center because stale models are
    the #1 cause of losing streaks.
    """
    ts = datetime.now(timezone.utc).isoformat()
    try:
        from src.scanner.automation.model_freshness import (
            format_freshness_for_prompt,
            get_model_freshness,
        )
        freshness_block = format_freshness_for_prompt(get_model_freshness())
    except Exception:
        freshness_block = "MODEL_FRESHNESS: (lookup failed)"

    # Compact one-line summary per losing trade. Schema-aware: outcome may
    # be a string in the production journal or a dict in test fixtures.
    loss_lines = []
    for t in recent_losses:
        outcome = t.get("outcome")
        if isinstance(outcome, dict):
            pnl = outcome.get("pnl_pips", t.get("pnl_pips", "?"))
            exit_r = outcome.get("exit_reason", t.get("close_reason", "?"))
        else:
            # outcome is a string ('win'/'loss'/...) — pull from top-level fields
            pnl = t.get("pnl_pips", "?")
            exit_r = t.get("close_reason", outcome if isinstance(outcome, str) else "?")
        regime_obj = t.get("regime")
        regime_v = (regime_obj.get("volatility_regime", "?") if isinstance(regime_obj, dict) else (regime_obj or "?"))
        loss_lines.append(
            f"  - {t.get('trade_id', '?')} {t.get('pair', '?')} "
            f"{t.get('direction', '?')} conf={t.get('confidence', '?')} "
            f"pnl_pips={pnl} exit={exit_r} regime={regime_v}"
        )

    return f"""You are Buddy's emergency reflection agent. The last {len(recent_losses)} \
closed trades were ALL LOSSES.

TRIGGER: losing_streak ({len(recent_losses)} consecutive losses)
TIMESTAMP: {ts}
CYCLE: #{scan_count}

{freshness_block}

RECENT_LOSSES (newest first):
{chr(10).join(loss_lines) if loss_lines else '  (no detail available)'}

YOUR JOB:
  1. Look at MODEL_FRESHNESS first. If oldest_age_days > 14, the most likely
     root cause is model staleness (regime drift since training). Write a
     learning entry naming the stale model + age + recommend retraining.
  2. Cross-reference the losses against agent_reasons in the journal — is
     one agent over-confident on bad signals? If so, propose a weight
     reduction in .claude/proposed_weights.json.
  3. Look at regime field across the losses — is buddy losing in a regime
     it wasn't trained for? (e.g., trained on NORMAL but market is now
     EXTREME). Note this.
  4. Look at exit_reason — are all losses hitting SL? If yes, SL might be
     too tight for current ATR. If they're hitting time-stop or partial
     close, that's a different problem.
  5. If safe, propose a short-term defensive config adjustment via
     .claude/config_adjustments.json (e.g., raise min_confidence,
     tighten max_uncertainty_score) until models can be retrained.

Use MCP tools aggressively (get_agent_weights, get_gate_health,
get_closed_trades, get_learnings). Don't guess — ground in data.

End with <reflection-result>...</reflection-result> and <promise>REFLECTION_COMPLETE</promise>.
"""


__all__ = [
    "CycleAutonomyTriggers",
    "DEFAULT_PERIODIC_EVERY_N_CYCLES",
    "DEFAULT_REJECTION_STREAK",
    "DEFAULT_LOSING_STREAK",
]
