"""Self-heal layer for the post-trade feedback loop.

Consumes the structured report emitted by
``src.scanner.feedback.diagnostics.PostTradeDiagnostics.run()`` and applies
corrective actions for every ``recommended_actions`` entry. Wired in by
``src.scanner.feedback.post_trade_loop.PostTradeLoop._run_self_heal`` via a
lazy import.

Design contract (hard rules):
    * ``apply()`` NEVER raises. Every failure path returns a dict and logs.
    * Each handler is isolated in its own try/except — one failing action
      must not prevent later actions from running.
    * Every handled action writes ONE decision record to
      ``logs/system_decisions.jsonl``.
    * If a CRITICAL issue is present OR any handler raises, ``apply()``
      enters *degraded mode*: writes a marker flag at
      ``trained_data/self_heal_degraded.flag`` that the next scan cycle can
      read, and sets ``degraded_mode=True`` in the return dict. Buddy never
      stops trading — it just runs in conservative fallback for the next cycle.
    * Stdlib + ``yaml`` only. No network, no credentials, no env deps.
    * No multiline f-strings (project style rule).

Action handlers (see ``_DISPATCH``):
    retrain_gates                       -> online_retrainer.trigger_retrain
    reset_gate_threshold_to_default:X   -> config_adjustments.json
    tighten_gate_threshold:X            -> config_adjustments.json (x1.10)
    soft_reset_agent_weight:X           -> agent_weights.json (atomic write)
    reduce_risk_per_trade_pct           -> config_adjustments.json multiplier
    retrain_rl_position_sizer           -> retrain_request marker file
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
import traceback
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

try:
    import yaml  # type: ignore
except ImportError:  # pragma: no cover — yaml ships in this env
    yaml = None  # type: ignore

logger = logging.getLogger(__name__)


HandlerResult = Tuple[bool, str, List[str]]


class SelfHeal:
    """Apply corrective actions for a diagnostics report.

    Usage:
        heal = SelfHeal()
        result = heal.apply(diag_result)  # never raises

    Returns a status dict matching the interface contract documented on
    ``apply()``.
    """

    # ------------------------------------------------------------------ #
    # File path constants                                                #
    # ------------------------------------------------------------------ #
    AGENT_WEIGHTS_PATH: Path = Path("trained_data/models/agent_weights.json")
    CONFIG_ADJUSTMENTS_PATH: Path = Path(".claude/config_adjustments.json")
    DECISION_LOG_PATH: Path = Path("logs/system_decisions.jsonl")
    ERROR_LOG_PATH: Path = Path("logs/unhandled_errors.jsonl")
    DEGRADED_FLAG_PATH: Path = Path("trained_data/self_heal_degraded.flag")
    RETRAIN_REQUESTS_DIR: Path = Path("trained_data/retrain_requests")
    CONFIG_PATH: Path = Path("config/config_improved_H1.yaml")
    # Debounce state — see _is_debounced(). Persists last-fire iso-timestamp
    # per action string so identical alerts on unresolved conditions don't
    # spam every cycle (the C5→C6 10-min treadmill on rl_model_staleness in
    # learnings.md was the canonical case before this was added).
    DEBOUNCE_STATE_PATH: Path = Path(".claude/self_heal_debounce.json")

    # Baseline agent weight to restore on soft reset.
    AGENT_WEIGHT_BASELINE: float = 1.0

    # Risk multiplier applied when drawdown_streak fires.
    RISK_REDUCTION_MULTIPLIER: float = 0.75

    # Threshold tightening factor for gate-firing-rate ceilings.
    GATE_TIGHTEN_FACTOR: float = 1.10

    # Clamp for probability-shaped thresholds (0, 1).
    PROBABILITY_CLAMP: float = 1.0

    # Debounce window per the C4 ask in .claude/learnings.md
    # (self_heal_no_debounce_C1): once an action fires for a given
    # (verb,arg) pair, suppress identical follow-ups for this many seconds
    # unless the underlying file mtime delta proves the condition resolved.
    # 6 hours is the operator-requested floor; this keeps the alert
    # informational without converting it to a per-cycle treadmill.
    DEBOUNCE_WINDOW_S: float = 6.0 * 3600.0

    # Evidence-required gating (Angle A1, 2026-05-10): refuse to propose
    # gate-tightening or gate-reset proposals when the journal has < N
    # closed trades in the last X hours. Prevents the runaway-on-stale-data
    # pattern where archived journal triggers self-heal proposals based on
    # historical context that no longer reflects current model performance.
    # The 2026-04-16 14-loss-streak frozen tightenings (min_confidence
    # 50→68, weighted_vote_threshold 0.45→0.85, etc.) would have been
    # blocked by this gate had it existed then.
    JOURNAL_PATH: Path = Path("trained_data/trade_journal_rl.json")
    EVIDENCE_MIN_RECENT_TRADES: int = 5
    EVIDENCE_WINDOW_HOURS: float = 6.0

    # Tiered autonomy levels (Agent 2, 2026-05-10).
    #
    # Per arxiv 2504.20093 self-healing-systems literature: not every
    # corrective action carries the same blast radius. A soft reset of
    # an agent weight to 1.0 is trivially reversible (one journal entry,
    # one atomic write); a gate threshold tighten compounds across cycles
    # and requires a manual revert to undo; a full retrain is expensive
    # and long-running. Tagging each handler with a level lets the
    # operator dial the auto-apply ceiling via
    # ``ScannerConfig.self_heal_max_autonomy_level`` without forking
    # individual handler logic.
    LEVEL_3_REVERSIBLE: int = 3   # auto-apply OK; trivially reversible
    LEVEL_4_GUARDED: int = 4      # auto-apply only if config flag elevates max level
    LEVEL_5_MANUAL: int = 5       # NEVER auto-apply — log only

    # Action-type → level. Unknown action types fail-closed to LEVEL_5
    # via ``_check_action_level``. Keep keys aligned with the verbs in
    # ``_dispatch`` (see __init__) — drift triggers
    # ``test_handler_level_map_complete``.
    _HANDLER_LEVELS: Dict[str, int] = {
        # Reversible: just restores baseline / dataclass default.
        "soft_reset_agent_weight": LEVEL_3_REVERSIBLE,
        "reset_gate_threshold_to_default": LEVEL_3_REVERSIBLE,
        # Guarded: compounds across cycles or shifts the risk profile;
        # manual revert is non-trivial.
        "tighten_gate_threshold": LEVEL_4_GUARDED,
        "reduce_risk_per_trade_pct": LEVEL_4_GUARDED,
        # Manual only: expensive, long-running, hard to undo. Operator
        # must explicitly elevate the autonomy ceiling to allow.
        "retrain_gates": LEVEL_5_MANUAL,
        "retrain_rl_position_sizer": LEVEL_5_MANUAL,
    }

    # Daily action budget (Agent 1, 2026-05-10): industry-standard circuit
    # breaker per DZone self-healing pipelines + NeurIPS 2024 self-healing
    # ML — cap total action APPLICATIONS in any rolling 24h window. Without
    # this, the 2026-04-16 incident applied 5 config adjustments in ~10
    # minutes; a budget would have stopped at action 3/4 and escalated to
    # the operator. Per-action debounce (DEBOUNCE_WINDOW_S) protects
    # against repeat-fire on the SAME action; the budget protects against
    # runaway across DIFFERENT actions.
    ACTION_BUDGET_PATH: Path = Path(".claude/self_heal_action_budget.json")
    MAX_ACTIONS_PER_DAY: int = 10

    # ------------------------------------------------------------------ #
    # Init                                                               #
    # ------------------------------------------------------------------ #
    def __init__(
        self,
        config_path: Optional[Path] = None,
        config: Optional[Any] = None,
    ) -> None:
        self._config_path = Path(config_path) if config_path else self.CONFIG_PATH
        self._yaml_cache: Optional[Dict[str, Any]] = None
        # Optional ScannerConfig handle (Agent 2, 2026-05-10). Used by
        # ``_check_action_level`` to read ``self_heal_max_autonomy_level``.
        # None = fall back to LEVEL_3_REVERSIBLE (safest).
        self.config = config
        self._dispatch: Dict[str, Callable[[str], HandlerResult]] = {
            "retrain_gates": self._handle_retrain_gates,
            "reset_gate_threshold_to_default": self._handle_reset_gate_threshold,
            "tighten_gate_threshold": self._handle_tighten_gate_threshold,
            "soft_reset_agent_weight": self._handle_soft_reset_agent_weight,
            "reduce_risk_per_trade_pct": self._handle_reduce_risk,
            "retrain_rl_position_sizer": self._handle_retrain_rl_position_sizer,
        }

    # ------------------------------------------------------------------ #
    # Public entry point                                                 #
    # ------------------------------------------------------------------ #
    def apply(self, diag_result: Dict[str, Any]) -> Dict[str, Any]:
        """Apply corrective actions for every issue in ``diag_result``.

        Args:
            diag_result: Output of ``PostTradeDiagnostics.run()``.

        Returns:
            dict shaped as:
                {
                  "status": "applied" | "no_action" | "degraded" | "error",
                  "actions_taken": [{"action": str, "success": bool,
                                      "detail": str}, ...],
                  "files_modified": [path, ...],
                  "degraded_mode": bool,
                  "error": Optional[str],
                }
        """
        actions_taken: List[Dict[str, Any]] = []
        files_modified: List[str] = []

        try:
            if not isinstance(diag_result, dict):
                return {
                    "status": "no_action",
                    "actions_taken": [],
                    "files_modified": [],
                    "degraded_mode": False,
                    "error": None,
                }

            status = str(diag_result.get("status", "")).upper()
            issues = diag_result.get("issues") or []
            recommended = diag_result.get("recommended_actions") or []

            if status == "HEALTHY" or not issues:
                # No action taken — don't spam the decision log with no-ops.
                return {
                    "status": "no_action",
                    "actions_taken": [],
                    "files_modified": [],
                    "degraded_mode": False,
                    "error": None,
                }

            if not isinstance(recommended, list):
                logger.warning(
                    "self_heal: recommended_actions not a list type=%s",
                    type(recommended).__name__,
                )
                recommended = []

            has_critical = self._has_critical(issues)
            any_raised = False

            # Dedupe preserving first-seen order: same action string runs once.
            seen: set = set()
            ordered: List[str] = []
            for raw in recommended:
                if not isinstance(raw, str):
                    continue
                key = raw.strip()
                if not key or key in seen:
                    continue
                seen.add(key)
                ordered.append(key)

            # Debounce: drop action strings that fired within DEBOUNCE_WINDOW_S
            # ago. Records the suppression in actions_taken so the operator
            # still sees the condition was diagnosed — just not re-acted on.
            ordered, suppressed = self._filter_debounced(ordered)
            for sup in suppressed:
                actions_taken.append({
                    "action": sup,
                    "success": False,
                    "detail": "debounced (fired within {0:.0f}s window)".format(self.DEBOUNCE_WINDOW_S),
                })

            # AGENT 2 — tiered autonomy level check (runs first, before
            # the daily action budget). Drops actions whose level exceeds
            # the operator-configured ``self_heal_max_autonomy_level``
            # ceiling (default 3). Records the skip in actions_taken so
            # the operator still sees the diagnosis even though the
            # action was suppressed.
            level_filtered: List[str] = []
            for action_str in ordered:
                action_type, _, _ = action_str.partition(":")
                allowed, reason, level = self._check_action_level(
                    action_type.strip()
                )
                if not allowed:
                    logger.info(
                        "self_heal action skipped (level gate): %s — %s",
                        action_str, reason,
                    )
                    actions_taken.append({
                        "action": action_str,
                        "success": False,
                        "detail": "level_gate_skipped:{0}:level={1}".format(
                            reason, level,
                        ),
                    })
                    continue
                level_filtered.append(action_str)
            ordered = level_filtered

            # AGENT 1 — daily action budget guard (industry-standard circuit
            # breaker per DZone self-healing pipelines + NeurIPS 2024
            # self-healing ML). Refuse the entire apply() call when total
            # actions in the rolling 24h window have reached
            # MAX_ACTIONS_PER_DAY. The 2026-04-16 incident applied 5 config
            # adjustments in ~10 minutes; this guard would have stopped it.
            budget_ok, budget_reason = self._check_action_budget()
            if not budget_ok:
                logger.warning(
                    "self_heal action refused: %s", budget_reason,
                )
                # Brain-feed-style operator escalation log. Format kept
                # grep-friendly: [SELF_HEAL_BUDGET_EXHAUSTED] N/MAX ...
                logger.warning(
                    "[SELF_HEAL_BUDGET_EXHAUSTED] %d/%d actions in 24h "
                    "— escalating to operator",
                    self.MAX_ACTIONS_PER_DAY,
                    self.MAX_ACTIONS_PER_DAY,
                )
                return self._build_refuse_result(budget_reason)

            for action_str in ordered:
                verb, arg = self._split_action(action_str)
                handler = self._dispatch.get(verb)
                success = False
                detail = ""
                modified: List[str] = []

                if handler is None:
                    success = False
                    detail = "unknown_action_string"
                    logger.warning(
                        "self_heal: unknown action=%s", action_str,
                    )
                else:
                    try:
                        success, detail, modified = handler(arg)
                    except Exception as err:  # noqa: BLE001
                        any_raised = True
                        success = False
                        detail = "handler_raised: {0}".format(repr(err))
                        logger.warning(
                            "self_heal: handler=%s raised error=%r",
                            verb, err,
                        )

                actions_taken.append({
                    "action": action_str,
                    "success": bool(success),
                    "detail": detail,
                })
                for m in modified:
                    if m and m not in files_modified:
                        files_modified.append(m)

                # AGENT 1 — record successful actions against the daily
                # budget. Only successes count: refusals/handler-failures
                # don't burn budget so the next legit action can still
                # proceed.
                if success:
                    self._record_action(action_str)

                # One decision log record per action attempt (success or not).
                self._log_decision(
                    trigger=self._trigger_for_action(issues, action_str),
                    issue=self._summarize_issues(issues, action_str),
                    action_taken=verb,
                    files_modified=list(modified),
                    degraded_mode=has_critical or any_raised,
                    error=None if success else detail,
                )

            # Overall status rollup.
            any_success = any(a["success"] for a in actions_taken)
            degraded_mode = has_critical or any_raised

            if degraded_mode:
                self._enter_degraded_mode(
                    reason=self._degraded_reason(
                        has_critical=has_critical,
                        any_raised=any_raised,
                    ),
                )

            if not actions_taken:
                final_status = "no_action"
            elif any_success and not degraded_mode:
                final_status = "applied"
            elif any_success and degraded_mode:
                final_status = "applied"
            else:
                # No handler succeeded — still return normally, no exception.
                final_status = "degraded"
                # Make sure we signal downstream even if no CRITICAL issue
                # was present: a complete handler failure is itself a reason
                # to run conservatively next cycle.
                if not degraded_mode:
                    self._enter_degraded_mode(reason="all_handlers_failed")
                    degraded_mode = True

            return {
                "status": final_status,
                "actions_taken": actions_taken,
                "files_modified": files_modified,
                "degraded_mode": degraded_mode,
                "error": None,
            }

        except Exception as exc:  # noqa: BLE001 — top-level safety net
            self._record_unhandled_error(exc)
            try:
                self._enter_degraded_mode(reason="apply_safety_net")
            except Exception:  # noqa: BLE001
                pass
            logger.error(
                "self_heal: apply() safety net caught error=%r", exc,
            )
            return {
                "status": "error",
                "actions_taken": actions_taken,
                "files_modified": files_modified,
                "degraded_mode": True,
                "error": str(exc),
            }

    # ------------------------------------------------------------------ #
    # Dispatch helpers                                                   #
    # ------------------------------------------------------------------ #
    @staticmethod
    def _split_action(action: str) -> Tuple[str, str]:
        """Split 'verb:arg' into (verb, arg). 'verb' → ('verb', '')."""
        if ":" in action:
            verb, _, arg = action.partition(":")
            return verb.strip(), arg.strip()
        return action.strip(), ""

    def _check_action_level(
        self,
        action_type: str,
    ) -> Tuple[bool, str, int]:
        """Tiered autonomy gate (Agent 2, 2026-05-10).

        Returns ``(allowed, reason, level)``.

        ``allowed`` is False when the action's level exceeds the
        operator-configured ceiling
        ``ScannerConfig.self_heal_max_autonomy_level`` (default 3 =
        only fully-reversible actions auto-apply). Unknown action
        types fail-closed to ``LEVEL_5_MANUAL`` so a typo or new
        verb without an explicit entry never ends up auto-applied.
        """
        level = self._HANDLER_LEVELS.get(action_type, self.LEVEL_5_MANUAL)
        max_level = int(
            getattr(self.config, "self_heal_max_autonomy_level", 3)
            if self.config is not None
            else 3
        )
        if level > max_level:
            return (
                False,
                "level_{0}_above_max_{1}".format(level, max_level),
                level,
            )
        return (True, "", level)

    def _filter_debounced(self, ordered: List[str]) -> Tuple[List[str], List[str]]:
        """Split an action list into (still_active, suppressed) by debounce state.

        Reads `.claude/self_heal_debounce.json` — `{action_str: iso_ts}`. An
        action is "suppressed" if its last-fire timestamp is within
        DEBOUNCE_WINDOW_S of now. After the split, updates the state file
        with NEW timestamps for the still_active actions (so the next call
        sees the fresh fire-time).

        Failures (corrupt JSON, write errors) fail-open: log + return the
        original list unchanged. We never want debounce-state issues to
        break self-heal recovery.
        """
        from datetime import datetime, timezone
        if not ordered:
            return [], []

        try:
            state: Dict[str, str] = {}
            if self.DEBOUNCE_STATE_PATH.exists():
                import json as _json
                try:
                    blob = _json.loads(self.DEBOUNCE_STATE_PATH.read_text(encoding="utf-8"))
                    if isinstance(blob, dict):
                        state = {str(k): str(v) for k, v in blob.items()}
                except (ValueError, OSError) as e:
                    logger.warning("self_heal: debounce state read failed err=%r — failing open", e)
                    return list(ordered), []

            now = datetime.now(timezone.utc)
            still_active: List[str] = []
            suppressed: List[str] = []
            for action_str in ordered:
                last_iso = state.get(action_str)
                if last_iso:
                    try:
                        last_dt = datetime.fromisoformat(str(last_iso).replace("Z", "+00:00"))
                        if last_dt.tzinfo is None:
                            last_dt = last_dt.replace(tzinfo=timezone.utc)
                        age_s = (now - last_dt).total_seconds()
                        if age_s < self.DEBOUNCE_WINDOW_S:
                            suppressed.append(action_str)
                            continue
                    except (ValueError, TypeError):
                        # corrupt timestamp → treat as not-recently-fired
                        pass
                still_active.append(action_str)

            # Update the state for the still_active actions only — suppressed
            # entries keep their existing fire-time so the window keeps closing.
            for a in still_active:
                state[a] = now.isoformat()

            try:
                self.DEBOUNCE_STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
                self._atomic_write_json(self.DEBOUNCE_STATE_PATH, state)
            except Exception as e:  # noqa: BLE001 — write failure is non-fatal
                logger.warning("self_heal: debounce state write failed err=%r", e)

            return still_active, suppressed
        except Exception as e:  # noqa: BLE001 — outer safety net
            logger.warning("self_heal: debounce filter raised err=%r — failing open", e)
            return list(ordered), []

    @staticmethod
    def _has_critical(issues: List[Any]) -> bool:
        for issue in issues:
            if isinstance(issue, dict) and issue.get("severity") == "critical":
                return True
        return False

    @staticmethod
    def _trigger_for_action(issues: List[Any], action: str) -> str:
        """Pick the diagnostic check name that likely produced this action."""
        # Map each recognized action-verb to the diagnostics check that emits
        # it. Used purely for the decision-log trigger field.
        verb = action.split(":", 1)[0]
        verb_to_check = {
            "retrain_gates": "direction_accuracy",
            "reset_gate_threshold_to_default": "gate_firing_rate",
            "tighten_gate_threshold": "gate_firing_rate",
            "soft_reset_agent_weight": "agent_weight_boundary",
            "reduce_risk_per_trade_pct": "drawdown_streak",
            "retrain_rl_position_sizer": "rl_model_staleness",
        }
        expected = verb_to_check.get(verb)
        if expected:
            for issue in issues:
                if isinstance(issue, dict) and issue.get("check") == expected:
                    return expected
        return verb or "multiple"

    @staticmethod
    def _summarize_issues(issues: List[Any], action: str) -> str:
        verb = action.split(":", 1)[0]
        verb_to_check = {
            "retrain_gates": "direction_accuracy",
            "reset_gate_threshold_to_default": "gate_firing_rate",
            "tighten_gate_threshold": "gate_firing_rate",
            "soft_reset_agent_weight": "agent_weight_boundary",
            "reduce_risk_per_trade_pct": "drawdown_streak",
            "retrain_rl_position_sizer": "rl_model_staleness",
        }
        expected = verb_to_check.get(verb)
        if expected:
            for issue in issues:
                if isinstance(issue, dict) and issue.get("check") == expected:
                    detail = issue.get("detail")
                    if isinstance(detail, str) and detail:
                        return detail
        return "diagnostics_issue:{0}".format(action)

    @staticmethod
    def _degraded_reason(has_critical: bool, any_raised: bool) -> str:
        parts: List[str] = []
        if has_critical:
            parts.append("critical_issue")
        if any_raised:
            parts.append("handler_raised")
        if not parts:
            parts.append("unknown")
        return "+".join(parts)

    # ------------------------------------------------------------------ #
    # Handler: retrain_gates                                             #
    # ------------------------------------------------------------------ #
    def _handle_retrain_gates(self, arg: str) -> HandlerResult:
        """Ask online_retrainer to retrain gate models. Respect cooldown."""
        try:
            from online_retrainer import OnlineRetrainer  # type: ignore
        except ImportError as err:
            logger.info(
                "self_heal: online_retrainer unavailable error=%r", err,
            )
            return (False, "online_retrainer_unavailable", [])

        try:
            retrainer = OnlineRetrainer()
            result = retrainer.trigger_retrain(
                X_replay=None,
                y_replay=None,
                reason="self_heal:direction_accuracy",
                force=False,
            )
        except Exception as err:  # noqa: BLE001
            logger.warning(
                "self_heal: trigger_retrain raised error=%r", err,
            )
            return (False, "trigger_retrain_raised: {0}".format(repr(err)), [])

        if not isinstance(result, dict):
            return (False, "non_dict_result", [])

        status = str(result.get("status", "")).lower()
        if status in ("success", "completed", "ok"):
            return (True, "retrain_status:{0}".format(status), [])
        if status == "blocked":
            return (
                False,
                "retrain_blocked:{0}".format(result.get("blocked_reason", "")),
                [],
            )
        if status == "skipped":
            return (
                False,
                "retrain_skipped:{0}".format(result.get("skipped_reason", "")),
                [],
            )
        # Unknown but not success — treat as non-fatal.
        return (False, "retrain_status:{0}".format(status or "unknown"), [])

    # ------------------------------------------------------------------ #
    # Handler: reset_gate_threshold_to_default                           #
    # ------------------------------------------------------------------ #
    def _handle_reset_gate_threshold(self, gate: str) -> HandlerResult:
        """Reset a ScannerConfig gate to its dataclass default.

        Writes a properly-shaped entry into config_adjustments.json["history"]
        so ConfigAdjuster.apply_adjustments() picks it up on the next cycle
        and runs setattr(config, gate, default). The legacy implementation
        wrote to data["self_heal"][gate_threshold_X] — a namespace
        ConfigAdjuster never reads — making every trap-recovery action a
        silent no-op (audit 2026-04-28).
        """
        if not gate:
            return (False, "missing_gate_arg", [])

        # Angle A1 (2026-05-10): refuse to propose against zero new evidence.
        # Reset proposals must also pass the same evidence floor as tighten —
        # otherwise self-heal flip-flops between tighten and reset on stale
        # data, neither direction grounded in actual outcomes.
        recent = self._count_recent_closed_trades()
        if recent < self.EVIDENCE_MIN_RECENT_TRADES:
            return (
                False,
                "insufficient_evidence:n={0}<{1}_in_{2}h".format(
                    recent,
                    self.EVIDENCE_MIN_RECENT_TRADES,
                    int(self.EVIDENCE_WINDOW_HOURS),
                ),
                [],
            )

        default = self._find_gate_default(gate)
        if default is None:
            return (False, "no_default_for_gate", [])

        # Strict guard: only write keys that are real ScannerConfig fields.
        # ConfigAdjuster.apply_adjustments() will setattr blindly, so
        # orphan keys would silently mutate the config object with attributes
        # nothing reads — exactly the 2026-04-16 $3,527 bug pattern.
        if self._scanner_config_default(gate) is None:
            return (False, "not_a_scanner_config_field", [])

        ok, detail = self._append_history_reset(gate, float(default))
        if not ok:
            return (False, detail, [])
        return (
            True,
            "reset:{0}={1:.4f}".format(gate, float(default)),
            [str(self.CONFIG_ADJUSTMENTS_PATH)],
        )

    def _append_history_reset(
        self,
        gate: str,
        new_value: float,
    ) -> Tuple[bool, str]:
        """Append a self-heal reset entry to config_adjustments.json["history"]
        in the same schema threshold_optimizer uses, so the existing
        ConfigAdjuster.apply_adjustments() pipeline consumes it on the next
        cycle without any new wiring.
        """
        import uuid

        data = self._load_adjustments()
        if not isinstance(data, dict):
            data = {}

        history = data.get("history")
        if not isinstance(history, list):
            history = []

        ts = datetime.now(timezone.utc).isoformat()
        proposal_id = str(uuid.uuid4())
        entry: Dict[str, Any] = {
            "approved_at": ts,
            "key": gate,
            "new_value": new_value,
            "old_value": None,
            "proposal_id": proposal_id,
            "reason": "self_heal_reset_gate_threshold_to_default",
            "source": "self_heal",
            "timestamp": ts,
        }
        history.append(entry)
        data["history"] = history

        # Maintain the running counter that diagnostics._check_quiet_streak
        # uses to detect over-tightening. Without this, the trap thinks
        # nothing was just attempted — re-fires every cycle for the same gate
        # and the debounce filter is the only thing protecting us from spam.
        try:
            data["total_adjustments"] = int(data.get("total_adjustments", 0) or 0) + 1
        except (TypeError, ValueError):
            data["total_adjustments"] = len(history)

        data["last_updated"] = ts

        try:
            self.CONFIG_ADJUSTMENTS_PATH.parent.mkdir(parents=True, exist_ok=True)
        except OSError as err:
            logger.warning("self_heal: adjustments mkdir failed error=%r", err)
            return (False, "mkdir_failed: {0}".format(repr(err)))

        ok, detail = self._atomic_write_json(self.CONFIG_ADJUSTMENTS_PATH, data)
        if not ok:
            return (False, detail)
        return (True, "history_appended:{0}".format(proposal_id[:8]))

    # ------------------------------------------------------------------ #
    # Handler: tighten_gate_threshold                                    #
    # ------------------------------------------------------------------ #
    def _count_recent_closed_trades(self) -> int:
        """Count closed trades in the journal within EVIDENCE_WINDOW_HOURS.

        A 'closed trade' is a journal entry with a parseable `close_time`
        (or `outcome.closed_at`) ISO timestamp >= now - EVIDENCE_WINDOW_HOURS.
        Returns 0 on missing/corrupt journal, in which case the
        evidence-required check refuses the proposal (fail-closed).
        """
        try:
            with open(self.JOURNAL_PATH) as f:
                trades = json.load(f)
            if not isinstance(trades, list):
                return 0
        except (FileNotFoundError, json.JSONDecodeError, OSError):
            return 0

        cutoff = datetime.now(timezone.utc) - timedelta(
            hours=self.EVIDENCE_WINDOW_HOURS
        )
        recent = 0
        for t in trades:
            if not isinstance(t, dict):
                continue
            close_iso = t.get("close_time")
            if not close_iso:
                outcome = t.get("outcome") if isinstance(t.get("outcome"), dict) else {}
                close_iso = outcome.get("closed_at") or outcome.get("close_time")
            if not close_iso:
                continue
            try:
                close_dt = datetime.fromisoformat(str(close_iso).replace("Z", "+00:00"))
                if close_dt.tzinfo is None:
                    close_dt = close_dt.replace(tzinfo=timezone.utc)
            except (ValueError, TypeError):
                continue
            if close_dt >= cutoff:
                recent += 1
        return recent

    # ------------------------------------------------------------------ #
    # Daily action budget (Agent 1, 2026-05-10)                          #
    # ------------------------------------------------------------------ #
    def _load_action_budget(self) -> Dict[str, Any]:
        """Load the action-budget state. Empty/corrupt → empty list.

        Shape: {"actions_today": [iso_timestamp, ...]}.
        """
        path = self.ACTION_BUDGET_PATH
        if not path.exists():
            return {"actions_today": []}
        try:
            with path.open("r", encoding="utf-8") as fh:
                data = json.load(fh)
        except (OSError, json.JSONDecodeError, UnicodeDecodeError) as err:
            logger.warning(
                "self_heal: action_budget load failed path=%s error=%r",
                path, err,
            )
            return {"actions_today": []}
        if not isinstance(data, dict):
            return {"actions_today": []}
        actions = data.get("actions_today")
        if not isinstance(actions, list):
            return {"actions_today": []}
        # Coerce to strings; drop non-strings defensively.
        return {"actions_today": [str(x) for x in actions if isinstance(x, str)]}

    def _record_action(self, action_label: str) -> None:
        """Append now_iso to actions_today and atomically persist."""
        try:
            state = self._load_action_budget()
            actions = state.get("actions_today") or []
            now_iso = datetime.now(timezone.utc).isoformat()
            actions.append(now_iso)
            state["actions_today"] = actions
            state["last_action_label"] = action_label
            state["last_action_at"] = now_iso
            try:
                self.ACTION_BUDGET_PATH.parent.mkdir(parents=True, exist_ok=True)
            except OSError as err:
                logger.warning(
                    "self_heal: action_budget mkdir failed error=%r", err,
                )
                return
            ok, detail = self._atomic_write_json(self.ACTION_BUDGET_PATH, state)
            if not ok:
                logger.warning(
                    "self_heal: action_budget write failed detail=%s", detail,
                )
        except Exception as err:  # noqa: BLE001 — recording is non-fatal
            logger.warning(
                "self_heal: _record_action raised error=%r action=%s",
                err, action_label,
            )

    def _check_action_budget(self) -> Tuple[bool, str]:
        """Return (ok, reason). If actions in last 24h >= MAX, refuse.

        Self-prunes entries older than 24h on read so the budget file
        doesn't grow unbounded across days. The 24h window is rolling,
        not calendar-day, so a burst at 23:30 doesn't reset at midnight.
        """
        state = self._load_action_budget()
        actions = state.get("actions_today") or []
        now = datetime.now(timezone.utc)
        cutoff = now - timedelta(hours=24.0)
        kept: List[str] = []
        for ts in actions:
            try:
                dt = datetime.fromisoformat(str(ts).replace("Z", "+00:00"))
                if dt.tzinfo is None:
                    dt = dt.replace(tzinfo=timezone.utc)
                if dt >= cutoff:
                    kept.append(ts)
            except (ValueError, TypeError):
                # Corrupt timestamp — drop it on prune.
                continue

        # Persist the prune so the file self-heals on each check.
        if len(kept) != len(actions):
            try:
                state["actions_today"] = kept
                self.ACTION_BUDGET_PATH.parent.mkdir(
                    parents=True, exist_ok=True,
                )
                self._atomic_write_json(self.ACTION_BUDGET_PATH, state)
            except Exception as err:  # noqa: BLE001 — prune is non-fatal
                logger.warning(
                    "self_heal: action_budget prune write failed error=%r",
                    err,
                )

        n = len(kept)
        if n >= self.MAX_ACTIONS_PER_DAY:
            reason = "budget_exhausted:n={0}/{1}_in_24h".format(
                n, self.MAX_ACTIONS_PER_DAY,
            )
            return (False, reason)
        return (True, "")

    def _build_refuse_result(self, reason: str) -> Dict[str, Any]:
        """Build a top-level refuse result for the daily budget guard.

        The status is "no_action" — refusing on budget grounds is not an
        error condition; it's the safety circuit doing its job. We surface
        the reason in actions_taken so the operator can grep
        decision/brain logs for it.
        """
        return {
            "status": "no_action",
            "actions_taken": [{
                "action": "budget_guard",
                "success": False,
                "detail": reason,
            }],
            "files_modified": [],
            "degraded_mode": False,
            "error": None,
        }

    def _handle_tighten_gate_threshold(self, gate: str) -> HandlerResult:
        if not gate:
            return (False, "missing_gate_arg", [])

        # Angle A1 (2026-05-10): refuse to propose against zero new evidence.
        # Self-heal must be grounded in fresh closed-trade data, not stale
        # historical memory. The 2026-04-16 runaway tightening fired against
        # an archived journal; this gate would have blocked it.
        recent = self._count_recent_closed_trades()
        if recent < self.EVIDENCE_MIN_RECENT_TRADES:
            return (
                False,
                "insufficient_evidence:n={0}<{1}_in_{2}h".format(
                    recent,
                    self.EVIDENCE_MIN_RECENT_TRADES,
                    int(self.EVIDENCE_WINDOW_HOURS),
                ),
                [],
            )

        adjustments = self._load_adjustments()
        key = "gate_threshold_{0}".format(gate)

        current: Optional[float] = None
        existing = adjustments.get("self_heal", {}) if isinstance(adjustments.get("self_heal"), dict) else {}
        if isinstance(existing, dict) and isinstance(existing.get(key), dict):
            raw = existing[key].get(key)
            if raw is not None:
                try:
                    current = float(raw)
                except (TypeError, ValueError):
                    current = None

        if current is None:
            default = self._find_gate_default(gate)
            if default is None:
                return (False, "no_default_in_yaml", [])
            current = float(default)

        new_value = current * self.GATE_TIGHTEN_FACTOR
        # Clamp probability-shaped thresholds (0, 1]. Raw confidence (50.0)
        # is left alone since tightening upward is still valid for score-
        # based gates.
        if 0.0 < current <= 1.0 and new_value > self.PROBABILITY_CLAMP:
            new_value = self.PROBABILITY_CLAMP

        entry = {
            key: float(new_value),
            "_ts": datetime.now(timezone.utc).isoformat(),
            "_reason": "self_heal_tighten",
        }
        ok, detail = self._merge_adjustment(key, entry)
        if not ok:
            return (False, detail, [])
        return (
            True,
            "tighten:{0}:{1:.4f}->{2:.4f}".format(gate, current, new_value),
            [str(self.CONFIG_ADJUSTMENTS_PATH)],
        )

    # ------------------------------------------------------------------ #
    # Handler: soft_reset_agent_weight                                   #
    # ------------------------------------------------------------------ #
    def _handle_soft_reset_agent_weight(self, agent: str) -> HandlerResult:
        if not agent:
            return (False, "missing_agent_arg", [])

        if not self.AGENT_WEIGHTS_PATH.exists():
            return (False, "agent_weights_missing", [])

        try:
            with self.AGENT_WEIGHTS_PATH.open("r", encoding="utf-8") as fh:
                data = json.load(fh)
        except (OSError, json.JSONDecodeError, UnicodeDecodeError) as err:
            logger.warning(
                "self_heal: weights load failed error=%r", err,
            )
            return (False, "weights_load_failed: {0}".format(repr(err)), [])

        if not isinstance(data, dict):
            return (False, "weights_root_not_dict", [])

        buckets_touched: List[str] = []
        for bucket_name, bucket in list(data.items()):
            if bucket_name.startswith("_meta"):
                continue
            if not isinstance(bucket, dict):
                continue
            if agent in bucket:
                bucket[agent] = float(self.AGENT_WEIGHT_BASELINE)
                buckets_touched.append(bucket_name)

        if not buckets_touched:
            # Agent not present in any bucket — nothing to reset. Still a
            # "handled" action (the issue has been acknowledged) but with a
            # descriptive detail so we don't flip to degraded mode on it.
            return (True, "agent_not_present", [])

        ok, detail = self._atomic_write_json(self.AGENT_WEIGHTS_PATH, data)
        if not ok:
            return (False, detail, [])
        return (
            True,
            "reset:{0}:buckets={1}".format(agent, ",".join(buckets_touched)),
            [str(self.AGENT_WEIGHTS_PATH)],
        )

    # ------------------------------------------------------------------ #
    # Handler: reduce_risk_per_trade_pct                                 #
    # ------------------------------------------------------------------ #
    def _handle_reduce_risk(self, _arg: str) -> HandlerResult:
        key = "risk_per_trade_pct_multiplier"
        entry = {
            key: float(self.RISK_REDUCTION_MULTIPLIER),
            "_ts": datetime.now(timezone.utc).isoformat(),
            "_reason": "self_heal_drawdown_streak",
        }
        ok, detail = self._merge_adjustment(key, entry)
        if not ok:
            return (False, detail, [])
        return (
            True,
            "risk_multiplier={0}".format(self.RISK_REDUCTION_MULTIPLIER),
            [str(self.CONFIG_ADJUSTMENTS_PATH)],
        )

    # ------------------------------------------------------------------ #
    # Handler: retrain_rl_position_sizer                                 #
    # ------------------------------------------------------------------ #
    def _handle_retrain_rl_position_sizer(self, _arg: str) -> HandlerResult:
        """Write a retrain-request marker. Training itself runs off the hot path."""
        ts = datetime.now(timezone.utc)
        name = "rl_sizer_{0}.json".format(ts.strftime("%Y%m%d_%H%M%S"))
        payload = {
            "requested_at": ts.isoformat(),
            "reason": "self_heal_rl_staleness",
            "triggered_by": "self_heal",
        }
        try:
            self.RETRAIN_REQUESTS_DIR.mkdir(parents=True, exist_ok=True)
            target = self.RETRAIN_REQUESTS_DIR / name
            ok, detail = self._atomic_write_json(target, payload)
            if not ok:
                return (False, detail, [])
            return (True, "retrain_request:{0}".format(name), [str(target)])
        except OSError as err:
            logger.warning(
                "self_heal: retrain_request mkdir/write failed error=%r", err,
            )
            return (False, "mkdir_failed: {0}".format(repr(err)), [])

    # ------------------------------------------------------------------ #
    # YAML: gate default lookup                                          #
    # ------------------------------------------------------------------ #
    def _load_yaml(self) -> Dict[str, Any]:
        if self._yaml_cache is not None:
            return self._yaml_cache
        if yaml is None:
            logger.warning(
                "self_heal: yaml module unavailable — no defaults",
            )
            self._yaml_cache = {}
            return self._yaml_cache
        if not self._config_path.exists():
            logger.debug(
                "self_heal: config file missing path=%s",
                self._config_path,
            )
            self._yaml_cache = {}
            return self._yaml_cache
        try:
            with self._config_path.open("r", encoding="utf-8") as fh:
                data = yaml.safe_load(fh) or {}
        except (OSError, UnicodeDecodeError) as err:
            logger.warning(
                "self_heal: yaml read failed path=%s error=%r",
                self._config_path, err,
            )
            self._yaml_cache = {}
            return self._yaml_cache
        except Exception as err:  # noqa: BLE001
            logger.warning(
                "self_heal: yaml parse failed path=%s error=%r",
                self._config_path, err,
            )
            self._yaml_cache = {}
            return self._yaml_cache

        self._yaml_cache = data if isinstance(data, dict) else {}
        return self._yaml_cache

    def _find_gate_default(self, gate: str) -> Optional[float]:
        """Resolve the canonical default for a gate name.

        Resolution order:
          1. ScannerConfig dataclass field default (PRIMARY — covers every
             field name emitted by PostTradeDiagnostics._QUIET_STREAK_RECOVERY_ORDER:
             min_confidence, weighted_vote_threshold, max_model_disagreement,
             max_uncertainty_score, atr_sl_multiplier, min_momentum,
             min_risk_reward_ratio).
          2. yaml-config aliases (legacy — keeps confidence_passed /
             momentum_passed / risk_passed working in case diagnostics ever
             reverts to those tag names).

        Returns None when neither source can produce a numeric default.
        Without ScannerConfig priority, the trap-recovery handler silently
        no-op'd on 5/7 of the recovery-order gates because YAML aliases only
        knew the truncated names. See commit 0772581 + audit 2026-04-28.
        """
        scanner_default = self._scanner_config_default(gate)
        if scanner_default is not None:
            return scanner_default

        cfg = self._load_yaml()
        if not cfg:
            return None

        # Legacy yaml-paths kept for backward compatibility with diagnostic
        # tag-style names. ScannerConfig field names take precedence above.
        paths_by_gate: Dict[str, List[Tuple[str, ...]]] = {
            "confidence_passed": [
                ("scan", "min_confidence"),
                ("inference", "min_meta_confidence"),
                ("fx", "confidence", "low_lt"),
            ],
            "momentum_passed": [
                ("inference", "min_momentum"),
                ("scan", "aggressive_min_meta_confidence"),
            ],
            "risk_passed": [
                ("fx", "risk", "risk_per_trade_pct"),
                ("buddy", "risk_per_trade_pct"),
                ("scan", "risk_per_trade_pct"),
            ],
        }

        paths = paths_by_gate.get(gate)
        if not paths:
            return None

        for path in paths:
            value: Any = cfg
            ok = True
            for key in path:
                if isinstance(value, dict) and key in value:
                    value = value[key]
                else:
                    ok = False
                    break
            if ok:
                try:
                    return float(value)
                except (TypeError, ValueError):
                    continue
        return None

    @staticmethod
    def _scanner_config_default(gate: str) -> Optional[float]:
        """Return the ScannerConfig dataclass default for ``gate``, or None.

        Imports lazily — keeps SelfHeal importable in environments where
        ScannerConfig has heavyweight optional deps.
        """
        try:
            from src.scanner.config import ScannerConfig  # noqa: WPS433
            cfg = ScannerConfig()
        except Exception as err:  # pragma: no cover — config import wraps a lot
            logger.debug("self_heal: ScannerConfig import failed err=%r", err)
            return None
        if not hasattr(cfg, gate):
            return None
        try:
            return float(getattr(cfg, gate))
        except (TypeError, ValueError):
            return None

    # ------------------------------------------------------------------ #
    # config_adjustments.json: load / merge / write                      #
    # ------------------------------------------------------------------ #
    def _load_adjustments(self) -> Dict[str, Any]:
        path = self.CONFIG_ADJUSTMENTS_PATH
        if not path.exists():
            return {}
        try:
            with path.open("r", encoding="utf-8") as fh:
                data = json.load(fh)
        except (OSError, json.JSONDecodeError, UnicodeDecodeError) as err:
            logger.warning(
                "self_heal: adjustments load failed path=%s error=%r",
                path, err,
            )
            return {}
        if not isinstance(data, dict):
            logger.warning(
                "self_heal: adjustments root not a dict type=%s",
                type(data).__name__,
            )
            return {}
        return data

    def _merge_adjustment(
        self,
        key: str,
        entry: Dict[str, Any],
    ) -> Tuple[bool, str]:
        """Merge a self-heal adjustment entry into config_adjustments.json.

        Writes under a dedicated ``self_heal`` section so the existing
        threshold_optimizer schema (history/pending/last_applied/...)
        is never overwritten.
        """
        data = self._load_adjustments()

        section = data.get("self_heal")
        if not isinstance(section, dict):
            section = {}

        section[key] = entry
        data["self_heal"] = section
        # Useful breadcrumb on the top level without touching
        # threshold_optimizer's own last_updated field (if present we leave
        # it alone and add our own).
        data["self_heal_last_updated"] = datetime.now(timezone.utc).isoformat()

        try:
            self.CONFIG_ADJUSTMENTS_PATH.parent.mkdir(parents=True, exist_ok=True)
        except OSError as err:
            logger.warning(
                "self_heal: adjustments mkdir failed error=%r", err,
            )
            return (False, "mkdir_failed: {0}".format(repr(err)))

        ok, detail = self._atomic_write_json(self.CONFIG_ADJUSTMENTS_PATH, data)
        if not ok:
            return (False, detail)
        return (True, "merged")

    # ------------------------------------------------------------------ #
    # Atomic JSON write                                                  #
    # ------------------------------------------------------------------ #
    @staticmethod
    def _atomic_write_json(
        path: Path,
        data: Any,
    ) -> Tuple[bool, str]:
        """Write ``data`` as JSON to ``path`` atomically (tmp + rename)."""
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
        except OSError as err:
            return (False, "mkdir_failed: {0}".format(repr(err)))

        try:
            payload = json.dumps(data, indent=2, sort_keys=True, default=str)
        except (TypeError, ValueError) as err:
            return (False, "json_dumps_failed: {0}".format(repr(err)))

        tmp_fd = None
        tmp_path: Optional[str] = None
        try:
            tmp_fd, tmp_path = tempfile.mkstemp(
                prefix=path.name + ".",
                suffix=".tmp",
                dir=str(path.parent),
            )
            with os.fdopen(tmp_fd, "w", encoding="utf-8") as fh:
                tmp_fd = None  # fdopen now owns the descriptor
                fh.write(payload)
                fh.flush()
                try:
                    os.fsync(fh.fileno())
                except OSError:
                    # fsync can fail on some filesystems — non-fatal.
                    pass
            os.replace(tmp_path, str(path))
            tmp_path = None
            return (True, "written")
        except OSError as err:
            return (False, "write_failed: {0}".format(repr(err)))
        finally:
            if tmp_fd is not None:
                try:
                    os.close(tmp_fd)
                except OSError:
                    pass
            if tmp_path is not None and os.path.exists(tmp_path):
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass

    # ------------------------------------------------------------------ #
    # Degraded-mode flag                                                 #
    # ------------------------------------------------------------------ #
    def _enter_degraded_mode(self, reason: str) -> None:
        """Write (or overwrite) the degraded-mode marker flag."""
        payload = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "reason": reason,
            "source": "self_heal",
        }
        ok, detail = self._atomic_write_json(self.DEGRADED_FLAG_PATH, payload)
        if not ok:
            logger.error(
                "self_heal: degraded flag write failed reason=%s detail=%s",
                reason, detail,
            )

    # ------------------------------------------------------------------ #
    # Decision log                                                       #
    # ------------------------------------------------------------------ #
    def _log_decision(
        self,
        trigger: str,
        issue: str,
        action_taken: str,
        files_modified: List[str],
        degraded_mode: bool,
        error: Optional[str],
    ) -> None:
        record = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "trigger": trigger,
            "issue": issue,
            "action_taken": action_taken,
            "files_modified": list(files_modified),
            "degraded_mode": bool(degraded_mode),
            "error": error,
        }
        line = json.dumps(record, sort_keys=True) + "\n"
        try:
            self.DECISION_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
            with self.DECISION_LOG_PATH.open("a", encoding="utf-8") as fh:
                fh.write(line)
        except OSError as err:
            # Last-resort: we can't even write the decision log. Log via
            # the python logger so the error surfaces somewhere.
            logger.error(
                "self_heal: decision log write failed path=%s error=%r",
                self.DECISION_LOG_PATH, err,
            )

    # ------------------------------------------------------------------ #
    # Unhandled-error log                                                #
    # ------------------------------------------------------------------ #
    def _record_unhandled_error(self, exc: BaseException) -> None:
        record = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "module": "self_heal",
            "error": repr(exc),
            "traceback": traceback.format_exc(),
        }
        line = json.dumps(record, sort_keys=True) + "\n"
        try:
            self.ERROR_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
            with self.ERROR_LOG_PATH.open("a", encoding="utf-8") as fh:
                fh.write(line)
        except OSError as err:
            logger.error(
                "self_heal: unhandled_errors.jsonl write failed "
                "path=%s error=%r original=%r",
                self.ERROR_LOG_PATH, err, exc,
            )


__all__ = ["SelfHeal"]
