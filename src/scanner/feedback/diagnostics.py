"""Post-trade diagnostics layer.

Inspects journal outcomes, agent weight boundaries, gate firing rates,
drawdown streaks, and RL model staleness — returning a structured health
report consumed by ``src.scanner.feedback.post_trade_loop.PostTradeLoop``.

Hard scope:
    * Stdlib + ``yaml`` only. No imports from other scanner subsystems.
    * ``run()`` NEVER raises. Every failure path returns a dict and logs
      with context at WARNING/ERROR level. Each check is isolated in its
      own try/except so one failing check does not poison the rest.
    * Thresholds are yaml-overridable under ``feedback_loop.diagnostics.*``
      with sensible hardcoded fallbacks. See ``DEFAULTS`` below.

Interface contract (called by PostTradeLoop._run_diagnostics):

    diag = PostTradeDiagnostics()
    result = diag.run(entry)   # `entry` is optional journal dict

    result = {
        "status": "HEALTHY" | "DEGRADED" | "CRITICAL",
        "issues": [
            {"check": str, "severity": "warning"|"critical",
             "detail": str, "value": Any, "threshold": Any},
            ...
        ],
        "recommended_actions": [str, ...],
        "metadata": {"trades_analyzed": int, "timestamp": iso},
    }
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

try:
    import yaml  # type: ignore
except ImportError:  # pragma: no cover — yaml is installed in this env
    yaml = None  # type: ignore

logger = logging.getLogger(__name__)


class PostTradeDiagnostics:
    """Structured health report for the feedback loop."""

    # ------------------------------------------------------------------ #
    # File path constants                                                #
    # ------------------------------------------------------------------ #
    JOURNAL_PATH: Path = Path("trained_data/trade_journal_rl.json")
    AGENT_WEIGHTS_PATH: Path = Path("trained_data/models/agent_weights.json")
    RL_MODEL_PATH: Path = Path("trained_data/models/rl_position_sizer.zip")
    CONFIG_PATH: Path = Path("config/config_improved_H1.yaml")

    # ------------------------------------------------------------------ #
    # Threshold defaults — overridable under yaml feedback_loop.diagnostics #
    # ------------------------------------------------------------------ #
    DEFAULTS: Dict[str, Any] = {
        # direction_accuracy
        "direction_window": 20,
        "direction_warning_lt": 0.52,
        "direction_critical_lt": 0.48,
        # gate_firing_rate
        "gate_window": 50,
        "gate_broken_off_lt": 0.05,
        "gate_broken_on_gt": 0.95,
        # agent_weight_boundary
        "weight_min": 0.1,
        "weight_max": 2.0,
        "weight_tolerance": 0.001,
        # drawdown_streak
        "drawdown_scan_window": 20,
        "drawdown_warning_streak": 5,
        "drawdown_critical_streak": 8,
        # rl_model_staleness (seconds)
        "rl_warning_age_s": 48 * 3600,      # 48 hours
        "rl_critical_age_s": 168 * 3600,    # 1 week
        # quiet_streak (audit 2026-04-28: gate-overtightening trap detector).
        # A "quiet streak" = zero closed trades for >quiet_warning_hours, AND
        # config_adjustments.json shows accumulated tightening (>= total_adj_floor).
        # Both conditions must hold — a healthy bot may have quiet hours when
        # markets are closed; only the combination signals the death spiral
        # where progressive tightening blocked all execution.
        "quiet_warning_hours": 24,          # 1 day of zero trades
        "quiet_critical_hours": 72,         # 3 days = unequivocal trap
        "quiet_total_adj_floor": 30,        # >=30 cumulative adjustments to be "overtightened"
    }

    # Names of the three gate pass-fields expected on journal entries.
    GATE_FIELDS: Tuple[str, ...] = (
        "momentum_passed",
        "confidence_passed",
        "risk_passed",
    )

    # ------------------------------------------------------------------ #
    # Init                                                               #
    # ------------------------------------------------------------------ #
    def __init__(self, config_path: Optional[Path] = None) -> None:
        self._config_path = Path(config_path) if config_path else self.CONFIG_PATH
        self._thresholds: Dict[str, Any] = dict(self.DEFAULTS)
        self._load_config()

    def _load_config(self) -> None:
        """Load yaml thresholds once. Missing file/keys → fall back to defaults."""
        if yaml is None:
            logger.warning(
                "diagnostics: yaml module unavailable — using hardcoded defaults"
            )
            return

        if not self._config_path.exists():
            logger.debug(
                "diagnostics: config file missing path=%s — using defaults",
                self._config_path,
            )
            return

        try:
            with self._config_path.open("r", encoding="utf-8") as fh:
                data = yaml.safe_load(fh) or {}
        except (OSError, UnicodeDecodeError) as err:
            logger.warning(
                "diagnostics: config read failed path=%s error=%r",
                self._config_path, err,
            )
            return
        except Exception as err:  # noqa: BLE001 — yaml.YAMLError and friends
            logger.warning(
                "diagnostics: config parse failed path=%s error=%r",
                self._config_path, err,
            )
            return

        if not isinstance(data, dict):
            logger.debug(
                "diagnostics: config root not a dict type=%s — using defaults",
                type(data).__name__,
            )
            return

        section = (
            (data.get("feedback_loop") or {}).get("diagnostics")
            if isinstance(data.get("feedback_loop"), dict)
            else None
        )
        if isinstance(section, dict):
            for key, default in self.DEFAULTS.items():
                if key in section:
                    try:
                        self._thresholds[key] = type(default)(section[key])
                    except (TypeError, ValueError) as err:
                        logger.warning(
                            "diagnostics: bad config value key=%s value=%r error=%r",
                            key, section[key], err,
                        )
        else:
            logger.debug(
                "diagnostics: no feedback_loop.diagnostics section — using defaults"
            )

    # ------------------------------------------------------------------ #
    # Public entry point                                                 #
    # ------------------------------------------------------------------ #
    def run(self, entry: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Run every health check. Aggregates results into a report dict."""
        issues: List[Dict[str, Any]] = []
        actions: List[str] = []
        trades_analyzed = 0

        try:
            journal = self._load_journal()
            trades_analyzed = len(journal)

            # Each check is isolated: a failure becomes a skipped result
            # rather than taking down the whole run.
            self._safe_check(
                "direction_accuracy", self._check_direction_accuracy,
                journal, issues, actions,
            )
            self._safe_check(
                "gate_firing_rate", self._check_gate_firing_rate,
                journal, issues, actions,
            )
            self._safe_check(
                "agent_weight_boundary", self._check_agent_weight_boundary,
                None, issues, actions,
            )
            self._safe_check(
                "drawdown_streak", self._check_drawdown_streak,
                journal, issues, actions,
            )
            self._safe_check(
                "rl_model_staleness", self._check_rl_model_staleness,
                None, issues, actions,
            )
            self._safe_check(
                "model_training_freshness", self._check_model_training_freshness,
                None, issues, actions,
            )
            # Quiet-streak detector — recommends widening when no trades have
            # closed in N hours AND config_adjustments has accumulated tightening.
            # The asymmetry-fix for the gate-overtightening trap discovered
            # 2026-04-28: existing checks tighten on losses but never widen on
            # silence, producing the "intelligently fucking itself" pattern
            # documented in .claude/learnings.md self_heal_no_debounce_C1..C6.
            self._safe_check(
                "quiet_streak", self._check_quiet_streak,
                journal, issues, actions,
            )

            status = self._aggregate_status(issues)

            return {
                "status": status,
                "issues": issues,
                "recommended_actions": actions,
                "metadata": {
                    "trades_analyzed": trades_analyzed,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "entry_trade_id": (entry or {}).get("trade_id")
                    if isinstance(entry, dict) else None,
                },
            }
        except Exception as err:  # noqa: BLE001 — outer safety net
            logger.error(
                "diagnostics: run() safety net caught error=%r", err,
            )
            return {
                "status": "DEGRADED",
                "issues": issues,
                "recommended_actions": actions,
                "metadata": {
                    "trades_analyzed": trades_analyzed,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "fatal_error": repr(err),
                },
            }

    # ------------------------------------------------------------------ #
    # Check harness                                                      #
    # ------------------------------------------------------------------ #
    def _safe_check(
        self,
        name: str,
        fn: Any,
        journal: Optional[List[Dict[str, Any]]],
        issues: List[Dict[str, Any]],
        actions: List[str],
    ) -> None:
        """Invoke one check; on any exception append a skipped-issue and move on."""
        try:
            if journal is None:
                result = fn(issues, actions)
            else:
                result = fn(journal, issues, actions)
            if result is False:
                logger.debug("diagnostics: %s returned False (no issues)", name)
        except Exception as err:  # noqa: BLE001
            logger.warning(
                "diagnostics: check=%s raised error=%r", name, err,
            )
            issues.append({
                "check": name,
                "severity": "skipped",
                "detail": "check raised: {0}".format(repr(err)),
                "value": None,
                "threshold": None,
            })

    # ------------------------------------------------------------------ #
    # Individual checks                                                  #
    # ------------------------------------------------------------------ #
    def _check_direction_accuracy(
        self,
        journal: List[Dict[str, Any]],
        issues: List[Dict[str, Any]],
        actions: List[str],
    ) -> bool:
        """Fraction of recent trades where outcome.trade_won is True."""
        window = int(self._thresholds["direction_window"])
        warn_lt = float(self._thresholds["direction_warning_lt"])
        crit_lt = float(self._thresholds["direction_critical_lt"])

        closed = [
            e for e in journal
            if isinstance(e, dict) and isinstance(e.get("outcome"), dict)
            and e["outcome"].get("trade_won") is not None
        ]
        recent = closed[-window:]
        if len(recent) < window:
            issues.append({
                "check": "direction_accuracy",
                "severity": "skipped",
                "detail": "only {0}/{1} closed trades available".format(
                    len(recent), window,
                ),
                "value": len(recent),
                "threshold": window,
            })
            return False

        wins = sum(1 for e in recent if bool(e["outcome"].get("trade_won")))
        rate = wins / float(window)

        if rate < crit_lt:
            issues.append({
                "check": "direction_accuracy",
                "severity": "critical",
                "detail": "win rate {0:.3f} below critical threshold".format(rate),
                "value": round(rate, 4),
                "threshold": crit_lt,
            })
            actions.append("retrain_gates")
            return True
        if rate < warn_lt:
            issues.append({
                "check": "direction_accuracy",
                "severity": "warning",
                "detail": "win rate {0:.3f} below warning threshold".format(rate),
                "value": round(rate, 4),
                "threshold": warn_lt,
            })
            actions.append("retrain_gates")
            return True
        return False

    def _check_gate_firing_rate(
        self,
        journal: List[Dict[str, Any]],
        issues: List[Dict[str, Any]],
        actions: List[str],
    ) -> bool:
        """Per-gate pass fraction over the last N entries."""
        window = int(self._thresholds["gate_window"])
        off_lt = float(self._thresholds["gate_broken_off_lt"])
        on_gt = float(self._thresholds["gate_broken_on_gt"])

        recent = [e for e in journal if isinstance(e, dict)][-window:]
        if len(recent) < window:
            issues.append({
                "check": "gate_firing_rate",
                "severity": "skipped",
                "detail": "only {0}/{1} entries available".format(
                    len(recent), window,
                ),
                "value": len(recent),
                "threshold": window,
            })
            return False

        any_flagged = False
        for gate in self.GATE_FIELDS:
            present = [e for e in recent if gate in e]
            if not present:
                issues.append({
                    "check": "gate_firing_rate",
                    "severity": "skipped",
                    "detail": "gate field '{0}' missing from all entries".format(gate),
                    "value": 0,
                    "threshold": window,
                })
                continue
            passes = sum(1 for e in present if bool(e.get(gate)))
            rate = passes / float(len(present))

            if rate < off_lt:
                issues.append({
                    "check": "gate_firing_rate",
                    "severity": "warning",
                    "detail": "gate '{0}' pass rate {1:.3f} below floor".format(
                        gate, rate,
                    ),
                    "value": round(rate, 4),
                    "threshold": off_lt,
                })
                actions.append("reset_gate_threshold_to_default:{0}".format(gate))
                any_flagged = True
            elif rate > on_gt:
                issues.append({
                    "check": "gate_firing_rate",
                    "severity": "warning",
                    "detail": "gate '{0}' pass rate {1:.3f} above ceiling".format(
                        gate, rate,
                    ),
                    "value": round(rate, 4),
                    "threshold": on_gt,
                })
                actions.append("tighten_gate_threshold:{0}".format(gate))
                any_flagged = True
        return any_flagged

    def _check_agent_weight_boundary(
        self,
        issues: List[Dict[str, Any]],
        actions: List[str],
    ) -> bool:
        """Flag every agent whose weight is clamped at min or max boundary."""
        lo = float(self._thresholds["weight_min"])
        hi = float(self._thresholds["weight_max"])
        tol = float(self._thresholds["weight_tolerance"])

        if not self.AGENT_WEIGHTS_PATH.exists():
            issues.append({
                "check": "agent_weight_boundary",
                "severity": "skipped",
                "detail": "agent_weights.json missing at {0}".format(
                    self.AGENT_WEIGHTS_PATH,
                ),
                "value": None,
                "threshold": None,
            })
            return False

        try:
            with self.AGENT_WEIGHTS_PATH.open("r", encoding="utf-8") as fh:
                data = json.load(fh)
        except (OSError, json.JSONDecodeError, UnicodeDecodeError) as err:
            logger.warning(
                "diagnostics: weights load failed path=%s error=%r",
                self.AGENT_WEIGHTS_PATH, err,
            )
            issues.append({
                "check": "agent_weight_boundary",
                "severity": "skipped",
                "detail": "weights load failed: {0}".format(repr(err)),
                "value": None,
                "threshold": None,
            })
            return False

        if not isinstance(data, dict):
            issues.append({
                "check": "agent_weight_boundary",
                "severity": "skipped",
                "detail": "weights root not a dict (type {0})".format(
                    type(data).__name__,
                ),
                "value": None,
                "threshold": None,
            })
            return False

        any_flagged = False
        seen: set = set()
        # Schema is {regime_bucket: {agent_name: float}}. We iterate every
        # non-meta bucket and dedupe by agent name so a clamped agent fires
        # at most one issue even if it's clamped in multiple buckets.
        for bucket_name, bucket in data.items():
            if not isinstance(bucket, dict):
                continue
            if bucket_name.startswith("_meta"):
                continue
            for agent, weight in bucket.items():
                if agent in seen:
                    continue
                try:
                    w = float(weight)
                except (TypeError, ValueError):
                    continue
                at_lo = abs(w - lo) <= tol
                at_hi = abs(w - hi) <= tol
                if at_lo or at_hi:
                    seen.add(agent)
                    boundary = "min" if at_lo else "max"
                    issues.append({
                        "check": "agent_weight_boundary",
                        "severity": "warning",
                        "detail": (
                            "agent '{0}' clamped at {1} boundary "
                            "in bucket '{2}'"
                        ).format(agent, boundary, bucket_name),
                        "value": round(w, 4),
                        "threshold": lo if at_lo else hi,
                    })
                    actions.append("soft_reset_agent_weight:{0}".format(agent))
                    any_flagged = True
        return any_flagged

    def _check_drawdown_streak(
        self,
        journal: List[Dict[str, Any]],
        issues: List[Dict[str, Any]],
        actions: List[str],
    ) -> bool:
        """Count consecutive losses at the most-recent tail of the journal."""
        window = int(self._thresholds["drawdown_scan_window"])
        warn = int(self._thresholds["drawdown_warning_streak"])
        crit = int(self._thresholds["drawdown_critical_streak"])

        closed = [
            e for e in journal
            if isinstance(e, dict) and isinstance(e.get("outcome"), dict)
            and e["outcome"].get("trade_won") is not None
        ]
        if not closed:
            issues.append({
                "check": "drawdown_streak",
                "severity": "skipped",
                "detail": "no closed trades with outcomes",
                "value": 0,
                "threshold": warn,
            })
            return False

        tail = closed[-window:]
        streak = 0
        for e in reversed(tail):
            if bool(e["outcome"].get("trade_won")) is False:
                streak += 1
            else:
                break

        if streak >= crit:
            issues.append({
                "check": "drawdown_streak",
                "severity": "critical",
                "detail": "{0} consecutive losses at tail".format(streak),
                "value": streak,
                "threshold": crit,
            })
            actions.append("reduce_risk_per_trade_pct")
            return True
        if streak >= warn:
            issues.append({
                "check": "drawdown_streak",
                "severity": "warning",
                "detail": "{0} consecutive losses at tail".format(streak),
                "value": streak,
                "threshold": warn,
            })
            actions.append("reduce_risk_per_trade_pct")
            return True
        return False

    def _check_model_training_freshness(
        self,
        issues: List[Dict[str, Any]],
        actions: List[str],
    ) -> bool:
        """Age of the core ML models (modular ensemble + joint gates).

        Stale models are a major cause of losing streaks. The mechanical
        RL feedback loop slowly degrades agent weights but doesn't know
        to retrain — only this diagnostic surfaces "models are X days old"
        so SelfHeal/Claude can recommend retraining.

        Thresholds are read from model_freshness module (single source of truth).
        """
        try:
            from src.scanner.automation.model_freshness import (
                get_model_freshness, STALE_DAYS, CRITICAL_DAYS,
            )
            freshness = get_model_freshness()
        except Exception as err:
            issues.append({
                "check": "model_training_freshness",
                "severity": "skipped",
                "detail": "freshness lookup failed: {0}".format(repr(err)),
                "value": None,
                "threshold": None,
            })
            return False

        oldest = freshness.get("oldest_age_days")
        status = freshness.get("status", "UNKNOWN")
        stale_models = freshness.get("stale_models") or []

        if oldest is None:
            issues.append({
                "check": "model_training_freshness",
                "severity": "skipped",
                "detail": "no model training metadata found",
                "value": None,
                "threshold": None,
            })
            return False

        if status == "CRITICAL":
            issues.append({
                "check": "model_training_freshness",
                "severity": "critical",
                "detail": (
                    "oldest model is {0:.0f} days old "
                    "(>{1}d critical threshold) — models are likely drifting "
                    "from current market regime: {2}"
                ).format(oldest, CRITICAL_DAYS, ", ".join(stale_models)),
                "value": oldest,
                "threshold": CRITICAL_DAYS,
            })
            actions.append(
                "Retrain core models: python main.py train-joint "
                "--instruments EUR_USD,GBP_USD,USD_JPY"
            )
            return True
        if status == "STALE":
            issues.append({
                "check": "model_training_freshness",
                "severity": "warning",
                "detail": (
                    "oldest model is {0:.0f} days old "
                    "(>{1}d stale threshold): {2}"
                ).format(oldest, STALE_DAYS, ", ".join(stale_models)),
                "value": oldest,
                "threshold": STALE_DAYS,
            })
            actions.append(
                "Schedule retraining of stale models within the next 7 days"
            )
            return True
        # FRESH or AGING — no issue raised, but record for transparency
        return True

    def _check_rl_model_staleness(
        self,
        issues: List[Dict[str, Any]],
        actions: List[str],
    ) -> bool:
        """Age of the RL position-sizer model file."""
        warn_s = int(self._thresholds["rl_warning_age_s"])
        crit_s = int(self._thresholds["rl_critical_age_s"])

        if not self.RL_MODEL_PATH.exists():
            issues.append({
                "check": "rl_model_staleness",
                "severity": "skipped",
                "detail": "rl model file missing at {0}".format(self.RL_MODEL_PATH),
                "value": None,
                "threshold": None,
            })
            return False

        try:
            mtime = os.path.getmtime(self.RL_MODEL_PATH)
        except OSError as err:
            logger.warning(
                "diagnostics: getmtime failed path=%s error=%r",
                self.RL_MODEL_PATH, err,
            )
            issues.append({
                "check": "rl_model_staleness",
                "severity": "skipped",
                "detail": "getmtime failed: {0}".format(repr(err)),
                "value": None,
                "threshold": None,
            })
            return False

        age = max(0.0, datetime.now(timezone.utc).timestamp() - mtime)

        if age >= crit_s:
            issues.append({
                "check": "rl_model_staleness",
                "severity": "critical",
                "detail": "rl model age {0:.0f}s exceeds critical".format(age),
                "value": round(age, 1),
                "threshold": crit_s,
            })
            actions.append("retrain_rl_position_sizer")
            return True
        if age >= warn_s:
            issues.append({
                "check": "rl_model_staleness",
                "severity": "warning",
                "detail": "rl model age {0:.0f}s exceeds warning".format(age),
                "value": round(age, 1),
                "threshold": warn_s,
            })
            actions.append("retrain_rl_position_sizer")
            return True
        return False

    # ------------------------------------------------------------------ #
    # Quiet-streak detector (asymmetry fix for the overtightening trap)  #
    # ------------------------------------------------------------------ #
    CONFIG_ADJUSTMENTS_PATH: Path = Path(".claude/config_adjustments.json")

    # Gates whose tightening signature most often leads to the trap. Sorted by
    # the order to UNTIGHTEN them when recovering — least-impactful gate gets
    # reset first so we don't unleash all blockers simultaneously.
    _QUIET_STREAK_RECOVERY_ORDER: Tuple[str, ...] = (
        "min_confidence",
        "weighted_vote_threshold",
        "max_model_disagreement",
        "max_uncertainty_score",
        "atr_sl_multiplier",
        "min_momentum",
        "min_risk_reward_ratio",
    )

    def _check_quiet_streak(
        self,
        journal: List[Dict[str, Any]],
        issues: List[Dict[str, Any]],
        actions: List[str],
    ) -> bool:
        """Detect zero-trade streak combined with overtightened config.

        Two conditions must hold to flag:
          1. No closed trade timestamp within `quiet_warning_hours` (or
             `quiet_critical_hours` for CRITICAL severity)
          2. config_adjustments.json shows >= `quiet_total_adj_floor`
             accumulated adjustments (signal that the bot has been
             progressively tightening its own gates)

        Both conditions together are the unique signature of the
        gate-overtightening trap — a quiet bot whose own config history
        proves it tightened itself into silence. A healthy bot may be quiet
        because markets are calm (condition 1 alone) or may have many
        adjustments without being overtightened (condition 2 alone — many
        could be widening). Only the combination is conclusive.

        Recommended action on flag: reset the most recently tightened gate
        in `_QUIET_STREAK_RECOVERY_ORDER` back to its YAML default. This
        widens *one* gate at a time so we don't unleash all blockers
        simultaneously — the next cycle either trades again (recovery
        complete) or remains quiet (recommend the next gate down the list).
        """
        warn_h = float(self._thresholds["quiet_warning_hours"])
        crit_h = float(self._thresholds["quiet_critical_hours"])
        adj_floor = int(self._thresholds["quiet_total_adj_floor"])

        # 1. Compute hours since last closed trade. The journal schema:
        #    `outcome` is a string ("win"|"loss"|None) — NOT a dict.
        #    `close_time` lives at the top level (ISO with optional 'Z' or offset).
        #    A trade is "closed" when outcome is one of the known strings.
        last_close_age_s: Optional[float] = None
        from datetime import datetime, timezone
        # Strip extra-precise fractional seconds (Python's fromisoformat caps at 6
        # digits; OANDA returns 9). Also tolerate trailing 'Z' as +00:00.
        def _parse_ts(raw: Any) -> Optional[datetime]:
            if not raw:
                return None
            s = str(raw).strip().replace("Z", "+00:00")
            # Truncate fractional seconds to 6 digits if present.
            if "." in s:
                head, _, rest = s.partition(".")
                # rest may include offset like "+00:00" — split on the first non-digit.
                frac = ""
                tail = ""
                for i, ch in enumerate(rest):
                    if ch.isdigit():
                        frac += ch
                    else:
                        tail = rest[i:]
                        break
                s = "{0}.{1}{2}".format(head, frac[:6], tail)
            try:
                dt = datetime.fromisoformat(s)
                if dt.tzinfo is None:
                    dt = dt.replace(tzinfo=timezone.utc)
                return dt
            except (ValueError, TypeError):
                return None

        for entry in reversed(journal or []):
            if not isinstance(entry, dict):
                continue
            outcome = entry.get("outcome")
            # Closed trades have outcome=str (win/loss/manual_close/etc).
            if not isinstance(outcome, str) or not outcome:
                continue
            ts = entry.get("close_time") or entry.get("timestamp")
            last_dt = _parse_ts(ts)
            if last_dt is None:
                continue
            last_close_age_s = (
                datetime.now(timezone.utc) - last_dt
            ).total_seconds()
            break

        if last_close_age_s is None:
            issues.append({
                "check": "quiet_streak",
                "severity": "skipped",
                "detail": "no parseable close timestamp on any journal entry",
                "value": None,
                "threshold": None,
            })
            return False

        last_close_age_h = last_close_age_s / 3600.0

        # Short-circuit: if quiet streak doesn't exceed warning threshold,
        # nothing to flag — markets are simply not delivering setups.
        if last_close_age_h < warn_h:
            return False

        # 2. Read config_adjustments to test the "tightened itself" hypothesis.
        total_adjustments = 0
        recently_tightened: Optional[str] = None
        try:
            import json
            if self.CONFIG_ADJUSTMENTS_PATH.exists():
                blob = json.loads(self.CONFIG_ADJUSTMENTS_PATH.read_text(encoding="utf-8"))
                if isinstance(blob, dict):
                    total_adjustments = int(blob.get("total_adjustments", 0) or 0)
                    history = blob.get("history") or []
                    if isinstance(history, list):
                        for h in reversed(history):
                            if not isinstance(h, dict):
                                continue
                            key = str(h.get("key", ""))
                            if key in self._QUIET_STREAK_RECOVERY_ORDER:
                                recently_tightened = key
                                break
                    if recently_tightened is None:
                        proposed = blob.get("proposed_adjustments") or []
                        if isinstance(proposed, list):
                            for p in reversed(proposed):
                                if not isinstance(p, dict):
                                    continue
                                key = str(p.get("key", ""))
                                if key in self._QUIET_STREAK_RECOVERY_ORDER:
                                    recently_tightened = key
                                    break
        except (OSError, ValueError) as e:
            logger.warning("quiet_streak: config_adjustments read failed err=%r", e)

        # If we've been quiet but the config hasn't been tightened, this isn't
        # the overtightening trap — leave a warning but don't recommend reset.
        if total_adjustments < adj_floor:
            issues.append({
                "check": "quiet_streak",
                "severity": "warning",
                "detail": (
                    "no closed trades for {0:.1f}h but only {1} config adjustments "
                    "(<{2} floor) — likely market quiet, not overtightening"
                ).format(last_close_age_h, total_adjustments, adj_floor),
                "value": round(last_close_age_h, 1),
                "threshold": warn_h,
            })
            return True

        # Both conditions hold. Pick a gate to reset, but skip any gate whose
        # corresponding reset action is in active SelfHeal debounce — that
        # means we tried it last cycle and the streak persists, so move on
        # to the next candidate. Without this rotation the system would
        # waste 6h on a single gate before trying the next.
        debounced_gates: set = set()
        try:
            import json as _json
            from datetime import datetime, timezone
            debounce_path = Path(".claude/self_heal_debounce.json")
            if debounce_path.exists():
                blob = _json.loads(debounce_path.read_text(encoding="utf-8"))
                if isinstance(blob, dict):
                    now = datetime.now(timezone.utc)
                    # Mirror SelfHeal.DEBOUNCE_WINDOW_S — kept hardcoded here
                    # to avoid an import dependency on the consumer module.
                    debounce_window_s = 6.0 * 3600.0
                    for action_str, ts in blob.items():
                        if not isinstance(action_str, str):
                            continue
                        if not action_str.startswith("reset_gate_threshold_to_default:"):
                            continue
                        try:
                            last_dt = datetime.fromisoformat(str(ts).replace("Z", "+00:00"))
                            if last_dt.tzinfo is None:
                                last_dt = last_dt.replace(tzinfo=timezone.utc)
                            if (now - last_dt).total_seconds() < debounce_window_s:
                                debounced_gates.add(action_str.split(":", 1)[1])
                        except (ValueError, TypeError):
                            continue
        except (OSError, ValueError) as e:
            logger.debug("quiet_streak: debounce-state read failed err=%r", e)

        # Choose the next gate to try: prefer the most-recently-tightened
        # gate if not already debounced, else walk the recovery order.
        candidates: List[str] = []
        if recently_tightened and recently_tightened not in debounced_gates:
            candidates.append(recently_tightened)
        for g in self._QUIET_STREAK_RECOVERY_ORDER:
            if g in debounced_gates or g in candidates:
                continue
            candidates.append(g)
        target_gate = candidates[0] if candidates else None

        if target_gate is None:
            # All recovery gates tried recently — escalate to operator and
            # don't fire another reset (would just stack into debounce).
            issues.append({
                "check": "quiet_streak",
                "severity": "critical",
                "detail": (
                    "OVERTIGHTENING TRAP UNRESOLVED: {0:.1f}h quiet, {1} adjustments, "
                    "all {2} recovery-order gates already reset and still no trades — "
                    "operator intervention required (review .claude/config_adjustments.json "
                    "and consider a one-shot full reset)"
                ).format(last_close_age_h, total_adjustments, len(self._QUIET_STREAK_RECOVERY_ORDER)),
                "value": round(last_close_age_h, 1),
                "threshold": crit_h,
            })
            return True

        severity = "critical" if last_close_age_h >= crit_h else "warning"
        issues.append({
            "check": "quiet_streak",
            "severity": severity,
            "detail": (
                "OVERTIGHTENING TRAP: no closed trades for {0:.1f}h, "
                "{1} cumulative config adjustments, latest tightened gate '{2}' — "
                "widening to break the spiral"
            ).format(last_close_age_h, total_adjustments, target_gate),
            "value": round(last_close_age_h, 1),
            "threshold": crit_h if severity == "critical" else warn_h,
        })
        actions.append("reset_gate_threshold_to_default:{0}".format(target_gate))
        return True

    # ------------------------------------------------------------------ #
    # Helpers                                                            #
    # ------------------------------------------------------------------ #
    def _load_journal(self) -> List[Dict[str, Any]]:
        """Load the trade journal as a list of dicts. Failures → []."""
        if not self.JOURNAL_PATH.exists():
            logger.debug(
                "diagnostics: journal file missing path=%s", self.JOURNAL_PATH,
            )
            return []
        try:
            with self.JOURNAL_PATH.open("r", encoding="utf-8") as fh:
                data = json.load(fh)
        except (OSError, json.JSONDecodeError, UnicodeDecodeError) as err:
            logger.warning(
                "diagnostics: journal load failed path=%s error=%r",
                self.JOURNAL_PATH, err,
            )
            return []
        if not isinstance(data, list):
            logger.warning(
                "diagnostics: journal root not a list type=%s",
                type(data).__name__,
            )
            return []
        return data

    @staticmethod
    def _aggregate_status(issues: List[Dict[str, Any]]) -> str:
        """HEALTHY if no non-skipped issues, CRITICAL if any critical,
        DEGRADED if any warnings."""
        has_critical = False
        has_warning = False
        for issue in issues:
            sev = issue.get("severity")
            if sev == "critical":
                has_critical = True
            elif sev == "warning":
                has_warning = True
        if has_critical:
            return "CRITICAL"
        if has_warning:
            return "DEGRADED"
        return "HEALTHY"


__all__ = ["PostTradeDiagnostics"]
