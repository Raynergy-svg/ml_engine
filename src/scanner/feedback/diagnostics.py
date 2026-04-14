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
