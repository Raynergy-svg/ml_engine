"""Adaptive config tuner that reads rules and adjusts scanner config.

Closes the learning loop: trade outcome -> learning -> rule -> config change.
Implementation target: PRD US-005.
"""

from __future__ import annotations

import json
import logging
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# Project root: config_tuner.py lives at src/scanner/automation/config_tuner.py
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent

# Rule-to-config mapping table.  Each entry maps a regex pattern found in
# rule text to (config_param, extracted_value_or_None, reason_template).
# When extracted_value is None the regex must contain a named group "value"
# that will be cast to float.
_RULE_PARAM_MAP: List[Dict[str, Any]] = [
    {
        "pattern": re.compile(
            r"R:R ratio below (?P<value>\d+(?:\.\d+)?):1", re.IGNORECASE
        ),
        "param": "min_risk_reward_ratio",
        "comparison": "gte",  # config value must be >= extracted value
        "reason": "Rule: minimum R:R ratio gate ({value}:1)",
        "section_hint": "execution",
    },
    {
        "pattern": re.compile(
            r"weighted_vote_score\s*\(?\s*>\s*(?P<value>\d+(?:\.\d+)?)\s*\)?",
            re.IGNORECASE,
        ),
        "param": "weighted_vote_threshold",
        "comparison": "gte",
        "reason": "Rule: higher vote score (>{value}) correlates with better outcomes",
        "section_hint": "agent",
    },
    {
        "pattern": re.compile(
            r"[Uu]ncertainty score\s*>\s*(?P<value>\d+(?:\.\d+)?)", re.IGNORECASE
        ),
        "param": "max_uncertainty_score",
        "comparison": "lte",  # config ceiling must be <= extracted value
        "reason": "Rule: uncertainty above {value} is a warning signal",
        "section_hint": "agent",
    },
    {
        "pattern": re.compile(
            r"[Mm]odel disagreement\s*>\s*(?P<value>\d+(?:\.\d+)?)", re.IGNORECASE
        ),
        "param": "max_model_disagreement",
        "comparison": "lte",
        "reason": "Rule: model disagreement above {value} predicts losses",
        "section_hint": "agent",
    },
    {
        "pattern": re.compile(
            r"[Mm]aximum portfolio risk:\s*(?P<value>\d+(?:\.\d+)?)%", re.IGNORECASE
        ),
        "param": "risk_per_trade_pct",
        # This is informational -- the tuner will not auto-lower risk_per_trade_pct
        # because the rule references *portfolio* risk, not per-trade risk.
        # Included for completeness of rule parsing.
        "comparison": None,
        "reason": "Rule: maximum portfolio risk is {value}%",
        "section_hint": "risk",
    },
]


class ConfigTuner:
    """Reads promoted rules and adjusts ScannerConfig parameters.

    Lifecycle:
        1. ``load_rules()``       -- parse trading.md into structured dicts
        2. ``apply_to_config()``  -- compare rules against live config, adjust
        3. ``log_adjustment()``   -- persist every change with timestamp + reason
        4. ``get_adjustment_history()`` -- review recent changes
        5. ``evaluate_adjustment_impact()`` -- before/after win-rate analysis
    """

    CONFIG_LOG = ".claude/config_adjustments.json"
    RULES_PATH = ".claude/rules/trading.md"
    TRADE_JOURNAL_PATH = "trained_data/trade_journal_rl.json"

    def __init__(self, project_root: Optional[Path] = None) -> None:
        self._root = Path(project_root) if project_root else PROJECT_ROOT

    # ------------------------------------------------------------------
    # 1. Rule loading
    # ------------------------------------------------------------------

    def load_rules(self) -> List[Dict[str, Any]]:
        """Read ``.claude/rules/trading.md`` and parse each rule line.

        Returns:
            List of dicts with keys: ``section``, ``rule_text``, ``keywords``.
            Only lines starting with ``- `` under an ``## <Section>`` header
            are treated as rules.
        """
        rules_path = self._root / self.RULES_PATH
        if not rules_path.exists():
            logger.warning("Rules file not found: %s", rules_path)
            return []

        text = rules_path.read_text(encoding="utf-8")
        rules: List[Dict[str, Any]] = []
        current_section = "General"

        for line in text.splitlines():
            stripped = line.strip()

            # Detect markdown H2 sections
            if stripped.startswith("## "):
                current_section = stripped[3:].strip()
                continue

            # Rule lines begin with "- "
            if not stripped.startswith("- "):
                continue

            rule_text = stripped[2:].strip()
            if not rule_text:
                continue

            keywords = self._extract_keywords(rule_text)
            rules.append(
                {
                    "section": current_section,
                    "rule_text": rule_text,
                    "keywords": keywords,
                }
            )

        logger.info("Loaded %d rules from %s", len(rules), rules_path)
        return rules

    # ------------------------------------------------------------------
    # 2. Apply rules to config
    # ------------------------------------------------------------------

    def apply_to_config(self, scanner_config: Any) -> Dict[str, Any]:
        """Compare parsed rules against *scanner_config* and adjust parameters.

        Args:
            scanner_config: A ``ScannerConfig`` instance (or any object whose
                attributes map to the tunable parameters).

        Returns:
            Dict of ``{param: new_value}`` for every parameter that was changed.
        """
        rules = self.load_rules()
        changes: Dict[str, Any] = {}

        for rule in rules:
            rule_text = rule["rule_text"]
            section = rule["section"]

            for mapping in _RULE_PARAM_MAP:
                match = mapping["pattern"].search(rule_text)
                if not match:
                    continue

                param: str = mapping["param"]
                comparison = mapping["comparison"]
                if comparison is None:
                    # Informational-only rule (e.g. portfolio risk); skip tuning.
                    continue

                extracted = float(match.group("value"))
                current_value = getattr(scanner_config, param, None)
                if current_value is None:
                    logger.debug(
                        "Config has no attribute '%s'; skipping rule", param
                    )
                    continue

                current_value = float(current_value)
                new_value: Optional[float] = None

                if comparison == "gte" and current_value < extracted:
                    # Config must be at least the rule-stated minimum.
                    new_value = extracted
                elif comparison == "lte" and current_value > extracted:
                    # Config ceiling must not exceed the rule-stated threshold.
                    new_value = extracted

                if new_value is not None and new_value != current_value:
                    reason = mapping["reason"].format(value=extracted)
                    rule_source = self._section_to_anchor(section)

                    setattr(scanner_config, param, new_value)
                    self.log_adjustment(
                        param=param,
                        old_value=current_value,
                        new_value=new_value,
                        reason=reason,
                        rule_source=rule_source,
                    )
                    changes[param] = new_value
                    logger.info(
                        "ConfigTuner adjusted %s: %s -> %s (%s)",
                        param,
                        current_value,
                        new_value,
                        reason,
                    )

        if not changes:
            logger.info("ConfigTuner: all parameters already satisfy rules")

        return changes

    # ------------------------------------------------------------------
    # 3. Adjustment logging
    # ------------------------------------------------------------------

    def log_adjustment(
        self,
        param: str,
        old_value: Any,
        new_value: Any,
        reason: str,
        rule_source: str,
    ) -> None:
        """Append an adjustment record to the config adjustments log.

        Creates the JSON file if it does not exist.
        """
        log_path = self._root / self.CONFIG_LOG
        log_path.parent.mkdir(parents=True, exist_ok=True)

        history = self._read_log(log_path)

        record = {
            "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "param": param,
            "old_value": self._serialize_value(old_value),
            "new_value": self._serialize_value(new_value),
            "reason": reason,
            "rule_source": rule_source,
        }
        history.append(record)

        log_path.write_text(
            json.dumps(history, indent=2) + "\n", encoding="utf-8"
        )
        logger.debug("Logged config adjustment: %s", record)

    # ------------------------------------------------------------------
    # 4. Adjustment history
    # ------------------------------------------------------------------

    def get_adjustment_history(self, limit: int = 20) -> List[Dict[str, Any]]:
        """Return the most recent *limit* config adjustments.

        Returns newest first.
        """
        log_path = self._root / self.CONFIG_LOG
        history = self._read_log(log_path)
        # Return most recent entries first.
        return list(reversed(history[-limit:]))

    # ------------------------------------------------------------------
    # 5. Impact evaluation
    # ------------------------------------------------------------------

    def evaluate_adjustment_impact(self, param: str) -> Dict[str, Any]:
        """Analyse trade outcomes before vs. after the most recent adjustment
        of *param*.

        Reads ``trade_journal_rl.json`` and partitions trades by the timestamp
        of the last adjustment for *param*.

        Returns:
            Dict with ``param``, ``adjusted_at``, ``trades_before``,
            ``win_rate_before``, ``trades_after``, ``win_rate_after``.
            If no adjustment or no journal exists, counts default to 0.
        """
        result: Dict[str, Any] = {
            "param": param,
            "adjusted_at": None,
            "trades_before": 0,
            "win_rate_before": 0.0,
            "trades_after": 0,
            "win_rate_after": 0.0,
        }

        # Find the most recent adjustment timestamp for this param.
        log_path = self._root / self.CONFIG_LOG
        history = self._read_log(log_path)
        adjusted_at: Optional[str] = None
        for entry in reversed(history):
            if entry.get("param") == param:
                adjusted_at = entry.get("timestamp")
                break

        if adjusted_at is None:
            logger.info("No adjustment found for param '%s'", param)
            return result

        result["adjusted_at"] = adjusted_at

        # Load trade journal.
        journal_path = self._root / self.TRADE_JOURNAL_PATH
        trades = self._load_trade_journal(journal_path)
        if not trades:
            logger.info("Trade journal empty or missing; cannot evaluate impact")
            return result

        # Partition trades into before / after the adjustment timestamp.
        before_wins = 0
        before_total = 0
        after_wins = 0
        after_total = 0

        for trade in trades:
            trade_ts = self._extract_trade_timestamp(trade)
            if trade_ts is None:
                continue

            won = self._trade_is_win(trade)

            if trade_ts < adjusted_at:
                before_total += 1
                if won:
                    before_wins += 1
            else:
                after_total += 1
                if won:
                    after_wins += 1

        result["trades_before"] = before_total
        result["win_rate_before"] = (
            round(before_wins / before_total, 4) if before_total else 0.0
        )
        result["trades_after"] = after_total
        result["win_rate_after"] = (
            round(after_wins / after_total, 4) if after_total else 0.0
        )

        return result

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_keywords(rule_text: str) -> List[str]:
        """Pull meaningful keywords from a rule line.

        Strategy: grab all lowercase alpha tokens >= 3 chars that are not
        common stop-words, plus any numeric tokens that look like thresholds.
        """
        stop_words = {
            "the", "and", "for", "with", "from", "that", "this", "not",
            "are", "was", "were", "has", "have", "been", "but", "all",
            "can", "will", "its", "per", "use", "run", "get", "set",
            "every", "must", "above", "below", "before", "after",
            "when", "than", "into", "over", "also", "each", "never",
            "always",
        }
        tokens = re.findall(r"[a-z_]{3,}|\d+(?:\.\d+)?", rule_text.lower())
        return [t for t in tokens if t not in stop_words]

    @staticmethod
    def _section_to_anchor(section: str) -> str:
        """Convert a markdown section name to a file#anchor reference."""
        anchor = re.sub(r"[^a-z0-9]+", "-", section.lower()).strip("-")
        return f"trading.md#{anchor}"

    @staticmethod
    def _serialize_value(value: Any) -> Any:
        """Ensure a value is JSON-serializable."""
        if isinstance(value, float):
            return round(value, 6)
        return value

    @staticmethod
    def _read_log(log_path: Path) -> List[Dict[str, Any]]:
        """Read the config adjustments log file.

        Handles empty files, bare ``{}`` objects (legacy format), and
        missing files gracefully.
        """
        if not log_path.exists():
            return []

        try:
            raw = log_path.read_text(encoding="utf-8").strip()
            if not raw:
                return []
            data = json.loads(raw)
            if isinstance(data, list):
                return data
            # Legacy format: bare object -- wrap in list.
            if isinstance(data, dict):
                return [data] if data else []
            return []
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning("Could not read config log %s: %s", log_path, exc)
            return []

    @staticmethod
    def _load_trade_journal(journal_path: Path) -> List[Dict[str, Any]]:
        """Load trade journal entries from disk."""
        if not journal_path.exists():
            return []
        try:
            raw = journal_path.read_text(encoding="utf-8").strip()
            if not raw:
                return []
            data = json.loads(raw)
            if isinstance(data, list):
                return data
            if isinstance(data, dict):
                # Some journals use {"trades": [...]} wrapper.
                return data.get("trades", [])
            return []
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning("Could not read trade journal %s: %s", journal_path, exc)
            return []

    @staticmethod
    def _extract_trade_timestamp(trade: Dict[str, Any]) -> Optional[str]:
        """Return an ISO-ish timestamp string from a trade record.

        Checks common field names in order of preference.
        """
        for key in ("closed_at", "close_time", "timestamp", "opened_at", "open_time"):
            ts = trade.get(key)
            if ts:
                return str(ts)
        return None

    @staticmethod
    def _trade_is_win(trade: Dict[str, Any]) -> bool:
        """Determine if a trade was a winner.

        Checks ``outcome``, ``result``, and ``pnl`` fields.
        """
        outcome = str(trade.get("outcome", trade.get("result", ""))).lower()
        if outcome in ("win", "won", "tp_hit", "profit"):
            return True
        if outcome in ("loss", "lost", "sl_hit"):
            return False

        # Fallback: check numeric PnL.
        pnl = trade.get("pnl", trade.get("profit_loss", trade.get("realized_pl")))
        if pnl is not None:
            try:
                return float(pnl) > 0
            except (TypeError, ValueError):
                pass

        return False
