"""Adaptive config tuner that reads promoted rules and adjusts scanner config.

Closes the learning loop: trade outcome -> learning -> rule -> config change -> better trades.

US-005: Create adaptive config tuner that reads rules and adjusts scanner config.
"""

from __future__ import annotations

import fcntl
import hashlib
import json
import logging
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from src.scanner.config import ScannerConfig

logger = logging.getLogger(__name__)

RULES_PATH = Path(".claude/rules/trading.md")
ADJUSTMENTS_PATH = Path(".claude/config_adjustments.json")
APPLIED_RULES_PATH = Path(".claude/config_applied_rules.json")
ROLLBACK_LOG_PATH = Path("trained_data/config_rollback_log.jsonl")

# Bounds to prevent runaway tuning
# Every field that can be adjusted must have explicit min/max bounds
BOUNDS = {
    "atr_sl_multiplier": (0.5, 2.0),
    "atr_tp_multiplier": (0.8, 3.0),
    "max_uncertainty_score": (0.30, 0.95),
    "weighted_vote_threshold": (0.45, 0.95),
    "max_model_disagreement": (0.20, 0.95),
    "min_confidence_score": (0.30, 0.95),
    "min_momentum_score": (0.30, 0.95),
    "max_position_size": (0.01, 0.50),
}


class ConfigTuner:
    """Reads promoted rules and applies config adjustments."""

    def __init__(
        self,
        rules_path: Optional[Path] = None,
        adjustments_path: Optional[Path] = None,
        applied_rules_path: Optional[Path] = None,
    ):
        self.rules_path = rules_path or RULES_PATH
        self.adjustments_path = adjustments_path or ADJUSTMENTS_PATH
        self.applied_rules_path = applied_rules_path or APPLIED_RULES_PATH
        self._applied_rules: set[str] = self._load_applied_rules()

    def _load_applied_rules(self) -> set[str]:
        """Load set of previously applied rule hashes from disk."""
        if not self.applied_rules_path.exists():
            return set()
        try:
            data = json.loads(self.applied_rules_path.read_text())
            return set(data.get("applied_hashes", []))
        except Exception:
            logger.warning(f"Failed to load applied rules from {self.applied_rules_path}")
            return set()

    def _save_applied_rules(self) -> None:
        """Persist applied rule hashes to disk."""
        self.applied_rules_path.parent.mkdir(parents=True, exist_ok=True)
        data = {"applied_hashes": list(self._applied_rules)}
        self.applied_rules_path.write_text(json.dumps(data, indent=2))

    def load_rules(self) -> List[str]:
        """Parse .claude/rules/trading.md and extract promoted rule lines."""
        if not self.rules_path.exists():
            return []

        content = self.rules_path.read_text()
        rules: List[str] = []
        in_promoted = False

        for line in content.split("\n"):
            stripped = line.strip()
            if "## Promoted Rules" in stripped:
                in_promoted = True
                continue
            if in_promoted and stripped.startswith("- ["):
                rules.append(stripped)
            elif in_promoted and stripped.startswith("##"):
                break  # Next section

        return rules

    def apply_to_config(self, config: "ScannerConfig") -> List[Dict[str, Any]]:
        """Read promoted rules and adjust config fields. Returns list of adjustments made."""
        rules = self.load_rules()
        adjustments: List[Dict[str, Any]] = []

        for rule in rules:
            # Skip already applied rules using stable MD5 hash
            rule_hash = hashlib.md5(rule.encode()).hexdigest()
            if rule_hash in self._applied_rules:
                continue

            adj = self._apply_rule(rule, config)
            if adj:
                adjustments.append(adj)
                self._applied_rules.add(rule_hash)

        if adjustments:
            self._log_adjustments(adjustments)
            self._save_applied_rules()

        return adjustments

    def _apply_rule(self, rule: str, config: "ScannerConfig") -> Optional[Dict[str, Any]]:
        """Parse a single rule and apply to config. Returns adjustment dict or None."""
        # Pattern: "Increase atr_sl_multiplier by 0.1 for {pair}"
        sl_match = re.search(r"increase atr_sl_multiplier.*?for (\w+)", rule, re.IGNORECASE)
        if sl_match:
            # Global adjustment (per-pair is handled by pair_sl_tp_config.json)
            old = config.atr_sl_multiplier
            new = min(BOUNDS["atr_sl_multiplier"][1], old + 0.1)
            if new != old:
                config.atr_sl_multiplier = new
                return {"field": "atr_sl_multiplier", "old": old, "new": new, "rule": rule}

        # Pattern: "Increase atr_tp_multiplier by 0.1 for {pair}"
        tp_match = re.search(r"increase atr_tp_multiplier.*?for (\w+)", rule, re.IGNORECASE)
        if tp_match:
            old = config.atr_tp_multiplier
            new = min(BOUNDS["atr_tp_multiplier"][1], old + 0.1)
            if new != old:
                config.atr_tp_multiplier = new
                return {"field": "atr_tp_multiplier", "old": old, "new": new, "rule": rule}

        # Pattern: "Lower max_uncertainty_score by 0.02"
        if re.search(r"lower\s+max_uncertainty_score", rule, re.IGNORECASE):
            old = config.max_uncertainty_score
            new = max(BOUNDS["max_uncertainty_score"][0], old - 0.02)
            if new != old:
                config.max_uncertainty_score = new
                return {"field": "max_uncertainty_score", "old": old, "new": new, "rule": rule}

        # Pattern: "Prefer weighted_vote_score > 0.7" / "Lower weighted_vote_threshold"
        if re.search(r"weighted_vote", rule, re.IGNORECASE) and re.search(r"prefer", rule, re.IGNORECASE):
            old = config.weighted_vote_threshold
            new = max(BOUNDS["weighted_vote_threshold"][0], old - 0.01)
            if new != old:
                config.weighted_vote_threshold = new
                return {"field": "weighted_vote_threshold", "old": old, "new": new, "rule": rule}

        # Pattern: "Lower max_model_disagreement by 0.02"
        if re.search(r"lower\s+max_model_disagreement", rule, re.IGNORECASE):
            old = config.max_model_disagreement
            new = max(BOUNDS["max_model_disagreement"][0], old - 0.02)
            if new != old:
                config.max_model_disagreement = new
                return {"field": "max_model_disagreement", "old": old, "new": new, "rule": rule}

        return None

    def compare_session_performance(
        self, current_win_rate: float, current_trade_count: int, config: "ScannerConfig"
    ) -> List[Dict[str, Any]]:
        """Compare current session performance against recent snapshots.

        If win rate dropped > 10% compared to historical average, identifies
        recent config changes as suspects and rolls them back.

        Args:
            current_win_rate: Current session's win rate (0.0 to 1.0)
            current_trade_count: Number of trades in current session
            config: Current ScannerConfig to potentially rollback

        Returns:
            List of rollback actions taken (empty if none)
        """
        if current_trade_count < 5:
            return []  # Not enough data to evaluate

        # Load recent session snapshots
        try:
            from src.scanner.automation.session_snapshot import SessionSnapshotManager
            snapshot_mgr = SessionSnapshotManager()
            snapshots = snapshot_mgr.load_recent_snapshots(n=3)
        except Exception as e:
            logger.debug(f"Cannot load session snapshots for comparison: {e}")
            return []

        if not snapshots:
            return []

        # Calculate historical average win rate
        historical_rates = [s.win_rate for s in snapshots if s.win_rate > 0]
        if not historical_rates:
            return []

        avg_historical = sum(historical_rates) / len(historical_rates)
        delta = current_win_rate - avg_historical

        if delta >= -0.10:
            # No significant drop
            logger.debug(
                f"Session performance OK: current={current_win_rate:.1%} vs "
                f"historical={avg_historical:.1%} (delta={delta:+.1%})"
            )
            return []

        logger.warning(
            f"Session performance DROP: current={current_win_rate:.1%} vs "
            f"historical={avg_historical:.1%} (delta={delta:+.1%}). "
            f"Checking for suspect config changes..."
        )

        # Find recent config adjustments that might be causing the drop
        rollbacks: List[Dict[str, Any]] = []
        if not self.adjustments_path.exists():
            return rollbacks

        try:
            adjustments = json.loads(self.adjustments_path.read_text())
        except Exception:
            return rollbacks

        if not adjustments:
            return rollbacks

        # Get the last snapshot's timestamp to find adjustments made after it
        last_snapshot_ts = snapshots[0].timestamp if snapshots else ""

        # Find suspect adjustments (made after last good snapshot)
        suspects = [
            adj for adj in adjustments
            if adj.get("timestamp", "") > last_snapshot_ts
        ]

        if not suspects:
            return rollbacks

        # Roll back suspect config changes
        for adj in suspects:
            field = adj.get("field", "")
            old_val = adj.get("old")
            if field and old_val is not None and hasattr(config, field):
                current_val = getattr(config, field)
                setattr(config, field, old_val)
                rollback = {
                    "field": field,
                    "rolled_back_from": current_val,
                    "rolled_back_to": old_val,
                    "reason": f"Win rate dropped {delta:+.1%} (current={current_win_rate:.1%} vs avg={avg_historical:.1%})",
                    "suspect_adjustment": adj,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                }
                rollbacks.append(rollback)
                logger.warning(
                    f"ROLLBACK: {field} {current_val} → {old_val} "
                    f"(suspect change caused {delta:+.1%} win rate drop)"
                )

                # Remove the rule hash from applied rules so it can be re-evaluated
                rule_hash = hashlib.md5(adj.get("rule", "").encode()).hexdigest()
                self._applied_rules.discard(rule_hash)

        if rollbacks:
            self._save_applied_rules()
            self._log_rollbacks(rollbacks)

        return rollbacks

    def _log_rollbacks(self, rollbacks: List[Dict[str, Any]]) -> None:
        """Log rollback events to trained_data/config_rollback_log.jsonl."""
        try:
            from src.scanner.automation.safe_json import safe_jsonl_append
            for rb in rollbacks:
                safe_jsonl_append(ROLLBACK_LOG_PATH, rb)
        except ImportError:
            try:
                ROLLBACK_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
                with open(ROLLBACK_LOG_PATH, "a", encoding="utf-8") as f:
                    for rb in rollbacks:
                        f.write(json.dumps(rb, default=str) + "\n")
            except Exception as e:
                logger.debug(f"Failed to log rollback: {e}")

    def _log_adjustments(self, adjustments: List[Dict[str, Any]]) -> None:
        """Persist adjustments to .claude/config_adjustments.json with file locking."""
        self.adjustments_path.parent.mkdir(parents=True, exist_ok=True)

        # Use file locking to prevent race conditions in concurrent access
        lock_file = self.adjustments_path.with_suffix(".lock")
        try:
            with open(str(lock_file), "w") as lock_f:
                fcntl.flock(lock_f.fileno(), fcntl.LOCK_EX)
                try:
                    # Load existing adjustments
                    existing: List[Dict[str, Any]] = []
                    if self.adjustments_path.exists():
                        try:
                            existing = json.loads(self.adjustments_path.read_text())
                        except Exception:
                            pass

                    # Add timestamp and append new adjustments
                    now = datetime.now(timezone.utc).isoformat()
                    for adj in adjustments:
                        adj["timestamp"] = now
                        existing.append(adj)
                        logger.info(
                            "Config adjustment: %s %.4f -> %.4f (rule: %s)",
                            adj["field"], adj["old"], adj["new"], adj.get("rule", "")[:60],
                        )

                    # Write back atomically
                    self.adjustments_path.write_text(json.dumps(existing, indent=2))
                finally:
                    fcntl.flock(lock_f.fileno(), fcntl.LOCK_UN)
        finally:
            # Clean up lock file
            try:
                lock_file.unlink(missing_ok=True)
            except Exception:
                pass
