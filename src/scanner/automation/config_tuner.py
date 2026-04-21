"""Adaptive config tuner that reads promoted rules and adjusts scanner config.

Closes the learning loop: trade outcome -> learning -> rule -> config change -> better trades.

US-005: Create adaptive config tuner that reads rules and adjusts scanner config.
"""

from __future__ import annotations

import json
import logging
import re
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from src.scanner.config import ScannerConfig

logger = logging.getLogger(__name__)

RULES_PATH = Path(".claude/rules/trading.md")
ADJUSTMENTS_PATH = Path(".claude/config_adjustments.json")
PENDING_PATH = Path(".claude/pending_adjustments.json")

# Bounds to prevent runaway tuning
BOUNDS = {
    "atr_sl_multiplier": (0.5, 2.0),
    "atr_tp_multiplier": (0.8, 3.0),
    "max_uncertainty_score": (0.30, 0.95),
    "weighted_vote_threshold": (0.45, 0.95),
    "max_model_disagreement": (0.20, 0.95),
}


class ConfigTuner:
    """Reads promoted rules and applies config adjustments."""

    def __init__(
        self,
        rules_path: Optional[Path] = None,
        adjustments_path: Optional[Path] = None,
        pending_path: Optional[Path] = None,
    ):
        self.rules_path = rules_path or RULES_PATH
        self.adjustments_path = adjustments_path or ADJUSTMENTS_PATH
        self.pending_path = pending_path or PENDING_PATH
        self._applied_rules: set[str] = set()

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
            # Skip already applied rules
            rule_hash = hash(rule)
            if rule_hash in self._applied_rules:
                continue

            adj = self._apply_rule(rule, config)
            if adj:
                adjustments.append(adj)
                self._applied_rules.add(rule_hash)

        if adjustments:
            self._log_adjustments(adjustments)

        return adjustments

    def _apply_rule(self, rule: str, config: "ScannerConfig") -> Optional[Dict[str, Any]]:
        """Parse a single rule and apply to config. Returns adjustment dict or None."""
        rule_lower = rule.lower()

        # Pattern: "Increase atr_sl_multiplier by 0.1 for {pair}"
        sl_match = re.search(r"increase atr_sl_multiplier.*?for (\w+)", rule_lower)
        if sl_match:
            # Global adjustment (per-pair is handled by pair_sl_tp_config.json)
            old = config.atr_sl_multiplier
            new = min(BOUNDS["atr_sl_multiplier"][1], old + 0.1)
            if new != old:
                config.atr_sl_multiplier = new
                return {"field": "atr_sl_multiplier", "old": old, "new": new, "rule": rule}

        # Pattern: "Increase atr_tp_multiplier by 0.1 for {pair}"
        tp_match = re.search(r"increase atr_tp_multiplier.*?for (\w+)", rule_lower)
        if tp_match:
            old = config.atr_tp_multiplier
            new = min(BOUNDS["atr_tp_multiplier"][1], old + 0.1)
            if new != old:
                config.atr_tp_multiplier = new
                return {"field": "atr_tp_multiplier", "old": old, "new": new, "rule": rule}

        # Pattern: "Lower max_uncertainty_score by 0.02"
        if "lower max_uncertainty_score" in rule_lower:
            old = config.max_uncertainty_score
            new = max(BOUNDS["max_uncertainty_score"][0], old - 0.02)
            if new != old:
                config.max_uncertainty_score = new
                return {"field": "max_uncertainty_score", "old": old, "new": new, "rule": rule}

        # Pattern: "Prefer weighted_vote_score > 0.7" / "Lower weighted_vote_threshold"
        if "weighted_vote" in rule_lower and "prefer" in rule_lower:
            old = config.weighted_vote_threshold
            new = max(BOUNDS["weighted_vote_threshold"][0], old - 0.01)
            if new != old:
                config.weighted_vote_threshold = new
                return {"field": "weighted_vote_threshold", "old": old, "new": new, "rule": rule}

        # Pattern: "Lower max_model_disagreement by 0.02"
        if "lower max_model_disagreement" in rule_lower:
            old = config.max_model_disagreement
            new = max(BOUNDS["max_model_disagreement"][0], old - 0.02)
            if new != old:
                config.max_model_disagreement = new
                return {"field": "max_model_disagreement", "old": old, "new": new, "rule": rule}

        return None

    def propose_adjustment(self, adj: Dict[str, Any]) -> str:
        """Write one adjustment as a proposal to pending_adjustments.json.

        Returns the proposal_id. Validates the key against ScannerConfig before
        writing — orphan keys are written with status='invalid' so they appear
        in the audit log but are never applied.
        """
        from src.scanner.automation.adjustment_validator import validate_adjustment

        key = adj.get("field") or adj.get("key", "")
        value = adj.get("new", adj.get("new_value"))
        old_value = adj.get("old", adj.get("old_value"))
        reason = str(adj.get("rule", adj.get("reason", "rule-based adjustment")))[:200]

        validation = validate_adjustment(key, value, reason)

        proposal_id = str(uuid.uuid4())
        now = datetime.now(timezone.utc).isoformat()
        proposal = {
            "id": proposal_id,
            "timestamp": now,
            "key": key,
            "current_value": old_value,
            "proposed_value": value,
            "reason": reason,
            "source": "config_tuner",
            "validation": validation.to_dict(),
            "status": "pending" if validation.valid else "invalid",
            "snooze_until": None,
        }

        # Atomic write to pending_adjustments.json
        data: Dict[str, Any] = {"proposals": []}
        if self.pending_path.exists():
            try:
                data = json.loads(self.pending_path.read_text())
            except Exception:
                data = {"proposals": []}

        data["proposals"].append(proposal)
        self.pending_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.pending_path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(data, indent=2))
        tmp.rename(self.pending_path)

        # Emit event (best-effort)
        try:
            from src.scanner.automation.event_bus import get_event_bus
            get_event_bus().publish("adjustment.proposed", {
                "proposal_id": proposal_id,
                "key": key,
                "value": value,
                "source": "config_tuner",
                "valid": validation.valid,
            })
        except Exception:
            pass

        return proposal_id

    def _log_adjustments(self, adjustments: List[Dict[str, Any]]) -> None:
        """Route each rule-based adjustment to pending_adjustments.json (not config_adjustments.json).

        Previously wrote directly to config_adjustments.json (US-508 retire).
        Now writes proposals that require operator approval via AdjustmentApprover.
        """
        now = datetime.now(timezone.utc).isoformat()
        for adj in adjustments:
            adj["timestamp"] = now
            proposal_id = self.propose_adjustment(adj)
            logger.info(
                "Config proposal %s: %s %.4f -> %.4f (rule: %s)",
                proposal_id[:8], adj["field"], adj["old"], adj["new"],
                adj.get("rule", "")[:60],
            )
