"""Domain-agnostic adaptive config tuner.

Extracted from Buddy's ConfigTuner (src/scanner/automation/config_tuner.py).
Generalizes the rule → config adjustment → bounds check → rollback pipeline.

Buddy instantiation:
    config_schema = ScannerConfig fields
    bounds = {"atr_sl_multiplier": (0.5, 2.0), ...}
    rollback_window = 10 trades

Aura instantiation:
    config_schema = AuraConfig fields (insight_frequency, emotional_sensitivity, etc.)
    bounds = {"insight_frequency": (1, 7), "emotional_sensitivity": (0.1, 1.0), ...}
    rollback_window = 7 days
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


@dataclass
class ConfigAdjustment:
    """Record of a single config parameter change."""

    parameter: str
    old_value: float
    new_value: float
    reason: str
    source_rule: str
    timestamp: str
    domain: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "parameter": self.parameter,
            "old_value": self.old_value,
            "new_value": self.new_value,
            "reason": self.reason,
            "source_rule": self.source_rule,
            "timestamp": self.timestamp,
            "domain": self.domain,
        }


class AdaptiveTuner:
    """Domain-agnostic config tuner with bounds checking and rollback.

    Args:
        domain: Name of the domain ("market", "human", "bridge")
        bounds: Dict of parameter_name -> (min, max) tuples
        rollback_window: Number of observations before checking if adjustment helped
        adjustments_path: Where to persist adjustment history
    """

    def __init__(
        self,
        domain: str,
        bounds: Dict[str, Tuple[float, float]],
        rollback_window: int = 10,
        adjustments_path: Optional[Path] = None,
    ):
        self.domain = domain
        self.bounds = bounds
        self.rollback_window = rollback_window
        self.adjustments_path = adjustments_path or Path(
            f".aura/config_adjustments_{domain}.json"
        )
        self._history: List[ConfigAdjustment] = self._load_history()

    def apply_adjustment(
        self,
        config: Dict[str, Any],
        parameter: str,
        new_value: float,
        reason: str,
        source_rule: str = "",
    ) -> Optional[ConfigAdjustment]:
        """Apply a config adjustment with bounds checking.

        Args:
            config: Current config dict (modified in place)
            parameter: Parameter name to adjust
            new_value: Proposed new value
            reason: Why this adjustment is being made
            source_rule: The rule that triggered this adjustment

        Returns:
            ConfigAdjustment record if applied, None if rejected by bounds.
        """
        if parameter not in self.bounds:
            logger.warning(f"[{self.domain}] No bounds defined for '{parameter}', skipping")
            return None

        min_val, max_val = self.bounds[parameter]
        clamped = max(min_val, min(max_val, new_value))

        old_value = config.get(parameter, 0.0)
        if abs(clamped - old_value) < 1e-6:
            return None  # No meaningful change

        config[parameter] = clamped

        adjustment = ConfigAdjustment(
            parameter=parameter,
            old_value=old_value,
            new_value=clamped,
            reason=reason,
            source_rule=source_rule,
            timestamp=datetime.now(timezone.utc).isoformat(),
            domain=self.domain,
        )

        self._history.append(adjustment)
        self._persist_history()

        logger.info(
            f"[{self.domain}] Adjusted {parameter}: {old_value:.4f} → {clamped:.4f} ({reason})"
        )
        return adjustment

    def check_rollback(
        self,
        parameter: str,
        performance_metric: float,
        baseline_metric: float,
        config: Dict[str, Any],
    ) -> bool:
        """Check if a recent adjustment degraded performance and rollback if so.

        Args:
            parameter: The parameter to check
            performance_metric: Current performance value
            baseline_metric: Pre-adjustment performance value
            config: Config dict to rollback (modified in place if rolling back)

        Returns:
            True if rollback was performed.
        """
        recent = [a for a in self._history if a.parameter == parameter]
        if not recent:
            return False

        last_adjustment = recent[-1]

        if performance_metric < baseline_metric * 0.9:  # 10% degradation threshold
            # Rollback
            config[parameter] = last_adjustment.old_value
            logger.warning(
                f"[{self.domain}] Rolling back {parameter}: "
                f"{last_adjustment.new_value:.4f} → {last_adjustment.old_value:.4f} "
                f"(performance degraded: {performance_metric:.4f} < {baseline_metric:.4f})"
            )
            return True

        return False

    def get_history(self, parameter: Optional[str] = None) -> List[ConfigAdjustment]:
        """Return adjustment history, optionally filtered by parameter."""
        if parameter:
            return [a for a in self._history if a.parameter == parameter]
        return list(self._history)

    def _load_history(self) -> List[ConfigAdjustment]:
        if not self.adjustments_path.exists():
            return []
        try:
            data = json.loads(self.adjustments_path.read_text())
            return [ConfigAdjustment(**a) for a in data if isinstance(a, dict)]
        except Exception as e:
            logger.warning(f"Failed to load adjustment history: {e}")
            return []

    def _persist_history(self) -> None:
        self.adjustments_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            data = [a.to_dict() for a in self._history[-100:]]  # Keep last 100
            self.adjustments_path.write_text(json.dumps(data, indent=2, default=str))
        except Exception as e:
            logger.error(f"Failed to persist adjustment history: {e}")
