"""Central config adjustment manager with persistence.

US-131: Collects threshold recommendations from ThresholdOptimizer,
ObservationConsumer, and DriftMonitor, persists them, and applies
them to the running ScannerConfig each cycle.

Approach:
1. Modules call collect_adjustment(source, key, value, reason)
2. Rate-limiter prevents more than 1 change per key per 10 cycles
3. apply_adjustments() writes overrides into ScannerConfig attrs
4. All adjustments persisted to .claude/config_adjustments.json
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
_ADJUSTMENTS_PATH = _PROJECT_ROOT / ".claude" / "config_adjustments.json"

# Rate limit: max 1 change per key per N cycles
RATE_LIMIT_CYCLES = 10


class ConfigAdjuster:
    """Collects, rate-limits, persists, and applies config adjustments.

    Usage:
        adjuster = ConfigAdjuster()
        adjuster.collect_adjustment("threshold_optimizer", "min_confidence", 0.48, "win_rate below target in HIGH regime")
        adjuster.apply_adjustments(scanner_config, current_cycle=42)
    """

    def __init__(self, persistence_path: Optional[Path] = None):
        self.persistence_path = persistence_path or _ADJUSTMENTS_PATH

        # Pending adjustments: {key: {source, value, reason, cycle_proposed}}
        self._pending: Dict[str, Dict[str, Any]] = {}

        # Applied history: [{key, old_value, new_value, source, reason, cycle, timestamp}]
        self._history: List[Dict[str, Any]] = []

        # Rate limiter: {key: last_applied_cycle}
        self._last_applied: Dict[str, int] = {}

        self._load_state()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def collect_adjustment(
        self,
        source: str,
        key: str,
        value: Any,
        reason: str,
        cycle: int = 0,
    ) -> None:
        """Propose an adjustment from any module.

        Args:
            source: Module name (e.g., "threshold_optimizer", "drift_monitor").
            key: Config key to adjust (e.g., "min_confidence").
            value: New value to set.
            reason: Human-readable reason for the change.
            cycle: Current scan cycle number.
        """
        self._pending[key] = {
            "source": source,
            "value": value,
            "reason": reason,
            "cycle_proposed": cycle,
        }

    def apply_adjustments(self, config: Any, current_cycle: int = 0) -> List[Dict[str, Any]]:
        """Apply all pending adjustments to config, respecting rate limits.

        Args:
            config: ScannerConfig instance to modify.
            current_cycle: Current scan cycle for rate limiting.

        Returns:
            List of adjustments that were actually applied.
        """
        applied = []

        for key, adj in list(self._pending.items()):
            # Rate limit check
            last_cycle = self._last_applied.get(key, -RATE_LIMIT_CYCLES - 1)
            if current_cycle - last_cycle < RATE_LIMIT_CYCLES:
                continue

            # Get old value for logging
            old_value = getattr(config, key, None)
            new_value = adj["value"]

            # Skip if value unchanged
            if old_value == new_value:
                del self._pending[key]
                continue

            # Apply to config
            try:
                setattr(config, key, new_value)
                record = {
                    "key": key,
                    "old_value": _safe_serialize(old_value),
                    "new_value": _safe_serialize(new_value),
                    "source": adj["source"],
                    "reason": adj["reason"],
                    "cycle": current_cycle,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                }
                applied.append(record)
                self._history.append(record)
                self._last_applied[key] = current_cycle

                logger.info(
                    "ConfigAdjuster: %s = %s → %s (source=%s, reason=%s)",
                    key, old_value, new_value, adj["source"], adj["reason"],
                )
            except Exception as e:
                logger.debug(f"ConfigAdjuster: Failed to apply {key}: {e}")

            # Remove from pending after attempt
            del self._pending[key]

        # Trim history
        if len(self._history) > 200:
            self._history = self._history[-100:]

        return applied

    def get_status(self) -> Dict[str, Any]:
        """Get adjuster status for observability."""
        return {
            "pending_count": len(self._pending),
            "total_applied": len(self._history),
            "recent": self._history[-5:],
            "pending": dict(self._pending),
        }

    def save_state(self) -> None:
        """Persist all adjustment history."""
        try:
            data = {
                "version": 1,
                "total_adjustments": len(self._history),
                "history": self._history[-100:],
                "pending": dict(self._pending),
                "last_applied": dict(self._last_applied),
                "last_updated": datetime.now(timezone.utc).isoformat(),
            }
            self.persistence_path.parent.mkdir(parents=True, exist_ok=True)
            try:
                from src.scanner.automation.safe_json import safe_json_write
                safe_json_write(self.persistence_path, data)
            except ImportError:
                self.persistence_path.write_text(json.dumps(data, indent=2))
        except Exception as e:
            logger.debug(f"ConfigAdjuster: Failed to save: {e}")

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _load_state(self) -> None:
        """Load previous state from disk."""
        if not self.persistence_path.exists():
            return
        try:
            try:
                from src.scanner.automation.safe_json import safe_json_read
                data = safe_json_read(self.persistence_path, default={})
            except ImportError:
                data = json.loads(self.persistence_path.read_text())

            if isinstance(data, dict):
                self._history = data.get("history", [])
                self._last_applied = data.get("last_applied", {})
                # Convert string cycle keys to str (JSON always serializes keys as strings)
                self._last_applied = {
                    k: int(v) for k, v in self._last_applied.items()
                    if isinstance(v, (int, float))
                }
        except Exception as e:
            logger.debug(f"ConfigAdjuster: Failed to load: {e}")


def _safe_serialize(value: Any) -> Any:
    """Make a value JSON-serializable."""
    if isinstance(value, (int, float, str, bool, type(None))):
        return value
    return str(value)
