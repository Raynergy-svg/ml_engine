"""
Walk-Forward Efficiency Stability Monitor — Phase 53 (US-332).

Tracks WFE over recent retraining cycles and detects declining trends.
When WFE drops >5% per cycle for 3+ consecutive cycles, pauses agent
weight retraining and freezes weights from the last stable cycle.
Resumes retraining when WFE trends positive for 2+ consecutive cycles.

States:
  STABLE    — WFE trending flat or up; retraining allowed
  UNSTABLE  — WFE declining; retraining paused, using frozen weights
  RECOVERING — WFE recovering but not yet confirmed; retraining still paused

Usage:
    monitor = WFEStabilityMonitor()
    monitor.record_wfe(0.72)   # After each walk-forward cycle
    if monitor.should_retrain():
        # Proceed with weight retraining
    else:
        # Use frozen weights from last stable cycle
"""

import json
import logging
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

_DEFAULT_PERSISTENCE_PATH = "trained_data/wfe_history.json"


@dataclass
class WFEMonitorConfig:
    """Configuration for WFE stability monitoring."""
    max_history: int = 20              # Keep last N WFE values
    decline_threshold: float = 0.05    # >5% drop per cycle = declining
    decline_cycles_trigger: int = 3    # 3+ consecutive declining cycles = UNSTABLE
    recovery_cycles_required: int = 2  # 2+ positive cycles = back to STABLE
    persistence_path: str = _DEFAULT_PERSISTENCE_PATH


@dataclass
class WFESnapshot:
    """Single WFE measurement."""
    wfe: float = 0.0
    timestamp: str = ""
    epoch_count: int = 0
    accepted_count: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "wfe": round(self.wfe, 4),
            "timestamp": self.timestamp,
            "epoch_count": self.epoch_count,
            "accepted_count": self.accepted_count,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "WFESnapshot":
        return cls(
            wfe=float(d.get("wfe", 0.0)),
            timestamp=str(d.get("timestamp", "")),
            epoch_count=int(d.get("epoch_count", 0)),
            accepted_count=int(d.get("accepted_count", 0)),
        )


class WFEStabilityMonitor:
    """Monitors walk-forward efficiency and controls retraining permission.

    Tracks WFE over cycles. Detects declining trends and pauses retraining
    when the signal is unreliable (usually during market regime shifts).

    States:
        STABLE: WFE is flat or improving — retraining allowed
        UNSTABLE: WFE declining for 3+ cycles — retraining paused
        RECOVERING: WFE recovering but not yet confirmed stable
    """

    def __init__(self, config: Optional[WFEMonitorConfig] = None):
        self.config = config or WFEMonitorConfig()
        self._history: List[WFESnapshot] = []
        self._state: str = "STABLE"  # STABLE | UNSTABLE | RECOVERING
        self._frozen_weights: Optional[Dict[str, float]] = None
        self._consecutive_declines: int = 0
        self._consecutive_recoveries: int = 0
        self._load_state()

    @property
    def state(self) -> str:
        """Current stability state."""
        return self._state

    @property
    def history(self) -> List[WFESnapshot]:
        """WFE history (read-only copy)."""
        return list(self._history)

    def record_wfe(
        self,
        wfe: float,
        epoch_count: int = 0,
        accepted_count: int = 0,
    ) -> str:
        """Record a new WFE measurement and update stability state.

        Args:
            wfe: Walk-Forward Efficiency from latest retraining cycle
            epoch_count: Number of epochs in this cycle
            accepted_count: Number of accepted epochs

        Returns:
            Current state after update: "STABLE", "UNSTABLE", or "RECOVERING"
        """
        snapshot = WFESnapshot(
            wfe=wfe,
            timestamp=datetime.now(timezone.utc).isoformat(),
            epoch_count=epoch_count,
            accepted_count=accepted_count,
        )
        self._history.append(snapshot)

        # Trim to max_history
        if len(self._history) > self.config.max_history:
            self._history = self._history[-self.config.max_history:]

        # Update trend analysis
        self._update_state()

        logger.info(
            "Phase 53 (US-332): WFE recorded=%.4f state=%s "
            "consecutive_declines=%d consecutive_recoveries=%d",
            wfe, self._state, self._consecutive_declines,
            self._consecutive_recoveries,
        )

        return self._state

    def should_retrain(self) -> bool:
        """Check if agent weight retraining should proceed.

        Returns True only when state is STABLE.
        When UNSTABLE or RECOVERING, frozen weights should be used.
        """
        return self._state == "STABLE"

    def get_frozen_weights(self) -> Optional[Dict[str, float]]:
        """Return frozen weights from last stable cycle, if available."""
        return dict(self._frozen_weights) if self._frozen_weights else None

    def freeze_weights(self, weights: Dict[str, float]) -> None:
        """Freeze current weights as fallback for unstable periods."""
        self._frozen_weights = dict(weights)
        logger.info(
            "Phase 53 (US-332): Weights frozen — %d agents, state=%s",
            len(weights), self._state,
        )

    def get_trend_summary(self) -> Dict[str, Any]:
        """Get current trend analysis summary.

        Returns:
            {state, wfe_current, wfe_previous, trend_direction,
             consecutive_declines, consecutive_recoveries, history_length}
        """
        current_wfe = self._history[-1].wfe if self._history else 0.0
        previous_wfe = self._history[-2].wfe if len(self._history) >= 2 else 0.0

        if len(self._history) >= 2:
            delta = current_wfe - previous_wfe
            if delta > self.config.decline_threshold * previous_wfe if previous_wfe > 0 else False:
                direction = "improving"
            elif delta < -self.config.decline_threshold * (previous_wfe if previous_wfe > 0 else 1.0):
                direction = "declining"
            else:
                direction = "stable"
        else:
            direction = "insufficient_data"

        return {
            "state": self._state,
            "wfe_current": round(current_wfe, 4),
            "wfe_previous": round(previous_wfe, 4),
            "trend_direction": direction,
            "consecutive_declines": self._consecutive_declines,
            "consecutive_recoveries": self._consecutive_recoveries,
            "history_length": len(self._history),
            "has_frozen_weights": self._frozen_weights is not None,
        }

    def save_state(self) -> None:
        """Persist WFE history and monitor state with retry (US-342)."""
        path = self.config.persistence_path
        try:
            os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
            data = {
                "version": 1,
                "updated_at": datetime.now(timezone.utc).isoformat(),
                "state": self._state,
                "consecutive_declines": self._consecutive_declines,
                "consecutive_recoveries": self._consecutive_recoveries,
                "history": [s.to_dict() for s in self._history[-self.config.max_history:]],
                "frozen_weights": self._frozen_weights,
            }
            try:
                from src.scanner.safe_io import resilient_save
                resilient_save(path, data, context="wfe_monitor")
            except ImportError:
                tmp_path = path + ".tmp"
                with open(tmp_path, "w") as f:
                    json.dump(data, f, indent=2, sort_keys=True)
                os.rename(tmp_path, path)
            logger.info(
                "Phase 53 (US-332): WFE monitor state saved — %d entries, state=%s",
                len(self._history), self._state,
            )
        except Exception as e:
            logger.error("Phase 53 (US-332): Failed to save WFE monitor state: %s", e)

    # ── Private methods ──────────────────────────────────────────────

    def _update_state(self) -> None:
        """Analyze WFE trend and transition between states."""
        if len(self._history) < 2:
            return  # Not enough data to detect trends

        current = self._history[-1].wfe
        previous = self._history[-2].wfe

        # Compute relative change
        if previous > 0:
            pct_change = (current - previous) / previous
        else:
            pct_change = 0.0

        is_declining = pct_change < -self.config.decline_threshold
        is_improving = pct_change > 0

        if is_declining:
            self._consecutive_declines += 1
            self._consecutive_recoveries = 0
        elif is_improving:
            self._consecutive_recoveries += 1
            self._consecutive_declines = 0
        else:
            # Flat — don't reset counters but don't increment either
            pass

        # State transitions
        if self._state == "STABLE":
            if self._consecutive_declines >= self.config.decline_cycles_trigger:
                self._state = "UNSTABLE"
                logger.warning(
                    "Phase 53 (US-332): WFE UNSTABLE — %d consecutive declines "
                    "(threshold: %d). Retraining paused.",
                    self._consecutive_declines,
                    self.config.decline_cycles_trigger,
                )

        elif self._state == "UNSTABLE":
            if is_improving:
                self._state = "RECOVERING"
                logger.info(
                    "Phase 53 (US-332): WFE RECOVERING — first positive cycle detected"
                )

        elif self._state == "RECOVERING":
            if self._consecutive_recoveries >= self.config.recovery_cycles_required:
                self._state = "STABLE"
                self._consecutive_declines = 0
                logger.info(
                    "Phase 53 (US-332): WFE STABLE — %d consecutive positive cycles. "
                    "Retraining resumed.",
                    self._consecutive_recoveries,
                )
            elif is_declining:
                self._state = "UNSTABLE"
                self._consecutive_recoveries = 0
                logger.warning(
                    "Phase 53 (US-332): WFE relapsed to UNSTABLE during recovery"
                )

    def _load_state(self) -> None:
        """Load persisted WFE history."""
        path = self.config.persistence_path
        if not os.path.exists(path):
            return
        try:
            with open(path, "r") as f:
                data = json.load(f)
            if isinstance(data, dict):
                self._state = data.get("state", "STABLE")
                if self._state not in ("STABLE", "UNSTABLE", "RECOVERING"):
                    self._state = "STABLE"
                self._consecutive_declines = int(data.get("consecutive_declines", 0))
                self._consecutive_recoveries = int(data.get("consecutive_recoveries", 0))
                raw_history = data.get("history", [])
                self._history = [
                    WFESnapshot.from_dict(s) for s in raw_history
                    if isinstance(s, dict)
                ]
                frozen = data.get("frozen_weights")
                if isinstance(frozen, dict):
                    self._frozen_weights = {
                        k: float(v) for k, v in frozen.items()
                        if isinstance(v, (int, float))
                    }
                logger.info(
                    "Phase 53 (US-332): WFE monitor loaded — %d entries, state=%s",
                    len(self._history), self._state,
                )
        except (json.JSONDecodeError, OSError) as e:
            logger.warning("Phase 53 (US-332): Could not load WFE monitor state: %s", e)
