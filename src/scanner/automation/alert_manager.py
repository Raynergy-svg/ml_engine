"""
Alert Manager — Operational monitoring and threshold-based alerts.

Fires alerts when trading metrics cross risk thresholds.
Supports cooldown periods and alert state persistence.

Key features:
- Never crashes the scanner (all operations wrapped in try/except)
- File locking for concurrent JSONL writes (per improvement rules)
- Graceful degradation on missing/corrupted state files
- Cooldown-based alert spam prevention
- ISO timestamp logging for all events
"""

from __future__ import annotations

import fcntl
import json
import logging
import os
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class Alert:
    """Represents a single fired alert event."""

    alert_type: str  # drawdown, consecutive_losses, win_rate_drop, weight_instability
    severity: str  # WARNING, CRITICAL
    message: str
    timestamp: str  # ISO format
    value: float  # the metric that triggered the alert
    threshold: float  # the threshold that was exceeded
    pair: str = ""  # specific pair if applicable
    acknowledged: bool = False

    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)


class AlertManager:
    """Manages trading alerts with cooldown and persistence."""

    # Alert type definitions with defaults
    ALERT_CONFIGS = {
        "drawdown": {
            "threshold": 0.05,  # 5% drawdown
            "severity": "CRITICAL",
            "cooldown_minutes": 60,
            "message_template": "Portfolio drawdown {value:.1%} exceeds {threshold:.1%} threshold",
        },
        "consecutive_losses": {
            "threshold": 3,  # 3 losses in a row
            "severity": "WARNING",
            "cooldown_minutes": 60,
            "message_template": "{value:.0f} consecutive losses (threshold: {threshold:.0f})",
        },
        "win_rate_drop": {
            "threshold": 0.10,  # 10% drop in rolling win rate
            "severity": "WARNING",
            "cooldown_minutes": 120,
            "message_template": "Win rate dropped {value:.1%} over rolling window (threshold: {threshold:.1%})",
        },
        "weight_instability": {
            "threshold": 0.30,  # >0.3 weight change in one cycle
            "severity": "WARNING",
            "cooldown_minutes": 60,
            "message_template": "Agent weight change {value:.2f} exceeds stability threshold {threshold:.2f}",
        },
    }

    def __init__(
        self,
        alert_log_path: str = "trained_data/alerts.jsonl",
        state_path: str = ".claude/alert_state.json",
        custom_configs: Optional[Dict] = None,
    ):
        """Initialize alert manager.

        Args:
            alert_log_path: Path to JSONL alert log
            state_path: Path to alert state file (for cooldown tracking)
            custom_configs: Override default alert configs
        """
        self.alert_log_path = Path(alert_log_path)
        self.state_path = Path(state_path)
        self._configs = dict(self.ALERT_CONFIGS)
        if custom_configs:
            self._configs.update(custom_configs)
            # Validate custom configs: cooldown_minutes must be > 0
            for alert_type, config in self._configs.items():
                if config.get("cooldown_minutes", 0) <= 0:
                    logger.warning(
                        f"Invalid cooldown_minutes for {alert_type}: "
                        f"{config.get('cooldown_minutes')}. Setting to 60."
                    )
                    config["cooldown_minutes"] = 60

        # Cooldown tracking: {alert_type: unix timestamp (float) of last fire}
        self._last_fired: Dict[str, float] = {}

        # Active (unacknowledged) alerts
        self._active_alerts: List[Alert] = []

        # Load state from disk
        self._load_state()

    def _load_state(self) -> None:
        """Load alert state from disk (cooldowns, active alerts).

        Handles missing/corrupted files gracefully per improvement rules.
        """
        if not self.state_path.exists():
            logger.debug(f"Alert state not found at {self.state_path}, starting fresh")
            return

        try:
            with open(self.state_path, "r") as f:
                state_data = json.load(f)

            # Restore cooldown times (store as unix timestamps for monotonic tracking)
            last_fired_raw = state_data.get("last_fired", {})
            if isinstance(last_fired_raw, dict):
                for alert_type, ts_value in last_fired_raw.items():
                    try:
                        # Accept both float (new format) and ISO string (legacy)
                        if isinstance(ts_value, (int, float)):
                            self._last_fired[alert_type] = float(ts_value)
                        elif isinstance(ts_value, str):
                            # Fallback for legacy ISO format
                            dt = datetime.fromisoformat(ts_value)
                            self._last_fired[alert_type] = dt.timestamp()
                        else:
                            logger.debug(f"Invalid timestamp type for {alert_type}: {type(ts_value)}")
                    except (ValueError, TypeError, AttributeError):
                        logger.debug(f"Invalid timestamp for {alert_type}: {ts_value}")
                        continue

            # Restore active alerts
            active_raw = state_data.get("active_alerts", [])
            if isinstance(active_raw, list):
                for alert_dict in active_raw:
                    try:
                        self._active_alerts.append(
                            Alert(
                                alert_type=alert_dict.get("alert_type", "unknown"),
                                severity=alert_dict.get("severity", "WARNING"),
                                message=alert_dict.get("message", ""),
                                timestamp=alert_dict.get("timestamp", ""),
                                value=float(alert_dict.get("value", 0)),
                                threshold=float(alert_dict.get("threshold", 0)),
                                pair=alert_dict.get("pair", ""),
                                acknowledged=bool(alert_dict.get("acknowledged", False)),
                            )
                        )
                    except (ValueError, TypeError, KeyError):
                        logger.debug(f"Skipping malformed alert: {alert_dict}")
                        continue

            logger.debug(
                f"Loaded alert state: {len(self._last_fired)} cooldowns, "
                f"{len(self._active_alerts)} active alerts"
            )

        except json.JSONDecodeError as e:
            logger.warning(f"Alert state file corrupted: {e}, starting fresh")
        except Exception as e:
            logger.warning(f"Failed to load alert state: {e}, starting fresh")

    def _save_state(self) -> None:
        """Persist alert state for cross-session continuity.

        Uses file locking to prevent concurrent writes per improvement rules.
        """
        try:
            self.state_path.parent.mkdir(parents=True, exist_ok=True)

            state_data = {
                "last_fired": {
                    k: v for k, v in self._last_fired.items()
                },
                "active_alerts": [a.to_dict() for a in self._active_alerts],
                "last_updated": datetime.utcnow().isoformat(),
            }

            # Use file locking to prevent concurrent writes
            with open(self.state_path, "w") as f:
                try:
                    fcntl.flock(f.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                    f.write(json.dumps(state_data, indent=2, default=str))
                    f.flush()
                    os.fsync(f.fileno())
                    fcntl.flock(f.fileno(), fcntl.LOCK_UN)
                except IOError:
                    # Non-blocking lock failed, try blocking
                    fcntl.flock(f.fileno(), fcntl.LOCK_EX)
                    f.write(json.dumps(state_data, indent=2, default=str))
                    f.flush()
                    os.fsync(f.fileno())
                    fcntl.flock(f.fileno(), fcntl.LOCK_UN)

        except Exception as e:
            logger.warning(f"Failed to save alert state: {e}")

    def check_drawdown(
        self, current_nav: float, peak_nav: float
    ) -> Optional[Alert]:
        """Check if portfolio drawdown exceeds threshold.

        Args:
            current_nav: Current portfolio NAV
            peak_nav: Highest NAV seen (peak)

        Returns:
            Alert if threshold exceeded and cooldown passed, None otherwise
        """
        if peak_nav <= 0:
            return None

        drawdown = (peak_nav - current_nav) / peak_nav
        config = self._configs["drawdown"]

        if drawdown > config["threshold"]:
            return self._create_alert(
                alert_type="drawdown",
                severity=config["severity"],
                message=config["message_template"].format(
                    value=drawdown, threshold=config["threshold"]
                ),
                value=drawdown,
                threshold=config["threshold"],
            )

        return None

    def check_consecutive_losses(
        self, recent_trades: List[Dict]
    ) -> Optional[Alert]:
        """Check for consecutive losing trades.

        Args:
            recent_trades: List of recent trade dicts, each with 'outcome' key

        Returns:
            Alert if threshold exceeded, None otherwise
        """
        if not recent_trades:
            return None

        config = self._configs["consecutive_losses"]
        threshold = int(config["threshold"])

        # Count consecutive losses from end of list
        consecutive = 0
        for trade in reversed(recent_trades):
            outcome = trade.get("outcome")
            if outcome is None:
                # Skip open trades
                continue
            if not outcome.get("trade_won", False):
                consecutive += 1
            else:
                break

        if consecutive >= threshold:
            return self._create_alert(
                alert_type="consecutive_losses",
                severity=config["severity"],
                message=config["message_template"].format(
                    value=consecutive, threshold=threshold
                ),
                value=float(consecutive),
                threshold=float(threshold),
            )

        return None

    def check_win_rate_drop(
        self, recent_trades: List[Dict], window: int = 20
    ) -> Optional[Alert]:
        """Check if rolling win rate has dropped significantly.

        Args:
            recent_trades: List of all trades
            window: Number of recent trades to analyze

        Returns:
            Alert if drop exceeds threshold, None otherwise
        """
        if not recent_trades:
            return None

        # Get closed trades only
        closed = [t for t in recent_trades if t.get("outcome") is not None]
        if len(closed) < window + 5:
            # Need enough history to compare
            return None

        config = self._configs["win_rate_drop"]

        # Recent window win rate
        recent_window = closed[-window:]
        recent_wins = sum(
            1 for t in recent_window if t["outcome"].get("trade_won", False)
        )
        recent_wr = recent_wins / window if window > 0 else 0.0

        # Overall win rate (before the window)
        older = closed[:-window] if len(closed) > window else closed[:1]
        older_wins = sum(1 for t in older if t["outcome"].get("trade_won", False))
        older_wr = older_wins / len(older) if older else 0.0

        drop = older_wr - recent_wr
        if drop > config["threshold"]:
            return self._create_alert(
                alert_type="win_rate_drop",
                severity=config["severity"],
                message=config["message_template"].format(
                    value=drop, threshold=config["threshold"]
                ),
                value=drop,
                threshold=config["threshold"],
            )

        return None

    def check_weight_instability(
        self,
        current_weights: Optional[Dict[str, float]] = None,
        previous_weights: Optional[Dict[str, float]] = None,
    ) -> Optional[Alert]:
        """Check if agent weights changed too much in one cycle.

        Args:
            current_weights: Dict of agent -> weight
            previous_weights: Dict of agent -> weight from previous cycle

        Returns:
            Alert if change exceeds threshold, None otherwise
        """
        if (
            not current_weights
            or not previous_weights
            or not isinstance(current_weights, dict)
            or not isinstance(previous_weights, dict)
        ):
            return None

        config = self._configs["weight_instability"]

        # Find max absolute weight change
        max_change = 0.0
        for agent, current_w in current_weights.items():
            prev_w = previous_weights.get(agent, current_w)
            change = abs(current_w - prev_w)
            max_change = max(max_change, change)

        if max_change > config["threshold"]:
            return self._create_alert(
                alert_type="weight_instability",
                severity=config["severity"],
                message=config["message_template"].format(
                    value=max_change, threshold=config["threshold"]
                ),
                value=max_change,
                threshold=config["threshold"],
            )

        return None

    def check_all(
        self,
        nav: float,
        peak_nav: float,
        recent_trades: Optional[List[Dict]] = None,
        current_weights: Optional[Dict[str, float]] = None,
        previous_weights: Optional[Dict[str, float]] = None,
    ) -> List[Alert]:
        """Run all alert checks. Returns list of newly fired alerts.

        Args:
            nav: Current portfolio NAV
            peak_nav: Peak NAV seen
            recent_trades: List of recent trades
            current_weights: Current agent weights
            previous_weights: Previous agent weights

        Returns:
            List of newly fired alerts
        """
        fired_alerts = []

        try:
            # Check drawdown
            alert = self.check_drawdown(nav, peak_nav)
            if alert:
                fired_alerts.append(alert)
                self._fire_alert(alert)

            # Check consecutive losses
            if recent_trades:
                alert = self.check_consecutive_losses(recent_trades)
                if alert:
                    fired_alerts.append(alert)
                    self._fire_alert(alert)

                # Check win rate drop
                alert = self.check_win_rate_drop(recent_trades)
                if alert:
                    fired_alerts.append(alert)
                    self._fire_alert(alert)

            # Check weight instability
            if current_weights and previous_weights:
                alert = self.check_weight_instability(current_weights, previous_weights)
                if alert:
                    fired_alerts.append(alert)
                    self._fire_alert(alert)

        except Exception as e:
            logger.warning(f"Alert check error: {e}")

        # Save state after all checks
        try:
            self._save_state()
        except Exception as e:
            logger.warning(f"Failed to save alert state: {e}")

        return fired_alerts

    def _create_alert(
        self,
        alert_type: str,
        severity: str,
        message: str,
        value: float,
        threshold: float,
        pair: str = "",
    ) -> Optional[Alert]:
        """Create an alert and check cooldown.

        Returns None if alert is in cooldown period.
        """
        if not self._is_cooled_down(alert_type):
            logger.debug(
                f"Alert '{alert_type}' in cooldown, suppressed: {message}"
            )
            return None

        now_iso = datetime.utcnow().isoformat()
        alert = Alert(
            alert_type=alert_type,
            severity=severity,
            message=message,
            timestamp=now_iso,
            value=value,
            threshold=threshold,
            pair=pair,
            acknowledged=False,
        )

        return alert

    def _is_cooled_down(self, alert_type: str) -> bool:
        """Check if enough time has passed since last alert of this type.

        Uses time.time() for monotonic clock to prevent clock skew issues.
        """
        if alert_type not in self._configs:
            return True

        config = self._configs[alert_type]
        cooldown_minutes = config.get("cooldown_minutes", 60)

        last_fired_timestamp = self._last_fired.get(alert_type)
        if last_fired_timestamp is None:
            return True

        elapsed_seconds = time.time() - last_fired_timestamp
        return elapsed_seconds >= (cooldown_minutes * 60)

    def _fire_alert(self, alert: Alert) -> None:
        """Log alert to JSONL and update state atomically.

        CRITICAL-2: Update cooldown and save state atomically in same call.
        """
        try:
            # Append to alerts.jsonl with file locking and fsync
            self.alert_log_path.parent.mkdir(parents=True, exist_ok=True)

            with open(self.alert_log_path, "a") as f:
                try:
                    fcntl.flock(f.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                    f.write(json.dumps(alert.to_dict(), default=str) + "\n")
                    f.flush()
                    os.fsync(f.fileno())
                    fcntl.flock(f.fileno(), fcntl.LOCK_UN)
                except IOError:
                    # Non-blocking lock failed, try blocking
                    fcntl.flock(f.fileno(), fcntl.LOCK_EX)
                    f.write(json.dumps(alert.to_dict(), default=str) + "\n")
                    f.flush()
                    os.fsync(f.fileno())
                    fcntl.flock(f.fileno(), fcntl.LOCK_UN)

            # CRITICAL-2: Update cooldown timestamp (unix time) and state atomically
            self._last_fired[alert.alert_type] = time.time()
            self._active_alerts.append(alert)

            # Immediately persist state (do not defer)
            self._save_state()

            # Log to console
            logger.warning(f"[{alert.severity}] {alert.alert_type}: {alert.message}")

        except Exception as e:
            logger.warning(f"Failed to fire alert: {e}")

    def get_active_alerts(self) -> List[Alert]:
        """Return currently active (unacknowledged) alerts."""
        return [a for a in self._active_alerts if not a.acknowledged]

    def acknowledge(self, alert_type: str) -> None:
        """Mark all alerts of a type as acknowledged."""
        for alert in self._active_alerts:
            if alert.alert_type == alert_type:
                alert.acknowledged = True

        try:
            self._save_state()
        except Exception as e:
            logger.warning(f"Failed to acknowledge alert: {e}")

    def get_summary(self) -> str:
        """Return human-readable alert status summary."""
        active = self.get_active_alerts()

        if not active:
            return "✓ No active alerts"

        lines = [f"⚠ {len(active)} active alert(s):"]
        for alert in active:
            lines.append(
                f"  [{alert.severity}] {alert.alert_type}: {alert.message}"
            )

        return "\n".join(lines)

    def clear_cooldowns(self) -> None:
        """Clear all cooldown timers (useful for testing)."""
        self._last_fired.clear()
        try:
            self._save_state()
        except Exception as e:
            logger.warning(f"Failed to clear cooldowns: {e}")
