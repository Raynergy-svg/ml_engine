"""Regime event broadcast system.

US-099: Event-driven regime transition broadcasting. When regime_tracker
detects a transition, broadcast to all dependent modules. Currently
regime is detected but not propagated — modules use stale regime state.

Event-driven architecture pattern:
1. RegimeBroadcaster maintains a registry of listener callbacks
2. When a transition is detected, all listeners are notified
3. Each listener receives RegimeEvent with old/new regime + metadata
4. Listeners can choose to act or ignore based on transition type
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
_TRANSITION_LOG_PATH = _PROJECT_ROOT / "trained_data" / "regime_transitions.json"


@dataclass
class RegimeEvent:
    """Event payload for regime transitions."""

    old_regime: str = "NORMAL"
    new_regime: str = "NORMAL"
    confidence: float = 1.0
    timestamp: str = ""
    source: str = "regime_tracker"
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "old_regime": self.old_regime,
            "new_regime": self.new_regime,
            "confidence": round(self.confidence, 4),
            "timestamp": self.timestamp,
            "source": self.source,
            "metadata": self.metadata,
        }


# Type for listener callbacks
RegimeListener = Callable[[RegimeEvent], None]


class RegimeBroadcaster:
    """Event-driven regime transition broadcaster.

    Central hub that propagates regime transitions to all dependent
    modules (agents, gates, execution, position sizing, hedging).

    Usage:
        broadcaster = RegimeBroadcaster()
        broadcaster.register_listener("market_impact", impact_model.on_regime_change)
        broadcaster.register_listener("entropy_sizer", sizer.on_regime_change)
        # When regime changes:
        broadcaster.broadcast_transition("NORMAL", "HIGH", confidence=0.85)
    """

    def __init__(
        self,
        persistence_path: Optional[Path] = None,
    ):
        self.persistence_path = persistence_path or _TRANSITION_LOG_PATH

        # Listener registry: {module_name: callback}
        self._listeners: Dict[str, RegimeListener] = {}

        # Transition history
        self._transitions: List[Dict[str, Any]] = []
        self._current_regime: str = "NORMAL"
        self._transition_count: int = 0

        self._load_state()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def register_listener(
        self,
        module_name: str,
        callback: RegimeListener,
    ) -> None:
        """Register a module to receive regime transition events.

        Args:
            module_name: Unique identifier for the listener.
            callback: Function to call with RegimeEvent on transition.
        """
        self._listeners[module_name] = callback
        logger.debug(f"RegimeBroadcaster: Registered listener '{module_name}'")

    def unregister_listener(self, module_name: str) -> None:
        """Remove a registered listener."""
        self._listeners.pop(module_name, None)

    def broadcast_transition(
        self,
        old_regime: str,
        new_regime: str,
        confidence: float = 1.0,
        source: str = "regime_tracker",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> int:
        """Broadcast a regime transition to all registered listeners.

        Args:
            old_regime: Previous regime.
            new_regime: New regime.
            confidence: Detection confidence (0-1).
            source: Which detector triggered the transition.
            metadata: Additional event data.

        Returns:
            Number of listeners successfully notified.
        """
        if old_regime == new_regime:
            return 0  # No actual transition

        event = RegimeEvent(
            old_regime=old_regime,
            new_regime=new_regime,
            confidence=confidence,
            timestamp=datetime.now(timezone.utc).isoformat(),
            source=source,
            metadata=metadata or {},
        )

        self._current_regime = new_regime
        self._transition_count += 1

        # Log transition
        self._transitions.append(event.to_dict())
        if len(self._transitions) > 200:
            self._transitions = self._transitions[-100:]

        # Notify all listeners
        notified = 0
        failed = []
        for module_name, callback in self._listeners.items():
            try:
                callback(event)
                notified += 1
            except Exception as e:
                failed.append(module_name)
                logger.debug(
                    f"RegimeBroadcaster: Listener '{module_name}' failed: {e}"
                )

        logger.info(
            f"RegimeBroadcaster: {old_regime}→{new_regime} "
            f"(conf={confidence:.2f}) — notified {notified}/{len(self._listeners)}"
            + (f", failed: {failed}" if failed else "")
        )

        return notified

    def get_current_regime(self) -> str:
        """Get the current regime."""
        return self._current_regime

    def get_transition_history(self, n: int = 20) -> List[Dict[str, Any]]:
        """Get recent regime transitions.

        Args:
            n: Number of transitions to return.

        Returns:
            List of transition dicts (most recent first).
        """
        return list(reversed(self._transitions[-n:]))

    def get_transition_frequency(self, window_minutes: float = 60.0) -> float:
        """Get regime transition frequency (transitions per hour).

        Args:
            window_minutes: Lookback window in minutes.

        Returns:
            Transitions per hour in the window.
        """
        if not self._transitions:
            return 0.0

        now = datetime.now(timezone.utc)
        recent = 0
        for t in reversed(self._transitions):
            try:
                ts = datetime.fromisoformat(t["timestamp"].replace("Z", "+00:00"))
                diff = (now - ts).total_seconds() / 60.0
                if diff <= window_minutes:
                    recent += 1
                else:
                    break
            except (ValueError, KeyError):
                continue

        hours = window_minutes / 60.0
        return recent / max(hours, 0.01)

    def get_status(self) -> Dict[str, Any]:
        """Get broadcaster status for observability."""
        return {
            "current_regime": self._current_regime,
            "registered_listeners": list(self._listeners.keys()),
            "n_listeners": len(self._listeners),
            "total_transitions": self._transition_count,
            "transition_freq_per_hour": round(self.get_transition_frequency(), 2),
            "recent_transitions": self.get_transition_history(5),
        }

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save_state(self) -> None:
        """Persist transition history."""
        try:
            data = {
                "version": 1,
                "current_regime": self._current_regime,
                "transition_count": self._transition_count,
                "transitions": self._transitions[-100:],
                "last_updated": datetime.now(timezone.utc).isoformat(),
            }
            self.persistence_path.parent.mkdir(parents=True, exist_ok=True)
            try:
                from src.scanner.automation.safe_json import safe_json_write
                safe_json_write(self.persistence_path, data)
            except ImportError:
                self.persistence_path.write_text(json.dumps(data, indent=2))
        except Exception as e:
            logger.debug(f"RegimeBroadcaster: Failed to save: {e}")

    def _load_state(self) -> None:
        """Load persisted state."""
        if not self.persistence_path.exists():
            return
        try:
            try:
                from src.scanner.automation.safe_json import safe_json_read
                data = safe_json_read(self.persistence_path, default={})
            except ImportError:
                data = json.loads(self.persistence_path.read_text(encoding="utf-8"))

            if not isinstance(data, dict):
                return

            self._current_regime = data.get("current_regime", "NORMAL")
            self._transition_count = data.get("transition_count", 0)
            self._transitions = data.get("transitions", [])

            logger.debug(
                f"RegimeBroadcaster: Loaded {self._transition_count} transitions, "
                f"current={self._current_regime}"
            )
        except Exception as e:
            logger.debug(f"RegimeBroadcaster: Failed to load: {e}")
