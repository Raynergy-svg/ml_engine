"""ScanTrigger and TriggerRegistry — event-driven scan triggers."""

import collections
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional

try:
    import structlog
    logger = structlog.get_logger(__name__)
except ImportError:
    import logging
    logger = logging.getLogger(__name__)


class TriggerType(Enum):
    SLOT_OPENED = "slot_opened"
    REGIME_TRANSITION = "regime_transition"
    SESSION_OPEN = "session_open"
    VOLATILITY_SPIKE = "volatility_spike"
    SCHEDULED_SWEEP = "scheduled_sweep"
    MANUAL = "manual"
    COOLDOWN_EXPIRED = "cooldown_expired"


@dataclass
class ScanTrigger:
    trigger_type: TriggerType
    pairs: Optional[List[str]] = None
    constraints: Dict = field(default_factory=dict)
    priority: int = 5
    timestamp: float = field(default_factory=time.time)
    source: str = "system"


class TriggerRegistry:
    """Registry of conditions that trigger scanner runs."""

    def __init__(self, sweep_interval_minutes: float = 30.0):
        self._pending: collections.deque = collections.deque(maxlen=100)
        self._sweep_interval = sweep_interval_minutes * 60.0
        self._last_sweep: float = time.time()

    def fire(self, trigger: ScanTrigger) -> None:
        """Append a trigger to the pending queue."""
        self._pending.append(trigger)
        logger.info(
            "trigger_fired",
            trigger_type=trigger.trigger_type.value,
            pairs_count=len(trigger.pairs) if trigger.pairs else 0,
            source=trigger.source,
        )

    def drain(self) -> List[ScanTrigger]:
        """Return all pending triggers and clear the queue."""
        triggers = list(self._pending)
        self._pending.clear()
        return triggers

    def has_pending(self) -> bool:
        """Return True if there are pending triggers."""
        return len(self._pending) > 0

    def schedule_sweep(self, pairs: Optional[List[str]] = None) -> None:
        """Create and fire a SCHEDULED_SWEEP trigger."""
        trigger = ScanTrigger(
            trigger_type=TriggerType.SCHEDULED_SWEEP,
            pairs=pairs,
        )
        self.fire(trigger)

    def should_sweep(self) -> bool:
        """Return True if sweep_interval_minutes has elapsed since last sweep, resets timer."""
        now = time.time()
        if now - self._last_sweep >= self._sweep_interval:
            self._last_sweep = now
            return True
        return False
