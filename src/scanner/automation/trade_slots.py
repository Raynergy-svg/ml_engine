"""TradeSlot state machine and SlotManager for pair lifecycle tracking."""

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


class TradeSlotState(Enum):
    IDLE = "IDLE"
    SCANNING = "SCANNING"
    CANDIDATE = "CANDIDATE"
    ARMED = "ARMED"
    OPEN = "OPEN"
    SUPERVISING = "SUPERVISING"
    CLOSING = "CLOSING"
    DIAGNOSING = "DIAGNOSING"
    ADAPTING = "ADAPTING"
    COOLDOWN = "COOLDOWN"


# Legal transitions: from_state -> set of allowed to_states
LEGAL_TRANSITIONS: Dict[TradeSlotState, set] = {
    TradeSlotState.IDLE: {TradeSlotState.SCANNING},
    TradeSlotState.SCANNING: {TradeSlotState.CANDIDATE, TradeSlotState.IDLE},
    TradeSlotState.CANDIDATE: {TradeSlotState.ARMED, TradeSlotState.IDLE},
    TradeSlotState.ARMED: {TradeSlotState.OPEN, TradeSlotState.IDLE},
    TradeSlotState.OPEN: {TradeSlotState.SUPERVISING},
    TradeSlotState.SUPERVISING: {TradeSlotState.CLOSING},
    TradeSlotState.CLOSING: {TradeSlotState.DIAGNOSING},
    TradeSlotState.DIAGNOSING: {TradeSlotState.ADAPTING},
    TradeSlotState.ADAPTING: {TradeSlotState.IDLE, TradeSlotState.COOLDOWN},
    TradeSlotState.COOLDOWN: {TradeSlotState.IDLE},
}


@dataclass
class TradeSlot:
    pair: str
    state: TradeSlotState = TradeSlotState.IDLE
    entered_state_at: float = field(default_factory=time.time)
    trade_id: Optional[str] = None
    metadata: dict = field(default_factory=dict)
    cooldown_until: Optional[float] = None


class SlotManager:
    """Manages TradeSlot state machines for all pairs."""

    def __init__(self, max_concurrent: int = 3):
        self.max_concurrent = max_concurrent
        self._slots: Dict[str, TradeSlot] = {}

    def _get_or_create_slot(self, pair: str) -> TradeSlot:
        if pair not in self._slots:
            self._slots[pair] = TradeSlot(pair=pair)
        return self._slots[pair]

    def transition(self, pair: str, new_state: TradeSlotState, **metadata) -> None:
        """Validate and perform a state transition for a pair.

        Raises ValueError on illegal transition.
        """
        slot = self._get_or_create_slot(pair)
        old_state = slot.state

        # any -> IDLE is always legal (reset)
        if new_state != TradeSlotState.IDLE:
            allowed = LEGAL_TRANSITIONS.get(old_state, set())
            if new_state not in allowed:
                raise ValueError(
                    f"Illegal transition for {pair}: {old_state.value} -> {new_state.value}"
                )

        slot.state = new_state
        slot.entered_state_at = time.time()
        if metadata:
            slot.metadata.update(metadata)

        # Clear cooldown when transitioning to IDLE
        if new_state == TradeSlotState.IDLE:
            slot.cooldown_until = None

        logger.info(
            "slot_transition",
            pair=pair,
            old_state=old_state.value,
            new_state=new_state.value,
        )

    def available_slots(self) -> int:
        """Return count of pairs in IDLE state that are not in cooldown."""
        now = time.time()
        idle_count = 0
        for slot in self._slots.values():
            if slot.state == TradeSlotState.IDLE:
                if slot.cooldown_until is None or slot.cooldown_until < now:
                    idle_count += 1
        # Available = max_concurrent minus non-idle slots, but also count idle slots
        non_idle = sum(
            1 for s in self._slots.values() if s.state != TradeSlotState.IDLE
        )
        return max(0, self.max_concurrent - non_idle)

    def open_positions(self) -> List[TradeSlot]:
        """Return list of TradeSlots where state is OPEN or SUPERVISING."""
        return [
            slot
            for slot in self._slots.values()
            if slot.state in (TradeSlotState.OPEN, TradeSlotState.SUPERVISING)
        ]

    def get_state(self, pair: str) -> TradeSlotState:
        """Return TradeSlotState for pair, defaults to IDLE."""
        if pair in self._slots:
            return self._slots[pair].state
        return TradeSlotState.IDLE

    def set_cooldown(self, pair: str, duration_seconds: float) -> None:
        """Transition pair to COOLDOWN and set cooldown_until."""
        self.transition(pair, TradeSlotState.COOLDOWN)
        slot = self._slots[pair]
        slot.cooldown_until = time.time() + duration_seconds

    def check_cooldowns(self) -> List[str]:
        """Auto-transition COOLDOWN pairs past their cooldown back to IDLE.

        Returns list of freed pair names.
        """
        now = time.time()
        freed = []
        for pair, slot in self._slots.items():
            if slot.state == TradeSlotState.COOLDOWN:
                if slot.cooldown_until is not None and slot.cooldown_until < now:
                    self.transition(pair, TradeSlotState.IDLE)
                    freed.append(pair)
        return freed
