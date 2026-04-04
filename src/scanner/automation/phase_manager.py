"""Phase-based state machine for the supervision-first runtime.

Five phases: IDLE → HUNTING → SUPERVISING → DIAGNOSING → ADAPTING
Replaces the monolithic 5-minute scan loop with event-driven phase transitions.
"""

import time
from enum import Enum
from typing import Any, Callable, Dict, Optional

import structlog
from transitions import Machine

logger = structlog.get_logger(__name__)


class Phase(str, Enum):
    IDLE = "idle"
    HUNTING = "hunting"
    SUPERVISING = "supervising"
    DIAGNOSING = "diagnosing"
    ADAPTING = "adapting"


# Phase timeout in seconds (auto-transition if exceeded)
PHASE_TIMEOUTS = {
    Phase.HUNTING: 300,      # 5 min max hunting
    Phase.DIAGNOSING: 30,    # 30s max diagnosis
    Phase.ADAPTING: 60,      # 60s max adapting
    Phase.SUPERVISING: None, # No timeout — stays while trades open
    Phase.IDLE: None,        # No timeout
}

_TRANSITIONS = [
    {"trigger": "go_hunting", "source": Phase.IDLE.value, "dest": Phase.HUNTING.value},
    {"trigger": "go_supervising", "source": Phase.HUNTING.value, "dest": Phase.SUPERVISING.value},
    {"trigger": "go_idle_from_hunting", "source": Phase.HUNTING.value, "dest": Phase.IDLE.value},
    {"trigger": "go_diagnosing", "source": Phase.SUPERVISING.value, "dest": Phase.DIAGNOSING.value},
    {"trigger": "go_hunting_from_supervising", "source": Phase.SUPERVISING.value, "dest": Phase.HUNTING.value},
    {"trigger": "go_idle_from_supervising", "source": Phase.SUPERVISING.value, "dest": Phase.IDLE.value},
    {"trigger": "go_adapting", "source": Phase.DIAGNOSING.value, "dest": Phase.ADAPTING.value},
    {"trigger": "go_hunting_from_adapting", "source": Phase.ADAPTING.value, "dest": Phase.HUNTING.value},
    {"trigger": "go_supervising_from_adapting", "source": Phase.ADAPTING.value, "dest": Phase.SUPERVISING.value},
    {"trigger": "go_idle_from_adapting", "source": Phase.ADAPTING.value, "dest": Phase.IDLE.value},
    # Emergency: any → idle
    {"trigger": "emergency_stop", "source": "*", "dest": Phase.IDLE.value},
]


class PhaseManager:
    """Manages phase transitions for the supervision-first runtime.

    Does NOT run its own loop — exposes on_tick() and transition methods
    that the outer SupervisionRuntime calls.
    """

    def __init__(self) -> None:
        self._entered_at: float = time.time()
        self._transition_count: int = 0
        self._last_phase: Optional[str] = None

        self.machine = Machine(
            model=self,
            states=[p.value for p in Phase],
            transitions=_TRANSITIONS,
            initial=Phase.IDLE.value,
            send_event=False,
            auto_transitions=False,
        )

    @property
    def current_phase(self) -> Phase:
        return Phase(self.state)  # type: ignore[attr-defined]

    @property
    def phase_duration(self) -> float:
        """Seconds spent in current phase."""
        return time.time() - self._entered_at

    @property
    def transition_count(self) -> int:
        return self._transition_count

    def transition_to(self, target: Phase, reason: str = "") -> bool:
        """Attempt a phase transition. Returns True if successful."""
        old = self.current_phase
        trigger_map = {
            (Phase.IDLE, Phase.HUNTING): "go_hunting",
            (Phase.HUNTING, Phase.SUPERVISING): "go_supervising",
            (Phase.HUNTING, Phase.IDLE): "go_idle_from_hunting",
            (Phase.SUPERVISING, Phase.DIAGNOSING): "go_diagnosing",
            (Phase.SUPERVISING, Phase.HUNTING): "go_hunting_from_supervising",
            (Phase.SUPERVISING, Phase.IDLE): "go_idle_from_supervising",
            (Phase.DIAGNOSING, Phase.ADAPTING): "go_adapting",
            (Phase.ADAPTING, Phase.HUNTING): "go_hunting_from_adapting",
            (Phase.ADAPTING, Phase.SUPERVISING): "go_supervising_from_adapting",
            (Phase.ADAPTING, Phase.IDLE): "go_idle_from_adapting",
        }

        key = (old, target)
        trigger_name = trigger_map.get(key)

        if trigger_name is None:
            if target == Phase.IDLE:
                trigger_name = "emergency_stop"
            else:
                logger.warning(
                    "phase_manager.illegal_transition",
                    old=old.value,
                    target=target.value,
                    reason=reason,
                )
                return False

        try:
            getattr(self, trigger_name)()  # Fire FSM trigger
            self._last_phase = old.value
            self._entered_at = time.time()
            self._transition_count += 1
            logger.info(
                "phase_manager.transition",
                old=old.value,
                new=target.value,
                reason=reason,
                duration=round(time.time() - self._entered_at, 1),
            )
            return True
        except Exception as e:
            logger.error(
                "phase_manager.transition_error",
                old=old.value,
                target=target.value,
                error=str(e),
            )
            return False

    def check_timeout(self) -> Optional[Phase]:
        """Check if current phase has exceeded its timeout.

        Returns the phase to transition to, or None if no timeout.
        """
        timeout = PHASE_TIMEOUTS.get(self.current_phase)
        if timeout is None:
            return None
        if self.phase_duration > timeout:
            # Timeout → go to sensible fallback
            fallbacks = {
                Phase.HUNTING: Phase.SUPERVISING,
                Phase.DIAGNOSING: Phase.ADAPTING,
                Phase.ADAPTING: Phase.IDLE,
            }
            return fallbacks.get(self.current_phase, Phase.IDLE)
        return None

    def get_status(self) -> Dict[str, Any]:
        return {
            "current_phase": self.current_phase.value,
            "phase_duration_seconds": round(self.phase_duration, 1),
            "transition_count": self._transition_count,
            "last_phase": self._last_phase,
        }
