"""Public-interface Nautilus replay runtime for Axiom candidate observations."""

from __future__ import annotations

from typing import Any

from .contracts import CandidateObservation, MaterialEvent
from .journal import CandidateJournal
from .policy import CandidateMonitor


class NautilusRuntimeUnavailable(RuntimeError):
    """Raised when the pinned Nautilus runtime is not installed."""


class NautilusReplayRuntime:
    """Deliver admitted observations through Nautilus into Axiom policy.

    This is deliberately replay-only. It uses Nautilus Clock and MessageBus
    mechanics, while its start/stop methods are adapter lifecycle gates. It
    exposes no order or execution client methods and does not create a
    TradingNode, broker, or credentialed client.
    """

    TOPIC = "axiom2.nautilus.admitted_observation"

    def __init__(
        self,
        monitor: CandidateMonitor,
        journal: CandidateJournal,
        *,
        trader_id: str = "TRADER-AXIOM2-REPLAY",
        topic: str = TOPIC,
    ) -> None:
        try:
            from nautilus_trader.common import Clock, MessageBus
            from nautilus_trader.model import TraderId
        except ModuleNotFoundError as exc:
            raise NautilusRuntimeUnavailable(
                "the pinned Nautilus package is required for replay runtime"
            ) from exc

        self.monitor = monitor
        self.journal = journal
        self.topic = topic
        self.clock = Clock.new_test()
        self.clock.set_time(journal.now_ns)
        self.bus = MessageBus(TraderId(trader_id), clock=self.clock)
        self._started = False
        self._disposed = False
        self._last_material_event: MaterialEvent | None = None
        self.bus.subscribe(self.topic, self._on_message)

    @property
    def is_running(self) -> bool:
        return self._started and not self._disposed

    @property
    def last_material_event(self) -> MaterialEvent | None:
        return self._last_material_event

    def start(self) -> None:
        if self._disposed:
            raise RuntimeError("runtime is disposed")
        if not self._started:
            self._started = True

    def stop(self) -> None:
        if self._disposed:
            return
        if self._started:
            self._started = False

    def dispose(self) -> None:
        if self._disposed:
            return
        self.stop()
        self.bus.dispose()
        self._disposed = True

    def set_time(self, to_time_ns: int) -> None:
        if not self.is_running:
            raise RuntimeError("runtime must be running")
        self.journal.advance_time(to_time_ns)
        self.clock.set_time(to_time_ns)

    def publish(self, observation: CandidateObservation) -> MaterialEvent | None:
        if not self.is_running:
            raise RuntimeError("runtime must be running")
        self._last_material_event = None
        self.bus.publish(
            self.topic,
            observation.to_payload(),
            external_pub=False,
        )
        return self._last_material_event

    def _on_message(self, payload: Any) -> None:
        if not isinstance(payload, dict):
            raise TypeError("admitted observation payload must be a dictionary")
        observation = CandidateObservation(**payload)
        self._last_material_event = self.monitor.observe(observation)
