"""Tier 7 Trading Control Plane — supervisor above ContinuousScanner.

Coordinates broker transport health, queue state, session lifecycle, and
degraded-mode transitions. Does NOT replace ContinuousScanner or
ExecutionManager — it wraps them with supervision.
"""

import threading
from typing import Any, Dict, Optional

import structlog

logger = structlog.get_logger(__name__)


class TradingControlPlane:
    """Supervise watch mode: broker health, queue state, session lifecycle.

    Called each scan cycle by continuous.py via check_health(). Does not
    run its own background thread — it's driven by the main scan loop.
    """

    def __init__(
        self,
        transport=None,
        event_queue=None,
        policy_engine=None,
        session_registry=None,
    ):
        self._transport = transport
        self._event_queue = event_queue
        self._policy_engine = policy_engine
        self._session_registry = session_registry
        self._lock = threading.Lock()
        self._degraded = False
        self._degraded_reason = ""
        self._session_id = ""
        self._started = False

        # Register environment provider with policy engine
        if self._policy_engine is not None:
            try:
                self._policy_engine.register_environment_provider(
                    self._get_environment_facts
                )
            except Exception:
                pass

    # ── Lifecycle ──────────────────────────────────────────────────────

    def start(self) -> str:
        """Create a session and connect transport. Returns session_id."""
        with self._lock:
            # Create session
            if self._session_registry is not None:
                self._session_id = self._session_registry.create_session()
            else:
                import uuid
                self._session_id = str(uuid.uuid4())[:12]

            # Connect transport
            if self._transport is not None:
                connected = self._transport.connect()
                state = self._transport.get_state().value
                if self._session_registry is not None:
                    self._session_registry.update_transport_state(state)
                if not connected:
                    self._enter_degraded_locked("transport_connect_failed")

            self._started = True
            logger.info("control_plane.started", session_id=self._session_id)
            return self._session_id

    def stop(self) -> None:
        """Graceful shutdown: flush queue, close transport, save session."""
        with self._lock:
            # Flush queue
            if self._event_queue is not None:
                try:
                    self._event_queue.stop(flush_first=True)
                    logger.info("control_plane.queue_flushed")
                except Exception as e:
                    logger.debug("control_plane.queue_flush_error", error=str(e))

            # Close transport
            if self._transport is not None:
                try:
                    self._transport.close()
                    if self._session_registry is not None:
                        self._session_registry.update_transport_state("closed")
                except Exception as e:
                    logger.debug("control_plane.transport_close_error", error=str(e))

            # Save session
            if self._session_registry is not None:
                try:
                    self._session_registry.save()
                except Exception:
                    pass

            self._started = False
            logger.info("control_plane.stopped", session_id=self._session_id)

    # ── Health Check (called each scan cycle) ──────────────────────────

    def check_health(self) -> Dict[str, Any]:
        """Check broker + queue health. Call this each scan cycle.

        Returns a status dict. Auto-enters/exits degraded mode based on
        transport heartbeat and queue failure count.
        """
        status: Dict[str, Any] = {
            "session_id": self._session_id,
            "degraded_mode": self._degraded,
            "degraded_reason": self._degraded_reason,
            "transport_state": "unknown",
            "last_heartbeat": None,
            "queue_status": {},
        }

        # Transport health
        if self._transport is not None:
            state = self._transport.get_state()
            status["transport_state"] = state.value
            status["last_heartbeat"] = self._transport.get_last_heartbeat()

            if state.value == "connected":
                # Heartbeat check
                hb_ok = self._transport.check_heartbeat()
                if hb_ok:
                    # Auto-recover from degraded
                    if self._degraded:
                        self.exit_degraded_mode()
                    if self._session_registry is not None:
                        self._session_registry.record_heartbeat()
                else:
                    # Heartbeat failed → degraded
                    self.enter_degraded_mode("heartbeat_failed")
                    status["transport_state"] = self._transport.get_state().value

            elif state.value == "degraded":
                # Attempt reconnect
                self.handle_transport_failure()
                status["transport_state"] = self._transport.get_state().value

        # Queue health
        if self._event_queue is not None:
            q_status = self._event_queue.get_status()
            status["queue_status"] = q_status

            # Check for excessive queue failures
            if q_status.get("failed_count", 0) > 10 and not self._degraded:
                self.enter_degraded_mode("excessive_queue_failures")

            # Update session registry
            if self._session_registry is not None:
                self._session_registry.update_queue_summary(
                    depth=q_status.get("depth", 0),
                    in_flight=q_status.get("in_flight", 0),
                    failed_count=q_status.get("failed_count", 0),
                )

        status["degraded_mode"] = self._degraded
        status["degraded_reason"] = self._degraded_reason
        return status

    # ── Degraded Mode ──────────────────────────────────────────────────

    def enter_degraded_mode(self, reason: str) -> None:
        """Mark the control plane as degraded."""
        with self._lock:
            self._enter_degraded_locked(reason)

    def _enter_degraded_locked(self, reason: str) -> None:
        """Internal — caller must NOT hold lock (or re-entrant lock)."""
        if self._degraded:
            return  # Already degraded
        self._degraded = True
        self._degraded_reason = reason
        if self._session_registry is not None:
            self._session_registry.enter_degraded(reason)
        logger.warning("control_plane.degraded", reason=reason)

    def exit_degraded_mode(self) -> None:
        """Clear degraded state."""
        with self._lock:
            if not self._degraded:
                return
            self._degraded = False
            self._degraded_reason = ""
            if self._session_registry is not None:
                self._session_registry.exit_degraded()
            logger.info("control_plane.recovered")

    # ── Transport Failure Handling ─────────────────────────────────────

    def handle_transport_failure(self) -> None:
        """Attempt to reconnect the transport from degraded state."""
        if self._transport is None:
            return

        if self._session_registry is not None:
            self._session_registry.record_reconnect_attempt()

        success = self._transport.attempt_reconnect()
        state = self._transport.get_state().value

        if self._session_registry is not None:
            self._session_registry.update_transport_state(state)

        if success:
            self.exit_degraded_mode()
        else:
            logger.info(
                "control_plane.reconnect_pending",
                attempts=self._transport.reconnect_attempts,
            )

    # ── Environment Provider (for PolicyEngine) ────────────────────────

    def _get_environment_facts(self) -> Dict[str, Any]:
        """Return environment facts for policy evaluation."""
        facts: Dict[str, Any] = {
            "broker_state": "unknown",
            "degraded_mode": self._degraded,
        }
        if self._degraded:
            facts["degraded_reason"] = self._degraded_reason
        if self._transport is not None:
            facts["broker_state"] = self._transport.get_state().value
            facts["reconnect_attempts"] = self._transport.reconnect_attempts
        return facts

    # ── Status ─────────────────────────────────────────────────────────

    def get_status(self) -> Dict[str, Any]:
        """Full control plane summary for operator display."""
        status = self.check_health() if self._started else {
            "session_id": self._session_id,
            "degraded_mode": self._degraded,
            "degraded_reason": self._degraded_reason,
            "transport_state": "not_started",
            "queue_status": {},
        }
        status["started"] = self._started
        if self._session_registry is not None:
            status["session_summary"] = self._session_registry.get_summary()
        return status
