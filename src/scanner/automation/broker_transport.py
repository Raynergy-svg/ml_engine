"""Broker transport abstraction with transitions FSM for connection state management.

Provides an explicit state machine for broker connectivity:
CLOSED → CONNECTING → CONNECTED → DEGRADED → RECONNECTING → (back to CONNECTED or DEGRADED)

Uses the `transitions` library for guarded state transitions with callbacks.
FlushGate buffers operations during RECONNECTING, drains on CONNECTED.
"""

import os
import random
import threading
import time
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Optional

import structlog
from transitions import Machine

from src.scanner.automation.event_queue import FlushGate

logger = structlog.get_logger(__name__)


class TransportState(str, Enum):
    """Broker transport connection states."""
    CLOSED = "closed"
    CONNECTING = "connecting"
    CONNECTED = "connected"
    DEGRADED = "degraded"
    RECONNECTING = "reconnecting"


class BrokerTransport(ABC):
    """Abstract broker transport interface."""

    @abstractmethod
    def connect(self) -> bool:
        """Initiate connection. Returns True on success."""
        ...

    @abstractmethod
    def close(self) -> None:
        """Close the transport."""
        ...

    @abstractmethod
    def is_connected(self) -> bool:
        """Check if transport is connected."""
        ...

    @abstractmethod
    def get_state(self) -> TransportState:
        """Return current transport state."""
        ...

    @abstractmethod
    def get_last_heartbeat(self) -> Optional[str]:
        """Return ISO timestamp of last successful heartbeat."""
        ...

    @abstractmethod
    def check_heartbeat(self) -> bool:
        """Perform a heartbeat check. Returns True if broker is reachable."""
        ...

    @abstractmethod
    def flush(self) -> None:
        """Flush any pending operations."""
        ...


class OANDATransport(BrokerTransport):
    """OANDA REST API transport with FSM state management.

    Heartbeat is NOT automatic — call check_heartbeat() each scan cycle
    from the control plane. Reconnect uses exponential backoff.
    """

    # FSM states and transitions
    _STATES = [s.value for s in TransportState]
    _TRANSITIONS = [
        {"trigger": "do_connect", "source": TransportState.CLOSED.value, "dest": TransportState.CONNECTING.value},
        {"trigger": "auth_success", "source": TransportState.CONNECTING.value, "dest": TransportState.CONNECTED.value},
        {"trigger": "auth_fail", "source": TransportState.CONNECTING.value, "dest": TransportState.CLOSED.value},
        {"trigger": "heartbeat_fail", "source": TransportState.CONNECTED.value, "dest": TransportState.DEGRADED.value},
        {"trigger": "do_close", "source": TransportState.CONNECTED.value, "dest": TransportState.CLOSED.value},
        {"trigger": "do_close", "source": TransportState.DEGRADED.value, "dest": TransportState.CLOSED.value},
        {"trigger": "do_close", "source": TransportState.RECONNECTING.value, "dest": TransportState.CLOSED.value},
        {"trigger": "do_close", "source": TransportState.CONNECTING.value, "dest": TransportState.CLOSED.value},
        {"trigger": "do_reconnect", "source": TransportState.DEGRADED.value, "dest": TransportState.RECONNECTING.value},
        {"trigger": "reconnect_success", "source": TransportState.RECONNECTING.value, "dest": TransportState.CONNECTED.value},
        {"trigger": "reconnect_fail", "source": TransportState.RECONNECTING.value, "dest": TransportState.DEGRADED.value},
    ]

    def __init__(
        self,
        base_reconnect_delay: float = 2.0,
        max_reconnect_delay: float = 60.0,
        heartbeat_timeout: float = 10.0,
    ):
        self._lock = threading.Lock()
        self._last_heartbeat: Optional[str] = None
        self._reconnect_attempts = 0
        self._base_delay = base_reconnect_delay
        self._max_delay = max_reconnect_delay
        self._heartbeat_timeout = heartbeat_timeout
        self._flush_gate = FlushGate()

        # OANDA credentials (loaded lazily)
        self._token: Optional[str] = None
        self._account_id: Optional[str] = None
        self._base_url: Optional[str] = None

        # Initialize FSM
        self.machine = Machine(
            model=self,
            states=self._STATES,
            transitions=self._TRANSITIONS,
            initial=TransportState.CLOSED.value,
            send_event=False,
        )

    # ── BrokerTransport interface ──────────────────────────────────────

    def connect(self) -> bool:
        with self._lock:
            try:
                self._load_credentials()
                self.do_connect()  # type: ignore[attr-defined]
                logger.info("broker_transport.connecting", base_url=self._base_url)

                # Verify auth with a lightweight API call
                if self._verify_auth():
                    self.auth_success()  # type: ignore[attr-defined]
                    self._last_heartbeat = datetime.now(timezone.utc).isoformat()
                    self._reconnect_attempts = 0
                    logger.info("broker_transport.connected")
                    return True
                else:
                    self.auth_fail()  # type: ignore[attr-defined]
                    logger.warning("broker_transport.auth_failed")
                    return False
            except Exception as e:
                logger.error("broker_transport.connect_error", error=str(e))
                try:
                    self.auth_fail()  # type: ignore[attr-defined]
                except Exception:
                    pass
                return False

    def close(self) -> None:
        with self._lock:
            try:
                self._flush_gate.drop()
                self.do_close()  # type: ignore[attr-defined]
                logger.info("broker_transport.closed")
            except Exception as e:
                logger.debug("broker_transport.close_error", error=str(e))

    def is_connected(self) -> bool:
        return self.state == TransportState.CONNECTED.value  # type: ignore[attr-defined]

    def get_state(self) -> TransportState:
        return TransportState(self.state)  # type: ignore[attr-defined]

    def get_last_heartbeat(self) -> Optional[str]:
        return self._last_heartbeat

    def check_heartbeat(self) -> bool:
        """Perform a heartbeat check against OANDA. Called by control plane each cycle.

        Returns True if broker is reachable. On failure, transitions to DEGRADED.
        """
        if self.state != TransportState.CONNECTED.value:  # type: ignore[attr-defined]
            return False

        with self._lock:
            try:
                if self._ping_broker():
                    self._last_heartbeat = datetime.now(timezone.utc).isoformat()
                    return True
                else:
                    self.heartbeat_fail()  # type: ignore[attr-defined]
                    self._flush_gate.start()
                    logger.warning("broker_transport.heartbeat_failed")
                    return False
            except Exception as e:
                try:
                    self.heartbeat_fail()  # type: ignore[attr-defined]
                    self._flush_gate.start()
                except Exception:
                    pass
                logger.warning("broker_transport.heartbeat_error", error=str(e))
                return False

    def attempt_reconnect(self) -> bool:
        """Attempt to reconnect from DEGRADED state with exponential backoff.

        Returns True on success (transitions to CONNECTED).
        """
        if self.state != TransportState.DEGRADED.value:  # type: ignore[attr-defined]
            return False

        with self._lock:
            self._reconnect_attempts += 1
            delay = min(
                self._base_delay * (2 ** (self._reconnect_attempts - 1))
                + random.uniform(0, self._base_delay),
                self._max_delay,
            )
            logger.info(
                "broker_transport.reconnecting",
                attempt=self._reconnect_attempts,
                delay=round(delay, 1),
            )

            try:
                self.do_reconnect()  # type: ignore[attr-defined]
            except Exception:
                return False

            # Wait backoff period
            time.sleep(min(delay, 5.0))  # Cap actual sleep at 5s to not block scanner

            try:
                self._load_credentials()
                if self._verify_auth():
                    self.reconnect_success()  # type: ignore[attr-defined]
                    self._last_heartbeat = datetime.now(timezone.utc).isoformat()
                    self._reconnect_attempts = 0

                    # Drain flush gate
                    buffered = self._flush_gate.end()
                    if buffered:
                        logger.info("broker_transport.flush_gate_drained", count=len(buffered))

                    logger.info("broker_transport.reconnected")
                    return True
                else:
                    self.reconnect_fail()  # type: ignore[attr-defined]
                    logger.warning("broker_transport.reconnect_failed", attempt=self._reconnect_attempts)
                    return False
            except Exception as e:
                try:
                    self.reconnect_fail()  # type: ignore[attr-defined]
                except Exception:
                    pass
                logger.warning("broker_transport.reconnect_error", error=str(e))
                return False

    def flush(self) -> None:
        """Drain any buffered operations from the flush gate."""
        if self._flush_gate.is_active:
            buffered = self._flush_gate.end()
            if buffered:
                logger.info("broker_transport.manual_flush", count=len(buffered))

    @property
    def reconnect_attempts(self) -> int:
        return self._reconnect_attempts

    @property
    def flush_gate(self) -> FlushGate:
        return self._flush_gate

    # ── OANDA-specific internals ───────────────────────────────────────

    def _load_credentials(self) -> None:
        """Load OANDA credentials from environment."""
        try:
            from src.utils.oanda_practice import _load_project_dotenv
            _load_project_dotenv()
        except Exception:
            pass

        self._token = os.getenv("OANDA_API_TOKEN", "") or os.getenv("OANDA_API_KEY", "")
        self._account_id = os.getenv("OANDA_ACCOUNT_ID", "")
        env = os.getenv("OANDA_ENVIRONMENT", "practice")
        if env == "live":
            self._base_url = "https://api-fxtrade.oanda.com"
        else:
            self._base_url = "https://api-fxpractice.oanda.com"

    def _verify_auth(self) -> bool:
        """Verify OANDA credentials with a lightweight account summary call."""
        if not self._token or not self._account_id:
            return False
        try:
            import requests
            url = f"{self._base_url}/v3/accounts/{self._account_id}/summary"
            resp = requests.get(
                url,
                headers={
                    "Authorization": f"Bearer {self._token}",
                    "Content-Type": "application/json",
                },
                timeout=self._heartbeat_timeout,
            )
            return resp.status_code == 200
        except Exception:
            return False

    def _ping_broker(self) -> bool:
        """Lightweight ping — same as _verify_auth but for heartbeat."""
        return self._verify_auth()
