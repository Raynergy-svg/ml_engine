"""Tests for Tier 7 PR 5: Control Plane, Broker Transport, Session Registry.

Validates FSM state machine, session persistence, control plane health
checks, degraded mode transitions, and auto-recovery.
"""

import json
import os
from pathlib import Path
from unittest.mock import MagicMock, patch, PropertyMock

import pytest

from src.scanner.automation.broker_transport import (
    TransportState,
    OANDATransport,
)
from src.scanner.automation.session_registry import SessionRegistry
from src.scanner.automation.trading_control_plane import TradingControlPlane


# ── Broker Transport Tests ────────────────────────────────────────────

class TestBrokerTransport:

    def test_initial_state_is_closed(self):
        t = OANDATransport()
        assert t.get_state() == TransportState.CLOSED

    def test_connect_success(self):
        t = OANDATransport()
        with patch.object(t, "_load_credentials"), \
             patch.object(t, "_verify_auth", return_value=True):
            result = t.connect()
        assert result is True
        assert t.get_state() == TransportState.CONNECTED
        assert t.is_connected()
        assert t.get_last_heartbeat() is not None

    def test_connect_auth_failure(self):
        t = OANDATransport()
        with patch.object(t, "_load_credentials"), \
             patch.object(t, "_verify_auth", return_value=False):
            result = t.connect()
        assert result is False
        assert t.get_state() == TransportState.CLOSED

    def test_heartbeat_success(self):
        t = OANDATransport()
        with patch.object(t, "_load_credentials"), \
             patch.object(t, "_verify_auth", return_value=True):
            t.connect()

        with patch.object(t, "_ping_broker", return_value=True):
            result = t.check_heartbeat()
        assert result is True
        assert t.get_state() == TransportState.CONNECTED

    def test_heartbeat_failure_transitions_to_degraded(self):
        t = OANDATransport()
        with patch.object(t, "_load_credentials"), \
             patch.object(t, "_verify_auth", return_value=True):
            t.connect()

        with patch.object(t, "_ping_broker", return_value=False):
            result = t.check_heartbeat()
        assert result is False
        assert t.get_state() == TransportState.DEGRADED
        assert t.flush_gate.is_active  # Gate activated during degraded

    def test_reconnect_success_from_degraded(self):
        t = OANDATransport(base_reconnect_delay=0.01, max_reconnect_delay=0.02)
        # Get to DEGRADED state
        with patch.object(t, "_load_credentials"), \
             patch.object(t, "_verify_auth", return_value=True):
            t.connect()
        with patch.object(t, "_ping_broker", return_value=False):
            t.check_heartbeat()
        assert t.get_state() == TransportState.DEGRADED

        # Reconnect
        with patch.object(t, "_load_credentials"), \
             patch.object(t, "_verify_auth", return_value=True):
            result = t.attempt_reconnect()
        assert result is True
        assert t.get_state() == TransportState.CONNECTED
        assert not t.flush_gate.is_active  # Gate deactivated

    def test_reconnect_failure_stays_degraded(self):
        t = OANDATransport(base_reconnect_delay=0.01, max_reconnect_delay=0.02)
        with patch.object(t, "_load_credentials"), \
             patch.object(t, "_verify_auth", return_value=True):
            t.connect()
        with patch.object(t, "_ping_broker", return_value=False):
            t.check_heartbeat()

        with patch.object(t, "_load_credentials"), \
             patch.object(t, "_verify_auth", return_value=False):
            result = t.attempt_reconnect()
        assert result is False
        assert t.get_state() == TransportState.DEGRADED
        assert t.reconnect_attempts == 1

    def test_close_from_any_state(self):
        t = OANDATransport()
        with patch.object(t, "_load_credentials"), \
             patch.object(t, "_verify_auth", return_value=True):
            t.connect()
        assert t.get_state() == TransportState.CONNECTED
        t.close()
        assert t.get_state() == TransportState.CLOSED


# ── Session Registry Tests ────────────────────────────────────────────

class TestSessionRegistry:

    def test_create_session_returns_id(self, tmp_path):
        reg = SessionRegistry(path=tmp_path / "session.json")
        sid = reg.create_session()
        assert sid
        assert len(sid) > 0
        assert reg.get_summary()["session_id"] == sid

    def test_save_and_load_persistence(self, tmp_path):
        path = tmp_path / "session.json"
        reg = SessionRegistry(path=path)
        sid = reg.create_session()
        reg.update_transport_state("connected")
        reg.save()

        # Load in a new instance
        reg2 = SessionRegistry(path=path)
        assert reg2.get_summary()["session_id"] == sid
        assert reg2.get_summary()["transport_state"] == "connected"

    def test_update_transport_state(self, tmp_path):
        reg = SessionRegistry(path=tmp_path / "session.json")
        reg.create_session()
        reg.update_transport_state("degraded")
        assert reg.get_summary()["transport_state"] == "degraded"

    def test_record_heartbeat(self, tmp_path):
        reg = SessionRegistry(path=tmp_path / "session.json")
        reg.create_session()
        reg.record_heartbeat()
        summary = reg.get_summary()
        assert summary["last_heartbeat"]
        assert summary["heartbeat_age_seconds"] >= 0

    def test_enter_exit_degraded(self, tmp_path):
        reg = SessionRegistry(path=tmp_path / "session.json")
        reg.create_session()
        reg.enter_degraded("test reason")
        summary = reg.get_summary()
        assert summary["degraded_mode"] is True
        assert summary["degraded_reason"] == "test reason"
        assert summary["degraded_since"]

        reg.exit_degraded()
        summary = reg.get_summary()
        assert summary["degraded_mode"] is False
        assert summary["degraded_reason"] == ""

    def test_event_id_cap_at_50(self, tmp_path):
        reg = SessionRegistry(path=tmp_path / "session.json")
        reg.create_session()
        for i in range(60):
            reg.record_event_processed(f"evt_{i}")
        summary = reg.get_summary()
        assert len(summary["last_processed_event_ids"]) == 50
        assert summary["last_processed_event_ids"][-1] == "evt_59"

    def test_invalid_transport_state_raises(self, tmp_path):
        reg = SessionRegistry(path=tmp_path / "session.json")
        reg.create_session()
        with pytest.raises(ValueError):
            reg.update_transport_state("invalid_state")


# ── Trading Control Plane Tests ───────────────────────────────────────

class TestTradingControlPlane:

    def _make_cp(self, transport=None, event_queue=None, queue=None, policy_engine=None, policy=None, registry=None):
        return TradingControlPlane(
            transport=transport,
            event_queue=event_queue or queue,
            policy_engine=policy_engine or policy,
            session_registry=registry,
        )

    def test_start_creates_session(self, tmp_path):
        reg = SessionRegistry(path=tmp_path / "session.json")
        transport = MagicMock()
        transport.connect.return_value = True
        transport.get_state.return_value = TransportState.CONNECTED
        cp = self._make_cp(transport=transport, registry=reg)

        sid = cp.start()
        assert sid
        assert reg.get_summary()["session_id"]
        transport.connect.assert_called_once()

    def test_check_health_returns_status(self):
        transport = MagicMock()
        transport.get_state.return_value = TransportState.CONNECTED
        transport.check_heartbeat.return_value = True
        transport.get_last_heartbeat.return_value = "2026-04-03T12:00:00Z"

        queue = MagicMock()
        queue.get_status.return_value = {"depth": 0, "in_flight": 0, "failed_count": 0}

        cp = self._make_cp(transport=transport, event_queue=queue)
        cp._started = True
        health = cp.check_health()

        assert health["transport_state"] == "connected"
        assert health["degraded_mode"] is False
        assert "queue_status" in health

    def test_degraded_on_heartbeat_failure(self, tmp_path):
        transport = MagicMock()
        transport.get_state.return_value = TransportState.CONNECTED
        transport.check_heartbeat.return_value = False
        transport.get_last_heartbeat.return_value = None

        reg = SessionRegistry(path=tmp_path / "session.json")
        reg.create_session()

        cp = self._make_cp(transport=transport, registry=reg)
        cp._started = True
        health = cp.check_health()

        assert cp._degraded is True
        assert reg.get_summary()["degraded_mode"] is True

    def test_auto_recovery_on_heartbeat_success(self, tmp_path):
        transport = MagicMock()
        transport.get_state.return_value = TransportState.CONNECTED
        transport.get_last_heartbeat.return_value = "2026-04-03T12:00:00Z"

        reg = SessionRegistry(path=tmp_path / "session.json")
        reg.create_session()

        cp = self._make_cp(transport=transport, registry=reg)
        cp._started = True
        cp._degraded = True
        cp._degraded_reason = "test"

        # Heartbeat succeeds → should auto-recover
        transport.check_heartbeat.return_value = True
        health = cp.check_health()
        assert cp._degraded is False
        assert reg.get_summary()["degraded_mode"] is False

    def test_stop_flushes_queue(self):
        queue = MagicMock()
        transport = MagicMock()
        cp = self._make_cp(transport=transport, event_queue=queue)
        cp._started = True
        cp.stop()
        queue.stop.assert_called_once_with(flush_first=True)
        transport.close.assert_called_once()

    def test_environment_provider_registered(self):
        policy = MagicMock()
        cp = self._make_cp(policy_engine=policy)
        policy.register_environment_provider.assert_called_once()

    def test_excessive_queue_failures_triggers_degraded(self):
        queue = MagicMock()
        queue.get_status.return_value = {"depth": 0, "in_flight": 0, "failed_count": 15}

        cp = self._make_cp(event_queue=queue)
        cp._started = True
        health = cp.check_health()

        assert cp._degraded is True
        assert "queue" in cp._degraded_reason
