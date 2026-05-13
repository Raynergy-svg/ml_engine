"""US-003 AbortConfirmModal tests — real EventBus + tmp_path strategic_log.

NO MOCKS per CLAUDE.md No-Mock Rule. Tests drive the real ModalScreen
lifecycle via Textual's Pilot harness and the real EventBus singleton
to verify:
  - confirm publishes control.signal_veto AND writes audit log
  - cancel publishes nothing AND writes nothing
  - empty replay buffer renders 'no pending signal' AND blocks confirm

Async tests use ``asyncio.run`` directly (not ``@pytest.mark.asyncio``)
so they run without the pytest-asyncio plugin — the same approach is
robust to either CI configuration.
"""
from __future__ import annotations

import asyncio
import uuid
from typing import Iterator

import pytest
from textual.app import App, ComposeResult
from textual.widgets import Static

from src.scanner.automation.event_bus import (
    EventBus,
    get_event_bus,
    reset_event_bus,
)
from src.tui.app import _emit_signal_veto
from src.tui.screens.abort_confirm_modal import (
    AbortConfirmModal,
    _signal_age_seconds,
)


@pytest.fixture
def fresh_bus() -> EventBus:
    """Real EventBus singleton, reset between tests for isolation."""
    reset_event_bus()
    bus = get_event_bus()
    yield bus
    reset_event_bus()


def _publish_pending(bus: EventBus, pair: str = "EUR_USD", direction: str = "LONG") -> dict:
    signal_id = str(uuid.uuid4())
    bus.publish("signal.pending", {
        "signal_id": signal_id,
        "pair": pair,
        "direction": direction,
        "entry": 1.0823,
        "sl": 1.0810,
        "tp": 1.0850,
    })
    for evt in reversed(bus.replay_buffer()):
        if evt.get("event_type") == "signal.pending":
            return evt
    raise AssertionError("signal.pending not in replay buffer after publish")


# ─────────────────────── _signal_age_seconds ───────────────────────────

class TestSignalAge:

    def test_returns_positive_age_for_recent_event(self, fresh_bus):
        evt = _publish_pending(fresh_bus)
        age = _signal_age_seconds(evt)
        assert age is not None
        assert age >= 0.0
        assert age < 5.0  # event was just emitted

    def test_returns_none_for_missing_ts(self):
        assert _signal_age_seconds({"event_type": "signal.pending", "payload": {}}) is None

    def test_returns_none_for_unparseable_ts(self):
        assert _signal_age_seconds({"ts": "not-a-timestamp"}) is None


# ─────────────────────── _emit_signal_veto (helper, no App) ──────────────

class TestEmitSignalVeto:

    def test_confirm_publishes_veto_and_writes_audit_log(self, fresh_bus, tmp_path):
        pending = _publish_pending(fresh_bus, pair="GBP_USD", direction="SHORT")
        log_path = tmp_path / "strategic_log.md"

        baseline = len(fresh_bus.replay_buffer())
        ok = _emit_signal_veto(app=None, bus=fresh_bus, pending_event=pending, log_path=log_path)

        assert ok is True

        # control.signal_veto landed in the replay buffer with the expected payload
        veto_events = [
            e for e in fresh_bus.replay_buffer()
            if e.get("event_type") == "control.signal_veto"
        ]
        assert len(veto_events) == 1
        veto_payload = veto_events[0]["payload"]
        assert veto_payload["signal_id"] == pending["payload"]["signal_id"]
        assert veto_payload["pair"] == "GBP_USD"
        assert veto_payload["direction"] == "SHORT"
        assert veto_payload["vetoed_by"] == "operator"

        # Audit log appended with required fields
        assert log_path.exists()
        content = log_path.read_text()
        assert "Operator SUPERVISOR_ABORT" in content
        assert "**actor:** TUI" in content
        assert "**action:** supervisor_abort" in content
        assert "**pair:** GBP_USD" in content
        assert f"**signal_id:** {pending['payload']['signal_id']}" in content
        # ISO timestamp marker present
        assert "T" in content and "+00:00" in content

        # bus saw exactly one new event (the veto) — no spurious republishes
        assert len(fresh_bus.replay_buffer()) == baseline + 1

    def test_refuses_to_publish_on_empty_signal_id(self, fresh_bus, tmp_path):
        log_path = tmp_path / "strategic_log.md"
        ok = _emit_signal_veto(
            app=None,
            bus=fresh_bus,
            pending_event={"ts": "2026-05-13T12:00:00+00:00", "payload": {"signal_id": ""}},
            log_path=log_path,
        )
        assert ok is False
        veto_events = [
            e for e in fresh_bus.replay_buffer()
            if e.get("event_type") == "control.signal_veto"
        ]
        assert veto_events == []
        assert not log_path.exists()

    def test_audit_log_survives_repeated_vetoes(self, fresh_bus, tmp_path):
        log_path = tmp_path / "strategic_log.md"
        pending1 = _publish_pending(fresh_bus, pair="EUR_USD")
        _emit_signal_veto(None, fresh_bus, pending1, log_path)
        pending2 = _publish_pending(fresh_bus, pair="USD_JPY")
        _emit_signal_veto(None, fresh_bus, pending2, log_path)

        content = log_path.read_text()
        assert content.count("Operator SUPERVISOR_ABORT") == 2
        assert "EUR_USD" in content
        assert "USD_JPY" in content


# ─────────────────────── Modal lifecycle (Pilot, real Textual) ──────────

class _Host(App):
    """Minimal host app that pushes the modal and captures the dismiss value."""

    def __init__(self, event):
        super().__init__()
        self._event = event
        self.captured: list = []

    def compose(self) -> ComposeResult:
        yield Static("host")

    def on_mount(self) -> None:
        self.push_screen(AbortConfirmModal(self._event), self.captured.append)


class TestModalLifecycle:

    def test_modal_returns_true_on_enter(self, fresh_bus):
        pending = _publish_pending(fresh_bus)
        host = _Host(pending)

        async def _run():
            async with host.run_test(size=(80, 24)) as pilot:
                await pilot.pause()
                await pilot.press("enter")
                await pilot.pause()

        asyncio.run(_run())
        assert host.captured == [True]

    def test_modal_returns_false_on_escape(self, fresh_bus):
        pending = _publish_pending(fresh_bus)
        host = _Host(pending)

        async def _run():
            async with host.run_test(size=(80, 24)) as pilot:
                await pilot.pause()
                await pilot.press("escape")
                await pilot.pause()

        asyncio.run(_run())
        assert host.captured == [False]

    def test_modal_renders_empty_state_with_no_pending(self):
        host = _Host(None)

        async def _run():
            async with host.run_test(size=(80, 24)) as pilot:
                await pilot.pause()
                screen = host.screen
                assert isinstance(screen, AbortConfirmModal)
                empty_widgets = list(screen.query("#abort-empty"))
                assert len(empty_widgets) == 1
                stats_widgets = list(screen.query("#abort-stats"))
                assert stats_widgets == []
                # Confirm button is disabled — ENTER + Confirm cannot fire a veto.
                confirm_btn = screen.query_one("#abort-confirm")
                assert confirm_btn.disabled is True
                await pilot.press("escape")
                await pilot.pause()

        asyncio.run(_run())
        # ESC dismisses False — no veto publishable from empty state.
        assert host.captured == [False]

    def test_modal_with_pending_shows_pair_and_direction(self, fresh_bus):
        pending = _publish_pending(fresh_bus, pair="AUD_USD", direction="SHORT")
        host = _Host(pending)

        async def _run():
            async with host.run_test(size=(80, 24)) as pilot:
                await pilot.pause()
                screen = host.screen
                assert isinstance(screen, AbortConfirmModal)
                stats = screen.query_one("#abort-stats", Static)
                rendered = str(stats.render())
                assert "AUD_USD" in rendered
                assert "SHORT" in rendered
                assert "Age:" in rendered
                await pilot.press("escape")
                await pilot.pause()

        asyncio.run(_run())


# ─────────────── End-to-end: modal callback → veto emit + log write ──────

class _AbortHost(App):
    """Host that wires the modal callback to _emit_signal_veto (mirrors app.py)."""

    def __init__(self, pending_event, bus, log_path):
        super().__init__()
        self._pending = pending_event
        self._bus = bus
        self._log_path = log_path

    def compose(self) -> ComposeResult:
        yield Static("host")

    def on_mount(self) -> None:
        def _cb(confirmed):
            if not confirmed or self._pending is None:
                return
            _emit_signal_veto(self, self._bus, self._pending, self._log_path)

        self.push_screen(AbortConfirmModal(self._pending), _cb)


class TestEndToEndModalToBus:

    def test_enter_confirm_emits_veto_and_writes_log(self, fresh_bus, tmp_path):
        pending = _publish_pending(fresh_bus, pair="USD_CAD", direction="LONG")
        log_path = tmp_path / "strategic_log.md"

        host = _AbortHost(pending, fresh_bus, log_path)

        async def _run():
            async with host.run_test(size=(80, 24)) as pilot:
                await pilot.pause()
                await pilot.press("enter")
                await pilot.pause()

        asyncio.run(_run())

        # Veto was published
        veto_events = [
            e for e in fresh_bus.replay_buffer()
            if e.get("event_type") == "control.signal_veto"
        ]
        assert len(veto_events) == 1
        assert veto_events[0]["payload"]["pair"] == "USD_CAD"

        # Audit log written
        assert log_path.exists()
        assert "USD_CAD" in log_path.read_text()

    def test_escape_cancel_publishes_nothing_and_writes_nothing(self, fresh_bus, tmp_path):
        pending = _publish_pending(fresh_bus, pair="NZD_USD", direction="LONG")
        log_path = tmp_path / "strategic_log.md"

        host = _AbortHost(pending, fresh_bus, log_path)

        async def _run():
            async with host.run_test(size=(80, 24)) as pilot:
                await pilot.pause()
                await pilot.press("escape")
                await pilot.pause()

        asyncio.run(_run())

        veto_events = [
            e for e in fresh_bus.replay_buffer()
            if e.get("event_type") == "control.signal_veto"
        ]
        assert veto_events == []
        assert not log_path.exists()

    def test_empty_state_enter_publishes_nothing(self, fresh_bus, tmp_path):
        log_path = tmp_path / "strategic_log.md"

        host = _AbortHost(None, fresh_bus, log_path)

        async def _run():
            async with host.run_test(size=(80, 24)) as pilot:
                await pilot.pause()
                await pilot.press("enter")
                await pilot.pause()
                await pilot.press("escape")
                await pilot.pause()

        asyncio.run(_run())

        veto_events = [
            e for e in fresh_bus.replay_buffer()
            if e.get("event_type") == "control.signal_veto"
        ]
        assert veto_events == []
        assert not log_path.exists()
