"""US-004 veto-countdown subscriber tests — real EventBus + real BuddyApp Pilot.

NO MOCKS per CLAUDE.md No-Mock Rule. The test wires the real
``EventBus`` singleton to a real ``BuddyApp`` running through Textual's
``run_test`` Pilot harness, publishes a ``signal.pending`` event, and
asserts the ``_start_veto_countdown`` method is invoked with the correct
arguments within 100ms (AC-6).

The recording is done by subclassing ``BuddyApp`` and overriding
``_start_veto_countdown`` to append to a list — a real subclass with a
real method override, NOT ``unittest.mock``.

The async tests use ``asyncio.run`` directly (no pytest-asyncio
dependency).
"""
from __future__ import annotations

import asyncio
import time
import uuid
from typing import Any, Dict, Iterator, List

import pytest

from src.scanner.automation.event_bus import (
    EventBus,
    get_event_bus,
    reset_event_bus,
)
from src.scanner.config import ScannerConfig
from src.tui.app import BuddyApp


@pytest.fixture
def fresh_bus() -> Iterator[EventBus]:
    """Reset the global trading bus between tests for isolation."""
    reset_event_bus()
    bus = get_event_bus()
    yield bus
    reset_event_bus()


_TUI_THEME_PATH = (
    __import__("pathlib").Path(__file__).resolve().parent.parent
    / "src" / "tui" / "theme.tcss"
)


class _RecordingApp(BuddyApp):
    """Subclass of BuddyApp that records every _start_veto_countdown invocation.

    Real subclass with a real method override — not a mock. Captures
    ``(pair, direction, duration_s, signal_id, monotonic_ts)`` so the
    test can both verify the args and bound the dispatch latency.

    ``CSS_PATH`` is repointed at the real ``src/tui/theme.tcss`` because
    Textual resolves the relative path against the *defining module's*
    directory — without this, the subclass lookup hits ``tests/theme.tcss``
    which does not exist.
    """

    CSS_PATH = str(_TUI_THEME_PATH)

    def __init__(self) -> None:
        super().__init__(live=False)
        self.captured: List[Dict[str, Any]] = []

    def _start_veto_countdown(  # type: ignore[override]
        self,
        pair: str,
        direction: str,
        duration_s: int,
        signal_id: str = "",
    ) -> None:
        self.captured.append({
            "pair": pair,
            "direction": direction,
            "duration_s": duration_s,
            "signal_id": signal_id,
            "ts": time.monotonic(),
        })


def _publish_signal_pending(
    bus: EventBus,
    *,
    pair: str = "EUR_USD",
    direction: str = "LONG",
    signal_id: str = "",
) -> str:
    sid = signal_id or str(uuid.uuid4())
    bus.publish("signal.pending", {
        "signal_id": sid,
        "pair": pair,
        "direction": direction,
        "entry": 1.0823,
        "sl": 1.0810,
        "tp": 1.0850,
    })
    return sid


async def _wait_until(predicate, *, timeout_s: float, pilot) -> bool:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        await pilot.pause()
        if predicate():
            return True
    return False


class TestVetoSubscription:

    def test_signal_pending_invokes_countdown_within_100ms(self, fresh_bus):
        """AC-6: a signal.pending event invokes _start_veto_countdown within 100ms."""
        host = _RecordingApp()

        async def _run():
            async with host.run_test(size=(80, 24)) as pilot:
                await pilot.pause()
                # Drain any startup invocations (none expected — but be defensive).
                host.captured.clear()

                t0 = time.monotonic()
                sid = _publish_signal_pending(
                    fresh_bus, pair="EUR_USD", direction="LONG"
                )
                # Allow up to 1s of pilot ticks; assert latency < 100ms below.
                got_it = await _wait_until(
                    lambda: any(c["signal_id"] == sid for c in host.captured),
                    timeout_s=1.0,
                    pilot=pilot,
                )
                assert got_it, "_start_veto_countdown was never invoked"
                first = next(c for c in host.captured if c["signal_id"] == sid)
                dispatch_ms = (first["ts"] - t0) * 1000.0
                # AC-6 says "within 100ms" — allow a small margin for the
                # Pilot ticker (Textual's default poll cadence is ~50ms).
                assert dispatch_ms < 100.0, (
                    f"dispatch latency {dispatch_ms:.1f}ms exceeded 100ms budget"
                )

        asyncio.run(_run())

    def test_countdown_called_with_config_duration(self, fresh_bus):
        """AC-2: duration_s comes from getattr(config, 'pre_trade_veto_seconds', 5)."""
        host = _RecordingApp()

        async def _run():
            async with host.run_test(size=(80, 24)) as pilot:
                await pilot.pause()
                host.captured.clear()
                _publish_signal_pending(fresh_bus, pair="GBP_USD", direction="SHORT")
                await _wait_until(
                    lambda: bool(host.captured), timeout_s=1.0, pilot=pilot
                )
                assert host.captured, "expected one capture"
                cap = host.captured[0]
                assert cap["pair"] == "GBP_USD"
                assert cap["direction"] == "SHORT"
                # ScannerConfig default is 5 (the AC-stated fallback).
                expected = getattr(ScannerConfig(), "pre_trade_veto_seconds", 5)
                assert cap["duration_s"] == expected

        asyncio.run(_run())

    def test_second_signal_cancels_prior_countdown(self, fresh_bus):
        """AC-3: a fresh signal.pending cancels the prior countdown — no zombies.

        Drives the REAL _start_veto_countdown (not the recording override)
        and asserts the bookkeeping attribute reflects exactly one active
        countdown after a second event arrives.
        """
        # Use a real BuddyApp so the production _start_veto_countdown runs.
        host = BuddyApp(live=False)

        async def _run():
            async with host.run_test(size=(80, 24)) as pilot:
                await pilot.pause()

                _publish_signal_pending(fresh_bus, pair="EUR_USD", signal_id="sig-1")
                await _wait_until(
                    lambda: getattr(host, "_active_veto_signal_id", "") == "sig-1",
                    timeout_s=1.0,
                    pilot=pilot,
                )
                first_cancel = host._active_veto_countdown_cancel
                assert first_cancel is not None, "first countdown did not start"
                assert host._active_veto_signal_id == "sig-1"

                _publish_signal_pending(fresh_bus, pair="USD_JPY", signal_id="sig-2")
                await _wait_until(
                    lambda: getattr(host, "_active_veto_signal_id", "") == "sig-2",
                    timeout_s=1.0,
                    pilot=pilot,
                )
                # The bookkeeping must have flipped to the new signal AND the
                # cancel handle must be a *different* callable (the prior was
                # stopped before the new one started).
                assert host._active_veto_signal_id == "sig-2"
                assert host._active_veto_countdown_cancel is not first_cancel

        asyncio.run(_run())

    def test_non_pending_events_are_ignored(self, fresh_bus):
        """Worker filters event_type — only signal.pending triggers the countdown."""
        host = _RecordingApp()

        async def _run():
            async with host.run_test(size=(80, 24)) as pilot:
                await pilot.pause()
                host.captured.clear()

                # Publish events the subscriber must ignore.
                fresh_bus.publish("signal.executed", {"pair": "EUR_USD"})
                fresh_bus.publish("control.kill", {})
                fresh_bus.publish("signal.rejected", {"pair": "EUR_USD"})
                fresh_bus.publish("adjustment.proposed", {"k": "v"})

                # Now publish one real signal.pending so we have something to
                # wait for (otherwise the test would have to sleep arbitrarily).
                sid = _publish_signal_pending(fresh_bus, pair="USD_CHF")
                await _wait_until(
                    lambda: any(c["signal_id"] == sid for c in host.captured),
                    timeout_s=1.0,
                    pilot=pilot,
                )

                # Exactly one capture — none of the other event types fired it.
                pending_caps = [
                    c for c in host.captured if c["signal_id"] == sid
                ]
                assert len(pending_caps) == 1
                # And no spurious captures for the ignored types either.
                assert len(host.captured) == 1

        asyncio.run(_run())

    def test_countdown_uses_real_brain_log_widget(self, fresh_bus):
        """AC-4: countdown banner is rendered via the existing #brain-log widget.

        Drives the REAL _start_veto_countdown and confirms a banner line lands
        in the brain log widget.
        """
        from textual.widgets import RichLog

        host = BuddyApp(live=False)

        async def _run():
            async with host.run_test(size=(80, 24)) as pilot:
                await pilot.pause()

                _publish_signal_pending(fresh_bus, pair="AUD_USD", direction="LONG")
                await _wait_until(
                    lambda: host._active_veto_countdown_cancel is not None,
                    timeout_s=1.0,
                    pilot=pilot,
                )

                # Let one tick of the 1.0s set_interval fire so the banner
                # actually lands in the brain log.
                await asyncio.sleep(1.2)
                await pilot.pause()

                brain = host.query_one("#brain-log", RichLog)
                rendered = "\n".join(str(line) for line in brain.lines)
                assert "AUD_USD" in rendered
                assert "EXECUTING" in rendered
                assert "A to abort" in rendered

                # Clean up the ticker so the pilot exit isn't slowed by it.
                if host._active_veto_countdown_cancel is not None:
                    host._active_veto_countdown_cancel()

        asyncio.run(_run())
