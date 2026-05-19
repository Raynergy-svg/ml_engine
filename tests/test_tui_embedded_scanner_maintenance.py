"""Tests for IdleMaintenance integration into EmbeddedScanner.

Pre-2026-05-19, IdleMaintenance was only wired into ContinuousScanner
(the headless CLI path) per the audit captured in maintenance.py.
EmbeddedScanner — the actual live TUI runtime per CLAUDE.md's "Tier 7
runtime entry point" section — never instantiated it, so journal sync
and background gate retrains weren't running in production.

This module validates the wiring added on 2026-05-19:
  • __init__ leaves _maintenance None until automation modules init
  • _init_automation_modules instantiates a real IdleMaintenance
  • _run_post_scan_automation calls run_if_needed only at the throttle
  • _maintenance_interval == 0 disables the call (guard rail)
  • shutdown signals cooperative stop via _stop_requested
  • exceptions inside run_if_needed do NOT propagate to the cycle

NO MOCKS per .claude/rules/improvement.md. Test doubles are real
subclasses that extend IdleMaintenance with counters or failure
injection; the base behaviour is preserved.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from src.scanner.automation.maintenance import IdleMaintenance
from src.tui.embedded_scanner import EmbeddedScanner


# ──────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────


class _Result:
    """Minimal stand-in for ScanResult.

    _run_post_scan_automation only reads result.analyses, so an empty
    list is the cheapest pass-through that exercises the wiring without
    pulling Scanner internals. Not a mock — a one-attribute real class.
    """

    analyses: list[Any] = []


class _RecordingMaintenance(IdleMaintenance):
    """Real IdleMaintenance subclass with call counters.

    Per the No-Mock rule, this extends — not replaces — the production
    class. The base ``__init__`` runs; ``run_if_needed`` and ``stop`` are
    overridden to increment counters and then delegate to ``super()``,
    preserving the real journal-sync / cooldown / shutdown behaviour.
    """

    def __init__(self) -> None:
        super().__init__()
        self.run_if_needed_calls = 0
        self.stop_calls = 0

    def run_if_needed(self, oanda_client: Any = None) -> None:  # type: ignore[override]
        self.run_if_needed_calls += 1
        super().run_if_needed(oanda_client=oanda_client)

    def stop(self) -> None:  # type: ignore[override]
        self.stop_calls += 1
        super().stop()


class _RaisingMaintenance(IdleMaintenance):
    """Real IdleMaintenance subclass that raises on run_if_needed.

    Used to verify that the post-scan automation hook swallows
    maintenance failures (mirrors ObservationLog / OnlineRL / Drift
    error-handling pattern at the same call site).
    """

    def run_if_needed(self, oanda_client: Any = None) -> None:  # type: ignore[override]
        raise RuntimeError("simulated maintenance crash")


def _make_scanner() -> EmbeddedScanner:
    """Construct an EmbeddedScanner without firing _init_scanner.

    _init_scanner loads ML models from disk; expensive and brittle in a
    unit-test environment. The wiring tests only exercise __init__,
    _init_automation_modules, _run_post_scan_automation, and shutdown,
    none of which depend on a real Scanner being attached.

    The brain_callback positional argument is the TUI's _write_brain
    handler in production; tests pass a no-op since we only verify
    state-mutation behaviour, not brain-feed wiring.
    """
    return EmbeddedScanner(
        brain_callback=lambda _msg: None,
        project_root=str(Path.cwd()),
    )


# ──────────────────────────────────────────────────────────────────────
# Tests
# ──────────────────────────────────────────────────────────────────────


def test_init_leaves_maintenance_none() -> None:
    """Pre-_init_automation_modules, _maintenance is None.

    Guards against accidentally constructing the helper at import time
    or in __init__ (which would couple TUI startup to OANDA client
    availability and slow boot).
    """
    s = _make_scanner()
    assert s._maintenance is None
    # Defaults from the wiring commit:
    assert s._maintenance_interval == 5


def test_init_automation_modules_creates_real_idle_maintenance() -> None:
    """_init_automation_modules populates _maintenance with a real
    IdleMaintenance — not a mock, not None, not a placeholder."""
    s = _make_scanner()
    s._init_automation_modules()
    assert isinstance(s._maintenance, IdleMaintenance)
    # Real class invariants — these are the contract surfaces shutdown()
    # and the cycle hook rely on:
    assert hasattr(s._maintenance, "run_if_needed")
    assert hasattr(s._maintenance, "stop")
    assert s._maintenance._stop_requested is not None
    assert not s._maintenance._stop_requested.is_set()


def test_post_scan_automation_throttles_to_interval() -> None:
    """_run_post_scan_automation only fires run_if_needed when
    scan_count % maintenance_interval == 0 — protects OANDA quota."""
    s = _make_scanner()
    s._maintenance = _RecordingMaintenance()
    s._maintenance_interval = 5

    # Cycles 1..4 must not fire (off-interval).
    for cycle in range(1, 5):
        s._scan_count = cycle
        s._post_scan_automation(_Result())
    assert s._maintenance.run_if_needed_calls == 0, (
        "maintenance must not fire off-interval"
    )

    # Cycle 5 fires (5 % 5 == 0).
    s._scan_count = 5
    s._post_scan_automation(_Result())
    assert s._maintenance.run_if_needed_calls == 1

    # Cycle 10 fires again.
    s._scan_count = 10
    s._post_scan_automation(_Result())
    assert s._maintenance.run_if_needed_calls == 2

    # Cycle 11..14 stays at 2.
    for cycle in range(11, 15):
        s._scan_count = cycle
        s._post_scan_automation(_Result())
    assert s._maintenance.run_if_needed_calls == 2


def test_maintenance_interval_zero_disables_call() -> None:
    """_maintenance_interval == 0 disables the call entirely.

    Avoids both div-by-zero (impossible in Python but worth the
    explicit gate) and any "every cycle" surprise on operators who
    set the field to 0 as a kill-switch."""
    s = _make_scanner()
    s._maintenance = _RecordingMaintenance()
    s._maintenance_interval = 0

    for cycle in range(1, 25):
        s._scan_count = cycle
        s._post_scan_automation(_Result())
    assert s._maintenance.run_if_needed_calls == 0


def test_shutdown_signals_cooperative_stop() -> None:
    """shutdown() must call IdleMaintenance.stop() so any in-flight
    retrain subprocess can drain. Observe _stop_requested.is_set()."""
    s = _make_scanner()
    s._init_automation_modules()
    assert not s._maintenance._stop_requested.is_set()
    s.shutdown()
    assert s._maintenance._stop_requested.is_set()


def test_maintenance_exception_does_not_break_cycle() -> None:
    """A maintenance crash must not propagate out of the post-scan
    hook — matches the existing pattern for ObservationLog, OnlineRL,
    Drift, ConfigAdjuster, and Freshness at the same call site."""
    s = _make_scanner()
    s._maintenance = _RaisingMaintenance()
    s._maintenance_interval = 5
    s._scan_count = 5
    # Must not raise — the cycle would silently swallow the error and
    # surface it via logger.debug in production.
    s._post_scan_automation(_Result())


def test_shutdown_safe_when_maintenance_init_failed() -> None:
    """If IdleMaintenance import or init fails, _maintenance stays
    None. shutdown() must not crash on the None branch."""
    s = _make_scanner()
    assert s._maintenance is None
    # shutdown() guards on `if self._maintenance is not None` — this
    # call must complete without raising.
    s.shutdown()
