"""US-011 integration test — OANDA auto-reconnect on transient disconnect.

Drives a real :class:`DataProvider` through the four-step contract from the
story acceptance criteria:

    (a) connect succeeds against real OANDA practice creds
    (b) simulate disconnect by flipping the connection flag (the broker
        client itself stays live so the subsequent reconnect can succeed)
    (c) refresh fires a reconnect attempt
    (d) reconnect succeeds and the snapshot ``oanda_connected`` flag flips
        OANDA OFF -> OANDA ON

NO MOCKS. Uses real OANDA practice creds via env (skip if absent). A
separate test exercises the backoff schedule against intentionally bad
creds — still no mocks (real API call that 401s).
"""
from __future__ import annotations

import os
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from src.tui.data_provider import (
    _RECONNECT_BACKOFF_CAP_SECONDS,
    _RECONNECT_BACKOFF_SCHEDULE,
    DataProvider,
    _next_reconnect_delay_seconds,
)


pytestmark = pytest.mark.integration


# ─────────────────────────── helpers ─────────────────────────────────


def _have_oanda_practice_creds() -> bool:
    """True iff the practice creds the live broker needs are in env."""
    token = os.environ.get("OANDA_API_TOKEN") or os.environ.get("OANDA_API_KEY")
    account = os.environ.get("OANDA_ACCOUNT_ID")
    return bool(token and account)


@pytest.fixture
def real_provider(tmp_path: Path) -> DataProvider:
    """A DataProvider rooted at tmp_path (brain feed writes land there).

    Skips if OANDA creds aren't in env — this fixture is for the "real
    connect must succeed" path (a)/(c)/(d) of the acceptance criteria.
    """
    if not _have_oanda_practice_creds():
        pytest.skip(
            "OANDA practice creds (OANDA_API_TOKEN + OANDA_ACCOUNT_ID) "
            "not in env — required for the real reconnect-success path"
        )
    return DataProvider(project_root=str(tmp_path))


# ───────────────── (a)+(b)+(c)+(d) full real-OANDA cycle ──────────────


def test_reconnect_after_simulated_disconnect_real_oanda(
    real_provider: DataProvider, tmp_path: Path
) -> None:
    """Full a→b→c→d cycle against real OANDA practice.

    Verifies the contract end-to-end: initial connect → simulated drop →
    refresh-triggered reconnect → header flag flips back to ON → attempts
    counter resets → brain feed records the recovery.
    """
    # (a) Initial connect succeeds against real OANDA.
    assert real_provider.connect() is True
    assert real_provider.oanda_connected is True
    assert real_provider._reconnect_attempts == 0

    # (b) Simulate disconnect: the broker client is still live but we flag
    #     the connection as dead the same way _refresh_account would on a
    #     transient API failure. Force the next-attempt deadline to NOW so
    #     refresh() doesn't have to wait out the backoff for this test.
    real_provider._connected = False
    real_provider._next_reconnect_at = datetime.now(timezone.utc)
    assert real_provider.oanda_connected is False

    # (c)+(d) refresh fires reconnect attempt, succeeds, snapshot reflects it.
    snap = real_provider.refresh()
    assert real_provider.oanda_connected is True, (
        "reconnect should flip self._connected back to True after a "
        "successful get_nav() on the existing broker client"
    )
    assert snap.oanda_connected is True, (
        "snap.oanda_connected drives the OANDA OFF→ON header indicator "
        "(see app.py header rendering); must reflect post-reconnect state"
    )
    # Successful reconnect clears the attempt counter so backoff restarts
    # from 5s if the connection drops again later.
    assert real_provider._reconnect_attempts == 0

    # Brain feed should have recorded the recovery.
    feed_path = tmp_path / ".claude" / "brain" / "feed.jsonl"
    assert feed_path.exists(), "reconnect must tee an INFO line to brain feed"
    content = feed_path.read_text()
    assert "OANDA reconnected" in content, (
        "success log must include the exact phrase 'OANDA reconnected' "
        "(spec AC-4)"
    )


# ─────────── backoff schedule + gate (no-creds-friendly path) ──────────


def _force_bad_creds_provider(tmp_path: Path, monkeypatch) -> DataProvider:
    """Build a DataProvider with intentionally bad creds.

    Every reconnect attempt will hit real OANDA and 401/network-fail, which
    is exactly what we want for exercising the backoff schedule and the
    gate-time short-circuit. NO MOCKS — the failures are real API calls.
    """
    monkeypatch.setenv("OANDA_API_TOKEN", "us011-invalid-token-deadbeef")
    monkeypatch.setenv("OANDA_ACCOUNT_ID", "000-000-0000000-000")
    monkeypatch.setenv("OANDA_ENVIRONMENT", "practice")
    return DataProvider(project_root=str(tmp_path))


def test_backoff_schedule_increments_through_cap(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Failed attempts walk 5s→10s→30s→60s and cap at 60s.

    Drives refresh() six times against bad creds. Each iteration forces the
    next-reconnect deadline back to NOW so the gate doesn't block — then
    asserts the resulting delay matches the documented schedule.
    """
    provider = _force_bad_creds_provider(tmp_path, monkeypatch)

    # Sanity check on the pure schedule helper before touching refresh().
    assert _RECONNECT_BACKOFF_SCHEDULE == (5, 10, 30, 60)
    assert _RECONNECT_BACKOFF_CAP_SECONDS == 60
    assert [_next_reconnect_delay_seconds(i) for i in (1, 2, 3, 4, 5, 6)] == [
        5,
        10,
        30,
        60,
        60,
        60,
    ]

    expected_delays = [5, 10, 30, 60, 60, 60]
    for i, expected in enumerate(expected_delays, start=1):
        provider._next_reconnect_at = datetime.now(timezone.utc)
        before = datetime.now(timezone.utc)
        snap = provider.refresh()
        # Connection must still be off (bad creds → can't succeed).
        assert provider.oanda_connected is False
        assert snap.oanda_connected is False
        # Attempts counter advances by exactly one per refresh cycle.
        assert provider._reconnect_attempts == i, (
            f"attempt {i} should bump counter to {i}, "
            f"got {provider._reconnect_attempts}"
        )
        # next_reconnect_at must now be ~expected seconds in the future.
        delay_seconds = (
            provider._next_reconnect_at - before
        ).total_seconds()
        # Lower bound: schedule value minus a tiny test-runtime margin.
        # Upper bound: schedule value plus a small slop for the API call
        # latency (the failure has to round-trip first).
        assert expected - 0.5 <= delay_seconds <= expected + 5.0, (
            f"attempt {i}: expected ~{expected}s, got {delay_seconds:.2f}s"
        )


def test_refresh_respects_gate_before_deadline(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Refresh refuses to retry before ``_next_reconnect_at``.

    First refresh fails and schedules the next attempt 5s out. Subsequent
    refreshes that fire BEFORE the deadline must be no-ops on the reconnect
    state — the attempt counter must not move, and ``_next_reconnect_at``
    must not change.
    """
    provider = _force_bad_creds_provider(tmp_path, monkeypatch)
    provider._next_reconnect_at = datetime.now(timezone.utc)

    # First failed attempt — schedules next attempt 5s out.
    provider.refresh()
    assert provider._reconnect_attempts == 1
    deadline_after_first = provider._next_reconnect_at
    assert deadline_after_first > datetime.now(timezone.utc), (
        "first failure should push the next-attempt deadline into the future"
    )

    # Second refresh — deadline still in the future, must not attempt.
    provider.refresh()
    assert provider._reconnect_attempts == 1, (
        "refresh fired before the backoff deadline must NOT increment "
        "the attempt counter (gate broken)"
    )
    assert provider._next_reconnect_at == deadline_after_first, (
        "refresh fired before the backoff deadline must NOT mutate "
        "_next_reconnect_at"
    )


def test_brain_feed_records_each_attempt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Every reconnect attempt emits a brain-feed line with attempt # + delay.

    AC-4: "Each attempt logs to brain feed at INFO with attempt number and
    delay." The lines land in ``.claude/brain/feed.jsonl`` under
    project_root — same surface the operator reads in the F1 panel.
    """
    provider = _force_bad_creds_provider(tmp_path, monkeypatch)

    # Drive two failed attempts.
    provider._next_reconnect_at = datetime.now(timezone.utc)
    provider.refresh()
    provider._next_reconnect_at = datetime.now(timezone.utc)
    provider.refresh()

    feed_path = tmp_path / ".claude" / "brain" / "feed.jsonl"
    assert feed_path.exists(), "brain feed must be created on first attempt"
    lines = feed_path.read_text().strip().splitlines()
    assert len(lines) >= 2, (
        f"expected >=2 brain-feed lines for 2 attempts, got {len(lines)}"
    )

    # Each attempt line must mention the attempt number and the retry delay.
    joined = "\n".join(lines)
    assert "attempt 1" in joined
    assert "attempt 2" in joined
    assert "retry in 5s" in joined  # attempt 1 → next delay 5s
    assert "retry in 10s" in joined  # attempt 2 → next delay 10s


def test_initial_state_matches_contract(tmp_path: Path) -> None:
    """Boot-time state of the new auto-reconnect fields.

    AC-1: "data_provider tracks ``self._reconnect_attempts: int`` and
    ``self._next_reconnect_at: datetime``." Verify both are present, of the
    documented types, and start in a sensible state (0 attempts, deadline
    in the past or now so the very first refresh is allowed to try).
    """
    provider = DataProvider(project_root=str(tmp_path))
    assert isinstance(provider._reconnect_attempts, int)
    assert provider._reconnect_attempts == 0
    assert isinstance(provider._next_reconnect_at, datetime)
    # Boot deadline must not be far in the future — operators should never
    # have to wait the full backoff window before the first connect attempt.
    assert provider._next_reconnect_at <= datetime.now(timezone.utc) + timedelta(
        seconds=1
    )
    # Property surface (AC-2 uses self.oanda_connected as the gate).
    assert provider.oanda_connected is False
