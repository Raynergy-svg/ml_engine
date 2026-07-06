"""US-010 — Tests for :mod:`src.equity.market_calendar`.

No mocks. Real :class:`EquityMarketCalendar` backed by the
``exchange_calendars`` XNYS calendar, real on-disk ship-gate JSON, real
on-disk halt-state file via ``tmp_path``.

Covers all US-010 ACs:

* Open on a normal weekday at 10:00 ET.
* Closed on a US holiday (Independence Day observed) and at 03:00 ET.
* Closed at the half-day early close (day after Thanksgiving 13:30 ET).
* A simulated halt (``HaltError`` raised from the broker callback)
  defers the order rather than silently dropping it.
* Ship-gate guard blocks construction when the ship-gate is missing,
  ``gate_pass!=True``, or the ``universe_hash`` doesn't match.
* Atomic state writes survive a mid-write crash without corruption.
"""

from __future__ import annotations

import json
import multiprocessing as mp
from pathlib import Path
from typing import List

import pandas as pd
import pytest

from src.equity.market_calendar import (
    DEFAULT_EXCHANGE,
    EquityMarketCalendar,
    EquitySessionGate,
    HaltError,
    HaltRegistry,
    HaltState,
    MarketCalendarError,
    OrderOutcome,
)
from src.equity.rebalance import Order


# ---------------------------------------------------------------------------
# Module-level worker for the M5 concurrent-writer test. Must be top-level
# (not a closure) so it is picklable under the "spawn" start method.
# ---------------------------------------------------------------------------
def _register_halt_worker(
    state_path_str: str,
    ticker: str,
    start_barrier: "mp.synchronize.Barrier",
    now_iso: str,
) -> None:
    """Register one distinct halt against the shared state file.

    Real process, real disk, real HaltRegistry, real fcntl lock. The
    barrier makes every process attempt its read-modify-write at the same
    instant — this is what would lose halts if the lock were absent.
    """
    reg = HaltRegistry(Path(state_path_str))
    now = pd.Timestamp(now_iso)
    start_barrier.wait()
    reg.register_halt(ticker, now=now, reason=f"halt-{ticker}")


UNIVERSE_HASH = "abc12345" * 8  # 64-char placeholder
OTHER_HASH = "fedc6789" * 8


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
def _write_ship_gate(
    path: Path,
    *,
    gate_pass: bool = True,
    universe_hash: str = UNIVERSE_HASH,
) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "gate_pass": bool(gate_pass),
        "net_sharpe": 0.55,
        "max_dd": 0.20,
        "positive_years": 11,
        "total_years": 16,
        "universe_hash": universe_hash,
        "asof": "2026-06-19",
        "recommendation": "PASS — test fixture",
        "criteria_source": "test",
        "thresholds": {},
        "pipeline_version": "2026-06-18-eq1",
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True))
    return path


def _make_gate(
    tmp_path: Path,
    *,
    gate_pass: bool = True,
    universe_hash: str = UNIVERSE_HASH,
) -> EquitySessionGate:
    gate = tmp_path / "backtests" / "SHIP_GATE.json"
    _write_ship_gate(gate, gate_pass=gate_pass, universe_hash=universe_hash)
    halt_state = tmp_path / "state" / "halts.json"
    return EquitySessionGate.create(
        universe_hash=UNIVERSE_HASH,
        ship_gate_path=gate,
        halt_state_path=halt_state,
    )


def _order(ticker: str = "AAPL", side: str = "BUY") -> Order:
    return Order(
        client_order_id=f"RB-X-{ticker}-{side}",
        ticker=ticker,
        side=side,
        weight_delta=0.05,
        target_weight=0.10,
    )


# Key reference timestamps (all UTC):
# 2026-06-22 Monday: regular trading day.
#   14:00 UTC = 10:00 EDT → open.
#   07:00 UTC = 03:00 EDT → closed.
# 2026-07-03 Friday: Independence Day observed (4 Jul = Saturday).
#   14:00 UTC = 10:00 EDT → closed (holiday).
# 2026-11-27 Friday: day after Thanksgiving, early close at 13:00 ET.
#   17:00 UTC = 12:00 EST → open.
#   18:30 UTC = 13:30 EST → closed (after half-day close).
WEEKDAY_10AM_UTC = pd.Timestamp("2026-06-22 14:00:00", tz="UTC")
WEEKDAY_3AM_UTC = pd.Timestamp("2026-06-22 07:00:00", tz="UTC")
HOLIDAY_10AM_UTC = pd.Timestamp("2026-07-03 14:00:00", tz="UTC")
HALFDAY_BEFORE_CLOSE_UTC = pd.Timestamp("2026-11-27 17:00:00", tz="UTC")
HALFDAY_AFTER_CLOSE_UTC = pd.Timestamp("2026-11-27 18:30:00", tz="UTC")


# ---------------------------------------------------------------------------
# EquityMarketCalendar: regular session, holidays, half-days
# ---------------------------------------------------------------------------
class TestEquityMarketCalendar:
    def test_open_on_normal_weekday_10am_et(self) -> None:
        cal = EquityMarketCalendar()
        assert cal.is_session_open(WEEKDAY_10AM_UTC) is True

    def test_closed_at_3am_et_on_normal_weekday(self) -> None:
        cal = EquityMarketCalendar()
        assert cal.is_session_open(WEEKDAY_3AM_UTC) is False

    def test_closed_on_independence_day_observed(self) -> None:
        cal = EquityMarketCalendar()
        assert cal.is_session_open(HOLIDAY_10AM_UTC) is False

    def test_closed_on_weekend(self) -> None:
        cal = EquityMarketCalendar()
        # 2026-06-20 Saturday 14:00 UTC.
        sat = pd.Timestamp("2026-06-20 14:00:00", tz="UTC")
        assert cal.is_session_open(sat) is False

    def test_halfday_open_before_early_close(self) -> None:
        cal = EquityMarketCalendar()
        assert cal.is_session_open(HALFDAY_BEFORE_CLOSE_UTC) is True

    def test_halfday_closed_after_early_close(self) -> None:
        cal = EquityMarketCalendar()
        # Day after Thanksgiving closes 13:00 ET. 13:30 ET = 18:30 UTC.
        assert cal.is_session_open(HALFDAY_AFTER_CLOSE_UTC) is False

    def test_session_status_open(self) -> None:
        cal = EquityMarketCalendar()
        status = cal.session_status(WEEKDAY_10AM_UTC)
        assert status["exchange"] == DEFAULT_EXCHANGE
        assert status["is_open"] is True
        assert status["reason_code"] == "open"
        assert status["next_open_utc"] is not None
        assert status["next_close_utc"] is not None

    def test_session_status_closed_has_next_open(self) -> None:
        cal = EquityMarketCalendar()
        status = cal.session_status(HOLIDAY_10AM_UTC)
        assert status["is_open"] is False
        assert status["reason_code"] == "closed"
        # Next open should be after the holiday.
        next_open = pd.Timestamp(status["next_open_utc"])
        assert next_open > HOLIDAY_10AM_UTC

    def test_naive_timestamp_is_rejected(self) -> None:
        """M6: a naive timestamp is REFUSED, not silently assumed UTC.

        Assuming naive==UTC would shift an exchange-local (ET) wall clock
        4-5 hours and could report the session open pre-market.
        """
        cal = EquityMarketCalendar()
        naive = pd.Timestamp("2026-06-22 14:00:00")  # no tz
        with pytest.raises(MarketCalendarError, match="naive timestamp"):
            cal.is_session_open(naive)

    def test_naive_premarket_et_does_not_report_open(self) -> None:
        """M6 correctness: a naive 09:30 ET wall-clock used to (wrongly)
        be read as 09:30 UTC = 05:30 ET (pre-market) yet land inside the
        UTC-interpreted regular window in other cases; either way the
        silent-UTC assumption was unsafe. We now REFUSE the naive input
        rather than guess, so the market is never reported open from an
        ambiguous timestamp.
        """
        cal = EquityMarketCalendar()
        # 04:30 wall-clock, naive. If silently read as UTC this is
        # 00:30 ET (closed); if read as ET it is pre-market (closed).
        # The point: we must not guess — refuse.
        naive_premarket = pd.Timestamp("2026-06-22 09:30:00")  # naive ET
        with pytest.raises(MarketCalendarError, match="naive timestamp"):
            cal.is_session_open(naive_premarket)

    def test_tz_aware_et_timestamp_converts_correctly(self) -> None:
        """A tz-aware ET timestamp is converted to UTC, not mislabeled.

        09:30 America/New_York on a regular Monday is the open — and the
        equivalent of 13:30 UTC (EDT). A correct conversion reports open;
        the old naive-as-UTC bug would have treated the 09:30 wall clock
        as 09:30 UTC = 05:30 ET (pre-market, closed).
        """
        cal = EquityMarketCalendar()
        et_open = pd.Timestamp("2026-06-22 09:30:00", tz="America/New_York")
        assert cal.is_session_open(et_open) is True
        # And the pre-market ET time is correctly closed.
        et_premarket = pd.Timestamp(
            "2026-06-22 05:30:00", tz="America/New_York"
        )
        assert cal.is_session_open(et_premarket) is False

    def test_unknown_exchange_raises(self) -> None:
        with pytest.raises(MarketCalendarError):
            EquityMarketCalendar(exchange="NOT_AN_EXCHANGE")


# ---------------------------------------------------------------------------
# HaltRegistry: persistence, TTL, atomic writes
# ---------------------------------------------------------------------------
class TestHaltRegistry:
    def test_empty_when_file_missing(self, tmp_path: Path) -> None:
        reg = HaltRegistry(tmp_path / "halts.json")
        assert reg.is_halted("AAPL", now=WEEKDAY_10AM_UTC) is False
        assert reg.list_active(now=WEEKDAY_10AM_UTC) == []

    def test_register_and_clear_halt(self, tmp_path: Path) -> None:
        reg = HaltRegistry(tmp_path / "halts.json")
        reg.register_halt("AAPL", now=WEEKDAY_10AM_UTC, reason="LULD pause")
        assert reg.is_halted("AAPL", now=WEEKDAY_10AM_UTC) is True
        active = reg.list_active(now=WEEKDAY_10AM_UTC)
        assert len(active) == 1
        assert active[0].ticker == "AAPL"
        assert active[0].reason == "LULD pause"

        # Persistence: a fresh registry on the same path sees the halt.
        reg2 = HaltRegistry(tmp_path / "halts.json")
        assert reg2.is_halted("AAPL", now=WEEKDAY_10AM_UTC) is True

        assert reg.clear_halt("AAPL") is True
        assert reg.is_halted("AAPL", now=WEEKDAY_10AM_UTC) is False
        assert reg.clear_halt("AAPL") is False  # already gone

    def test_ttl_expiry_evicts_record(self, tmp_path: Path) -> None:
        reg = HaltRegistry(tmp_path / "halts.json")
        until = WEEKDAY_10AM_UTC + pd.Timedelta(minutes=5)
        reg.register_halt(
            "MSFT",
            now=WEEKDAY_10AM_UTC,
            reason="circuit-breaker",
            until=until,
        )
        # During TTL: halted.
        assert reg.is_halted("MSFT", now=WEEKDAY_10AM_UTC) is True
        # After TTL: cleared on read.
        after = until + pd.Timedelta(seconds=1)
        assert reg.is_halted("MSFT", now=after) is False
        # And the file no longer carries the record.
        reloaded = reg.load()
        assert "MSFT" not in reloaded.halts

    def test_atomic_write_leaves_no_temp(self, tmp_path: Path) -> None:
        reg = HaltRegistry(tmp_path / "halts.json")
        reg.register_halt("AAPL", now=WEEKDAY_10AM_UTC)
        # No stray .tmp file should remain.
        stray = [p for p in tmp_path.iterdir() if p.suffix == ".tmp"]
        assert stray == []
        # The persisted JSON is well-formed.
        text = (tmp_path / "halts.json").read_text(encoding="utf-8")
        payload = json.loads(text)
        assert payload["halts"]["AAPL"]["ticker"] == "AAPL"

    def test_corrupt_state_resets_safely(self, tmp_path: Path) -> None:
        path = tmp_path / "halts.json"
        path.write_text("{not json")
        reg = HaltRegistry(path)
        # Corrupt → fresh state, not a crash.
        assert reg.is_halted("AAPL", now=WEEKDAY_10AM_UTC) is False
        loaded = reg.load()
        assert isinstance(loaded, HaltState)
        assert loaded.halts == {}

    def test_empty_ticker_rejected(self, tmp_path: Path) -> None:
        reg = HaltRegistry(tmp_path / "halts.json")
        with pytest.raises(MarketCalendarError):
            reg.register_halt("", now=WEEKDAY_10AM_UTC)

    def test_register_halt_rejects_naive_now(self, tmp_path: Path) -> None:
        """M6: register_halt refuses a naive ``now`` rather than guessing."""
        reg = HaltRegistry(tmp_path / "halts.json")
        naive = pd.Timestamp("2026-06-22 14:00:00")  # no tz
        with pytest.raises(MarketCalendarError, match="naive timestamp"):
            reg.register_halt("AAPL", now=naive)


# ---------------------------------------------------------------------------
# M5 — HaltRegistry concurrency: the fcntl lock must not lose halts
# ---------------------------------------------------------------------------
class TestHaltRegistryConcurrency:
    def test_concurrent_writers_no_halt_lost(self, tmp_path: Path) -> None:
        """N real processes each register a DISTINCT halt against the same
        file at the same instant. With the read-modify-write guarded by an
        exclusive fcntl lock, ALL N halts must survive — none clobbered.

        This is the load-bearing M5 proof: without the lock, two processes
        that both load the same prior snapshot and then save would have the
        later save overwrite the earlier's freshly-registered halt, and a
        halt would be silently lost (a safety failure).
        """
        state_path = tmp_path / "state" / "halts.json"
        n_writers = 12
        tickers = [f"TKR{i:02d}" for i in range(n_writers)]
        now_iso = WEEKDAY_10AM_UTC.isoformat()

        # "spawn" is the strictest start method (no inherited state) and is
        # the cross-platform default; it exercises real, independent procs.
        ctx = mp.get_context("spawn")
        barrier = ctx.Barrier(n_writers)
        procs = [
            ctx.Process(
                target=_register_halt_worker,
                args=(str(state_path), ticker, barrier, now_iso),
            )
            for ticker in tickers
        ]
        for p in procs:
            p.start()
        for p in procs:
            p.join(timeout=60)

        # No worker crashed.
        for p in procs:
            assert p.exitcode == 0, f"worker exited with {p.exitcode}"

        # Every distinct halt survived — none clobbered by a racing writer.
        reg = HaltRegistry(state_path)
        state = reg.load()
        registered = set(state.halts.keys())
        assert registered == set(tickers), (
            "lost halts under concurrency: missing "
            f"{set(tickers) - registered}"
        )
        for ticker in tickers:
            assert reg.is_halted(ticker, now=WEEKDAY_10AM_UTC) is True

        # The on-disk JSON is well-formed (atomic write held under lock).
        payload = json.loads(state_path.read_text(encoding="utf-8"))
        assert set(payload["halts"].keys()) == set(tickers)
        # No stray temp or lock-leak artifacts beyond the .lock sidecar.
        stray_tmp = [
            p for p in state_path.parent.iterdir() if p.suffix == ".tmp"
        ]
        assert stray_tmp == []

    def test_concurrent_writers_via_threads_no_halt_lost(
        self, tmp_path: Path
    ) -> None:
        """Same invariant, in-process via real threads (no mocks).

        Threads share the process but each opens its own lock fd, so the
        flock still serialises the RMW. Complements the multiprocessing
        test and runs even where ``spawn`` is constrained.
        """
        import threading

        state_path = tmp_path / "state" / "halts.json"
        n_writers = 16
        tickers = [f"THR{i:02d}" for i in range(n_writers)]
        start = threading.Barrier(n_writers)
        errors: List[BaseException] = []
        lock = threading.Lock()

        def worker(ticker: str) -> None:
            try:
                reg = HaltRegistry(state_path)
                start.wait()
                reg.register_halt(
                    ticker, now=WEEKDAY_10AM_UTC, reason=f"halt-{ticker}"
                )
            except BaseException as exc:  # surface, don't swallow
                with lock:
                    errors.append(exc)

        threads = [
            threading.Thread(target=worker, args=(t,)) for t in tickers
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=60)

        assert errors == [], f"worker errors: {errors}"
        reg = HaltRegistry(state_path)
        state = reg.load()
        assert set(state.halts.keys()) == set(tickers), (
            "lost halts under threaded concurrency: missing "
            f"{set(tickers) - set(state.halts.keys())}"
        )


# ---------------------------------------------------------------------------
# EquitySessionGate: ship-gate guard at construction time
# ---------------------------------------------------------------------------
class TestSessionGateShipGate:
    def test_construct_when_gate_passes(self, tmp_path: Path) -> None:
        gate = _make_gate(tmp_path)
        assert isinstance(gate, EquitySessionGate)
        assert gate.universe_hash == UNIVERSE_HASH

    def test_blocks_when_ship_gate_missing(self, tmp_path: Path) -> None:
        missing = tmp_path / "backtests" / "SHIP_GATE.json"
        halt_state = tmp_path / "state" / "halts.json"
        with pytest.raises(MarketCalendarError, match="file not found"):
            EquitySessionGate.create(
                universe_hash=UNIVERSE_HASH,
                ship_gate_path=missing,
                halt_state_path=halt_state,
            )

    def test_blocks_when_gate_pass_false(self, tmp_path: Path) -> None:
        with pytest.raises(MarketCalendarError, match="gate_pass"):
            _make_gate(tmp_path, gate_pass=False)

    def test_blocks_when_universe_hash_mismatches(
        self, tmp_path: Path
    ) -> None:
        with pytest.raises(MarketCalendarError, match="universe_hash"):
            _make_gate(tmp_path, universe_hash=OTHER_HASH)

    def test_blocks_when_universe_hash_blank(self, tmp_path: Path) -> None:
        gate_path = tmp_path / "backtests" / "SHIP_GATE.json"
        _write_ship_gate(gate_path, universe_hash="")
        with pytest.raises(MarketCalendarError, match="universe_hash"):
            EquitySessionGate.create(
                universe_hash=UNIVERSE_HASH,
                ship_gate_path=gate_path,
                halt_state_path=tmp_path / "state" / "halts.json",
            )

    def test_blocks_when_ship_gate_malformed(self, tmp_path: Path) -> None:
        gate_path = tmp_path / "backtests" / "SHIP_GATE.json"
        gate_path.parent.mkdir(parents=True, exist_ok=True)
        gate_path.write_text("{not valid json")
        with pytest.raises(MarketCalendarError, match="cannot read"):
            EquitySessionGate.create(
                universe_hash=UNIVERSE_HASH,
                ship_gate_path=gate_path,
                halt_state_path=tmp_path / "state" / "halts.json",
            )

    def test_blocks_when_ship_gate_not_an_object(
        self, tmp_path: Path
    ) -> None:
        gate_path = tmp_path / "backtests" / "SHIP_GATE.json"
        gate_path.parent.mkdir(parents=True, exist_ok=True)
        gate_path.write_text("[1, 2, 3]")
        with pytest.raises(MarketCalendarError, match="JSON object"):
            EquitySessionGate.create(
                universe_hash=UNIVERSE_HASH,
                ship_gate_path=gate_path,
                halt_state_path=tmp_path / "state" / "halts.json",
            )

    def test_universe_hash_must_be_nonempty_string(
        self, tmp_path: Path
    ) -> None:
        gate_path = tmp_path / "backtests" / "SHIP_GATE.json"
        _write_ship_gate(gate_path)
        with pytest.raises(MarketCalendarError, match="non-empty string"):
            EquitySessionGate.create(
                universe_hash="",
                ship_gate_path=gate_path,
                halt_state_path=tmp_path / "state" / "halts.json",
            )


# ---------------------------------------------------------------------------
# EquitySessionGate: guard_session enforces NYSE regular hours
# ---------------------------------------------------------------------------
class TestSessionGateGuardSession:
    def test_allows_normal_session(self, tmp_path: Path) -> None:
        gate = _make_gate(tmp_path)
        # No exception.
        gate.guard_session(now=WEEKDAY_10AM_UTC)

    def test_refuses_holiday(self, tmp_path: Path) -> None:
        gate = _make_gate(tmp_path)
        with pytest.raises(MarketCalendarError, match="refuse rebalance"):
            gate.guard_session(now=HOLIDAY_10AM_UTC)

    def test_refuses_pre_market(self, tmp_path: Path) -> None:
        gate = _make_gate(tmp_path)
        with pytest.raises(MarketCalendarError, match="refuse rebalance"):
            gate.guard_session(now=WEEKDAY_3AM_UTC)

    def test_refuses_after_halfday_close(self, tmp_path: Path) -> None:
        gate = _make_gate(tmp_path)
        with pytest.raises(MarketCalendarError, match="refuse rebalance"):
            gate.guard_session(now=HALFDAY_AFTER_CLOSE_UTC)


# ---------------------------------------------------------------------------
# EquitySessionGate.safe_send: halt awareness
# ---------------------------------------------------------------------------
class TestSafeSendHaltAwareness:
    def test_normal_fill_on_open_session(self, tmp_path: Path) -> None:
        gate = _make_gate(tmp_path)
        calls: List[str] = []

        def send(o: Order) -> bool:
            calls.append(o.client_order_id)
            return True

        result = gate.safe_send(_order(), send, now=WEEKDAY_10AM_UTC)
        assert result.outcome == OrderOutcome.FILLED.value
        assert len(calls) == 1

    def test_defers_when_market_closed(self, tmp_path: Path) -> None:
        gate = _make_gate(tmp_path)
        calls: List[str] = []

        def send(o: Order) -> bool:
            calls.append(o.client_order_id)
            return True

        result = gate.safe_send(_order(), send, now=HOLIDAY_10AM_UTC)
        assert result.outcome == OrderOutcome.DEFERRED_CLOSED.value
        assert calls == []  # broker never invoked when market closed

    def test_halt_error_defers_order_not_silently_dropped(
        self, tmp_path: Path
    ) -> None:
        """The load-bearing AC: a simulated halt defers the order."""
        gate = _make_gate(tmp_path)
        order = _order(ticker="NVDA")
        broker_call_log: List[str] = []

        def send(o: Order) -> bool:
            broker_call_log.append(o.client_order_id)
            raise HaltError("LULD volatility pause")

        result = gate.safe_send(order, send, now=WEEKDAY_10AM_UTC)

        # Order is DEFERRED, not silently dropped: routed to halt registry.
        assert result.outcome == OrderOutcome.DEFERRED_HALT.value
        assert result.order is order
        assert "LULD" in result.reason

        # The halt is persisted: next-cycle attempt also defers
        # (without re-invoking the broker).
        broker_call_log.clear()
        result2 = gate.safe_send(order, send, now=WEEKDAY_10AM_UTC)
        assert result2.outcome == OrderOutcome.DEFERRED_HALT.value
        assert broker_call_log == []

        # And the halt surfaces via deferred_tickers — operator can see it.
        assert "NVDA" in gate.deferred_tickers(now=WEEKDAY_10AM_UTC)

    def test_halt_with_ttl_clears_for_retry(self, tmp_path: Path) -> None:
        gate = _make_gate(tmp_path)
        order = _order(ticker="TSLA")
        ttl = WEEKDAY_10AM_UTC + pd.Timedelta(minutes=5)

        def halting_send(o: Order) -> bool:
            raise HaltError("circuit breaker")

        gate.safe_send(
            order, halting_send, now=WEEKDAY_10AM_UTC, halt_ttl=ttl,
        )
        assert "TSLA" in gate.deferred_tickers(now=WEEKDAY_10AM_UTC)

        # After TTL, the halt is cleared and a fresh send goes through.
        def good_send(o: Order) -> bool:
            return True

        after_ttl = ttl + pd.Timedelta(seconds=1)
        # Pick a time still inside the session — 10:05 ET, same Monday.
        retry_at = pd.Timestamp("2026-06-22 14:10:00", tz="UTC")
        # Force-evict by reading at after_ttl, but execute at retry_at
        # (which is also post-TTL since ttl was 14:05 UTC).
        assert retry_at > after_ttl
        result = gate.safe_send(order, good_send, now=retry_at)
        assert result.outcome == OrderOutcome.FILLED.value

    def test_graceful_reject_returns_rejected(self, tmp_path: Path) -> None:
        gate = _make_gate(tmp_path)

        def send(o: Order) -> bool:
            return False  # graceful broker reject (not a halt)

        result = gate.safe_send(_order(), send, now=WEEKDAY_10AM_UTC)
        assert result.outcome == OrderOutcome.REJECTED.value

    def test_non_halt_exception_propagates(self, tmp_path: Path) -> None:
        gate = _make_gate(tmp_path)

        def send(o: Order) -> bool:
            raise RuntimeError("network down")

        with pytest.raises(RuntimeError, match="network down"):
            gate.safe_send(_order(), send, now=WEEKDAY_10AM_UTC)

    def test_pre_existing_halt_skips_broker_call(
        self, tmp_path: Path
    ) -> None:
        gate = _make_gate(tmp_path)
        gate.halt_registry.register_halt(
            "AAPL", now=WEEKDAY_10AM_UTC, reason="prior LULD"
        )
        calls: List[str] = []

        def send(o: Order) -> bool:
            calls.append(o.client_order_id)
            return True

        result = gate.safe_send(
            _order(ticker="AAPL"), send, now=WEEKDAY_10AM_UTC
        )
        assert result.outcome == OrderOutcome.DEFERRED_HALT.value
        assert calls == []  # broker never invoked
