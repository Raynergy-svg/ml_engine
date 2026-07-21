"""Unit tests for ExecutionManager.flatten_all() — US-503.

Covers:
- All-success path: all orders cancelled, all trades closed, FlattenResult returned.
- Partial-failure path: one position fails all retries, KillSwitchPartialFailure raised.
- Rate-limit retry: 5xx on first call triggers retry, succeeds on second.
- 4xx no-retry: 404/400 responses are not retried.
"""

from __future__ import annotations

import asyncio
import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.scanner.execution import ExecutionManager, FlattenResult, KillSwitchPartialFailure


# ── Helpers ───────────────────────────────────────────────────────────

def _make_response(status_code: int, body: dict | None = None) -> MagicMock:
    """Build a mock requests.Response."""
    r = MagicMock()
    r.status_code = status_code
    r.json.return_value = body or {}
    r.text = json.dumps(body or {})
    return r


def _pending_orders_resp(order_ids: list[str]) -> MagicMock:
    return _make_response(200, {"orders": [{"id": oid} for oid in order_ids]})


def _open_trades_resp(trade_ids: list[str]) -> MagicMock:
    return _make_response(200, {"trades": [
        {"id": tid, "instrument": "EUR_USD"} for tid in trade_ids
    ]})


# ── Correct-contract helpers (2026-07-20) ─────────────────────────────
#
# OANDA returns HTTP 200 for a close request it ACCEPTED AND THEN CANCELLED.
# A close is confirmed by ``orderFillTransaction`` and refuted by
# ``orderCancelTransaction``. The original fixtures used a bare 200 with an
# empty body, which encoded the very bug that left the book non-flat on
# 2026-05-13 and 2026-07-15.

def _close_ok(trade_id: str = "T") -> MagicMock:
    """A genuinely executed close."""
    return _make_response(200, {
        "orderFillTransaction": {"id": f"F-{trade_id}", "type": "ORDER_FILL",
                                 "tradesClosed": [{"tradeID": trade_id}]},
    })


def _close_fifo_refused(trade_id: str = "T") -> MagicMock:
    """HTTP 200, but the broker cancelled it. THE Jul-15 shape."""
    return _make_response(200, {
        "orderCancelTransaction": {"id": f"C-{trade_id}", "type": "ORDER_CANCEL",
                                   "reason": "FIFO_VIOLATION"},
    })


def _sequenced_router(call_map: dict, open_trades_sequence: list):
    """Router whose openTrades GET returns a SEQUENCE.

    The first call is the pre-sweep read; later calls are the post-sweep
    verification, which must be able to report a different (ideally empty) book.
    """
    state = {"i": 0}

    async def _fake_to_thread(func, url, **kwargs):
        method = {"get": "GET", "put": "PUT", "delete": "DELETE"}[func.__name__]
        path = url.replace("https://api-fxpractice.oanda.com", "")
        key = f"{method}:{path}"
        if key.endswith("/openTrades"):
            idx = min(state["i"], len(open_trades_sequence) - 1)
            state["i"] += 1
            return open_trades_sequence[idx]
        return call_map[key]

    return _fake_to_thread


# ── Fixtures ──────────────────────────────────────────────────────────

@pytest.fixture
def mgr(tmp_path, monkeypatch):
    """ExecutionManager with OANDA env vars patched to dummies."""
    monkeypatch.setenv("OANDA_API_TOKEN", "test-token")
    monkeypatch.setenv("OANDA_ACCOUNT_ID", "test-acct")
    return ExecutionManager()


# ── Tests ─────────────────────────────────────────────────────────────

class TestFlattenAllSuccess:
    """flatten_all() with 2 orders and 2 trades that all succeed."""

    def test_returns_flatten_result(self, mgr, tmp_path, monkeypatch):
        monkeypatch.setattr("os.getenv", lambda k, d="": {
            "OANDA_API_TOKEN": "tok", "OANDA_ACCOUNT_ID": "acct",
        }.get(k, d))

        call_map = {
            "GET:/v3/accounts/acct/pendingOrders": _pending_orders_resp(["O1", "O2"]),
            "GET:/v3/accounts/acct/openTrades": _open_trades_resp(["T1", "T2"]),
            "DELETE:/v3/accounts/acct/orders/O1": _make_response(200),
            "DELETE:/v3/accounts/acct/orders/O2": _make_response(200),
            "PUT:/v3/accounts/acct/trades/T1/close": _close_ok("T1"),
            "PUT:/v3/accounts/acct/trades/T2/close": _close_ok("T2"),
        }

        _fake_to_thread = _sequenced_router(
            call_map,
            # pre-sweep: two open; post-sweep verification: flat
            [_open_trades_resp(["T1", "T2"]), _open_trades_resp([])],
        )

        se_mock = MagicMock()
        bus_mock = MagicMock()

        with (
            patch("asyncio.to_thread", side_effect=_fake_to_thread),
            patch("src.scanner.automation.state_engine.StateEngine.set_halted", se_mock),
            patch("src.scanner.automation.event_bus.get_event_bus", return_value=bus_mock),
            patch("builtins.open", MagicMock()),
        ):
            result = asyncio.run(mgr.flatten_all("unit-test"))

        assert isinstance(result, FlattenResult)
        assert result.orders_cancelled == 2
        assert result.orders_failed == 0
        assert result.positions_closed == 2
        assert result.positions_failed == 0
        assert result.duration_ms > 0


class TestFlattenAllPartialFailure:
    """One position close exhausts all retries → KillSwitchPartialFailure."""

    def test_raises_partial_failure(self, mgr, monkeypatch):
        monkeypatch.setattr("os.getenv", lambda k, d="": {
            "OANDA_API_TOKEN": "tok", "OANDA_ACCOUNT_ID": "acct",
        }.get(k, d))

        # T2 always returns 500 to exhaust retries
        call_counts: dict[str, int] = {}

        async def _fake_to_thread(func, url, **kwargs):
            path = url.replace("https://api-fxpractice.oanda.com", "")
            if "pendingOrders" in path:
                return _pending_orders_resp([])
            if "openTrades" in path:
                # pre-sweep sees both; post-sweep verification still sees T2,
                # which is the honest broker state after T2's close failed.
                call_counts["openTrades"] = call_counts.get("openTrades", 0) + 1
                return (_open_trades_resp(["T1", "T2"])
                        if call_counts["openTrades"] == 1
                        else _open_trades_resp(["T2"]))
            if "T1/close" in path:
                return _close_ok("T1")
            if "T2/close" in path:
                call_counts["T2"] = call_counts.get("T2", 0) + 1
                return _make_response(500)
            return _make_response(200)

        with (
            patch("asyncio.to_thread", side_effect=_fake_to_thread),
            patch("src.scanner.automation.state_engine.StateEngine.set_halted"),
            patch("src.scanner.automation.event_bus.get_event_bus", return_value=MagicMock()),
            patch("builtins.open", MagicMock()),
            patch("asyncio.sleep", new_callable=AsyncMock),
        ):
            with pytest.raises(KillSwitchPartialFailure) as exc_info:
                asyncio.run(mgr.flatten_all("partial-failure-test"))

        err = exc_info.value
        assert "T2" in err.remaining_trades
        assert len(err.remaining_trades) == 1


class TestFlattenAllRateLimitRetry:
    """5xx on first attempt retries and succeeds on second."""

    def test_retries_on_5xx(self, mgr, monkeypatch):
        monkeypatch.setattr("os.getenv", lambda k, d="": {
            "OANDA_API_TOKEN": "tok", "OANDA_ACCOUNT_ID": "acct",
        }.get(k, d))

        attempts: dict[str, int] = {"T1": 0}

        async def _fake_to_thread(func, url, **kwargs):
            if "pendingOrders" in url:
                return _pending_orders_resp([])
            if "openTrades" in url:
                attempts["openTrades"] = attempts.get("openTrades", 0) + 1
                return (_open_trades_resp(["T1"]) if attempts["openTrades"] == 1
                        else _open_trades_resp([]))
            if "T1/close" in url:
                attempts["T1"] += 1
                if attempts["T1"] == 1:
                    return _make_response(503)  # first attempt: 5xx → retry
                return _close_ok("T1")          # second attempt: real fill
            return _make_response(200)

        with (
            patch("asyncio.to_thread", side_effect=_fake_to_thread),
            patch("src.scanner.automation.state_engine.StateEngine.set_halted"),
            patch("src.scanner.automation.event_bus.get_event_bus", return_value=MagicMock()),
            patch("builtins.open", MagicMock()),
            patch("asyncio.sleep", new_callable=AsyncMock),
        ):
            result = asyncio.run(mgr.flatten_all("retry-test"))

        assert result.positions_closed == 1
        assert attempts["T1"] == 2  # 5xx triggered one retry


class TestFlattenAll4xxNoRetry:
    """4xx client errors are returned immediately without retry."""

    def test_4xx_not_retried(self, mgr, monkeypatch):
        monkeypatch.setattr("os.getenv", lambda k, d="": {
            "OANDA_API_TOKEN": "tok", "OANDA_ACCOUNT_ID": "acct",
        }.get(k, d))

        attempts: dict[str, int] = {"T1": 0}

        async def _fake_to_thread(func, url, **kwargs):
            if "pendingOrders" in url:
                return _pending_orders_resp([])
            if "openTrades" in url:
                return _open_trades_resp(["T1"])
            if "T1/close" in url:
                attempts["T1"] += 1
                return _make_response(422)  # 4xx — should not retry
            return _make_response(200)

        with (
            patch("asyncio.to_thread", side_effect=_fake_to_thread),
            patch("src.scanner.automation.state_engine.StateEngine.set_halted"),
            patch("src.scanner.automation.event_bus.get_event_bus", return_value=MagicMock()),
            patch("builtins.open", MagicMock()),
            patch("asyncio.sleep", new_callable=AsyncMock),
        ):
            with pytest.raises(KillSwitchPartialFailure):
                asyncio.run(mgr.flatten_all("4xx-test"))

        # 4xx should not have triggered any retries — called exactly once
        assert attempts["T1"] == 1


class TestFlattenAllHaltFirst:
    """StateEngine.set_halted(True) is called before any OANDA API call."""

    def test_halted_before_oanda(self, mgr, monkeypatch):
        monkeypatch.setattr("os.getenv", lambda k, d="": {
            "OANDA_API_TOKEN": "tok", "OANDA_ACCOUNT_ID": "acct",
        }.get(k, d))

        call_log: list[str] = []

        async def _fake_to_thread(func, url, **kwargs):
            call_log.append(f"OANDA:{func.__name__}:{url}")
            if "pendingOrders" in url:
                return _pending_orders_resp([])
            if "openTrades" in url:
                return _open_trades_resp([])
            return _make_response(200)

        def _fake_set_halted(self_se, value):
            call_log.append(f"set_halted:{value}")

        with (
            patch("asyncio.to_thread", side_effect=_fake_to_thread),
            patch(
                "src.scanner.automation.state_engine.StateEngine.set_halted",
                _fake_set_halted,
            ),
            patch("src.scanner.automation.event_bus.get_event_bus", return_value=MagicMock()),
            patch("builtins.open", MagicMock()),
        ):
            asyncio.run(mgr.flatten_all("halt-first-test"))

        halt_idx = next(i for i, e in enumerate(call_log) if e.startswith("set_halted:True"))
        first_oanda_idx = next(i for i, e in enumerate(call_log) if e.startswith("OANDA:"))
        assert halt_idx < first_oanda_idx, "set_halted(True) must precede first OANDA call"


class TestFlattenAllEventBus:
    """control.kill event is emitted with FlattenResult payload."""

    def test_emits_control_kill(self, mgr, monkeypatch):
        monkeypatch.setattr("os.getenv", lambda k, d="": {
            "OANDA_API_TOKEN": "tok", "OANDA_ACCOUNT_ID": "acct",
        }.get(k, d))

        async def _fake_to_thread(func, url, **kwargs):
            if "pendingOrders" in url:
                return _pending_orders_resp([])
            if "openTrades" in url:
                return _open_trades_resp([])
            return _make_response(200)

        bus_mock = MagicMock()

        with (
            patch("asyncio.to_thread", side_effect=_fake_to_thread),
            patch("src.scanner.automation.state_engine.StateEngine.set_halted"),
            patch("src.scanner.automation.event_bus.get_event_bus", return_value=bus_mock),
            patch("builtins.open", MagicMock()),
        ):
            asyncio.run(mgr.flatten_all("event-test"))

        bus_mock.publish.assert_called_once()
        event_type, payload = bus_mock.publish.call_args[0]
        assert event_type == "control.kill"
        assert payload["reason"] == "event-test"
        assert "result" in payload


# NOTE: the truthful-flatten acceptance scenarios live in
# tests/test_flatten_all_truthful.py, which drives the real ExecutionManager
# against REAL recorded OANDA payloads through injectable seams and uses no
# mocks. The fixtures below were corrected to the real broker contract (a close
# is confirmed by orderFillTransaction, not by HTTP 200) but this file remains
# mock-based for historical reasons; prefer the truthful suite for new coverage.
