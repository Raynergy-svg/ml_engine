"""flatten_all truthfulness — driven by REAL recorded OANDA payloads.

NO MOCKS. No ``unittest.mock``, no ``patch``, no ``MagicMock``. A real
``ExecutionManager`` runs the real ``flatten_all`` against a replay transport
that serves response bodies **reconstructed from actual broker transactions**
in ``trained_data/oanda/transactions.jsonl`` — including the genuine
``ORDER_CANCEL / FIFO_VIOLATION`` records from 2026-05-13 and 2026-07-15 that
left the book non-flat.

Why this file exists in this form
---------------------------------

The original unit tests stubbed a bare ``HTTP 200`` for a close and asserted
success. That fixture *encoded the production bug*: OANDA returns 200 for a
close it accepted and then cancelled. The tests passed for months while the
behaviour was wrong, because they asserted against an imagined response shape
rather than a real one.

Writing more hand-authored fixtures would repeat that mistake one layer up. So
the payload shapes here are lifted from the recorded journal, and where the
journal is unavailable the test skips rather than falling back to invention.

State isolation: every test points ``flatten_state_path`` and
``flatten_log_path`` at ``tmp_path``. Nothing here can touch the live halt
state or the real strategic log.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import pytest

from src.scanner.execution import (
    ExecutionManager,
    FlattenResult,
    KillSwitchPartialFailure,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
JOURNAL = REPO_ROOT / "trained_data" / "oanda" / "transactions.jsonl"


# ---------------------------------------------------------------------------
# Real recorded payloads
# ---------------------------------------------------------------------------


def _journal_rows() -> List[Dict[str, Any]]:
    if not JOURNAL.exists():
        return []
    out = []
    for line in JOURNAL.read_text().splitlines():
        if line.strip():
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return out


_ROWS = _journal_rows()

requires_journal = pytest.mark.skipif(
    not _ROWS,
    reason="recorded OANDA journal not present; refusing to substitute invented payloads",
)


def _real_fifo_cancel() -> Dict[str, Any]:
    """An ACTUAL ORDER_CANCEL/FIFO_VIOLATION emitted by OANDA."""
    for r in _ROWS:
        if r.get("type") == "ORDER_CANCEL" and r.get("reason") == "FIFO_VIOLATION":
            return r
    pytest.skip("no recorded FIFO_VIOLATION in the journal")


def _real_close_fill() -> Dict[str, Any]:
    """An ACTUAL ORDER_FILL from a successful MARKET_ORDER_TRADE_CLOSE."""
    for r in _ROWS:
        if r.get("type") == "ORDER_FILL" and r.get("reason") == "MARKET_ORDER_TRADE_CLOSE":
            return r
    pytest.skip("no recorded trade-close fill in the journal")


class Response:
    """Minimal stand-in for requests.Response — a value object, not a mock.

    It holds data and answers ``.json()``/``.text``; it records no calls and
    asserts nothing.
    """

    def __init__(self, status_code: int, body: Optional[Dict[str, Any]] = None) -> None:
        self.status_code = status_code
        self._body = body if body is not None else {}
        self.text = json.dumps(self._body)

    def json(self) -> Dict[str, Any]:
        return self._body


class ReplayTransport:
    """Serves recorded broker responses and records the call order.

    Routing is by URL shape, exactly as the real endpoints are addressed.
    ``open_trades_sequence`` lets the pre-sweep read and the post-sweep
    verification return different books, which is the whole point of the
    verification step.
    """

    def __init__(
        self,
        *,
        open_trades_sequence: List[List[Dict[str, Any]]],
        close_responses: Dict[str, Any],
        pending_orders: Optional[List[str]] = None,
        open_trades_status: int = 200,
    ) -> None:
        self.open_trades_sequence = open_trades_sequence
        self.close_responses = close_responses
        self.pending_orders = pending_orders or []
        self.open_trades_status = open_trades_status
        self.close_order: List[str] = []
        self.open_trades_reads = 0

    async def __call__(self, method: str, url: str, **kwargs: Any) -> Response:
        if "pendingOrders" in url:
            return Response(200, {"orders": [{"id": o} for o in self.pending_orders]})
        if "openTrades" in url:
            self.open_trades_reads += 1
            if self.open_trades_status != 200:
                return Response(self.open_trades_status, {"errorMessage": "boom"})
            idx = min(self.open_trades_reads - 1, len(self.open_trades_sequence) - 1)
            return Response(200, {"trades": self.open_trades_sequence[idx]})
        if "/close" in url:
            trade_id = url.rsplit("/trades/", 1)[1].split("/")[0]
            self.close_order.append(trade_id)
            resp = self.close_responses.get(trade_id)
            if callable(resp):
                resp = resp(len([c for c in self.close_order if c == trade_id]))
            return resp if isinstance(resp, Response) else Response(200, resp or {})
        if url.endswith("/orders") or "/orders/" in url:
            return Response(200, {})
        return Response(200, {})


def _trade(tid: str, instrument: str = "EUR_USD") -> Dict[str, Any]:
    return {"id": tid, "instrument": instrument}


def _manager(tmp_path: Path, transport: ReplayTransport, monkeypatch) -> ExecutionManager:
    """A REAL ExecutionManager wired to the replay transport and tmp state.

    Credentials are set via ``monkeypatch.setenv`` -- scoped to the test and
    restored afterwards. An earlier version used ``os.environ.setdefault``,
    which leaked into the whole session and un-skipped
    ``test_flatten_all_integration.py`` (it skips only when creds are ABSENT).
    That real-network test then ran and wrote to the LIVE .claude/state.json.
    Global env mutation in a test is exactly the contamination these seams
    exist to prevent.
    """
    monkeypatch.setenv("OANDA_API_TOKEN", "replay-token")
    monkeypatch.setenv("OANDA_ACCOUNT_ID", "replay-acct")
    mgr = ExecutionManager()
    mgr.flatten_transport = transport
    mgr.flatten_state_path = tmp_path / "state.json"
    mgr.flatten_log_path = str(tmp_path / "strategic_log.md")

    async def _no_delay(_seconds: float) -> None:
        return None

    mgr.flatten_sleep = _no_delay
    return mgr


def _run(mgr: ExecutionManager) -> FlattenResult:
    return asyncio.run(mgr.flatten_all("truthful-test"))


# ---------------------------------------------------------------------------
# The two shapes that actually happened
# ---------------------------------------------------------------------------


@requires_journal
def test_real_recorded_fifo_cancel_is_not_a_close(tmp_path, monkeypatch):
    """HTTP 200 carrying a REAL ORDER_CANCEL/FIFO_VIOLATION must fail.

    This is the exact payload OANDA returned on 2026-05-13 and 2026-07-15.
    """
    cancel = _real_fifo_cancel()
    assert cancel["reason"] == "FIFO_VIOLATION"

    transport = ReplayTransport(
        open_trades_sequence=[[_trade("2689"), _trade("2691")], [_trade("2689"), _trade("2691")]],
        close_responses={
            "2689": Response(200, {"orderCancelTransaction": cancel}),
            "2691": Response(200, {"orderCancelTransaction": cancel}),
        },
    )
    mgr = _manager(tmp_path, transport, monkeypatch)

    with pytest.raises(KillSwitchPartialFailure) as ei:
        _run(mgr)

    assert "2689" in ei.value.remaining_trades


@requires_journal
def test_broker_acknowledges_but_book_never_goes_flat(tmp_path, monkeypatch):
    """Every close returns a REAL fill payload, yet the book still shows trades.

    The old code reported success here. The post-sweep re-read is the only
    thing that can catch it.
    """
    fill = _real_close_fill()

    transport = ReplayTransport(
        open_trades_sequence=[[_trade("T1")], [_trade("T1")]],  # never goes flat
        close_responses={"T1": Response(200, {"orderFillTransaction": fill})},
    )
    mgr = _manager(tmp_path, transport, monkeypatch)

    with pytest.raises(KillSwitchPartialFailure) as ei:
        _run(mgr)

    assert "T1" in ei.value.remaining_trades
    assert transport.open_trades_reads >= 2, "the book must be RE-READ after the sweep"


# ---------------------------------------------------------------------------
# FIFO ordering
# ---------------------------------------------------------------------------


@requires_journal
def test_same_instrument_closes_run_oldest_first(tmp_path, monkeypatch):
    fill = _real_close_fill()
    transport = ReplayTransport(
        open_trades_sequence=[
            [_trade("2691"), _trade("2687"), _trade("2689")],  # deliberately unsorted
            [],
        ],
        close_responses={
            t: Response(200, {"orderFillTransaction": fill}) for t in ("2687", "2689", "2691")
        },
    )
    mgr = _manager(tmp_path, transport, monkeypatch)

    result = _run(mgr)

    assert transport.close_order == ["2687", "2689", "2691"]
    assert result.verified_flat is True
    assert result.positions_closed == 3


@requires_journal
def test_oldest_failure_stops_the_instrument_rather_than_spraying_rejections(tmp_path, monkeypatch):
    cancel = _real_fifo_cancel()
    transport = ReplayTransport(
        open_trades_sequence=[[_trade("2687"), _trade("2689")], [_trade("2687"), _trade("2689")]],
        close_responses={
            "2687": Response(200, {"orderCancelTransaction": cancel}),
            "2689": Response(200, {"orderCancelTransaction": cancel}),
        },
    )
    mgr = _manager(tmp_path, transport, monkeypatch)

    with pytest.raises(KillSwitchPartialFailure) as ei:
        _run(mgr)

    assert transport.close_order == ["2687"], "must stop at the oldest failure"
    assert set(ei.value.remaining_trades) >= {"2687", "2689"}


@requires_journal
def test_different_instruments_are_not_serialised_behind_each_other(tmp_path, monkeypatch):
    fill = _real_close_fill()
    transport = ReplayTransport(
        open_trades_sequence=[
            [_trade("A1", "EUR_USD"), _trade("B1", "USD_JPY")],
            [],
        ],
        close_responses={
            "A1": Response(200, {"orderFillTransaction": fill}),
            "B1": Response(200, {"orderFillTransaction": fill}),
        },
    )
    mgr = _manager(tmp_path, transport, monkeypatch)

    result = _run(mgr)

    assert set(transport.close_order) == {"A1", "B1"}
    assert result.verified_flat is True


# ---------------------------------------------------------------------------
# Verification and fail-closed behaviour
# ---------------------------------------------------------------------------


@requires_journal
def test_already_flat_account_succeeds_and_is_verified(tmp_path, monkeypatch):
    transport = ReplayTransport(open_trades_sequence=[[], []], close_responses={})
    mgr = _manager(tmp_path, transport, monkeypatch)

    result = _run(mgr)

    assert result.positions_closed == 0
    assert result.positions_failed == 0
    assert result.verified_flat is True
    assert result.residual_trade_ids == []
    assert transport.close_order == []


@requires_journal
def test_partial_closure_reports_the_residual(tmp_path, monkeypatch):
    fill = _real_close_fill()
    cancel = _real_fifo_cancel()
    transport = ReplayTransport(
        open_trades_sequence=[
            [_trade("A1", "EUR_USD"), _trade("B1", "USD_JPY")],
            [_trade("B1", "USD_JPY")],  # B1 survived
        ],
        close_responses={
            "A1": Response(200, {"orderFillTransaction": fill}),
            "B1": Response(200, {"orderCancelTransaction": cancel}),
        },
    )
    mgr = _manager(tmp_path, transport, monkeypatch)

    with pytest.raises(KillSwitchPartialFailure) as ei:
        _run(mgr)

    assert "B1" in ei.value.remaining_trades


@requires_journal
def test_unreadable_pre_sweep_book_refuses_rather_than_closing_nothing(tmp_path, monkeypatch):
    """If we cannot see what is open we must not claim to have flattened it."""
    transport = ReplayTransport(
        open_trades_sequence=[[]], close_responses={}, open_trades_status=500,
    )
    mgr = _manager(tmp_path, transport, monkeypatch)

    with pytest.raises(KillSwitchPartialFailure) as ei:
        _run(mgr)

    assert "UNKNOWN" in ei.value.remaining_trades
    assert transport.close_order == []


@requires_journal
def test_post_sweep_verification_failure_is_not_success(tmp_path, monkeypatch):
    fill = _real_close_fill()
    reads = {"n": 0}

    class _FlakyVerify(ReplayTransport):
        async def __call__(self, method, url, **kwargs):
            if "openTrades" in url:
                reads["n"] += 1
                if reads["n"] == 1:
                    self.open_trades_reads += 1
                    return Response(200, {"trades": [_trade("T1")]})
                self.open_trades_reads += 1
                return Response(503, {"errorMessage": "unavailable"})
            return await super().__call__(method, url, **kwargs)

    transport = _FlakyVerify(
        open_trades_sequence=[[_trade("T1")]],
        close_responses={"T1": Response(200, {"orderFillTransaction": fill})},
    )
    mgr = _manager(tmp_path, transport, monkeypatch)

    with pytest.raises(KillSwitchPartialFailure) as ei:
        _run(mgr)

    assert "UNVERIFIED" in ei.value.remaining_trades


@requires_journal
def test_200_with_neither_fill_nor_cancel_is_not_a_close(tmp_path, monkeypatch):
    """The original fixture shape — a bare 200 — must now fail."""
    transport = ReplayTransport(
        open_trades_sequence=[[_trade("T1")], [_trade("T1")]],
        close_responses={"T1": Response(200, {})},
    )
    mgr = _manager(tmp_path, transport, monkeypatch)

    with pytest.raises(KillSwitchPartialFailure) as ei:
        _run(mgr)

    assert "T1" in ei.value.remaining_trades


# ---------------------------------------------------------------------------
# Isolation guarantees
# ---------------------------------------------------------------------------


@requires_journal
def test_tests_do_not_touch_the_live_halt_state(tmp_path, monkeypatch):
    """The halt write must land in tmp_path, never in the repo's state.json."""
    live = REPO_ROOT / ".claude" / "state.json"
    before = live.read_text() if live.exists() else None

    transport = ReplayTransport(open_trades_sequence=[[], []], close_responses={})
    mgr = _manager(tmp_path, transport, monkeypatch)
    _run(mgr)

    assert (tmp_path / "state.json").exists(), "halt should be written to the tmp state"
    assert json.loads((tmp_path / "state.json").read_text())["halted"] is True
    if before is not None:
        assert live.read_text() == before, "LIVE state.json must be untouched"


@requires_journal
def test_no_unittest_mock_anywhere_in_this_module():
    """Standing guard: this file must never regress to patching."""
    import ast

    tree = ast.parse(Path(__file__).read_text())
    forbidden = {"MagicMock", "AsyncMock", "patch", "mocker", "Mock"}

    for node in ast.walk(tree):
        names = []
        if isinstance(node, ast.Import):
            names = [a.name for a in node.names]
        elif isinstance(node, ast.ImportFrom) and node.module:
            names = [node.module]
        for name in names:
            assert "mock" not in name.lower(), f"forbidden import {name!r}"

    # Identifiers only -- prose in docstrings may legitimately NAME the things
    # this file refuses to use.
    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            assert node.id not in forbidden, f"forbidden identifier {node.id!r}"
        elif isinstance(node, ast.Attribute):
            assert node.attr not in forbidden, f"forbidden attribute {node.attr!r}"
        elif isinstance(node, ast.arg):
            assert node.arg not in forbidden, f"forbidden fixture arg {node.arg!r}"

    # monkeypatch.setenv is environment ISOLATION and is allowed; setattr/delattr
    # are patching and are not.
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name):
            if node.value.id == "monkeypatch":
                assert node.attr == "setenv", (
                    f"monkeypatch.{node.attr} is patching, not isolation"
                )
