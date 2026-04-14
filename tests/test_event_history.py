"""Tests for EventHistory paged JSONL reader."""

import json
import pytest

from src.scanner.automation.event_history import EventHistory, HistoryPage
from src.scanner.automation.trading_events import TradingEventType


def _make_event_line(event_id: str, event_type: str = "trade_opened",
                     pair: str = "EUR_USD", session_id: str = "sess-1") -> str:
    """Build a valid JSONL line for a trading event."""
    return json.dumps({
        "event_id": event_id,
        "event_type": event_type,
        "timestamp": "2026-04-03T12:00:00Z",
        "source": "test",
        "session_id": session_id,
        "correlation_id": "corr-1",
        "payload": {"pair": pair, "trade_id": f"trade-{event_id}"},
        "payload_version": 1,
    })


def _write_jsonl(path, lines):
    """Write list of JSONL strings to a file."""
    path.write_text("\n".join(lines) + "\n")


class TestGetLatest:
    def test_returns_n_most_recent(self, tmp_path):
        f = tmp_path / "events.jsonl"
        lines = [_make_event_line(f"ev-{i}") for i in range(10)]
        _write_jsonl(f, lines)

        history = EventHistory(str(f))
        page = history.get_latest(n=3)

        assert len(page.events) == 3
        assert page.events[0].event_id == "ev-7"
        assert page.events[1].event_id == "ev-8"
        assert page.events[2].event_id == "ev-9"

    def test_empty_file_returns_empty_page(self, tmp_path):
        f = tmp_path / "events.jsonl"
        f.write_text("")

        history = EventHistory(str(f))
        page = history.get_latest(n=5)

        assert page.events == []
        assert page.first_id is None
        assert page.has_more is False

    def test_missing_file_returns_empty_page(self, tmp_path):
        history = EventHistory(str(tmp_path / "nonexistent.jsonl"))
        page = history.get_latest(n=5)

        assert page.events == []
        assert page.first_id is None
        assert page.has_more is False


class TestGetBefore:
    def test_returns_older_events(self, tmp_path):
        f = tmp_path / "events.jsonl"
        lines = [_make_event_line(f"ev-{i}") for i in range(10)]
        _write_jsonl(f, lines)

        history = EventHistory(str(f))
        page = history.get_before(cursor_id="ev-5", n=3)

        assert len(page.events) == 3
        assert page.events[0].event_id == "ev-2"
        assert page.events[1].event_id == "ev-3"
        assert page.events[2].event_id == "ev-4"

    def test_has_more_true_when_more_exist(self, tmp_path):
        f = tmp_path / "events.jsonl"
        lines = [_make_event_line(f"ev-{i}") for i in range(10)]
        _write_jsonl(f, lines)

        history = EventHistory(str(f))
        page = history.get_before(cursor_id="ev-5", n=3)

        assert page.has_more is True

    def test_has_more_false_at_beginning(self, tmp_path):
        f = tmp_path / "events.jsonl"
        lines = [_make_event_line(f"ev-{i}") for i in range(5)]
        _write_jsonl(f, lines)

        history = EventHistory(str(f))
        # cursor at ev-2, only ev-0 and ev-1 before it — requesting 3 but only 2 exist
        page = history.get_before(cursor_id="ev-2", n=3)

        assert page.has_more is False
        assert len(page.events) == 2
        assert page.events[0].event_id == "ev-0"
        assert page.events[1].event_id == "ev-1"


class TestFilterBy:
    def test_filter_by_event_type(self, tmp_path):
        f = tmp_path / "events.jsonl"
        lines = [
            _make_event_line("ev-0", event_type="trade_opened"),
            _make_event_line("ev-1", event_type="trade_closed"),
            _make_event_line("ev-2", event_type="trade_opened"),
            _make_event_line("ev-3", event_type="trade_closed"),
        ]
        _write_jsonl(f, lines)

        history = EventHistory(str(f))
        page = history.filter_by(event_type="trade_closed")

        assert len(page.events) == 2
        assert all(e.event_type == TradingEventType.TRADE_CLOSED for e in page.events)
        assert page.events[0].event_id == "ev-1"
        assert page.events[1].event_id == "ev-3"

    def test_filter_by_pair(self, tmp_path):
        f = tmp_path / "events.jsonl"
        lines = [
            _make_event_line("ev-0", pair="EUR_USD"),
            _make_event_line("ev-1", pair="GBP_USD"),
            _make_event_line("ev-2", pair="EUR_USD"),
            _make_event_line("ev-3", pair="USD_JPY"),
        ]
        _write_jsonl(f, lines)

        history = EventHistory(str(f))
        page = history.filter_by(pair="EUR_USD")

        assert len(page.events) == 2
        assert page.events[0].event_id == "ev-0"
        assert page.events[1].event_id == "ev-2"

    def test_filter_by_multiple_uses_and_logic(self, tmp_path):
        f = tmp_path / "events.jsonl"
        lines = [
            _make_event_line("ev-0", event_type="trade_opened", pair="EUR_USD"),
            _make_event_line("ev-1", event_type="trade_closed", pair="EUR_USD"),
            _make_event_line("ev-2", event_type="trade_opened", pair="GBP_USD"),
            _make_event_line("ev-3", event_type="trade_closed", pair="GBP_USD"),
        ]
        _write_jsonl(f, lines)

        history = EventHistory(str(f))
        page = history.filter_by(event_type="trade_closed", pair="EUR_USD")

        assert len(page.events) == 1
        assert page.events[0].event_id == "ev-1"


class TestCorruptedLines:
    def test_corrupted_jsonl_line_skipped(self, tmp_path):
        f = tmp_path / "events.jsonl"
        lines = [
            _make_event_line("ev-0"),
            "THIS IS NOT JSON {{{",
            _make_event_line("ev-2"),
            '{"event_id": "ev-3"}',  # valid JSON but missing required keys
            _make_event_line("ev-4"),
        ]
        _write_jsonl(f, lines)

        history = EventHistory(str(f))
        page = history.get_latest(n=10)

        # Only 3 valid events should be parsed
        assert len(page.events) == 3
        ids = [e.event_id for e in page.events]
        assert ids == ["ev-0", "ev-2", "ev-4"]
