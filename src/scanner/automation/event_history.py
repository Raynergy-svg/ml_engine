"""Paged JSONL reader for trading event history.

Provides memory-efficient access to the trading event log without loading
the entire file. Supports tail-reading, cursor-based pagination, and
multi-field filtering.
"""

import json
import os
from dataclasses import dataclass
from typing import List, Optional

try:
    import structlog
    logger = structlog.get_logger(__name__)
except ImportError:
    import logging
    logger = logging.getLogger(__name__)

from src.scanner.automation.trading_events import TradingEvent, TradingEventType


@dataclass
class HistoryPage:
    """A page of event history results."""
    events: List[TradingEvent]
    first_id: Optional[str]
    has_more: bool


def _parse_event(line: str) -> Optional[TradingEvent]:
    """Parse a single JSONL line into a TradingEvent, returning None on failure."""
    line = line.strip()
    if not line:
        return None
    try:
        data = json.loads(line)
        return TradingEvent(
            event_id=data["event_id"],
            event_type=TradingEventType(data["event_type"]),
            timestamp=data["timestamp"],
            source=data["source"],
            session_id=data["session_id"],
            correlation_id=data["correlation_id"],
            payload=data.get("payload", {}),
            payload_version=data.get("payload_version", 1),
        )
    except (json.JSONDecodeError, KeyError, ValueError) as exc:
        logger.warning("event_history.parse_failed", error=str(exc), line=line[:120])
        return None


def _read_tail_lines(path: str, n: int) -> List[str]:
    """Read the last n lines from a file by seeking backwards in chunks.

    Returns lines in file order (oldest first).
    """
    chunk_size = 8192
    lines: List[str] = []
    try:
        file_size = os.path.getsize(path)
    except OSError:
        return []
    if file_size == 0:
        return []

    with open(path, "rb") as f:
        remaining = file_size
        tail_bytes = b""
        while remaining > 0 and len(lines) <= n:
            read_size = min(chunk_size, remaining)
            remaining -= read_size
            f.seek(remaining)
            chunk = f.read(read_size)
            tail_bytes = chunk + tail_bytes
            lines = tail_bytes.split(b"\n")
        # Filter empty lines from split and decode
        text_lines = [l.decode("utf-8", errors="replace") for l in lines if l.strip()]
        # Return only the last n
        return text_lines[-n:]


class EventHistory:
    """Paged reader for the trading events JSONL log."""

    def __init__(self, jsonl_path: str = "trained_data/trading_events.jsonl"):
        self.jsonl_path = jsonl_path

    def get_latest(self, n: int = 20) -> HistoryPage:
        """Return the N most recent events without loading the entire file.

        Reads from the tail of the JSONL file using backward chunk seeking.
        """
        if not os.path.exists(self.jsonl_path):
            return HistoryPage(events=[], first_id=None, has_more=False)

        # Read more lines than needed to account for corrupted/empty lines
        raw_lines = _read_tail_lines(self.jsonl_path, n + 10)
        events: List[TradingEvent] = []
        for line in reversed(raw_lines):
            ev = _parse_event(line)
            if ev is not None:
                events.append(ev)
            if len(events) >= n:
                break

        # events are newest-first from the reversed iteration; reverse to oldest-first
        events.reverse()

        # Determine has_more: if we got the full n, there are likely more
        has_more = len(events) >= n
        first_id = events[0].event_id if events else None
        return HistoryPage(events=events, first_id=first_id, has_more=has_more)

    def get_before(self, cursor_id: str, n: int = 20) -> HistoryPage:
        """Return N events older than cursor_id.

        Scans the file to find the cursor position, then returns the N events
        immediately before it.
        """
        if not os.path.exists(self.jsonl_path):
            return HistoryPage(events=[], first_id=None, has_more=False)

        all_before: List[TradingEvent] = []
        found_cursor = False

        try:
            with open(self.jsonl_path, "r", encoding="utf-8") as f:
                for line in f:
                    ev = _parse_event(line)
                    if ev is None:
                        continue
                    if ev.event_id == cursor_id:
                        found_cursor = True
                        break
                    all_before.append(ev)
        except OSError:
            return HistoryPage(events=[], first_id=None, has_more=False)

        if not found_cursor:
            return HistoryPage(events=[], first_id=None, has_more=False)

        # Take the last n events before the cursor
        page_events = all_before[-n:]
        has_more = len(all_before) > n
        first_id = page_events[0].event_id if page_events else None
        return HistoryPage(events=page_events, first_id=first_id, has_more=has_more)

    def filter_by(
        self,
        event_type: Optional[str] = None,
        pair: Optional[str] = None,
        trade_id: Optional[str] = None,
        session_id: Optional[str] = None,
    ) -> HistoryPage:
        """Return events matching all non-None filters (AND logic).

        Scans the full file applying filters. Returns all matching events.
        """
        if not os.path.exists(self.jsonl_path):
            return HistoryPage(events=[], first_id=None, has_more=False)

        matches: List[TradingEvent] = []
        try:
            with open(self.jsonl_path, "r", encoding="utf-8") as f:
                for line in f:
                    ev = _parse_event(line)
                    if ev is None:
                        continue
                    if event_type is not None and ev.event_type.value != event_type:
                        continue
                    if session_id is not None and ev.session_id != session_id:
                        continue
                    if pair is not None:
                        ev_pair = ev.payload.get("pair", "")
                        if ev_pair != pair:
                            continue
                    if trade_id is not None:
                        ev_trade_id = ev.payload.get("trade_id", "")
                        if ev_trade_id != trade_id:
                            continue
                    matches.append(ev)
        except OSError:
            return HistoryPage(events=[], first_id=None, has_more=False)

        first_id = matches[0].event_id if matches else None
        return HistoryPage(events=matches, first_id=first_id, has_more=False)
