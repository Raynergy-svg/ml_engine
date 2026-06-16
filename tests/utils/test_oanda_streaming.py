"""Smoke tests for OANDA streaming tick client."""
import json
from datetime import datetime, timezone

import pytest

from src.utils.oanda_streaming import TickQuote, OandaStreamClient, StreamBuffer


class FakeResp:
    """Mock requests.Response for SSE stream testing."""
    def __init__(self, lines, status_code=200):
        self._lines = lines
        self.status_code = status_code
        self.encoding = "utf-8"

    def raise_for_status(self):
        if self.status_code >= 400:
            raise Exception(f"HTTP {self.status_code}")

    def iter_lines(self, decode_unicode=False):
        for line in self._lines:
            yield line


class TestTickQuote:
    def test_tick_basic(self):
        t = TickQuote(
            instrument="EUR_USD",
            time=datetime.now(timezone.utc),
            bid=1.10,
            ask=1.1001,
            mid=1.10005,
            bid_liq=1e6,
            ask_liq=1e6,
            status="tradeable",
            source="oanda_stream",
        )
        assert t.mid == pytest.approx(1.10005)


class TestStreamBuffer:
    def test_append_and_flush(self):
        buf = StreamBuffer(max_size=3)
        tick = TickQuote(
            instrument="EUR_USD",
            time=datetime.now(timezone.utc),
            bid=1.0,
            ask=1.0001,
            mid=1.00005,
            bid_liq=1.0,
            ask_liq=1.0,
            status="tradeable",
            source="test",
        )
        assert buf.append(tick) is False
        assert buf.append(tick) is False
        assert buf.append(tick) is True
        flushed = buf.flush()
        assert len(flushed) == 3
        assert buf.flush() == []


class TestOandaStreamClientParse:
    def test_parse_heartbeat(self):
        client = OandaStreamClient("acc", "tok")
        assert client._parse_tick_line(json.dumps({"type": "HEARTBEAT", "time": "2026-06-15T12:00:00Z"})) is None

    def test_parse_valid_tick(self):
        client = OandaStreamClient("acc", "tok")
        payload = {
            "type": "PRICE",
            "instrument": "EUR_USD",
            "time": "2026-06-15T12:00:00.123456789Z",
            "bids": [{"price": "1.10000", "liquidity": 1000000}],
            "asks": [{"price": "1.10005", "liquidity": 1000000}],
            "closeoutBid": "1.09998",
            "closeoutAsk": "1.10007",
            "tradeable": True,
        }
        tick = client._parse_tick_line(json.dumps(payload))
        assert tick is not None
        assert tick.instrument == "EUR_USD"
        assert tick.bid == pytest.approx(1.09998)
        assert tick.ask == pytest.approx(1.10007)
        assert tick.status == "tradeable"

    def test_parse_missing_instrument(self):
        client = OandaStreamClient("acc", "tok")
        assert client._parse_tick_line(json.dumps({"type": "PRICE", "bids": [], "asks": []})) is None
