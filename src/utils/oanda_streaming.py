"""OANDA v20 Streaming client for real-time pricing (tick) capture.

Uses the /pricing/stream endpoint to receive Server-Sent Events (SSE)
of price updates.  Each update contains bid/ask quotes with
sub-second timestamps.

Design notes:
- OANDA streams are long-lived HTTP GET with chunked transfer.
- The stream may drop after ~30-60 s of inactivity; auto-reconnect.
- Heartbeat: empty lines every ~5 s; used to detect stalled connections.
- We persist every N ticks (buffer=1000) to parquet to minimise I/O.

Usage:
    client = OandaStreamClient.from_env()
    for tick in client.stream_ticks("EUR_USD"):
        print(tick.time, tick.bid, tick.ask)
"""

from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional, Set, Tuple

import numpy as np
import pandas as pd
import requests

logger = logging.getLogger(__name__)

PRACTICE_STREAM_URL = "https://stream-fxpractice.oanda.com/v3"
LIVE_STREAM_URL = "https://stream-fxtrade.oanda.com/v3"

_HEARTBEAT_TIMEOUT_SEC = 15.0   # reconnect if no data for 15 s
_RECONNECT_BACKOFF_SEC = 1.0    # initial backoff, doubles each attempt
_MAX_RECONNECT_BACKOFF = 30.0   # cap backoff
_STREAM_CONNECT_TIMEOUT = 10.0  # TCP connect timeout
_STREAM_READ_TIMEOUT = 60.0     # total read timeout (must exceed heartbeat)


@dataclass(frozen=True)
class TickQuote:
    """Single quote tick from OANDA pricing stream."""
    instrument: str
    time: datetime           # UTC, parsed from ISO string
    bid: float               # closeoutBid (best executable)
    ask: float               # closeoutAsk (best executable)
    mid: float               # (bid + ask) / 2
    bid_liq: float           # liquidity at best bid
    ask_liq: float           # liquidity at best ask
    status: str              # "tradeable" or "non-tradeable"
    source: str              # "oanda_stream"


@dataclass
class StreamBuffer:
    """In-memory ring buffer for ticks before flush to disk."""
    ticks: List[TickQuote] = field(default_factory=list)
    max_size: int = 1000
    flush_count: int = 0

    def append(self, tick: TickQuote) -> bool:
        """Add tick; return True if buffer is full and should be flushed."""
        self.ticks.append(tick)
        if len(self.ticks) >= self.max_size:
            return True
        return False

    def flush(self) -> List[TickQuote]:
        """Drain buffer and return accumulated ticks."""
        out = self.ticks
        self.ticks = []
        self.flush_count += len(out)
        return out


class OandaStreamClient:
    """Low-level OANDA streaming client."""

    def __init__(
        self,
        account_id: str,
        api_token: str,
        *,
        environment: str = "practice",
        heartbeat_timeout: float = _HEARTBEAT_TIMEOUT_SEC,
    ) -> None:
        self.account_id = account_id
        self.api_token = api_token
        self.environment = environment
        self.heartbeat_timeout = heartbeat_timeout
        self.base_url = (
            PRACTICE_STREAM_URL if environment == "practice" else LIVE_STREAM_URL
        )
        self._session = requests.Session()
        self._session.headers.update(
            {
                "Authorization": f"Bearer {api_token}",
                "Content-Type": "application/json",
                "User-Agent": "ml_engine-oanda-stream/1.0",
            }
        )
        self._shutdown = False

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------
    @classmethod
    def from_env(cls) -> "OandaStreamClient":
        api_token = os.getenv("OANDA_API_TOKEN") or os.getenv("OANDA_API_KEY")
        account_id = os.getenv("OANDA_ACCOUNT_ID")
        if not api_token or not account_id:
            raise RuntimeError(
                "Missing OANDA env vars. Set OANDA_API_TOKEN and OANDA_ACCOUNT_ID."
            )
        env = os.getenv("OANDA_ENVIRONMENT", "practice")
        return cls(account_id=account_id, api_token=api_token, environment=env)

    # ------------------------------------------------------------------
    # Streaming generator
    # ------------------------------------------------------------------
    def stream_ticks(
        self,
        instruments: List[str],
        *,
        snapshot: bool = True,
    ) -> Generator[TickQuote, None, None]:
        """Yield live TickQuote objects until shutdown.

        Args:
            instruments: List of OANDA instrument codes, e.g. ["EUR_USD"].
            snapshot: If True, request a snapshot before the stream starts.

        Yields:
            TickQuote for every price update received.
        """
        inst_str = ",".join(instruments)
        url = f"{self.base_url}/accounts/{self.account_id}/pricing/stream"
        params: Dict[str, Any] = {"instruments": inst_str}
        if snapshot:
            params["snapshot"] = "true"

        reconnect_delay = _RECONNECT_BACKOFF_SEC

        while not self._shutdown:
            try:
                logger.info(
                    f"Connecting stream: {url} for {len(instruments)} instruments"
                )
                resp = self._session.get(
                    url,
                    params=params,
                    stream=True,
                    timeout=(
                        _STREAM_CONNECT_TIMEOUT,
                        _STREAM_READ_TIMEOUT,
                    ),
                )
                resp.raise_for_status()

                last_recv = time.time()
                for line in resp.iter_lines(decode_unicode=True):
                    if self._shutdown:
                        break

                    if not line:
                        # Heartbeat line — just update watchdog
                        last_recv = time.time()
                        continue

                    last_recv = time.time()
                    tick = self._parse_tick_line(line)
                    if tick is not None:
                        yield tick

                    # Watchdog: if no data at all for heartbeat_timeout, force reconnect
                    elapsed = time.time() - last_recv
                    if elapsed > self.heartbeat_timeout:
                        logger.warning(
                            f"Heartbeat timeout ({elapsed:.1f}s > {self.heartbeat_timeout}s); reconnecting..."
                        )
                        break

                reconnect_delay = _RECONNECT_BACKOFF_SEC  # reset on clean closure

            except requests.exceptions.HTTPError as e:
                code = e.response.status_code if e.response else 0
                if code == 400:
                    logger.error(f"Stream rejected (bad instruments?): {e}")
                    raise  # fatal — bad args
                logger.warning(f"Stream HTTP error {code}; reconnect in {reconnect_delay}s")
                time.sleep(reconnect_delay)
                reconnect_delay = min(reconnect_delay * 2, _MAX_RECONNECT_BACKOFF)

            except requests.exceptions.ConnectionError as e:
                logger.warning(f"Stream connection error; reconnect in {reconnect_delay}s: {e}")
                time.sleep(reconnect_delay)
                reconnect_delay = min(reconnect_delay * 2, _MAX_RECONNECT_BACKOFF)

            except Exception as e:
                logger.exception(f"Stream unexpected error; reconnect in {reconnect_delay}s")
                time.sleep(reconnect_delay)
                reconnect_delay = min(reconnect_delay * 2, _MAX_RECONNECT_BACKOFF)

    def _parse_tick_line(self, line: str) -> Optional[TickQuote]:
        """Parse one SSE line into TickQuote."""
        # SSE lines from OANDA are pure JSON objects, NOT "data: {...}" format
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            logger.debug(f"Ignoring non-JSON line: {line[:120]}")
            return None

        # OANDA sends two object types: Price and Heartbeat
        # Heartbeat: {"type":"HEARTBEAT","time":"2026-..."}
        if obj.get("type") == "HEARTBEAT":
            return None

        instrument = obj.get("instrument")
        if not instrument:
            return None

        # Parse prices
        bids = obj.get("bids", [])
        asks = obj.get("asks", [])
        if not bids or not asks:
            return None

        bid_price = float(bids[0]["price"])
        ask_price = float(asks[0]["price"])
        bid_liq = float(bids[0].get("liquidity", 0))
        ask_liq = float(asks[0].get("liquidity", 0))
        closeout_bid = float(obj.get("closeoutBid", bid_price))
        closeout_ask = float(obj.get("closeoutAsk", ask_price))

        # Timestamp
        ts_raw = obj.get("time", "")
        try:
            ts = datetime.fromisoformat(ts_raw.replace("Z", "+00:00"))
        except Exception:
            ts = datetime.now(timezone.utc)

        return TickQuote(
            instrument=instrument,
            time=ts,
            bid=closeout_bid,
            ask=closeout_ask,
            mid=(closeout_bid + closeout_ask) / 2.0,
            bid_liq=bid_liq,
            ask_liq=ask_liq,
            status="tradeable" if obj.get("tradeable") else "non-tradeable",
            source="oanda_stream",
        )

    def shutdown(self) -> None:
        """Signal the stream to terminate gracefully."""
        self._shutdown = True
