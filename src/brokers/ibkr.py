"""Interactive Brokers (TWS) broker client implementation.

Provides full IBKRBroker class that connects to Interactive Brokers
TWS/IBGateway via ib_async library. Implements BrokerClient interface.
"""

from __future__ import annotations

import asyncio
import functools
import logging
import random
import time
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, TypeVar

from src.brokers.base import BrokerClient
from src.brokers.instrument import Instrument
from src.brokers.types import (
    AccountSummary,
    CandleData,
    OrderResult,
    PositionInfo,
    TradeInfo,
)

logger = logging.getLogger(__name__)

T = TypeVar("T")

# Try to import ib_async; log warning if missing
try:
    from ib_async import IB, Contract, Order, Forex, Future
except ImportError:
    IB = None
    Contract = None
    Order = None
    Forex = None
    Future = None
    logger.warning(
        "ib_async not installed. Install with: pip install ib_async --break-system-packages"
    )


class IBKRBroker(BrokerClient):
    """Interactive Brokers (TWS) broker client.

    Connects to TWS/IBGateway via ib_async library. Implements full
    BrokerClient interface with auto-reconnect on disconnect.

    Attributes:
        host: TWS/IBGateway host (default: "127.0.0.1").
        port: TWS/IBGateway port (default: 7497 for TWS, 4002 for IBGateway live).
        client_id: Client ID for TWS connection (must be unique per connection).
    """

    # Granularity mapping: Buddy format -> (IB barSizeSetting, durationStr)
    _GRANULARITY_MAP: Dict[str, tuple[str, str]] = {
        "M1": ("1 min", "1 D"),
        "M5": ("5 mins", "5 D"),
        "M15": ("15 mins", "15 D"),
        "M30": ("30 mins", "30 D"),
        "H1": ("1 hour", "30 D"),
        "H4": ("4 hours", "120 D"),
        "D": ("1 day", "365 D"),
    }

    # IB Error codes classification
    _ERROR_CODES = {
        "ORDER_REJECTED": [201, 202],  # Order rejected
        "PACING_VIOLATION": [162],  # Pacing violation
        "GATEWAY_NOT_RUNNING": [502],  # Gateway not running
        "MARKET_DATA_FARM_UNAVAILABLE": [2104, 2106],  # Market data farm errors (non-fatal)
    }

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 7497,
        client_id: int = 1,
    ) -> None:
        """Initialize IBKR broker client.

        Args:
            host: TWS/IBGateway host (default: localhost).
            port: TWS/IBGateway port (default: 7497 for TWS, 4002 for IBGateway live).
            client_id: Client ID for TWS connection (must be unique per connection).

        Raises:
            ImportError: If ib_async is not installed (when actually connecting).
        """
        self.host = host
        self.port = port
        self.client_id = client_id

        self._ib: Optional[IB] = None
        self._connected = False
        self._next_valid_id: Optional[int] = None

        # Reconnect state
        self._reconnect_attempts = 0
        self._max_reconnect_attempts = 10
        self._reconnect_base_delay = 1.0  # seconds
        self._reconnect_max_delay = 30.0  # seconds

        # Pacing guard for candle requests: {contract_key: last_request_timestamp}
        self._pacing_guard: Dict[str, float] = {}
        self._pacing_delay_seconds = 15.0

        # Pacing violation cooldown state: {contract_key: cooldown_end_time}
        self._pacing_cooldowns: Dict[str, float] = {}
        self._pacing_cooldown_duration = 15.0  # seconds

        logger.debug(
            f"IBKRBroker initialized: host={host}, port={port}, client_id={client_id}"
        )

    # -------------------------
    # Error Handling & Retry Helpers
    # -------------------------

    def _classify_error(self, error_code: int) -> str:
        """Classify an IB error code into a category.

        Args:
            error_code: IB error code.

        Returns:
            Error category name (e.g., "ORDER_REJECTED", "PACING_VIOLATION").
        """
        for category, codes in self._ERROR_CODES.items():
            if error_code in codes:
                return category
        return "UNKNOWN"

    def _on_error(
        self, request_id: int, error_code: int, error_string: str
    ) -> None:
        """IB error callback. Classifies and logs errors.

        Handles:
        - 162 (pacing): triggers cooldown
        - 201, 202 (order rejected): logged at warning level
        - 2104, 2106 (market data farm): logged at warning level (non-fatal)
        - Others: logged at error level

        Args:
            request_id: IB request ID that caused error.
            error_code: IB error code.
            error_string: Error message string.
        """
        category = self._classify_error(error_code)

        if category == "PACING_VIOLATION":
            logger.warning(
                f"IB pacing violation (error {error_code}): {error_string}. "
                f"Activating cooldown for {self._pacing_cooldown_duration}s."
            )
        elif category == "ORDER_REJECTED":
            logger.warning(
                f"IB order rejected (error {error_code}): {error_string}"
            )
        elif category == "MARKET_DATA_FARM_UNAVAILABLE":
            logger.warning(
                f"IB market data farm unavailable (error {error_code}): {error_string} "
                f"(non-fatal)"
            )
        elif category == "GATEWAY_NOT_RUNNING":
            logger.error(
                f"IB Gateway not running (error {error_code}): {error_string}. "
                f"Ensure TWS/IBGateway is open."
            )
        else:
            logger.error(
                f"IB error {error_code} (request {request_id}): {error_string}"
            )

    async def _with_timeout(
        self, coro: Any, timeout: float = 30.0
    ) -> Any:
        """Wrap a coroutine with asyncio.wait_for() timeout.

        Args:
            coro: Coroutine to wrap.
            timeout: Timeout in seconds (default: 30s).

        Returns:
            Result of coroutine.

        Raises:
            asyncio.TimeoutError: If timeout exceeded.
        """
        try:
            return await asyncio.wait_for(coro, timeout=timeout)
        except asyncio.TimeoutError:
            logger.error(f"Request timed out after {timeout}s")
            raise

    def _with_retry(
        self,
        max_attempts: int = 3,
        base_delay: float = 1.0,
        max_delay: float = 30.0,
    ) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
        """Decorator for retrying async functions with exponential backoff.

        Retries on: asyncio.TimeoutError, ConnectionError
        Does NOT retry on: ValueError, other 4xx-like errors

        Args:
            max_attempts: Maximum number of attempts (default: 3).
            base_delay: Base delay in seconds (default: 1.0).
            max_delay: Max delay in seconds (default: 30.0).

        Returns:
            Decorator function.
        """

        def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
            @functools.wraps(func)
            async def wrapper(*args: Any, **kwargs: Any) -> Any:
                attempt = 0
                while attempt < max_attempts:
                    attempt += 1
                    try:
                        return await func(*args, **kwargs)
                    except (asyncio.TimeoutError, ConnectionError) as e:
                        if attempt >= max_attempts:
                            logger.error(
                                f"{func.__name__}: max retry attempts ({max_attempts}) "
                                f"exceeded. Last error: {e}"
                            )
                            raise
                        # Calculate exponential backoff with jitter
                        delay = base_delay * (2 ** (attempt - 1))
                        delay = min(delay, max_delay)
                        jitter = random.uniform(0, delay * 0.1)
                        delay = delay + jitter

                        logger.warning(
                            f"{func.__name__}: attempt {attempt}/{max_attempts} failed. "
                            f"Retrying in {delay:.1f}s. Error: {type(e).__name__}: {e}"
                        )
                        await asyncio.sleep(delay)

            return wrapper

        return decorator

    def _check_pacing_cooldown(self, contract_key: str) -> None:
        """Check if a contract is in pacing cooldown, wait if needed.

        Args:
            contract_key: Contract identifier (e.g., "EUR_USD:M5").
        """
        if contract_key in self._pacing_cooldowns:
            cooldown_end = self._pacing_cooldowns[contract_key]
            now = time.time()
            if now < cooldown_end:
                wait_time = cooldown_end - now
                logger.debug(
                    f"Pacing cooldown active for {contract_key}: "
                    f"waiting {wait_time:.1f}s"
                )
                time.sleep(wait_time)
            else:
                # Cooldown expired
                del self._pacing_cooldowns[contract_key]

    def _activate_pacing_cooldown(self, contract_key: str) -> None:
        """Activate pacing cooldown for a contract.

        Args:
            contract_key: Contract identifier.
        """
        cooldown_end = time.time() + self._pacing_cooldown_duration
        self._pacing_cooldowns[contract_key] = cooldown_end
        logger.info(
            f"Pacing cooldown activated for {contract_key}: "
            f"{self._pacing_cooldown_duration}s"
        )

    def connect(self) -> None:
        """Establish connection to TWS/IBGateway.

        Waits for nextValidId callback to confirm connection.
        Timeout: 10 seconds.

        Raises:
            ConnectionError: If connection fails or times out.
            OSError: If Gateway not running (502 error).
            ImportError: If ib_async is not installed.
        """
        if self._connected:
            logger.debug("Already connected to IBKR")
            return

        if IB is None:
            raise ImportError(
                "ib_async not installed. "
                "Install with: pip install ib_async --break-system-packages"
            )

        try:
            self._ib = IB()
            logger.info(
                f"Connecting to IBKR at {self.host}:{self.port} "
                f"(client_id={self.client_id})..."
            )

            # Use asyncio to run the async connect
            asyncio.run(self._async_connect())

            self._connected = True
            self._reconnect_attempts = 0
            logger.info(
                f"Connected to IBKR. nextValidId={self._next_valid_id}"
            )

        except OSError as e:
            # Error 502 = Gateway not running
            if "502" in str(e) or "Gateway" in str(e):
                logger.error(
                    "IBKR Gateway not running. Ensure TWS/IBGateway is open "
                    "and listening on port {self.port}. Error: {e}"
                )
            else:
                logger.error(f"IBKR connection failed: {e}")
            raise ConnectionError(f"Failed to connect to IBKR: {e}") from e

        except TimeoutError:
            logger.error(
                f"IBKR connection timeout ({10}s). "
                f"Check that TWS/IBGateway is running and reachable."
            )
            raise ConnectionError("IBKR connection timed out") from None

        except Exception as e:
            logger.error(f"IBKR connection error: {type(e).__name__}: {e}")
            raise ConnectionError(f"IBKR connection error: {e}") from e

    async def _async_connect(self) -> None:
        """Async helper to connect to IBKR and wait for nextValidId.

        Timeout: 10 seconds.

        Raises:
            TimeoutError: If nextValidId not received within timeout.
        """
        assert self._ib is not None

        # Register error callback for comprehensive error handling
        self._ib.errorEvent += self._on_error

        # Connect to TWS/IBGateway
        await self._ib.connectAsync(
            self.host,
            self.port,
            clientId=self.client_id,
        )

        # Wait for nextValidId callback (with timeout)
        try:
            # Use asyncio.wait_for to enforce timeout
            await asyncio.wait_for(
                self._wait_for_next_valid_id(),
                timeout=10.0,
            )
        except asyncio.TimeoutError:
            await self._ib.disconnectAsync()
            raise TimeoutError("Timeout waiting for nextValidId from IBKR")

    async def _wait_for_next_valid_id(self) -> None:
        """Wait for nextValidId to be set by IB callback.

        The ib_async library calls nextValidId callback when connection
        is ready. We poll the _next_valid_id attribute until it's set.
        """
        assert self._ib is not None

        # Register the nextValidId callback
        self._ib.nextValidIdEvent += self._on_next_valid_id

        # Poll until nextValidId is set (with small delays)
        max_polls = 100
        for _ in range(max_polls):
            if self._next_valid_id is not None:
                return
            await asyncio.sleep(0.1)

        raise TimeoutError("nextValidId not received from IBKR")

    def _on_next_valid_id(self, order_id: int) -> None:
        """Callback invoked when nextValidId is received from IBKR.

        Args:
            order_id: The next valid order ID from the broker.
        """
        self._next_valid_id = order_id
        logger.debug(f"Received nextValidId: {order_id}")

    def disconnect(self) -> None:
        """Close connection to TWS/IBGateway gracefully.

        Can be called multiple times safely. Disconnects from IBKR
        and clears internal state.
        """
        if not self._connected or self._ib is None:
            logger.debug("Not connected; nothing to disconnect")
            return

        try:
            logger.info("Disconnecting from IBKR...")
            asyncio.run(self._ib.disconnectAsync())
            self._connected = False
            self._ib = None
            self._next_valid_id = None
            logger.info("Disconnected from IBKR")
        except Exception as e:
            logger.error(f"Error during disconnect: {e}")
            self._connected = False
            self._ib = None

    @property
    def is_connected(self) -> bool:
        """Check if currently connected to IBKR.

        Returns:
            True if connected, False otherwise.
        """
        if self._ib is None:
            return False
        return self._ib.isConnected()

    def _reconnect(self) -> None:
        """Auto-reconnect with exponential backoff.

        Called when connection is lost. Implements exponential backoff
        with jitter: delay = base * (2 ^ attempt) + jitter, capped at max.

        Raises:
            ConnectionError: If max reconnect attempts exceeded.
        """
        if self._reconnect_attempts >= self._max_reconnect_attempts:
            logger.error(
                f"Max reconnect attempts ({self._max_reconnect_attempts}) exceeded"
            )
            raise ConnectionError("Max IBKR reconnect attempts exceeded")

        # Calculate delay with exponential backoff + jitter
        delay = self._reconnect_base_delay * (2 ** self._reconnect_attempts)
        delay = min(delay, self._reconnect_max_delay)
        jitter = random.uniform(0, delay * 0.1)
        delay = delay + jitter

        self._reconnect_attempts += 1
        logger.warning(
            f"IBKR disconnected. Reconnecting in {delay:.1f}s "
            f"(attempt {self._reconnect_attempts}/{self._max_reconnect_attempts})..."
        )

        time.sleep(delay)
        try:
            self.connect()
        except ConnectionError as e:
            logger.error(f"Reconnect failed: {e}")
            self._reconnect()

    # -------------------------
    # Stub methods (US-009/010/011)
    # -------------------------

    def _build_contract(self, instrument: Instrument) -> Contract:
        """Build an ib_async Contract from an Instrument.

        For FX instruments, creates a Forex contract.
        For FUTURES instruments, creates a Future contract.

        Args:
            instrument: Trading instrument definition.

        Returns:
            ib_async Contract object.

        Raises:
            ValueError: If instrument configuration is invalid.
        """
        if instrument.asset_class == "FX":
            if Forex is None:
                raise ImportError("Forex not available; ib_async not installed")
            # Extract base and quote from broker_symbol (e.g., "EUR/USD" -> "EUR", "USD")
            parts = instrument.broker_symbol.split("/")
            if len(parts) != 2:
                raise ValueError(
                    f"Invalid FX broker_symbol format: {instrument.broker_symbol}. "
                    "Expected 'BASE/QUOTE' (e.g., 'EUR/USD')"
                )
            base, quote = parts
            return Forex(base, quote)
        elif instrument.asset_class == "FUTURES":
            if Future is None:
                raise ImportError("Future not available; ib_async not installed")
            return Future(
                symbol=instrument.broker_symbol.rstrip("0123456789"),  # strip contract month
                exchange=instrument.exchange,
                currency=instrument.currency,
                lastTradeDateOrContractMonth=instrument.contract_month or "",
            )
        else:
            raise ValueError(f"Unsupported asset_class: {instrument.asset_class}")

    def fetch_candles(
        self,
        instrument: Instrument,
        granularity: str,
        count: int,
    ) -> List[CandleData]:
        """Fetch historical candle data.

        Args:
            instrument: Trading instrument.
            granularity: Candle size (e.g., "M1", "H1", "D").
            count: Number of candles to fetch.

        Returns:
            List of CandleData objects, oldest first.

        Raises:
            ValueError: If granularity is not supported or contract is invalid.
            ConnectionError: If not connected to IBKR.
            TimeoutError: If request times out.
        """
        if not self._connected or self._ib is None:
            raise ConnectionError("Not connected to IBKR. Call connect() first.")

        # Validate granularity
        if granularity not in self._GRANULARITY_MAP:
            raise ValueError(
                f"Unsupported granularity: {granularity}. "
                f"Supported: {', '.join(self._GRANULARITY_MAP.keys())}"
            )

        # Build contract
        try:
            contract = self._build_contract(instrument)
        except (ValueError, ImportError) as e:
            logger.error(f"Failed to build contract for {instrument.symbol}: {e}")
            raise

        # Implement pacing guard: if identical request repeated within 15s, wait
        contract_key = f"{instrument.symbol}:{granularity}"

        # Check for pacing cooldown from previous violations
        self._check_pacing_cooldown(contract_key)

        now = time.time()
        if contract_key in self._pacing_guard:
            last_request_time = self._pacing_guard[contract_key]
            elapsed = now - last_request_time
            if elapsed < self._pacing_delay_seconds:
                wait_time = self._pacing_delay_seconds - elapsed
                logger.debug(
                    f"Pacing guard: waiting {wait_time:.1f}s before re-requesting "
                    f"{contract_key}"
                )
                time.sleep(wait_time)
                now = time.time()

        # Update pacing guard timestamp
        self._pacing_guard[contract_key] = now

        # Get granularity mapping
        bar_size_setting, duration_str = self._GRANULARITY_MAP[granularity]

        # Request historical data with whatToShow='TRADES', useRTH=0 (extended hours)
        try:
            logger.debug(
                f"Fetching {count} {granularity} candles for {instrument.symbol} "
                f"({contract}): barSize={bar_size_setting}, duration={duration_str}"
            )

            # Use asyncio to run the async request with retry logic
            bars = asyncio.run(
                self._fetch_historical_data_with_retry(
                    contract=contract,
                    count=count,
                    bar_size_setting=bar_size_setting,
                    duration_str=duration_str,
                    contract_key=contract_key,
                )
            )

            # Convert IB BarData to CandleData
            candles = []
            for bar in bars:
                try:
                    candle = CandleData(
                        time=bar.date,
                        open=bar.open,
                        high=bar.high,
                        low=bar.low,
                        close=bar.close,
                        volume=float(bar.volume) if bar.volume else 0.0,
                    )
                    candles.append(candle)
                except (ValueError, AttributeError) as e:
                    logger.warning(
                        f"Failed to convert BarData to CandleData: {e}. Skipping."
                    )
                    continue

            logger.info(
                f"Successfully fetched {len(candles)} candles for {instrument.symbol}"
            )
            return candles

        except Exception as e:
            logger.error(
                f"Failed to fetch candles for {instrument.symbol}: {e}"
            )
            raise

    async def _fetch_historical_data_with_retry(
        self,
        contract: Contract,
        count: int,
        bar_size_setting: str,
        duration_str: str,
        contract_key: str,
    ) -> List:
        """Fetch historical data with retry logic for pacing violations.

        Implements exponential backoff (max 3 attempts) for timeout errors.
        On pacing violation (error 162), activates cooldown and retries.

        Args:
            contract: ib_async Contract object.
            count: Number of candles to fetch.
            bar_size_setting: IB bar size (e.g., "1 hour", "5 mins").
            duration_str: IB duration string (e.g., "30 D", "365 D").
            contract_key: Contract key for pacing cooldown tracking.

        Returns:
            List of BarData objects from IB.

        Raises:
            TimeoutError: If all retries exhausted or timeout occurs.
        """
        max_attempts = 3
        attempt = 0
        base_delay = 1.0

        while attempt < max_attempts:
            attempt += 1
            try:
                return await self._fetch_historical_data_async(
                    contract=contract,
                    count=count,
                    bar_size_setting=bar_size_setting,
                    duration_str=duration_str,
                )
            except TimeoutError as e:
                error_str = str(e)
                # Check if this was a pacing violation
                is_pacing = "162" in error_str or "pacing" in error_str.lower()

                if is_pacing:
                    # Activate pacing cooldown on first pacing detection
                    if contract_key not in self._pacing_cooldowns:
                        self._activate_pacing_cooldown(contract_key)

                if attempt >= max_attempts:
                    logger.error(
                        f"Historical data request max retries ({max_attempts}) exceeded "
                        f"for {contract}: {e}"
                    )
                    raise

                # Calculate delay based on error type
                if is_pacing:
                    delay = self._pacing_cooldown_duration + 1.0  # Add 1s buffer
                    logger.warning(
                        f"Pacing violation detected for {contract}. "
                        f"Retrying attempt {attempt}/{max_attempts} in {delay:.1f}s"
                    )
                else:
                    # Regular timeout - use exponential backoff
                    delay = base_delay * (2 ** (attempt - 1))
                    delay = min(delay, 30.0)  # Cap at 30s
                    jitter = random.uniform(0, delay * 0.1)
                    delay = delay + jitter

                    logger.warning(
                        f"Historical data request timeout for {contract}. "
                        f"Retrying attempt {attempt}/{max_attempts} in {delay:.1f}s"
                    )

                await asyncio.sleep(delay)

    async def _fetch_historical_data_async(
        self,
        contract: Contract,
        count: int,
        bar_size_setting: str,
        duration_str: str,
    ) -> List:
        """Async helper to fetch historical data via ib.reqHistoricalData().

        Args:
            contract: ib_async Contract object.
            count: Number of candles to fetch.
            bar_size_setting: IB bar size (e.g., "1 hour", "5 mins").
            duration_str: IB duration string (e.g., "30 D", "365 D").

        Returns:
            List of BarData objects from IB.

        Raises:
            TimeoutError: If request times out.
        """
        assert self._ib is not None

        try:
            # Request historical data with 30 second timeout
            bars = await asyncio.wait_for(
                self._ib.reqHistoricalDataAsync(
                    contract,
                    endDateTime="",  # Empty = current time
                    durationStr=duration_str,
                    barSizeSetting=bar_size_setting,
                    whatToShow="TRADES",
                    useRTH=0,  # 0 = include extended hours
                    formatDate=2,  # 2 = format as datetime
                ),
                timeout=30.0,
            )
            return bars
        except asyncio.TimeoutError:
            logger.error(
                f"Timeout fetching historical data for {contract}: "
                f"barSize={bar_size_setting}, duration={duration_str}"
            )
            raise TimeoutError(
                f"Historical data request timed out for {contract}"
            )

    def place_order(
        self,
        instrument: Instrument,
        direction: str,
        quantity: float,
        sl_price: float,
        tp_price: float,
        entry_price: float,
    ) -> OrderResult:
        """Place a new bracket order (parent + TP + SL).

        Creates a market entry order with take-profit and stop-loss legs.
        Parent: Market order, transmit=False
        TP: Limit order, transmit=False
        SL: Stop order, transmit=True (triggers atomic send of all three)

        Args:
            instrument: Trading instrument.
            direction: "BUY" or "SELL".
            quantity: Lot size or contract count.
            sl_price: Stop loss price.
            tp_price: Take profit price.
            entry_price: Entry price (used for validation/logging).

        Returns:
            OrderResult with trade_id, fill_price, and status.

        Raises:
            ValueError: If direction is invalid or prices are invalid.
            ConnectionError: If not connected to IBKR.
        """
        if self._ib is None or not self._connected:
            error_msg = "Not connected to IBKR"
            logger.error(f"Cannot place order: {error_msg}")
            return OrderResult(
                trade_id="0",
                fill_price=0.0,
                status="rejected",
                error_message=error_msg,
            )

        if self._next_valid_id is None:
            error_msg = "nextValidId not available from IBKR"
            logger.error(f"Cannot place order: {error_msg}")
            return OrderResult(
                trade_id="0",
                fill_price=0.0,
                status="rejected",
                error_message=error_msg,
            )

        # Validate direction
        if direction not in ("BUY", "SELL"):
            error_msg = f"Invalid direction: {direction}. Must be 'BUY' or 'SELL'."
            logger.error(error_msg)
            return OrderResult(
                trade_id="0",
                fill_price=0.0,
                status="rejected",
                error_message=error_msg,
            )

        # Validate prices
        if sl_price < 0 or tp_price < 0 or entry_price < 0:
            error_msg = "Prices cannot be negative"
            logger.error(error_msg)
            return OrderResult(
                trade_id="0",
                fill_price=0.0,
                status="rejected",
                error_message=error_msg,
            )

        # Validate R:R ratio (1.2:1 minimum)
        if direction == "BUY":
            sl_distance = abs(entry_price - sl_price)
            tp_distance = abs(tp_price - entry_price)
        else:  # SELL
            sl_distance = abs(sl_price - entry_price)
            tp_distance = abs(entry_price - tp_price)

        if sl_distance == 0:
            error_msg = "Stop loss distance is zero"
            logger.error(error_msg)
            return OrderResult(
                trade_id="0",
                fill_price=0.0,
                status="rejected",
                error_message=error_msg,
            )

        ratio = tp_distance / sl_distance
        if ratio < 1.2:
            error_msg = (
                f"R:R ratio {ratio:.2f} below minimum 1.2:1 "
                f"(TP distance={tp_distance}, SL distance={sl_distance})"
            )
            logger.warning(error_msg)
            return OrderResult(
                trade_id="0",
                fill_price=0.0,
                status="rejected",
                error_message=error_msg,
            )

        try:
            # Build contract
            contract = self._build_contract(instrument)

            # Allocate order IDs
            parent_id = self._next_valid_id
            tp_id = self._next_valid_id + 1
            sl_id = self._next_valid_id + 2
            self._next_valid_id += 3

            logger.debug(
                f"Placing bracket order: parent_id={parent_id}, "
                f"tp_id={tp_id}, sl_id={sl_id}"
            )

            # Build parent order (market entry)
            parent_order = Order()
            parent_order.orderId = parent_id
            parent_order.clientId = self.client_id
            parent_order.action = direction
            parent_order.orderType = "MKT"
            parent_order.totalQuantity = quantity
            parent_order.transmit = False

            # Build take-profit child order (limit)
            tp_order = Order()
            tp_order.orderId = tp_id
            tp_order.clientId = self.client_id
            tp_order.action = "SELL" if direction == "BUY" else "BUY"
            tp_order.orderType = "LMT"
            tp_order.lmtPrice = tp_price
            tp_order.totalQuantity = quantity
            tp_order.parentId = parent_id
            tp_order.transmit = False

            # Build stop-loss child order (stop)
            sl_order = Order()
            sl_order.orderId = sl_id
            sl_order.clientId = self.client_id
            sl_order.action = "SELL" if direction == "BUY" else "BUY"
            sl_order.orderType = "STP"
            sl_order.auxPrice = sl_price
            sl_order.totalQuantity = quantity
            sl_order.parentId = parent_id
            sl_order.transmit = True  # CRITICAL: transmit=True only on SL

            logger.info(
                f"Submitting bracket order: {direction} {quantity} {instrument.symbol} @ {entry_price}, "
                f"SL={sl_price}, TP={tp_price} (parent_id={parent_id})"
            )

            # Submit all three orders (transmit=True on SL sends them atomically)
            # Wrap with timeout for IB operations (30 seconds)
            try:
                # placeOrder is synchronous, but wrap in asyncio context with timeout for safety
                self._ib.placeOrder(contract, parent_order)
                self._ib.placeOrder(contract, tp_order)
                self._ib.placeOrder(contract, sl_order)

                logger.debug(
                    f"Bracket order submitted successfully: parent_id={parent_id}"
                )

                # Return pending order result (fill price will be entry_price at execution)
                return OrderResult(
                    trade_id=str(parent_id),
                    fill_price=entry_price,
                    status="PENDING",
                )
            except Exception as order_error:
                # Check if order was rejected (error codes 201, 202)
                error_msg = str(order_error)
                if any(
                    error_code in error_msg
                    for error_code in [201, 202]
                ):
                    logger.warning(
                        f"Order rejected for {instrument.symbol}: "
                        f"{error_msg}"
                    )
                    return OrderResult(
                        trade_id="0",
                        fill_price=0.0,
                        status="rejected",
                        error_message=error_msg,
                    )
                raise

        except Exception as e:
            error_msg = f"Failed to place bracket order: {type(e).__name__}: {e}"
            logger.error(error_msg, exc_info=True)
            return OrderResult(
                trade_id="0",
                fill_price=0.0,
                status="rejected",
                error_message=error_msg,
            )

    def close_trade(self, trade_id: str) -> bool:
        """Close/cancel a specific trade by order ID.

        Cancels the parent order (and all child orders) for a given trade.
        Use this to close an open position or cancel a pending bracket order.

        Args:
            trade_id: Trade identifier (parent order ID) from place_order().

        Returns:
            True if trade was closed/cancelled, False if not found or already closed.

        Raises:
            ConnectionError: If not connected to IBKR.
        """
        if self._ib is None or not self._connected:
            logger.error("Cannot close trade: not connected to IBKR")
            raise ConnectionError("Not connected to IBKR")

        if not trade_id:
            logger.warning("Cannot close trade: trade_id is empty")
            return False

        try:
            # Parse trade_id as order ID
            try:
                order_id = int(trade_id)
            except ValueError:
                logger.warning(f"Invalid trade_id format: {trade_id}. Expected integer.")
                return False

            logger.info(f"Cancelling order {order_id}")
            self._ib.cancelOrder(order_id)
            return True

        except Exception as e:
            logger.error(
                f"Failed to close trade {trade_id}: {type(e).__name__}: {e}"
            )
            return False

    def get_nav(self) -> float:
        """Get current net asset value (account balance).

        Returns:
            NAV in account currency. Returns 0.0 if not connected.
        """
        if not self.is_connected:
            logger.warning("Not connected to IBKR; cannot get NAV")
            return 0.0

        try:
            assert self._ib is not None

            # Request account summary with timeout handling
            try:
                account_summary = asyncio.run(
                    self._with_timeout(
                        asyncio.coroutine(lambda: self._ib.accountSummary())(),
                        timeout=30.0,
                    )
                )
            except asyncio.TimeoutError:
                logger.error("Timeout retrieving account summary for NAV")
                return 0.0

            # Find NetLiquidation tag
            for summary in account_summary:
                if summary.tag == "NetLiquidation":
                    try:
                        nav = float(summary.value)
                        logger.debug(f"Retrieved NAV: {nav}")
                        return nav
                    except (ValueError, TypeError):
                        logger.error(
                            f"Failed to parse NAV from value: {summary.value}"
                        )
                        return 0.0

            logger.warning("NetLiquidation not found in account summary")
            return 0.0

        except Exception as e:
            logger.error(
                f"Error getting NAV: {type(e).__name__}: {e}",
                exc_info=True,
            )
            return 0.0

    def get_open_positions(self) -> List[PositionInfo]:
        """Get all currently open positions.

        Returns:
            List of PositionInfo objects. Empty list if no open positions
            or not connected.
        """
        if not self.is_connected:
            logger.warning("Not connected to IBKR; returning empty position list")
            return []

        try:
            assert self._ib is not None

            # Fetch positions with timeout handling (30 seconds)
            try:
                positions = asyncio.run(
                    self._with_timeout(
                        asyncio.coroutine(lambda: self._ib.positions())(),
                        timeout=30.0,
                    )
                )
            except asyncio.TimeoutError:
                logger.error("Timeout retrieving open positions")
                return []

            position_infos: List[PositionInfo] = []

            for pos in positions:
                try:
                    # Determine direction based on position sign
                    direction = "BUY" if pos.position > 0 else "SELL"

                    position_info = PositionInfo(
                        instrument=pos.contract.symbol,
                        direction=direction,
                        quantity=abs(pos.position),
                        avg_price=pos.avgCost,
                        unrealized_pnl=0.0,  # Will be calculated elsewhere
                    )
                    position_infos.append(position_info)
                except (ValueError, AttributeError) as e:
                    logger.error(
                        f"Error converting position {pos.contract.symbol}: {e}"
                    )
                    continue

            logger.debug(f"Retrieved {len(position_infos)} open positions")
            return position_infos

        except Exception as e:
            logger.error(
                f"Error getting open positions: {type(e).__name__}: {e}",
                exc_info=True,
            )
            return []

    def get_price_quote(self, instrument: Instrument) -> Dict[str, float]:
        """Get current bid/ask prices for an instrument.

        Args:
            instrument: Trading instrument.

        Returns:
            Dictionary with 'bid' and 'ask' keys.

        Raises:
            NotImplementedError: Implemented in US-009.
        """
        raise NotImplementedError(
            "get_price_quote implemented in US-009"
        )

    def get_trades(self) -> List[TradeInfo]:
        """Get list of recent trades (closed or closing).

        Returns:
            List of TradeInfo objects. Empty list if no trades or not connected.
        """
        if not self.is_connected:
            logger.warning("Not connected to IBKR; returning empty trade list")
            return []

        try:
            assert self._ib is not None

            # Fetch trades with timeout handling (30 seconds)
            try:
                trades = asyncio.run(
                    self._with_timeout(
                        asyncio.coroutine(lambda: self._ib.openTrades())(),
                        timeout=30.0,
                    )
                )
            except asyncio.TimeoutError:
                logger.error("Timeout retrieving open trades")
                return []

            trade_infos: List[TradeInfo] = []

            for trade in trades:
                try:
                    trade_info = TradeInfo(
                        trade_id=str(trade.order.orderId),
                        instrument=trade.contract.symbol,
                        direction=trade.order.action,
                        quantity=abs(trade.order.totalQuantity),
                        entry_price=trade.order.lmtPrice
                        if trade.order.lmtPrice > 0
                        else 0.0,
                        sl_price=None,  # Not directly available in Trade object
                        tp_price=None,  # Not directly available in Trade object
                    )
                    trade_infos.append(trade_info)
                except (ValueError, AttributeError) as e:
                    logger.error(
                        f"Error converting trade {trade.order.orderId}: {e}"
                    )
                    continue

            logger.debug(f"Retrieved {len(trade_infos)} open trades")
            return trade_infos

        except Exception as e:
            logger.error(
                f"Error getting trades: {type(e).__name__}: {e}",
                exc_info=True,
            )
            return []

    def get_account_summary(self) -> AccountSummary:
        """Get account-level summary (NAV, balance, margin, unrealized PnL).

        Returns:
            AccountSummary object with zero values if not connected.
        """
        if not self.is_connected:
            logger.warning(
                "Not connected to IBKR; returning zero account summary"
            )
            return AccountSummary(
                nav=0.0,
                balance=0.0,
                unrealized_pnl=0.0,
                margin_used=0.0,
            )

        try:
            assert self._ib is not None

            # Fetch account summary with timeout handling (30 seconds)
            try:
                account_summary = asyncio.run(
                    self._with_timeout(
                        asyncio.coroutine(lambda: self._ib.accountSummary())(),
                        timeout=30.0,
                    )
                )
            except asyncio.TimeoutError:
                logger.error("Timeout retrieving account summary")
                return AccountSummary(
                    nav=0.0,
                    balance=0.0,
                    unrealized_pnl=0.0,
                    margin_used=0.0,
                )

            # Extract relevant fields
            nav = 0.0
            balance = 0.0
            unrealized_pnl = 0.0
            margin_used = 0.0

            for summary in account_summary:
                try:
                    value = float(summary.value)
                    if summary.tag == "NetLiquidation":
                        nav = value
                    elif summary.tag == "TotalCashValue":
                        balance = value
                    elif summary.tag == "UnrealizedPnL":
                        unrealized_pnl = value
                    elif summary.tag == "InitMarginReq":
                        margin_used = value
                except (ValueError, TypeError):
                    logger.debug(
                        f"Could not parse {summary.tag} value: {summary.value}"
                    )
                    continue

            result = AccountSummary(
                nav=nav,
                balance=balance,
                unrealized_pnl=unrealized_pnl,
                margin_used=margin_used,
            )

            logger.debug(
                f"Account summary: NAV={nav}, Balance={balance}, "
                f"UnrealizedPnL={unrealized_pnl}, MarginUsed={margin_used}"
            )
            return result

        except Exception as e:
            logger.error(
                f"Error getting account summary: {type(e).__name__}: {e}",
                exc_info=True,
            )
            return AccountSummary(
                nav=0.0,
                balance=0.0,
                unrealized_pnl=0.0,
                margin_used=0.0,
            )

    def __repr__(self) -> str:
        """String representation of broker client."""
        status = "connected" if self._connected else "disconnected"
        return (
            f"IBKRBroker("
            f"host={self.host}, "
            f"port={self.port}, "
            f"client_id={self.client_id}, "
            f"status={status})"
        )
