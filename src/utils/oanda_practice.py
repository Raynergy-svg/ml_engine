"""Minimal OANDA v20 REST client (practice).

This module intentionally targets the PRACTICE environment only.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, Optional

import time
import uuid

import os
import sys
try:
    import requests  # type: ignore
except Exception:  # pragma: no cover
    requests = None


def _parse_simple_dotenv(path: "Path") -> None:
    """Minimal dotenv parser fallback when python-dotenv is unavailable."""
    try:
        for raw_line in path.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if key and key not in os.environ:
                os.environ[key] = value
    except Exception:
        return


def _session_label_from_iso(ts: str | None) -> str | None:
    """Map an ISO timestamp to a coarse FX session label."""
    if not ts:
        return None
    try:
        dt = datetime.fromisoformat(str(ts).replace('Z', '+00:00'))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        hour = dt.astimezone(timezone.utc).hour
    except Exception:
        return None
    if 0 <= hour < 7:
        return "asia"
    if 7 <= hour < 12:
        return "london"
    if 12 <= hour < 17:
        return "ny_overlap"
    if 17 <= hour < 21:
        return "new_york"
    return "rollover"


def _load_project_dotenv() -> None:
    """Best-effort load of `.env.local` / `.env` for local runs.

    This keeps CLI workflows consistent with `unified_talk`/Buddy, and is
    fail-open: missing `python-dotenv` or parse issues should not crash.
    """
    try:
        from pathlib import Path

        load_dotenv = None
        try:
            from dotenv import load_dotenv as _load_dotenv  # type: ignore
            load_dotenv = _load_dotenv
        except Exception:
            load_dotenv = None

        roots = []
        try:
            roots.append(Path.cwd())
        except Exception:
            pass

        try:
            roots.append(Path(__file__).resolve().parent)
        except Exception:
            pass

        seen: set[str] = set()
        candidates = []
        for root in roots:
            try:
                resolved = root.resolve()
            except Exception:
                continue
            chain = [resolved, *resolved.parents]
            for base in chain:
                key = str(base)
                if key in seen:
                    continue
                seen.add(key)
                candidates.append(base)

        for base in candidates:
            for name in (".env.local", ".env"):
                p = base / name
                if p.exists() and p.is_file():
                    if load_dotenv is not None:
                        load_dotenv(dotenv_path=str(p), override=False)
                    else:
                        _parse_simple_dotenv(p)
    except Exception:
        return


# OANDA REST v20 base URLs (see: https://developer.oanda.com/rest-live-v20/development-guide/)
PRACTICE_API_URL = "https://api-fxpractice.oanda.com/v3"


class OandaApiError(RuntimeError):
    def __init__(
        self,
        status_code: int,
        message: str,
        *,
        error_message: Optional[str] = None,
        reject_reason: Optional[str] = None,
        response_text: Optional[str] = None,
    ):
        super().__init__(message)
        self.status_code = status_code
        self.error_message = error_message
        self.reject_reason = reject_reason
        self.response_text = response_text


def _format_price(instrument: str, price: float) -> str:
    """Format a price string with conservative, instrument-aware precision.

    OANDA enforces instrument display precision. As a safe default:
    - JPY pairs use 3 decimals (e.g., 149.123)
    - others use 5 decimals (e.g., 1.23456)
    """
    decimals = 3 if str(instrument).endswith("_JPY") else 5
    return f"{float(price):.{decimals}f}"


@dataclass(frozen=True)
class OandaPracticeConfig:
    api_token: str
    account_id: str
    timeout_seconds: int = 15


class OandaPracticeClient:
    def __init__(self, config: OandaPracticeConfig):
        if requests is None:
            raise ModuleNotFoundError(
                "The 'requests' package is required for OANDA client usage. Install it with: pip install requests"
            )
        self._config = config
        self._session = requests.Session()
        self._session.headers.update(
            {
                "Authorization": f"Bearer {config.api_token}",
                "Content-Type": "application/json",
                "Accept": "application/json",
                "User-Agent": "ml_engine-oanda-practice/1.0",
            }
        )

    def close(self) -> None:
        """Close the underlying HTTP session and release socket resources."""
        if hasattr(self, "_session") and self._session is not None:
            try:
                self._session.close()
            except Exception:
                pass

    def __del__(self) -> None:
        self.close()

    def __enter__(self) -> "OandaPracticeClient":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    @classmethod
    def from_env(cls) -> "OandaPracticeClient":
        # Tests expect missing env vars to raise; avoid auto-loading .env under pytest.
        if "pytest" not in sys.modules:
            _load_project_dotenv()
        api_token = os.getenv("OANDA_API_TOKEN") or os.getenv("OANDA_API_KEY")
        account_id = os.getenv("OANDA_ACCOUNT_ID")
        if not api_token or not account_id:
            raise OSError(
                "Missing OANDA env vars. Set OANDA_API_TOKEN (or OANDA_API_KEY) and OANDA_ACCOUNT_ID."
            )
        return cls(OandaPracticeConfig(api_token=api_token, account_id=account_id))

    def _parse_error_details(self, resp: requests.Response) -> tuple[Optional[str], Optional[str]]:
        """Parse common OANDA error shapes from response."""
        try:
            err = resp.json()
            if isinstance(err, dict):
                return err.get("errorMessage"), err.get("rejectReason")
        except Exception:
            pass
        return None, None

    def _get_retry_sleep(self, resp: requests.Response, default_backoff: float) -> float:
        """Calculate sleep duration for retry based on response headers."""
        retry_after = resp.headers.get("Retry-After")
        if retry_after:
            try:
                return float(retry_after)
            except ValueError:
                pass
        return default_backoff

    def _should_retry(self, status_code: int, attempt: int, max_attempts: int) -> bool:
        """Determine if request should be retried based on status code."""
        retryable_codes = {429, 500, 502, 503, 504}
        return status_code in retryable_codes and attempt < max_attempts

    def _build_error_message(self, resp: requests.Response, error_message: Optional[str]) -> str:
        """Build error message from response."""
        msg = f"OANDA API error {resp.status_code}"
        if error_message:
            msg += f": {error_message}"
        elif resp.text:
            msg += f": {resp.text}"
        return msg

    def _request(
        self,
        method: str,
        path: str,
        *,
        params: Optional[Dict[str, Any]] = None,
        json: Any = None,
    ) -> Any:
        url = PRACTICE_API_URL + path
        timeout = (5, self._config.timeout_seconds)
        max_attempts = 3
        backoff_seconds = 0.5

        for attempt in range(1, max_attempts + 1):
            try:
                resp = self._session.request(
                    method, url, params=params, json=json, timeout=timeout
                )
            except requests.exceptions.RequestException as e:
                if attempt == max_attempts:
                    raise OandaApiError(0, f"Network error: {e}")
                time.sleep(backoff_seconds)
                backoff_seconds *= 2.0
                continue

            if resp.status_code < 400:
                return resp.json() if resp.content else {}

            error_message, reject_reason = self._parse_error_details(resp)

            if self._should_retry(resp.status_code, attempt, max_attempts):
                sleep_s = self._get_retry_sleep(resp, backoff_seconds)
                time.sleep(max(0.0, sleep_s))
                backoff_seconds *= 2.0
                continue

            raise OandaApiError(
                resp.status_code,
                self._build_error_message(resp, error_message),
                error_message=error_message,
                reject_reason=reject_reason,
                response_text=resp.text,
            )
        return None

    def get_candles(
        self,
        instrument: str,
        *,
        granularity: str = "M5",
        count: int = 500,
        price: str = "M",
        from_time: Optional[str] = None,
        to_time: Optional[str] = None,
        smooth: Optional[bool] = None,
        include_first: Optional[bool] = None,
    ) -> Any:
        params: Dict[str, Any] = {
            "granularity": granularity,
            "count": int(count),
            "price": price,
        }
        if from_time:
            params["from"] = str(from_time)
        if to_time:
            params["to"] = str(to_time)
        if smooth is not None:
            params["smooth"] = bool(smooth)
        if include_first is not None:
            params["includeFirst"] = bool(include_first)
        return self._request(
            "GET",
            f"/instruments/{instrument}/candles",
            params=params,
        )

    def get_account_summary(self) -> Any:
        return self._request("GET", f"/accounts/{self._config.account_id}/summary")

    def get_open_positions(self) -> Any:
        """List currently open positions for the account."""
        return self._request("GET", f"/accounts/{self._config.account_id}/openPositions")

    def get_pricing(self, *, instruments: str) -> Any:
        """Get real-time pricing for one or more instruments.

        instruments: comma-separated list, e.g. "EUR_USD" or "EUR_USD,USD_JPY".
        """
        return self._request(
            "GET",
            f"/accounts/{self._config.account_id}/pricing",
            params={"instruments": instruments},
        )

    def get_order_book(self, instrument: str) -> Any:
        """Fetch order book snapshot (pending order distribution).

        Returns price-bucketed distribution of pending orders.
        Free on OANDA practice accounts.
        """
        return self._request("GET", f"/instruments/{instrument}/orderBook")

    def get_position_book(self, instrument: str) -> Any:
        """Fetch position book snapshot (trader long/short distribution).

        Returns price-bucketed distribution of open positions.
        Free on OANDA practice accounts.
        """
        return self._request("GET", f"/instruments/{instrument}/positionBook")

    def _extract_first_price(self, price_data: Dict[str, Any], side_key: str) -> Optional[float]:
        """Extract the first price from a bid/ask array."""
        side = price_data.get(side_key) or []
        if not isinstance(side, list) or not side:
            return None
        first_entry = side[0]
        if not isinstance(first_entry, dict):
            return None
        try:
            return float(first_entry.get("price"))
        except (TypeError, ValueError):
            return None

    def _extract_closeout_price(self, price_data: Dict[str, Any], key: str) -> Optional[float]:
        """Extract a closeout price as fallback."""
        try:
            return float(price_data.get(key))
        except (TypeError, ValueError):
            return None

    def _get_bid_ask(self, price_data: Dict[str, Any]) -> tuple[Optional[float], Optional[float]]:
        """Extract bid and ask prices from price data with fallback to closeout prices."""
        bid = self._extract_first_price(price_data, "bids")
        ask = self._extract_first_price(price_data, "asks")

        if bid is None:
            bid = self._extract_closeout_price(price_data, "closeoutBid")
        if ask is None:
            ask = self._extract_closeout_price(price_data, "closeoutAsk")

        return bid, ask

    def get_price_quote(self, *, instrument: str) -> Dict[str, float]:
        """Return best bid/ask for a single instrument (floats).

        Fail-closed by raising ValueError if a usable quote can't be extracted.
        """
        payload = self.get_pricing(instruments=instrument)
        prices = payload.get("prices") or []
        if not prices or not isinstance(prices, list):
            raise ValueError("No prices in OANDA pricing response")

        p0 = prices[0] if isinstance(prices[0], dict) else None
        if not p0:
            raise ValueError("Malformed OANDA pricing response")

        bid, ask = self._get_bid_ask(p0)

        if bid is None or ask is None or bid <= 0 or ask <= 0:
            raise ValueError("Missing bid/ask in OANDA pricing response")

        return {"bid": float(bid), "ask": float(ask)}

    # Allow callers to store the last trade id created via create_market_order for convenience.
    _last_trade_id: str | None = None

    def close_position(
        self,
        *,
        instrument: str,
        long_units: str = "ALL",
        short_units: str = "ALL",
    ) -> Any:
        """Close an open position (long/short) for an instrument.

        OANDA supports closing either side or both. We default to closing both.
        """
        payload: Dict[str, Any] = {"longUnits": long_units, "shortUnits": short_units}
        return self._request(
            "PUT",
            f"/accounts/{self._config.account_id}/positions/{instrument}/close",
            json=payload,
        )

    def close_trade(
        self,
        *,
        trade_id: str,
        ensemble: Optional[Any] = None,  # ModularEnsembleInference for drift tracking
        prediction: float = 0.5,
        confidence: float = 50.0,
        features: Optional[Any] = None,  # np.ndarray
    ) -> Any:
        """Close a specific trade by trade id (full close).

        Uses the v20 endpoint: PUT /accounts/{accountID}/trades/{tradeID}/close
        Also records trade outcome for online learning and drift detection.

        Args:
            trade_id: OANDA trade ID to close
            ensemble: Optional ModularEnsembleInference instance for drift tracking
            prediction: Model prediction at trade entry (for drift detection)
            confidence: Ridge confidence at trade entry (for drift detection)
            features: Feature array at trade entry (for drift detection)
        """
        # First, get trade details before closing (for online learning)
        trade_details = None
        try:
            trade_details = self._request(
                "GET",
                f"/accounts/{self._config.account_id}/trades/{trade_id}",
            )
        except Exception:
            pass

        # Close the trade
        payload: Dict[str, Any] = {}
        result = self._request(
            "PUT",
            f"/accounts/{self._config.account_id}/trades/{trade_id}/close",
            json=payload,
        )

        # Record outcome for online learning with drift detection
        try:
            self._record_trade_outcome_for_learning(
                trade_id,
                trade_details,
                result,
                ensemble=ensemble,
                prediction=prediction,
                confidence=confidence,
                features=features,
            )
        except Exception:
            pass  # Don't fail the close if learning recording fails

        return result

    def _record_trade_outcome_for_learning(
        self,
        trade_id: str,
        trade_details: Optional[Dict[str, Any]],
        close_result: Dict[str, Any],
        ensemble: Optional[Any] = None,  # ModularEnsembleInference instance
        prediction: float = 0.5,
        confidence: float = 50.0,
        features: Optional[Any] = None,  # np.ndarray
    ) -> None:
        """Record trade outcome for online learning system.

        This enables the adaptive learning capability where the system
        learns from trade outcomes to improve future reasoning.

        If an ensemble instance is provided, uses its record_trade_result method
        for proper drift detection integration. Otherwise falls back to creating
        a new MarketIntelligence instance (loses state).

        Args:
            trade_id: OANDA trade ID
            trade_details: Trade details from OANDA API
            close_result: Close result from OANDA API
            ensemble: Optional ModularEnsembleInference instance for drift tracking
            prediction: Model prediction probability at entry
            confidence: Ridge confidence score at entry
            features: Feature array at entry for drift detection
        """
        try:
            # Extract trade info
            trade = (trade_details or {}).get('trade', {})
            close_tx = (close_result or {}).get('orderFillTransaction', {})

            instrument = trade.get('instrument', '')
            if not instrument:
                return

            # Calculate P&L
            realized_pl = float(close_tx.get('pl', 0) or trade.get('realizedPL', 0))
            abs(int(trade.get('currentUnits', 0) or trade.get('initialUnits', 0)))

            # Determine direction
            units_signed = int(trade.get('currentUnits', 0) or trade.get('initialUnits', 0))
            direction = 'long' if units_signed > 0 else 'short'

            # Calculate pips
            open_price = float(trade.get('price', 0))
            close_price = float(close_tx.get('price', 0))
            if open_price > 0 and close_price > 0:
                is_jpy = instrument.endswith('_JPY')
                pip_mult = 100 if is_jpy else 10000
                if direction == 'long':
                    pips = (close_price - open_price) * pip_mult
                else:
                    pips = (open_price - close_price) * pip_mult
            else:
                pips = realized_pl * 10  # Rough estimate

            exit_reason = "manual"
            close_reason = str(close_tx.get("reason", "") or "").upper()
            if "TAKE_PROFIT" in close_reason:
                exit_reason = "tp"
            elif "STOP_LOSS" in close_reason:
                exit_reason = "sl"
            elif "TRAILING_STOP" in close_reason:
                exit_reason = "ts"

            open_time = trade.get("openTime")
            close_time = close_tx.get("time") or trade.get("closeTime")
            entry_session = _session_label_from_iso(open_time)
            exit_session = _session_label_from_iso(close_time)

            context_metadata = {
                "exit_reason": exit_reason,
                "close_reason": close_reason or None,
                "open_time": open_time,
                "close_time": close_time,
                "entry_session": entry_session,
                "session_label": entry_session,
                "exit_session": exit_session,
                "units": abs(units_signed),
            }

            # PREFERRED: Use ensemble's record_trade_result for proper drift detection
            if ensemble is not None and hasattr(ensemble, 'record_trade_result'):
                drift_result = ensemble.record_trade_result(
                    trade_id=trade_id,
                    instrument=instrument,
                    direction=direction,
                    entry_price=open_price,
                    exit_price=close_price,
                    pnl_pips=pips,
                    prediction=prediction,
                    confidence=confidence,
                    features=features,
                    context_metadata=context_metadata,
                )

                # Log drift detection result
                if drift_result and drift_result.get('drift_detected'):
                    import logging
                    logger = logging.getLogger(__name__)
                    logger.warning(
                        f"📊 Drift detected: {drift_result.get('reason')} "
                        f"→ {drift_result.get('recommendation')}"
                    )
                return

            # FALLBACK: Create new MarketIntelligence instance (loses state)
            from market_intelligence import MarketIntelligence
            intel = MarketIntelligence(enable_online_learning=True)
            if intel.online_learner:
                # Legacy call - doesn't support full drift detection
                from datetime import datetime
                import numpy as np
                intel.record_trade_outcome(
                    trade_id=trade_id,
                    instrument=instrument,
                    direction=1 if direction == 'long' else 0,
                    entry_time=datetime.utcnow(),
                    exit_time=datetime.utcnow(),
                    entry_price=open_price,
                    exit_price=close_price,
                    pnl_pips=pips,
                    features=features if features is not None else np.zeros(10),
                    prediction=prediction,
                    confidence=confidence,
                    context_metadata=context_metadata,
                )

        except ImportError:
            pass  # market_intelligence not available
        except Exception:
            pass  # Silently fail - don't disrupt trading

    def create_market_order(
        self,
        *,
        instrument: str,
        units: int,
        stop_loss_price: Optional[float] = None,
        take_profit_price: Optional[float] = None,
        trailing_stop_distance: Optional[float] = None,
        price_bound: Optional[float] = None,
        client_order_id: Optional[str] = None,
        client_tag: str = "ml_engine_paper",
        position_fill: str = "DEFAULT",
    ) -> Any:
        # Client order ID helps with idempotency/deduping in client logs.
        # (OANDA supports clientExtensions broadly; client order ids are supported in v20 models.)
        if not client_order_id:
            client_order_id = f"mleng-{uuid.uuid4().hex[:16]}"

        order: Dict[str, Any] = {
            "type": "MARKET",
            "instrument": instrument,
            "units": str(int(units)),
            "timeInForce": "IOC",
            "positionFill": str(position_fill).upper() if position_fill else "DEFAULT",
            "clientOrderID": client_order_id,
            "clientExtensions": {"tag": client_tag},
        }
        if order["positionFill"] not in {"DEFAULT", "OPEN_ONLY", "REDUCE_FIRST", "REDUCE_ONLY"}:
            order["positionFill"] = "DEFAULT"

        # Optional price bound (single bound supported by v20) to limit worst-case execution.
        if price_bound is not None:
            order["priceBound"] = _format_price(instrument, float(price_bound))

        if stop_loss_price is not None:
            order["stopLossOnFill"] = {"price": _format_price(instrument, float(stop_loss_price))}
        if take_profit_price is not None:
            order["takeProfitOnFill"] = {"price": _format_price(instrument, float(take_profit_price))}
        if trailing_stop_distance is not None:
            # OANDA trailing stop uses distance in price units
            decimals = 3 if str(instrument).endswith("_JPY") else 5
            order["trailingStopLossOnFill"] = {"distance": f"{float(trailing_stop_distance):.{decimals}f}"}

        result = self._request(
            "POST",
            f"/accounts/{self._config.account_id}/orders",
            json={"order": order},
        )

        # Try to capture created trade id(s) (if the order was filled) so callers
        # can target the specific trade for precise auto-close. Store on the
        # client instance as `_last_trade_id` for convenience.
        try:
            tx = (result or {}).get("orderFillTransaction") or (result or {}).get("orderCreateTransaction") or {}
            trade_id = None
            if isinstance(tx, dict):
                to = tx.get("tradeOpened")
                if isinstance(to, dict):
                    trade_id = to.get("tradeID") or to.get("id")
                tro = tx.get("tradesOpened")
                if trade_id is None and isinstance(tro, list) and len(tro) > 0:
                    trade_id = tro[0].get("tradeID") or tro[0].get("id")
            if trade_id:
                try:
                    self._last_trade_id = str(trade_id)
                except Exception:
                    pass
        except Exception:
            pass

        return result

    def get_trades(
        self,
        *,
        state: str = "ALL",
        count: int = 500,
        instrument: Optional[str] = None,
    ) -> Any:
        """Get trades for the account.

        Args:
            state: Trade state filter: OPEN, CLOSED, CLOSE_WHEN_TRADEABLE, ALL
            count: Maximum number of trades to return (max 500)
            instrument: Optional instrument filter (e.g., EUR_USD)

        Returns:
            Dict with 'trades' list containing trade details
        """
        params: Dict[str, Any] = {"state": state, "count": min(count, 500)}
        if instrument:
            params["instrument"] = instrument
        return self._request(
            "GET",
            f"/accounts/{self._config.account_id}/trades",
            params=params,
        )

    def get_transactions(
        self,
        *,
        from_time: Optional[str] = None,
        to_time: Optional[str] = None,
        type_filter: Optional[str] = None,
    ) -> Any:
        """Get transactions for the account.

        Args:
            from_time: Start time (RFC3339 format)
            to_time: End time (RFC3339 format)
            type_filter: Transaction type filter (e.g., ORDER_FILL)

        Returns:
            Dict with transaction pages
        """
        params: Dict[str, Any] = {}
        if from_time:
            params["from"] = from_time
        if to_time:
            params["to"] = to_time
        if type_filter:
            params["type"] = type_filter
        return self._request(
            "GET",
            f"/accounts/{self._config.account_id}/transactions",
            params=params,
        )
# — Raynergy-svg —
