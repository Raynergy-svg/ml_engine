"""Minimal OANDA v20 REST client (practice).

This module intentionally targets the PRACTICE environment only.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import time
import uuid

import os
import requests


def _load_project_dotenv() -> None:
    """Best-effort load of `.env.local` / `.env` for local runs.

    This keeps CLI workflows consistent with `unified_talk`/Buddy, and is
    fail-open: missing `python-dotenv` or parse issues should not crash.
    """
    try:
        from pathlib import Path

        try:
            from dotenv import load_dotenv  # type: ignore
        except Exception:
            return

        candidates = []
        try:
            candidates.append(Path.cwd())
        except Exception:
            pass

        try:
            candidates.append(Path(__file__).resolve().parent)
        except Exception:
            pass

        seen: set[str] = set()
        for base in candidates:
            try:
                base = base.resolve()
            except Exception:
                continue
            if str(base) in seen:
                continue
            seen.add(str(base))

            for name in (".env.local", ".env"):
                p = base / name
                if p.exists() and p.is_file():
                    load_dotenv(dotenv_path=str(p), override=False)
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

    @classmethod
    def from_env(cls) -> "OandaPracticeClient":
        _load_project_dotenv()
        api_token = os.getenv("OANDA_API_TOKEN") or os.getenv("OANDA_API_KEY")
        account_id = os.getenv("OANDA_ACCOUNT_ID")
        if not api_token or not account_id:
            raise EnvironmentError(
                "Missing OANDA env vars. Set OANDA_API_TOKEN (or OANDA_API_KEY) and OANDA_ACCOUNT_ID."
            )
        return cls(OandaPracticeConfig(api_token=api_token, account_id=account_id))

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

        # Best-practice-ish defaults:
        # - persistent HTTP connection via requests.Session
        # - retry a small number of times on transient errors and rate limiting
        max_attempts = 3
        backoff_seconds = 0.5

        for attempt in range(1, max_attempts + 1):
            resp = self._session.request(
                method,
                url,
                params=params,
                json=json,
                timeout=timeout,
            )

            if resp.status_code < 400:
                # Some endpoints may return empty bodies; default to {}.
                if not resp.content:
                    return {}
                return resp.json()

            # Parse common OANDA error shapes.
            error_message: Optional[str] = None
            reject_reason: Optional[str] = None
            try:
                err = resp.json()
                if isinstance(err, dict):
                    error_message = err.get("errorMessage")
                    reject_reason = err.get("rejectReason")
            except Exception:
                err = None

            # Retry policy: 429 + 5xx.
            if resp.status_code in {429, 500, 502, 503, 504} and attempt < max_attempts:
                retry_after = resp.headers.get("Retry-After")
                if retry_after:
                    try:
                        sleep_s = float(retry_after)
                    except ValueError:
                        sleep_s = backoff_seconds
                else:
                    sleep_s = backoff_seconds
                time.sleep(max(0.0, sleep_s))
                backoff_seconds *= 2.0
                continue

            msg = f"OANDA API error {resp.status_code}"
            if error_message:
                msg += f": {error_message}"
            elif resp.text:
                msg += f": {resp.text}"
            raise OandaApiError(
                resp.status_code,
                msg,
                error_message=error_message,
                reject_reason=reject_reason,
                response_text=resp.text,
            )

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

        def _first_price(side_key: str) -> Optional[float]:
            side = p0.get(side_key) or []
            if isinstance(side, list) and side and isinstance(side[0], dict):
                try:
                    return float(side[0].get("price"))
                except Exception:
                    return None
            return None

        bid = _first_price("bids")
        ask = _first_price("asks")

        # Fallback to closeout prices if top-of-book arrays are missing.
        if bid is None:
            try:
                bid = float(p0.get("closeoutBid"))
            except Exception:
                bid = None
        if ask is None:
            try:
                ask = float(p0.get("closeoutAsk"))
            except Exception:
                ask = None

        if bid is None or ask is None or bid <= 0 or ask <= 0:
            raise ValueError("Missing bid/ask in OANDA pricing response")

        return {"bid": float(bid), "ask": float(ask)}

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

    def create_market_order(
        self,
        *,
        instrument: str,
        units: int,
        stop_loss_price: Optional[float] = None,
        take_profit_price: Optional[float] = None,
        price_bound: Optional[float] = None,
        client_order_id: Optional[str] = None,
        client_tag: str = "ml_engine_paper",
    ) -> Any:
        # Client order ID helps with idempotency/deduping in client logs.
        # (OANDA supports clientExtensions broadly; client order ids are supported in v20 models.)
        if not client_order_id:
            client_order_id = f"mleng-{uuid.uuid4().hex[:16]}"

        order: Dict[str, Any] = {
            "type": "MARKET",
            "instrument": instrument,
            "units": str(int(units)),
            "timeInForce": "FOK",
            "positionFill": "DEFAULT",
            "clientOrderID": client_order_id,
            "clientExtensions": {"tag": client_tag},
        }

        # Optional price bound (single bound supported by v20) to limit worst-case execution.
        if price_bound is not None:
            order["priceBound"] = _format_price(instrument, float(price_bound))

        if stop_loss_price is not None:
            order["stopLossOnFill"] = {"price": _format_price(instrument, float(stop_loss_price))}
        if take_profit_price is not None:
            order["takeProfitOnFill"] = {"price": _format_price(instrument, float(take_profit_price))}

        return self._request(
            "POST",
            f"/accounts/{self._config.account_id}/orders",
            json={"order": order},
        )
