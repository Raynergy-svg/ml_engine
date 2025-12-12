"""Minimal OANDA v20 REST client (practice).

This module intentionally targets the PRACTICE environment only.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import os
import requests


PRACTICE_API_URL = "https://api-fxpractice.oanda.com/v3"


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
            }
        )

    @classmethod
    def from_env(cls) -> "OandaPracticeClient":
        api_token = os.getenv("OANDA_API_TOKEN") or os.getenv("OANDA_API_KEY")
        account_id = os.getenv("OANDA_ACCOUNT_ID")
        if not api_token or not account_id:
            raise EnvironmentError(
                "Missing OANDA env vars. Set OANDA_API_TOKEN (or OANDA_API_KEY) and OANDA_ACCOUNT_ID."
            )
        return cls(OandaPracticeConfig(api_token=api_token, account_id=account_id))

    def _request(self, method: str, path: str, *, params: Optional[Dict[str, Any]] = None, json: Any = None) -> Any:
        url = PRACTICE_API_URL + path
        resp = self._session.request(
            method,
            url,
            params=params,
            json=json,
            timeout=self._config.timeout_seconds,
        )
        if resp.status_code >= 400:
            raise RuntimeError(f"OANDA API error {resp.status_code}: {resp.text}")
        return resp.json()

    def get_candles(
        self,
        instrument: str,
        *,
        granularity: str = "M5",
        count: int = 500,
        price: str = "M",
    ) -> Any:
        return self._request(
            "GET",
            f"/instruments/{instrument}/candles",
            params={"granularity": granularity, "count": int(count), "price": price},
        )

    def get_account_summary(self) -> Any:
        return self._request("GET", f"/accounts/{self._config.account_id}/summary")

    def create_market_order(
        self,
        *,
        instrument: str,
        units: int,
        stop_loss_price: Optional[float] = None,
        take_profit_price: Optional[float] = None,
        client_tag: str = "ml_engine_paper",
    ) -> Any:
        order: Dict[str, Any] = {
            "type": "MARKET",
            "instrument": instrument,
            "units": str(int(units)),
            "timeInForce": "FOK",
            "positionFill": "DEFAULT",
            "clientExtensions": {"tag": client_tag},
        }

        if stop_loss_price is not None:
            order["stopLossOnFill"] = {"price": f"{float(stop_loss_price):.5f}"}
        if take_profit_price is not None:
            order["takeProfitOnFill"] = {"price": f"{float(take_profit_price):.5f}"}

        return self._request(
            "POST",
            f"/accounts/{self._config.account_id}/orders",
            json={"order": order},
        )
