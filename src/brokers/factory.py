"""Broker factory for creating broker client instances.

Provides create_broker() factory function that instantiates the correct
broker client based on configuration. Currently supports:
- OANDA (v20 REST API) — default
- IBKR (Interactive Brokers) — stub for future implementation
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.brokers.base import BrokerClient

logger = logging.getLogger(__name__)


def create_broker(
    broker_type: str = "oanda",
    oanda_account_id: str | None = None,
    oanda_api_key: str | None = None,
    oanda_environment: str = "practice",
    ibkr_host: str = "127.0.0.1",
    ibkr_port: int = 7497,
    ibkr_client_id: int = 1,
) -> BrokerClient:
    """Factory function to create a broker client instance.

    Args:
        broker_type: Broker type — "oanda" or "ibkr" (case-insensitive).
        oanda_account_id: OANDA account ID (e.g., "001-001-123456-001").
        oanda_api_key: OANDA API key for authentication.
        oanda_environment: "practice" or "live".
        ibkr_host: Interactive Brokers TWS host (default: "127.0.0.1").
        ibkr_port: Interactive Brokers TWS port (default: 7497).
        ibkr_client_id: Interactive Brokers client ID (default: 1).

    Returns:
        BrokerClient: An instantiated broker client (OandaBrokerClient or IBKRBroker).

    Raises:
        ValueError: If broker_type is unsupported or required params are missing.
        Exception: If broker connection or initialization fails.
    """
    broker_type = broker_type.lower().strip()

    if broker_type == "oanda":
        # OANDA integration — uses existing OandaBroker
        from src.brokers.oanda import OandaBroker

        if not oanda_account_id or not oanda_api_key:
            raise ValueError(
                "oanda broker requires oanda_account_id and oanda_api_key. "
                "Set them via --oanda-account-id, --oanda-api-key environment variables, "
                "or in ScannerConfig."
            )

        logger.info(
            f"Creating OANDA broker client: account={oanda_account_id}, "
            f"environment={oanda_environment}"
        )

        client = OandaBroker(
            account_id=oanda_account_id,
            api_key=oanda_api_key,
            environment=oanda_environment,
        )
        client.connect()
        return client

    elif broker_type == "ibkr":
        # Interactive Brokers integration — real implementation
        from src.brokers.ibkr import IBKRBroker

        logger.info(
            f"Creating IBKR broker client: host={ibkr_host}, "
            f"port={ibkr_port}, client_id={ibkr_client_id}"
        )

        client = IBKRBroker(
            host=ibkr_host,
            port=ibkr_port,
            client_id=ibkr_client_id,
        )
        client.connect()
        return client

    else:
        raise ValueError(
            f"Unsupported broker type: {broker_type}. "
            f"Supported types: 'oanda', 'ibkr'."
        )
