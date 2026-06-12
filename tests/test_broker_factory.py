"""Tests for broker factory module.

Tests the create_broker() factory function and broker client instantiation.
"""

from __future__ import annotations

import importlib.util

import pytest

from src.brokers.factory import create_broker
from src.brokers.ibkr import IBKRBroker
from src.brokers.oanda import OandaBroker

# ib_async is an optional dependency (only needed when IBKR is live).
# Without it, IBKRBroker.connect() raises a deliberate guidance
# ImportError (src/brokers/ibkr.py) BEFORE any connection attempt;
# with it installed but no Gateway running, connect() raises
# ConnectionError. Tests pin whichever failure mode this env produces.
_HAS_IB_ASYNC = importlib.util.find_spec("ib_async") is not None
_IBKR_NO_GATEWAY_ERROR = ConnectionError if _HAS_IB_ASYNC else ImportError


class TestBrokerFactory:
    """Test suite for broker factory function."""

    def test_create_broker_oanda_valid(self) -> None:
        """Test creating an OANDA broker returns the right client type.

        OandaBroker.connect() is a documented no-op (REST API —
        authentication happens per-request), so the factory succeeds with
        fake credentials and never touches the network. We assert the
        factory creates the right type, which was always this test's
        stated intent.
        """
        client = create_broker(
            broker_type="oanda",
            oanda_account_id="001-001-123456-001",
            oanda_api_key="fake_key_for_testing",
            oanda_environment="practice",
        )
        assert isinstance(client, OandaBroker)

    def test_create_broker_oanda_missing_account_id(self) -> None:
        """Test that OANDA broker requires account ID."""
        with pytest.raises(ValueError, match="oanda broker requires"):
            create_broker(
                broker_type="oanda",
                oanda_account_id=None,
                oanda_api_key="fake_key",
                oanda_environment="practice",
            )

    def test_create_broker_oanda_missing_api_key(self) -> None:
        """Test that OANDA broker requires API key."""
        with pytest.raises(ValueError, match="oanda broker requires"):
            create_broker(
                broker_type="oanda",
                oanda_account_id="001-001-123456-001",
                oanda_api_key=None,
                oanda_environment="practice",
            )

    def test_create_broker_ibkr_connection_error(self) -> None:
        """Test creating an IBKR broker raises ConnectionError when Gateway not running.

        IBKRBroker is fully implemented but needs IB Gateway running.
        Without Gateway, connect() raises ConnectionError; without the
        optional ib_async package, the deliberate guidance ImportError.
        """
        with pytest.raises(_IBKR_NO_GATEWAY_ERROR):
            create_broker(
                broker_type="ibkr",
                ibkr_host="127.0.0.1",
                ibkr_port=7497,
                ibkr_client_id=1,
            )

    def test_create_broker_invalid_type(self) -> None:
        """Test that invalid broker type raises ValueError."""
        with pytest.raises(ValueError, match="Unsupported broker type"):
            create_broker(broker_type="invalid_broker")

    def test_create_broker_case_insensitive(self) -> None:
        """Test that broker type is case-insensitive."""
        # Should accept uppercase and lowercase
        with pytest.raises(ValueError, match="oanda broker requires"):
            create_broker(
                broker_type="OANDA",
                oanda_account_id=None,
                oanda_api_key="fake",
            )

        with pytest.raises(_IBKR_NO_GATEWAY_ERROR):
            create_broker(broker_type="IBKR")


class TestIBKRBrokerReal:
    """Test suite for real IBKR broker implementation (no Gateway required)."""

    def test_ibkr_broker_instantiation(self) -> None:
        """Test that IBKR broker can be instantiated with default parameters."""
        broker = IBKRBroker()
        assert broker.host == "127.0.0.1"
        assert broker.port == 7497
        assert broker.client_id == 1
        assert broker._connected is False

    def test_ibkr_broker_instantiation_custom_params(self) -> None:
        """Test IBKR broker instantiation with custom parameters."""
        broker = IBKRBroker(
            host="192.168.1.100",
            port=4002,
            client_id=42,
        )
        assert broker.host == "192.168.1.100"
        assert broker.port == 4002
        assert broker.client_id == 42

    def test_ibkr_broker_connect_raises_without_gateway(self) -> None:
        """Test that IBKR broker connect() raises without Gateway.

        ConnectionError with ib_async installed; the deliberate guidance
        ImportError without it (src/brokers/ibkr.py lazy import).
        """
        broker = IBKRBroker()
        with pytest.raises(_IBKR_NO_GATEWAY_ERROR):
            broker.connect()

    def test_ibkr_broker_disconnect(self) -> None:
        """Test that IBKR broker disconnect() is safe when not connected."""
        broker = IBKRBroker()
        broker.disconnect()  # Should not raise
        assert broker._connected is False

    def test_ibkr_broker_repr(self) -> None:
        """Test IBKR broker string representation."""
        broker = IBKRBroker(
            host="example.com",
            port=4002,
            client_id=5,
        )
        repr_str = repr(broker)
        assert "IBKRBroker" in repr_str
        assert "example.com" in repr_str
        assert "4002" in repr_str
        assert "client_id=5" in repr_str


class TestBrokerFactoryEdgeCases:
    """Test edge cases and error conditions."""

    def test_create_broker_whitespace_stripping(self) -> None:
        """Test that broker type handles whitespace gracefully."""
        with pytest.raises(ValueError, match="oanda broker requires"):
            create_broker(
                broker_type="  oanda  ",
                oanda_account_id=None,
                oanda_api_key="fake",
            )

    def test_create_broker_mixed_case(self) -> None:
        """Test mixed case broker types."""
        with pytest.raises(ValueError, match="oanda broker requires"):
            create_broker(
                broker_type="OaNdA",
                oanda_account_id=None,
                oanda_api_key="fake",
            )

        with pytest.raises(_IBKR_NO_GATEWAY_ERROR):
            create_broker(broker_type="IbKr")

    def test_ibkr_broker_defaults_are_correct(self) -> None:
        """Test that IBKR defaults match documentation."""
        broker = IBKRBroker()
        # Default localhost connection
        assert broker.host == "127.0.0.1"
        # Default TWS port (7497 for paper, 4002 for live)
        assert broker.port == 7497
        # Default client ID
        assert broker.client_id == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
