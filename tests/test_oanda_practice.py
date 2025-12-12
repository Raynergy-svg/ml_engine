"""Unit tests for OANDA practice client (no network)."""

import os
import sys
import unittest
from unittest import mock

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import oanda_practice


class TestOandaPracticeClient(unittest.TestCase):
    def test_from_env_missing_vars_raises(self):
        with mock.patch.dict("os.environ", {}, clear=True):
            with self.assertRaises(EnvironmentError):
                oanda_practice.OandaPracticeClient.from_env()

    def test_get_candles_calls_expected_endpoint(self):
        session = mock.Mock()
        session.headers = {}

        def _fake_request(method, url, params=None, json=None, timeout=None):
            resp = mock.Mock()
            resp.status_code = 200
            resp.json.return_value = {"candles": []}
            return resp

        session.request.side_effect = _fake_request

        with mock.patch.object(oanda_practice.requests, "Session", return_value=session):
            client = oanda_practice.OandaPracticeClient(
                oanda_practice.OandaPracticeConfig(api_token="t", account_id="a")
            )
            client.get_candles("EUR_USD", granularity="M5", count=10)

        session.request.assert_called_once()
        (method, url), kwargs = session.request.call_args
        self.assertEqual(method, "GET")
        self.assertIn("/instruments/EUR_USD/candles", url)
        self.assertEqual(kwargs["params"]["granularity"], "M5")
        self.assertEqual(kwargs["params"]["count"], 10)

    def test_create_market_order_payload_includes_sl_tp(self):
        session = mock.Mock()
        session.headers = {}

        def _fake_request(method, url, params=None, json=None, timeout=None):
            resp = mock.Mock()
            resp.status_code = 200
            resp.json.return_value = {"orderCreateTransaction": {"id": "1"}}
            return resp

        session.request.side_effect = _fake_request

        with mock.patch.object(oanda_practice.requests, "Session", return_value=session):
            client = oanda_practice.OandaPracticeClient(
                oanda_practice.OandaPracticeConfig(api_token="t", account_id="acct")
            )
            client.create_market_order(
                instrument="EUR_USD",
                units=123,
                stop_loss_price=1.23456,
                take_profit_price=1.34567,
            )

        session.request.assert_called_once()
        (method, url), kwargs = session.request.call_args
        self.assertEqual(method, "POST")
        self.assertIn("/accounts/acct/orders", url)
        payload = kwargs["json"]
        self.assertIn("order", payload)
        order = payload["order"]
        self.assertEqual(order["type"], "MARKET")
        self.assertEqual(order["instrument"], "EUR_USD")
        self.assertEqual(order["units"], "123")
        self.assertEqual(order["stopLossOnFill"]["price"], "1.23456")
        self.assertEqual(order["takeProfitOnFill"]["price"], "1.34567")


if __name__ == "__main__":
    unittest.main()
