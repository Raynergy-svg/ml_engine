"""Unit tests for forex paper-trading helpers."""

import unittest

import pandas as pd

from ml_engine.fx_paper import (
    candles_to_ohlcv_df,
    atr,
    pip_size,
    position_size_units,
    setup_signal,
)


class TestCandlesToOHLCV(unittest.TestCase):
    def test_converts_complete_candles(self):
        resp = {
            "candles": [
                {
                    "complete": True,
                    "time": "2025-01-01T00:00:00Z",
                    "mid": {"o": "1.1000", "h": "1.1010", "l": "1.0990", "c": "1.1005"},
                    "volume": 123,
                },
                {
                    "complete": True,
                    "time": "2025-01-01T00:05:00Z",
                    "mid": {"o": "1.1005", "h": "1.1020", "l": "1.1000", "c": "1.1015"},
                    "volume": 321,
                },
            ]
        }

        df = candles_to_ohlcv_df(resp)
        self.assertEqual(list(df.columns), ["time", "open", "high", "low", "close", "volume"])
        self.assertEqual(len(df), 2)
        self.assertAlmostEqual(float(df["open"].iloc[0]), 1.1, places=6)

    def test_skips_incomplete_candles(self):
        resp = {
            "candles": [
                {
                    "complete": False,
                    "time": "2025-01-01T00:00:00Z",
                    "mid": {"o": "1.1000", "h": "1.1010", "l": "1.0990", "c": "1.1005"},
                    "volume": 123,
                },
                {
                    "complete": True,
                    "time": "2025-01-01T00:05:00Z",
                    "mid": {"o": "1.1005", "h": "1.1020", "l": "1.1000", "c": "1.1015"},
                    "volume": 321,
                },
            ]
        }

        df = candles_to_ohlcv_df(resp)
        self.assertEqual(len(df), 1)
        self.assertEqual(df["time"].iloc[0], "2025-01-01T00:05:00Z")


class TestIndicatorsAndSizing(unittest.TestCase):
    def _df_from_close(self, close):
        # Make simple OHLC around close
        close = pd.Series(close, dtype=float)
        return pd.DataFrame(
            {
                "high": close + 0.0005,
                "low": close - 0.0005,
                "close": close,
            }
        )

    def test_atr_positive_finite(self):
        df = self._df_from_close([1.1000 + i * 0.0001 for i in range(60)])
        value = atr(df, period=14)
        self.assertGreater(value, 0)

    def test_pip_size_heuristic(self):
        self.assertEqual(pip_size("EUR_USD"), 0.0001)
        self.assertEqual(pip_size("USD_JPY"), 0.01)

    def test_position_size_units_basic(self):
        units = position_size_units(
            instrument="EUR_USD",
            equity=10_000.0,
            risk_per_trade_pct=0.01,
            stop_distance_price=0.002,
            price=1.1,
        )
        self.assertEqual(units, 50_000)

    def test_setup_signal_buy_and_sell(self):
        # Construct a series that is above EMA with moderate RSI.
        close_buy = []
        price = 1.1000
        for i in range(80):
            # net upward drift with periodic dips to keep RSI moderate
            if i % 5 == 0:
                price -= 0.00005
            else:
                price += 0.00010
            close_buy.append(price)

        df_buy = self._df_from_close(close_buy)
        sig_buy = setup_signal(df_buy)
        self.assertIn(sig_buy, {"buy", "hold"})

        # Construct a series that is below EMA with moderate RSI.
        close_sell = []
        price = 1.2000
        for i in range(80):
            if i % 5 == 0:
                price += 0.00005
            else:
                price -= 0.00010
            close_sell.append(price)

        df_sell = self._df_from_close(close_sell)
        sig_sell = setup_signal(df_sell)
        self.assertIn(sig_sell, {"sell", "hold"})


if __name__ == "__main__":
    unittest.main()
