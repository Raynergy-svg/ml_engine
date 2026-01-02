"""Unit tests for forex paper-trading helpers."""

import unittest

import pandas as pd

import numpy as np

from ml_engine.fx_paper import (
    candles_to_ohlcv_df,
    atr,
    pip_size,
    position_size_units,
    setup_signal,
    simulate_tp_sl_outcome,
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
        self.assertEqual(list(df.columns), ["time", "open", "high", "low", "close", "volume", "close_ema_14"])
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


class TestTpSlSimulation(unittest.TestCase):
    def test_buy_tp_hit(self):
        # Next-candle entry:
        # - signal_index=0 (signal at close[0])
        # - enter at open[1] (+spread)
        # - scan starts at index 2
        df = pd.DataFrame(
            {
                "open": [100.00, 100.50, 101.00],
                "close": [100.00, 100.80, 101.20],
                "high": [100.50, 101.00, 101.50],
                "low": [99.50, 100.00, 100.50],
            }
        )
        r = simulate_tp_sl_outcome(
            df,
            instrument="USD_JPY",
            signal_index=0,
            direction="long",
            stop_loss_pips=20,
            take_profit_pips=60,
            spread_pips=0.0,
            max_horizon_candles=1,
            tie_break="sl",
        )
        self.assertEqual(r.outcome, 1)
        self.assertEqual(r.hit_type, "tp")
        self.assertTrue(r.is_win)
        self.assertEqual(r.to_label(), 1)

    def test_buy_sl_hit(self):
        df = pd.DataFrame(
            {
                "open": [100.00, 100.50, 99.80],
                "close": [100.00, 100.20, 99.70],
                "high": [100.50, 101.00, 100.00],
                "low": [99.50, 99.00, 99.50],
            }
        )
        r = simulate_tp_sl_outcome(
            df,
            instrument="USD_JPY",
            signal_index=0,
            direction="long",
            stop_loss_pips=20,
            take_profit_pips=60,
            spread_pips=0.0,
            max_horizon_candles=1,
            tie_break="sl",
        )
        self.assertEqual(r.outcome, 0)
        self.assertEqual(r.hit_type, "sl")
        self.assertFalse(r.is_win)
        self.assertEqual(r.to_label(), 0)

    def test_intrabar_tie_break_is_worst_case_sl(self):
        # Same candle crosses both TP and SL; calibration training uses worst-case SL.
        df = pd.DataFrame(
            {
                "open": [100.00, 100.50, 100.40],
                "close": [100.00, 100.50, 100.50],
                "high": [100.50, 101.00, 102.00],
                "low": [99.50, 100.00, 99.00],
            }
        )
        r = simulate_tp_sl_outcome(
            df,
            instrument="USD_JPY",
            signal_index=0,
            direction="long",
            stop_loss_pips=20,
            take_profit_pips=60,
            spread_pips=0.0,
            max_horizon_candles=1,
            tie_break="sl",
        )
        self.assertEqual(r.outcome, 0)
        self.assertEqual(r.hit_type, "sl")

    def test_missing_open_raises(self):
        df = pd.DataFrame({"close": [100.0, 101.0, 102.0], "high": [100.0, 101.0, 102.0], "low": [100.0, 101.0, 102.0]})
        with self.assertRaisesRegex(ValueError, "requires 'open' column"):
            simulate_tp_sl_outcome(
                df,
                instrument="USD_JPY",
                signal_index=0,
                direction="long",
                stop_loss_pips=20,
                take_profit_pips=60,
                spread_pips=2.0,
                max_horizon_candles=1,
                tie_break="sl",
            )

    def test_insufficient_data_raises(self):
        df = pd.DataFrame({"open": [100.0, 101.0], "high": [100.0, 101.0], "low": [100.0, 101.0], "close": [100.0, 101.0]})
        with self.assertRaisesRegex(ValueError, "Insufficient data"):
            simulate_tp_sl_outcome(
                df,
                instrument="USD_JPY",
                signal_index=0,
                direction="long",
                stop_loss_pips=20,
                take_profit_pips=60,
                spread_pips=2.0,
                max_horizon_candles=1,
                tie_break="sl",
            )

    def test_invalid_spread_raises(self):
        df = pd.DataFrame(
            {
                "open": [100.0, 100.5, 101.0],
                "high": [100.5, 101.0, 101.5],
                "low": [99.5, 100.0, 100.5],
                "close": [100.0, 100.8, 101.2],
            }
        )
        with self.assertRaisesRegex(ValueError, "Invalid spread_pips"):
            simulate_tp_sl_outcome(
                df,
                instrument="USD_JPY",
                signal_index=0,
                direction="long",
                stop_loss_pips=20,
                take_profit_pips=60,
                spread_pips=100.0,  # Exceeds 50 pip limit
                max_horizon_candles=1,
                tie_break="sl",
            )

    def test_nan_open_raises(self):
        df = pd.DataFrame(
            {
                "open": [100.0, np.nan, 101.0],
                "high": [100.0, 101.0, 101.0],
                "low": [100.0, 99.0, 100.0],
                "close": [100.0, 100.0, 101.0],
            }
        )
        with self.assertRaisesRegex(ValueError, "Invalid entry open price"):
            simulate_tp_sl_outcome(
                df,
                instrument="USD_JPY",
                signal_index=0,
                direction="long",
                stop_loss_pips=20,
                take_profit_pips=60,
                spread_pips=2.0,
                max_horizon_candles=1,
                tie_break="sl",
            )


if __name__ == "__main__":
    unittest.main()
