"""Unit tests for text-derived market features."""

import csv
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from data_loader import MarketDataLoader
from text_features import simple_sentiment_score


class TestSimpleSentimentScore(unittest.TestCase):
    def test_empty_text(self):
        self.assertEqual(simple_sentiment_score(None), 0.0)
        self.assertEqual(simple_sentiment_score(""), 0.0)

    def test_positive_text(self):
        score = simple_sentiment_score("Company beats earnings, strong growth")
        self.assertGreater(score, 0.0)

    def test_negative_text(self):
        score = simple_sentiment_score("Company misses earnings, weak outlook")
        self.assertLess(score, 0.0)

    def test_negation_flips(self):
        score = simple_sentiment_score("not bullish")
        self.assertLess(score, 0.0)


class TestTextCsvMerge(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.tmp_path = Path(self.tmp.name)

        self.config = {
            "text": {"enabled": True},
            "data": {"sequence_length": 3},
        }
        self.loader = MarketDataLoader(self.config)

        # Small synthetic market df
        dates = pd.date_range("2024-01-01", periods=10, freq="D")
        self.market_df = pd.DataFrame(
            {
                "open": range(10),
                "high": range(10),
                "low": range(10),
                "close": range(10),
                "volume": [100] * 10,
            },
            index=dates,
        )

    def tearDown(self):
        self.tmp.cleanup()

    def test_load_text_csv_and_merge(self):
        text_csv = self.tmp_path / "text.csv"
        with open(text_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["date", "text"])
            writer.writeheader()
            writer.writerow({"date": "2024-01-02", "text": "bullish upgrade"})
            writer.writerow({"date": "2024-01-02", "text": "strong growth"})
            writer.writerow({"date": "2024-01-05", "text": "misses earnings"})

        text_df = self.loader.load_text_csv(str(text_csv))
        self.assertIn("text_sentiment", text_df.columns)
        self.assertIn("text_count", text_df.columns)

        merged = self.loader.merge_text_features(self.market_df, text_df)
        self.assertIn("text_sentiment", merged.columns)
        self.assertIn("text_count", merged.columns)

        # Missing days should be 0
        self.assertEqual(float(merged.loc["2024-01-03", "text_count"]), 0.0)


if __name__ == "__main__":
    unittest.main()
