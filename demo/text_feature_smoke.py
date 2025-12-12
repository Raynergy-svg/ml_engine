"""Smoke demo: merge dated text into market data and show feature shapes.

This does not do a full training run; it proves the text path is wired:
- load OHLCV CSV
- load dated text CSV
- merge -> preprocess -> show resulting feature count

Usage:
  python demo/text_feature_smoke.py \
    --market market_data/TSLA_data.csv \
    --text demo/sample_text.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))

    from data_loader import MarketDataLoader
    from utils import load_config

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--market", required=True)
    parser.add_argument("--text", required=True)
    parser.add_argument("--text-date-col", default="date")
    parser.add_argument("--text-text-col", default="text")
    args = parser.parse_args()

    config = load_config(args.config)
    loader = MarketDataLoader(config)

    market_df = loader.load_csv(args.market)
    text_df = loader.load_text_csv(
        args.text,
        date_column=args.text_date_col,
        text_column=args.text_text_col,
    )
    merged = loader.merge_text_features(market_df, text_df)

    x_train, y_train, x_val, y_val, x_test, y_test = loader.preprocess(
        merged,
        add_features=True,
        scaler_type="standard",
        sequence_length=config.get("data", {}).get("sequence_length", 60),
        test_size=0.2,
        validation_size=0.1,
    )

    print("Merged columns contain text features:")
    cols = [c for c in merged.columns if c.startswith("text_")]
    print(cols)
    print("Train/Val/Test shapes:")
    print("x_train", x_train.shape, "y_train", y_train.shape)
    print("x_val", x_val.shape, "y_val", y_val.shape)
    print("x_test", x_test.shape, "y_test", y_test.shape)


if __name__ == "__main__":
    main()
