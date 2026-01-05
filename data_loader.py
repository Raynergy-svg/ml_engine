"""
Enhanced data loader with validation, preprocessing, and feature engineering.
"""

import os
import logging
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, Optional, Dict, Any, List
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.model_selection import train_test_split
import joblib

from feature_engineering import FeatureEngineering
from text_features import hashed_ngram_features, text_feature_summary

logger = logging.getLogger(__name__)


PreprocessResult = Tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]


class DataLoader:
    """Enhanced data loader with comprehensive preprocessing."""

    def __init__(self, config: Dict[str, Any]):
        """Initialize data loader with configuration."""
        self.config = config
        self.scaler = None
        self.feature_engineer = FeatureEngineering(config)
        self.feature_names = []

    def load_csv(self, file_path: str) -> pd.DataFrame:
        """Load data from CSV file with validation."""
        logger.info("Loading data from %s", file_path)

        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Data file not found: {file_path}")

        try:
            df = pd.read_csv(file_path)
            logger.info("Loaded %s rows from %s", len(df), file_path)

            # Validate required columns
            required_cols = ["open", "high", "low", "close", "volume"]
            missing_cols = [
                col
                for col in required_cols
                if col.lower() not in [c.lower() for c in df.columns]
            ]

            if missing_cols:
                raise ValueError(f"Missing required columns: {missing_cols}")

            # Standardize column names
            df.columns = [col.lower() for col in df.columns]

            # Handle date column
            if "date" in df.columns:
                df["date"] = pd.to_datetime(df["date"], utc=True)
                df.set_index("date", inplace=True)
            elif df.index.name and "date" in df.index.name.lower():
                df.index = pd.to_datetime(df.index, utc=True)

            # Sort by date
            df = df.sort_index()

            # Remove duplicates
            df = df[~df.index.duplicated(keep="last")]

            # Validate data quality
            self._validate_data(df)

            return df

        except Exception as e:
            logger.error("Error loading data: %s", e)
            raise

    def load_text_csv(
        self,
        file_path: str,
        date_column: str = "date",
        text_column: str = "text",
    ) -> pd.DataFrame:
        """Load a date-keyed text CSV and convert to numeric text features.

        Expected columns:
        - date_column (default: 'date')
        - text_column (default: 'text')

        Multiple rows per day are supported; they are aggregated to per-day
        features.
        """
        logger.info("Loading text data from %s", file_path)

        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Text data file not found: {file_path}")

        df_text = pd.read_csv(file_path)
        df_text.columns = [c.lower() for c in df_text.columns]
        date_col = date_column.lower()
        text_col = text_column.lower()

        if date_col not in df_text.columns:
            raise ValueError(
                f"Text CSV missing required date column '{date_column}'"
            )
        if text_col not in df_text.columns:
            raise ValueError(
                f"Text CSV missing required text column '{text_column}'"
            )

        df_text[date_col] = pd.to_datetime(df_text[date_col], utc=True)
        df_text[text_col] = df_text[text_col].astype(str)

        # Aggregate to per-day numeric features
        normalized_dates = df_text[date_col].dt.normalize()
        grouped = df_text.groupby(normalized_dates)[text_col].apply(list)

        text_cfg = (self.config or {}).get("text", {})
        use_hash = bool(text_cfg.get("use_hash_ngrams", False))
        hash_dim = int(text_cfg.get("hash_ngram_dim", 64))
        ngram_min = int(text_cfg.get("hash_ngram_min", 1))
        ngram_max = int(text_cfg.get("hash_ngram_max", 2))

        rows = []
        for day, texts in grouped.items():
            summary = text_feature_summary(texts)
            row = {
                "date": day,
                "text_sentiment": summary.sentiment,
                "text_sentiment_std": summary.sentiment_std,
                "text_sentiment_min": summary.sentiment_min,
                "text_sentiment_max": summary.sentiment_max,
                "text_sentiment_abs_mean": summary.sentiment_abs_mean,
                "text_sentiment_nonzero_frac": summary.sentiment_nonzero_frac,
                "text_token_count": summary.token_count,
                "text_char_count": summary.char_count,
                "text_count": int(len(texts)),
            }

            if use_hash:
                vec = hashed_ngram_features(
                    texts,
                    dim=hash_dim,
                    min_n=ngram_min,
                    max_n=ngram_max,
                )
                for i, value in enumerate(vec):
                    row[f"text_hash_{i}"] = float(value)

            rows.append(row)

        features_df = pd.DataFrame(rows).set_index("date").sort_index()
        return features_df

    def merge_text_features(
        self,
        market_df: pd.DataFrame,
        text_features_df: pd.DataFrame,
        how: str = "left",
    ) -> pd.DataFrame:
        """Merge numeric text features into a market DataFrame by date."""
        if market_df is None or market_df.empty:
            raise ValueError("market_df is empty")
        if text_features_df is None or text_features_df.empty:
            logger.warning("text_features_df is empty; skipping merge")
            return market_df

        df = market_df.copy()
        if not isinstance(df.index, pd.DatetimeIndex):
            raise ValueError(
                "market_df must be indexed by datetime for text merge"
            )

        text_df = text_features_df.copy()
        if not isinstance(text_df.index, pd.DatetimeIndex):
            raise ValueError("text_features_df must be indexed by datetime")

        # Normalize to date (no time component) for consistent joins
        df.index = pd.to_datetime(df.index, utc=True).normalize()
        text_df.index = pd.to_datetime(text_df.index, utc=True).normalize()

        merged = df.join(text_df, how=how)

        # Fill missing text features with 0 (no text signal available)
        base_cols = [
            "text_sentiment",
            "text_sentiment_std",
            "text_sentiment_min",
            "text_sentiment_max",
            "text_sentiment_abs_mean",
            "text_sentiment_nonzero_frac",
            "text_token_count",
            "text_char_count",
            "text_count",
        ]
        for col in base_cols:
            if col in merged.columns:
                merged[col] = merged[col].fillna(0)

        hash_cols = [c for c in merged.columns if c.startswith("text_hash_")]
        for col in hash_cols:
            merged[col] = merged[col].fillna(0)

        return merged

    def _validate_data(self, df: pd.DataFrame) -> None:
        """Validate data quality."""
        # Check for missing values
        missing_pct = df.isnull().sum() / len(df) * 100
        if missing_pct.max() > 50:
            logger.warning(
                "High percentage of missing values detected: %s",
                missing_pct[missing_pct > 0],
            )

        # Check for negative values in price columns
        price_cols = ["open", "high", "low", "close"]
        for col in price_cols:
            if col in df.columns and (df[col] < 0).any():
                logger.warning("Negative values found in %s", col)

        # Check for zero volume
        if "volume" in df.columns:
            zero_volume_pct = (df["volume"] == 0).sum() / len(df) * 100
            if zero_volume_pct > 10:
                logger.warning(
                    "%.2f%% of data has zero volume",
                    zero_volume_pct,
                )

        # Check for price consistency (high >= low, etc.)
        if all(col in df.columns for col in ["high", "low"]):
            inconsistent = df["high"] < df["low"]
            if inconsistent.any():
                logger.warning(
                    "Found %s rows where high < low",
                    int(inconsistent.sum()),
                )
                df.loc[inconsistent, ["high", "low"]] = df.loc[
                    inconsistent, ["low", "high"]
                ].values

    def preprocess(
        self,
        df: pd.DataFrame,
        add_features: bool = True,
        scaler_type: str = "standard",
        sequence_length: int = 60,
        test_size: float = 0.2,
        validation_size: float = 0.1,
    ) -> PreprocessResult:
        """Preprocess data with feature engineering and scaling."""
        logger.info("Starting data preprocessing...")

        # Feature engineering
        if add_features:
            df = self.feature_engineer.create_features(df, include_all=True)

        # Remove rows with NaN values
        df = df.dropna()

        if len(df) < sequence_length + 100:
            raise ValueError(
                f"Insufficient data after preprocessing: {len(df)} rows"
            )

        # Prepare features and target
        target_col = "close"
        if target_col not in df.columns:
            raise ValueError(f"Target column '{target_col}' not found")

        # Select numeric columns only
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

        # Remove target from features
        feature_cols = [col for col in numeric_cols if col != target_col]

        x = df[feature_cols].values
        y = df[target_col].values

        logger.info("Features shape: %s, Target shape: %s", x.shape, y.shape)

        # Scale features
        if scaler_type == "standard":
            self.scaler = StandardScaler()
        elif scaler_type == "minmax":
            self.scaler = MinMaxScaler()
        elif scaler_type == "robust":
            self.scaler = RobustScaler()
        else:
            self.scaler = StandardScaler()

        x_scaled = self.scaler.fit_transform(x)

        # Create sequences
        x_seq, y_seq = self._create_sequences(x_scaled, y, sequence_length)

        logger.info(
            "Created sequences: X=%s, y=%s",
            x_seq.shape,
            y_seq.shape,
        )

        # Split into train, validation, and test sets
        # First split: separate test set
        random_state = int(self.config.get("random_seed", 42))
        x_temp, x_test, y_temp, y_test = train_test_split(
            x_seq,
            y_seq,
            test_size=test_size,
            shuffle=False,
            random_state=random_state,
        )

        # Second split: separate validation from train
        val_size_adjusted = validation_size / (1 - test_size)
        x_train, x_val, y_train, y_val = train_test_split(
            x_temp,
            y_temp,
            test_size=val_size_adjusted,
            shuffle=False,
            random_state=random_state,
        )

        logger.info(
            "Train: %s, Val: %s, Test: %s",
            x_train.shape,
            x_val.shape,
            x_test.shape,
        )

        self.feature_names = feature_cols

        return x_train, y_train, x_val, y_val, x_test, y_test

    def _create_sequences(
        self, x: np.ndarray, y: np.ndarray, sequence_length: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Create sequences for time series prediction."""
        x_seq, y_seq = [], []

        for i in range(len(x) - sequence_length):
            x_seq.append(x[i:i + sequence_length])
            y_seq.append(y[i + sequence_length])

        return np.array(x_seq), np.array(y_seq)

    def save_scaler(self, path: str) -> None:
        """Save the fitted scaler."""
        if self.scaler is None:
            logger.warning("No scaler to save")
            return

        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.scaler, path)
        logger.info("Scaler saved to %s", path)

    def load_scaler(self, path: str) -> None:
        """Load a fitted scaler."""
        if not os.path.exists(path):
            raise FileNotFoundError(f"Scaler file not found: {path}")

        self.scaler = joblib.load(path)
        logger.info("Scaler loaded from %s", path)

    def transform_new_data(
        self, df: pd.DataFrame, sequence_length: int = 60
    ) -> np.ndarray:
        """Transform new data using fitted scaler."""
        if self.scaler is None:
            raise ValueError(
                "Scaler not fitted. Call preprocess() first or load a scaler."
            )

        # Apply same preprocessing
        df = self.feature_engineer.create_features(df, include_all=True)
        df = df.dropna()

        # Select same features
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        feature_cols = [col for col in numeric_cols if col != "close"]

        x = df[feature_cols].values
        x_scaled = self.scaler.transform(x)

        # Create sequences
        if len(x_scaled) >= sequence_length:
            x_seq = np.array([x_scaled[-sequence_length:]])
            return x_seq
        else:
            raise ValueError(
                "Insufficient data: need %s, got %s"
                % (sequence_length, len(x_scaled))
            )


class MarketDataLoader(DataLoader):
    """Specialized data loader for market data."""

    def load_multiple_tickers(
        self, data_dir: str, tickers: Optional[List[str]] = None
    ) -> Dict[str, pd.DataFrame]:
        """Load data for multiple tickers."""
        data_dir = Path(data_dir)

        if not data_dir.exists():
            raise FileNotFoundError(f"Data directory not found: {data_dir}")

        # Get all CSV files if tickers not specified
        if tickers is None:
            csv_files = list(data_dir.glob("*.csv"))
            tickers = [f.stem.replace("_data", "") for f in csv_files]

        data_dict = {}
        for ticker in tickers:
            file_path = data_dir / f"{ticker}_data.csv"
            if file_path.exists():
                try:
                    df = self.load_csv(str(file_path))
                    data_dict[ticker] = df
                    logger.info(f"Loaded {ticker}: {len(df)} rows")
                except Exception as e:
                    logger.error(f"Failed to load {ticker}: {e}")
            else:
                logger.warning(f"File not found for {ticker}: {file_path}")

        return data_dict

    def combine_ticker_data(
        self, data_dict: Dict[str, pd.DataFrame], method: str = "concat"
    ) -> pd.DataFrame:
        """Combine data from multiple tickers."""
        if method == "concat":
            # Concatenate all data
            combined = pd.concat(data_dict.values(), axis=0, ignore_index=True)
            return combined

        elif method == "average":
            # Average across tickers
            # Align indices first
            aligned_data = []
            for ticker, df in data_dict.items():
                aligned_data.append(df)

            combined = pd.concat(aligned_data, axis=1, keys=data_dict.keys())
            # Average numeric columns
            numeric_cols = combined.select_dtypes(include=[np.number]).columns
            averaged = combined[numeric_cols].groupby(level=1, axis=1).mean()
            return averaged

        else:
            raise ValueError(f"Unknown combination method: {method}")
# — Raynergy-svg —
