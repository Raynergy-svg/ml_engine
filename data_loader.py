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

logger = logging.getLogger(__name__)


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
        logger.info(f"Loading data from {file_path}")

        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Data file not found: {file_path}")

        try:
            df = pd.read_csv(file_path)
            logger.info(f"Loaded {len(df)} rows from {file_path}")

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
                df["date"] = pd.to_datetime(df["date"])
                df.set_index("date", inplace=True)
            elif df.index.name and "date" in df.index.name.lower():
                df.index = pd.to_datetime(df.index)

            # Sort by date
            df = df.sort_index()

            # Remove duplicates
            df = df[~df.index.duplicated(keep="last")]

            # Validate data quality
            self._validate_data(df)

            return df

        except Exception as e:
            logger.error(f"Error loading data: {e}")
            raise

    def _validate_data(self, df: pd.DataFrame) -> None:
        """Validate data quality."""
        # Check for missing values
        missing_pct = df.isnull().sum() / len(df) * 100
        if missing_pct.max() > 50:
            logger.warning(
                f"High percentage of missing values detected:\n{missing_pct[missing_pct > 0]}"
            )

        # Check for negative values in price columns
        price_cols = ["open", "high", "low", "close"]
        for col in price_cols:
            if col in df.columns and (df[col] < 0).any():
                logger.warning(f"Negative values found in {col}")

        # Check for zero volume
        if "volume" in df.columns:
            zero_volume_pct = (df["volume"] == 0).sum() / len(df) * 100
            if zero_volume_pct > 10:
                logger.warning(f"{zero_volume_pct:.2f}% of data has zero volume")

        # Check for price consistency (high >= low, etc.)
        if all(col in df.columns for col in ["high", "low"]):
            inconsistent = df["high"] < df["low"]
            if inconsistent.any():
                logger.warning(f"Found {inconsistent.sum()} rows where high < low")
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
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Preprocess data with feature engineering and scaling."""
        logger.info("Starting data preprocessing...")

        # Feature engineering
        if add_features:
            df = self.feature_engineer.create_features(df, include_all=True)

        # Remove rows with NaN values
        df = df.dropna()

        if len(df) < sequence_length + 100:
            raise ValueError(f"Insufficient data after preprocessing: {len(df)} rows")

        # Prepare features and target
        target_col = "close"
        if target_col not in df.columns:
            raise ValueError(f"Target column '{target_col}' not found")

        # Select numeric columns only
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

        # Remove target from features
        feature_cols = [col for col in numeric_cols if col != target_col]

        X = df[feature_cols].values
        y = df[target_col].values

        logger.info(f"Features shape: {X.shape}, Target shape: {y.shape}")

        # Scale features
        if scaler_type == "standard":
            self.scaler = StandardScaler()
        elif scaler_type == "minmax":
            self.scaler = MinMaxScaler()
        elif scaler_type == "robust":
            self.scaler = RobustScaler()
        else:
            self.scaler = StandardScaler()

        X_scaled = self.scaler.fit_transform(X)

        # Create sequences
        X_seq, y_seq = self._create_sequences(X_scaled, y, sequence_length)

        logger.info(f"Created sequences: X={X_seq.shape}, y={y_seq.shape}")

        # Split into train, validation, and test sets
        # First split: separate test set
        X_temp, X_test, y_temp, y_test = train_test_split(
            X_seq, y_seq, test_size=test_size, shuffle=False
        )

        # Second split: separate validation from train
        val_size_adjusted = validation_size / (1 - test_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=val_size_adjusted, shuffle=False
        )

        logger.info(f"Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")

        self.feature_names = feature_cols

        return X_train, y_train, X_val, y_val, X_test, y_test

    def _create_sequences(
        self, X: np.ndarray, y: np.ndarray, sequence_length: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Create sequences for time series prediction."""
        X_seq, y_seq = [], []

        for i in range(len(X) - sequence_length):
            X_seq.append(X[i : i + sequence_length])
            y_seq.append(y[i + sequence_length])

        return np.array(X_seq), np.array(y_seq)

    def save_scaler(self, path: str) -> None:
        """Save the fitted scaler."""
        if self.scaler is None:
            logger.warning("No scaler to save")
            return

        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.scaler, path)
        logger.info(f"Scaler saved to {path}")

    def load_scaler(self, path: str) -> None:
        """Load a fitted scaler."""
        if not os.path.exists(path):
            raise FileNotFoundError(f"Scaler file not found: {path}")

        self.scaler = joblib.load(path)
        logger.info(f"Scaler loaded from {path}")

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

        X = df[feature_cols].values
        X_scaled = self.scaler.transform(X)

        # Create sequences
        if len(X_scaled) >= sequence_length:
            X_seq = np.array([X_scaled[-sequence_length:]])
            return X_seq
        else:
            raise ValueError(
                f"Insufficient data: need {sequence_length}, got {len(X_scaled)}"
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
