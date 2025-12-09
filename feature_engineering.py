"""
Advanced feature engineering for financial time series.
Includes technical indicators, statistical features, and derived metrics.
"""

import numpy as np
import pandas as pd
from typing import List, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class FeatureEngineering:
    """Advanced feature engineering for stock market data."""

    def __init__(self, config: Optional[dict] = None):
        """Initialize feature engineering with configuration."""
        self.config = config or {}
        self.feature_names = []

    def add_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add comprehensive technical indicators."""
        df = df.copy()

        # Simple Moving Averages
        for window in [5, 10, 20, 50, 100, 200]:
            df[f"sma_{window}"] = df["close"].rolling(window=window).mean()

        # Exponential Moving Averages
        for window in [12, 26, 50]:
            df[f"ema_{window}"] = df["close"].ewm(span=window, adjust=False).mean()

        # Moving Average Convergence Divergence (MACD)
        ema_12 = df["close"].ewm(span=12, adjust=False).mean()
        ema_26 = df["close"].ewm(span=26, adjust=False).mean()
        df["macd"] = ema_12 - ema_26
        df["macd_signal"] = df["macd"].ewm(span=9, adjust=False).mean()
        df["macd_hist"] = df["macd"] - df["macd_signal"]

        # Relative Strength Index (RSI)
        delta = df["close"].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df["rsi"] = 100 - (100 / (1 + rs))

        # Bollinger Bands
        for window in [20]:
            rolling_mean = df["close"].rolling(window=window).mean()
            rolling_std = df["close"].rolling(window=window).std()
            df[f"bb_upper_{window}"] = rolling_mean + (rolling_std * 2)
            df[f"bb_lower_{window}"] = rolling_mean - (rolling_std * 2)
            df[f"bb_width_{window}"] = (
                df[f"bb_upper_{window}"] - df[f"bb_lower_{window}"]
            )
            df[f"bb_position_{window}"] = (df["close"] - df[f"bb_lower_{window}"]) / df[
                f"bb_width_{window}"
            ]

        # Stochastic Oscillator
        low_14 = df["low"].rolling(window=14).min()
        high_14 = df["high"].rolling(window=14).max()
        df["stoch_k"] = 100 * ((df["close"] - low_14) / (high_14 - low_14))
        df["stoch_d"] = df["stoch_k"].rolling(window=3).mean()

        # Average True Range (ATR)
        high_low = df["high"] - df["low"]
        high_close = np.abs(df["high"] - df["close"].shift())
        low_close = np.abs(df["low"] - df["close"].shift())
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = ranges.max(axis=1)
        df["atr"] = true_range.rolling(window=14).mean()

        # Commodity Channel Index (CCI)
        tp = (df["high"] + df["low"] + df["close"]) / 3
        df["cci"] = (tp - tp.rolling(window=20).mean()) / (
            0.015 * tp.rolling(window=20).std()
        )

        # On-Balance Volume (OBV)
        df["obv"] = (np.sign(df["close"].diff()) * df["volume"]).fillna(0).cumsum()

        # Money Flow Index (MFI)
        typical_price = (df["high"] + df["low"] + df["close"]) / 3
        money_flow = typical_price * df["volume"]

        positive_flow = money_flow.where(typical_price > typical_price.shift(1), 0)
        negative_flow = money_flow.where(typical_price < typical_price.shift(1), 0)

        positive_mf = positive_flow.rolling(window=14).sum()
        negative_mf = negative_flow.rolling(window=14).sum()

        mfi_ratio = positive_mf / negative_mf
        df["mfi"] = 100 - (100 / (1 + mfi_ratio))

        # Williams %R
        df["williams_r"] = -100 * ((high_14 - df["close"]) / (high_14 - low_14))

        # Rate of Change (ROC)
        for window in [5, 10, 20]:
            df[f"roc_{window}"] = (
                (df["close"] - df["close"].shift(window)) / df["close"].shift(window)
            ) * 100

        # Average Directional Index (ADX)
        plus_dm = df["high"].diff()
        minus_dm = -df["low"].diff()
        plus_dm[plus_dm < 0] = 0
        minus_dm[minus_dm < 0] = 0

        tr = true_range
        atr = tr.rolling(window=14).mean()

        plus_di = 100 * (plus_dm.rolling(window=14).mean() / atr)
        minus_di = 100 * (minus_dm.rolling(window=14).mean() / atr)

        dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
        df["adx"] = dx.rolling(window=14).mean()
        df["plus_di"] = plus_di
        df["minus_di"] = minus_di

        return df

    def add_statistical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add statistical features."""
        df = df.copy()

        # Returns
        df["returns"] = df["close"].pct_change()
        df["log_returns"] = np.log(df["close"] / df["close"].shift(1))

        # Volatility measures
        for window in [5, 10, 20, 60]:
            df[f"volatility_{window}"] = df["returns"].rolling(window=window).std()
            df[f"volatility_log_{window}"] = (
                df["log_returns"].rolling(window=window).std()
            )

        # Skewness and Kurtosis
        for window in [20, 60]:
            df[f"skew_{window}"] = df["returns"].rolling(window=window).skew()
            df[f"kurt_{window}"] = df["returns"].rolling(window=window).kurt()

        # Z-score
        for window in [20, 60]:
            mean = df["close"].rolling(window=window).mean()
            std = df["close"].rolling(window=window).std()
            df[f"zscore_{window}"] = (df["close"] - mean) / std

        # Momentum
        for window in [5, 10, 20]:
            df[f"momentum_{window}"] = df["close"] - df["close"].shift(window)

        # Volume features
        df["volume_sma_20"] = df["volume"].rolling(window=20).mean()
        df["volume_ratio"] = df["volume"] / df["volume_sma_20"]
        df["volume_change"] = df["volume"].pct_change()

        # Price ranges
        df["high_low_ratio"] = df["high"] / df["low"]
        df["close_open_ratio"] = df["close"] / df["open"]

        # Cumulative features
        df["cum_returns"] = (1 + df["returns"]).cumprod() - 1
        df["cum_log_returns"] = df["log_returns"].cumsum()

        return df

    def add_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add time-based features."""
        df = df.copy()

        if not isinstance(df.index, pd.DatetimeIndex):
            if "date" in df.columns or "Date" in df.columns:
                date_col = "date" if "date" in df.columns else "Date"
                df[date_col] = pd.to_datetime(df[date_col])
                df.set_index(date_col, inplace=True)
            else:
                logger.warning("No datetime index or date column found")
                return df

        # Temporal features
        df["day_of_week"] = df.index.dayofweek
        df["day_of_month"] = df.index.day
        df["week_of_year"] = df.index.isocalendar().week
        df["month"] = df.index.month
        df["quarter"] = df.index.quarter
        df["year"] = df.index.year

        # Cyclical encoding
        df["day_sin"] = np.sin(2 * np.pi * df["day_of_week"] / 7)
        df["day_cos"] = np.cos(2 * np.pi * df["day_of_week"] / 7)
        df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
        df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)

        # Trading session features
        df["is_month_start"] = df.index.is_month_start.astype(int)
        df["is_month_end"] = df.index.is_month_end.astype(int)
        df["is_quarter_start"] = df.index.is_quarter_start.astype(int)
        df["is_quarter_end"] = df.index.is_quarter_end.astype(int)

        return df

    def add_lag_features(
        self, df: pd.DataFrame, lags: List[int] = None
    ) -> pd.DataFrame:
        """Add lagged features."""
        df = df.copy()

        if lags is None:
            lags = [1, 2, 3, 5, 10, 20]

        for lag in lags:
            df[f"close_lag_{lag}"] = df["close"].shift(lag)
            df[f"volume_lag_{lag}"] = df["volume"].shift(lag)
            df[f"returns_lag_{lag}"] = (
                df["returns"].shift(lag) if "returns" in df.columns else None
            )

        return df

    def add_rolling_features(
        self, df: pd.DataFrame, windows: List[int] = None
    ) -> pd.DataFrame:
        """Add rolling window features."""
        df = df.copy()

        if windows is None:
            windows = [5, 10, 20, 60]

        for window in windows:
            # Rolling statistics
            df[f"close_mean_{window}"] = df["close"].rolling(window=window).mean()
            df[f"close_std_{window}"] = df["close"].rolling(window=window).std()
            df[f"close_min_{window}"] = df["close"].rolling(window=window).min()
            df[f"close_max_{window}"] = df["close"].rolling(window=window).max()

            # Rolling volume statistics
            df[f"volume_mean_{window}"] = df["volume"].rolling(window=window).mean()
            df[f"volume_std_{window}"] = df["volume"].rolling(window=window).std()

        return df

    def create_features(
        self, df: pd.DataFrame, include_all: bool = True
    ) -> pd.DataFrame:
        """Create all features."""
        logger.info("Starting feature engineering...")

        original_shape = df.shape

        # Add all feature sets
        if include_all:
            df = self.add_technical_indicators(df)
            df = self.add_statistical_features(df)
            df = self.add_time_features(df)
            df = self.add_lag_features(df)
            df = self.add_rolling_features(df)

        # Remove infinite values
        df = df.replace([np.inf, -np.inf], np.nan)

        # Forward fill then backward fill NaN values
        df = df.fillna(method="ffill").fillna(method="bfill")

        # If still NaN, fill with 0
        df = df.fillna(0)

        new_shape = df.shape
        logger.info(f"Feature engineering complete: {original_shape} -> {new_shape}")
        logger.info(f"Added {new_shape[1] - original_shape[1]} new features")

        self.feature_names = df.columns.tolist()

        return df

    def select_features(
        self,
        df: pd.DataFrame,
        target_col: str = "close",
        method: str = "correlation",
        top_k: int = 50,
    ) -> Tuple[pd.DataFrame, List[str]]:
        """Select most important features."""
        from sklearn.feature_selection import (
            SelectKBest,
            f_regression,
            mutual_info_regression,
        )

        # Separate features and target
        y = df[target_col].values
        X = df.drop(columns=[target_col])

        # Remove non-numeric columns
        numeric_cols = X.select_dtypes(include=[np.number]).columns
        X = X[numeric_cols]

        if method == "correlation":
            # Correlation-based selection
            correlations = X.corrwith(df[target_col]).abs()
            top_features = correlations.nlargest(top_k).index.tolist()

        elif method == "f_test":
            # F-test based selection
            selector = SelectKBest(score_func=f_regression, k=min(top_k, X.shape[1]))
            selector.fit(X, y)
            top_features = X.columns[selector.get_support()].tolist()

        elif method == "mutual_info":
            # Mutual information based selection
            selector = SelectKBest(
                score_func=mutual_info_regression, k=min(top_k, X.shape[1])
            )
            selector.fit(X, y)
            top_features = X.columns[selector.get_support()].tolist()

        else:
            top_features = X.columns.tolist()[:top_k]

        selected_df = df[top_features + [target_col]]

        logger.info(f"Selected {len(top_features)} features using {method} method")

        return selected_df, top_features
