"""
Advanced feature engineering for financial time series.
Includes technical indicators, statistical features, and derived metrics.
"""

import logging
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

from text_features import simple_sentiment_score
from candle_smoothing import resample_5min_ohlcv_and_ema_close

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
            df[f"ema_{window}"] = df["close"].ewm(
                span=window,
                adjust=False,
            ).mean()

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
            df[f"bb_position_{window}"] = (
                (df["close"] - df[f"bb_lower_{window}"])
                / df[f"bb_width_{window}"]
            )

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
        df["obv"] = (
            (np.sign(df["close"].diff()) * df["volume"]).fillna(0).cumsum()
        )

        # Money Flow Index (MFI)
        typical_price = (df["high"] + df["low"] + df["close"]) / 3
        money_flow = typical_price * df["volume"]

        positive_flow = money_flow.where(
            typical_price > typical_price.shift(1),
            0,
        )
        negative_flow = money_flow.where(
            typical_price < typical_price.shift(1),
            0,
        )

        positive_mf = positive_flow.rolling(window=14).sum()
        negative_mf = negative_flow.rolling(window=14).sum()

        mfi_ratio = positive_mf / negative_mf
        df["mfi"] = 100 - (100 / (1 + mfi_ratio))

        # Williams %R
        df["williams_r"] = -100 * (
            (high_14 - df["close"]) / (high_14 - low_14)
        )

        # Rate of Change (ROC)
        for window in [5, 10, 20]:
            df[f"roc_{window}"] = (
                (df["close"] - df["close"].shift(window))
                / df["close"].shift(window)
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

        # =====================================================================
        # DIRECTIONAL FEATURES (Enhanced for direction prediction)
        # =====================================================================
        
        # MACD Crossover Signals (+1 = bullish cross, -1 = bearish cross, 0 = no cross)
        macd_above_signal = (df["macd"] > df["macd_signal"]).astype(int)
        df["macd_crossover"] = macd_above_signal.diff().fillna(0)  # +1, -1, or 0
        
        # MACD Histogram momentum (acceleration of trend)
        df["macd_hist_momentum"] = df["macd_hist"].diff()
        
        # MACD Divergence (price vs MACD disagreement)
        price_direction = np.sign(df["close"].diff(5))
        macd_direction = np.sign(df["macd"].diff(5))
        df["macd_divergence"] = (price_direction != macd_direction).astype(float)
        
        # Trend Strength Composite (combines multiple indicators)
        # Normalized ADX (0-1 scale, >0.5 = strong trend)
        adx_norm = df["adx"] / 100.0
        # RSI trend (>50 = bullish, <50 = bearish, normalized to -1 to +1)
        rsi_trend = (df["rsi"] - 50) / 50.0
        # MACD histogram normalized
        macd_hist_norm = df["macd_hist"] / (df["macd_hist"].rolling(20).std() + 1e-8)
        macd_hist_norm = macd_hist_norm.clip(-3, 3) / 3.0  # Clip to [-1, 1]
        
        df["trend_strength"] = (adx_norm * 0.4 + rsi_trend.abs() * 0.3 + macd_hist_norm.abs() * 0.3)
        df["trend_direction"] = np.sign(rsi_trend) * df["trend_strength"]
        
        # Moving Average Crossovers
        df["sma_5_20_cross"] = (df["sma_5"] > df["sma_20"]).astype(int).diff().fillna(0)
        df["ema_12_26_cross"] = (ema_12 > ema_26).astype(int).diff().fillna(0)
        
        # Momentum Divergence (price making new highs but momentum not)
        price_high_20 = df["close"].rolling(20).max()
        mom_high_20 = df["momentum_10"].rolling(20).max() if "momentum_10" in df.columns else df["close"].diff(10).rolling(20).max()
        df["momentum_divergence"] = (
            (df["close"] >= price_high_20 * 0.99)
            & (df.get("momentum_10", df["close"].diff(10)) < mom_high_20 * 0.9)
        ).astype(float)
        
        # Stochastic Crossover
        df["stoch_crossover"] = (df["stoch_k"] > df["stoch_d"]).astype(int).diff().fillna(0)
        
        # RSI Overbought/Oversold signals
        df["rsi_signal"] = np.where(df["rsi"] > 70, -1, np.where(df["rsi"] < 30, 1, 0))
        
        # =====================================================================
        # ADDITIONAL MOMENTUM FEATURES (Step 3: Feature engineering for momentum)
        # =====================================================================
        
        # CCI Signal (Commodity Channel Index)
        # Buy when CCI < -100 (oversold), Sell when CCI > 100 (overbought)
        if "cci" in df.columns:
            df["cci_signal"] = np.where(df["cci"] > 100, -1, np.where(df["cci"] < -100, 1, 0))
        
        # ADX Trend Strength (>25 = strong trend, >40 = very strong)
        df["adx_trend_strong"] = (df["adx"] > 25).astype(float)
        df["adx_trend_very_strong"] = (df["adx"] > 40).astype(float)
        
        # DI Crossover (+DI crosses above -DI = bullish, vice versa)
        di_bullish = (df["plus_di"] > df["minus_di"]).astype(int)
        df["di_crossover"] = di_bullish.diff().fillna(0)  # +1 = bullish cross, -1 = bearish cross
        
        # DI Spread (larger spread = stronger trend direction)
        df["di_spread"] = (df["plus_di"] - df["minus_di"]) / (df["plus_di"] + df["minus_di"] + 1e-8)
        
        # Price Momentum (percentage change over different windows)
        for window in [5, 10, 20]:
            df[f"price_pct_change_{window}"] = df["close"].pct_change(window)
        
        # Acceleration (second derivative of price)
        df["price_acceleration"] = df["close"].diff().diff()
        df["momentum_acceleration"] = df["momentum_10"].diff() if "momentum_10" in df.columns else df["close"].diff(10).diff()
        
        # Volume-Price Confirmation (rising price + rising volume = strong)
        if "volume" in df.columns:
            price_rising = (df["close"].diff() > 0).astype(int)
            volume_rising = (df["volume"].diff() > 0).astype(int)
            df["volume_price_confirm"] = (price_rising * volume_rising - (1 - price_rising) * volume_rising).fillna(0)
        
        # Multi-timeframe RSI agreement
        rsi_short = (
            df["close"].diff(7).apply(lambda x: max(x, 0)).rolling(7).mean()
            / df["close"].diff(7).abs().rolling(7).mean() * 100
        )
        df["rsi_7"] = rsi_short.fillna(50)
        df["rsi_agreement"] = ((df["rsi"] > 50) == (df["rsi_7"] > 50)).astype(float)

        return df

    def add_statistical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add statistical features."""
        df = df.copy()

        # Returns
        df["returns"] = df["close"].pct_change()
        df["log_returns"] = np.log(df["close"] / df["close"].shift(1))

        # Volatility measures
        for window in [5, 10, 20, 60]:
            df[f"volatility_{window}"] = df["returns"].rolling(
                window=window
            ).std()
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
            # Check for various datetime column names
            date_col = None
            for col in ["date", "Date", "time", "Time", "datetime", "Datetime", "timestamp"]:
                if col in df.columns:
                    date_col = col
                    break
            
            if date_col:
                # Force a real DatetimeIndex even if the source strings have mixed/explicit timezones.
                # Using utc=True avoids pandas returning an object Index (which lacks .dayofweek).
                dt = pd.to_datetime(df[date_col], errors='coerce', utc=True)
                if getattr(dt, 'isna', lambda: False)().all():
                    logger.debug("Date column could not be parsed - skipping time features")
                    return df
                # Drop timezone to keep downstream features consistent.
                df[date_col] = dt.dt.tz_convert(None)
                df.set_index(date_col, inplace=True)

                if not isinstance(df.index, pd.DatetimeIndex):
                    logger.debug("Parsed date index is not a DatetimeIndex - skipping time features")
                    return df
            else:
                logger.debug("No datetime index or date column found - skipping time features")
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
            df[f"close_mean_{window}"] = df["close"].rolling(
                window=window
            ).mean()
            df[f"close_std_{window}"] = df["close"].rolling(
                window=window
            ).std()
            df[f"close_min_{window}"] = df["close"].rolling(
                window=window
            ).min()
            df[f"close_max_{window}"] = df["close"].rolling(
                window=window
            ).max()

            # Rolling volume statistics
            df[f"volume_mean_{window}"] = df["volume"].rolling(
                window=window
            ).mean()
            df[f"volume_std_{window}"] = df["volume"].rolling(
                window=window
            ).std()

        return df

    def _parse_int_list(self, values, default, min_value: int) -> List[int]:
        raw = values if isinstance(values, list) else default
        parsed: List[int] = []
        for item in raw:
            try:
                int_item = int(item)
            except (TypeError, ValueError):
                continue
            if int_item >= min_value:
                parsed.append(int_item)
        return parsed

    def _ensure_text_sentiment(
        self, df: pd.DataFrame, text_col: str
    ) -> pd.DataFrame:
        if "text_sentiment" in df.columns:
            return df
        if text_col not in df.columns:
            return df
        try:
            df["text_sentiment"] = df[text_col].astype(str).map(
                simple_sentiment_score
            )
        except Exception as exc:
            logger.warning("Failed to compute text sentiment: %s", exc)
        return df

    def _add_text_rolling(
        self, df: pd.DataFrame, windows: List[int]
    ) -> pd.DataFrame:
        for window in windows:
            df[f"text_sentiment_sma_{window}"] = df["text_sentiment"].rolling(
                window=window
            ).mean()
            df[f"text_sentiment_std_{window}"] = df["text_sentiment"].rolling(
                window=window
            ).std()
        return df

    def _add_text_lags(
        self, df: pd.DataFrame, lags: List[int]
    ) -> pd.DataFrame:
        for lag_i in lags:
            df[f"text_sentiment_lag_{lag_i}"] = df["text_sentiment"].shift(
                lag_i
            )
        return df

    def _add_text_ewm(
        self, df: pd.DataFrame, spans: List[int]
    ) -> pd.DataFrame:
        for span in spans:
            if span <= 1:
                continue
            df[f"text_sentiment_ewm_{span}"] = df["text_sentiment"].ewm(
                span=span,
                adjust=False,
            ).mean()
        return df

    def add_text_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add text-derived numeric features when enabled in config."""
        text_cfg = (self.config or {}).get("text", {})
        if not bool(text_cfg.get("enabled", False)):
            return df

        df = df.copy()
        text_col = str(text_cfg.get("text_column", "text")).lower()
        df = self._ensure_text_sentiment(df, text_col)
        if "text_sentiment" not in df.columns:
            return df

        if "text_count" in df.columns:
            df["text_has_news"] = (df["text_count"] > 0).astype(int)

        if "text_count" in df.columns:
            df["text_sentiment_x_count"] = df["text_sentiment"] * df[
                "text_count"
            ]

        rolling_windows = self._parse_int_list(
            text_cfg.get("rolling_windows"),
            default=[3, 7, 14],
            min_value=2,
        )
        lags = self._parse_int_list(
            text_cfg.get("lags"),
            default=[1, 2, 3],
            min_value=1,
        )

        if rolling_windows:
            df = self._add_text_rolling(df, rolling_windows)
            df = self._add_text_ewm(df, rolling_windows)
        if lags:
            df = self._add_text_lags(df, lags)

        df["text_sentiment_change"] = df["text_sentiment"].diff()
        return df

    def create_features(
        self,
        df: pd.DataFrame,
        include_all: bool = True,
        *,
        apply_candle_smoothing: bool = True,
        median_window: int | None = None,
    ) -> pd.DataFrame:
        """Create all features."""
        logger.info("Starting feature engineering...")

        original_shape = df.shape
        original_len = int(original_shape[0])

        # Candle smoothing for time-based OHLCV (FX-style `time` column).
        # Resample to 5-minute bars and EMA(14) the close before computing indicators.
        if apply_candle_smoothing and isinstance(df, pd.DataFrame) and "time" in [str(c).strip().lower() for c in df.columns]:
            try:
                df = resample_5min_ohlcv_and_ema_close(
                    df,
                    ema_span=14,
                    median_window=median_window,
                    time_col="time",
                )
            except Exception as e:
                logger.debug("Candle smoothing skipped (%s)", e)

        # Optional text-derived features (sentiment/rolling/lags)
        df = self.add_text_features(df)

        # If smoothing produced an EMA close feature, compute indicators on it while
        # keeping the raw close column intact for labels and downstream OHLC logic.
        close_raw = None
        if "close_ema_14" in df.columns and "close" in df.columns:
            try:
                # Store raw close by position (not index) because downstream time features
                # may replace the index (assignment would otherwise align and become NaN).
                close_raw = df["close"].to_numpy(copy=True)
                df["close"] = df["close_ema_14"].to_numpy(dtype=float, copy=False)
            except Exception:
                close_raw = None

        # Add all feature sets
        if include_all:
            df = self.add_technical_indicators(df)
            df = self.add_statistical_features(df)
            df = self.add_time_features(df)
            df = self.add_lag_features(df)
            df = self.add_rolling_features(df)

        # Restore raw close (labels should be based on raw close, not smoothed close).
        if close_raw is not None:
            try:
                df["close"] = close_raw
            except Exception:
                pass

        # Remove infinite values
        df = df.replace([np.inf, -np.inf], np.nan)

        # Forward-fill only (never backward-fill; bfill leaks future).
        df = df.ffill()

        # Drop columns that are entirely NaN (otherwise dropna(axis=0) would drop all rows).
        df = df.dropna(axis=1, how="all")

        # Drop leading rows until all columns have a valid value.
        try:
            first_valids = [col.first_valid_index() for _, col in df.items()]
            first_valids = [ix for ix in first_valids if ix is not None]
            first_valid = max(first_valids) if first_valids else None
        except Exception:
            first_valid = None
        if first_valid is not None:
            df = df.loc[first_valid:]

        # If any NaNs remain (e.g., all-NaN columns), drop affected rows.
        df = df.dropna(axis=0)

        dropped = int(original_len - len(df))
        if dropped > 0:
            logger.info("Dropped %s rows with NaN after forward-fill", dropped)

        new_shape = df.shape
        logger.info(
            "Feature engineering complete: %s -> %s",
            original_shape,
            new_shape,
        )
        logger.info(
            "Added %s new features",
            int(new_shape[1] - original_shape[1]),
        )

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
            selector = SelectKBest(
                score_func=f_regression,
                k=min(top_k, X.shape[1]),
            )
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

        logger.info(
            "Selected %s features using %s method",
            len(top_features),
            method,
        )

        return selected_df, top_features
# — Raynergy-svg —
