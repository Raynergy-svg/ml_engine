"""
Advanced feature engineering for financial time series.
Includes technical indicators, statistical features, and derived metrics.

M1 Metal Performance Notes:
- All operations are pandas/numpy based (CPU)
- Feature engineering runs once before training (not a bottleneck)
- For >100k rows, consider using numba or vectorized operations
- Output arrays are float32 for TensorFlow Metal compatibility
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

        # =====================================================================
        # MARKET REGIME DETECTION (2025 Best Practice)
        # =====================================================================
        
        # Trending vs Ranging Regime
        # High ADX + clear DI separation = trending
        # Low ADX = ranging/consolidating
        df["is_trending"] = (df["adx"] > 25).astype(float)
        df["is_strong_trend"] = (df["adx"] > 40).astype(float)
        df["is_ranging"] = (df["adx"] < 20).astype(float)
        
        # Volatility Regime (based on ATR percentile)
        atr_pct = df["atr"].rolling(100).apply(
            lambda x: pd.Series(x).rank(pct=True).iloc[-1] if len(x) > 0 else 0.5,
            raw=False
        )
        df["volatility_regime"] = atr_pct.fillna(0.5)
        df["is_high_volatility"] = (df["volatility_regime"] > 0.7).astype(float)
        df["is_low_volatility"] = (df["volatility_regime"] < 0.3).astype(float)
        
        # Trend Quality Score (combining ADX, DI spread, and MA alignment)
        ma_aligned_up = ((df["sma_5"] > df["sma_20"]) & (df["sma_20"] > df["sma_50"])).astype(float)
        ma_aligned_down = ((df["sma_5"] < df["sma_20"]) & (df["sma_20"] < df["sma_50"])).astype(float)
        df["ma_alignment"] = ma_aligned_up - ma_aligned_down  # +1 = bullish aligned, -1 = bearish aligned
        
        # Squeeze Detection (low BB width = potential breakout)
        bb_width_pct = df["bb_width_20"].rolling(50).apply(
            lambda x: pd.Series(x).rank(pct=True).iloc[-1] if len(x) > 0 else 0.5,
            raw=False
        )
        df["bb_squeeze"] = (bb_width_pct < 0.2).astype(float).fillna(0)
        
        # =====================================================================
        # FEATURE INTERACTIONS (2025 Best Practice)
        # =====================================================================
        
        # RSI × ATR (strong signal in high volatility)
        df["rsi_x_atr_norm"] = ((df["rsi"] - 50) / 50) * (df["atr"] / df["atr"].rolling(20).mean())
        
        # MACD × ADX (trend momentum strength)
        df["macd_x_adx"] = (df["macd_hist"] / (df["macd_hist"].rolling(20).std() + 1e-8)) * (df["adx"] / 50)
        
        # Volume × Price Movement (confirmation)
        if "volume" in df.columns:
            vol_norm = df["volume"] / df["volume"].rolling(20).mean()
            price_move = df["close"].pct_change().abs() * 100
            df["volume_price_strength"] = vol_norm * price_move
        
        # Overbought/Oversold with trend confirmation
        df["rsi_trend_confirm"] = (
            ((df["rsi"] > 70) & (df["trend_direction"] > 0)).astype(float) * -1 +  # Overbought in uptrend
            ((df["rsi"] < 30) & (df["trend_direction"] < 0)).astype(float) * 1     # Oversold in downtrend
        )

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

        # =====================================================================
        # TREND-AGNOSTIC MOMENTUM FEATURES (2025 Best Practice)
        # Use percentage returns normalized by volatility instead of raw price diffs
        # This makes features stationary across different price levels and regimes
        # =====================================================================
        
        # ATR for volatility normalization (compute if not already present)
        if "atr" not in df.columns:
            high_low = df["high"] - df["low"]
            high_close = np.abs(df["high"] - df["close"].shift())
            low_close = np.abs(df["low"] - df["close"].shift())
            ranges = pd.concat([high_low, high_close, low_close], axis=1)
            true_range = ranges.max(axis=1)
            atr_temp = true_range.rolling(window=14).mean()
        else:
            atr_temp = df["atr"]
        
        # Rolling ATR for normalization (smoothed to reduce noise)
        rolling_atr = atr_temp.rolling(20).mean().fillna(atr_temp)
        # Prevent division by zero
        rolling_atr = rolling_atr.replace(0, np.nan).ffill().fillna(0.0001)
        
        # Momentum as LOG RETURNS (trend-agnostic, stationary)
        # Log returns compound properly and are symmetric for up/down moves
        for window in [5, 10, 20]:
            # Log return momentum: ln(close[t] / close[t-N])
            log_return_momentum = np.log(
                df["close"] / df["close"].shift(window).replace(0, np.nan)
            ).fillna(0)
            
            # Volatility-normalized momentum: divide by rolling ATR
            # This makes "0.5% move" mean different things in low vs high volatility
            # A 0.5% move in low vol is significant; in high vol it's noise
            df[f"momentum_{window}"] = log_return_momentum / rolling_atr
            
            # Also keep raw percentage momentum for comparison
            df[f"momentum_pct_{window}"] = (
                df["close"] / df["close"].shift(window) - 1.0
            ).fillna(0)
        
        # Clip extreme values to prevent outliers from dominating
        for window in [5, 10, 20]:
            df[f"momentum_{window}"] = df[f"momentum_{window}"].clip(-10, 10)

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

        # =====================================================================
        # FX SESSION FEATURES (Critical for Forex - 2025 Best Practice)
        # =====================================================================
        
        # Hour of day (critical for intraday trading)
        df["hour"] = df.index.hour
        df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
        df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
        
        # FX Trading Sessions (times in UTC)
        # Asian Session: 00:00 - 09:00 UTC (Tokyo 09:00-18:00 JST)
        # London Session: 07:00 - 16:00 UTC
        # NY Session: 13:00 - 22:00 UTC
        df["is_asian_session"] = ((df["hour"] >= 0) & (df["hour"] < 9)).astype(int)
        df["is_london_session"] = ((df["hour"] >= 7) & (df["hour"] < 16)).astype(int)
        df["is_ny_session"] = ((df["hour"] >= 13) & (df["hour"] < 22)).astype(int)
        
        # Session Overlaps (highest volatility periods)
        df["is_london_ny_overlap"] = ((df["hour"] >= 13) & (df["hour"] < 16)).astype(int)
        df["is_asian_london_overlap"] = ((df["hour"] >= 7) & (df["hour"] < 9)).astype(int)
        
        # Weekend proximity (Friday afternoon = risk-off, Sunday evening = gaps)
        df["is_friday"] = (df["day_of_week"] == 4).astype(int)
        df["is_monday"] = (df["day_of_week"] == 0).astype(int)
        df["friday_afternoon"] = ((df["day_of_week"] == 4) & (df["hour"] >= 18)).astype(int)
        
        # Market regime time features
        # First/last hour of major sessions often have higher volatility
        df["is_session_open"] = (
            ((df["hour"] == 0) | (df["hour"] == 7) | (df["hour"] == 13))
        ).astype(int)
        df["is_session_close"] = (
            ((df["hour"] == 8) | (df["hour"] == 15) | (df["hour"] == 21))
        ).astype(int)

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

    def add_regime_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add market regime detection features (2025 best practice).
        
        Identifies:
        1. Trend vs range-bound (using ADX)
        2. Volatility regime (high/low/normal)
        3. Momentum regime
        4. Mean-reversion potential
        """
        df = df.copy()
        
        # --- Trend Regime (ADX-based) ---
        if 'adx' in df.columns:
            # Strong trend: ADX > 40
            # Trending: ADX > 25
            # Range-bound: ADX < 20
            df['is_strong_trend'] = (df['adx'] > 40).astype(float)
            df['is_trending'] = (df['adx'] > 25).astype(float)
            df['is_ranging'] = (df['adx'] < 20).astype(float)
            
            # Trend strength (normalized ADX)
            df['trend_strength'] = df['adx'].clip(0, 100) / 100.0
        
        # --- Volatility Regime ---
        if 'atr' in df.columns:
            # Use rolling percentile of ATR
            atr_pctl = df['atr'].rolling(window=100, min_periods=20).apply(
                lambda x: pd.Series(x).rank(pct=True).iloc[-1], raw=False
            )
            df['volatility_regime'] = atr_pctl.fillna(0.5)
            df['is_high_volatility'] = (atr_pctl > 0.75).astype(float)
            df['is_low_volatility'] = (atr_pctl < 0.25).astype(float)
        
        # --- Momentum Regime ---
        if 'rsi' in df.columns:
            # Overbought/Oversold
            df['is_overbought'] = (df['rsi'] > 70).astype(float)
            df['is_oversold'] = (df['rsi'] < 30).astype(float)
            df['rsi_extreme'] = ((df['rsi'] > 70) | (df['rsi'] < 30)).astype(float)
            
            # RSI momentum (rate of change)
            df['rsi_momentum'] = df['rsi'].diff(5) / 5.0
        
        # --- Moving Average Alignment ---
        if all(col in df.columns for col in ['sma_5', 'sma_20', 'sma_50']):
            # Bullish alignment: 5 > 20 > 50
            # Bearish alignment: 5 < 20 < 50
            bull_align = ((df['sma_5'] > df['sma_20']) & (df['sma_20'] > df['sma_50'])).astype(float)
            bear_align = ((df['sma_5'] < df['sma_20']) & (df['sma_20'] < df['sma_50'])).astype(float)
            df['ma_bullish_alignment'] = bull_align
            df['ma_bearish_alignment'] = bear_align
            df['ma_alignment_strength'] = bull_align - bear_align
        
        # --- Bollinger Band Regime ---
        if 'bb_width_20' in df.columns:
            bb_pctl = df['bb_width_20'].rolling(window=100, min_periods=20).apply(
                lambda x: pd.Series(x).rank(pct=True).iloc[-1], raw=False
            )
            df['bb_squeeze'] = (bb_pctl < 0.2).astype(float)  # Potential breakout
            df['bb_expansion'] = (bb_pctl > 0.8).astype(float)
        
        if 'bb_position_20' in df.columns:
            # Position within bands (0-1)
            df['bb_upper_touch'] = (df['bb_position_20'] > 0.95).astype(float)
            df['bb_lower_touch'] = (df['bb_position_20'] < 0.05).astype(float)
        
        # --- Session Features (FX-specific) ---
        if hasattr(df.index, 'hour'):
            hour = df.index.hour
            df['hour_sin'] = np.sin(2 * np.pi * hour / 24)
            df['hour_cos'] = np.cos(2 * np.pi * hour / 24)
            
            # Major FX sessions (UTC)
            df['is_asian_session'] = ((hour >= 0) & (hour < 9)).astype(float)
            df['is_london_session'] = ((hour >= 7) & (hour < 16)).astype(float)
            df['is_ny_session'] = ((hour >= 13) & (hour < 22)).astype(float)
            
            # High volatility overlaps
            df['is_london_ny_overlap'] = ((hour >= 13) & (hour < 16)).astype(float)
            df['is_asian_london_overlap'] = ((hour >= 7) & (hour < 9)).astype(float)
        
        # --- Day of Week Features ---
        if hasattr(df.index, 'dayofweek'):
            dow = df.index.dayofweek
            df['is_monday'] = (dow == 0).astype(float)
            df['is_friday'] = (dow == 4).astype(float)
            df['is_mid_week'] = ((dow >= 1) & (dow <= 3)).astype(float)
            
            # Weekend gap risk (Friday afternoon)
            if hasattr(df.index, 'hour'):
                df['friday_afternoon'] = ((dow == 4) & (df.index.hour >= 17)).astype(float)
        
        return df

    def add_intermarket_features(
        self, 
        df: pd.DataFrame, 
        correlates: dict = None
    ) -> pd.DataFrame:
        """Add intermarket correlation features (2025 enhancement).
        
        For FX, relevant correlates include:
        - DXY (Dollar Index) for USD pairs
        - VIX (volatility index)
        - Gold (safe haven)
        - Oil (commodity currencies)
        - Bond yields (rate differentials)
        
        Args:
            df: Main price DataFrame
            correlates: Dict of {name: DataFrame} with correlated asset prices
                       Each DataFrame should have 'close' column and same index
        
        Note: This method adds placeholder features if correlates not provided,
              allowing the model architecture to remain consistent.
        """
        df = df.copy()
        
        if correlates is None:
            # Add placeholder features (will be 0.5 or neutral)
            # This maintains feature consistency when external data unavailable
            df['dxy_corr_20'] = 0.0
            df['vix_level'] = 0.5
            df['gold_momentum'] = 0.0
            df['yield_diff_momentum'] = 0.0
            df['risk_on_score'] = 0.5
            return df
        
        # --- DXY (Dollar Index) Correlation ---
        if 'dxy' in correlates and 'close' in correlates['dxy'].columns:
            dxy = correlates['dxy']['close'].reindex(df.index, method='ffill')
            
            # Rolling correlation with USD
            df['dxy_corr_20'] = df['close'].rolling(20).corr(dxy).fillna(0)
            df['dxy_momentum'] = dxy.pct_change(5).fillna(0)
            df['dxy_zscore'] = ((dxy - dxy.rolling(50).mean()) / dxy.rolling(50).std()).fillna(0)
        
        # --- VIX (Volatility Index) ---
        if 'vix' in correlates and 'close' in correlates['vix'].columns:
            vix = correlates['vix']['close'].reindex(df.index, method='ffill')
            
            df['vix_level'] = (vix.clip(10, 80) - 10) / 70.0  # Normalize to 0-1
            df['vix_spike'] = (vix.diff() > vix.rolling(20).std() * 2).astype(float)
            df['vix_high'] = (vix > 25).astype(float)
        
        # --- Gold (Safe Haven) ---
        if 'gold' in correlates and 'close' in correlates['gold'].columns:
            gold = correlates['gold']['close'].reindex(df.index, method='ffill')
            
            df['gold_momentum'] = gold.pct_change(10).fillna(0).clip(-0.1, 0.1)
            df['gold_corr_20'] = df['close'].rolling(20).corr(gold).fillna(0)
        
        # --- Yield Spread (Rate Differential) ---
        if 'us_10y' in correlates and 'jp_10y' in correlates:
            us_yield = correlates['us_10y']['close'].reindex(df.index, method='ffill')
            jp_yield = correlates['jp_10y']['close'].reindex(df.index, method='ffill')
            
            # Yield differential (important for carry trades)
            yield_diff = us_yield - jp_yield
            df['yield_diff'] = yield_diff.fillna(yield_diff.mean())
            df['yield_diff_momentum'] = yield_diff.diff(5).fillna(0)
        
        # --- Composite Risk Score ---
        # Combines multiple factors into a risk-on/risk-off indicator
        risk_factors = []
        if 'vix_level' in df.columns:
            risk_factors.append(1 - df['vix_level'])  # Lower VIX = risk-on
        if 'gold_momentum' in df.columns:
            risk_factors.append(-df['gold_momentum'] * 5)  # Gold selling = risk-on
        
        if risk_factors:
            df['risk_on_score'] = np.mean(risk_factors, axis=0).clip(0, 1)
        else:
            df['risk_on_score'] = 0.5
        
        return df

    def add_momentum_divergence_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add momentum divergence features (classic technical patterns).
        
        Divergences between price and momentum indicators often precede reversals.
        """
        df = df.copy()
        
        # Price highs/lows over window
        window = 14
        price_high = df['close'].rolling(window).max()
        price_low = df['close'].rolling(window).min()
        
        # Check if RSI diverges from price
        if 'rsi' in df.columns:
            rsi_high = df['rsi'].rolling(window).max()
            rsi_low = df['rsi'].rolling(window).min()
            
            # Bearish divergence: price makes new high but RSI doesn't
            new_price_high = (df['close'] >= price_high * 0.999)
            rsi_lower_high = (df['rsi'] < rsi_high - 5)
            df['bearish_divergence'] = (new_price_high & rsi_lower_high).astype(float)
            
            # Bullish divergence: price makes new low but RSI doesn't
            new_price_low = (df['close'] <= price_low * 1.001)
            rsi_higher_low = (df['rsi'] > rsi_low + 5)
            df['bullish_divergence'] = (new_price_low & rsi_higher_low).astype(float)
        
        # MACD divergence
        if 'macd' in df.columns:
            macd_high = df['macd'].rolling(window).max()
            macd_low = df['macd'].rolling(window).min()
            
            new_price_high = (df['close'] >= price_high * 0.999)
            macd_lower = (df['macd'] < macd_high * 0.9)
            df['macd_bearish_div'] = (new_price_high & macd_lower).astype(float)
            
            new_price_low = (df['close'] <= price_low * 1.001)
            macd_higher = (df['macd'] > macd_low * 1.1) if (df['macd'] < 0).any() else (df['macd'] > macd_low + 0.0001)
            df['macd_bullish_div'] = (new_price_low & macd_higher.fillna(False)).astype(float)
        
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

    def normalize_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Normalize features to prevent scale issues (2025 Best Practice).
        
        This method:
        1. Clips extreme values to prevent outliers from dominating
        2. Normalizes price-derived features to [-1, 1] or [0, 1] ranges
        3. Handles inf/nan values
        """
        df = df.copy()
        
        # Features that should be in [0, 100] range (oscillators)
        oscillator_cols = ['rsi', 'rsi_7', 'stoch_k', 'stoch_d', 'mfi', 'adx']
        for col in oscillator_cols:
            if col in df.columns:
                df[col] = df[col].clip(0, 100)
        
        # Features that should be in [-100, 100] range
        bounded_cols = ['williams_r', 'cci']
        for col in bounded_cols:
            if col in df.columns:
                if col == 'williams_r':
                    df[col] = df[col].clip(-100, 0)
                elif col == 'cci':
                    df[col] = df[col].clip(-200, 200)
        
        # Percentage features should be clipped
        pct_cols = [c for c in df.columns if 'pct_change' in c or 'roc_' in c]
        for col in pct_cols:
            if col in df.columns:
                df[col] = df[col].clip(-0.5, 0.5)  # ±50% max
        
        # Normalize volume ratio (often spikes)
        if 'volume_ratio' in df.columns:
            df['volume_ratio'] = df['volume_ratio'].clip(0, 5)
        
        # Z-scores should be bounded
        zscore_cols = [c for c in df.columns if 'zscore' in c]
        for col in zscore_cols:
            if col in df.columns:
                df[col] = df[col].clip(-4, 4)
        
        # Handle inf/nan
        df = df.replace([np.inf, -np.inf], np.nan)
        
        # Log transform skewed features (volatility, ATR)
        # This helps with the long-tail distribution
        if 'atr' in df.columns:
            atr_mean = df['atr'].mean()
            if atr_mean > 0:
                df['atr_log'] = np.log1p(df['atr'] / atr_mean)
        
        for col in ['volatility_5', 'volatility_10', 'volatility_20']:
            if col in df.columns:
                col_mean = df[col].mean()
                if col_mean > 0:
                    df[f'{col}_log'] = np.log1p(df[col] / col_mean)
        
        return df

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

    def add_news_sentiment_features(
        self,
        df: pd.DataFrame,
        instrument: str = "EUR_USD",
    ) -> pd.DataFrame:
        """Add real-time news sentiment features using FinBERT (if available).
        
        This adds:
        - news_sentiment: Mean sentiment score from recent headlines [-1, 1]
        - news_volume: Number of news items (activity = volatility signal)
        
        Requires optional news_features module. Falls back gracefully if not available.
        
        Args:
            df: DataFrame to add features to
            instrument: Currency pair for news fetching
            
        Returns:
            DataFrame with news sentiment features added
        """
        news_cfg = (self.config or {}).get("news", {})
        if not bool(news_cfg.get("enabled", True)):
            return df
        
        try:
            from news_features import add_sentiment_features
            use_finbert = news_cfg.get("use_finbert", True)
            return add_sentiment_features(df, instrument, use_finbert=use_finbert)
        except ImportError:
            logger.debug("news_features module not available")
            return df
        except Exception as e:
            logger.warning(f"Failed to add news sentiment features: {e}")
            return df

    def create_features(
        self,
        df: pd.DataFrame,
        include_all: bool = True,
        *,
        apply_candle_smoothing: bool = True,
        median_window: Optional[int] = None,
        instrument: Optional[str] = None,
    ) -> pd.DataFrame:
        """Create all features.
        
        Args:
            df: Input DataFrame with OHLCV data
            include_all: Whether to add all feature sets
            apply_candle_smoothing: Whether to apply candle smoothing
            median_window: Window size for median smoothing
            instrument: Currency pair for news sentiment features (e.g., 'EUR_USD')
        """
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

        # Optional news sentiment features (requires news_features module)
        if instrument is not None:
            df = self.add_news_sentiment_features(df, instrument)

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
            
            # Enhanced features (2025 improvements)
            # Market regime detection
            df = self.add_regime_features(df)
            # Momentum divergences (classic patterns)
            df = self.add_momentum_divergence_features(df)
            # Intermarket correlations (if external data available)
            correlates = self.config.get("intermarket_data")
            if correlates:
                df = self.add_intermarket_features(df, correlates)
            else:
                # Add placeholder features for consistent model architecture
                df = self.add_intermarket_features(df, None)
            
            # Normalize features (2025 Best Practice)
            # Prevents scale issues and clips extreme values
            if self.config.get("normalize_features", True):
                df = self.normalize_features(df)

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
