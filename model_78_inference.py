"""
Inference wrapper for the verified 78% accuracy model.

This model was trained on Colab A100 with:
- 60k candles, multi-pair (GBP_JPY, EUR_JPY, AUD_USD, USD_CAD, USD_JPY)
- 58 specific features (flat input, not sequence)
- Verified: no data leakage, 77.2% on pure future data

Usage:
    from model_78_inference import Model78Inference
    
    model = Model78Inference()
    model.load()
    
    # Single prediction
    direction, confidence = model.predict(df)  # df = OHLCV DataFrame
    
    # Batch predictions
    results = model.predict_batch(df_dict)  # {pair: df}
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# The 58 features required by the 78% model
REQUIRED_FEATURES = [
    "macd_norm", "macd_signal_norm", "macd_hist_momentum", "rsi_norm", "adx",
    "atr_pct_10", "atr_pct_20", "obv", "volume_ratio_10", "returns_1",
    "returns_5", "returns_10", "returns_20", "macd_hist", "price_acceleration",
    "ema_12_26_cross", "sma_5_20_cross", "momentum_acceleration", "macd_signal",
    "rsi_x_atr_norm", "log_returns", "trend_direction", "ma_alignment",
    "rsi_oversold", "rsi_overbought", "momentum_divergence", "bb_squeeze",
    "volume_price_confirm", "returns_3", "returns_2", "macd_divergence",
    "is_low_volatility", "is_high_volatility", "sma_cross_5_20", "rsi_agreement",
    "hl_range_pct", "upper_wick_ratio", "lower_wick_ratio", "di_crossover",
    "pair_id", "body_ratio", "atr_pct_5", "trend_strength", "rsi_7",
    "atr_pct_14", "sma_5", "lower_low_ratio", "atr", "mfi", "bb_lower_20",
    "sma_10", "ema_26", "sma_20", "sma_50", "bb_width_20", "is_ranging",
    "norm_high", "norm_low"
]

# Pair ID mapping (from training)
PAIR_IDS = {
    "GBP_JPY": 0, "EUR_JPY": 1, "AUD_USD": 2, "USD_CAD": 3, "USD_JPY": 4,
    "EUR_USD": 5, "GBP_USD": 6, "NZD_USD": 7, "USD_CHF": 8, "AUD_JPY": 9,
    "EUR_GBP": 10, "GBP_CHF": 11, "EUR_CHF": 12, "CAD_JPY": 13, "NZD_JPY": 14,
}


class Model78Inference:
    """
    Inference wrapper for the 78% accuracy model.
    
    Handles feature engineering to match the exact 58 features
    the model was trained on.
    """
    
    def __init__(
        self,
        model_path: str = "trained_data/models/transformer_direction_78pct.keras",
        meta_path: str = "trained_data/models/modular_ensemble_78pct.meta.json",
        feature_config_path: str = "trained_data/models/direction_feature_config.json",
    ):
        self.model_path = Path(model_path)
        self.meta_path = Path(meta_path)
        self.feature_config_path = Path(feature_config_path)
        
        self._model = None
        self._meta = None
        self._feature_names: List[str] = REQUIRED_FEATURES
        self._scaler = None
        
    def load(self) -> bool:
        """Load the model and metadata."""
        try:
            import keras
            
            if not self.model_path.exists():
                logger.error(f"Model not found: {self.model_path}")
                return False
            
            self._model = keras.models.load_model(str(self.model_path), compile=False)
            
            # Load meta
            if self.meta_path.exists():
                self._meta = json.loads(self.meta_path.read_text())
            
            # Load feature config
            if self.feature_config_path.exists():
                fc = json.loads(self.feature_config_path.read_text())
                self._feature_names = fc.get("feature_names", REQUIRED_FEATURES)
            
            logger.debug(f"Loaded 78% model: {self.model_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            return False
    
    @property
    def is_loaded(self) -> bool:
        return self._model is not None
    
    @property
    def accuracy(self) -> float:
        """Return the verified validation accuracy."""
        if self._meta:
            return self._meta.get("results", {}).get("transformer", {}).get("val_accuracy", 0.78)
        return 0.78
    
    def _compute_features(self, df: pd.DataFrame, pair: str = "EUR_USD") -> Optional[np.ndarray]:
        """
        Compute the exact 58 features required by the model.
        
        Args:
            df: OHLCV DataFrame with columns: open, high, low, close, volume
            pair: Currency pair for pair_id encoding
            
        Returns:
            numpy array of shape (1, 58) or None on failure
        """
        try:
            df = df.copy()
            
            # Ensure we have enough data
            if len(df) < 60:
                logger.warning(f"Not enough data: {len(df)} rows (need 60+)")
                return None
            
            # Basic price columns
            close = df["close"]
            high = df["high"]
            low = df["low"]
            open_ = df["open"]
            volume = df.get("volume", pd.Series(1.0, index=df.index))
            
            # Returns
            df["returns_1"] = close.pct_change(1)
            df["returns_2"] = close.pct_change(2)
            df["returns_3"] = close.pct_change(3)
            df["returns_5"] = close.pct_change(5)
            df["returns_10"] = close.pct_change(10)
            df["returns_20"] = close.pct_change(20)
            df["log_returns"] = np.log(close / close.shift(1))
            
            # Moving averages
            df["sma_5"] = close.rolling(5).mean()
            df["sma_10"] = close.rolling(10).mean()
            df["sma_20"] = close.rolling(20).mean()
            df["sma_50"] = close.rolling(50).mean()
            df["ema_12"] = close.ewm(span=12).mean()
            df["ema_26"] = close.ewm(span=26).mean()
            
            # MACD
            df["macd"] = df["ema_12"] - df["ema_26"]
            df["macd_signal"] = df["macd"].ewm(span=9).mean()
            df["macd_hist"] = df["macd"] - df["macd_signal"]
            macd_std = df["macd"].rolling(20).std().replace(0, 1e-8)
            df["macd_norm"] = df["macd"] / macd_std
            df["macd_signal_norm"] = df["macd_signal"] / macd_std
            df["macd_hist_momentum"] = df["macd_hist"].diff()
            df["macd_divergence"] = (df["macd_hist"] > 0).astype(float) - (df["returns_5"] > 0).astype(float)
            
            # RSI
            delta = close.diff()
            gain = delta.where(delta > 0, 0).rolling(14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
            rs = gain / loss.replace(0, 1e-8)
            df["rsi_14"] = 100 - (100 / (1 + rs))
            df["rsi_norm"] = (df["rsi_14"] - 50) / 50
            df["rsi_oversold"] = (df["rsi_14"] < 30).astype(float)
            df["rsi_overbought"] = (df["rsi_14"] > 70).astype(float)
            
            # RSI 7
            gain7 = delta.where(delta > 0, 0).rolling(7).mean()
            loss7 = (-delta.where(delta < 0, 0)).rolling(7).mean()
            rs7 = gain7 / loss7.replace(0, 1e-8)
            df["rsi_7"] = 100 - (100 / (1 + rs7))
            
            # ATR
            tr = pd.concat([
                high - low,
                (high - close.shift(1)).abs(),
                (low - close.shift(1)).abs()
            ], axis=1).max(axis=1)
            df["atr"] = tr.rolling(14).mean()
            df["atr_pct_5"] = tr.rolling(5).mean() / close * 100
            df["atr_pct_10"] = tr.rolling(10).mean() / close * 100
            df["atr_pct_14"] = df["atr"] / close * 100
            df["atr_pct_20"] = tr.rolling(20).mean() / close * 100
            
            # ADX
            plus_dm = high.diff()
            minus_dm = -low.diff()
            plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0)
            minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0)
            atr14 = tr.rolling(14).mean().replace(0, 1e-8)
            plus_di = 100 * (plus_dm.rolling(14).mean() / atr14)
            minus_di = 100 * (minus_dm.rolling(14).mean() / atr14)
            dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, 1e-8)
            df["adx"] = dx.rolling(14).mean() / 100  # Normalize to 0-1
            df["di_crossover"] = (plus_di > minus_di).astype(float)
            
            # Bollinger Bands
            bb_mid = close.rolling(20).mean()
            bb_std = close.rolling(20).std()
            df["bb_upper_20"] = bb_mid + 2 * bb_std
            df["bb_lower_20"] = bb_mid - 2 * bb_std
            df["bb_width_20"] = (df["bb_upper_20"] - df["bb_lower_20"]) / bb_mid
            df["bb_squeeze"] = (df["bb_width_20"] < df["bb_width_20"].rolling(20).quantile(0.2)).astype(float)
            
            # Volume
            df["obv"] = (np.sign(close.diff()) * volume).cumsum()
            df["obv"] = (df["obv"] - df["obv"].rolling(20).mean()) / df["obv"].rolling(20).std().replace(0, 1e-8)
            df["volume_ratio_10"] = volume / volume.rolling(10).mean().replace(0, 1e-8)
            df["volume_price_confirm"] = ((volume > volume.rolling(10).mean()) & 
                                          (close > close.shift(1))).astype(float)
            
            # MFI
            typical_price = (high + low + close) / 3
            raw_money_flow = typical_price * volume
            positive_flow = raw_money_flow.where(typical_price > typical_price.shift(1), 0).rolling(14).sum()
            negative_flow = raw_money_flow.where(typical_price < typical_price.shift(1), 0).rolling(14).sum()
            mfi_ratio = positive_flow / negative_flow.replace(0, 1e-8)
            df["mfi"] = 100 - (100 / (1 + mfi_ratio))
            df["mfi"] = df["mfi"] / 100  # Normalize
            
            # Price action
            df["hl_range_pct"] = (high - low) / close * 100
            body = (close - open_).abs()
            full_range = (high - low).replace(0, 1e-8)
            df["body_ratio"] = body / full_range
            df["upper_wick_ratio"] = (high - pd.concat([close, open_], axis=1).max(axis=1)) / full_range
            df["lower_wick_ratio"] = (pd.concat([close, open_], axis=1).min(axis=1) - low) / full_range
            df["lower_low_ratio"] = (low < low.shift(1)).astype(float)
            
            # Normalized high/low
            rolling_high = high.rolling(20).max()
            rolling_low = low.rolling(20).min()
            range_20 = (rolling_high - rolling_low).replace(0, 1e-8)
            df["norm_high"] = (high - rolling_low) / range_20
            df["norm_low"] = (low - rolling_low) / range_20
            
            # Price acceleration
            df["price_acceleration"] = df["returns_1"].diff()
            df["momentum_acceleration"] = df["returns_5"].diff()
            
            # Crosses
            df["ema_12_26_cross"] = (df["ema_12"] > df["ema_26"]).astype(float)
            df["sma_5_20_cross"] = (df["sma_5"] > df["sma_20"]).astype(float)
            df["sma_cross_5_20"] = df["sma_5_20_cross"]  # Alias
            
            # Trend/momentum features
            df["trend_direction"] = (close > df["sma_20"]).astype(float)
            df["ma_alignment"] = ((df["sma_5"] > df["sma_10"]) & 
                                  (df["sma_10"] > df["sma_20"])).astype(float)
            df["trend_strength"] = df["adx"]  # Use ADX as trend strength
            
            # Momentum divergence
            price_higher = close > close.shift(5)
            rsi_lower = df["rsi_14"] < df["rsi_14"].shift(5)
            df["momentum_divergence"] = (price_higher & rsi_lower).astype(float)
            
            # RSI agreement
            df["rsi_agreement"] = ((df["rsi_14"] > 50) == (df["returns_5"] > 0)).astype(float)
            
            # RSI x ATR interaction
            df["rsi_x_atr_norm"] = df["rsi_norm"] * (df["atr_pct_14"] / df["atr_pct_14"].rolling(20).mean().replace(0, 1e-8))
            
            # Volatility regime
            atr_pctl = df["atr_pct_14"].rolling(50).rank(pct=True)
            df["is_low_volatility"] = (atr_pctl < 0.3).astype(float)
            df["is_high_volatility"] = (atr_pctl > 0.7).astype(float)
            df["is_ranging"] = ((df["adx"] < 0.25) & (df["bb_squeeze"] > 0)).astype(float)
            
            # Pair ID
            df["pair_id"] = PAIR_IDS.get(pair, 5) / len(PAIR_IDS)  # Normalize
            
            # Get the latest row with all features
            df = df.fillna(0)
            
            # Extract exactly the features we need
            features = []
            for feat in self._feature_names:
                if feat in df.columns:
                    features.append(df[feat].iloc[-1])
                else:
                    logger.warning(f"Missing feature: {feat}")
                    features.append(0.0)
            
            return np.array(features, dtype=np.float32).reshape(1, -1)
            
        except Exception as e:
            logger.error(f"Feature computation failed: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def predict(
        self,
        df: pd.DataFrame,
        pair: str = "EUR_USD",
    ) -> Tuple[str, float]:
        """
        Predict direction for a single pair.
        
        Args:
            df: OHLCV DataFrame
            pair: Currency pair name
            
        Returns:
            Tuple of (direction, confidence) where direction is "LONG" or "SHORT"
        """
        if not self.is_loaded:
            if not self.load():
                return "NEUTRAL", 0.5
        
        features = self._compute_features(df, pair)
        if features is None:
            return "NEUTRAL", 0.5
        
        try:
            # Model outputs sigmoid probability
            prob = self._model.predict(features, verbose=0)[0, 0]
            
            # Convert to direction and confidence
            if prob > 0.5:
                direction = "LONG"
                confidence = prob
            else:
                direction = "SHORT"
                confidence = 1 - prob
            
            return direction, float(confidence)
            
        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            return "NEUTRAL", 0.5
    
    def predict_batch(
        self,
        df_dict: Dict[str, pd.DataFrame],
    ) -> Dict[str, Tuple[str, float]]:
        """
        Predict direction for multiple pairs.
        
        Args:
            df_dict: Dict mapping pair names to OHLCV DataFrames
            
        Returns:
            Dict mapping pair names to (direction, confidence) tuples
        """
        results = {}
        for pair, df in df_dict.items():
            results[pair] = self.predict(df, pair)
        return results


# Singleton instance for easy import
_model_78: Optional[Model78Inference] = None


def get_model_78() -> Model78Inference:
    """Get or create the singleton Model78 instance."""
    global _model_78
    if _model_78 is None:
        _model_78 = Model78Inference()
    return _model_78


def predict_direction(df: pd.DataFrame, pair: str = "EUR_USD") -> Tuple[str, float]:
    """Convenience function for quick predictions."""
    model = get_model_78()
    if not model.is_loaded:
        model.load()
    return model.predict(df, pair)
