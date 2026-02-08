"""
Multi-Pair Ensemble Inference for Scanner.

Uses pair-specific models for optimal accuracy per instrument.
Each pair has its own trained model (transformer + gates) achieving 57%+ accuracy,
compared to a single multi-pair model stuck at 50%.

Usage:
    from multi_pair_inference import MultiPairInference
    
    model = MultiPairInference()
    model.load()
    
    # Predict for a specific pair
    direction, confidence = model.predict(df, "EUR_USD")
    
    # Get status of all loaded models
    model.get_model_status()
"""

import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import pickle

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# Default pairs with trained models
DEFAULT_PAIRS = [
    "EUR_USD", "GBP_USD", "USD_JPY", "AUD_USD", 
    "USD_CAD", "NZD_USD", "USD_CHF"
]


class PairModel:
    """Container for a single pair's model and gates."""
    
    def __init__(self, pair: str, base_path: Path):
        self.pair = pair
        self.base_path = base_path
        
        # Model components
        self.transformer = None       # Raw Keras model (kept for backward compat)
        self.trainer = None           # TransformerDirectionTrainer (preferred predict path)
        self.xgb_momentum = None
        self.rf_risk = None
        self.ridge_confidence = None
        self.scaler = None
        self.meta = None
        self.feature_names = []
        self.seq_len = 60
        
        # Metrics
        self.accuracy = 0.0
        self.is_loaded = False
        
    def load(self) -> bool:
        """Load all model components for this pair.
        
        Uses TransformerDirectionTrainer for the Keras model to ensure:
        - Custom layers are properly registered before loading
        - Multi-strategy loading (standard, tf.keras, safe_mode, rebuild)
        - Output calibration and EMA weights are loaded
        - Consistent prediction behavior with modular_inference.py
        """
        try:
            pair_dir = self.base_path / self.pair
            if not pair_dir.exists():
                logger.debug(f"No model directory for {self.pair}")
                return False
            
            # Load transformer using TransformerDirectionTrainer
            # This ensures custom Keras layers are registered and all loading
            # strategies (standard, tf.keras, safe_mode, rebuild) are tried
            tf_path = pair_dir / "transformer_direction.keras"
            if not tf_path.exists():
                # Try checkpoints
                ckpt_path = pair_dir / "checkpoints" / "transformer_direction_best.keras"
                if ckpt_path.exists():
                    tf_path = ckpt_path
                else:
                    logger.warning(f"No transformer for {self.pair}")
                    return False
            
            # Import here to avoid circular imports and ensure custom layers registered
            from src.training.modular_trainers import TransformerDirectionTrainer
            self.trainer = TransformerDirectionTrainer()
            self.trainer.load(str(tf_path), instrument=self.pair)
            self.transformer = self.trainer.model  # Keep backward compat reference
            
            # Get feature names, scaler, seq_len from the trainer (authoritative source)
            self.feature_names = self.trainer.feature_names or []
            self.seq_len = self.trainer.seq_len or 60
            self.scaler = self.trainer.scaler
            
            # Get accuracy from trainer metrics
            if hasattr(self.trainer, 'metrics') and self.trainer.metrics:
                self.accuracy = self.trainer.metrics.get(
                    'best_val_accuracy',
                    self.trainer.metrics.get('val_accuracy', 0.55)
                )
            
            # Load gate models
            xgb_path = pair_dir / "xgb_momentum.pkl"
            if xgb_path.exists():
                with open(xgb_path, 'rb') as f:
                    self.xgb_momentum = pickle.load(f)
            
            rf_path = pair_dir / "rf_risk.pkl"
            if rf_path.exists():
                with open(rf_path, 'rb') as f:
                    self.rf_risk = pickle.load(f)
            
            ridge_path = pair_dir / "ridge_confidence.pkl"
            if ridge_path.exists():
                with open(ridge_path, 'rb') as f:
                    self.ridge_confidence = pickle.load(f)
            
            # Load meta for additional info
            meta_path = pair_dir / "transformer_direction.meta.pkl"
            if meta_path.exists():
                with open(meta_path, 'rb') as f:
                    self.meta = pickle.load(f)
                # Override accuracy from meta if trainer didn't have it
                if self.accuracy <= 0.55:
                    metrics = self.meta.get('metrics', {})
                    self.accuracy = metrics.get(
                        'best_val_accuracy',
                        metrics.get('val_accuracy', 0.55))
            
            self.is_loaded = True
            logger.debug(f"Loaded model for {self.pair}: {self.accuracy:.1%}")
            return True
            
        except Exception as e:
            err_msg = str(e)
            # Truncate enormous Keras deserialization errors (full model JSON)
            if len(err_msg) > 200:
                err_msg = err_msg[:200] + "..."
            logger.warning(f"Failed to load model for {self.pair}: {err_msg}")
            return False


class MultiPairInference:
    """
    Multi-pair inference using pair-specific models.
    
    Falls back gracefully if a pair's model is not available.
    """
    
    def __init__(
        self,
        models_dir: str = "trained_data/models",
        pairs: Optional[List[str]] = None,
    ):
        self.models_dir = Path(models_dir)
        self.pairs = pairs or DEFAULT_PAIRS
        
        # Loaded models per pair
        self._pair_models: Dict[str, PairModel] = {}
        
        # Fallback model (for pairs without specific models)
        self._fallback_model = None
        
        # Feature engineering
        self._feature_engineer = None
        
    def load(self) -> bool:
        """Load all available pair models."""
        import io
        loaded = 0
        
        for pair in self.pairs:
            pm = PairModel(pair, self.models_dir)
            # Suppress stderr during Keras model loading to avoid
            # massive JSON config dumps on deserialization failures
            old_stderr = sys.stderr
            try:
                sys.stderr = io.StringIO()
                success = pm.load()
            finally:
                sys.stderr = old_stderr
            if success:
                self._pair_models[pair] = pm
                loaded += 1
        
        logger.debug(f"MultiPairInference: loaded {loaded}/{len(self.pairs)} pair models")
        return loaded > 0
    
    def get_model_status(self) -> Dict[str, Any]:
        """Get status of all models."""
        status = {
            "total_pairs": len(self.pairs),
            "loaded_pairs": len(self._pair_models),
            "models": {}
        }
        
        for pair in self.pairs:
            if pair in self._pair_models:
                pm = self._pair_models[pair]
                status["models"][pair] = {
                    "loaded": True,
                    "accuracy": pm.accuracy,
                    "has_transformer": pm.transformer is not None,
                    "has_xgb": pm.xgb_momentum is not None,
                    "has_rf": pm.rf_risk is not None,
                    "has_ridge": pm.ridge_confidence is not None,
                }
            else:
                status["models"][pair] = {"loaded": False}
        
        return status
    
    def _init_feature_engineer(self):
        """Initialize feature engineering."""
        if self._feature_engineer is not None:
            return
        
        try:
            from feature_engineering import FeatureEngineering
            self._feature_engineer = FeatureEngineering({})
        except ImportError:
            pass
    
    def _compute_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute features for prediction.
        
        Combines features from multiple sources:
        1. compute_normalized_features (macd_norm, rsi_norm, etc.)
        2. FeatureEngineering (temporal, session, advanced features)
        """
        df_feat = df.copy()
        
        # First: Apply normalized features (these provide macd_norm, rsi_norm, etc.)
        try:
            from modular_data_loaders import compute_normalized_features
            df_norm = compute_normalized_features(df.copy())
            # Copy normalized features to main dataframe
            for col in df_norm.columns:
                if col not in df_feat.columns:
                    df_feat[col] = df_norm[col].values
        except Exception as e:
            logger.debug(f"Normalized features failed: {e}")
        
        # Second: Apply FeatureEngineering (adds temporal, session, advanced features)
        self._init_feature_engineer()
        if self._feature_engineer:
            try:
                df_fe = self._feature_engineer.create_features(df.copy(), include_all=True)
                # Copy FE features that aren't already present
                for col in df_fe.columns:
                    if col not in df_feat.columns:
                        if len(df_fe[col]) == len(df_feat):
                            df_feat[col] = df_fe[col].values
            except Exception as e:
                logger.debug(f"FeatureEngineering failed: {e}")
        
        # Clean any NaN/inf values
        df_feat = df_feat.replace([np.inf, -np.inf], np.nan).ffill().bfill().fillna(0)
        
        return df_feat
    
    def _compute_basic_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute basic features for inference."""
        df = df.copy()
        close = df["close"]
        
        # Returns
        df["returns_1"] = close.pct_change(1)
        df["returns_5"] = close.pct_change(5)
        df["returns_10"] = close.pct_change(10)
        
        # RSI
        delta = close.diff()
        gain = delta.where(delta > 0, 0).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss.replace(0, 1e-8)
        df["rsi"] = 100 - (100 / (1 + rs))
        
        # MACD
        ema12 = close.ewm(span=12).mean()
        ema26 = close.ewm(span=26).mean()
        df["macd"] = ema12 - ema26
        df["macd_signal"] = df["macd"].ewm(span=9).mean()
        df["macd_hist"] = df["macd"] - df["macd_signal"]
        
        # ATR
        high, low = df["high"], df["low"]
        tr = pd.concat([
            high - low,
            (high - close.shift(1)).abs(),
            (low - close.shift(1)).abs()
        ], axis=1).max(axis=1)
        df["atr"] = tr.rolling(14).mean()
        
        # Clean NaN
        df = df.replace([np.inf, -np.inf], np.nan).ffill().bfill().fillna(0)
        
        return df
    
    def predict(
        self, 
        df: pd.DataFrame, 
        pair: str,
        df_feat: Optional[pd.DataFrame] = None,
    ) -> Tuple[str, float]:
        """
        Predict direction and confidence for a pair.
        
        Args:
            df: OHLCV DataFrame (raw data)
            pair: Currency pair name
            df_feat: Pre-computed features (optional, avoids recomputation)
            
        Returns:
            Tuple of (direction: "LONG"/"SHORT", confidence: 0-1)
        """
        # Use pair-specific model if available
        if pair in self._pair_models:
            pm = self._pair_models[pair]
            return self._predict_with_pair_model(df, pm, df_feat=df_feat)
        
        # Fallback to 78% model
        if self._fallback_model is not None:
            try:
                direction, prob = self._fallback_model.predict(df, pair)
                return direction, prob
            except Exception as e:
                logger.warning(f"Fallback model failed: {e}")
        
        # Ultimate fallback: technical indicators
        return self._predict_technical(df)
    
    def _predict_with_pair_model(
        self, 
        df: pd.DataFrame, 
        pm: PairModel,
        df_feat: Optional[pd.DataFrame] = None,
    ) -> Tuple[str, float]:
        """Predict using a pair-specific model with proper feature handling.
        
        Uses TransformerDirectionTrainer.predict() when available for consistency
        with modular_inference.py (same calibration, EMA weights, thresholds).
        """
        try:
            # Use pre-computed features if available, otherwise compute
            if df_feat is not None and len(df_feat) > 0:
                df_features = df_feat
            else:
                df_features = self._compute_features(df)
            
            # Get model's expected features and scaler from meta
            feature_names = getattr(pm, 'feature_names', None) or []
            seq_len = getattr(pm, 'seq_len', 60)
            
            # Check if we have enough data
            if len(df_features) < seq_len:
                logger.warning(f"Not enough data: {len(df_features)} < {seq_len}")
                return "HOLD", 0.5
            
            # Build feature matrix with CORRECT feature names in order
            if feature_names:
                # Use saved feature names - reorder/select columns
                available = df_features.columns.tolist()
                missing_features = []
                X_list = []
                
                for fname in feature_names:
                    if fname in available:
                        X_list.append(df_features[fname].iloc[-seq_len:].values)
                    else:
                        # Feature not available - fill with zeros
                        missing_features.append(fname)
                        X_list.append(np.zeros(seq_len))
                
                if missing_features and len(missing_features) < 20:  # Log if not too many
                    logger.debug(f"Missing features for {pm.pair}: {missing_features[:5]}...")
                
                # Stack features: shape (seq_len, n_features)
                X = np.column_stack(X_list)
            else:
                # Fallback: use first N columns from numeric
                n_features = pm.transformer.input_shape[-1] if pm.transformer else 80
                numeric_cols = df_features.select_dtypes(include=[np.number]).columns.tolist()
                if len(numeric_cols) >= n_features:
                    X = df_features[numeric_cols[:n_features]].iloc[-seq_len:].values
                else:
                    X = np.zeros((seq_len, n_features))
                    X[:, :len(numeric_cols)] = df_features[numeric_cols].iloc[-seq_len:].values
            
            # Clean input
            X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
            
            # === USE TRAINER PREDICT for consistency with modular_inference ===
            # This ensures: calibrated thresholds, EMA weights, proper scaling
            if pm.trainer is not None and pm.trainer.is_trained:
                pred_result = pm.trainer.predict(X)
                prob = pred_result['probability']
                threshold = pred_result.get('threshold', 0.5)
                direction = "LONG" if pred_result['direction'] == 1 else "SHORT"
                confidence = pred_result.get('confidence', abs(prob - threshold) * 2)
                
                # Boost confidence based on model's historical accuracy
                confidence = min(confidence * (1 + (pm.accuracy - 0.5)), 0.95)
                
                return direction, confidence
            
            # === FALLBACK: Raw model prediction (legacy path) ===
            scaler = getattr(pm, 'scaler', None)
            if scaler is not None:
                try:
                    X = scaler.transform(X)
                except Exception as e:
                    logger.warning(f"Scaler transform failed for {pm.pair}: {e}")
            
            # Reshape for model: (batch=1, seq_len, features)
            X = X.reshape(1, seq_len, -1).astype(np.float32)
            
            # Predict
            pred = pm.transformer.predict(X, verbose=0)
            
            # Parse output
            if pred.shape[-1] == 1:
                prob = float(pred[0, 0])
            else:
                prob = float(pred[0, 1]) if pred.shape[-1] >= 2 else float(pred[0, 0])
            
            direction = "LONG" if prob > 0.5 else "SHORT"
            confidence = abs(prob - 0.5) * 2  # Scale to 0-1
            
            # Boost confidence based on model's historical accuracy
            confidence = min(confidence * (1 + (pm.accuracy - 0.5)), 0.95)
            
            return direction, confidence
            
        except Exception as e:
            logger.warning(f"Pair model prediction failed for {pm.pair}: {e}")
            return self._predict_technical(df)
    
    def _predict_technical(self, df: pd.DataFrame) -> Tuple[str, float]:
        """Fallback prediction using technical indicators."""
        try:
            df_feat = self._compute_basic_features(df)
            
            rsi = df_feat["rsi"].iloc[-1]
            macd = df_feat["macd"].iloc[-1]
            macd_hist = df_feat["macd_hist"].iloc[-1]
            
            # RSI signals
            if rsi < 30:
                return "LONG", 0.55 + (30 - rsi) / 100
            elif rsi > 70:
                return "SHORT", 0.55 + (rsi - 70) / 100
            
            # MACD signals
            if macd_hist > 0 and macd > 0:
                return "LONG", 0.52
            elif macd_hist < 0 and macd < 0:
                return "SHORT", 0.52
            
            return "HOLD", 0.5
            
        except Exception:
            return "HOLD", 0.5
    
    def predict_all(
        self, 
        data: Dict[str, pd.DataFrame],
    ) -> Dict[str, Tuple[str, float]]:
        """
        Predict for multiple pairs.
        
        Args:
            data: Dict mapping pair names to DataFrames
            
        Returns:
            Dict mapping pair names to (direction, confidence)
        """
        results = {}
        for pair, df in data.items():
            results[pair] = self.predict(df, pair)
        return results
    
    def get_best_pair(
        self,
        data: Dict[str, pd.DataFrame],
        min_confidence: float = 0.55,
    ) -> Optional[Tuple[str, str, float]]:
        """
        Get the best trading opportunity across all pairs.
        
        Args:
            data: Dict mapping pair names to DataFrames
            min_confidence: Minimum confidence threshold
            
        Returns:
            Tuple of (pair, direction, confidence) or None
        """
        predictions = self.predict_all(data)
        
        best = None
        best_conf = min_confidence
        
        for pair, (direction, conf) in predictions.items():
            if direction != "HOLD" and conf > best_conf:
                best = (pair, direction, conf)
                best_conf = conf
        
        return best


def test_multi_pair_inference():
    """Test the multi-pair inference system."""
    logging.basicConfig(level=logging.INFO)
    
    model = MultiPairInference()
    model.load()
    
    print(f"\n{'='*60}")
    print("Multi-Pair Inference Status")
    print('='*60)
    
    status = model.get_model_status()
    print(f"Total pairs: {status['total_pairs']}")
    print(f"Loaded: {status['loaded_pairs']}")
    print(f"Fallback available: {status['fallback_available']}")
    
    print("\nPer-pair status:")
    for pair, info in status["models"].items():
        if info.get("loaded"):
            acc = info.get("accuracy", 0)
            print(f"  {pair}: ✓ {acc:.1%}")
        else:
            print(f"  {pair}: ✗ not loaded")
    
    return model


if __name__ == "__main__":
    test_multi_pair_inference()
