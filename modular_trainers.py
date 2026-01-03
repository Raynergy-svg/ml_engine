"""
Modular Trainers for Specialized Ensemble Models.

Each trainer handles ONE specific model type:
- TCNTrainer: Direction prediction (binary classification)
- XGBoostTrainer: Momentum analysis (regression + classification)
- RandomForestTrainer: Risk assessment (regression + classification)
- RidgeTrainer: Confidence scoring (regression 0-100)

No shared gradients. No joint loss. Each model trains independently.
"""

from __future__ import annotations

import json
import logging
import pickle
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


# =============================================================================
# BASE TRAINER CLASS
# =============================================================================

@dataclass
class TrainerConfig:
    """Configuration for model trainers."""
    # Common
    epochs: int = 100
    batch_size: int = 64
    learning_rate: float = 0.001
    patience: int = 20
    verbose: int = 1
    
    # TCN specific
    tcn_hidden_size: int = 32
    tcn_num_layers: int = 2
    tcn_kernel_size: int = 3
    tcn_dropout: float = 0.3
    
    # Transformer specific (for direction prediction) - reduced capacity to prevent overfitting
    transformer_d_model: int = 16  # Model dimension (reduced from 32)
    transformer_num_heads: int = 2  # Number of attention heads (reduced from 4)
    transformer_num_layers: int = 1  # Number of encoder layers (reduced from 2)
    transformer_dff: int = 32  # Feedforward network dimension (reduced from 64)
    transformer_dropout: float = 0.4  # Dropout rate (increased to reduce overfitting)
    
    # XGBoost specific
    xgb_n_estimators: int = 200
    xgb_max_depth: int = 5
    xgb_learning_rate: float = 0.05
    
    # Random Forest specific
    rf_n_estimators: int = 200
    rf_max_depth: int = 10
    rf_min_samples_leaf: int = 10
    
    # Ridge specific
    ridge_alpha: float = 1.0


class BaseTrainer(ABC):
    """Abstract base class for all modular trainers."""
    
    def __init__(self, config: Optional[TrainerConfig] = None):
        self.config = config or TrainerConfig()
        self.model = None
        self.is_trained = False
        self.metrics: Dict[str, float] = {}
    
    @abstractmethod
    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
    ) -> Dict[str, float]:
        """Train the model and return metrics."""
        pass
    
    @abstractmethod
    def predict(self, X: np.ndarray) -> Dict[str, Any]:
        """Make predictions."""
        pass
    
    @abstractmethod
    def save(self, path: str) -> None:
        """Save model to disk."""
        pass
    
    @abstractmethod
    def load(self, path: str) -> None:
        """Load model from disk."""
        pass


# =============================================================================
# TCN TRAINER - Direction Prediction
# =============================================================================

class TCNTrainer(BaseTrainer):
    """
    TCN model for direction prediction.
    
    Input: Volatility regimes + close-to-close features
    Output: Binary direction (0=down, 1=up)
    """
    
    def __init__(self, config: Optional[TrainerConfig] = None):
        super().__init__(config)
        self.scaler = None
        self.feature_names = None  # Save feature names for inference
    
    def _build_model(self, input_shape: Tuple[int, int]) -> Any:
        """Build TCN model architecture."""
        import tensorflow as tf
        from tensorflow import keras
        
        seq_len, n_features = input_shape
        
        inp = keras.Input(shape=(seq_len, n_features), name="features")
        
        # Add noise for regularization
        x = keras.layers.GaussianNoise(0.02)(inp)
        
        # TCN layers using Conv1D with dilation
        filters = self.config.tcn_hidden_size
        kernel_size = self.config.tcn_kernel_size
        
        for i in range(self.config.tcn_num_layers):
            dilation_rate = 2 ** i
            x = keras.layers.Conv1D(
                filters=filters,
                kernel_size=kernel_size,
                padding='causal',
                dilation_rate=dilation_rate,
                activation='relu',
                name=f'tcn_conv_{i}'
            )(x)
            x = keras.layers.BatchNormalization()(x)
            x = keras.layers.Dropout(self.config.tcn_dropout)(x)
        
        # Global pooling and output
        x = keras.layers.GlobalAveragePooling1D()(x)
        x = keras.layers.Dense(32, activation='relu')(x)
        x = keras.layers.Dropout(self.config.tcn_dropout)(x)
        
        # Binary direction output
        direction = keras.layers.Dense(1, activation='sigmoid', name='direction', dtype='float32')(x)
        
        model = keras.Model(inputs=inp, outputs=direction, name='tcn_direction')
        
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=self.config.learning_rate),
            loss='binary_crossentropy',
            metrics=['accuracy'],
        )
        
        return model
    
    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        feature_names: Optional[list] = None,
    ) -> Dict[str, float]:
        """Train TCN for direction prediction."""
        import tensorflow as tf
        from tensorflow import keras
        from sklearn.preprocessing import StandardScaler
        
        logger.info("Training TCN (Direction)...")
        
        # Save feature names for inference
        self.feature_names = feature_names
        self.n_features = X_train.shape[-1]
        
        # Scale features
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train.reshape(-1, X_train.shape[-1]))
        X_val_scaled = self.scaler.transform(X_val.reshape(-1, X_val.shape[-1]))
        
        # Reshape for sequence input (batch, seq_len, features)
        # Use sliding windows
        seq_len = min(60, len(X_train_scaled) // 10)
        
        def create_sequences(X, y, seq_len):
            X_seq, y_seq = [], []
            for i in range(len(X) - seq_len):
                X_seq.append(X[i:i+seq_len])
                y_seq.append(y[i+seq_len-1])  # Label at end of sequence
            return np.array(X_seq), np.array(y_seq)
        
        X_train_seq, y_train_seq = create_sequences(X_train_scaled, y_train, seq_len)
        X_val_seq, y_val_seq = create_sequences(X_val_scaled, y_val, seq_len)
        
        # Build model
        self.model = self._build_model((seq_len, X_train.shape[-1]))
        
        # Callbacks - use config patience values
        callbacks = [
            # Primary: Stop when validation loss stops improving
            keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=self.config.patience,
                mode='min',
                restore_best_weights=True,
                verbose=1,
            ),
            # LR reduction (less aggressive: factor=0.5, patience=1/4 of early stopping)
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=max(4, self.config.patience // 4),
                min_lr=1e-6,
                verbose=1,
            ),
        ]
        
        # Train
        history = self.model.fit(
            X_train_seq, y_train_seq,
            validation_data=(X_val_seq, y_val_seq),
            epochs=self.config.epochs,
            batch_size=self.config.batch_size,
            callbacks=callbacks,
            verbose=self.config.verbose,
        )
        
        self.is_trained = True
        self.seq_len = seq_len
        
        # Calculate metrics
        val_pred = (self.model.predict(X_val_seq, verbose=0) > 0.5).astype(float)
        val_acc = np.mean(val_pred.flatten() == y_val_seq)
        
        self.metrics = {
            'train_accuracy': float(history.history['accuracy'][-1]),
            'val_accuracy': float(val_acc),
            'epochs_trained': len(history.history['loss']),
        }
        
        logger.info(f"TCN trained: val_accuracy={val_acc:.4f}")
        return self.metrics
    
    def predict(self, X: np.ndarray) -> Dict[str, Any]:
        """Predict direction (0 or 1)."""
        if not self.is_trained:
            raise RuntimeError("Model not trained")
        
        # Scale
        X_scaled = self.scaler.transform(X.reshape(-1, X.shape[-1]))
        
        # Create sequence from last seq_len rows
        if len(X_scaled) >= self.seq_len:
            X_seq = X_scaled[-self.seq_len:].reshape(1, self.seq_len, -1)
        else:
            # Pad with zeros if not enough data
            pad_len = self.seq_len - len(X_scaled)
            X_padded = np.vstack([np.zeros((pad_len, X_scaled.shape[1])), X_scaled])
            X_seq = X_padded.reshape(1, self.seq_len, -1)
        
        prob = float(self.model.predict(X_seq, verbose=0)[0, 0])
        direction = 1 if prob > 0.5 else 0
        
        return {
            'direction': direction,
            'probability': prob,
        }
    
    def save(self, path: str) -> None:
        """Save TCN model and scaler."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save Keras model
        self.model.save(str(path))
        
        # Save scaler and config
        meta = {
            'scaler': self.scaler,
            'seq_len': self.seq_len,
            'metrics': self.metrics,
            'config': self.config.__dict__,
            'feature_names': self.feature_names,
            'n_features': self.n_features,
        }
        meta_path = path.with_suffix('.meta.pkl')
        with open(meta_path, 'wb') as f:
            pickle.dump(meta, f)
        
        logger.info(f"TCN saved to {path}")
    
    def load(self, path: str) -> None:
        """Load TCN model and scaler."""
        from tensorflow import keras
        
        path = Path(path)
        self.model = keras.models.load_model(str(path))
        
        meta_path = path.with_suffix('.meta.pkl')
        with open(meta_path, 'rb') as f:
            meta = pickle.load(f)
        
        self.scaler = meta['scaler']
        self.seq_len = meta['seq_len']
        self.metrics = meta['metrics']
        self.feature_names = meta.get('feature_names')
        self.n_features = meta.get('n_features')
        self.is_trained = True
        
        logger.info(f"TCN loaded from {path}")


# =============================================================================
# TRANSFORMER TRAINER - Direction Prediction (Replacement for TCN)
# =============================================================================

class TransformerDirectionTrainer(BaseTrainer):
    """
    Transformer model for direction prediction.
    
    Self-attention captures long-range dependencies in price trends,
    making it better suited for direction prediction than TCN.
    
    Input: Directional features (ADX, MACD, SMA crosses, market structure)
    Output: Binary direction (0=down, 1=up)
    """
    
    def __init__(self, config: Optional[TrainerConfig] = None):
        super().__init__(config)
        self.scaler = None
        self.feature_names = None
        self.n_features = None
        self.seq_len = None
        
        # Transformer-specific config (use TCN config as base, can be overridden)
        self.transformer_d_model = getattr(config, 'transformer_d_model', 32) if config else 32
        self.transformer_num_heads = getattr(config, 'transformer_num_heads', 4) if config else 4
        self.transformer_num_layers = getattr(config, 'transformer_num_layers', 2) if config else 2
        self.transformer_dff = getattr(config, 'transformer_dff', 64) if config else 64
        self.transformer_dropout = getattr(config, 'transformer_dropout', 0.2) if config else 0.2
    
    def _build_model(self, input_shape: Tuple[int, int]) -> Any:
        """Build Transformer model architecture with strong regularization."""
        import tensorflow as tf
        from tensorflow import keras
        
        seq_len, n_features = input_shape
        
        # L2 regularization to prevent overfitting
        l2_reg = keras.regularizers.l2(0.01)
        
        # Input
        inp = keras.Input(shape=(seq_len, n_features), name="features")
        
        # Add input noise for regularization
        x = keras.layers.GaussianNoise(0.1)(inp)
        
        # Project features to d_model dimension (with L2)
        x = keras.layers.Dense(
            self.transformer_d_model, 
            kernel_regularizer=l2_reg,
            name='input_projection'
        )(x)
        
        # Add positional encoding
        x = self._add_positional_encoding(x, seq_len, self.transformer_d_model)
        
        # Transformer encoder layers
        for i in range(self.transformer_num_layers):
            x = self._transformer_encoder_layer(
                x, 
                self.transformer_d_model, 
                self.transformer_num_heads,
                self.transformer_dff,
                self.transformer_dropout,
                l2_reg,
                name_prefix=f'transformer_{i}'
            )
        
        # Global pooling and output
        x = keras.layers.GlobalAveragePooling1D()(x)
        x = keras.layers.Dense(16, activation='relu', kernel_regularizer=l2_reg)(x)  # Reduced from 32
        x = keras.layers.Dropout(0.5)(x)  # Higher dropout before output
        
        # Binary direction output
        direction = keras.layers.Dense(1, activation='sigmoid', name='direction', dtype='float32')(x)
        
        model = keras.Model(inputs=inp, outputs=direction, name='transformer_direction')
        
        # Use label smoothing to prevent overconfident predictions
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=self.config.learning_rate),
            loss=keras.losses.BinaryCrossentropy(label_smoothing=0.1),
            metrics=['accuracy'],
        )
        
        return model
    
    def _add_positional_encoding(self, x, seq_len: int, d_model: int):
        """Add sinusoidal positional encoding."""
        import tensorflow as tf
        from tensorflow import keras
        
        # Create positional encoding
        positions = np.arange(seq_len)[:, np.newaxis]
        dims = np.arange(d_model)[np.newaxis, :]
        
        angles = positions / np.power(10000, (2 * (dims // 2)) / d_model)
        
        # Apply sin to even indices, cos to odd
        pos_encoding = np.zeros((seq_len, d_model))
        pos_encoding[:, 0::2] = np.sin(angles[:, 0::2])
        pos_encoding[:, 1::2] = np.cos(angles[:, 1::2])
        
        pos_encoding = pos_encoding[np.newaxis, :, :].astype(np.float32)
        
        # Add positional encoding to input
        return x + tf.constant(pos_encoding)
    
    def _transformer_encoder_layer(self, x, d_model: int, num_heads: int, dff: int, 
                                    dropout: float, l2_reg, name_prefix: str):
        """Single transformer encoder layer with multi-head attention and feedforward."""
        from tensorflow import keras
        
        # Multi-head self-attention
        attn_output = keras.layers.MultiHeadAttention(
            num_heads=num_heads,
            key_dim=d_model // num_heads,
            name=f'{name_prefix}_mha'
        )(x, x)
        attn_output = keras.layers.Dropout(dropout)(attn_output)
        x = keras.layers.LayerNormalization(epsilon=1e-6, name=f'{name_prefix}_ln1')(x + attn_output)
        
        # Feedforward network with L2 regularization
        ffn = keras.layers.Dense(dff, activation='relu', kernel_regularizer=l2_reg, name=f'{name_prefix}_ffn1')(x)
        ffn = keras.layers.Dense(d_model, kernel_regularizer=l2_reg, name=f'{name_prefix}_ffn2')(ffn)
        ffn = keras.layers.Dropout(dropout)(ffn)
        x = keras.layers.LayerNormalization(epsilon=1e-6, name=f'{name_prefix}_ln2')(x + ffn)
        
        return x
    
    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        feature_names: Optional[list] = None,
        w_train: Optional[np.ndarray] = None,
        w_val: Optional[np.ndarray] = None,
    ) -> Dict[str, float]:
        """
        Train Transformer for direction prediction.
        
        IMPORTANT: Create sequences FIRST from all data (preserving temporal order),
        THEN filter sequences based on whether the target label is clear.
        This preserves temporal continuity for proper sequence modeling.
        """
        import tensorflow as tf
        from tensorflow import keras
        from sklearn.preprocessing import StandardScaler
        
        logger.info("Training Transformer (Direction)...")
        
        # Save feature names for inference
        self.feature_names = feature_names
        self.n_features = X_train.shape[-1]
        
        # Scale features FIRST (on all data to preserve temporal order)
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train.reshape(-1, X_train.shape[-1]))
        X_val_scaled = self.scaler.transform(X_val.reshape(-1, X_val.shape[-1]))
        
        # Create sequences with sliding windows (BEFORE filtering)
        seq_len = min(60, len(X_train_scaled) // 10)
        self.seq_len = seq_len
        
        def create_sequences_with_weights(X, y, w, seq_len):
            """Create sequences and keep track of weights for filtering."""
            X_seq, y_seq, w_seq = [], [], []
            for i in range(len(X) - seq_len):
                X_seq.append(X[i:i+seq_len])
                y_seq.append(y[i+seq_len-1])  # Label at end of sequence
                if w is not None:
                    w_seq.append(w[i+seq_len-1])  # Weight at end of sequence
                else:
                    w_seq.append(1.0)
            return np.array(X_seq), np.array(y_seq), np.array(w_seq)
        
        X_train_seq, y_train_seq, w_train_seq = create_sequences_with_weights(
            X_train_scaled, y_train, w_train, seq_len
        )
        X_val_seq, y_val_seq, w_val_seq = create_sequences_with_weights(
            X_val_scaled, y_val, w_val, seq_len
        )
        
        # NOW filter sequences based on weights (clear labels only)
        train_clear_mask = w_train_seq > 0
        val_clear_mask = w_val_seq > 0
        
        X_train_filtered = X_train_seq[train_clear_mask]
        y_train_filtered = y_train_seq[train_clear_mask]
        X_val_filtered = X_val_seq[val_clear_mask]
        y_val_filtered = y_val_seq[val_clear_mask]
        
        logger.info(f"Filtered training: {train_clear_mask.sum()}/{len(train_clear_mask)} sequences with clear labels")
        logger.info(f"Filtered validation: {val_clear_mask.sum()}/{len(val_clear_mask)} sequences with clear labels")
        
        # Log class distribution
        train_up_pct = (y_train_filtered == 1).mean() * 100
        val_up_pct = (y_val_filtered == 1).mean() * 100
        logger.info(f"Class distribution: train={train_up_pct:.1f}% up, val={val_up_pct:.1f}% up")
        
        # Calculate class weights to handle imbalance
        n_up = (y_train_filtered == 1).sum()
        n_down = (y_train_filtered == 0).sum()
        if n_up > 0 and n_down > 0:
            # Inverse frequency weighting
            total = n_up + n_down
            class_weight = {
                0: total / (2 * n_down),
                1: total / (2 * n_up),
            }
            logger.info(f"Class weights: down={class_weight[0]:.3f}, up={class_weight[1]:.3f}")
        else:
            class_weight = None
            logger.warning("Cannot compute class weights - one class has zero samples")
        
        logger.info(f"Sequence shape: train={X_train_filtered.shape}, val={X_val_filtered.shape}")
        
        # Build model
        self.model = self._build_model((seq_len, self.n_features))
        
        # Print model summary
        self.model.summary(print_fn=logger.info)
        
        # Callbacks - use config patience values
        # Key insight: val_loss is the most reliable metric for detecting overfitting
        # If val_loss increases while train_loss decreases, we're overfitting
        callbacks = [
            # Primary: Stop when validation loss stops improving (best for overfitting)
            keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=self.config.patience,
                mode='min',
                restore_best_weights=True,  # Restore weights from best epoch
                verbose=1,
            ),
            # LR reduction (less aggressive: factor=0.5, patience=1/4 of early stopping)
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=max(4, self.config.patience // 4),
                min_lr=1e-6,
                verbose=1,
            ),
        ]
        
        # Train on FILTERED sequences (clear labels only) with class weighting
        history = self.model.fit(
            X_train_filtered, y_train_filtered,
            validation_data=(X_val_filtered, y_val_filtered),
            epochs=self.config.epochs,
            batch_size=self.config.batch_size,
            callbacks=callbacks,
            verbose=self.config.verbose,
            class_weight=class_weight,
        )
        
        self.is_trained = True
        
        # Calculate metrics on FILTERED validation (clear labels only)
        val_pred = (self.model.predict(X_val_filtered, verbose=0) > 0.5).astype(float)
        val_acc = np.mean(val_pred.flatten() == y_val_filtered)
        
        # Calculate balanced accuracy
        y_true = y_val_filtered.flatten()
        y_pred = val_pred.flatten()
        up_acc = np.mean(y_pred[y_true == 1] == 1) if (y_true == 1).sum() > 0 else 0
        down_acc = np.mean(y_pred[y_true == 0] == 0) if (y_true == 0).sum() > 0 else 0
        balanced_acc = (up_acc + down_acc) / 2
        
        self.metrics = {
            'train_accuracy': float(history.history['accuracy'][-1]),
            'val_accuracy': float(val_acc),
            'val_balanced_accuracy': float(balanced_acc),
            'val_up_accuracy': float(up_acc),
            'val_down_accuracy': float(down_acc),
            'epochs_trained': len(history.history['loss']),
            'n_train_samples': len(X_train_filtered),
            'n_val_samples': len(X_val_filtered),
        }
        
        logger.info(f"Transformer trained: val_accuracy={val_acc:.4f}, "
                   f"balanced_acc={balanced_acc:.4f} (up={up_acc:.4f}, down={down_acc:.4f})")
        return self.metrics
    
    def predict(self, X: np.ndarray) -> Dict[str, Any]:
        """Predict direction (0 or 1) with probability."""
        if not self.is_trained:
            raise RuntimeError("Model not trained")
        
        # Scale
        X_scaled = self.scaler.transform(X.reshape(-1, X.shape[-1]))
        
        # Create sequence from last seq_len rows
        if len(X_scaled) >= self.seq_len:
            X_seq = X_scaled[-self.seq_len:].reshape(1, self.seq_len, -1)
        else:
            # Pad with zeros if not enough data
            pad_len = self.seq_len - len(X_scaled)
            X_padded = np.vstack([np.zeros((pad_len, X_scaled.shape[1])), X_scaled])
            X_seq = X_padded.reshape(1, self.seq_len, -1)
        
        prob = float(self.model.predict(X_seq, verbose=0)[0, 0])
        direction = 1 if prob > 0.5 else 0
        
        return {
            'direction': direction,
            'probability': prob,
        }
    
    def save(self, path: str) -> None:
        """Save Transformer model and scaler."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save Keras model
        self.model.save(str(path))
        
        # Save scaler and config
        meta = {
            'scaler': self.scaler,
            'seq_len': self.seq_len,
            'metrics': self.metrics,
            'config': self.config.__dict__,
            'feature_names': self.feature_names,
            'n_features': self.n_features,
            'model_type': 'transformer',
        }
        meta_path = path.with_suffix('.meta.pkl')
        with open(meta_path, 'wb') as f:
            pickle.dump(meta, f)
        
        logger.info(f"Transformer saved to {path}")
    
    def load(self, path: str) -> None:
        """Load Transformer model and scaler."""
        from tensorflow import keras
        
        path = Path(path)
        self.model = keras.models.load_model(str(path))
        
        meta_path = path.with_suffix('.meta.pkl')
        with open(meta_path, 'rb') as f:
            meta = pickle.load(f)
        
        self.scaler = meta['scaler']
        self.seq_len = meta['seq_len']
        self.metrics = meta['metrics']
        self.feature_names = meta.get('feature_names')
        self.n_features = meta.get('n_features')
        self.is_trained = True
        
        logger.info(f"Transformer loaded from {path}")


# =============================================================================
# TRANSFORMER REGIME TRAINER - Market Regime Classification (3 classes)
# =============================================================================

class TransformerRegimeTrainer(BaseTrainer):
    """
    Transformer model for market regime classification (3 classes).
    
    The Transformer acts as a "bouncer" - it determines WHAT KIND of market we're in,
    not which direction to trade. This is a much more tractable problem than direction.
    
    Regimes:
    - 0 = TREND: Strong directional movement, let gates decide direction
    - 1 = CHOP: Sideways noise, skip trading entirely  
    - 2 = MEAN_REVERT: Overextended, fade 2-bar momentum
    
    Input: Regime indicators (ADX, RSI, volatility, z-scores, etc.)
    Output: Softmax over 3 classes
    """
    
    def __init__(self, config: Optional[TrainerConfig] = None):
        super().__init__(config)
        self.scaler = None
        self.feature_names = None
        self.n_features = None
        self.seq_len = None
        self.class_names = ['trend', 'chop', 'mean_revert']
        
        # Transformer hyperparameters
        self.d_model = getattr(config, 'transformer_d_model', 32) if config else 32
        self.num_heads = getattr(config, 'transformer_num_heads', 4) if config else 4
        self.ff_dim = getattr(config, 'transformer_ff_dim', 64) if config else 64
        self.num_blocks = getattr(config, 'transformer_num_blocks', 2) if config else 2
        self.transformer_dropout = getattr(config, 'transformer_dropout', 0.2) if config else 0.2
    
    def _build_model(self, input_shape: Tuple[int, int]) -> Any:
        """Build Transformer model for 3-class regime classification."""
        import tensorflow as tf
        from tensorflow import keras
        
        seq_len, n_features = input_shape
        
        inp = keras.Input(shape=(seq_len, n_features), name="features")
        
        # Project input to d_model dimensions
        x = keras.layers.Dense(self.d_model, name='input_projection')(inp)
        
        # Add positional encoding
        x = self._add_positional_encoding(x, seq_len, self.d_model)
        
        # Transformer encoder blocks
        for i in range(self.num_blocks):
            x = self._transformer_encoder_layer(
                x, 
                d_model=self.d_model,
                num_heads=self.num_heads,
                dff=self.ff_dim,
                dropout=self.transformer_dropout,
                name_prefix=f'transformer_{i}'
            )
        
        # Global pooling and output
        x = keras.layers.GlobalAveragePooling1D()(x)
        x = keras.layers.Dense(32, activation='relu')(x)
        x = keras.layers.Dropout(self.transformer_dropout)(x)
        
        # 3-class regime output (softmax)
        regime = keras.layers.Dense(3, activation='softmax', name='regime', dtype='float32')(x)
        
        model = keras.Model(inputs=inp, outputs=regime, name='transformer_regime')
        
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=self.config.learning_rate),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy'],
        )
        
        return model
    
    def _add_positional_encoding(self, x, seq_len: int, d_model: int):
        """Add sinusoidal positional encoding."""
        import tensorflow as tf
        
        positions = np.arange(seq_len)[:, np.newaxis]
        dims = np.arange(d_model)[np.newaxis, :]
        
        angles = positions / np.power(10000, (2 * (dims // 2)) / d_model)
        
        pos_encoding = np.zeros((seq_len, d_model))
        pos_encoding[:, 0::2] = np.sin(angles[:, 0::2])
        pos_encoding[:, 1::2] = np.cos(angles[:, 1::2])
        
        pos_encoding = pos_encoding[np.newaxis, :, :].astype(np.float32)
        
        return x + tf.constant(pos_encoding)
    
    def _transformer_encoder_layer(self, x, d_model: int, num_heads: int, dff: int,
                                    dropout: float, name_prefix: str):
        """Single transformer encoder layer."""
        from tensorflow import keras
        
        # Multi-head self-attention
        attn_output = keras.layers.MultiHeadAttention(
            num_heads=num_heads,
            key_dim=d_model // num_heads,
            name=f'{name_prefix}_mha'
        )(x, x)
        attn_output = keras.layers.Dropout(dropout)(attn_output)
        x = keras.layers.LayerNormalization(epsilon=1e-6, name=f'{name_prefix}_ln1')(x + attn_output)
        
        # Feedforward network
        ffn = keras.layers.Dense(dff, activation='relu', name=f'{name_prefix}_ffn1')(x)
        ffn = keras.layers.Dense(d_model, name=f'{name_prefix}_ffn2')(ffn)
        ffn = keras.layers.Dropout(dropout)(ffn)
        x = keras.layers.LayerNormalization(epsilon=1e-6, name=f'{name_prefix}_ln2')(x + ffn)
        
        return x
    
    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        feature_names: Optional[list] = None,
        class_names: Optional[list] = None,
    ) -> Dict[str, float]:
        """
        Train Transformer for 3-class regime classification.
        Reports F1 score (macro) as primary metric.
        """
        import tensorflow as tf
        from tensorflow import keras
        from sklearn.preprocessing import StandardScaler
        from sklearn.metrics import f1_score, classification_report
        from sklearn.utils.class_weight import compute_class_weight
        
        logger.info("Training Transformer (Regime Classification)...")
        
        self.feature_names = feature_names
        self.n_features = X_train.shape[-1]
        if class_names:
            self.class_names = class_names
        
        # Scale features
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train.reshape(-1, X_train.shape[-1]))
        X_val_scaled = self.scaler.transform(X_val.reshape(-1, X_val.shape[-1]))
        
        # Create sequences
        seq_len = min(60, len(X_train_scaled) // 10)
        self.seq_len = seq_len
        
        def create_sequences(X, y, seq_len):
            X_seq, y_seq = [], []
            for i in range(len(X) - seq_len):
                X_seq.append(X[i:i+seq_len])
                y_seq.append(y[i+seq_len-1])
            return np.array(X_seq), np.array(y_seq)
        
        X_train_seq, y_train_seq = create_sequences(X_train_scaled, y_train, seq_len)
        X_val_seq, y_val_seq = create_sequences(X_val_scaled, y_val, seq_len)
        
        # Log class distribution
        unique, counts = np.unique(y_train_seq, return_counts=True)
        class_dist = dict(zip(unique, counts))
        logger.info(f"Training class distribution: {class_dist}")
        
        unique_val, counts_val = np.unique(y_val_seq, return_counts=True)
        class_dist_val = dict(zip(unique_val, counts_val))
        logger.info(f"Validation class distribution: {class_dist_val}")
        
        logger.info(f"Sequence shape: train={X_train_seq.shape}, val={X_val_seq.shape}")
        
        # Compute class weights for imbalanced classes
        classes = np.unique(y_train_seq)
        if len(classes) > 1:
            weights = compute_class_weight('balanced', classes=classes, y=y_train_seq)
            class_weight = {int(c): w for c, w in zip(classes, weights)}
            logger.info(f"Class weights: {class_weight}")
        else:
            class_weight = None
        
        # Build model
        self.model = self._build_model((seq_len, self.n_features))
        self.model.summary(print_fn=logger.info)
        
        # Callbacks - use config patience values
        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=self.config.patience,
                mode='min',
                restore_best_weights=True,
                verbose=1,
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=max(4, self.config.patience // 4),
                min_lr=1e-6,
                verbose=1,
            ),
        ]
        
        # Train
        history = self.model.fit(
            X_train_seq, y_train_seq,
            validation_data=(X_val_seq, y_val_seq),
            epochs=self.config.epochs,
            batch_size=self.config.batch_size,
            callbacks=callbacks,
            verbose=self.config.verbose,
            class_weight=class_weight,
        )
        
        self.is_trained = True
        
        # Calculate metrics
        val_pred_probs = self.model.predict(X_val_seq, verbose=0)
        val_pred = np.argmax(val_pred_probs, axis=1)
        
        # F1 score (macro - treats all classes equally)
        f1_macro = f1_score(y_val_seq, val_pred, average='macro')
        f1_weighted = f1_score(y_val_seq, val_pred, average='weighted')
        
        # Per-class F1
        f1_per_class = f1_score(y_val_seq, val_pred, average=None)
        
        # Accuracy
        val_acc = np.mean(val_pred == y_val_seq)
        
        # Classification report
        report = classification_report(y_val_seq, val_pred, target_names=self.class_names)
        logger.info(f"\nClassification Report:\n{report}")
        
        self.metrics = {
            'train_accuracy': float(history.history['accuracy'][-1]),
            'val_accuracy': float(val_acc),
            'f1_macro': float(f1_macro),
            'f1_weighted': float(f1_weighted),
            'f1_trend': float(f1_per_class[0]) if len(f1_per_class) > 0 else 0.0,
            'f1_chop': float(f1_per_class[1]) if len(f1_per_class) > 1 else 0.0,
            'f1_mean_revert': float(f1_per_class[2]) if len(f1_per_class) > 2 else 0.0,
            'epochs_trained': len(history.history['loss']),
            'n_train_samples': len(X_train_seq),
            'n_val_samples': len(X_val_seq),
        }
        
        logger.info(f"Regime Transformer trained: val_acc={val_acc:.4f}, F1_macro={f1_macro:.4f}")
        logger.info(f"  F1 per class: trend={self.metrics['f1_trend']:.3f}, "
                   f"chop={self.metrics['f1_chop']:.3f}, mean_revert={self.metrics['f1_mean_revert']:.3f}")
        
        return self.metrics
    
    def predict(self, X: np.ndarray) -> Dict[str, Any]:
        """
        Predict regime (0=trend, 1=chop, 2=mean_revert) with probabilities.
        """
        if not self.is_trained:
            raise RuntimeError("Model not trained")
        
        # Scale
        X_scaled = self.scaler.transform(X.reshape(-1, X.shape[-1]))
        
        # Create sequence from last seq_len rows
        if len(X_scaled) >= self.seq_len:
            X_seq = X_scaled[-self.seq_len:].reshape(1, self.seq_len, -1)
        else:
            # Pad with zeros if not enough data
            pad_len = self.seq_len - len(X_scaled)
            X_padded = np.vstack([np.zeros((pad_len, X_scaled.shape[-1])), X_scaled])
            X_seq = X_padded.reshape(1, self.seq_len, -1)
        
        # Predict
        probs = self.model.predict(X_seq, verbose=0)[0]
        regime = int(np.argmax(probs))
        
        return {
            'regime': regime,
            'regime_name': self.class_names[regime],
            'prob_trend': float(probs[0]),
            'prob_chop': float(probs[1]),
            'prob_mean_revert': float(probs[2]),
            'confidence': float(np.max(probs)),
        }
    
    def save(self, path: str) -> None:
        """Save Transformer regime model and scaler."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        self.model.save(str(path))
        
        meta = {
            'scaler': self.scaler,
            'seq_len': self.seq_len,
            'metrics': self.metrics,
            'config': self.config.__dict__,
            'feature_names': self.feature_names,
            'n_features': self.n_features,
            'class_names': self.class_names,
            'model_type': 'transformer_regime',
        }
        meta_path = path.with_suffix('.meta.pkl')
        with open(meta_path, 'wb') as f:
            pickle.dump(meta, f)
        
        logger.info(f"Transformer Regime saved to {path}")
    
    def load(self, path: str) -> None:
        """Load Transformer regime model and scaler."""
        from tensorflow import keras
        
        path = Path(path)
        self.model = keras.models.load_model(str(path))
        
        meta_path = path.with_suffix('.meta.pkl')
        with open(meta_path, 'rb') as f:
            meta = pickle.load(f)
        
        self.scaler = meta['scaler']
        self.seq_len = meta['seq_len']
        self.metrics = meta['metrics']
        self.feature_names = meta.get('feature_names')
        self.n_features = meta.get('n_features')
        self.class_names = meta.get('class_names', ['trend', 'chop', 'mean_revert'])
        self.is_trained = True
        
        logger.info(f"Transformer Regime loaded from {path}")


# =============================================================================
# XGBOOST TRAINER - Momentum Analysis
# =============================================================================

class XGBoostTrainer(BaseTrainer):
    """
    XGBoost model for momentum analysis.
    
    Input: Lagged returns + spread dynamics
    Output: 
        - momentum_score (0-1): How fast price is moving
        - acceleration (bool): Is momentum growing?
    """
    
    def __init__(self, config: Optional[TrainerConfig] = None):
        super().__init__(config)
        self.momentum_model = None
        self.accel_model = None
        self.scaler = None
        self.feature_names = None
        self.n_features = None
        self.momentum_norm_factor = None  # Saved from training for reference
    
    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        feature_names: Optional[list] = None,
        momentum_norm_factor: Optional[float] = None,
    ) -> Dict[str, float]:
        """Train XGBoost for momentum analysis (2 models)."""
        self.momentum_norm_factor = momentum_norm_factor
        try:
            import xgboost as xgb
        except ImportError:
            raise ImportError("XGBoost not installed. Run: pip install xgboost")
        
        from sklearn.preprocessing import StandardScaler
        
        logger.info("Training XGBoost (Momentum)...")
        
        # Save feature names for inference
        self.feature_names = feature_names
        self.n_features = X_train.shape[-1]
        
        # Scale features
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_val_scaled = self.scaler.transform(X_val)
        
        # Split targets: y[:, 0] = momentum_score, y[:, 1] = acceleration
        y_train_momentum = y_train[:, 0]
        y_train_accel = y_train[:, 1].astype(int)
        y_val_momentum = y_val[:, 0]
        y_val_accel = y_val[:, 1].astype(int)
        
        # Train momentum regressor
        self.momentum_model = xgb.XGBRegressor(
            n_estimators=self.config.xgb_n_estimators,
            max_depth=self.config.xgb_max_depth,
            learning_rate=self.config.xgb_learning_rate,
            verbosity=0,
            n_jobs=-1,
            random_state=42,
        )
        self.momentum_model.fit(
            X_train_scaled, y_train_momentum,
            eval_set=[(X_val_scaled, y_val_momentum)],
            verbose=False,
        )
        
        # Train acceleration classifier
        self.accel_model = xgb.XGBClassifier(
            n_estimators=self.config.xgb_n_estimators,
            max_depth=self.config.xgb_max_depth,
            learning_rate=self.config.xgb_learning_rate,
            verbosity=0,
            n_jobs=-1,
            random_state=42,
        )
        self.accel_model.fit(
            X_train_scaled, y_train_accel,
            eval_set=[(X_val_scaled, y_val_accel)],
            verbose=False,
        )
        
        self.is_trained = True
        
        # Calculate metrics
        momentum_pred = self.momentum_model.predict(X_val_scaled)
        accel_pred = self.accel_model.predict(X_val_scaled)
        
        momentum_mae = float(np.mean(np.abs(momentum_pred - y_val_momentum)))
        accel_acc = float(np.mean(accel_pred == y_val_accel))
        
        self.metrics = {
            'momentum_mae': momentum_mae,
            'acceleration_accuracy': accel_acc,
        }
        
        logger.info(f"XGBoost trained: momentum_mae={momentum_mae:.4f}, accel_acc={accel_acc:.4f}")
        return self.metrics
    
    def predict(self, X: np.ndarray) -> Dict[str, Any]:
        """Predict momentum score and acceleration."""
        if not self.is_trained:
            raise RuntimeError("Model not trained")
        
        # Scale
        if X.ndim == 1:
            X = X.reshape(1, -1)
        X_scaled = self.scaler.transform(X)
        
        # Get last row for prediction
        X_last = X_scaled[-1:] if len(X_scaled) > 1 else X_scaled
        
        # DEBUG: Log scaled features once
        if not hasattr(self, '_predict_debug_logged'):
            logger.info(f"XGB Predict - Input shape: {X.shape}, Scaled shape: {X_scaled.shape}")
            logger.info(f"XGB Predict - X_last scaled: {X_last.flatten()}")
            self._predict_debug_logged = True
        
        momentum = float(self.momentum_model.predict(X_last)[0])
        acceleration = bool(self.accel_model.predict(X_last)[0])
        
        # Clamp momentum to 0-1
        momentum = max(0.0, min(1.0, momentum))
        
        return {
            'momentum': momentum,
            'acceleration': acceleration,
        }
    
    def save(self, path: str) -> None:
        """Save XGBoost models."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        data = {
            'momentum_model': self.momentum_model,
            'accel_model': self.accel_model,
            'scaler': self.scaler,
            'metrics': self.metrics,
            'config': self.config.__dict__,
            'feature_names': self.feature_names,
            'n_features': self.n_features,
            'momentum_norm_factor': self.momentum_norm_factor,
        }
        
        with open(path, 'wb') as f:
            pickle.dump(data, f)
        
        logger.info(f"XGBoost saved to {path}")
    
    def load(self, path: str) -> None:
        """Load XGBoost models."""
        with open(path, 'rb') as f:
            data = pickle.load(f)
        
        # DEBUG: Log scaler info
        scaler = data.get('scaler')
        if scaler is not None:
            logger.info(f"XGB Scaler - mean_: {scaler.mean_}")
            logger.info(f"XGB Scaler - scale_: {scaler.scale_}")
        
        self.momentum_model = data['momentum_model']
        self.accel_model = data['accel_model']
        self.scaler = data['scaler']
        self.metrics = data['metrics']
        self.feature_names = data.get('feature_names')
        self.n_features = data.get('n_features')
        self.is_trained = True
        
        logger.info(f"XGBoost loaded from {path}")


# =============================================================================
# RANDOM FOREST TRAINER - Risk Assessment
# =============================================================================

class RandomForestTrainer(BaseTrainer):
    """
    Random Forest model for risk assessment.
    
    Input: ATR, historical drawdowns, streak patterns
    Output:
        - expected_drawdown_pips: Max adverse excursion in next N bars
        - streak_prob: Probability losing streak continues
    """
    
    def __init__(self, config: Optional[TrainerConfig] = None):
        super().__init__(config)
        self.drawdown_model = None
        self.streak_model = None
        self.scaler = None
        self.feature_names = None
        self.n_features = None
    
    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        feature_names: Optional[list] = None,
    ) -> Dict[str, float]:
        """Train Random Forest for risk assessment (2 models)."""
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.preprocessing import StandardScaler
        
        logger.info("Training Random Forest (Risk)...")
        
        # Save feature names for inference
        self.feature_names = feature_names
        self.n_features = X_train.shape[-1]
        
        # Scale features
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_val_scaled = self.scaler.transform(X_val)
        
        # Split targets: y[:, 0] = drawdown_pips, y[:, 1] = streak_prob
        y_train_drawdown = y_train[:, 0]
        y_train_streak = y_train[:, 1]
        y_val_drawdown = y_val[:, 0]
        y_val_streak = y_val[:, 1]
        
        # Train drawdown regressor
        self.drawdown_model = RandomForestRegressor(
            n_estimators=self.config.rf_n_estimators,
            max_depth=self.config.rf_max_depth,
            min_samples_leaf=self.config.rf_min_samples_leaf,
            n_jobs=-1,
            random_state=42,
        )
        self.drawdown_model.fit(X_train_scaled, y_train_drawdown)
        
        # Train streak probability regressor (0-1 range, so regression not classification)
        self.streak_model = RandomForestRegressor(
            n_estimators=self.config.rf_n_estimators,
            max_depth=self.config.rf_max_depth,
            min_samples_leaf=self.config.rf_min_samples_leaf,
            n_jobs=-1,
            random_state=42,
        )
        self.streak_model.fit(X_train_scaled, y_train_streak)
        
        self.is_trained = True
        
        # Calculate metrics
        drawdown_pred = self.drawdown_model.predict(X_val_scaled)
        streak_pred = self.streak_model.predict(X_val_scaled)
        
        drawdown_mae = float(np.mean(np.abs(drawdown_pred - y_val_drawdown)))
        streak_mae = float(np.mean(np.abs(streak_pred - y_val_streak)))
        
        # Convert to basis points for meaningful display (0.001 = 10 bps)
        drawdown_mae_bps = drawdown_mae * 10000
        
        self.metrics = {
            'drawdown_mae_pct': drawdown_mae,  # Raw percentage (0-1)
            'drawdown_mae_bps': drawdown_mae_bps,  # Basis points for display
            'streak_prob_mae': streak_mae,
        }
        
        logger.info(f"RF trained: drawdown_mae={drawdown_mae_bps:.1f} bps ({drawdown_mae*100:.3f}%), streak_mae={streak_mae:.4f}")
        return self.metrics
    
    def predict(self, X: np.ndarray) -> Dict[str, Any]:
        """Predict expected drawdown and streak probability."""
        if not self.is_trained:
            raise RuntimeError("Model not trained")
        
        # Scale
        if X.ndim == 1:
            X = X.reshape(1, -1)
        X_scaled = self.scaler.transform(X)
        
        # Get last row for prediction
        X_last = X_scaled[-1:] if len(X_scaled) > 1 else X_scaled
        
        # Model outputs drawdown as PERCENTAGE (instrument-agnostic)
        expected_drawdown_pct = float(self.drawdown_model.predict(X_last)[0])
        streak_prob = float(self.streak_model.predict(X_last)[0])
        
        # Clamp values
        expected_drawdown_pct = max(0.0, min(1.0, expected_drawdown_pct))  # 0-100% range
        streak_prob = max(0.0, min(1.0, streak_prob))
        
        return {
            'expected_drawdown_pct': expected_drawdown_pct,
            # Keep legacy key for backward compatibility
            'expected_drawdown_pips': expected_drawdown_pct * 10000,  # Rough conversion for display
            'streak_prob': streak_prob,
        }
    
    def save(self, path: str) -> None:
        """Save Random Forest models."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        data = {
            'drawdown_model': self.drawdown_model,
            'streak_model': self.streak_model,
            'scaler': self.scaler,
            'metrics': self.metrics,
            'config': self.config.__dict__,
            'feature_names': self.feature_names,
            'n_features': self.n_features,
        }
        
        with open(path, 'wb') as f:
            pickle.dump(data, f)
        
        logger.info(f"Random Forest saved to {path}")
    
    def load(self, path: str) -> None:
        """Load Random Forest models."""
        with open(path, 'rb') as f:
            data = pickle.load(f)
        
        self.drawdown_model = data['drawdown_model']
        self.streak_model = data['streak_model']
        self.scaler = data['scaler']
        self.metrics = data['metrics']
        self.feature_names = data.get('feature_names')
        self.n_features = data.get('n_features')
        self.is_trained = True
        
        logger.info(f"Random Forest loaded from {path}")


# =============================================================================
# RIDGE TRAINER - Confidence Scoring
# =============================================================================

class RidgeTrainer(BaseTrainer):
    """
    Ridge regression model for confidence scoring.
    
    Input: Rolling variance, volume changes
    Output: Confidence score (0-100)
    """
    
    def __init__(self, config: Optional[TrainerConfig] = None):
        super().__init__(config)
        self.scaler = None
        self.feature_names = None
        self.n_features = None
    
    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        feature_names: Optional[list] = None,
    ) -> Dict[str, float]:
        """Train Ridge for confidence scoring."""
        from sklearn.linear_model import Ridge
        from sklearn.preprocessing import StandardScaler
        
        logger.info("Training Ridge (Confidence)...")
        
        # Save feature names for inference
        self.feature_names = feature_names
        self.n_features = X_train.shape[-1]
        
        # Scale features
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_val_scaled = self.scaler.transform(X_val)
        
        # Train Ridge regressor
        self.model = Ridge(alpha=self.config.ridge_alpha)
        self.model.fit(X_train_scaled, y_train)
        
        self.is_trained = True
        
        # Calculate metrics
        y_pred = self.model.predict(X_val_scaled)
        mae = float(np.mean(np.abs(y_pred - y_val)))
        r2 = float(self.model.score(X_val_scaled, y_val))
        
        self.metrics = {
            'confidence_mae': mae,
            'r2_score': r2,
        }
        
        logger.info(f"Ridge trained: confidence_mae={mae:.2f}, r2={r2:.4f}")
        return self.metrics
    
    def predict(self, X: np.ndarray) -> Dict[str, Any]:
        """Predict confidence score (0-100)."""
        if not self.is_trained:
            raise RuntimeError("Model not trained")
        
        # Scale
        if X.ndim == 1:
            X = X.reshape(1, -1)
        X_scaled = self.scaler.transform(X)
        
        # Get last row for prediction
        X_last = X_scaled[-1:] if len(X_scaled) > 1 else X_scaled
        
        confidence = float(self.model.predict(X_last)[0])
        
        # Clamp to 0-100
        confidence = max(0.0, min(100.0, confidence))
        
        return {
            'confidence': confidence,
        }
    
    def save(self, path: str) -> None:
        """Save Ridge model."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        data = {
            'model': self.model,
            'scaler': self.scaler,
            'metrics': self.metrics,
            'config': self.config.__dict__,
            'feature_names': self.feature_names,
            'n_features': self.n_features,
        }
        
        with open(path, 'wb') as f:
            pickle.dump(data, f)
        
        logger.info(f"Ridge saved to {path}")
    
    def load(self, path: str) -> None:
        """Load Ridge model."""
        with open(path, 'rb') as f:
            data = pickle.load(f)
        
        self.model = data['model']
        self.scaler = data['scaler']
        self.metrics = data['metrics']
        self.feature_names = data.get('feature_names')
        self.n_features = data.get('n_features')
        self.is_trained = True
        
        logger.info(f"Ridge loaded from {path}")


# =============================================================================
# CONVENIENCE FUNCTION - Train All Models
# =============================================================================

def train_all_modular(
    data: Dict[str, Dict[str, np.ndarray]],
    config: Optional[TrainerConfig] = None,
    save_dir: str = "trained_data/models",
    use_transformer: bool = True,
    use_regime: bool = False,
) -> Dict[str, BaseTrainer]:
    """
    Train all 4 models independently.
    
    Args:
        data: Dict from load_all_modular_data() with 'direction'/'regime', 'xgboost', 'rf', 'ridge' keys
        config: Optional trainer configuration
        save_dir: Directory to save models
        use_transformer: If True, use Transformer; if False, use TCN (only for direction mode)
        use_regime: If True, train regime classifier instead of direction predictor
    
    Returns:
        Dict with trained trainer instances
    """
    config = config or TrainerConfig()
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    trainers = {}
    
    # 1. Regime or Direction Model
    logger.info("\n" + "="*50)
    
    if use_regime:
        # REGIME MODE: 3-class classification (trend/chop/mean_revert)
        logger.info("Training Transformer (REGIME Classifier - 3 classes)")
        logger.info("  Classes: trend, chop, mean_revert")
        logger.info("="*50)
        
        regime_data = data.get('regime')
        if regime_data is None:
            raise ValueError("No regime data found. Set use_regime=True in load_all_modular_data()")
        
        regime_trainer = TransformerRegimeTrainer(config)
        regime_trainer.train(
            regime_data['X_train'], regime_data['y_train'],
            regime_data['X_val'], regime_data['y_val'],
            feature_names=regime_data.get('feature_names'),
            class_names=regime_data.get('class_names'),
        )
        regime_trainer.save(str(save_dir / "transformer_regime.keras"))
        trainers['regime'] = regime_trainer
        trainers['transformer'] = regime_trainer  # Alias
        
    else:
        # DIRECTION MODE: Binary classification (legacy)
        if use_transformer:
            logger.info("Training Transformer (Direction Predictor)")
        else:
            logger.info("Training TCN (Direction Predictor)")
        logger.info("="*50)
        
        # Get direction data (try 'direction' key first, fallback to 'tcn')
        dir_data = data.get('direction', data.get('tcn'))
        if dir_data is None:
            raise ValueError("No direction data found (tried 'direction' and 'tcn' keys)")
        
        if use_transformer:
            dir_trainer = TransformerDirectionTrainer(config)
            dir_trainer.train(
                dir_data['X_train'], dir_data['y_train'],
                dir_data['X_val'], dir_data['y_val'],
                feature_names=dir_data.get('feature_names'),
                w_train=dir_data.get('w_train'),
                w_val=dir_data.get('w_val'),
            )
            dir_trainer.save(str(save_dir / "transformer_direction.keras"))
            trainers['direction'] = dir_trainer
            trainers['transformer'] = dir_trainer  # Alias
        else:
            dir_trainer = TCNTrainer(config)
            dir_trainer.train(
                dir_data['X_train'], dir_data['y_train'],
                dir_data['X_val'], dir_data['y_val'],
                feature_names=dir_data.get('feature_names'),
            )
            dir_trainer.save(str(save_dir / "tcn_direction.keras"))
            trainers['direction'] = dir_trainer
            trainers['tcn'] = dir_trainer  # Alias
    
    # 2. XGBoost
    logger.info("\n" + "="*50)
    logger.info("Training XGBoost (Momentum Analyzer)")
    logger.info("="*50)
    xgb_data = data['xgboost']
    xgb_trainer = XGBoostTrainer(config)
    xgb_trainer.train(
        xgb_data['X_train'], xgb_data['y_train'],
        xgb_data['X_val'], xgb_data['y_val'],
        feature_names=xgb_data.get('feature_names'),
    )
    xgb_trainer.save(str(save_dir / "xgb_momentum.pkl"))
    trainers['xgboost'] = xgb_trainer
    
    # 3. Random Forest
    logger.info("\n" + "="*50)
    logger.info("Training Random Forest (Risk Assessor)")
    logger.info("="*50)
    rf_data = data['rf']
    rf_trainer = RandomForestTrainer(config)
    rf_trainer.train(
        rf_data['X_train'], rf_data['y_train'],
        rf_data['X_val'], rf_data['y_val'],
        feature_names=rf_data.get('feature_names'),
    )
    rf_trainer.save(str(save_dir / "rf_risk.pkl"))
    trainers['rf'] = rf_trainer
    
    # 4. Ridge
    logger.info("\n" + "="*50)
    logger.info("Training Ridge (Confidence Scorer)")
    logger.info("="*50)
    ridge_data = data['ridge']
    ridge_trainer = RidgeTrainer(config)
    ridge_trainer.train(
        ridge_data['X_train'], ridge_data['y_train'],
        ridge_data['X_val'], ridge_data['y_val'],
        feature_names=ridge_data.get('feature_names'),
    )
    ridge_trainer.save(str(save_dir / "ridge_confidence.pkl"))
    trainers['ridge'] = ridge_trainer
    
    logger.info("\n" + "="*50)
    logger.info("All 4 models trained independently!")
    logger.info("="*50)
    
    return trainers

