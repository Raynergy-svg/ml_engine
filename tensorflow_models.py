"""
TensorFlow/Keras model implementations for time series prediction.
Includes state-of-the-art architectures with TensorBoard visualization support.

Models:
- TFStockPredictor: Enhanced LSTM with residual connections
- TFAttentiveLSTM: LSTM with multi-head self-attention
- TFTransformerPredictor: Pure Transformer for time series
- TFTemporalFusionTransformer: State-of-the-art for multi-feature forecasting
- TFTCNPredictor: Temporal Convolutional Network (faster than LSTM)
- TFEnsemblePredictor: Ensemble of multiple models

M1 Metal Optimizations:
- All models support mixed precision (float16 compute, float32 output)
- Prefer TCN over LSTM (parallelizable, better Metal utilization)
- Use BatchNormalization over LayerNormalization where possible (faster on Metal)
- Avoid recurrent_dropout > 0.2 on Metal (can cause slowdowns)
- Use jit_compile=True in model.compile() for XLA acceleration
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model, regularizers
from tensorflow.keras.saving import register_keras_serializable
import numpy as np
from typing import List, Optional
import platform

# =============================================================================
# M1 Metal Optimization Utilities
# =============================================================================

def is_apple_silicon() -> bool:
    """Check if running on Apple Silicon."""
    return platform.system() == "Darwin" and platform.machine() == "arm64"

def get_compute_dtype():
    """Get optimal compute dtype for current hardware."""
    # On M1 Metal with mixed precision, compute in float16 but keep outputs float32
    policy = tf.keras.mixed_precision.global_policy()
    return policy.compute_dtype if policy.compute_dtype else "float32"

# M1 Metal best practices:
# 1. Avoid excessive recurrent_dropout (can slow down Metal GPU)
# 2. Prefer Conv1D (TCN) over LSTM when possible (better parallelization)
# 3. Use batch sizes of 64-256 (optimal for Metal unified memory)
# 4. Use jit_compile=True for XLA optimization

# Default regularization strength for overfitting prevention
DEFAULT_L2_REG = 0.001
# M1 Metal: Keep recurrent_dropout low (≤0.15) to avoid GPU slowdowns
DEFAULT_RECURRENT_DROPOUT = 0.1 if is_apple_silicon() else 0.15

# M1 Metal optimal batch sizes (for unified memory architecture)
M1_OPTIMAL_BATCH_SIZES = [64, 128, 256]  # Powers of 2 work best


# =============================================================================
# Custom Layers
# =============================================================================

@register_keras_serializable()
class Mish(layers.Layer):
    """Mish activation: x * tanh(softplus(x)) - smoother than ReLU."""
    
    def call(self, x):
        return x * tf.math.tanh(tf.math.softplus(x))


@register_keras_serializable()
class Swish(layers.Layer):
    """Swish activation: x * sigmoid(x) - also known as SiLU."""
    
    def call(self, x):
        return x * tf.math.sigmoid(x)


@register_keras_serializable()
class GatedLinearUnit(layers.Layer):
    """GLU activation for Temporal Fusion Transformer."""
    
    def __init__(self, units: int, **kwargs):
        super().__init__(**kwargs)
        self.units = units
        self.linear = layers.Dense(units * 2)
    
    def call(self, x):
        output = self.linear(x)
        return output[..., :self.units] * tf.math.sigmoid(output[..., self.units:])


@register_keras_serializable()
class GatedResidualNetwork(layers.Layer):
    """
    Gated Residual Network (GRN) - Core building block of TFT.
    Applies non-linear processing with skip connections and gating.
    """
    
    def __init__(
        self,
        hidden_size: int,
        output_size: Optional[int] = None,
        dropout: float = 0.1,
        use_time_distributed: bool = False,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.hidden_size = hidden_size
        self.output_size = output_size or hidden_size
        self.dropout_rate = dropout
        self.use_time_distributed = use_time_distributed
    
    def build(self, input_shape):
        self.dense1 = layers.Dense(self.hidden_size, activation='elu')
        self.dense2 = layers.Dense(self.hidden_size)
        self.dropout = layers.Dropout(self.dropout_rate)
        self.glu = GatedLinearUnit(self.output_size)
        self.layer_norm = layers.LayerNormalization()
        self.context_proj = layers.Dense(self.hidden_size)
        
        # Skip connection projection if needed
        input_dim = input_shape[-1]
        if input_dim != self.output_size:
            self.skip_proj = layers.Dense(self.output_size)
        else:
            self.skip_proj = None
        
        super().build(input_shape)
    
    def call(self, x, context=None, training=None):
        # Skip connection
        skip = self.skip_proj(x) if self.skip_proj else x
        
        # Non-linear processing
        hidden = self.dense1(x)
        if context is not None:
            hidden = hidden + self.context_proj(context)
        hidden = self.dense2(hidden)
        hidden = self.dropout(hidden, training=training)
        
        # Gated output with residual
        gated = self.glu(hidden)
        return self.layer_norm(skip + gated)


@register_keras_serializable()
class VariableSelectionNetwork(layers.Layer):
    """
    Variable Selection Network for TFT.
    Learns which features are most important for prediction.
    """
    
    def __init__(
        self,
        num_features: int,
        hidden_size: int,
        dropout: float = 0.1,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.num_features = num_features
        self.hidden_size = hidden_size
        self.dropout_rate = dropout
    
    def build(self, input_shape):
        # Feature-wise GRNs
        self.feature_grns = [
            GatedResidualNetwork(self.hidden_size, dropout=self.dropout_rate)
            for _ in range(self.num_features)
        ]
        
        # Softmax weights GRN
        self.weight_grn = GatedResidualNetwork(
            self.hidden_size, 
            output_size=self.num_features,
            dropout=self.dropout_rate
        )
        
        super().build(input_shape)
    
    def call(self, x, training=None):
        # x shape: [batch, time, features] or [batch, features]
        
        # Flatten for weight calculation
        if len(x.shape) == 3:
            batch_size, _, _ = tf.shape(x)[0], tf.shape(x)[1], x.shape[-1]
            flattened = tf.reshape(x, [batch_size, -1])
        else:
            flattened = x
        
        # Calculate variable selection weights
        weights = self.weight_grn(flattened, training=training)
        weights = tf.nn.softmax(weights, axis=-1)
        
        # Apply GRN to each feature and weight
        if len(x.shape) == 3:
            # Process each feature across time
            processed_features = []
            for i, grn in enumerate(self.feature_grns):
                feature_slice = x[..., i:i+1]
                processed = grn(feature_slice, training=training)
                processed_features.append(processed)
            
            # Stack and apply weights
            stacked = tf.stack(processed_features, axis=-1)  # [batch, time, hidden, features]
            weights_expanded = weights[:, tf.newaxis, tf.newaxis, :]  # [batch, 1, 1, features]
            selected = tf.reduce_sum(stacked * weights_expanded, axis=-1)
        else:
            processed_features = []
            for i, grn in enumerate(self.feature_grns):
                feature_slice = x[..., i:i+1]
                processed = grn(feature_slice, training=training)
                processed_features.append(processed)
            
            stacked = tf.stack(processed_features, axis=-1)
            weights_expanded = weights[:, tf.newaxis, :]
            selected = tf.reduce_sum(stacked * weights_expanded, axis=-1)
        
        return selected, weights


@register_keras_serializable()
class InterpretableMultiHeadAttention(layers.Layer):
    """
    Interpretable Multi-Head Attention for TFT.
    Provides attention weights for interpretability.
    """
    
    def __init__(self, num_heads: int, key_dim: int, dropout: float = 0.0, **kwargs):
        super().__init__(**kwargs)
        self.num_heads = num_heads
        self.key_dim = key_dim
        self.dropout_rate = dropout
    
    def build(self, input_shape):
        self.mha = layers.MultiHeadAttention(
            num_heads=self.num_heads,
            key_dim=self.key_dim,
            dropout=self.dropout_rate
        )
        super().build(input_shape)
    
    def call(self, query=None, value=None, key=None, training=None, return_attention=False):
        if query is None or value is None:
            raise ValueError("query and value must be provided")
        if key is None:
            key = value
        
        if return_attention:
            output, attention_weights = self.mha(
                query, value, key, 
                training=training,
                return_attention_scores=True
            )
            return output, attention_weights
        
        return self.mha(query, value, key, training=training)


@register_keras_serializable()
class PositionalEncoding(layers.Layer):
    """Sinusoidal positional encoding for Transformer models."""
    
    def __init__(self, max_len: int = 5000, **kwargs):
        super().__init__(**kwargs)
        self.max_len = max_len
    
    def build(self, input_shape):
        d_model = input_shape[-1]
        
        position = np.arange(self.max_len)[:, np.newaxis]
        div_term = np.exp(np.arange(0, d_model, 2) * -(np.log(10000.0) / d_model))
        
        pe = np.zeros((self.max_len, d_model))
        pe[:, 0::2] = np.sin(position * div_term)
        pe[:, 1::2] = np.cos(position * div_term)
        
        self.pe = tf.constant(pe[np.newaxis, :, :], dtype=tf.float32)
        super().build(input_shape)
    
    def call(self, x):
        seq_len = tf.shape(x)[1]
        return x + self.pe[:, :seq_len, :]


# =============================================================================
# Model Implementations
# =============================================================================

@register_keras_serializable()
class TFStockPredictor(Model):
    """
    Enhanced LSTM-based predictor with residual connections and layer normalization.
    TensorFlow equivalent of PyTorch StockPredictor.
    
    Enhanced with:
    - L2 kernel regularization on Dense layers
    - RecurrentDropout for LSTM layers
    - GaussianNoise for input augmentation
    - SpatialDropout1D for feature dropout
    """
    
    def __init__(
        self,
        input_size: int = 7,
        hidden_size: int = 128,
        num_layers: int = 3,
        dropout: float = 0.2,
        recurrent_dropout: float = DEFAULT_RECURRENT_DROPOUT,
        kernel_regularizer: float = DEFAULT_L2_REG,
        bidirectional: bool = False,
        use_layer_norm: bool = True,
        activation: str = "relu",
        noise_std: float = 0.05,
        **kwargs
    ):
        super().__init__(**kwargs)
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.use_layer_norm = use_layer_norm
        self.noise_std = noise_std
        
        # L2 regularizer for Dense layers
        l2_reg = regularizers.L2(kernel_regularizer) if kernel_regularizer > 0 else None
        
        # Input augmentation layers (training-time regularization)
        self.gaussian_noise = layers.GaussianNoise(noise_std)
        self.spatial_dropout = layers.SpatialDropout1D(dropout * 0.5)
        
        # Build LSTM stack with RecurrentDropout
        self.lstm_layers = []
        for i in range(num_layers):
            return_sequences = (i < num_layers - 1)  # Only last layer returns final state
            
            if bidirectional:
                lstm = layers.Bidirectional(
                    layers.LSTM(
                        hidden_size, 
                        return_sequences=return_sequences, 
                        dropout=dropout,
                        recurrent_dropout=recurrent_dropout,
                        kernel_regularizer=l2_reg
                    )
                )
            else:
                lstm = layers.LSTM(
                    hidden_size, 
                    return_sequences=return_sequences, 
                    dropout=dropout,
                    recurrent_dropout=recurrent_dropout,
                    kernel_regularizer=l2_reg
                )
            
            self.lstm_layers.append(lstm)
        
        if use_layer_norm:
            self.layer_norm = layers.LayerNormalization()
        
        self.dropout = layers.Dropout(dropout)
        
        # Dense layers with L2 regularization
        self.fc1 = layers.Dense(128, kernel_regularizer=l2_reg)
        self.fc2 = layers.Dense(64, kernel_regularizer=l2_reg)
        self.fc3 = layers.Dense(32, kernel_regularizer=l2_reg)
        self.fc4 = layers.Dense(1)
        
        # Skip connection
        self.skip = layers.Dense(1)
        
        # Activation
        if activation == "mish":
            self.activation = Mish()
        elif activation == "swish":
            self.activation = Swish()
        elif activation == "gelu":
            self.activation = layers.Activation('gelu')
        else:
            self.activation = layers.ReLU()
    
    def call(self, x, training=None):
        # Input augmentation (only during training)
        x = self.gaussian_noise(x, training=training)
        x = self.spatial_dropout(x, training=training)
        
        # LSTM stack
        lstm_out = x
        for lstm in self.lstm_layers:
            lstm_out = lstm(lstm_out, training=training)
        
        if self.use_layer_norm:
            lstm_out = self.layer_norm(lstm_out)
        
        lstm_out = self.dropout(lstm_out, training=training)
        
        # Skip connection
        skip_out = self.skip(lstm_out)
        
        # Main path
        out = self.fc1(lstm_out)
        out = self.activation(out)
        out = self.dropout(out, training=training)
        
        out = self.fc2(out)
        out = self.activation(out)
        out = self.dropout(out, training=training)
        
        out = self.fc3(out)
        out = self.activation(out)
        out = self.fc4(out)
        
        # Residual connection
        return out + skip_out


@register_keras_serializable()
class TFAttentiveLSTM(Model):
    """
    LSTM with multi-head self-attention for enhanced sequence modeling.
    TensorFlow equivalent of PyTorch AttentiveLSTM.
    
    Enhanced with:
    - L2 kernel regularization
    - RecurrentDropout for LSTM
    - GaussianNoise and SpatialDropout1D
    """
    
    def __init__(
        self,
        input_size: int = 7,
        hidden_size: int = 128,
        num_layers: int = 3,
        num_heads: int = 4,
        dropout: float = 0.2,
        recurrent_dropout: float = DEFAULT_RECURRENT_DROPOUT,
        kernel_regularizer: float = DEFAULT_L2_REG,
        bidirectional: bool = False,
        noise_std: float = 0.05,
        multi_task: bool = False,
        state_classes: int = 3,
        **kwargs
    ):
        super().__init__(**kwargs)
        
        self.multi_task = multi_task
        
        # L2 regularizer
        l2_reg = regularizers.L2(kernel_regularizer) if kernel_regularizer > 0 else None
        
        lstm_output_size = hidden_size * 2 if bidirectional else hidden_size
        
        # Input augmentation layers
        self.gaussian_noise = layers.GaussianNoise(noise_std)
        self.spatial_dropout = layers.SpatialDropout1D(dropout * 0.5)
        
        # LSTM layers with RecurrentDropout (all return sequences for attention)
        self.lstm_layers = []
        for _ in range(num_layers):
            if bidirectional:
                lstm = layers.Bidirectional(
                    layers.LSTM(
                        hidden_size, 
                        return_sequences=True, 
                        dropout=dropout,
                        recurrent_dropout=recurrent_dropout,
                        kernel_regularizer=l2_reg
                    )
                )
            else:
                lstm = layers.LSTM(
                    hidden_size, 
                    return_sequences=True, 
                    dropout=dropout,
                    recurrent_dropout=recurrent_dropout,
                    kernel_regularizer=l2_reg
                )
            self.lstm_layers.append(lstm)
        
        # Multi-head attention
        self.attention = layers.MultiHeadAttention(
            num_heads=num_heads,
            key_dim=lstm_output_size // num_heads,
            dropout=dropout
        )
        
        self.layer_norm1 = layers.LayerNormalization()
        self.layer_norm2 = layers.LayerNormalization()
        
        self.dropout = layers.Dropout(dropout)
        
        # Feed-forward network with L2 regularization
        self.ffn = keras.Sequential([
            layers.Dense(lstm_output_size * 4, activation='gelu', kernel_regularizer=l2_reg),
            layers.Dropout(dropout),
            layers.Dense(lstm_output_size, kernel_regularizer=l2_reg),
            layers.Dropout(dropout),
        ])
        
        # Output layers with L2 regularization
        if multi_task:
            # Price prediction head
            self.price_head = keras.Sequential([
                layers.Dense(64, activation='relu', kernel_regularizer=l2_reg),
                layers.Dropout(dropout),
                layers.Dense(1, name='price_output')
            ], name='price_head')
            
            # Trend prediction head (continuous return)
            self.trend_head = keras.Sequential([
                layers.Dense(64, activation='relu', kernel_regularizer=l2_reg),
                layers.Dropout(dropout),
                layers.Dense(32, kernel_regularizer=l2_reg),
                layers.Activation('relu'),
                layers.Dropout(dropout * 0.5),
                layers.Dense(1, name='trend_output')
            ], name='trend_head')
            
            # Direction classification head (binary: up/down)
            self.direction_head = keras.Sequential([
                layers.Dense(64, activation='relu', kernel_regularizer=l2_reg),
                layers.Dropout(dropout),
                layers.Dense(32, kernel_regularizer=l2_reg),
                layers.Activation('relu'),
                layers.Dropout(dropout * 0.5),
                layers.Dense(1, activation='sigmoid', name='direction_output')
            ], name='direction_head')
            
            # Risk prediction head (sigmoid for 0-1 output)
            self.risk_head = keras.Sequential([
                layers.Dense(32, activation='relu', kernel_regularizer=l2_reg),
                layers.Dropout(dropout),
                layers.Dense(1, activation='sigmoid', name='risk_output')
            ], name='risk_head')
            
            # Market state classification head
            self.state_head = keras.Sequential([
                layers.Dense(32, activation='relu', kernel_regularizer=l2_reg),
                layers.Dropout(dropout),
                layers.Dense(state_classes, activation='softmax', name='state_output')
            ], name='state_head')
        else:
            self.fc = keras.Sequential([
                layers.Dense(64, activation='relu', kernel_regularizer=l2_reg),
                layers.Dropout(dropout),
                layers.Dense(1)
            ])
    
    def call(self, x, training=None):
        # Input augmentation (only during training)
        x = self.gaussian_noise(x, training=training)
        x = self.spatial_dropout(x, training=training)
        
        # LSTM stack
        lstm_out = x
        for lstm in self.lstm_layers:
            lstm_out = lstm(lstm_out, training=training)
        
        # Self-attention with residual
        residual = lstm_out
        lstm_out = self.layer_norm1(lstm_out)
        attn_out = self.attention(lstm_out, lstm_out, training=training)
        lstm_out = residual + self.dropout(attn_out, training=training)
        
        # FFN with residual
        residual = lstm_out
        lstm_out = self.layer_norm2(lstm_out)
        lstm_out = residual + self.ffn(lstm_out, training=training)
        
        # Take last timestep
        lstm_out = lstm_out[:, -1, :]
        
        if self.multi_task:
            price = self.price_head(lstm_out, training=training)
            trend = self.trend_head(lstm_out, training=training)
            direction = self.direction_head(lstm_out, training=training)
            risk = self.risk_head(lstm_out, training=training)
            state_logits = self.state_head(lstm_out, training=training)
            
            return {
                'price': price,
                'trend': trend,
                'direction': direction,
                'risk': risk,
                'state_logits': state_logits,
            }
        else:
            return self.fc(lstm_out, training=training)


@register_keras_serializable()
class TFTransformerPredictor(Model):
    """
    Pure Transformer for time series prediction.
    TensorFlow equivalent of PyTorch TransformerPredictor.
    """
    
    def __init__(
        self,
        input_size: int = 7,
        hidden_size: int = 128,
        num_layers: int = 3,
        num_heads: int = 8,
        dropout: float = 0.2,
        positional_encoding: str = "sinusoidal",
        activation: str = "gelu",
        **kwargs
    ):
        super().__init__(**kwargs)
        
        self.hidden_size = hidden_size
        self.positional_encoding_type = positional_encoding
        
        # Input projection
        self.input_projection = layers.Dense(hidden_size)
        
        # Positional encoding
        if positional_encoding == "learned":
            self.pos_embedding = layers.Embedding(1000, hidden_size)
        else:
            self.pos_encoding = PositionalEncoding(max_len=1000)
        
        # Transformer encoder layers
        self.encoder_layers = []
        for _ in range(num_layers):
            self.encoder_layers.append(
                TransformerEncoderLayer(
                    d_model=hidden_size,
                    num_heads=num_heads,
                    dff=hidden_size * 4,
                    dropout=dropout,
                    activation=activation
                )
            )
        
        # Output layers
        if activation == "mish":
            act_fn = Mish()
        elif activation == "swish":
            act_fn = Swish()
        else:
            act_fn = layers.Activation('gelu')
        
        self.fc = keras.Sequential([
            layers.Dense(64),
            act_fn,
            layers.Dropout(dropout),
            layers.Dense(32),
            act_fn,
            layers.Dropout(dropout),
            layers.Dense(1)
        ])
    
    def call(self, x, training=None):
        # Input projection
        x = self.input_projection(x)
        
        # Add positional encoding
        if self.positional_encoding_type == "learned":
            seq_len = tf.shape(x)[1]
            positions = tf.range(seq_len)
            x = x + self.pos_embedding(positions)
        else:
            x = self.pos_encoding(x)
        
        # Transformer encoder
        for encoder_layer in self.encoder_layers:
            x = encoder_layer(x, training=training)
        
        # Take last timestep
        x = x[:, -1, :]
        
        return self.fc(x, training=training)


@register_keras_serializable()
class TransformerEncoderLayer(layers.Layer):
    """Single Transformer encoder layer."""
    
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        dff: int,
        dropout: float = 0.1,
        activation: str = "gelu",
        **kwargs
    ):
        super().__init__(**kwargs)
        
        self.mha = layers.MultiHeadAttention(
            num_heads=num_heads,
            key_dim=d_model // num_heads,
            dropout=dropout
        )
        
        self.ffn = keras.Sequential([
            layers.Dense(dff, activation=activation),
            layers.Dropout(dropout),
            layers.Dense(d_model),
            layers.Dropout(dropout)
        ])
        
        self.layernorm1 = layers.LayerNormalization()
        self.layernorm2 = layers.LayerNormalization()
        
        self.dropout = layers.Dropout(dropout)
    
    def call(self, x, training=None):
        # Self-attention with residual
        attn_out = self.mha(x, x, training=training)
        x = self.layernorm1(x + self.dropout(attn_out, training=training))
        
        # FFN with residual
        ffn_out = self.ffn(x, training=training)
        x = self.layernorm2(x + ffn_out)
        
        return x


@register_keras_serializable()
class TFTemporalFusionTransformer(Model):
    """
    Temporal Fusion Transformer (TFT) - State-of-the-art for time series forecasting.
    
    Key features:
    - Variable Selection Networks for feature importance
    - Gated Residual Networks for non-linear processing
    - Interpretable Multi-Head Attention
    - Static covariate encoders
    - **Multi-Task Heads**: price, trend, risk, state (like UnifiedMarketNet)
    
    Enhanced with:
    - L2 kernel regularization
    - RecurrentDropout for LSTM
    - GaussianNoise and SpatialDropout1D for input regularization
    
    Paper: "Temporal Fusion Transformers for Interpretable Multi-horizon Time Series Forecasting"
    https://arxiv.org/abs/1912.09363
    """
    
    def __init__(
        self,
        input_size: int = 7,
        hidden_size: int = 128,
        num_heads: int = 4,
        dropout: float = 0.1,
        recurrent_dropout: float = DEFAULT_RECURRENT_DROPOUT,
        kernel_regularizer: float = DEFAULT_L2_REG,
        num_encoder_steps: int = 60,
        num_decoder_steps: int = 1,
        state_classes: int = 3,  # 3 classes matches PyTorch 93%+ accuracy
        multi_task: bool = True,
        noise_std: float = 0.05,
        **kwargs
    ):
        super().__init__(**kwargs)
        
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.num_encoder_steps = num_encoder_steps
        self.num_decoder_steps = num_decoder_steps
        self.multi_task = multi_task
        self.state_classes = state_classes
        
        # L2 regularizer
        l2_reg = regularizers.L2(kernel_regularizer) if kernel_regularizer > 0 else None
        
        # Input augmentation layers
        self.gaussian_noise = layers.GaussianNoise(noise_std)
        self.spatial_dropout = layers.SpatialDropout1D(dropout * 0.5)
        
        # Input projection with L2 regularization
        self.input_projection = layers.Dense(hidden_size, kernel_regularizer=l2_reg)
        
        # Variable Selection Network for encoder inputs
        self.encoder_vsn = GatedResidualNetwork(hidden_size, dropout=dropout)
        
        # LSTM Encoder with RecurrentDropout
        self.encoder_lstm = layers.LSTM(
            hidden_size, 
            return_sequences=True, 
            return_state=True,
            dropout=dropout,
            recurrent_dropout=recurrent_dropout,
            kernel_regularizer=l2_reg
        )
        
        # LSTM Decoder (for multi-step forecasting) with RecurrentDropout
        self.decoder_lstm = layers.LSTM(
            hidden_size,
            return_sequences=True,
            dropout=dropout,
            recurrent_dropout=recurrent_dropout,
            kernel_regularizer=l2_reg
        )
        
        # Gated skip connection
        self.gated_skip = GatedLinearUnit(hidden_size)
        
        # Static context GRN
        self.static_context_grn = GatedResidualNetwork(hidden_size, dropout=dropout)
        
        # Temporal self-attention
        self.self_attention = InterpretableMultiHeadAttention(
            num_heads=num_heads,
            key_dim=hidden_size // num_heads,
            dropout=dropout
        )
        
        # Post-attention GRN
        self.post_attention_grn = GatedResidualNetwork(hidden_size, dropout=dropout)
        
        # Layer norms
        self.layer_norm1 = layers.LayerNormalization()
        self.layer_norm2 = layers.LayerNormalization()
        
        # =====================================================================
        # MULTI-TASK HEADS (like UnifiedMarketNet: price, trend, risk, state)
        # With L2 regularization for overfitting prevention
        # Enhanced for directional accuracy
        # =====================================================================
        if multi_task:
            # Price prediction head
            self.price_head = keras.Sequential([
                layers.Dense(hidden_size, activation='elu', kernel_regularizer=l2_reg),
                layers.Dropout(dropout),
                layers.Dense(1, name='price_output')
            ], name='price_head')
            
            # Trend prediction head (continuous return prediction)
            self.trend_head = keras.Sequential([
                layers.Dense(hidden_size, activation='elu', kernel_regularizer=l2_reg),
                layers.Dropout(dropout),
                layers.Dense(hidden_size // 2, activation='elu', kernel_regularizer=l2_reg),
                layers.Dropout(dropout * 0.5),
                layers.Dense(1, name='trend_output')
            ], name='trend_head')
            
            # Direction classification head (binary: up/down)
            # Uses sigmoid for probability of "up" move
            self.direction_head = keras.Sequential([
                layers.Dense(hidden_size, activation='elu', kernel_regularizer=l2_reg),
                layers.Dropout(dropout),
                layers.Dense(hidden_size // 2, activation='elu', kernel_regularizer=l2_reg),
                layers.Dropout(dropout * 0.5),
                layers.Dense(1, activation='sigmoid', name='direction_output')  # P(up)
            ], name='direction_head')
            
            # Risk prediction head (sigmoid for 0-1 output)
            self.risk_head = keras.Sequential([
                layers.Dense(hidden_size // 2, activation='elu', kernel_regularizer=l2_reg),
                layers.Dropout(dropout),
                layers.Dense(1, activation='sigmoid', name='risk_output')
            ], name='risk_head')
            
            # Market state classification head
            self.state_head = keras.Sequential([
                layers.Dense(hidden_size // 2, activation='elu', kernel_regularizer=l2_reg),
                layers.Dropout(dropout),
                layers.Dense(state_classes, activation='softmax', name='state_output')
            ], name='state_head')
        else:
            # Single output (price only)
            self.output_projection = keras.Sequential([
                layers.Dense(hidden_size, activation='elu', kernel_regularizer=l2_reg),
                layers.Dropout(dropout),
                layers.Dense(1)
            ])
    
    def call(self, x, training=None):
        # Input augmentation (only during training)
        x = self.gaussian_noise(x, training=training)
        x = self.spatial_dropout(x, training=training)
        
        # Project inputs
        embedded = self.input_projection(x)
        
        # Variable selection (simplified - full TFT has per-feature processing)
        selected = self.encoder_vsn(embedded, training=training)
        
        # LSTM encoding
        encoder_output, state_h, _ = self.encoder_lstm(selected, training=training)
        
        # Static context from final LSTM state
        static_context = self.static_context_grn(state_h, training=training)
        static_context = tf.expand_dims(static_context, 1)
        
        # Gated skip connection
        skip = self.gated_skip(encoder_output)
        
        # Self-attention over time
        attention_output = self.self_attention(
            encoder_output, encoder_output, 
            training=training
        )
        
        # Add skip connection and normalize
        temporal_output = self.layer_norm1(skip + attention_output)
        
        # Post-attention processing
        temporal_output = self.post_attention_grn(temporal_output, context=static_context, training=training)
        temporal_output = self.layer_norm2(temporal_output + encoder_output)
        
        # Take final timestep for single-step prediction
        final_output = temporal_output[:, -1, :]
        
        # Multi-task output
        if self.multi_task:
            price = self.price_head(final_output, training=training)
            trend = self.trend_head(final_output, training=training)
            direction = self.direction_head(final_output, training=training)  # P(up move)
            risk = self.risk_head(final_output, training=training)
            state_logits = self.state_head(final_output, training=training)
            
            return {
                'price': price,
                'trend': trend,
                'direction': direction,  # New: binary up/down classification
                'risk': risk,
                'state_logits': state_logits,
            }
        else:
            return self.output_projection(final_output, training=training)

    def build(self, input_shape):
        """Build sublayers by running a dummy forward pass.

        Keras may otherwise mark this model as built even when some sublayers
        remain unbuilt (common with subclassed Models), which can lead to
        warnings and brittle save/load behavior.
        """
        try:
            # input_shape: (batch, time, features)
            time_steps = input_shape[1] if len(input_shape) > 1 and input_shape[1] is not None else self.num_encoder_steps
            n_features = input_shape[2] if len(input_shape) > 2 and input_shape[2] is not None else self.input_size
            dummy = tf.zeros((1, int(time_steps), int(n_features)), dtype=tf.float32)
            _ = self.call(dummy, training=False)
        except Exception:
            # Fall back to letting Keras build lazily on first real call.
            pass
        super().build(input_shape)
    
    def get_attention_weights(self, x):
        """Get attention weights for interpretability."""
        embedded = self.input_projection(x)
        selected = self.encoder_vsn(embedded, training=False)
        encoder_output, _, _ = self.encoder_lstm(selected, training=False)
        
        _, attention_weights = self.self_attention(
            encoder_output, encoder_output,
            training=False,
            return_attention=True
        )
        
        return attention_weights


@register_keras_serializable()
class TFTemporalFusionTransformerEnhanced(Model):
    """
    Enhanced Temporal Fusion Transformer with FULL feature set.
    
    IMPROVEMENTS over base TFT:
    1. Static covariates (instrument ID, session type)
    2. Known future inputs (time-of-day, day-of-week embeddings)
    3. Interpretable attention with visualization support
    4. Multi-horizon output capability
    
    Static Covariates:
    - Instrument embedding (EUR_USD=0, GBP_USD=1, etc.)
    - Session embedding (Asian=0, London=1, NY=2)
    
    Known Future Inputs:
    - Hour of day (0-23) -> 24-dim embedding
    - Day of week (0-6) -> 7-dim embedding
    - These are known at inference time!
    
    Paper: "Temporal Fusion Transformers for Interpretable Multi-horizon Time Series Forecasting"
    """
    
    # Class-level constants for embedding dimensions
    NUM_INSTRUMENTS = 10  # EUR_USD, GBP_USD, USD_JPY, etc.
    NUM_SESSIONS = 4      # Asian, London, NY, Overlap
    NUM_HOURS = 24
    NUM_DAYS = 7
    
    def __init__(
        self,
        input_size: int = 7,
        hidden_size: int = 128,
        num_heads: int = 4,
        dropout: float = 0.1,
        recurrent_dropout: float = DEFAULT_RECURRENT_DROPOUT,
        kernel_regularizer: float = DEFAULT_L2_REG,
        num_encoder_steps: int = 60,
        num_decoder_steps: int = 1,
        state_classes: int = 3,
        multi_task: bool = True,
        noise_std: float = 0.05,
        # NEW: Static covariate dimensions
        instrument_embedding_dim: int = 8,
        session_embedding_dim: int = 4,
        # NEW: Known future input dimensions
        hour_embedding_dim: int = 8,
        day_embedding_dim: int = 4,
        **kwargs
    ):
        super().__init__(**kwargs)
        
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.num_encoder_steps = num_encoder_steps
        self.num_decoder_steps = num_decoder_steps
        self.multi_task = multi_task
        self.state_classes = state_classes
        
        # L2 regularizer
        l2_reg = regularizers.L2(kernel_regularizer) if kernel_regularizer > 0 else None
        
        # =====================================================================
        # STATIC COVARIATE ENCODERS (NEW)
        # =====================================================================
        
        # Instrument embedding (learnable representation per currency pair)
        self.instrument_embedding = layers.Embedding(
            self.NUM_INSTRUMENTS, instrument_embedding_dim, name='instrument_emb'
        )
        
        # Session embedding (Asian, London, NY, Overlap)
        self.session_embedding = layers.Embedding(
            self.NUM_SESSIONS, session_embedding_dim, name='session_emb'
        )
        
        # Static encoder: combines all static features
        self.static_encoder = keras.Sequential([
            layers.Dense(hidden_size, activation='elu', kernel_regularizer=l2_reg),
            layers.Dropout(dropout),
            layers.Dense(hidden_size, kernel_regularizer=l2_reg),
        ], name='static_encoder')
        
        # Context vectors derived from static (for gating)
        self.static_context_variable_selection = layers.Dense(hidden_size, name='static_ctx_vs')
        self.static_context_enrichment = layers.Dense(hidden_size, name='static_ctx_enrich')
        self.static_context_state_h = layers.Dense(hidden_size, name='static_ctx_h')
        self.static_context_state_c = layers.Dense(hidden_size, name='static_ctx_c')
        
        # =====================================================================
        # KNOWN FUTURE INPUT ENCODERS (NEW)
        # Time features that are known at inference time
        # =====================================================================
        
        # Hour of day embedding (captures intraday patterns)
        self.hour_embedding = layers.Embedding(
            self.NUM_HOURS, hour_embedding_dim, name='hour_emb'
        )
        
        # Day of week embedding (captures weekly seasonality)
        self.day_embedding = layers.Embedding(
            self.NUM_DAYS, day_embedding_dim, name='day_emb'
        )
        
        # Known future encoder
        self.known_future_encoder = layers.Dense(
            hidden_size // 4, activation='elu', kernel_regularizer=l2_reg, name='known_future_enc'
        )
        
        # =====================================================================
        # TEMPORAL PROCESSING (Enhanced with static context)
        # =====================================================================
        
        # Input augmentation layers
        self.gaussian_noise = layers.GaussianNoise(noise_std)
        self.spatial_dropout = layers.SpatialDropout1D(dropout * 0.5)
        
        # Input projection (now includes known future features)
        total_input_dim = input_size + hour_embedding_dim + day_embedding_dim
        self.input_projection = layers.Dense(hidden_size, kernel_regularizer=l2_reg)
        
        # Variable Selection Network (conditioned on static context)
        self.encoder_vsn = GatedResidualNetwork(hidden_size, dropout=dropout)
        
        # LSTM Encoder with static covariate initialization
        self.encoder_lstm = layers.LSTM(
            hidden_size, 
            return_sequences=True, 
            return_state=True,
            dropout=dropout,
            recurrent_dropout=recurrent_dropout,
            kernel_regularizer=l2_reg
        )
        
        # Gated skip connection
        self.gated_skip = GatedLinearUnit(hidden_size)
        
        # Interpretable Multi-Head Attention (stores weights for visualization)
        self.self_attention = InterpretableMultiHeadAttention(
            num_heads=num_heads,
            key_dim=hidden_size // num_heads,
            dropout=dropout
        )
        
        # Post-attention GRN with static enrichment
        self.post_attention_grn = GatedResidualNetwork(hidden_size, dropout=dropout)
        
        # Layer norms
        self.layer_norm1 = layers.LayerNormalization()
        self.layer_norm2 = layers.LayerNormalization()
        
        # =====================================================================
        # MULTI-TASK HEADS
        # =====================================================================
        if multi_task:
            self.price_head = keras.Sequential([
                layers.Dense(hidden_size, activation='elu', kernel_regularizer=l2_reg),
                layers.Dropout(dropout),
                layers.Dense(1, name='price_output')
            ], name='price_head')
            
            self.trend_head = keras.Sequential([
                layers.Dense(hidden_size, activation='elu', kernel_regularizer=l2_reg),
                layers.Dropout(dropout),
                layers.Dense(hidden_size // 2, activation='elu', kernel_regularizer=l2_reg),
                layers.Dropout(dropout * 0.5),
                layers.Dense(1, name='trend_output')
            ], name='trend_head')
            
            self.direction_head = keras.Sequential([
                layers.Dense(hidden_size, activation='elu', kernel_regularizer=l2_reg),
                layers.Dropout(dropout),
                layers.Dense(hidden_size // 2, activation='elu', kernel_regularizer=l2_reg),
                layers.Dropout(dropout * 0.5),
                layers.Dense(1, activation='sigmoid', name='direction_output')
            ], name='direction_head')
            
            self.risk_head = keras.Sequential([
                layers.Dense(hidden_size // 2, activation='elu', kernel_regularizer=l2_reg),
                layers.Dropout(dropout),
                layers.Dense(1, activation='sigmoid', name='risk_output')
            ], name='risk_head')
            
            self.state_head = keras.Sequential([
                layers.Dense(hidden_size // 2, activation='elu', kernel_regularizer=l2_reg),
                layers.Dropout(dropout),
                layers.Dense(state_classes, activation='softmax', name='state_output')
            ], name='state_head')
        else:
            self.output_projection = keras.Sequential([
                layers.Dense(hidden_size, activation='elu', kernel_regularizer=l2_reg),
                layers.Dropout(dropout),
                layers.Dense(1)
            ])
        
        # Store last attention weights for interpretability
        self._last_attention_weights = None
    
    def call(self, inputs, training=None):
        """
        Forward pass with static covariates and known future inputs.
        
        Args:
            inputs: Can be:
                - Tensor of shape (batch, seq, features) - basic mode
                - Dict with keys: 'sequence', 'instrument_id', 'session_id', 'hour', 'day_of_week'
        """
        # Handle both basic tensor input and dict input
        if isinstance(inputs, dict):
            x = inputs['sequence']
            instrument_id = inputs.get('instrument_id', tf.zeros((tf.shape(x)[0],), dtype=tf.int32))
            session_id = inputs.get('session_id', tf.zeros((tf.shape(x)[0],), dtype=tf.int32))
            hour = inputs.get('hour', tf.zeros((tf.shape(x)[0], tf.shape(x)[1]), dtype=tf.int32))
            day_of_week = inputs.get('day_of_week', tf.zeros((tf.shape(x)[0], tf.shape(x)[1]), dtype=tf.int32))
        else:
            x = inputs
            batch_size = tf.shape(x)[0]
            seq_len = tf.shape(x)[1]
            # Default static covariates (will be learned as average)
            instrument_id = tf.zeros((batch_size,), dtype=tf.int32)
            session_id = tf.zeros((batch_size,), dtype=tf.int32)
            hour = tf.zeros((batch_size, seq_len), dtype=tf.int32)
            day_of_week = tf.zeros((batch_size, seq_len), dtype=tf.int32)
        
        # =====================================================================
        # STATIC COVARIATE ENCODING
        # =====================================================================
        
        # Embed static features
        instrument_emb = self.instrument_embedding(instrument_id)  # (batch, emb_dim)
        session_emb = self.session_embedding(session_id)          # (batch, emb_dim)
        
        # Concatenate and encode static features
        static_features = tf.concat([instrument_emb, session_emb], axis=-1)
        static_encoded = self.static_encoder(static_features, training=training)
        
        # Derive context vectors for different uses
        static_ctx_vs = self.static_context_variable_selection(static_encoded)
        static_ctx_enrich = self.static_context_enrichment(static_encoded)
        static_ctx_h = self.static_context_state_h(static_encoded)
        static_ctx_c = self.static_context_state_c(static_encoded)
        
        # =====================================================================
        # KNOWN FUTURE INPUT ENCODING
        # =====================================================================
        
        # Embed time features (these are known for all future steps)
        hour_emb = self.hour_embedding(hour)            # (batch, seq, emb_dim)
        day_emb = self.day_embedding(day_of_week)       # (batch, seq, emb_dim)
        known_future = tf.concat([hour_emb, day_emb], axis=-1)
        
        # =====================================================================
        # TEMPORAL PROCESSING
        # =====================================================================
        
        # Input augmentation
        x = self.gaussian_noise(x, training=training)
        x = self.spatial_dropout(x, training=training)
        
        # Concatenate sequence with known future features
        x_with_future = tf.concat([x, known_future], axis=-1)
        
        # Project inputs
        embedded = self.input_projection(x_with_future)
        
        # Variable selection (conditioned on static context)
        static_ctx_expanded = tf.expand_dims(static_ctx_vs, 1)  # (batch, 1, hidden)
        selected = self.encoder_vsn(embedded, context=static_ctx_expanded, training=training)
        
        # LSTM encoding with static-initialized states
        initial_state = [static_ctx_h, static_ctx_c]
        encoder_output, state_h, _ = self.encoder_lstm(
            selected, initial_state=initial_state, training=training
        )
        
        # Gated skip connection
        skip = self.gated_skip(encoder_output)
        
        # Self-attention with interpretable weights
        attention_output, attention_weights = self.self_attention(
            encoder_output, encoder_output, 
            training=training,
            return_attention=True
        )
        self._last_attention_weights = attention_weights  # Store for visualization
        
        # Add skip connection and normalize
        temporal_output = self.layer_norm1(skip + attention_output)
        
        # Post-attention processing with static enrichment
        static_ctx_enrich_expanded = tf.expand_dims(static_ctx_enrich, 1)
        temporal_output = self.post_attention_grn(
            temporal_output, context=static_ctx_enrich_expanded, training=training
        )
        temporal_output = self.layer_norm2(temporal_output + encoder_output)
        
        # Take final timestep
        final_output = temporal_output[:, -1, :]
        
        # Multi-task output
        if self.multi_task:
            return {
                'price': self.price_head(final_output, training=training),
                'trend': self.trend_head(final_output, training=training),
                'direction': self.direction_head(final_output, training=training),
                'risk': self.risk_head(final_output, training=training),
                'state_logits': self.state_head(final_output, training=training),
            }
        else:
            return self.output_projection(final_output, training=training)
    
    def get_attention_weights(self):
        """Get last computed attention weights for interpretability."""
        return self._last_attention_weights
    
    def get_feature_importance(self, x, instrument_id=None, session_id=None):
        """
        Get feature importance scores via attention analysis.
        
        Returns dict with:
        - temporal_attention: Which timesteps matter most
        - feature_weights: Which features contribute most (from VSN)
        """
        # Run forward pass to get attention weights
        inputs = {'sequence': x}
        if instrument_id is not None:
            inputs['instrument_id'] = instrument_id
        if session_id is not None:
            inputs['session_id'] = session_id
        
        _ = self.call(inputs, training=False)
        
        attention = self._last_attention_weights
        if attention is not None:
            # Average attention across heads
            temporal_importance = tf.reduce_mean(attention, axis=1)  # (batch, seq, seq)
            # Sum attention received by each position
            temporal_importance = tf.reduce_sum(temporal_importance, axis=-1)  # (batch, seq)
        else:
            temporal_importance = None
        
        return {
            'temporal_attention': temporal_importance,
        }


@register_keras_serializable()
class TFTCNPredictor(Model):
    """
    Temporal Convolutional Network (TCN) for time series prediction.
    
    Advantages over LSTM:
    - Parallelizable (faster training)
    - Longer effective memory via dilated convolutions
    - More stable gradients
    
    Enhanced with:
    - L2 kernel regularization
    - GaussianNoise and SpatialDropout1D for input regularization
    
    TensorFlow equivalent of PyTorch TCNPredictor.
    """
    
    def __init__(
        self,
        input_size: int = 7,
        hidden_size: int = 128,
        num_layers: int = 3,
        kernel_size: int = 3,
        dropout: float = 0.2,
        kernel_regularizer: float = DEFAULT_L2_REG,
        activation: str = "relu",
        use_residual: bool = True,
        state_classes: int = 3,  # 3 classes matches PyTorch 93%+ accuracy
        multi_task: bool = True,
        noise_std: float = 0.05,
        **kwargs
    ):
        super().__init__(**kwargs)
        
        self.use_residual = use_residual
        self.num_layers = num_layers
        self.multi_task = multi_task
        self.state_classes = state_classes
        self.hidden_size = hidden_size
        
        # L2 regularizer
        l2_reg = regularizers.L2(kernel_regularizer) if kernel_regularizer > 0 else None
        
        # Input augmentation layers
        self.gaussian_noise = layers.GaussianNoise(noise_std)
        self.spatial_dropout = layers.SpatialDropout1D(dropout * 0.5)
        
        # Activation
        if activation == "mish":
            self.activation = Mish()
        elif activation == "swish":
            self.activation = Swish()
        elif activation == "gelu":
            self.activation = layers.Activation('gelu')
        else:
            self.activation = layers.ReLU()
        
        # TCN layers with exponentially increasing dilation and L2 regularization
        self.tcn_layers = []
        self.residual_layers = []
        
        for i in range(num_layers):
            dilation_rate = 2 ** i
            in_channels = input_size if i == 0 else hidden_size
            
            # Causal convolution with dilation and L2 regularization
            conv = keras.Sequential([
                layers.Conv1D(
                    hidden_size,
                    kernel_size,
                    padding='causal',
                    dilation_rate=dilation_rate,
                    kernel_regularizer=l2_reg
                ),
                layers.BatchNormalization()
            ])
            self.tcn_layers.append(conv)
            
            # Residual projection if needed
            if use_residual and in_channels != hidden_size:
                self.residual_layers.append(layers.Conv1D(hidden_size, 1, kernel_regularizer=l2_reg))
            else:
                self.residual_layers.append(None)
        
        self.dropout = layers.Dropout(dropout)
        
        # =====================================================================
        # MULTI-TASK HEADS (like UnifiedMarketNet: price, trend, risk, state)
        # With L2 regularization for overfitting prevention
        # Enhanced for directional accuracy
        # =====================================================================
        if multi_task:
            # Price prediction head
            self.price_head = keras.Sequential([
                layers.Dense(64, kernel_regularizer=l2_reg),
                self.activation,
                layers.Dropout(dropout),
                layers.Dense(1, name='price_output')
            ], name='price_head')
            
            # Trend prediction head (continuous return)
            self.trend_head = keras.Sequential([
                layers.Dense(64, kernel_regularizer=l2_reg),
                self.activation,
                layers.Dropout(dropout),
                layers.Dense(32, kernel_regularizer=l2_reg),
                self.activation,
                layers.Dropout(dropout * 0.5),
                layers.Dense(1, name='trend_output')
            ], name='trend_head')
            
            # Direction classification head (binary: up/down)
            self.direction_head = keras.Sequential([
                layers.Dense(64, kernel_regularizer=l2_reg),
                self.activation,
                layers.Dropout(dropout),
                layers.Dense(32, kernel_regularizer=l2_reg),
                self.activation,
                layers.Dropout(dropout * 0.5),
                layers.Dense(1, activation='sigmoid', name='direction_output')  # P(up)
            ], name='direction_head')
            
            # Risk prediction head (sigmoid for 0-1 output)
            self.risk_head = keras.Sequential([
                layers.Dense(32, kernel_regularizer=l2_reg),
                self.activation,
                layers.Dropout(dropout),
                layers.Dense(1, activation='sigmoid', name='risk_output')
            ], name='risk_head')
            
            # Market state classification head
            self.state_head = keras.Sequential([
                layers.Dense(32, kernel_regularizer=l2_reg),
                self.activation,
                layers.Dropout(dropout),
                layers.Dense(state_classes, activation='softmax', name='state_output')
            ], name='state_head')
        else:
            # Single output (price only)
            self.fc = keras.Sequential([
                layers.Dense(64, kernel_regularizer=l2_reg),
                self.activation,
                layers.Dropout(dropout),
                layers.Dense(32, kernel_regularizer=l2_reg),
                self.activation,
                layers.Dropout(dropout),
                layers.Dense(1)
            ])
    
    def call(self, x, training=None):
        # x shape: [batch, time, features]
        
        # Input augmentation (only during training)
        x = self.gaussian_noise(x, training=training)
        x = self.spatial_dropout(x, training=training)
        
        for i, (conv, residual_layer) in enumerate(zip(self.tcn_layers, self.residual_layers)):
            residual = x
            
            # Apply convolution
            out = conv(x, training=training)
            out = self.activation(out)
            out = self.dropout(out, training=training)
            
            # Add residual connection
            if self.use_residual:
                if residual_layer is not None:
                    residual = residual_layer(residual)
                x = out + residual
            else:
                x = out
        
        # Take last timestep
        x = x[:, -1, :]
        
        # Multi-task output
        if self.multi_task:
            price = self.price_head(x, training=training)
            trend = self.trend_head(x, training=training)
            direction = self.direction_head(x, training=training)  # P(up move)
            risk = self.risk_head(x, training=training)
            state_logits = self.state_head(x, training=training)
            
            return {
                'price': price,
                'trend': trend,
                'direction': direction,  # New: binary up/down classification
                'risk': risk,
                'state_logits': state_logits,
            }
        else:
            return self.fc(x, training=training)


@register_keras_serializable()
class TFEnsemblePredictor(Model):
    """Ensemble of multiple models for robust predictions."""
    
    def __init__(
        self,
        models: List[Model],
        ensemble_method: str = "average",
        **kwargs
    ):
        super().__init__(**kwargs)
        
        self.models = models
        self.ensemble_method = ensemble_method
        
        if ensemble_method == "weighted":
            self.weights = tf.Variable(
                tf.ones(len(models)) / len(models),
                trainable=True
            )
    
    def call(self, x, training=None):
        predictions = [model(x, training=training) for model in self.models]
        predictions = tf.stack(predictions, axis=0)
        
        if self.ensemble_method == "average":
            return tf.reduce_mean(predictions, axis=0)
        elif self.ensemble_method == "weighted":
            weights = tf.nn.softmax(self.weights)
            weights = tf.reshape(weights, [-1, 1, 1])
            return tf.reduce_sum(predictions * weights, axis=0)
        else:
            return tf.reduce_mean(predictions, axis=0)


# =============================================================================
# Model Factory
# =============================================================================

def create_tensorflow_model(config: dict) -> Model:
    """
    Factory function to create TensorFlow models based on configuration.
    
    Args:
        config: Dictionary with model configuration including:
            - type: Model type (lstm, attention_lstm, transformer, tft, tcn)
            - input_size: Number of input features
            - hidden_size: Hidden layer size
            - num_layers: Number of layers
            - dropout: Dropout rate
            - Other model-specific parameters
    
    Returns:
        Configured TensorFlow model
    
    M1 Metal Recommendations:
        - Use 'tcn' for fastest training (parallelizable convolutions)
        - Use 'tft' for best accuracy with interpretability
        - Avoid deep LSTM (>3 layers) on Metal
        - Set jit_compile=True when compiling model
    """
    model_type = config.get('type', 'lstm').lower()
    
    # M1 Metal: Warn if using suboptimal model type
    if is_apple_silicon() and model_type in ['lstm', 'stock_predictor']:
        import logging
        logging.getLogger(__name__).info(
            "M1 Metal tip: Consider 'tcn' for 2-3x faster training than LSTM"
        )
    
    # Common parameters for all models
    common_params = {
        'input_size': config.get('input_size', 7),
        'hidden_size': config.get('hidden_size', 128),
        'dropout': config.get('dropout', 0.35),  # Increased default for small datasets
    }
    
    # Regularization parameters (new for overfitting prevention)
    regularization_params = {
        'recurrent_dropout': config.get('recurrent_dropout', DEFAULT_RECURRENT_DROPOUT),
        'kernel_regularizer': config.get('kernel_regularizer', DEFAULT_L2_REG),
        'noise_std': config.get('noise_std', 0.05),
    }
    
    # Multi-task head parameters (like UnifiedMarketNet)
    multi_task_params = {
        'multi_task': config.get('multi_task', True),
        'state_classes': config.get('state_classes', 3),
    }
    
    if model_type in ['lstm', 'stock_predictor']:
        return TFStockPredictor(
            **common_params,
            **regularization_params,
            num_layers=config.get('num_layers', 2),  # Reduced for small datasets
            bidirectional=config.get('bidirectional', False),
            use_layer_norm=config.get('use_layer_norm', True),
            activation=config.get('activation', 'relu'),
        )
    
    elif model_type in ['attention_lstm', 'attentive_lstm']:
        return TFAttentiveLSTM(
            **common_params,
            **regularization_params,
            **multi_task_params,
            num_layers=config.get('num_layers', 2),  # Reduced for small datasets
            num_heads=config.get('num_heads', 4),
            bidirectional=config.get('bidirectional', False),
        )
    
    elif model_type == 'transformer':
        return TFTransformerPredictor(
            **common_params,
            num_layers=config.get('num_layers', 2),  # Reduced for small datasets
            num_heads=config.get('num_heads', 8),
            positional_encoding=config.get('positional_encoding', 'sinusoidal'),
            activation=config.get('activation', 'gelu'),
        )
    
    elif model_type in ['tft', 'temporal_fusion_transformer']:
        return TFTemporalFusionTransformer(
            **common_params,
            **regularization_params,
            **multi_task_params,
            num_heads=config.get('num_heads', 4),
            num_encoder_steps=config.get('sequence_length', 60),
        )
    
    elif model_type == 'tcn':
        return TFTCNPredictor(
            **common_params,
            kernel_regularizer=regularization_params['kernel_regularizer'],
            noise_std=regularization_params['noise_std'],
            **multi_task_params,
            num_layers=config.get('num_layers', 3),
            kernel_size=config.get('kernel_size', 3),
            activation=config.get('activation', 'relu'),
            use_residual=config.get('use_residual', True),
        )
    
    else:
        raise ValueError(f"Unknown model type: {model_type}")


# =============================================================================
# Model Comparison Summary
# =============================================================================

MODEL_COMPARISON = """
┌─────────────────────────────────────────────────────────────────────────────┐
│                    TensorFlow Model Comparison                              │
├─────────────────┬───────────────┬───────────────┬───────────────────────────┤
│ Model           │ Speed         │ Memory        │ Best For                  │
├─────────────────┼───────────────┼───────────────┼───────────────────────────┤
│ LSTM            │ Medium        │ Medium        │ Baseline, short sequences │
│ Attention-LSTM  │ Slower        │ Higher        │ Long-range dependencies   │
│ Transformer     │ Fast (GPU)    │ High          │ Parallel training         │
│ TFT             │ Medium        │ Higher        │ Multi-feature, interpret. │
│ TCN             │ Fastest       │ Low           │ Real-time, long sequences │
└─────────────────┴───────────────┴───────────────┴───────────────────────────┘

Recommendations:
- For your FX trading with 104 features: TFT (interpretability) or TCN (speed)
- For experimentation: Start with Attention-LSTM, compare with TCN
- For production: TCN (fastest inference) or TFT (best accuracy on complex data)
"""

if __name__ == "__main__":
    print(MODEL_COMPARISON)
    
    # Test model creation
    test_config = {
        'type': 'tft',
        'input_size': 104,
        'hidden_size': 64,
        'num_layers': 3,
        'dropout': 0.2,
        'num_heads': 4,
        'sequence_length': 60,
    }
    
    model = create_tensorflow_model(test_config)
    
    # Test forward pass
    test_input = tf.random.normal([32, 60, 104])  # [batch, seq_len, features]
    output = model(test_input)
    print(f"\nModel: {type(model).__name__}")
    print(f"Input shape: {test_input.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Parameters: {model.count_params():,}")
# — Raynergy-svg —
