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
# Custom Loss Functions for Direction Prediction
# =============================================================================

@register_keras_serializable()
class AntiCollapseFocalLoss(keras.losses.Loss):
    """
    Enhanced Focal Loss that actively prevents prediction collapse.

    The problem: Models collapse to predicting one class (e.g., all 0.47 = SHORT).

    Previous entropy regularization was WRONG: it pushed predictions toward 0.5,
    which means all predictions cluster around the decision boundary = 100% one class.

    This version uses VARIANCE REGULARIZATION instead:
    1. Focal loss for class imbalance
    2. Variance penalty: Penalizes when batch predictions have LOW variance
       (i.e., when model outputs similar values for all inputs)
    3. Dynamic alpha: Higher weight for ignored class
    4. Gradient floor: Ensures minority class contributes gradients

    Args:
        gamma: Focusing parameter (default 2.0)
        base_alpha: Starting class weight for positive class (default 0.5 = balanced)
        variance_weight: Weight for variance penalty (default 0.1) - penalizes constant outputs
        min_gradient_scale: Minimum gradient scale to prevent vanishing (default 0.1)
        label_smoothing: Label smoothing factor (default 0.05)
    """

    def __init__(
        self,
        gamma: float = 2.0,
        base_alpha: float = 0.5,
        entropy_weight: float = 0.1,  # Kept for backward compat, now controls variance
        min_gradient_scale: float = 0.1,
        label_smoothing: float = 0.05,
        name: str = 'anti_collapse_focal_loss',
        **kwargs
    ):
        super().__init__(name=name, **kwargs)
        self.gamma = gamma
        self.base_alpha = base_alpha
        self.variance_weight = entropy_weight  # Repurposed: now variance weight
        self.min_gradient_scale = min_gradient_scale
        self.label_smoothing = label_smoothing

    def call(self, y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)

        # Apply label smoothing
        if self.label_smoothing > 0:
            y_true = y_true * (1 - self.label_smoothing) + 0.5 * self.label_smoothing

        # Clip predictions (wider range than standard to allow more gradient flow)
        y_pred = tf.clip_by_value(y_pred, 1e-6, 1 - 1e-6)

        # === DYNAMIC ALPHA ADJUSTMENT ===
        # Detect which class is being ignored and boost it
        mean_pred = tf.reduce_mean(y_pred)
        # If mean_pred < 0.3, model is biased toward DOWN → boost UP (alpha > 0.5)
        # If mean_pred > 0.7, model is biased toward UP → boost DOWN (alpha < 0.5)
        # Aggressive swing: 0.5 coefficient (was 0.3) → range [0.15, 0.85] for extreme collapse
        collapse_severity = 2.0 * tf.abs(mean_pred - 0.5)  # 0-1 scale: how collapsed
        alpha_coefficient = 0.3 + 0.4 * collapse_severity  # Ramp from 0.3 to 0.7 as collapse worsens
        dynamic_alpha = self.base_alpha + alpha_coefficient * (0.5 - mean_pred)
        dynamic_alpha = tf.clip_by_value(dynamic_alpha, 0.10, 0.90)

        # === FOCAL LOSS COMPONENT ===
        # Binary cross entropy
        bce = -y_true * tf.math.log(y_pred) - (1 - y_true) * tf.math.log(1 - y_pred)

        # Focal weight with alpha adjustment
        pt = y_true * y_pred + (1 - y_true) * (1 - y_pred)
        focal_weight = tf.pow(1 - pt, self.gamma)

        # Apply dynamic alpha
        alpha_weight = y_true * dynamic_alpha + (1 - y_true) * (1 - dynamic_alpha)
        focal_weight = focal_weight * alpha_weight

        # === GRADIENT FLOOR ===
        # Prevent focal_weight from going too close to zero
        # This ensures minority class samples always contribute to learning
        focal_weight = tf.maximum(focal_weight, self.min_gradient_scale)

        focal_loss = focal_weight * bce

        # === VARIANCE REGULARIZATION (replaces entropy) ===
        # Penalize LOW variance in predictions across the batch
        # If all predictions are ~0.47, variance is near 0 → add penalty
        # This forces the model to have diverse outputs, not collapse to constant
        pred_variance = tf.math.reduce_variance(y_pred)
        # Target variance: 0.08 (std≈0.28) means predictions spread from ~0.22 to ~0.78
        # Increased from 0.06 for stronger anti-collapse
        target_variance = 0.08
        # Penalty increases as variance drops below target - use squared penalty for stronger effect
        variance_gap = tf.maximum(target_variance - pred_variance, 0.0)
        # Scale penalty by collapse severity: gentle when balanced, aggressive when collapsed
        variance_scale = 50.0 + 200.0 * collapse_severity  # 50-250x based on how collapsed
        variance_penalty = self.variance_weight * tf.square(variance_gap) * variance_scale

        # === CLASS BALANCE PENALTY ===
        # Additional penalty when predictions are too skewed (>70% one class)
        # Lowered threshold from 0.30 to 0.20 for earlier intervention
        balance_gap = tf.maximum(tf.abs(mean_pred - 0.5) - 0.20, 0.0)
        balance_penalty = self.variance_weight * tf.square(balance_gap) * 5.0  # Quadratic ramp

        total_loss = tf.reduce_mean(focal_loss) + variance_penalty + balance_penalty

        return total_loss

    def get_config(self):
        config = super().get_config()
        config.update({
            'gamma': self.gamma,
            'base_alpha': self.base_alpha,
            'entropy_weight': self.variance_weight,  # Keep param name for compat
            'min_gradient_scale': self.min_gradient_scale,
            'label_smoothing': self.label_smoothing,
        })
        return config


@register_keras_serializable()
class BinaryFocalLoss(keras.losses.Loss):
    """
    Standard Binary Focal Loss for addressing class imbalance.

    Paper: "Focal Loss for Dense Object Detection" (Lin et al., 2017)
    """

    def __init__(
        self,
        gamma: float = 2.0,
        alpha: float = 0.25,
        label_smoothing: float = 0.0,
        name: str = 'binary_focal_loss',
        **kwargs
    ):
        super().__init__(name=name, **kwargs)
        self.gamma = gamma
        self.alpha = alpha
        self.label_smoothing = label_smoothing

    def call(self, y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)
        if self.label_smoothing > 0:
            y_true = y_true * (1 - self.label_smoothing) + 0.5 * self.label_smoothing

        y_pred = tf.clip_by_value(y_pred, 1e-7, 1 - 1e-7)
        bce = -y_true * tf.math.log(y_pred) - (1 - y_true) * tf.math.log(1 - y_pred)
        pt = y_true * y_pred + (1 - y_true) * (1 - y_pred)
        focal_weight = tf.pow(1 - pt, self.gamma)

        if self.alpha is not None:
            alpha_weight = y_true * self.alpha + (1 - y_true) * (1 - self.alpha)
            focal_weight = focal_weight * alpha_weight

        return tf.reduce_mean(focal_weight * bce)

    def get_config(self):
        config = super().get_config()
        config.update({
            'gamma': self.gamma,
            'alpha': self.alpha,
            'label_smoothing': self.label_smoothing,
        })
        return config


@register_keras_serializable()
class ClassBalancedFocalLoss(keras.losses.Loss):
    """
    Class-Balanced Focal Loss for extreme class imbalance.

    Combines Class-Balanced Loss (Cui et al., CVPR 2019) with Focal Loss.
    This is more stable than inverse frequency weighting with caps.

    Key insight: As class size increases, effective number of samples grows sublinearly
    due to redundancy. CB Loss uses this to compute stable weights.

    Args:
        gamma: Focusing parameter for Focal Loss (default 2.0)
        beta: Class-balanced hyperparameter (default 0.9999)
        label_smoothing: Label smoothing factor (default 0.05)
        focal_alpha: Base class weight for Focal (default 0.5)
    """

    def __init__(
        self,
        gamma: float = 2.0,
        beta: float = 0.9999,
        label_smoothing: float = 0.05,
        focal_alpha: float = 0.5,
        name: str = 'class_balanced_focal_loss',
        **kwargs
    ):
        super().__init__(name=name, **kwargs)
        self.gamma = gamma
        self.beta = beta
        self.label_smoothing = label_smoothing
        self.focal_alpha = focal_alpha

    def call(self, y_true, y_pred):
        import tensorflow as tf

        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)

        # Apply label smoothing
        if self.label_smoothing > 0:
            y_true = y_true * (1 - self.label_smoothing) + 0.5 * self.label_smoothing

        # Clip predictions
        y_pred = tf.clip_by_value(y_pred, 1e-7, 1 - 1e-7)

        # === COMPUTE CLASS-BALANCED WEIGHTS ===
        # Get batch size
        batch_size = tf.cast(tf.shape(y_true)[0], tf.float32)

        # Count samples per class
        n_up = tf.reduce_sum(y_true)
        n_down = batch_size - n_up

        # Compute effective number of samples for each class
        # n_eff = (1 - beta^(n/N)) / (1 - beta)
        total = n_up + n_down + tf.keras.backend.epsilon()

        # Effective samples for UP class
        up_ratio = n_up / total
        (1.0 - tf.pow(self.beta, up_ratio)) / (1.0 - self.beta + tf.keras.backend.epsilon())

        # Effective samples for DOWN class
        down_ratio = n_down / total
        (1.0 - tf.pow(self.beta, down_ratio)) / (1.0 - self.beta + tf.keras.backend.epsilon())

        # Class-balanced weights (inverse of effective samples)
        # w_CB = (1 - beta) / (1 - beta^(n/N))
        up_weight = (1.0 - self.beta) / (1.0 - tf.pow(self.beta, up_ratio) + tf.keras.backend.epsilon())
        down_weight = (1.0 - self.beta) / (1.0 - tf.pow(self.beta, down_ratio) + tf.keras.backend.epsilon())

        # Normalize weights to sum to batch size (preserve effective batch size)
        total_weight = up_weight * n_up + down_weight * n_down
        up_weight = up_weight * batch_size / (total_weight + tf.keras.backend.epsilon())
        down_weight = down_weight * batch_size / (total_weight + tf.keras.backend.epsilon())

        # Apply class-balanced weights per sample
        cb_weight = y_true * up_weight + (1 - y_true) * down_weight

        # === FOCAL LOSS COMPONENT ===
        # Binary cross entropy
        bce = -y_true * tf.math.log(y_pred) - (1 - y_true) * tf.math.log(1 - y_pred)

        # Focal weight
        pt = y_true * y_pred + (1 - y_true) * (1 - y_pred)
        focal_weight = tf.pow(1 - pt, self.gamma)

        # Apply focal alpha
        alpha_weight = y_true * self.focal_alpha + (1 - y_true) * (1 - self.focal_alpha)
        focal_weight = focal_weight * alpha_weight

        # === COMBINE CB WEIGHTS WITH FOCAL LOSS ===
        # L = w_CB * L_Focal
        focal_loss = focal_weight * bce

        # Apply class-balanced weighting
        weighted_loss = cb_weight * focal_loss

        # Normalize by batch size to get average loss
        total_loss = tf.reduce_mean(weighted_loss)

        return total_loss

    def get_config(self):
        config = super().get_config()
        config.update({
            'gamma': self.gamma,
            'beta': self.beta,
            'label_smoothing': self.label_smoothing,
            'focal_alpha': self.focal_alpha,
        })
        return config


@register_keras_serializable()
class HybridClassBalancedAntiCollapseLoss(keras.losses.Loss):
    """
    Combines Class-Balanced Focal Loss with variance regularization to prevent collapse.

    This hybrid loss:
    1. Uses effective number of samples weighting (beta=0.9999) for class imbalance
    2. Adds variance penalty when predictions collapse to constant values

    The variance penalty encourages prediction diversity:
        variance_loss = variance_weight * max(0, variance_target - pred_variance)

    This activates only when prediction variance drops below the target,
    pushing the model to produce more diverse predictions.

    Args:
        gamma: Focal loss focusing parameter (default 2.0)
        beta: Effective number smoothing for class balancing (default 0.9999)
        variance_weight: Weight for variance penalty (default 0.1)
        variance_target: Target minimum variance for predictions (default 0.08)
        label_smoothing: Label smoothing factor (default 0.05)
    """

    def __init__(
        self,
        gamma: float = 2.0,
        beta: float = 0.9999,
        variance_weight: float = 0.1,
        variance_target: float = 0.08,
        label_smoothing: float = 0.05,
        name: str = 'hybrid_cb_anticollapse_loss',
        **kwargs
    ):
        super().__init__(name=name, **kwargs)
        self.gamma = gamma
        self.beta = beta
        self.variance_weight = variance_weight
        self.variance_target = variance_target
        self.label_smoothing = label_smoothing

    def call(self, y_true, y_pred):
        import tensorflow as tf

        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)

        # Apply label smoothing
        if self.label_smoothing > 0:
            y_true = y_true * (1 - self.label_smoothing) + 0.5 * self.label_smoothing

        # Clip predictions to prevent log(0)
        y_pred = tf.clip_by_value(y_pred, 1e-7, 1 - 1e-7)

        # === COMPUTE EFFECTIVE NUMBER WEIGHTS (Class Balancing) ===
        n_pos = tf.reduce_sum(tf.cast(y_true > 0.5, tf.float32)) + 1e-6
        n_neg = tf.reduce_sum(tf.cast(y_true <= 0.5, tf.float32)) + 1e-6

        e_pos = (1.0 - tf.pow(self.beta, n_pos)) / (1.0 - self.beta + 1e-7)
        e_neg = (1.0 - tf.pow(self.beta, n_neg)) / (1.0 - self.beta + 1e-7)

        w_pos = 1.0 / e_pos
        w_neg = 1.0 / e_neg
        total_w = w_pos + w_neg + 1e-7
        w_pos = 2.0 * w_pos / total_w
        w_neg = 2.0 * w_neg / total_w

        # === FOCAL LOSS COMPUTATION ===
        bce = -y_true * tf.math.log(y_pred) - (1 - y_true) * tf.math.log(1 - y_pred)
        pt = y_true * y_pred + (1 - y_true) * (1 - y_pred)
        focal_weight = tf.pow(1 - pt, self.gamma)
        cb_weight = y_true * w_pos + (1 - y_true) * w_neg
        cb_focal_loss = tf.reduce_mean(cb_weight * focal_weight * bce)

        # === VARIANCE PENALTY (Anti-Collapse) ===
        # Penalize when prediction variance drops below target
        # This prevents the model from collapsing to constant predictions
        pred_var = tf.math.reduce_variance(y_pred)
        variance_loss = self.variance_weight * tf.maximum(0.0, self.variance_target - pred_var)

        return cb_focal_loss + variance_loss

    def get_config(self):
        config = super().get_config()
        config.update({
            'gamma': self.gamma,
            'beta': self.beta,
            'variance_weight': self.variance_weight,
            'variance_target': self.variance_target,
            'label_smoothing': self.label_smoothing,
        })
        return config


@register_keras_serializable()
class MADLLoss(keras.losses.Loss):
    """
    Mean Absolute Directional Loss - optimizes for directional profitability.

    MADL penalizes incorrect directional predictions weighted by return magnitude:
    L = |actual_return| * (1 - correct_direction_indicator)

    This aligns model training with trading objectives:
    - Large losses on wrong predictions when returns are large (missed opportunities)
    - Small losses on wrong predictions when returns are small (doesn't matter much)
    - Zero loss on correct predictions (no penalty for profitable trades)

    Research shows MADL optimization achieves RAPI scores of 1.45-1.58 on USD pairs,
    outperforming standard BCE/focal loss which only optimize for accuracy.

    Args:
        return_scale: Scale factor for returns (default 100.0 to normalize FX returns)
        direction_weight: Weight for direction vs return magnitude (default 0.7)
        label_smoothing: Label smoothing factor (default 0.0)
    """

    def __init__(
        self,
        return_scale: float = 100.0,
        direction_weight: float = 0.7,
        label_smoothing: float = 0.0,
        name: str = 'madl_loss',
        **kwargs
    ):
        super().__init__(name=name, **kwargs)
        self.return_scale = return_scale
        self.direction_weight = direction_weight
        self.label_smoothing = label_smoothing

    def call(self, y_true, y_pred):
        """
        Compute MADL loss.

        Args:
            y_true: Ground truth labels (0=down, 1=up) or (direction, return) tuple
            y_pred: Predicted probabilities [0, 1]

        Note: For full MADL with return weighting, y_true should contain returns.
              If y_true is binary labels only, we fall back to weighted BCE.
        """
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)

        # Apply label smoothing
        if self.label_smoothing > 0:
            y_true = y_true * (1 - self.label_smoothing) + 0.5 * self.label_smoothing

        # Clip predictions for numerical stability
        y_pred = tf.clip_by_value(y_pred, 1e-7, 1 - 1e-7)

        # === DIRECTIONAL LOSS COMPONENT ===
        # Convert predictions to directional signal
        pred_direction = tf.cast(y_pred >= 0.5, tf.float32)  # 1=up, 0=down
        true_direction = tf.cast(y_true >= 0.5, tf.float32)  # 1=up, 0=down

        # Direction mismatch indicator (1 if wrong, 0 if correct)
        direction_mismatch = tf.abs(pred_direction - true_direction)

        # === CONFIDENCE PENALTY ===
        # Penalize confident wrong predictions more than uncertain wrong predictions
        # If pred=0.9 and true=0, penalty is higher than if pred=0.6 and true=0
        confidence = tf.abs(y_pred - 0.5) * 2  # [0, 1] confidence scale

        # === COMBINED MADL ===
        # Base loss: direction mismatch weighted by prediction confidence
        # Wrong + confident = high loss, Wrong + uncertain = medium loss, Right = low loss
        base_loss = direction_mismatch * (0.5 + confidence * 0.5)

        # Add BCE component for gradient smoothness (MADL alone has harsh gradients)
        bce = -y_true * tf.math.log(y_pred) - (1 - y_true) * tf.math.log(1 - y_pred)

        # Combine: direction_weight for MADL, (1-direction_weight) for BCE
        total_loss = (
            self.direction_weight * base_loss +
            (1 - self.direction_weight) * bce
        )

        return tf.reduce_mean(total_loss)

    def get_config(self):
        config = super().get_config()
        config.update({
            'return_scale': self.return_scale,
            'direction_weight': self.direction_weight,
            'label_smoothing': self.label_smoothing,
        })
        return config


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
        return x + tf.cast(self.pe[:, :seq_len, :], x.dtype)

    def get_config(self):
        config = super().get_config()
        config['max_len'] = self.max_len
        return config


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
    """Single Transformer encoder layer with MHA + FFN + residual connections."""

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        dff: int,
        dropout: float = 0.1,
        activation: str = "gelu",
        kernel_regularizer=None,
        **kwargs
    ):
        super().__init__(**kwargs)

        # Store config for get_config() serialization
        self.d_model = d_model
        self.num_heads = num_heads
        self.dff = dff
        self._dropout_rate = dropout
        self._activation = activation
        self._kernel_regularizer = kernel_regularizer

        self.mha = layers.MultiHeadAttention(
            num_heads=num_heads,
            key_dim=d_model // num_heads,
            dropout=dropout
        )

        self.ffn = keras.Sequential([
            layers.Dense(dff, activation=activation, kernel_regularizer=kernel_regularizer),
            layers.Dropout(dropout),
            layers.Dense(d_model, kernel_regularizer=kernel_regularizer),
            layers.Dropout(dropout)
        ])

        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)

        self.dropout = layers.Dropout(dropout)

    def build(self, input_shape):
        """Build sublayers with the given input shape."""
        # MHA in Keras 3.x: build(query_shape, value_shape, key_shape=None)
        # For self-attention, query == value == key shape
        self.mha.build(input_shape, input_shape)
        self.ffn.build(input_shape)
        self.layernorm1.build(input_shape)
        self.layernorm2.build(input_shape)
        self.dropout.build(input_shape)
        super().build(input_shape)

    def call(self, x, training=None):
        # Self-attention with residual
        attn_out = self.mha(x, x, training=training)
        x = self.layernorm1(x + self.dropout(attn_out, training=training))

        # FFN with residual
        ffn_out = self.ffn(x, training=training)
        x = self.layernorm2(x + ffn_out)

        return x

    def get_config(self):
        config = super().get_config()
        reg_serialized = (keras.regularizers.serialize(self._kernel_regularizer)
                          if self._kernel_regularizer else None)
        config.update({
            'd_model': self.d_model,
            'num_heads': self.num_heads,
            'dff': self.dff,
            'dropout': self._dropout_rate,
            'activation': self._activation,
            'kernel_regularizer': reg_serialized,
        })
        return config


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

            # Volatility regime classification head (4-class: LOW/NORMAL/HIGH/EXTREME)
            # Used as entry timing filter - only trade when regime is HIGH (2) or EXTREME (3)
            self.volatility_regime_head = keras.Sequential([
                layers.Dense(64, kernel_regularizer=l2_reg),
                self.activation,
                layers.Dropout(dropout),
                layers.Dense(32, kernel_regularizer=l2_reg),
                self.activation,
                layers.Dropout(dropout * 0.5),
                layers.Dense(4, activation='softmax', name='volatility_regime_output')  # 4-class
            ], name='volatility_regime_head')

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

        for _i, (conv, residual_layer) in enumerate(zip(self.tcn_layers, self.residual_layers)):
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
            volatility_regime = self.volatility_regime_head(x, training=training)  # 4-class regime
            risk = self.risk_head(x, training=training)
            state_logits = self.state_head(x, training=training)

            return {
                'price': price,
                'trend': trend,
                'volatility_regime': volatility_regime,  # 4-class: LOW/NORMAL/HIGH/EXTREME
                'risk': risk,
                'state_logits': state_logits,
            }
        else:
            return self.fc(x, training=training)


# =============================================================================
# TCN VOLATILITY DUAL-HEAD MODEL - Forward Volatility Prediction
# =============================================================================

@register_keras_serializable()
class TCNVolatilityDualHead(Model):
    """
    Dual-head TCN for forward volatility regime prediction.

    Predicts FUTURE volatility (not current) with two output heads:
    1. Classification head: 4-class (QUIET_NEXT/STABLE_NEXT/ACTIVE_NEXT/EXTREME_NEXT)
    2. Regression head: % change in volatility (backup/tiebreaker)

    Combined loss: 0.7 * CategoricalFocalCrossentropy + 0.3 * MSE

    Architecture:
    - Shared TCN encoder (5 residual blocks, dilation 2^i)
    - Global average pooling
    - Two separate dense heads

    Key improvements over basic TCN:
    - Focal loss for class imbalance
    - Dual-head for regression fallback when classification uncertain
    - Anti-collapse mechanisms (variance monitoring)
    """

    def __init__(
        self,
        n_features: int,
        seq_len: int = 60,
        n_classes: int = 4,
        num_filters: int = 64,
        kernel_size: int = 5,
        num_residual_blocks: int = 5,
        dilation_base: int = 2,
        dropout: float = 0.2,
        l2_reg: float = 0.001,
        **kwargs
    ):
        super().__init__(**kwargs)

        self.n_features = n_features
        self.seq_len = seq_len
        self.n_classes = n_classes
        self.num_filters = num_filters
        self.kernel_size = kernel_size
        self.num_residual_blocks = num_residual_blocks
        self.dilation_base = dilation_base
        self.dropout_rate = dropout

        # L2 regularizer
        self.l2_regularizer = regularizers.L2(l2_reg) if l2_reg > 0 else None

        # Input regularization
        self.noise = layers.GaussianNoise(0.02)
        self.spatial_dropout = layers.SpatialDropout1D(dropout * 0.5)

        # Initial projection
        self.input_projection = layers.Conv1D(
            filters=num_filters,
            kernel_size=1,
            padding='same',
            kernel_regularizer=self.l2_regularizer,
            name='input_projection'
        )

        # Residual blocks
        self.residual_blocks = []
        for i in range(num_residual_blocks):
            dilation_rate = dilation_base ** i
            block = self._build_residual_block(
                num_filters, kernel_size, dilation_rate, dropout, i
            )
            self.residual_blocks.append(block)

        # Global pooling
        self.global_pool = layers.GlobalAveragePooling1D(name='global_pool')

        # Shared dense layer
        self.shared_dense = layers.Dense(
            64, activation='relu',
            kernel_regularizer=self.l2_regularizer,
            name='shared_fc'
        )
        self.shared_dropout = layers.Dropout(dropout)

        # Classification head (4-class)
        self.class_fc1 = layers.Dense(
            32, activation='relu',
            kernel_regularizer=self.l2_regularizer,
            name='class_fc1'
        )
        self.class_dropout = layers.Dropout(dropout)
        self.class_output = layers.Dense(
            n_classes, activation='softmax',
            dtype='float32',
            name='class_output'
        )

        # Regression head (% vol change)
        self.reg_fc1 = layers.Dense(
            32, activation='relu',
            kernel_regularizer=self.l2_regularizer,
            name='reg_fc1'
        )
        self.reg_dropout = layers.Dropout(dropout)
        self.reg_output = layers.Dense(
            1, activation='linear',
            dtype='float32',
            name='reg_output'
        )

    def _build_residual_block(self, filters, kernel_size, dilation_rate, dropout, idx):
        """Build a single residual block with dilated causal convolution."""
        return {
            'conv1': layers.Conv1D(
                filters, kernel_size,
                padding='causal',
                dilation_rate=dilation_rate,
                kernel_regularizer=self.l2_regularizer,
                name=f'res{idx}_conv1'
            ),
            'bn1': layers.BatchNormalization(name=f'res{idx}_bn1'),
            'conv2': layers.Conv1D(
                filters, kernel_size,
                padding='causal',
                dilation_rate=dilation_rate,
                kernel_regularizer=self.l2_regularizer,
                name=f'res{idx}_conv2'
            ),
            'bn2': layers.BatchNormalization(name=f'res{idx}_bn2'),
            'dropout': layers.Dropout(dropout),
        }

    def call(self, inputs, training=None):
        # Input shape: (batch, seq_len, n_features)
        x = self.noise(inputs, training=training)
        x = self.spatial_dropout(x, training=training)

        # Initial projection
        x = self.input_projection(x)

        # Residual blocks
        for block in self.residual_blocks:
            residual = x

            # First conv + BN + ReLU
            out = block['conv1'](x)
            out = block['bn1'](out, training=training)
            out = tf.nn.relu(out)

            # Second conv + BN + dropout
            out = block['conv2'](out)
            out = block['bn2'](out, training=training)
            out = block['dropout'](out, training=training)

            # Residual connection
            x = tf.nn.relu(residual + out)

        # Global pooling
        x = self.global_pool(x)

        # Shared dense
        shared = self.shared_dense(x)
        shared = self.shared_dropout(shared, training=training)

        # Classification head
        class_features = self.class_fc1(shared)
        class_features = self.class_dropout(class_features, training=training)
        class_output = self.class_output(class_features)

        # Regression head
        reg_features = self.reg_fc1(shared)
        reg_features = self.reg_dropout(reg_features, training=training)
        reg_output = self.reg_output(reg_features)

        return {
            'classification': class_output,  # (batch, n_classes)
            'regression': reg_output,         # (batch, 1)
        }

    def get_config(self):
        config = super().get_config()
        config.update({
            'n_features': self.n_features,
            'seq_len': self.seq_len,
            'n_classes': self.n_classes,
            'num_filters': self.num_filters,
            'kernel_size': self.kernel_size,
            'num_residual_blocks': self.num_residual_blocks,
            'dilation_base': self.dilation_base,
            'dropout': self.dropout_rate,
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


@register_keras_serializable()
class DualHeadLoss(keras.losses.Loss):
    """
    Combined loss for dual-head volatility model.

    Loss = classification_weight * focal_loss + regression_weight * mse_loss

    Default weights: 0.7 classification + 0.3 regression
    """

    def __init__(
        self,
        classification_weight: float = 0.7,
        regression_weight: float = 0.3,
        focal_gamma: float = 2.0,
        focal_alpha: Optional[List[float]] = None,
        name: str = 'dual_head_loss',
        **kwargs
    ):
        super().__init__(name=name, **kwargs)
        self.classification_weight = classification_weight
        self.regression_weight = regression_weight
        self.focal_gamma = focal_gamma
        # Default alpha: boost QUIET (0) and EXTREME (3), which are often minority
        self.focal_alpha = focal_alpha or [0.35, 0.25, 0.25, 0.15]

    def call(self, y_true, y_pred):
        """
        Compute combined loss.

        Args:
            y_true: Dict with 'classification' (one-hot) and 'regression' targets
            y_pred: Dict with 'classification' and 'regression' predictions
        """
        # Classification loss (categorical focal cross-entropy)
        y_class_true = y_true['classification']
        y_class_pred = y_pred['classification']

        # Clip predictions
        y_class_pred = tf.clip_by_value(y_class_pred, 1e-7, 1 - 1e-7)

        # Cross entropy
        ce = -y_class_true * tf.math.log(y_class_pred)

        # Focal weight
        pt = tf.reduce_sum(y_class_true * y_class_pred, axis=-1, keepdims=True)
        focal_weight = tf.pow(1 - pt, self.focal_gamma)

        # Alpha weighting per class
        alpha = tf.constant(self.focal_alpha, dtype=tf.float32)
        alpha_weight = tf.reduce_sum(y_class_true * alpha, axis=-1, keepdims=True)

        focal_loss = tf.reduce_mean(alpha_weight * focal_weight * tf.reduce_sum(ce, axis=-1))

        # Regression loss (MSE)
        y_reg_true = y_true['regression']
        y_reg_pred = y_pred['regression']
        mse_loss = tf.reduce_mean(tf.square(y_reg_true - y_reg_pred))

        # Variance regularization to prevent mode collapse
        # Penalize when prediction variance per class drops below threshold
        pred_variance = tf.math.reduce_variance(y_class_pred, axis=0)
        min_variance = 0.04  # Minimum acceptable variance per class
        variance_penalty = 0.1 * tf.reduce_mean(tf.maximum(min_variance - pred_variance, 0.0))

        # Combined loss
        total_loss = (self.classification_weight * focal_loss +
                      self.regression_weight * mse_loss +
                      variance_penalty)

        return total_loss

    def get_config(self):
        config = super().get_config()
        config.update({
            'classification_weight': self.classification_weight,
            'regression_weight': self.regression_weight,
            'focal_gamma': self.focal_gamma,
            'focal_alpha': self.focal_alpha,
        })
        return config


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


def build_transformer_direction_model(
    seq_len: int,
    n_features: int,
    d_model: int = 32,
    num_heads: int = 4,
    num_layers: int = 2,
    dff: int = 64,
    dropout: float = 0.2,
    l2_strength: float = 0.001,
    learning_rate: float = 0.001,
    input_noise_std: float = 0.02,
    spatial_dropout: float = 0.05,
    projection_dropout: float = 0.15,
    head_units: int = 16,
    head_activation: str = 'tanh',
    head_dropout: float = 0.15,
) -> Model:
    """
    Build a compiled Transformer for binary direction prediction.

    Uses PositionalEncoding and TransformerEncoderLayer from this module
    via functional API (compatible with .keras serialization).

    Layer names are chosen to match warm-start freezing patterns:
        'input_projection', 'positional_encoding', 'transformer_{i}', 'direction'
    """
    l2_reg = regularizers.l2(l2_strength) if l2_strength > 0 else None

    inp = layers.Input(shape=(seq_len, n_features), name="features")

    # Input augmentation
    x = layers.GaussianNoise(input_noise_std)(inp)
    x = layers.SpatialDropout1D(spatial_dropout)(x)

    # Project features to d_model dimension
    x = layers.Dense(d_model, kernel_regularizer=l2_reg, name='input_projection')(x)
    x = layers.Dropout(projection_dropout)(x)

    # Positional encoding
    x = PositionalEncoding(max_len=seq_len + 100, name='positional_encoding')(x)

    # Transformer encoder layers
    for i in range(num_layers):
        x = TransformerEncoderLayer(
            d_model=d_model,
            num_heads=num_heads,
            dff=dff,
            dropout=dropout,
            activation='relu',
            kernel_regularizer=l2_reg,
            name=f'transformer_{i}',
        )(x)

    # Global pooling and classification head
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dense(head_units, activation=head_activation, kernel_regularizer=l2_reg)(x)
    x = layers.Dropout(head_dropout)(x)

    # Binary direction output
    direction = layers.Dense(
        1,
        activation='sigmoid',
        name='direction',
        dtype='float32',
        kernel_initializer=keras.initializers.TruncatedNormal(mean=0.0, stddev=0.05),
        bias_initializer=keras.initializers.Zeros(),
    )(x)

    model = Model(inputs=inp, outputs=direction, name='transformer_direction')

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss=keras.losses.BinaryCrossentropy(label_smoothing=0.0),
        metrics=[keras.metrics.BinaryAccuracy(name='accuracy', threshold=0.5)],
    )

    return model


def build_transformer_regime_model(
    seq_len: int,
    n_features: int,
    d_model: int = 32,
    num_heads: int = 4,
    num_layers: int = 2,
    dff: int = 64,
    dropout: float = 0.2,
    num_classes: int = 3,
    head_units: int = 32,
    learning_rate: float = 0.001,
) -> Model:
    """
    Build a compiled Transformer for regime classification (N classes).

    Uses PositionalEncoding and TransformerEncoderLayer from this module
    via functional API (compatible with .keras serialization).

    No L2 regularization or input noise (simpler task than direction).
    """
    inp = layers.Input(shape=(seq_len, n_features), name="features")

    # Project input to d_model dimensions
    x = layers.Dense(d_model, name='input_projection')(inp)

    # Positional encoding
    x = PositionalEncoding(max_len=seq_len + 100, name='positional_encoding')(x)

    # Transformer encoder blocks
    for i in range(num_layers):
        x = TransformerEncoderLayer(
            d_model=d_model,
            num_heads=num_heads,
            dff=dff,
            dropout=dropout,
            activation='relu',
            kernel_regularizer=None,
            name=f'transformer_{i}',
        )(x)

    # Global pooling and classification head
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dense(head_units, activation='relu')(x)
    x = layers.Dropout(dropout)(x)

    # Softmax output
    regime = layers.Dense(num_classes, activation='softmax', name='regime', dtype='float32')(x)

    model = Model(inputs=inp, outputs=regime, name='transformer_regime')

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy'],
    )

    return model


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
    model_type = config.get('type', 'transformer').lower()

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
