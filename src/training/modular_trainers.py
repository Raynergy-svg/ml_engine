"""
Modular Trainers for Specialized Ensemble Models.

Each trainer handles ONE specific model type:
- TCNTrainer: Direction prediction (binary classification)
- XGBoostTrainer: Momentum analysis (regression + classification)
- RandomForestTrainer: Risk assessment (regression + classification)
- RidgeTrainer: Confidence scoring (regression 0-100)

Advanced Features (2025):
- EMA (Exponential Moving Average) shadow weights for stable inference
- EWC (Elastic Weight Consolidation) for multi-instrument continual learning
- Memory Replay Buffer to prevent catastrophic forgetting
- Training lineage tracking for model evolution analysis

No shared gradients. No joint loss. Each model trains independently.
"""

from __future__ import annotations

import hashlib
import json
import logging
import pickle
import warnings
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING

import numpy as np
import tensorflow as tf

if TYPE_CHECKING:
    import pandas as pd

logger = logging.getLogger(__name__)

# === Module-level constants for frequently used literals ===
MODEL_NOT_TRAINED_ERROR = "Model not trained"
PRODUCTION_MODELS_DIR = "trained_data/models"
META_PKL_SUFFIX = ".meta.pkl"
WEIGHTS_H5_SUFFIX = ".weights.h5"
ARCH_JSON_SUFFIX = ".arch.json"
EWC_PKL_SUFFIX = ".ewc.pkl"
EMA_PKL_SUFFIX = ".ema.pkl"
SERIALIZED_MODEL_WARNING = ".*serialized model.*"
UNPICKLE_ESTIMATOR_WARNING = ".*unpickle estimator.*"
LGBM_NOT_INSTALLED_ERROR = "LightGBM not installed. Run: pip install lightgbm"
JOINT_MODELS_DIR = "trained_data/models/joint"
TRANSFORMER_DIRECTION_FILENAME = "transformer_direction.keras"
LGBM_MOMENTUM_FILENAME = "lgbm_momentum.pkl"
LGBM_RISK_FILENAME = "lgbm_risk.pkl"
RIDGE_CONFIDENCE_FILENAME = "ridge_confidence.pkl"
NO_DIRECTION_DATA_ERROR = "No direction data found (tried 'direction' and 'tcn' keys)"
WEIGHTS_LOADED_FULL_MODEL_MSG = "✓ Loaded weights via full model load"


# === Pair Volatility Classification for Auto-Tuning ===
VOLATILE_PAIRS = {
    "GBP_JPY",
    "GBP_AUD",
    "GBP_NZD",
    "EUR_NZD",
    "AUD_JPY",
    "NZD_JPY",
    "CAD_JPY",
    "EUR_AUD",
    "GBP_CAD",
    "EUR_CAD",
}
STABLE_PAIRS = {"EUR_USD", "USD_CHF", "EUR_CHF", "EUR_GBP", "USD_SGD"}


def compute_auto_variance_weight(
    instrument: str, class_ratio: float, base_weight: float = 0.1
) -> float:
    """
    Auto-tune variance weight for AntiCollapseFocalLoss based on:
    1. Class imbalance severity (higher imbalance → higher weight)
    2. Pair volatility tier (volatile pairs need more collapse prevention)

    Args:
        instrument: Currency pair name (e.g., "EUR_USD") or "JOINT_*" for multi-pair
        class_ratio: Proportion of UP labels (0.0-1.0, 0.5 = balanced)
        base_weight: Base variance weight (default 0.1)

    Returns:
        Auto-tuned variance weight in range ~[0.06, 0.25]
    """
    # Imbalance factor: 0 when balanced (0.5), 1 when extreme (0 or 1)
    imbalance_severity = abs(class_ratio - 0.5) / 0.5

    # Pair volatility multiplier
    if instrument.startswith("JOINT"):
        # Joint multi-pair training: use moderate setting (mix of volatile/stable)
        pair_mult = 1.1  # Slightly above normal to handle mixed pair characteristics
    elif instrument in VOLATILE_PAIRS:
        pair_mult = 1.5  # Volatile pairs need stronger collapse prevention
    elif instrument in STABLE_PAIRS:
        pair_mult = 0.8  # Stable pairs need lighter touch
    else:
        pair_mult = 1.0  # Normal pairs

    # Final weight: base * (1 + imbalance_factor * 0.5) * pair_mult
    auto_weight = base_weight * (1 + imbalance_severity * 0.5) * pair_mult
    return max(0.05, min(0.25, auto_weight))  # Clamp to safe range


# =============================================================================
# DTYPE COMPATIBILITY HELPER
# =============================================================================


def _get_numpy_dtype(keras_dtype) -> np.dtype:
    """Convert Keras/TF dtype to numpy dtype, handling both Keras 2.x and 3.x.

    Keras 2.x: w.dtype is a TF dtype object with as_numpy_dtype attribute
    Keras 3.x: w.dtype may be a string like 'float32'

    Args:
        keras_dtype: dtype from a Keras weight (e.g., w.dtype)

    Returns:
        numpy dtype for array casting
    """
    # If it has as_numpy_dtype attribute (TF dtype object), use it
    if hasattr(keras_dtype, "as_numpy_dtype"):
        return keras_dtype.as_numpy_dtype

    # If it's already a numpy dtype, return as-is
    if isinstance(keras_dtype, np.dtype):
        return keras_dtype

    # If it's a string (Keras 3.x), convert manually
    if isinstance(keras_dtype, str):
        dtype_map = {
            "float32": np.float32,
            "float64": np.float64,
            "float16": np.float16,
            "int32": np.int32,
            "int64": np.int64,
            "bool": np.bool_,
        }
        if keras_dtype in dtype_map:
            return dtype_map[keras_dtype]
        # Try numpy's conversion
        return np.dtype(keras_dtype)

    # Fallback: try to convert via tf.dtypes
    try:
        return tf.dtypes.as_dtype(keras_dtype).as_numpy_dtype
    except Exception:
        # Last resort: default to float32
        logger.warning(f"Could not convert dtype {keras_dtype}, defaulting to float32")
        return np.float32


# =============================================================================
# CLEAN TRAINING OUTPUT HELPER
# =============================================================================

class TrainingDisplay:
    """Clean, professional training output using Rich."""

    def __init__(self, model_name: str = "Model"):
        from rich.console import Console
        from rich.table import Table
        from rich.panel import Panel
        from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
        self.console = Console()
        self.model_name = model_name
        self._table_cls = Table
        self._panel_cls = Panel
        self._progress_cls = Progress
        self._spinner_column_cls = SpinnerColumn
        self._text_column_cls = TextColumn
        self._bar_column_cls = BarColumn
        self._task_progress_column_cls = TaskProgressColumn

    def show_config(self, config: dict):
        """Display configuration as a clean table."""
        table = self._table_cls(show_header=False, box=None, padding=(0, 2))
        table.add_column("Key", style="dim")
        table.add_column("Value", style="cyan")
        for k, v in config.items():
            table.add_row(str(k), str(v))
        self.console.print(self._panel_cls(table, title=f"[bold]{self.model_name}[/bold]", border_style="blue"))

    def show_summary(self, metrics: dict, title: str = "Results"):
        """Display training results summary."""
        table = self._table_cls(show_header=False, box=None, padding=(0, 2))
        table.add_column("Metric", style="dim")
        table.add_column("Value", style="green")
        for k, v in metrics.items():
            if isinstance(v, float):
                table.add_row(str(k), f"{v:.4f}")
            else:
                table.add_row(str(k), str(v))
        self.console.print(self._panel_cls(table, title=f"[bold]{title}[/bold]", border_style="green"))

    def status(self, message: str, style: str = ""):
        """Print a status message."""
        self.console.print(f"  {message}", style=style)

    def warn(self, message: str):
        """Print a warning message."""
        self.console.print(f"  [yellow]⚠ {message}[/yellow]")

    def error(self, message: str):
        """Print an error message."""
        self.console.print(f"  [red]✗ {message}[/red]")

    def success(self, message: str):
        """Print a success message."""
        self.console.print(f"  [green]✓ {message}[/green]")


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
    patience: int = 20  # Original setting that achieved 60.9%
    min_epochs: int = 30  # Minimum epochs before early stopping can trigger
    verbose: int = 1
    quiet: bool = False  # Quiet mode: minimal console output, logs to file
    seq_len: int = (
        60  # Sequence length for sliding window (from config train_defaults.seq_len)
    )

    # TCN specific - AGGRESSIVE ANTI-OVERFITTING (val was 38%, worse than random)
    tcn_hidden_size: int = 16  # Reduced from 32 - smaller capacity
    tcn_num_layers: int = 1  # Reduced from 2 - simpler model
    tcn_kernel_size: int = 2  # Reduced from 3 - less receptive field
    tcn_dropout: float = 0.6  # Increased from 0.5 - heavy dropout
    tcn_l2_reg: float = 0.01  # NEW: L2 kernel regularization
    tcn_spatial_dropout: float = 0.3  # NEW: Spatial dropout on input
    tcn_noise_std: float = 0.05  # NEW: Input noise level

    # Transformer specific (for direction prediction)
    transformer_d_model: int = 32  # Model dimension
    transformer_num_heads: int = 4  # Number of attention heads
    transformer_num_layers: int = 2  # Number of encoder layers
    transformer_dff: int = 64  # Feedforward network dimension
    transformer_dropout: float = 0.2  # Dropout in transformer encoder layers

    # Transformer regularization — wired from YAML transformer section
    transformer_l2_reg: float = 0.003  # L2 kernel regularization weight
    transformer_input_noise: float = 0.02  # Gaussian noise on input
    transformer_spatial_dropout: float = 0.10  # SpatialDropout1D on input sequence
    transformer_projection_dropout: float = 0.10  # Dropout after input projection

    # Output head settings (for direction model) - stored in metadata for fallback rebuild
    final_dense_units: int = 16  # Units in final dense layer before output
    final_dense_activation: str = (
        "tanh"  # Activation: tanh allows [-1,1] range for balanced sigmoid
    )
    final_dense_dropout: float = 0.10  # Dropout before output layer (reduced from 0.15)

    # === CLASS-BALANCED LOSS SETTINGS (NEW - for extreme class imbalance) ===
    use_class_balanced_loss: bool = (
        True  # Use CB Loss instead of Focal Loss for extreme imbalance
    )
    cb_beta: float = 0.9999  # Class-balanced hyperparameter (higher = more aggressive)
    cb_gamma: float = 2.0  # Focusing parameter for CB Loss

    # === FOCAL LOSS SETTINGS (Anti-Bias - fallback if CB Loss disabled) ===
    use_focal_loss: bool = False  # DISABLED: CB Loss is better for extreme imbalance
    focal_gamma: float = (
        2.0  # Focusing parameter - higher = more focus on hard examples
    )
    focal_alpha: float = 0.5  # Class balance (0.5 = balanced, <0.5 = favor SHORT)
    focal_label_smoothing: float = 0.1  # Label smoothing to prevent overconfidence

    # === ANTI-COLLAPSE LOSS SETTINGS (v2) ===
    use_anti_collapse_loss: bool = (
        True  # Use AntiCollapseFocalLoss with variance regularization
    )
    use_hybrid_cb_anticollapse: bool = (
        True  # NEW: Hybrid CB + Anti-Collapse loss (takes priority)
    )
    anti_collapse_base_variance_weight: float = (
        0.2  # INCREASED: Base variance weight (auto-tuned per pair) - was 0.1
    )
    anti_collapse_label_smoothing: float = (
        0.05  # Label smoothing for anti-collapse loss
    )
    sample_weight_max_multiplier: float = 5.0  # Max sample weight (increased from 3.0)

    # === MADL LOSS SETTINGS (Directional Profitability) ===
    use_madl_loss: bool = (
        False  # Use MADL instead of Focal Loss (optimizes for trading profit)
    )
    madl_direction_weight: float = 0.7  # Weight for direction component vs BCE in MADL

    # XGBoost specific
    xgb_n_estimators: int = 200
    xgb_max_depth: int = 5
    xgb_learning_rate: float = 0.05
    use_gpu: bool = False  # Enable GPU acceleration for XGBoost (A100)

    # Random Forest specific
    rf_n_estimators: int = 200
    rf_max_depth: int = 10
    rf_min_samples_leaf: int = 10

    # ElasticNet specific (replaces Ridge for better feature selection)
    elasticnet_l1_ratios: List[float] = field(
        default_factory=lambda: [0.1, 0.5, 0.7, 0.9, 0.95, 1.0]
    )
    elasticnet_alphas: Optional[List[float]] = (
        None  # Auto-generate via logspace if None
    )
    elasticnet_cv_splits: int = 5  # TimeSeriesSplit folds
    elasticnet_max_iter: int = 10000

    # Legacy Ridge alpha (deprecated, use elasticnet_* instead)
    ridge_alpha: float = 1.0

    # Checkpoint directory for pair-specific models
    checkpoint_dir: str = "trained_data/checkpoints"

    # === CONTINUAL LEARNING SETTINGS (2025) ===

    # EMA (Exponential Moving Average) for stable inference
    use_ema: bool = True  # Enable EMA shadow weights
    ema_decay: float = 0.999  # α for EMA: θ_ema = α * θ_ema + (1-α) * θ
    ema_update_every: int = 16  # Update EMA every N training steps
    use_ema_for_inference: bool = True  # Use EMA weights for prediction

    # EWC (Elastic Weight Consolidation) for multi-instrument learning
    # Enabled by default for continual learning. Use --disable-ewc to disable.
    use_ewc: bool = True  # Enable EWC penalty on warm-start
    ewc_lambda: float = (
        100.0  # Strength of EWC constraint (λ) - reduced from 1000 to allow learning
    )
    ewc_gamma: float = 0.95  # Decay for old Fisher values when adding new tasks

    # Replay Buffer for catastrophic forgetting prevention
    use_replay_buffer: bool = True  # Enable memory replay
    replay_buffer_ratio: float = 0.10  # Save 10% of training data
    replay_mix_ratio: float = 0.20  # Mix 20% replay samples during training
    replay_buffer_dir: str = "trained_data/replay"  # Replay buffer storage

    # Warm-start settings (CRITICAL for preventing catastrophic forgetting)
    warm_start_lr_factor: float = (
        0.1  # FIXED: 0.1 = 10x LR reduction (was 0.01 = 100x, too aggressive!)
    )
    warm_start_freeze_encoder: bool = (
        True  # Freeze ALL transformer encoder layers (not just transformer_0)
    )
    warm_start_unfreeze_epochs: int = (
        10  # Unfreeze encoder after N epochs (0 = never unfreeze)
    )
    warm_start_gradual_unfreeze: bool = (
        True  # FIXED: Now enabled - gradually unfreeze layers from top to bottom
    )

    # Drift detection for auto-retraining
    drift_threshold: float = 0.03  # Retrain if val_acc drops by >3%

    # === OVERFIT PREVENTION SETTINGS ===
    overfit_threshold: float = 0.08       # 8% train-val gap triggers warning
    critical_threshold: float = 0.15      # 15% gap triggers intervention (LR reduction)
    severe_threshold: float = 0.25        # 25% gap triggers early stop
    max_acceptable_gap: float = 0.12      # Won't save checkpoint if gap > 12%
    overfit_patience_epochs: int = 3      # Consecutive overfit epochs before intervention
    auto_adjust_dropout: bool = True      # Dynamically increase dropout on overfitting
    auto_reduce_lr: bool = True           # Dynamically reduce LR on overfitting
    max_dropout_increase: float = 0.3     # Cap on dropout increase

    # === SWA (Stochastic Weight Averaging) ===
    enable_swa: bool = True               # Average weights in final 25% for flatter optima
    swa_start_fraction: float = 0.75      # Start SWA at 75% of training
    swa_lr_factor: float = 0.5            # SWA constant LR = initial_lr * factor

    # === COSINE ANNEALING WITH WARM RESTARTS ===
    enable_cosine_restarts: bool = True    # Periodic LR resets to escape sharp minima
    cosine_restart_period: int = 10       # Restart every N epochs
    cosine_restart_lr_mult: float = 0.8   # Each restart uses 80% of prev LR

    # === WARM-START OVERFIT RECOVERY ===
    enable_warmstart_detection: bool = True
    warmstart_reset_threshold: float = 0.15  # If initial gap > 15%, intervene
    weight_perturbation_scale: float = 0.02  # Noise to break memorization
    reset_optimizer_on_overfit: bool = True   # Reset momentum on critical overfit

    # Feature selection settings
    use_feature_selection: bool = True  # Enable RF importance-based feature selection
    feature_selection_method: str = "random_forest"  # 'random_forest' or 'f_test'
    top_k_features: int = 50  # Number of top features to select


# =============================================================================
# SHARED SEQUENCE CREATION UTILITIES
# =============================================================================


def create_sequences(
    X: np.ndarray, y: np.ndarray, seq_len: int
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create sliding window sequences from 2D data.

    Args:
        X: 2D array of shape (n_samples, n_features)
        y: 1D array of labels
        seq_len: Length of each sequence window

    Returns:
        x_seq: 3D array of shape (n_sequences, seq_len, n_features)
        y_seq: 1D array of labels (using label at end of each sequence)
    """
    x_seq, y_seq = [], []
    for i in range(len(X) - seq_len):
        x_seq.append(X[i : i + seq_len])
        y_seq.append(y[i + seq_len - 1])  # Label at end of sequence
    return np.array(x_seq), np.array(y_seq)


def create_sequences_with_weights(
    X: np.ndarray, y: np.ndarray, w: Optional[np.ndarray], seq_len: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Create sliding window sequences with sample weights.

    Args:
        X: 2D array of shape (n_samples, n_features)
        y: 1D array of labels
        w: 1D array of sample weights (or None for uniform weights)
        seq_len: Length of each sequence window

    Returns:
        x_seq: 3D array of shape (n_sequences, seq_len, n_features)
        y_seq: 1D array of labels
        w_seq: 1D array of weights
    """
    x_seq, y_seq, w_seq = [], [], []
    for i in range(len(X) - seq_len):
        x_seq.append(X[i : i + seq_len])
        y_seq.append(y[i + seq_len - 1])  # Label at end of sequence
        if w is not None:
            w_seq.append(w[i + seq_len - 1])  # Weight at end of sequence
        else:
            w_seq.append(1.0)
    return np.array(x_seq), np.array(y_seq), np.array(w_seq)


def get_config_seq_len(config: Optional[TrainerConfig], default: int = 60) -> int:
    """
    Get seq_len from config with validation.

    Args:
        config: TrainerConfig instance
        default: Default value if config is None

    Returns:
        seq_len value

    Raises:
        ValueError: If config exists but seq_len is missing
    """
    if config is None:
        return default

    seq_len = getattr(config, "seq_len", None)
    if seq_len is None:
        raise ValueError(
            "Config missing required 'seq_len' parameter. "
            "Add seq_len to config/config_improved_H1.yaml under train_defaults."
        )
    return seq_len


# =============================================================================
# WEIGHT SHAPE VALIDATION HELPER
# =============================================================================


def _safe_load_weights_ignoring_optimizer(
    model, path: str, *, skip_mismatch: bool = True
) -> bool:
    """
    Load model layer weights from a checkpoint, suppressing optimizer state warnings.

    When loading from a checkpoint with a different architecture, the Adam optimizer's
    moment estimates (m, v) may have a different variable count than the current model's
    trainable variables.  This produces noisy warnings like:
        "Adam optimizer currently has 2 variables whereas the saved optimizer state
         has 9 variables, causing a loading skip."

    These warnings are *benign* when the model is subsequently recompiled with a fresh
    optimizer (as happens during warm-start training), but they confuse users.  This
    helper intercepts them, logs them at DEBUG level, and returns success/failure.

    Args:
        model: The Keras model to load weights into.
        path: Path to the weights file (.h5 or .keras).
        skip_mismatch: If True, skip layers with incompatible shapes.

    Returns:
        True if weights were loaded successfully (even partially).
    """
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            model.load_weights(path, skip_mismatch=skip_mismatch)

        # Downgrade optimizer-related warnings to debug; keep others at warning
        for w in caught:
            msg = str(w.message)
            if any(kw in msg.lower() for kw in ("optimizer", "skipping variable", "loading skip")):
                logger.debug(f"Suppressed optimizer state warning: {msg}")
            else:
                logger.warning(f"Weight loading warning: {msg}")
        return True
    except Exception as e:
        logger.debug(f"Weight loading failed: {e}")
        return False


def _safe_reset_optimizer_state(model) -> None:
    """
    Reset optimizer slot variables (moments) to zeros after a checkpoint load.

    After loading weights from a structurally different checkpoint, Adam's first/second
    moment estimates are invalid.  Zeroing them forces Adam to rebuild its running
    averages from scratch on the next training step—equivalent to starting with a
    fresh optimizer while preserving the current learning-rate schedule.

    This is safe because:
    - The model layer weights themselves are already loaded and valid.
    - The optimizer is recompiled before training begins in the warm-start flow.
    """
    try:
        opt = getattr(model, "optimizer", None)
        if opt is None:
            return
        opt_vars = getattr(opt, "variables", [])
        if not opt_vars:
            return
        reset_count = 0
        for var in opt_vars:
            try:
                var.assign(tf.zeros_like(var))
                reset_count += 1
            except Exception:
                pass  # Non-assignable variable (e.g., iteration counter)
        if reset_count:
            logger.info(
                f"🔄 Optimizer state reset: zeroed {reset_count} slot variable(s) "
                f"(will rebuild during training)"
            )
    except Exception as exc:
        logger.debug(f"Optimizer reset skipped (not yet built): {exc}")


def _validate_weight_shapes(
    model_weights: List[np.ndarray],
    checkpoint_weights: List[np.ndarray],
    context: str = "weights",
) -> Tuple[bool, Optional[str]]:
    """
    Validate that checkpoint weights are compatible with model architecture.

    Compares tensor count and per-tensor shapes to ensure safe weight loading.
    This is more robust than count_params() which can match even when shapes differ.

    Args:
        model_weights: List of weight arrays from current model
        checkpoint_weights: List of weight arrays from checkpoint
        context: Description for error messages (e.g., "EMA weights", "model weights")

    Returns:
        Tuple of (is_compatible, error_message)
        - is_compatible: True if shapes match exactly
        - error_message: None if compatible, otherwise description of mismatch
    """
    if len(model_weights) != len(checkpoint_weights):
        return False, (
            f"{context} count mismatch: model has {len(model_weights)} tensors, "
            f"checkpoint has {len(checkpoint_weights)}"
        )

    for i, (mw, cw) in enumerate(zip(model_weights, checkpoint_weights)):
        model_shape = mw.shape if hasattr(mw, "shape") else np.array(mw).shape
        ckpt_shape = cw.shape if hasattr(cw, "shape") else np.array(cw).shape
        if model_shape != ckpt_shape:
            return False, (
                f"{context} shape mismatch at tensor {i}: "
                f"model expects {model_shape}, checkpoint has {ckpt_shape}"
            )

    return True, None


# =============================================================================
# EMA CALLBACK - Exponential Moving Average of Weights
# =============================================================================


class EMACallback:
    """
    Exponential Moving Average of model weights for stable inference.

    Maintains shadow weights: θ_ema = α * θ_ema + (1-α) * θ_current

    Benefits:
    - Smoother predictions in volatile markets
    - Reduces model jitter from noisy financial data
    - Better generalization without additional training cost

    Usage:
        ema = EMACallback(model, decay=0.999)
        # During training loop:
        ema.update()  # Call every N steps
        # For inference:
        ema.apply()  # Copy EMA weights to model
        model.predict(...)
        ema.restore()  # Restore training weights
    """

    def __init__(
        self,
        model,
        decay: float = 0.999,
        update_every: int = 16,
    ):
        self.model = model
        self.decay = decay
        self.update_every = update_every
        self.step_counter = 0
        self.ema_weights = None  # Dict[str, np.ndarray] keyed by weight name
        self.backup_weights = None
        self._initialized = False

    def _initialize_ema(self):
        """Initialize EMA weights as copy of current model weights."""
        self.ema_weights = [w.numpy().copy() for w in self.model.trainable_weights]
        self._initialized = True
        logger.info(
            f"📊 EMA initialized with {len(self.ema_weights)} weight tensors (decay={self.decay})"
        )

    def update(self, force: bool = False):
        """
        Update EMA weights with current model weights.

        Args:
            force: If True, update regardless of step counter
        """
        self.step_counter += 1

        if not force and self.step_counter % self.update_every != 0:
            return

        if not self._initialized:
            self._initialize_ema()
            return

        # Handle dynamic weight changes (e.g., when layers are unfrozen)
        current_weights = self.model.trainable_weights
        if len(current_weights) != len(self.ema_weights):
            logger.info(
                f"📊 EMA: Weights changed from {len(self.ema_weights)} to "
                f"{len(current_weights)} (layers unfrozen), reinitializing..."
            )
            # Preserve existing EMA weights where possible, add new ones
            old_ema = self.ema_weights
            self.ema_weights = []
            for i, w in enumerate(current_weights):
                if i < len(old_ema) and old_ema[i].shape == w.numpy().shape:
                    self.ema_weights.append(old_ema[i])
                else:
                    # New weight or shape changed - initialize from current
                    self.ema_weights.append(w.numpy().copy())
            logger.info(
                f"📊 EMA: Reinitialized with {len(self.ema_weights)} weight tensors"
            )

        # EMA update: θ_ema = α * θ_ema + (1-α) * θ_current
        for i, w in enumerate(self.model.trainable_weights):
            self.ema_weights[i] = (
                self.decay * self.ema_weights[i] + (1 - self.decay) * w.numpy()
            )

    def apply(self):
        """Apply EMA weights to model (for inference). Backs up current weights."""
        if not self._initialized:
            logger.warning("EMA not initialized, cannot apply")
            return

        # Handle dynamic weight changes
        current_weights = self.model.trainable_weights
        if len(current_weights) != len(self.ema_weights):
            logger.warning(
                f"EMA: Weight count mismatch ({len(self.ema_weights)} EMA vs "
                f"{len(current_weights)} model), skipping apply"
            )
            return

        # Backup current (training) weights
        self.backup_weights = [w.numpy().copy() for w in self.model.trainable_weights]

        # Apply EMA weights - cast to match dtype for Metal compatibility
        for w, ema_w in zip(self.model.trainable_weights, self.ema_weights):
            if w.shape == ema_w.shape:
                w.assign(ema_w.astype(_get_numpy_dtype(w.dtype)))

    def restore(self):
        """Restore original training weights after inference."""
        if self.backup_weights is None:
            return

        # Handle dynamic weight changes
        current_weights = self.model.trainable_weights
        if len(current_weights) != len(self.backup_weights):
            logger.warning("EMA: Weight count changed during apply, cannot restore")
            self.backup_weights = None
            return

        for w, backup_w in zip(self.model.trainable_weights, self.backup_weights):
            if w.shape == backup_w.shape:
                # Cast to match dtype for Metal/Keras 3.x compatibility
                w.assign(backup_w.astype(_get_numpy_dtype(w.dtype)))

        self.backup_weights = None

    def get_ema_weights(self) -> List[np.ndarray]:
        """Get EMA weights for saving (as list for backward compat, with names)."""
        if not self._initialized:
            self._initialize_ema()
        # Return as dict for named storage
        return self.ema_weights

    def _load_ema_weights_by_name(
        self,
        model_weights: list,
        weights: List[np.ndarray],
        weight_names: List[str],
    ) -> Tuple[int, int]:
        """Load EMA weights using name-based matching.

        Returns:
            Tuple of (loaded_count, skipped_count)
        """
        loaded_count = 0
        skipped_count = 0
        model_weight_names = [w.name for w in model_weights]
        checkpoint_weight_map = dict(zip(weight_names, weights))

        for i, (model_w, model_name) in enumerate(
            zip(model_weights, model_weight_names)
        ):
            loaded = self._try_load_by_exact_name(
                i, model_w, model_name, checkpoint_weight_map
            )
            if loaded:
                loaded_count += 1
                continue

            loaded = self._try_load_by_partial_name(
                i, model_w, model_name, checkpoint_weight_map
            )
            if loaded:
                loaded_count += 1
            else:
                skipped_count += 1

        return loaded_count, skipped_count

    def _try_load_by_exact_name(
        self, idx: int, model_w, model_name: str, checkpoint_map: dict
    ) -> bool:
        """Try to load weight by exact name match."""
        if model_name not in checkpoint_map:
            return False
        ckpt_weight = checkpoint_map[model_name]
        if model_w.shape != ckpt_weight.shape:
            return False
        self.ema_weights[idx] = ckpt_weight.copy()
        return True

    def _try_load_by_partial_name(
        self, idx: int, model_w, model_name: str, checkpoint_map: dict
    ) -> bool:
        """Try to load weight by partial name match (without scope prefix)."""
        base_name = model_name.split("/")[-1].split(":")[0]
        for ckpt_name, ckpt_weight in checkpoint_map.items():
            ckpt_base = ckpt_name.split("/")[-1].split(":")[0]
            if base_name == ckpt_base and model_w.shape == ckpt_weight.shape:
                self.ema_weights[idx] = ckpt_weight.copy()
                return True
        return False

    def _load_ema_weights_by_position(
        self,
        model_weights: list,
        weights: List[np.ndarray],
    ) -> Tuple[int, int]:
        """Load EMA weights using position-based matching with shape validation.

        Returns:
            Tuple of (loaded_count, skipped_count)
        """
        loaded_count = 0
        skipped_count = 0

        for i, model_w in enumerate(model_weights):
            if i >= len(weights):
                skipped_count += 1
                continue

            ckpt_weight = weights[i]
            if model_w.shape == ckpt_weight.shape:
                self.ema_weights[i] = ckpt_weight.copy()
                loaded_count += 1
            else:
                logger.debug(
                    f"  Skipping EMA weight {i}: shape mismatch "
                    f"(model={model_w.shape}, checkpoint={ckpt_weight.shape})"
                )
                skipped_count += 1

        return loaded_count, skipped_count

    def set_ema_weights(
        self, weights: List[np.ndarray], weight_names: List[str] = None,
        *, severe_ratio_threshold: float = 3.0,
    ):
        """Load EMA weights from checkpoint with graceful mismatch handling.

        Handles architectural differences between saved checkpoint and current model:
        - **Exact match** → fast-path load of all tensors.
        - **Severe mismatch** (tensor count ratio > ``severe_ratio_threshold``) →
          the checkpoint is too different to salvage; EMA is fully re-initialized
          from the current model weights so training starts with a clean slate.
        - **Moderate mismatch** → partial load: compatible weights (by name or
          position) are copied from the checkpoint; the rest are initialized from
          the current model weights.

        Args:
            weights: List of numpy arrays from checkpoint.
            weight_names: Optional list of weight names from checkpoint for
                name-based matching.
            severe_ratio_threshold: When ``max(n_model, n_ckpt) / min(n_model,
                n_ckpt)`` exceeds this value the checkpoint is considered too
                different and EMA is re-initialized entirely (default 3.0).
        """
        model_weights = self.model.trainable_weights
        model_weight_arrays = [w.numpy() for w in model_weights]
        n_model = len(model_weight_arrays)
        n_ckpt = len(weights)

        # ------------------------------------------------------------------
        # Fast path: exact match
        # ------------------------------------------------------------------
        is_compatible, error_msg = _validate_weight_shapes(
            model_weight_arrays, weights, context="EMA weights"
        )

        if is_compatible:
            self.ema_weights = [w.copy() for w in weights]
            self._initialized = True
            logger.info(f"📊 EMA weights loaded ({n_ckpt} tensors)")
            return

        # ------------------------------------------------------------------
        # Severe mismatch → full re-initialization
        # ------------------------------------------------------------------
        count_ratio = (
            max(n_model, n_ckpt) / max(min(n_model, n_ckpt), 1)
        )
        if count_ratio > severe_ratio_threshold:
            logger.info(
                f"📊 EMA architecture changed ({n_ckpt}→{n_model} tensors). "
                f"Re-initializing EMA from current model weights."
            )
            self._initialize_ema()
            return

        # ------------------------------------------------------------------
        # Moderate mismatch → partial load
        # ------------------------------------------------------------------
        logger.warning(f"⚠️ EMA weight mismatch: {error_msg}")
        logger.info("📊 Attempting graceful partial EMA weight loading...")

        # Seed from current model weights first (safe baseline)
        self.ema_weights = [w.numpy().copy() for w in model_weights]

        if weight_names and len(weight_names) == n_ckpt:
            loaded_count, skipped_count = self._load_ema_weights_by_name(
                model_weights, weights, weight_names
            )
        else:
            loaded_count, skipped_count = self._load_ema_weights_by_position(
                model_weights, weights
            )

        self._initialized = True

        total = loaded_count + skipped_count
        if loaded_count == 0 and total > 0:
            # Nothing useful was transferred — treat as full reinit
            logger.warning(
                "📊 EMA: no compatible weights found in checkpoint — "
                "fully re-initialized from current model"
            )
        elif skipped_count > 0:
            logger.warning(
                f"📊 EMA partial load: {loaded_count}/{total} weights loaded, "
                f"{skipped_count} re-initialized from current model"
            )
        else:
            logger.info(f"📊 EMA weights loaded ({loaded_count} tensors)")


# =============================================================================
# EWC - Elastic Weight Consolidation
# =============================================================================


class EWCPenalty:
    """
    Elastic Weight Consolidation for continual learning.

    Prevents catastrophic forgetting by adding a penalty that discourages
    large changes to weights that were important for previous tasks.

    Loss = L_new + (λ/2) * Σ F_i * (θ_i - θ_old)²

    Where F_i is the Fisher Information (importance) of weight i.

    References:
    - Kirkpatrick et al., "Overcoming catastrophic forgetting" (2016)
    - EAT: Experience-accumulated Transformer for stock prediction
    """

    def __init__(
        self,
        model,
        ewc_lambda: float = 1000.0,
        gamma: float = 0.95,  # Decay for old Fisher values
    ):
        self.model = model
        self.ewc_lambda = ewc_lambda
        self.gamma = gamma
        self.fisher_diagonal: Optional[List[np.ndarray]] = None
        self.reference_weights: Optional[List[np.ndarray]] = None
        self._n_tasks = 0

    def _sample_data(
        self, X: np.ndarray, y: np.ndarray, n_samples: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Sample subset of data for Fisher computation efficiency."""
        rng = np.random.default_rng(42)
        if len(X) > n_samples:
            indices = rng.choice(len(X), n_samples, replace=False)
            return X[indices], y[indices]
        return X, y

    def _detect_output_shape(self, x_sample: np.ndarray) -> int:
        """Detect model output shape for loss function selection."""
        try:
            test_pred = self.model(x_sample[:1], training=False)
            test_pred = self._extract_prediction(test_pred)
            return test_pred.shape[-1] if len(test_pred.shape) > 1 else 1
        except Exception:
            return 1  # Default to binary

    def _extract_prediction(self, pred):
        """Extract prediction tensor from model output (handles dict outputs)."""
        if isinstance(pred, dict):
            if "direction" in pred:
                return pred["direction"]
            return list(pred.values())[0]
        return pred

    def _select_loss_function(self, output_shape: int, y_sample: np.ndarray):
        """Select appropriate loss function based on output type."""
        import tensorflow as tf

        unique_labels = np.unique(y_sample)
        is_binary = output_shape == 1 or (
            len(unique_labels) <= 2 and max(unique_labels) <= 1
        )

        if is_binary:
            logger.debug("EWC using BinaryCrossentropy (binary classification)")
            return tf.keras.losses.BinaryCrossentropy(from_logits=False), True

        logger.debug("EWC using SparseCategoricalCrossentropy (multi-class)")
        return tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False), False

    def _compute_sample_gradients(
        self,
        x_sample: np.ndarray,
        y_sample: np.ndarray,
        loss_fn,
        is_binary: bool,
    ) -> List[np.ndarray]:
        """Compute accumulated squared gradients for all samples."""
        new_fisher = [np.zeros_like(w.numpy()) for w in self.model.trainable_weights]

        for i in range(len(x_sample)):
            grads = self._compute_single_sample_gradient(
                x_sample[i : i + 1], y_sample[i : i + 1], loss_fn, is_binary
            )
            for j, grad in enumerate(grads):
                if grad is not None:
                    new_fisher[j] += grad.numpy() ** 2

        return [f / len(x_sample) for f in new_fisher]

    def _compute_single_sample_gradient(
        self, x_i: np.ndarray, y_i: np.ndarray, loss_fn, is_binary: bool
    ) -> List:
        """Compute gradient for a single sample."""
        import tensorflow as tf

        if is_binary:
            y_i = tf.cast(y_i, tf.float32)

        with tf.GradientTape() as tape:
            pred = self.model(x_i, training=False)
            pred = self._extract_prediction(pred)
            loss = loss_fn(y_i, pred)

        return tape.gradient(loss, self.model.trainable_weights)

    def _update_fisher_diagonal(self, new_fisher: List[np.ndarray]):
        """Update Fisher diagonal with new values using decay."""
        if self.fisher_diagonal is not None:
            self.fisher_diagonal = [
                self.gamma * old_f + new_f
                for old_f, new_f in zip(self.fisher_diagonal, new_fisher)
            ]
        else:
            self.fisher_diagonal = new_fisher

    def compute_fisher(
        self,
        X: np.ndarray,
        y: np.ndarray,
        n_samples: int = 1000,
    ):
        """
        Compute Fisher Information diagonal after training on a task.

        Fisher_i ≈ E[(∂L/∂θ_i)²] - measures how sensitive the loss is to each weight.
        High Fisher = weight is important for this task.

        Args:
            X: Training features (sequences)
            y: Training labels
            n_samples: Number of samples to use for estimation
        """
        # Sample subset for efficiency
        x_sample, y_sample = self._sample_data(X, y, n_samples)

        # Detect output type and select loss function
        output_shape = self._detect_output_shape(x_sample)
        loss_fn, is_binary = self._select_loss_function(output_shape, y_sample)

        # Compute Fisher Information from gradients
        new_fisher = self._compute_sample_gradients(
            x_sample, y_sample, loss_fn, is_binary
        )

        # Update Fisher diagonal with decay
        self._update_fisher_diagonal(new_fisher)

        # Store reference weights
        self.reference_weights = [
            w.numpy().copy() for w in self.model.trainable_weights
        ]
        self._n_tasks += 1

        # Log Fisher statistics
        total_importance = sum(f.sum() for f in self.fisher_diagonal)
        logger.info(
            f"🧠 EWC Fisher computed: {self._n_tasks} task(s), total_importance={total_importance:.2f}"
        )

    def penalty(self) -> float:
        """
        Compute EWC penalty for current weights.

        Returns:
            Scalar penalty value (add to loss)
        """
        if self.fisher_diagonal is None or self.reference_weights is None:
            return 0.0

        import tensorflow as tf

        penalty = 0.0
        for w, f, w_old in zip(
            self.model.trainable_weights, self.fisher_diagonal, self.reference_weights
        ):
            penalty += tf.reduce_sum(f * (w - w_old) ** 2, axis=None)

        return (self.ewc_lambda / 2) * penalty

    def save(self, path: str):
        """Save EWC state (Fisher + reference weights)."""
        if self.fisher_diagonal is None:
            logger.warning("No EWC state to save (Fisher not computed)")
            return

        path = Path(path)
        data = {
            "fisher_diagonal": self.fisher_diagonal,
            "reference_weights": self.reference_weights,
            "ewc_lambda": self.ewc_lambda,
            "gamma": self.gamma,
            "n_tasks": self._n_tasks,
        }
        with open(path, "wb") as f:
            pickle.dump(data, f)
        logger.info(f"🧠 EWC state saved to {path}")

    def load(self, path: str):
        """Load EWC state from checkpoint with shape validation."""
        path = Path(path)
        if not path.exists():
            logger.info(f"No EWC checkpoint at {path}, starting fresh")
            return False

        with open(path, "rb") as f:
            data = pickle.load(f)

        # Validate Fisher diagonal shapes match current model
        model_weights = [w.numpy() for w in self.model.trainable_weights]
        fisher_weights = data.get("fisher_diagonal", [])
        reference_weights = data.get("reference_weights", [])

        if fisher_weights:
            is_compatible, error_msg = _validate_weight_shapes(
                model_weights, fisher_weights, context="EWC Fisher diagonal"
            )
            if not is_compatible:
                logger.debug(f"{error_msg}. EWC state incompatible, starting fresh.")
                return False

        if reference_weights:
            is_compatible, error_msg = _validate_weight_shapes(
                model_weights, reference_weights, context="EWC reference weights"
            )
            if not is_compatible:
                logger.debug(f"{error_msg}. EWC state incompatible, starting fresh.")
                return False

        self.fisher_diagonal = fisher_weights
        self.reference_weights = reference_weights
        self.ewc_lambda = data.get("ewc_lambda", self.ewc_lambda)
        self.gamma = data.get("gamma", self.gamma)
        self._n_tasks = data.get("n_tasks", 1)

        logger.info(
            f"🧠 EWC state loaded: {self._n_tasks} task(s), λ={self.ewc_lambda}"
        )
        return True


class EWCLoss(tf.keras.losses.Loss):
    """
    Custom loss class that includes EWC penalty.

    This wraps the base loss and adds the EWC constraint to prevent
    catastrophic forgetting during warm-start training. Using a class
    instead of a closure avoids issues with global/free variables.

    Args:
        base_loss: The base loss function (e.g., BinaryCrossentropy)
        ewc_penalty_fn: Callable that returns the EWC penalty (EWCPenalty.penalty)
        ewc_weight: Weight for the EWC term (usually 1.0, λ is in EWCPenalty)
    """

    def __init__(
        self,
        base_loss,
        ewc_penalty_fn,
        ewc_weight: float = 1.0,
        name: str = "ewc_loss",
    ):
        super().__init__(name=name)
        self.base_loss = base_loss
        self.ewc_penalty_fn = ewc_penalty_fn
        self.ewc_weight = ewc_weight

    def call(self, y_true, y_pred):
        """Compute loss with EWC penalty."""
        # Compute base classification loss
        classification_loss = self.base_loss(y_true, y_pred)

        # Add EWC penalty (protects important weights from previous tasks)
        ewc_term = self.ewc_penalty_fn()

        return classification_loss + self.ewc_weight * ewc_term


def create_ewc_loss(base_loss, ewc_penalty_fn, ewc_weight: float = 1.0):
    """
    Create a custom loss function that includes EWC penalty.

    This wraps the base loss and adds the EWC constraint to prevent
    catastrophic forgetting during warm-start training.

    Args:
        base_loss: The base loss function (e.g., BinaryCrossentropy)
        ewc_penalty_fn: Callable that returns the EWC penalty (EWCPenalty.penalty)
        ewc_weight: Weight for the EWC term (usually 1.0, λ is in EWCPenalty)

    Returns:
        Custom loss class compatible with Keras
    """
    return EWCLoss(base_loss, ewc_penalty_fn, ewc_weight)


def _safe_get_learning_rate(optimizer, default: float = 0.001) -> float:
    """
    Safely get learning rate, handling Keras 3.x / TF 2.16+ compatibility.

    In Keras 3.x with learning rate schedules, the learning_rate property
    may raise a TypeError when accessed. This helper handles both cases.

    Args:
        optimizer: Keras optimizer instance
        default: Default value to return if learning rate cannot be accessed

    Returns:
        The current learning rate as a float, or default if not accessible
    """
    try:
        lr = optimizer.learning_rate
        # Handle Variable, EagerTensor, or scalar
        if hasattr(lr, "numpy"):
            return float(lr.numpy())
        elif hasattr(lr, "value"):
            return float(lr.value())
        else:
            return float(lr)
    except (AttributeError, TypeError):
        # Learning rate is likely a schedule - return the schedule's current value
        try:
            # Try to get _learning_rate which might be the schedule
            lr_obj = getattr(optimizer, "_learning_rate", None)
            if lr_obj is not None:
                if callable(lr_obj) and hasattr(optimizer, "iterations"):
                    # It's a schedule - call it with current iteration
                    return float(lr_obj(optimizer.iterations))
                elif hasattr(lr_obj, "numpy"):
                    return float(lr_obj.numpy())
        except Exception:
            pass
        return default
    except Exception:
        return default


def _safe_set_learning_rate(optimizer, new_lr: float) -> bool:
    """
    Safely set learning rate, handling Keras 3.x / TF 2.16+ compatibility.

    In Keras 3.x with learning rate schedules, the learning_rate property
    may return an EagerTensor instead of a Variable, which doesn't have
    an assign() method. This helper handles both cases gracefully.

    Args:
        optimizer: Keras optimizer instance
        new_lr: New learning rate value

    Returns:
        True if successful, False otherwise
    """
    try:
        # First check if optimizer uses a LearningRateSchedule
        lr_obj = getattr(optimizer, "_learning_rate", None)
        if lr_obj is not None and callable(lr_obj) and not hasattr(lr_obj, "assign"):
            # It's a schedule - can't be modified
            return False

        # Try Keras 3.x / TF 2.16+ approach first
        lr = optimizer.learning_rate
        if hasattr(lr, "assign"):
            lr.assign(new_lr)
            return True
        if hasattr(optimizer, "_learning_rate") and hasattr(
            optimizer._learning_rate, "assign"
        ):
            # Some optimizers store the actual variable in _learning_rate
            optimizer._learning_rate.assign(new_lr)
            return True
        # Fallback: try set_value (works with older Keras)
        tf.keras.backend.set_value(optimizer.learning_rate, new_lr)
        return True
    except (AttributeError, TypeError):
        # Learning rate is likely a schedule - can't be modified directly
        # This is expected behavior when using LR schedules
        return False
    except Exception:
        return False


@dataclass
class OverfitPreventionConfig:
    """Configuration for OverfitPreventionCallback.

    Groups all configurable parameters to reduce __init__ parameter count
    and improve code readability.
    """
    # Threshold settings
    overfit_threshold: float = 0.08  # 8% gap triggers warning
    critical_threshold: float = 0.12  # 12% gap triggers intervention
    severe_threshold: float = 0.20  # 20% gap triggers early stop
    max_acceptable_gap: float = 0.10  # Won't save checkpoint if gap > 10%
    patience_epochs: int = 2  # Epochs before intervention
    auto_adjust_dropout: bool = True
    auto_reduce_lr: bool = True
    max_dropout_increase: float = 0.3

    # SWA settings - helps find flatter minima that generalize better
    enable_swa: bool = True  # Averages weights for better generalization
    swa_start_fraction: float = 0.5  # Start SWA at 50% of training
    swa_lr_factor: float = 0.5  # SWA uses lower constant LR

    # Cosine restart settings - helps escape local minima
    enable_cosine_restarts: bool = True  # Warm restarts improve convergence
    restart_period: int = 15  # Restart every 15 epochs
    restart_lr_mult: float = 0.9  # Each restart uses 90% of prev LR

    # Mixup augmentation (applied at batch level)
    enable_mixup: bool = False  # Disabled by default
    mixup_alpha: float = 0.2

    # Warm-start overfit recovery
    enable_warmstart_detection: bool = True
    warmstart_reset_threshold: float = 0.15  # If initial gap > 15%, intervene
    weight_perturbation_scale: float = 0.02  # Noise to break memorization
    reset_optimizer_on_overfit: bool = True  # Reset momentum on critical overfit

    # Continual learning: Previous best accuracy (prevents saving worse models)
    warm_start_best_acc: float = 0.0  # Set from loaded checkpoint


class OverfitPreventionCallback(tf.keras.callbacks.Callback):
    """
    Advanced callback to detect and mitigate overfitting during training.

    Based on research from:
    - Stochastic Weight Averaging (SWA) - Izmailov et al. (arXiv:1803.05407)
    - SGDR: Cosine Annealing with Warm Restarts (arXiv:1608.03983)
    - Mixup: Beyond Empirical Risk Minimization (arXiv:1710.09412)

    Key Features:
    1. Stochastic Weight Averaging (SWA) in final 25% of training
       - Averages weights to find flatter optima that generalize better
       - Research shows SGD finds boundary of flat region, SWA finds center

    2. Cosine Annealing with Warm Restarts
       - Periodically resets LR to escape local minima
       - Allows re-exploration when stuck in sharp optima

    3. Aggressive Early Intervention
       - Immediate LR reduction on overfitting detection
       - Dynamic dropout adjustment
       - L2 weight decay boost

    4. Gap-Gated Checkpointing
       - Only saves when val improves AND gap is acceptable
       - Prevents saving memorizing models

    5. Warm-Start Overfit Detection (NEW)
       - Detects if warm-started model is already overfitting
       - Can perturb weights to break memorization pattern
       - Resets optimizer momentum to allow fresh learning

    Key insight: A model with 15%+ train-val gap is memorizing, not learning.
    SWA helps find flatter optima that generalize better to validation.
    """

    def __init__(
        self,
        checkpoint_dir: str = "trained_data/checkpoints",
        model_name: str = "transformer",
        config: Optional["OverfitPreventionConfig"] = None,
    ):
        super().__init__()

        # Use provided config or create default
        cfg = config if config is not None else OverfitPreventionConfig()

        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.model_name = model_name

        # Threshold settings
        self.overfit_threshold = cfg.overfit_threshold
        self.critical_threshold = cfg.critical_threshold
        self.severe_threshold = cfg.severe_threshold
        self.max_acceptable_gap = cfg.max_acceptable_gap
        self.patience_epochs = cfg.patience_epochs
        self.auto_adjust_dropout = cfg.auto_adjust_dropout
        self.auto_reduce_lr = cfg.auto_reduce_lr
        self.max_dropout_increase = cfg.max_dropout_increase

        # SWA settings
        self.enable_swa = cfg.enable_swa
        self.swa_start_fraction = cfg.swa_start_fraction
        self.swa_lr_factor = cfg.swa_lr_factor
        self.swa_weights = None
        self.swa_count = 0
        self.swa_started = False

        # Cosine restart settings
        self.enable_cosine_restarts = cfg.enable_cosine_restarts
        self.restart_period = cfg.restart_period
        self.restart_lr_mult = cfg.restart_lr_mult
        self.current_restart_epoch = 0
        self.num_restarts = 0

        # Mixup settings
        self.enable_mixup = cfg.enable_mixup
        self.mixup_alpha = cfg.mixup_alpha

        # Warm-start overfit recovery
        self.enable_warmstart_detection = cfg.enable_warmstart_detection
        self.warmstart_reset_threshold = cfg.warmstart_reset_threshold
        self.weight_perturbation_scale = cfg.weight_perturbation_scale
        self.reset_optimizer_on_overfit = cfg.reset_optimizer_on_overfit
        self._warmstart_checked = False
        self._initial_weights_perturbed = False
        self._optimizer_reset_count = 0

        # State tracking - CRITICAL: Initialize from warm-start to prevent saving worse models
        self.best_val_acc = cfg.warm_start_best_acc  # Start from previous best (not 0!)
        self.best_val_acc_clean = cfg.warm_start_best_acc  # Best val_acc with acceptable gap
        self.warm_start_best_acc = cfg.warm_start_best_acc  # Store original for logging
        self.best_epoch = 0
        self.best_epoch_clean = 0
        self.overfit_epochs = 0  # Consecutive epochs with ANY overfitting
        self.critical_epochs = 0  # Consecutive epochs with CRITICAL overfitting
        self.val_acc_history = []
        self.train_acc_history = []
        self.gap_history = []
        self.lr_history = []
        self.dropout_adjustments = 0
        self.lr_reductions = 0
        self._console = None
        self._initial_lr = None
        self._total_epochs = None
        self._base_weights = None  # Store weights before overfitting

    @property
    def console(self):
        if self._console is None:
            from rich.console import Console

            self._console = Console()
        return self._console

    def on_train_begin(self, logs=None):
        """Capture initial learning rate and total epochs."""
        self._initial_lr = _safe_get_learning_rate(self.model.optimizer, default=0.001)

        # Try to get total epochs from params
        self._total_epochs = self.params.get("epochs", 100)

        self.console.print(
            f"  [dim]🧪 Advanced Training: SWA={self.enable_swa}, "
            f"CosineRestarts={self.enable_cosine_restarts}[/dim]"
        )

    def _perturb_weights(self, scale: float = 0.02):
        """
        Add small noise to weights to break memorization patterns.

        This helps escape sharp local minima where the model has memorized
        training data. Research shows that flat minima generalize better.
        """
        perturbed_count = 0
        for layer in self.model.layers:
            for weight in layer.trainable_weights:
                if "kernel" in weight.name or "weight" in weight.name:
                    # Add Gaussian noise proportional to weight magnitude
                    # Explicitly match dtype to prevent Metal/TF assertion failures
                    noise = tf.random.normal(
                        shape=weight.shape,
                        mean=0.0,
                        stddev=scale * tf.reduce_mean(tf.abs(weight), axis=None),
                        dtype=weight.dtype,
                    )
                    weight.assign_add(noise)
                    perturbed_count += 1

        if perturbed_count > 0:
            self.console.print(
                f"  [yellow]🎲 Weight perturbation applied (scale={scale:.1%}) "
                f"to {perturbed_count} layers[/yellow]"
            )
        return perturbed_count

    def _reinitialize_kernel_weight(self, weight) -> bool:
        """Reinitialize a kernel weight using Glorot uniform initialization.

        Returns:
            True if reinitialized, False otherwise
        """
        if "kernel" not in weight.name:
            return False
        # Glorot uniform initialization
        # Explicitly match dtype to prevent Metal/TF assertion failures
        fan_in = weight.shape[0]
        fan_out = weight.shape[1] if len(weight.shape) > 1 else 1
        limit = np.sqrt(6.0 / (fan_in + fan_out))
        new_weights = tf.random.uniform(
            weight.shape, -limit, limit, dtype=weight.dtype
        )
        weight.assign(new_weights)
        return True

    def _reinitialize_bias_weight(self, weight) -> bool:
        """Reinitialize a bias weight to zeros.

        Returns:
            True if reinitialized, False otherwise
        """
        if "bias" not in weight.name:
            return False
        weight.assign(tf.zeros_like(weight))
        return True

    def _reinitialize_dense_layers(self):
        """
        Reinitialize only Dense/output layers that tend to memorize most.

        Keeps convolutional/attention layers (feature extraction) but resets
        the classification head which often overfits first.
        """
        reinitialized = 0
        for layer in self.model.layers:
            # Reset Dense layers (classification head)
            if not isinstance(layer, tf.keras.layers.Dense):
                continue
            for weight in layer.trainable_weights:
                if self._reinitialize_kernel_weight(weight):
                    reinitialized += 1
                else:
                    self._reinitialize_bias_weight(weight)

        if reinitialized > 0:
            self.console.print(
                f"  [red]🔄 Reinitialized {reinitialized} Dense layer weights (classification head reset)[/red]"
            )
        return reinitialized

    def _full_weight_reset(self):
        """
        Completely reinitialize all model weights.

        Nuclear option when model is too far into memorization to recover.
        """
        self.console.print(
            "  [red bold]🔥 FULL WEIGHT RESET - Model too far into memorization[/red bold]"
        )

        # Save current architecture, reinitialize weights
        for layer in self.model.layers:
            if hasattr(layer, "kernel_initializer") and hasattr(layer, "kernel"):
                # Reinitialize kernel - cast to match dtype for Metal compatibility
                if layer.kernel is not None:
                    new_kernel = layer.kernel_initializer(layer.kernel.shape)
                    layer.kernel.assign(tf.cast(new_kernel, layer.kernel.dtype))
            if hasattr(layer, "bias_initializer") and hasattr(layer, "bias"):
                # Reinitialize bias - cast to match dtype for Metal compatibility
                if layer.bias is not None:
                    new_bias = layer.bias_initializer(layer.bias.shape)
                    layer.bias.assign(tf.cast(new_bias, layer.bias.dtype))

        # Reset optimizer
        self._reset_optimizer_state()

        # Reset LR to initial (use safe helper to handle LR schedules)
        if _safe_set_learning_rate(self.model.optimizer, self._initial_lr):
            self.console.print(
                f"  [cyan]📈 LR reset to initial: {self._initial_lr:.2e}[/cyan]"
            )

        self.console.print(
            "  [yellow]   Starting fresh - warm-start weights were too corrupted[/yellow]"
        )

    def _reset_optimizer_state(self):
        """
        Reset optimizer momentum/velocity to allow fresh gradient accumulation.
        Keras 3.x compatible.

        When overfitting, the optimizer momentum may be pointing toward
        memorization. Resetting allows re-exploration.
        """
        try:
            optimizer = self.model.optimizer

            # Keras 3.x: optimizer.variables is a property, not a method
            if callable(getattr(optimizer, "variables", None)):
                opt_vars = optimizer.variables()
            else:
                opt_vars = getattr(optimizer, "variables", [])

            reset_count = 0
            for var in opt_vars:
                var_name = getattr(var, "name", str(var)).lower()
                if (
                    "momentum" in var_name
                    or "velocity" in var_name
                    or "/m:" in var_name
                    or "/v:" in var_name
                ):
                    var.assign(tf.zeros_like(var))
                    reset_count += 1

            if reset_count > 0:
                self._optimizer_reset_count += 1
                self.console.print(
                    f"  [yellow]🔄 Optimizer momentum reset ({reset_count} vars, #{self._optimizer_reset_count}) - "
                    f"fresh gradient accumulation[/yellow]"
                )
            return True
        except Exception as e:
            logger.warning(f"Could not reset optimizer: {e}")
            return False

    def _check_warmstart_overfit(self, train_acc: float, val_acc: float, gap: float):
        """
        Check if warm-started model is already overfitting and take action.

        Called only on first epoch to detect problematic warm-start.
        """
        if self._warmstart_checked:
            return

        self._warmstart_checked = True

        if not self.enable_warmstart_detection:
            return

        # Check if this looks like a warm-start (high train acc on epoch 1)
        is_likely_warmstart = (
            train_acc > 0.65
        )  # Fresh model wouldn't have 65%+ on epoch 1

        if is_likely_warmstart and gap > self.warmstart_reset_threshold:
            self.console.print("  [red bold]⚠️ WARM-START OVERFIT DETECTED[/red bold]")
            self.console.print(
                f"  [yellow]   Initial gap={gap:.1%} suggests loaded weights are memorizing.[/yellow]"
            )

            # Severity determines action
            if gap > 0.20:  # 20%+ gap - nuclear option
                self.console.print(
                    "  [red]   Gap > 20% - applying AGGRESSIVE recovery (dense layer reset)[/red]"
                )
                self._reinitialize_dense_layers()
                self._reset_optimizer_state()
            else:
                self.console.print(
                    "  [yellow]   Applying recovery: weight perturbation + optimizer reset[/yellow]"
                )
                # Standard recovery
                self._perturb_weights(scale=self.weight_perturbation_scale)
                self._initial_weights_perturbed = True
                if self.reset_optimizer_on_overfit:
                    self._reset_optimizer_state()

            # Set LR to a moderate value for re-learning (use safe helper to handle LR schedules)
            recovery_lr = self._initial_lr * 2  # Higher LR for exploration
            if _safe_set_learning_rate(self.model.optimizer, recovery_lr):
                self.console.print(
                    f"  [cyan]📈 LR boosted to {recovery_lr:.2e} for recovery exploration[/cyan]"
                )

    def _get_cosine_lr(self, epoch: int, base_lr: float) -> float:
        """Calculate cosine annealing LR with warm restarts."""
        if not self.enable_cosine_restarts:
            return base_lr

        # Epoch within current restart cycle
        cycle_epoch = (epoch - self.current_restart_epoch) % self.restart_period

        # Cosine annealing: lr_t = lr_min + 0.5 * (lr_max - lr_min) * (1 + cos(pi * t / T))
        lr_min = base_lr * 0.01  # Min LR is 1% of base
        lr_max = base_lr * (
            self.restart_lr_mult**self.num_restarts
        )  # Decay with restarts

        cos_val = np.cos(np.pi * cycle_epoch / self.restart_period)
        new_lr = lr_min + 0.5 * (lr_max - lr_min) * (1 + cos_val)

        return new_lr

    def _update_swa_weights(self):
        """Update running average of weights for SWA."""
        if self.swa_weights is None:
            # Initialize SWA weights as copy of current weights
            self.swa_weights = [w.numpy().copy() for w in self.model.trainable_weights]
            self.swa_count = 1
            self.console.print(
                "  [magenta]🔄 SWA initialized - collecting weights for averaging[/magenta]"
            )
        else:
            # Running average: swa_w = (swa_w * n + w) / (n + 1)
            # Explicit float32 cast to prevent dtype promotion on Apple Silicon/Metal
            self.swa_count += 1
            for i, w in enumerate(self.model.trainable_weights):
                avg = (
                    self.swa_weights[i] * (self.swa_count - 1) + w.numpy()
                ) / self.swa_count
                self.swa_weights[i] = avg.astype(_get_numpy_dtype(w.dtype))

    def _apply_swa_weights(self):
        """Apply averaged weights to model."""
        if self.swa_weights is not None and self.swa_count > 1:
            for i, w in enumerate(self.model.trainable_weights):
                # Ensure dtype matches to prevent Metal/TF/Keras 3.x assertion failures
                w.assign(self.swa_weights[i].astype(_get_numpy_dtype(w.dtype)))
            self.console.print(
                f"  [magenta]✨ SWA applied: averaged {self.swa_count} weight snapshots[/magenta]"
            )
            return True
        return False

    def _store_base_weights(self):
        """Store current weights as backup before overfitting gets worse."""
        self._base_weights = [w.numpy().copy() for w in self.model.trainable_weights]

    def _restore_base_weights(self):
        """Restore weights to pre-overfit state."""
        if self._base_weights is not None:
            for i, w in enumerate(self.model.trainable_weights):
                # Ensure dtype matches to prevent Metal/TF/Keras 3.x assertion failures
                w.assign(self._base_weights[i].astype(_get_numpy_dtype(w.dtype)))
            self.console.print(
                "  [cyan]↩️ Restored weights to pre-overfit checkpoint[/cyan]"
            )

    # =========================================================================
    # Helper methods for on_epoch_end (refactored for lower cognitive complexity)
    # =========================================================================

    def _handle_stuck_detection(self, epoch: int) -> None:
        """Handle stuck detection and escalating interventions."""
        if self.warm_start_best_acc > 0:
            # In warm-start mode: DO NOT perturb weights or reset layers
            if self.critical_epochs >= 15 and self.critical_epochs % 15 == 0:
                self.console.print(
                    f"  [yellow]⚠️ Warm-start: {self.critical_epochs} epochs without improvement over baseline[/yellow]"
                )
                self.console.print(
                    "  [yellow]   Continuing training (will restore original weights if no improvement)[/yellow]"
                )
            return

        should_intervene = (
            self.critical_epochs >= 5
            and self.critical_epochs % 5 == 0
            and (
                not hasattr(self, "_last_perturbation_epoch")
                or epoch - self._last_perturbation_epoch >= 5
            )
        )
        if not should_intervene:
            return

        self._last_perturbation_epoch = epoch
        self.console.print(
            f"  [yellow]⚠️ Stuck in critical overfit for {self.critical_epochs} epochs[/yellow]"
        )
        self._apply_escalating_intervention()

    def _apply_escalating_intervention(self) -> None:
        """Apply escalating interventions based on how long we've been stuck."""
        if self.critical_epochs >= 20:
            self._intervention_nuclear()
        elif self.critical_epochs >= 15:
            self._intervention_large()
        elif self.critical_epochs >= 10:
            self._intervention_medium()
        else:
            self._intervention_small()

    def _intervention_nuclear(self) -> None:
        """20+ epochs stuck: Nuclear option - full dense layer reset."""
        self.console.print(
            "  [red bold]🔥 20+ epochs stuck - resetting Dense layers entirely[/red bold]"
        )
        self._reinitialize_dense_layers()
        self._reset_optimizer_state()
        # Use safe helper to handle LR schedules
        _safe_set_learning_rate(self.model.optimizer, self._initial_lr)
        self.critical_epochs = 0
        self.dropout_adjustments = 0
        self.lr_reductions = 0

    def _intervention_large(self) -> None:
        """15+ epochs: Larger perturbation + dense layer partial reset."""
        self.console.print(
            "  [red]💥 15+ epochs stuck - larger perturbation + partial reset[/red]"
        )
        self._perturb_weights(scale=self.weight_perturbation_scale * 2)
        self._reinitialize_dense_layers()
        self._reset_optimizer_state()

    def _intervention_medium(self) -> None:
        """10+ epochs: Medium perturbation."""
        self._perturb_weights(scale=self.weight_perturbation_scale)
        self._reset_optimizer_state()

    def _intervention_small(self) -> None:
        """5+ epochs: Small perturbation."""
        self._perturb_weights(scale=self.weight_perturbation_scale * 0.5)
        if self.reset_optimizer_on_overfit:
            self._reset_optimizer_state()

    def _handle_cosine_restarts(self, epoch: int) -> None:
        """Handle cosine annealing with warm restarts."""
        if not self.enable_cosine_restarts:
            return

        cycle_epoch = epoch - self.current_restart_epoch
        if cycle_epoch > 0 and cycle_epoch % self.restart_period == 0:
            self._apply_warm_restart(epoch)
        else:
            self._apply_cosine_decay(epoch)

    def _apply_warm_restart(self, epoch: int) -> None:
        """Apply warm restart - reset LR to initial value."""
        self.num_restarts += 1
        self.current_restart_epoch = epoch
        new_lr = self._initial_lr * (self.restart_lr_mult ** (self.num_restarts - 1))
        if _safe_set_learning_rate(self.model.optimizer, new_lr):
            self.console.print(
                f"  [magenta]🔄 Warm restart #{self.num_restarts}: LR reset to {new_lr:.2e}[/magenta]"
            )

    def _apply_cosine_decay(self, epoch: int) -> None:
        """Apply cosine decay within cycle."""
        new_lr = self._get_cosine_lr(epoch, self._initial_lr)
        if self.lr_reductions == 0:
            _safe_set_learning_rate(self.model.optimizer, new_lr)

    def _handle_swa(self, epoch: int, overfit_gap: float) -> None:
        """Handle Stochastic Weight Averaging."""
        if not self.enable_swa or not self._total_epochs:
            return

        swa_start_epoch = int(self._total_epochs * self.swa_start_fraction)
        if epoch < swa_start_epoch:
            return

        if not self.swa_started:
            self._start_swa_phase(epoch)

        if overfit_gap <= self.critical_threshold:
            self._update_swa_weights()

    def _start_swa_phase(self, epoch: int) -> None:
        """Start the SWA phase with reduced learning rate.
        
        IMPORTANT: Disables cosine restarts when SWA begins. SWA requires
        a stable/declining LR to average weights along a flat loss landscape.
        Cosine restarts would spike LR and destroy the averaging benefit.
        """
        self.swa_started = True
        # Disable cosine restarts — they conflict with SWA's constant low LR
        if self.enable_cosine_restarts:
            self.enable_cosine_restarts = False
            self.console.print(
                "  [magenta]⚙️ Cosine restarts disabled for SWA phase (stable LR needed)[/magenta]"
            )
        swa_lr = self._initial_lr * self.swa_lr_factor
        if _safe_set_learning_rate(self.model.optimizer, swa_lr):
            self.console.print(
                f"  [magenta]🎯 SWA phase started (epoch {epoch + 1}/{self._total_epochs}): "
                f"LR={swa_lr:.2e}[/magenta]"
            )

    def _handle_checkpointing(
        self, epoch: int, val_acc: float, overfit_gap: float
    ) -> None:
        """Handle model checkpointing based on validation accuracy and gap."""
        if val_acc > self.best_val_acc_clean and overfit_gap <= self.max_acceptable_gap:
            self._save_checkpoint(epoch, val_acc, overfit_gap)
        elif val_acc > self.best_val_acc_clean:
            self.console.print(
                f"  [yellow]⚠️ Val improved to {val_acc:.1%} but gap={overfit_gap:.1%} > "
                f"{self.max_acceptable_gap:.0%} - NOT saving[/yellow]"
            )
        elif self.warm_start_best_acc > 0 and val_acc < self.warm_start_best_acc:
            degradation = self.warm_start_best_acc - val_acc
            if degradation > 0.02:
                self.console.print(
                    f"  [red]⚠️ DEGRADATION: val={val_acc:.1%} < warm-start baseline={self.warm_start_best_acc:.1%} "
                    f"(-{degradation:.1%})[/red]"
                )

    def _save_checkpoint(self, epoch: int, val_acc: float, overfit_gap: float) -> None:
        """Save a model checkpoint."""
        self.best_val_acc_clean = val_acc
        self.best_epoch_clean = epoch + 1
        checkpoint_path = self.checkpoint_dir / f"{self.model_name}_best.keras"
        try:
            self.model.save(checkpoint_path)
            if self.warm_start_best_acc > 0:
                improvement = val_acc - self.warm_start_best_acc
                self.console.print(
                    f"  [green]💾 Checkpoint saved: val={val_acc:.1%}, gap={overfit_gap:.1%} "
                    f"(+{improvement:+.1%} from warm-start)[/green]"
                )
            else:
                self.console.print(
                    f"  [green]💾 Checkpoint saved: val={val_acc:.1%}, gap={overfit_gap:.1%}[/green]"
                )
        except Exception as e:
            logger.warning(f"Could not save checkpoint: {e}")

    def _check_warmstart_early_stop(self, epoch: int) -> bool:
        """Check if we should early stop due to warm-start failure. Returns True if should stop."""
        if self.warm_start_best_acc <= 0 or epoch < 15:
            return False
        if self.best_val_acc >= self.warm_start_best_acc:
            return False

        self.console.print(
            "\n  [bold red]🛑 CONTINUAL LEARNING FAILED: No improvement over baseline[/bold red]"
        )
        self.console.print(
            f"  [red]   Best achieved: {self.best_val_acc:.1%} < baseline: {self.warm_start_best_acc:.1%}[/red]"
        )
        self.console.print(
            "  [yellow]💡 Stopping early. Original model weights preserved.[/yellow]"
        )
        self.console.print(
            "  [cyan]   Try: More data, lower LR (--lr 0.00001), or different pair[/cyan]"
        )
        self.model.stop_training = True
        return True

    def _handle_severe_overfit(self, overfit_gap: float) -> bool:
        """Handle severe overfitting (>25% gap). Returns True if should stop."""
        self.critical_epochs += 1
        if self.critical_epochs < 2:
            return False

        self.console.print(
            f"  [red bold]🛑 SEVERE OVERFITTING: gap={overfit_gap:.1%} > "
            f"{self.severe_threshold:.0%}[/red bold]"
        )
        if self._apply_swa_weights():
            self.console.print(
                "  [yellow]💡 Applied SWA averaged weights before stopping.[/yellow]"
            )
        self.console.print(
            "  [yellow]💡 Model is memorizing training data. Stopping.[/yellow]"
        )
        self.console.print(
            "  [cyan]   Try: 1) More data, 2) Simplify model, "
            "3) Dropout 0.5+, 4) L2 regularization[/cyan]"
        )
        self.model.stop_training = True
        return True

    def _handle_critical_overfit(
        self, train_acc: float, val_acc: float, overfit_gap: float
    ) -> None:
        """Handle critical overfitting (15-25% gap)."""
        self.overfit_epochs += 1
        self.critical_epochs += 1

        self.console.print(
            f"  [red]⚠️ CRITICAL: train={train_acc:.1%} vs val={val_acc:.1%} "
            f"(gap={overfit_gap:.1%})[/red]"
        )

        if self.auto_reduce_lr and self.lr_reductions < 4:
            factor = 0.3 if self.critical_epochs >= 2 else 0.5
            self._reduce_learning_rate(factor=factor)

        if (
            self.auto_adjust_dropout
            and self.critical_epochs >= self.patience_epochs
            and self.dropout_adjustments < 5
        ):
            self._increase_dropout(aggressive=True)

    def _handle_warning_overfit(
        self, train_acc: float, val_acc: float, overfit_gap: float
    ) -> None:
        """Handle warning level overfitting (8-15% gap)."""
        self.overfit_epochs += 1
        self.critical_epochs = 0

        if self.overfit_epochs == 1:
            self.console.print(
                f"  [yellow]⚠️ Overfit warning: gap={overfit_gap:.1%} "
                f"(train={train_acc:.1%}, val={val_acc:.1%})[/yellow]"
            )

        if (
            self.auto_adjust_dropout
            and self.overfit_epochs >= self.patience_epochs
            and self.dropout_adjustments < 5
        ):
            self._increase_dropout(aggressive=False)
            self.overfit_epochs = 0

    def _classify_overfit_severity(
        self, train_acc: float, val_acc: float, overfit_gap: float
    ) -> bool:
        """Classify overfit severity and take action. Returns True if should stop training."""
        if overfit_gap > self.severe_threshold:
            return self._handle_severe_overfit(overfit_gap)
        elif overfit_gap > self.critical_threshold:
            self._handle_critical_overfit(train_acc, val_acc, overfit_gap)
        elif overfit_gap > self.overfit_threshold:
            self._handle_warning_overfit(train_acc, val_acc, overfit_gap)
        else:
            # HEALTHY: gap < 8%
            self.overfit_epochs = 0
            self.critical_epochs = 0
        return False

    # =========================================================================
    # Main on_epoch_end method (refactored)
    # =========================================================================

    def on_epoch_end(self, epoch, logs=None):
        """End of epoch callback - monitors overfitting and applies interventions."""
        if logs is None:
            return

        # Extract metrics
        train_acc = logs.get("accuracy", 0)
        val_acc = logs.get("val_accuracy", 0)
        overfit_gap = train_acc - val_acc

        # Track history
        self.train_acc_history.append(train_acc)
        self.val_acc_history.append(val_acc)
        self.gap_history.append(overfit_gap)

        # Warm-start overfit check (first epoch only)
        if epoch == 0:
            self._check_warmstart_overfit(train_acc, val_acc, overfit_gap)

        # Stuck detection and interventions
        self._handle_stuck_detection(epoch)

        # Cosine annealing with warm restarts
        self._handle_cosine_restarts(epoch)

        # Stochastic Weight Averaging
        self._handle_swa(epoch, overfit_gap)

        # Track best overall
        if val_acc > self.best_val_acc:
            self.best_val_acc = val_acc
            self.best_epoch = epoch + 1

        # Store weights when gap is healthy
        if (
            overfit_gap <= self.overfit_threshold
            and val_acc >= self.best_val_acc_clean * 0.98
        ):
            self._store_base_weights()

        # Checkpointing
        self._handle_checkpointing(epoch, val_acc, overfit_gap)

        # Warm-start early stop check
        if self._check_warmstart_early_stop(epoch):
            return

        # Classify overfit severity and take action
        if self._classify_overfit_severity(train_acc, val_acc, overfit_gap):
            return

    def _increase_dropout(self, aggressive: bool = False):
        """Dynamically increase dropout rates in the model."""
        if aggressive:
            dropout_delta = self.max_dropout_increase / 2  # 15% increase
        else:
            dropout_delta = self.max_dropout_increase / 4  # 7.5% increase

        adjusted_count = 0
        for layer in self.model.layers:
            if isinstance(layer, tf.keras.layers.Dropout):
                old_rate = layer.rate
                new_rate = min(0.7, old_rate + dropout_delta)  # Cap at 70% (was 60%)
                if new_rate > old_rate:
                    layer.rate = new_rate
                    adjusted_count += 1

        if adjusted_count > 0:
            self.dropout_adjustments += 1
            adj_type = "aggressive" if aggressive else "mild"
            self.console.print(
                f"  [cyan]🔧 Dropout +{dropout_delta:.0%} ({adj_type}) → "
                f"adjustment #{self.dropout_adjustments}[/cyan]"
            )

    def _reduce_learning_rate(self, factor: float = 0.5):
        """Reduce learning rate to slow down memorization. Keras 3.x compatible."""
        try:
            optimizer = self.model.optimizer
            # Use safe helper to get current LR (handles LR schedules)
            current_lr = _safe_get_learning_rate(optimizer, default=0.001)
            new_lr = max(current_lr * factor, 1e-6)

            # Use safe helper for Keras 3.x compatibility
            if _safe_set_learning_rate(optimizer, new_lr):
                self.lr_reductions += 1
                self.console.print(
                    f"  [cyan]📉 LR reduced: {current_lr:.2e} → {new_lr:.2e} "
                    f"(x{factor}, #{self.lr_reductions})[/cyan]"
                )
        except Exception as e:
            logger.warning(f"Could not reduce LR: {e}")

    def _save_swa_model(self):
        """Save SWA model to production and backup paths."""
        try:
            # Save to production path (trained_data/models/) not just checkpoints
            production_dir = Path(PRODUCTION_MODELS_DIR)
            production_dir.mkdir(parents=True, exist_ok=True)
            production_path = production_dir / f"{self.model_name}.keras"

            self.model.save(production_path)
            self.console.print(
                f"  [bold magenta]💾 SWA weights auto-saved as primary model: {production_path}[/bold magenta]"
            )

            # Also save to checkpoint dir for backup
            swa_checkpoint_path = (
                self.checkpoint_dir / f"{self.model_name}_swa.keras"
            )
            self.model.save(swa_checkpoint_path)
            self.console.print(
                f"  [dim]💾 SWA backup saved: {swa_checkpoint_path}[/dim]"
            )
        except Exception as e:
            logger.warning(f"Could not save SWA model: {e}")

    def _print_best_checkpoint_summary(self):
        """Print summary of best checkpoint."""
        if self.best_epoch_clean > 0:
            self.console.print(
                f"  [bold green]💾 Best clean checkpoint: epoch {self.best_epoch_clean} "
                f"(val={self.best_val_acc_clean:.1%})[/bold green]"
            )
        elif self.best_epoch > 0:
            self.console.print(
                f"  [yellow]⚠️ No clean checkpoint saved. Best val was {self.best_val_acc:.1%} "
                f"at epoch {self.best_epoch} but with excessive overfitting.[/yellow]"
            )

    def _print_gap_statistics(self):
        """Print gap statistics and suggestions."""
        if len(self.gap_history) == 0:
            return

        avg_gap = np.mean(self.gap_history)
        max_gap = max(self.gap_history)
        min_gap = min(self.gap_history)
        recent_gaps = self.gap_history[-10:] if len(self.gap_history) >= 10 else self.gap_history
        recent_avg_gap = np.mean(recent_gaps)

        self.console.print(
            f"  [dim]📊 Gap stats: min={min_gap:.1%}, avg={avg_gap:.1%}, max={max_gap:.1%}[/dim]"
        )

        if self.num_restarts > 0:
            self.console.print(f"  [dim]🔄 Warm restarts: {self.num_restarts}[/dim]")

        self._print_suggestions(recent_avg_gap, max_gap)

    def _print_suggestions(self, recent_avg_gap: float, max_gap: float):
        """Print context-aware suggestions based on gap and accuracy."""
        best_val = self.best_val_acc_clean if self.best_val_acc_clean > 0 else self.best_val_acc

        if recent_avg_gap > self.critical_threshold:
            self.console.print(
                "  [yellow]💡 Overfitting detected: try stronger regularization, "
                "more dropout, or more data[/yellow]"
            )
        elif recent_avg_gap < 0.05 and best_val < 0.55:
            self.console.print(
                "  [dim]💡 Model generalizes well (low gap) but accuracy is limited. "
                "Try: more/better features, longer training, or accept market noise limits.[/dim]"
            )
        elif max_gap > self.critical_threshold and recent_avg_gap < 0.05:
            self.console.print(
                "  [green]💡 Model recovered from early overfitting (regularization worked)[/green]"
            )

    def on_train_end(self, logs=None):
        # Apply SWA weights at the end if we collected any
        if self.enable_swa and self.swa_count > 1:
            self.console.print(
                f"  [magenta]📊 SWA collected {self.swa_count} weight snapshots[/magenta]"
            )
            if self._apply_swa_weights():
                self._save_swa_model()

        self._print_best_checkpoint_summary()
        self._print_gap_statistics()


class EWCTrainingCallback(tf.keras.callbacks.Callback):
    """
    Callback to log EWC penalty during training.

    Helps monitor if the EWC constraint is being applied and its magnitude.
    """

    def __init__(self, ewc_penalty: EWCPenalty, log_every: int = 10):
        super().__init__()
        self.ewc_penalty = ewc_penalty
        self.log_every = log_every
        self._batch_count = 0

    def on_train_batch_end(self, batch, logs=None):
        self._batch_count += 1
        if self._batch_count % self.log_every == 0:
            if self.ewc_penalty and self.ewc_penalty.fisher_diagonal is not None:
                penalty_val = float(self.ewc_penalty.penalty())
                if logs is not None:
                    logs["ewc_penalty"] = penalty_val

    def on_epoch_end(self, epoch, logs=None):
        pass  # Suppressed - RichEpochCallback handles display


class QuietProgressCallback(tf.keras.callbacks.Callback):
    """
    Minimal progress bar for quiet mode training.

    Shows a single updating line:
    Training ━━━━━━━━━━━━━━━━ 45% val=56.6% best=58.2%
    """

    def __init__(self, model_name: str = "Model", total_epochs: int = 100):
        super().__init__()
        self.model_name = model_name
        self.total_epochs = total_epochs
        self.best_val_acc = 0.0
        self._console = None
        self._progress = None
        self._task = None

    @property
    def console(self):
        if self._console is None:
            from rich.console import Console

            self._console = Console()
        return self._console

    def on_train_begin(self, logs=None):
        from rich.progress import (
            Progress,
            BarColumn,
            TextColumn,
            TaskProgressColumn,
            TimeRemainingColumn,
        )

        self._progress = Progress(
            TextColumn(f"[cyan]{self.model_name}[/cyan]"),
            BarColumn(bar_width=30),
            TaskProgressColumn(),
            TextColumn("[dim]val=[/dim]"),
            TextColumn("{task.fields[val_acc]:.1%}", style="green"),
            TextColumn("[dim]best=[/dim]"),
            TextColumn("{task.fields[best_acc]:.1%}", style="bold green"),
            TimeRemainingColumn(),
            console=self.console,
            transient=True,  # Disappears when done
        )
        self._progress.start()
        self._task = self._progress.add_task(
            "training", total=self.total_epochs, val_acc=0.0, best_acc=0.0
        )

    def on_epoch_end(self, epoch, logs=None):
        if logs is None or self._progress is None:
            return

        val_acc = logs.get("val_accuracy", logs.get("accuracy", 0))
        if val_acc > self.best_val_acc:
            self.best_val_acc = val_acc

        self._progress.update(
            self._task, completed=epoch + 1, val_acc=val_acc, best_acc=self.best_val_acc
        )

    def on_train_end(self, logs=None):
        if self._progress:
            self._progress.stop()
        self.console.print(
            f"[green]✓ {self.model_name}:[/green] best val_accuracy={self.best_val_acc:.1%}"
        )


class GradualUnfreezeCallback(tf.keras.callbacks.Callback):
    """
    Gradually unfreeze encoder layers during warm-start training.

    This callback implements the declared but previously unimplemented
    warm_start_unfreeze_epochs and warm_start_gradual_unfreeze config options.

    After `unfreeze_after_epochs`:
    - If gradual=True: Unfreeze layers from top (classification head) to bottom
    - If gradual=False: Unfreeze all layers at once

    BatchNormalization layers are kept frozen to preserve running statistics.
    """

    def __init__(
        self,
        unfreeze_after_epochs: int = 10,
        gradual: bool = True,
        learning_rate_boost: float = 2.0,  # Boost LR when unfreezing
    ):
        super().__init__()
        self.unfreeze_after_epochs = unfreeze_after_epochs
        self.gradual = gradual
        self.learning_rate_boost = learning_rate_boost
        self._unfrozen = False
        self._initial_lr = None
        self._gradual_unfreezes = 0  # Track gradual unfreeze steps

    def on_train_begin(self, logs=None):
        # Store initial learning rate - use safe helper for Keras 3.x compatibility
        self._initial_lr = _safe_get_learning_rate(self.model.optimizer, default=0.001)

    def on_epoch_begin(self, epoch, logs=None):
        if self.unfreeze_after_epochs <= 0:
            return  # Never unfreeze

        if epoch == self.unfreeze_after_epochs:
            self._unfreeze_layers(epoch)
        elif self.gradual and epoch > self.unfreeze_after_epochs:
            # For gradual unfreezing, unfreeze more layers every 5 epochs
            if (epoch - self.unfreeze_after_epochs) % 5 == 0:
                self._gradual_unfreeze_more(epoch)

    def _can_unfreeze_layer(self, layer) -> bool:
        """Check if a layer can be unfrozen (not BatchNorm and currently frozen)."""
        if isinstance(layer, tf.keras.layers.BatchNormalization):
            return False
        return not layer.trainable

    def _unfreeze_layer_list(self, layers: list) -> int:
        """Unfreeze a list of layers. Returns count of unfrozen layers."""
        unfrozen_count = 0
        for layer in layers:
            if self._can_unfreeze_layer(layer):
                layer.trainable = True
                unfrozen_count += 1
        return unfrozen_count

    def _boost_learning_rate(self, unfrozen_count: int):
        """Boost learning rate if layers were unfrozen."""
        if not self._initial_lr or unfrozen_count == 0:
            return
        new_lr = self._initial_lr * self.learning_rate_boost
        if _safe_set_learning_rate(self.model.optimizer, new_lr):
            logger.info(
                f"   LR boosted: {self._initial_lr:.2e} → {new_lr:.2e} (factor={self.learning_rate_boost})"
            )

    def _unfreeze_layers(self, epoch: int):
        """Unfreeze layers (all at once or start gradual unfreezing)."""
        if self._unfrozen:
            return

        if self.gradual:
            # Start by unfreezing only the last few layers (closest to output)
            unfrozen_count = self._unfreeze_layer_list(self.model.layers[-5:])
            logger.info(
                f"🔓 Epoch {epoch}: Gradual unfreeze started - {unfrozen_count} layers now trainable"
            )
        else:
            # Unfreeze all at once (except BatchNorm)
            unfrozen_count = self._unfreeze_layer_list(self.model.layers)
            logger.info(
                f"🔓 Epoch {epoch}: Unfroze {unfrozen_count} layers (BatchNorm kept frozen)"
            )

        self._boost_learning_rate(unfrozen_count)
        self._unfrozen = True
        self._gradual_unfreezes = 1

    def _gradual_unfreeze_more(self, epoch: int):
        """Unfreeze additional layers for gradual unfreezing."""
        # Find frozen layers and unfreeze from top to bottom
        frozen_layers = [
            layer
            for layer in self.model.layers
            if self._can_unfreeze_layer(layer)
        ]

        if not frozen_layers:
            return  # All unfrozen

        # Unfreeze 2 more layers each time
        to_unfreeze = frozen_layers[-2:] if len(frozen_layers) >= 2 else frozen_layers
        unfrozen_names = []

        for layer in to_unfreeze:
            layer.trainable = True
            unfrozen_names.append(layer.name)

        self._gradual_unfreezes += 1
        remaining = len(frozen_layers) - len(to_unfreeze)
        logger.info(
            f"🔓 Epoch {epoch}: Gradual unfreeze step {self._gradual_unfreezes} - "
            f"unfroze {unfrozen_names}, {remaining} layers still frozen"
        )


class RichEpochCallback(tf.keras.callbacks.Callback):
    """
    Rich-formatted epoch display with color-coded metrics.

    Colors:
    - Green: Improving / Best
    - Yellow: Slight degradation
    - Red: Significant degradation
    - Cyan: Neutral / Info
    """

    def __init__(
        self,
        model_name: str = "Model",
        total_epochs: int = 100,
        warm_start_best_acc: float = 0.0,
        quiet: bool = False,
    ):
        super().__init__()
        self.model_name = model_name
        self.total_epochs = total_epochs
        self.warm_start_best_acc = warm_start_best_acc
        self.best_val_acc = (
            warm_start_best_acc  # Start from warm-start baseline, not 0!
        )
        self.best_val_loss = float("inf")
        self.best_epoch = 0
        self.prev_val_acc = warm_start_best_acc  # Start comparison from baseline
        self.prev_val_loss = float("inf")
        self._console = None
        self.quiet = quiet

    @property
    def console(self):
        if self._console is None:
            from rich.console import Console

            self._console = Console()
        return self._console

    def on_train_begin(self, logs=None):
        if self.quiet:
            return  # Skip verbose output
        if self.warm_start_best_acc > 0:
            self.console.print(
                f"[dim]Training {self.model_name} for up to {self.total_epochs} epochs...[/dim]"
            )
            self.console.print(
                f"  [cyan]🎯 Warm-start baseline: {self.warm_start_best_acc:.1%} (must beat to save)[/cyan]"
            )
        else:
            self.console.print(
                f"[dim]Training {self.model_name} for up to {self.total_epochs} epochs...[/dim]"
            )

    def _get_accuracy_color_and_status(
        self, val_acc: float, is_best: bool, below_baseline: bool
    ) -> tuple:
        """Determine color and status message for accuracy display."""
        if is_best and not below_baseline:
            return "bold green", "⭐ BEST"
        if is_best and below_baseline:
            diff = self.warm_start_best_acc - val_acc
            return "yellow", f"↗ best this run (baseline -{diff:.1%})"
        if below_baseline:
            diff = self.warm_start_best_acc - val_acc
            return "red", f"⚠ below baseline ({diff:.1%})"
        if val_acc >= self.prev_val_acc:
            return "green", "↗ improving"
        if val_acc >= self.prev_val_acc - 0.02:
            return "yellow", "→ stable"
        return "red", "↘ degrading"

    def _get_loss_color(self, val_loss: float) -> str:
        """Determine color for loss display."""
        import math
        if val_loss is None or (isinstance(val_loss, float) and math.isnan(val_loss)):
            return "yellow"
        if val_loss < self.prev_val_loss:
            return "green"
        if val_loss <= self.prev_val_loss * 1.1:
            return "yellow"
        return "red"

    def on_epoch_end(self, epoch, logs=None):
        if logs is None:
            return

        epoch_num = epoch + 1
        val_acc = logs.get("val_accuracy", logs.get("accuracy", 0))
        val_loss = logs.get("val_loss", logs.get("loss", 0))
        train_acc = logs.get("accuracy", 0)
        lr = logs.get("lr", logs.get("learning_rate", 0))

        # Determine if this is best epoch (must beat warm-start baseline!)
        is_best = val_acc > self.best_val_acc
        if is_best:
            self.best_val_acc = val_acc
            self.best_val_loss = val_loss
            self.best_epoch = epoch_num

        if self.quiet:
            self.prev_val_acc = val_acc
            self.prev_val_loss = val_loss
            return  # Skip verbose output

        # Color coding based on performance - RELATIVE TO WARM-START BASELINE
        below_baseline = (
            self.warm_start_best_acc > 0 and val_acc < self.warm_start_best_acc
        )
        acc_color, status = self._get_accuracy_color_and_status(
            val_acc, is_best, below_baseline
        )
        loss_color = self._get_loss_color(val_loss)

        # Format output
        epoch_str = f"[cyan]Epoch {epoch_num:3d}/{self.total_epochs}[/cyan]"
        acc_str = f"[{acc_color}]acc={val_acc:.1%}[/{acc_color}]"
        import math
        if val_loss is None or (isinstance(val_loss, float) and math.isnan(val_loss)):
            loss_str = "[yellow]loss=N/A[/yellow]"
        else:
            loss_str = f"[{loss_color}]loss={val_loss:.4f}[/{loss_color}]"
        train_str = f"[dim]train={train_acc:.1%}[/dim]"
        lr_str = f"[dim]lr={lr:.2e}[/dim]" if lr > 0 else ""
        status_str = f"[{acc_color}]{status}[/{acc_color}]"

        self.console.print(
            f"  {epoch_str} | {acc_str} {loss_str} | {train_str} {lr_str} | {status_str}"
        )

        self.prev_val_acc = val_acc
        self.prev_val_loss = val_loss

    def on_train_end(self, logs=None):
        if self.quiet:
            return  # Skip verbose output
        self.console.print(
            f"  [bold green]✓ Best: epoch {self.best_epoch} with val_accuracy={self.best_val_acc:.1%}[/bold green]"
        )


class AutoAdjustCallback(tf.keras.callbacks.Callback):
    """
    Auto-adjusts training when stuck (plateau detection).

    Actions taken when stuck:
    1. Reduce learning rate by factor
    2. If still stuck, increase dropout via noise injection
    3. Log adjustments for transparency
    """

    def __init__(
        self,
        patience: int = 5,          # Epochs without improvement before adjusting
        lr_factor: float = 0.5,     # LR reduction factor
        min_lr: float = 1e-6,       # Minimum learning rate
        max_adjustments: int = 3,   # Maximum number of adjustments
        min_delta: float = 0.005,   # Minimum improvement to reset patience
        verbose: bool = True
    ):
        super().__init__()
        self.patience = patience
        self.lr_factor = lr_factor
        self.min_lr = min_lr
        self.max_adjustments = max_adjustments
        self.min_delta = min_delta
        self.verbose = verbose

        self.best_val_acc = 0.0
        self.wait = 0
        self.adjustments_made = 0
        self._console = None

    @property
    def console(self):
        if self._console is None:
            from rich.console import Console
            self._console = Console()
        return self._console

    def _get_current_lr(self) -> float:
        """Get current learning rate from optimizer."""
        try:
            return float(self.model.optimizer.learning_rate)
        except Exception:
            return float(tf.keras.backend.get_value(self.model.optimizer.learning_rate))

    def _try_reduce_lr(self) -> None:
        """Attempt to reduce learning rate when stuck on a plateau."""
        current_lr = self._get_current_lr()

        if current_lr <= self.min_lr:
            if self.verbose and self.wait == self.patience:
                self.console.print(
                    f"  [dim]⚙ LR already at minimum ({self.min_lr:.2e}), cannot adjust further[/dim]"
                )
            return

        new_lr = max(current_lr * self.lr_factor, self.min_lr)
        self.model.optimizer.learning_rate.assign(new_lr)
        self.adjustments_made += 1
        self.wait = 0  # Reset patience

        if self.verbose:
            self.console.print(
                f"  [yellow]⚙ Auto-adjust #{self.adjustments_made}: "
                f"LR {current_lr:.2e} → {new_lr:.2e} "
                f"(stuck for {self.patience} epochs at {self.best_val_acc:.1%})[/yellow]"
            )

    def on_epoch_end(self, epoch, logs=None):
        if logs is None:
            return

        val_acc = logs.get('val_accuracy', logs.get('accuracy', 0))

        # Check if improved
        if val_acc > self.best_val_acc + self.min_delta:
            self.best_val_acc = val_acc
            self.wait = 0
        else:
            self.wait += 1

        # Check if stuck
        if self.wait >= self.patience and self.adjustments_made < self.max_adjustments:
            self._try_reduce_lr()

    def on_train_end(self, logs=None):
        if self.adjustments_made > 0 and self.verbose:
            self.console.print(
                f"  [dim]⚙ Made {self.adjustments_made} auto-adjustments during training[/dim]"
            )


# =============================================================================
# REPLAY BUFFER - Memory for Catastrophic Forgetting Prevention
# =============================================================================


class ReplayBuffer:
    """
    Memory replay buffer for continual learning.

    Stores representative samples from each training session and mixes them
    into future training to prevent forgetting old patterns.

    Uses reservoir sampling for memory-efficient storage of large datasets.

    Benefits:
    - Retains examples of past market regimes (flash crashes, trends, etc.)
    - Simple and data-centric approach
    - Complements EWC (data + weight protection)
    """

    def __init__(
        self,
        capacity_ratio: float = 0.10,  # Store 10% of training data
        mix_ratio: float = 0.20,  # Mix 20% replay samples
        buffer_dir: str = "trained_data/replay",
    ):
        self.capacity_ratio = capacity_ratio
        self.mix_ratio = mix_ratio
        self.buffer_dir = Path(buffer_dir)

        self.x_buffer: Optional[np.ndarray] = None
        self.y_buffer: Optional[np.ndarray] = None
        self.w_buffer: Optional[np.ndarray] = None  # Sample weights
        self.feature_names: Optional[List[str]] = None  # Track feature names for cross-session alignment
        self.metadata: Dict[str, Any] = {}

        self._sample_count = 0

    def _initialize_buffer(
        self, X: np.ndarray, y: np.ndarray, w: Optional[np.ndarray], capacity: int
    ):
        """Initialize the buffer with a random sample from the first batch."""
        rng = np.random.default_rng(42)
        indices = rng.choice(len(X), min(capacity, len(X)), replace=False)
        self.x_buffer = X[indices].copy()
        self.y_buffer = y[indices].copy()
        self.w_buffer = w[indices].copy() if w is not None else np.ones(len(indices))
        self._sample_count = len(X)
        logger.info(
            f"📦 Replay buffer initialized: {len(self.x_buffer)} samples from {len(X)} total"
        )

    def _reservoir_sample_item(
        self,
        idx: int,
        X: np.ndarray,
        y: np.ndarray,
        w: Optional[np.ndarray],
        capacity: int,
        rng: np.random.Generator,
    ):
        """Add single item using reservoir sampling."""
        self._sample_count += 1

        if len(self.x_buffer) < capacity:
            # Buffer not full, just append
            self.x_buffer = np.vstack([self.x_buffer, X[idx : idx + 1]])
            self.y_buffer = np.concatenate([self.y_buffer, y[idx : idx + 1]])
            w_i = w[idx : idx + 1] if w is not None else np.array([1.0])
            self.w_buffer = np.concatenate([self.w_buffer, w_i])
        else:
            # Reservoir sampling
            j = rng.integers(0, self._sample_count)
            if j < len(self.x_buffer):
                self.x_buffer[j] = X[idx]
                self.y_buffer[j] = y[idx]
                self.w_buffer[j] = w[idx] if w is not None else 1.0

    def _update_buffer_with_reservoir(
        self, X: np.ndarray, y: np.ndarray, w: Optional[np.ndarray], capacity: int
    ):
        """Update buffer using reservoir sampling for new batch."""
        rng = np.random.default_rng(42)
        for i in range(len(X)):
            self._reservoir_sample_item(i, X, y, w, capacity, rng)
        logger.info(
            f"📦 Replay buffer updated: {len(self.x_buffer)} samples, seen {self._sample_count} total"
        )

    def _track_data_source(self, data_id: str, n_samples: int):
        """Track data source in metadata."""
        if "data_sources" not in self.metadata:
            self.metadata["data_sources"] = []
        self.metadata["data_sources"].append({
            "id": data_id,
            "n_samples": n_samples,
            "timestamp": datetime.now().isoformat(),
        })

    def add_samples(
        self,
        X: np.ndarray,
        y: np.ndarray,
        w: Optional[np.ndarray] = None,
        data_id: Optional[str] = None,
        feature_names: Optional[List[str]] = None,
    ):
        """
        Add samples to replay buffer using reservoir sampling.

        Args:
            X: Features (sequences)
            y: Labels
            w: Optional sample weights
            data_id: Identifier for this data batch (e.g., instrument_date)
            feature_names: Feature names corresponding to X's last axis
        """
        # Track feature names for future alignment
        if feature_names is not None:
            self.feature_names = list(feature_names)
        capacity = int(len(X) * self.capacity_ratio)
        if capacity < 10:
            capacity = min(len(X), 100)  # Minimum buffer size

        if self.x_buffer is None:
            self._initialize_buffer(X, y, w, capacity)
        else:
            self._update_buffer_with_reservoir(X, y, w, capacity)

        if data_id:
            self._track_data_source(data_id, len(X))

    def get_replay_samples(
        self,
        n_new_samples: int,
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Get replay samples to mix with new training data.

        Args:
            n_new_samples: Number of new training samples

        Returns:
            (X_replay, y_replay, w_replay) or (None, None, None) if buffer empty
        """
        if self.x_buffer is None or len(self.x_buffer) == 0:
            return None, None, None

        # Calculate how many replay samples to return
        n_replay = int(n_new_samples * self.mix_ratio)
        n_replay = min(n_replay, len(self.x_buffer))

        if n_replay == 0:
            return None, None, None

        # Random sample from buffer
        replay_rng = np.random.default_rng(42)
        indices = replay_rng.choice(len(self.x_buffer), n_replay, replace=False)

        logger.info(
            "📦 Providing %d replay samples (%.0f%% of %d)",
            n_replay, self.mix_ratio * 100, n_new_samples
        )

        return (
            self.x_buffer[indices].copy(),
            self.y_buffer[indices].copy(),
            self.w_buffer[indices].copy() if self.w_buffer is not None else None,
        )

    def clear(self):
        """Clear the replay buffer (e.g., when feature dimensions change)."""
        self.x_buffer = None
        self.y_buffer = None
        self.w_buffer = None
        self._sample_count = 0
        self.metadata = {}
        logger.info("📦 Replay buffer cleared")

    def save(self, instrument: str):
        """Save replay buffer to disk."""
        save_dir = self.buffer_dir / instrument
        save_dir.mkdir(parents=True, exist_ok=True)

        if self.x_buffer is None:
            logger.warning("No replay buffer to save")
            return

        # Save arrays
        np.savez_compressed(
            save_dir / "buffer.npz",
            X=self.x_buffer,
            y=self.y_buffer,
            w=self.w_buffer,
        )

        # Save metadata
        meta = {
            "capacity_ratio": self.capacity_ratio,
            "mix_ratio": self.mix_ratio,
            "sample_count": self._sample_count,
            "buffer_size": len(self.x_buffer),
            "saved_at": datetime.now().isoformat(),
            **self.metadata,
        }
        with open(save_dir / "buffer_meta.json", "w") as f:
            json.dump(meta, f, indent=2)

        logger.info(
            f"📦 Replay buffer saved to {save_dir} ({len(self.x_buffer)} samples)"
        )

    def load(self, instrument: str) -> bool:
        """Load replay buffer from disk."""
        load_dir = self.buffer_dir / instrument
        buffer_path = load_dir / "buffer.npz"

        if not buffer_path.exists():
            logger.info(f"No replay buffer at {buffer_path}")
            return False

        data = np.load(buffer_path)
        self.x_buffer = data["X"]
        self.y_buffer = data["y"]
        self.w_buffer = data.get("w")

        # Load metadata
        meta_path = load_dir / "buffer_meta.json"
        if meta_path.exists():
            with open(meta_path) as f:
                meta = json.load(f)
            self._sample_count = meta.get("sample_count", len(self.x_buffer))
            self.metadata = meta

        logger.info(
            f"📦 Replay buffer loaded: {len(self.x_buffer)} samples from {instrument}"
        )
        return True


# =============================================================================
# DRIFT DETECTION SYSTEM
# =============================================================================


class DriftDetector:
    """
    Advanced drift detection for continual learning.

    Detects three types of drift:
    1. Performance drift: Model accuracy degradation
    2. Data drift: Distribution shift in input features
    3. Concept drift: Relationship between features and labels changes

    When drift is detected, triggers retraining or alerts.
    """

    def __init__(
        self,
        performance_threshold: float = 0.03,  # 3% accuracy drop
        feature_drift_threshold: float = 0.1,  # 10% feature distribution shift
        window_size: int = 5,  # Number of recent sessions to consider
    ):
        self.performance_threshold = performance_threshold
        self.feature_drift_threshold = feature_drift_threshold
        self.window_size = window_size
        self.baseline_stats: Optional[Dict[str, Any]] = None

    def compute_feature_stats(self, X: np.ndarray) -> Dict[str, np.ndarray]:
        """Compute feature statistics for drift detection."""
        return {
            "mean": np.mean(X, axis=(0, 1)) if X.ndim == 3 else np.mean(X, axis=0),
            "std": np.std(X, axis=(0, 1)) if X.ndim == 3 else np.std(X, axis=0),
            "min": np.min(X, axis=(0, 1)) if X.ndim == 3 else np.min(X, axis=0),
            "max": np.max(X, axis=(0, 1)) if X.ndim == 3 else np.max(X, axis=0),
        }

    def set_baseline(self, X: np.ndarray, metrics: Dict[str, float]):
        """Set baseline statistics from initial training."""
        self.baseline_stats = {
            "feature_stats": self.compute_feature_stats(X),
            "best_val_accuracy": metrics.get("val_accuracy", 0),
            "baseline_metrics": metrics.copy(),
            "timestamp": datetime.now().isoformat(),
        }
        logger.info(
            f"📊 Drift baseline set: val_acc={metrics.get('val_accuracy', 0):.4f}"
        )

    def check_performance_drift(
        self,
        current_val_acc: float,
        metric_history: List[Dict[str, float]],
    ) -> Tuple[bool, str]:
        """
        Check for performance drift.

        Returns:
            (is_drifted, reason)
        """
        if not metric_history:
            return False, "No history"

        # Get best historical accuracy
        best_acc = max(entry.get("val_accuracy", 0) for entry in metric_history)

        # Check absolute drift from best
        drop = best_acc - current_val_acc
        if drop > self.performance_threshold:
            return True, f"Accuracy dropped {drop:.4f} from best {best_acc:.4f}"

        # Check trend: declining over recent window
        recent = (
            metric_history[-self.window_size :]
            if len(metric_history) >= self.window_size
            else metric_history
        )
        if len(recent) >= 3:
            recent_accs = [e.get("val_accuracy", 0) for e in recent]
            trend = recent_accs[-1] - recent_accs[0]
            if trend < -self.performance_threshold:
                return True, f"Declining trend: {trend:.4f} over {len(recent)} sessions"

        return False, "No drift"

    def check_feature_drift(self, x_new: np.ndarray) -> Tuple[bool, str]:
        """
        Check for feature distribution drift.

        Uses simple mean/std comparison. Could be enhanced with KS-test.
        """
        if self.baseline_stats is None:
            return False, "No baseline"

        new_stats = self.compute_feature_stats(x_new)
        baseline = self.baseline_stats["feature_stats"]

        # Compare means (normalized by baseline std)
        mean_shift = np.abs(new_stats["mean"] - baseline["mean"])
        normalized_shift = mean_shift / (baseline["std"] + 1e-8)
        max_shift = float(np.max(normalized_shift))

        if max_shift > self.feature_drift_threshold * 10:  # 10 sigma shift
            return True, f"Feature distribution shifted: max={max_shift:.2f} sigma"

        # Compare std (relative change)
        std_change = np.abs(new_stats["std"] - baseline["std"]) / (
            baseline["std"] + 1e-8
        )
        max_std_change = float(np.max(std_change))

        if max_std_change > self.feature_drift_threshold * 5:  # 50% std change
            return True, f"Feature variance changed: max={max_std_change:.2%}"

        return False, "No feature drift"

    def full_drift_check(
        self,
        x_new: np.ndarray,
        current_val_acc: float,
        metric_history: List[Dict[str, float]],
    ) -> Dict[str, Any]:
        """
        Perform full drift analysis.

        Returns:
            Dict with drift status and recommendations
        """
        perf_drifted, perf_reason = self.check_performance_drift(
            current_val_acc, metric_history
        )
        feat_drifted, feat_reason = self.check_feature_drift(x_new)

        any_drift = perf_drifted or feat_drifted

        result = {
            "any_drift": any_drift,
            "performance_drift": perf_drifted,
            "performance_reason": perf_reason,
            "feature_drift": feat_drifted,
            "feature_reason": feat_reason,
            "recommendation": "normal",
        }

        if perf_drifted and feat_drifted:
            result["recommendation"] = "full_retrain"
            logger.warning(
                "⚠️ DRIFT: Performance AND feature drift detected → Full retraining recommended"
            )
        elif perf_drifted:
            result["recommendation"] = "warm_start_retrain"
            logger.warning(
                "⚠️ DRIFT: Performance drift detected → Warm-start retraining recommended"
            )
        elif feat_drifted:
            result["recommendation"] = "monitor"
            logger.info("📊 Feature drift detected but performance stable → Monitoring")

        return result

    def save(self, path: str):
        """Save drift detector state."""
        if self.baseline_stats is None:
            return

        path = Path(path)
        data = {
            "performance_threshold": self.performance_threshold,
            "feature_drift_threshold": self.feature_drift_threshold,
            "window_size": self.window_size,
            "baseline_stats": {
                k: v.tolist() if isinstance(v, np.ndarray) else v
                for k, v in self.baseline_stats.items()
                if k != "feature_stats"
            },
            "feature_stats": {
                k: v.tolist() for k, v in self.baseline_stats["feature_stats"].items()
            },
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        logger.info(f"📊 Drift baseline saved to {path}")

    def load(self, path: str) -> bool:
        """Load drift detector state."""
        path = Path(path)
        if not path.exists():
            return False

        with open(path) as f:
            data = json.load(f)

        self.performance_threshold = data.get(
            "performance_threshold", self.performance_threshold
        )
        self.feature_drift_threshold = data.get(
            "feature_drift_threshold", self.feature_drift_threshold
        )
        self.window_size = data.get("window_size", self.window_size)

        self.baseline_stats = dict(data.get("baseline_stats", {}).items())
        self.baseline_stats["feature_stats"] = {
            k: np.array(v) for k, v in data.get("feature_stats", {}).items()
        }

        logger.info(f"📊 Drift baseline loaded from {path}")
        return True

    def record_training_result(
        self,
        val_accuracy: float,
        instrument: str,
        data_hash: str,
        feature_means: np.ndarray,
    ):
        """
        Record a training result for drift tracking.

        This maintains a running history of training sessions
        and can detect drift patterns over time.
        """
        if not hasattr(self, "_history"):
            self._history = []

        self._history.append(
            {
                "val_accuracy": val_accuracy,
                "instrument": instrument,
                "data_hash": data_hash,
                "feature_means": feature_means.copy()
                if isinstance(feature_means, np.ndarray)
                else np.array(feature_means),
                "timestamp": datetime.now().isoformat(),
            }
        )

        # Keep only recent history
        if len(self._history) > self.window_size * 2:
            self._history = self._history[-self.window_size * 2 :]

        # Set baseline if this is first record
        if self.baseline_stats is None and len(self._history) == 1:
            self.baseline_stats = {
                "best_val_accuracy": val_accuracy,
                "feature_stats": {"mean": feature_means.copy()},
                "timestamp": datetime.now().isoformat(),
            }

    def _check_performance_drop(self, current_acc: float) -> Tuple[bool, str]:
        """Check if performance dropped below threshold."""
        best_acc = max(h["val_accuracy"] for h in self._history)
        drop = best_acc - current_acc
        if drop > self.performance_threshold:
            return True, f"Performance dropped {drop:.2%} from best {best_acc:.2%}"
        return False, ""

    def _check_feature_drift(self, current_means: np.ndarray) -> Tuple[bool, str]:
        """Check if feature means shifted beyond threshold."""
        if not self.baseline_stats or "feature_stats" not in self.baseline_stats:
            return False, ""

        baseline_means = self.baseline_stats["feature_stats"].get("mean")
        if baseline_means is None or len(baseline_means) != len(current_means):
            return False, ""

        mean_shift = np.abs(current_means - baseline_means).mean()
        if mean_shift > self.feature_drift_threshold:
            return True, f"Feature means shifted by {mean_shift:.2%}"
        return False, ""

    def _check_declining_trend(self) -> Tuple[bool, str]:
        """Check for declining accuracy trend over recent sessions."""
        if len(self._history) < 3:
            return False, ""

        recent = self._history[-3:]
        accs = [h["val_accuracy"] for h in recent]

        # Check if all sequential pairs show decline
        is_declining = all(accs[i] > accs[i + 1] for i in range(len(accs) - 1))
        if not is_declining:
            return False, ""

        decline = accs[0] - accs[-1]
        if decline > self.performance_threshold:
            return True, f"Declining trend: {decline:.2%} over last 3 sessions"
        return False, ""

    def check_drift(self) -> Tuple[bool, str]:
        """
        Check for drift based on recorded history.

        Returns:
            (drift_detected, reason)
        """
        if not hasattr(self, "_history") or len(self._history) < 2:
            return False, "Insufficient history"

        current = self._history[-1]

        # Check performance drop
        drift, reason = self._check_performance_drop(current["val_accuracy"])
        if drift:
            return True, reason

        # Check feature drift
        drift, reason = self._check_feature_drift(current["feature_means"])
        if drift:
            return True, reason

        # Check declining trend
        drift, reason = self._check_declining_trend()
        if drift:
            return True, reason

        return False, "No drift detected"


# =============================================================================
# TRAINING LINEAGE TRACKER
# =============================================================================


@dataclass
class TrainingLineage:
    """
    Track model training history across warm-start sessions.

    Enables:
    - Rollback to any ancestor checkpoint
    - Analysis of model evolution over time
    - Drift detection based on metric history
    - Scheduling decisions for retraining
    """

    checkpoint_id: str = ""  # Unique ID for this checkpoint
    parent_checkpoint_id: Optional[str] = None  # ID of warm-start source
    created_at: str = ""  # ISO timestamp

    # Cumulative counters
    cumulative_epochs: int = 0  # Total epochs across all sessions
    cumulative_samples: int = 0  # Total samples seen
    session_epochs: int = 0  # Epochs in this session
    generation: int = 1  # How many warm-starts from initial training

    # Data fingerprint
    data_hash: str = ""  # Hash of training data for drift detection
    data_range: str = ""  # e.g., "2024-01-01 to 2024-06-01"
    instrument: str = ""  # e.g., "EUR_USD"
    granularity: str = ""  # e.g., "H1"

    # Performance history
    metric_history: List[Dict[str, float]] = field(default_factory=list)

    # EMA/EWC state tracking
    ema_enabled: bool = False
    ewc_n_tasks: int = 0
    replay_buffer_size: int = 0

    # Training configuration snapshot
    training_config: Dict[str, Any] = field(default_factory=dict)

    # Drift detection state
    drift_detected: bool = False
    last_drift_check: str = ""
    drift_reason: str = ""

    # Scheduling metadata
    last_training_duration_seconds: float = 0.0
    recommended_retrain_interval_hours: int = 168  # Default: weekly

    # === Schema Version (v2) ===
    lineage_version: int = 2  # Bump when adding fields

    # === Dynamic Training State (v2) ===
    auto_variance_weight: float = (
        0.1  # Auto-tuned variance penalty for AntiCollapseFocalLoss
    )
    lr_reductions_count: int = 0  # Number of LR reductions during training
    final_learning_rate: float = 0.0  # Learning rate at end of training
    collapse_recovery_count: int = 0  # Times recovered from prediction collapse

    def generate_checkpoint_id(self) -> str:
        """Generate unique checkpoint ID based on timestamp + random."""
        import secrets

        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        rand = secrets.token_hex(4)
        self.checkpoint_id = f"{ts}_{rand}"
        self.created_at = datetime.now().isoformat()
        return self.checkpoint_id

    def add_metrics(self, metrics: Dict[str, float]):
        """Add metrics from a training session."""
        entry = {
            "timestamp": datetime.now().isoformat(),
            "checkpoint_id": self.checkpoint_id,
            "generation": self.generation,
            **{k: float(v) for k, v in metrics.items() if isinstance(v, (int, float))},
        }
        self.metric_history.append(entry)

    def check_drift(self, current_val_acc: float, threshold: float = 0.03) -> bool:
        """
        Check if model performance has drifted beyond threshold.

        Returns True if val_accuracy dropped by more than threshold from best.
        """
        if not self.metric_history:
            return False

        best_acc = max(entry.get("val_accuracy", 0) for entry in self.metric_history)

        if best_acc - current_val_acc > threshold:
            logger.info(
                f"📊 Model drift check: val_acc={current_val_acc:.4f} vs best={best_acc:.4f} "
                f"(drop={best_acc - current_val_acc:.4f} > threshold={threshold})"
            )
            return True
        return False

    @staticmethod
    def compute_data_hash(X: np.ndarray, y: np.ndarray) -> str:
        """Compute hash of training data for change detection."""
        # Use subset for efficiency
        n = min(1000, len(X))
        indices = np.linspace(0, len(X) - 1, n, dtype=int)
        x_sample = X[indices]
        y_sample = y[indices]

        data_bytes = x_sample.tobytes() + y_sample.tobytes()
        return hashlib.md5(data_bytes, usedforsecurity=False).hexdigest()[:12]

    def get_training_summary(self) -> str:
        """Get human-readable summary of training lineage."""
        lines = [
            "📊 Training Lineage Summary",
            f"  Checkpoint: {self.checkpoint_id}",
            f"  Created: {self.created_at}",
            f"  Generation: {self.generation}",
            f"  Cumulative epochs: {self.cumulative_epochs}",
            f"  Cumulative samples: {self.cumulative_samples:,}",
            f"  Instrument: {self.instrument} ({self.granularity})",
        ]
        if self.metric_history:
            latest = self.metric_history[-1]
            lines.append(
                f"  Latest val_accuracy: {latest.get('val_accuracy', 'N/A'):.4f}"
            )
        if self.drift_detected:
            lines.append(f"  ⚠️ Drift detected: {self.drift_reason}")
        if self.ema_enabled:
            lines.append("  EMA: enabled")
        if self.ewc_n_tasks > 0:
            lines.append(f"  EWC: {self.ewc_n_tasks} task(s)")
        if self.replay_buffer_size > 0:
            lines.append(f"  Replay buffer: {self.replay_buffer_size:,} samples")
        return "\n".join(lines)

    def should_retrain(self, hours_since_last: float) -> Tuple[bool, str]:
        """
        Determine if retraining is recommended.

        Returns:
            (should_retrain, reason)
        """
        # Check time-based scheduling
        if hours_since_last >= self.recommended_retrain_interval_hours:
            return True, f"Scheduled: {hours_since_last:.1f}h since last training"

        # Check drift
        if self.drift_detected:
            return True, f"Drift detected: {self.drift_reason}"

        # Check if accuracy is declining
        if len(self.metric_history) >= 3:
            recent_accs = [e.get("val_accuracy", 0) for e in self.metric_history[-3:]]
            if all(
                recent_accs[i] > recent_accs[i + 1] for i in range(len(recent_accs) - 1)
            ):
                return True, f"Declining accuracy trend: {recent_accs}"

        return False, "No retraining needed"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "checkpoint_id": self.checkpoint_id,
            "parent_checkpoint_id": self.parent_checkpoint_id,
            "created_at": self.created_at,
            "cumulative_epochs": self.cumulative_epochs,
            "cumulative_samples": self.cumulative_samples,
            "session_epochs": self.session_epochs,
            "generation": self.generation,
            "data_hash": self.data_hash,
            "data_range": self.data_range,
            "instrument": self.instrument,
            "granularity": self.granularity,
            "metric_history": self.metric_history,
            "ema_enabled": self.ema_enabled,
            "ewc_n_tasks": self.ewc_n_tasks,
            "replay_buffer_size": self.replay_buffer_size,
            "training_config": self.training_config,
            "drift_detected": self.drift_detected,
            "last_drift_check": self.last_drift_check,
            "drift_reason": self.drift_reason,
            "last_training_duration_seconds": self.last_training_duration_seconds,
            "recommended_retrain_interval_hours": self.recommended_retrain_interval_hours,
            "lineage_version": self.lineage_version,
            "auto_variance_weight": self.auto_variance_weight,
            "lr_reductions_count": self.lr_reductions_count,
            "final_learning_rate": self.final_learning_rate,
            "collapse_recovery_count": self.collapse_recovery_count,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TrainingLineage":
        """Create from dictionary."""
        return cls(
            checkpoint_id=data.get("checkpoint_id", ""),
            parent_checkpoint_id=data.get("parent_checkpoint_id"),
            created_at=data.get("created_at", ""),
            cumulative_epochs=data.get("cumulative_epochs", 0),
            cumulative_samples=data.get("cumulative_samples", 0),
            session_epochs=data.get("session_epochs", 0),
            generation=data.get("generation", 1),
            data_hash=data.get("data_hash", ""),
            data_range=data.get("data_range", ""),
            instrument=data.get("instrument", ""),
            granularity=data.get("granularity", ""),
            metric_history=data.get("metric_history", []),
            ema_enabled=data.get("ema_enabled", False),
            ewc_n_tasks=data.get("ewc_n_tasks", 0),
            replay_buffer_size=data.get("replay_buffer_size", 0),
            training_config=data.get("training_config", {}),
            drift_detected=data.get("drift_detected", False),
            last_drift_check=data.get("last_drift_check", ""),
            drift_reason=data.get("drift_reason", ""),
            last_training_duration_seconds=data.get(
                "last_training_duration_seconds", 0.0
            ),
            recommended_retrain_interval_hours=data.get(
                "recommended_retrain_interval_hours", 168
            ),
            lineage_version=data.get("lineage_version", 1),
            auto_variance_weight=data.get("auto_variance_weight", 0.1),
            lr_reductions_count=data.get("lr_reductions_count", 0),
            final_learning_rate=data.get("final_learning_rate", 0.0),
            collapse_recovery_count=data.get("collapse_recovery_count", 0),
        )


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
        x_val: np.ndarray,
        y_val: np.ndarray,
    ) -> Dict[str, float]:
        """Train the model and return metrics."""

    @abstractmethod
    def predict(self, X: np.ndarray) -> Dict[str, Any]:
        """Make predictions."""

    @abstractmethod
    def save(self, path: str) -> None:
        """Save model to disk."""

    @abstractmethod
    def load(self, path: str) -> None:
        """Load model from disk."""


# =============================================================================
# TCN TRAINER - Volatility Regime Classification (Entry Timing Filter)
# =============================================================================


class TCNTrainer(BaseTrainer):
    """
    TCN model for volatility regime classification.

    Purpose: Entry timing filter - only trade when volatility is HIGH or EXTREME.
    Addresses timing issues where trades are closed early due to unfavorable conditions.

    Input: OHLCV features with ATR-based volatility indicators
    Output: 4-class regime (0=LOW, 1=NORMAL, 2=HIGH, 3=EXTREME)

    The Transformer handles direction prediction alone - TCN is purely a timing gate.
    """

    # Class constants for regime names
    REGIME_NAMES = {0: "LOW", 1: "NORMAL", 2: "HIGH", 3: "EXTREME"}

    def __init__(self, config: Optional[TrainerConfig] = None):
        super().__init__(config)
        self.scaler = None
        self.feature_names = None  # Save feature names for inference
        self.n_classes = 4  # LOW, NORMAL, HIGH, EXTREME

    def _build_model(self, input_shape: Tuple[int, int]) -> Any:
        """
        Build TCN model for 4-class volatility regime classification.

        Architecture optimized for volatility pattern recognition:
        - Dilated causal convolutions capture multi-scale volatility patterns
        - 4-class softmax output for regime classification
        - Moderate regularization (volatility has cleaner patterns than direction)
        """
        from tensorflow import keras
        regularizers = keras.regularizers

        seq_len, n_features = input_shape

        # Get config values with sensible defaults
        filters = getattr(
            self.config, "tcn_hidden_size", 32
        )  # Increased for regime patterns
        kernel_size = getattr(self.config, "tcn_kernel_size", 3)
        num_layers = getattr(
            self.config, "tcn_num_layers", 2
        )  # More layers for volatility
        dropout = getattr(
            self.config, "tcn_dropout", 0.4
        )  # Reduced - regimes are more learnable
        l2_reg = getattr(self.config, "tcn_l2_reg", 0.005)  # Lighter regularization
        spatial_dropout = getattr(self.config, "tcn_spatial_dropout", 0.2)
        noise_std = getattr(self.config, "tcn_noise_std", 0.03)

        # L2 regularizer for kernels
        kernel_reg = regularizers.l2(l2_reg)

        inp = keras.Input(shape=(seq_len, n_features), name="features")

        # === INPUT REGULARIZATION ===
        x = keras.layers.GaussianNoise(noise_std)(inp)
        x = keras.layers.SpatialDropout1D(spatial_dropout)(x)

        # === TCN LAYERS with dilated causal convolutions ===
        for i in range(num_layers):
            dilation_rate = 2**i
            x = keras.layers.Conv1D(
                filters=filters,
                kernel_size=kernel_size,
                padding="causal",
                dilation_rate=dilation_rate,
                activation="relu",
                kernel_regularizer=kernel_reg,
                name=f"tcn_conv_{i}",
            )(x)
            x = keras.layers.BatchNormalization()(x)
            x = keras.layers.Dropout(dropout)(x)

        # === OUTPUT HEAD for 4-class regime classification ===
        x = keras.layers.GlobalAveragePooling1D()(x)

        # Dense layer with moderate capacity
        x = keras.layers.Dense(32, activation="relu", kernel_regularizer=kernel_reg)(x)
        x = keras.layers.Dropout(dropout)(x)

        # 4-class softmax output for volatility regime
        volatility_regime = keras.layers.Dense(
            self.n_classes,  # 4 classes: LOW, NORMAL, HIGH, EXTREME
            activation="softmax",
            name="volatility_regime",
            dtype="float32",
        )(x)

        model = keras.Model(
            inputs=inp, outputs=volatility_regime, name="tcn_volatility_regime"
        )

        # Compile with sparse categorical crossentropy (integer labels)
        lr = self.config.learning_rate

        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=lr),
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )

        return model

    def _compute_class_weights(self, y: np.ndarray) -> Dict[int, float]:
        """Compute inverse-frequency class weights for ALL 4 volatility regimes.

        Always returns weights for classes 0-3 (LOW, NORMAL, HIGH, EXTREME),
        even if some classes are not present in the training data.
        Missing classes get weight 1.0 (neutral).
        """
        unique, counts = np.unique(y, return_counts=True)
        n_samples = len(y)
        n_expected_classes = 4  # Always 4 classes: LOW, NORMAL, HIGH, EXTREME

        # Start with neutral weights for ALL 4 classes
        weights = dict.fromkeys(range(n_expected_classes), 1.0)

        # Update weights for classes that exist in training data
        for cls, count in zip(unique, counts):
            cls_int = int(cls)
            if 0 <= cls_int < n_expected_classes:
                # Inverse frequency with cap
                weight = n_samples / (n_expected_classes * count)
                weights[cls_int] = min(weight, 5.0)  # Cap at 5x

        logger.info(f"Volatility regime class weights: {weights}")
        return weights

    def _validate_labels(self, y_train: np.ndarray) -> None:
        """Validate that training labels are valid regime classes."""
        unique_labels = np.unique(y_train)
        valid_classes = set(range(self.n_classes))
        invalid_labels = [
            label
            for label in unique_labels
            if label not in valid_classes and int(label) != label
        ]
        if invalid_labels:
            label_dist = dict(zip(*np.unique(y_train, return_counts=True)))
            logger.error(f"Invalid labels found: {invalid_labels}. Expected integers 0-{self.n_classes - 1}.")
            logger.error(f"Label distribution: {label_dist}")
            logger.error(f"Label dtype: {y_train.dtype}, min={np.min(y_train):.4f}, max={np.max(y_train):.4f}")
            logger.error("HINT: If you see 0.5 labels, direction data (0.0/0.5/1.0) is being passed to TCN.")
            raise ValueError(
                f"TCN labels must be integers 0-{self.n_classes - 1}, but found: {invalid_labels}.\n"
                f"Label distribution: {label_dist}\n"
                f"Fix: Ensure load_volatility_regime_data() is used when training TCN volatility filter."
            )

    def _log_class_distribution(self, y_train: np.ndarray) -> None:
        """Log the class distribution for training data."""
        unique, counts = np.unique(y_train, return_counts=True)
        for cls, count in zip(unique, counts):
            pct = count / len(y_train) * 100
            logger.info(f"  Train {self.REGIME_NAMES.get(int(cls), cls)}: {count} ({pct:.1f}%)")

    def _try_load_h5_weights(self, warm_start_path: str) -> bool:
        """Try loading weights from .weights.h5 companion file."""
        weights_h5_path = Path(warm_start_path).with_suffix(WEIGHTS_H5_SUFFIX)
        if not weights_h5_path.exists():
            return False
        if _safe_load_weights_ignoring_optimizer(self.model, str(weights_h5_path)):
            logger.info(f"✓ Loaded weights from {weights_h5_path}")
            _safe_reset_optimizer_state(self.model)
            return True
        return False

    def _try_load_direct_weights(self, warm_start_path: str) -> bool:
        """Try loading weights from .keras file directly."""
        if _safe_load_weights_ignoring_optimizer(self.model, str(warm_start_path)):
            logger.info(f"✓ Loaded weights from {warm_start_path}")
            _safe_reset_optimizer_state(self.model)
            return True
        return False

    def _try_load_full_model_weights(self, warm_start_path: str, keras_module: Any) -> bool:
        """Try full model load and extract weights (bypasses optimizer state)."""
        try:
            existing_model = keras_module.models.load_model(warm_start_path, compile=False)
            checkpoint_weights = existing_model.get_weights()
            model_weights = self.model.get_weights()

            is_compatible, shape_error = _validate_weight_shapes(
                model_weights, checkpoint_weights, context="model weights"
            )

            if is_compatible:
                self.model.set_weights(checkpoint_weights)
                logger.info(WEIGHTS_LOADED_FULL_MODEL_MSG)
                del existing_model
                return True

            # Partial transfer by name matching
            logger.warning(f"Architecture mismatch: {shape_error}")
            logger.info("🔄 Attempting partial weight transfer by layer name...")
            ckpt_weight_map = {w.name: w.numpy() for w in existing_model.weights}
            loaded, skipped = 0, 0
            for w in self.model.weights:
                if w.name in ckpt_weight_map and w.shape == ckpt_weight_map[w.name].shape:
                    w.assign(ckpt_weight_map[w.name].astype(_get_numpy_dtype(w.dtype)))
                    loaded += 1
                else:
                    skipped += 1
            del existing_model

            if loaded > 0:
                logger.info(
                    f"✓ Partial weight transfer: {loaded} loaded, {skipped} re-initialized"
                )
                return True
            logger.warning("⚠️ No compatible weights found in full model")
            return False
        except Exception as e:
            logger.debug(f"Full model load failed: {e}")
            return False

    def _load_warm_start_metadata(self, warm_start_path: str) -> float:
        """Load previous best accuracy from metadata file."""
        meta_path = Path(warm_start_path).with_suffix(META_PKL_SUFFIX)
        if not meta_path.exists():
            return 0.0
        try:
            with open(meta_path, "rb") as f:
                meta = pickle.load(f)
            prev_val_acc = meta.get("metrics", {}).get("val_accuracy", 0.0)
            if prev_val_acc > 0:
                logger.info(f"🎯 Warm-start baseline: {prev_val_acc:.1%}")
            return prev_val_acc
        except Exception as e:
            logger.warning(f"Could not load warm-start metadata: {e}")
            return 0.0

    def _load_warm_start_weights(
        self, warm_start_path: str, keras_module: Any
    ) -> Tuple[bool, float]:
        """Load weights from warm-start checkpoint.

        Returns:
            Tuple of (weights_loaded, previous_val_acc)
        """
        # Try loading strategies in order of preference
        weights_loaded = (
            self._try_load_h5_weights(warm_start_path)
            or self._try_load_direct_weights(warm_start_path)
            or self._try_load_full_model_weights(warm_start_path, keras_module)
        )

        # Load previous best accuracy from metadata if weights loaded
        prev_val_acc = self._load_warm_start_metadata(warm_start_path) if weights_loaded else 0.0

        return weights_loaded, prev_val_acc

    def _create_tcn_callbacks(
        self, keras_module: Any
    ) -> List[Any]:
        """Create callbacks for TCN training."""
        return [
            RichEpochCallback(
                model_name="TCN Volatility Regime",
                total_epochs=self.config.epochs,
                warm_start_best_acc=self._warm_start_val_acc,
                quiet=self.config.quiet,
            )
            if not self.config.quiet
            else QuietProgressCallback(
                model_name="TCN Volatility Regime",
                total_epochs=self.config.epochs,
            ),
            keras_module.callbacks.EarlyStopping(
                monitor="val_loss",
                patience=self.config.patience,
                mode="min",
                restore_best_weights=True,
                verbose=0,
            ),
            OverfitPreventionCallback(
                checkpoint_dir=self.config.checkpoint_dir,
                model_name="tcn_volatility_regime",
                config=OverfitPreventionConfig(
                    overfit_threshold=0.10,
                    critical_threshold=0.15,
                    severe_threshold=0.25,
                    max_acceptable_gap=0.12,
                    patience_epochs=2,
                    auto_adjust_dropout=True,
                    auto_reduce_lr=True,
                    enable_swa=True,
                    swa_start_fraction=0.5,
                    enable_cosine_restarts=True,
                    restart_period=15,
                ),
            ),
        ]

    def _load_model_native(self, path: Path, keras_module: Any) -> bool:
        """Try loading model in native .keras format."""
        try:
            # Import custom class so @register_keras_serializable runs before load
            try:
                from src.models.tensorflow_models import TCNVolatilityDualHead  # noqa: F401
            except ImportError:
                try:
                    from models.tensorflow_models import TCNVolatilityDualHead  # noqa: F401
                except ImportError:
                    pass

            self.model = keras_module.models.load_model(str(path), compile=False)
            logger.info(f"TCN Volatility Regime loaded from {path} (native format, compile=False)")
            return True
        except Exception as e:
            logger.warning(f"Could not load native .keras format: {e}")
            logger.info("Attempting cross-version load from weights + architecture...")
            return False

    def _load_model_from_arch_json(
        self, arch_path: Path, weights_path: Path, meta: Dict, keras_module: Any
    ) -> bool:
        """Try loading model from architecture JSON and weights."""
        if not (arch_path.exists() and weights_path.exists()):
            return False
        try:
            with open(arch_path) as f:
                arch_json = f.read()
            self.model = keras_module.models.model_from_json(arch_json)
            self.model.load_weights(str(weights_path))
            lr = meta.get("config", {}).get("learning_rate", 0.0003)
            self.model.compile(
                optimizer=keras_module.optimizers.Adam(learning_rate=lr),
                loss="sparse_categorical_crossentropy",
                metrics=["accuracy"],
            )
            logger.info(f"TCN Volatility Regime loaded from {weights_path} (cross-version: arch.json + weights)")
            return True
        except Exception as e:
            logger.warning(f"Could not load from arch.json + weights: {e}")
            return False

    def _load_model_rebuild_architecture(
        self, weights_path: Path, meta: Dict
    ) -> bool:
        """Try rebuilding architecture and loading weights."""
        if not weights_path.exists():
            return False
        try:
            arch_cfg = meta.get("architecture", {})
            input_shape = arch_cfg.get("input_shape", (self.seq_len, self.n_features))
            if self.config is None:
                self.config = TrainerConfig()
            for key, value in arch_cfg.items():
                if key != "input_shape":
                    setattr(self.config, key, value)
            self.model = self._build_model(input_shape)
            self.model.load_weights(str(weights_path))
            logger.info(f"TCN Volatility Regime loaded from {weights_path} (cross-version: rebuilt architecture)")
            return True
        except Exception as e:
            logger.warning(f"Could not rebuild and load weights: {e}")
            return False

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        x_val: np.ndarray,
        y_val: np.ndarray,
        feature_names: Optional[list] = None,
        warm_start_path: Optional[str] = None,
        instrument: str = "UNKNOWN",
    ) -> Dict[str, float]:
        """Train TCN for volatility regime classification.

        Args:
            warm_start_path: Path to existing model to load weights from (warm-start training)
            instrument: Trading instrument (e.g., "EUR_USD") for logging
        """
        from tensorflow import keras
        from sklearn.preprocessing import StandardScaler

        logger.info("Training TCN (Volatility Regime Filter)...")
        logger.info(f"  Classes: {self.REGIME_NAMES}")

        # Initialize warm-start tracking
        self._is_warm_start = False
        self._warm_start_val_acc = 0.0

        # Validate and log labels
        self._validate_labels(y_train)
        self._log_class_distribution(y_train)

        # Save feature names for inference
        self.feature_names = feature_names
        self.n_features = X_train.shape[-1]

        # Scale features
        self.scaler = StandardScaler()
        x_train_scaled = self.scaler.fit_transform(
            X_train.reshape(-1, X_train.shape[-1])
        )
        x_val_scaled = self.scaler.transform(x_val.reshape(-1, x_val.shape[-1]))

        # Get seq_len from config (validated)
        seq_len = get_config_seq_len(self.config)

        # Create sequences using shared helper
        x_train_seq, y_train_seq = create_sequences(x_train_scaled, y_train, seq_len)
        x_val_seq, y_val_seq = create_sequences(x_val_scaled, y_val, seq_len)

        # Ensure labels are integers for sparse categorical crossentropy
        y_train_seq = y_train_seq.astype(np.int32)
        y_val_seq = y_val_seq.astype(np.int32)

        # Build model
        self.model = self._build_model((seq_len, X_train.shape[-1]))

        # === WARM-START: Load existing weights if available ===
        effective_lr = self.config.learning_rate
        if warm_start_path and Path(warm_start_path).exists():
            try:
                logger.info(f"🔥 WARM-START: Loading weights from {warm_start_path}")
                weights_loaded, prev_val_acc = self._load_warm_start_weights(warm_start_path, keras)

                if weights_loaded:
                    self._is_warm_start = True
                    self._warm_start_val_acc = prev_val_acc

                    # Reduce learning rate for warm-start (use 10% of base LR)
                    warm_start_lr_factor = getattr(self.config, 'warm_start_lr_factor', 0.1)
                    effective_lr = self.config.learning_rate * warm_start_lr_factor
                    logger.info(
                        f"🔥 Warm-start LR reduction: {self.config.learning_rate} → "
                        f"{effective_lr} (factor={warm_start_lr_factor})"
                    )
                else:
                    logger.warning(f"⚠️ Could not load warm-start weights from {warm_start_path}")
            except Exception as e:
                logger.warning(f"⚠️ Warm-start failed: {e}. Training from scratch.")

        # === WARMUP LR SCHEDULE ===
        # Calculate steps for warmup schedule
        steps_per_epoch = len(x_train_seq) // self.config.batch_size
        warmup_epochs = getattr(self.config, "warmup_epochs", 5)  # 5 epochs warmup
        warmup_steps = warmup_epochs * steps_per_epoch
        total_steps = self.config.epochs * steps_per_epoch

        # Use WarmupCosineDecaySchedule for stable training
        try:
            from src.training.m1_metal_optimizer import WarmupCosineDecaySchedule

            lr_schedule = WarmupCosineDecaySchedule(
                initial_learning_rate=effective_lr * 0.1,  # Start at 10% of target
                warmup_steps=warmup_steps,
                decay_steps=total_steps - warmup_steps,
                min_learning_rate=1e-6,
                warmup_target=effective_lr,  # Use effective_lr (reduced for warm-start)
            )
            # Recompile model with warmup schedule
            self.model.compile(
                optimizer=keras.optimizers.Adam(learning_rate=lr_schedule),
                loss="sparse_categorical_crossentropy",
                metrics=["accuracy"],
            )
            lr_display = f"{effective_lr:.2e}" if self._is_warm_start else f"{self.config.learning_rate}"
            logger.info(
                f"✓ TCN using warmup LR schedule: {warmup_epochs} warmup epochs, target LR={lr_display}"
            )
        except ImportError as e:
            logger.warning(
                f"WarmupCosineDecaySchedule not available: {e}. Using constant LR."
            )

        # Compute class weights for imbalanced regimes
        class_weight = self._compute_class_weights(y_train_seq)

        # Callbacks
        callbacks = self._create_tcn_callbacks(keras)

        # Train with class weights
        history = self.model.fit(
            x_train_seq,
            y_train_seq,
            validation_data=(x_val_seq, y_val_seq),
            epochs=self.config.epochs,
            batch_size=self.config.batch_size,
            class_weight=class_weight,
            callbacks=callbacks,
            verbose=0,
        )

        self.is_trained = True
        self.seq_len = seq_len

        # Calculate metrics
        val_pred_probs = self.model.predict(x_val_seq, verbose=0)
        val_pred = np.argmax(val_pred_probs, axis=1)
        val_acc = np.mean(val_pred == y_val_seq)

        # Per-class accuracy
        for cls in range(self.n_classes):
            mask = y_val_seq == cls
            if np.sum(mask) > 0:
                cls_acc = np.mean(val_pred[mask] == cls)
                logger.info(f"  {self.REGIME_NAMES[cls]} accuracy: {cls_acc:.1%}")

        self.metrics = {
            "train_accuracy": float(history.history["accuracy"][-1]),
            "val_accuracy": float(val_acc),
            "epochs_trained": len(history.history["loss"]),
        }

        logger.info(f"TCN Volatility Regime trained: val_accuracy={val_acc:.1%}")
        return self.metrics

    def predict(self, X: np.ndarray) -> Dict[str, Any]:
        """Predict volatility regime (0-3) and probabilities."""
        if not self.is_trained:
            raise RuntimeError(MODEL_NOT_TRAINED_ERROR)

        # Scale
        x_scaled = self.scaler.transform(X.reshape(-1, X.shape[-1]))

        # Create sequence from last seq_len rows
        if len(x_scaled) >= self.seq_len:
            x_seq = x_scaled[-self.seq_len :].reshape(1, self.seq_len, -1)
        else:
            # Pad with zeros if not enough data
            pad_len = self.seq_len - len(x_scaled)
            x_padded = np.vstack([np.zeros((pad_len, x_scaled.shape[1])), x_scaled])
            x_seq = x_padded.reshape(1, self.seq_len, -1)

        probs = self.model.predict(x_seq, verbose=0)[0]  # Shape: (4,)
        regime = int(np.argmax(probs))
        regime_name = self.REGIME_NAMES[regime]

        return {
            "volatility_regime": regime,
            "volatility_regime_name": regime_name,
            "regime_probabilities": probs.tolist(),
            "regime_confidence": float(probs[regime]),
        }

    def save(self, path: str) -> None:
        """Save TCN model with cross-Keras-version compatibility.

        Saves:
        - .keras file (native format for current Keras version)
        - .weights.h5 file (portable weights for cross-version loading)
        - .arch.json file (model architecture JSON for rebuild)
        - .meta.pkl file (scaler, feature names, config)

        This allows loading on both Keras 2.x and 3.x environments.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        # Save Keras model in native format
        self.model.save(str(path))

        # === CROSS-VERSION COMPATIBILITY: Save weights and architecture separately ===
        # This allows loading on different Keras versions

        # 1. Save weights in H5 format (portable across Keras 2.x/3.x)
        weights_path = path.with_suffix(WEIGHTS_H5_SUFFIX)
        try:
            self.model.save_weights(str(weights_path))
            logger.debug(f"Saved portable weights to {weights_path}")
        except Exception as e:
            logger.warning(f"Could not save portable weights: {e}")

        # 2. Save architecture as JSON (for rebuilding on different Keras version)
        arch_path = path.with_suffix(ARCH_JSON_SUFFIX)
        try:
            arch_json = self.model.to_json()
            with open(arch_path, "w") as f:
                f.write(arch_json)
            logger.debug(f"Saved architecture to {arch_path}")
        except Exception as e:
            logger.warning(f"Could not save architecture JSON: {e}")

        # Save scaler and config
        meta = {
            "scaler": self.scaler,
            "seq_len": self.seq_len,
            "metrics": self.metrics,
            "config": self.config.__dict__,
            "feature_names": self.feature_names,
            "n_features": self.n_features,
            "n_classes": self.n_classes,
            "model_type": "volatility_regime",  # Identify model type
            # Store architecture config for programmatic rebuild
            "architecture": {
                "input_shape": (self.seq_len, self.n_features),
                "tcn_hidden_size": getattr(self.config, "tcn_hidden_size", 32),
                "tcn_kernel_size": getattr(self.config, "tcn_kernel_size", 3),
                "tcn_num_layers": getattr(self.config, "tcn_num_layers", 2),
                "tcn_dropout": getattr(self.config, "tcn_dropout", 0.4),
                "tcn_l2_reg": getattr(self.config, "tcn_l2_reg", 0.005),
                "tcn_spatial_dropout": getattr(self.config, "tcn_spatial_dropout", 0.2),
                "tcn_noise_std": getattr(self.config, "tcn_noise_std", 0.03),
            },
        }
        meta_path = path.with_suffix(META_PKL_SUFFIX)
        with open(meta_path, "wb") as f:
            pickle.dump(meta, f)

        logger.info(
            f"TCN Volatility Regime saved to {path} (with cross-version compatibility)"
        )

    def load(self, path: str) -> None:
        """Load TCN model with cross-Keras-version compatibility.

        Loading strategy (in order):
        1. Try native .keras format (same Keras version)
        2. Try rebuilding from .arch.json + .weights.h5 (cross-version)
        3. Try rebuilding from meta['architecture'] + .weights.h5 (cross-version, no JSON)

        This allows models trained on Keras 3.x to load on Keras 2.x and vice versa.
        """
        from tensorflow import keras

        # Import custom class so @register_keras_serializable runs before load
        try:
            from src.models.tensorflow_models import TCNVolatilityDualHead  # noqa: F401
        except ImportError:
            try:
                from models.tensorflow_models import TCNVolatilityDualHead  # noqa: F401
            except ImportError:
                logger.warning("Could not import TCNVolatilityDualHead for registration")

        path = Path(path)
        meta_path = path.with_suffix(META_PKL_SUFFIX)
        weights_path = path.with_suffix(WEIGHTS_H5_SUFFIX)
        arch_path = path.with_suffix(ARCH_JSON_SUFFIX)

        # Load metadata first (always needed)
        with open(meta_path, "rb") as f:
            meta = pickle.load(f)

        self.scaler = meta["scaler"]
        self.seq_len = meta["seq_len"]
        self.metrics = meta["metrics"]
        self.feature_names = meta.get("feature_names")
        self.n_features = meta.get("n_features")
        self.n_classes = meta.get("n_classes", 4)

        # Try loading strategies in order
        model_loaded = self._load_model_native(path, keras)

        if not model_loaded:
            model_loaded = self._load_model_from_arch_json(arch_path, weights_path, meta, keras)

        if not model_loaded:
            model_loaded = self._load_model_rebuild_architecture(weights_path, meta)

        if not model_loaded:
            raise RuntimeError(
                f"Failed to load TCN model from {path}. "
                f"Tried: native .keras, arch.json + weights.h5, rebuild + weights.h5. "
                f"Ensure model was saved with cross-version compatibility support."
            )

        self.is_trained = True


# =============================================================================
# TCN VOLATILITY REGIME TRAINER - Forward-Looking 4-Class Prediction
# =============================================================================


class _VolTrainStepModule(tf.Module):
    """Encapsulates all state for the volatility training step as a tf.Module.

    Using a tf.Module subclass instead of a closure eliminates free-variable
    capture inside @tf.function, which avoids TensorFlow tracing issues.
    All TF ops (GradientTape, add_n) are stored as attributes to avoid
    depending on global/free variables inside the @tf.function body.
    """

    def __init__(
        self, model, optimizer, cls_loss_fn, reg_loss_fn,
        cls_w, reg_w, train_loss_metric, train_acc_metric,
    ):
        super().__init__()
        self._model = model
        self._optimizer = optimizer
        self._cls_loss_fn = cls_loss_fn
        self._reg_loss_fn = reg_loss_fn
        self._cls_w = cls_w
        self._reg_w = reg_w
        self._train_loss_metric = train_loss_metric
        self._train_acc_metric = train_acc_metric
        # Store TF ops as attributes to avoid free-variable capture in @tf.function
        self._gradient_tape_cls = tf.GradientTape
        self._add_n = tf.add_n

    @tf.function
    def __call__(self, x, y, sample_weight):  # NOSONAR - all TF ops stored as tf.Module attributes
        with self._gradient_tape_cls() as tape:
            outputs = self._model(x, training=True)
            class_loss = self._cls_loss_fn(
                y['classification'], outputs['classification'],
                sample_weight=sample_weight,
            )
            reg_loss = self._reg_loss_fn(y['regression'], outputs['regression'])
            total_loss = self._cls_w * class_loss + self._reg_w * reg_loss
            if self._model.losses:
                total_loss += self._add_n(self._model.losses)
        gradients = tape.gradient(total_loss, self._model.trainable_variables)
        self._optimizer.apply_gradients(
            zip(gradients, self._model.trainable_variables)
        )
        self._train_loss_metric.update_state(total_loss)
        self._train_acc_metric.update_state(
            y['classification'], outputs['classification']
        )
        return total_loss


class _VolValStepModule(tf.Module):
    """Encapsulates all state for the volatility validation step as a tf.Module.

    Using a tf.Module subclass instead of a closure eliminates free-variable
    capture inside @tf.function, which avoids TensorFlow tracing issues.
    All TF ops are stored as attributes to avoid depending on global/free
    variables inside the @tf.function body.
    """

    def __init__(
        self, model, cls_loss_fn, reg_loss_fn,
        cls_w, reg_w, val_loss_metric, val_acc_metric,
    ):
        super().__init__()
        self._model = model
        self._cls_loss_fn = cls_loss_fn
        self._reg_loss_fn = reg_loss_fn
        self._cls_w = cls_w
        self._reg_w = reg_w
        self._val_loss_metric = val_loss_metric
        self._val_acc_metric = val_acc_metric
        # Store TF ops as attributes to avoid free-variable capture in @tf.function
        self._add_n = tf.add_n

    @tf.function
    def __call__(self, x, y, sample_weight):  # NOSONAR - all TF ops stored as tf.Module attributes
        outputs = self._model(x, training=False)
        class_loss = self._cls_loss_fn(
            y['classification'], outputs['classification'],
            sample_weight=sample_weight,
        )
        reg_loss = self._reg_loss_fn(y['regression'], outputs['regression'])
        total_loss = self._cls_w * class_loss + self._reg_w * reg_loss
        if self._model.losses:
            total_loss += self._add_n(self._model.losses)
        self._val_loss_metric.update_state(total_loss)
        self._val_acc_metric.update_state(
            y['classification'], outputs['classification']
        )
        return total_loss


def _build_vol_train_step(
    model, optimizer, cls_loss_fn, reg_loss_fn,
    cls_w, reg_w,
    train_loss_metric, train_acc_metric,
):
    """Build a training step module with zero free-variable captures."""
    return _VolTrainStepModule(
        model, optimizer, cls_loss_fn, reg_loss_fn,
        cls_w, reg_w, train_loss_metric, train_acc_metric,
    )


def _build_vol_val_step(
    model, cls_loss_fn, reg_loss_fn,
    cls_w, reg_w,
    val_loss_metric, val_acc_metric,
):
    """Build a validation step module with zero free-variable captures."""
    return _VolValStepModule(
        model, cls_loss_fn, reg_loss_fn,
        cls_w, reg_w, val_loss_metric, val_acc_metric,
    )


class _RegimeCollapseCallback(tf.keras.callbacks.Callback):
    """Detect and warn when regime predictions collapse to a single class.

    Checks validation predictions every ``check_every`` epochs and warns
    if any single class accounts for more than ``collapse_threshold`` of
    all predictions.  This is a sign that the model is ignoring minority
    classes and needs intervention (e.g. stronger focal loss alpha or
    class weights).

    Args:
        x_val: Validation input sequences.
        display: :class:`TrainingDisplay` instance for console output.
        class_names: List of human-readable class names (length = n_classes).
        check_every: Check interval in epochs (default 5).
        collapse_threshold: Fraction above which a single class is
            considered "collapsed" (default 0.80 = 80%).
    """

    def __init__(
        self,
        x_val: np.ndarray,
        display: "TrainingDisplay",
        class_names: List[str],
        check_every: int = 5,
        collapse_threshold: float = 0.80,
    ):
        super().__init__()
        self.x_val = x_val
        self.display = display
        self.class_names = class_names
        self.check_every = check_every
        self.collapse_threshold = collapse_threshold
        self.n_classes = len(class_names)

    def on_epoch_end(self, epoch: int, logs=None):
        if (epoch + 1) % self.check_every != 0:
            return
        try:
            outputs = self.model(self.x_val, training=False)
            probs = outputs["classification"] if isinstance(outputs, dict) else outputs
            pred_classes = tf.argmax(probs, axis=1).numpy()
            counts = np.bincount(pred_classes, minlength=self.n_classes)
            fracs = counts / max(len(pred_classes), 1)

            dominant_idx = int(np.argmax(fracs))
            dominant_frac = float(fracs[dominant_idx])

            if dominant_frac >= self.collapse_threshold:
                self.display.warn(
                    f"Regime collapse at epoch {epoch + 1}: "
                    f"{self.class_names[dominant_idx]} = {dominant_frac:.0%} "
                    f"(threshold {self.collapse_threshold:.0%})"
                )
        except Exception as exc:  # noqa: BLE001
            logger.debug("Regime collapse check failed: %s", exc)


class TCNVolatilityRegimeTrainer(BaseTrainer):
    """
    TCN model for FORWARD-LOOKING 4-class volatility regime prediction.

    CRITICAL CHANGE (2025): Now predicts FUTURE volatility regime, not current.
    Uses dual-head architecture: classification + regression for robustness.

    Research-backed architecture (Bai et al. / Unit8):
    - Dilated causal convolutions with exponential dilation
    - Residual connections for stable deep training
    - Weight normalization to prevent gradient explosion
    - Full receptive field coverage (≥ seq_len)

    Forward Volatility Regimes (predicted 48 bars ahead):
        - 0 = QUIET_NEXT: Future ATR < 25th percentile (skip trading)
        - 1 = STABLE_NEXT: Future ATR 25th-60th percentile (normal)
        - 2 = ACTIVE_NEXT: Future ATR 60th-85th percentile (opportunity!)
        - 3 = EXTREME_NEXT: Future ATR > 85th percentile (caution)

    Dual-Head Output:
        - Classification: 4-class softmax for regime
        - Regression: % change in volatility (fallback/tiebreaker)

    Anti-Collapse Mechanisms:
        - PredictionCollapseCallback: Detects >80% single-class predictions
        - CategoricalFocalCrossentropy: Boosts minority classes
        - Class weights: Inverse frequency weighting
        - Sample weights: Higher weight for large volatility changes

    Success Criteria:
        - Classification accuracy >60% on validation (harder task than current regime)
        - All 4 classes represented in predictions (no collapse)
        - F1-score >0.50 for ACTIVE_NEXT and EXTREME_NEXT classes
    """

    def __init__(self, config: Optional[TrainerConfig] = None):
        super().__init__(config)
        self.scaler = None
        self.feature_names = None
        self.n_features = None
        self.seq_len = None
        self.n_classes = 4
        self.class_names = ['QUIET_NEXT', 'STABLE_NEXT', 'ACTIVE_NEXT', 'EXTREME_NEXT']

        # Forward-looking parameters
        self.lookahead = getattr(self.config, 'tcn_lookahead', 48)  # 48 bars = 2 days for H1

        # TCN architecture hyperparameters
        self.kernel_size = getattr(self.config, 'tcn_kernel_size', 5)
        self.dilation_base = getattr(self.config, 'tcn_dilation_base', 2)
        self.num_filters = getattr(self.config, 'tcn_num_filters', 64)
        self.num_residual_blocks = getattr(self.config, 'tcn_num_residual_blocks', 5)
        self.dropout = getattr(self.config, 'tcn_dropout', 0.2)
        self.use_weight_norm = getattr(self.config, 'tcn_weight_norm', True)

        # Loss weights for dual-head
        self.classification_weight = 0.7
        self.regression_weight = 0.3

        # Focal loss parameters - will be overridden by class_weights if provided
        self.focal_gamma = 2.0
        self.focal_alpha = None  # Will use class_weights or config

        # Regression thresholds for fallback mapping
        self.reg_thresholds = {
            'quiet': -0.15,
            'stable_high': 0.15,
            'active_high': 0.40,
        }

        # Dual-head model flag
        self.use_dual_head = True

    def _compute_receptive_field(self) -> int:
        """Compute receptive field size for current architecture."""
        k = self.kernel_size
        b = self.dilation_base
        n = self.num_residual_blocks
        receptive_field = 1 + 2 * (k - 1) * (b ** n - 1) // (b - 1)
        return receptive_field

    def _build_model(self, input_shape: Tuple[int, int]) -> Any:
        """
        Build dual-head TCN model for forward volatility prediction.

        Uses TCNVolatilityDualHead from tensorflow_models.py.
        """
        # Import the dual-head model
        try:
            from src.models.tensorflow_models import TCNVolatilityDualHead
        except ImportError:
            from models.tensorflow_models import TCNVolatilityDualHead

        seq_len, n_features = input_shape

        # Verify receptive field coverage
        receptive_field = self._compute_receptive_field()
        if receptive_field < seq_len:
            logger.warning(f"Receptive field ({receptive_field}) < seq_len ({seq_len}). "
                           f"Consider increasing num_residual_blocks.")

        # Build dual-head model
        model = TCNVolatilityDualHead(
            n_features=n_features,
            seq_len=seq_len,
            n_classes=self.n_classes,
            num_filters=self.num_filters,
            kernel_size=self.kernel_size,
            num_residual_blocks=self.num_residual_blocks,
            dilation_base=self.dilation_base,
            dropout=self.dropout,
        )

        # Build model by calling it once
        dummy_input = tf.zeros((1, seq_len, n_features))
        _ = model(dummy_input)

        return model

    def _resolve_sample_weights(
        self,
        w_train: Optional[np.ndarray],
        w_val: Optional[np.ndarray],
        sample_weights: Optional[np.ndarray],
        n_train: int,
        n_val: int,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Resolve sample weights from multiple sources."""
        if w_train is None:
            w_train = sample_weights if sample_weights is not None else np.ones(n_train)
        if w_val is None:
            w_val = np.ones(n_val)
        return w_train, w_val

    def _compute_focal_alpha(
        self, class_weights: Optional[Dict[int, float]]
    ) -> List[float]:
        """Compute focal alpha from class weights or use defaults."""
        if class_weights is not None:
            cw_values = [class_weights.get(i, 1.0) for i in range(self.n_classes)]
            cw_sum = sum(cw_values)
            alpha = [v / cw_sum for v in cw_values]
            logger.debug(f"Focal alpha from class weights: {[f'{a:.3f}' for a in alpha]}")
        else:
            alpha = [0.30, 0.20, 0.25, 0.25]  # QUIET, STABLE, ACTIVE, EXTREME
            logger.debug(f"Using default focal alpha: {alpha}")
        return alpha

    def _show_train_config(
        self,
        display: "TrainingDisplay",
        w_train: np.ndarray,
        y_train: np.ndarray,
        tcn_lr: float,
    ) -> None:
        """Show clean configuration summary."""
        train_valid_mask = w_train > 0
        valid_labels = y_train[train_valid_mask]
        class_counts = np.bincount(valid_labels, minlength=self.n_classes) if len(valid_labels) > 0 else [0] * 4
        n_valid = max(len(valid_labels), 1)

        display.show_config({
            "Lookahead": f"{self.lookahead} bars",
            "Params": f"{self.model.count_params():,}",
            "LR": f"{tcn_lr:.1e}",
            "Loss": f"FocalCE(γ={self.focal_gamma})",
            "Classes": (
                f"QUI:{class_counts[0]/n_valid:.0%} STA:{class_counts[1]/n_valid:.0%} "
                f"ACT:{class_counts[2]/n_valid:.0%} EXT:{class_counts[3]/n_valid:.0%}"
            ),
        })

    def _create_regime_collapse_callback(
        self, x_val: np.ndarray, display: "TrainingDisplay"
    ) -> Any:
        """Create a regime collapse detection callback."""
        return _RegimeCollapseCallback(x_val, display, list(self.class_names))

    def _create_vol_regime_callbacks(
        self, x_val: np.ndarray, display: "TrainingDisplay"
    ) -> list:
        """Create all callbacks for TCN volatility regime training."""
        from tensorflow import keras

        return [
            RichEpochCallback(
                model_name="TCN Forward Volatility",
                total_epochs=self.config.epochs,
            ),
            AutoAdjustCallback(
                patience=8, lr_factor=0.5, min_lr=1e-6,
                max_adjustments=4, min_delta=0.005, verbose=True,
            ),
            keras.callbacks.EarlyStopping(
                monitor='val_loss', patience=self.config.patience, mode='min',
                restore_best_weights=True, verbose=0,
                start_from_epoch=max(10, self.config.min_epochs),
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss', factor=0.5,
                patience=max(5, self.config.patience // 3),
                min_lr=1e-6, verbose=0,
            ),
            self._create_regime_collapse_callback(x_val, display),
        ]

    def _create_vol_datasets(
        self,
        x_train: np.ndarray,
        y_train_onehot: np.ndarray,
        y_train_reg: np.ndarray,
        w_train: np.ndarray,
        x_val: np.ndarray,
        y_val_onehot: np.ndarray,
        y_val_reg: np.ndarray,
        w_val: np.ndarray,
    ) -> Tuple[Any, Any]:
        """Create tf.data.Dataset for training and validation."""
        train_dataset = tf.data.Dataset.from_tensor_slices((
            x_train,
            {'classification': y_train_onehot, 'regression': y_train_reg.reshape(-1, 1)},
            w_train,
        )).shuffle(1024).batch(self.config.batch_size).prefetch(tf.data.AUTOTUNE)

        val_dataset = tf.data.Dataset.from_tensor_slices((
            x_val,
            {'classification': y_val_onehot, 'regression': y_val_reg.reshape(-1, 1)},
            w_val,
        )).batch(self.config.batch_size).prefetch(tf.data.AUTOTUNE)

        return train_dataset, val_dataset

    def _build_train_val_steps(
        self,
        optimizer: Any,
        classification_loss_fn: Any,
        regression_loss_fn: Any,
        metrics: Dict[str, Any],
    ) -> Tuple[Any, Any]:
        """Build tf.function train and validation step functions.

        Uses a tf.Module subclass to hold all dependencies as tracked
        attributes, eliminating free-variable capture in @tf.function
        closures which can cause issues with TensorFlow tracing.

        Returns:
            Tuple of (train_step_fn, val_step_fn)
        """

        cls_w = tf.constant(self.classification_weight, dtype=tf.float32)
        reg_w = tf.constant(self.regression_weight, dtype=tf.float32)

        train_step_fn = _build_vol_train_step(
            model=self.model,
            optimizer=optimizer,
            cls_loss_fn=classification_loss_fn,
            reg_loss_fn=regression_loss_fn,
            cls_w=cls_w,
            reg_w=reg_w,
            train_loss_metric=metrics['train_loss'],
            train_acc_metric=metrics['train_acc'],
        )

        val_step_fn = _build_vol_val_step(
            model=self.model,
            cls_loss_fn=classification_loss_fn,
            reg_loss_fn=regression_loss_fn,
            cls_w=cls_w,
            reg_w=reg_w,
            val_loss_metric=metrics['val_loss'],
            val_acc_metric=metrics['val_acc'],
        )

        return train_step_fn, val_step_fn

    def _run_epoch(
        self,
        train_dataset: Any,
        val_dataset: Any,
        train_step: Any,
        val_step: Any,
        metrics: Dict[str, Any],
    ) -> Tuple[float, float, float, float]:
        """Run a single training epoch and return (train_loss, train_acc, val_loss, val_acc)."""
        metrics['train_loss'].reset_state()
        metrics['train_acc'].reset_state()
        metrics['val_loss'].reset_state()
        metrics['val_acc'].reset_state()

        for x_batch, y_batch, w_batch in train_dataset:
            train_step(x_batch, y_batch, w_batch)
        for x_batch, y_batch, w_batch in val_dataset:
            val_step(x_batch, y_batch, w_batch)

        return (
            metrics['train_loss'].result().numpy(),
            metrics['train_acc'].result().numpy(),
            metrics['val_loss'].result().numpy(),
            metrics['val_acc'].result().numpy(),
        )

    def _notify_callbacks(self, callbacks: list, epoch: int, logs: dict) -> None:
        """Notify all callbacks of epoch end."""
        for callback in callbacks:
            if hasattr(callback, 'set_model'):
                callback.set_model(self.model)
            callback.on_epoch_end(epoch, logs)

    def _run_dual_head_training_loop(
        self,
        train_dataset: Any,
        val_dataset: Any,
        optimizer: Any,
        classification_loss_fn: Any,
        regression_loss_fn: Any,
        callbacks: list,
    ) -> dict:
        """Run the custom dual-head training loop and return history."""
        from tensorflow import keras

        metrics_dict = {
            'train_loss': keras.metrics.Mean(name='train_loss'),
            'train_acc': keras.metrics.CategoricalAccuracy(name='train_accuracy'),
            'val_loss': keras.metrics.Mean(name='val_loss'),
            'val_acc': keras.metrics.CategoricalAccuracy(name='val_accuracy'),
        }

        train_step, val_step = self._build_train_val_steps(
            optimizer, classification_loss_fn, regression_loss_fn, metrics_dict,
        )

        best_val_loss = float('inf')
        best_weights = None
        patience_counter = 0
        history: Dict[str, list] = {'loss': [], 'accuracy': [], 'val_loss': [], 'val_accuracy': []}

        for epoch in range(self.config.epochs):
            train_loss, train_acc, val_loss, val_acc = self._run_epoch(
                train_dataset, val_dataset, train_step, val_step, metrics_dict,
            )

            history['loss'].append(train_loss)
            history['accuracy'].append(train_acc)
            history['val_loss'].append(val_loss)
            history['val_accuracy'].append(val_acc)

            logs = {'loss': train_loss, 'accuracy': train_acc,
                    'val_loss': val_loss, 'val_accuracy': val_acc}
            self._notify_callbacks(callbacks, epoch, logs)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_weights = self.model.get_weights()
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= self.config.patience and epoch >= self.config.min_epochs:
                logger.info(f"Early stopping at epoch {epoch+1}")
                break

        if best_weights is not None:
            self.model.set_weights(best_weights)

        return history

    def _compute_vol_regime_metrics(
        self,
        x_val: np.ndarray,
        y_val: np.ndarray,
        history: dict,
    ) -> Dict[str, float]:
        """Compute final metrics for the volatility regime model."""
        from sklearn.metrics import f1_score

        val_outputs = self.model.predict(x_val, verbose=0)
        val_pred_probs = val_outputs['classification'] if isinstance(val_outputs, dict) else val_outputs
        val_pred_classes = np.argmax(val_pred_probs, axis=1)
        val_acc = np.mean(val_pred_classes == y_val)

        pred_dist = np.bincount(val_pred_classes, minlength=self.n_classes) / len(val_pred_classes)
        f1_scores = f1_score(y_val, val_pred_classes, average=None, zero_division=0)
        f1_macro = f1_score(y_val, val_pred_classes, average='macro', zero_division=0)

        active_extreme_mask = (y_val >= 2)
        active_extreme_acc = (
            float(np.mean(val_pred_classes[active_extreme_mask] >= 2))
            if active_extreme_mask.sum() > 0 else 0.0
        )

        all_classes_present = all(pred_dist[i] > 0.05 for i in range(4))

        return {
            'train_accuracy': float(history['accuracy'][-1]) if history['accuracy'] else 0.0,
            'val_accuracy': float(val_acc),
            'val_f1_macro': float(f1_macro),
            'val_f1_quiet': float(f1_scores[0]) if len(f1_scores) > 0 else 0.0,
            'val_f1_stable': float(f1_scores[1]) if len(f1_scores) > 1 else 0.0,
            'val_f1_active': float(f1_scores[2]) if len(f1_scores) > 2 else 0.0,
            'val_f1_extreme': float(f1_scores[3]) if len(f1_scores) > 3 else 0.0,
            'active_extreme_detection': active_extreme_acc,
            'all_classes_present': all_classes_present,
            'epochs_trained': len(history['loss']),
            'receptive_field': self._compute_receptive_field(),
            'lookahead': self.lookahead,
            '_f1_scores': f1_scores,  # Temp key for display
            '_pred_dist': pred_dist,  # Temp key for display
        }

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        x_val: np.ndarray,
        y_val: np.ndarray,
        feature_names: Optional[list] = None,
        class_weights: Optional[Dict[int, float]] = None,
        sample_weights: Optional[np.ndarray] = None,
        seq_len: int = 60,
        y_train_reg: Optional[np.ndarray] = None,
        y_val_reg: Optional[np.ndarray] = None,
        w_train: Optional[np.ndarray] = None,
        w_val: Optional[np.ndarray] = None,
    ) -> Dict[str, float]:
        """
        Train dual-head TCN for forward 4-class volatility regime prediction.

        Args:
            X_train: Training sequences (batch, seq_len, features)
            y_train: Training labels (batch,) with values 0-3
            x_val: Validation sequences
            y_val: Validation labels
            feature_names: Feature names for interpretability
            class_weights: Class weights for imbalanced data
            sample_weights: Per-sample weights (deprecated, use w_train)
            seq_len: Sequence length
            y_train_reg: Regression targets for training (% vol change)
            y_val_reg: Regression targets for validation
            w_train: Sample weights for training
            w_val: Sample weights for validation

        Returns:
            Dict with training metrics
        """
        from tensorflow import keras

        display = TrainingDisplay("TCN Forward Volatility")

        # Save metadata
        self.feature_names = feature_names
        self.n_features = X_train.shape[-1]
        self.seq_len = seq_len if X_train.ndim == 3 else 60
        self.scaler = None  # Assume pre-scaled from data loader

        if X_train.ndim != 3:
            raise ValueError(f"Expected 3D input (batch, seq_len, features), got {X_train.shape}")

        # Resolve weights and regression targets
        w_train, w_val = self._resolve_sample_weights(
            w_train, w_val, sample_weights, len(y_train), len(y_val),
        )
        if y_train_reg is None:
            y_train_reg = np.zeros(len(y_train), dtype=np.float32)
        if y_val_reg is None:
            y_val_reg = np.zeros(len(y_val), dtype=np.float32)

        # Convert to one-hot and build model
        y_train_onehot = tf.keras.utils.to_categorical(y_train, num_classes=self.n_classes)
        y_val_onehot = tf.keras.utils.to_categorical(y_val, num_classes=self.n_classes)
        self.model = self._build_model((self.seq_len, self.n_features))

        # Learning rate and focal alpha
        tcn_lr = min(self.config.learning_rate, 0.001)
        self.focal_alpha = self._compute_focal_alpha(class_weights)

        # Losses and optimizer
        classification_loss_fn = keras.losses.CategoricalFocalCrossentropy(
            gamma=self.focal_gamma, alpha=self.focal_alpha, from_logits=False,
        )
        regression_loss_fn = keras.losses.MeanSquaredError()
        optimizer = keras.optimizers.Adam(learning_rate=tcn_lr)
        self.model.optimizer = optimizer

        # Display and callbacks
        self._show_train_config(display, w_train, y_train, tcn_lr)
        callbacks = self._create_vol_regime_callbacks(x_val, display)

        # Create datasets
        train_dataset, val_dataset = self._create_vol_datasets(
            X_train, y_train_onehot, y_train_reg, w_train,
            x_val, y_val_onehot, y_val_reg, w_val,
        )

        # Run training loop
        history = self._run_dual_head_training_loop(
            train_dataset, val_dataset, optimizer,
            classification_loss_fn, regression_loss_fn, callbacks,
        )

        self.is_trained = True

        # Compute and display metrics
        self.metrics = self._compute_vol_regime_metrics(x_val, y_val, history)

        # Extract temp keys for display then remove them
        f1_scores = self.metrics.pop('_f1_scores')
        self.metrics.pop('_pred_dist')

        display.show_summary({
            'Val Accuracy': f"{self.metrics['val_accuracy']:.1%}",
            'F1 Macro': f"{self.metrics['val_f1_macro']:.3f}",
            'F1 (QUI/STA/ACT/EXT)': f"{f1_scores[0]:.2f} / {f1_scores[1]:.2f} / {f1_scores[2]:.2f} / {f1_scores[3]:.2f}",
            'Active/Extreme Det': f"{self.metrics['active_extreme_detection']:.1%}",
            'Epochs': self.metrics['epochs_trained'],
            'Status': "✓ Healthy" if self.metrics['all_classes_present'] else "⚠ Collapse",
        }, title="Training Complete")

        return self.metrics

    def predict(self, X: np.ndarray) -> Dict[str, Any]:
        """
        Predict forward volatility regime for input sequence.

        Returns:
            Dict with:
                - regime: int (0=QUIET_NEXT, 1=STABLE_NEXT, 2=ACTIVE_NEXT, 3=EXTREME_NEXT)
                - regime_name: str
                - probabilities: np.ndarray of 4 class probabilities
                - confidence: float (max probability)
                - vol_change_pct: float (regression prediction)
                - regression_regime: int (regime from regression fallback)
                - is_opportunity: bool (ACTIVE_NEXT or STABLE_NEXT - allow trading)
        """
        if not self.is_trained:
            raise RuntimeError("Model not trained")

        # Handle input shape
        if X.ndim == 2:
            if len(X) >= self.seq_len:
                x_seq = X[-self.seq_len:].reshape(1, self.seq_len, -1)
            else:
                pad_len = self.seq_len - len(X)
                x_padded = np.vstack([np.zeros((pad_len, X.shape[1])), X])
                x_seq = x_padded.reshape(1, self.seq_len, -1)
        elif X.ndim == 3:
            x_seq = X
        else:
            raise ValueError(f"Expected 2D or 3D input, got shape {X.shape}")

        # Get predictions
        outputs = self.model.predict(x_seq, verbose=0)

        if isinstance(outputs, dict):
            probs = outputs['classification'][0]
            vol_change = float(outputs['regression'][0, 0])
        else:
            probs = outputs[0]
            vol_change = 0.0

        regime = int(np.argmax(probs))
        confidence = float(probs[regime])

        # Regression fallback mapping
        if vol_change < self.reg_thresholds['quiet']:
            reg_regime = 0  # QUIET_NEXT
        elif vol_change < self.reg_thresholds['stable_high']:
            reg_regime = 1  # STABLE_NEXT
        elif vol_change < self.reg_thresholds['active_high']:
            reg_regime = 2  # ACTIVE_NEXT
        else:
            reg_regime = 3  # EXTREME_NEXT

        # Use regression fallback if classification confidence is low
        final_regime = regime
        if confidence < 0.60:
            logger.debug(f"Low classification confidence ({confidence:.1%}), using regression fallback")
            final_regime = reg_regime

        return {
            'regime': final_regime,
            'regime_name': self.class_names[final_regime],
            'probabilities': probs,
            'confidence': confidence,
            'vol_change_pct': vol_change,
            'regression_regime': reg_regime,
            'classification_regime': regime,
            'is_opportunity': final_regime in [1, 2],  # STABLE or ACTIVE - allow trading
            'is_high_volatility': final_regime >= 2,  # ACTIVE or EXTREME
        }

    def save(self, path: str) -> None:
        """Save TCN Forward Volatility model."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        self.model.save(str(path))

        meta = {
            'scaler': self.scaler,
            'seq_len': self.seq_len,
            'n_features': self.n_features,
            'n_classes': self.n_classes,
            'class_names': self.class_names,
            'metrics': self.metrics,
            'feature_names': self.feature_names,
            'lookahead': self.lookahead,
            'reg_thresholds': self.reg_thresholds,
            'config': {
                'kernel_size': self.kernel_size,
                'dilation_base': self.dilation_base,
                'num_filters': self.num_filters,
                'num_residual_blocks': self.num_residual_blocks,
                'dropout': self.dropout,
                'use_weight_norm': self.use_weight_norm,
                'receptive_field': self._compute_receptive_field(),
                'focal_gamma': self.focal_gamma,
                'focal_alpha': self.focal_alpha,
            },
        }

        meta_path = path.with_suffix('.meta.pkl')
        with open(meta_path, 'wb') as f:
            pickle.dump(meta, f)

        logger.info(f"TCN Forward Volatility saved to {path}")

    def load(self, path: str) -> None:
        """Load TCN Forward Volatility model."""
        from tensorflow import keras

        path = Path(path)

        # Try to load with custom objects
        try:
            from src.models.tensorflow_models import TCNVolatilityDualHead as _tcn_vol_cls
        except ImportError:
            try:
                from models.tensorflow_models import TCNVolatilityDualHead as _tcn_vol_cls
            except ImportError:
                _tcn_vol_cls = None

        custom_objects = {}
        if _tcn_vol_cls is not None:
            custom_objects['TCNVolatilityDualHead'] = _tcn_vol_cls

        self.model = keras.models.load_model(str(path), custom_objects=custom_objects)

        meta_path = path.with_suffix('.meta.pkl')
        with open(meta_path, 'rb') as f:
            meta = pickle.load(f)

        self.scaler = meta.get('scaler')
        self.seq_len = meta['seq_len']
        self.n_features = meta['n_features']
        self.n_classes = meta.get('n_classes', 4)
        self.class_names = meta.get('class_names', ['QUIET_NEXT', 'STABLE_NEXT', 'ACTIVE_NEXT', 'EXTREME_NEXT'])
        self.metrics = meta.get('metrics', {})
        self.feature_names = meta.get('feature_names')
        self.lookahead = meta.get('lookahead', 48)
        self.reg_thresholds = meta.get('reg_thresholds', {
            'quiet': -0.15, 'stable_high': 0.15, 'active_high': 0.40
        })

        # Restore architecture config
        arch_config = meta.get('config', {})
        self.kernel_size = arch_config.get('kernel_size', 5)
        self.dilation_base = arch_config.get('dilation_base', 2)
        self.num_filters = arch_config.get('num_filters', 64)
        self.num_residual_blocks = arch_config.get('num_residual_blocks', 5)
        self.dropout = arch_config.get('dropout', 0.2)
        self.use_weight_norm = arch_config.get('use_weight_norm', True)
        self.focal_gamma = arch_config.get('focal_gamma', 2.0)
        self.focal_alpha = arch_config.get('focal_alpha', [0.35, 0.25, 0.25, 0.15])

        self.is_trained = True

        logger.info(f"TCN Forward Volatility loaded from {path}")


# =============================================================================
# TRANSFORMER TRAINER - Direction Prediction (Replacement for TCN)
# =============================================================================


class _PredictionCollapseCallback(tf.keras.callbacks.Callback):
    """Detect and recover from prediction collapse during training.

    Monitors validation predictions for excessive class imbalance (>95% one class)
    and applies progressive recovery strategies: LR reduction, weight perturbation,
    and training stop with best-weight restoration.
    """

    def __init__(self, x_val, y_val, initial_lr_reductions=0):
        super().__init__()
        self.x_val = x_val
        self.y_val = y_val
        self.check_every = 5
        self.max_consecutive = 5
        self.warmup_epochs = 15
        self.consecutive_collapses = 0
        self.lr_reductions = initial_lr_reductions
        self.recovery_count = 0
        self.weight_perturbations = 0
        self.best_weights = None
        self.best_diversity = 0.0

    def on_epoch_end(self, epoch, logs=None):
        if epoch < self.warmup_epochs or (epoch + 1) % self.check_every != 0:
            return
        self._check_collapse(epoch)

    def _check_collapse(self, epoch):
        try:
            preds = self.model.predict(self.x_val, verbose=0).flatten()
            pred_up_pct = (preds > 0.5).mean() * 100
            diversity = min(pred_up_pct, 100 - pred_up_pct)

            if diversity > self.best_diversity:
                self.best_diversity = diversity
                self.best_weights = self.model.get_weights()

            if pred_up_pct > 95 or pred_up_pct < 5:
                self._handle_collapse(epoch, pred_up_pct)
            elif self.consecutive_collapses > 0:
                self._handle_recovery(diversity)
        except Exception as e:
            logger.warning(f"⚠️ Collapse check failed at epoch {epoch + 1}: {e}")

    def _handle_collapse(self, epoch, pred_up_pct):
        self.consecutive_collapses += 1
        collapse_dir = "UP" if pred_up_pct > 95 else "DOWN"
        logger.warning(
            f"⚠️ PREDICTION COLLAPSE #{self.consecutive_collapses}/{self.max_consecutive} "
            f"at epoch {epoch + 1}: {pred_up_pct:.1f}% {collapse_dir}"
        )
        self._apply_recovery_strategy()

    def _apply_recovery_strategy(self):
        if self.consecutive_collapses <= 2 and self.lr_reductions < 2:
            self._reduce_learning_rate()
        elif self.consecutive_collapses <= 4 and self.weight_perturbations < 2:
            self._perturb_weights()

        if self.consecutive_collapses >= self.max_consecutive:
            self._stop_training()

    def _reduce_learning_rate(self):
        old_lr = _safe_get_learning_rate(self.model.optimizer)
        new_lr = old_lr * 0.5
        if _safe_set_learning_rate(self.model.optimizer, new_lr):
            self.lr_reductions += 1
            logger.warning(f"🔻 LR reduced #{self.lr_reductions}: {old_lr:.2e} → {new_lr:.2e}")

    def _perturb_weights(self):
        try:
            weights = self.model.get_weights()
            rng = np.random.default_rng(seed=42 + self.weight_perturbations)
            noise_scale = 0.01 * (1 + self.weight_perturbations)
            perturbed = [w + rng.normal(0, noise_scale, w.shape).astype(w.dtype) for w in weights]
            self.model.set_weights(perturbed)
            self.weight_perturbations += 1
            logger.warning(f"🎲 Weight perturbation #{self.weight_perturbations} applied")
        except Exception as e:
            logger.warning(f"⚠️ Weight perturbation failed: {e}")

    def _stop_training(self):
        if self.best_weights is not None:
            self.model.set_weights(self.best_weights)
            logger.warning(f"🔄 Restored best weights (diversity={self.best_diversity:.1f}%)")
        logger.error(f"🛑 STOPPING TRAINING: {self.consecutive_collapses} consecutive collapses")
        self.model.stop_training = True

    def _handle_recovery(self, diversity):
        self.recovery_count += 1
        logger.info(f"✅ Recovered from collapse (diversity={diversity:.1f}%)")
        self.consecutive_collapses = 0


class TransformerDirectionTrainer(BaseTrainer):
    """
    Transformer model for direction prediction with advanced continual learning.

    Self-attention captures long-range dependencies in price trends,
    making it better suited for direction prediction than TCN.

    Features (2025):
    - EMA shadow weights for stable inference
    - EWC for multi-instrument learning without forgetting
    - Replay buffer to retain past market patterns
    - Training lineage tracking
    - Warm-start with LR reduction

    Input: Directional features (ADX, MACD, SMA crosses, market structure)
    Output: Binary direction (0=down, 1=up)
    """

    def __init__(self, config: Optional[TrainerConfig] = None):
        super().__init__(config)
        self.scaler = None
        self.feature_names = None
        self.n_features = None
        self.seq_len = None
        self.selected_indices = None  # Feature selection indices (set during training)

        # Transformer-specific config - defaults match TrainerConfig for proven 60.9% config
        self.transformer_d_model = (
            getattr(config, "transformer_d_model", 16) if config else 16
        )
        self.transformer_num_heads = (
            getattr(config, "transformer_num_heads", 2) if config else 2
        )
        self.transformer_num_layers = (
            getattr(config, "transformer_num_layers", 1) if config else 1
        )
        self.transformer_dff = getattr(config, "transformer_dff", 32) if config else 32
        self.transformer_dropout = (
            getattr(config, "transformer_dropout", 0.2) if config else 0.2
        )  # Reduced from 0.4

        # === CONTINUAL LEARNING COMPONENTS ===

        # EMA for stable inference
        self.ema: Optional[EMACallback] = None
        self._use_ema = getattr(config, "use_ema", True) if config else True

        # EWC for multi-instrument learning
        self.ewc: Optional[EWCPenalty] = None
        self._use_ewc = getattr(config, "use_ewc", True) if config else True

        # Replay buffer
        self.replay_buffer: Optional[ReplayBuffer] = None
        self._use_replay = (
            getattr(config, "use_replay_buffer", True) if config else True
        )

        # Drift detector
        self.drift_detector: Optional[DriftDetector] = None
        self._drift_threshold = (
            getattr(config, "drift_threshold", 0.03) if config else 0.03
        )

        # Training lineage
        self.lineage: Optional[TrainingLineage] = None

        # Track if this is a warm-start session
        self._is_warm_start = False

    def _build_model(self, input_shape: Tuple[int, int]) -> Any:
        """
        Build Transformer model architecture.

        Key changes for anti-collapse:
        1. Very light L2 regularization (0.001) - too much suppresses outputs
        2. Moderate dropout (0.25)
        3. Small model capacity (d_model=32, 2 layers)
        4. NO regularization on output layer - let it learn freely
        5. Output bias initialized to small positive value (assumes ~52% UP)
        """
        from tensorflow import keras

        # Focal Loss and Class-Balanced Loss are imported later when needed for class imbalance handling

        seq_len, n_features = input_shape

        # L2 regularization — read from config (wired from YAML transformer.l2_reg)
        l2_weight = getattr(self.config, "transformer_l2_reg", 0.003) if self.config else 0.003
        l2_reg = keras.regularizers.l2(l2_weight)

        # Input
        inp = keras.Input(shape=(seq_len, n_features), name="features")

        # Input noise — read from config (wired from YAML transformer.input_noise)
        noise_level = getattr(self.config, "transformer_input_noise", 0.02) if self.config else 0.02
        x = keras.layers.GaussianNoise(noise_level)(inp)

        # Spatial dropout on input sequence — read from config (wired from YAML)
        spatial_dropout = getattr(self.config, "transformer_spatial_dropout", 0.10) if self.config else 0.10
        x = keras.layers.SpatialDropout1D(spatial_dropout)(x)

        # Project features to d_model dimension (with L2)
        x = keras.layers.Dense(
            self.transformer_d_model, kernel_regularizer=l2_reg, name="input_projection"
        )(x)
        proj_dropout = getattr(self.config, "transformer_projection_dropout", 0.10) if self.config else 0.10
        x = keras.layers.Dropout(proj_dropout)(x)

        # Add positional encoding
        x = self._add_positional_encoding(x, seq_len, self.transformer_d_model)

        # Transformer encoder layers
        for i in range(self.transformer_num_layers):
            x = self._transformer_encoder_layer(
                x,
                self.transformer_d_model,
                self.transformer_num_heads,
                self.transformer_dff,
                self.transformer_dropout,  # Uses config dropout (0.4)
                l2_reg,
                name_prefix=f"transformer_{i}",
            )

        # Global pooling and output
        x = keras.layers.GlobalAveragePooling1D()(x)

        # Use tanh instead of ReLU - tanh outputs [-1, 1] which allows both positive
        # and negative contributions to the sigmoid input, making it easier to balance around 0.5
        x = keras.layers.Dense(
            self.config.final_dense_units,
            activation=self.config.final_dense_activation,
            kernel_regularizer=l2_reg,
        )(x)
        x = keras.layers.Dropout(self.config.final_dense_dropout)(x)

        # Binary direction output
        # With tanh inputs ranging [-1, 1], the dot product with weights can be near 0
        # Use small kernel init and zero bias to start at sigmoid(0) = 0.5
        direction = keras.layers.Dense(
            1,
            activation="sigmoid",
            name="direction",
            dtype="float32",
            kernel_initializer=keras.initializers.TruncatedNormal(
                mean=0.0, stddev=0.05
            ),
            bias_initializer=keras.initializers.Zeros(),  # Start at sigmoid(0) = 0.5
        )(x)

        model = keras.Model(inputs=inp, outputs=direction, name="transformer_direction")

        # Adam optimizer with standard learning rate
        optimizer = keras.optimizers.Adam(learning_rate=self.config.learning_rate)
        logger.info(f"Using Adam optimizer with lr={self.config.learning_rate:.2e}")

        # Standard binary cross entropy - we'll handle calibration post-training
        model.compile(
            optimizer=optimizer,
            loss=keras.losses.BinaryCrossentropy(label_smoothing=0.0),
            metrics=[keras.metrics.BinaryAccuracy(name="accuracy", threshold=0.5)],
        )

        return model

    def _add_positional_encoding(self, x, seq_len: int, d_model: int):
        """Add sinusoidal positional encoding."""
        # Create positional encoding
        positions = np.arange(seq_len)[:, np.newaxis]
        dims = np.arange(d_model)[np.newaxis, :]

        angles = positions / np.power(10000, (2 * (dims // 2)) / d_model)

        # Apply sin to even indices, cos to odd
        pos_encoding = np.zeros((seq_len, d_model))
        pos_encoding[:, 0::2] = np.sin(angles[:, 0::2])
        pos_encoding[:, 1::2] = np.cos(angles[:, 1::2])

        pos_encoding = pos_encoding[np.newaxis, :, :].astype(np.float32)

        # Add positional encoding to input (cast to match input dtype for mixed precision)
        pos_tensor = tf.constant(pos_encoding)
        pos_tensor = tf.cast(pos_tensor, x.dtype)
        return x + pos_tensor

    def _transformer_encoder_layer(
        self,
        x,
        d_model: int,
        num_heads: int,
        dff: int,
        dropout: float,
        l2_reg,
        name_prefix: str,
    ):
        """Single transformer encoder layer with multi-head attention and feedforward."""
        from tensorflow import keras

        # Multi-head self-attention
        attn_output = keras.layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=d_model // num_heads, name=f"{name_prefix}_mha"
        )(x, x)
        attn_output = keras.layers.Dropout(dropout)(attn_output)
        x = keras.layers.LayerNormalization(epsilon=1e-6, name=f"{name_prefix}_ln1")(
            x + attn_output
        )

        # Feedforward network with L2 regularization
        ffn = keras.layers.Dense(
            dff,
            activation="relu",
            kernel_regularizer=l2_reg,
            name=f"{name_prefix}_ffn1",
        )(x)
        ffn = keras.layers.Dense(
            d_model, kernel_regularizer=l2_reg, name=f"{name_prefix}_ffn2"
        )(ffn)
        ffn = keras.layers.Dropout(dropout)(ffn)
        x = keras.layers.LayerNormalization(epsilon=1e-6, name=f"{name_prefix}_ln2")(
            x + ffn
        )

        return x

    # =========================================================================
    # TRAIN METHOD HELPER FUNCTIONS (Refactored for cognitive complexity < 15)
    # =========================================================================

    def _initialize_training_state(
        self, instrument: str, data_range: str
    ) -> None:
        """Initialize training lineage and drift detector."""
        self.lineage = TrainingLineage()
        self.lineage.generate_checkpoint_id()
        self.lineage.instrument = instrument
        self.lineage.data_range = data_range
        self.lineage.granularity = (
            getattr(self.config, "granularity", "H1") if self.config else "H1"
        )

        self.drift_detector = DriftDetector(
            performance_threshold=self._drift_threshold,
            feature_drift_threshold=0.10,
            window_size=5,
        )

    def _scale_features(
        self,
        X_train: np.ndarray,
        x_val: np.ndarray,
        skip_scaling: bool,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Scale features using StandardScaler."""
        from sklearn.preprocessing import StandardScaler

        if skip_scaling:
            logger.info("Skipping scaling (data is already pre-scaled)")
            self.scaler = None
            x_train_scaled = X_train.reshape(-1, X_train.shape[-1])
            x_val_scaled = x_val.reshape(-1, x_val.shape[-1])
        else:
            self.scaler = StandardScaler()
            x_train_scaled = self.scaler.fit_transform(
                X_train.reshape(-1, X_train.shape[-1])
            )
            x_val_scaled = self.scaler.transform(x_val.reshape(-1, x_val.shape[-1]))

        return x_train_scaled, x_val_scaled

    def _load_warm_start_feature_meta(
        self, warm_start_path: Optional[str]
    ) -> Tuple[Optional[list], Optional[Any]]:
        """Load feature names and scaler from warm-start checkpoint."""
        if not warm_start_path or not Path(warm_start_path).exists():
            return None, None

        meta_path = Path(warm_start_path).with_suffix(META_PKL_SUFFIX)
        if not meta_path.exists():
            return None, None

        try:
            with open(meta_path, "rb") as f:
                warm_meta = pickle.load(f)
            feature_names = warm_meta.get("feature_names")
            scaler = warm_meta.get("scaler")
            if feature_names:
                logger.info(
                    f"🔥 WARM-START: Loaded {len(feature_names)} feature names from checkpoint"
                )
                logger.info(f"   Features: {feature_names[:5]}...")
            return feature_names, scaler
        except Exception as e:
            logger.warning(f"Could not load warm-start meta for features: {e}")
            return None, None

    def _apply_warm_start_features(
        self,
        x_train_scaled: np.ndarray,
        x_val_scaled: np.ndarray,
        warm_start_feature_names: list,
        warm_start_scaler: Any,
    ) -> Tuple[np.ndarray, np.ndarray, bool]:
        """Apply warm-start feature selection if compatible.

        Returns (x_train, x_val, is_compatible).
        """
        saved_features_set = set(warm_start_feature_names)
        current_features_set = set(self.feature_names)
        common_features = saved_features_set.intersection(current_features_set)

        # Check for exact match (all saved features available)
        if len(common_features) == len(warm_start_feature_names):
            return self._apply_exact_feature_match(
                x_train_scaled, x_val_scaled,
                warm_start_feature_names, warm_start_scaler
            )

        # Check for partial match (80%+ overlap)
        if len(common_features) >= len(warm_start_feature_names) * 0.8:
            return self._apply_partial_feature_match(
                x_train_scaled, x_val_scaled,
                warm_start_feature_names, common_features
            )

        # Insufficient overlap - run fresh selection
        logger.warning(
            f"⚠️ Only {len(common_features)}/{len(warm_start_feature_names)} "
            "saved features available. Running fresh selection."
        )
        return x_train_scaled, x_val_scaled, False

    def _apply_exact_feature_match(
        self,
        x_train_scaled: np.ndarray,
        x_val_scaled: np.ndarray,
        warm_start_feature_names: list,
        warm_start_scaler: Any,
    ) -> Tuple[np.ndarray, np.ndarray, bool]:
        """Apply exact feature match from warm-start."""
        from sklearn.preprocessing import StandardScaler

        selected_indices = [
            self.feature_names.index(f)
            for f in warm_start_feature_names
            if f in self.feature_names
        ]

        x_train_scaled = x_train_scaled[:, selected_indices]
        x_val_scaled = x_val_scaled[:, selected_indices]

        self.selected_indices = selected_indices  # Store for inference/validation
        self.feature_names = [self.feature_names[i] for i in selected_indices]
        self.n_features = len(selected_indices)

        if warm_start_scaler is not None:
            self.scaler = warm_start_scaler
            logger.info("✓ Using saved scaler from checkpoint")
        elif self.scaler is not None:
            self.scaler = StandardScaler()
            x_train_scaled = self.scaler.fit_transform(x_train_scaled)
            x_val_scaled = self.scaler.transform(x_val_scaled)

        logger.info(
            f"🔥 WARM-START: Using ALL {len(selected_indices)} saved features (weights compatible)"
        )
        logger.info(f"   Top features: {self.feature_names[:5]}")
        return x_train_scaled, x_val_scaled, True

    def _apply_partial_feature_match(
        self,
        x_train_scaled: np.ndarray,
        x_val_scaled: np.ndarray,
        warm_start_feature_names: list,
        common_features: set,
    ) -> Tuple[np.ndarray, np.ndarray, bool]:
        """Apply partial feature match when 80%+ overlap exists."""
        from sklearn.preprocessing import StandardScaler

        saved_features_set = set(warm_start_feature_names)
        missing_features = saved_features_set - common_features
        logger.warning(
            f"⚠️ Feature dimension mismatch: saved model has {len(warm_start_feature_names)} features, "
            f"but only {len(common_features)} available. Missing: {list(missing_features)[:5]}..."
        )
        logger.warning("⚠️ Disabling warm-start weight loading (incompatible architecture)")

        selected_indices = [
            self.feature_names.index(f)
            for f in warm_start_feature_names
            if f in self.feature_names
        ]

        x_train_scaled = x_train_scaled[:, selected_indices]
        x_val_scaled = x_val_scaled[:, selected_indices]

        self.selected_indices = selected_indices  # Store for inference/validation
        self.feature_names = [self.feature_names[i] for i in selected_indices]
        self.n_features = len(selected_indices)

        if self.scaler is not None:
            self.scaler = StandardScaler()
            x_train_scaled = self.scaler.fit_transform(x_train_scaled)
            x_val_scaled = self.scaler.transform(x_val_scaled)

        logger.info(
            f"🔥 WARM-START: Using {len(selected_indices)} available features (fresh model, no weight transfer)"
        )
        return x_train_scaled, x_val_scaled, False

    def _apply_feature_selection(
        self,
        x_train_scaled: np.ndarray,
        x_val_scaled: np.ndarray,
        y_train: np.ndarray,
        top_k_features: int,
        method: str = "random_forest",
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Apply feature selection using RF importance or F-test."""
        logger.info(
            f"🔍 Feature Selection: Reducing from {x_train_scaled.shape[-1]} to "
            f"top {top_k_features} features using {method}"
        )

        x_train_flat = x_train_scaled
        x_val_flat = x_val_scaled
        y_train_flat = np.tile(y_train, (x_train_scaled.shape[0], 1)).flatten()[
            : len(x_train_flat)
        ]

        if method == "random_forest":
            selected_indices, x_train_selected, x_val_selected = \
                self._rf_feature_selection(x_train_flat, x_val_flat, y_train_flat, top_k_features)
        else:
            selected_indices, x_train_selected, x_val_selected = \
                self._ftest_feature_selection(x_train_flat, x_val_flat, y_train_flat, top_k_features)

        self.selected_indices = selected_indices
        self._update_features_after_selection(selected_indices, x_train_selected)

        return x_train_selected, x_val_selected

    def _rf_feature_selection(
        self,
        x_train_flat: np.ndarray,
        x_val_flat: np.ndarray,
        y_train_flat: np.ndarray,
        top_k: int,
    ) -> Tuple[list, np.ndarray, np.ndarray]:
        """Apply Random Forest importance-based feature selection."""
        from src.data.feature_engineering import FeatureEngineering
        import pandas as pd

        feature_cols = (
            self.feature_names
            or [f"feat_{i}" for i in range(x_train_flat.shape[-1])]
        )
        df_train = pd.DataFrame(x_train_flat, columns=feature_cols)
        df_train["_target_"] = y_train_flat

        fe = FeatureEngineering()
        _, selected_features = fe.select_features(
            df_train,
            target_col="_target_",
            method="random_forest",
            top_k=top_k,
        )

        selected_indices = [
            feature_cols.index(f) for f in selected_features if f in feature_cols
        ]
        x_train_selected = x_train_flat[:, selected_indices]
        x_val_selected = x_val_flat[:, selected_indices]

        logger.info(f"🌲 RF Feature Selection: {len(selected_indices)} features selected")
        logger.info(f"   Top features: {selected_features[:5]}")

        return selected_indices, x_train_selected, x_val_selected

    def _ftest_feature_selection(
        self,
        x_train_flat: np.ndarray,
        x_val_flat: np.ndarray,
        y_train_flat: np.ndarray,
        top_k: int,
    ) -> Tuple[list, np.ndarray, np.ndarray]:
        """Apply F-test based feature selection."""
        from sklearn.feature_selection import SelectKBest, f_classif

        selector = SelectKBest(score_func=f_classif, k=top_k)
        x_train_selected = selector.fit_transform(x_train_flat, y_train_flat)
        x_val_selected = selector.transform(x_val_flat)

        selected_indices = list(selector.get_support(indices=True))
        logger.info(f"✓ F-test Selection: {len(selected_indices)} features selected")

        return selected_indices, x_train_selected, x_val_selected

    def _update_features_after_selection(
        self, selected_indices: list, x_train_selected: np.ndarray
    ) -> None:
        """Update feature names and scaler after selection."""
        from sklearn.preprocessing import StandardScaler

        if self.feature_names is not None:
            self.feature_names = [self.feature_names[i] for i in selected_indices]
            logger.info(
                f"✓ Updated feature names: {len(self.feature_names)} features selected"
            )

        self.n_features = len(selected_indices)

        if self.scaler is not None:
            self.scaler = StandardScaler()
            # Re-fitting will be done by caller
            logger.info(
                f"✓ Scaler will be re-fitted on {x_train_selected.shape[-1]} selected features"
            )

    def _compute_class_statistics(
        self, y_train_filtered: np.ndarray, y_val_filtered: np.ndarray
    ) -> Tuple[float, float, float, float]:
        """Compute class distribution and imbalance ratios."""
        train_up_pct = (y_train_filtered == 1).mean() * 100
        val_up_pct = (y_val_filtered == 1).mean() * 100

        train_imbalance = (
            max(train_up_pct, 100 - train_up_pct)
            / min(train_up_pct, 100 - train_up_pct)
            if min(train_up_pct, 100 - train_up_pct) > 0
            else float("inf")
        )
        val_imbalance = (
            max(val_up_pct, 100 - val_up_pct) / min(val_up_pct, 100 - val_up_pct)
            if min(val_up_pct, 100 - val_up_pct) > 0
            else float("inf")
        )

        logger.info(
            "Class distribution: train=%.1f%% up (imbalance=%.2fx), val=%.1f%% up (imbalance=%.2fx)",
            train_up_pct, train_imbalance, val_up_pct, val_imbalance,
        )
        if train_imbalance > 2.0 or val_imbalance > 2.0:
            logger.warning(
                "High class imbalance detected (>2x). Consider adjusting direction_threshold."
            )

        return train_up_pct, val_up_pct, train_imbalance, val_imbalance

    def _compute_sample_weights(
        self, y_train_filtered: np.ndarray, max_weight: float = 3.0
    ) -> np.ndarray:
        """Compute sample weights for class balancing."""
        n_samples = len(y_train_filtered)
        n_up = int((y_train_filtered == 1).sum())
        n_down = n_samples - n_up

        if n_up > 0 and n_down > 0:
            up_weight = min(n_samples / (2 * n_up), max_weight)
            down_weight = min(n_samples / (2 * n_down), max_weight)
            sample_weights = np.where(y_train_filtered == 1, up_weight, down_weight)
            sample_weights = sample_weights / sample_weights.mean()
            logger.info(
                f"📊 Sample weights: UP={up_weight:.2f}x, DOWN={down_weight:.2f}x (cap={max_weight}x)"
            )
            logger.info(f"🎯 Using AntiCollapseFocalLoss (gamma={self.config.focal_gamma}, alpha={self.config.focal_alpha}, entropy_weight=0.1)")
        else:
            sample_weights = np.ones(n_samples)
            logger.warning("⚠️ Single class in training data - using uniform weights")

        return sample_weights

    def _handle_replay_buffer(
        self,
        x_train_filtered: np.ndarray,
        y_train_filtered: np.ndarray,
        sample_weights: np.ndarray,
        instrument: str,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Load and mix replay buffer samples with training data."""
        self.replay_buffer = ReplayBuffer(
            capacity_ratio=self.config.replay_buffer_ratio,
            mix_ratio=self.config.replay_mix_ratio,
            buffer_dir=self.config.replay_buffer_dir,
        )

        if self.replay_buffer.load(instrument):
            x_replay, y_replay, w_replay = self.replay_buffer.get_replay_samples(
                len(x_train_filtered)
            )

            if x_replay is not None and len(x_replay) > 0:
                buffer_n_features = x_replay.shape[2]
                if buffer_n_features == x_train_filtered.shape[2]:
                    x_train_filtered = np.vstack([x_train_filtered, x_replay])
                    y_train_filtered = np.concatenate([y_train_filtered, y_replay])
                    if sample_weights is not None and w_replay is not None:
                        sample_weights = np.concatenate([sample_weights, w_replay])
                    logger.info(
                        f"📦 Mixed {len(x_replay)} replay samples, new train size: {len(x_train_filtered)}"
                    )
                else:
                    logger.warning(
                        f"⚠️ Replay buffer feature mismatch: buffer has {buffer_n_features} features, "
                        f"current data has {x_train_filtered.shape[2]}. Clearing buffer."
                    )
                    self.replay_buffer.clear()

        # Always add current training data to buffer for future sessions
        self.replay_buffer.add_samples(
            x_train_filtered,
            y_train_filtered,
            sample_weights,
            data_id=f"{instrument}_{self.lineage.checkpoint_id}",
        )

        return x_train_filtered, y_train_filtered, sample_weights

    def _create_loss_function(
        self, auto_variance_weight: float, label_smoothing: float
    ) -> Any:
        """Create appropriate loss function based on config.

        Returns the base loss function (before EWC wrapping).
        Tries loss types in priority order: MADL > Hybrid CB > Anti-Collapse > CB Focal > Focal > BCE.
        """
        # Priority list of loss function attempts
        loss_attempts = [
            (getattr(self.config, "use_madl_loss", False) if self.config else False,
             lambda: self._try_madl_loss(label_smoothing)),
            (self.config and getattr(self.config, "use_hybrid_cb_anticollapse", True),
             lambda: self._try_hybrid_cb_loss(auto_variance_weight, label_smoothing)),
            (self.config and getattr(self.config, "use_anti_collapse_loss", True),
             lambda: self._try_anti_collapse_loss(auto_variance_weight)),
            (self.config and self.config.use_class_balanced_loss,
             lambda: self._try_cb_focal_loss(label_smoothing)),
        ]

        for should_try, try_func in loss_attempts:
            if should_try:
                loss = try_func()
                if loss is not None:
                    return loss

        # Fallback to standard Focal Loss or BCE
        return self._get_fallback_loss(label_smoothing)

    def _try_madl_loss(self, label_smoothing: float) -> Optional[Any]:
        """Try to create MADL Loss."""
        try:
            from src.models.tensorflow_models import MADLLoss
            madl_direction_weight = (
                getattr(self.config, "madl_direction_weight", 0.7) if self.config else 0.7
            )
            logger.info(
                f"💰 Using MADL Loss for directional profitability "
                f"(direction_weight={madl_direction_weight}, label_smoothing={label_smoothing})"
            )
            return MADLLoss(direction_weight=madl_direction_weight, label_smoothing=label_smoothing)
        except ImportError:
            logger.warning("⚠️ MADLLoss not found, falling back to Class-Balanced Focal Loss")
            return None

    def _try_hybrid_cb_loss(
        self, auto_variance_weight: float, label_smoothing: float
    ) -> Optional[Any]:
        """Try to create Hybrid CB + Anti-Collapse Loss."""
        try:
            from src.models.tensorflow_models import HybridClassBalancedAntiCollapseLoss
            cb_beta = getattr(self.config, "cb_beta", 0.9999)
            cb_gamma = getattr(self.config, "cb_gamma", 2.0)
            variance_weight = auto_variance_weight * 2.5
            variance_target = 0.12
            logger.info(
                f"🔥 Using HybridClassBalancedAntiCollapseLoss "
                f"(beta={cb_beta}, gamma={cb_gamma}, "
                f"variance_weight={variance_weight:.3f}, variance_target={variance_target})"
            )
            return HybridClassBalancedAntiCollapseLoss(
                gamma=cb_gamma, beta=cb_beta,
                variance_weight=variance_weight, variance_target=variance_target,
                label_smoothing=label_smoothing,
            )
        except Exception as e:
            logger.warning(f"⚠️ HybridClassBalancedAntiCollapseLoss failed: {e}")
            return None

    def _try_anti_collapse_loss(
        self, auto_variance_weight: float
    ) -> Optional[Any]:
        """Try to create Anti-Collapse Focal Loss."""
        try:
            from src.models.tensorflow_models import AntiCollapseFocalLoss
            focal_gamma = getattr(self.config, "focal_gamma", 2.0) if self.config else 2.0
            ac_label_smoothing = (
                getattr(self.config, "anti_collapse_label_smoothing", 0.05)
                if self.config else 0.05
            )
            logger.info(
                f"🛡️ Using AntiCollapseFocalLoss (gamma={focal_gamma}, "
                f"variance_weight={auto_variance_weight:.3f}, label_smoothing={ac_label_smoothing})"
            )
            return AntiCollapseFocalLoss(
                gamma=focal_gamma, base_alpha=0.5,
                entropy_weight=auto_variance_weight, label_smoothing=ac_label_smoothing,
            )
        except Exception as e:
            logger.warning(f"⚠️ AntiCollapseFocalLoss failed: {e}")
            return None

    def _try_cb_focal_loss(self, label_smoothing: float) -> Optional[Any]:
        """Try to create Class-Balanced Focal Loss."""
        try:
            from src.models.tensorflow_models import ClassBalancedFocalLoss
            cb_beta = getattr(self.config, "cb_beta", 0.9999) if self.config else 0.9999
            cb_gamma = getattr(self.config, "cb_gamma", 2.0) if self.config else 2.0
            logger.info(
                f"🎯 Using Class-Balanced Focal Loss "
                f"(beta={cb_beta}, gamma={cb_gamma}, label_smoothing={label_smoothing})"
            )
            return ClassBalancedFocalLoss(beta=cb_beta, gamma=cb_gamma, label_smoothing=label_smoothing)
        except ImportError:
            logger.warning("⚠️ ClassBalancedFocalLoss not found, falling back to Focal Loss")
            return None

    def _get_fallback_loss(self, label_smoothing: float) -> Any:
        """Get fallback loss function (Focal or BCE)."""
        from tensorflow import keras
        try:
            from src.models.tensorflow_models import BinaryFocalLoss
            logger.info(
                f"🎯 Using Focal Loss (gamma=2.0, alpha=0.5, label_smoothing={label_smoothing})"
            )
            return BinaryFocalLoss(gamma=2.0, alpha=0.5, label_smoothing=label_smoothing)
        except ImportError:
            logger.warning("⚠️ BinaryFocalLoss not found, falling back to BinaryCrossentropy")
            return keras.losses.BinaryCrossentropy(label_smoothing=label_smoothing)

    def _load_warm_start_weights(self, warm_start_path: str) -> bool:
        """Try to load warm-start weights using multiple strategies.

        Returns True if weights were successfully loaded.
        """
        logger.info(f"🔥 WARM-START: Loading weights from {warm_start_path}")

        # Strategy 1: Try .weights.h5 companion file or direct load_weights
        if self._try_load_weights_h5(warm_start_path):
            return True

        # Strategy 2: Try loading full model and extracting weights
        if self._try_load_full_model_weights(warm_start_path):
            return True

        # Strategy 3: Try loading from meta.pkl
        return bool(self._try_load_meta_weights(warm_start_path))

    def _try_load_weights_h5(self, warm_start_path: str) -> bool:
        """Strategy 1: Try loading weights from .h5 file.

        Uses _safe_load_weights_ignoring_optimizer to suppress optimizer-state
        mismatch warnings that are benign during warm-start (the optimizer is
        rebuilt with a fresh Adam before training begins).
        """
        weights_h5_path = Path(warm_start_path).with_suffix(WEIGHTS_H5_SUFFIX)
        if weights_h5_path.exists():
            target = str(weights_h5_path)
        else:
            target = str(warm_start_path)

        if _safe_load_weights_ignoring_optimizer(self.model, target):
            logger.info(f"✓ Loaded model weights from {target}")
            # Reset optimizer slot variables – they belong to the old architecture
            _safe_reset_optimizer_state(self.model)
            return True
        return False

    def _transfer_weights_by_name(
        self, ckpt_weight_map: Dict[str, np.ndarray]
    ) -> Tuple[int, int]:
        """Transfer weights from checkpoint to model by matching layer names.

        Returns:
            Tuple of (loaded_count, skipped_count).
        """
        loaded, skipped = 0, 0
        for w in self.model.weights:
            if w.name in ckpt_weight_map:
                ckpt_val = ckpt_weight_map[w.name]
                if w.shape == ckpt_val.shape:
                    w.assign(ckpt_val.astype(_get_numpy_dtype(w.dtype)))
                    loaded += 1
                else:
                    skipped += 1
                    logger.debug(
                        f"  Shape mismatch for {w.name}: "
                        f"model={w.shape}, checkpoint={ckpt_val.shape}"
                    )
            else:
                matched = self._try_partial_name_match(w, ckpt_weight_map)
                if matched:
                    loaded += 1
                else:
                    skipped += 1
        return loaded, skipped

    def _try_partial_name_match(
        self, weight, ckpt_weight_map: Dict[str, np.ndarray]
    ) -> bool:
        """Try matching a model weight to checkpoint by base name."""
        base = weight.name.split("/")[-1].split(":")[0]
        for cn, cv in ckpt_weight_map.items():
            cb = cn.split("/")[-1].split(":")[0]
            if base == cb and weight.shape == cv.shape:
                weight.assign(cv.astype(_get_numpy_dtype(weight.dtype)))
                return True
        return False

    def _try_load_full_model_weights(self, warm_start_path: str) -> bool:
        """Strategy 2: Try loading full model and extracting weights.

        When architectures differ (e.g. different layer count or frozen layers),
        falls back to *name-based partial weight transfer* so that compatible
        layers still benefit from the checkpoint.
        """
        from tensorflow import keras

        try:
            existing_model = keras.models.load_model(warm_start_path, compile=False)
        except Exception as e:
            logger.debug(f"Full model load failed: {e}")
            return False

        try:
            checkpoint_weights = existing_model.get_weights()
            model_weights = self.model.get_weights()

            is_compatible, shape_error = _validate_weight_shapes(
                model_weights, checkpoint_weights, context="model weights"
            )

            if is_compatible:
                self.model.set_weights(checkpoint_weights)
                logger.info(WEIGHTS_LOADED_FULL_MODEL_MSG)
                return True

            # --- Partial weight transfer via name-based matching ---
            logger.warning(f"Architecture mismatch: {shape_error}")
            logger.info("🔄 Attempting partial weight transfer by layer name...")

            ckpt_weight_map = {
                w.name: w.numpy() for w in existing_model.weights
            }
            loaded, skipped = self._transfer_weights_by_name(ckpt_weight_map)

            if loaded > 0:
                logger.info(
                    f"✓ Partial weight transfer: {loaded} weights loaded, "
                    f"{skipped} re-initialized from scratch"
                )
                _safe_reset_optimizer_state(self.model)
                return True

            logger.warning("⚠️ No compatible weights found – starting fresh")
            return False
        finally:
            del existing_model

    def _try_load_meta_weights(self, warm_start_path: str) -> bool:
        """Strategy 3: Try loading weights from meta.pkl."""
        meta_path = Path(warm_start_path).with_suffix(META_PKL_SUFFIX)
        if not meta_path.exists():
            return False

        try:
            with open(meta_path, "rb") as f:
                meta = pickle.load(f)
            if "model_weights" in meta:
                self.model.set_weights(meta["model_weights"])
                logger.info("✓ Loaded weights from meta.pkl")
                return True
            return False
        except Exception as e:
            logger.debug(f"Meta weights load failed: {e}")
            return False

    def _freeze_encoder_layers(self) -> Tuple[int, list]:
        """Freeze encoder layers for warm-start training.

        Returns (frozen_count, trainable_head_layers).
        """
        from tensorflow import keras

        frozen_count = 0
        trainable_head_layers = []

        encoder_patterns = [
            "transformer_", "input_projection", "positional", "multi_head",
            "attention", "ffn", "layer_norm", "spatial_dropout",
            "gaussian_noise", "global_average",
        ]

        for layer in self.model.layers:
            layer_name = layer.name.lower()
            is_encoder_layer = any(pattern in layer_name for pattern in encoder_patterns)

            # Check if this is the classification head
            try:
                output_units = (
                    layer.output.shape[-1] if hasattr(layer, "output")
                    else getattr(layer, "units", 32)
                )
            except (AttributeError, TypeError):
                output_units = getattr(layer, "units", 32)

            is_classification_head = (
                "direction" in layer_name or
                (isinstance(layer, keras.layers.Dense) and output_units <= 16
                 and "projection" not in layer_name)
            )

            if is_encoder_layer and not is_classification_head:
                layer.trainable = False
                frozen_count += 1
            elif layer.trainable:
                trainable_head_layers.append(layer.name)

        return frozen_count, trainable_head_layers

    def _log_frozen_layers(self, frozen_count: int, trainable_head_layers: list) -> None:
        """Log information about frozen layers."""
        import tensorflow as tf

        if frozen_count > 0:
            logger.info(f"🔒 WARM-START: Froze {frozen_count} encoder layers")
            trainable_params = sum(
                [tf.size(w).numpy() for w in self.model.trainable_weights]
            )
            total_params = self.model.count_params()
            pct = 100 * trainable_params / total_params
            logger.info(f"   Trainable: {trainable_params:,}/{total_params:,} params ({pct:.1f}%)")
            if trainable_head_layers:
                suffix = "..." if len(trainable_head_layers) > 5 else ""
                logger.info(f"   Trainable layers: {trainable_head_layers[:5]}{suffix}")
        else:
            logger.warning("⚠️ WARM-START: No layers frozen! This may cause catastrophic forgetting.")

    def _handle_warm_start(self, warm_start_path: str) -> float:
        """Handle warm-start loading: weights, layer freezing, lineage, EWC, EMA.

        Returns the effective learning rate to use.
        """
        try:
            weights_loaded = self._load_warm_start_weights(warm_start_path)

            if weights_loaded:
                self._is_warm_start = True
                self._warm_start_weights = self.model.get_weights()
                logger.info(f"✓ Successfully loaded {self.model.count_params():,} parameters from checkpoint")

                # Freeze encoder layers if configured
                if self.config.warm_start_freeze_encoder:
                    frozen_count, trainable_head_layers = self._freeze_encoder_layers()
                    self._log_frozen_layers(frozen_count, trainable_head_layers)

                # Load metadata (lineage, metrics, training state)
                self._load_warm_start_metadata(warm_start_path)

                # Load EWC state
                self._load_warm_start_ewc(warm_start_path)

                # Load EMA weights
                self._load_warm_start_ema(warm_start_path)

                # Compute effective learning rate
                effective_lr = self.config.learning_rate * self.config.warm_start_lr_factor
                logger.info(
                    f"🔥 Warm-start LR reduction: {self.config.learning_rate} → "
                    f"{effective_lr} (factor={self.config.warm_start_lr_factor})"
                )
                return effective_lr
            else:
                logger.warning("Could not load warm-start weights. Starting fresh.")
                return self.config.learning_rate
        except Exception as e:
            logger.warning(f"Could not load warm-start checkpoint: {e}. Starting fresh.")
            return self.config.learning_rate

    def _load_warm_start_metadata(self, warm_start_path: str) -> None:
        """Load lineage, metrics, and training state from warm-start checkpoint."""
        meta_path = Path(warm_start_path).with_suffix(META_PKL_SUFFIX)
        if not meta_path.exists():
            return

        with open(meta_path, "rb") as f:
            meta = pickle.load(f)

        # Load lineage
        if "lineage" in meta:
            parent_lineage = TrainingLineage.from_dict(meta["lineage"])
            self.lineage.parent_checkpoint_id = parent_lineage.checkpoint_id
            self.lineage.cumulative_epochs = parent_lineage.cumulative_epochs
            self.lineage.cumulative_samples = parent_lineage.cumulative_samples
            self.lineage.metric_history = parent_lineage.metric_history.copy()
            self._loaded_model_instrument = parent_lineage.instrument
            logger.info(
                f"📊 Loaded lineage from parent: {parent_lineage.checkpoint_id} "
                f"(cumulative epochs: {self.lineage.cumulative_epochs}, "
                f"instrument: {self._loaded_model_instrument})"
            )

            # Restore dynamic training state
            self._restored_variance_weight = getattr(parent_lineage, "auto_variance_weight", 0.0)
            self._restored_lr_reductions = getattr(parent_lineage, "lr_reductions_count", 0)
            self._restored_lr = getattr(parent_lineage, "final_learning_rate", 0.0)
            if self._restored_variance_weight > 0 or self._restored_lr_reductions > 0:
                logger.info(
                    f"🔄 Restored training state: variance_weight={self._restored_variance_weight:.3f}, "
                    f"lr_reductions={self._restored_lr_reductions}, last_lr={self._restored_lr:.2e}"
                )

        # Load previous best accuracy
        prev_metrics = meta.get("metrics", {})
        self._warm_start_val_acc = prev_metrics.get("val_accuracy", 0.0)
        if self._warm_start_val_acc > 0:
            logger.info(f"🎯 Previous best val_accuracy: {self._warm_start_val_acc:.1%} (will not save worse)")

    def _load_warm_start_ewc(self, warm_start_path: str) -> None:
        """Load EWC Fisher information from previous training."""
        if not self._use_ewc:
            return
        ewc_path = Path(warm_start_path).with_suffix(EWC_PKL_SUFFIX)
        self.ewc = EWCPenalty(
            self.model,
            ewc_lambda=self.config.ewc_lambda,
            gamma=self.config.ewc_gamma,
        )
        if self.ewc.load(str(ewc_path)):
            self.lineage.ewc_n_tasks = self.ewc._n_tasks
            logger.info(f"🧠 EWC loaded: {self.ewc._n_tasks} prior task(s) will be protected")

    def _load_warm_start_ema(self, warm_start_path: str) -> None:
        """Load EMA weights from previous training."""
        if not self._use_ema:
            return
        ema_meta_path = Path(warm_start_path).with_suffix(EMA_PKL_SUFFIX)
        if ema_meta_path.exists():
            with open(ema_meta_path, "rb") as f:
                ema_data = pickle.load(f)

            # Quick check: if tensor count differs drastically, skip loading
            # and remove the stale file so it won't warn on future runs.
            n_model = len(self.model.trainable_weights)
            n_ckpt = len(ema_data.get("ema_weights", []))
            ratio = max(n_model, n_ckpt) / max(min(n_model, n_ckpt), 1)
            if ratio > 3.0:
                logger.info(
                    f"📊 Stale EMA checkpoint ({n_ckpt} tensors) doesn't match "
                    f"current model ({n_model} tensors) — starting fresh EMA."
                )
                try:
                    ema_meta_path.unlink()
                except OSError:
                    pass
                return

            self.ema = EMACallback(
                self.model,
                decay=self.config.ema_decay,
                update_every=self.config.ema_update_every,
            )
            self.ema.set_ema_weights(
                ema_data["ema_weights"],
                weight_names=ema_data.get("ema_weight_names"),
            )
            logger.info("📊 EMA weights loaded from checkpoint")

    def _prepare_sequences_and_filter(
        self,
        x_train_scaled: np.ndarray,
        x_val_scaled: np.ndarray,
        y_train: np.ndarray,
        y_val: np.ndarray,
        w_train: Optional[np.ndarray],
        w_val: Optional[np.ndarray],
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Create sequences and filter by clear labels."""
        seq_len = get_config_seq_len(self.config)
        self.seq_len = seq_len

        x_train_seq, y_train_seq, w_train_seq = create_sequences_with_weights(
            x_train_scaled, y_train, w_train, seq_len
        )
        x_val_seq, y_val_seq, w_val_seq = create_sequences_with_weights(
            x_val_scaled, y_val, w_val, seq_len
        )

        train_clear_mask = w_train_seq > 0
        val_clear_mask = w_val_seq > 0

        x_train_filtered = x_train_seq[train_clear_mask]
        y_train_filtered = y_train_seq[train_clear_mask]
        x_val_filtered = x_val_seq[val_clear_mask]
        y_val_filtered = y_val_seq[val_clear_mask]

        logger.info(
            f"Filtered training: {train_clear_mask.sum()}/{len(train_clear_mask)} sequences with clear labels"
        )
        logger.info(
            f"Filtered validation: {val_clear_mask.sum()}/{len(val_clear_mask)} sequences with clear labels"
        )

        return x_train_filtered, y_train_filtered, x_val_filtered, y_val_filtered

    def _compute_auto_variance_weight(
        self, instrument: str, train_up_pct: float
    ) -> float:
        """Compute auto-tuned variance weight for loss function."""
        class_ratio = train_up_pct / 100.0
        if hasattr(self, "_restored_variance_weight") and self._restored_variance_weight > 0:
            auto_variance_weight = self._restored_variance_weight
            logger.info(
                f"🔄 Using restored variance_weight={auto_variance_weight:.3f} from warm-start"
            )
        else:
            auto_variance_weight = compute_auto_variance_weight(
                instrument=instrument,
                class_ratio=class_ratio,
                base_weight=self.config.anti_collapse_base_variance_weight if self.config else 0.1,
            )
            logger.info(
                f"🎯 Auto-tuned variance_weight={auto_variance_weight:.3f} "
                f"for {instrument} (class_ratio={class_ratio:.2f})"
            )
        return auto_variance_weight

    def _setup_optimizer_with_warmup(
        self, effective_lr: float, x_train_filtered: np.ndarray
    ) -> Any:
        """Setup optimizer with warmup learning rate schedule."""
        from tensorflow import keras

        warmup_epochs = getattr(self.config, "warmup_epochs", 5) if self.config else 5
        steps_per_epoch = max(1, len(x_train_filtered) // self.config.batch_size)
        total_steps = self.config.epochs * steps_per_epoch
        warmup_steps = warmup_epochs * steps_per_epoch

        try:
            from src.training.m1_metal_optimizer import WarmupCosineDecaySchedule
            lr_schedule = WarmupCosineDecaySchedule(
                initial_learning_rate=effective_lr * 0.1,
                warmup_steps=warmup_steps,
                decay_steps=total_steps - warmup_steps,
                min_learning_rate=1e-6,
                warmup_target=effective_lr,
            )
            optimizer = keras.optimizers.Adam(learning_rate=lr_schedule)
            logger.info(
                f"🔥 Warmup LR: {warmup_epochs} epochs ({warmup_steps} steps), target LR={effective_lr:.6f}"
            )
        except ImportError:
            logger.warning("⚠️ WarmupCosineDecaySchedule not available, using constant LR")
            optimizer = keras.optimizers.Adam(learning_rate=effective_lr)

        return optimizer

    def _compile_model_with_loss(
        self,
        optimizer: Any,
        base_loss: Any,
        instrument: str,
    ) -> None:
        """Compile model with appropriate loss function."""
        use_ewc_loss = (
            self._is_warm_start
            and self._use_ewc
            and self.ewc is not None
            and self.ewc.fisher_diagonal is not None
            and getattr(self, "_loaded_model_instrument", None) != instrument
        )

        if use_ewc_loss:
            logger.info(
                f"🧠 EWC loss enabled (cross-pair): λ={self.ewc.ewc_lambda}, "
                f"protecting {self.ewc._n_tasks} prior task(s)"
            )
            ewc_loss = create_ewc_loss(base_loss, self.ewc.penalty, ewc_weight=1.0)
            self.model.compile(optimizer=optimizer, loss=ewc_loss, metrics=["accuracy"])
        else:
            if self._is_warm_start and self._use_ewc:
                logger.info("🧠 EWC disabled for same-pair warm-start (prevents over-regularization)")
            self.model.compile(optimizer=optimizer, loss=base_loss, metrics=["accuracy"])

    def _evaluate_warm_start_baseline(
        self, x_val_filtered: np.ndarray, y_val_filtered: np.ndarray, instrument: str
    ) -> None:
        """Evaluate and set warm-start baseline accuracy."""
        if not self._is_warm_start:
            return

        try:
            loaded_model_instrument = getattr(self, "_loaded_model_instrument", None)
            is_cross_pair = loaded_model_instrument and loaded_model_instrument != instrument

            eval_results = self.model.evaluate(x_val_filtered, y_val_filtered, verbose=0)
            actual_baseline_acc = eval_results[1] if len(eval_results) > 1 else eval_results[0]

            if is_cross_pair:
                logger.info(f"🔄 Cross-pair training ({loaded_model_instrument} → {instrument})")
                logger.info(
                    f"   Stored baseline: {self._warm_start_val_acc:.1%}, "
                    f"Actual on {instrument}: {actual_baseline_acc:.1%}"
                )
                self._warm_start_val_acc = actual_baseline_acc
                logger.info(f"🎯 Using actual baseline on {instrument}: {self._warm_start_val_acc:.1%}")
            else:
                logger.info(f"🔄 Same-pair training ({instrument})")
                logger.info(
                    f"   Stored baseline: {self._warm_start_val_acc:.1%}, Current eval: {actual_baseline_acc:.1%}"
                )
                if self._warm_start_val_acc > 0:
                    logger.info(f"🎯 Using STORED baseline: {self._warm_start_val_acc:.1%} (ignoring data drift)")
                else:
                    self._warm_start_val_acc = actual_baseline_acc
                    logger.info(f"🎯 No stored baseline, using actual: {self._warm_start_val_acc:.1%}")
        except Exception as e:
            logger.warning(f"Could not evaluate baseline: {e}")

    def _initialize_ema(self) -> None:
        """Initialize EMA callback if not already loaded."""
        if self._use_ema and self.ema is None:
            self.ema = EMACallback(
                self.model,
                decay=self.config.ema_decay,
                update_every=self.config.ema_update_every,
            )
            self.lineage.ema_enabled = True

    def _create_training_callbacks(
        self,
        x_val_filtered: np.ndarray,
        y_val_filtered: np.ndarray,
    ) -> list:
        """Create all training callbacks."""
        from tensorflow import keras

        early_stop_patience = (
            self.config.patience // 2 if self._is_warm_start else self.config.patience
        )
        lr_reduce_patience = (
            max(5, self.config.patience // 4) if self._is_warm_start
            else max(4, self.config.patience // 4)
        )

        if self._is_warm_start:
            logger.info("📊 Warm-start callback adjustments:")
            logger.info(f"   Early stopping patience: {early_stop_patience} (reduced from {self.config.patience})")
            logger.info(f"   LR reduction patience: {lr_reduce_patience}")

        callbacks = [
            self._create_epoch_callback(),
            keras.callbacks.EarlyStopping(
                monitor="val_accuracy",
                patience=early_stop_patience,
                mode="max",
                restore_best_weights=True,
                verbose=0,
            ),
            OverfitPreventionCallback(
                checkpoint_dir=self.config.checkpoint_dir,
                model_name="transformer_direction",
                config=OverfitPreventionConfig(
                    overfit_threshold=0.08,
                    critical_threshold=0.12,
                    severe_threshold=0.20,
                    max_acceptable_gap=0.10,
                    patience_epochs=2,
                    auto_adjust_dropout=True,
                    auto_reduce_lr=True,
                    enable_swa=True,
                    swa_start_fraction=0.5,
                    enable_cosine_restarts=True,
                    restart_period=15,
                    warm_start_best_acc=self._warm_start_val_acc,
                ),
            ),
        ]

        # Add collapse detection callbacks
        callbacks.extend(self._create_collapse_callbacks(x_val_filtered, y_val_filtered))

        # Add gradual unfreeze callback for warm-start
        self._add_unfreeze_callback(callbacks)

        # Add EMA and EWC callbacks
        self._add_ema_ewc_callbacks(callbacks)

        return callbacks

    def _create_epoch_callback(self) -> Any:
        """Create appropriate epoch display callback."""
        if not self.config.quiet:
            return RichEpochCallback(
                model_name="Transformer Direction",
                total_epochs=self.config.epochs,
                warm_start_best_acc=self._warm_start_val_acc,
                quiet=self.config.quiet,
            )
        return QuietProgressCallback(
            model_name="Transformer Direction",
            total_epochs=self.config.epochs,
        )

    def _create_collapse_callbacks(
        self, x_val_filtered: np.ndarray, y_val_filtered: np.ndarray
    ) -> list:
        """Create collapse detection and prevention callbacks."""

        # Define callbacks inline to avoid class definition complexity
        self._collapse_callback = self._make_prediction_collapse_callback(
            x_val_filtered, y_val_filtered
        )
        self._proactive_collapse_callback = self._make_proactive_collapse_callback()

        return [self._collapse_callback, self._proactive_collapse_callback]

    def _make_prediction_collapse_callback(
        self, x_val: np.ndarray, y_val: np.ndarray
    ) -> Any:
        """Create prediction collapse detection callback."""
        initial_lr_reductions = getattr(self, "_restored_lr_reductions", 0)
        return _PredictionCollapseCallback(x_val, y_val, initial_lr_reductions)

    def _make_proactive_collapse_callback(self) -> Any:
        """Create proactive collapse prevention callback."""
        from tensorflow import keras

        class ProactiveCollapsePreventionCallback(keras.callbacks.Callback):
            def __init__(self):
                super().__init__()
                self.check_every = 50
                self.bias_threshold = 1.5
                self.batch_count = 0
                self.interventions = 0

            def on_train_batch_end(self, batch, logs=None):
                self.batch_count += 1
                if self.batch_count % self.check_every != 0:
                    return
                self._check_bias()

            def _check_bias(self):
                try:
                    output_layer = next(
                        (layer for layer in self.model.layers if layer.name == "direction"), None
                    )
                    if output_layer is None or not hasattr(output_layer, 'bias'):
                        return

                    bias_val = float(output_layer.bias.numpy()[0])
                    if abs(bias_val) > self.bias_threshold:
                        self._intervene(output_layer, bias_val)
                except Exception:
                    pass

            def _intervene(self, output_layer, bias_val):
                self.interventions += 1
                new_bias = bias_val * 0.3
                output_layer.bias.assign([new_bias])

                kernel = output_layer.kernel.numpy()
                rng = np.random.default_rng(seed=42)
                noise = rng.normal(0, 0.01, kernel.shape)
                output_layer.kernel.assign(kernel + noise)

                logger.warning(
                    f"🔧 Proactive collapse intervention #{self.interventions}: "
                    f"bias {bias_val:.3f} → {new_bias:.3f}"
                )

        return ProactiveCollapsePreventionCallback()

    def _add_unfreeze_callback(self, callbacks: list) -> None:
        """Add gradual unfreeze callback for warm-start training."""
        if self._is_warm_start and self.config.warm_start_unfreeze_epochs > 0:
            unfreeze_callback = GradualUnfreezeCallback(
                unfreeze_after_epochs=self.config.warm_start_unfreeze_epochs,
                gradual=self.config.warm_start_gradual_unfreeze,
                learning_rate_boost=1.5,
            )
            callbacks.append(unfreeze_callback)
            logger.info(
                f"🔓 Gradual unfreeze enabled: will unfreeze after epoch "
                f"{self.config.warm_start_unfreeze_epochs}"
            )

    def _add_ema_ewc_callbacks(self, callbacks: list) -> None:
        """Add EMA and EWC monitoring callbacks."""
        from tensorflow import keras

        if self._use_ema and self.ema is not None:
            class EMAUpdateCallback(keras.callbacks.Callback):
                def __init__(self, ema_callback):
                    super().__init__()
                    self.ema_callback = ema_callback

                def on_train_batch_end(self, batch, logs=None):
                    if self.ema_callback:
                        self.ema_callback.update()

            callbacks.append(EMAUpdateCallback(self.ema))

        if (self._is_warm_start and self._use_ewc and self.ewc is not None
                and self.ewc.fisher_diagonal is not None):
            callbacks.append(EWCTrainingCallback(self.ewc, log_every=50))

    def _handle_warm_start_recovery(
        self, x_val_filtered: np.ndarray, y_val_filtered: np.ndarray
    ) -> None:
        """Restore original weights if training degraded."""
        if not (self._is_warm_start and self._warm_start_weights is not None
                and self._warm_start_val_acc > 0):
            return

        current_val_pred = (self.model.predict(x_val_filtered, verbose=0) > 0.5).astype(float)
        current_val_acc = np.mean(current_val_pred.flatten() == y_val_filtered)

        logger.info(
            f"🔍 Post-training check: current_val_acc={current_val_acc:.1%}, "
            f"warm_start_baseline={self._warm_start_val_acc:.1%}"
        )

        if current_val_acc < self._warm_start_val_acc - 0.01:
            from rich.console import Console
            console = Console()
            console.print("  [bold red]⚠️ WARM-START RECOVERY TRIGGERED[/bold red]")
            console.print(f"  [red]   Current: {current_val_acc:.1%} < Baseline: {self._warm_start_val_acc:.1%}[/red]")
            console.print("  [yellow]   Restoring original warm-start weights...[/yellow]")
            self.model.set_weights(self._warm_start_weights)
            console.print(f"  [green]✓ Model preserved at {self._warm_start_val_acc:.1%} accuracy[/green]")

    def _update_ewc_and_ema(
        self, x_train_filtered: np.ndarray, y_train_filtered: np.ndarray
    ) -> None:
        """Update EWC Fisher information and EMA weights."""
        if self._use_ewc:
            if self.ewc is None:
                self.ewc = EWCPenalty(
                    self.model,
                    ewc_lambda=self.config.ewc_lambda,
                    gamma=self.config.ewc_gamma,
                )
            self.ewc.compute_fisher(x_train_filtered, y_train_filtered, n_samples=1000)
            self.lineage.ewc_n_tasks = self.ewc._n_tasks

        if self._use_ema and self.ema is not None:
            self.ema.update(force=True)

    def _update_lineage(
        self, history: Any, x_train_filtered: np.ndarray
    ) -> None:
        """Update training lineage with session info."""
        self.lineage.session_epochs = len(history.history["loss"])
        self.lineage.cumulative_epochs += self.lineage.session_epochs
        self.lineage.cumulative_samples += len(x_train_filtered)
        if self.replay_buffer:
            self.lineage.replay_buffer_size = (
                len(self.replay_buffer.x_buffer) if self.replay_buffer.x_buffer is not None else 0
            )

    def _compute_final_metrics(
        self,
        history: Any,
        x_val_filtered: np.ndarray,
        y_val_filtered: np.ndarray,
        x_train_filtered: np.ndarray,
    ) -> Dict[str, float]:
        """Compute final training metrics."""
        import tensorflow as tf

        val_raw_pred = self.model.predict(x_val_filtered, verbose=0)
        val_pred = (val_raw_pred > 0.5).astype(float)
        val_acc = np.mean(val_pred.flatten() == y_val_filtered)

        y_true = y_val_filtered.flatten()
        y_pred = val_pred.flatten()
        np.mean(y_pred[y_true == 1] == 1) if (y_true == 1).sum() > 0 else 0
        np.mean(y_pred[y_true == 0] == 0) if (y_true == 0).sum() > 0 else 0

        raw_mean = float(np.mean(val_raw_pred))
        raw_std = float(np.std(val_raw_pred))
        raw_median = float(np.median(val_raw_pred))

        self._log_prediction_distribution(val_raw_pred, y_pred, raw_mean, raw_median, raw_std)

        # Calibration
        self.output_calibration = {
            "threshold": raw_median,
            "mean": raw_mean,
            "std": max(raw_std, 0.01),
            "enabled": abs(raw_mean - 0.5) > 0.05,
        }

        # Calibrated metrics
        val_pred_cal = (val_raw_pred.flatten() > raw_median).astype(float)
        up_acc_cal = np.mean(val_pred_cal[y_true == 1] == 1) if (y_true == 1).sum() > 0 else 0
        down_acc_cal = np.mean(val_pred_cal[y_true == 0] == 0) if (y_true == 0).sum() > 0 else 0
        balanced_acc = (up_acc_cal + down_acc_cal) / 2

        self._log_calibrated_distribution(val_pred_cal, raw_median, up_acc_cal, down_acc_cal, balanced_acc)

        metrics = {
            "train_accuracy": float(history.history["accuracy"][-1]),
            "val_accuracy": float(val_acc),
            "val_balanced_accuracy": float(balanced_acc),
            "val_up_accuracy": float(up_acc_cal),
            "val_down_accuracy": float(down_acc_cal),
            "epochs_trained": len(history.history["loss"]),
            "n_train_samples": len(x_train_filtered),
            "n_val_samples": len(x_val_filtered),
        }

        self._add_weight_norm_metrics(metrics, tf)
        return metrics

    def _log_prediction_distribution(
        self, val_raw_pred: np.ndarray, y_pred: np.ndarray,
        raw_mean: float, raw_median: float, raw_std: float
    ) -> None:
        """Log prediction distribution for debugging."""
        long_preds = (y_pred == 1).sum()
        short_preds = (y_pred == 0).sum()
        logger.info("📊 Final validation prediction distribution:")
        logger.info(
            f"   Raw prob: mean={raw_mean:.4f}, median={raw_median:.4f}, "
            f"std={raw_std:.4f}, min={float(np.min(val_raw_pred)):.4f}, "
            f"max={float(np.max(val_raw_pred)):.4f}"
        )
        long_pct = 100 * long_preds / len(y_pred)
        short_pct = 100 * short_preds / len(y_pred)
        logger.info(f"   Predictions (thresh=0.5): LONG={long_preds} ({long_pct:.1f}%), SHORT={short_preds} ({short_pct:.1f}%)")
        if long_preds == 0 or short_preds == 0:
            logger.warning(f"   ⚠️ MODEL COLLAPSE DETECTED: Always predicting {'LONG' if long_preds > 0 else 'SHORT'}!")

    def _log_calibrated_distribution(
        self, val_pred_cal: np.ndarray, threshold: float,
        up_acc: float, down_acc: float, balanced_acc: float
    ) -> None:
        """Log calibrated prediction distribution."""
        long_preds = (val_pred_cal == 1).sum()
        short_preds = (val_pred_cal == 0).sum()
        long_pct = 100 * long_preds / len(val_pred_cal)
        short_pct = 100 * short_preds / len(val_pred_cal)
        logger.info(f"📐 Calibrated (thresh={threshold:.4f}): LONG={long_preds} ({long_pct:.1f}%), SHORT={short_preds} ({short_pct:.1f}%)")
        logger.info(f"📐 Calibrated balanced accuracy: {balanced_acc:.4f} (up={up_acc:.4f}, down={down_acc:.4f})")

    def _add_weight_norm_metrics(self, metrics: Dict[str, float], tf: Any) -> None:
        """Add weight norm metrics for regularization monitoring."""
        try:
            total_weight_norm = 0.0
            for layer in self.model.layers:
                for w in layer.trainable_weights:
                    total_weight_norm += float(tf.norm(w).numpy())
            avg_weight_norm = total_weight_norm / max(
                1, len([w for layer in self.model.layers for w in layer.trainable_weights])
            )
            metrics["total_weight_norm"] = total_weight_norm
            metrics["avg_weight_norm"] = avg_weight_norm
            logger.info(f"Weight norms: total={total_weight_norm:.2f}, avg={avg_weight_norm:.4f}")
        except Exception as e:
            logger.debug(f"Could not compute weight norms: {e}")

    def _check_and_record_drift(
        self, val_acc: float, instrument: str, x_train_filtered: np.ndarray
    ) -> None:
        """Check for drift and record training result."""
        if self.drift_detector is None:
            return

        feature_means = (
            x_train_filtered.mean(axis=(0, 1)) if len(x_train_filtered.shape) == 3
            else x_train_filtered.mean(axis=0)
        )
        self.drift_detector.record_training_result(
            val_accuracy=val_acc,
            instrument=instrument,
            data_hash=self.lineage.data_hash if self.lineage else "",
            feature_means=feature_means,
        )

        drift_detected, drift_reason = self.drift_detector.check_drift()
        if drift_detected:
            logger.warning(f"⚠️ DRIFT DETECTED: {drift_reason}")
            self.metrics["drift_detected"] = True
            self.metrics["drift_reason"] = drift_reason
            if self.lineage:
                self.lineage.drift_detected = True
                self.lineage.drift_reason = drift_reason
                self.lineage.last_drift_check = datetime.now().isoformat()
        else:
            self.metrics["drift_detected"] = False
            if self.lineage:
                self.lineage.drift_detected = False
                self.lineage.last_drift_check = datetime.now().isoformat()

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        x_val: np.ndarray,
        y_val: np.ndarray,
        feature_names: Optional[list] = None,
        w_train: Optional[np.ndarray] = None,
        w_val: Optional[np.ndarray] = None,
        warm_start_path: Optional[str] = None,
        instrument: str = "UNKNOWN",
        data_range: str = "",
        skip_scaling: bool = False,
    ) -> Dict[str, float]:
        """Train Transformer for direction prediction with continual learning."""
        logger.info("Training Transformer (Direction)...")

        # Initialize state
        self._initialize_training_state(instrument, data_range)
        self.feature_names = feature_names
        self.n_features = X_train.shape[-1]

        # Scale features
        x_train_scaled, x_val_scaled = self._scale_features(X_train, x_val, skip_scaling)

        # Handle warm-start features and feature selection
        x_train_scaled, x_val_scaled, _warm_start_features_compatible = \
            self._handle_feature_preparation(
                x_train_scaled, x_val_scaled, y_train, warm_start_path
            )

        # Create sequences and filter
        x_train_filtered, y_train_filtered, x_val_filtered, y_val_filtered = \
            self._prepare_sequences_and_filter(
                x_train_scaled, x_val_scaled, y_train, y_val, w_train, w_val
            )

        # Compute class stats and variance weight
        train_up_pct, _, _, _ = self._compute_class_statistics(y_train_filtered, y_val_filtered)
        auto_variance_weight = self._compute_auto_variance_weight(instrument, train_up_pct)
        self.lineage.auto_variance_weight = auto_variance_weight

        # Sample weights and replay buffer
        sample_weights = self._compute_sample_weights(y_train_filtered)
        logger.info(f"Sequence shape: train={x_train_filtered.shape}, val={x_val_filtered.shape}")

        if self._use_replay:
            x_train_filtered, y_train_filtered, sample_weights = self._handle_replay_buffer(
                x_train_filtered, y_train_filtered, sample_weights, instrument
            )

        self.lineage.data_hash = TrainingLineage.compute_data_hash(x_train_filtered, y_train_filtered)

        # Build and configure model
        self.model = self._build_model((self.seq_len, self.n_features))
        effective_lr = self._setup_warm_start(warm_start_path, _warm_start_features_compatible)

        # Setup optimizer and compile
        optimizer = self._setup_optimizer_with_warmup(effective_lr, x_train_filtered)
        label_smoothing = getattr(self.config, "label_smoothing", 0.05) if self.config else 0.05
        base_loss = self._create_loss_function(auto_variance_weight, label_smoothing)
        self._compile_model_with_loss(optimizer, base_loss, instrument)

        # Evaluate warm-start baseline
        self._evaluate_warm_start_baseline(x_val_filtered, y_val_filtered, instrument)
        self.model.summary(print_fn=logger.info)

        # Initialize EMA and create callbacks
        self._initialize_ema()
        callbacks = self._create_training_callbacks(x_val_filtered, y_val_filtered)

        # Train
        history = self.model.fit(
            x_train_filtered, y_train_filtered,
            validation_data=(x_val_filtered, y_val_filtered),
            epochs=self.config.epochs,
            batch_size=self.config.batch_size,
            callbacks=callbacks,
            verbose=0,
            sample_weight=sample_weights,
        )

        self.is_trained = True

        # Post-training: recovery, EWC, EMA, lineage
        self._handle_warm_start_recovery(x_val_filtered, y_val_filtered)
        self._update_ewc_and_ema(x_train_filtered, y_train_filtered)
        self._update_lineage(history, x_train_filtered)

        # Compute metrics
        self.metrics = self._compute_final_metrics(
            history, x_val_filtered, y_val_filtered, x_train_filtered
        )

        # Drift detection
        self._check_and_record_drift(self.metrics["val_accuracy"], instrument, x_train_filtered)

        logger.info(
            f"Transformer trained [canonical]: val_accuracy={self.metrics['val_accuracy']:.4f}, "
            f"balanced_acc={self.metrics['val_balanced_accuracy']:.4f}"
        )
        return self.metrics

    def _handle_feature_preparation(
        self,
        x_train_scaled: np.ndarray,
        x_val_scaled: np.ndarray,
        y_train: np.ndarray,
        warm_start_path: Optional[str],
    ) -> Tuple[np.ndarray, np.ndarray, bool]:
        """Handle feature selection and warm-start feature compatibility."""
        use_feature_selection = getattr(self.config, "use_feature_selection", True) if self.config else True
        feature_selection_method = getattr(self.config, "feature_selection_method", "random_forest") if self.config else "random_forest"
        top_k_features = getattr(self.config, "top_k_features", 50) if self.config else 50

        _warm_start_feature_names, _warm_start_scaler = self._load_warm_start_feature_meta(warm_start_path)
        _warm_start_features_compatible = False

        if _warm_start_feature_names and self.feature_names:
            x_train_scaled, x_val_scaled, _warm_start_features_compatible = \
                self._apply_warm_start_features(
                    x_train_scaled, x_val_scaled, _warm_start_feature_names, _warm_start_scaler
                )
            if _warm_start_features_compatible:
                use_feature_selection = False

        if use_feature_selection and x_train_scaled.shape[-1] > top_k_features:
            x_train_scaled, x_val_scaled = self._apply_feature_selection(
                x_train_scaled, x_val_scaled, y_train, top_k_features, feature_selection_method
            )
            if self.scaler is not None:
                from sklearn.preprocessing import StandardScaler
                self.scaler = StandardScaler()
                x_train_scaled = self.scaler.fit_transform(x_train_scaled)
                x_val_scaled = self.scaler.transform(x_val_scaled)
                logger.info(f"✓ Re-fitted scaler on {x_train_scaled.shape[-1]} selected features")
            logger.info(f"🔍 Feature Selection Complete: train={x_train_scaled.shape}, val={x_val_scaled.shape}")

        return x_train_scaled, x_val_scaled, _warm_start_features_compatible

    def _setup_warm_start(
        self, warm_start_path: Optional[str], features_compatible: bool
    ) -> float:
        """Setup warm-start training if applicable."""
        self._is_warm_start = False
        self._warm_start_val_acc = 0.0
        self._warm_start_weights = None
        self._loaded_model_instrument = None
        effective_lr = self.config.learning_rate

        if warm_start_path and Path(warm_start_path).exists():
            if not features_compatible:
                logger.warning("🔥 WARM-START SKIPPED: Feature dimensions incompatible")
                logger.info("   Training will start fresh with new architecture")
            else:
                effective_lr = self._handle_warm_start(warm_start_path)
        elif warm_start_path:
            logger.info(f"No checkpoint found at {warm_start_path}. Starting fresh training.")

        return effective_lr

    def _align_features(self, x_reshaped: np.ndarray) -> np.ndarray:
        """Align input features to match model's expected feature count."""
        model_n_features = self.model.input_shape[-1]
        current_n_features = x_reshaped.shape[-1]

        if current_n_features <= model_n_features:
            return x_reshaped

        if (
            self.selected_indices is not None
            and len(self.selected_indices) == model_n_features
        ):
            try:
                return x_reshaped[:, self.selected_indices]
            except (IndexError, ValueError) as e:
                logger.warning(f"Feature selection failed: {e}")

        logger.warning(
            f"Feature count mismatch: input has {current_n_features}, "
            f"model expects {model_n_features}. Using first {model_n_features} features."
        )
        return x_reshaped[:, :model_n_features]

    def _prepare_predict_input(self, x_scaled: np.ndarray) -> np.ndarray:
        """Prepare scaled data for model prediction (flat or sequence)."""
        model_input_shape = self.model.input_shape
        is_flat_model = len(model_input_shape) == 2  # (None, n_features)

        if is_flat_model:
            return x_scaled[-1:] if len(x_scaled) > 1 else x_scaled

        # Sequence model - create sequence from last seq_len rows
        if len(x_scaled) >= self.seq_len:
            return x_scaled[-self.seq_len:].reshape(1, self.seq_len, -1)

        # Pad with zeros if not enough data
        pad_len = self.seq_len - len(x_scaled)
        x_padded = np.vstack([np.zeros((pad_len, x_scaled.shape[1])), x_scaled])
        return x_padded.reshape(1, self.seq_len, -1)

    def _should_use_ema(self, use_ema: bool) -> bool:
        """Check whether EMA weights should be applied for inference."""
        return (
            use_ema
            and self._use_ema
            and self.ema is not None
            and self.ema._initialized
            and self.config.use_ema_for_inference
        )

    def predict(self, X: np.ndarray, use_ema: bool = True) -> Dict[str, Any]:
        """
        Predict direction (0 or 1) with probability.

        Applies output calibration if enabled to correct for systematic bias.

        Args:
            X: Input features
            use_ema: If True and EMA is available, use EMA weights for stable inference

        Returns:
            Dict with 'direction' (0 or 1) and 'probability' (0.0 to 1.0)
        """
        if not self.is_trained:
            raise RuntimeError(MODEL_NOT_TRAINED_ERROR)

        x_reshaped = self._align_features(X.reshape(-1, X.shape[-1]))
        x_scaled = self.scaler.transform(x_reshaped)
        x_input = self._prepare_predict_input(x_scaled)

        use_ema_weights = self._should_use_ema(use_ema)
        if use_ema_weights:
            self.ema.apply()

        try:
            prob_raw = float(self.model.predict(x_input, verbose=0)[0, 0])
        finally:
            if use_ema_weights:
                self.ema.restore()

        # === APPLY OUTPUT CALIBRATION ===
        calibration = getattr(self, "output_calibration", None)
        threshold = 0.5
        if calibration and calibration.get("enabled", False):
            threshold = calibration.get("threshold", 0.5)

        direction = 1 if prob_raw > threshold else 0

        std = calibration.get("std", 0.15) if calibration else 0.15
        confidence_distance = abs(prob_raw - threshold) / (2 * std)
        confidence = min(1.0, confidence_distance)

        return {
            "direction": direction,
            "probability": prob_raw,  # Raw probability
            "probability_raw": prob_raw,  # Alias for debugging
            "confidence": confidence,  # Normalized confidence
            "threshold": threshold,  # Calibrated threshold
            "ema_used": use_ema_weights,
            "calibration_applied": calibration.get("enabled", False)
            if calibration
            else False,
        }

    def save(self, path: str, instrument: str = "UNKNOWN") -> None:
        """
        Save Transformer model with all continual learning state.

        Saves:
        - .keras: Main model weights
        - .meta.pkl: Scaler, config, metrics, lineage
        - .ema.pkl: EMA shadow weights
        - .ewc.pkl: EWC Fisher information + reference weights
        - replay buffer to trained_data/replay/<instrument>/
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        # === Persist callback state to lineage for warm-start ===
        if hasattr(self, "_collapse_callback") and self._collapse_callback is not None:
            self.lineage.lr_reductions_count = self._collapse_callback.lr_reductions
            self.lineage.collapse_recovery_count = (
                self._collapse_callback.recovery_count
            )
        try:
            self.lineage.final_learning_rate = _safe_get_learning_rate(
                self.model.optimizer, default=0.0
            )
        except Exception:
            self.lineage.final_learning_rate = 0.0

        logger.info(
            f"💾 Persisting training state: "
            f"variance_weight={getattr(self.lineage, 'auto_variance_weight', 0.0):.3f}, "
            f"lr_reductions={getattr(self.lineage, 'lr_reductions_count', 0)}, "
            f"final_lr={getattr(self.lineage, 'final_learning_rate', 0.0):.2e}"
        )

        # Save Keras model in native format
        self.model.save(str(path))

        # === CROSS-VERSION COMPATIBILITY: Save weights and architecture separately ===
        # This allows loading on different Keras versions (2.x vs 3.x)

        # 1. Save weights in H5 format (portable across Keras versions)
        weights_path = path.with_suffix(WEIGHTS_H5_SUFFIX)
        try:
            self.model.save_weights(str(weights_path))
            logger.debug(f"Saved portable weights to {weights_path}")
        except Exception as e:
            logger.warning(f"Could not save portable weights: {e}")

        # 2. Save architecture as JSON (for rebuilding on different Keras version)
        arch_path = path.with_suffix(ARCH_JSON_SUFFIX)
        try:
            arch_json = self.model.to_json()
            with open(arch_path, "w") as f:
                f.write(arch_json)
            logger.debug(f"Saved architecture to {arch_path}")
        except Exception as e:
            logger.warning(f"Could not save architecture JSON: {e}")

        # Update lineage metrics before saving
        if self.lineage:
            self.lineage.add_metrics(self.metrics)

        # Save scaler, config, and lineage (with architecture config for cross-version rebuild)
        meta = {
            "scaler": self.scaler,
            "seq_len": self.seq_len,
            "metrics": self.metrics,
            "config": self.config.__dict__,
            "feature_names": self.feature_names,
            "n_features": self.n_features,
            "selected_indices": getattr(
                self, "selected_indices", None
            ),  # Feature selection indices
            "model_type": "transformer",
            "lineage": self.lineage.to_dict() if self.lineage else None,
            "is_warm_start": self._is_warm_start,
            "ema_enabled": self._use_ema,
            "ewc_enabled": self._use_ewc,
            "replay_enabled": self._use_replay,
            "output_calibration": getattr(
                self, "output_calibration", None
            ),  # Save calibration params
            # Store transformer architecture config for programmatic rebuild
            "architecture": {
                "input_shape": (self.seq_len, self.n_features),
                "transformer_d_model": self.transformer_d_model,
                "transformer_num_heads": self.transformer_num_heads,
                "transformer_num_layers": self.transformer_num_layers,
                "transformer_dff": self.transformer_dff,
                "transformer_dropout": self.transformer_dropout,
            },
        }
        meta_path = path.with_suffix(META_PKL_SUFFIX)
        with open(meta_path, "wb") as f:
            pickle.dump(meta, f)

        # Save EMA weights if available
        if self._use_ema and self.ema is not None and self.ema._initialized:
            ema_data = {
                "ema_weights": self.ema.get_ema_weights(),
                "ema_weight_names": [
                    w.name for w in self.model.trainable_weights
                ],  # For graceful loading
                "decay": self.ema.decay,
                "update_every": self.ema.update_every,
                "step_counter": self.ema.step_counter,
            }
            ema_path = path.with_suffix(EMA_PKL_SUFFIX)
            with open(ema_path, "wb") as f:
                pickle.dump(ema_data, f)
            logger.info(f"📊 EMA weights saved to {ema_path}")

        # Save EWC state if available
        if (
            self._use_ewc
            and self.ewc is not None
            and self.ewc.fisher_diagonal is not None
        ):
            ewc_path = path.with_suffix(EWC_PKL_SUFFIX)
            self.ewc.save(str(ewc_path))

        # Save replay buffer
        if self._use_replay and self.replay_buffer is not None:
            self.replay_buffer.save(instrument)

        logger.info(
            f"✅ Transformer saved to {path} "
            f"(EMA={self._use_ema}, EWC={self._use_ewc}, Replay={self._use_replay}, cross-version=True)"
        )

    def _load_model_cross_version(self, path: Path) -> Tuple[Any, list]:
        """Strategy 0: Load model with cross-version loader."""
        errors = []
        try:
            from src.utils.keras_model_loader import load_keras_model
            model, load_metadata = load_keras_model(str(path), compile=False)
            if load_metadata.get("success"):
                logger.info(
                    f"✓ Model loaded with cross-version loader ({load_metadata.get('approach_used')})"
                )
                return model, errors
            return None, errors
        except ImportError:
            errors.append("Cross-version loader not available")
        except Exception as e:
            errors.append(f"Cross-version: {e}")
        return None, errors

    def _load_model_standard(self, path: Path) -> Tuple[Any, list]:
        """Strategy 1-3: Try standard, tf.keras, and safe_mode=False loaders."""
        import tensorflow as tf
        from tensorflow import keras

        errors = []
        loaders = [
            ("Standard", lambda: keras.models.load_model(str(path), compile=False)),
            ("TF-native", lambda: tf.keras.models.load_model(str(path), compile=False)),
            ("Safe-mode", lambda: keras.models.load_model(str(path), compile=False, safe_mode=False)),
        ]
        for name, loader_fn in loaders:
            try:
                model = loader_fn()
                logger.info(f"✓ Model loaded with {name} loader")
                return model, errors
            except Exception as e:
                errors.append(f"{name}: {e}")
        return None, errors

    def _load_model_from_arch_weights(self, path: Path) -> Tuple[Any, list]:
        """Strategy 3.5: Load from arch.json + weights.h5."""
        from tensorflow import keras

        errors = []
        arch_path = path.with_suffix(ARCH_JSON_SUFFIX)
        weights_path = path.with_suffix(WEIGHTS_H5_SUFFIX)
        if not (arch_path.exists() and weights_path.exists()):
            return None, errors
        try:
            with open(arch_path) as f:
                arch_json = f.read()
            model = keras.models.model_from_json(arch_json)
            model.load_weights(str(weights_path))
            logger.info("✓ Model loaded from arch.json + weights.h5 (cross-version)")
            return model, errors
        except Exception as e:
            errors.append(f"Arch+Weights: {e}")
        return None, errors

    def _rebuild_transformer_from_meta(self, path: Path) -> Tuple[Any, list]:
        """Strategy 4: Rebuild model from metadata and load weights."""
        import tensorflow as tf
        from tensorflow import keras

        errors = []
        meta_path = path.with_suffix(META_PKL_SUFFIX)
        if not meta_path.exists():
            return None, errors

        try:
            with open(meta_path, "rb") as f:
                meta = pickle.load(f)

            model = self._build_transformer_from_config(meta, keras, tf)
            self._try_apply_ema_to_rebuilt_model(model, path)
            return model, errors
        except Exception as e:
            errors.append(f"Rebuild: {e}")
        return None, errors

    def _build_transformer_from_config(self, meta: dict, keras: Any, tf: Any) -> Any:
        """Reconstruct Transformer architecture from saved metadata."""
        n_features = meta.get("n_features", 59)
        seq_len = meta.get("seq_len", 60)
        config = meta.get("config", {})

        d_model = config.get("transformer_d_model", 32)
        num_heads = config.get("transformer_num_heads", 4)
        num_layers = config.get("transformer_num_layers", 2)
        dff = config.get("transformer_dff", 64)
        dropout = config.get("transformer_dropout", 0.2)
        final_dense_units = config.get("final_dense_units", 16)
        final_dense_activation = config.get("final_dense_activation", "tanh")
        final_dense_dropout = config.get("final_dense_dropout", 0.15)

        logger.info(
            f"Rebuilding Transformer: n_features={n_features}, seq_len={seq_len}, "
            f"d_model={d_model}, final_dense={final_dense_units}"
        )

        inp = keras.Input(shape=(seq_len, n_features), name="features")
        x = keras.layers.GaussianNoise(0.15)(inp)
        x = keras.layers.SpatialDropout1D(0.2)(x)
        x = keras.layers.Dense(d_model, name="input_projection")(x)
        x = keras.layers.Dropout(0.3)(x)

        # Positional encoding
        positions = np.arange(seq_len)[:, np.newaxis]
        dims = np.arange(d_model)[np.newaxis, :]
        angles = positions / np.power(10000, (2 * (dims // 2)) / d_model)
        pos_encoding = np.zeros((seq_len, d_model))
        pos_encoding[:, 0::2] = np.sin(angles[:, 0::2])
        pos_encoding[:, 1::2] = np.cos(angles[:, 1::2])
        x = x + tf.constant(pos_encoding[np.newaxis, :, :].astype(np.float32))

        for i in range(num_layers):
            attn_output = keras.layers.MultiHeadAttention(
                num_heads=num_heads,
                key_dim=d_model // num_heads,
                name=f"transformer_{i}_mha",
            )(x, x)
            attn_output = keras.layers.Dropout(dropout)(attn_output)
            x = keras.layers.LayerNormalization(
                epsilon=1e-6, name=f"transformer_{i}_ln1"
            )(x + attn_output)

            ffn = keras.layers.Dense(
                dff, activation="relu", name=f"transformer_{i}_ffn1"
            )(x)
            ffn = keras.layers.Dense(d_model, name=f"transformer_{i}_ffn2")(ffn)
            ffn = keras.layers.Dropout(dropout)(ffn)
            x = keras.layers.LayerNormalization(
                epsilon=1e-6, name=f"transformer_{i}_ln2"
            )(x + ffn)

        x = keras.layers.GlobalAveragePooling1D()(x)
        x = keras.layers.Dense(final_dense_units, activation=final_dense_activation)(x)
        x = keras.layers.Dropout(final_dense_dropout)(x)
        direction = keras.layers.Dense(
            1, activation="sigmoid", name="direction", dtype="float32"
        )(x)

        model = keras.Model(inputs=inp, outputs=direction, name="transformer_direction")
        logger.info("✓ Model architecture rebuilt from metadata")
        return model

    def _try_apply_ema_to_rebuilt_model(self, model: Any, path: Path) -> None:
        """Try to load and apply EMA weights to a rebuilt model."""
        ema_path = path.with_suffix(EMA_PKL_SUFFIX)
        if not ema_path.exists():
            return

        with open(ema_path, "rb") as f:
            ema_data = pickle.load(f)
        ema_weights = ema_data.get("ema_weights", [])
        model_weights = model.trainable_weights

        if len(ema_weights) != len(model_weights):
            logger.debug(
                f"EMA weights count ({len(ema_weights)}) != "
                f"model weights ({len(model_weights)}), will re-init"
            )
            return

        loaded_count, skipped_count = 0, 0
        for w, ema_w in zip(model_weights, ema_weights):
            if w.shape != tuple(ema_w.shape):
                logger.warning(f"Shape mismatch for {w.name}: model={w.shape}, EMA={ema_w.shape}")
                skipped_count += 1
                continue
            try:
                w.assign(ema_w.astype(_get_numpy_dtype(w.dtype)))
                loaded_count += 1
            except Exception as assign_err:
                logger.warning(f"Could not assign EMA weight to {w.name}: {assign_err}")
                skipped_count += 1

        if skipped_count > 0:
            logger.warning(
                f"⚠️ Loaded {loaded_count} EMA weights, "
                f"skipped {skipped_count} due to shape mismatch"
            )
        else:
            logger.info(f"✓ Loaded {loaded_count} EMA weights into rebuilt model")

    def _load_model_with_strategies(self, path: Path) -> Any:
        """Try all model loading strategies in order, returning loaded model or raising."""
        all_errors = []

        strategies = [
            self._load_model_cross_version,
            self._load_model_standard,
            self._load_model_from_arch_weights,
            self._rebuild_transformer_from_meta,
        ]

        for strategy in strategies:
            model, errors = strategy(path)
            all_errors.extend(errors)
            if model is not None:
                return model

        raise RuntimeError(
            f"Failed to load model from {path}. Errors: {'; '.join(all_errors)}"
        )

    def _load_metadata(self, path: Path) -> None:
        """Load model metadata from .meta.pkl file."""
        meta_path = path.with_suffix(META_PKL_SUFFIX)
        with open(meta_path, "rb") as f:
            meta = pickle.load(f)

        self.scaler = meta["scaler"]
        self.seq_len = meta["seq_len"]
        self.metrics = meta["metrics"]
        self.feature_names = meta.get("feature_names")
        self.n_features = meta.get("n_features")
        self.selected_indices = meta.get("selected_indices")
        self._is_warm_start = meta.get("is_warm_start", False)
        self._use_ema = meta.get("ema_enabled", True)
        self._use_ewc = meta.get("ewc_enabled", True)
        self._use_replay = meta.get("replay_enabled", True)

        self.output_calibration = meta.get("output_calibration", None)
        if self.output_calibration and self.output_calibration.get("enabled"):
            logger.info(
                f"📐 Output calibration loaded: bias={self.output_calibration['bias']:.4f}"
            )

        if meta.get("lineage"):
            self.lineage = TrainingLineage.from_dict(meta["lineage"])
            logger.info(
                f"📊 Lineage loaded: checkpoint={self.lineage.checkpoint_id}, "
                f"cumulative_epochs={self.lineage.cumulative_epochs}"
            )

    def _load_ema_state(self, path: Path) -> None:
        """Load EMA weights for inference."""
        ema_path = path.with_suffix(EMA_PKL_SUFFIX)
        if not (ema_path.exists() and self._use_ema):
            return

        try:
            with open(ema_path, "rb") as f:
                ema_data = pickle.load(f)

            ema_weights = ema_data.get("ema_weights")
            if not isinstance(ema_weights, list) or not ema_weights:
                raise ValueError("EMA weights missing or invalid")

            if any(getattr(w, "shape", ()) == () for w in ema_weights):
                raise ValueError("EMA weights contain scalar tensors")

            self.ema = EMACallback(
                self.model,
                decay=ema_data.get("decay", self.config.ema_decay),
                update_every=ema_data.get("update_every", self.config.ema_update_every),
            )
            self.ema.set_ema_weights(
                ema_weights, weight_names=ema_data.get("ema_weight_names")
            )
            self.ema.step_counter = ema_data.get("step_counter", 0)
            logger.info(f"📊 EMA weights loaded (decay={self.ema.decay})")
        except Exception as e:
            logger.info(f"ℹ EMA load skipped: {e}")
            try:
                ema_path.unlink()
            except OSError:
                pass

    def _load_ewc_state(self, path: Path) -> None:
        """Load EWC penalty state for continual learning."""
        ewc_path = path.with_suffix(EWC_PKL_SUFFIX)
        if ewc_path.exists() and self._use_ewc:
            self.ewc = EWCPenalty(
                self.model,
                ewc_lambda=self.config.ewc_lambda,
                gamma=self.config.ewc_gamma,
            )
            self.ewc.load(str(ewc_path))

    def _load_replay_state(self, instrument: str) -> None:
        """Load replay buffer for continual learning."""
        if self._use_replay:
            self.replay_buffer = ReplayBuffer(
                capacity_ratio=self.config.replay_buffer_ratio,
                mix_ratio=self.config.replay_mix_ratio,
                buffer_dir=self.config.replay_buffer_dir,
            )
            self.replay_buffer.load(instrument)

    def load(self, path: str, instrument: str = "UNKNOWN") -> None:
        """
        Load Transformer model with all continual learning state.

        Loads:
        - .keras: Main model weights
        - .meta.pkl: Scaler, config, metrics, lineage
        - .ema.pkl: EMA shadow weights (for inference)
        - .ewc.pkl: EWC state (for future warm-start)
        - replay buffer from trained_data/replay/<instrument>/

        Handles Keras 2.x/3.x compatibility for models trained on Colab.
        """
        path = Path(path)

        self.model = self._load_model_with_strategies(path)
        self._load_metadata(path)
        self._load_ema_state(path)
        self._load_ewc_state(path)
        self._load_replay_state(instrument)

        self.is_trained = True
        logger.info(f"✅ Transformer loaded from {path}")

        if (
            self.lineage
            and self.metrics.get("val_accuracy")
            and self.lineage.check_drift(
                self.metrics["val_accuracy"], self.config.drift_threshold
            )
        ):
            logger.info(
                "📊 Model performance below historical best - consider retraining if persistent"
            )


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
        self.class_names = ["trend", "chop", "mean_revert"]

        # Transformer hyperparameters - match proven direction config as baseline
        self.d_model = getattr(config, "transformer_d_model", 16) if config else 16
        self.num_heads = getattr(config, "transformer_num_heads", 2) if config else 2
        self.ff_dim = getattr(config, "transformer_ff_dim", 32) if config else 32
        self.num_blocks = getattr(config, "transformer_num_blocks", 1) if config else 1
        self.transformer_dropout = (
            getattr(config, "transformer_dropout", 0.2) if config else 0.2
        )  # Reduced from 0.4

    def _build_model(self, input_shape: Tuple[int, int]) -> Any:
        """Build Transformer model for 3-class regime classification."""
        from tensorflow import keras

        seq_len, n_features = input_shape

        inp = keras.Input(shape=(seq_len, n_features), name="features")

        # Project input to d_model dimensions
        x = keras.layers.Dense(self.d_model, name="input_projection")(inp)

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
                name_prefix=f"transformer_{i}",
            )

        # Global pooling and output
        x = keras.layers.GlobalAveragePooling1D()(x)
        x = keras.layers.Dense(32, activation="relu")(x)
        x = keras.layers.Dropout(self.transformer_dropout)(x)

        # 3-class regime output (softmax)
        regime = keras.layers.Dense(
            3, activation="softmax", name="regime", dtype="float32"
        )(x)

        model = keras.Model(inputs=inp, outputs=regime, name="transformer_regime")

        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=self.config.learning_rate),
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
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

        # Cast to match input dtype for mixed precision
        pos_tensor = tf.constant(pos_encoding)
        pos_tensor = tf.cast(pos_tensor, x.dtype)
        return x + pos_tensor

    def _transformer_encoder_layer(
        self,
        x,
        d_model: int,
        num_heads: int,
        dff: int,
        dropout: float,
        name_prefix: str,
    ):
        """Single transformer encoder layer."""
        from tensorflow import keras

        # Multi-head self-attention
        attn_output = keras.layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=d_model // num_heads, name=f"{name_prefix}_mha"
        )(x, x)
        attn_output = keras.layers.Dropout(dropout)(attn_output)
        x = keras.layers.LayerNormalization(epsilon=1e-6, name=f"{name_prefix}_ln1")(
            x + attn_output
        )

        # Feedforward network
        ffn = keras.layers.Dense(dff, activation="relu", name=f"{name_prefix}_ffn1")(x)
        ffn = keras.layers.Dense(d_model, name=f"{name_prefix}_ffn2")(ffn)
        ffn = keras.layers.Dropout(dropout)(ffn)
        x = keras.layers.LayerNormalization(epsilon=1e-6, name=f"{name_prefix}_ln2")(
            x + ffn
        )

        return x

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        x_val: np.ndarray,
        y_val: np.ndarray,
        feature_names: Optional[list] = None,
        class_names: Optional[list] = None,
    ) -> Dict[str, float]:
        """
        Train Transformer for 3-class regime classification.
        Reports F1 score (macro) as primary metric.
        """
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
        x_train_scaled = self.scaler.fit_transform(
            X_train.reshape(-1, X_train.shape[-1])
        )
        x_val_scaled = self.scaler.transform(x_val.reshape(-1, x_val.shape[-1]))

        # Get seq_len from config (validated)
        seq_len = get_config_seq_len(self.config)
        self.seq_len = seq_len

        # Create sequences using shared helper
        x_train_seq, y_train_seq = create_sequences(x_train_scaled, y_train, seq_len)
        x_val_seq, y_val_seq = create_sequences(x_val_scaled, y_val, seq_len)

        # Log class distribution
        unique, counts = np.unique(y_train_seq, return_counts=True)
        class_dist = dict(zip(unique, counts))
        logger.info(f"Training class distribution: {class_dist}")

        unique_val, counts_val = np.unique(y_val_seq, return_counts=True)
        class_dist_val = dict(zip(unique_val, counts_val))
        logger.info(f"Validation class distribution: {class_dist_val}")

        logger.info(f"Sequence shape: train={x_train_seq.shape}, val={x_val_seq.shape}")

        # Compute class weights for imbalanced classes
        classes = np.unique(y_train_seq)
        if len(classes) > 1:
            weights = compute_class_weight("balanced", classes=classes, y=y_train_seq)
            class_weight = {int(c): w for c, w in zip(classes, weights)}
            logger.info(f"Class weights: {class_weight}")
        else:
            class_weight = None

        # Build model
        self.model = self._build_model((seq_len, self.n_features))
        # Compact model info for console
        try:
            from rich.console import Console as _RichConsole
            _rc = _RichConsole()
            _rc.print(
                f"  [dim]Model:[/dim] {self.model.name} | "
                f"[green]{self.model.count_params():,}[/green] params | "
                f"input=({seq_len}, {self.n_features})"
            )
        except Exception:
            pass
        # Full summary to log file (console handler at WARNING skips INFO)
        self.model.summary(print_fn=logger.info)

        # Callbacks - use config patience values
        callbacks = [
            # Rich-formatted epoch display with color coding (or quiet progress bar)
            RichEpochCallback(
                model_name="Transformer Regime",
                total_epochs=self.config.epochs,
                quiet=self.config.quiet,
            )
            if not self.config.quiet
            else QuietProgressCallback(
                model_name="Transformer Regime",
                total_epochs=self.config.epochs,
            ),
            keras.callbacks.EarlyStopping(
                monitor="val_loss",
                patience=self.config.patience,
                mode="min",
                restore_best_weights=True,
                verbose=0,  # Suppress - Rich callback handles display
                start_from_epoch=self.config.min_epochs,  # Enforce minimum epochs before early stopping
            ),
            # NOTE: ReduceLROnPlateau removed - incompatible with LearningRateSchedule
        ]

        # Train
        history = self.model.fit(
            x_train_seq,
            y_train_seq,
            validation_data=(x_val_seq, y_val_seq),
            epochs=self.config.epochs,
            batch_size=self.config.batch_size,
            callbacks=callbacks,
            verbose=0,  # Suppress Keras output - RichEpochCallback handles display
            class_weight=class_weight,
        )

        self.is_trained = True

        # Calculate metrics
        val_pred_probs = self.model.predict(x_val_seq, verbose=0)
        val_pred = np.argmax(val_pred_probs, axis=1)

        # F1 score (macro - treats all classes equally)
        f1_macro = f1_score(y_val_seq, val_pred, average="macro")
        f1_weighted = f1_score(y_val_seq, val_pred, average="weighted")

        # Per-class F1
        f1_per_class = f1_score(y_val_seq, val_pred, average=None)

        # Accuracy
        val_acc = np.mean(val_pred == y_val_seq)

        # Classification report
        report = classification_report(
            y_val_seq, val_pred, target_names=self.class_names
        )
        logger.info(f"\nClassification Report:\n{report}")

        self.metrics = {
            "train_accuracy": float(history.history["accuracy"][-1]),
            "val_accuracy": float(val_acc),
            "f1_macro": float(f1_macro),
            "f1_weighted": float(f1_weighted),
            "f1_trend": float(f1_per_class[0]) if len(f1_per_class) > 0 else 0.0,
            "f1_chop": float(f1_per_class[1]) if len(f1_per_class) > 1 else 0.0,
            "f1_mean_revert": float(f1_per_class[2]) if len(f1_per_class) > 2 else 0.0,
            "epochs_trained": len(history.history["loss"]),
            "n_train_samples": len(x_train_seq),
            "n_val_samples": len(x_val_seq),
        }

        logger.info(
            f"Regime Transformer trained: val_acc={val_acc:.4f}, F1_macro={f1_macro:.4f}"
        )
        logger.info(
            f"  F1 per class: trend={self.metrics['f1_trend']:.3f}, "
            f"chop={self.metrics['f1_chop']:.3f}, mean_revert={self.metrics['f1_mean_revert']:.3f}"
        )

        return self.metrics

    def predict(self, X: np.ndarray) -> Dict[str, Any]:
        """
        Predict regime (0=trend, 1=chop, 2=mean_revert) with probabilities.
        """
        if not self.is_trained:
            raise RuntimeError(MODEL_NOT_TRAINED_ERROR)

        # Scale
        x_scaled = self.scaler.transform(X.reshape(-1, X.shape[-1]))

        # Create sequence from last seq_len rows
        if len(x_scaled) >= self.seq_len:
            x_seq = x_scaled[-self.seq_len :].reshape(1, self.seq_len, -1)
        else:
            # Pad with zeros if not enough data
            pad_len = self.seq_len - len(x_scaled)
            x_padded = np.vstack([np.zeros((pad_len, x_scaled.shape[-1])), x_scaled])
            x_seq = x_padded.reshape(1, self.seq_len, -1)

        # Predict
        probs = self.model.predict(x_seq, verbose=0)[0]
        regime = int(np.argmax(probs))

        return {
            "regime": regime,
            "regime_name": self.class_names[regime],
            "prob_trend": float(probs[0]),
            "prob_chop": float(probs[1]),
            "prob_mean_revert": float(probs[2]),
            "confidence": float(np.max(probs)),
        }

    def save(self, path: str) -> None:
        """Save Transformer regime model and scaler."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        self.model.save(str(path))

        meta = {
            "scaler": self.scaler,
            "seq_len": self.seq_len,
            "metrics": self.metrics,
            "config": self.config.__dict__,
            "feature_names": self.feature_names,
            "n_features": self.n_features,
            "class_names": self.class_names,
            "model_type": "transformer_regime",
        }
        meta_path = path.with_suffix(META_PKL_SUFFIX)
        with open(meta_path, "wb") as f:
            pickle.dump(meta, f)

        logger.info(f"Transformer Regime saved to {path}")

    def load(self, path: str) -> None:
        """Load Transformer regime model and scaler."""
        from tensorflow import keras

        path = Path(path)
        self.model = keras.models.load_model(str(path))

        meta_path = path.with_suffix(META_PKL_SUFFIX)
        with open(meta_path, "rb") as f:
            meta = pickle.load(f)

        self.scaler = meta["scaler"]
        self.seq_len = meta["seq_len"]
        self.metrics = meta["metrics"]
        self.feature_names = meta.get("feature_names")
        self.n_features = meta.get("n_features")
        self.class_names = meta.get("class_names", ["trend", "chop", "mean_revert"])
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
        x_val: np.ndarray,
        y_val: np.ndarray,
        feature_names: Optional[list] = None,
        momentum_norm_factor: Optional[float] = None,
    ) -> Dict[str, float]:
        """Train XGBoost for momentum analysis (2 models) with GPU support."""
        self.momentum_norm_factor = momentum_norm_factor
        try:
            import xgboost as xgb
        except ImportError:
            raise ImportError("XGBoost not installed. Run: pip install xgboost")

        from sklearn.preprocessing import StandardScaler

        # Detect GPU availability
        use_gpu = self.config.use_gpu if hasattr(self.config, "use_gpu") else False
        tree_method = "gpu_hist" if use_gpu else "auto"

        logger.info(
            f"Training XGBoost (Momentum) - GPU: {use_gpu}, tree_method: {tree_method}"
        )

        # Save feature names for inference
        self.feature_names = feature_names
        self.n_features = X_train.shape[-1]

        # Scale features
        self.scaler = StandardScaler()
        x_train_scaled = self.scaler.fit_transform(X_train)
        x_val_scaled = self.scaler.transform(x_val)

        # Split targets: y[:, 0] = momentum_score, y[:, 1] = acceleration
        y_train_momentum = y_train[:, 0]
        y_train_accel = y_train[:, 1].astype(int)
        y_val_momentum = y_val[:, 0]
        y_val_accel = y_val[:, 1].astype(int)

        # Train momentum regressor with GPU acceleration
        self.momentum_model = xgb.XGBRegressor(
            n_estimators=self.config.xgb_n_estimators,
            max_depth=self.config.xgb_max_depth,
            learning_rate=self.config.xgb_learning_rate,
            tree_method=tree_method,  # GPU-accelerated if available
            predictor="gpu_predictor" if use_gpu else "auto",
            verbosity=0,
            n_jobs=-1 if not use_gpu else 1,  # GPU doesn't need multi-threading
            random_state=42,
        )
        self.momentum_model.fit(
            x_train_scaled,
            y_train_momentum,
            eval_set=[(x_val_scaled, y_val_momentum)],
            verbose=False,
        )

        # Train acceleration classifier with GPU acceleration
        self.accel_model = xgb.XGBClassifier(
            n_estimators=self.config.xgb_n_estimators,
            max_depth=self.config.xgb_max_depth,
            learning_rate=self.config.xgb_learning_rate,
            tree_method=tree_method,  # GPU-accelerated if available
            predictor="gpu_predictor" if use_gpu else "auto",
            verbosity=0,
            n_jobs=-1 if not use_gpu else 1,  # GPU doesn't need multi-threading
            random_state=42,
        )
        self.accel_model.fit(
            x_train_scaled,
            y_train_accel,
            eval_set=[(x_val_scaled, y_val_accel)],
            verbose=False,
        )

        self.is_trained = True

        # Calculate metrics
        momentum_pred = self.momentum_model.predict(x_val_scaled)
        accel_pred = self.accel_model.predict(x_val_scaled)

        momentum_mae = float(np.mean(np.abs(momentum_pred - y_val_momentum)))
        accel_acc = float(np.mean(accel_pred == y_val_accel))

        self.metrics = {
            "momentum_mae": momentum_mae,
            "acceleration_accuracy": accel_acc,
        }

        logger.info(
            f"XGBoost trained: momentum_mae={momentum_mae:.4f}, accel_acc={accel_acc:.4f}"
        )
        return self.metrics

    def predict(self, X: np.ndarray) -> Dict[str, Any]:
        """Predict momentum score and acceleration."""
        if not self.is_trained:
            raise RuntimeError(MODEL_NOT_TRAINED_ERROR)

        # Scale
        if X.ndim == 1:
            X = X.reshape(1, -1)
        x_scaled = self.scaler.transform(X)

        # Get last row for prediction
        x_last = x_scaled[-1:] if len(x_scaled) > 1 else x_scaled

        # DEBUG: Log scaled features once
        if not hasattr(self, "_predict_debug_logged"):
            logger.info(
                f"XGB Predict - Input shape: {X.shape}, Scaled shape: {x_scaled.shape}"
            )
            logger.info(f"XGB Predict - x_last scaled: {x_last.flatten()}")
            self._predict_debug_logged = True

        momentum = float(self.momentum_model.predict(x_last)[0])
        acceleration = bool(self.accel_model.predict(x_last)[0])

        # Clamp momentum to 0-1
        momentum = max(0.0, min(1.0, momentum))

        return {
            "momentum": momentum,
            "acceleration": acceleration,
        }

    def save(self, path: str) -> None:
        """Save XGBoost models with version metadata."""
        import sklearn
        import xgboost

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "momentum_model": self.momentum_model,
            "accel_model": self.accel_model,
            "scaler": self.scaler,
            "metrics": self.metrics,
            "config": self.config.__dict__,
            "feature_names": self.feature_names,
            "n_features": self.n_features,
            "momentum_norm_factor": self.momentum_norm_factor,
            # Version metadata for compatibility checks
            "sklearn_version": sklearn.__version__,
            "xgboost_version": xgboost.__version__,
            "saved_at": datetime.now().isoformat(),
        }

        with open(path, "wb") as f:
            pickle.dump(data, f)

        logger.info(
            f"XGBoost saved to {path} (sklearn={sklearn.__version__}, xgboost={xgboost.__version__})"
        )

    def load(self, path: str) -> None:
        """Load XGBoost models."""
        import warnings

        # Suppress XGBoost version warnings (common when loading older serialized models)
        warnings.filterwarnings("ignore", category=UserWarning, module="xgboost")
        warnings.filterwarnings("ignore", message=SERIALIZED_MODEL_WARNING)
        warnings.filterwarnings("ignore", message=".*older version of XGBoost.*")

        with open(path, "rb") as f:
            data = pickle.load(f)

        # DEBUG: Log scaler info
        scaler = data.get("scaler")
        if scaler is not None:
            logger.info(f"XGB Scaler - mean_: {scaler.mean_}")
            logger.info(f"XGB Scaler - scale_: {scaler.scale_}")

        self.momentum_model = data["momentum_model"]
        self.accel_model = data["accel_model"]
        self.scaler = data["scaler"]
        self.metrics = data["metrics"]
        self.feature_names = data.get("feature_names")
        self.n_features = data.get("n_features")
        self.is_trained = True

        # Store version info for compatibility checking
        self._saved_sklearn_version = data.get("sklearn_version")
        self._saved_xgboost_version = data.get("xgboost_version")
        self._saved_at = data.get("saved_at")

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
        x_val: np.ndarray,
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
        x_train_scaled = self.scaler.fit_transform(X_train)
        x_val_scaled = self.scaler.transform(x_val)

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
        self.drawdown_model.fit(x_train_scaled, y_train_drawdown)

        # Train streak probability regressor (0-1 range, so regression not classification)
        self.streak_model = RandomForestRegressor(
            n_estimators=self.config.rf_n_estimators,
            max_depth=self.config.rf_max_depth,
            min_samples_leaf=self.config.rf_min_samples_leaf,
            n_jobs=-1,
            random_state=42,
        )
        self.streak_model.fit(x_train_scaled, y_train_streak)

        self.is_trained = True

        # Calculate metrics
        drawdown_pred = self.drawdown_model.predict(x_val_scaled)
        streak_pred = self.streak_model.predict(x_val_scaled)

        # Clip predictions to valid range (0-10% for drawdown, 0-1 for streak)
        drawdown_pred = np.clip(drawdown_pred, 0, 0.10)
        streak_pred = np.clip(streak_pred, 0, 1.0)

        drawdown_mae = float(np.mean(np.abs(drawdown_pred - y_val_drawdown)))
        streak_mae = float(np.mean(np.abs(streak_pred - y_val_streak)))

        # Convert to basis points for meaningful display (0.001 = 10 bps)
        drawdown_mae_bps = drawdown_mae * 10000

        self.metrics = {
            "drawdown_mae_pct": drawdown_mae,  # Raw percentage (0-1)
            "drawdown_mae_bps": drawdown_mae_bps,  # Basis points for display
            "streak_prob_mae": streak_mae,
        }

        logger.info(
            f"RF trained: drawdown_mae={drawdown_mae_bps:.1f} bps ({drawdown_mae * 100:.3f}%), "
            f"streak_mae={streak_mae:.4f}"
        )
        return self.metrics

    def predict(self, X: np.ndarray) -> Dict[str, Any]:
        """Predict expected drawdown and streak probability."""
        if not self.is_trained:
            raise RuntimeError(MODEL_NOT_TRAINED_ERROR)

        # Scale
        if X.ndim == 1:
            X = X.reshape(1, -1)
        x_scaled = self.scaler.transform(X)

        # Get last row for prediction
        x_last = x_scaled[-1:] if len(x_scaled) > 1 else x_scaled

        # Model outputs drawdown as PERCENTAGE (instrument-agnostic)
        expected_drawdown_pct = float(self.drawdown_model.predict(x_last)[0])
        streak_prob = float(self.streak_model.predict(x_last)[0])

        # Clamp values to realistic ranges
        expected_drawdown_pct = max(
            0.0, min(0.10, expected_drawdown_pct)
        )  # 0-10% max drawdown
        streak_prob = max(0.0, min(1.0, streak_prob))  # 0-100% probability

        return {
            "expected_drawdown_pct": expected_drawdown_pct,
            # Keep legacy key for backward compatibility
            "expected_drawdown_pips": expected_drawdown_pct
            * 10000,  # Rough conversion for display
            "streak_prob": streak_prob,
        }

    def save(self, path: str) -> None:
        """Save Random Forest models with version metadata."""
        import sklearn

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "drawdown_model": self.drawdown_model,
            "streak_model": self.streak_model,
            "scaler": self.scaler,
            "metrics": self.metrics,
            "config": self.config.__dict__,
            "feature_names": self.feature_names,
            "n_features": self.n_features,
            # Version metadata for compatibility checks
            "sklearn_version": sklearn.__version__,
            "saved_at": datetime.now().isoformat(),
        }

        with open(path, "wb") as f:
            pickle.dump(data, f)

        logger.info(f"Random Forest saved to {path} (sklearn={sklearn.__version__})")

    def load(self, path: str) -> None:
        """Load Random Forest models."""
        import warnings

        # Suppress sklearn version warnings
        try:
            from sklearn.exceptions import InconsistentVersionWarning

            warnings.filterwarnings("ignore", category=InconsistentVersionWarning)
        except ImportError:
            pass
        warnings.filterwarnings("ignore", message=UNPICKLE_ESTIMATOR_WARNING)

        with open(path, "rb") as f:
            data = pickle.load(f)

        self.drawdown_model = data["drawdown_model"]
        self.streak_model = data["streak_model"]
        self.scaler = data["scaler"]
        self.metrics = data["metrics"]
        self.feature_names = data.get("feature_names")
        self.n_features = data.get("n_features")
        self.is_trained = True

        # Store version info for compatibility checking
        self._saved_sklearn_version = data.get("sklearn_version")
        self._saved_at = data.get("saved_at")

        logger.info(f"Random Forest loaded from {path}")


# =============================================================================
# RIDGE TRAINER - Confidence Scoring (Now uses LightGBM)
# =============================================================================


def _create_lgbm_regressor(**kwargs) -> Any:
    """
    Create LightGBM regressor with GPU→CPU fallback.

    Tries GPU first (faster), falls back to CPU if unavailable.
    GPU uses smaller max_bin for speedup.

    Args:
        **kwargs: Additional LGBMRegressor parameters

    Returns:
        Configured LGBMRegressor instance
    """
    try:
        from lightgbm import LGBMRegressor
    except ImportError:
        logger.warning("LightGBM not installed, falling back to ElasticNetCV")
        return None

    import warnings
    import platform

    # Default hyperparameters optimized for confidence scoring
    default_params = {
        "n_estimators": 100,
        "learning_rate": 0.1,
        "max_depth": 6,
        "num_leaves": 31,
        "n_jobs": -1,
        "random_state": 42,
        "verbose": -1,
    }
    default_params.update(kwargs)

    # On macOS, LightGBM GPU is not supported (Metal not supported, only CUDA/OpenCL)
    # Skip GPU attempts to avoid [Fatal] error messages
    is_macos = platform.system() == "Darwin"
    devices_to_try = ["cpu"] if is_macos else ["gpu", "cuda", "cpu"]

    for device in devices_to_try:
        try:
            params = default_params.copy()
            params["device"] = device
            # Smaller bins for GPU speedup
            params["max_bin"] = 63 if device != "cpu" else 255

            model = LGBMRegressor(**params)

            # Test fit with tiny data to verify device works
            import numpy as np

            rng = np.random.default_rng(42)
            x_test = rng.random((10, 5)).astype(np.float32)
            y_test = rng.random(10).astype(np.float32)

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model.fit(x_test, y_test)

            logger.info(f"✓ LightGBM using device: {device}")

            # Return fresh model (the test model is discarded)
            return LGBMRegressor(**params)

        except Exception as e:
            if device != "cpu":
                logger.debug(f"LightGBM {device} unavailable: {e}")
            continue

    # Final fallback - CPU without test
    logger.info("⚠ LightGBM falling back to CPU (no GPU available)")
    params = default_params.copy()
    params["device"] = "cpu"
    params["max_bin"] = 255
    return LGBMRegressor(**params)


class RidgeTrainer(BaseTrainer):
    """
    LightGBM regression model for confidence/stability scoring.

    UPGRADED: Now uses LightGBM instead of ElasticNetCV for better accuracy.
    Falls back to ElasticNetCV if LightGBM is not installed.

    Features:
    - GPU acceleration with CPU fallback
    - Better handling of non-linear relationships
    - Faster training than ElasticNetCV
    - Backward compatible loading (handles both old ElasticNet and new LightGBM)

    Input: Rolling variance, volume changes, technical indicators
    Output: Confidence/stability score (0-100)
    """

    def __init__(self, config: Optional[TrainerConfig] = None):
        super().__init__(config)
        self.scaler = None
        self.feature_names = None
        self.n_features = None
        self.best_alpha = None
        self.best_l1_ratio = None
        self.n_nonzero_coefs = None

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        x_val: np.ndarray,
        y_val: np.ndarray,
        feature_names: Optional[list] = None,
    ) -> Dict[str, float]:
        """Train LightGBM for confidence scoring (falls back to ElasticNet if unavailable)."""
        from sklearn.model_selection import TimeSeriesSplit
        from sklearn.preprocessing import StandardScaler
        import pandas as pd

        # Save feature names for inference
        self.feature_names = feature_names
        self.n_features = X_train.shape[-1]

        # Scale features
        self.scaler = StandardScaler()
        x_train_scaled = self.scaler.fit_transform(X_train)
        x_val_scaled = self.scaler.transform(x_val)

        # === FIX: Convert to DataFrame with feature names ===
        # This ensures LightGBM stores feature names consistently to prevent
        # sklearn UserWarning: "X does not have valid feature names"
        if self.feature_names is not None:
            x_train_df = pd.DataFrame(x_train_scaled, columns=self.feature_names)
            x_val_df = pd.DataFrame(x_val_scaled, columns=self.feature_names)
        else:
            # Fallback: generate default feature names
            self.feature_names = [
                f"feature_{i}" for i in range(x_train_scaled.shape[1])
            ]
            x_train_df = pd.DataFrame(x_train_scaled, columns=self.feature_names)
            x_val_df = pd.DataFrame(x_val_scaled, columns=self.feature_names)

        # Try LightGBM first (GPU-accelerated)
        lgbm_model = _create_lgbm_regressor(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=6,
            num_leaves=31,
        )

        if lgbm_model is not None:
            # Use LightGBM
            logger.info("Training LightGBM (Confidence) - GPU-accelerated...")

            self.model = lgbm_model
            self.model.fit(
                x_train_df,
                y_train,  # Use DataFrame with feature names
                eval_set=[(x_val_df, y_val)],  # Use DataFrame with feature names
            )

            self.is_trained = True
            self._model_type = "lightgbm"

            # Calculate metrics - use DataFrame to avoid feature name warning
            y_pred = self.model.predict(x_val_df)
            mae = float(np.mean(np.abs(y_pred - y_val)))

            # R² score
            ss_res = np.sum((y_val - y_pred) ** 2)
            ss_tot = np.sum((y_val - np.mean(y_val)) ** 2)
            r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

            # Get feature importances
            importances = self.model.feature_importances_
            n_important = int(np.sum(importances > 0.01))

            self.metrics = {
                "confidence_mae": mae,
                "r2_score": float(r2),
                "model_type": "lightgbm",
                "n_estimators": self.model.n_estimators,
                "n_important_features": n_important,
                "n_total_features": self.n_features,
            }

            logger.info(
                f"LightGBM trained: MAE={mae:.2f}, R²={r2:.4f}, "
                f"important_features={n_important}/{self.n_features}"
            )
            return self.metrics

        # Fallback to ElasticNetCV
        from sklearn.linear_model import ElasticNetCV

        logger.info("Training ElasticNet (Confidence) with TimeSeriesSplit CV...")

        # Configure TimeSeriesSplit for temporal CV (prevents leakage)
        tscv = TimeSeriesSplit(n_splits=self.config.elasticnet_cv_splits)

        # Auto-generate alphas if not specified
        alphas = self.config.elasticnet_alphas
        if alphas is None:
            alphas = np.logspace(-4, 2, 50).tolist()

        # Train ElasticNetCV with automatic hyperparameter tuning
        self.model = ElasticNetCV(
            l1_ratio=self.config.elasticnet_l1_ratios,
            alphas=alphas,
            cv=tscv,
            max_iter=self.config.elasticnet_max_iter,
            n_jobs=-1,  # Parallel CV
            selection="random",  # Faster convergence
        )
        self.model.fit(x_train_scaled, y_train)

        self.is_trained = True
        self._model_type = "elasticnet"

        # Extract best hyperparameters
        self.best_alpha = float(self.model.alpha_)
        self.best_l1_ratio = float(self.model.l1_ratio_)
        self.n_nonzero_coefs = int(np.sum(self.model.coef_ != 0))

        # Calculate metrics
        y_pred = self.model.predict(x_val_scaled)
        mae = float(np.mean(np.abs(y_pred - y_val)))
        r2 = float(self.model.score(x_val_scaled, y_val))

        self.metrics = {
            "confidence_mae": mae,
            "r2_score": r2,
            "model_type": "elasticnet",
            "best_alpha": self.best_alpha,
            "best_l1_ratio": self.best_l1_ratio,
            "n_nonzero_coefs": self.n_nonzero_coefs,
            "n_total_coefs": self.n_features,
            "sparsity_ratio": 1.0 - (self.n_nonzero_coefs / self.n_features)
            if self.n_features > 0
            else 0.0,
        }

        logger.info(
            f"ElasticNet trained: MAE={mae:.2f}, R²={r2:.4f}, "
            f"alpha={self.best_alpha:.4f}, l1_ratio={self.best_l1_ratio:.2f}, "
            f"sparse={self.n_nonzero_coefs}/{self.n_features} features"
        )
        return self.metrics

    def predict(self, X: np.ndarray) -> Dict[str, Any]:
        """Predict confidence score (0-100)."""
        if not self.is_trained:
            raise RuntimeError(MODEL_NOT_TRAINED_ERROR)

        import pandas as pd

        # Scale
        if X.ndim == 1:
            X = X.reshape(1, -1)
        x_scaled = self.scaler.transform(X)

        # Get last row for prediction
        x_last = x_scaled[-1:] if len(x_scaled) > 1 else x_scaled

        # === FIX: Always use DataFrame for LightGBM ===
        # This prevents sklearn UserWarning: "X does not have valid feature names"
        # The model was trained with DataFrames that have feature names,
        # so predictions must also use DataFrames with the same feature names.
        model_type = getattr(self, "_model_type", "elasticnet")
        if model_type == "lightgbm":
            # Ensure feature_names are available
            if self.feature_names is None:
                # Generate default feature names if not set
                self.feature_names = [f"feature_{i}" for i in range(x_last.shape[1])]

            # Convert to DataFrame with feature names
            x_last = pd.DataFrame(x_last, columns=self.feature_names)

        confidence = float(self.model.predict(x_last)[0])

        # Clamp to 0-100
        confidence = max(0.0, min(100.0, confidence))

        return {
            "confidence": confidence,
        }

    def save(self, path: str) -> None:
        """Save confidence model with version metadata."""
        import sklearn

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        # Detect model type
        model_type = getattr(self, "_model_type", "elasticnet")
        if model_type == "lightgbm":
            try:
                import lightgbm

                lgbm_version = lightgbm.__version__
            except ImportError:
                lgbm_version = "unknown"
        else:
            lgbm_version = None

        data = {
            "model": self.model,
            "scaler": self.scaler,
            "metrics": self.metrics,
            "config": self.config.__dict__,
            "feature_names": self.feature_names,
            "n_features": self.n_features,
            # Model type and version metadata
            "model_type": model_type,
            "sklearn_version": sklearn.__version__,
            "lightgbm_version": lgbm_version,
            "saved_at": datetime.now().isoformat(),
        }

        with open(path, "wb") as f:
            pickle.dump(data, f)

        logger.info(f"Confidence model ({model_type}) saved to {path}")

    def load(self, path: str) -> None:
        """Load confidence model (supports both LightGBM and ElasticNet)."""
        import warnings

        # Suppress sklearn version warnings
        try:
            from sklearn.exceptions import InconsistentVersionWarning

            warnings.filterwarnings("ignore", category=InconsistentVersionWarning)
        except ImportError:
            pass
        warnings.filterwarnings("ignore", message=UNPICKLE_ESTIMATOR_WARNING)

        with open(path, "rb") as f:
            data = pickle.load(f)

        self.model = data["model"]
        self.scaler = data["scaler"]
        self.metrics = data["metrics"]
        self.feature_names = data.get("feature_names")
        self.n_features = data.get("n_features")
        self.is_trained = True

        # Detect model type from saved data or model class
        self._model_type = data.get("model_type")
        if self._model_type is None:
            # Infer from model class name
            model_class = type(self.model).__name__
            if "LGBM" in model_class or "LightGBM" in model_class:
                self._model_type = "lightgbm"
            else:
                self._model_type = "elasticnet"

        # Store version info for compatibility checking
        self._saved_sklearn_version = data.get("sklearn_version")
        self._saved_lightgbm_version = data.get("lightgbm_version")
        self._saved_at = data.get("saved_at")

        logger.info(f"Confidence model ({self._model_type}) loaded from {path}")


# =============================================================================
# REGIME-SPECIFIC LIGHTGBM TRAINER - 5 Separate Models by Market Regime
# =============================================================================

# LightGBM hyperparameters optimized for each market regime
REGIME_LGBM_PARAMS = {
    "STRONG_TREND": {
        # Deeper trees for trend following, more capacity to capture momentum
        "n_estimators": 150,
        "max_depth": 8,
        "num_leaves": 63,
        "learning_rate": 0.05,
        "feature_fraction": 0.7,
        "bagging_fraction": 0.8,
        "bagging_freq": 5,
        "min_child_samples": 10,
        "reg_alpha": 0.05,
        "reg_lambda": 0.05,
    },
    "WEAK_TREND": {
        # Conservative settings for noisy trend signals
        "n_estimators": 200,
        "max_depth": 6,
        "num_leaves": 31,
        "learning_rate": 0.03,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 5,
        "min_child_samples": 20,
        "reg_alpha": 0.1,
        "reg_lambda": 0.1,
    },
    "CHOP": {
        # Shallow trees, heavy regularization to avoid overfitting on noise
        "n_estimators": 100,
        "max_depth": 4,
        "num_leaves": 15,
        "learning_rate": 0.02,
        "feature_fraction": 0.9,
        "bagging_fraction": 0.9,
        "bagging_freq": 3,
        "min_child_samples": 30,
        "reg_alpha": 0.2,
        "reg_lambda": 0.2,
    },
    "MEAN_REVERT": {
        # More iterations to capture reversal patterns
        "n_estimators": 250,
        "max_depth": 6,
        "num_leaves": 31,
        "learning_rate": 0.04,
        "feature_fraction": 0.75,
        "bagging_fraction": 0.8,
        "bagging_freq": 5,
        "min_child_samples": 15,
        "reg_alpha": 0.08,
        "reg_lambda": 0.08,
    },
    "BREAKOUT": {
        # Fast learning for volatility expansion detection
        "n_estimators": 120,
        "max_depth": 7,
        "num_leaves": 45,
        "learning_rate": 0.06,
        "feature_fraction": 0.65,
        "bagging_fraction": 0.75,
        "bagging_freq": 5,
        "min_child_samples": 12,
        "reg_alpha": 0.05,
        "reg_lambda": 0.05,
    },
}

# Regime names for iteration
REGIME_NAMES_LIST = ["STRONG_TREND", "WEAK_TREND", "CHOP", "MEAN_REVERT", "BREAKOUT"]


def get_regime_lgbm_params(regime: str) -> Dict[str, Any]:
    """
    Get LightGBM hyperparameters optimized for a specific market regime.

    Args:
        regime: One of STRONG_TREND, WEAK_TREND, CHOP, MEAN_REVERT, BREAKOUT

    Returns:
        Dict of LightGBM hyperparameters
    """
    params = REGIME_LGBM_PARAMS.get(regime, REGIME_LGBM_PARAMS["WEAK_TREND"]).copy()

    # Add common params
    params.update(
        {
            "objective": "binary",
            "boosting_type": "gbdt",
            "class_weight": "balanced",
            "random_state": 42,
            "verbose": -1,
            "n_jobs": -1,
        }
    )

    return params


class RegimeLGBMTrainer(BaseTrainer):
    """
    Regime-specific LightGBM trainer that trains 5 separate models,
    one optimized for each market regime.

    Models are saved to: trained_data/models/{instrument}/lgbm_regime_{REGIME}.pkl

    Each regime has different hyperparameters:
    - STRONG_TREND: Deeper trees, less regularization, captures momentum
    - WEAK_TREND: Conservative, moderate regularization for noisy signals
    - CHOP: Shallow trees, heavy regularization to avoid noise overfitting
    - MEAN_REVERT: More iterations for reversal pattern learning
    - BREAKOUT: Fast learning for volatility expansion detection

    Input: Features + regime labels for filtering
    Output: Direction prediction per regime
    """

    def __init__(self, config: Optional[TrainerConfig] = None):
        super().__init__(config)
        self.regime_models: Dict[str, Any] = {}
        self.regime_scalers: Dict[str, Any] = {}
        self.regime_metrics: Dict[str, Dict[str, float]] = {}
        self.feature_names: Optional[List[str]] = None
        self.n_features: Optional[int] = None

    def train_regime_models(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        regimes_train: np.ndarray,
        x_val: np.ndarray,
        y_val: np.ndarray,
        regimes_val: np.ndarray,
        feature_names: Optional[List[str]] = None,
        min_samples_per_regime: int = 100,
    ) -> Dict[str, Dict[str, float]]:
        """
        Train 5 separate LightGBM models, one per market regime.

        Args:
            X_train: Training features
            y_train: Training labels (0=down, 1=up)
            regimes_train: Regime labels for each training sample (0-4)
            x_val: Validation features
            y_val: Validation labels
            regimes_val: Regime labels for validation samples
            feature_names: Feature names for LightGBM
            min_samples_per_regime: Skip regime if fewer samples

        Returns:
            Dict mapping regime name to metrics dict
        """
        try:
            import lightgbm  # noqa: F401 - verify availability
            del lightgbm
        except ImportError:
            logger.error(LGBM_NOT_INSTALLED_ERROR)
            raise

        # Import regime names from data loaders
        try:
            from src.core.modular_data_loaders import REGIME_NAMES
        except ImportError:
            REGIME_NAMES = {
                0: "STRONG_TREND",
                1: "WEAK_TREND",
                2: "CHOP",
                3: "MEAN_REVERT",
                4: "BREAKOUT",
            }

        self.feature_names = feature_names
        self.n_features = X_train.shape[-1]

        # Generate feature names if not provided
        if self.feature_names is None:
            self.feature_names = [f"feature_{i}" for i in range(self.n_features)]

        all_metrics = {}

        for regime_id in range(5):
            regime_name = REGIME_NAMES.get(regime_id, f"REGIME_{regime_id}")
            metrics = self._train_single_regime(
                regime_id, regime_name,
                X_train, y_train, regimes_train,
                x_val, y_val, regimes_val,
                min_samples_per_regime,
            )
            all_metrics[regime_name] = metrics

        self.is_trained = True
        return all_metrics

    def _train_single_regime(
        self,
        regime_id: int,
        regime_name: str,
        X_train: np.ndarray,
        y_train: np.ndarray,
        regimes_train: np.ndarray,
        x_val: np.ndarray,
        y_val: np.ndarray,
        regimes_val: np.ndarray,
        min_samples_per_regime: int,
    ) -> Dict[str, Any]:
        """Train a single regime-specific LightGBM model."""
        try:
            import lightgbm as lgb
        except ImportError:
            logger.error(LGBM_NOT_INSTALLED_ERROR)
            raise
        from sklearn.preprocessing import StandardScaler
        from sklearn.metrics import accuracy_score, f1_score
        import pandas as pd

        train_mask = regimes_train == regime_id
        val_mask = regimes_val == regime_id
        n_train = train_mask.sum()
        n_val = val_mask.sum()

        logger.info(f"\n{'=' * 60}")
        logger.info(f"Training {regime_name} model (train={n_train}, val={n_val})")
        logger.info(f"{'=' * 60}")

        if n_train < min_samples_per_regime:
            logger.warning(
                f"Skipping {regime_name}: only {n_train} samples "
                f"(min={min_samples_per_regime})"
            )
            return {
                "status": "skipped",
                "reason": f"insufficient_samples ({n_train})",
                "n_train": n_train,
                "n_val": n_val,
            }

        x_train_regime = X_train[train_mask]
        y_train_regime = y_train[train_mask]
        x_val_regime = x_val[val_mask] if n_val > 0 else x_train_regime[:100]
        y_val_regime = y_val[val_mask] if n_val > 0 else y_train_regime[:100]

        scaler = StandardScaler()
        x_train_scaled = scaler.fit_transform(x_train_regime)
        x_val_scaled = scaler.transform(x_val_regime)

        x_train_df = pd.DataFrame(x_train_scaled, columns=self.feature_names)
        x_val_df = pd.DataFrame(x_val_scaled, columns=self.feature_names)

        params = get_regime_lgbm_params(regime_name)
        model = lgb.LGBMClassifier(**params)

        try:
            model.fit(
                x_train_df,
                y_train_regime,
                eval_set=[(x_val_df, y_val_regime)],
            )
        except Exception as e:
            logger.error(f"Training failed for {regime_name}: {e}")
            return {"status": "failed", "reason": str(e)}

        y_pred_train = model.predict(x_train_df)
        y_pred_val = model.predict(x_val_df)

        train_acc = accuracy_score(y_train_regime, y_pred_train)
        val_acc = accuracy_score(y_val_regime, y_pred_val) if n_val > 0 else 0.0
        val_f1 = (
            f1_score(y_val_regime, y_pred_val, zero_division=0)
            if n_val > 0
            else 0.0
        )

        importances = model.feature_importances_
        top_features = sorted(
            zip(self.feature_names, importances), key=lambda x: x[1], reverse=True
        )[:5]

        self.regime_models[regime_name] = model
        self.regime_scalers[regime_name] = scaler

        metrics = {
            "status": "trained",
            "n_train": n_train,
            "n_val": n_val,
            "train_accuracy": float(train_acc),
            "val_accuracy": float(val_acc),
            "val_f1": float(val_f1),
            "params": params,
            "top_features": [(f, float(i)) for f, i in top_features],
        }

        self.regime_metrics[regime_name] = metrics
        logger.info(
            f"{regime_name}: train_acc={train_acc:.3f}, val_acc={val_acc:.3f}, "
            f"val_f1={val_f1:.3f}"
        )

        self.is_trained = True
        return metrics

    def predict(
        self,
        X: np.ndarray,
        regime: str = "WEAK_TREND",
    ) -> Dict[str, Any]:
        """
        Predict direction using regime-specific model.

        Args:
            X: Feature matrix (last row used for prediction)
            regime: Market regime name for model selection

        Returns:
            Dict with 'direction' (0/1) and 'probability'
        """
        import pandas as pd

        if not self.is_trained:
            raise RuntimeError("Models not trained. Call train_regime_models() first.")

        # Get model for regime (fallback to WEAK_TREND)
        model = self.regime_models.get(regime)
        scaler = self.regime_scalers.get(regime)

        if model is None:
            logger.warning(f"No model for regime {regime}, using WEAK_TREND")
            model = self.regime_models.get("WEAK_TREND")
            scaler = self.regime_scalers.get("WEAK_TREND")

        if model is None:
            raise RuntimeError("No trained models available")

        # Prepare input
        if X.ndim == 1:
            X = X.reshape(1, -1)
        x_last = X[-1:] if len(X) > 1 else X

        # Scale
        x_scaled = scaler.transform(x_last)
        x_df = pd.DataFrame(x_scaled, columns=self.feature_names)

        # Predict
        direction = int(model.predict(x_df)[0])
        proba = model.predict_proba(x_df)[0]
        probability = float(proba[1])  # Probability of up

        return {
            "direction": direction,
            "probability": probability,
            "regime_used": regime,
        }

    def save(
        self,
        base_path: str = PRODUCTION_MODELS_DIR,
        instrument: str = "EUR_USD"
    ) -> Dict[str, Path]:
        """
        Save all regime models to disk.

        Args:
            base_path: Base directory (e.g., 'trained_data/models')
            instrument: Currency pair name

        Returns:
            Dict mapping regime name to saved file path
        """
        save_dir = Path(base_path) / instrument
        save_dir.mkdir(parents=True, exist_ok=True)

        saved_paths = {}

        for regime_name in REGIME_NAMES_LIST:
            if regime_name not in self.regime_models:
                continue

            model_path = save_dir / f"lgbm_regime_{regime_name}.pkl"

            data = {
                "model": self.regime_models[regime_name],
                "scaler": self.regime_scalers[regime_name],
                "metrics": self.regime_metrics.get(regime_name, {}),
                "feature_names": self.feature_names,
                "n_features": self.n_features,
                "regime": regime_name,
                "saved_at": datetime.now().isoformat(),
            }

            with open(model_path, "wb") as f:
                pickle.dump(data, f)

            saved_paths[regime_name] = model_path
            logger.info(f"Saved {regime_name} model to {model_path}")

        # Save metadata
        meta_path = save_dir / "regime_lgbm_meta.json"
        import json

        meta = {
            "instrument": instrument,
            "regimes_trained": list(self.regime_models.keys()),
            "metrics": self.regime_metrics,
            "feature_names": self.feature_names,
            "n_features": self.n_features,
            "saved_at": datetime.now().isoformat(),
        }
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2, default=str)

        return saved_paths

    def load(
        self,
        base_path: str = PRODUCTION_MODELS_DIR,
        instrument: str = "EUR_USD"
    ) -> List[str]:
        """
        Load all available regime models from disk.

        Args:
            base_path: Base directory
            instrument: Currency pair name

        Returns:
            List of loaded regime names
        """
        load_dir = Path(base_path) / instrument
        loaded_regimes = []

        for regime_name in REGIME_NAMES_LIST:
            model_path = load_dir / f"lgbm_regime_{regime_name}.pkl"

            if not model_path.exists():
                continue

            with open(model_path, "rb") as f:
                data = pickle.load(f)

            self.regime_models[regime_name] = data["model"]
            self.regime_scalers[regime_name] = data["scaler"]
            self.regime_metrics[regime_name] = data.get("metrics", {})

            if self.feature_names is None:
                self.feature_names = data.get("feature_names")
                self.n_features = data.get("n_features")

            loaded_regimes.append(regime_name)
            logger.info(f"Loaded {regime_name} model from {model_path}")

        if loaded_regimes:
            self.is_trained = True

        return loaded_regimes

    def get_model_for_regime(self, regime: str) -> Optional[Any]:
        """Get the LightGBM model for a specific regime."""
        return self.regime_models.get(regime)

    def list_trained_regimes(self) -> List[str]:
        """List all trained regime names."""
        return list(self.regime_models.keys())


# =============================================================================
# LIGHTGBM MOMENTUM TRAINER - Replaces XGBoost for Joint Training (Phase 4)
# =============================================================================


def _create_lgbm_classifier(**kwargs) -> Any:
    """
    Create LightGBM classifier with GPU→CPU fallback.

    Tries GPU first (faster), falls back to CPU if unavailable.
    GPU uses smaller max_bin for speedup.

    Args:
        **kwargs: Additional LGBMClassifier parameters

    Returns:
        Configured LGBMClassifier instance
    """
    try:
        from lightgbm import LGBMClassifier
    except ImportError:
        logger.warning("LightGBM not installed, falling back to XGBoost")
        return None

    import warnings
    import platform

    # Default hyperparameters optimized for binary classification
    default_params = {
        "n_estimators": 150,
        "learning_rate": 0.05,
        "max_depth": 6,
        "num_leaves": 31,
        "n_jobs": -1,
        "random_state": 42,
        "verbose": -1,
        "class_weight": "balanced",
    }
    default_params.update(kwargs)

    # On macOS, LightGBM GPU is not supported (Metal not supported, only CUDA/OpenCL)
    is_macos = platform.system() == "Darwin"
    devices_to_try = ["cpu"] if is_macos else ["gpu", "cuda", "cpu"]

    for device in devices_to_try:
        try:
            params = default_params.copy()
            params["device"] = device
            params["max_bin"] = 63 if device != "cpu" else 255

            model = LGBMClassifier(**params)

            # Test fit with tiny data to verify device works
            rng = np.random.default_rng(42)
            x_test = rng.random((10, 5)).astype(np.float32)
            y_test = rng.integers(0, 2, 10)

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model.fit(x_test, y_test)

            logger.debug(f"✓ LightGBM Classifier using device: {device}")

            # Return fresh model (the test model is discarded)
            return LGBMClassifier(**params)

        except Exception as e:
            if device != "cpu":
                logger.debug(f"LightGBM {device} unavailable: {e}")
            continue

    # Final fallback - CPU without test
    params = default_params.copy()
    params["device"] = "cpu"
    params["max_bin"] = 255
    return LGBMClassifier(**params)


class LightGBMMomentumTrainer(BaseTrainer):
    """
    LightGBM model for momentum analysis (replaces XGBoost for Phase 4 joint training).

    Input: Lagged returns + spread dynamics
    Output:
        - momentum_score (0-1): How fast price is moving
        - acceleration (bool): Is momentum growing?

    Advantages over XGBoost:
    - Faster training with categorical feature support
    - Better memory efficiency for large datasets
    - Supports init_model for warm-start fine-tuning
    """

    def __init__(self, config: Optional[TrainerConfig] = None):
        super().__init__(config)
        self.momentum_model = None
        self.accel_model = None
        self.scaler = None
        self.feature_names = None
        self.n_features = None
        self.momentum_norm_factor = None

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        x_val: np.ndarray,
        y_val: np.ndarray,
        feature_names: Optional[list] = None,
        momentum_norm_factor: Optional[float] = None,
        init_momentum_model: Optional[Any] = None,
        init_accel_model: Optional[Any] = None,
    ) -> Dict[str, float]:
        """
        Train LightGBM for momentum analysis (2 models) with optional warm-start.

        Args:
            init_momentum_model: Pre-trained momentum model for fine-tuning
            init_accel_model: Pre-trained acceleration model for fine-tuning
        """
        try:
            import lightgbm  # noqa: F401 - availability check
        except ImportError:
            raise ImportError(LGBM_NOT_INSTALLED_ERROR)

        from sklearn.preprocessing import StandardScaler
        import pandas as pd

        self.momentum_norm_factor = momentum_norm_factor

        logger.info("Training LightGBM (Momentum)...")

        # Save feature names for inference
        self.feature_names = feature_names
        self.n_features = X_train.shape[-1]

        # Generate feature names if not provided
        if self.feature_names is None:
            self.feature_names = [f"feature_{i}" for i in range(self.n_features)]

        # Scale features
        self.scaler = StandardScaler()
        x_train_scaled = self.scaler.fit_transform(X_train)
        x_val_scaled = self.scaler.transform(x_val)

        # Convert to DataFrame for LightGBM
        x_train_df = pd.DataFrame(x_train_scaled, columns=self.feature_names)
        x_val_df = pd.DataFrame(x_val_scaled, columns=self.feature_names)

        # Split targets: y[:, 0] = momentum_score, y[:, 1] = acceleration
        # Use .copy() to avoid LightGBM warnings about sliced arrays doubling memory
        y_train_momentum = y_train[:, 0].copy()
        y_train_accel = y_train[:, 1].astype(int).copy()
        y_val_momentum = y_val[:, 0].copy()
        y_val_accel = y_val[:, 1].astype(int).copy()

        # Train momentum regressor
        self.momentum_model = _create_lgbm_regressor(
            n_estimators=150,
            learning_rate=0.05,
            max_depth=6,
            num_leaves=31,
        )

        if self.momentum_model is None:
            raise RuntimeError("Could not create LightGBM regressor")

        # Warm-start from init_model if provided (full re-optimization)
        self.momentum_model.fit(
            x_train_df,
            y_train_momentum,
            eval_set=[(x_val_df, y_val_momentum)],
            init_model=init_momentum_model,
        )

        # Train acceleration classifier
        self.accel_model = _create_lgbm_classifier(
            n_estimators=150,
            learning_rate=0.05,
            max_depth=6,
            num_leaves=31,
        )

        if self.accel_model is None:
            raise RuntimeError("Could not create LightGBM classifier")

        self.accel_model.fit(
            x_train_df,
            y_train_accel,
            eval_set=[(x_val_df, y_val_accel)],
            init_model=init_accel_model,
        )

        self.is_trained = True

        # Calculate metrics
        momentum_pred = self.momentum_model.predict(x_val_df)
        accel_pred = self.accel_model.predict(x_val_df)

        momentum_mae = float(np.mean(np.abs(momentum_pred - y_val_momentum)))
        accel_acc = float(np.mean(accel_pred == y_val_accel))

        self.metrics = {
            "momentum_mae": momentum_mae,
            "acceleration_accuracy": accel_acc,
            "model_type": "lightgbm",
        }

        logger.info(
            f"LightGBM Momentum trained: momentum_mae={momentum_mae:.4f}, accel_acc={accel_acc:.4f}"
        )
        return self.metrics

    def predict(self, X: np.ndarray) -> Dict[str, Any]:
        """Predict momentum score and acceleration."""
        import pandas as pd

        if not self.is_trained:
            raise RuntimeError(MODEL_NOT_TRAINED_ERROR)

        # Scale
        if X.ndim == 1:
            X = X.reshape(1, -1)
        x_scaled = self.scaler.transform(X)

        # Get last row for prediction
        x_last = x_scaled[-1:] if len(x_scaled) > 1 else x_scaled

        # Convert to DataFrame
        x_df = pd.DataFrame(x_last, columns=self.feature_names)

        momentum = float(self.momentum_model.predict(x_df)[0])
        acceleration = bool(self.accel_model.predict(x_df)[0])

        # Clamp momentum to 0-1
        momentum = max(0.0, min(1.0, momentum))

        return {
            "momentum": momentum,
            "acceleration": acceleration,
        }

    def save(self, path: str) -> None:
        """Save LightGBM momentum models with version metadata."""
        import sklearn

        try:
            import lightgbm

            lgbm_version = lightgbm.__version__
        except ImportError:
            lgbm_version = "unknown"

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "momentum_model": self.momentum_model,
            "accel_model": self.accel_model,
            "scaler": self.scaler,
            "metrics": self.metrics,
            "config": self.config.__dict__ if self.config else {},
            "feature_names": self.feature_names,
            "n_features": self.n_features,
            "momentum_norm_factor": self.momentum_norm_factor,
            "model_type": "lightgbm_momentum",
            "sklearn_version": sklearn.__version__,
            "lightgbm_version": lgbm_version,
            "saved_at": datetime.now().isoformat(),
        }

        with open(path, "wb") as f:
            pickle.dump(data, f)

        logger.info(f"LightGBM Momentum saved to {path}")

    def load(self, path: str) -> None:
        """Load LightGBM momentum models."""
        import warnings

        warnings.filterwarnings("ignore", message=SERIALIZED_MODEL_WARNING)

        with open(path, "rb") as f:
            data = pickle.load(f)

        self.momentum_model = data["momentum_model"]
        self.accel_model = data["accel_model"]
        self.scaler = data["scaler"]
        self.metrics = data["metrics"]
        self.feature_names = data.get("feature_names")
        self.n_features = data.get("n_features")
        self.momentum_norm_factor = data.get("momentum_norm_factor")
        self.is_trained = True

        logger.info(f"LightGBM Momentum loaded from {path}")

    def get_booster(self, model_type: str = "momentum") -> Any:
        """Get underlying booster for warm-start fine-tuning."""
        if model_type == "momentum":
            return self.momentum_model.booster_ if self.momentum_model else None
        else:
            return self.accel_model.booster_ if self.accel_model else None


# =============================================================================
# LIGHTGBM RISK TRAINER - Replaces RandomForest for Joint Training (Phase 4)
# =============================================================================


class LightGBMRiskTrainer(BaseTrainer):
    """
    LightGBM model for risk assessment (replaces RandomForest for Phase 4 joint training).

    Input: ATR, historical drawdowns, streak patterns
    Output:
        - expected_drawdown_pct: Max adverse excursion in next N bars
        - streak_prob: Probability losing streak continues

    Advantages over RandomForest:
    - Faster training especially on large datasets
    - Better handling of feature interactions
    - Supports init_model for warm-start fine-tuning
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
        x_val: np.ndarray,
        y_val: np.ndarray,
        feature_names: Optional[list] = None,
        init_drawdown_model: Optional[Any] = None,
        init_streak_model: Optional[Any] = None,
    ) -> Dict[str, float]:
        """
        Train LightGBM for risk assessment (2 models) with optional warm-start.

        Args:
            init_drawdown_model: Pre-trained drawdown model for fine-tuning
            init_streak_model: Pre-trained streak model for fine-tuning
        """
        try:
            import lightgbm  # noqa: F401 - availability check
        except ImportError:
            raise ImportError(LGBM_NOT_INSTALLED_ERROR)

        from sklearn.preprocessing import StandardScaler
        import pandas as pd

        logger.info("Training LightGBM (Risk)...")

        # Save feature names for inference
        self.feature_names = feature_names
        self.n_features = X_train.shape[-1]

        # Generate feature names if not provided
        if self.feature_names is None:
            self.feature_names = [f"feature_{i}" for i in range(self.n_features)]

        # Scale features
        self.scaler = StandardScaler()
        x_train_scaled = self.scaler.fit_transform(X_train)
        x_val_scaled = self.scaler.transform(x_val)

        # Convert to DataFrame for LightGBM
        x_train_df = pd.DataFrame(x_train_scaled, columns=self.feature_names)
        x_val_df = pd.DataFrame(x_val_scaled, columns=self.feature_names)

        # Split targets: y[:, 0] = drawdown_pct, y[:, 1] = streak_prob
        # Use .copy() to avoid LightGBM warnings about sliced arrays doubling memory
        y_train_drawdown = y_train[:, 0].copy()
        y_train_streak = y_train[:, 1].copy()
        y_val_drawdown = y_val[:, 0].copy()
        y_val_streak = y_val[:, 1].copy()

        # Train drawdown regressor
        self.drawdown_model = _create_lgbm_regressor(
            n_estimators=150,
            learning_rate=0.05,
            max_depth=6,
            num_leaves=31,
        )

        if self.drawdown_model is None:
            raise RuntimeError("Could not create LightGBM regressor for drawdown")

        self.drawdown_model.fit(
            x_train_df,
            y_train_drawdown,
            eval_set=[(x_val_df, y_val_drawdown)],
            init_model=init_drawdown_model,
        )

        # Train streak probability regressor
        self.streak_model = _create_lgbm_regressor(
            n_estimators=150,
            learning_rate=0.05,
            max_depth=6,
            num_leaves=31,
        )

        if self.streak_model is None:
            raise RuntimeError("Could not create LightGBM regressor for streak")

        self.streak_model.fit(
            x_train_df,
            y_train_streak,
            eval_set=[(x_val_df, y_val_streak)],
            init_model=init_streak_model,
        )

        self.is_trained = True

        # Calculate metrics
        drawdown_pred = self.drawdown_model.predict(x_val_df)
        streak_pred = self.streak_model.predict(x_val_df)

        # Clip predictions to valid range
        drawdown_pred = np.clip(drawdown_pred, 0, 0.10)
        streak_pred = np.clip(streak_pred, 0, 1.0)

        drawdown_mae = float(np.mean(np.abs(drawdown_pred - y_val_drawdown)))
        streak_mae = float(np.mean(np.abs(streak_pred - y_val_streak)))

        # Convert to basis points for display
        drawdown_mae_bps = drawdown_mae * 10000

        self.metrics = {
            "drawdown_mae_pct": drawdown_mae,
            "drawdown_mae_bps": drawdown_mae_bps,
            "streak_prob_mae": streak_mae,
            "model_type": "lightgbm",
        }

        logger.info(
            f"LightGBM Risk trained: drawdown_mae={drawdown_mae_bps:.1f} bps, streak_mae={streak_mae:.4f}"
        )
        return self.metrics

    def predict(self, X: np.ndarray) -> Dict[str, Any]:
        """Predict expected drawdown and streak probability."""
        import pandas as pd

        if not self.is_trained:
            raise RuntimeError(MODEL_NOT_TRAINED_ERROR)

        # Scale
        if X.ndim == 1:
            X = X.reshape(1, -1)
        x_scaled = self.scaler.transform(X)

        # Get last row for prediction
        x_last = x_scaled[-1:] if len(x_scaled) > 1 else x_scaled

        # Convert to DataFrame
        x_df = pd.DataFrame(x_last, columns=self.feature_names)

        expected_drawdown_pct = float(self.drawdown_model.predict(x_df)[0])
        streak_prob = float(self.streak_model.predict(x_df)[0])

        # Clamp values
        expected_drawdown_pct = max(0.0, min(0.10, expected_drawdown_pct))
        streak_prob = max(0.0, min(1.0, streak_prob))

        return {
            "expected_drawdown_pct": expected_drawdown_pct,
            "expected_drawdown_pips": expected_drawdown_pct * 10000,  # Legacy key
            "streak_prob": streak_prob,
        }

    def save(self, path: str) -> None:
        """Save LightGBM risk models with version metadata."""
        import sklearn

        try:
            import lightgbm

            lgbm_version = lightgbm.__version__
        except ImportError:
            lgbm_version = "unknown"

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "drawdown_model": self.drawdown_model,
            "streak_model": self.streak_model,
            "scaler": self.scaler,
            "metrics": self.metrics,
            "config": self.config.__dict__ if self.config else {},
            "feature_names": self.feature_names,
            "n_features": self.n_features,
            "model_type": "lightgbm_risk",
            "sklearn_version": sklearn.__version__,
            "lightgbm_version": lgbm_version,
            "saved_at": datetime.now().isoformat(),
        }

        with open(path, "wb") as f:
            pickle.dump(data, f)

        logger.info(f"LightGBM Risk saved to {path}")

    def load(self, path: str) -> None:
        """Load LightGBM risk models."""
        import warnings

        warnings.filterwarnings("ignore", message=SERIALIZED_MODEL_WARNING)

        with open(path, "rb") as f:
            data = pickle.load(f)

        self.drawdown_model = data["drawdown_model"]
        self.streak_model = data["streak_model"]
        self.scaler = data["scaler"]
        self.metrics = data["metrics"]
        self.feature_names = data.get("feature_names")
        self.n_features = data.get("n_features")
        self.is_trained = True

        logger.info(f"LightGBM Risk loaded from {path}")

    def get_booster(self, model_type: str = "drawdown") -> Any:
        """Get underlying booster for warm-start fine-tuning."""
        if model_type == "drawdown":
            return self.drawdown_model.booster_ if self.drawdown_model else None
        else:
            return self.streak_model.booster_ if self.streak_model else None


# =============================================================================
# JOINT MULTI-PAIR TRAINER - Orchestrates Joint Training (Phase 4)
# =============================================================================

# Valid instruments for joint training
JOINT_TRAINING_INSTRUMENTS = [
    "EUR_USD",
    "GBP_USD",
    "USD_JPY",
    "USD_CHF",
    "AUD_USD",
    "USD_CAD",
    "NZD_USD",
    "EUR_GBP",
    "EUR_JPY",
    "GBP_JPY",
]


class JointMultiPairTrainer:
    """
    Orchestrates joint training across multiple currency pairs.

    Combines data from multiple pairs with instrument one-hot encoding for
    transfer learning. Supports pair-specific fine-tuning with full re-optimization.

    Models trained:
    - TransformerDirectionTrainer: Direction prediction (Keras)
    - LightGBMMomentumTrainer: Momentum analysis (LightGBM)
    - LightGBMRiskTrainer: Risk assessment (LightGBM)
    - RidgeTrainer: Confidence scoring (LightGBM)

    Save location: trained_data/models/joint/
    Fine-tuned models: trained_data/models/{instrument}/
    """

    def __init__(self, config: Optional[TrainerConfig] = None):
        self.config = config or TrainerConfig()
        self.instruments: List[str] = []
        self.joint_data: Optional[Dict[str, Any]] = None
        self.normalization_stats: Dict[str, Dict] = {}

        # Individual trainers
        self.direction_trainer: Optional[TransformerDirectionTrainer] = None
        self.momentum_trainer: Optional[LightGBMMomentumTrainer] = None
        self.risk_trainer: Optional[LightGBMRiskTrainer] = None
        self.confidence_trainer: Optional[RidgeTrainer] = None

        # Per-instrument results
        self.per_instrument_metrics: Dict[str, Dict[str, float]] = {}
        self.joint_metrics: Dict[str, float] = {}

        # Fine-tune decisions
        self.fine_tune_decisions: Dict[str, bool] = {}

        self.is_trained = False

    def _append_model_result(
        self,
        combined_data: dict,
        model_key: str,
        result: dict,
        instrument: str,
        instruments: list,
        append_instrument_features,
    ) -> None:
        """Append a single model's data result to the combined data dict."""
        if result is None:
            return

        x_train, feature_names = append_instrument_features(
            result["X_train"],
            instrument,
            instruments,
            original_feature_names=result.get("feature_names"),
        )
        x_val, _ = append_instrument_features(
            result["X_val"],
            instrument,
            instruments,
            original_feature_names=result.get("feature_names"),
        )

        combined_data[model_key]["X_train"].append(x_train)
        combined_data[model_key]["y_train"].append(result["y_train"])
        combined_data[model_key]["X_val"].append(x_val)
        combined_data[model_key]["y_val"].append(result["y_val"])

        # Append weights if the model type supports them (direction)
        if "w_train" in combined_data[model_key]:
            combined_data[model_key]["w_train"].append(
                result.get("w_train", np.ones(len(result["y_train"])))
            )
            combined_data[model_key]["w_val"].append(
                result.get("w_val", np.ones(len(result["y_val"])))
            )

        if combined_data[model_key]["feature_names"] is None:
            combined_data[model_key]["feature_names"] = feature_names

    def _load_instrument_data(
        self,
        df,
        instrument: str,
        instruments: list,
        combined_data: dict,
        load_direction_data,
        load_xgboost_data,
        load_rf_data,
        load_ridge_data,
        append_instrument_features,
    ) -> None:
        """Load all model data for a single instrument and append to combined_data."""
        loaders = [
            ("direction", load_direction_data),
            ("momentum", load_xgboost_data),
            ("risk", load_rf_data),
            ("confidence", load_ridge_data),
        ]

        for model_key, loader_fn in loaders:
            result = loader_fn(df)
            self._append_model_result(
                combined_data, model_key, result,
                instrument, instruments, append_instrument_features,
            )

    def prepare_joint_data(
        self,
        dfs: Dict[str, "pd.DataFrame"],
        instruments: List[str],
    ) -> Dict[str, Dict[str, np.ndarray]]:
        """
        Prepare combined data with per-pair normalization and instrument encoding.

        Args:
            dfs: Dict mapping instrument to DataFrame with features
            instruments: List of instrument names to include

        Returns:
            Dict with combined data for each model type (direction, momentum, risk, confidence)
        """
        try:
            from src.core.modular_data_loaders import (
                append_instrument_features,
                load_direction_data,
                load_xgboost_data,
                load_rf_data,
                load_ridge_data,
            )
        except ImportError as e:
            logger.error(f"Could not import data loaders: {e}")
            raise

        self.instruments = instruments
        combined_data = {
            "direction": {
                "X_train": [],
                "y_train": [],
                "w_train": [],
                "X_val": [],
                "y_val": [],
                "w_val": [],
                "feature_names": None,
            },
            "momentum": {
                "X_train": [],
                "y_train": [],
                "X_val": [],
                "y_val": [],
                "feature_names": None,
            },
            "risk": {
                "X_train": [],
                "y_train": [],
                "X_val": [],
                "y_val": [],
                "feature_names": None,
            },
            "confidence": {
                "X_train": [],
                "y_train": [],
                "X_val": [],
                "y_val": [],
                "feature_names": None,
            },
        }

        for instrument in instruments:
            if instrument not in dfs:
                logger.warning(f"Skipping {instrument}: not in provided DataFrames")
                continue

            df = dfs[instrument]
            logger.info(f"Processing {instrument}: {len(df)} rows")

            try:
                self._load_instrument_data(
                    df, instrument, instruments, combined_data,
                    load_direction_data, load_xgboost_data,
                    load_rf_data, load_ridge_data,
                    append_instrument_features,
                )
            except Exception as e:
                logger.error(f"Error loading data for {instrument}: {e}")
                continue

        # Concatenate all instruments
        for model_type in combined_data:
            if combined_data[model_type]["X_train"]:
                combined_data[model_type]["X_train"] = np.vstack(
                    combined_data[model_type]["X_train"]
                )
                combined_data[model_type]["y_train"] = np.concatenate(
                    combined_data[model_type]["y_train"]
                )
                combined_data[model_type]["X_val"] = np.vstack(
                    combined_data[model_type]["X_val"]
                )
                combined_data[model_type]["y_val"] = np.concatenate(
                    combined_data[model_type]["y_val"]
                )

                # Concatenate weights if present (for direction model)
                if (
                    "w_train" in combined_data[model_type]
                    and combined_data[model_type]["w_train"]
                ):
                    combined_data[model_type]["w_train"] = np.concatenate(
                        combined_data[model_type]["w_train"]
                    )
                    combined_data[model_type]["w_val"] = np.concatenate(
                        combined_data[model_type]["w_val"]
                    )

                logger.info(
                    f"Combined {model_type}: train={len(combined_data[model_type]['X_train'])}, "
                    f"val={len(combined_data[model_type]['X_val'])}"
                )

        self.joint_data = combined_data
        return combined_data

    def train(
        self,
        dfs: Dict[str, "pd.DataFrame"],
        instruments: List[str],
        save_dir: str = JOINT_MODELS_DIR,
    ) -> Dict[str, Dict[str, float]]:
        """
        Train all models on joint data from multiple instruments.

        Args:
            dfs: Dict mapping instrument to DataFrame
            instruments: List of instruments to include
            save_dir: Directory to save joint models

        Returns:
            Dict with metrics for each model type
        """
        logger.info("\n" + "=" * 60)
        logger.info("JOINT MULTI-PAIR TRAINING")
        logger.info(f"Instruments: {instruments}")
        logger.info("=" * 60)

        # Prepare joint data
        data = self.prepare_joint_data(dfs, instruments)

        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)

        results = {}

        # 1. Train Direction (Transformer)
        if (
            data["direction"]["X_train"] is not None
            and len(data["direction"]["X_train"]) > 0
        ):
            logger.info("\n" + "-" * 50)
            logger.info("Training Joint Transformer (Direction)")
            logger.info("-" * 50)

            self.direction_trainer = TransformerDirectionTrainer(self.config)
            # Use "JOINT" + instrument list for multi-pair training identification
            joint_instrument = (
                f"JOINT_{'+'.join(instruments[:3])}"
                if len(instruments) <= 3
                else f"JOINT_{len(instruments)}pairs"
            )
            metrics = self.direction_trainer.train(
                data["direction"]["X_train"],
                data["direction"]["y_train"],
                data["direction"]["X_val"],
                data["direction"]["y_val"],
                feature_names=data["direction"]["feature_names"],
                w_train=data["direction"].get(
                    "w_train"
                ),  # Pass weights to filter unclear samples
                w_val=data["direction"].get("w_val"),
                skip_scaling=True,  # Data from load_direction_data is already scaled
                instrument=joint_instrument,
            )
            self.direction_trainer.save(str(save_path / TRANSFORMER_DIRECTION_FILENAME))
            results["direction"] = metrics

        # 2. Train Momentum (LightGBM)
        if (
            data["momentum"]["X_train"] is not None
            and len(data["momentum"]["X_train"]) > 0
        ):
            logger.info("\n" + "-" * 50)
            logger.info("Training Joint LightGBM (Momentum)")
            logger.info("-" * 50)

            self.momentum_trainer = LightGBMMomentumTrainer(self.config)
            metrics = self.momentum_trainer.train(
                data["momentum"]["X_train"],
                data["momentum"]["y_train"],
                data["momentum"]["X_val"],
                data["momentum"]["y_val"],
                feature_names=data["momentum"]["feature_names"],
            )
            self.momentum_trainer.save(str(save_path / LGBM_MOMENTUM_FILENAME))
            results["momentum"] = metrics

        # 3. Train Risk (LightGBM)
        if data["risk"]["X_train"] is not None and len(data["risk"]["X_train"]) > 0:
            logger.info("\n" + "-" * 50)
            logger.info("Training Joint LightGBM (Risk)")
            logger.info("-" * 50)

            self.risk_trainer = LightGBMRiskTrainer(self.config)
            metrics = self.risk_trainer.train(
                data["risk"]["X_train"],
                data["risk"]["y_train"],
                data["risk"]["X_val"],
                data["risk"]["y_val"],
                feature_names=data["risk"]["feature_names"],
            )
            self.risk_trainer.save(str(save_path / LGBM_RISK_FILENAME))
            results["risk"] = metrics

        # 4. Train Confidence (Ridge/LightGBM)
        if (
            data["confidence"]["X_train"] is not None
            and len(data["confidence"]["X_train"]) > 0
        ):
            logger.info("\n" + "-" * 50)
            logger.info("Training Joint Ridge/LightGBM (Confidence)")
            logger.info("-" * 50)

            self.confidence_trainer = RidgeTrainer(self.config)
            metrics = self.confidence_trainer.train(
                data["confidence"]["X_train"],
                data["confidence"]["y_train"],
                data["confidence"]["X_val"],
                data["confidence"]["y_val"],
                feature_names=data["confidence"]["feature_names"],
            )
            self.confidence_trainer.save(str(save_path / RIDGE_CONFIDENCE_FILENAME))
            results["confidence"] = metrics

        self.joint_metrics = results
        self.is_trained = True

        # Save metadata
        self._save_metadata(save_path, instruments, results)

        logger.info("\n" + "=" * 60)
        logger.info("Joint training complete!")
        logger.info(f"Models saved to: {save_path}")
        logger.info("=" * 60)

        return results

    def evaluate_per_instrument(
        self,
        dfs: Dict[str, "pd.DataFrame"],
        instruments: List[str],
    ) -> Dict[str, Dict[str, float]]:
        """
        Evaluate joint models on each instrument separately.

        Returns accuracy per instrument to identify fine-tuning candidates.
        """
        if not self.is_trained:
            raise RuntimeError("Models not trained. Call train() first.")

        try:
            from src.core.modular_data_loaders import (
                append_instrument_features,
                load_direction_data,
            )
        except ImportError as e:
            logger.error(f"Could not import data loaders: {e}")
            raise

        results = {}

        for instrument in instruments:
            if instrument not in dfs:
                continue

            try:
                result = self._evaluate_instrument_direction(
                    dfs[instrument], instrument, instruments,
                    load_direction_data, append_instrument_features,
                )
                if result is not None:
                    results[instrument] = result
            except Exception as e:
                logger.error(f"Error evaluating {instrument}: {e}")
                results[instrument] = {"error": str(e)}

        self.per_instrument_metrics = results
        return results

    def _extract_features_from_df(
        self, df, instrument: str, expected_feature_names: list
    ) -> Optional[np.ndarray]:
        """Extract validation features from DataFrame using expected feature names."""
        instrument_cols = [f for f in expected_feature_names if f.startswith('instrument_')]
        base_features = [f for f in expected_feature_names if not f.startswith('instrument_')]
        available_base = [f for f in base_features if f in df.columns]

        if len(available_base) != len(base_features):
            missing = set(base_features) - set(available_base)
            logger.warning(f"{instrument}: Missing {len(missing)} base features: {list(missing)[:5]}...")
            return None

        n = len(df)
        train_end = int(n * 0.7)
        val_end = int(n * 0.9)
        x_val_base = df[available_base].iloc[train_end:val_end].values.astype(np.float32)
        x_val_base = np.nan_to_num(x_val_base, nan=0.0, posinf=0.0, neginf=0.0)

        if not instrument_cols:
            logger.debug(f"{instrument}: Extracted {len(available_base)} features directly from df")
            return x_val_base

        n_samples = x_val_base.shape[0]
        instrument_onehot = np.zeros((n_samples, len(instrument_cols)), dtype=np.float32)
        current_col = f"instrument_{instrument}"
        if current_col in instrument_cols:
            col_idx = instrument_cols.index(current_col)
            instrument_onehot[:, col_idx] = 1.0
        logger.debug(f"{instrument}: Extracted {len(base_features)} base + {len(instrument_cols)} instrument features")
        return np.hstack([x_val_base, instrument_onehot])

    def _align_eval_features(
        self, x_val_2d: np.ndarray, instrument: str, instruments: list,
        expected_n_features: Optional[int], append_instrument_features,
    ) -> np.ndarray:
        """Align evaluation features to match model expectations."""
        if expected_n_features is None:
            return x_val_2d

        n_instrument_features = len(instruments)
        if expected_n_features == x_val_2d.shape[-1] + n_instrument_features:
            x_val_2d, _ = append_instrument_features(x_val_2d, instrument, instruments)
            logger.debug(f"{instrument}: Added instrument features, shape {x_val_2d.shape}")
            return x_val_2d

        if x_val_2d.shape[-1] == expected_n_features:
            return x_val_2d

        if x_val_2d.shape[-1] > expected_n_features:
            logger.warning(
                f"{instrument}: Feature mismatch - data has {x_val_2d.shape[-1]}, "
                f"model expects {expected_n_features}. Using first {expected_n_features} features."
            )
            return x_val_2d[:, :expected_n_features]

        n_pad = expected_n_features - x_val_2d.shape[-1]
        logger.warning(
            f"{instrument}: Feature mismatch - data has {x_val_2d.shape[-1]}, "
            f"model expects {expected_n_features}. Padding with {n_pad} zero columns."
        )
        padding = np.zeros((x_val_2d.shape[0], n_pad), dtype=np.float32)
        return np.hstack([x_val_2d, padding])

    def _evaluate_instrument_direction(
        self, df, instrument: str, instruments: list,
        load_direction_data, append_instrument_features,
    ) -> Optional[Dict[str, float]]:
        """Evaluate direction model for a single instrument."""
        dir_result = load_direction_data(df)
        if not dir_result or not self.direction_trainer:
            return None

        expected_n_features = self.direction_trainer.n_features
        expected_feature_names = self.direction_trainer.feature_names

        # Try extracting features directly from DataFrame
        x_val_2d = None
        if expected_feature_names:
            x_val_2d = self._extract_features_from_df(df, instrument, expected_feature_names)

        if x_val_2d is None:
            x_val_2d = dir_result["X_val"]

        x_val_2d = self._align_eval_features(
            x_val_2d, instrument, instruments,
            expected_n_features, append_instrument_features,
        )

        seq_len = get_config_seq_len(self.config)
        x_val_seq, y_val_seq, _ = create_sequences_with_weights(
            x_val_2d, dir_result["y_val"], dir_result.get("w_val"), seq_len
        )

        if len(x_val_seq) == 0:
            logger.warning(f"{instrument}: No sequences created for evaluation")
            return None

        preds = self.direction_trainer.model.predict(x_val_seq)
        y_pred = (preds > 0.5).astype(int).flatten()
        y_true = y_val_seq[:len(y_pred)]

        accuracy = float(np.mean(y_pred == y_true))
        logger.info(f"{instrument}: direction_accuracy={accuracy:.4f}")
        return {"direction_accuracy": accuracy}

    def should_fine_tune(
        self,
        performance_threshold: float = 0.05,
    ) -> Dict[str, bool]:
        """
        Determine which pairs need fine-tuning.

        Fine-tune if accuracy is >threshold below average.

        Args:
            performance_threshold: Fine-tune if accuracy below avg - threshold

        Returns:
            Dict mapping instrument to should_fine_tune (True/False)
        """
        if not self.per_instrument_metrics:
            logger.warning(
                "No per-instrument metrics. Call evaluate_per_instrument() first."
            )
            return {}

        # Get accuracies
        accuracies = []
        for instrument, metrics in self.per_instrument_metrics.items():
            if "direction_accuracy" in metrics:
                accuracies.append((instrument, metrics["direction_accuracy"]))

        if not accuracies:
            return {}

        avg_accuracy = np.mean([a for _, a in accuracies])

        decisions = {}
        for instrument, accuracy in accuracies:
            needs_tuning = (avg_accuracy - accuracy) > performance_threshold
            decisions[instrument] = needs_tuning

            if needs_tuning:
                logger.info(
                    f"{instrument} needs fine-tuning: "
                    f"{accuracy:.2%} vs avg {avg_accuracy:.2%} "
                    f"(gap={avg_accuracy - accuracy:.2%})"
                )
            else:
                logger.info(f"{instrument} OK: {accuracy:.2%}")

        self.fine_tune_decisions = decisions
        return decisions

    def fine_tune_for_pair(
        self,
        df: "pd.DataFrame",
        instrument: str,
        save_dir: str = PRODUCTION_MODELS_DIR,
        _fine_tune_epochs: int = 10,  # Reserved for future Transformer fine-tuning
        _fine_tune_lr_factor: float = 0.1,  # Reserved for future use
    ) -> Dict[str, float]:
        """
        Fine-tune joint models for specific pair with full re-optimization.

        Uses LightGBM init_model for warm-start, allowing full re-optimization
        rather than freezing early trees.

        Args:
            df: DataFrame for the instrument
            instrument: Currency pair name
            save_dir: Base directory for saving
            fine_tune_epochs: Number of epochs for Transformer fine-tuning
            fine_tune_lr_factor: Learning rate multiplier (0.1 = 10% of original)

        Returns:
            Dict with fine-tuned metrics
        """
        if not self.is_trained:
            raise RuntimeError("Joint models not trained. Call train() first.")

        try:
            from src.core.modular_data_loaders import (
                append_instrument_features,
                load_xgboost_data,
                load_rf_data,
                load_ridge_data,
            )
        except ImportError as e:
            logger.error(f"Could not import data loaders: {e}")
            raise

        logger.info("\n" + "-" * 50)
        logger.info(f"Fine-tuning for {instrument}")
        logger.info("-" * 50)

        pair_save_dir = Path(save_dir) / instrument
        pair_save_dir.mkdir(parents=True, exist_ok=True)

        results = {}

        fine_tune_specs = [
            {
                "name": "momentum",
                "trainer": self.momentum_trainer,
                "loader": load_xgboost_data,
                "trainer_cls": LightGBMMomentumTrainer,
                "filename": LGBM_MOMENTUM_FILENAME,
                "init_kwargs_fn": lambda t: {
                    "init_momentum_model": t.get_booster("momentum"),
                    "init_accel_model": t.get_booster("accel"),
                },
            },
            {
                "name": "risk",
                "trainer": self.risk_trainer,
                "loader": load_rf_data,
                "trainer_cls": LightGBMRiskTrainer,
                "filename": LGBM_RISK_FILENAME,
                "init_kwargs_fn": lambda t: {
                    "init_drawdown_model": t.get_booster("drawdown"),
                    "init_streak_model": t.get_booster("streak"),
                },
            },
            {
                "name": "confidence",
                "trainer": self.confidence_trainer,
                "loader": load_ridge_data,
                "trainer_cls": RidgeTrainer,
                "filename": RIDGE_CONFIDENCE_FILENAME,
                "init_kwargs_fn": lambda _t: {},
            },
        ]

        for spec in fine_tune_specs:
            self._fine_tune_single_model(
                spec, df, instrument, pair_save_dir, append_instrument_features, results
            )

        logger.info(f"Fine-tuned models saved to: {pair_save_dir}")
        return results

    def _fine_tune_single_model(
        self,
        spec: Dict,
        df: "pd.DataFrame",
        instrument: str,
        pair_save_dir: Path,
        append_instrument_features,
        results: Dict,
    ) -> None:
        """Fine-tune a single model type using the spec dict."""
        trainer = spec["trainer"]
        if not trainer or not trainer.is_trained:
            return
        try:
            data_result = spec["loader"](df)
            if not data_result:
                return
            x_train, feature_names = append_instrument_features(
                data_result["X_train"], instrument, self.instruments
            )
            x_val, _ = append_instrument_features(
                data_result["X_val"], instrument, self.instruments
            )
            init_kwargs = spec["init_kwargs_fn"](trainer)
            new_trainer = spec["trainer_cls"](self.config)
            metrics = new_trainer.train(
                x_train,
                data_result["y_train"],
                x_val,
                data_result["y_val"],
                feature_names=feature_names,
                **init_kwargs,
            )
            new_trainer.save(str(pair_save_dir / spec["filename"]))
            results[spec["name"]] = metrics
            logger.info(f"  {spec['name'].title()} fine-tuned: {metrics}")
        except Exception as e:
            logger.error(f"  {spec['name'].title()} fine-tune failed: {e}")

    def _save_metadata(
        self,
        save_path: Path,
        instruments: List[str],
        results: Dict[str, Dict],
    ) -> None:
        """Save joint training metadata."""
        import json

        meta = {
            "training_type": "joint_multi_pair",
            "instruments": instruments,
            "metrics": results,
            "per_instrument_metrics": self.per_instrument_metrics,
            "fine_tune_decisions": self.fine_tune_decisions,
            "saved_at": datetime.now().isoformat(),
        }

        meta_path = save_path / "joint_training_meta.json"
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2, default=str)

        logger.info(f"Metadata saved to: {meta_path}")

    def save(self, save_dir: str = JOINT_MODELS_DIR) -> Dict[str, Path]:
        """Save all joint models."""
        if not self.is_trained:
            raise RuntimeError("Models not trained")

        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)

        paths = {}

        if self.direction_trainer:
            p = save_path / TRANSFORMER_DIRECTION_FILENAME
            self.direction_trainer.save(str(p))
            paths["direction"] = p

        if self.momentum_trainer:
            p = save_path / LGBM_MOMENTUM_FILENAME
            self.momentum_trainer.save(str(p))
            paths["momentum"] = p

        if self.risk_trainer:
            p = save_path / LGBM_RISK_FILENAME
            self.risk_trainer.save(str(p))
            paths["risk"] = p

        if self.confidence_trainer:
            p = save_path / RIDGE_CONFIDENCE_FILENAME
            self.confidence_trainer.save(str(p))
            paths["confidence"] = p

        return paths

    def load(self, load_dir: str = JOINT_MODELS_DIR) -> bool:
        """Load all joint models."""
        load_path = Path(load_dir)

        if not load_path.exists():
            logger.warning(f"Joint model directory not found: {load_path}")
            return False

        loaded = False

        # Direction (Transformer)
        dir_path = load_path / TRANSFORMER_DIRECTION_FILENAME
        if dir_path.exists():
            self.direction_trainer = TransformerDirectionTrainer(self.config)
            self.direction_trainer.load(str(dir_path))
            loaded = True

        # Momentum (LightGBM)
        mom_path = load_path / LGBM_MOMENTUM_FILENAME
        if mom_path.exists():
            self.momentum_trainer = LightGBMMomentumTrainer(self.config)
            self.momentum_trainer.load(str(mom_path))
            loaded = True

        # Risk (LightGBM)
        risk_path = load_path / LGBM_RISK_FILENAME
        if risk_path.exists():
            self.risk_trainer = LightGBMRiskTrainer(self.config)
            self.risk_trainer.load(str(risk_path))
            loaded = True

        # Confidence (Ridge)
        conf_path = load_path / RIDGE_CONFIDENCE_FILENAME
        if conf_path.exists():
            self.confidence_trainer = RidgeTrainer(self.config)
            self.confidence_trainer.load(str(conf_path))
            loaded = True

        self.is_trained = loaded
        return loaded


# =============================================================================
# HISTGRADIENTBOOSTING TRAINER - Direction Baseline (sklearn)
# =============================================================================


class HistGradientBoostingDirectionTrainer(BaseTrainer):
    """
    HistGradientBoostingClassifier for direction prediction.

    This is a fast, accurate sklearn-based baseline that:
    - Handles NaN/missing values natively
    - Uses histogram-based splits for speed
    - Provides competitive accuracy without deep learning overhead

    Use as sanity check: If Transformer beats this by <2%, Transformer may be overfitting.
    If this baseline achieves 55-60%, the features are informative.

    Input: Flattened feature matrix (no sequences needed)
    Output: Binary direction (0=down, 1=up) with probabilities
    """

    def __init__(self, config: Optional[TrainerConfig] = None):
        super().__init__(config)
        self.scaler = None
        self.pca = None
        self.feature_names = None
        self.n_features = None

        # HistGB specific config
        self.max_iter = getattr(config, "histgb_max_iter", 200) if config else 200
        self.max_depth = getattr(config, "histgb_max_depth", 8) if config else 8
        self.learning_rate = (
            getattr(config, "histgb_learning_rate", 0.05) if config else 0.05
        )
        self.l2_regularization = (
            getattr(config, "histgb_l2_reg", 0.1) if config else 0.1
        )
        self.use_pca = getattr(config, "histgb_use_pca", True) if config else True
        self.pca_variance = (
            getattr(config, "histgb_pca_variance", 0.95) if config else 0.95
        )

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        x_val: np.ndarray,
        y_val: np.ndarray,
        feature_names: Optional[list] = None,
        w_train: Optional[np.ndarray] = None,
        w_val: Optional[np.ndarray] = None,
    ) -> Dict[str, float]:
        """
        Train HistGradientBoostingClassifier for direction.

        Uses optional PCA to reduce dimensionality and prevent overfitting.
        """
        from sklearn.ensemble import HistGradientBoostingClassifier
        from sklearn.preprocessing import StandardScaler
        from sklearn.decomposition import PCA

        logger.info("Training HistGradientBoosting (Direction Baseline)...")

        # Save feature names for inference
        self.feature_names = feature_names
        self.n_features = X_train.shape[-1]

        # Flatten if 3D (sequences) - use last timestep or mean
        if X_train.ndim == 3:
            logger.info(f"Flattening 3D input: {X_train.shape} -> using last timestep")
            X_train = X_train[:, -1, :]  # Use last timestep
            x_val = x_val[:, -1, :]

        # Filter by weights if provided (keep only clear labels)
        if w_train is not None:
            clear_mask_train = w_train > 0
            clear_mask_val = (
                w_val > 0 if w_val is not None else np.ones(len(y_val), dtype=bool)
            )
            x_train_filtered = X_train[clear_mask_train]
            y_train_filtered = y_train[clear_mask_train]
            x_val_filtered = x_val[clear_mask_val]
            y_val_filtered = y_val[clear_mask_val]
            logger.info(
                f"Filtered to clear labels: train={len(x_train_filtered)}, val={len(x_val_filtered)}"
            )
        else:
            x_train_filtered = X_train
            y_train_filtered = y_train
            x_val_filtered = x_val
            y_val_filtered = y_val

        # Ensure labels are integer (sklearn classifier requires discrete labels)
        y_train_filtered = np.asarray(y_train_filtered).astype(int)
        y_val_filtered = np.asarray(y_val_filtered).astype(int)

        # Scale features
        self.scaler = StandardScaler()
        x_train_scaled = self.scaler.fit_transform(x_train_filtered)
        x_val_scaled = self.scaler.transform(x_val_filtered)

        # Optional PCA for dimensionality reduction
        if self.use_pca and x_train_scaled.shape[1] > 20:
            self.pca = PCA(n_components=self.pca_variance, random_state=42)
            x_train_pca = self.pca.fit_transform(x_train_scaled)
            x_val_pca = self.pca.transform(x_val_scaled)
            logger.info(
                f"PCA: {x_train_scaled.shape[1]} features -> {x_train_pca.shape[1]} components "
                f"(explaining {self.pca_variance * 100:.0f}% variance)"
            )
            x_train_final = x_train_pca
            x_val_final = x_val_pca
        else:
            x_train_final = x_train_scaled
            x_val_final = x_val_scaled
            self.pca = None

        # Log class distribution
        train_up_pct = (y_train_filtered == 1).mean() * 100
        val_up_pct = (y_val_filtered == 1).mean() * 100
        logger.info(
            "Class distribution: train=%.1f%% up, val=%.1f%% up",
            train_up_pct, val_up_pct
        )

        # Train HistGradientBoosting with early stopping
        self.model = HistGradientBoostingClassifier(
            max_iter=self.max_iter,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            l2_regularization=self.l2_regularization,
            early_stopping=True,
            validation_fraction=0.15,
            n_iter_no_change=20,
            random_state=42,
            verbose=0,
        )

        self.model.fit(x_train_final, y_train_filtered)
        self.is_trained = True

        # Evaluate
        y_pred = self.model.predict(x_val_final)
        y_prob = self.model.predict_proba(x_val_final)[:, 1]

        # Metrics
        val_acc = np.mean(y_pred == y_val_filtered)

        # Balanced accuracy
        up_mask = y_val_filtered == 1
        down_mask = y_val_filtered == 0
        up_acc = np.mean(y_pred[up_mask] == 1) if up_mask.sum() > 0 else 0
        down_acc = np.mean(y_pred[down_mask] == 0) if down_mask.sum() > 0 else 0
        balanced_acc = (up_acc + down_acc) / 2

        # AUC
        try:
            from sklearn.metrics import roc_auc_score

            auc = roc_auc_score(y_val_filtered, y_prob)
        except Exception:
            auc = 0.5

        self.metrics = {
            "val_accuracy": float(val_acc),
            "val_balanced_accuracy": float(balanced_acc),
            "val_up_accuracy": float(up_acc),
            "val_down_accuracy": float(down_acc),
            "auc": float(auc),
            "n_train_samples": len(x_train_filtered),
            "n_val_samples": len(x_val_filtered),
            "n_features_used": x_train_final.shape[1],
            "n_iterations": self.model.n_iter_,
        }

        logger.info(
            f"HistGB trained: val_accuracy={val_acc:.4f}, balanced={balanced_acc:.4f}, "
            f"auc={auc:.4f}, iters={self.model.n_iter_}"
        )
        return self.metrics

    def predict(self, X: np.ndarray) -> Dict[str, Any]:
        """Predict direction (0 or 1) with probability."""
        if not self.is_trained:
            raise RuntimeError(MODEL_NOT_TRAINED_ERROR)

        # Flatten if 3D
        if X.ndim == 3:
            X = X[:, -1, :]

        # Scale
        x_scaled = self.scaler.transform(X.reshape(-1, X.shape[-1]))

        # Apply PCA if used
        if self.pca is not None:
            x_final = self.pca.transform(x_scaled)
        else:
            x_final = x_scaled

        # Get last row for prediction
        x_last = x_final[-1:] if len(x_final) > 1 else x_final

        prob = float(self.model.predict_proba(x_last)[0, 1])
        direction = int(self.model.predict(x_last)[0])

        return {
            "direction": direction,
            "probability": prob,
        }

    def save(self, path: str) -> None:
        """Save HistGB model."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "model": self.model,
            "scaler": self.scaler,
            "pca": self.pca,
            "metrics": self.metrics,
            "config": self.config.__dict__ if self.config else {},
            "feature_names": self.feature_names,
            "n_features": self.n_features,
            "model_type": "histgb",
        }

        with open(path, "wb") as f:
            pickle.dump(data, f)

        logger.info(f"HistGB saved to {path}")

    def load(self, path: str) -> None:
        """Load HistGB model."""
        import warnings

        # Suppress sklearn version warnings
        try:
            from sklearn.exceptions import InconsistentVersionWarning

            warnings.filterwarnings("ignore", category=InconsistentVersionWarning)
        except ImportError:
            pass
        warnings.filterwarnings("ignore", message=UNPICKLE_ESTIMATOR_WARNING)

        with open(path, "rb") as f:
            data = pickle.load(f)

        self.model = data["model"]
        self.scaler = data["scaler"]
        self.pca = data.get("pca")
        self.metrics = data["metrics"]
        self.feature_names = data.get("feature_names")
        self.n_features = data.get("n_features")
        self.is_trained = True

        logger.info(f"HistGB loaded from {path}")


# =============================================================================
# MODEL MIGRATION UTILITIES
# =============================================================================


def migrate_xgboost_model(model_path: str, output_path: Optional[str] = None) -> bool:
    """
    Migrate XGBoost model from pickle to native format to avoid version warnings.

    XGBoost models serialized with pickle may cause warnings when loaded with
    different XGBoost versions. This function re-saves the model in the native
    XGBoost format which is more portable.

    Args:
        model_path: Path to existing pickle model
        output_path: Path for migrated model (default: same as input)

    Returns:
        True if migration successful, False otherwise
    """
    import warnings
    import pickle
    from pathlib import Path

    # Suppress warnings during migration
    warnings.filterwarnings("ignore", category=UserWarning, module="xgboost")
    warnings.filterwarnings("ignore", message=SERIALIZED_MODEL_WARNING)

    model_path = Path(model_path)
    output_path = Path(output_path) if output_path else model_path

    if not model_path.exists():
        logger.error(f"Model not found: {model_path}")
        return False

    try:
        # Load existing model
        with open(model_path, "rb") as f:
            data = pickle.load(f)

        # Re-save (this ensures internal XGBoost state is updated)
        backup_path = model_path.with_suffix(".pkl.bak")
        if model_path == output_path:
            # Backup original
            import shutil

            shutil.copy(model_path, backup_path)

        with open(output_path, "wb") as f:
            pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)

        logger.info(f"✅ XGBoost model migrated: {output_path}")
        return True

    except Exception as e:
        logger.error(f"Migration failed: {e}")
        return False


def migrate_all_models(model_dir: str = PRODUCTION_MODELS_DIR) -> Dict[str, bool]:
    """
    Migrate all pickle-based models to reduce version warnings.

    Args:
        model_dir: Directory containing models

    Returns:
        Dict mapping model name to migration success status
    """
    from pathlib import Path

    model_dir = Path(model_dir)
    results = {}

    # Models that may need migration
    pkl_models = [
        "xgb_momentum.pkl",
        "rf_risk.pkl",
        RIDGE_CONFIDENCE_FILENAME,
        "histgb_direction.pkl",
    ]

    for model_name in pkl_models:
        model_path = model_dir / model_name
        if model_path.exists():
            results[model_name] = migrate_xgboost_model(str(model_path))
        else:
            results[model_name] = None  # Not found

    return results


# =============================================================================
# CONVENIENCE FUNCTION - Train All Models
# =============================================================================


def train_all_modular(
    data: Dict[str, Dict[str, np.ndarray]],
    config: Optional[TrainerConfig] = None,
    save_dir: str = PRODUCTION_MODELS_DIR,
    use_transformer: bool = True,
    use_regime: bool = False,
    warm_start: bool = False,
    train_histgb: bool = True,
    instrument: str = "EUR_USD",
    data_range: str = "",
) -> Dict[str, BaseTrainer]:
    """
    Train all 4 models independently.

    Args:
        data: Dict from load_all_modular_data() with 'direction'/'regime', 'xgboost', 'rf', 'ridge' keys
        config: Optional trainer configuration
        save_dir: Directory to save models
        use_transformer: If True, use Transformer; if False, use TCN (only for direction mode)
        use_regime: If True, train regime classifier instead of direction predictor
        warm_start: If True, load existing model weights and continue training (compounding learning)
        train_histgb: If True, also train HistGradientBoosting for hybrid voting with Transformer
        instrument: Trading instrument name for replay buffer storage (e.g., "EUR_USD")
        data_range: Date range of training data for lineage tracking

    Returns:
        Dict with trained trainer instances
    """
    config = config or TrainerConfig()
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    trainers = {}

    # Determine warm-start checkpoint path
    transformer_checkpoint = (
        save_dir / TRANSFORMER_DIRECTION_FILENAME if warm_start else None
    )

    # 1. Regime or Direction Model
    logger.info("\n" + "=" * 50)

    if use_regime:
        _train_regime_direction(data, config, save_dir, trainers)
    else:
        _train_standard_direction(
            data, config, save_dir, trainers,
            use_transformer, warm_start, transformer_checkpoint,
            instrument, data_range,
        )

    # 2-5. Auxiliary models (XGBoost, RF, Ridge, HistGB)
    _train_auxiliary_models(
        data, config, save_dir, trainers, train_histgb, use_regime
    )

    logger.info("\n" + "=" * 50)
    logger.info("All 4 models trained independently!")
    logger.info("=" * 50)

    return trainers


def _train_regime_direction(
    data: Dict, config: TrainerConfig, save_dir: Path, trainers: Dict
) -> None:
    """Train regime classifier (3-class: trend/chop/mean_revert)."""
    logger.info("Training Transformer (REGIME Classifier - 3 classes)")
    logger.info("  Classes: trend, chop, mean_revert")
    logger.info("=" * 50)

    regime_data = data.get("regime")
    if regime_data is None:
        raise ValueError(
            "No regime data found. Set use_regime=True in load_all_modular_data()"
        )

    regime_trainer = TransformerRegimeTrainer(config)
    regime_trainer.train(
        regime_data["X_train"],
        regime_data["y_train"],
        regime_data["X_val"],
        regime_data["y_val"],
        feature_names=regime_data.get("feature_names"),
        class_names=regime_data.get("class_names"),
    )
    regime_trainer.save(str(save_dir / "transformer_regime.keras"))
    trainers["regime"] = regime_trainer
    trainers["transformer"] = regime_trainer


def _train_standard_direction(
    data: Dict,
    config: TrainerConfig,
    save_dir: Path,
    trainers: Dict,
    use_transformer: bool,
    warm_start: bool,
    transformer_checkpoint: Optional[Path],
    instrument: str,
    data_range: str,
) -> None:
    """Train direction model (ensemble, transformer-only, or TCN-only)."""
    use_ensemble = (
        getattr(config, "use_tcn_transformer_ensemble", True) if config else True
    )

    if use_ensemble:
        _train_ensemble_direction(
            data, config, save_dir, trainers,
            warm_start, transformer_checkpoint, instrument, data_range,
        )
    elif use_transformer:
        _train_transformer_only_direction(
            data, config, save_dir, trainers,
            warm_start, transformer_checkpoint, instrument, data_range,
        )
    else:
        _train_tcn_only_direction(data, config, save_dir, trainers)


def _get_direction_data(data: Dict) -> Dict:
    """Get direction data from data dict, raising if missing."""
    dir_data = data.get("direction", data.get("tcn"))
    if dir_data is None:
        raise ValueError(NO_DIRECTION_DATA_ERROR)
    return dir_data


def _log_warm_start(warm_start: bool, checkpoint: Optional[Path]) -> None:
    """Log warm-start status if applicable."""
    if warm_start and checkpoint and checkpoint.exists():
        logger.info(
            f"🔥 WARM-START enabled: Loading weights from {checkpoint}"
        )


def _train_transformer_direction_core(
    dir_data: Dict,
    config: TrainerConfig,
    save_dir: Path,
    warm_start: bool,
    transformer_checkpoint: Optional[Path],
    instrument: str,
    data_range: str,
) -> TransformerDirectionTrainer:
    """Train a single Transformer direction model and save it."""
    _log_warm_start(warm_start, transformer_checkpoint)
    trainer = TransformerDirectionTrainer(config)
    trainer.train(
        dir_data["X_train"],
        dir_data["y_train"],
        dir_data["X_val"],
        dir_data["y_val"],
        feature_names=dir_data.get("feature_names"),
        w_train=dir_data.get("w_train"),
        w_val=dir_data.get("w_val"),
        warm_start_path=str(transformer_checkpoint) if warm_start else None,
        instrument=instrument,
        data_range=data_range,
    )
    trainer.save(str(save_dir / TRANSFORMER_DIRECTION_FILENAME))
    return trainer


def _train_ensemble_direction(
    data: Dict,
    config: TrainerConfig,
    save_dir: Path,
    trainers: Dict,
    warm_start: bool,
    transformer_checkpoint: Optional[Path],
    instrument: str,
    data_range: str,
) -> None:
    """Train TCN + Transformer ensemble for direction."""
    logger.info("Training TCN + Transformer Ensemble (Direction Predictors)")
    logger.info("=" * 50)

    dir_data = _get_direction_data(data)

    # 1. Train Transformer
    logger.info("\n[1/2] Training Transformer (Direction)")
    logger.info("-" * 40)
    transformer_trainer = _train_transformer_direction_core(
        dir_data, config, save_dir,
        warm_start, transformer_checkpoint, instrument, data_range,
    )
    trainers["transformer"] = transformer_trainer

    # 2. Train TCN Volatility Regime Filter
    logger.info("\n[2/2] Training TCN (Volatility Regime Filter)")
    logger.info("-" * 40)
    _train_tcn_volatility_regime(data, config, save_dir, trainers, warm_start, instrument)

    # Use transformer as primary direction model for backward compatibility
    trainers["direction"] = transformer_trainer

    logger.info("\n✓ TCN + Transformer Ensemble training complete")
    logger.info(
        f"  - Transformer saved: {save_dir / TRANSFORMER_DIRECTION_FILENAME}"
    )
    logger.info(f"  - TCN saved: {save_dir / 'tcn_direction.keras'}")


def _train_tcn_volatility_regime(
    data: Dict,
    config: TrainerConfig,
    save_dir: Path,
    trainers: Dict,
    warm_start: bool,
    instrument: str,
) -> None:
    """Train TCN volatility regime filter (4-class: LOW/NORMAL/HIGH/EXTREME)."""
    tcn_trainer = TCNTrainer(config)
    vol_regime_data = data.get("volatility_regime")
    if vol_regime_data is None:
        logger.warning("No volatility_regime data found in data dict")
        logger.warning(
            "Add 'volatility_regime' to load_all_modular_data or train TCN separately"
        )
        return

    try:
        tcn_checkpoint = save_dir / "tcn_volatility_regime.keras"
        warm_start_tcn_path = str(tcn_checkpoint) if warm_start and tcn_checkpoint.exists() else None

        tcn_trainer.train(
            vol_regime_data["X_train"],
            vol_regime_data["y_train"],
            vol_regime_data["X_val"],
            vol_regime_data["y_val"],
            feature_names=vol_regime_data.get("feature_names"),
            warm_start_path=warm_start_tcn_path,
            instrument=instrument,
        )
        tcn_trainer.save(str(save_dir / "tcn_volatility_regime.keras"))
        trainers["tcn_volatility"] = tcn_trainer
        logger.info("✓ TCN Volatility Regime model trained and saved")
    except Exception as e:
        logger.warning(f"TCN Volatility Regime training failed: {e}")
        logger.warning(
            "Continuing with Transformer-only (no volatility filter)"
        )


def _train_transformer_only_direction(
    data: Dict,
    config: TrainerConfig,
    save_dir: Path,
    trainers: Dict,
    warm_start: bool,
    transformer_checkpoint: Optional[Path],
    instrument: str,
    data_range: str,
) -> None:
    """Train Transformer-only direction model."""
    logger.info("Training Transformer (Direction Predictor)")
    logger.info("=" * 50)

    dir_data = _get_direction_data(data)
    dir_trainer = _train_transformer_direction_core(
        dir_data, config, save_dir,
        warm_start, transformer_checkpoint, instrument, data_range,
    )
    trainers["direction"] = dir_trainer
    trainers["transformer"] = dir_trainer


def _train_tcn_only_direction(
    data: Dict, config: TrainerConfig, save_dir: Path, trainers: Dict
) -> None:
    """Train TCN-only direction model."""
    logger.info("Training TCN (Direction Predictor)")
    logger.info("=" * 50)

    dir_data = _get_direction_data(data)
    dir_trainer = TCNTrainer(config)
    dir_trainer.train(
        dir_data["X_train"],
        dir_data["y_train"],
        dir_data["X_val"],
        dir_data["y_val"],
        feature_names=dir_data.get("feature_names"),
    )
    dir_trainer.save(str(save_dir / "tcn_direction.keras"))
    trainers["direction"] = dir_trainer
    trainers["tcn"] = dir_trainer


def _train_simple_model(
    data: Dict,
    key: str,
    trainer_cls,
    config: TrainerConfig,
    save_dir: Path,
    save_name: str,
    label: str,
) -> BaseTrainer:
    """Train a simple (non-direction) model and save it."""
    logger.info("\n" + "=" * 50)
    logger.info(f"Training {label}")
    logger.info("=" * 50)
    model_data = data[key]
    trainer = trainer_cls(config)
    trainer.train(
        model_data["X_train"],
        model_data["y_train"],
        model_data["X_val"],
        model_data["y_val"],
        feature_names=model_data.get("feature_names"),
    )
    trainer.save(str(save_dir / save_name))
    return trainer


def _train_auxiliary_models(
    data: Dict,
    config: TrainerConfig,
    save_dir: Path,
    trainers: Dict,
    train_histgb: bool,
    use_regime: bool,
) -> None:
    """Train auxiliary models: XGBoost, RF, Ridge, and optionally HistGB."""
    trainers["xgboost"] = _train_simple_model(
        data, "xgboost", XGBoostTrainer, config, save_dir,
        "xgb_momentum.pkl", "XGBoost (Momentum Analyzer)",
    )
    trainers["rf"] = _train_simple_model(
        data, "rf", RandomForestTrainer, config, save_dir,
        "rf_risk.pkl", "Random Forest (Risk Assessor)",
    )
    trainers["ridge"] = _train_simple_model(
        data, "ridge", RidgeTrainer, config, save_dir,
        RIDGE_CONFIDENCE_FILENAME, "Ridge (Confidence Scorer)",
    )

    if train_histgb and not use_regime:
        _train_histgb_model(data, config, save_dir, trainers)


def _train_histgb_model(
    data: Dict, config: TrainerConfig, save_dir: Path, trainers: Dict
) -> None:
    """Train HistGradientBoosting for hybrid voting."""
    logger.info("\n" + "=" * 50)
    logger.info(
        "Training HistGradientBoosting (Direction Baseline for Hybrid Voting)"
    )
    logger.info("=" * 50)

    dir_data = data.get("direction", data.get("tcn"))
    if dir_data is None:
        logger.warning("No direction data found for HistGB training")
        return

    histgb_trainer = HistGradientBoostingDirectionTrainer(config)
    histgb_trainer.train(
        dir_data["X_train"],
        dir_data["y_train"],
        dir_data["X_val"],
        dir_data["y_val"],
        feature_names=dir_data.get("feature_names"),
    )
    histgb_trainer.save(str(save_dir / "histgb_direction.pkl"))
    trainers["histgb"] = histgb_trainer
    logger.info("✓ HistGB trained for hybrid voting with Transformer")
