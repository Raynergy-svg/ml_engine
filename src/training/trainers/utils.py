"""
Utility functions and constants for model trainers.

This module contains:
- Module-level constants for frequently used literals
- Pair volatility classification for auto-tuning
- Helper functions for data processing, weight management, and model configuration
- LightGBM regressor/classifier creation with GPU fallback
- Regime-specific LightGBM hyperparameters
"""

from __future__ import annotations

import json
import logging
import os
import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union, TYPE_CHECKING

import numpy as np
import warnings

# TensorFlow is lazy-loaded to allow CLI startup without it installed.
# Functions that need tf import it locally.
tf = None  # type: ignore

def _get_tf():
    """Lazy-load TensorFlow on first use."""
    global tf
    if tf is None or tf is False:
        import tensorflow as _tf
        tf = _tf
    return tf

if TYPE_CHECKING:
    from src.training.trainers.config import TrainerConfig

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


# =============================================================================
# ATOMIC SAVE HELPERS (2026-06-10 training-infra audit)
# =============================================================================
# Every trainer save path must write to a temp file in the SAME directory and
# os.replace() it into place. A direct open(path, "wb") that dies mid-write
# (ENOSPC, SIGKILL) leaves a truncated artifact that per-pair gate routing
# happily loads — the worst failure mode. os.replace() within one directory
# is atomic on POSIX filesystems.

PathLike = Union[str, "Path"]


def _atomic_tmp_path(path: Path, preserved_suffix: str = "") -> Path:
    """Return a sibling temp path in the same directory as ``path``.

    If ``preserved_suffix`` is given and the filename ends with it, ``.tmp``
    is inserted BEFORE that suffix. This matters for format-sniffing savers:
    Keras refuses to save unless the path ends with ``.keras`` /
    ``.weights.h5``, so the temp file must keep the trailing extension.
    """
    if preserved_suffix and path.name.endswith(preserved_suffix):
        base = path.name[: -len(preserved_suffix)]
        return path.with_name(f"{base}.tmp{preserved_suffix}")
    return path.with_name(path.name + ".tmp")


def _fsync_existing_file(file_path: Path) -> None:
    """Best-effort fsync for files written by APIs that hide the handle
    (e.g. ``keras.Model.save``). Failure to fsync is logged, not fatal —
    os.replace() still guarantees we never expose a half-written file."""
    try:
        fd = os.open(str(file_path), os.O_RDONLY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
    except OSError as exc:
        logger.debug(f"fsync skipped for {file_path}: {exc}")


def atomic_pickle_dump(obj: Any, path: PathLike) -> None:
    """Atomically pickle ``obj`` to ``path`` (tmp + flush + fsync + replace)."""
    path = Path(path)
    tmp = _atomic_tmp_path(path)
    try:
        with open(tmp, "wb") as f:
            pickle.dump(obj, f)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise


def atomic_joblib_dump(obj: Any, path: PathLike) -> None:
    """Atomically joblib-dump ``obj`` to ``path`` (tmp + fsync + replace)."""
    import joblib  # lazy: only the transformer scaler saves need it

    path = Path(path)
    tmp = _atomic_tmp_path(path)
    try:
        joblib.dump(obj, str(tmp))
        _fsync_existing_file(tmp)
        os.replace(tmp, path)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise


def atomic_text_write(text: str, path: PathLike) -> None:
    """Atomically write ``text`` to ``path`` (tmp + flush + fsync + replace)."""
    path = Path(path)
    tmp = _atomic_tmp_path(path)
    try:
        with open(tmp, "w") as f:
            f.write(text)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise


def atomic_json_dump(obj: Any, path: PathLike, **json_kwargs: Any) -> None:
    """Atomically serialize ``obj`` as JSON to ``path``."""
    atomic_text_write(json.dumps(obj, **json_kwargs), path)


def atomic_keras_save(model: Any, path: PathLike, *, weights_only: bool = False) -> None:
    """Atomically save a Keras model (or just its weights) via tmp + replace.

    Keras validates the trailing extension, so the temp file inserts ``.tmp``
    before the required suffix (``.keras`` or ``.weights.h5``), e.g.
    ``transformer_direction.tmp.keras``.

    Args:
        model: Keras model instance.
        path: Final destination (must end with ``.keras`` or ``.weights.h5``).
        weights_only: If True, calls ``model.save_weights`` instead of
            ``model.save``.
    """
    path = Path(path)
    preserved = (
        WEIGHTS_H5_SUFFIX if path.name.endswith(WEIGHTS_H5_SUFFIX) else path.suffix
    )
    tmp = _atomic_tmp_path(path, preserved)
    try:
        if weights_only:
            model.save_weights(str(tmp))
        else:
            model.save(str(tmp))
        _fsync_existing_file(tmp)
        os.replace(tmp, path)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise


# =============================================================================
# METRIC HONESTY HELPER (2026-06-10 training-infra audit)
# =============================================================================


def train_accuracy_at_best_epoch(
    train_acc_history: List[float],
    val_acc_history: Optional[List[float]] = None,
    best_epoch_idx: Optional[int] = None,
) -> float:
    """Return train_accuracy at the epoch whose weights were actually saved.

    Trainers that restore best weights (EarlyStopping restore_best_weights=True,
    or hand-rolled best-val-loss tracking) must NOT report
    ``history["accuracy"][-1]`` — the last epoch is the post-overfit one, which
    produces a phantom train/val gap on the saved checkpoint and can falsely
    trip the operator's 10% hard ship gate (see .claude/rules/improvement.md
    "Hard Ship Gate").

    Args:
        train_acc_history: Per-epoch training accuracy.
        val_acc_history: Per-epoch validation accuracy. Used to locate the
            best epoch via argmax when ``best_epoch_idx`` is not supplied.
        best_epoch_idx: Explicit saved-epoch index for trainers that track it
            themselves (e.g. best-val-loss keyed restore). Takes precedence
            over ``val_acc_history``.

    Returns:
        Training accuracy at the best epoch; falls back to the last epoch when
        the best epoch cannot be determined, and 0.0 for empty history.
    """
    if not train_acc_history:
        return 0.0
    if best_epoch_idx is not None and 0 <= best_epoch_idx < len(train_acc_history):
        return float(train_acc_history[best_epoch_idx])
    if best_epoch_idx is None and val_acc_history:
        idx = int(np.argmax(val_acc_history))
        idx = min(idx, len(train_acc_history) - 1)
        return float(train_acc_history[idx])
    return float(train_acc_history[-1])


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
        return _get_tf().dtypes.as_dtype(keras_dtype).as_numpy_dtype
    except Exception:
        # Last resort: default to float32
        logger.warning(f"Could not convert dtype {keras_dtype}, defaulting to float32")
        return np.float32


def predict_with_named_input_if_needed(
    model: Any,
    batch: np.ndarray,
    *,
    verbose: int = 0,
) -> Any:
    """Prefer named-input inference for single-input Keras models.

    Some saved models expose a single named input such as ``features``. Passing
    a bare ndarray still runs, but Keras emits structure warnings. This helper
    uses the named-input form first and falls back to the plain array call only
    when the runtime rejects the mapping.
    """
    model_inputs = getattr(model, "inputs", None)
    if isinstance(model_inputs, list) and len(model_inputs) == 1:
        input_name = getattr(model_inputs[0], "name", None)
        if isinstance(input_name, str) and input_name:
            candidates = [input_name.split(":", 1)[0]]
            if candidates[0] != input_name:
                candidates.append(input_name)
            for candidate in candidates:
                try:
                    with warnings.catch_warnings():
                        warnings.filterwarnings(
                            "ignore",
                            message=".*structure of `inputs` doesn't match the expected structure.*",
                            category=UserWarning,
                        )
                        return model.predict({candidate: batch}, verbose=verbose)
                except (TypeError, ValueError, KeyError):
                    continue
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=".*structure of `inputs` doesn't match the expected structure.*",
            category=UserWarning,
        )
        return model.predict(batch, verbose=verbose)


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


def validate_sequence_alignment(
    X: np.ndarray,
    y: np.ndarray,
    seq_len: int,
    context: str = "sequence creation",
) -> None:
    """
    Validate that feature and label arrays are properly aligned for sequencing.

    Prevents IndexError from off-by-one errors in look-ahead logic or label alignment.

    Args:
        X: Feature matrix of shape (n_samples, n_features)
        y: Label array of shape (n_samples,) or (n_samples, n_targets)
        seq_len: Sequence length for temporal models
        context: Description for error messages

    Raises:
        ValueError: If arrays are misaligned and would cause IndexError
    """
    if X is None or len(X) == 0:
        raise ValueError(f"Empty feature array in {context}")

    if y is None or len(y) == 0:
        raise ValueError(f"Empty label array in {context}")

    n_X = len(X)
    n_y = len(y)

    # After sequencing, we access y[i + seq_len - 1] for i in range(n_X - seq_len)
    # The maximum index accessed is: (n_X - seq_len - 1) + seq_len - 1 = n_X - 2
    # So we need: n_y >= n_X - 1
    max_y_idx_needed = n_X - 2

    if n_y <= max_y_idx_needed:
        raise ValueError(
            f"Label array too short for sequencing in {context}: "
            f"X has {n_X} samples, y has {n_y} samples, "
            f"but sequencing needs y[{max_y_idx_needed}] (max index). "
            f"Possible causes:\n"
            f"  1. Look-ahead rows not properly dropped from features\n"
            f"  2. Train/val split applied inconsistently to X vs y\n"
            f"  3. Feature extraction used raw df indices instead of data loader indices"
        )

    # Warn if lengths differ significantly (may indicate misalignment)
    if abs(n_X - n_y) > 1:
        logger.warning(
            f"Feature/label length mismatch in {context}: X={n_X}, y={n_y}. "
            f"This may indicate data alignment issues."
        )


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

    Raises:
        ValueError: If arrays are misaligned (via validate_sequence_alignment)
    """
    # Validate alignment before sequencing to prevent IndexError
    validate_sequence_alignment(X, y, seq_len, "create_sequences_with_weights")

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

        # Downgrade expected warm-start warnings to debug/info
        for w in caught:
            msg = str(w.message)
            msg_lower = msg.lower()
            # These are all expected during architecture changes / warm-start
            benign_keywords = (
                "optimizer", "skipping variable", "loading skip",
                "expected", "variables", "received 0",  # Layer shape mismatches
                "einsumdense", "could not be loaded",  # Attention layer changes
            )
            if any(kw in msg_lower for kw in benign_keywords):
                logger.debug(f"Warm-start weight mismatch (expected): {msg[:200]}")
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
                var.assign(_get_tf().zeros_like(var))
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
    # Import here to avoid circular dependency
    from src.training.modular_trainers import EWCLoss
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
        _get_tf().keras.backend.set_value(optimizer.learning_rate, new_lr)
        return True
    except (AttributeError, TypeError):
        # Learning rate is likely a schedule - can't be modified directly
        # This is expected behavior when using LR schedules
        return False
    except Exception:
        return False


# =============================================================================
# LIGHTGBM UTILITIES
# =============================================================================

# Regime-specific LightGBM hyperparameters
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
