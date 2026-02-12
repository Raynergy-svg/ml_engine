"""
Training Callbacks for ML Engine.

This module contains all TensorFlow/Keras callbacks and related utilities
for advanced training features.

Callback Classes:
-----------------
1. **EMACallback**: Exponential Moving Average of model weights for stable inference
2. **EWCPenalty**: Elastic Weight Consolidation for continual learning
3. **EWCLoss**: Custom loss class that includes EWC penalty
4. **OverfitPreventionCallback**: Advanced callback to detect and mitigate overfitting
5. **EWCTrainingCallback**: Callback to log EWC penalty during training
6. **QuietProgressCallback**: Minimal progress bar for quiet mode training
7. **GradualUnfreezeCallback**: Gradually unfreeze encoder layers during warm-start
8. **RichEpochCallback**: Rich-formatted epoch display with color-coded metrics
9. **AutoAdjustCallback**: Auto-adjusts training when stuck (plateau detection)
10. **ReplayBuffer**: Memory replay buffer for continual learning
11. **DriftDetector**: Advanced drift detection for continual learning
12. **TrainingLineage**: Track model training history across warm-start sessions

Key Features:
-------------
- **Stochastic Weight Averaging (SWA)**: Find flatter optima for better generalization
- **Cosine Annealing with Warm Restarts**: Escape local minima periodically
- **Aggressive Overfit Intervention**: Immediate LR reduction and dropout adjustment
- **Gap-Gated Checkpointing**: Only save when validation improves AND gap is acceptable
- **Warm-Start Overfit Detection**: Detect and recover from pre-existing overfitting
- **Catastrophic Forgetting Prevention**: EWC + Replay Buffer for continual learning
- **Drift Detection**: Monitor performance, data, and concept drift
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING

import numpy as np
import tensorflow as tf

from src.training.trainers.config import TrainerConfig, OverfitPreventionConfig
from src.training.trainers.utils import (
    _validate_weight_shapes,
    _safe_get_learning_rate,
    _safe_set_learning_rate,
)

if TYPE_CHECKING:
    import pandas as pd

logger = logging.getLogger(__name__)


# =============================================================================
# EMA - Exponential Moving Average
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


# =============================================================================
# OVERFIT PREVENTION CALLBACK
# =============================================================================


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


# =============================================================================
# ADDITIONAL TRAINING CALLBACKS
# =============================================================================


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
            loss_str = f"[yellow]loss=N/A[/yellow]"
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
            # Get current learning rate
            try:
                current_lr = float(self.model.optimizer.learning_rate)
            except Exception:
                current_lr = float(tf.keras.backend.get_value(self.model.optimizer.learning_rate))

            if current_lr > self.min_lr:
                # Reduce learning rate
                new_lr = max(current_lr * self.lr_factor, self.min_lr)

                # Keras 3.x compatible way to set LR
                self.model.optimizer.learning_rate.assign(new_lr)

                self.adjustments_made += 1
                self.wait = 0  # Reset patience

                if self.verbose:
                    self.console.print(
                        f"  [yellow]⚙ Auto-adjust #{self.adjustments_made}: "
                        f"LR {current_lr:.2e} → {new_lr:.2e} "
                        f"(stuck for {self.patience} epochs at {self.best_val_acc:.1%})[/yellow]"
                    )
            else:
                if self.verbose and self.wait == self.patience:
                    self.console.print(
                        f"  [dim]⚙ LR already at minimum ({self.min_lr:.2e}), cannot adjust further[/dim]"
                    )

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
        current_feature_names: Optional[List[str]] = None,
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

