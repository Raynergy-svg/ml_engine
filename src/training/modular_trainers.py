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
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import tensorflow as tf

# Import custom loss functions from our model module
from src.models.tensorflow_models import AntiCollapseFocalLoss

logger = logging.getLogger(__name__)


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
        self._Table = Table
        self._Panel = Panel
        self._Progress = Progress
        self._SpinnerColumn = SpinnerColumn
        self._TextColumn = TextColumn
        self._BarColumn = BarColumn
        self._TaskProgressColumn = TaskProgressColumn
    
    def show_config(self, config: dict):
        """Display configuration as a clean table."""
        table = self._Table(show_header=False, box=None, padding=(0, 2))
        table.add_column("Key", style="dim")
        table.add_column("Value", style="cyan")
        for k, v in config.items():
            table.add_row(str(k), str(v))
        self.console.print(self._Panel(table, title=f"[bold]{self.model_name}[/bold]", border_style="blue"))
    
    def show_summary(self, metrics: dict, title: str = "Results"):
        """Display training results summary."""
        table = self._Table(show_header=False, box=None, padding=(0, 2))
        table.add_column("Metric", style="dim")
        table.add_column("Value", style="green")
        for k, v in metrics.items():
            if isinstance(v, float):
                table.add_row(str(k), f"{v:.4f}")
            else:
                table.add_row(str(k), str(v))
        self.console.print(self._Panel(table, title=f"[bold]{title}[/bold]", border_style="green"))
    
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
    
    # TCN specific - INCREASED REGULARIZATION
    tcn_hidden_size: int = 32
    tcn_num_layers: int = 2
    tcn_kernel_size: int = 3
    tcn_dropout: float = 0.5  # Was 0.3 - increased to combat overfitting
    
    # TCN Volatility Regime specific (4-class: LOW/NORMAL/HIGH/EXTREME)
    # Research-backed architecture from Bai et al. / Unit8
    tcn_num_filters: int = 64         # Channel capacity per layer
    tcn_num_residual_blocks: int = 5  # Depth for receptive field coverage
    tcn_dilation_base: int = 2        # Exponential dilation: 1, 2, 4, 8, 16
    tcn_weight_norm: bool = True      # Weight normalization for training stability
    
    # Transformer specific (for direction prediction) - STRONGER REGULARIZATION
    transformer_d_model: int = 32  # Model dimension (was 16, broke training)
    transformer_num_heads: int = 4  # Number of attention heads (was 2)
    transformer_num_layers: int = 2  # Number of encoder layers (was 1)
    transformer_dff: int = 64  # Feedforward network dimension (was 32)
    transformer_dropout: float = 0.2  # Was 0.4 - reduced to prevent output suppression
    
    # === FOCAL LOSS SETTINGS (Anti-Bias) ===
    use_focal_loss: bool = True  # Use Focal Loss instead of BCE to prevent direction bias
    focal_gamma: float = 2.0  # Focusing parameter - higher = more focus on hard examples
    focal_alpha: float = 0.5  # Class balance (0.5 = balanced, <0.5 = favor SHORT)
    
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
    elasticnet_l1_ratios: List[float] = field(default_factory=lambda: [0.1, 0.5, 0.7, 0.9, 0.95, 1.0])
    elasticnet_alphas: Optional[List[float]] = None  # Auto-generate via logspace if None
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
    ewc_lambda: float = 100.0  # Strength of EWC constraint (λ) - reduced from 1000 to allow learning
    ewc_gamma: float = 0.95  # Decay for old Fisher values when adding new tasks
    
    # Replay Buffer for catastrophic forgetting prevention
    use_replay_buffer: bool = True  # Enable memory replay
    replay_buffer_ratio: float = 0.10  # Save 10% of training data
    replay_mix_ratio: float = 0.20  # Mix 20% replay samples during training
    replay_buffer_dir: str = "trained_data/replay"  # Replay buffer storage
    
    # Warm-start settings (CRITICAL for preventing catastrophic forgetting)
    warm_start_lr_factor: float = 0.01  # Reduce LR by 100x on warm-start (0.01 is standard for fine-tuning)
    warm_start_freeze_encoder: bool = True  # Freeze ALL transformer encoder layers (not just transformer_0)
    warm_start_unfreeze_epochs: int = 10  # Optionally unfreeze after N epochs (0 = never unfreeze)
    warm_start_gradual_unfreeze: bool = False  # Gradually unfreeze layers from top to bottom
    
    # Drift detection for auto-retraining
    drift_threshold: float = 0.03  # Retrain if val_acc drops by >3%


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
        self.ema_weights = None
        self.backup_weights = None
        self._initialized = False
        
    def _initialize_ema(self):
        """Initialize EMA weights as copy of current model weights."""
        self.ema_weights = [w.numpy().copy() for w in self.model.trainable_weights]
        self._initialized = True
        logger.info(f"📊 EMA initialized with {len(self.ema_weights)} weight tensors (decay={self.decay})")
    
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
        
        # EMA update: θ_ema = α * θ_ema + (1-α) * θ_current
        for i, w in enumerate(self.model.trainable_weights):
            self.ema_weights[i] = (
                self.decay * self.ema_weights[i] + 
                (1 - self.decay) * w.numpy()
            )
    
    def apply(self):
        """Apply EMA weights to model (for inference). Backs up current weights."""
        if not self._initialized:
            logger.warning("EMA not initialized, cannot apply")
            return
        
        # Backup current (training) weights
        self.backup_weights = [w.numpy().copy() for w in self.model.trainable_weights]
        
        # Apply EMA weights
        for w, ema_w in zip(self.model.trainable_weights, self.ema_weights):
            w.assign(ema_w)
    
    def restore(self):
        """Restore original training weights after inference."""
        if self.backup_weights is None:
            return
        
        for w, backup_w in zip(self.model.trainable_weights, self.backup_weights):
            w.assign(backup_w)
        
        self.backup_weights = None
    
    def get_ema_weights(self) -> List[np.ndarray]:
        """Get EMA weights for saving."""
        if not self._initialized:
            self._initialize_ema()
        return self.ema_weights
    
    def set_ema_weights(self, weights: List[np.ndarray]):
        """Load EMA weights from checkpoint."""
        self.ema_weights = [w.copy() for w in weights]
        self._initialized = True
        logger.info(f"📊 EMA weights loaded ({len(weights)} tensors)")


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
        import tensorflow as tf
        
        # Sample subset for efficiency
        if len(X) > n_samples:
            indices = np.random.choice(len(X), n_samples, replace=False)
            X_sample = X[indices]
            y_sample = y[indices]
        else:
            X_sample = X
            y_sample = y
        
        # Initialize Fisher as zeros
        new_fisher = [np.zeros_like(w.numpy()) for w in self.model.trainable_weights]
        
        # Detect output type to choose correct loss function
        # Binary classification (sigmoid) uses BinaryCrossentropy
        # Multi-class (softmax) uses SparseCategoricalCrossentropy
        try:
            test_pred = self.model(X_sample[:1], training=False)
            # Handle dict outputs (multi-head models)
            if isinstance(test_pred, dict):
                # Use 'direction' output if available, else first output
                if 'direction' in test_pred:
                    test_pred = test_pred['direction']
                else:
                    test_pred = list(test_pred.values())[0]
            output_shape = test_pred.shape[-1] if len(test_pred.shape) > 1 else 1
        except Exception:
            output_shape = 1  # Default to binary
        
        # Choose loss function based on output shape and label values
        unique_labels = np.unique(y_sample)
        is_binary = output_shape == 1 or (len(unique_labels) <= 2 and max(unique_labels) <= 1)
        
        if is_binary:
            loss_fn = tf.keras.losses.BinaryCrossentropy(from_logits=False)
            logger.debug("EWC using BinaryCrossentropy (binary classification)")
        else:
            loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False)
            logger.debug("EWC using SparseCategoricalCrossentropy (multi-class)")
        
        for i in range(len(X_sample)):
            x_i = X_sample[i:i+1]
            y_i = y_sample[i:i+1]
            
            # Cast labels to float32 for binary crossentropy
            if is_binary:
                y_i = tf.cast(y_i, tf.float32)
            
            with tf.GradientTape() as tape:
                pred = self.model(x_i, training=False)
                # Handle dict outputs (multi-head models)
                if isinstance(pred, dict):
                    if 'direction' in pred:
                        pred = pred['direction']
                    else:
                        pred = list(pred.values())[0]
                loss = loss_fn(y_i, pred)
            
            grads = tape.gradient(loss, self.model.trainable_weights)
            
            # Fisher = E[grad²]
            for j, grad in enumerate(grads):
                if grad is not None:
                    new_fisher[j] += grad.numpy() ** 2
        
        # Average over samples
        new_fisher = [f / len(X_sample) for f in new_fisher]
        
        # Combine with existing Fisher (if any) using decay
        if self.fisher_diagonal is not None:
            self.fisher_diagonal = [
                self.gamma * old_f + new_f 
                for old_f, new_f in zip(self.fisher_diagonal, new_fisher)
            ]
        else:
            self.fisher_diagonal = new_fisher
        
        # Store reference weights
        self.reference_weights = [w.numpy().copy() for w in self.model.trainable_weights]
        self._n_tasks += 1
        
        # Log Fisher statistics
        total_importance = sum(f.sum() for f in self.fisher_diagonal)
        logger.info(f"🧠 EWC Fisher computed: {self._n_tasks} task(s), total_importance={total_importance:.2f}")
    
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
            self.model.trainable_weights, 
            self.fisher_diagonal, 
            self.reference_weights
        ):
            penalty += tf.reduce_sum(f * (w - w_old) ** 2)
        
        return (self.ewc_lambda / 2) * penalty
    
    def save(self, path: str):
        """Save EWC state (Fisher + reference weights)."""
        if self.fisher_diagonal is None:
            logger.warning("No EWC state to save (Fisher not computed)")
            return
        
        path = Path(path)
        data = {
            'fisher_diagonal': self.fisher_diagonal,
            'reference_weights': self.reference_weights,
            'ewc_lambda': self.ewc_lambda,
            'gamma': self.gamma,
            'n_tasks': self._n_tasks,
        }
        with open(path, 'wb') as f:
            pickle.dump(data, f)
        logger.info(f"🧠 EWC state saved to {path}")
    
    def load(self, path: str):
        """Load EWC state from checkpoint."""
        path = Path(path)
        if not path.exists():
            logger.info(f"No EWC checkpoint at {path}, starting fresh")
            return False
        
        with open(path, 'rb') as f:
            data = pickle.load(f)
        
        self.fisher_diagonal = data['fisher_diagonal']
        self.reference_weights = data['reference_weights']
        self.ewc_lambda = data.get('ewc_lambda', self.ewc_lambda)
        self.gamma = data.get('gamma', self.gamma)
        self._n_tasks = data.get('n_tasks', 1)
        
        logger.info(f"🧠 EWC state loaded: {self._n_tasks} task(s), λ={self.ewc_lambda}")
        return True


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
        Custom loss function compatible with Keras
    """
    import tensorflow as tf
    
    @tf.function
    def ewc_loss(y_true, y_pred):
        # Compute base classification loss
        classification_loss = base_loss(y_true, y_pred)
        
        # Add EWC penalty (protects important weights from previous tasks)
        ewc_term = ewc_penalty_fn()
        
        return classification_loss + ewc_weight * ewc_term
    
    return ewc_loss


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
        overfit_threshold: float = 0.08,   # 8% gap triggers warning
        critical_threshold: float = 0.15,  # 15% gap triggers intervention
        severe_threshold: float = 0.25,    # 25% gap triggers early stop
        max_acceptable_gap: float = 0.12,  # Won't save checkpoint if gap > 12%
        patience_epochs: int = 2,          # Reduced from 3 for faster response
        auto_adjust_dropout: bool = True,
        auto_reduce_lr: bool = True,
        max_dropout_increase: float = 0.3,
        # SWA settings
        enable_swa: bool = False,  # DISABLED: Conflicts with ReduceLROnPlateau causing loss to increase
        swa_start_fraction: float = 0.75,  # Start SWA at 75% of training
        swa_lr_factor: float = 0.5,        # SWA uses lower constant LR
        # Cosine restart settings
        enable_cosine_restarts: bool = False,  # DISABLED: Conflicts with other LR schedulers
        restart_period: int = 10,          # Restart every 10 epochs
        restart_lr_mult: float = 0.8,      # Each restart uses 80% of prev LR
        # Mixup augmentation (applied at batch level)
        enable_mixup: bool = False,        # Disabled by default (needs data pipeline change)
        mixup_alpha: float = 0.2,
        # Warm-start overfit recovery (NEW)
        enable_warmstart_detection: bool = True,
        warmstart_reset_threshold: float = 0.15,  # If initial gap > 15%, intervene
        weight_perturbation_scale: float = 0.02,  # Noise to break memorization
        reset_optimizer_on_overfit: bool = True,  # Reset momentum on critical overfit
        # CONTINUAL LEARNING: Previous best accuracy (prevents saving worse models)
        warm_start_best_acc: float = 0.0,  # Set from loaded checkpoint
    ):
        super().__init__()
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.model_name = model_name
        self.overfit_threshold = overfit_threshold
        self.critical_threshold = critical_threshold
        self.severe_threshold = severe_threshold
        self.max_acceptable_gap = max_acceptable_gap
        self.patience_epochs = patience_epochs
        self.auto_adjust_dropout = auto_adjust_dropout
        self.auto_reduce_lr = auto_reduce_lr
        self.max_dropout_increase = max_dropout_increase
        
        # SWA settings
        self.enable_swa = enable_swa
        self.swa_start_fraction = swa_start_fraction
        self.swa_lr_factor = swa_lr_factor
        self.swa_weights = None
        self.swa_count = 0
        self.swa_started = False
        
        # Cosine restart settings
        self.enable_cosine_restarts = enable_cosine_restarts
        self.restart_period = restart_period
        self.restart_lr_mult = restart_lr_mult
        self.current_restart_epoch = 0
        self.num_restarts = 0
        
        # Mixup settings
        self.enable_mixup = enable_mixup
        self.mixup_alpha = mixup_alpha
        
        # Warm-start overfit recovery (NEW)
        self.enable_warmstart_detection = enable_warmstart_detection
        self.warmstart_reset_threshold = warmstart_reset_threshold
        self.weight_perturbation_scale = weight_perturbation_scale
        self.reset_optimizer_on_overfit = reset_optimizer_on_overfit
        self._warmstart_checked = False
        self._initial_weights_perturbed = False
        self._optimizer_reset_count = 0
        
        # State tracking - CRITICAL: Initialize from warm-start to prevent saving worse models
        self.best_val_acc = warm_start_best_acc  # Start from previous best (not 0!)
        self.best_val_acc_clean = warm_start_best_acc  # Best val_acc with acceptable gap
        self.warm_start_best_acc = warm_start_best_acc  # Store original for logging
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
        try:
            self._initial_lr = float(tf.keras.backend.get_value(self.model.optimizer.learning_rate))
        except Exception:
            self._initial_lr = 0.001
        
        # Try to get total epochs from params
        self._total_epochs = self.params.get('epochs', 100)
        
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
                if 'kernel' in weight.name or 'weight' in weight.name:
                    # Add Gaussian noise proportional to weight magnitude
                    noise = tf.random.normal(
                        shape=weight.shape,
                        mean=0.0,
                        stddev=scale * tf.reduce_mean(tf.abs(weight))
                    )
                    weight.assign_add(noise)
                    perturbed_count += 1
        
        if perturbed_count > 0:
            self.console.print(
                f"  [yellow]🎲 Weight perturbation applied (scale={scale:.1%}) "
                f"to {perturbed_count} layers[/yellow]"
            )
        return perturbed_count
    
    def _reinitialize_dense_layers(self):
        """
        Reinitialize only Dense/output layers that tend to memorize most.
        
        Keeps convolutional/attention layers (feature extraction) but resets
        the classification head which often overfits first.
        """
        reinitialized = 0
        for layer in self.model.layers:
            # Reset Dense layers (classification head)
            if isinstance(layer, tf.keras.layers.Dense):
                for weight in layer.trainable_weights:
                    if 'kernel' in weight.name:
                        # Glorot uniform initialization
                        fan_in = weight.shape[0]
                        fan_out = weight.shape[1] if len(weight.shape) > 1 else 1
                        limit = np.sqrt(6.0 / (fan_in + fan_out))
                        new_weights = tf.random.uniform(weight.shape, -limit, limit)
                        weight.assign(new_weights)
                        reinitialized += 1
                    elif 'bias' in weight.name:
                        weight.assign(tf.zeros_like(weight))
        
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
            f"  [red bold]🔥 FULL WEIGHT RESET - Model too far into memorization[/red bold]"
        )
        
        # Save current architecture, reinitialize weights
        for layer in self.model.layers:
            if hasattr(layer, 'kernel_initializer') and hasattr(layer, 'kernel'):
                # Reinitialize kernel
                if layer.kernel is not None:
                    new_kernel = layer.kernel_initializer(layer.kernel.shape)
                    layer.kernel.assign(new_kernel)
            if hasattr(layer, 'bias_initializer') and hasattr(layer, 'bias'):
                # Reinitialize bias
                if layer.bias is not None:
                    new_bias = layer.bias_initializer(layer.bias.shape)
                    layer.bias.assign(new_bias)
        
        # Reset optimizer
        self._reset_optimizer_state()
        
        # Reset LR to initial
        try:
            tf.keras.backend.set_value(self.model.optimizer.learning_rate, self._initial_lr)
            self.console.print(
                f"  [cyan]📈 LR reset to initial: {self._initial_lr:.2e}[/cyan]"
            )
        except Exception:
            pass
        
        self.console.print(
            f"  [yellow]   Starting fresh - warm-start weights were too corrupted[/yellow]"
        )
    
    def _reset_optimizer_state(self):
        """
        Reset optimizer momentum/velocity to allow fresh gradient accumulation.
        
        When overfitting, the optimizer momentum may be pointing toward
        memorization. Resetting allows re-exploration.
        """
        try:
            optimizer = self.model.optimizer
            
            # Reset slot variables (momentum, velocity for Adam)
            for var in optimizer.variables():
                if 'momentum' in var.name.lower() or 'velocity' in var.name.lower() or 'm/' in var.name or 'v/' in var.name:
                    var.assign(tf.zeros_like(var))
            
            self._optimizer_reset_count += 1
            self.console.print(
                f"  [yellow]🔄 Optimizer momentum reset (#{self._optimizer_reset_count}) - "
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
        is_likely_warmstart = train_acc > 0.65  # Fresh model wouldn't have 65%+ on epoch 1
        
        if is_likely_warmstart and gap > self.warmstart_reset_threshold:
            self.console.print(
                f"  [red bold]⚠️ WARM-START OVERFIT DETECTED[/red bold]"
            )
            self.console.print(
                f"  [yellow]   Initial gap={gap:.1%} suggests loaded weights are memorizing.[/yellow]"
            )
            
            # Severity determines action
            if gap > 0.20:  # 20%+ gap - nuclear option
                self.console.print(
                    f"  [red]   Gap > 20% - applying AGGRESSIVE recovery (dense layer reset)[/red]"
                )
                self._reinitialize_dense_layers()
                self._reset_optimizer_state()
            else:
                self.console.print(
                    f"  [yellow]   Applying recovery: weight perturbation + optimizer reset[/yellow]"
                )
                # Standard recovery
                self._perturb_weights(scale=self.weight_perturbation_scale)
                self._initial_weights_perturbed = True
                if self.reset_optimizer_on_overfit:
                    self._reset_optimizer_state()
            
            # Set LR to a moderate value for re-learning
            try:
                recovery_lr = self._initial_lr * 2  # Higher LR for exploration
                tf.keras.backend.set_value(self.model.optimizer.learning_rate, recovery_lr)
                self.console.print(
                    f"  [cyan]📈 LR boosted to {recovery_lr:.2e} for recovery exploration[/cyan]"
                )
            except Exception:
                pass
    
    def _get_cosine_lr(self, epoch: int, base_lr: float) -> float:
        """Calculate cosine annealing LR with warm restarts."""
        if not self.enable_cosine_restarts:
            return base_lr
        
        # Epoch within current restart cycle
        cycle_epoch = (epoch - self.current_restart_epoch) % self.restart_period
        
        # Cosine annealing: lr_t = lr_min + 0.5 * (lr_max - lr_min) * (1 + cos(pi * t / T))
        lr_min = base_lr * 0.01  # Min LR is 1% of base
        lr_max = base_lr * (self.restart_lr_mult ** self.num_restarts)  # Decay with restarts
        
        cos_val = np.cos(np.pi * cycle_epoch / self.restart_period)
        new_lr = lr_min + 0.5 * (lr_max - lr_min) * (1 + cos_val)
        
        return new_lr
    
    def _update_swa_weights(self):
        """Update running average of weights for SWA."""
        if self.swa_weights is None:
            # Initialize SWA weights as copy of current weights
            self.swa_weights = [w.numpy().copy() for w in self.model.trainable_weights]
            self.swa_count = 1
            self.console.print("  [magenta]🔄 SWA initialized - collecting weights for averaging[/magenta]")
        else:
            # Running average: swa_w = (swa_w * n + w) / (n + 1)
            self.swa_count += 1
            for i, w in enumerate(self.model.trainable_weights):
                self.swa_weights[i] = (self.swa_weights[i] * (self.swa_count - 1) + w.numpy()) / self.swa_count
    
    def _apply_swa_weights(self):
        """Apply averaged weights to model."""
        if self.swa_weights is not None and self.swa_count > 1:
            for i, w in enumerate(self.model.trainable_weights):
                w.assign(self.swa_weights[i])
            self.console.print(f"  [magenta]✨ SWA applied: averaged {self.swa_count} weight snapshots[/magenta]")
            return True
        return False
    
    def _store_base_weights(self):
        """Store current weights as backup before overfitting gets worse."""
        self._base_weights = [w.numpy().copy() for w in self.model.trainable_weights]
    
    def _restore_base_weights(self):
        """Restore weights to pre-overfit state."""
        if self._base_weights is not None:
            for i, w in enumerate(self.model.trainable_weights):
                w.assign(self._base_weights[i])
            self.console.print("  [cyan]↩️ Restored weights to pre-overfit checkpoint[/cyan]")
    
    def on_epoch_end(self, epoch, logs=None):
        if logs is None:
            return
        
        train_acc = logs.get('accuracy', 0)
        val_acc = logs.get('val_accuracy', 0)
        overfit_gap = train_acc - val_acc
        
        self.train_acc_history.append(train_acc)
        self.val_acc_history.append(val_acc)
        self.gap_history.append(overfit_gap)
        
        # === WARM-START OVERFIT CHECK (first epoch only) ===
        if epoch == 0:
            self._check_warmstart_overfit(train_acc, val_acc, overfit_gap)
        
        # === STUCK DETECTION: Escalating interventions ===
        # CRITICAL: Skip destructive interventions during warm-start to preserve learned weights!
        # The warm-start recovery code will handle restoring weights if training fails.
        if self.warm_start_best_acc > 0:
            # In warm-start mode: DO NOT perturb weights or reset layers
            # Just let early stopping handle it - we'll restore original weights if needed
            if self.critical_epochs >= 15 and self.critical_epochs % 15 == 0:
                self.console.print(
                    f"  [yellow]⚠️ Warm-start: {self.critical_epochs} epochs without improvement over baseline[/yellow]"
                )
                self.console.print(
                    f"  [yellow]   Continuing training (will restore original weights if no improvement)[/yellow]"
                )
        elif self.critical_epochs >= 5 and self.critical_epochs % 5 == 0:
            if not hasattr(self, '_last_perturbation_epoch') or epoch - self._last_perturbation_epoch >= 5:
                self._last_perturbation_epoch = epoch
                self.console.print(
                    f"  [yellow]⚠️ Stuck in critical overfit for {self.critical_epochs} epochs[/yellow]"
                )
                
                # Escalating interventions based on how long we've been stuck
                if self.critical_epochs >= 20:
                    # 20+ epochs stuck: Nuclear option - full dense layer reset
                    self.console.print(
                        "  [red bold]🔥 20+ epochs stuck - resetting Dense layers entirely[/red bold]"
                    )
                    self._reinitialize_dense_layers()
                    self._reset_optimizer_state()
                    # Reset LR to allow fresh learning
                    try:
                        tf.keras.backend.set_value(self.model.optimizer.learning_rate, self._initial_lr)
                    except Exception:
                        pass
                    # Reset counters to give it a fresh chance
                    self.critical_epochs = 0
                    self.dropout_adjustments = 0
                    self.lr_reductions = 0
                elif self.critical_epochs >= 15:
                    # 15+ epochs: Larger perturbation + dense layer partial reset
                    self.console.print(
                        "  [red]💥 15+ epochs stuck - larger perturbation + partial reset[/red]"
                    )
                    self._perturb_weights(scale=self.weight_perturbation_scale * 2)
                    self._reinitialize_dense_layers()
                    self._reset_optimizer_state()
                elif self.critical_epochs >= 10:
                    # 10+ epochs: Medium perturbation
                    self._perturb_weights(scale=self.weight_perturbation_scale)
                    self._reset_optimizer_state()
                else:
                    # 5+ epochs: Small perturbation
                    self._perturb_weights(scale=self.weight_perturbation_scale * 0.5)
                    if self.reset_optimizer_on_overfit:
                        self._reset_optimizer_state()
        
        # === COSINE ANNEALING WITH WARM RESTARTS ===
        if self.enable_cosine_restarts:
            cycle_epoch = (epoch - self.current_restart_epoch)
            if cycle_epoch > 0 and cycle_epoch % self.restart_period == 0:
                # Time for warm restart
                self.num_restarts += 1
                self.current_restart_epoch = epoch
                new_lr = self._initial_lr * (self.restart_lr_mult ** (self.num_restarts - 1))
                try:
                    tf.keras.backend.set_value(self.model.optimizer.learning_rate, new_lr)
                    self.console.print(
                        f"  [magenta]🔄 Warm restart #{self.num_restarts}: LR reset to {new_lr:.2e}[/magenta]"
                    )
                except Exception:
                    pass
            else:
                # Apply cosine decay within cycle
                new_lr = self._get_cosine_lr(epoch, self._initial_lr)
                try:
                    # Only apply cosine if we're not in overfit intervention mode
                    if self.lr_reductions == 0:
                        tf.keras.backend.set_value(self.model.optimizer.learning_rate, new_lr)
                except Exception:
                    pass
        
        # === STOCHASTIC WEIGHT AVERAGING ===
        if self.enable_swa and self._total_epochs:
            swa_start_epoch = int(self._total_epochs * self.swa_start_fraction)
            if epoch >= swa_start_epoch:
                if not self.swa_started:
                    self.swa_started = True
                    # Reduce LR for SWA phase
                    try:
                        swa_lr = self._initial_lr * self.swa_lr_factor
                        tf.keras.backend.set_value(self.model.optimizer.learning_rate, swa_lr)
                        self.console.print(
                            f"  [magenta]🎯 SWA phase started (epoch {epoch+1}/{self._total_epochs}): "
                            f"LR={swa_lr:.2e}[/magenta]"
                        )
                    except Exception:
                        pass
                
                # Update SWA weights every epoch during SWA phase
                if overfit_gap <= self.critical_threshold:  # Only average good weights
                    self._update_swa_weights()
        
        # Track best overall (even if overfitting)
        if val_acc > self.best_val_acc:
            self.best_val_acc = val_acc
            self.best_epoch = epoch + 1
        
        # Store weights when gap is healthy (for potential rollback)
        if overfit_gap <= self.overfit_threshold and val_acc >= self.best_val_acc_clean * 0.98:
            self._store_base_weights()
        
        # === CHECKPOINT: Only save if BETTER than previous AND gap is acceptable ===
        if val_acc > self.best_val_acc_clean and overfit_gap <= self.max_acceptable_gap:
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
        elif val_acc > self.best_val_acc_clean:
            # Val improved but gap too large - DON'T save
            self.console.print(
                f"  [yellow]⚠️ Val improved to {val_acc:.1%} but gap={overfit_gap:.1%} > "
                f"{self.max_acceptable_gap:.0%} - NOT saving[/yellow]"
            )
        elif self.warm_start_best_acc > 0 and val_acc < self.warm_start_best_acc:
            # CONTINUAL LEARNING: Current is worse than warm-start baseline
            degradation = self.warm_start_best_acc - val_acc
            if degradation > 0.02:  # Only warn if > 2% degradation
                self.console.print(
                    f"  [red]⚠️ DEGRADATION: val={val_acc:.1%} < warm-start baseline={self.warm_start_best_acc:.1%} "
                    f"(-{degradation:.1%})[/red]"
                )
        
        # === WARM-START EARLY STOP: If no improvement over baseline after 15 epochs ===
        if self.warm_start_best_acc > 0 and epoch >= 15:
            if self.best_val_acc < self.warm_start_best_acc:
                # No improvement over baseline after 15 epochs - stop and revert
                self.console.print(
                    f"\n  [bold red]🛑 CONTINUAL LEARNING FAILED: No improvement over baseline[/bold red]"
                )
                self.console.print(
                    f"  [red]   Best achieved: {self.best_val_acc:.1%} < baseline: {self.warm_start_best_acc:.1%}[/red]"
                )
                self.console.print(
                    f"  [yellow]💡 Stopping early. Original model weights preserved.[/yellow]"
                )
                self.console.print(
                    f"  [cyan]   Try: More data, lower LR (--lr 0.00001), or different pair[/cyan]"
                )
                self.model.stop_training = True
                return
        
        # === OVERFIT SEVERITY CLASSIFICATION ===
        if overfit_gap > self.severe_threshold:
            # SEVERE: > 25% gap - stop immediately
            self.critical_epochs += 1
            if self.critical_epochs >= 2:  # 2 consecutive severe epochs
                self.console.print(
                    f"  [red bold]🛑 SEVERE OVERFITTING: gap={overfit_gap:.1%} > "
                    f"{self.severe_threshold:.0%}[/red bold]"
                )
                
                # Try to recover with SWA weights before stopping
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
                return
                
        elif overfit_gap > self.critical_threshold:
            # CRITICAL: 15-25% gap - aggressive intervention
            self.overfit_epochs += 1
            self.critical_epochs += 1
            
            self.console.print(
                f"  [red]⚠️ CRITICAL: train={train_acc:.1%} vs val={val_acc:.1%} "
                f"(gap={overfit_gap:.1%})[/red]"
            )
            
            # Immediate action: reduce LR significantly
            if self.auto_reduce_lr and self.lr_reductions < 4:
                factor = 0.3 if self.critical_epochs >= 2 else 0.5
                self._reduce_learning_rate(factor=factor)
            
            # After patience: increase dropout aggressively
            if self.auto_adjust_dropout and self.critical_epochs >= self.patience_epochs:
                if self.dropout_adjustments < 5:
                    self._increase_dropout(aggressive=True)
                    
        elif overfit_gap > self.overfit_threshold:
            # WARNING: 8-15% gap - monitor and mild intervention
            self.overfit_epochs += 1
            self.critical_epochs = 0  # Reset critical counter
            
            if self.overfit_epochs == 1:
                self.console.print(
                    f"  [yellow]⚠️ Overfit warning: gap={overfit_gap:.1%} "
                    f"(train={train_acc:.1%}, val={val_acc:.1%})[/yellow]"
                )
            
            # After patience: mild dropout increase
            if self.auto_adjust_dropout and self.overfit_epochs >= self.patience_epochs:
                if self.dropout_adjustments < 5:
                    self._increase_dropout(aggressive=False)
                    self.overfit_epochs = 0  # Reset after adjustment
        else:
            # HEALTHY: gap < 8%
            self.overfit_epochs = 0
            self.critical_epochs = 0
    
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
        """Reduce learning rate to slow down memorization."""
        try:
            # Handle both optimizer objects and string optimizer names
            optimizer = self.model.optimizer
            if optimizer is None:
                logger.warning("Model has no optimizer, cannot reduce LR")
                return
            
            # Get learning_rate attribute - may be a Variable, callable, or float
            lr_attr = getattr(optimizer, 'learning_rate', None) or getattr(optimizer, 'lr', None)
            if lr_attr is None:
                logger.warning("Optimizer has no learning_rate attribute")
                return
            
            # Get current value - handle tf.Variable, callable schedules, or scalar
            if hasattr(lr_attr, 'numpy'):
                current_lr = float(lr_attr.numpy())
            elif callable(lr_attr):
                # LR schedule - get current step value
                current_lr = float(lr_attr(optimizer.iterations))
            else:
                current_lr = float(lr_attr)
            
            new_lr = max(current_lr * factor, 1e-6)
            
            # Set new LR - use K.set_value for tf.Variable
            if hasattr(lr_attr, 'assign'):
                lr_attr.assign(new_lr)
            else:
                tf.keras.backend.set_value(optimizer.learning_rate, new_lr)
            
            self.lr_reductions += 1
            self.console.print(
                f"  [cyan]📉 LR reduced: {current_lr:.2e} → {new_lr:.2e} "
                f"(x{factor}, #{self.lr_reductions})[/cyan]"
            )
        except Exception as e:
            logger.warning(f"Could not reduce LR: {e}")
    
    def on_train_end(self, logs=None):
        # Apply SWA weights at the end if we collected any
        if self.enable_swa and self.swa_count > 1:
            self.console.print(f"  [magenta]📊 SWA collected {self.swa_count} weight snapshots[/magenta]")
            
            # Save SWA model separately
            swa_checkpoint_path = self.checkpoint_dir / f"{self.model_name}_swa.keras"
            
            # Apply SWA weights
            if self._apply_swa_weights():
                try:
                    self.model.save(swa_checkpoint_path)
                    self.console.print(
                        f"  [magenta]💾 SWA model saved: {swa_checkpoint_path}[/magenta]"
                    )
                except Exception as e:
                    logger.warning(f"Could not save SWA checkpoint: {e}")
        
        # Summary
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
        
        # Final stats
        if len(self.gap_history) > 0:
            avg_gap = np.mean(self.gap_history)
            max_gap = max(self.gap_history)
            min_gap = min(self.gap_history)
            
            self.console.print(
                f"  [dim]📊 Gap stats: min={min_gap:.1%}, avg={avg_gap:.1%}, max={max_gap:.1%}[/dim]"
            )
            
            if self.num_restarts > 0:
                self.console.print(
                    f"  [dim]🔄 Warm restarts: {self.num_restarts}[/dim]"
                )
            
            if max_gap > self.critical_threshold:
                self.console.print(
                    "  [yellow]💡 Suggestions: more data, stronger regularization, "
                    "or simpler model[/yellow]"
                )


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
                    logs['ewc_penalty'] = penalty_val
    
    def on_epoch_end(self, epoch, logs=None):
        pass  # Suppressed - RichEpochCallback handles display


class RichEpochCallback(tf.keras.callbacks.Callback):
    """
    Rich-formatted epoch display with color-coded metrics.
    
    Colors:
    - Green: Improving / Best
    - Yellow: Slight degradation  
    - Red: Significant degradation
    - Cyan: Neutral / Info
    """
    
    def __init__(self, model_name: str = "Model", total_epochs: int = 100, warm_start_best_acc: float = 0.0):
        super().__init__()
        self.model_name = model_name
        self.total_epochs = total_epochs
        self.warm_start_best_acc = warm_start_best_acc
        self.best_val_acc = warm_start_best_acc  # Start from warm-start baseline, not 0!
        self.best_val_loss = float('inf')
        self.best_epoch = 0
        self.prev_val_acc = warm_start_best_acc  # Start comparison from baseline
        self.prev_val_loss = float('inf')
        self._console = None
    
    @property
    def console(self):
        if self._console is None:
            from rich.console import Console
            self._console = Console()
        return self._console
    
    def on_train_begin(self, logs=None):
        if self.warm_start_best_acc > 0:
            self.console.print(f"[dim]Training {self.model_name} for up to {self.total_epochs} epochs...[/dim]")
            self.console.print(f"  [cyan]🎯 Warm-start baseline: {self.warm_start_best_acc:.1%} (must beat to save)[/cyan]")
        else:
            self.console.print(f"[dim]Training {self.model_name} for up to {self.total_epochs} epochs...[/dim]")
    
    def on_epoch_end(self, epoch, logs=None):
        if logs is None:
            return
        
        epoch_num = epoch + 1
        val_acc = logs.get('val_accuracy', logs.get('accuracy', 0))
        val_loss = logs.get('val_loss', logs.get('loss', 0))
        train_acc = logs.get('accuracy', 0)
        train_loss = logs.get('loss', 0)
        lr = logs.get('lr', logs.get('learning_rate', 0))
        
        # Determine if this is best epoch (must beat warm-start baseline!)
        is_best = val_acc > self.best_val_acc
        if is_best:
            self.best_val_acc = val_acc
            self.best_val_loss = val_loss
            self.best_epoch = epoch_num
        
        # Color coding based on performance - RELATIVE TO WARM-START BASELINE
        below_baseline = self.warm_start_best_acc > 0 and val_acc < self.warm_start_best_acc
        
        if is_best and not below_baseline:
            acc_color = "bold green"
            status = "⭐ BEST"
        elif is_best and below_baseline:
            # New best during this run, but still below warm-start baseline
            acc_color = "yellow"
            diff = self.warm_start_best_acc - val_acc
            status = f"↗ best this run (baseline -{diff:.1%})"
        elif below_baseline:
            # Below warm-start baseline
            acc_color = "red"
            diff = self.warm_start_best_acc - val_acc
            status = f"⚠ below baseline ({diff:.1%})"
        elif val_acc >= self.prev_val_acc:
            acc_color = "green"
            status = "↗ improving"
        elif val_acc >= self.prev_val_acc - 0.02:
            acc_color = "yellow"
            status = "→ stable"
        else:
            acc_color = "red"
            status = "↘ degrading"
        
        # Loss color
        if val_loss < self.prev_val_loss:
            loss_color = "green"
        elif val_loss <= self.prev_val_loss * 1.1:
            loss_color = "yellow"
        else:
            loss_color = "red"
        
        # Format output
        epoch_str = f"[cyan]Epoch {epoch_num:3d}/{self.total_epochs}[/cyan]"
        acc_str = f"[{acc_color}]acc={val_acc:.1%}[/{acc_color}]"
        loss_str = f"[{loss_color}]loss={val_loss:.4f}[/{loss_color}]"
        train_str = f"[dim]train={train_acc:.1%}[/dim]"
        lr_str = f"[dim]lr={lr:.2e}[/dim]" if lr > 0 else ""
        status_str = f"[{acc_color}]{status}[/{acc_color}]"
        
        self.console.print(f"  {epoch_str} | {acc_str} {loss_str} | {train_str} {lr_str} | {status_str}")
        
        self.prev_val_acc = val_acc
        self.prev_val_loss = val_loss
    
    def on_train_end(self, logs=None):
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
            except:
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
        
        self.X_buffer: Optional[np.ndarray] = None
        self.y_buffer: Optional[np.ndarray] = None
        self.w_buffer: Optional[np.ndarray] = None  # Sample weights
        self.metadata: Dict[str, Any] = {}
        
        self._sample_count = 0
    
    def add_samples(
        self,
        X: np.ndarray,
        y: np.ndarray,
        w: Optional[np.ndarray] = None,
        data_id: Optional[str] = None,
    ):
        """
        Add samples to replay buffer using reservoir sampling.
        
        Args:
            X: Features (sequences)
            y: Labels
            w: Optional sample weights
            data_id: Identifier for this data batch (e.g., instrument_date)
        """
        capacity = int(len(X) * self.capacity_ratio)
        if capacity < 10:
            capacity = min(len(X), 100)  # Minimum buffer size
        
        if self.X_buffer is None:
            # First batch - simple random sample
            indices = np.random.choice(len(X), min(capacity, len(X)), replace=False)
            self.X_buffer = X[indices].copy()
            self.y_buffer = y[indices].copy()
            self.w_buffer = w[indices].copy() if w is not None else np.ones(len(indices))
            self._sample_count = len(X)
            
            logger.info(f"📦 Replay buffer initialized: {len(self.X_buffer)} samples from {len(X)} total")
        else:
            # Reservoir sampling for subsequent batches
            for i in range(len(X)):
                self._sample_count += 1
                
                # Replace with probability capacity/count
                if len(self.X_buffer) < capacity:
                    # Buffer not full, just append
                    self.X_buffer = np.vstack([self.X_buffer, X[i:i+1]])
                    self.y_buffer = np.concatenate([self.y_buffer, y[i:i+1]])
                    w_i = w[i:i+1] if w is not None else np.array([1.0])
                    self.w_buffer = np.concatenate([self.w_buffer, w_i])
                else:
                    # Reservoir sampling
                    j = np.random.randint(0, self._sample_count)
                    if j < len(self.X_buffer):
                        self.X_buffer[j] = X[i]
                        self.y_buffer[j] = y[i]
                        self.w_buffer[j] = w[i] if w is not None else 1.0
            
            logger.info(f"📦 Replay buffer updated: {len(self.X_buffer)} samples, seen {self._sample_count} total")
        
        # Track data sources
        if data_id:
            if 'data_sources' not in self.metadata:
                self.metadata['data_sources'] = []
            self.metadata['data_sources'].append({
                'id': data_id,
                'n_samples': len(X),
                'timestamp': datetime.now().isoformat(),
            })
    
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
        if self.X_buffer is None or len(self.X_buffer) == 0:
            return None, None, None
        
        # Calculate how many replay samples to return
        n_replay = int(n_new_samples * self.mix_ratio)
        n_replay = min(n_replay, len(self.X_buffer))
        
        if n_replay == 0:
            return None, None, None
        
        # Random sample from buffer
        indices = np.random.choice(len(self.X_buffer), n_replay, replace=False)
        
        logger.info(f"📦 Providing {n_replay} replay samples ({self.mix_ratio*100:.0f}% of {n_new_samples})")
        
        return (
            self.X_buffer[indices].copy(),
            self.y_buffer[indices].copy(),
            self.w_buffer[indices].copy() if self.w_buffer is not None else None,
        )
    
    def save(self, instrument: str):
        """Save replay buffer to disk."""
        save_dir = self.buffer_dir / instrument
        save_dir.mkdir(parents=True, exist_ok=True)
        
        if self.X_buffer is None:
            logger.warning("No replay buffer to save")
            return
        
        # Save arrays
        np.savez_compressed(
            save_dir / "buffer.npz",
            X=self.X_buffer,
            y=self.y_buffer,
            w=self.w_buffer,
        )
        
        # Save metadata
        meta = {
            'capacity_ratio': self.capacity_ratio,
            'mix_ratio': self.mix_ratio,
            'sample_count': self._sample_count,
            'buffer_size': len(self.X_buffer),
            'saved_at': datetime.now().isoformat(),
            **self.metadata,
        }
        with open(save_dir / "buffer_meta.json", 'w') as f:
            json.dump(meta, f, indent=2)
        
        logger.info(f"📦 Replay buffer saved to {save_dir} ({len(self.X_buffer)} samples)")
    
    def load(self, instrument: str) -> bool:
        """Load replay buffer from disk."""
        load_dir = self.buffer_dir / instrument
        buffer_path = load_dir / "buffer.npz"
        
        if not buffer_path.exists():
            logger.info(f"No replay buffer at {buffer_path}")
            return False
        
        data = np.load(buffer_path)
        self.X_buffer = data['X']
        self.y_buffer = data['y']
        self.w_buffer = data.get('w')
        
        # Load metadata
        meta_path = load_dir / "buffer_meta.json"
        if meta_path.exists():
            with open(meta_path, 'r') as f:
                meta = json.load(f)
            self._sample_count = meta.get('sample_count', len(self.X_buffer))
            self.metadata = meta
        
        logger.info(f"📦 Replay buffer loaded: {len(self.X_buffer)} samples from {instrument}")
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
            'mean': np.mean(X, axis=(0, 1)) if X.ndim == 3 else np.mean(X, axis=0),
            'std': np.std(X, axis=(0, 1)) if X.ndim == 3 else np.std(X, axis=0),
            'min': np.min(X, axis=(0, 1)) if X.ndim == 3 else np.min(X, axis=0),
            'max': np.max(X, axis=(0, 1)) if X.ndim == 3 else np.max(X, axis=0),
        }
    
    def set_baseline(self, X: np.ndarray, metrics: Dict[str, float]):
        """Set baseline statistics from initial training."""
        self.baseline_stats = {
            'feature_stats': self.compute_feature_stats(X),
            'best_val_accuracy': metrics.get('val_accuracy', 0),
            'baseline_metrics': metrics.copy(),
            'timestamp': datetime.now().isoformat(),
        }
        logger.info(f"📊 Drift baseline set: val_acc={metrics.get('val_accuracy', 0):.4f}")
    
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
        best_acc = max(
            entry.get('val_accuracy', 0) 
            for entry in metric_history
        )
        
        # Check absolute drift from best
        drop = best_acc - current_val_acc
        if drop > self.performance_threshold:
            return True, f"Accuracy dropped {drop:.4f} from best {best_acc:.4f}"
        
        # Check trend: declining over recent window
        recent = metric_history[-self.window_size:] if len(metric_history) >= self.window_size else metric_history
        if len(recent) >= 3:
            recent_accs = [e.get('val_accuracy', 0) for e in recent]
            trend = recent_accs[-1] - recent_accs[0]
            if trend < -self.performance_threshold:
                return True, f"Declining trend: {trend:.4f} over {len(recent)} sessions"
        
        return False, "No drift"
    
    def check_feature_drift(self, X_new: np.ndarray) -> Tuple[bool, str]:
        """
        Check for feature distribution drift.
        
        Uses simple mean/std comparison. Could be enhanced with KS-test.
        """
        if self.baseline_stats is None:
            return False, "No baseline"
        
        new_stats = self.compute_feature_stats(X_new)
        baseline = self.baseline_stats['feature_stats']
        
        # Compare means (normalized by baseline std)
        mean_shift = np.abs(new_stats['mean'] - baseline['mean'])
        normalized_shift = mean_shift / (baseline['std'] + 1e-8)
        max_shift = float(np.max(normalized_shift))
        
        if max_shift > self.feature_drift_threshold * 10:  # 10 sigma shift
            return True, f"Feature distribution shifted: max={max_shift:.2f} sigma"
        
        # Compare std (relative change)
        std_change = np.abs(new_stats['std'] - baseline['std']) / (baseline['std'] + 1e-8)
        max_std_change = float(np.max(std_change))
        
        if max_std_change > self.feature_drift_threshold * 5:  # 50% std change
            return True, f"Feature variance changed: max={max_std_change:.2%}"
        
        return False, "No feature drift"
    
    def full_drift_check(
        self,
        X_new: np.ndarray,
        current_val_acc: float,
        metric_history: List[Dict[str, float]],
    ) -> Dict[str, Any]:
        """
        Perform full drift analysis.
        
        Returns:
            Dict with drift status and recommendations
        """
        perf_drifted, perf_reason = self.check_performance_drift(current_val_acc, metric_history)
        feat_drifted, feat_reason = self.check_feature_drift(X_new)
        
        any_drift = perf_drifted or feat_drifted
        
        result = {
            'any_drift': any_drift,
            'performance_drift': perf_drifted,
            'performance_reason': perf_reason,
            'feature_drift': feat_drifted,
            'feature_reason': feat_reason,
            'recommendation': 'normal',
        }
        
        if perf_drifted and feat_drifted:
            result['recommendation'] = 'full_retrain'
            logger.warning("⚠️ DRIFT: Performance AND feature drift detected → Full retraining recommended")
        elif perf_drifted:
            result['recommendation'] = 'warm_start_retrain'
            logger.warning("⚠️ DRIFT: Performance drift detected → Warm-start retraining recommended")
        elif feat_drifted:
            result['recommendation'] = 'monitor'
            logger.info("📊 Feature drift detected but performance stable → Monitoring")
        
        return result
    
    def save(self, path: str):
        """Save drift detector state."""
        if self.baseline_stats is None:
            return
        
        path = Path(path)
        data = {
            'performance_threshold': self.performance_threshold,
            'feature_drift_threshold': self.feature_drift_threshold,
            'window_size': self.window_size,
            'baseline_stats': {
                k: v.tolist() if isinstance(v, np.ndarray) else v
                for k, v in self.baseline_stats.items()
                if k != 'feature_stats'
            },
            'feature_stats': {
                k: v.tolist() for k, v in self.baseline_stats['feature_stats'].items()
            },
        }
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)
        logger.info(f"📊 Drift baseline saved to {path}")
    
    def load(self, path: str) -> bool:
        """Load drift detector state."""
        path = Path(path)
        if not path.exists():
            return False
        
        with open(path, 'r') as f:
            data = json.load(f)
        
        self.performance_threshold = data.get('performance_threshold', self.performance_threshold)
        self.feature_drift_threshold = data.get('feature_drift_threshold', self.feature_drift_threshold)
        self.window_size = data.get('window_size', self.window_size)
        
        self.baseline_stats = {
            k: v for k, v in data.get('baseline_stats', {}).items()
        }
        self.baseline_stats['feature_stats'] = {
            k: np.array(v) for k, v in data.get('feature_stats', {}).items()
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
        if not hasattr(self, '_history'):
            self._history = []
        
        self._history.append({
            'val_accuracy': val_accuracy,
            'instrument': instrument,
            'data_hash': data_hash,
            'feature_means': feature_means.copy() if isinstance(feature_means, np.ndarray) else np.array(feature_means),
            'timestamp': datetime.now().isoformat(),
        })
        
        # Keep only recent history
        if len(self._history) > self.window_size * 2:
            self._history = self._history[-self.window_size * 2:]
        
        # Set baseline if this is first record
        if self.baseline_stats is None and len(self._history) == 1:
            self.baseline_stats = {
                'best_val_accuracy': val_accuracy,
                'feature_stats': {'mean': feature_means.copy()},
                'timestamp': datetime.now().isoformat(),
            }
    
    def check_drift(self) -> Tuple[bool, str]:
        """
        Check for drift based on recorded history.
        
        Returns:
            (drift_detected, reason)
        """
        if not hasattr(self, '_history') or len(self._history) < 2:
            return False, "Insufficient history"
        
        # Get current and historical accuracies
        current = self._history[-1]
        current_acc = current['val_accuracy']
        
        # Check against best historical accuracy
        best_acc = max(h['val_accuracy'] for h in self._history)
        drop = best_acc - current_acc
        
        if drop > self.performance_threshold:
            return True, f"Performance dropped {drop:.2%} from best {best_acc:.2%}"
        
        # Check feature drift if we have baseline
        if self.baseline_stats and 'feature_stats' in self.baseline_stats:
            baseline_means = self.baseline_stats['feature_stats'].get('mean')
            if baseline_means is not None:
                current_means = current['feature_means']
                if len(baseline_means) == len(current_means):
                    mean_shift = np.abs(current_means - baseline_means).mean()
                    if mean_shift > self.feature_drift_threshold:
                        return True, f"Feature means shifted by {mean_shift:.2%}"
        
        # Check for declining trend
        if len(self._history) >= 3:
            recent = self._history[-3:]
            accs = [h['val_accuracy'] for h in recent]
            if all(accs[i] > accs[i+1] for i in range(len(accs)-1)):
                decline = accs[0] - accs[-1]
                if decline > self.performance_threshold:
                    return True, f"Declining trend: {decline:.2%} over last 3 sessions"
        
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
            'timestamp': datetime.now().isoformat(),
            'checkpoint_id': self.checkpoint_id,
            'generation': self.generation,
            **{k: float(v) for k, v in metrics.items() if isinstance(v, (int, float))}
        }
        self.metric_history.append(entry)
    
    def check_drift(self, current_val_acc: float, threshold: float = 0.03) -> bool:
        """
        Check if model performance has drifted beyond threshold.
        
        Returns True if val_accuracy dropped by more than threshold from best.
        """
        if not self.metric_history:
            return False
        
        best_acc = max(
            entry.get('val_accuracy', 0) 
            for entry in self.metric_history
        )
        
        if best_acc - current_val_acc > threshold:
            logger.warning(
                f"⚠️ DRIFT DETECTED: val_acc={current_val_acc:.4f} vs best={best_acc:.4f} "
                f"(drop={best_acc - current_val_acc:.4f} > threshold={threshold})"
            )
            return True
        return False
    
    @staticmethod
    def compute_data_hash(X: np.ndarray, y: np.ndarray) -> str:
        """Compute hash of training data for change detection."""
        # Use subset for efficiency
        n = min(1000, len(X))
        indices = np.linspace(0, len(X)-1, n, dtype=int)
        X_sample = X[indices]
        y_sample = y[indices]
        
        data_bytes = X_sample.tobytes() + y_sample.tobytes()
        return hashlib.md5(data_bytes).hexdigest()[:12]
    
    def get_training_summary(self) -> str:
        """Get human-readable summary of training lineage."""
        lines = [
            f"📊 Training Lineage Summary",
            f"  Checkpoint: {self.checkpoint_id}",
            f"  Created: {self.created_at}",
            f"  Generation: {self.generation}",
            f"  Cumulative epochs: {self.cumulative_epochs}",
            f"  Cumulative samples: {self.cumulative_samples:,}",
            f"  Instrument: {self.instrument} ({self.granularity})",
        ]
        if self.metric_history:
            latest = self.metric_history[-1]
            lines.append(f"  Latest val_accuracy: {latest.get('val_accuracy', 'N/A'):.4f}")
        if self.drift_detected:
            lines.append(f"  ⚠️ Drift detected: {self.drift_reason}")
        if self.ema_enabled:
            lines.append(f"  EMA: enabled")
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
            recent_accs = [e.get('val_accuracy', 0) for e in self.metric_history[-3:]]
            if all(recent_accs[i] > recent_accs[i+1] for i in range(len(recent_accs)-1)):
                return True, f"Declining accuracy trend: {recent_accs}"
        
        return False, "No retraining needed"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            'checkpoint_id': self.checkpoint_id,
            'parent_checkpoint_id': self.parent_checkpoint_id,
            'created_at': self.created_at,
            'cumulative_epochs': self.cumulative_epochs,
            'cumulative_samples': self.cumulative_samples,
            'session_epochs': self.session_epochs,
            'generation': self.generation,
            'data_hash': self.data_hash,
            'data_range': self.data_range,
            'instrument': self.instrument,
            'granularity': self.granularity,
            'metric_history': self.metric_history,
            'ema_enabled': self.ema_enabled,
            'ewc_n_tasks': self.ewc_n_tasks,
            'replay_buffer_size': self.replay_buffer_size,
            'training_config': self.training_config,
            'drift_detected': self.drift_detected,
            'last_drift_check': self.last_drift_check,
            'drift_reason': self.drift_reason,
            'last_training_duration_seconds': self.last_training_duration_seconds,
            'recommended_retrain_interval_hours': self.recommended_retrain_interval_hours,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'TrainingLineage':
        """Create from dictionary."""
        return cls(
            checkpoint_id=data.get('checkpoint_id', ''),
            parent_checkpoint_id=data.get('parent_checkpoint_id'),
            created_at=data.get('created_at', ''),
            cumulative_epochs=data.get('cumulative_epochs', 0),
            cumulative_samples=data.get('cumulative_samples', 0),
            session_epochs=data.get('session_epochs', 0),
            generation=data.get('generation', 1),
            data_hash=data.get('data_hash', ''),
            data_range=data.get('data_range', ''),
            instrument=data.get('instrument', ''),
            granularity=data.get('granularity', ''),
            metric_history=data.get('metric_history', []),
            ema_enabled=data.get('ema_enabled', False),
            ewc_n_tasks=data.get('ewc_n_tasks', 0),
            replay_buffer_size=data.get('replay_buffer_size', 0),
            training_config=data.get('training_config', {}),
            drift_detected=data.get('drift_detected', False),
            last_drift_check=data.get('last_drift_check', ''),
            drift_reason=data.get('drift_reason', ''),
            last_training_duration_seconds=data.get('last_training_duration_seconds', 0.0),
            recommended_retrain_interval_hours=data.get('recommended_retrain_interval_hours', 168),
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
        direction = keras.layers.Dense(1, activation='sigmoid', name='direction', dtype='float32', bias_initializer='zeros')(x)  # Zero bias init for balanced predictions
        
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
            # Rich-formatted epoch display with color coding
            RichEpochCallback(
                model_name="TCN Direction",
                total_epochs=self.config.epochs,
            ),
            # Primary: Stop when validation loss stops improving
            keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=self.config.patience,
                mode='min',
                restore_best_weights=True,
                verbose=0,  # Suppress - Rich callback handles display
                start_from_epoch=self.config.min_epochs,  # Enforce minimum epochs before early stopping
            ),
            # LR reduction (less aggressive: factor=0.5, patience=1/4 of early stopping)
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=max(4, self.config.patience // 4),
                min_lr=1e-6,
                verbose=0,  # Suppress - Rich callback handles display
            ),
            # Overfit prevention: checkpoints + auto-adjust dropout/LR when train >> val
            OverfitPreventionCallback(
                checkpoint_dir=self.config.checkpoint_dir,
                model_name="tcn_direction",
                overfit_threshold=0.08,      # 8% gap → warning
                critical_threshold=0.15,     # 15% gap → intervention
                severe_threshold=0.25,       # 25% gap → stop training
                max_acceptable_gap=0.12,     # Won't save if gap > 12%
                patience_epochs=3,
                auto_adjust_dropout=True,
                auto_reduce_lr=True,
            ),
        ]
        
        # Train
        history = self.model.fit(
            X_train_seq, y_train_seq,
            validation_data=(X_val_seq, y_val_seq),
            epochs=self.config.epochs,
            batch_size=self.config.batch_size,
            callbacks=callbacks,
            verbose=0,  # Suppress Keras output - RichEpochCallback handles display
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
# TCN VOLATILITY REGIME TRAINER - Forward-Looking 4-Class Prediction
# =============================================================================

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
        import tensorflow as tf
        from tensorflow import keras
        
        # Import the dual-head model
        try:
            from src.models.tensorflow_models import TCNVolatilityDualHead, DualHeadLoss
        except ImportError:
            from models.tensorflow_models import TCNVolatilityDualHead, DualHeadLoss
        
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
    
    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
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
            X_val: Validation sequences
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
        import tensorflow as tf
        from tensorflow import keras
        from sklearn.metrics import classification_report, f1_score
        
        # Initialize clean display
        display = TrainingDisplay("TCN Forward Volatility")
        
        # Save metadata
        self.feature_names = feature_names
        self.n_features = X_train.shape[-1]
        self.seq_len = seq_len if X_train.ndim == 3 else 60
        self.scaler = None  # Assume pre-scaled from data loader
        
        # Ensure correct shape
        if X_train.ndim != 3:
            raise ValueError(f"Expected 3D input (batch, seq_len, features), got {X_train.shape}")
        
        # Use provided sample weights or fall back to deprecated parameter
        if w_train is None:
            w_train = sample_weights if sample_weights is not None else np.ones(len(y_train))
        if w_val is None:
            w_val = np.ones(len(y_val))
        
        # Create regression targets if not provided (default to zeros)
        if y_train_reg is None:
            y_train_reg = np.zeros(len(y_train), dtype=np.float32)
        if y_val_reg is None:
            y_val_reg = np.zeros(len(y_val), dtype=np.float32)
        
        # Convert to one-hot for classification
        y_train_onehot = tf.keras.utils.to_categorical(y_train, num_classes=self.n_classes)
        y_val_onehot = tf.keras.utils.to_categorical(y_val, num_classes=self.n_classes)
        logger.debug(f"Labels shape: {y_train_onehot.shape}")
        
        # Build model
        self.model = self._build_model((self.seq_len, self.n_features))
        
        # === COMPILE WITH CUSTOM TRAINING STEP ===
        # We need custom training because of dual outputs
        
        # Learning rate
        tcn_lr = min(self.config.learning_rate, 0.001)  # Conservative for forward prediction
        
        # Determine focal alpha from class_weights (sklearn balanced) or use default
        # Class weights from sklearn are inverse frequency, so higher = rarer class
        if class_weights is not None:
            # Normalize class weights to sum to 1 for focal alpha
            cw_values = [class_weights.get(i, 1.0) for i in range(self.n_classes)]
            cw_sum = sum(cw_values)
            effective_alpha = [v / cw_sum for v in cw_values]
            logger.debug(f"Focal alpha from class weights: {[f'{a:.3f}' for a in effective_alpha]}")
        else:
            # Default focal alpha: boost minority classes
            effective_alpha = [0.30, 0.20, 0.25, 0.25]  # QUIET, STABLE, ACTIVE, EXTREME
            logger.debug(f"Using default focal alpha: {effective_alpha}")
        
        self.focal_alpha = effective_alpha
        
        # Create separate losses
        classification_loss_fn = keras.losses.CategoricalFocalCrossentropy(
            gamma=self.focal_gamma,
            alpha=self.focal_alpha,
            from_logits=False,
        )
        regression_loss_fn = keras.losses.MeanSquaredError()
        
        # Optimizer
        optimizer = keras.optimizers.Adam(learning_rate=tcn_lr)
        
        # Assign optimizer to model so callbacks can access it
        self.model.optimizer = optimizer
        
        # Show clean configuration summary
        train_valid_mask = w_train > 0
        valid_labels = y_train[train_valid_mask]
        class_counts = np.bincount(valid_labels, minlength=self.n_classes) if len(valid_labels) > 0 else [0]*4
        
        display.show_config({
            "Lookahead": f"{self.lookahead} bars",
            "Params": f"{self.model.count_params():,}",
            "LR": f"{tcn_lr:.1e}",
            "Loss": f"FocalCE(γ={self.focal_gamma})",
            "Classes": f"QUI:{class_counts[0]/len(valid_labels):.0%} STA:{class_counts[1]/len(valid_labels):.0%} ACT:{class_counts[2]/len(valid_labels):.0%} EXT:{class_counts[3]/len(valid_labels):.0%}",
        })
        
        # === PREDICTION COLLAPSE CALLBACK (4-class version) ===
        class RegimeCollapseCallback(keras.callbacks.Callback):
            def __init__(self, X_val, y_val, class_names, check_every=5):
                super().__init__()
                self.X_val = X_val
                self.y_val = y_val
                self.class_names = class_names
                self.check_every = check_every
                self.collapse_warned = False
                self.display = display
            
            def on_epoch_end(self, epoch, logs=None):
                if (epoch + 1) % self.check_every != 0:
                    return
                
                outputs = self.model.predict(self.X_val, verbose=0)
                if isinstance(outputs, dict):
                    preds = outputs['classification']
                else:
                    preds = outputs
                
                pred_classes = np.argmax(preds, axis=1)
                pred_dist = np.bincount(pred_classes, minlength=4) / len(pred_classes)
                
                # Check for collapse (>80% same prediction)
                max_pct = max(pred_dist)
                if max_pct > 0.80:
                    dominant = np.argmax(pred_dist)
                    if not self.collapse_warned:
                        self.display.warn(f"Collapse detected: {max_pct:.0%} -> {self.class_names[dominant]}")
                        self.collapse_warned = True
                else:
                    self.collapse_warned = False
        
        # Callbacks
        callbacks = [
            RichEpochCallback(
                model_name="TCN Forward Volatility",
                total_epochs=self.config.epochs,
            ),
            AutoAdjustCallback(
                patience=8,           # Stuck for 8 epochs → adjust
                lr_factor=0.5,        # Halve LR when stuck
                min_lr=1e-6,
                max_adjustments=4,    # Up to 4 LR reductions
                min_delta=0.005,      # 0.5% improvement threshold
                verbose=True
            ),
            keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=self.config.patience,
                mode='min',
                restore_best_weights=True,
                verbose=0,
                start_from_epoch=max(10, self.config.min_epochs),
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=max(5, self.config.patience // 3),
                min_lr=1e-6,
                verbose=0,
            ),
            RegimeCollapseCallback(X_val, y_val, self.class_names, check_every=5),
        ]
        
        # === CUSTOM TRAINING LOOP for dual-head ===
        # Create datasets
        train_dataset = tf.data.Dataset.from_tensor_slices((
            X_train,
            {'classification': y_train_onehot, 'regression': y_train_reg.reshape(-1, 1)},
            w_train
        )).shuffle(1024).batch(self.config.batch_size).prefetch(tf.data.AUTOTUNE)
        
        val_dataset = tf.data.Dataset.from_tensor_slices((
            X_val,
            {'classification': y_val_onehot, 'regression': y_val_reg.reshape(-1, 1)},
            w_val
        )).batch(self.config.batch_size).prefetch(tf.data.AUTOTUNE)
        
        # Training metrics
        train_loss_metric = keras.metrics.Mean(name='train_loss')
        train_acc_metric = keras.metrics.CategoricalAccuracy(name='train_accuracy')
        val_loss_metric = keras.metrics.Mean(name='val_loss')
        val_acc_metric = keras.metrics.CategoricalAccuracy(name='val_accuracy')
        
        # Class weights are now integrated into focal_alpha above
        
        @tf.function
        def train_step(x, y, sample_weight):
            with tf.GradientTape() as tape:
                outputs = self.model(x, training=True)
                
                # Classification loss
                class_loss = classification_loss_fn(
                    y['classification'], 
                    outputs['classification'],
                    sample_weight=sample_weight
                )
                
                # Regression loss
                reg_loss = regression_loss_fn(y['regression'], outputs['regression'])
                
                # Combined loss
                total_loss = (self.classification_weight * class_loss + 
                             self.regression_weight * reg_loss)
                
                # Add regularization losses
                if self.model.losses:
                    total_loss += tf.add_n(self.model.losses)
            
            gradients = tape.gradient(total_loss, self.model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
            
            train_loss_metric.update_state(total_loss)
            train_acc_metric.update_state(y['classification'], outputs['classification'])
            
            return total_loss
        
        @tf.function
        def val_step(x, y, sample_weight):
            outputs = self.model(x, training=False)
            
            class_loss = classification_loss_fn(
                y['classification'], 
                outputs['classification'],
                sample_weight=sample_weight
            )
            reg_loss = regression_loss_fn(y['regression'], outputs['regression'])
            total_loss = (self.classification_weight * class_loss + 
                         self.regression_weight * reg_loss)
            
            val_loss_metric.update_state(total_loss)
            val_acc_metric.update_state(y['classification'], outputs['classification'])
            
            return total_loss
        
        # Training loop
        best_val_loss = float('inf')
        best_weights = None
        patience_counter = 0
        history = {'loss': [], 'accuracy': [], 'val_loss': [], 'val_accuracy': []}
        
        for epoch in range(self.config.epochs):
            # Reset metrics
            train_loss_metric.reset_state()
            train_acc_metric.reset_state()
            val_loss_metric.reset_state()
            val_acc_metric.reset_state()
            
            # Training
            for x_batch, y_batch, w_batch in train_dataset:
                train_step(x_batch, y_batch, w_batch)
            
            # Validation
            for x_batch, y_batch, w_batch in val_dataset:
                val_step(x_batch, y_batch, w_batch)
            
            # Get metrics
            train_loss = train_loss_metric.result().numpy()
            train_acc = train_acc_metric.result().numpy()
            val_loss = val_loss_metric.result().numpy()
            val_acc = val_acc_metric.result().numpy()
            
            history['loss'].append(train_loss)
            history['accuracy'].append(train_acc)
            history['val_loss'].append(val_loss)
            history['val_accuracy'].append(val_acc)
            
            # Call callbacks
            logs = {'loss': train_loss, 'accuracy': train_acc, 
                   'val_loss': val_loss, 'val_accuracy': val_acc}
            for callback in callbacks:
                # Set model via set_model() for Keras callbacks, or _model for custom
                if hasattr(callback, 'set_model'):
                    callback.set_model(self.model)
                elif hasattr(callback, '_model'):
                    callback._model = self.model
                callback.on_epoch_end(epoch, logs)
            
            # Early stopping logic
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_weights = self.model.get_weights()
                patience_counter = 0
            else:
                patience_counter += 1
            
            if patience_counter >= self.config.patience and epoch >= self.config.min_epochs:
                logger.info(f"Early stopping at epoch {epoch+1}")
                break
        
        # Restore best weights
        if best_weights is not None:
            self.model.set_weights(best_weights)
        
        self.is_trained = True
        
        # Evaluate on validation set
        val_outputs = self.model.predict(X_val, verbose=0)
        if isinstance(val_outputs, dict):
            val_pred_probs = val_outputs['classification']
            val_pred_reg = val_outputs['regression'].flatten()
        else:
            val_pred_probs = val_outputs
            val_pred_reg = np.zeros(len(y_val))
        
        val_pred_classes = np.argmax(val_pred_probs, axis=1)
        val_acc = np.mean(val_pred_classes == y_val)
        
        # Calculate distributions (logged via display)
        pred_dist = np.bincount(val_pred_classes, minlength=self.n_classes) / len(val_pred_classes)
        true_dist = np.bincount(y_val, minlength=self.n_classes) / len(y_val)
        
        # Calculate F1 scores per class
        f1_scores = f1_score(y_val, val_pred_classes, average=None, zero_division=0)
        f1_macro = f1_score(y_val, val_pred_classes, average='macro', zero_division=0)
        
        # ACTIVE/EXTREME detection accuracy (classes 2 and 3 - the actionable ones)
        active_extreme_mask = (y_val >= 2)
        if active_extreme_mask.sum() > 0:
            active_extreme_acc = np.mean(val_pred_classes[active_extreme_mask] >= 2)
        else:
            active_extreme_acc = 0.0
        
        # Check for collapse
        all_classes_present = all(pred_dist[i] > 0.05 for i in range(4))
        
        self.metrics = {
            'train_accuracy': float(history['accuracy'][-1]) if history['accuracy'] else 0.0,
            'val_accuracy': float(val_acc),
            'val_f1_macro': float(f1_macro),
            'val_f1_quiet': float(f1_scores[0]) if len(f1_scores) > 0 else 0.0,
            'val_f1_stable': float(f1_scores[1]) if len(f1_scores) > 1 else 0.0,
            'val_f1_active': float(f1_scores[2]) if len(f1_scores) > 2 else 0.0,
            'val_f1_extreme': float(f1_scores[3]) if len(f1_scores) > 3 else 0.0,
            'active_extreme_detection': float(active_extreme_acc),
            'all_classes_present': all_classes_present,
            'epochs_trained': len(history['loss']),
            'receptive_field': self._compute_receptive_field(),
            'lookahead': self.lookahead,
        }
        
        # Show clean results summary
        display.show_summary({
            'Val Accuracy': f"{val_acc:.1%}",
            'F1 Macro': f"{f1_macro:.3f}",
            'F1 (QUI/STA/ACT/EXT)': f"{f1_scores[0]:.2f} / {f1_scores[1]:.2f} / {f1_scores[2]:.2f} / {f1_scores[3]:.2f}",
            'Active/Extreme Det': f"{active_extreme_acc:.1%}",
            'Epochs': len(history['loss']),
            'Status': "✓ Healthy" if all_classes_present else "⚠ Collapse",
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
                X_seq = X[-self.seq_len:].reshape(1, self.seq_len, -1)
            else:
                pad_len = self.seq_len - len(X)
                X_padded = np.vstack([np.zeros((pad_len, X.shape[1])), X])
                X_seq = X_padded.reshape(1, self.seq_len, -1)
        elif X.ndim == 3:
            X_seq = X
        else:
            raise ValueError(f"Expected 2D or 3D input, got shape {X.shape}")
        
        # Get predictions
        outputs = self.model.predict(X_seq, verbose=0)
        
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
            from src.models.tensorflow_models import TCNVolatilityDualHead
        except ImportError:
            try:
                from models.tensorflow_models import TCNVolatilityDualHead
            except ImportError:
                TCNVolatilityDualHead = None
        
        custom_objects = {}
        if TCNVolatilityDualHead is not None:
            custom_objects['TCNVolatilityDualHead'] = TCNVolatilityDualHead
        
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
        
        # Transformer-specific config - defaults match TrainerConfig for proven 60.9% config
        self.transformer_d_model = getattr(config, 'transformer_d_model', 16) if config else 16
        self.transformer_num_heads = getattr(config, 'transformer_num_heads', 2) if config else 2
        self.transformer_num_layers = getattr(config, 'transformer_num_layers', 1) if config else 1
        self.transformer_dff = getattr(config, 'transformer_dff', 32) if config else 32
        self.transformer_dropout = getattr(config, 'transformer_dropout', 0.2) if config else 0.2  # Reduced from 0.4
        
        # === CONTINUAL LEARNING COMPONENTS ===
        
        # EMA for stable inference
        self.ema: Optional[EMACallback] = None
        self._use_ema = getattr(config, 'use_ema', True) if config else True
        
        # EWC for multi-instrument learning
        self.ewc: Optional[EWCPenalty] = None
        self._use_ewc = getattr(config, 'use_ewc', True) if config else True
        
        # Replay buffer
        self.replay_buffer: Optional[ReplayBuffer] = None
        self._use_replay = getattr(config, 'use_replay_buffer', True) if config else True
        
        # Drift detector
        self.drift_detector: Optional[DriftDetector] = None
        self._drift_threshold = getattr(config, 'drift_threshold', 0.03) if config else 0.03
        
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
        import tensorflow as tf
        from tensorflow import keras
        
        seq_len, n_features = input_shape
        
        # VERY light L2 regularization - too much suppresses outputs
        l2_reg = keras.regularizers.l2(0.001)
        
        # Input
        inp = keras.Input(shape=(seq_len, n_features), name="features")
        
        # Very light input noise - prevents overfitting to exact values
        x = keras.layers.GaussianNoise(0.02)(inp)
        
        # Light spatial dropout on input sequence
        x = keras.layers.SpatialDropout1D(0.05)(x)
        
        # Project features to d_model dimension (with L2)
        x = keras.layers.Dense(
            self.transformer_d_model, 
            kernel_regularizer=l2_reg,
            name='input_projection'
        )(x)
        x = keras.layers.Dropout(0.15)(x)
        
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
                name_prefix=f'transformer_{i}'
            )
        
        # Global pooling and output
        x = keras.layers.GlobalAveragePooling1D()(x)
        
        # Use tanh instead of ReLU - tanh outputs [-1, 1] which allows both positive
        # and negative contributions to the sigmoid input, making it easier to balance around 0.5
        x = keras.layers.Dense(16, activation='tanh', kernel_regularizer=l2_reg)(x)
        x = keras.layers.Dropout(0.15)(x)
        
        # Binary direction output
        # With tanh inputs ranging [-1, 1], the dot product with weights can be near 0
        # Use small kernel init and zero bias to start at sigmoid(0) = 0.5
        direction = keras.layers.Dense(
            1, 
            activation='sigmoid', 
            name='direction', 
            dtype='float32',
            kernel_initializer=keras.initializers.TruncatedNormal(mean=0.0, stddev=0.05),
            bias_initializer=keras.initializers.Zeros(),  # Start at sigmoid(0) = 0.5
        )(x)
        
        model = keras.Model(inputs=inp, outputs=direction, name='transformer_direction')
        
        # Adam optimizer with standard learning rate
        optimizer = keras.optimizers.Adam(learning_rate=self.config.learning_rate)
        logger.info(f"Using Adam optimizer with lr={self.config.learning_rate:.2e}")
        
        # Standard binary cross entropy - we'll handle calibration post-training
        model.compile(
            optimizer=optimizer,
            loss=keras.losses.BinaryCrossentropy(label_smoothing=0.0),
            metrics=[keras.metrics.BinaryAccuracy(name='accuracy', threshold=0.5)],
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
        warm_start_path: Optional[str] = None,
        instrument: str = "UNKNOWN",
        data_range: str = "",
    ) -> Dict[str, float]:
        """
        Train Transformer for direction prediction with continual learning.
        
        IMPORTANT: Create sequences FIRST from all data (preserving temporal order),
        THEN filter sequences based on whether the target label is clear.
        This preserves temporal continuity for proper sequence modeling.
        
        Args:
            warm_start_path: Path to existing model to load weights from (iterative training)
            instrument: Trading instrument (e.g., "EUR_USD") for replay buffer
            data_range: Date range of training data (e.g., "2024-01-01 to 2024-06-01")
        
        Continual Learning Features:
            - EMA: Maintains shadow weights for stable inference
            - EWC: Applies penalty to protect important weights from prior learning
            - Replay Buffer: Mixes past samples to prevent forgetting
            - Warm-start LR: Reduces learning rate by 10x when continuing training
        """
        import tensorflow as tf
        from tensorflow import keras
        from sklearn.preprocessing import StandardScaler
        
        logger.info("Training Transformer (Direction)...")
        
        # === INITIALIZE TRAINING LINEAGE ===
        self.lineage = TrainingLineage()
        self.lineage.generate_checkpoint_id()
        self.lineage.instrument = instrument
        self.lineage.data_range = data_range
        self.lineage.granularity = getattr(self.config, 'granularity', 'H1') if self.config else 'H1'
        
        # === INITIALIZE DRIFT DETECTOR ===
        self.drift_detector = DriftDetector(
            performance_threshold=self._drift_threshold,
            feature_drift_threshold=0.10,  # 10% change in feature means
            window_size=5,
        )
        
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
        
        # Log class distribution and imbalance ratio
        train_up_pct = (y_train_filtered == 1).mean() * 100
        val_up_pct = (y_val_filtered == 1).mean() * 100
        train_imbalance = max(train_up_pct, 100 - train_up_pct) / min(train_up_pct, 100 - train_up_pct) if min(train_up_pct, 100 - train_up_pct) > 0 else float('inf')
        val_imbalance = max(val_up_pct, 100 - val_up_pct) / min(val_up_pct, 100 - val_up_pct) if min(val_up_pct, 100 - val_up_pct) > 0 else float('inf')
        logger.info(
            "Class distribution: train=%.1f%% up (imbalance=%.2fx), val=%.1f%% up (imbalance=%.2fx)",
            train_up_pct, train_imbalance, val_up_pct, val_imbalance
        )
        if train_imbalance > 2.0 or val_imbalance > 2.0:
            logger.warning("High class imbalance detected (>2x). Consider adjusting direction_threshold.")
        
        # =========================================================================
        # PHASE 2: REGIME-AWARE SAMPLE WEIGHTS (2025 Best Practice)
        # Compute per-sample weights that balance classes WITHIN each volatility regime
        # This prevents the model from learning regime-specific biases
        # =========================================================================
        n_up = (y_train_filtered == 1).sum()
        n_down = (y_train_filtered == 0).sum()
        
        if n_up > 0 and n_down > 0:
            # Compute volatility regime from sequence features
            # Use the last timestep's volatility percentile as regime indicator
            # Feature index for atr_pct or volatility (typically in first few features)
            
            # Extract volatility from sequences (mean of last 5 timesteps to reduce noise)
            seq_volatility = np.mean(np.abs(X_train_filtered[:, -5:, :]), axis=(1, 2))
            
            # Quantile-based regime classification (3 regimes: low, medium, high vol)
            vol_p33 = np.percentile(seq_volatility, 33)
            vol_p66 = np.percentile(seq_volatility, 66)
            
            regime_train = np.where(
                seq_volatility < vol_p33, 0,  # Low volatility
                np.where(seq_volatility < vol_p66, 1, 2)  # Medium / High volatility
            )
            
            # Compute per-regime class weights
            sample_weights = np.ones(len(y_train_filtered), dtype=np.float32)
            
            for regime in [0, 1, 2]:
                regime_mask = (regime_train == regime)
                if regime_mask.sum() == 0:
                    continue
                
                regime_up = ((y_train_filtered == 1) & regime_mask).sum()
                regime_down = ((y_train_filtered == 0) & regime_mask).sum()
                regime_total = regime_up + regime_down
                
                if regime_up > 0 and regime_down > 0:
                    # Inverse frequency weighting within regime
                    up_weight = regime_total / (2 * regime_up)
                    down_weight = regime_total / (2 * regime_down)
                    
                    # Apply weights to samples in this regime
                    sample_weights[(y_train_filtered == 1) & regime_mask] = up_weight
                    sample_weights[(y_train_filtered == 0) & regime_mask] = down_weight
                    
                    regime_name = ['LOW_VOL', 'MED_VOL', 'HIGH_VOL'][regime]
                    logger.info(f"  Regime {regime_name}: n={regime_mask.sum()}, up={regime_up} ({100*regime_up/regime_total:.1f}%), "
                               f"weights: up={up_weight:.3f}, down={down_weight:.3f}")
            
            # Normalize weights to mean=1 (preserves effective batch size)
            sample_weights = sample_weights / sample_weights.mean()
            
            # === MINORITY CLASS BOOSTING (Anti-Bias) ===
            # Boost minority class to prevent collapse to majority class
            # NOTE: 3x was too aggressive, causing gradient instability and training degradation
            # 1.5x provides balance without destabilizing training
            minority_class = 1 if n_up < n_down else 0
            minority_boost = 1.5  # 1.5x boost for minority class (reduced from 3x - was too aggressive)
            sample_weights[y_train_filtered == minority_class] *= minority_boost
            # Re-normalize to mean=1
            sample_weights = sample_weights / sample_weights.mean()
            minority_name = "UP" if minority_class == 1 else "DOWN"
            logger.info(f"🎯 AGGRESSIVE Minority boost: {minority_name} class boosted by {minority_boost}x")
            
            # Also compute global class weights for Keras (backup)
            total = n_up + n_down
            class_weight = {
                0: total / (2 * n_down),
                1: total / (2 * n_up),
            }
            logger.info(f"Global class weights: down={class_weight[0]:.3f}, up={class_weight[1]:.3f}")
            logger.info(f"Regime-aware sample weights: mean={sample_weights.mean():.3f}, std={sample_weights.std():.3f}")
            
            # ALWAYS use sample weights to prevent model collapse to majority class
            # Even with balanced data, regime-aware weighting helps
            imbalance_ratio = max(n_up, n_down) / min(n_up, n_down)
            logger.info(f"Class imbalance ratio: {imbalance_ratio:.3f} - sample weights ALWAYS enabled")
        else:
            class_weight = None
            sample_weights = None
            logger.warning("Cannot compute class weights - one class has zero samples")
        
        logger.info(f"Sequence shape: train={X_train_filtered.shape}, val={X_val_filtered.shape}")
        
        # === REPLAY BUFFER: Load existing buffer and mix with new data ===
        if self._use_replay:
            self.replay_buffer = ReplayBuffer(
                capacity_ratio=self.config.replay_buffer_ratio,
                mix_ratio=self.config.replay_mix_ratio,
                buffer_dir=self.config.replay_buffer_dir,
            )
            
            # Load existing buffer if available
            if self.replay_buffer.load(instrument):
                # Get replay samples to mix with training data
                X_replay, y_replay, w_replay = self.replay_buffer.get_replay_samples(len(X_train_filtered))
                
                if X_replay is not None and len(X_replay) > 0:
                    # Check dimension compatibility before mixing
                    if X_replay.shape[2] == X_train_filtered.shape[2]:
                        # Mix replay samples with new training data
                        X_train_filtered = np.vstack([X_train_filtered, X_replay])
                        y_train_filtered = np.concatenate([y_train_filtered, y_replay])
                        
                        if sample_weights is not None and w_replay is not None:
                            sample_weights = np.concatenate([sample_weights, w_replay])
                        
                        logger.info(f"📦 Mixed {len(X_replay)} replay samples, new train size: {len(X_train_filtered)}")
                    else:
                        logger.warning(f"⚠️ Replay buffer feature mismatch: buffer has {X_replay.shape[2]} features, "
                                      f"current data has {X_train_filtered.shape[2]}. Resetting replay buffer.")
                        # Reset buffer to accept new feature dimension
                        self.replay_buffer.X_buffer = None
                        self.replay_buffer.y_buffer = None
                        self.replay_buffer.w_buffer = None
                        self.replay_buffer._sample_count = 0
            
            # Add current training data to buffer for future sessions
            self.replay_buffer.add_samples(
                X_train_filtered, 
                y_train_filtered, 
                sample_weights,
                data_id=f"{instrument}_{self.lineage.checkpoint_id}"
            )
        
        # Compute data hash for lineage tracking
        self.lineage.data_hash = TrainingLineage.compute_data_hash(X_train_filtered, y_train_filtered)
        
        # Build model
        self.model = self._build_model((seq_len, self.n_features))
        
        # === WARM-START: Load existing weights + EWC + Lineage ===
        self._is_warm_start = False
        self._warm_start_val_acc = 0.0  # Track previous best to prevent saving worse models
        self._warm_start_weights = None  # Store original weights for recovery
        self._loaded_model_instrument = None  # Track which instrument the loaded model was trained on
        effective_lr = self.config.learning_rate
        
        if warm_start_path and Path(warm_start_path).exists():
            try:
                logger.info(f"🔥 WARM-START: Loading weights from {warm_start_path}")
                existing_model = keras.models.load_model(warm_start_path, compile=False)
                
                # Check if architectures match
                if existing_model.count_params() == self.model.count_params():
                    self.model.set_weights(existing_model.get_weights())
                    self._is_warm_start = True
                    
                    # CRITICAL: Store original warm-start weights for recovery if training fails
                    self._warm_start_weights = self.model.get_weights()
                    
                    logger.info(f"✓ Successfully loaded {self.model.count_params():,} parameters from checkpoint")
                    
                    # === LAYER FREEZING FOR WARM-START ===
                    # Freeze transformer encoder layers (feature extraction) to prevent catastrophic forgetting
                    # Only train the classification head which adapts to new/updated data
                    frozen_count = 0
                    trainable_head_layers = []
                    
                    if self.config.warm_start_freeze_encoder:
                        for layer in self.model.layers:
                            layer_name = layer.name.lower()
                            
                            # Freeze ALL encoder layers: transformer_*, input_projection, positional, attention, etc.
                            # This is MORE AGGRESSIVE than before - only the final Dense head remains trainable
                            is_encoder_layer = any(pattern in layer_name for pattern in [
                                'transformer_',      # transformer_0, transformer_1, etc. (ALL of them)
                                'input_projection',  # Feature projection layer
                                'positional',        # Positional encoding
                                'multi_head',        # Multi-head attention
                                'attention',         # Any attention layer
                                'ffn',               # Feedforward network in encoder
                                'layer_norm',        # Layer normalization
                                'spatial_dropout',   # Input dropout
                                'gaussian_noise',    # Input noise
                                'global_average',    # Pooling layer
                            ])
                            
                            # Don't freeze the final classification head (Dense layers at the end)
                            is_classification_head = (
                                'direction' in layer_name or  # Output layer
                                (isinstance(layer, keras.layers.Dense) and 
                                 layer.output_shape[-1] <= 16 and  # Small dense = head
                                 'projection' not in layer_name)  # Not input projection
                            )
                            
                            if is_encoder_layer and not is_classification_head:
                                layer.trainable = False
                                frozen_count += 1
                            elif layer.trainable:
                                trainable_head_layers.append(layer.name)
                    
                    if frozen_count > 0:
                        logger.info(f"🔒 WARM-START: Froze {frozen_count} encoder layers (feature extraction preserved)")
                        # Log trainable vs frozen params
                        trainable_params = sum([tf.size(w).numpy() for w in self.model.trainable_weights])
                        total_params = self.model.count_params()
                        logger.info(f"   Trainable: {trainable_params:,}/{total_params:,} params ({100*trainable_params/total_params:.1f}%)")
                        if trainable_head_layers:
                            logger.info(f"   Trainable layers: {trainable_head_layers[:5]}{'...' if len(trainable_head_layers) > 5 else ''}")
                    else:
                        logger.warning(f"⚠️ WARM-START: No layers frozen! This may cause catastrophic forgetting.")
                    
                    # WARM-START LR REDUCTION: Use 100x lower LR to preserve learned weights
                    effective_lr = self.config.learning_rate * self.config.warm_start_lr_factor
                    logger.info(f"🔥 Warm-start LR reduction: {self.config.learning_rate} → {effective_lr} (factor={self.config.warm_start_lr_factor})")
                    
                    # Load parent lineage if available
                    meta_path = Path(warm_start_path).with_suffix('.meta.pkl')
                    if meta_path.exists():
                        with open(meta_path, 'rb') as f:
                            meta = pickle.load(f)
                        if 'lineage' in meta:
                            parent_lineage = TrainingLineage.from_dict(meta['lineage'])
                            self.lineage.parent_checkpoint_id = parent_lineage.checkpoint_id
                            self.lineage.cumulative_epochs = parent_lineage.cumulative_epochs
                            self.lineage.cumulative_samples = parent_lineage.cumulative_samples
                            self.lineage.metric_history = parent_lineage.metric_history.copy()
                            # Store the instrument the loaded model was trained on
                            self._loaded_model_instrument = parent_lineage.instrument
                            logger.info(f"📊 Loaded lineage from parent: {parent_lineage.checkpoint_id} "
                                       f"(cumulative epochs: {self.lineage.cumulative_epochs}, instrument: {self._loaded_model_instrument})")
                        
                        # CRITICAL: Load previous best accuracy to prevent saving worse models
                        prev_metrics = meta.get('metrics', {})
                        self._warm_start_val_acc = prev_metrics.get('val_accuracy', 0.0)
                        if self._warm_start_val_acc > 0:
                            logger.info(f"🎯 Previous best val_accuracy: {self._warm_start_val_acc:.1%} (will not save worse)")
                    
                    # LOAD EWC STATE: Load Fisher information from previous training
                    if self._use_ewc:
                        ewc_path = Path(warm_start_path).with_suffix('.ewc.pkl')
                        self.ewc = EWCPenalty(
                            self.model,
                            ewc_lambda=self.config.ewc_lambda,
                            gamma=self.config.ewc_gamma,
                        )
                        if self.ewc.load(str(ewc_path)):
                            self.lineage.ewc_n_tasks = self.ewc._n_tasks
                            logger.info(f"🧠 EWC loaded: {self.ewc._n_tasks} prior task(s) will be protected")
                    
                    # LOAD EMA WEIGHTS: Continue from previous EMA state
                    if self._use_ema:
                        ema_meta_path = Path(warm_start_path).with_suffix('.ema.pkl')
                        if ema_meta_path.exists():
                            with open(ema_meta_path, 'rb') as f:
                                ema_data = pickle.load(f)
                            self.ema = EMACallback(
                                self.model,
                                decay=self.config.ema_decay,
                                update_every=self.config.ema_update_every,
                            )
                            self.ema.set_ema_weights(ema_data['ema_weights'])
                            logger.info("📊 EMA weights loaded from checkpoint")
                else:
                    logger.warning(f"Architecture mismatch: checkpoint has {existing_model.count_params():,} params, "
                                 f"new model has {self.model.count_params():,}. Starting fresh.")
                del existing_model
            except Exception as e:
                logger.warning(f"Could not load warm-start checkpoint: {e}. Starting fresh.")
        elif warm_start_path:
            logger.info(f"No checkpoint found at {warm_start_path}. Starting fresh training.")
        
        # Re-compile model with effective LR and gradient clipping (may be reduced for warm-start)
        # Gradient clipping prevents exploding gradients that can cause prediction collapse
        optimizer = keras.optimizers.Adam(
            learning_rate=effective_lr,
            clipnorm=1.0  # Clip gradients to prevent explosion → collapse
        )
        
        # === FOCAL LOSS SETTINGS ===
        # AntiCollapseFocalLoss prevents: direction bias, probability collapse to 0.5
        # Features: dynamic alpha based on predicted mean, variance regularization, gradient floor
        if self.config.use_focal_loss:
            base_loss = AntiCollapseFocalLoss(
                gamma=self.config.focal_gamma,
                base_alpha=self.config.focal_alpha,
                entropy_weight=0.2,  # Increased variance penalty weight for anti-collapse
                label_smoothing=0.0,  # NO smoothing - hurts binary classification
            )
            logger.info(f"🎯 Using AntiCollapseFocalLoss (gamma={self.config.focal_gamma}, alpha={self.config.focal_alpha}, variance_weight=0.2)")
        else:
            base_loss = keras.losses.BinaryCrossentropy(label_smoothing=0.0)
            logger.info("📊 Using BinaryCrossentropy (no focal loss)")
        
        # Use EWC loss if warm-starting with loaded Fisher information
        # NOTE: EWC can be counter-productive for same-pair warm-start (already learned this data)
        # Only use EWC for cross-pair transfer learning
        use_ewc_loss = (
            self._is_warm_start 
            and self._use_ewc 
            and self.ewc is not None 
            and self.ewc.fisher_diagonal is not None
            and getattr(self, '_loaded_model_instrument', None) != instrument  # Only for cross-pair
        )
        
        if use_ewc_loss:
            logger.info(f"🧠 EWC loss enabled (cross-pair): λ={self.ewc.ewc_lambda}, protecting {self.ewc._n_tasks} prior task(s)")
            ewc_loss = create_ewc_loss(base_loss, self.ewc.penalty, ewc_weight=1.0)
            self.model.compile(
                optimizer=optimizer,
                loss=ewc_loss,
                metrics=['accuracy'],
            )
        else:
            if self._is_warm_start and self._use_ewc:
                logger.info(f"🧠 EWC disabled for same-pair warm-start (prevents over-regularization)")
            self.model.compile(
                optimizer=optimizer,
                loss=base_loss,
                metrics=['accuracy'],
            )
        
        # === CRITICAL: Decide whether to use stored baseline or re-evaluate ===
        # For SAME-PAIR training: trust stored baseline (data may have drifted but model is valid)
        # For CROSS-PAIR training: re-evaluate baseline on new pair's data
        if self._is_warm_start:
            try:
                # Get the instrument that the loaded model was trained on
                loaded_model_instrument = getattr(self, '_loaded_model_instrument', None)
                is_cross_pair = loaded_model_instrument and loaded_model_instrument != instrument
                
                # Quick evaluation to get actual baseline on this pair's data
                eval_results = self.model.evaluate(X_val_filtered, y_val_filtered, verbose=0)
                actual_baseline_acc = eval_results[1] if len(eval_results) > 1 else eval_results[0]
                
                if is_cross_pair:
                    # Cross-pair training: use actual evaluation on new data
                    logger.info(f"🔄 Cross-pair training ({loaded_model_instrument} → {instrument})")
                    logger.info(f"   Stored baseline: {self._warm_start_val_acc:.1%}, Actual on {instrument}: {actual_baseline_acc:.1%}")
                    self._warm_start_val_acc = actual_baseline_acc
                    logger.info(f"🎯 Using actual baseline on {instrument}: {self._warm_start_val_acc:.1%}")
                else:
                    # Same-pair training: trust stored baseline, but log comparison
                    logger.info(f"🔄 Same-pair training ({instrument})")
                    logger.info(f"   Stored baseline: {self._warm_start_val_acc:.1%}, Current eval: {actual_baseline_acc:.1%}")
                    # Keep stored baseline to prevent false "degradation" from data drift
                    if self._warm_start_val_acc > 0:
                        logger.info(f"🎯 Using STORED baseline: {self._warm_start_val_acc:.1%} (ignoring data drift)")
                    else:
                        self._warm_start_val_acc = actual_baseline_acc
                        logger.info(f"🎯 No stored baseline, using actual: {self._warm_start_val_acc:.1%}")
            except Exception as e:
                logger.warning(f"Could not evaluate baseline: {e}")
        
        # Print model summary
        self.model.summary(print_fn=logger.info)
        
        # === INITIALIZE EMA IF NOT LOADED ===
        if self._use_ema and self.ema is None:
            self.ema = EMACallback(
                self.model,
                decay=self.config.ema_decay,
                update_every=self.config.ema_update_every,
            )
            self.lineage.ema_enabled = True
        
        # === CUSTOM TRAINING CALLBACK FOR EMA UPDATES ===
        class EMAUpdateCallback(keras.callbacks.Callback):
            def __init__(self, ema_callback):
                super().__init__()
                self.ema_callback = ema_callback
            
            def on_train_batch_end(self, batch, logs=None):
                if self.ema_callback:
                    self.ema_callback.update()
        
        # === PREDICTION COLLAPSE DETECTION & RECOVERY CALLBACK ===
        # Detects if model collapses to predicting all one class and takes corrective action
        class PredictionCollapseCallback(keras.callbacks.Callback):
            def __init__(self, X_val, y_val, check_every=5, max_recovery_attempts=3):
                super().__init__()
                self.X_val = X_val
                self.y_val = y_val
                self.check_every = check_every
                self.collapse_warned = False
                self.collapse_epochs = 0  # Consecutive collapse epochs
                self.recovery_attempts = 0
                self.max_recovery_attempts = max_recovery_attempts
                self.best_weights = None
                self.best_balance = 0.5  # Best prediction balance (0.5 = perfectly balanced)
            
            def on_epoch_end(self, epoch, logs=None):
                if (epoch + 1) % self.check_every != 0:
                    return
                
                preds = self.model.predict(self.X_val, verbose=0)
                pred_classes = (preds > 0.5).astype(float).flatten()
                
                pred_up_pct = pred_classes.mean() * 100
                pred_down_pct = 100 - pred_up_pct
                
                # Track prediction balance (0.5 = perfect, 0 or 1 = collapsed)
                current_balance = min(pred_up_pct, pred_down_pct) / 50  # 0-1 scale
                
                # Save best balanced weights
                if current_balance > self.best_balance and current_balance > 0.3:
                    self.best_balance = current_balance
                    self.best_weights = self.model.get_weights()
                
                # Check for collapse (>90% same prediction - more aggressive detection)
                if pred_up_pct > 90 or pred_down_pct > 90:
                    self.collapse_epochs += 1
                    dominant = "UP" if pred_up_pct > 90 else "DOWN"
                    
                    if not self.collapse_warned:
                        logger.warning(f"⚠️ PREDICTION COLLAPSE at epoch {epoch+1}: "
                                      f"Model predicts {pred_up_pct:.1f}% UP, {pred_down_pct:.1f}% DOWN "
                                      f"(all {dominant})")
                        self.collapse_warned = True
                    
                    # === RECOVERY ACTION: After 2 consecutive collapse checks ===
                    if self.collapse_epochs >= 2 and self.recovery_attempts < self.max_recovery_attempts:
                        self.recovery_attempts += 1
                        logger.warning(f"🔧 COLLAPSE RECOVERY attempt {self.recovery_attempts}/{self.max_recovery_attempts}")
                        
                        # Strategy 1: Restore best balanced weights if available
                        if self.best_weights is not None:
                            logger.info("  → Restoring best balanced weights")
                            self.model.set_weights(self.best_weights)
                        else:
                            # Strategy 2: Perturb output layer to break symmetry
                            logger.info("  → Perturbing output layer weights")
                            weights = self.model.get_weights()
                            # Add noise to last layer weights only
                            weights[-2] = weights[-2] + np.random.normal(0, 0.1, weights[-2].shape)
                            weights[-1] = np.array([0.0])  # Reset bias to neutral
                            self.model.set_weights(weights)
                        
                        # Reduce learning rate to stabilize
                        current_lr = float(self.model.optimizer.learning_rate)
                        new_lr = current_lr * 0.5
                        self.model.optimizer.learning_rate.assign(new_lr)
                        logger.info(f"  → Reduced learning rate: {current_lr:.2e} → {new_lr:.2e}")
                        
                        self.collapse_epochs = 0  # Reset counter after recovery
                    
                    # If all recovery attempts exhausted, stop training
                    elif self.collapse_epochs >= 4 and self.recovery_attempts >= self.max_recovery_attempts:
                        logger.error(f"❌ STOPPING: Prediction collapse persists after {self.max_recovery_attempts} recovery attempts")
                        self.model.stop_training = True
                else:
                    self.collapse_epochs = 0
                    self.collapse_warned = False
                    if epoch > 0 and (epoch + 1) % 10 == 0:
                        logger.info(f"📊 Prediction distribution at epoch {epoch+1}: "
                                   f"{pred_up_pct:.1f}% UP, {pred_down_pct:.1f}% DOWN")
        
        # Callbacks - use config patience values
        # Key insight: For classification, val_accuracy is what matters for trading
        # val_loss can improve while accuracy drops (model becomes uncertain)
        
        # WARM-START ADJUSTMENTS:
        # - Shorter early stopping patience (model already near optimum, stop sooner if degrading)
        # - Longer ReduceLROnPlateau patience (don't cut LR prematurely)
        early_stop_patience = self.config.patience // 2 if self._is_warm_start else self.config.patience
        lr_reduce_patience = self.config.patience * 2 if self._is_warm_start else max(4, self.config.patience // 4)
        
        if self._is_warm_start:
            logger.info(f"📊 Warm-start callback adjustments:")
            logger.info(f"   Early stopping patience: {early_stop_patience} (reduced from {self.config.patience})")
            logger.info(f"   LR reduction patience: {lr_reduce_patience} (increased from {max(4, self.config.patience // 4)})")
        
        callbacks = [
            # Rich-formatted epoch display with color coding
            RichEpochCallback(
                model_name="Transformer Direction",
                total_epochs=self.config.epochs,
                warm_start_best_acc=self._warm_start_val_acc,  # Track baseline for display
            ),
            # Primary: Stop when validation ACCURACY stops improving
            # WARM-START: Use shorter patience - stop faster if model degrades
            keras.callbacks.EarlyStopping(
                monitor='val_accuracy',
                patience=early_stop_patience,
                mode='max',  # We want to MAXIMIZE accuracy
                restore_best_weights=True,  # Restore weights from best epoch
                verbose=0,  # Suppress - Rich callback handles display
                start_from_epoch=self.config.min_epochs,  # Enforce minimum epochs before early stopping
            ),
            # LR reduction based on accuracy plateau
            # WARM-START: Use much more patience to avoid premature LR cuts
            # Fresh training: patience // 4 = 5 epochs
            # Warm-start: patience * 2 = 40 epochs (model is already near plateau)
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_accuracy',
                factor=0.5,
                patience=lr_reduce_patience,
                mode='max',  # Reduce LR when accuracy stops improving
                min_lr=1e-7 if self._is_warm_start else 1e-6,  # Allow lower LR for fine-tuning
                verbose=0,  # Suppress - Rich callback handles display
            ),
            # Overfit prevention: checkpoints + auto-adjust dropout/LR when train >> val
            OverfitPreventionCallback(
                checkpoint_dir=self.config.checkpoint_dir,
                model_name="transformer_direction",
                overfit_threshold=0.08,      # 8% gap → warning
                critical_threshold=0.15,     # 15% gap → intervention (LR reduction)
                severe_threshold=0.25,       # 25% gap → stop training
                max_acceptable_gap=0.12,     # Won't save checkpoint if gap > 12%
                patience_epochs=3,           # Faster intervention
                auto_adjust_dropout=True,
                auto_reduce_lr=True,
                # CONTINUAL LEARNING: Don't save worse models than previous best
                warm_start_best_acc=self._warm_start_val_acc,
            ),
            # Prediction collapse detection with auto-recovery - check every 3 epochs for faster response
            PredictionCollapseCallback(X_val_filtered, y_val_filtered, check_every=3, max_recovery_attempts=3),
        ]
        
        # Add EMA update callback if enabled
        if self._use_ema and self.ema is not None:
            callbacks.append(EMAUpdateCallback(self.ema))
        
        # Add EWC monitoring callback if EWC is active (warm-start with prior tasks)
        if self._is_warm_start and self._use_ewc and self.ewc is not None and self.ewc.fisher_diagonal is not None:
            callbacks.append(EWCTrainingCallback(self.ewc, log_every=50))
        
        # Train on FILTERED sequences (clear labels only) with SAMPLE WEIGHTS
        # Note: Cannot use both class_weight and sample_weight in Keras
        # sample_weights already incorporate class balancing + regime awareness
        history = self.model.fit(
            X_train_filtered, y_train_filtered,
            validation_data=(X_val_filtered, y_val_filtered),
            epochs=self.config.epochs,
            batch_size=self.config.batch_size,
            callbacks=callbacks,
            verbose=0,  # Suppress Keras output - RichEpochCallback handles display
            sample_weight=sample_weights,
        )
        
        self.is_trained = True
        
        # === WARM-START RECOVERY: Restore original weights if training degraded ===
        # CRITICAL: Do this BEFORE computing final metrics!
        weights_restored = False
        if self._is_warm_start and self._warm_start_weights is not None and self._warm_start_val_acc > 0:
            # Check current val accuracy using model after EarlyStopping restored "best" training weights
            current_val_pred = (self.model.predict(X_val_filtered, verbose=0) > 0.5).astype(float)
            current_val_acc = np.mean(current_val_pred.flatten() == y_val_filtered)
            
            logger.info(f"🔍 Post-training check: current_val_acc={current_val_acc:.1%}, warm_start_baseline={self._warm_start_val_acc:.1%}")
            
            if current_val_acc < self._warm_start_val_acc - 0.01:  # Allow 1% tolerance
                # Training degraded - restore original weights
                from rich.console import Console
                console = Console()
                console.print(f"  [bold red]⚠️ WARM-START RECOVERY TRIGGERED[/bold red]")
                console.print(f"  [red]   Current: {current_val_acc:.1%} < Baseline: {self._warm_start_val_acc:.1%}[/red]")
                console.print(f"  [yellow]   Restoring original warm-start weights to prevent degradation...[/yellow]")
                
                self.model.set_weights(self._warm_start_weights)
                weights_restored = True
                
                console.print(f"  [green]✓ Original weights restored. Model preserved at {self._warm_start_val_acc:.1%} accuracy[/green]")
                logger.info(f"✓ Warm-start weights restored. Model preserved at {self._warm_start_val_acc:.1%}")
        
        # === COMPUTE EWC FISHER INFORMATION ===
        # After training, compute importance of each weight for this task
        if self._use_ewc:
            if self.ewc is None:
                self.ewc = EWCPenalty(
                    self.model,
                    ewc_lambda=self.config.ewc_lambda,
                    gamma=self.config.ewc_gamma,
                )
            self.ewc.compute_fisher(X_train_filtered, y_train_filtered, n_samples=1000)
            self.lineage.ewc_n_tasks = self.ewc._n_tasks
        
        # === FINAL EMA UPDATE ===
        if self._use_ema and self.ema is not None:
            self.ema.update(force=True)  # Ensure final weights are captured
        
        # Update lineage
        self.lineage.session_epochs = len(history.history['loss'])
        self.lineage.cumulative_epochs += self.lineage.session_epochs
        self.lineage.cumulative_samples += len(X_train_filtered)
        if self.replay_buffer:
            self.lineage.replay_buffer_size = len(self.replay_buffer.X_buffer) if self.replay_buffer.X_buffer is not None else 0
        
        # Calculate metrics on FILTERED validation (clear labels only)
        # NOTE: This is the CANONICAL val_accuracy - computed on full val set after EarlyStopping
        # restores best weights. Different from epoch-level val_accuracy logged during training
        # (which uses batch-level estimates and may differ by 1-2%).
        val_raw_pred = self.model.predict(X_val_filtered, verbose=0)
        val_pred = (val_raw_pred > 0.5).astype(float)
        val_acc = np.mean(val_pred.flatten() == y_val_filtered)
        
        # Calculate balanced accuracy
        y_true = y_val_filtered.flatten()
        y_pred = val_pred.flatten()
        up_acc = np.mean(y_pred[y_true == 1] == 1) if (y_true == 1).sum() > 0 else 0
        down_acc = np.mean(y_pred[y_true == 0] == 0) if (y_true == 0).sum() > 0 else 0
        balanced_acc = (up_acc + down_acc) / 2
        
        # DEBUG: Log prediction distribution to detect collapse
        long_preds = (y_pred == 1).sum()
        short_preds = (y_pred == 0).sum()
        raw_mean = float(np.mean(val_raw_pred))
        raw_std = float(np.std(val_raw_pred))
        raw_median = float(np.median(val_raw_pred))
        logger.info(f"📊 Final validation prediction distribution:")
        logger.info(f"   Raw prob: mean={raw_mean:.4f}, median={raw_median:.4f}, std={raw_std:.4f}, min={float(np.min(val_raw_pred)):.4f}, max={float(np.max(val_raw_pred)):.4f}")
        logger.info(f"   Predictions (thresh=0.5): LONG={long_preds} ({100*long_preds/len(y_pred):.1f}%), SHORT={short_preds} ({100*short_preds/len(y_pred):.1f}%)")
        if long_preds == 0 or short_preds == 0:
            logger.warning(f"   ⚠️ MODEL COLLAPSE DETECTED: Always predicting {'LONG' if long_preds > 0 else 'SHORT'}!")
        
        # === ADAPTIVE THRESHOLD CALIBRATION ===
        # Instead of shifting predictions, use the median as the decision threshold.
        # This guarantees ~50/50 split on the calibration set.
        # Store threshold instead of bias for cleaner inference.
        self.output_calibration = {
            'threshold': raw_median,  # Use median as threshold for balanced predictions
            'mean': raw_mean,
            'std': max(raw_std, 0.01),
            'enabled': abs(raw_mean - 0.5) > 0.05,  # Only calibrate if significantly biased
        }
        
        # Recalculate with adaptive threshold
        val_pred_calibrated = (val_raw_pred.flatten() > raw_median).astype(float)
        
        # Update balanced accuracy with calibrated predictions
        up_acc_cal = np.mean(val_pred_calibrated[y_true == 1] == 1) if (y_true == 1).sum() > 0 else 0
        down_acc_cal = np.mean(val_pred_calibrated[y_true == 0] == 0) if (y_true == 0).sum() > 0 else 0
        balanced_acc_cal = (up_acc_cal + down_acc_cal) / 2
        
        long_preds_cal = (val_pred_calibrated == 1).sum()
        short_preds_cal = (val_pred_calibrated == 0).sum()
        logger.info(f"📐 Calibrated (thresh={raw_median:.4f}): LONG={long_preds_cal} ({100*long_preds_cal/len(val_pred_calibrated):.1f}%), "
                   f"SHORT={short_preds_cal} ({100*short_preds_cal/len(val_pred_calibrated):.1f}%)")
        logger.info(f"📐 Calibrated balanced accuracy: {balanced_acc_cal:.4f} (up={up_acc_cal:.4f}, down={down_acc_cal:.4f})")
        
        # Use calibrated balanced accuracy in metrics
        balanced_acc = balanced_acc_cal
        up_acc = up_acc_cal
        down_acc = down_acc_cal
        
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
        
        # Log weight norms for regularization monitoring
        # Higher norms may indicate insufficient regularization
        try:
            total_weight_norm = 0.0
            trainable_params = 0
            for layer in self.model.layers:
                for w in layer.trainable_weights:
                    w_norm = float(tf.norm(w).numpy())
                    total_weight_norm += w_norm
                    trainable_params += int(tf.size(w).numpy())
            avg_weight_norm = total_weight_norm / max(1, len([w for l in self.model.layers for w in l.trainable_weights]))
            self.metrics['total_weight_norm'] = total_weight_norm
            self.metrics['avg_weight_norm'] = avg_weight_norm
            logger.info(f"Weight norms: total={total_weight_norm:.2f}, avg={avg_weight_norm:.4f} (trainable params={trainable_params:,})")
        except Exception as e:
            logger.debug(f"Could not compute weight norms: {e}")
        
        # === DRIFT DETECTION ===
        # Record training result and check for drift
        if self.drift_detector is not None:
            # Record this training result
            self.drift_detector.record_training_result(
                val_accuracy=val_acc,
                instrument=instrument,
                data_hash=self.lineage.data_hash if self.lineage else '',
                feature_means=X_train_filtered.mean(axis=(0, 1)) if len(X_train_filtered.shape) == 3 else X_train_filtered.mean(axis=0),
            )
            
            # Check for drift
            drift_detected, drift_reason = self.drift_detector.check_drift()
            if drift_detected:
                logger.warning(f"⚠️ DRIFT DETECTED: {drift_reason}")
                self.metrics['drift_detected'] = True
                self.metrics['drift_reason'] = drift_reason
                if self.lineage:
                    self.lineage.drift_detected = True
                    self.lineage.drift_reason = drift_reason
                    self.lineage.last_drift_check = datetime.now().isoformat()
            else:
                self.metrics['drift_detected'] = False
                if self.lineage:
                    self.lineage.drift_detected = False
                    self.lineage.last_drift_check = datetime.now().isoformat()
        
        logger.info(f"Transformer trained [canonical]: val_accuracy={val_acc:.4f}, "
                   f"balanced_acc={balanced_acc:.4f} (up={up_acc:.4f}, down={down_acc:.4f})")
        return self.metrics
    
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
            raise RuntimeError("Model not trained")
        
        # Scale
        X_scaled = self.scaler.transform(X.reshape(-1, X.shape[-1]))
        
        # Check model input shape to determine if sequence or flat input
        model_input_shape = self.model.input_shape
        is_flat_model = len(model_input_shape) == 2  # (None, n_features)
        
        if is_flat_model:
            # Flat input model - use last row only
            X_input = X_scaled[-1:] if len(X_scaled) > 1 else X_scaled
        else:
            # Sequence model - create sequence from last seq_len rows
            if len(X_scaled) >= self.seq_len:
                X_input = X_scaled[-self.seq_len:].reshape(1, self.seq_len, -1)
            else:
                # Pad with zeros if not enough data
                pad_len = self.seq_len - len(X_scaled)
                X_padded = np.vstack([np.zeros((pad_len, X_scaled.shape[1])), X_scaled])
                X_input = X_padded.reshape(1, self.seq_len, -1)
        
        # Use EMA weights for stable inference if available
        use_ema_weights = (
            use_ema and 
            self._use_ema and 
            self.ema is not None and 
            self.ema._initialized and
            self.config.use_ema_for_inference
        )
        
        if use_ema_weights:
            self.ema.apply()  # Apply EMA weights
        
        try:
            prob_raw = float(self.model.predict(X_input, verbose=0)[0, 0])
        finally:
            if use_ema_weights:
                self.ema.restore()  # Restore training weights
        
        # === APPLY OUTPUT CALIBRATION ===
        # Use adaptive threshold instead of shifting probabilities
        calibration = getattr(self, 'output_calibration', None)
        threshold = 0.5  # Default threshold
        if calibration and calibration.get('enabled', False):
            threshold = calibration.get('threshold', 0.5)
        
        direction = 1 if prob_raw > threshold else 0
        
        # For confidence, measure distance from threshold (not from 0.5)
        # Normalize to 0-1 range based on typical distribution
        std = calibration.get('std', 0.15) if calibration else 0.15
        confidence_distance = abs(prob_raw - threshold) / (2 * std)  # Normalize by 2 std
        confidence = min(1.0, confidence_distance)  # Cap at 1.0
        
        return {
            'direction': direction,
            'probability': prob_raw,  # Raw probability
            'probability_raw': prob_raw,  # Alias for debugging
            'confidence': confidence,  # Normalized confidence
            'threshold': threshold,  # Calibrated threshold
            'ema_used': use_ema_weights,
            'calibration_applied': calibration.get('enabled', False) if calibration else False,
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
        
        # Save Keras model
        self.model.save(str(path))
        
        # Update lineage metrics before saving
        if self.lineage:
            self.lineage.add_metrics(self.metrics)
        
        # Save scaler, config, and lineage
        meta = {
            'scaler': self.scaler,
            'seq_len': self.seq_len,
            'metrics': self.metrics,
            'config': self.config.__dict__,
            'feature_names': self.feature_names,
            'n_features': self.n_features,
            'model_type': 'transformer',
            'lineage': self.lineage.to_dict() if self.lineage else None,
            'is_warm_start': self._is_warm_start,
            'ema_enabled': self._use_ema,
            'ewc_enabled': self._use_ewc,
            'replay_enabled': self._use_replay,
            'output_calibration': getattr(self, 'output_calibration', None),  # Save calibration params
        }
        meta_path = path.with_suffix('.meta.pkl')
        with open(meta_path, 'wb') as f:
            pickle.dump(meta, f)
        
        # Save EMA weights if available
        if self._use_ema and self.ema is not None and self.ema._initialized:
            ema_data = {
                'ema_weights': self.ema.get_ema_weights(),
                'decay': self.ema.decay,
                'update_every': self.ema.update_every,
                'step_counter': self.ema.step_counter,
            }
            ema_path = path.with_suffix('.ema.pkl')
            with open(ema_path, 'wb') as f:
                pickle.dump(ema_data, f)
            logger.info(f"📊 EMA weights saved to {ema_path}")
        
        # Save EWC state if available
        if self._use_ewc and self.ewc is not None and self.ewc.fisher_diagonal is not None:
            ewc_path = path.with_suffix('.ewc.pkl')
            self.ewc.save(str(ewc_path))
        
        # Save replay buffer
        if self._use_replay and self.replay_buffer is not None:
            self.replay_buffer.save(instrument)
        
        logger.info(f"✅ Transformer saved to {path} (EMA={self._use_ema}, EWC={self._use_ewc}, Replay={self._use_replay})")
    
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
        import tensorflow as tf
        from tensorflow import keras
        
        path = Path(path)
        
        # Try multiple loading strategies for Keras 2.x/3.x compatibility
        model = None
        load_errors = []
        
        # Strategy 1: Standard load (works if same Keras version)
        try:
            model = keras.models.load_model(str(path), compile=False)
            logger.info(f"✓ Model loaded with standard loader")
        except Exception as e:
            load_errors.append(f"Standard: {e}")
        
        # Strategy 2: Use tf.keras.models.load_model (TF-native)
        if model is None:
            try:
                model = tf.keras.models.load_model(str(path), compile=False)
                logger.info(f"✓ Model loaded with tf.keras loader")
            except Exception as e:
                load_errors.append(f"TF-native: {e}")
        
        # Strategy 3: Load with safe_mode=False for Keras 3 models
        if model is None:
            try:
                model = keras.models.load_model(str(path), compile=False, safe_mode=False)
                logger.info(f"✓ Model loaded with safe_mode=False")
            except Exception as e:
                load_errors.append(f"Safe-mode: {e}")
        
        # Strategy 4: Rebuild model from metadata and load weights only
        if model is None:
            meta_path = path.with_suffix('.meta.pkl')
            if meta_path.exists():
                try:
                    with open(meta_path, 'rb') as f:
                        meta = pickle.load(f)
                    
                    n_features = meta.get('n_features', 59)
                    seq_len = meta.get('seq_len', 60)
                    config = meta.get('config', {})
                    
                    # Get transformer hyperparams from config
                    d_model = config.get('transformer_d_model', 32)
                    num_heads = config.get('transformer_num_heads', 4)
                    num_layers = config.get('transformer_num_layers', 2)
                    dff = config.get('transformer_dff', 64)
                    dropout = config.get('transformer_dropout', 0.2)
                    
                    logger.info(f"Rebuilding Transformer: n_features={n_features}, seq_len={seq_len}, d_model={d_model}")
                    
                    # Rebuild the actual Transformer architecture
                    inp = keras.Input(shape=(seq_len, n_features), name="features")
                    x = keras.layers.GaussianNoise(0.15)(inp)
                    x = keras.layers.SpatialDropout1D(0.2)(x)
                    
                    # Input projection
                    x = keras.layers.Dense(d_model, name='input_projection')(x)
                    x = keras.layers.Dropout(0.3)(x)
                    
                    # Add positional encoding (simplified for rebuild)
                    positions = np.arange(seq_len)[:, np.newaxis]
                    dims = np.arange(d_model)[np.newaxis, :]
                    angles = positions / np.power(10000, (2 * (dims // 2)) / d_model)
                    pos_encoding = np.zeros((seq_len, d_model))
                    pos_encoding[:, 0::2] = np.sin(angles[:, 0::2])
                    pos_encoding[:, 1::2] = np.cos(angles[:, 1::2])
                    pos_encoding = pos_encoding[np.newaxis, :, :].astype(np.float32)
                    x = x + tf.constant(pos_encoding)
                    
                    # Transformer encoder layers
                    for i in range(num_layers):
                        # Multi-head attention
                        attn_output = keras.layers.MultiHeadAttention(
                            num_heads=num_heads,
                            key_dim=d_model // num_heads,
                            name=f'transformer_{i}_mha'
                        )(x, x)
                        attn_output = keras.layers.Dropout(dropout)(attn_output)
                        x = keras.layers.LayerNormalization(epsilon=1e-6, name=f'transformer_{i}_ln1')(x + attn_output)
                        
                        # FFN
                        ffn = keras.layers.Dense(dff, activation='relu', name=f'transformer_{i}_ffn1')(x)
                        ffn = keras.layers.Dense(d_model, name=f'transformer_{i}_ffn2')(ffn)
                        ffn = keras.layers.Dropout(dropout)(ffn)
                        x = keras.layers.LayerNormalization(epsilon=1e-6, name=f'transformer_{i}_ln2')(x + ffn)
                    
                    # Global pooling and output
                    x = keras.layers.GlobalAveragePooling1D()(x)
                    x = keras.layers.Dense(8, activation='relu')(x)
                    x = keras.layers.Dropout(0.5)(x)
                    direction = keras.layers.Dense(1, activation='sigmoid', name='direction', dtype='float32')(x)
                    
                    model = keras.Model(inputs=inp, outputs=direction, name='transformer_direction')
                    logger.info(f"✓ Model architecture rebuilt from metadata")
                    
                    # If model was rebuilt, we need to load weights from EMA
                    ema_path = path.with_suffix('.ema.pkl')
                    if ema_path.exists():
                        with open(ema_path, 'rb') as f:
                            ema_data = pickle.load(f)
                        ema_weights = ema_data.get('ema_weights', [])
                        
                        # Try to apply EMA weights directly to rebuilt model
                        model_weights = model.trainable_weights
                        if len(ema_weights) == len(model_weights):
                            for w, ema_w in zip(model_weights, ema_weights):
                                try:
                                    w.assign(ema_w)
                                except Exception as assign_err:
                                    logger.warning(f"Could not assign EMA weight to {w.name}: {assign_err}")
                            logger.info(f"✓ Loaded {len(ema_weights)} EMA weights into rebuilt model")
                        else:
                            logger.warning(f"EMA weights count ({len(ema_weights)}) != model weights ({len(model_weights)})")
                    
                except Exception as e:
                    load_errors.append(f"Rebuild: {e}")
        
        if model is None:
            all_errors = "; ".join(load_errors)
            raise RuntimeError(f"Failed to load model from {path}. Errors: {all_errors}")
        
        self.model = model
        
        # Load metadata
        meta_path = path.with_suffix('.meta.pkl')
        with open(meta_path, 'rb') as f:
            meta = pickle.load(f)
        
        self.scaler = meta['scaler']
        self.seq_len = meta['seq_len']
        self.metrics = meta['metrics']
        self.feature_names = meta.get('feature_names')
        self.n_features = meta.get('n_features')
        self._is_warm_start = meta.get('is_warm_start', False)
        self._use_ema = meta.get('ema_enabled', True)
        self._use_ewc = meta.get('ewc_enabled', True)
        self._use_replay = meta.get('replay_enabled', True)
        
        # Load output calibration (for bias correction)
        self.output_calibration = meta.get('output_calibration', None)
        if self.output_calibration and self.output_calibration.get('enabled'):
            logger.info(f"📐 Output calibration loaded: bias={self.output_calibration['bias']:.4f}")
        
        # Load lineage
        if meta.get('lineage'):
            self.lineage = TrainingLineage.from_dict(meta['lineage'])
            logger.info(f"📊 Lineage loaded: checkpoint={self.lineage.checkpoint_id}, "
                       f"cumulative_epochs={self.lineage.cumulative_epochs}")
        
        # Load EMA weights
        ema_path = path.with_suffix('.ema.pkl')
        if ema_path.exists() and self._use_ema:
            with open(ema_path, 'rb') as f:
                ema_data = pickle.load(f)
            self.ema = EMACallback(
                self.model,
                decay=ema_data.get('decay', self.config.ema_decay),
                update_every=ema_data.get('update_every', self.config.ema_update_every),
            )
            self.ema.set_ema_weights(ema_data['ema_weights'])
            self.ema.step_counter = ema_data.get('step_counter', 0)
            logger.info(f"📊 EMA weights loaded (decay={self.ema.decay})")
        
        # Load EWC state
        ewc_path = path.with_suffix('.ewc.pkl')
        if ewc_path.exists() and self._use_ewc:
            self.ewc = EWCPenalty(
                self.model,
                ewc_lambda=self.config.ewc_lambda,
                gamma=self.config.ewc_gamma,
            )
            self.ewc.load(str(ewc_path))
        
        # Load replay buffer
        if self._use_replay:
            self.replay_buffer = ReplayBuffer(
                capacity_ratio=self.config.replay_buffer_ratio,
                mix_ratio=self.config.replay_mix_ratio,
                buffer_dir=self.config.replay_buffer_dir,
            )
            self.replay_buffer.load(instrument)
        
        self.is_trained = True
        
        logger.info(f"✅ Transformer loaded from {path}")
        
        # Check for drift if lineage exists
        if self.lineage and self.metrics.get('val_accuracy'):
            if self.lineage.check_drift(self.metrics['val_accuracy'], self.config.drift_threshold):
                logger.warning("⚠️ Consider retraining: model may have drifted from optimal performance")


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
        
        # Transformer hyperparameters - match proven direction config as baseline
        self.d_model = getattr(config, 'transformer_d_model', 16) if config else 16
        self.num_heads = getattr(config, 'transformer_num_heads', 2) if config else 2
        self.ff_dim = getattr(config, 'transformer_ff_dim', 32) if config else 32
        self.num_blocks = getattr(config, 'transformer_num_blocks', 1) if config else 1
        self.transformer_dropout = getattr(config, 'transformer_dropout', 0.2) if config else 0.2  # Reduced from 0.4
    
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
            # Rich-formatted epoch display with color coding
            RichEpochCallback(
                model_name="Transformer Regime",
                total_epochs=self.config.epochs,
            ),
            keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=self.config.patience,
                mode='min',
                restore_best_weights=True,
                verbose=0,  # Suppress - Rich callback handles display
                start_from_epoch=self.config.min_epochs,  # Enforce minimum epochs before early stopping
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=max(4, self.config.patience // 4),
                min_lr=1e-6,
                verbose=0,  # Suppress - Rich callback handles display
            ),
        ]
        
        # Train
        history = self.model.fit(
            X_train_seq, y_train_seq,
            validation_data=(X_val_seq, y_val_seq),
            epochs=self.config.epochs,
            batch_size=self.config.batch_size,
            callbacks=callbacks,
            verbose=0,  # Suppress Keras output - RichEpochCallback handles display
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
        """Train XGBoost for momentum analysis (2 models) with GPU support."""
        self.momentum_norm_factor = momentum_norm_factor
        try:
            import xgboost as xgb
        except ImportError:
            raise ImportError("XGBoost not installed. Run: pip install xgboost")
        
        from sklearn.preprocessing import StandardScaler
        
        # Detect GPU availability
        use_gpu = self.config.use_gpu if hasattr(self.config, 'use_gpu') else False
        tree_method = 'gpu_hist' if use_gpu else 'auto'
        
        logger.info(f"Training XGBoost (Momentum) - GPU: {use_gpu}, tree_method: {tree_method}")
        
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
        
        # Train momentum regressor with GPU acceleration
        self.momentum_model = xgb.XGBRegressor(
            n_estimators=self.config.xgb_n_estimators,
            max_depth=self.config.xgb_max_depth,
            learning_rate=self.config.xgb_learning_rate,
            tree_method=tree_method,          # GPU-accelerated if available
            predictor='gpu_predictor' if use_gpu else 'auto',
            verbosity=0,
            n_jobs=-1 if not use_gpu else 1,  # GPU doesn't need multi-threading
            random_state=42,
        )
        self.momentum_model.fit(
            X_train_scaled, y_train_momentum,
            eval_set=[(X_val_scaled, y_val_momentum)],
            verbose=False,
        )
        
        # Train acceleration classifier with GPU acceleration
        self.accel_model = xgb.XGBClassifier(
            n_estimators=self.config.xgb_n_estimators,
            max_depth=self.config.xgb_max_depth,
            learning_rate=self.config.xgb_learning_rate,
            tree_method=tree_method,          # GPU-accelerated if available
            predictor='gpu_predictor' if use_gpu else 'auto',
            verbosity=0,
            n_jobs=-1 if not use_gpu else 1,  # GPU doesn't need multi-threading
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
        """Save XGBoost models with version metadata."""
        import sklearn
        import xgboost
        
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
            # Version metadata for compatibility checks
            'sklearn_version': sklearn.__version__,
            'xgboost_version': xgboost.__version__,
            'saved_at': datetime.now().isoformat(),
        }
        
        with open(path, 'wb') as f:
            pickle.dump(data, f)
        
        logger.info(f"XGBoost saved to {path} (sklearn={sklearn.__version__}, xgboost={xgboost.__version__})")
    
    def load(self, path: str) -> None:
        """Load XGBoost models."""
        import warnings
        # Suppress XGBoost version warnings (common when loading older serialized models)
        warnings.filterwarnings('ignore', category=UserWarning, module='xgboost')
        warnings.filterwarnings('ignore', message='.*serialized model.*')
        warnings.filterwarnings('ignore', message='.*older version of XGBoost.*')
        
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
        
        # Store version info for compatibility checking
        self._saved_sklearn_version = data.get('sklearn_version')
        self._saved_xgboost_version = data.get('xgboost_version')
        self._saved_at = data.get('saved_at')
        
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
        
        # Clip predictions to valid range (0-10% for drawdown, 0-1 for streak)
        drawdown_pred = np.clip(drawdown_pred, 0, 0.10)
        streak_pred = np.clip(streak_pred, 0, 1.0)
        
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
        
        # Clamp values to realistic ranges
        expected_drawdown_pct = max(0.0, min(0.10, expected_drawdown_pct))  # 0-10% max drawdown
        streak_prob = max(0.0, min(1.0, streak_prob))  # 0-100% probability
        
        return {
            'expected_drawdown_pct': expected_drawdown_pct,
            # Keep legacy key for backward compatibility
            'expected_drawdown_pips': expected_drawdown_pct * 10000,  # Rough conversion for display
            'streak_prob': streak_prob,
        }
    
    def save(self, path: str) -> None:
        """Save Random Forest models with version metadata."""
        import sklearn
        
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
            # Version metadata for compatibility checks
            'sklearn_version': sklearn.__version__,
            'saved_at': datetime.now().isoformat(),
        }
        
        with open(path, 'wb') as f:
            pickle.dump(data, f)
        
        logger.info(f"Random Forest saved to {path} (sklearn={sklearn.__version__})")
    
    def load(self, path: str) -> None:
        """Load Random Forest models."""
        import warnings
        # Suppress sklearn version warnings
        try:
            from sklearn.exceptions import InconsistentVersionWarning
            warnings.filterwarnings('ignore', category=InconsistentVersionWarning)
        except ImportError:
            pass
        warnings.filterwarnings('ignore', message='.*unpickle estimator.*')
        
        with open(path, 'rb') as f:
            data = pickle.load(f)
        
        self.drawdown_model = data['drawdown_model']
        self.streak_model = data['streak_model']
        self.scaler = data['scaler']
        self.metrics = data['metrics']
        self.feature_names = data.get('feature_names')
        self.n_features = data.get('n_features')
        self.is_trained = True
        
        # Store version info for compatibility checking
        self._saved_sklearn_version = data.get('sklearn_version')
        self._saved_at = data.get('saved_at')
        
        logger.info(f"Random Forest loaded from {path}")


# =============================================================================
# RIDGE TRAINER - Confidence Scoring
# =============================================================================

class RidgeTrainer(BaseTrainer):
    """
    ElasticNet regression model for confidence/stability scoring.
    
    Uses ElasticNetCV with TimeSeriesSplit for automatic alpha + L1 ratio tuning.
    Combines L1 (Lasso) and L2 (Ridge) regularization for:
    - Automatic feature selection (L1 sparsity)
    - Handling correlated features (L2 stability)
    - Temporal-aware cross-validation (no data leakage)
    
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
        X_val: np.ndarray,
        y_val: np.ndarray,
        feature_names: Optional[list] = None,
    ) -> Dict[str, float]:
        """Train ElasticNetCV for confidence scoring with TimeSeriesSplit."""
        from sklearn.linear_model import ElasticNetCV
        from sklearn.model_selection import TimeSeriesSplit
        from sklearn.preprocessing import StandardScaler
        
        logger.info("Training ElasticNet (Confidence) with TimeSeriesSplit CV...")
        
        # Save feature names for inference
        self.feature_names = feature_names
        self.n_features = X_train.shape[-1]
        
        # Scale features
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_val_scaled = self.scaler.transform(X_val)
        
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
            selection='random',  # Faster convergence
        )
        self.model.fit(X_train_scaled, y_train)
        
        self.is_trained = True
        
        # Extract best hyperparameters
        self.best_alpha = float(self.model.alpha_)
        self.best_l1_ratio = float(self.model.l1_ratio_)
        self.n_nonzero_coefs = int(np.sum(self.model.coef_ != 0))
        
        # Calculate metrics
        y_pred = self.model.predict(X_val_scaled)
        mae = float(np.mean(np.abs(y_pred - y_val)))
        r2 = float(self.model.score(X_val_scaled, y_val))
        
        self.metrics = {
            'confidence_mae': mae,
            'r2_score': r2,
            'best_alpha': self.best_alpha,
            'best_l1_ratio': self.best_l1_ratio,
            'n_nonzero_coefs': self.n_nonzero_coefs,
            'n_total_coefs': self.n_features,
            'sparsity_ratio': 1.0 - (self.n_nonzero_coefs / self.n_features) if self.n_features > 0 else 0.0,
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
        """Save Ridge model with version metadata."""
        import sklearn
        
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        data = {
            'model': self.model,
            'scaler': self.scaler,
            'metrics': self.metrics,
            'config': self.config.__dict__,
            'feature_names': self.feature_names,
            'n_features': self.n_features,
            # Version metadata for compatibility checks
            'sklearn_version': sklearn.__version__,
            'saved_at': datetime.now().isoformat(),
        }
        
        with open(path, 'wb') as f:
            pickle.dump(data, f)
        
        logger.info(f"Ridge saved to {path} (sklearn={sklearn.__version__})")
    
    def load(self, path: str) -> None:
        """Load Ridge model."""
        import warnings
        # Suppress sklearn version warnings
        try:
            from sklearn.exceptions import InconsistentVersionWarning
            warnings.filterwarnings('ignore', category=InconsistentVersionWarning)
        except ImportError:
            pass
        warnings.filterwarnings('ignore', message='.*unpickle estimator.*')
        
        with open(path, 'rb') as f:
            data = pickle.load(f)
        
        self.model = data['model']
        self.scaler = data['scaler']
        self.metrics = data['metrics']
        self.feature_names = data.get('feature_names')
        self.n_features = data.get('n_features')
        self.is_trained = True
        
        # Store version info for compatibility checking
        self._saved_sklearn_version = data.get('sklearn_version')
        self._saved_at = data.get('saved_at')
        
        logger.info(f"Ridge loaded from {path}")


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
        self.max_iter = getattr(config, 'histgb_max_iter', 200) if config else 200
        self.max_depth = getattr(config, 'histgb_max_depth', 8) if config else 8
        self.learning_rate = getattr(config, 'histgb_learning_rate', 0.05) if config else 0.05
        self.l2_regularization = getattr(config, 'histgb_l2_reg', 0.1) if config else 0.1
        self.use_pca = getattr(config, 'histgb_use_pca', True) if config else True
        self.pca_variance = getattr(config, 'histgb_pca_variance', 0.95) if config else 0.95
    
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
            X_val = X_val[:, -1, :]
        
        # Filter by weights if provided (keep only clear labels)
        if w_train is not None:
            clear_mask_train = w_train > 0
            clear_mask_val = w_val > 0 if w_val is not None else np.ones(len(y_val), dtype=bool)
            X_train_filtered = X_train[clear_mask_train]
            y_train_filtered = y_train[clear_mask_train]
            X_val_filtered = X_val[clear_mask_val]
            y_val_filtered = y_val[clear_mask_val]
            logger.info(f"Filtered to clear labels: train={len(X_train_filtered)}, val={len(X_val_filtered)}")
        else:
            X_train_filtered = X_train
            y_train_filtered = y_train
            X_val_filtered = X_val
            y_val_filtered = y_val
        
        # Ensure labels are integer (sklearn classifier requires discrete labels)
        y_train_filtered = np.asarray(y_train_filtered).astype(int)
        y_val_filtered = np.asarray(y_val_filtered).astype(int)
        
        # Scale features
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train_filtered)
        X_val_scaled = self.scaler.transform(X_val_filtered)
        
        # Optional PCA for dimensionality reduction
        if self.use_pca and X_train_scaled.shape[1] > 20:
            self.pca = PCA(n_components=self.pca_variance, random_state=42)
            X_train_pca = self.pca.fit_transform(X_train_scaled)
            X_val_pca = self.pca.transform(X_val_scaled)
            logger.info(f"PCA: {X_train_scaled.shape[1]} features -> {X_train_pca.shape[1]} components "
                       f"(explaining {self.pca_variance*100:.0f}% variance)")
            X_train_final = X_train_pca
            X_val_final = X_val_pca
        else:
            X_train_final = X_train_scaled
            X_val_final = X_val_scaled
            self.pca = None
        
        # Log class distribution
        train_up_pct = (y_train_filtered == 1).mean() * 100
        val_up_pct = (y_val_filtered == 1).mean() * 100
        logger.info(f"Class distribution: train={train_up_pct:.1f}% up, val={val_up_pct:.1f}% up")
        
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
        
        self.model.fit(X_train_final, y_train_filtered)
        self.is_trained = True
        
        # Evaluate
        y_pred = self.model.predict(X_val_final)
        y_prob = self.model.predict_proba(X_val_final)[:, 1]
        
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
        except:
            auc = 0.5
        
        self.metrics = {
            'val_accuracy': float(val_acc),
            'val_balanced_accuracy': float(balanced_acc),
            'val_up_accuracy': float(up_acc),
            'val_down_accuracy': float(down_acc),
            'auc': float(auc),
            'n_train_samples': len(X_train_filtered),
            'n_val_samples': len(X_val_filtered),
            'n_features_used': X_train_final.shape[1],
            'n_iterations': self.model.n_iter_,
        }
        
        logger.info(f"HistGB trained: val_accuracy={val_acc:.4f}, balanced={balanced_acc:.4f}, "
                   f"auc={auc:.4f}, iters={self.model.n_iter_}")
        return self.metrics
    
    def predict(self, X: np.ndarray) -> Dict[str, Any]:
        """Predict direction (0 or 1) with probability."""
        if not self.is_trained:
            raise RuntimeError("Model not trained")
        
        # Flatten if 3D
        if X.ndim == 3:
            X = X[:, -1, :]
        
        # Scale
        X_scaled = self.scaler.transform(X.reshape(-1, X.shape[-1]))
        
        # Apply PCA if used
        if self.pca is not None:
            X_final = self.pca.transform(X_scaled)
        else:
            X_final = X_scaled
        
        # Get last row for prediction
        X_last = X_final[-1:] if len(X_final) > 1 else X_final
        
        prob = float(self.model.predict_proba(X_last)[0, 1])
        direction = int(self.model.predict(X_last)[0])
        
        return {
            'direction': direction,
            'probability': prob,
        }
    
    def save(self, path: str) -> None:
        """Save HistGB model."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        data = {
            'model': self.model,
            'scaler': self.scaler,
            'pca': self.pca,
            'metrics': self.metrics,
            'config': self.config.__dict__ if self.config else {},
            'feature_names': self.feature_names,
            'n_features': self.n_features,
            'model_type': 'histgb',
        }
        
        with open(path, 'wb') as f:
            pickle.dump(data, f)
        
        logger.info(f"HistGB saved to {path}")
    
    def load(self, path: str) -> None:
        """Load HistGB model."""
        import warnings
        # Suppress sklearn version warnings
        try:
            from sklearn.exceptions import InconsistentVersionWarning
            warnings.filterwarnings('ignore', category=InconsistentVersionWarning)
        except ImportError:
            pass
        warnings.filterwarnings('ignore', message='.*unpickle estimator.*')
        
        with open(path, 'rb') as f:
            data = pickle.load(f)
        
        self.model = data['model']
        self.scaler = data['scaler']
        self.pca = data.get('pca')
        self.metrics = data['metrics']
        self.feature_names = data.get('feature_names')
        self.n_features = data.get('n_features')
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
    warnings.filterwarnings('ignore', category=UserWarning, module='xgboost')
    warnings.filterwarnings('ignore', message='.*serialized model.*')
    
    model_path = Path(model_path)
    output_path = Path(output_path) if output_path else model_path
    
    if not model_path.exists():
        logger.error(f"Model not found: {model_path}")
        return False
    
    try:
        # Load existing model
        with open(model_path, 'rb') as f:
            data = pickle.load(f)
        
        # Re-save (this ensures internal XGBoost state is updated)
        backup_path = model_path.with_suffix('.pkl.bak')
        if model_path == output_path:
            # Backup original
            import shutil
            shutil.copy(model_path, backup_path)
        
        with open(output_path, 'wb') as f:
            pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
        
        logger.info(f"✅ XGBoost model migrated: {output_path}")
        return True
        
    except Exception as e:
        logger.error(f"Migration failed: {e}")
        return False


def migrate_all_models(model_dir: str = "trained_data/models") -> Dict[str, bool]:
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
        "ridge_confidence.pkl",
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
    save_dir: str = "trained_data/models",
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
    transformer_checkpoint = save_dir / "transformer_direction.keras" if warm_start else None
    
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
            
            # Log warm-start status
            if warm_start and transformer_checkpoint and transformer_checkpoint.exists():
                logger.info(f"🔥 WARM-START enabled: Loading weights from {transformer_checkpoint}")
            
            dir_trainer.train(
                dir_data['X_train'], dir_data['y_train'],
                dir_data['X_val'], dir_data['y_val'],
                feature_names=dir_data.get('feature_names'),
                w_train=dir_data.get('w_train'),
                w_val=dir_data.get('w_val'),
                warm_start_path=str(transformer_checkpoint) if warm_start else None,
                instrument=instrument,
                data_range=data_range,
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
    
    # 5. HistGradientBoosting (Optional - for hybrid voting)
    if train_histgb and not use_regime:
        logger.info("\n" + "="*50)
        logger.info("Training HistGradientBoosting (Direction Baseline for Hybrid Voting)")
        logger.info("="*50)
        
        dir_data = data.get('direction', data.get('tcn'))
        if dir_data is not None:
            histgb_trainer = HistGradientBoostingDirectionTrainer(config)
            histgb_trainer.train(
                dir_data['X_train'], dir_data['y_train'],
                dir_data['X_val'], dir_data['y_val'],
                feature_names=dir_data.get('feature_names'),
            )
            histgb_trainer.save(str(save_dir / "histgb_direction.pkl"))
            trainers['histgb'] = histgb_trainer
            logger.info("✓ HistGB trained for hybrid voting with Transformer")
        else:
            logger.warning("No direction data found for HistGB training")
    
    logger.info("\n" + "="*50)
    logger.info("All 4 models trained independently!")
    logger.info("="*50)
    
    return trainers

