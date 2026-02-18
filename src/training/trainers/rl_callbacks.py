"""
RL Training Callbacks for Hybrid SFT-RL Training.

This module provides callbacks for monitoring and controlling the RL
aspects of hybrid SFT-RL training.

Classes:
--------
- RLMetricsCallback: Tracks and logs RL-specific metrics during training
- RLBaselineScheduler: Schedules rl_prob over training (curriculum learning)

Example Usage:
--------------
    >>> from src.training.trainers.rl_callbacks import (
    ...     RLMetricsCallback, RLBaselineScheduler
    ... )
    >>> callbacks = [
    ...     RLMetricsCallback(log_dir="logs/rl"),
    ...     RLBaselineScheduler(
    ...         initial_rl_prob=0.1,
    ...         final_rl_prob=0.5,
    ...         warmup_epochs=10
    ...     )
    ... ]
    >>> model.fit(..., callbacks=callbacks)
"""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, TYPE_CHECKING

import numpy as np
import tensorflow as tf

if TYPE_CHECKING:
    from src.training.trainers.hybrid_sft_rl_loss import HybridSFTLoss, HybridSFTLossWrapper

logger = logging.getLogger(__name__)


class RLMetricsCallback(tf.keras.callbacks.Callback):
    """
    Callback to track and log RL-specific metrics during training.
    
    Monitors:
    - Reward mean and distribution
    - Baseline value over time
    - Entropy (exploration measure)
    - Action distribution (detect mode collapse)
    
    Logs to:
    - Console (via logger)
    - TensorBoard (optional)
    
    Args:
        loss_fn: HybridSFTLoss or HybridSFTLossWrapper instance
        log_dir: Directory for TensorBoard logs (optional)
        log_every_n_epochs: How often to log metrics (default: 1)
        collapse_threshold: Entropy threshold for collapse detection
        tensorboard_update_freq: Updates per epoch for TensorBoard
        
    Example:
        >>> loss_fn = HybridSFTLoss(rl_prob=0.3)
        >>> callback = RLMetricsCallback(loss_fn=loss_fn, log_dir="logs/rl")
        >>> model.compile(optimizer=optimizer, loss=loss_fn)
        >>> model.fit(..., callbacks=[callback])
    """
    
    def __init__(
        self,
        loss_fn: Optional[Any] = None,
        log_dir: Optional[str] = None,
        log_every_n_epochs: int = 1,
        collapse_threshold: float = 0.01,
        tensorboard_update_freq: int = 10,
    ):
        super().__init__()
        
        self.loss_fn = loss_fn
        self.log_dir = Path(log_dir) if log_dir else None
        self.log_every_n_epochs = log_every_n_epochs
        self.collapse_threshold = collapse_threshold
        self.tensorboard_update_freq = tensorboard_update_freq
        
        # Internal state
        self._epoch_count = 0
        self._collapse_warnings = 0
        self._summary_writer = None
        self._metrics_history: List[Dict[str, Any]] = []
        
        # Console reference
        self._console = None
    
    @property
    def console(self):
        """Get rich console for formatted output."""
        if self._console is None:
            try:
                from rich.console import Console
                self._console = Console()
            except ImportError:
                self._console = None
        return self._console
    
    def _print(self, message: str, style: Optional[str] = None) -> None:
        """Print message with optional styling."""
        if self.console:
            if style:
                self.console.print(f"[{style}]{message}[/{style}]")
            else:
                self.console.print(message)
        else:
            logger.info(message)
    
    def on_train_begin(self, logs: Optional[Dict] = None) -> None:
        """Initialize TensorBoard writer if log_dir is provided."""
        if self.log_dir:
            self.log_dir.mkdir(parents=True, exist_ok=True)
            try:
                self._summary_writer = tf.summary.create_file_writer(
                    str(self.log_dir / datetime.now().strftime("%Y%m%d-%H%M%S"))
                )
                logger.info(f"📊 RL metrics logging to {self.log_dir}")
            except Exception as e:
                logger.warning(f"Could not create TensorBoard writer: {e}")
                self._summary_writer = None
        
        self._print("🎮 RL Metrics Callback initialized", style="cyan")
    
    def on_epoch_end(
        self,
        epoch: int,
        logs: Optional[Dict[str, Any]] = None
    ) -> None:
        """Log RL metrics at the end of each epoch."""
        self._epoch_count += 1
        
        if self._epoch_count % self.log_every_n_epochs != 0:
            return
        
        # Get metrics from loss function
        metrics = self._get_loss_metrics()
        if not metrics:
            return
        
        # Store in history
        self._metrics_history.append({
            "epoch": epoch,
            "timestamp": datetime.now().isoformat(),
            **metrics
        })
        
        # Log to console
        self._log_to_console(epoch, metrics)
        
        # Log to TensorBoard
        if self._summary_writer:
            self._log_to_tensorboard(epoch, metrics)
        
        # Check for policy collapse
        self._check_policy_collapse(metrics, epoch)
    
    def _get_loss_metrics(self) -> Optional[Dict[str, Any]]:
        """Extract metrics from the loss function."""
        if self.loss_fn is None:
            # Try to find loss in model
            if hasattr(self, 'model') and hasattr(self.model, 'loss'):
                loss = self.model.loss
                if hasattr(loss, 'get_metrics_summary'):
                    return loss.get_metrics_summary()
                elif hasattr(loss, 'hybrid_loss'):
                    return loss.hybrid_loss.get_metrics_summary()
            return None
        
        # Handle both HybridSFTLoss and HybridSFTLossWrapper
        if hasattr(self.loss_fn, 'get_metrics_summary'):
            return self.loss_fn.get_metrics_summary()
        elif hasattr(self.loss_fn, 'hybrid_loss'):
            return self.loss_fn.hybrid_loss.get_metrics_summary()
        
        return None
    
    def _log_to_console(self, epoch: int, metrics: Dict[str, Any]) -> None:
        """Log metrics to console with formatting."""
        mean_reward = metrics.get("mean_reward", 0.0)
        mean_entropy = metrics.get("mean_entropy", 0.0)
        baseline = metrics.get("current_baseline", 0.0)
        rl_ratio = metrics.get("rl_ratio", 0.0)
        
        # Format action distribution
        action_dist = metrics.get("action_distribution", {})
        action_str = ", ".join(
            f"{k}:{v}" for k, v in sorted(action_dist.items())
        ) if action_dist else "N/A"
        
        self._print(
            f"  🎮 Epoch {epoch + 1}: RL metrics - "
            f"reward={mean_reward:.3f}, entropy={mean_entropy:.4f}, "
            f"baseline={baseline:.3f}, rl_ratio={rl_ratio:.1%}"
        )
        
        if action_str != "N/A":
            self._print(f"     Actions: {action_str}", style="dim")
    
    def _log_to_tensorboard(
        self,
        epoch: int,
        metrics: Dict[str, Any]
    ) -> None:
        """Log metrics to TensorBoard."""
        if not self._summary_writer:
            return
        
        try:
            with self._summary_writer.as_default():
                tf.summary.scalar(
                    "rl/reward_mean",
                    metrics.get("mean_reward", 0.0),
                    step=epoch
                )
                tf.summary.scalar(
                    "rl/entropy",
                    metrics.get("mean_entropy", 0.0),
                    step=epoch
                )
                tf.summary.scalar(
                    "rl/baseline",
                    metrics.get("current_baseline", 0.0),
                    step=epoch
                )
                tf.summary.scalar(
                    "rl/rl_ratio",
                    metrics.get("rl_ratio", 0.0),
                    step=epoch
                )
                
                # Log action distribution as histogram
                action_dist = metrics.get("action_distribution", {})
                if action_dist:
                    actions = []
                    for action, count in action_dist.items():
                        actions.extend([action] * count)
                    if actions:
                        tf.summary.histogram(
                            "rl/actions",
                            np.array(actions),
                            step=epoch
                        )
                
                self._summary_writer.flush()
        except Exception as e:
            logger.warning(f"Could not log to TensorBoard: {e}")
    
    def _check_policy_collapse(
        self,
        metrics: Dict[str, Any],
        epoch: int
    ) -> None:
        """Check for policy collapse and warn."""
        is_collapsed = metrics.get("policy_collapsed", False)
        reason = metrics.get("collapse_reason", "")
        
        if is_collapsed:
            self._collapse_warnings += 1
            self._print(
                f"  ⚠️ POLICY COLLAPSE DETECTED at epoch {epoch + 1}: {reason}",
                style="red bold"
            )
            
            if self._collapse_warnings >= 3:
                self._print(
                    "  🔴 Multiple collapse warnings. Consider increasing entropy_coef.",
                    style="yellow"
                )
    
    def on_train_end(self, logs: Optional[Dict] = None) -> None:
        """Clean up and log final summary."""
        if self._summary_writer:
            self._summary_writer.close()
        
        # Log final summary
        if self._metrics_history:
            final_metrics = self._metrics_history[-1]
            self._print(
                f"  📊 Final RL metrics: reward={final_metrics.get('mean_reward', 0):.3f}, "
                f"entropy={final_metrics.get('mean_entropy', 0):.4f}",
                style="green"
            )
            
            # Log collapse warning count
            if self._collapse_warnings > 0:
                self._print(
                    f"  ⚠️ Total collapse warnings: {self._collapse_warnings}",
                    style="yellow"
                )
    
    def get_metrics_history(self) -> List[Dict[str, Any]]:
        """Get the full metrics history."""
        return self._metrics_history.copy()


class RLBaselineScheduler(tf.keras.callbacks.Callback):
    """
    Schedules rl_prob over training for curriculum learning.
    
    Starts with pure supervised learning (rl_prob=0 or low) and gradually
    increases RL proportion. This allows the model to first learn a good
    policy through supervision before exploring with RL.
    
    Scheduling Strategies:
    ----------------------
    1. **linear**: Linear increase from initial to final
    2. **cosine**: Cosine annealing schedule (smoother transition)
    3. **step**: Step-wise increase at specific epochs
    
    Args:
        loss_fn: HybridSFTLoss or HybridSFTLossWrapper instance
        initial_rl_prob: Starting RL probability (default: 0.0)
        final_rl_prob: Final RL probability (default: 0.5)
        warmup_epochs: Epochs before reaching final_rl_prob
        schedule_type: "linear", "cosine", or "step"
        step_epochs: List of epochs to increase rl_prob (for "step" schedule)
        step_values: List of rl_prob values for each step
        min_rl_prob: Minimum rl_prob (never go below this)
        
    Example:
        >>> scheduler = RLBaselineScheduler(
        ...     loss_fn=loss_fn,
        ...     initial_rl_prob=0.0,
        ...     final_rl_prob=0.5,
        ...     warmup_epochs=20,
        ...     schedule_type="linear"
        ... )
        >>> model.fit(..., callbacks=[scheduler])
    """
    
    def __init__(
        self,
        loss_fn: Optional[Any] = None,
        initial_rl_prob: float = 0.0,
        final_rl_prob: float = 0.5,
        warmup_epochs: int = 10,
        schedule_type: str = "linear",
        step_epochs: Optional[List[int]] = None,
        step_values: Optional[List[float]] = None,
        min_rl_prob: float = 0.0,
    ):
        super().__init__()
        
        self.loss_fn = loss_fn
        self.initial_rl_prob = initial_rl_prob
        self.final_rl_prob = final_rl_prob
        self.warmup_epochs = warmup_epochs
        self.schedule_type = schedule_type
        self.step_epochs = step_epochs or []
        self.step_values = step_values or []
        self.min_rl_prob = min_rl_prob
        
        # Validate
        if schedule_type not in ("linear", "cosine", "step"):
            raise ValueError(
                f"Invalid schedule_type: {schedule_type}. "
                "Must be 'linear', 'cosine', or 'step'"
            )
        
        if schedule_type == "step" and len(self.step_epochs) != len(self.step_values):
            raise ValueError(
                "step_epochs and step_values must have same length"
            )
        
        # Internal state
        self._current_rl_prob = initial_rl_prob
        self._epoch_count = 0
        self._console = None
    
    @property
    def console(self):
        """Get rich console for formatted output."""
        if self._console is None:
            try:
                from rich.console import Console
                self._console = Console()
            except ImportError:
                self._console = None
        return self._console
    
    def _print(self, message: str, style: Optional[str] = None) -> None:
        """Print message with optional styling."""
        if self.console:
            if style:
                self.console.print(f"[{style}]{message}[/{style}]")
            else:
                self.console.print(message)
        else:
            logger.info(message)
    
    def _get_loss_object(self) -> Optional[Any]:
        """Get the loss object to update."""
        if self.loss_fn is not None:
            return self.loss_fn
        
        # Try to find loss in model
        if hasattr(self, 'model') and hasattr(self.model, 'loss'):
            loss = self.model.loss
            if hasattr(loss, 'rl_prob'):  # HybridSFTLoss
                return loss
            elif hasattr(loss, 'hybrid_loss'):  # HybridSFTLossWrapper
                return loss.hybrid_loss
        
        return None
    
    def _compute_linear_schedule(self, epoch: int) -> float:
        """Compute rl_prob using linear schedule."""
        if epoch >= self.warmup_epochs:
            return self.final_rl_prob
        
        progress = epoch / self.warmup_epochs
        return self.initial_rl_prob + progress * (self.final_rl_prob - self.initial_rl_prob)
    
    def _compute_cosine_schedule(self, epoch: int) -> float:
        """Compute rl_prob using cosine schedule."""
        if epoch >= self.warmup_epochs:
            return self.final_rl_prob
        
        # Cosine annealing from initial to final
        progress = epoch / self.warmup_epochs
        cosine_factor = (1 - np.cos(np.pi * progress)) / 2
        return self.initial_rl_prob + cosine_factor * (self.final_rl_prob - self.initial_rl_prob)
    
    def _compute_step_schedule(self, epoch: int) -> float:
        """Compute rl_prob using step schedule."""
        # Find the appropriate step
        for step_epoch, step_value in zip(self.step_epochs, self.step_values):
            if epoch < step_epoch:
                break
            current_value = step_value
        else:
            current_value = self.final_rl_prob
        
        return max(current_value, self.min_rl_prob)
    
    def _compute_rl_prob(self, epoch: int) -> float:
        """Compute the rl_prob for current epoch."""
        if self.schedule_type == "linear":
            rl_prob = self._compute_linear_schedule(epoch)
        elif self.schedule_type == "cosine":
            rl_prob = self._compute_cosine_schedule(epoch)
        elif self.schedule_type == "step":
            rl_prob = self._compute_step_schedule(epoch)
        else:
            rl_prob = self.final_rl_prob
        
        return max(rl_prob, self.min_rl_prob)
    
    def _update_loss_rl_prob(self, rl_prob: float) -> bool:
        """Update the loss function's rl_prob."""
        loss_obj = self._get_loss_object()
        if loss_obj is None:
            return False
        
        try:
            if hasattr(loss_obj, 'rl_prob'):
                # Direct attribute (HybridSFTLoss)
                loss_obj.rl_prob = rl_prob
            elif hasattr(loss_obj, '_hybrid_loss'):
                # Wrapper (HybridSFTLossWrapper)
                loss_obj._hybrid_loss.rl_prob = rl_prob
            else:
                return False
            return True
        except Exception as e:
            logger.warning(f"Could not update rl_prob: {e}")
            return False
    
    def on_train_begin(self, logs: Optional[Dict] = None) -> None:
        """Log schedule info at training start."""
        self._print(
            f"🎮 RL Baseline Scheduler: {self.schedule_type} schedule, "
            f"rl_prob: {self.initial_rl_prob:.2f} → {self.final_rl_prob:.2f} "
            f"over {self.warmup_epochs} epochs",
            style="cyan"
        )
        
        # Set initial rl_prob
        self._update_loss_rl_prob(self.initial_rl_prob)
        self._current_rl_prob = self.initial_rl_prob
    
    def on_epoch_begin(self, epoch: int, logs: Optional[Dict] = None) -> None:
        """Update rl_prob at the beginning of each epoch."""
        new_rl_prob = self._compute_rl_prob(epoch)
        
        # Only update and log if changed significantly
        if abs(new_rl_prob - self._current_rl_prob) > 0.001:
            if self._update_loss_rl_prob(new_rl_prob):
                self._current_rl_prob = new_rl_prob
                
                # Log the update
                if epoch < self.warmup_epochs:
                    self._print(
                        f"  🎮 Epoch {epoch + 1}: rl_prob → {new_rl_prob:.2f} "
                        f"(curriculum: {epoch + 1}/{self.warmup_epochs})",
                        style="cyan"
                    )
                else:
                    self._print(
                        f"  🎮 Epoch {epoch + 1}: rl_prob → {new_rl_prob:.2f} (final)",
                        style="green"
                    )
        
        self._epoch_count += 1
    
    def on_train_end(self, logs: Optional[Dict] = None) -> None:
        """Log final schedule state."""
        self._print(
            f"  📊 RL schedule complete: final rl_prob = {self._current_rl_prob:.2f}",
            style="green"
        )
    
    def get_current_rl_prob(self) -> float:
        """Get the current rl_prob value."""
        return self._current_rl_prob


class RLEarlyStoppingCallback(tf.keras.callbacks.Callback):
    """
    Early stopping callback based on RL metrics.
    
    Stops training if:
    - Policy collapse is detected (entropy too low)
    - Reward plateaus for too long
    - Action distribution becomes too skewed
    
    Args:
        loss_fn: HybridSFTLoss instance
        min_entropy: Minimum entropy threshold
        min_reward: Minimum reward threshold
        patience: Epochs to wait before stopping
        restore_best_weights: Whether to restore best weights on stop
        
    Example:
        >>> callback = RLEarlyStoppingCallback(
        ...     loss_fn=loss_fn,
        ...     min_entropy=0.01,
        ...     patience=5
        ... )
        >>> model.fit(..., callbacks=[callback])
    """
    
    def __init__(
        self,
        loss_fn: Optional[Any] = None,
        min_entropy: float = 0.005,
        min_reward: float = 0.3,
        patience: int = 10,
        restore_best_weights: bool = True,
    ):
        super().__init__()
        
        self.loss_fn = loss_fn
        self.min_entropy = min_entropy
        self.min_reward = min_reward
        self.patience = patience
        self.restore_best_weights = restore_best_weights
        
        # Internal state
        self._low_entropy_count = 0
        self._low_reward_count = 0
        self._best_weights = None
        self._best_reward = 0.0
        self._stopped_epoch = 0
    
    def _get_loss_metrics(self) -> Optional[Dict[str, Any]]:
        """Extract metrics from the loss function."""
        if self.loss_fn is None:
            if hasattr(self, 'model') and hasattr(self.model, 'loss'):
                loss = self.model.loss
                if hasattr(loss, 'get_metrics_summary'):
                    return loss.get_metrics_summary()
                elif hasattr(loss, 'hybrid_loss'):
                    return loss.hybrid_loss.get_metrics_summary()
            return None
        
        if hasattr(self.loss_fn, 'get_metrics_summary'):
            return self.loss_fn.get_metrics_summary()
        elif hasattr(self.loss_fn, 'hybrid_loss'):
            return self.loss_fn.hybrid_loss.get_metrics_summary()
        
        return None
    
    def on_epoch_end(self, epoch: int, logs: Optional[Dict] = None) -> None:
        """Check RL metrics and potentially stop training."""
        metrics = self._get_loss_metrics()
        if not metrics:
            return
        
        entropy = metrics.get("mean_entropy", 1.0)
        reward = metrics.get("mean_reward", 0.0)
        
        # Track best weights
        if reward > self._best_reward:
            self._best_reward = reward
            if self.restore_best_weights and hasattr(self, 'model'):
                self._best_weights = self.model.get_weights()
        
        # Check entropy
        if entropy < self.min_entropy:
            self._low_entropy_count += 1
            logger.warning(
                f"⚠️ Low entropy: {entropy:.4f} < {self.min_entropy:.4f} "
                f"({self._low_entropy_count}/{self.patience})"
            )
        else:
            self._low_entropy_count = 0
        
        # Check reward
        if reward < self.min_reward:
            self._low_reward_count += 1
        else:
            self._low_reward_count = 0
        
        # Check if should stop
        should_stop = (
            self._low_entropy_count >= self.patience or
            self._low_reward_count >= self.patience * 2
        )
        
        if should_stop:
            self._stopped_epoch = epoch
            self.model.stop_training = True
            
            reason = "policy collapse" if self._low_entropy_count >= self.patience \
                else "low reward"
            logger.warning(
                f"🛑 RL Early stopping at epoch {epoch + 1}: {reason}"
            )
            
            # Restore best weights
            if self.restore_best_weights and self._best_weights is not None:
                self.model.set_weights(self._best_weights)
                logger.info(
                    f"↩️ Restored best weights (reward={self._best_reward:.3f})"
                )


# Export public API
__all__ = [
    "RLMetricsCallback",
    "RLBaselineScheduler",
    "RLEarlyStoppingCallback",
]
