"""
Hybrid SFT-RL Loss Module.

Implements a stochastic switch between Cross-Entropy (Supervised) and 
REINFORCE Policy Gradient (RL) loss to mitigate memorization.

This module provides:
- HybridSFTLoss: Main loss class that stochastically switches between SFT and RL
- RLMetricsTracker: Helper class to track RL-specific metrics

Key Concepts:
-------------
1. **Stochastic Switch**: Each batch has a probability (rl_prob) of using RL loss
2. **REINFORCE Policy Gradient**: Uses reward signal to update policy (model)
3. **Entropy Bonus**: Encourages exploration to prevent policy collapse
4. **Baseline Reduction**: EMA baseline reduces variance of gradient estimates

Example Usage:
--------------
    >>> from src.training.trainers.hybrid_sft_rl_loss import HybridSFTLoss
    >>> loss_fn = HybridSFTLoss(
    ...     rl_prob=0.3,
    ...     entropy_coef=0.01,
    ...     num_classes=2
    ... )
    >>> # Use in model.compile()
    >>> model.compile(optimizer=optimizer, loss=loss_fn)

References:
-----------
- Williams, R. J. (1992). Simple statistical gradient-following algorithms for 
  connectionist reinforcement learning.
- Mnih, V., et al. (2016). Asynchronous methods for deep reinforcement learning.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple

import numpy as np
import tensorflow as tf

logger = logging.getLogger(__name__)


@dataclass
class RLMetricsTracker:
    """
    Tracks RL-specific metrics during training.
    
    Attributes:
        reward_history: List of mean rewards per batch
        baseline_history: List of baseline values over time
        entropy_history: List of entropy values per batch
        action_distribution: Counts of actions taken
        rl_steps: Number of batches processed with RL loss
        sft_steps: Number of batches processed with SFT loss
    """
    reward_history: List[float] = field(default_factory=list)
    baseline_history: List[float] = field(default_factory=list)
    entropy_history: List[float] = field(default_factory=list)
    action_distribution: Dict[int, int] = field(default_factory=dict)
    rl_steps: int = 0
    sft_steps: int = 0
    
    def update(
        self,
        reward_mean: float,
        baseline: float,
        entropy: float,
        actions: np.ndarray,
        is_rl: bool
    ) -> None:
        """Update tracked metrics with latest batch results."""
        self.reward_history.append(reward_mean)
        self.baseline_history.append(baseline)
        self.entropy_history.append(entropy)
        
        # Update action distribution
        for action in actions:
            action_int = int(action)
            self.action_distribution[action_int] = \
                self.action_distribution.get(action_int, 0) + 1
        
        if is_rl:
            self.rl_steps += 1
        else:
            self.sft_steps += 1
    
    def get_summary(self) -> Dict[str, Any]:
        """Get summary statistics of tracked metrics."""
        if not self.reward_history:
            return {"status": "no_data"}
        
        recent_rewards = self.reward_history[-100:] if len(self.reward_history) > 100 \
            else self.reward_history
        recent_entropy = self.entropy_history[-100:] if len(self.entropy_history) > 100 \
            else self.entropy_history
        
        total_steps = self.rl_steps + self.sft_steps
        rl_ratio = self.rl_steps / total_steps if total_steps > 0 else 0.0
        
        return {
            "mean_reward": float(np.mean(recent_rewards)),
            "mean_entropy": float(np.mean(recent_entropy)),
            "current_baseline": self.baseline_history[-1] if self.baseline_history else 0.0,
            "rl_ratio": rl_ratio,
            "action_distribution": dict(self.action_distribution),
            "total_rl_steps": self.rl_steps,
            "total_sft_steps": self.sft_steps,
        }
    
    def check_policy_collapse(self, threshold: float = 0.01) -> Tuple[bool, str]:
        """
        Check if policy has collapsed (entropy near zero).
        
        Args:
            threshold: Entropy threshold below which collapse is detected
            
        Returns:
            Tuple of (is_collapsed, reason)
        """
        if len(self.entropy_history) < 10:
            return False, "Insufficient data"
        
        recent_entropy = self.entropy_history[-10:]
        mean_entropy = np.mean(recent_entropy)
        
        if mean_entropy < threshold:
            return True, f"Entropy collapsed to {mean_entropy:.4f}"
        
        # Check for mode collapse in action distribution
        if self.action_distribution:
            total_actions = sum(self.action_distribution.values())
            if total_actions > 100:  # Need enough samples
                max_action_ratio = max(self.action_distribution.values()) / total_actions
                if max_action_ratio > 0.95:
                    return True, f"Mode collapse: {max_action_ratio:.1%} actions same class"
        
        return False, "Policy healthy"


class HybridSFTLoss(tf.keras.losses.Loss):
    """
    Hybrid Supervised-Reinforcement Learning Loss.
    
    Stochastically switches between Cross-Entropy (Supervised) and 
    REINFORCE Policy Gradient (RL) loss to mitigate memorization.
    
    The RL path encourages exploration and prevents the model from
    memorizing the training data by:
    1. Sampling actions from the model's probability distribution
    2. Computing rewards based on action correctness
    3. Using policy gradient to reinforce good actions
    4. Adding entropy bonus to maintain exploration
    
    Args:
        rl_prob: Probability of using RL loss per batch (default: 0.5)
        entropy_coef: Entropy bonus coefficient (default: 0.01)
        initial_baseline: Initial reward baseline for variance reduction (default: 0.5)
        baseline_momentum: EMA momentum for baseline updates (default: 0.9)
        num_classes: Number of output classes (default: 2 for binary)
        label_smoothing: Label smoothing for supervised loss (default: 0.0)
        name: Loss name for logging
        
    Attributes:
        metrics_tracker: RLMetricsTracker instance for tracking RL metrics (eager mode)
        reward_baseline: Current EMA baseline for reward normalization
        
    Example:
        >>> loss_fn = HybridSFTLoss(
        ...     rl_prob=0.3,
        ...     entropy_coef=0.01,
        ...     num_classes=2
        ... )
        >>> model.compile(optimizer=optimizer, loss=loss_fn)
    """
    
    def __init__(
        self,
        rl_prob: float = 0.5,
        entropy_coef: float = 0.01,
        initial_baseline: float = 0.5,
        baseline_momentum: float = 0.9,
        num_classes: int = 2,
        label_smoothing: float = 0.0,
        name: str = "hybrid_sft_rl_loss",
        **kwargs
    ):
        super().__init__(name=name, **kwargs)
        
        self.rl_prob = rl_prob
        self.entropy_coef = entropy_coef
        self.num_classes = num_classes
        self.label_smoothing = label_smoothing
        self.baseline_momentum = baseline_momentum
        
        # Initialize baseline as a non-trainable variable (updated manually)
        self._reward_baseline = tf.Variable(
            initial_baseline,
            dtype=tf.float32,
            name="reward_baseline",
            trainable=False
        )
        
        # In-graph metric tracking using tf.Variable for graph-mode compatibility
        # These accumulate during training and can be read at epoch end
        self._rl_step_count = tf.Variable(
            0.0, dtype=tf.float32, name="rl_step_count", trainable=False
        )
        self._sft_step_count = tf.Variable(
            0.0, dtype=tf.float32, name="sft_step_count", trainable=False
        )
        self._total_reward_sum = tf.Variable(
            0.0, dtype=tf.float32, name="total_reward_sum", trainable=False
        )
        self._total_entropy_sum = tf.Variable(
            0.0, dtype=tf.float32, name="total_entropy_sum", trainable=False
        )
        
        # Python-based metrics tracker for eager mode (post-training analysis)
        self.metrics_tracker = RLMetricsTracker()
        
        logger.info(
            f"🎮 HybridSFTLoss initialized: rl_prob={rl_prob:.2f}, "
            f"entropy_coef={entropy_coef:.4f}, baseline={initial_baseline:.2f}"
        )
    
    @property
    def reward_baseline(self) -> float:
        """Get current reward baseline value."""
        return float(self._reward_baseline.numpy())
    
    def get_config(self) -> Dict[str, Any]:
        """Get loss configuration for serialization."""
        return {
            "rl_prob": self.rl_prob,
            "entropy_coef": self.entropy_coef,
            "initial_baseline": self.reward_baseline,
            "baseline_momentum": self.baseline_momentum,
            "num_classes": self.num_classes,
            "label_smoothing": self.label_smoothing,
            "name": self.name,
        }
    
    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "HybridSFTLoss":
        """Create loss from configuration dictionary."""
        return cls(**config)
    
    def _compute_rl_loss(
        self,
        y_true: tf.Tensor,
        y_pred: tf.Tensor
    ) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
        """
        Compute REINFORCE policy gradient loss with entropy bonus.
        
        Args:
            y_true: Ground truth labels (batch_size,) or (batch_size, 1)
            y_pred: Model predictions (sigmoid probabilities, shape (batch_size,1))
            
        Returns:
            Tuple of (loss, reward_mean, entropy, actions)
        """
        # Ensure y_true is integer type for indexing
        y_true_int = tf.cast(tf.squeeze(y_true), tf.int32)
        batch_size = tf.shape(y_true_int)[0]
        
        # Handle edge case of empty batch using tf.cond for graph compatibility
        def empty_batch():
            return (tf.constant(0.0), tf.constant(0.0), tf.constant(0.0), 
                    tf.constant([], dtype=tf.float32))
        
        def compute_loss():
            # Model outputs sigmoid probability p (shape: batch_size,1)
            # Squeeze to get shape (batch_size,)
            prob_up = tf.squeeze(y_pred, axis=-1)
            
            # Clamp probabilities to avoid log(0)
            prob_up_safe = tf.clip_by_value(prob_up, 1e-8, 1.0 - 1e-8)
            prob_down_safe = 1.0 - prob_up_safe
            
            # Sample actions from Bernoulli distribution
            # action =1 (up) with probability prob_up, action =0 (down) otherwise
            random_vals = tf.random.uniform(tf.shape(prob_up_safe))
            actions = tf.cast(random_vals < prob_up_safe, tf.int32)
            
            # Get log probabilities of sampled actions
            # If action ==1, log_prob = log(prob_up); else log_prob = log(1 - prob_up)
            log_prob_up = tf.math.log(prob_up_safe)
            log_prob_down = tf.math.log(prob_down_safe)
            log_probs = tf.where(tf.equal(actions, 1), log_prob_up, log_prob_down)
            
            # Compute reward: 1.0 if action matches target, 0.0 otherwise
            rewards = tf.cast(tf.equal(actions, y_true_int), tf.float32)
            reward_mean = tf.reduce_mean(rewards)
            
            # Update baseline with EMA (detached from computation graph)
            # This is done with tf.stop_gradient to prevent backprop
            new_baseline = (
                self.baseline_momentum * self._reward_baseline +
                (1.0 - self.baseline_momentum) * tf.stop_gradient(reward_mean)
            )
            self._reward_baseline.assign(new_baseline)
            
            # Compute advantage (reward - baseline)
            advantage = rewards - tf.stop_gradient(self._reward_baseline)
            
            # Policy gradient loss: -log_prob * advantage
            # Negative because we want to maximize expected reward
            pg_loss = -log_probs * advantage
            
            # Compute binary entropy for exploration bonus
            # H(p) = -p*log(p) - (1-p)*log(1-p)
            entropy = -(prob_up_safe * log_prob_up + prob_down_safe * log_prob_down)
            entropy_mean = tf.reduce_mean(entropy)
            
            # Total loss: policy gradient - entropy bonus
            # Subtract entropy to encourage exploration (higher entropy = more exploration)
            loss = tf.reduce_mean(pg_loss) - self.entropy_coef * entropy_mean
            
            return loss, reward_mean, entropy_mean, tf.cast(actions, tf.float32)
        
        return tf.cond(batch_size > 0, compute_loss, empty_batch)
    
    def _compute_sft_loss(
        self,
        y_true: tf.Tensor,
        y_pred: tf.Tensor
    ) -> tf.Tensor:
        """
        Compute standard supervised cross-entropy loss.
        
        Args:
            y_true: Ground truth labels
            y_pred: Model predictions (probabilities from sigmoid)
            
        Returns:
            Scalar loss value
        """
        # Ensure y_true is1D for compatibility - use reshape instead of squeeze
        # to maintain known rank in graph mode
        batch_size = tf.shape(y_pred)[0]
        y_true_flat = tf.reshape(y_true, [batch_size])
        
        # The model outputs a single sigmoid probability (Dense(1, activation='sigmoid'))
        # Use binary_crossentropy with from_logits=False since output is already probability
        loss = tf.keras.losses.binary_crossentropy(
            y_true_flat, tf.squeeze(y_pred, axis=-1), from_logits=False
        )
        
        return tf.reduce_mean(loss)
    
    def call(
        self,
        y_true: tf.Tensor,
        y_pred: tf.Tensor
    ) -> tf.Tensor:
        """
        Compute hybrid loss with stochastic switch.
        
        Stochastically chooses between RL and SFT loss based on rl_prob.
        All operations are graph-compatible using pure TensorFlow ops.
        
        Args:
            y_true: Ground truth labels
            y_pred: Model predictions (sigmoid probabilities)
            
        Returns:
            Scalar loss value
        """
        # Stochastic switch: use RL loss with probability rl_prob
        use_rl = tf.random.uniform([]) < self.rl_prob
        
        # Compute both losses (TensorFlow will optimize unused computation)
        rl_loss, reward_mean, entropy, actions = self._compute_rl_loss(y_true, y_pred)
        sft_loss = self._compute_sft_loss(y_true, y_pred)
        
        # Select which loss to use based on stochastic switch
        loss = tf.cond(use_rl, lambda: rl_loss, lambda: sft_loss)
        
        # === In-graph metric tracking using pure TensorFlow ops ===
        # Update step counters based on which loss was selected
        rl_increment = tf.cast(use_rl, tf.float32)
        sft_increment = tf.cast(tf.logical_not(use_rl), tf.float32)
        
        self._rl_step_count.assign_add(rl_increment)
        self._sft_step_count.assign_add(sft_increment)
        
        # Accumulate reward and entropy for averaging (only on RL steps)
        # Use tf.cond to conditionally update these metrics
        def update_rl_metrics():
            self._total_reward_sum.assign_add(reward_mean)
            self._total_entropy_sum.assign_add(entropy)
            return tf.constant(0.0)  # Return dummy value
        
        def skip_rl_metrics():
            return tf.constant(0.0)
        
        tf.cond(use_rl, update_rl_metrics, skip_rl_metrics)
        
        # === Eager-mode metrics tracking (for post-training analysis) ===
        # Only update Python metrics tracker when executing eagerly
        # This avoids the SymbolicTensor.numpy() error during graph tracing
        if tf.executing_eagerly():
            self.metrics_tracker.update(
                reward_mean=float(reward_mean.numpy()),
                baseline=self.reward_baseline,
                entropy=float(entropy.numpy()),
                actions=actions.numpy(),
                is_rl=bool(use_rl.numpy())
            )
        
        return loss
    
    def get_metrics_summary(self) -> Dict[str, Any]:
        """
        Get summary of RL metrics for logging.
        
        Combines in-graph metrics (tf.Variable based) with Python metrics
        (eager-mode only). The in-graph metrics are always available,
        while Python metrics are only populated during eager execution.
        
        Returns:
            Dictionary containing RL metrics summary
        """
        # Get in-graph metrics (always available)
        rl_steps = float(self._rl_step_count.numpy())
        sft_steps = float(self._sft_step_count.numpy())
        total_steps = rl_steps + sft_steps
        
        # Calculate average reward and entropy from in-graph accumulators
        avg_reward = float(self._total_reward_sum.numpy()) / max(rl_steps, 1.0)
        avg_entropy = float(self._total_entropy_sum.numpy()) / max(rl_steps, 1.0)
        
        # Build summary with in-graph metrics
        summary = {
            "current_baseline": self.reward_baseline,
            "rl_prob": self.rl_prob,
            "entropy_coef": self.entropy_coef,
            # In-graph metrics
            "total_rl_steps": rl_steps,
            "total_sft_steps": sft_steps,
            "rl_ratio": rl_steps / total_steps if total_steps > 0 else 0.0,
            "mean_reward": avg_reward,
            "mean_entropy": avg_entropy,
        }
        
        # Merge Python-based metrics if available (eager mode tracking)
        python_summary = self.metrics_tracker.get_summary()
        if python_summary.get("status") != "no_data":
            # Python tracker has more detailed stats when available
            summary["action_distribution"] = python_summary.get("action_distribution", {})
            # Check for policy collapse using Python tracker
            is_collapsed, reason = self.metrics_tracker.check_policy_collapse()
            summary["policy_collapsed"] = is_collapsed
            summary["collapse_reason"] = reason
        else:
            summary["action_distribution"] = {}
            summary["policy_collapsed"] = False
            summary["collapse_reason"] = "Insufficient data for collapse detection"
        
        return summary
    
    def reset_metrics(self) -> None:
        """Reset both in-graph and Python metrics trackers."""
        # Reset in-graph tf.Variables
        self._rl_step_count.assign(0.0)
        self._sft_step_count.assign(0.0)
        self._total_reward_sum.assign(0.0)
        self._total_entropy_sum.assign(0.0)
        
        # Reset Python metrics tracker
        self.metrics_tracker = RLMetricsTracker()
        logger.info("🎮 HybridSFTLoss metrics reset")


class HybridSFTLossWrapper(tf.keras.losses.Loss):
    """
    Wrapper for HybridSFTLoss that provides compatibility with existing training code.
    
    This wrapper ensures the loss works with both:
    - Standard Keras training (model.fit)
    - Custom training loops
    
    It also provides callbacks for monitoring RL metrics during training.
    
    Args:
        config: TrainerConfig instance containing SFT-RL parameters
        name: Loss name
        
    Example:
        >>> from src.training.trainers.config import TrainerConfig
        >>> config = TrainerConfig(use_hybrid_sft_rl=True, rl_prob=0.3)
        >>> loss_fn = HybridSFTLossWrapper(config)
        >>> model.compile(optimizer=optimizer, loss=loss_fn)
    """
    
    def __init__(
        self,
        config: Any,
        name: str = "hybrid_sft_rl_loss_wrapper",
        **kwargs
    ):
        super().__init__(name=name, **kwargs)
        
        # Extract configuration
        rl_prob = getattr(config, "rl_prob", 0.5)
        entropy_coef = getattr(config, "entropy_coef", 0.01)
        initial_baseline = getattr(config, "initial_baseline", 0.5)
        baseline_momentum = getattr(config, "baseline_momentum", 0.9)
        label_smoothing = getattr(config, "label_smoothing", 0.0)
        
        # Determine num_classes from config or default to binary
        num_classes = getattr(config, "num_classes", 2)
        
        self._hybrid_loss = HybridSFTLoss(
            rl_prob=rl_prob,
            entropy_coef=entropy_coef,
            initial_baseline=initial_baseline,
            baseline_momentum=baseline_momentum,
            num_classes=num_classes,
            label_smoothing=label_smoothing,
            name="hybrid_sft_rl_core"
        )
        
        logger.info(
            f"🎮 HybridSFTLossWrapper created with rl_prob={rl_prob:.2f}, "
            f"entropy_coef={entropy_coef:.4f}"
        )
    
    @property
    def hybrid_loss(self) -> HybridSFTLoss:
        """Get the underlying HybridSFTLoss instance."""
        return self._hybrid_loss
    
    def call(self, y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        """Compute loss."""
        return self._hybrid_loss(y_true, y_pred)
    
    def get_config(self) -> Dict[str, Any]:
        """Get loss configuration."""
        return self._hybrid_loss.get_config()
    
    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "HybridSFTLossWrapper":
        """Create from configuration."""
        from src.training.trainers.config import TrainerConfig
        
        trainer_config = TrainerConfig()
        for key, value in config.items():
            if hasattr(trainer_config, key):
                setattr(trainer_config, key, value)
        
        return cls(trainer_config)
    
    def get_metrics_summary(self) -> Dict[str, Any]:
        """Get RL metrics summary."""
        return self._hybrid_loss.get_metrics_summary()
    
    def reset_metrics(self) -> None:
        """Reset metrics tracker."""
        self._hybrid_loss.reset_metrics()


def create_hybrid_sft_rl_loss(
    config: Any,
    **kwargs
) -> tf.keras.losses.Loss:
    """
    Factory function to create Hybrid SFT-RL loss.
    
    Args:
        config: TrainerConfig instance
        **kwargs: Additional arguments to pass to HybridSFTLoss
        
    Returns:
        Configured loss function
    """
    if not getattr(config, "use_hybrid_sft_rl", False):
        raise ValueError(
            "Hybrid SFT-RL loss requested but use_hybrid_sft_rl=False in config"
        )
    
    return HybridSFTLossWrapper(config, **kwargs)


# Export public API
__all__ = [
    "HybridSFTLoss",
    "HybridSFTLossWrapper",
    "RLMetricsTracker",
    "create_hybrid_sft_rl_loss",
]
