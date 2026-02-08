"""
Advanced training utilities based on PyTorch best practices.
Includes learning rate scheduling, model ensembling, and advanced optimization techniques.

Enhanced with:
- Time Series Cross-Validation for temporal data
- Stochastic Weight Averaging (SWA) for improved generalization
- Focal Loss for imbalanced classification
- Monte Carlo Dropout for uncertainty estimation
"""

raise SystemExit(
    "Retired: training utilities are disabled to avoid alternate training paths. "
    "Use: python main.py train-buddy"
)

# + PyTorch retired (TensorFlow-only migration)
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
import numpy as np  # noqa: E402
from typing import List, Dict, Optional, Tuple, Iterator  # noqa: E402
from sklearn.model_selection import TimeSeriesSplit  # noqa: E402
import logging  # noqa: E402

logger = logging.getLogger(__name__)


class WarmupScheduler:
    """
    Learning rate warmup scheduler (PyTorch best practice for transformers).
    Gradually increases learning rate from 0 to base_lr during warmup steps.
    """
    
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        warmup_steps: int,
        base_lr: float,
        min_lr: float = 1e-7
    ):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.base_lr = base_lr
        self.min_lr = min_lr
        self.current_step = 0
    
    def step(self):
        """Update learning rate based on current step."""
        self.current_step += 1
        
        if self.current_step <= self.warmup_steps:
            # Linear warmup
            lr = self.base_lr * (self.current_step / self.warmup_steps)
            lr = max(lr, self.min_lr)
        else:
            # After warmup, use base learning rate
            lr = self.base_lr
        
        # Update optimizer learning rate
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        
        return lr


class CosineAnnealingWarmup:
    """
    Cosine annealing with warmup (PyTorch best practice).
    Combines warmup with cosine annealing for better convergence.
    """
    
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        warmup_steps: int,
        total_steps: int,
        base_lr: float,
        min_lr: float = 1e-7
    ):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.base_lr = base_lr
        self.min_lr = min_lr
        self.current_step = 0
    
    def step(self):
        """Update learning rate with cosine annealing and warmup."""
        self.current_step += 1
        
        if self.current_step <= self.warmup_steps:
            # Warmup phase
            lr = self.base_lr * (self.current_step / self.warmup_steps)
        else:
            # Cosine annealing phase
            progress = (self.current_step - self.warmup_steps) / (self.total_steps - self.warmup_steps)
            lr = self.min_lr + (self.base_lr - self.min_lr) * 0.5 * (1 + np.cos(np.pi * progress))
        
        # Update optimizer learning rate
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        
        return lr


class EarlyStopping:
    """
    Early stopping to prevent overfitting (PyTorch best practice).
    Monitors validation metric and stops training when no improvement.
    """
    
    def __init__(
        self,
        patience: int = 10,
        min_delta: float = 0.0,
        mode: str = 'min'
    ):
        """
        Initialize early stopping.
        
        Args:
            patience: Number of epochs to wait for improvement
            min_delta: Minimum change to qualify as improvement
            mode: 'min' for loss, 'max' for accuracy
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        
    def __call__(self, score: float) -> bool:
        """
        Check if training should stop.
        
        Args:
            score: Current validation metric
            
        Returns:
            True if training should stop
        """
        if self.best_score is None:
            self.best_score = score
            return False
        
        # Check for improvement
        if self.mode == 'min':
            improved = score < (self.best_score - self.min_delta)
        else:
            improved = score > (self.best_score + self.min_delta)
        
        if improved:
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
                logger.info(f"Early stopping triggered after {self.counter} epochs without improvement")
        
        return self.early_stop


class ModelEnsemble:
    """
    Ensemble multiple models for improved predictions (PyTorch best practice).
    Supports averaging, weighted averaging, and stacking.
    """
    
    def __init__(
        self,
        models: List[nn.Module],
        weights: Optional[List[float]] = None,
        ensemble_method: str = 'average'
    ):
        """
        Initialize ensemble.
        
        Args:
            models: List of trained models
            weights: Optional weights for weighted averaging
            ensemble_method: 'average', 'weighted', or 'median'
        """
        self.models = models
        self.ensemble_method = ensemble_method
        
        if weights is None:
            self.weights = [1.0 / len(models)] * len(models)
        else:
            # Normalize weights
            total = sum(weights)
            self.weights = [w / total for w in weights]
    
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """
        Make ensemble prediction.
        
        Args:
            x: Input tensor
            
        Returns:
            Ensemble prediction
        """
        predictions = []
        
        # Get predictions from all models
        for model in self.models:
            model.eval()
            with torch.no_grad():
                pred = model(x)
                predictions.append(pred)
        
        # Stack predictions
        predictions = torch.stack(predictions, dim=0)
        
        # Combine predictions based on method
        if self.ensemble_method == 'average':
            return predictions.mean(dim=0)
        elif self.ensemble_method == 'weighted':
            weights = torch.tensor(self.weights, device=x.device).view(-1, 1, 1)
            return (predictions * weights).sum(dim=0)
        elif self.ensemble_method == 'median':
            return predictions.median(dim=0)[0]
        else:
            raise ValueError(f"Unknown ensemble method: {self.ensemble_method}")
    
    def evaluate_diversity(self, x: torch.Tensor) -> float:
        """
        Measure prediction diversity (useful for ensemble quality assessment).
        
        Args:
            x: Input tensor
            
        Returns:
            Diversity score (higher is better)
        """
        predictions = []
        
        for model in self.models:
            model.eval()
            with torch.no_grad():
                pred = model(x)
                predictions.append(pred.cpu().numpy())
        
        predictions = np.array(predictions)
        
        # Calculate standard deviation across models
        diversity = np.std(predictions, axis=0).mean()
        
        return float(diversity)


class GradientClipping:
    """
    Gradient clipping utility (PyTorch best practice).
    Provides different clipping strategies.
    """
    
    @staticmethod
    def clip_grad_norm(
        parameters,
        max_norm: float,
        norm_type: float = 2.0
    ) -> float:
        """
        Clip gradient norm (most common PyTorch practice).
        
        Args:
            parameters: Model parameters
            max_norm: Maximum gradient norm
            norm_type: Type of norm (2.0 for L2)
            
        Returns:
            Total norm before clipping
        """
        return torch.nn.utils.clip_grad_norm_(parameters, max_norm, norm_type=norm_type)
    
    @staticmethod
    def clip_grad_value(
        parameters,
        clip_value: float
    ):
        """
        Clip gradient values (alternative to norm clipping).
        
        Args:
            parameters: Model parameters
            clip_value: Maximum absolute value for gradients
        """
        torch.nn.utils.clip_grad_value_(parameters, clip_value)


class LossHistory:
    """
    Track and analyze loss history (PyTorch best practice for monitoring).
    Helps detect overfitting and training issues.
    """
    
    def __init__(self):
        self.train_losses = []
        self.val_losses = []
        self.learning_rates = []
    
    def update(
        self,
        train_loss: float,
        val_loss: Optional[float] = None,
        lr: Optional[float] = None
    ):
        """Update loss history."""
        self.train_losses.append(train_loss)
        if val_loss is not None:
            self.val_losses.append(val_loss)
        if lr is not None:
            self.learning_rates.append(lr)
    
    def get_smoothed_losses(self, window: int = 5) -> Dict[str, List[float]]:
        """
        Get smoothed losses for visualization.
        
        Args:
            window: Smoothing window size
            
        Returns:
            Dictionary with smoothed losses
        """
        def smooth(losses):
            if len(losses) < window:
                return losses
            smoothed = []
            for i in range(len(losses)):
                start = max(0, i - window + 1)
                smoothed.append(np.mean(losses[start:i+1]))
            return smoothed
        
        return {
            'train_losses': smooth(self.train_losses),
            'val_losses': smooth(self.val_losses) if self.val_losses else []
        }
    
    def detect_overfitting(self, threshold: float = 0.1) -> bool:
        """
        Detect if model is overfitting.
        
        Args:
            threshold: Gap threshold between train and validation loss
            
        Returns:
            True if overfitting detected
        """
        if len(self.val_losses) < 10:
            return False
        
        # Check last 10 epochs
        recent_train = np.mean(self.train_losses[-10:])
        recent_val = np.mean(self.val_losses[-10:])
        
        gap = recent_val - recent_train
        relative_gap = gap / recent_train if recent_train > 0 else 0
        
        return relative_gap > threshold


def create_optimizer(
    model: nn.Module,
    optimizer_type: str = 'adamw',
    learning_rate: float = 0.001,
    weight_decay: float = 0.01,
    **kwargs
) -> torch.optim.Optimizer:
    """
    Create optimizer with best practices (PyTorch recommendation).
    
    Args:
        model: Model to optimize
        optimizer_type: Type of optimizer ('adam', 'adamw', 'sgd')
        learning_rate: Learning rate
        weight_decay: Weight decay for regularization
        **kwargs: Additional optimizer arguments
        
    Returns:
        Optimizer instance
    """
    if optimizer_type.lower() == 'adam':
        return torch.optim.Adam(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
            **kwargs
        )
    elif optimizer_type.lower() == 'adamw':
        # AdamW is recommended for transformers (PyTorch best practice)
        return torch.optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
            **kwargs
        )
    elif optimizer_type.lower() == 'sgd':
        return torch.optim.SGD(
            model.parameters(),
            lr=learning_rate,
            momentum=kwargs.get('momentum', 0.9),
            weight_decay=weight_decay,
            nesterov=kwargs.get('nesterov', True)
        )
    else:
        raise ValueError(f"Unknown optimizer type: {optimizer_type}")


# =============================================================================
# Time Series Cross-Validation (for temporal data with small datasets)
# =============================================================================

class TimeSeriesCrossValidator:
    """
    Time Series Cross-Validation for proper temporal validation.
    
    Unlike standard K-Fold, this ensures training data always precedes
    validation data in time, preventing data leakage.
    
    Benefits for small datasets:
    - Better estimate of generalization performance
    - Multiple validation folds from limited data
    - Proper temporal ordering preserved
    """
    
    def __init__(
        self,
        n_splits: int = 5,
        test_size: Optional[int] = None,
        gap: int = 0,
        expanding_window: bool = True,
    ):
        """
        Initialize Time Series Cross-Validator.
        
        Args:
            n_splits: Number of splits/folds
            test_size: Size of test set in each fold (None = auto)
            gap: Gap between train and test to avoid look-ahead bias
            expanding_window: If True, use expanding window (more training data per fold)
                             If False, use sliding window (fixed window size)
        """
        self.n_splits = n_splits
        self.test_size = test_size
        self.gap = gap
        self.expanding_window = expanding_window
        self.tscv = TimeSeriesSplit(
            n_splits=n_splits,
            test_size=test_size,
            gap=gap
        )
    
    def split(
        self,
        X: np.ndarray,
        y: Optional[np.ndarray] = None,
    ) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        """
        Generate train/test indices for time series cross-validation.
        
        Args:
            X: Feature array
            y: Target array (optional, not used but kept for sklearn compatibility)
        
        Yields:
            (train_indices, test_indices) tuples
        """
        for train_idx, test_idx in self.tscv.split(X):
            yield train_idx, test_idx
    
    def get_n_splits(self) -> int:
        """Return number of splits."""
        return self.n_splits
    
    def cross_validate(
        self,
        X: np.ndarray,
        y: np.ndarray,
        model_fn,
        train_fn,
        eval_fn,
    ) -> Dict[str, List[float]]:
        """
        Perform cross-validation and return metrics for each fold.
        
        Args:
            X: Features array [samples, timesteps, features]
            y: Targets array
            model_fn: Function that returns a new model instance
            train_fn: Function(model, X_train, y_train) -> trained_model
            eval_fn: Function(model, X_test, y_test) -> metrics_dict
        
        Returns:
            Dictionary with lists of metrics for each fold
        """
        all_metrics = {}
        
        for fold, (train_idx, test_idx) in enumerate(self.split(X)):
            logger.info(f"Cross-validation fold {fold + 1}/{self.n_splits}")
            logger.info(f"  Train size: {len(train_idx)}, Test size: {len(test_idx)}")
            
            # Split data
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            
            # Create and train model
            model = model_fn()
            model = train_fn(model, X_train, y_train)
            
            # Evaluate
            metrics = eval_fn(model, X_test, y_test)
            
            # Collect metrics
            for key, value in metrics.items():
                if key not in all_metrics:
                    all_metrics[key] = []
                all_metrics[key].append(value)
            
            logger.info(f"  Fold {fold + 1} metrics: {metrics}")
        
        # Log summary statistics
        logger.info("\nCross-validation summary:")
        for key, values in all_metrics.items():
            mean_val = np.mean(values)
            std_val = np.std(values)
            logger.info(f"  {key}: {mean_val:.4f} (+/- {std_val:.4f})")
        
        return all_metrics


class FocalLoss(nn.Module):
    """
    Focal Loss for addressing class imbalance in classification.
    
    Reduces loss contribution from easy examples, focusing training
    on hard, misclassified examples.
    
    Paper: "Focal Loss for Dense Object Detection" (Lin et al., 2017)
    """
    
    def __init__(
        self,
        gamma: float = 2.0,
        alpha: Optional[float] = 0.25,
        reduction: str = 'mean',
    ):
        """
        Initialize Focal Loss.
        
        Args:
            gamma: Focusing parameter (higher = more focus on hard examples)
            alpha: Class weight for positive class (None = no weighting)
            reduction: 'mean', 'sum', or 'none'
        """
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction
    
    def forward(
        self,
        inputs: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute focal loss.
        
        Args:
            inputs: Predicted logits [batch, num_classes]
            targets: Ground truth labels [batch]
        
        Returns:
            Focal loss
        """
        ce_loss = nn.functional.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)  # Probability of correct class
        
        focal_weight = (1 - pt) ** self.gamma
        
        if self.alpha is not None:
            alpha_weight = self.alpha * targets + (1 - self.alpha) * (1 - targets)
            focal_weight = focal_weight * alpha_weight
        
        focal_loss = focal_weight * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class MonteCarloDropout:
    """
    Monte Carlo Dropout for uncertainty estimation.
    
    Runs multiple forward passes with dropout enabled to estimate
    prediction uncertainty. Useful for:
    - Detecting out-of-distribution samples
    - Assessing confidence in predictions
    - Ensemble-like behavior from a single model
    """
    
    def __init__(
        self,
        model: nn.Module,
        n_iterations: int = 50,
    ):
        """
        Initialize MC Dropout wrapper.
        
        Args:
            model: PyTorch model with dropout layers
            n_iterations: Number of forward passes
        """
        self.model = model
        self.n_iterations = n_iterations
    
    def predict_with_uncertainty(
        self,
        x: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Make predictions with uncertainty estimation.
        
        Args:
            x: Input tensor
        
        Returns:
            (mean_prediction, std_prediction) - mean and standard deviation
        """
        self.model.train()  # Enable dropout
        
        predictions = []
        with torch.no_grad():
            for _ in range(self.n_iterations):
                pred = self.model(x)
                predictions.append(pred)
        
        predictions = torch.stack(predictions, dim=0)
        
        mean_pred = predictions.mean(dim=0)
        std_pred = predictions.std(dim=0)
        
        return mean_pred, std_pred
    
    def get_confidence(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        """
        Get confidence score (inverse of uncertainty).
        
        Args:
            x: Input tensor
        
        Returns:
            Confidence scores [0, 1]
        """
        _, std = self.predict_with_uncertainty(x)
        # Convert std to confidence (higher std = lower confidence)
        confidence = 1.0 / (1.0 + std)
        return confidence


class StochasticWeightAveraging:
    """
    Stochastic Weight Averaging (SWA) for improved generalization.
    
    Averages model weights over training trajectory to find flatter
    minima, which generalize better.
    
    Paper: "Averaging Weights Leads to Wider Optima and Better Generalization"
    """
    
    def __init__(
        self,
        model: nn.Module,
        swa_start_epoch: int = 10,
        swa_freq: int = 5,
    ):
        """
        Initialize SWA.
        
        Args:
            model: PyTorch model
            swa_start_epoch: Epoch to start weight averaging
            swa_freq: Frequency of weight collection (every N epochs)
        """
        self.model = model
        self.swa_start_epoch = swa_start_epoch
        self.swa_freq = swa_freq
        self.swa_n = 0
        self.swa_state_dict = None
    
    def update(self, epoch: int):
        """
        Update SWA weights if conditions are met.
        
        Args:
            epoch: Current training epoch
        """
        if epoch < self.swa_start_epoch:
            return
        
        if (epoch - self.swa_start_epoch) % self.swa_freq != 0:
            return
        
        if self.swa_state_dict is None:
            # First SWA update - copy current weights
            self.swa_state_dict = {
                k: v.clone() for k, v in self.model.state_dict().items()
            }
        else:
            # Running average update
            current_state = self.model.state_dict()
            for k in self.swa_state_dict:
                self.swa_state_dict[k] = (
                    self.swa_state_dict[k] * self.swa_n + current_state[k]
                ) / (self.swa_n + 1)
        
        self.swa_n += 1
        logger.info(f"SWA update #{self.swa_n} at epoch {epoch}")
    
    def apply_swa_weights(self):
        """Apply averaged weights to the model."""
        if self.swa_state_dict is not None:
            self.model.load_state_dict(self.swa_state_dict)
            logger.info("Applied SWA weights to model")
        else:
            logger.warning("No SWA weights collected yet")


def create_time_series_cv_splits(
    n_samples: int,
    n_splits: int = 5,
    test_ratio: float = 0.15,
    gap: int = 0,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Create time series cross-validation splits.
    
    Args:
        n_samples: Total number of samples
        n_splits: Number of CV splits
        test_ratio: Ratio of samples for test in each split
        gap: Gap between train and test
    
    Returns:
        List of (train_indices, test_indices) tuples
    """
    test_size = int(n_samples * test_ratio)
    
    cv = TimeSeriesCrossValidator(
        n_splits=n_splits,
        test_size=test_size,
        gap=gap,
    )
    
    indices = np.arange(n_samples)
    splits = list(cv.split(indices))
    
    return splits
# — Raynergy-svg —
