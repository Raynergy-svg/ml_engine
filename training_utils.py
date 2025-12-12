"""
Advanced training utilities based on PyTorch best practices.
Includes learning rate scheduling, model ensembling, and advanced optimization techniques.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import List, Dict, Optional
import logging

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
