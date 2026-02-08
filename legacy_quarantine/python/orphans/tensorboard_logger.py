"""
TensorBoard Integration for PyTorch Training.
Adds visual monitoring to existing PyTorch training pipelines.

Usage:
    from tensorboard_logger import TensorBoardLogger
    
    logger = TensorBoardLogger('trained_data/tensorboard/pytorch_run')
    
    for epoch in range(epochs):
        # Training...
        logger.log_scalar('Loss/train', train_loss, epoch)
        logger.log_scalar('Loss/val', val_loss, epoch)
        logger.log_histogram('weights/fc1', model.fc1.weight, epoch)
    
    # Launch TensorBoard:
    # tensorboard --logdir=trained_data/tensorboard
"""

raise SystemExit(
    "Retired: TensorBoard logger disabled (legacy training). Buddy training runs only via main.py. "
    "Use: python main.py train-buddy"
)

import datetime  # noqa: E402
from pathlib import Path  # noqa: E402
from typing import Dict, Any, Optional, Union  # noqa: E402
import numpy as np  # noqa: E402

# PyTorch retired (TensorFlow-only migration)
# import torch
# import torch.nn as nn
# from torch.utils.tensorboard import SummaryWriter

# Stubs for static analysis (code is unreachable due to SystemExit above)
SummaryWriter = None  # type: ignore
torch = None  # type: ignore
nn = None  # type: ignore


class TensorBoardLogger:
    """
    TensorBoard logger for PyTorch training.
    Provides easy-to-use interface for logging metrics, histograms, and graphs.
    """
    
    def __init__(
        self,
        log_dir: Optional[str] = None,
        experiment_name: Optional[str] = None,
        flush_secs: int = 30,
    ):
        """
        Initialize TensorBoard logger.
        
        Args:
            log_dir: Directory for TensorBoard logs
            experiment_name: Name for this experiment
            flush_secs: How often to flush to disk
        """
        if log_dir is None:
            timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
            experiment_name = experiment_name or "pytorch"
            log_dir = f"trained_data/tensorboard/{experiment_name}_{timestamp}"
        
        self.log_dir = log_dir
        Path(log_dir).mkdir(parents=True, exist_ok=True)
        
        self.writer = SummaryWriter(log_dir=log_dir, flush_secs=flush_secs)
        self.step = 0
        
        print(f"📊 TensorBoard logging to: {log_dir}")
        print(f"   View: tensorboard --logdir={log_dir}")
    
    def log_scalar(
        self,
        tag: str,
        value: float,
        step: Optional[int] = None,
    ):
        """Log a scalar value."""
        step = step if step is not None else self.step
        self.writer.add_scalar(tag, value, step)
    
    def log_scalars(
        self,
        main_tag: str,
        tag_scalar_dict: Dict[str, float],
        step: Optional[int] = None,
    ):
        """Log multiple scalars under one tag."""
        step = step if step is not None else self.step
        self.writer.add_scalars(main_tag, tag_scalar_dict, step)
    
    def log_histogram(
        self,
        tag: str,
        values: Union[torch.Tensor, np.ndarray],
        step: Optional[int] = None,
    ):
        """Log a histogram of values."""
        step = step if step is not None else self.step
        if isinstance(values, torch.Tensor):
            values = values.detach().cpu().numpy()
        self.writer.add_histogram(tag, values, step)
    
    def log_model_weights(
        self,
        model: nn.Module,
        step: Optional[int] = None,
    ):
        """Log all model weight histograms."""
        step = step if step is not None else self.step
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.writer.add_histogram(f"weights/{name}", param.detach().cpu().numpy(), step)
                if param.grad is not None:
                    self.writer.add_histogram(f"gradients/{name}", param.grad.detach().cpu().numpy(), step)
    
    def log_model_graph(
        self,
        model: nn.Module,
        input_tensor: torch.Tensor,
    ):
        """Log model architecture graph."""
        self.writer.add_graph(model, input_tensor)
    
    def log_learning_rate(
        self,
        lr: float,
        step: Optional[int] = None,
    ):
        """Log learning rate."""
        step = step if step is not None else self.step
        self.writer.add_scalar("Learning_Rate", lr, step)
    
    def log_epoch_metrics(
        self,
        epoch: int,
        train_loss: float,
        val_loss: Optional[float] = None,
        lr: Optional[float] = None,
        additional_metrics: Optional[Dict[str, float]] = None,
    ):
        """
        Log all metrics for an epoch.
        
        Args:
            epoch: Current epoch number
            train_loss: Training loss
            val_loss: Validation loss (optional)
            lr: Learning rate (optional)
            additional_metrics: Dict of additional metrics to log
        """
        self.writer.add_scalar("Loss/train", train_loss, epoch)
        
        if val_loss is not None:
            self.writer.add_scalar("Loss/val", val_loss, epoch)
            self.writer.add_scalars("Loss/comparison", {
                "train": train_loss,
                "val": val_loss,
            }, epoch)
        
        if lr is not None:
            self.writer.add_scalar("Learning_Rate", lr, epoch)
        
        if additional_metrics:
            for name, value in additional_metrics.items():
                self.writer.add_scalar(f"Metrics/{name}", value, epoch)
    
    def log_trading_metrics(
        self,
        epoch: int,
        predictions: np.ndarray,
        targets: np.ndarray,
    ):
        """Log trading-specific metrics."""
        # Directional accuracy
        pred_direction = np.sign(predictions)
        true_direction = np.sign(targets)
        directional_accuracy = np.mean(pred_direction == true_direction)
        
        # Mean absolute error
        mae = np.mean(np.abs(predictions - targets))
        
        # R-squared (simplified)
        ss_res = np.sum((targets - predictions) ** 2)
        ss_tot = np.sum((targets - np.mean(targets)) ** 2)
        r2 = 1 - (ss_res / (ss_tot + 1e-8))
        
        self.writer.add_scalar("Trading/directional_accuracy", directional_accuracy, epoch)
        self.writer.add_scalar("Trading/mae", mae, epoch)
        self.writer.add_scalar("Trading/r2", r2, epoch)
        
        # Prediction distribution
        self.writer.add_histogram("Trading/predictions", predictions, epoch)
        self.writer.add_histogram("Trading/targets", targets, epoch)
        self.writer.add_histogram("Trading/errors", predictions - targets, epoch)
    
    def log_text(
        self,
        tag: str,
        text: str,
        step: Optional[int] = None,
    ):
        """Log text (markdown supported)."""
        step = step if step is not None else self.step
        self.writer.add_text(tag, text, step)
    
    def log_hyperparameters(
        self,
        hparams: Dict[str, Any],
        metrics: Optional[Dict[str, float]] = None,
    ):
        """Log hyperparameters with optional metrics."""
        # Convert non-primitive types to strings
        hparams_clean = {}
        for k, v in hparams.items():
            if isinstance(v, (int, float, str, bool)):
                hparams_clean[k] = v
            else:
                hparams_clean[k] = str(v)
        
        metrics = metrics or {}
        self.writer.add_hparams(hparams_clean, metrics)
    
    def increment_step(self):
        """Increment global step counter."""
        self.step += 1
    
    def flush(self):
        """Flush all pending events to disk."""
        self.writer.flush()
    
    def close(self):
        """Close the logger."""
        self.writer.close()
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


class TrainingCallback:
    """
    Callback for integrating TensorBoard with PyTorch training loops.
    
    Usage:
        callback = TrainingCallback(log_dir='runs/experiment1')
        
        for epoch in range(epochs):
            train_loss = train_one_epoch(...)
            val_loss = validate(...)
            
            callback.on_epoch_end(
                epoch, model, train_loss, val_loss, optimizer
            )
    """
    
    def __init__(
        self,
        log_dir: Optional[str] = None,
        experiment_name: Optional[str] = None,
        log_weights: bool = True,
        log_gradients: bool = True,
        log_weights_freq: int = 5,
    ):
        """
        Initialize callback.
        
        Args:
            log_dir: TensorBoard log directory
            experiment_name: Name for the experiment
            log_weights: Whether to log weight histograms
            log_gradients: Whether to log gradient histograms
            log_weights_freq: Log weights every N epochs
        """
        self.logger = TensorBoardLogger(log_dir, experiment_name)
        self.log_weights = log_weights
        self.log_gradients = log_gradients
        self.log_weights_freq = log_weights_freq
        self.best_val_loss = float('inf')
    
    def on_train_begin(
        self,
        model: nn.Module,
        config: Dict[str, Any],
        sample_input: Optional[torch.Tensor] = None,
    ):
        """Called at the start of training."""
        # Log model graph
        if sample_input is not None:
            try:
                self.logger.log_model_graph(model, sample_input)
            except Exception as e:
                print(f"Warning: Could not log model graph: {e}")
        
        # Log hyperparameters
        self.logger.log_hyperparameters(config)
        
        # Log model summary as text
        model_summary = self._get_model_summary(model)
        self.logger.log_text("Model/summary", model_summary, 0)
    
    def on_epoch_end(
        self,
        epoch: int,
        model: nn.Module,
        train_loss: float,
        val_loss: Optional[float] = None,
        optimizer: Optional[torch.optim.Optimizer] = None,
        predictions: Optional[np.ndarray] = None,
        targets: Optional[np.ndarray] = None,
        additional_metrics: Optional[Dict[str, float]] = None,
    ):
        """
        Called at the end of each epoch.
        
        Args:
            epoch: Current epoch
            model: PyTorch model
            train_loss: Training loss
            val_loss: Validation loss
            optimizer: Optimizer (for learning rate logging)
            predictions: Model predictions (for trading metrics)
            targets: Ground truth targets
            additional_metrics: Any additional metrics to log
        """
        # Get learning rate
        lr = None
        if optimizer is not None:
            lr = optimizer.param_groups[0]['lr']
        
        # Log epoch metrics
        self.logger.log_epoch_metrics(
            epoch, train_loss, val_loss, lr, additional_metrics
        )
        
        # Log weights periodically
        if self.log_weights and epoch % self.log_weights_freq == 0:
            self.logger.log_model_weights(model, epoch)
        
        # Log trading metrics if predictions provided
        if predictions is not None and targets is not None:
            self.logger.log_trading_metrics(epoch, predictions, targets)
        
        # Track best model
        if val_loss is not None and val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            self.logger.log_text(
                "Training/best_epoch",
                f"Best model at epoch {epoch} with val_loss={val_loss:.6f}",
                epoch
            )
        
        self.logger.flush()
    
    def on_train_end(
        self,
        final_metrics: Optional[Dict[str, float]] = None,
    ):
        """Called at the end of training."""
        if final_metrics:
            self.logger.log_hyperparameters(
                {"training_complete": True},
                final_metrics
            )
        
        self.logger.close()
        print(f"\n✓ TensorBoard logs saved to: {self.logger.log_dir}")
        print(f"   View: tensorboard --logdir={self.logger.log_dir}")
    
    def _get_model_summary(self, model: nn.Module) -> str:
        """Generate model summary as markdown."""
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        summary = f"""
## Model Summary

**Model Type:** `{model.__class__.__name__}`

**Parameters:**
- Total: {total_params:,}
- Trainable: {trainable_params:,}
- Non-trainable: {total_params - trainable_params:,}

**Architecture:**
```
{model}
```
"""
        return summary


def integrate_tensorboard_with_engine(engine, log_dir: Optional[str] = None):
    """
    Add TensorBoard logging to an existing EnhancedMLEngine.
    
    Usage:
        from ml_engine_enhanced import EnhancedMLEngine
        from tensorboard_logger import integrate_tensorboard_with_engine
        
        engine = EnhancedMLEngine(config)
        callback = integrate_tensorboard_with_engine(engine)
        
        # Training will now log to TensorBoard
        engine.train(X_train, y_train, X_val, y_val)
    """
    callback = TrainingCallback(
        log_dir=log_dir,
        experiment_name=engine.config.get('model', {}).get('type', 'pytorch'),
    )
    
    # Store reference for cleanup
    engine._tensorboard_callback = callback
    
    # Log initial setup
    sample_input = torch.randn(1, 60, engine.config.get('model', {}).get('input_size', 104))
    sample_input = sample_input.to(engine.device)
    
    callback.on_train_begin(
        model=engine.model,
        config=engine.config,
        sample_input=sample_input,
    )
    
    return callback


if __name__ == "__main__":
    print("TensorBoard Logger for PyTorch")
    print("=" * 60)
    
    # Example usage
    print("""
Usage Example:
--------------

from tensorboard_logger import TensorBoardLogger, TrainingCallback

# Option 1: Direct logging
logger = TensorBoardLogger('trained_data/tensorboard/my_experiment')

for epoch in range(100):
    train_loss = train_one_epoch(...)
    val_loss = validate(...)
    
    logger.log_epoch_metrics(epoch, train_loss, val_loss)
    logger.log_model_weights(model, epoch)

logger.close()

# Option 2: Callback-based
callback = TrainingCallback(log_dir='trained_data/tensorboard/my_experiment')

callback.on_train_begin(model, config)

for epoch in range(100):
    train_loss = train_one_epoch(...)
    val_loss = validate(...)
    
    callback.on_epoch_end(epoch, model, train_loss, val_loss, optimizer)

callback.on_train_end()

# Then launch TensorBoard:
# tensorboard --logdir=trained_data/tensorboard
""")
    
    # Demo with synthetic data
    print("\nDemo: Creating synthetic training run...")
    
    rng = np.random.default_rng(seed=42)
    with TensorBoardLogger("trained_data/tensorboard/demo") as logger:
        for epoch in range(10):
            # Simulate training
            train_loss = 1.0 / (epoch + 1) + rng.random() * 0.1
            val_loss = 1.0 / (epoch + 1) + rng.random() * 0.15
            lr = 0.001 * (0.9 ** epoch)
            
            logger.log_epoch_metrics(
                epoch, train_loss, val_loss, lr,
                additional_metrics={
                    'directional_accuracy': 0.5 + epoch * 0.03,
                    'r2_score': 0.3 + epoch * 0.05,
                }
            )
            
            print(f"Epoch {epoch+1}/10 - train_loss: {train_loss:.4f}, val_loss: {val_loss:.4f}")
    
    print("\n✓ Demo complete!")
    print("📊 View results: tensorboard --logdir=trained_data/tensorboard/demo")
# — Raynergy-svg —
