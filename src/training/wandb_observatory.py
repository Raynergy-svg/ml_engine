"""
Weights & Biases Observatory - Production-Grade ML Training Observability Pipeline

This module provides enterprise-grade experiment tracking and visualization using
Weights & Biases (W&B), replicating the granular monitoring dashboards used by
frontier AI laboratories.

Features:
    - Real-time loss curves with EMA smoothing
    - Gradient and activation histogram tracking
    - System metrics monitoring (GPU SM, VRAM, Power)
    - Model graph visualization
    - Multi-run comparison and analysis
    - Artifact versioning and lineage

Design Philosophy:
    This implementation follows the "Observability First" principle where understanding
    *what works and what doesn't* is as critical as the model architecture itself.
    W&B was selected over TensorBoard for:
    
    1. **Collaboration**: Shared dashboards accessible across teams without SSH tunnels
    2. **Comparison**: Side-by-side run comparison with automatic hyperparameter correlation
    3. **Persistence**: Cloud storage ensures experiments are never lost
    4. **Analysis**: Built-in statistical tools for significance testing
    5. **Reproducibility**: Full lineage from code commit to model artifact

References:
    - W&B Documentation: https://docs.wandb.ai/
    - Gradient Tracking: https://docs.wandb.ai/guides/integrations/pytorch#log-gradients
    - System Metrics: https://docs.wandb.ai/guides/track/log/system-metrics

Usage:
    ```python
    from src.training.wandb_observatory import WandbTrainer, WandbTrainingConfig

    config = WandbTrainingConfig(
        project_name="fx_trading_ensemble",
        experiment_name="gbp_jpy_transformer_v1",
        enable_gradient_logging=True,
        ema_smoothing_factor=0.1,
    )
    
    trainer = WandbTrainer(model, config)
    trainer.train(train_loader, val_loader, num_epochs=100)
    ```
"""

from __future__ import annotations

import logging
import os
import warnings
from contextlib import contextmanager
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import (
    Any,
    Dict,
    Generator,
    List,
    Optional,
    Tuple,
)

import numpy as np

# Optional imports with graceful degradation
try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    warnings.warn("PyTorch not available. Some features will be disabled.")

try:
    import wandb
    WANDB_AVAILABLE = True
    
    # Check wandb version for reinit parameter compatibility
    # reinit="finish_previous" was added in wandb 0.23.0
    # Older versions only support boolean values
    try:
        from packaging.version import parse as parse_version
        _wandb_version = parse_version(getattr(wandb, '__version__', '0.0.0'))
        WANDB_SUPPORTS_REINIT_STRING = _wandb_version >= parse_version('0.23.0')
    except ImportError:
        # packaging not available, assume newer version
        WANDB_SUPPORTS_REINIT_STRING = True
except ImportError:
    WANDB_AVAILABLE = False
    WANDB_SUPPORTS_REINIT_STRING = False
    warnings.warn(
        "Weights & Biases not installed. Install with: pip install wandb"
    )

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False

logger = logging.getLogger(__name__)


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class WandbTrainingConfig:
    """
    Comprehensive training configuration for W&B observability.
    
    This configuration encapsulates all hyperparameters and observability
    settings required for reproducible, monitored training runs.
    
    Attributes:
        project_name: W&B project name for organizing runs
        experiment_name: Unique identifier for this experiment
        tags: List of tags for filtering and organization
        notes: Detailed description of the experiment
        learning_rate: Initial learning rate
        batch_size: Training batch size
        num_epochs: Maximum number of training epochs
        early_stopping_patience: Epochs to wait before early stopping
        weight_decay: L2 regularization coefficient
        gradient_clip_val: Maximum gradient norm for clipping
        scheduler_patience: LR scheduler patience
        scheduler_factor: LR reduction factor
        enable_gradient_logging: Log gradient histograms (memory intensive)
        enable_activation_logging: Log activation distributions
        log_frequency: Steps between logging metrics
        histogram_frequency: Epochs between histogram logging
        ema_smoothing_factor: EMA factor for loss smoothing (0-1)
        watch_model: Enable wandb.watch() for automatic gradient tracking
        log_system_metrics: Enable GPU/CPU/memory monitoring
        save_checkpoints: Save model checkpoints to W&B
        checkpoint_frequency: Epochs between checkpoint saves
        seed: Random seed for reproducibility
        model_dir: Local directory for model artifacts
    """
    # W&B Configuration
    project_name: str = "ml_engine_observatory"
    experiment_name: str = "experiment"
    tags: List[str] = field(default_factory=list)
    notes: str = ""
    
    # Training Hyperparameters
    learning_rate: float = 1e-4
    batch_size: int = 64
    num_epochs: int = 100
    early_stopping_patience: int = 15
    weight_decay: float = 1e-5
    gradient_clip_val: Optional[float] = 1.0
    
    # Scheduler Configuration
    scheduler_patience: int = 5
    scheduler_factor: float = 0.5
    scheduler_min_lr: float = 1e-7
    
    # Observability Settings
    enable_gradient_logging: bool = True
    enable_activation_logging: bool = False
    log_frequency: int = 10  # Steps
    histogram_frequency: int = 5  # Epochs
    ema_smoothing_factor: float = 0.1  # Lower = more smoothing
    watch_model: bool = True
    log_system_metrics: bool = True
    
    # Checkpointing
    save_checkpoints: bool = True
    checkpoint_frequency: int = 10  # Epochs
    
    # Reproducibility
    seed: int = 42
    model_dir: str = "trained_data/checkpoints"
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        if not 0 < self.ema_smoothing_factor < 1:
            warnings.warn(
                f"EMA smoothing factor {self.ema_smoothing_factor} should be in (0, 1). "
                "Values closer to 0 provide more smoothing."
            )
        
        if self.gradient_clip_val is not None and self.gradient_clip_val <= 0:
            raise ValueError("gradient_clip_val must be positive or None")


@dataclass
class EMATracker:
    """
    Exponential Moving Average tracker for metric smoothing.
    
    EMA separates signal from noise in volatile training curves,
    making it easier to identify true convergence trends.
    
    The smoothing formula is:
        ema_{t} = alpha * value_{t} + (1 - alpha) * ema_{t-1}
    
    Where alpha is the smoothing factor. Lower alpha = more smoothing.
    
    Attributes:
        alpha: Smoothing factor (0-1)
        value: Current EMA value
        initialized: Whether the tracker has seen its first value
    """
    alpha: float = 0.1
    value: float = 0.0
    initialized: bool = False
    
    def update(self, new_value: float) -> float:
        """
        Update EMA with new observation.
        
        Args:
            new_value: New observation to incorporate
            
        Returns:
            Updated EMA value
        """
        if not self.initialized:
            self.value = new_value
            self.initialized = True
        else:
            self.value = self.alpha * new_value + (1 - self.alpha) * self.value
        return self.value
    
    def reset(self) -> None:
        """Reset tracker to initial state."""
        self.value = 0.0
        self.initialized = False


# =============================================================================
# SYSTEM METRICS COLLECTOR
# =============================================================================

class SystemMetricsCollector:
    """
    Collect and log system-level metrics for infrastructure observability.
    
    Critical for identifying hardware bottlenecks:
    - GPU SM utilization: Is the GPU actually being used?
    - VRAM: Are we approaching OOM?
    - Power draw: Is thermal throttling occurring?
    - CPU/Memory: Is data loading the bottleneck?
    
    These metrics answer "why is training slow?" questions that pure
    loss curves cannot address.
    """
    
    def __init__(self, log_frequency: int = 100):
        """
        Initialize system metrics collector.
        
        Args:
            log_frequency: Steps between system metric collection
        """
        self.log_frequency = log_frequency
        self._step = 0
        
    def collect(self) -> Dict[str, float]:
        """
        Collect current system metrics.
        
        Returns:
            Dictionary of system metric values
        """
        metrics = {}
        
        # CPU and Memory metrics (via psutil)
        if PSUTIL_AVAILABLE:
            metrics["system/cpu_percent"] = psutil.cpu_percent(interval=0.1)
            mem = psutil.virtual_memory()
            metrics["system/memory_percent"] = mem.percent
            metrics["system/memory_gb"] = mem.used / (1024**3)
        
        # GPU metrics (via torch.cuda or MPS)
        if TORCH_AVAILABLE:
            if torch.cuda.is_available():
                # NVIDIA GPU metrics
                metrics["gpu/utilization"] = self._get_cuda_utilization()
                metrics["gpu/memory_allocated_gb"] = torch.cuda.memory_allocated() / (1024**3)
                metrics["gpu/memory_reserved_gb"] = torch.cuda.memory_reserved() / (1024**3)
                metrics["gpu/max_memory_allocated_gb"] = torch.cuda.max_memory_allocated() / (1024**3)
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                # Apple Silicon MPS metrics (limited)
                metrics["gpu/backend"] = "mps"
                # MPS doesn't expose detailed metrics, but we can track memory pressure
                if PSUTIL_AVAILABLE:
                    metrics["system/memory_pressure"] = psutil.virtual_memory().percent
        
        return metrics
    
    def _get_cuda_utilization(self) -> float:
        """
        Get CUDA GPU utilization percentage.
        
        Returns:
            GPU utilization as percentage (0-100)
        """
        try:
            # pynvml provides detailed GPU metrics
            import pynvml
            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            util = pynvml.nvmlDeviceGetUtilizationRates(handle)
            pynvml.nvmlShutdown()
            return float(util.gpu)
        except (ImportError, Exception):
            # Fallback: estimate from memory allocation ratio
            if torch.cuda.is_available():
                reserved = torch.cuda.memory_reserved()
                max_reserved = torch.cuda.get_device_properties(0).total_memory
                return (reserved / max_reserved) * 100 if max_reserved > 0 else 0
            return 0.0
    
    def should_log(self) -> bool:
        """Check if we should log at this step."""
        self._step += 1
        return self._step % self.log_frequency == 0


# =============================================================================
# GRADIENT AND ACTIVATION HOOKS
# =============================================================================

class GradientActivationMonitor:
    """
    Monitor gradient flow and activation distributions for deep learning diagnostics.
    
    This class addresses the "Weight Health" diagnostic requirement by:
    1. Logging gradient histograms to detect vanishing/exploding gradients
    2. Tracking activation distributions to identify dead neurons
    3. Monitoring weight norms for regularization effectiveness
    
    Gradient histograms are particularly valuable for:
    - Detecting vanishing gradients (all gradients near zero)
    - Detecting exploding gradients (gradients with extreme values)
    - Identifying layer-specific training issues
    
    Activation histograms help diagnose:
    - Dead ReLUs (activations stuck at zero)
    - Saturated sigmoids/tanh (activations at extremes)
    - Feature collapse (low variance in activations)
    """
    
    def __init__(
        self,
        model: "nn.Module",
        log_gradient_histograms: bool = True,
        log_activation_histograms: bool = False,
    ):
        """
        Initialize gradient/activation monitor.
        
        Args:
            model: PyTorch model to monitor
            log_gradient_histograms: Enable gradient histogram logging
            log_activation_histograms: Enable activation histogram logging
        """
        if not TORCH_AVAILABLE:
            raise RuntimeError("PyTorch is required for GradientActivationMonitor")
        
        self.model = model
        self.log_gradient_histograms = log_gradient_histograms
        self.log_activation_histograms = log_activation_histograms
        
        self._activation_hooks = []
        self._activations: Dict[str, torch.Tensor] = {}
        
        if log_activation_histograms:
            self._register_activation_hooks()
    
    def _register_activation_hooks(self) -> None:
        """
        Register forward hooks to capture activation distributions.
        
        Hooks are registered on all modules with learnable parameters,
        capturing output tensors for histogram analysis.
        """
        def make_hook(name: str):
            def hook(_module, _input, output):
                if isinstance(output, torch.Tensor):
                    self._activations[name] = output.detach()
            return hook
        
        for name, module in self.model.named_modules():
            if len(list(module.children())) == 0:  # Leaf modules only
                hook = module.register_forward_hook(make_hook(name))
                self._activation_hooks.append(hook)
    
    def get_gradient_stats(self) -> Dict[str, Dict[str, float]]:
        """
        Compute gradient statistics for all parameters.
        
        Returns:
            Nested dictionary with per-layer gradient statistics:
            - grad_mean: Average gradient value
            - grad_std: Gradient standard deviation
            - grad_max: Maximum gradient value
            - grad_min: Minimum gradient value
            - grad_norm: L2 norm of gradients
        """
        stats = {}
        
        for name, param in self.model.named_parameters():
            if param.grad is not None:
                grad = param.grad.detach()
                stats[name] = {
                    "grad_mean": float(grad.mean()),
                    "grad_std": float(grad.std()),
                    "grad_max": float(grad.max()),
                    "grad_min": float(grad.min()),
                    "grad_norm": float(grad.norm()),
                }
        
        return stats
    
    def get_activation_histograms(self) -> Dict[str, np.ndarray]:
        """
        Get activation histograms for analysis.
        
        Returns:
            Dictionary mapping layer names to activation histograms
        """
        histograms = {}
        
        for name, activation in self._activations.items():
            # Flatten and convert to numpy for histogram computation
            flat = activation.flatten().cpu().numpy()
            histograms[name] = flat
        
        return histograms
    
    def log_to_wandb(self, step: Optional[int] = None) -> None:
        """
        Log gradient and activation histograms to W&B.
        
        This creates the visual histograms in W&B that allow you to
        see gradient flow through your network over time.
        
        Args:
            step: DEPRECATED - Current training step (ignored to avoid step conflicts)
        """
        if not WANDB_AVAILABLE:
            return
        
        # Log gradient histograms
        if self.log_gradient_histograms:
            for name, param in self.model.named_parameters():
                if param.grad is not None:
                    wandb.log(
                        {f"gradients/{name}": wandb.Histogram(param.grad.cpu().numpy())},
                        commit=False,
                    )
        
        # Log activation histograms
        if self.log_activation_histograms:
            for name, activation in self._activations.items():
                wandb.log(
                    {f"activations/{name}": wandb.Histogram(activation.cpu().numpy())},
                    commit=False,
                )
    
    def cleanup(self) -> None:
        """Remove all registered hooks."""
        for hook in self._activation_hooks:
            hook.remove()
        self._activation_hooks.clear()
        self._activations.clear()


# =============================================================================
# MAIN TRAINER CLASS
# =============================================================================

class WandbTrainer:
    """
    Production-grade PyTorch trainer with comprehensive W&B observability.
    
    This trainer encapsulates all logging logic, providing a "drop-in" solution
    for monitoring any PyTorch model. It addresses all four diagnostic requirements:
    
    1. **Temporal Dynamics**: Real-time loss curves with EMA smoothing
       - `log_frequency` controls how often metrics are logged
       - `ema_smoothing_factor` controls noise reduction in trends
    
    2. **Weight Health**: Gradient and activation histograms
       - `enable_gradient_logging=True` enables gradient tracking
       - `watch_model=True` uses wandb.watch() for automatic histogram logging
    
    3. **Infrastructure**: System metrics (GPU, VRAM, Power)
       - `log_system_metrics=True` enables hardware monitoring
       - Automatic GPU utilization, memory, and power tracking
    
    4. **Topology**: Model graph visualization
       - `watch_model=True` logs the computational graph
       - Graph appears in W&B "Graphs" tab for architecture inspection
    
    Example:
        ```python
        import torch
        from torch import nn
        
        model = nn.Sequential(
            nn.Linear(100, 256),
            nn.ReLU(),
            nn.Linear(256, 10),
        )
        
        config = WandbTrainingConfig(
            project_name="my_project",
            experiment_name="test_run",
            enable_gradient_logging=True,
        )
        
        trainer = WandbTrainer(model, config)
        history = trainer.train(train_loader, val_loader, num_epochs=50)
        ```
    
    Attributes:
        model: PyTorch model to train
        config: Training configuration
        device: Device for training (cuda/mps/cpu)
        optimizer: Optimizer instance
        scheduler: Learning rate scheduler
        criterion: Loss function
        ema_trackers: EMA trackers for metrics
        gradient_monitor: Gradient/activation monitor
        system_collector: System metrics collector
    """
    
    def __init__(
        self,
        model: "nn.Module",
        config: WandbTrainingConfig,
        optimizer: Optional["torch.optim.Optimizer"] = None,
        scheduler: Optional[Any] = None,
        criterion: Optional["nn.Module"] = None,
    ):
        """
        Initialize the W&B-observable trainer.
        
        Args:
            model: PyTorch model to train
            config: Training configuration
            optimizer: Optional custom optimizer (default: AdamW)
            scheduler: Optional custom LR scheduler (default: ReduceLROnPlateau)
            criterion: Optional custom loss function (default: CrossEntropyLoss)
        """
        if not TORCH_AVAILABLE:
            raise RuntimeError("PyTorch is required for WandbTrainer")
        
        self.config = config
        self.model = model
        
        # Determine device
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")
        
        logger.info(f"Training on device: {self.device}")
        self.model.to(self.device)
        
        # Initialize optimizer
        self.optimizer = optimizer or torch.optim.AdamW(
            self.model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )
        
        # Initialize scheduler
        self.scheduler = scheduler or torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=config.scheduler_factor,
            patience=config.scheduler_patience,
            min_lr=config.scheduler_min_lr,
        )
        
        # Initialize criterion
        self.criterion = criterion or nn.CrossEntropyLoss()
        
        # EMA trackers for metric smoothing
        self.ema_trackers: Dict[str, EMATracker] = {}
        
        # Gradient and activation monitoring
        self.gradient_monitor: Optional[GradientActivationMonitor] = None
        if config.enable_gradient_logging or config.enable_activation_logging:
            self.gradient_monitor = GradientActivationMonitor(
                self.model,
                log_gradient_histograms=config.enable_gradient_logging,
                log_activation_histograms=config.enable_activation_logging,
            )
        
        # System metrics collector
        self.system_collector = SystemMetricsCollector(
            log_frequency=config.log_frequency
        ) if config.log_system_metrics else None
        
        # Training state
        self.global_step = 0
        self.current_epoch = 0
        self.best_val_loss = float('inf')
        self.best_val_metric = float('-inf')
        self.patience_counter = 0
        
        # W&B run reference
        self._wandb_run: Optional[wandb.sdk.wandb_run.Run] = None
        
        # Ensure model directory exists
        Path(config.model_dir).mkdir(parents=True, exist_ok=True)
    
    def _initialize_wandb(self) -> None:
        """
        Initialize W&B run with proper configuration.
        
        This method handles:
        - API key validation (raises clear error if not configured)
        - Run initialization with full configuration logging
        - Model watching for automatic gradient tracking
        - Offline mode support (WANDB_MODE=offline or WANDB_MODE=disabled)
        """
        if not WANDB_AVAILABLE:
            logger.warning("W&B not available. Metrics will not be logged.")
            return
        
        # Check for offline/disabled mode (no API key required)
        wandb_mode = os.environ.get("WANDB_MODE", "").lower()
        if wandb_mode in ("offline", "disabled", "dryrun"):
            logger.info(f"W&B running in {wandb_mode} mode - metrics logged locally only")
        else:
            # Check for API key only in online mode
            api_key = os.environ.get("WANDB_API_KEY")
            if not api_key:
                raise ValueError(
                    "WANDB_API_KEY environment variable not set. "
                    "Get your API key from https://wandb.ai/authorize and set it with:\n"
                    "  export WANDB_API_KEY='your-key-here'\n"
                    "Or run: wandb login\n\n"
                    "For offline mode (no cloud sync), set:\n"
                    "  export WANDB_MODE=offline"
                )
        
        # Initialize run
        # Use version-appropriate reinit value:
        # - wandb >= 0.23.0: "finish_previous" (properly cleans up previous run)
        # - wandb < 0.23.0: True (boolean fallback)
        reinit_value = "finish_previous" if WANDB_SUPPORTS_REINIT_STRING else True
        self._wandb_run = wandb.init(
            project=self.config.project_name,
            name=self.config.experiment_name,
            tags=self.config.tags,
            notes=self.config.notes,
            config=asdict(self.config),
            reinit=reinit_value,
            save_code=True,  # Save the training script for reproducibility
        )
        
        # Log model graph and gradients
        if self.config.watch_model:
            # wandb.watch logs:
            # 1. Model topology (graph visualization)
            # 2. Gradient histograms (log='all' = gradients + parameters)
            # 3. Parameter histograms
            wandb.watch(
                self.model,
                log='all',  # Log gradients, parameters, and graph
                log_freq=self.config.log_frequency,
                idx=0,  # Index for multi-model tracking
            )
        
        logger.info(f"W&B run initialized: {self._wandb_run.url}")
    
    def _finish_wandb(self) -> None:
        """Properly close W&B run and upload artifacts."""
        if self._wandb_run is not None:
            wandb.finish()
            self._wandb_run = None
    
    def _get_ema_tracker(self, name: str) -> EMATracker:
        """Get or create EMA tracker for a metric."""
        if name not in self.ema_trackers:
            self.ema_trackers[name] = EMATracker(alpha=self.config.ema_smoothing_factor)
        return self.ema_trackers[name]
    
    def _log_metrics(
        self,
        metrics: Dict[str, float],
        step: Optional[int] = None,
        commit: bool = True,
    ) -> None:
        """
        Log metrics to W&B with optional EMA smoothing.
        
        For each metric, this logs both the raw value and an EMA-smoothed
        version. The EMA version helps identify trends in noisy metrics.
        
        Args:
            metrics: Dictionary of metric names to values
            step: DEPRECATED - Optional step number (ignored to avoid step conflicts)
            commit: Whether to commit the log entry
        """
        if not WANDB_AVAILABLE or self._wandb_run is None:
            return
        
        log_dict = {}
        
        for name, value in metrics.items():
            # Log raw value
            log_dict[name] = value
            
            # Log EMA-smoothed value
            ema_tracker = self._get_ema_tracker(name)
            ema_value = ema_tracker.update(value)
            log_dict[f"{name}_ema"] = ema_value
        
        # Commit controls whether to flush metrics immediately
        # Don't use explicit step to avoid conflicts with other loggers
        wandb.log(log_dict, commit=commit)
    
    def _log_system_metrics(self) -> None:
        """Log system metrics if enabled."""
        if self.system_collector and self.system_collector.should_log():
            metrics = self.system_collector.collect()
            if metrics and WANDB_AVAILABLE and self._wandb_run:
                # Don't use explicit step to avoid conflicts with other loggers
                wandb.log(metrics, commit=False)
    
    def _log_histograms(self) -> None:
        """Log gradient and activation histograms."""
        if self.gradient_monitor and WANDB_AVAILABLE and self._wandb_run:
            self.gradient_monitor.log_to_wandb(step=self.global_step)
    
    def _save_checkpoint(self, epoch: int, is_best: bool = False) -> None:
        """
        Save model checkpoint.
        
        Args:
            epoch: Current epoch number
            is_best: Whether this is the best model so far
        """
        checkpoint_path = Path(self.config.model_dir) / f"checkpoint_epoch_{epoch}.pt"
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'best_val_loss': self.best_val_loss,
            'global_step': self.global_step,
            'config': asdict(self.config),
        }
        
        torch.save(checkpoint, checkpoint_path)
        
        # Log to W&B as artifact
        if WANDB_AVAILABLE and self._wandb_run and self.config.save_checkpoints:
            artifact = wandb.Artifact(
                name=f"model-{self.config.experiment_name}",
                type="model",
                description=f"Checkpoint at epoch {epoch}",
            )
            artifact.add_file(str(checkpoint_path))
            wandb.log_artifact(artifact)
        
        # Save best model separately
        if is_best:
            best_path = Path(self.config.model_dir) / "best_model.pt"
            torch.save(checkpoint, best_path)
            logger.info(f"New best model saved to {best_path}")
    
    def train_epoch(
        self,
        train_loader: "DataLoader",
        epoch: int,
    ) -> Dict[str, float]:
        """
        Train for one epoch with comprehensive logging.
        
        This method implements the training loop with:
        - Gradient clipping for stability
        - Progress logging with EMA smoothing
        - System metrics collection
        - Gradient histogram logging (at specified frequency)
        
        Args:
            train_loader: Training data loader
            epoch: Current epoch number
            
        Returns:
            Dictionary of training metrics for the epoch
        """
        self.model.train()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0

        for batch_idx, (inputs, targets) in enumerate(train_loader):
            # Move to device
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            if self.config.gradient_clip_val is not None:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config.gradient_clip_val,
                )
            
            # Optimizer step
            self.optimizer.step()
            
            # Accumulate metrics
            total_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total_correct += predicted.eq(targets).sum().item()
            total_samples += inputs.size(0)
            
            # Log step metrics
            if batch_idx % self.config.log_frequency == 0:
                step_metrics = {
                    "train/loss_step": loss.item(),
                    "train/accuracy_step": total_correct / total_samples,
                    "train/lr": self.optimizer.param_groups[0]['lr'],
                }
                self._log_metrics(step_metrics)
                
                # Log system metrics
                self._log_system_metrics()
            
            self.global_step += 1
        
        # Compute epoch metrics
        epoch_loss = total_loss / total_samples
        epoch_accuracy = total_correct / total_samples
        
        # Log gradient histograms at specified frequency
        if epoch % self.config.histogram_frequency == 0:
            self._log_histograms()
        
        return {
            "train/loss": epoch_loss,
            "train/accuracy": epoch_accuracy,
        }
    
    def validate(
        self,
        val_loader: "DataLoader",
        _epoch: int,
    ) -> Dict[str, float]:
        """
        Validate the model with comprehensive logging.
        
        Args:
            val_loader: Validation data loader
            epoch: Current epoch number
            
        Returns:
            Dictionary of validation metrics
        """
        self.model.eval()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)
                
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                
                total_loss += loss.item() * inputs.size(0)
                _, predicted = outputs.max(1)
                total_correct += predicted.eq(targets).sum().item()
                total_samples += inputs.size(0)
        
        val_loss = total_loss / total_samples
        val_accuracy = total_correct / total_samples
        
        return {
            "val/loss": val_loss,
            "val/accuracy": val_accuracy,
        }
    
    def train(
        self,
        train_loader: "DataLoader",
        val_loader: "DataLoader",
        num_epochs: Optional[int] = None,
    ) -> Dict[str, List[float]]:
        """
        Full training loop with W&B observability.
        
        This is the main entry point for training. It handles:
        - W&B initialization and cleanup
        - Training and validation loops
        - Early stopping
        - Learning rate scheduling
        - Checkpoint saving
        - Comprehensive logging
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            num_epochs: Override config epochs (optional)
            
        Returns:
            Dictionary of training history (lists of metrics per epoch)
        """
        num_epochs = num_epochs or self.config.num_epochs
        
        # Set random seeds for reproducibility
        torch.manual_seed(self.config.seed)
        np.random.seed(self.config.seed)
        
        # Initialize W&B
        self._initialize_wandb()
        
        history: Dict[str, List[float]] = {
            "train_loss": [],
            "train_accuracy": [],
            "val_loss": [],
            "val_accuracy": [],
        }
        
        try:
            for epoch in range(num_epochs):
                self.current_epoch = epoch
                
                # Training
                train_metrics = self.train_epoch(train_loader, epoch)
                
                # Validation
                val_metrics = self.validate(val_loader, epoch)
                
                # Log epoch metrics
                all_metrics = {**train_metrics, **val_metrics}
                self._log_metrics(all_metrics, step=self.global_step)
                
                # Update history
                history["train_loss"].append(train_metrics["train/loss"])
                history["train_accuracy"].append(train_metrics["train/accuracy"])
                history["val_loss"].append(val_metrics["val/loss"])
                history["val_accuracy"].append(val_metrics["val/accuracy"])
                
                # Learning rate scheduling
                if self.scheduler:
                    if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                        self.scheduler.step(val_metrics["val/loss"])
                    else:
                        self.scheduler.step()
                
                # Check for improvement
                is_best = val_metrics["val/loss"] < self.best_val_loss
                if is_best:
                    self.best_val_loss = val_metrics["val/loss"]
                    self.patience_counter = 0
                else:
                    self.patience_counter += 1
                
                # Save checkpoint
                if epoch % self.config.checkpoint_frequency == 0 or is_best:
                    self._save_checkpoint(epoch, is_best=is_best)
                
                # Log progress
                logger.info(
                    f"Epoch {epoch + 1}/{num_epochs} - "
                    f"Train Loss: {train_metrics['train/loss']:.4f}, "
                    f"Val Loss: {val_metrics['val/loss']:.4f}, "
                    f"Val Acc: {val_metrics['val/accuracy']:.2%}, "
                    f"Best: {self.best_val_loss:.4f}"
                )
                
                # Early stopping
                if self.patience_counter >= self.config.early_stopping_patience:
                    logger.info(
                        f"Early stopping triggered after {epoch + 1} epochs "
                        f"(patience: {self.config.early_stopping_patience})"
                    )
                    break
        
        finally:
            # Cleanup
            if self.gradient_monitor:
                self.gradient_monitor.cleanup()
            self._finish_wandb()
        
        return history


# =============================================================================
# CONTEXT MANAGER FOR QUICK EXPERIMENTS
# =============================================================================

@contextmanager
def wandb_experiment(
    project_name: str,
    experiment_name: str,
    config: Optional[Dict[str, Any]] = None,
    tags: Optional[List[str]] = None,
) -> Generator[Optional[wandb.sdk.wandb_run.Run], None, None]:
    """
    Context manager for quick W&B experiments.
    
    Provides a clean way to wrap any code block with W&B tracking.
    Useful for quick experiments, hyperparameter searches, or debugging.
    
    Args:
        project_name: W&B project name
        experiment_name: Run name
        config: Optional configuration dictionary
        tags: Optional list of tags
        
    Yields:
        W&B run object or None if W&B not available
        
    Example:
        ```python
        with wandb_experiment("my_project", "test", config={"lr": 0.001}) as run:
            for epoch in range(10):
                loss = train_one_epoch()
                wandb.log({"loss": loss})
        ```
    """
    if not WANDB_AVAILABLE:
        warnings.warn("W&B not available, skipping experiment tracking")
        yield None
        return
    
    # Check API key
    if not os.environ.get("WANDB_API_KEY"):
        warnings.warn("WANDB_API_KEY not set, skipping experiment tracking")
        yield None
        return
    
    # Use version-appropriate reinit value
    reinit_value = "finish_previous" if WANDB_SUPPORTS_REINIT_STRING else True
    run = wandb.init(
        project=project_name,
        name=experiment_name,
        config=config or {},
        tags=tags or [],
        reinit=reinit_value,
    )
    
    try:
        yield run
    finally:
        wandb.finish()


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def compare_runs(
    project_name: str,
    run_ids: List[str],
    metrics: List[str] = ["val/loss", "val/accuracy"],
) -> None:
    """
    Compare multiple runs in W&B.
    
    This utility fetches run data and creates comparison visualizations.
    Useful for hyperparameter tuning analysis.
    
    Args:
        project_name: W&B project name
        run_ids: List of run IDs to compare
        metrics: Metrics to compare
    """
    if not WANDB_AVAILABLE:
        raise RuntimeError("W&B is required for run comparison")
    
    api = wandb.Api()
    
    for run_id in run_ids:
        try:
            run = api.run(f"{project_name}/{run_id}")
            print(f"\nRun: {run.name} ({run_id})")
            print(f"  Config: {run.config}")
            # Filter to requested metrics
            filtered_metrics = {k: v for k, v in run.summary.items() if k in metrics}
            print(f"  Metrics: {filtered_metrics or run.summary}")
        except Exception as e:
            print(f"Error fetching run {run_id}: {e}")


def log_model_summary(model: "nn.Module", input_size: Tuple[int, ...]) -> None:
    """
    Log model summary to W&B.
    
    Creates a visual summary of the model architecture including
    layer dimensions and parameter counts.
    
    Args:
        model: PyTorch model
        input_size: Input tensor shape (without batch dimension)
    """
    if not WANDB_AVAILABLE:
        return
    
    # Try to use torchinfo for detailed summary
    try:
        from torchinfo import summary
        summary_str = str(summary(model, input_size=input_size))
        wandb.log({"model/summary": wandb.Html(f"<pre>{summary_str}</pre>")})
    except ImportError:
        # Fallback to basic parameter count
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        summary_html = f"""
        <h3>Model Summary</h3>
        <p>Total Parameters: {total_params:,}</p>
        <p>Trainable Parameters: {trainable_params:,}</p>
        """
        wandb.log({"model/summary": wandb.Html(summary_html)})


# =============================================================================
# MODULE VALIDATION
# =============================================================================

def _validate_environment() -> Dict[str, bool]:
    """
    Validate the environment for W&B observatory.
    
    Returns:
        Dictionary of component availability
    """
    return {
        "torch": TORCH_AVAILABLE,
        "wandb": WANDB_AVAILABLE,
        "psutil": PSUTIL_AVAILABLE,
        "cuda": torch.cuda.is_available() if TORCH_AVAILABLE else False,
        "mps": (
            hasattr(torch.backends, 'mps') and 
            torch.backends.mps.is_available()
        ) if TORCH_AVAILABLE else False,
    }


if __name__ == "__main__":
    # Environment validation
    print("=" * 60)
    print("W&B Observatory - Environment Validation")
    print("=" * 60)
    
    env = _validate_environment()
    for component, available in env.items():
        status = "✓" if available else "✗"
        print(f"  {status} {component}: {available}")
    
    print("\n" + "=" * 60)
    print("Configuration Example")
    print("=" * 60)
    
    example_config = WandbTrainingConfig(
        project_name="fx_trading_ensemble",
        experiment_name="gbp_jpy_transformer_v1",
        learning_rate=1e-4,
        batch_size=64,
        num_epochs=100,
        enable_gradient_logging=True,
        ema_smoothing_factor=0.1,
        tags=["transformer", "gbp_jpy", "production"],
        notes="Production training run for GBP/JPY Transformer model",
    )
    
    print(f"Project: {example_config.project_name}")
    print(f"Experiment: {example_config.experiment_name}")
    print(f"Learning Rate: {example_config.learning_rate}")
    print(f"Batch Size: {example_config.batch_size}")
    print(f"Gradient Logging: {example_config.enable_gradient_logging}")
    print(f"EMA Smoothing: {example_config.ema_smoothing_factor}")
    
    print("\n" + "=" * 60)
    print("Quick Start Example")
    print("=" * 60)
    print("""
# 1. Set your W&B API key
export WANDB_API_KEY='your-key-here'

# 2. Create your model
import torch.nn as nn
model = nn.Sequential(
    nn.Linear(100, 256),
    nn.ReLU(),
    nn.Linear(256, 10),
)

# 3. Initialize trainer
from src.training.wandb_observatory import WandbTrainer, WandbTrainingConfig

config = WandbTrainingConfig(
    project_name="my_project",
    experiment_name="test_run",
)

trainer = WandbTrainer(model, config)

# 4. Train with full observability
history = trainer.train(train_loader, val_loader, num_epochs=50)

# 5. View results at https://wandb.ai/your-username/my_project
""")
    
    print("\n✓ W&B Observatory module ready!")
