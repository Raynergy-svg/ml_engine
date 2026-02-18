"""
Professional Enterprise-Grade Training Display System

Provides clean, structured, real-time training output with:
- Progress bars and ETA
- Live metric updates
- Color-coded status indicators
- Summary tables
- Session headers and footers

Best Practices Applied:
- Rich library for consistent terminal formatting
- Keras callbacks for training integration
- Structured logging for debugging
- Clean separation of display logic
"""

from __future__ import annotations

import time
import logging
from datetime import datetime, timedelta
from typing import Dict, Any, List
from dataclasses import dataclass, field

from .premium_output import PremiumConsole, StatusGlyphs

try:
    from rich.console import Console
    from rich.live import Live  # noqa: F401
    from rich.layout import Layout  # noqa: F401
    from rich.table import Table
    from rich.progress import Progress, BarColumn, TextColumn, TimeRemainingColumn, SpinnerColumn, MofNCompleteColumn  # noqa: F401
    from rich.text import Text
    from rich.style import Style  # noqa: F401
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False

logger = logging.getLogger(__name__)


# =============================================================================
# LOGGING CONTROL
# =============================================================================

class SuppressLogging:
    """Context manager to suppress logging from specific loggers during training.

    Usage:
        with SuppressLogging(['modular_trainers', 'tensorflow']):
            # Train model - logs will be suppressed
            trainer.train(...)
    """

    def __init__(self, logger_names: List[str] = None, level: int = logging.WARNING):
        """
        Args:
            logger_names: List of logger names to suppress. Default suppresses common ML loggers.
            level: Minimum log level to show. Default WARNING hides INFO and DEBUG.
        """
        self.logger_names = logger_names or [
            'modular_trainers',
            'tensorflow',
            'absl',
            'h5py',
        ]
        self.level = level
        self._original_levels = {}

    def __enter__(self):
        for name in self.logger_names:
            lgr = logging.getLogger(name)
            self._original_levels[name] = lgr.level
            lgr.setLevel(self.level)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        for name, original_level in self._original_levels.items():
            logging.getLogger(name).setLevel(original_level)
        return False


# =============================================================================
# CONSTANTS
# =============================================================================

# Thresholds for color coding
ACCURACY_EXCELLENT = 0.70
ACCURACY_GOOD = 0.55
ACCURACY_FAIR = 0.50

LOSS_EXCELLENT = 0.3
LOSS_GOOD = 0.5
LOSS_FAIR = 0.7


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class TrainingConfig:
    """Training session configuration for display."""
    model_name: str = "Transformer"
    task: str = "Direction Prediction"
    instrument: str = ""
    granularity: str = ""
    total_epochs: int = 100
    batch_size: int = 64
    learning_rate: float = 0.001
    n_train_samples: int = 0
    n_val_samples: int = 0
    n_features: int = 0
    seq_len: int = 60
    use_madl: bool = False
    warm_start: bool = False
    extra_info: Dict[str, Any] = field(default_factory=dict)


@dataclass
class EpochMetrics:
    """Metrics for a single epoch."""
    epoch: int
    loss: float
    accuracy: float
    val_loss: float
    val_accuracy: float
    lr: float = 0.001
    duration_sec: float = 0.0

    @property
    def loss_improved(self) -> bool:
        return getattr(self, '_loss_improved', False)

    @loss_improved.setter
    def loss_improved(self, value: bool):
        self._loss_improved = value

    @property
    def acc_improved(self) -> bool:
        return getattr(self, '_acc_improved', False)

    @acc_improved.setter
    def acc_improved(self, value: bool):
        self._acc_improved = value


# =============================================================================
# TRAINING DISPLAY CLASS
# =============================================================================

class TrainingDisplay:
    """
    Enterprise-grade training display with Rich formatting.

    Features:
    - Session header with configuration
    - Real-time epoch progress
    - Color-coded metrics
    - Live updates
    - Summary tables
    """

    def __init__(self, config: TrainingConfig):
        self.config = config
        self._rich_console = Console() if RICH_AVAILABLE else None
        self.console = PremiumConsole(self._rich_console)
        self.start_time = None
        self.epoch_history: List[EpochMetrics] = []
        self.best_val_loss = float('inf')
        self.best_val_acc = 0.0
        self.best_epoch = 0
        self.current_lr = config.learning_rate
        self._live = None

    def print_header(self):
        """Print training session header."""
        if not RICH_AVAILABLE:
            self._print_header_plain()
            return

        self.start_time = datetime.now()

        # Print main header with PremiumConsole
        self.console.blank()
        self.console.section_header(
            f"{StatusGlyphs.STEP} Training Session Started",
            subtitle=f"Started at {self.start_time.strftime('%Y-%m-%d %H:%M:%S')}",
            color="cyan"
        )

        # Print data section
        self.console.blank()
        self._rich_console.print("  [bold]DATA[/bold]", style="cyan")
        self.console.key_value("Model", self.config.model_name)
        self.console.key_value("Task", self.config.task)
        if self.config.instrument:
            self.console.key_value("Instrument", f"{self.config.instrument} ({self.config.granularity})")
        self.console.key_value("Training", f"{self.config.n_train_samples:,} samples")
        self.console.key_value("Validation", f"{self.config.n_val_samples:,} samples")
        self.console.key_value("Features", f"{self.config.n_features} × {self.config.seq_len} timesteps")

        # Print config section
        self.console.blank()
        self._rich_console.print("  [bold]CONFIG[/bold]", style="cyan")
        self.console.key_value("Epochs", str(self.config.total_epochs))
        self.console.key_value("Batch Size", str(self.config.batch_size))
        self.console.key_value("Learning Rate", f"{self.config.learning_rate:.1e}")
        self.console.key_value("Loss", "MADL" if self.config.use_madl else "BCE")
        if self.config.warm_start:
            self.console.key_value("Mode", "Warm Start")

        self.console.blank()
        self.console.divider(100)
        self.console.blank()

        # Print column headers for epoch display
        self._print_epoch_header()

    def _print_header_plain(self):
        """Plain text header for non-Rich environments."""
        self.start_time = datetime.now()
        print("\n" + "=" * 80)
        print(f"Training Session: {self.config.model_name} - {self.config.task}")
        print(f"Started: {self.start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print("-" * 80)
        print(f"Data: {self.config.n_train_samples:,} train / {self.config.n_val_samples:,} val samples")
        print(f"Features: {self.config.n_features} × {self.config.seq_len} timesteps")
        print(f"Config: epochs={self.config.total_epochs}, batch={self.config.batch_size}, lr={self.config.learning_rate:.1e}")
        print("=" * 80 + "\n")

    def _print_epoch_header(self):
        """Print column headers for epoch rows."""
        if not RICH_AVAILABLE:
            print(f"{'Epoch':>8} {'Loss':>10} {'Acc':>8} {'Val Loss':>10} {'Val Acc':>8} {'LR':>10} {'Time':>8} {'Status':<12}")
            print("-" * 80)
            return

        header = Text()
        header.append(f"{'Epoch':>8}", style="bold cyan")
        header.append("  ")  # Spacing instead of │
        header.append(f"{'Loss':>9}", style="bold")
        header.append("  ")
        header.append(f"{'Acc':>7}", style="bold")
        header.append("  ")  # Spacing instead of │
        header.append(f"{'Val Loss':>9}", style="bold")
        header.append("  ")
        header.append(f"{'Val Acc':>7}", style="bold")
        header.append("  ")  # Spacing instead of │
        header.append(f"{'LR':>9}", style="bold")
        header.append("  ")
        header.append(f"{'Time':>7}", style="bold")
        header.append("  ")  # Spacing instead of │
        header.append("Status", style="bold")

        self._rich_console.print(header)
        self.console.divider(100)

    def update_epoch(self, metrics: EpochMetrics):
        """Update display with epoch metrics."""
        # Track improvements
        metrics.loss_improved = metrics.val_loss < self.best_val_loss
        metrics.acc_improved = metrics.val_accuracy > self.best_val_acc

        if metrics.val_loss < self.best_val_loss:
            self.best_val_loss = metrics.val_loss
        if metrics.val_accuracy > self.best_val_acc:
            self.best_val_acc = metrics.val_accuracy
            self.best_epoch = metrics.epoch

        self.epoch_history.append(metrics)
        self.current_lr = metrics.lr

        if RICH_AVAILABLE:
            self._print_epoch_rich(metrics)
        else:
            self._print_epoch_plain(metrics)

    def _get_acc_style(self, acc: float) -> str:
        """Get color style for accuracy value."""
        if acc >= ACCURACY_EXCELLENT:
            return "bold green"
        elif acc >= ACCURACY_GOOD:
            return "green"
        elif acc >= ACCURACY_FAIR:
            return "yellow"
        else:
            return "red"

    def _get_loss_style(self, loss: float) -> str:
        """Get color style for loss value."""
        if loss <= LOSS_EXCELLENT:
            return "bold green"
        elif loss <= LOSS_GOOD:
            return "green"
        elif loss <= LOSS_FAIR:
            return "yellow"
        else:
            return "red"

    def _print_epoch_rich(self, m: EpochMetrics):
        """Print epoch row with Rich formatting."""
        row = Text()

        # Epoch number
        m.epoch / self.config.total_epochs * 100
        row.append(f"{m.epoch:>4}/{self.config.total_epochs:<3}", style="cyan")

        row.append("  ")  # Spacing instead of │

        # Training metrics
        row.append(f"{m.loss:>9.4f}", style=self._get_loss_style(m.loss))
        row.append("  ")
        row.append(f"{m.accuracy*100:>6.1f}%", style=self._get_acc_style(m.accuracy))

        row.append("  ")  # Spacing instead of │

        # Validation metrics with improvement indicators
        val_loss_str = f"{m.val_loss:>9.4f}"
        if m.loss_improved:
            val_loss_str += " ↓"
        row.append(val_loss_str, style=self._get_loss_style(m.val_loss))
        row.append("  ")

        val_acc_str = f"{m.val_accuracy*100:>6.1f}%"
        if m.acc_improved:
            val_acc_str += " ↑"
        row.append(val_acc_str, style=self._get_acc_style(m.val_accuracy))

        row.append("  ")  # Spacing instead of │

        # Learning rate
        row.append(f"{m.lr:>9.2e}", style="dim")
        row.append("  ")

        # Duration
        if m.duration_sec < 60:
            time_str = f"{m.duration_sec:>5.1f}s"
        else:
            time_str = f"{m.duration_sec/60:>5.1f}m"
        row.append(time_str, style="dim")

        row.append("  ")  # Spacing instead of │

        # Status
        if m.acc_improved:
            row.append(f"{StatusGlyphs.BEST} Best", style="bold green")
        elif m.loss_improved:
            row.append(f"{StatusGlyphs.ARROW_DOWN} Loss", style="green")
        else:
            row.append(f"  {StatusGlyphs.NEUTRAL}", style="dim")

        self._rich_console.print(row)

    def _print_epoch_plain(self, m: EpochMetrics):
        """Print epoch row in plain text."""
        improved = "★" if m.acc_improved else ("↓" if m.loss_improved else " ")
        print(f"{m.epoch:>4}/{self.config.total_epochs:<3} {m.loss:>10.4f} {m.accuracy*100:>7.1f}% "
              f"{m.val_loss:>10.4f} {m.val_accuracy*100:>7.1f}% {m.lr:>10.2e} {m.duration_sec:>7.1f}s {improved}")

    def print_early_stop(self, reason: str = "val_accuracy", patience: int = 10):
        """Print early stopping notification."""
        if not RICH_AVAILABLE:
            print(f"\n⏹  Early stopping: {reason} did not improve for {patience} epochs")
            return

        self.console.blank()
        self.console.section_header("Early Stopping", color="yellow")
        self.console.key_value("Reason", f"{reason} did not improve for {patience} epochs")
        self.console.key_value("Best Epoch", str(self.best_epoch))
        self.console.key_value("Best Val Accuracy", f"{self.best_val_acc:.4f}")

    def print_lr_reduction(self, new_lr: float, old_lr: float):
        """Print learning rate reduction notification."""
        if not RICH_AVAILABLE:
            print(f"  ↓ LR reduced: {old_lr:.2e} → {new_lr:.2e}")
            return

        self.console.print(f"    [dim]↓ Learning rate reduced: {old_lr:.2e} → {new_lr:.2e}[/dim]")

    def print_summary(self, final_metrics: Dict[str, Any]):
        """Print training summary table."""
        duration = datetime.now() - self.start_time if self.start_time else timedelta(0)

        if not RICH_AVAILABLE:
            self._print_summary_plain(final_metrics, duration)
            return

        self.console.blank()
        self.console.divider(100)

        # Determine overall status
        val_acc = final_metrics.get('val_accuracy', 0)
        if val_acc >= 0.55:
            status_color = "green"
            status_text = f"{StatusGlyphs.SUCCESS} Training Successful"
        elif val_acc >= 0.50:
            status_color = "yellow"
            status_text = f"{StatusGlyphs.PENDING} Training Marginal"
        else:
            status_color = "red"
            status_text = f"{StatusGlyphs.ERROR} Training Below Baseline"

        self.console.section_header(status_text, color=status_color)

        # Performance section
        self.console.blank()
        self._rich_console.print("  [bold]PERFORMANCE[/bold]", style="green")
        bal_acc = final_metrics.get('val_balanced_accuracy', 0)
        up_acc = final_metrics.get('val_up_accuracy', 0)
        down_acc = final_metrics.get('val_down_accuracy', 0)

        self.console.key_value("Validation Accuracy", f"{val_acc:.4f}")
        self.console.key_value("Balanced Accuracy", f"{bal_acc:.4f}")
        self.console.key_value("  Up (Long)", f"{up_acc:.4f}")
        self.console.key_value("  Down (Short)", f"{down_acc:.4f}")

        # Statistics section
        self.console.blank()
        self._rich_console.print("  [bold]STATISTICS[/bold]", style="cyan")
        epochs_trained = final_metrics.get('epochs_trained', len(self.epoch_history))
        self.console.key_value("Epochs Trained", f"{epochs_trained}/{self.config.total_epochs}")
        self.console.key_value("Best Epoch", str(self.best_epoch))
        self.console.key_value("Training Time", self._format_duration(duration))
        self.console.key_value("Final LR", f"{self.current_lr:.2e}")

        if 'avg_weight_norm' in final_metrics:
            self.console.key_value("Avg Weight Norm", f"{final_metrics['avg_weight_norm']:.4f}")

        self.console.blank()
        self._rich_console.print(f"  [dim]Completed at {datetime.now().strftime('%H:%M:%S')}[/dim]")
        self.console.blank()

    def _print_summary_plain(self, final_metrics: Dict[str, Any], duration: timedelta):
        """Print summary in plain text."""
        print("\n" + "=" * 80)
        print("TRAINING SUMMARY")
        print("-" * 80)
        print(f"Validation Accuracy: {final_metrics.get('val_accuracy', 0):.4f}")
        print(f"Balanced Accuracy:   {final_metrics.get('val_balanced_accuracy', 0):.4f}")
        print(f"Epochs Trained:      {final_metrics.get('epochs_trained', len(self.epoch_history))}")
        print(f"Training Time:       {self._format_duration(duration)}")
        print("=" * 80 + "\n")

    def _format_duration(self, duration: timedelta) -> str:
        """Format duration for display."""
        total_seconds = int(duration.total_seconds())
        hours, remainder = divmod(total_seconds, 3600)
        minutes, seconds = divmod(remainder, 60)

        if hours > 0:
            return f"{hours}h {minutes}m {seconds}s"
        elif minutes > 0:
            return f"{minutes}m {seconds}s"
        else:
            return f"{seconds}s"


# =============================================================================
# KERAS CALLBACK
# =============================================================================

class ProfessionalTrainingCallback:
    """
    Keras callback wrapper for professional training display.

    Integrates with Keras training loop to provide:
    - Real-time epoch updates
    - Learning rate tracking
    - Early stopping notifications
    """

    def __init__(self, display: TrainingDisplay):
        self.display = display
        self.epoch_start_time = None
        self.last_lr = display.config.learning_rate

    def create_callbacks(self) -> list:
        """Create Keras callbacks with display integration."""
        from tensorflow import keras

        display = self.display
        parent = self

        class DisplayCallback(keras.callbacks.Callback):
            """Custom callback for display updates."""

            def on_epoch_begin(self, epoch, logs=None):
                parent.epoch_start_time = time.time()

            def on_epoch_end(self, epoch, logs=None):
                logs = logs or {}
                duration = time.time() - parent.epoch_start_time if parent.epoch_start_time else 0

                # Get current learning rate - handle LR schedules gracefully
                try:
                    lr_obj = self.model.optimizer.learning_rate
                    if hasattr(lr_obj, "numpy"):
                        lr = float(lr_obj.numpy())
                    elif hasattr(lr_obj, "value"):
                        lr = float(lr_obj.value())
                    else:
                        lr = float(lr_obj)
                except (TypeError, AttributeError):
                    # Learning rate is likely a schedule - try to get current value
                    try:
                        _lr = getattr(self.model.optimizer, "_learning_rate", None)
                        if callable(_lr) and hasattr(self.model.optimizer, "iterations"):
                            lr = float(_lr(self.model.optimizer.iterations))
                        else:
                            lr = parent.last_lr
                    except Exception:
                        lr = parent.last_lr

                # Check for LR changes
                if abs(lr - parent.last_lr) / max(parent.last_lr, 1e-10) > 0.01:
                    display.print_lr_reduction(lr, parent.last_lr)
                parent.last_lr = lr

                metrics = EpochMetrics(
                    epoch=epoch + 1,
                    loss=logs.get('loss', 0),
                    accuracy=logs.get('accuracy', logs.get('madl_accuracy', 0)),
                    val_loss=logs.get('val_loss', 0),
                    val_accuracy=logs.get('val_accuracy', logs.get('val_madl_accuracy', 0)),
                    lr=lr,
                    duration_sec=duration,
                )
                display.update_epoch(metrics)

            def on_train_end(self, logs=None):
                # Check if early stopping occurred
                if hasattr(self, 'stopped_epoch') and self.stopped_epoch > 0:
                    display.print_early_stop()

        class SilentEarlyStopping(keras.callbacks.EarlyStopping):
            """Early stopping with display integration."""

            def on_train_end(self, logs=None):
                if self.stopped_epoch > 0:
                    display.print_early_stop(
                        reason=self.monitor,
                        patience=self.patience
                    )
                super().on_train_end(logs)

        class SilentReduceLROnPlateau(keras.callbacks.ReduceLROnPlateau):
            """LR reduction with suppressed verbose output."""

            def __init__(self, **kwargs):
                # Force verbose=0 for silent operation
                kwargs['verbose'] = 0
                super().__init__(**kwargs)

        return DisplayCallback(), SilentEarlyStopping, SilentReduceLROnPlateau


def create_training_display(
    model_name: str = "Transformer",
    task: str = "Direction Prediction",
    instrument: str = "",
    granularity: str = "",
    total_epochs: int = 100,
    batch_size: int = 64,
    learning_rate: float = 0.001,
    n_train_samples: int = 0,
    n_val_samples: int = 0,
    n_features: int = 0,
    seq_len: int = 60,
    use_madl: bool = False,
    warm_start: bool = False,
    **extra_info
) -> TrainingDisplay:
    """Factory function to create training display."""
    config = TrainingConfig(
        model_name=model_name,
        task=task,
        instrument=instrument,
        granularity=granularity,
        total_epochs=total_epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        n_train_samples=n_train_samples,
        n_val_samples=n_val_samples,
        n_features=n_features,
        seq_len=seq_len,
        use_madl=use_madl,
        warm_start=warm_start,
        extra_info=extra_info,
    )
    return TrainingDisplay(config)


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def format_metric_table(metrics: Dict[str, float], title: str = "Metrics") -> str:
    """Format metrics as a table string."""
    if not RICH_AVAILABLE:
        lines = [f"\n{title}:", "-" * 40]
        for key, value in metrics.items():
            if isinstance(value, float):
                lines.append(f"  {key}: {value:.4f}")
            else:
                lines.append(f"  {key}: {value}")
        return "\n".join(lines)

    console = Console(force_terminal=True, record=True)
    table = Table(title=title, show_header=True)
    table.add_column("Metric", style="cyan")
    table.add_column("Value", justify="right")

    for key, value in metrics.items():
        if isinstance(value, float):
            table.add_row(key, f"{value:.4f}")
        else:
            table.add_row(key, str(value))

    console.print(table)
    return console.export_text()


def print_data_summary(
    n_train: int,
    n_val: int,
    n_features: int,
    class_balance: Dict[str, float] = None,
    regime_dist: Dict[str, float] = None,
):
    """Print data loading summary."""
    if not RICH_AVAILABLE:
        print("\nData Summary:")
        print(f"  Training samples: {n_train:,}")
        print(f"  Validation samples: {n_val:,}")
        print(f"  Features: {n_features}")
        if class_balance:
            print(f"  Class balance: {class_balance}")
        if regime_dist:
            print(f"  Regime distribution: {regime_dist}")
        return

    console = PremiumConsole(Console())

    console.blank()
    console.section_header("Data Loaded", color="cyan")
    console.key_value("Training", f"{n_train:,} samples")
    console.key_value("Validation", f"{n_val:,} samples")
    console.key_value("Features", str(n_features))

    if class_balance:
        for label, pct in class_balance.items():
            console.key_value(f"  {label}", f"{pct:.1%}")

    if regime_dist:
        for regime, pct in regime_dist.items():
            console.key_value(f"  {regime}", f"{pct:.1%}")


# =============================================================================
# QUICK MODEL TRAINING DISPLAY (XGBoost, RF, Ridge, HistGB)
# =============================================================================

@dataclass
class QuickModelResult:
    """Result from a quick-training model."""
    model_name: str
    step: str  # e.g., "2/4"
    purpose: str
    features_desc: str
    output_desc: str
    metrics: Dict[str, float]
    duration_sec: float = 0.0
    extra_info: str = ""


def print_quick_model_header(
    step: str,
    model_name: str,
    purpose: str,
    features_desc: str,
    output_desc: str,
    color: str = "cyan",
):
    """Print professional header for quick-training model step."""
    if not RICH_AVAILABLE:
        print(f"\n{'─' * 60}")
        print(f"  Step {step}: {model_name} ({purpose})")
        print(f"{'─' * 60}")
        print(f"  Features: {features_desc}")
        print(f"  Output: {output_desc}")
        return

    console = PremiumConsole(Console())

    console.blank()
    console.step_progress(int(step.split('/')[0]), int(step.split('/')[1]), model_name, color=color)
    console.key_value("Purpose", purpose, indent=True)
    console.key_value("Features", features_desc, indent=True)
    console.key_value("Output", output_desc, indent=True)


def print_quick_model_result(
    model_name: str,
    metrics: Dict[str, Any],
    duration_sec: float = 0.0,
    success: bool = True,
):
    """Print professional result for quick-training model."""
    if not RICH_AVAILABLE:
        status = "✓" if success else "✗"
        print(f"{status} {model_name} complete")
        for k, v in metrics.items():
            if isinstance(v, float):
                print(f"   {k}: {v:.4f}")
        return

    console = PremiumConsole(Console())

    # Build metrics display
    metrics_parts = []
    for key, value in metrics.items():
        if isinstance(value, float):
            if 'accuracy' in key.lower() or 'acc' in key.lower():
                color = _get_accuracy_color(value)
                metrics_parts.append(f"[{color}]{key}={value:.1%}[/{color}]")
            elif 'mae' in key.lower():
                metrics_parts.append(f"[cyan]{key}={value:.4f}[/cyan]")
            elif 'r2' in key.lower():
                color = "green" if value > 0 else "yellow" if value > -0.5 else "red"
                metrics_parts.append(f"[{color}]{key}={value:.3f}[/{color}]")
            else:
                metrics_parts.append(f"{key}={value:.4f}")
        else:
            metrics_parts.append(f"{key}={value}")

    metrics_str = "  ".join(metrics_parts)  # Use spacing instead of │

    if duration_sec > 0:
        time_str = f" [dim]({duration_sec:.1f}s)[/dim]"
    else:
        time_str = ""

    status = f"{StatusGlyphs.SUCCESS}" if success else f"{StatusGlyphs.ERROR}"
    status_style = "bold green" if success else "bold red"
    console._console.print(f"  [{status_style}]{status}[/{status_style}] [bold]{model_name}[/bold]: {metrics_str}{time_str}")


def _get_accuracy_color(acc: float) -> str:
    """Get color for accuracy value."""
    if acc >= ACCURACY_EXCELLENT:
        return "bold green"
    elif acc >= ACCURACY_GOOD:
        return "green"
    elif acc >= ACCURACY_FAIR:
        return "yellow"
    return "red"


# =============================================================================
# ENSEMBLE TRAINING SUMMARY
# =============================================================================

def print_ensemble_summary(
    model_results: Dict[str, Dict[str, Any]],
    config: Dict[str, Any],
    saved_paths: List[str],
    inference_logic: str = "",
    use_regime: bool = False,
):
    """Print professional ensemble training summary."""
    if not RICH_AVAILABLE:
        print("\n" + "=" * 60)
        print("  MODULAR ENSEMBLE TRAINING COMPLETE!")
        print("=" * 60)
        print("\nModel Performance:")
        for name, metrics in model_results.items():
            print(f"  • {name}: {metrics}")
        print("\nSaved Models:")
        for path in saved_paths:
            print(f"  • {path}")
        return

    console = PremiumConsole(Console())

    console.blank()
    console.section_header(
        f"{StatusGlyphs.SUCCESS} Modular Ensemble Training Complete",
        subtitle=f"Completed at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        color="green"
    )

    # Model Performance section
    console.blank()
    console._console.print("  [bold]MODEL PERFORMANCE[/bold]", style="green")
    console.divider(60)

    for model_name, info in model_results.items():
        role = info.get('role', '')
        metric_name = info.get('metric_name', '')
        metric_value = info.get('metric_value', 0)

        # Format metric
        if 'accuracy' in metric_name.lower():
            color = _get_accuracy_color(metric_value)
            metric_str = f"{metric_value:.1%}"
        elif 'mae' in metric_name.lower():
            color = "cyan"
            metric_str = f"{metric_value:.4f}"
        elif 'r2' in metric_name.lower():
            color = "green" if metric_value > 0 else "yellow" if metric_value > -0.5 else "red"
            metric_str = f"{metric_value:.3f}"
        else:
            color = "white"
            metric_str = f"{metric_value:.4f}" if isinstance(metric_value, float) else str(metric_value)

        # Status indicator
        status = info.get('status', 'ok')
        if status == 'excellent':
            status_str = f"[bold green]{StatusGlyphs.STAR}{StatusGlyphs.STAR}{StatusGlyphs.STAR}[/bold green]"
        elif status == 'good':
            status_str = f"[green]{StatusGlyphs.STAR}{StatusGlyphs.STAR}[/green]"
        elif status == 'ok':
            status_str = f"[yellow]{StatusGlyphs.STAR}[/yellow]"
        else:
            status_str = f"[dim]{StatusGlyphs.NEUTRAL}[/dim]"

        console._console.print(f"  [bold]{model_name}[/bold] [dim]({role})[/dim]")
        console._console.print(f"    {metric_name}: [{color}]{metric_str}[/{color}]  {status_str}")

    # Configuration section
    console.blank()
    console._console.print("  [bold]CONFIGURATION[/bold]", style="cyan")
    console.divider(60)

    for key, value in config.items():
        if isinstance(value, float):
            value_str = f"{value:.4f}" if abs(value) < 1 else f"{value:.1f}"
        elif isinstance(value, bool):
            value_str = "Yes" if value else "No"
        else:
            value_str = str(value)
        console.key_value(key, value_str, indent=True)

    # Saved models section
    console.blank()
    console._console.print("  [bold]SAVED MODELS[/bold]", style="dim")
    console.divider(60)

    for path in saved_paths:
        short_path = path.split('/')[-1] if '/' in path else path
        console._console.print(f"  {StatusGlyphs.BULLET} [cyan]{short_path}[/cyan]")

    # Inference logic section
    if inference_logic:
        console.blank()
        console._console.print("  [bold]INFERENCE LOGIC[/bold]", style="yellow")
        console.divider(60)
        console._console.print(f"  [yellow]{inference_logic}[/yellow]")

    console.blank()
