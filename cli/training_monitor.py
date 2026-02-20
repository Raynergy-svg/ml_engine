#!/usr/bin/env python3
"""Real-time terminal-based training visualization using Rich Live display."""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from rich.columns import Columns
from rich.console import Console, Group
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn
from rich.table import Table
from rich.text import Text

console = Console()


# Unicode sparkline characters (from lowest to highest)
SPARKLINE_CHARS = " ▁▂▃▄▅▆▇█"


def sparkline(values: List[float], width: int = 40) -> str:
    """Generate ASCII sparkline from values."""
    if not values:
        return "─" * width

    # Downsample if needed
    if len(values) > width:
        step = len(values) / width
        values = [values[int(i * step)] for i in range(width)]

    min_val = min(values)
    max_val = max(values)
    val_range = max_val - min_val if max_val > min_val else 1

    chars = []
    for v in values:
        normalized = (v - min_val) / val_range
        idx = min(int(normalized * (len(SPARKLINE_CHARS) - 1)), len(SPARKLINE_CHARS) - 1)
        chars.append(SPARKLINE_CHARS[idx])

    # Pad to width
    result = "".join(chars)
    if len(result) < width:
        result = "─" * (width - len(result)) + result

    return result


def trend_indicator(current: float, previous: float, lower_is_better: bool = False) -> str:
    """Return trend indicator arrow with color."""
    if previous == 0 or current == previous:
        return "[dim]→[/dim]"

    if lower_is_better:
        if current < previous:
            return "[green]↘[/green]"
        else:
            return "[red]↗[/red]"
    else:
        if current > previous:
            return "[green]↗[/green]"
        else:
            return "[red]↘[/red]"


@dataclass
class TrainingState:
    """Tracks current training state for visualization."""
    model_name: str = "Model"
    model_type: str = "Unknown"
    total_epochs: int = 100
    current_epoch: int = 0

    # History
    train_loss: List[float] = field(default_factory=list)
    val_loss: List[float] = field(default_factory=list)
    train_acc: List[float] = field(default_factory=list)
    val_acc: List[float] = field(default_factory=list)
    learning_rates: List[float] = field(default_factory=list)

    # Current metrics
    current_lr: float = 0.001
    best_val_acc: float = 0.0
    best_epoch: int = 0

    # Model info
    layer_info: List[Dict[str, Any]] = field(default_factory=list)
    total_params: int = 0
    trainable_params: int = 0

    # Last prediction sample
    last_input_shape: Optional[Tuple[int, ...]] = None
    last_prediction: Optional[List[float]] = None
    last_prediction_label: str = ""

    # Training health
    overfitting_gap: float = 0.0
    ema_active: bool = False
    swa_active: bool = False


class TrainingMonitor:
    """Real-time terminal-based training visualization."""

    def __init__(
        self,
        model_name: str = "Model",
        model_type: str = "TCN",
        total_epochs: int = 100,
    ):
        self.state = TrainingState(
            model_name=model_name,
            model_type=model_type,
            total_epochs=total_epochs,
        )
        self.console = Console()
        self._live: Optional[Live] = None

    def extract_model_info(self, model: Any) -> None:
        """Extract layer information from Keras model."""
        try:
            self.state.layer_info = []
            self.state.total_params = 0
            self.state.trainable_params = 0

            for layer in model.layers:
                layer_dict = {
                    "name": layer.name,
                    "type": layer.__class__.__name__,
                    "output_shape": str(getattr(layer, "output_shape", "?")),
                    "params": layer.count_params(),
                    "trainable": layer.trainable,
                }
                self.state.layer_info.append(layer_dict)
                self.state.total_params += layer.count_params()
                if layer.trainable:
                    self.state.trainable_params += layer.count_params()

        except Exception:
            pass  # Model introspection failed

    def _render_header(self) -> Panel:
        """Render the header with model name and progress."""
        progress = self.state.current_epoch / self.state.total_epochs if self.state.total_epochs > 0 else 0
        filled = int(progress * 40)
        bar = "█" * filled + "░" * (40 - filled)

        header_text = Text()
        header_text.append("BUDDY TRAINING MONITOR", style="bold cyan")
        header_text.append(f"  [{self.state.model_type}]", style="dim")

        progress_text = f"\n{bar} {self.state.current_epoch}/{self.state.total_epochs} epochs ({progress:.0%})"
        header_text.append(progress_text, style="yellow")

        return Panel(header_text, border_style="cyan")

    def _render_architecture(self) -> Panel:
        """Render model architecture as ASCII diagram."""
        if not self.state.layer_info:
            return Panel("[dim]No model loaded[/dim]", title="Architecture", border_style="blue")

        # Build ASCII diagram
        lines = []
        max_width = 20

        for i, layer in enumerate(self.state.layer_info):
            layer_type = layer["type"][:12]
            params = layer["params"]
            trainable = "●" if layer["trainable"] else "○"

            # Format parameter count
            if params >= 1_000_000:
                param_str = f"{params/1_000_000:.1f}M"
            elif params >= 1_000:
                param_str = f"{params/1_000:.1f}K"
            else:
                param_str = str(params)

            # Layer box
            box_content = f"{trainable} {layer_type}"
            box = f"┌{'─' * max_width}┐\n│{box_content:^{max_width}}│\n│{param_str:^{max_width}}│\n└{'─' * max_width}┘"

            if i < len(self.state.layer_info) - 1:
                box += f"\n{'│':^{max_width + 2}}\n{'▼':^{max_width + 2}}"

            lines.append(box)

        # Summary
        total = self.state.total_params
        trainable = self.state.trainable_params
        frozen = total - trainable
        summary = f"\n[dim]● trainable  ○ frozen[/dim]\n[cyan]Total: {total:,}  Trainable: {trainable:,}  Frozen: {frozen:,}[/cyan]"

        content = "\n".join(lines) + summary
        return Panel(content, title="Model Architecture", border_style="blue")

    def _render_loss_curves(self) -> Panel:
        """Render loss curves as sparklines."""
        lines = []

        # Train loss sparkline
        if self.state.train_loss:
            spark = sparkline(self.state.train_loss, width=35)
            current = self.state.train_loss[-1]
            prev = self.state.train_loss[-2] if len(self.state.train_loss) > 1 else current
            trend = trend_indicator(current, prev, lower_is_better=True)
            lines.append(f"[red]Train Loss[/red]  {spark} {current:.4f} {trend}")
        else:
            lines.append("[dim]Train Loss  ─────────────────────────────────── N/A[/dim]")

        # Val loss sparkline
        if self.state.val_loss:
            spark = sparkline(self.state.val_loss, width=35)
            current = self.state.val_loss[-1]
            prev = self.state.val_loss[-2] if len(self.state.val_loss) > 1 else current
            trend = trend_indicator(current, prev, lower_is_better=True)
            lines.append(f"[blue]Val Loss[/blue]    {spark} {current:.4f} {trend}")
        else:
            lines.append("[dim]Val Loss    ─────────────────────────────────── N/A[/dim]")

        # Train accuracy sparkline
        if self.state.train_acc:
            spark = sparkline(self.state.train_acc, width=35)
            current = self.state.train_acc[-1]
            prev = self.state.train_acc[-2] if len(self.state.train_acc) > 1 else current
            trend = trend_indicator(current, prev, lower_is_better=False)
            lines.append(f"[red]Train Acc[/red]   {spark} {current:.1%} {trend}")

        # Val accuracy sparkline
        if self.state.val_acc:
            spark = sparkline(self.state.val_acc, width=35)
            current = self.state.val_acc[-1]
            prev = self.state.val_acc[-2] if len(self.state.val_acc) > 1 else current
            trend = trend_indicator(current, prev, lower_is_better=False)
            lines.append(f"[blue]Val Acc[/blue]     {spark} {current:.1%} {trend}")

        content = "\n".join(lines) if lines else "[dim]No training data yet[/dim]"
        return Panel(content, title="Loss Curves", border_style="green")

    def _render_metrics(self) -> Panel:
        """Render current training metrics."""
        table = Table(show_header=False, box=None, padding=(0, 1))
        table.add_column("Metric", style="bold")
        table.add_column("Value", justify="right")

        # Current metrics
        if self.state.train_loss:
            table.add_row("Train Loss", f"{self.state.train_loss[-1]:.4f}")
        if self.state.val_loss:
            table.add_row("Val Loss", f"{self.state.val_loss[-1]:.4f}")
        if self.state.train_acc:
            color = "green" if self.state.train_acc[-1] >= 0.6 else "yellow"
            table.add_row("Train Acc", f"[{color}]{self.state.train_acc[-1]:.1%}[/{color}]")
        if self.state.val_acc:
            color = "green" if self.state.val_acc[-1] >= 0.55 else "yellow"
            table.add_row("Val Acc", f"[{color}]{self.state.val_acc[-1]:.1%}[/{color}]")

        table.add_row("", "")  # Spacer
        table.add_row("Best Epoch", f"{self.state.best_epoch}")
        table.add_row("Best Val Acc", f"[bold green]{self.state.best_val_acc:.1%}[/bold green]")
        table.add_row("Learning Rate", f"{self.state.current_lr:.2e}")

        # Overfitting gap
        if self.state.train_loss and self.state.val_loss:
            gap = self.state.val_loss[-1] - self.state.train_loss[-1]
            gap_color = "green" if gap < 0.1 else "yellow" if gap < 0.3 else "red"
            table.add_row("Overfit Gap", f"[{gap_color}]{gap:.3f}[/{gap_color}]")

        # Training health indicators
        table.add_row("", "")
        if self.state.ema_active:
            table.add_row("EMA", "[green]Active[/green]")
        if self.state.swa_active:
            table.add_row("SWA", "[green]Active[/green]")

        return Panel(table, title="Metrics", border_style="magenta")

    def _render_prediction_flow(self) -> Panel:
        """Render prediction flow visualization."""
        lines = []

        if self.state.last_input_shape:
            lines.append(f"[cyan]Input Shape:[/cyan] {self.state.last_input_shape}")

        if self.state.last_prediction:
            probs = self.state.last_prediction
            lines.append("")
            lines.append("[cyan]Output Probabilities:[/cyan]")

            # Visualize probabilities as bars
            labels = ["Class 0", "Class 1", "Class 2", "Class 3"][:len(probs)]
            max_prob = max(probs) if probs else 1

            for i, (label, prob) in enumerate(zip(labels, probs)):
                bar_width = int(prob / max_prob * 20) if max_prob > 0 else 0
                bar = "█" * bar_width + "░" * (20 - bar_width)
                is_max = prob == max_prob
                style = "bold green" if is_max else "dim"
                lines.append(f"  [{style}]{label}: {bar} {prob:.1%}[/{style}]")

            if self.state.last_prediction_label:
                lines.append("")
                lines.append(f"[bold]Prediction: {self.state.last_prediction_label}[/bold]")
        else:
            lines.append("[dim]No prediction yet - will update during training[/dim]")

        content = "\n".join(lines)
        return Panel(content, title="Prediction Flow", border_style="yellow")

    def render(self) -> Layout:
        """Render the full dashboard layout."""
        layout = Layout()

        # Top: Header
        layout.split_column(
            Layout(name="header", size=5),
            Layout(name="body"),
        )

        layout["header"].update(self._render_header())

        # Body: Architecture | Metrics + Loss
        layout["body"].split_row(
            Layout(name="left", ratio=1),
            Layout(name="right", ratio=2),
        )

        layout["left"].update(self._render_architecture())

        # Right: Loss curves + Metrics + Prediction
        layout["right"].split_column(
            Layout(name="curves", ratio=2),
            Layout(name="bottom", ratio=1),
        )

        layout["curves"].update(self._render_loss_curves())

        layout["bottom"].split_row(
            Layout(name="metrics", ratio=1),
            Layout(name="prediction", ratio=1),
        )

        layout["metrics"].update(self._render_metrics())
        layout["prediction"].update(self._render_prediction_flow())

        return layout

    def update(self, epoch: int, logs: Dict[str, Any]) -> None:
        """Update state from training logs (called by callback)."""
        self.state.current_epoch = epoch + 1

        # Extract metrics from logs
        if "loss" in logs:
            self.state.train_loss.append(float(logs["loss"]))
        if "val_loss" in logs:
            self.state.val_loss.append(float(logs["val_loss"]))
        if "accuracy" in logs:
            self.state.train_acc.append(float(logs["accuracy"]))
        if "val_accuracy" in logs:
            val_acc = float(logs["val_accuracy"])
            self.state.val_acc.append(val_acc)
            if val_acc > self.state.best_val_acc:
                self.state.best_val_acc = val_acc
                self.state.best_epoch = epoch + 1

        # Learning rate
        if "lr" in logs:
            self.state.current_lr = float(logs["lr"])
            self.state.learning_rates.append(self.state.current_lr)

        # Refresh display if live
        if self._live:
            self._live.refresh()

    def start(self) -> None:
        """Start the live display."""
        self._live = Live(self.render(), console=self.console, refresh_per_second=2)
        self._live.start()

    def stop(self) -> None:
        """Stop the live display."""
        if self._live:
            self._live.stop()
            self._live = None

    def __enter__(self) -> "TrainingMonitor":
        self.start()
        return self

    def __exit__(self, *args: Any) -> None:
        self.stop()


def render_model_architecture(model_type: str, config_path: Optional[str] = None) -> None:
    """Render a static model architecture view."""
    console.print(Panel(
        f"[bold cyan]Model Architecture: {model_type}[/bold cyan]",
        border_style="cyan",
    ))

    # Load model config if available
    if config_path:
        try:
            import yaml
            with open(config_path) as f:
                config = yaml.safe_load(f)

            model_cfg = config.get("model", {})
            console.print(f"\n[bold]Configuration:[/bold]")
            console.print(f"  Sequence Length: {model_cfg.get('seq_len', 60)}")
            console.print(f"  Hidden Size: {model_cfg.get('tcn_filters', 32)}")
            console.print(f"  Kernel Size: {model_cfg.get('tcn_kernel_size', 3)}")
            console.print(f"  Dropout: {model_cfg.get('dropout', 0.4)}")
        except Exception:
            pass

    # ASCII architecture based on model type
    if model_type.upper() == "TCN":
        arch = """
┌─────────────────────────────────────────────────────────────────┐
│                    TCN (Temporal Convolutional Network)         │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│   ┌──────────┐                                                  │
│   │  Input   │  OHLCV + ATR features                           │
│   │ (60, 24) │  60 timesteps, 24 features                      │
│   └────┬─────┘                                                  │
│        │                                                        │
│        ▼                                                        │
│   ┌──────────┐                                                  │
│   │ Gaussian │  Input noise regularization                     │
│   │  Noise   │  std=0.01                                        │
│   └────┬─────┘                                                  │
│        │                                                        │
│        ▼                                                        │
│   ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐       │
│   │  Conv1D  │─▶│  Conv1D  │─▶│  Conv1D  │─▶│  Conv1D  │       │
│   │ dil=1    │  │ dil=2    │  │ dil=4    │  │ dil=8    │       │
│   │ k=3,f=32 │  │ k=3,f=32 │  │ k=3,f=32 │  │ k=3,f=32 │       │
│   └──────────┘  └──────────┘  └──────────┘  └──────────┘       │
│        │                                                        │
│        ▼  Dilated causal convolutions with residual connections │
│   ┌──────────┐                                                  │
│   │ GlobalAvg│  Temporal aggregation                           │
│   │ Pooling  │                                                  │
│   └────┬─────┘                                                  │
│        │                                                        │
│        ▼                                                        │
│   ┌──────────┐                                                  │
│   │  Dense   │  Hidden layer with dropout                      │
│   │   64     │                                                  │
│   └────┬─────┘                                                  │
│        │                                                        │
│        ▼                                                        │
│   ┌──────────┐                                                  │
│   │ Softmax  │  4-class regime output                          │
│   │  Output  │  [LOW, NORMAL, HIGH, EXTREME]                   │
│   └──────────┘                                                  │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
"""
    elif model_type.upper() == "TRANSFORMER":
        arch = """
┌─────────────────────────────────────────────────────────────────┐
│              Transformer Encoder (Direction Prediction)        │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│   ┌──────────┐                                                  │
│   │  Input   │  Directional features                           │
│   │ (60, 32) │  60 timesteps, 32 features                      │
│   └────┬─────┘                                                  │
│        │                                                        │
│        ▼                                                        │
│   ┌──────────┐                                                  │
│   │  Linear  │  Input projection + positional encoding         │
│   │Projection│  d_model=32                                      │
│   └────┬─────┘                                                  │
│        │                                                        │
│    ┌───┴───┐×N  (N=2 encoder layers)                           │
│    │       │                                                    │
│    ▼       │                                                    │
│ ┌──────────┴──┐                                                 │
│ │ Multi-Head  │  Self-attention mechanism                      │
│ │ Attention   │  heads=4, dropout=0.1                          │
│ │   + Add&Norm│                                                 │
│ └──────┬──────┘                                                 │
│        │                                                        │
│        ▼                                                        │
│ ┌─────────────┐                                                 │
│ │ Feed-Forward│  Position-wise FFN                             │
│ │   Network   │  dff=64, dropout=0.1                           │
│ │   + Add&Norm│                                                 │
│ └──────┬──────┘                                                 │
│        │                                                        │
│    └───┴───┘                                                    │
│        │                                                        │
│        ▼                                                        │
│   ┌──────────┐                                                  │
│   │ GlobalAvg│  Sequence aggregation                           │
│   │ Pooling  │                                                  │
│   └────┬─────┘                                                  │
│        │                                                        │
│        ▼                                                        │
│   ┌──────────┐                                                  │
│   │ Sigmoid  │  Binary direction output                        │
│   │  Output  │  [DOWN=0, UP=1]                                  │
│   └──────────┘                                                  │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
"""
    else:
        arch = f"""
┌─────────────────────────────────────────────────────────────────┐
│                    Model: {model_type:^30}          │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│   Use --training flag during buddy train to see live details   │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
"""

    console.print(arch)


def replay_training_log(log_path: str) -> None:
    """Replay a saved training log file."""
    import time

    log_file = Path(log_path)
    if not log_file.exists():
        console.print(f"[red]Log file not found: {log_path}[/red]")
        return

    try:
        with open(log_file) as f:
            data = json.load(f)
    except json.JSONDecodeError:
        console.print(f"[red]Invalid JSON in log file: {log_path}[/red]")
        return

    model_name = data.get("model_name", "Model")
    model_type = data.get("model_type", "Unknown")
    history = data.get("history", {})

    epochs = len(history.get("loss", []))
    if epochs == 0:
        console.print("[yellow]No training history found in log file[/yellow]")
        return

    console.print(f"[cyan]Replaying training: {model_name} ({epochs} epochs)[/cyan]")
    console.print("[dim]Press Ctrl+C to stop[/dim]\n")

    monitor = TrainingMonitor(
        model_name=model_name,
        model_type=model_type,
        total_epochs=epochs,
    )

    try:
        with monitor:
            for epoch in range(epochs):
                logs = {
                    "loss": history.get("loss", [])[epoch] if epoch < len(history.get("loss", [])) else 0,
                    "val_loss": history.get("val_loss", [])[epoch] if epoch < len(history.get("val_loss", [])) else 0,
                    "accuracy": history.get("accuracy", [])[epoch] if epoch < len(history.get("accuracy", [])) else 0,
                    "val_accuracy": history.get("val_accuracy", [])[epoch] if epoch < len(history.get("val_accuracy", [])) else 0,
                }
                monitor.update(epoch, logs)
                time.sleep(0.1)  # Replay speed
    except KeyboardInterrupt:
        console.print("\n[yellow]Replay stopped[/yellow]")


# Export
__all__ = [
    "TrainingMonitor",
    "TrainingState",
    "render_model_architecture",
    "replay_training_log",
    "sparkline",
    "trend_indicator",
]
