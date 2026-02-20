#!/usr/bin/env python3
"""Keras callback for live training visualization."""
from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, TYPE_CHECKING

import tensorflow as tf

if TYPE_CHECKING:
    from cli.training_monitor import TrainingMonitor

logger = logging.getLogger(__name__)


class TrainingMonitorCallback(tf.keras.callbacks.Callback):
    """Keras callback that updates the terminal training monitor.

    Integrates with cli.training_monitor.TrainingMonitor for real-time
    visualization of training progress, metrics, and model architecture.

    Usage:
        from cli.training_monitor import TrainingMonitor
        from src.training.trainers.monitor_callback import TrainingMonitorCallback

        monitor = TrainingMonitor(model_name="TCN", total_epochs=100)
        callback = TrainingMonitorCallback(monitor)

        with monitor:
            model.fit(..., callbacks=[callback])
    """

    def __init__(
        self,
        monitor: "TrainingMonitor",
        *,
        save_history: bool = True,
        history_dir: Optional[Path] = None,
        update_frequency: int = 1,  # Update every N epochs
    ):
        """Initialize the callback.

        Args:
            monitor: TrainingMonitor instance for visualization
            save_history: Whether to save training history to JSON
            history_dir: Directory for saving history (default: trained_data/logs)
            update_frequency: Update display every N epochs
        """
        super().__init__()
        self.monitor = monitor
        self.save_history = save_history
        self.history_dir = Path(history_dir or "trained_data/logs")
        self.update_frequency = update_frequency

        # Track history for saving
        self._history: Dict[str, List[float]] = {
            "loss": [],
            "val_loss": [],
            "accuracy": [],
            "val_accuracy": [],
            "lr": [],
        }
        self._start_time: Optional[datetime] = None

    def on_train_begin(self, logs: Optional[Dict[str, Any]] = None) -> None:
        """Called at the beginning of training."""
        self._start_time = datetime.now()

        # Extract model info for architecture visualization
        if self.model is not None:
            self.monitor.extract_model_info(self.model)

            # Set input shape from model
            try:
                input_shape = self.model.input_shape
                if isinstance(input_shape, list):
                    input_shape = input_shape[0]
                self.monitor.state.last_input_shape = input_shape
            except Exception:
                pass

    def on_epoch_end(self, epoch: int, logs: Optional[Dict[str, Any]] = None) -> None:
        """Called at the end of each epoch."""
        if logs is None:
            logs = {}

        # Store history
        for key in ["loss", "val_loss", "accuracy", "val_accuracy"]:
            if key in logs:
                self._history[key].append(float(logs[key]))

        # Get learning rate
        try:
            lr = float(tf.keras.backend.get_value(self.model.optimizer.learning_rate))
            logs["lr"] = lr
            self._history["lr"].append(lr)
        except Exception:
            pass

        # Update monitor (respecting update_frequency)
        if (epoch + 1) % self.update_frequency == 0:
            self.monitor.update(epoch, logs)

        # Sample prediction for visualization
        if (epoch + 1) % 10 == 0:  # Every 10 epochs
            self._update_prediction_sample()

    def _update_prediction_sample(self) -> None:
        """Run a sample prediction to visualize prediction flow."""
        if self.model is None:
            return

        try:
            import numpy as np

            # Get input shape and create random sample
            input_shape = self.model.input_shape
            if isinstance(input_shape, list):
                input_shape = input_shape[0]

            # Create sample (batch_size=1)
            sample_shape = (1,) + tuple(d if d is not None else 1 for d in input_shape[1:])
            sample = np.random.randn(*sample_shape).astype(np.float32)

            # Run prediction
            prediction = self.model.predict(sample, verbose=0)

            # Store prediction probabilities
            if prediction.ndim == 2 and prediction.shape[0] == 1:
                probs = prediction[0].tolist()
                self.monitor.state.last_prediction = probs

                # Determine label
                pred_class = int(np.argmax(probs))
                if len(probs) == 4:
                    labels = ["LOW", "NORMAL", "HIGH", "EXTREME"]
                elif len(probs) == 2:
                    labels = ["DOWN", "UP"]
                else:
                    labels = [f"Class {i}" for i in range(len(probs))]

                self.monitor.state.last_prediction_label = f"{labels[pred_class]} ({probs[pred_class]:.1%})"

        except Exception as e:
            logger.debug(f"Failed to update prediction sample: {e}")

    def on_train_end(self, logs: Optional[Dict[str, Any]] = None) -> None:
        """Called at the end of training."""
        if not self.save_history:
            return

        # Save training history
        self.history_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        history_file = self.history_dir / f"training_history_{self.monitor.state.model_name}_{timestamp}.json"

        history_data = {
            "model_name": self.monitor.state.model_name,
            "model_type": self.monitor.state.model_type,
            "total_epochs": self.monitor.state.current_epoch,
            "best_epoch": self.monitor.state.best_epoch,
            "best_val_acc": self.monitor.state.best_val_acc,
            "start_time": self._start_time.isoformat() if self._start_time else None,
            "end_time": datetime.now().isoformat(),
            "history": self._history,
            "final_metrics": {
                "train_loss": self._history["loss"][-1] if self._history["loss"] else None,
                "val_loss": self._history["val_loss"][-1] if self._history["val_loss"] else None,
                "train_acc": self._history["accuracy"][-1] if self._history["accuracy"] else None,
                "val_acc": self._history["val_accuracy"][-1] if self._history["val_accuracy"] else None,
            },
            "model_summary": {
                "total_params": self.monitor.state.total_params,
                "trainable_params": self.monitor.state.trainable_params,
                "layers": len(self.monitor.state.layer_info),
            },
        }

        try:
            with open(history_file, "w") as f:
                json.dump(history_data, f, indent=2)
            logger.info(f"Training history saved to {history_file}")
        except Exception as e:
            logger.warning(f"Failed to save training history: {e}")


class QuietTrainingMonitorCallback(TrainingMonitorCallback):
    """Silent version that only saves history without live display updates."""

    def __init__(
        self,
        monitor: "TrainingMonitor",
        *,
        save_history: bool = True,
        history_dir: Optional[Path] = None,
    ):
        super().__init__(
            monitor,
            save_history=save_history,
            history_dir=history_dir,
            update_frequency=9999,  # Effectively never update display
        )

    def on_epoch_end(self, epoch: int, logs: Optional[Dict[str, Any]] = None) -> None:
        """Only store history, don't update display."""
        if logs is None:
            logs = {}

        # Store history only
        for key in ["loss", "val_loss", "accuracy", "val_accuracy"]:
            if key in logs:
                self._history[key].append(float(logs[key]))

        try:
            lr = float(tf.keras.backend.get_value(self.model.optimizer.learning_rate))
            self._history["lr"].append(lr)
        except Exception:
            pass


# Export
__all__ = [
    "TrainingMonitorCallback",
    "QuietTrainingMonitorCallback",
]
