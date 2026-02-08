import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np
import time
import collections
from sklearn.metrics import r2_score, mean_squared_error
import yaml  # Import the YAML library
from scipy.signal import savgol_filter  # For data smoothing


class MLVisualizer:
    def __init__(self, ml_engine, config_path="config_tuned.yaml"):
        self.ml_engine = ml_engine
        self.paused = False  # For pausing/resuming animation
        self.data_queue = collections.deque(
            maxlen=100
        )  # Store the last 100 data points
        self.window_size = 20  # Rolling window size for statistics
        self.config_path = config_path  # Store the path to the config file
        self.config = self.load_config()  # Load the configuration

        # Enhanced configuration handling
        self.model_config = self.config.get("model", {})
        self.viz_config = self.config.get(
            "visualization",
            {
                "update_interval": 1000,
                "max_points": 100,
                "smoothing_window": 5,
                "plot_rolling_mean": True,
                "performance_metrics": ["r2", "mse", "accuracy"],
            },
        )

        # Additional tracking for enhanced ML engine metrics
        self.data_queue = collections.deque(
            maxlen=self.viz_config.get("max_points", 100)
        )
        self.architecture_type = self.model_config.get("architecture", "lstm")
        self.learning_rate = self.model_config.get("learning_rate", 0.001)
        self.batch_size = self.model_config.get("batch_size", 32)

        # Create a figure with multiple subplots
        self.fig = plt.figure(figsize=(15, 12))
        self.fig.suptitle(
            f"Real-Time ML Engine Metrics - {self.architecture_type.upper()}",
            fontsize=16,
        )

        # Training Loss Plot
        self.ax1 = plt.subplot2grid((4, 2), (0, 0))
        self.ax1.set_title("Training Metrics")
        self.ax1.set_xlabel("Epoch")
        self.ax1.set_ylabel("Loss")
        (self.loss_line,) = self.ax1.plot([], [], "r-", label="Train Loss")
        (self.val_loss_line,) = self.ax1.plot([], [], "b-", label="Val Loss")
        self.loss_annotation = self.ax1.text(
            0.05,
            0.95,
            "",
            transform=self.ax1.transAxes,
            fontsize=10,
            verticalalignment="top",
        )
        self.ax1.legend()
        self.ax1.grid(True)

        # Learning Rate Plot
        self.ax2 = plt.subplot2grid((4, 2), (0, 1))
        self.ax2.set_title("Training Parameters")
        (self.lr_line,) = self.ax2.plot([], [], "g-", label="Learning Rate")
        (self.batch_loss_line,) = self.ax2.plot([], [], "r--", label="Batch Loss")
        self.param_annotation = self.ax2.text(
            0.05,
            0.95,
            "",
            transform=self.ax2.transAxes,
            fontsize=10,
            verticalalignment="top",
        )
        # New: Define lr_annotation to display the current learning rate.
        self.lr_annotation = self.ax2.text(
            0.05,
            0.85,
            "",
            transform=self.ax2.transAxes,
            fontsize=10,
            verticalalignment="top",
        )
        self.ax2.legend()
        self.ax2.grid(True)

        # Prediction vs Actual Plot
        self.ax3 = plt.subplot2grid((4, 2), (1, 0), colspan=2)
        self.ax3.set_title("Predictions vs Actual")
        self.ax3.set_xlabel("Time")
        self.ax3.set_ylabel("Value")
        (self.pred_line,) = self.ax3.plot([], [], "r-", label="Predicted")
        (self.actual_line,) = self.ax3.plot([], [], "b-", label="Actual")
        # Add rolling mean lines
        (self.pred_rolling_mean_line,) = self.ax3.plot(
            [], [], "m--", label="Predicted Rolling Mean"
        )
        (self.actual_rolling_mean_line,) = self.ax3.plot(
            [], [], "c--", label="Actual Rolling Mean"
        )
        self.ax3.legend(loc="upper right")
        self.ax3.grid(True)

        # Model Architecture Info
        self.ax4 = plt.subplot2grid((4, 2), (2, 0))
        self.ax4.set_title("Model Architecture")
        self.ax4.axis("off")
        self.model_info = self.ax4.text(
            0.05,
            0.95,
            "",
            transform=self.ax4.transAxes,
            fontsize=10,
            verticalalignment="top",
        )

        # Performance Metrics Plot
        self.ax5 = plt.subplot2grid((4, 2), (2, 1))
        self.ax5.set_title("Performance Metrics")
        self.ax5.set_xlabel("Time")
        self.ax5.set_ylabel("Score")
        (self.r2_line,) = self.ax5.plot([], [], "g-", label="R² Score")
        (self.mse_line,) = self.ax5.plot([], [], "r-", label="MSE")
        self.perf_annotation = self.ax5.text(
            0.05,
            0.95,
            "",
            transform=self.ax5.transAxes,
            fontsize=10,
            verticalalignment="top",
        )
        self.ax5.legend()
        self.ax5.grid(True)

        # Error Distribution Plot
        self.ax6 = plt.subplot2grid((4, 2), (3, 0))
        self.ax6.set_title("Error Distribution")
        self.ax6.set_xlabel("Error")
        self.ax6.set_ylabel("Frequency")
        self.error_hist = self.ax6.hist([], bins=30, color="gray", alpha=0.7)

        # System Metrics
        self.ax7 = plt.subplot2grid((4, 2), (3, 1))
        self.ax7.set_title("System Metrics")
        (self.cpu_line,) = self.ax7.plot([], [], "b-", label="CPU Usage")
        (self.memory_line,) = self.ax7.plot([], [], "r-", label="Memory Usage")
        self.system_annotation = self.ax7.text(
            0.05,
            0.95,
            "",
            transform=self.ax7.transAxes,
            fontsize=10,
            verticalalignment="top",
        )
        self.ax7.legend()
        self.ax7.grid(True)

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        # Remove interactive mode here so that plt.show() creates a dedicated window.
        # plt.ion()

        # Set interactive key bindings
        self.fig.canvas.mpl_connect("key_press_event", self.on_key_press)

    def load_config(self):
        """Load the configuration from the YAML file."""
        with open(self.config_path) as f:
            return yaml.safe_load(f)

    def update_config(self, new_config):
        """Update the configuration and apply changes to the ML engine."""
        self.config = new_config
        # Apply the changes to the ML engine (this will depend on your ML engine's API)
        self.ml_engine.apply_config(
            self.config
        )  # Assuming ml_engine has an apply_config method
        print("Configuration updated.")

    def on_key_press(self, event):
        if event.key == "p":
            self.paused = not self.paused
            print(f"Animation {'paused' if self.paused else 'resumed'}.")
        elif event.key == "r":
            # Trigger online retraining if supported by ml_engine.
            try:
                new_data = (
                    self.ml_engine.get_new_data()
                )  # ml_engine should implement this method
                if new_data is not None:
                    self.ml_engine.online_retrain(new_data)
                    print("Online retraining triggered.")
                else:
                    print("No new data available for online retraining.")
            except Exception as ex:
                print(f"Retraining error: {ex}")
        elif event.key == "u":  # Add a key press event to update the config
            # For simplicity, let's reload the config from the file
            # In a real application, you would get the new config from the GUI
            self.config = self.load_config()
            self.update_config(self.config)
            print("Configuration reloaded from file.")
        elif event.key == "c":
            # New: Clear all plots
            self.data_queue.clear()
            print("Cleared all plots.")
        elif event.key == "s":
            # New: Save current figure
            timestamp = time.strftime("%Y%m%d-%H%M%S")
            self.fig.savefig(f"ml_visualization_{timestamp}.png")
            print(f"Saved figure as ml_visualization_{timestamp}.png")

    def smooth_data(self, data, window_length=5, polyorder=2):
        """Apply Savitzky-Golay filter for data smoothing."""
        try:
            if len(data) < window_length:
                window_length = len(data) - (
                    len(data) % 2 == 0
                )  # make window_length odd
                if window_length < 3:
                    print("Data too short for smoothing. Returning original data.")
                    return data
            return savgol_filter(data, window_length, polyorder)
        except Exception as e:
            print(f"Smoothing error: {e}. Returning original data.")
            return data

    def update(self, frame):
        if self.paused:
            return []

        # Fetch new data from ml_engine
        new_data = self.ml_engine.get_latest_data()
        if new_data:
            self.data_queue.append(new_data)

        # Prepare data for plotting
        predictions = [
            np.float64(data["prediction"]) if "prediction" in data else np.nan
            for data in self.data_queue
        ]
        actuals = [
            np.float64(data["actual"]) if "actual" in data else np.nan
            for data in self.data_queue
        ]

        # Data Smoothing: Only smooth if there are enough data points.
        if len(predictions) >= 5:
            predictions_smoothed = self.smooth_data(predictions)
        else:
            predictions_smoothed = predictions
        if len(actuals) >= 5:
            actuals_smoothed = self.smooth_data(actuals)
        else:
            actuals_smoothed = actuals

        # Remove NaN values
        predictions_clean = np.nan_to_num(predictions_smoothed)
        actuals_clean = np.nan_to_num(actuals_smoothed)

        # Update Training Loss
        if hasattr(
            self.ml_engine, "training_history"
        ) and self.ml_engine.training_history.get("losses"):
            epochs = range(len(self.ml_engine.training_history["losses"]))
            self.loss_line.set_data(epochs, self.ml_engine.training_history["losses"])
            if self.ml_engine.training_history.get("val_losses"):
                self.val_loss_line.set_data(
                    epochs, self.ml_engine.training_history["val_losses"]
                )
            self.ax1.relim()
            self.ax1.autoscale_view()
            latest_loss = self.ml_engine.training_history["losses"][-1]
            self.loss_annotation.set_text(f"Latest Loss: {latest_loss:.4f}")

        # Update Learning Rate
        if hasattr(self.ml_engine, "optimizer"):
            lrs = [group["lr"] for group in self.ml_engine.optimizer.param_groups]
            if lrs:
                epochs = range(len(self.ml_engine.training_history.get("losses", [])))
                self.lr_line.set_data(epochs, [lrs[0]] * len(epochs))
                self.ax2.relim()
                self.ax2.autoscale_view()
                self.lr_annotation.set_text(f"LR: {lrs[0]:.6f}")

        # Update Predictions vs Actual
        if predictions_clean.size > 0 and actuals_clean.size > 0:
            times = range(len(predictions_clean))

            # Ensure times, predictions and actuals have the same length
            min_len = min(len(times), len(predictions_clean), len(actuals_clean))
            times = times[:min_len]
            predictions_clean = predictions_clean[:min_len]
            actuals_clean = actuals_clean[:min_len]

            self.pred_line.set_data(times, predictions_clean)
            self.actual_line.set_data(times, actuals_clean)

            # Calculate rolling means
            if len(predictions_clean) >= self.window_size:
                pred_rolling_mean = np.convolve(
                    predictions_clean,
                    np.ones(self.window_size) / self.window_size,
                    mode="valid",
                )
                actual_rolling_mean = np.convolve(
                    actuals_clean,
                    np.ones(self.window_size) / self.window_size,
                    mode="valid",
                )

                # Adjust time axis for rolling mean
                rolling_times = range(
                    len(predictions_clean) - self.window_size + 1
                )  # Corrected this line

                # Ensure rolling_times and rolling_mean have the same length
                min_len_rolling = min(
                    len(rolling_times), len(pred_rolling_mean), len(actual_rolling_mean)
                )
                rolling_times = rolling_times[:min_len_rolling]
                pred_rolling_mean = pred_rolling_mean[:min_len_rolling]
                actual_rolling_mean = actual_rolling_mean[:min_len_rolling]

                self.pred_rolling_mean_line.set_data(rolling_times, pred_rolling_mean)
                self.actual_rolling_mean_line.set_data(
                    rolling_times, actual_rolling_mean
                )

            self.ax3.relim()
            self.ax3.autoscale_view()

        # Update Error Distribution
        if predictions_clean.size > 0 and actuals_clean.size > 0:
            errors = np.array(predictions_clean) - np.array(actuals_clean)
            self.ax6.cla()
            self.ax6.set_title("Error Distribution")
            self.ax6.set_xlabel("Error")
            self.ax6.set_ylabel("Frequency")
            self.ax6.hist(errors, bins=30, color="gray", alpha=0.7)

        # Update Performance Metrics
        if predictions_clean.size > 1 and actuals_clean.size > 1:
            r2 = r2_score(actuals_clean, predictions_clean)
            mse = mean_squared_error(actuals_clean, predictions_clean)
            times = range(len(predictions_clean))
            self.r2_line.set_data(times, [r2] * len(times))
            self.mse_line.set_data(times, [mse] * len(times))
            self.ax5.relim()
            self.ax5.autoscale_view()
            self.perf_annotation.set_text(f"R²: {r2:.4f}\nMSE: {mse:.4f}")

        # Update system metrics
        times = range(len(self.data_queue))
        if hasattr(self.ml_engine, "cpu_usage"):
            self.cpu_line.set_data(times, self.ml_engine.cpu_usage)
        if hasattr(self.ml_engine, "memory_usage"):
            self.memory_line.set_data(times, self.ml_engine.memory_usage)
        self.ax7.relim()
        self.ax7.autoscale_view()

        # Update model architecture information
        model_info_text = (
            f"Architecture: {self.architecture_type}\n"
            f"Hidden Size: {self.model_config.get('hidden_size')}\n"
            f"Num Layers: {self.model_config.get('num_layers')}\n"
            f"Dropout: {self.model_config.get('dropout')}\n"
            f"Sequence Length: {self.model_config.get('sequence_length')}"
        )
        self.model_info.set_text(model_info_text)

        self.fig.canvas.draw_idle()
        # Return a list of updated artists; since we disabled blitting, this list is less critical.
        return []


def start_visualization(ml_engine):
    """
    Launch a real-time visualization window that displays ML engine metrics.
    This window shows training loss, learning rate, predictions vs. actual values,
    error distribution, and performance metrics. Interactive commands:
      - Press 'p' to pause/resume.
      - Press 'r' to trigger online retraining.
    """
    visualizer = MLVisualizer(ml_engine)
    # Create the animation without blitting for robust subplot updates.
    visualizer.anim = FuncAnimation(
        visualizer.fig,
        visualizer.update,
        interval=1000,
        blit=False,
        save_count=500,  # Add save_count here
    )
    plt.show()  # This will open the visualization in a separate window


# Example usage:
if __name__ == "__main__":
    import threading

    # For demonstration, we create a dummy ml_engine object with the required attributes.
    # In practice, replace this with your actual ML engine instance.
    class DummyMLEngine:
        def __init__(self):
            self.training_history = {
                "losses": [],  # Initialize as empty lists
                "val_losses": [],
            }
            self.optimizer = type("opt", (), {"param_groups": [{"lr": 0.001}]})
            self.recent_predictions = collections.deque(
                maxlen=60
            )  # Use deque for efficiency
            self.recent_actuals = collections.deque(
                maxlen=60
            )  # Use deque for efficiency
            self.time = 0  # Initialize time
            self.epochs = 100

        def get_new_data(self):
            # Return dummy data for retraining
            return None

        def online_retrain(self, new_data):
            print("Retraining with new data...")

        def train_step(self, epoch):
            # Simulate a training step
            loss = np.random.rand()
            val_loss = np.random.rand()
            prediction = np.random.rand()
            actual = np.random.rand()

            # Update training history
            self.training_history["losses"].append(loss)
            self.training_history["val_losses"].append(val_loss)

            # Update recent predictions and actuals
            self.recent_predictions.append(prediction)
            self.recent_actuals.append(actual)
            self.time = epoch  # sync time with epoch

        def start_training(self):
            # Spawn a background thread that simulates continuous training
            def training_loop():
                for epoch in range(self.epochs):
                    self.train_step(epoch)
                    time.sleep(0.5)  # Simulate training delay
                print("Training complete.")

            threading.Thread(target=training_loop, daemon=True).start()

        def get_latest_data(self):
            # Simply read the latest values without triggering train_step
            if self.recent_predictions and self.recent_actuals:
                prediction = self.recent_predictions[-1]
                actual = self.recent_actuals[-1]
            else:
                prediction = np.random.rand()
                actual = np.random.rand()
            return {"prediction": prediction, "actual": actual, "time": self.time}

    # Create a dummy engine instance; replace with your real engine.
    dummy_engine = DummyMLEngine()
    dummy_engine.start_training()  # Start live training in background

    # Start the visualization in its own window.
    start_visualization(dummy_engine)
# — Raynergy-svg —
