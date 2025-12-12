from datetime import datetime, timezone
import logging
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


class ModelTrainer:
    def __init__(self, config: dict[str, Any]):
        self.config = config
        self.rng = np.random.default_rng(self.config.get("seed"))
        self.model = None
        self.history = {}

    def train(self, x_train, y_train, x_val=None, y_val=None) -> dict[str, Any]:
        """Train the model with actual training logic."""
        try:
            logger.info("Starting training with %s samples", len(x_train))

            # Initialize model based on config
            self.model = self._build_model()

            # Training loop
            epochs = int(self.config.get("epochs", 10))
            batch_size = max(1, int(self.config.get("batch_size", 32)))

            if len(x_train) == 0:
                raise ValueError("x_train is empty")

            self.history = {}

            for epoch in range(epochs):
                # Shuffle training data
                indices = self.rng.permutation(len(x_train))
                x_shuffled = x_train[indices]
                y_shuffled = y_train[indices]

                # Mini-batch training
                epoch_loss = 0.0
                num_batches = int(np.ceil(len(x_train) / batch_size))

                for i in range(0, len(x_train), batch_size):
                    x_batch = x_shuffled[i : i + batch_size]
                    y_batch = y_shuffled[i : i + batch_size]

                    loss = float(self._train_step(x_batch, y_batch))
                    epoch_loss += loss

                avg_loss = epoch_loss / max(num_batches, 1)

                # Validation
                val_metrics: dict[str, float] = {}
                if x_val is not None and y_val is not None:
                    val_metrics = self._validate(x_val, y_val)

                logger.info(
                    "Epoch %s/%s - Loss: %.4f - Val metrics: %s",
                    epoch + 1,
                    epochs,
                    avg_loss,
                    val_metrics,
                )

                # Store history
                self.history.setdefault("loss", []).append(avg_loss)
                for key, val in val_metrics.items():
                    self.history.setdefault(f"val_{key}", []).append(val)

            return {
                "status": "success",
                "history": self.history,
                "final_loss": self.history["loss"][-1],
                "trained_at": datetime.now(tz=timezone.utc).isoformat(),
            }

        except Exception:
            logger.exception("Training failed")
            raise

    def _build_model(self):
        """Build model architecture based on config"""
        model_type = self.config.get("model_type", "linear")
        input_dim = self.config.get("input_dim", 10)
        output_dim = self.config.get("output_dim", 1)

        if model_type == "linear":
            return LinearModel(input_dim, output_dim, rng=self.rng)
        elif model_type == "neural_net":
            hidden_dims = self.config.get("hidden_dims", [64, 32])
            return NeuralNetwork(input_dim, hidden_dims, output_dim, rng=self.rng)
        else:
            raise ValueError(f"Unknown model type: {model_type}")

    def _train_step(self, x_batch, y_batch) -> float:
        """Single training step"""
        predictions = self.model.forward(x_batch)
        loss = self._compute_loss(predictions, y_batch)
        gradients = self._compute_gradients(x_batch, y_batch, predictions)
        self.model.update_weights(gradients, self.config.get("learning_rate", 0.01))
        return loss

    def _compute_loss(self, predictions, targets) -> float:
        """Compute loss (MSE for regression)"""
        return np.mean((predictions - targets) ** 2)

    def _compute_gradients(self, X, y, predictions):
        """Compute gradients for backpropagation"""
        return self.model.backward(X, y, predictions)

    def _validate(self, x_val, y_val) -> dict[str, float]:
        """Validate model on validation set"""
        predictions = self.model.forward(x_val)
        mse = np.mean((predictions - y_val) ** 2)
        mae = np.mean(np.abs(predictions - y_val))

        return {"mse": float(mse), "mae": float(mae)}


class LinearModel:
    def __init__(self, input_dim: int, output_dim: int, *, rng: np.random.Generator):
        self.weights = rng.standard_normal((input_dim, output_dim)) * 0.01
        self.bias = np.zeros(output_dim)

    def forward(self, X):
        return X @ self.weights + self.bias

    def backward(self, X, y, predictions):
        m = len(X)
        dw = (1 / m) * X.T @ (predictions - y)
        db = (1 / m) * np.sum(predictions - y, axis=0)
        return {"dw": dw, "db": db}

    def update_weights(self, gradients, learning_rate):
        self.weights -= learning_rate * gradients["dw"]
        self.bias -= learning_rate * gradients["db"]


class NeuralNetwork:
    def __init__(self, input_dim: int, hidden_dims: list[int], output_dim: int, *, rng: np.random.Generator):
        self.layers = []
        dims = [input_dim] + hidden_dims + [output_dim]

        for i in range(len(dims) - 1):
            self.layers.append(
                {
                    "W": rng.standard_normal((dims[i], dims[i + 1])) * np.sqrt(2.0 / dims[i]),
                    "b": np.zeros(dims[i + 1]),
                }
            )

        self.cache = {}

    def forward(self, X):
        A = X
        self.cache["A0"] = A

        for i, layer in enumerate(self.layers):
            Z = A @ layer["W"] + layer["b"]
            A = self._relu(Z) if i < len(self.layers) - 1 else Z
            self.cache[f"Z{i + 1}"] = Z
            self.cache[f"A{i + 1}"] = A

        return A

    def backward(self, X, y, predictions):
        m = len(X)
        gradients = []

        # Output layer gradient
        d_a = predictions - y

        # Backpropagate through layers
        for i in reversed(range(len(self.layers))):
            Z = self.cache[f"Z{i + 1}"]
            a_prev = self.cache[f"A{i}"]

            if i < len(self.layers) - 1:
                d_a = d_a * self._relu_derivative(Z)

            d_w = (1 / m) * a_prev.T @ d_a
            db = (1 / m) * np.sum(d_a, axis=0)

            gradients.insert(0, {"dW": d_w, "db": db})

            if i > 0:
                d_a = d_a @ self.layers[i]["W"].T

        return gradients

    def update_weights(self, gradients, learning_rate):
        for layer, grad in zip(self.layers, gradients):
            layer["W"] -= learning_rate * grad["dW"]
            layer["b"] -= learning_rate * grad["db"]

    @staticmethod
    def _relu(x):
        return np.maximum(0, x)

    @staticmethod
    def _relu_derivative(x):
        return (x > 0).astype(float)
