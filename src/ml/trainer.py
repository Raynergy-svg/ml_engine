import numpy as np
from typing import Dict, Any, Optional
import logging
from datetime import datetime

logger = logging.getLogger(__name__)


class ModelTrainer:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.model = None
        self.history = {}

    def train(self, X_train, y_train, X_val=None, y_val=None) -> Dict[str, Any]:
        """Train the model with actual training logic"""
        try:
            logger.info(f"Starting training with {len(X_train)} samples")

            # Initialize model based on config
            self.model = self._build_model()

            # Training loop
            epochs = self.config.get("epochs", 10)
            batch_size = self.config.get("batch_size", 32)

            for epoch in range(epochs):
                # Shuffle training data
                indices = np.random.permutation(len(X_train))
                X_shuffled = X_train[indices]
                y_shuffled = y_train[indices]

                # Mini-batch training
                epoch_loss = 0
                num_batches = len(X_train) // batch_size

                for i in range(0, len(X_train), batch_size):
                    X_batch = X_shuffled[i : i + batch_size]
                    y_batch = y_shuffled[i : i + batch_size]

                    loss = self._train_step(X_batch, y_batch)
                    epoch_loss += loss

                avg_loss = epoch_loss / num_batches

                # Validation
                val_metrics = {}
                if X_val is not None and y_val is not None:
                    val_metrics = self._validate(X_val, y_val)

                logger.info(
                    f"Epoch {epoch + 1}/{epochs} - Loss: {avg_loss:.4f} - Val metrics: {val_metrics}"
                )

                # Store history
                self.history.setdefault("loss", []).append(avg_loss)
                for key, val in val_metrics.items():
                    self.history.setdefault(f"val_{key}", []).append(val)

            return {
                "status": "success",
                "history": self.history,
                "final_loss": self.history["loss"][-1],
                "trained_at": datetime.utcnow().isoformat(),
            }

        except Exception as e:
            logger.error(f"Training failed: {str(e)}")
            raise

    def _build_model(self):
        """Build model architecture based on config"""
        model_type = self.config.get("model_type", "linear")
        input_dim = self.config.get("input_dim", 10)
        output_dim = self.config.get("output_dim", 1)

        if model_type == "linear":
            return LinearModel(input_dim, output_dim)
        elif model_type == "neural_net":
            hidden_dims = self.config.get("hidden_dims", [64, 32])
            return NeuralNetwork(input_dim, hidden_dims, output_dim)
        else:
            raise ValueError(f"Unknown model type: {model_type}")

    def _train_step(self, X_batch, y_batch) -> float:
        """Single training step"""
        predictions = self.model.forward(X_batch)
        loss = self._compute_loss(predictions, y_batch)
        gradients = self._compute_gradients(X_batch, y_batch, predictions)
        self.model.update_weights(gradients, self.config.get("learning_rate", 0.01))
        return loss

    def _compute_loss(self, predictions, targets) -> float:
        """Compute loss (MSE for regression)"""
        return np.mean((predictions - targets) ** 2)

    def _compute_gradients(self, X, y, predictions):
        """Compute gradients for backpropagation"""
        return self.model.backward(X, y, predictions)

    def _validate(self, X_val, y_val) -> Dict[str, float]:
        """Validate model on validation set"""
        predictions = self.model.forward(X_val)
        mse = np.mean((predictions - y_val) ** 2)
        mae = np.mean(np.abs(predictions - y_val))

        return {"mse": float(mse), "mae": float(mae)}


class LinearModel:
    def __init__(self, input_dim: int, output_dim: int):
        self.weights = np.random.randn(input_dim, output_dim) * 0.01
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
    def __init__(self, input_dim: int, hidden_dims: list, output_dim: int):
        self.layers = []
        dims = [input_dim] + hidden_dims + [output_dim]

        for i in range(len(dims) - 1):
            self.layers.append(
                {
                    "W": np.random.randn(dims[i], dims[i + 1]) * np.sqrt(2.0 / dims[i]),
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
        dA = predictions - y

        # Backpropagate through layers
        for i in reversed(range(len(self.layers))):
            Z = self.cache[f"Z{i + 1}"]
            A_prev = self.cache[f"A{i}"]

            if i < len(self.layers) - 1:
                dA = dA * self._relu_derivative(Z)

            dW = (1 / m) * A_prev.T @ dA
            db = (1 / m) * np.sum(dA, axis=0)

            gradients.insert(0, {"dW": dW, "db": db})

            if i > 0:
                dA = dA @ self.layers[i]["W"].T

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
