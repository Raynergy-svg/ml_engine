import numpy as np
from typing import Dict, Any, List, Union
import logging
from concurrent.futures import ThreadPoolExecutor
import time

logger = logging.getLogger(__name__)


class InferenceEngine:
    def __init__(self, model, config: Dict[str, Any] = None):
        self.model = model
        self.config = config or {}
        self.batch_size = self.config.get("batch_size", 32)
        self.num_workers = self.config.get("num_workers", 4)
        self._warmup()

    def _warmup(self):
        """Warm up the model with dummy data"""
        try:
            input_dim = getattr(self.model, "input_dim", 10)
            dummy_input = np.random.randn(1, input_dim)
            self.predict(dummy_input)
            logger.info("Model warmup completed")
        except Exception as e:
            logger.warning(f"Warmup failed: {e}")

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Single prediction"""
        if len(X.shape) == 1:
            X = X.reshape(1, -1)

        start_time = time.time()
        predictions = self.model.forward(X)
        inference_time = time.time() - start_time

        logger.debug(
            f"Inference completed in {inference_time:.4f}s for {len(X)} samples"
        )
        return predictions

    def predict_batch(self, X: np.ndarray) -> np.ndarray:
        """Batch prediction with automatic batching"""
        if len(X) <= self.batch_size:
            return self.predict(X)

        results = []
        for i in range(0, len(X), self.batch_size):
            batch = X[i : i + self.batch_size]
            batch_predictions = self.predict(batch)
            results.append(batch_predictions)

        return np.vstack(results)

    def predict_async(self, X_list: List[np.ndarray]) -> List[np.ndarray]:
        """Async batch predictions using thread pool"""
        with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            futures = [executor.submit(self.predict, X) for X in X_list]
            results = [future.result() for future in futures]

        return results

    def predict_with_confidence(self, X: np.ndarray) -> Dict[str, np.ndarray]:
        """Predict with confidence scores (for ensemble/probabilistic models)"""
        predictions = self.predict(X)

        # For regression: use prediction variance as confidence proxy
        if hasattr(self.model, "predict_uncertainty"):
            uncertainty = self.model.predict_uncertainty(X)
            confidence = 1.0 / (1.0 + uncertainty)
        else:
            # Default: constant confidence
            confidence = np.ones(len(predictions)) * 0.95

        return {"predictions": predictions, "confidence": confidence}

    def explain_prediction(
        self, X: np.ndarray, method: str = "gradients"
    ) -> Dict[str, Any]:
        """Generate prediction explanations"""
        predictions = self.predict(X)

        if method == "gradients":
            # Compute feature importance via gradients
            feature_importance = self._compute_feature_importance(X)
        else:
            feature_importance = None

        return {
            "predictions": predictions,
            "feature_importance": feature_importance,
            "method": method,
        }

    def _compute_feature_importance(self, X: np.ndarray) -> np.ndarray:
        """Compute feature importance using gradient-based method"""
        epsilon = 1e-5
        base_pred = self.predict(X)
        importance = np.zeros(X.shape)

        for i in range(X.shape[1]):
            X_perturbed = X.copy()
            X_perturbed[:, i] += epsilon
            perturbed_pred = self.predict(X_perturbed)
            importance[:, i] = np.abs(perturbed_pred - base_pred).flatten()

        # Normalize
        importance = importance / (np.sum(importance, axis=1, keepdims=True) + 1e-10)
        return importance
