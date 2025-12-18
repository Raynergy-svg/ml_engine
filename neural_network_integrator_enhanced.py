"""
Neural Network Integration Module for connecting ML, MT, MR engines and reasoning head.
Optimized for CPU and GPU performance with enhanced model synchronization.
"""

import logging
import os
from typing import Any, Dict, Tuple

import numpy as np

# PyTorch was removed from this repository. This legacy integrator is retired.
torch = None  # type: ignore


class _NN:
    class Module:  # pragma: no cover
        pass


nn = _NN()  # type: ignore
F = None  # type: ignore

# Configure logging
logger = logging.getLogger(__name__)


class NeuralNetworkIntegrator:
    """
    Neural Network Integrator for connecting ML (Machine Learning), MT (Market Trends),
    and MR (Market Reasoning) engines with the reasoning head for synchronized predictions.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize the Neural Network Integrator.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        self.device = self.config.get("device", "cuda" if torch.cuda.is_available() else "cpu")
        self.ml_weight = self.config.get("ml_weight", 0.4)
        self.mt_weight = self.config.get("mt_weight", 0.3)
        self.mr_weight = self.config.get("mr_weight", 0.3)
        self.use_dynamic_weights = self.config.get("use_dynamic_weights", True)
        self.use_attention = self.config.get("use_attention", True)
        
        # Initialize engines
        self.ml_engine = None
        self.mt_engine = None
        self.mr_engine = None
        self.reasoning_engine = None

        # Unified engine (single-network mode)
        self.unified_engine = None
        
        # Initialize integration model
        self.integration_model = self._create_integration_model()
        
        # Move model to device
        self.integration_model = self.integration_model.to(self.device)
        
        logger.info(f"NeuralNetworkIntegrator initialized on {self.device}")
    
    def _create_integration_model(self) -> nn.Module:
        """
        Create the neural network integration model.
        
        Returns:
            PyTorch model for integration
        """
        if self.use_attention:
            return AttentionIntegrationModel(
                input_dim=self.config.get("feature_dim", 64),
                hidden_dim=self.config.get("hidden_dim", 128),
                num_heads=self.config.get("num_heads", 4),
                dropout=self.config.get("dropout", 0.2)
            )
        else:
            return BasicIntegrationModel(
                input_dim=self.config.get("feature_dim", 64),
                hidden_dim=self.config.get("hidden_dim", 128),
                dropout=self.config.get("dropout", 0.2)
            )
    
    def set_engines(
        self,
        ml_engine: Any,
        mt_engine: Any,
        mr_engine: Any,
        reasoning_engine: Any
    ):
        """
        Set the engines for integration.
        
        Args:
            ml_engine: Machine Learning engine
            mt_engine: Market Trends engine
            mr_engine: Market Reasoning engine
            reasoning_engine: Reasoning engine
        """
        self.ml_engine = ml_engine
        self.mt_engine = mt_engine
        self.mr_engine = mr_engine
        self.reasoning_engine = reasoning_engine

        # If multi-engine mode is configured, ensure unified mode is off.
        self.unified_engine = None
        
        logger.info("All engines set for integration")

    def set_unified_engine(self, unified_engine: Any, reasoning_engine: Any):
        """Use a single neural engine for predictions.

        The unified engine must expose `predict(features)` and return either a dict
        with `prediction`/`uncertainty` keys or a raw prediction.
        """
        self.unified_engine = unified_engine
        self.reasoning_engine = reasoning_engine

        # Clear legacy engines to avoid accidental use.
        self.ml_engine = None
        self.mt_engine = None
        self.mr_engine = None

        logger.info("Unified engine set for integration")
    
    def _calculate_dynamic_weights(
        self,
        ml_uncertainty: float,
        mt_uncertainty: float,
        mr_uncertainty: float
    ) -> Tuple[float, float, float]:
        """
        Calculate dynamic weights based on prediction uncertainties.
        
        Args:
            ml_uncertainty: Uncertainty of ML engine prediction
            mt_uncertainty: Uncertainty of MT engine prediction
            mr_uncertainty: Uncertainty of MR engine prediction
            
        Returns:
            Tuple of (ml_weight, mt_weight, mr_weight)
        """
        # Convert uncertainties to confidences
        ml_confidence = 1.0 - ml_uncertainty
        mt_confidence = 1.0 - mt_uncertainty
        mr_confidence = 1.0 - mr_uncertainty
        
        # Ensure confidences are positive
        ml_confidence = max(0.01, ml_confidence)
        mt_confidence = max(0.01, mt_confidence)
        mr_confidence = max(0.01, mr_confidence)
        
        # Calculate weights based on relative confidences
        total_confidence = ml_confidence + mt_confidence + mr_confidence
        
        ml_weight = ml_confidence / total_confidence
        mt_weight = mt_confidence / total_confidence
        mr_weight = mr_confidence / total_confidence
        
        return ml_weight, mt_weight, mr_weight
    
    def _ensure_engines_set(self) -> None:
        if self.unified_engine is not None:
            if self.reasoning_engine is None:
                raise ValueError("Reasoning engine must be set before making predictions")
            return

        if any(
            engine is None
            for engine in (
                self.ml_engine,
                self.mt_engine,
                self.mr_engine,
                self.reasoning_engine,
            )
        ):
            raise ValueError("All engines must be set before making predictions")

    @staticmethod
    def _split_prediction_and_uncertainty(result: Any) -> Tuple[Any, float]:
        if isinstance(result, dict):
            prediction = result.get("prediction", result)
            uncertainty = result.get("uncertainty", 0.2)
        else:
            prediction = result
            uncertainty = 0.2

        if isinstance(uncertainty, torch.Tensor):
            uncertainty = float(uncertainty.detach().cpu().item())
        elif isinstance(uncertainty, np.ndarray):
            uncertainty = float(np.mean(uncertainty))
        else:
            uncertainty = float(uncertainty)

        return prediction, uncertainty

    def _as_tensor_prediction(self, prediction: Any) -> torch.Tensor:
        if isinstance(prediction, torch.Tensor):
            tensor_pred = prediction
        else:
            tensor_pred = torch.as_tensor(prediction)

        tensor_pred = tensor_pred.to(self.device)

        # Normalize to (batch, pred_dim) so stacking/weighting works consistently
        if tensor_pred.dim() == 0:
            tensor_pred = tensor_pred.view(1, 1)
        elif tensor_pred.dim() == 1:
            tensor_pred = tensor_pred.unsqueeze(-1)
        elif tensor_pred.dim() > 2:
            tensor_pred = tensor_pred.reshape(tensor_pred.shape[0], -1)

        return tensor_pred

    def _get_engine_outputs(
        self, features: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, float, torch.Tensor, float, torch.Tensor, float]:
        ml_result = self.ml_engine.predict(features.get("ml_features"))
        mt_result = self.mt_engine.predict(features.get("mt_features"))
        mr_result = self.mr_engine.predict(features.get("mr_features"))

        ml_prediction, ml_uncertainty = self._split_prediction_and_uncertainty(ml_result)
        mt_prediction, mt_uncertainty = self._split_prediction_and_uncertainty(mt_result)
        mr_prediction, mr_uncertainty = self._split_prediction_and_uncertainty(mr_result)

        ml_prediction = self._as_tensor_prediction(ml_prediction)
        mt_prediction = self._as_tensor_prediction(mt_prediction)
        mr_prediction = self._as_tensor_prediction(mr_prediction)

        return ml_prediction, ml_uncertainty, mt_prediction, mt_uncertainty, mr_prediction, mr_uncertainty

    @staticmethod
    def _weights_to_1d_numpy(weights: Any) -> np.ndarray:
        if isinstance(weights, torch.Tensor):
            weights = weights.detach().cpu().numpy()
        weights = np.asarray(weights, dtype=float)
        if weights.ndim > 1:
            weights = weights.mean(axis=0)
        return weights.reshape(-1)

    def _integrate_predictions(
        self,
        ml_prediction: torch.Tensor,
        mt_prediction: torch.Tensor,
        mr_prediction: torch.Tensor,
    ) -> Tuple[torch.Tensor, np.ndarray]:
        with torch.no_grad():
            if self.use_attention:
                engine_outputs = torch.stack([ml_prediction, mt_prediction, mr_prediction], dim=1)
                integrated_prediction, attention_weights = self.integration_model(engine_outputs)
                engine_weights = self._weights_to_1d_numpy(attention_weights)
                return integrated_prediction, engine_weights

            engine_weights = np.array([self.ml_weight, self.mt_weight, self.mr_weight], dtype=float)
            integrated_prediction = (
                self.ml_weight * ml_prediction +
                self.mt_weight * mt_prediction +
                self.mr_weight * mr_prediction
            )
            return integrated_prediction, engine_weights

    @staticmethod
    def _compute_integrated_uncertainty(
        engine_weights: np.ndarray,
        ml_uncertainty: float,
        mt_uncertainty: float,
        mr_uncertainty: float
    ) -> float:
        weights = np.asarray(engine_weights, dtype=float).reshape(-1)
        if weights.size != 3:
            weights = np.array([1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0], dtype=float)

        uncs = np.array([ml_uncertainty, mt_uncertainty, mr_uncertainty], dtype=float)
        return float(np.dot(weights, uncs))

    @staticmethod
    def _reasoning_uncertainties(prediction_np: np.ndarray, integrated_uncertainty: float) -> np.ndarray:
        return np.full_like(prediction_np, float(integrated_uncertainty), dtype=float)

    def _build_result(
        self,
        integrated_prediction_np: np.ndarray,
        integrated_uncertainty: float,
        ml_prediction: torch.Tensor,
        mt_prediction: torch.Tensor,
        mr_prediction: torch.Tensor,
        ml_uncertainty: float,
        mt_uncertainty: float,
        mr_uncertainty: float,
        engine_weights: np.ndarray,
        reasoning_result: Any,
        features: Dict[str, torch.Tensor],
        return_features: bool,
    ) -> Dict[str, Any]:
        result = {
            "prediction": integrated_prediction_np,
            "uncertainty": integrated_uncertainty,
            "ml_prediction": ml_prediction.cpu().numpy(),
            "mt_prediction": mt_prediction.cpu().numpy(),
            "mr_prediction": mr_prediction.cpu().numpy(),
            "ml_uncertainty": ml_uncertainty,
            "mt_uncertainty": mt_uncertainty,
            "mr_uncertainty": mr_uncertainty,
            "weights": engine_weights,
            "reasoning": reasoning_result
        }

        if return_features:
            result.update({
                "ml_features": features.get("ml_features"),
                "mt_features": features.get("mt_features"),
                "mr_features": features.get("mr_features")
            })

        return result

    def predict(
        self,
        features: Dict[str, torch.Tensor],
        return_features: bool = False
    ) -> Dict[str, Any]:
        """
        Make integrated predictions using all engines.
        
        Args:
            features: Dictionary of input features for each engine
            return_features: Whether to return intermediate features
            
        Returns:
            Dictionary of prediction results
        """
        self._ensure_engines_set()

        # Single-network mode: bypass per-engine integration.
        if self.unified_engine is not None:
            unified_result = self.unified_engine.predict(features)
            unified_prediction, unified_uncertainty = self._split_prediction_and_uncertainty(unified_result)
            pred_tensor = self._as_tensor_prediction(unified_prediction)
            pred_np = pred_tensor.detach().cpu().numpy()

            # Maintain output compatibility with the multi-engine API.
            engine_weights = np.array([1.0, 0.0, 0.0], dtype=float)
            reasoning_result = self.reasoning_engine.analyze_predictions(
                predictions=pred_np,
                uncertainties=self._reasoning_uncertainties(pred_np, float(unified_uncertainty)),
            )

            result = self._build_result(
                integrated_prediction_np=pred_np,
                integrated_uncertainty=float(unified_uncertainty),
                ml_prediction=pred_tensor,
                mt_prediction=pred_tensor,
                mr_prediction=pred_tensor,
                ml_uncertainty=float(unified_uncertainty),
                mt_uncertainty=float(unified_uncertainty),
                mr_uncertainty=float(unified_uncertainty),
                engine_weights=engine_weights,
                reasoning_result=reasoning_result,
                features=features,
                return_features=return_features,
            )

            if isinstance(unified_result, dict):
                # Preserve the unified engine's additional head outputs if present.
                for extra_key in ("trend", "risk", "state_probs"):
                    if extra_key in unified_result:
                        result[extra_key] = unified_result[extra_key]

            return result

        (
            ml_prediction,
            ml_uncertainty,
            mt_prediction,
            mt_uncertainty,
            mr_prediction,
            mr_uncertainty,
        ) = self._get_engine_outputs(features)

        # Calculate dynamic weights if enabled (used by non-attention integration)
        if self.use_dynamic_weights and not self.use_attention:
            self.ml_weight, self.mt_weight, self.mr_weight = self._calculate_dynamic_weights(
                ml_uncertainty, mt_uncertainty, mr_uncertainty
            )

        integrated_prediction, engine_weights = self._integrate_predictions(
            ml_prediction, mt_prediction, mr_prediction
        )
        integrated_prediction_np = integrated_prediction.cpu().numpy()

        integrated_uncertainty = self._compute_integrated_uncertainty(
            engine_weights=engine_weights,
            ml_uncertainty=ml_uncertainty,
            mt_uncertainty=mt_uncertainty,
            mr_uncertainty=mr_uncertainty,
        )

        reasoning_result = self.reasoning_engine.analyze_predictions(
            predictions=integrated_prediction_np,
            uncertainties=self._reasoning_uncertainties(integrated_prediction_np, integrated_uncertainty),
        )

        return self._build_result(
            integrated_prediction_np=integrated_prediction_np,
            integrated_uncertainty=integrated_uncertainty,
            ml_prediction=ml_prediction,
            mt_prediction=mt_prediction,
            mr_prediction=mr_prediction,
            ml_uncertainty=ml_uncertainty,
            mt_uncertainty=mt_uncertainty,
            mr_uncertainty=mr_uncertainty,
            engine_weights=engine_weights,
            reasoning_result=reasoning_result,
            features=features,
            return_features=return_features,
        )
    
    def save_model(self, path: str):
        """
        Save the integration model.
        
        Args:
            path: Path to save the model
        """
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        # Save model state
        torch.save({
            "model_state_dict": self.integration_model.state_dict(),
            "config": self.config,
            "ml_weight": self.ml_weight,
            "mt_weight": self.mt_weight,
            "mr_weight": self.mr_weight,
            "use_dynamic_weights": self.use_dynamic_weights,
            "use_attention": self.use_attention
        }, path)
        
        logger.info(f"Integration model saved to {path}")
    
    def load_model(self, path: str):
        """
        Load the integration model.
        
        Args:
            path: Path to load the model from
        """
        # Load model state
        checkpoint = torch.load(path, map_location=self.device)
        
        # Load model state dict
        self.integration_model.load_state_dict(checkpoint["model_state_dict"])
        
        # Load configuration
        if "config" in checkpoint:
            self.config.update(checkpoint["config"])
        
        # Load weights
        if "ml_weight" in checkpoint:
            self.ml_weight = checkpoint["ml_weight"]
        if "mt_weight" in checkpoint:
            self.mt_weight = checkpoint["mt_weight"]
        if "mr_weight" in checkpoint:
            self.mr_weight = checkpoint["mr_weight"]
        if "use_dynamic_weights" in checkpoint:
            self.use_dynamic_weights = checkpoint["use_dynamic_weights"]
        if "use_attention" in checkpoint:
            self.use_attention = checkpoint["use_attention"]
        
        logger.info(f"Integration model loaded from {path}")


class BasicIntegrationModel(nn.Module):
    """Basic neural network for integrating predictions from multiple engines."""
    
    def __init__(self, input_dim: int, hidden_dim: int, dropout: float = 0.2):
        """
        Initialize the basic integration model.
        
        Args:
            input_dim: Input dimension
            hidden_dim: Hidden dimension
            dropout: Dropout rate
        """
        super().__init__()
        
        # Fully connected layers
        self.fc1 = nn.Linear(input_dim * 3, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.fc3 = nn.Linear(hidden_dim // 2, 1)
        
        # Dropout for regularization
        self.dropout = nn.Dropout(dropout)
        
        # Weight generation layer
        self.weight_gen = nn.Sequential(
            nn.Linear(input_dim * 3, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 3),
            nn.Softmax(dim=1)
        )
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape [batch_size, 3, input_dim]
            
        Returns:
            Tuple of (integrated_prediction, weights)
        """
        batch_size, _, _ = x.shape
        
        # Flatten inputs
        x_flat = x.reshape(batch_size, -1)
        
        # Generate weights
        weights = self.weight_gen(x_flat)
        
        # Apply weights to each engine's output
        weighted_sum = torch.sum(x * weights.unsqueeze(2), dim=1)
        
        # Process through fully connected layers
        out = F.relu(self.fc1(x_flat))
        out = self.dropout(out)
        out = F.relu(self.fc2(out))
        out = self.dropout(out)
        out = self.fc3(out)
        
        # Combine direct weighted sum with processed output
        integrated = out + weighted_sum.mean(dim=1, keepdim=True)
        
        return integrated, weights


class AttentionIntegrationModel(nn.Module):
    """Attention-based neural network for integrating predictions from multiple engines."""
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_heads: int = 4,
        dropout: float = 0.2,
    ):
        """
        Initialize the attention integration model.
        
        Args:
            input_dim: Input dimension
            hidden_dim: Hidden dimension
            num_heads: Number of attention heads
            dropout: Dropout rate
        """
        super().__init__()
        
        # Multi-head attention
        self.attention = nn.MultiheadAttention(
            embed_dim=input_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        self.layer_norm1 = nn.LayerNorm(input_dim)
        self.layer_norm2 = nn.LayerNorm(input_dim)

        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, input_dim),
        )

        # Output projection
        self.output_proj = nn.Linear(input_dim, 1)

        # Dropout for regularization
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass with attention mechanism.
        
        Args:
            x: Input tensor of shape [batch_size, 3, input_dim]
            
        Returns:
            Tuple of (integrated_prediction, attention_weights)
        """
        # x: [batch_size, 3, input_dim]
        
        # Apply attention
        query = self.layer_norm1(x)
        attn_output, attn_weights = self.attention(query, query, query)
        
        # Residual connection
        x = x + self.dropout(attn_output)
        
        # Apply feed-forward network with residual connection
        x = x + self.dropout(self.ffn(self.layer_norm2(x)))
        
        # Extract attention weights for each engine
        # attn_weights: [batch, tgt_len(=3), src_len(=3)] -> per-engine weights: [batch, 3]
        engine_weights = attn_weights.mean(dim=1)
        
        # Apply output projection to get final prediction
        # Average across engines
        x_avg = x.mean(dim=1)  # [batch_size, input_dim]
        
        # Project to scalar output
        integrated_prediction = self.output_proj(x_avg)  # [batch_size, 1]
        
        return integrated_prediction, engine_weights
