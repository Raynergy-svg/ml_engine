"""
Neural Network Integration Module for connecting ML, MT, MR engines and reasoning head.
Optimized for CPU and GPU performance with enhanced model synchronization.
"""

import os
import logging
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional, Union, Any, Callable
from pathlib import Path
import time
import json

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
        
        logger.info("All engines set for integration")
    
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
        # Ensure all engines are set
        if any(engine is None for engine in [self.ml_engine, self.mt_engine, self.mr_engine, self.reasoning_engine]):
            raise ValueError("All engines must be set before making predictions")
        
        # Get predictions from each engine
        ml_result = self.ml_engine.predict(features.get("ml_features"))
        mt_result = self.mt_engine.predict(features.get("mt_features"))
        mr_result = self.mr_engine.predict(features.get("mr_features"))
        
        # Extract predictions and uncertainties
        ml_prediction = ml_result.get("prediction", ml_result)
        mt_prediction = mt_result.get("prediction", mt_result)
        mr_prediction = mr_result.get("prediction", mr_result)
        
        ml_uncertainty = ml_result.get("uncertainty", 0.2)
        mt_uncertainty = mt_result.get("uncertainty", 0.2)
        mr_uncertainty = mr_result.get("uncertainty", 0.2)
        
        # Calculate dynamic weights if enabled
        if self.use_dynamic_weights:
            self.ml_weight, self.mt_weight, self.mr_weight = self._calculate_dynamic_weights(
                ml_uncertainty, mt_uncertainty, mr_uncertainty
            )
        
        # Convert predictions to tensors if needed
        if not isinstance(ml_prediction, torch.Tensor):
            ml_prediction = torch.tensor(ml_prediction, device=self.device)
        if not isinstance(mt_prediction, torch.Tensor):
            mt_prediction = torch.tensor(mt_prediction, device=self.device)
        if not isinstance(mr_prediction, torch.Tensor):
            mr_prediction = torch.tensor(mr_prediction, device=self.device)
        
        # Ensure predictions are on the correct device
        ml_prediction = ml_prediction.to(self.device)
        mt_prediction = mt_prediction.to(self.device)
        mr_prediction = mr_prediction.to(self.device)
        
        # Combine predictions using the integration model
        with torch.no_grad():
            if self.use_attention:
                # Prepare inputs for attention model
                engine_outputs = torch.stack([ml_prediction, mt_prediction, mr_prediction], dim=1)
                
                # Get integrated prediction
                integrated_prediction, attention_weights = self.integration_model(engine_outputs)
                
                # Convert attention weights to numpy for analysis
                attention_weights = attention_weights.cpu().numpy()
            else:
                # Simple weighted average
                integrated_prediction = (
                    self.ml_weight * ml_prediction +
                    self.mt_weight * mt_prediction +
                    self.mr_weight * mr_prediction
                )
                attention_weights = np.array([self.ml_weight, self.mt_weight, self.mr_weight])
        
        # Convert to numpy for further processing
        integrated_prediction_np = integrated_prediction.cpu().numpy()
        
        # Calculate integrated uncertainty
        if self.use_attention:
            # Weighted uncertainty based on attention weights
            integrated_uncertainty = (
                attention_weights[0] * ml_uncertainty +
                attention_weights[1] * mt_uncertainty +
                attention_weights[2] * mr_uncertainty
            )
        else:
            # Weighted uncertainty based on fixed/dynamic weights
            integrated_uncertainty = (
                self.ml_weight * ml_uncertainty +
                self.mt_weight * mt_uncertainty +
                self.mr_weight * mr_uncertainty
            )
        
        # Process through reasoning engine
        reasoning_result = self.reasoning_engine.analyze_predictions(
            predictions=integrated_prediction_np,
            uncertainties=integrated_uncertainty
        )
        
        # Prepare result dictionary
        result = {
            "prediction": integrated_prediction_np,
            "uncertainty": integrated_uncertainty,
            "ml_prediction": ml_prediction.cpu().numpy(),
            "mt_prediction": mt_prediction.cpu().numpy(),
            "mr_prediction": mr_prediction.cpu().numpy(),
            "ml_uncertainty": ml_uncertainty,
            "mt_uncertainty": mt_uncertainty,
            "mr_uncertainty": mr_uncertainty,
            "weights": attention_weights if self.use_attention else np.array([self.ml_weight, self.mt_weight, self.mr_weight]),
            "reasoning": reasoning_result
        }
        
        # Add intermediate features if requested
        if return_features:
            result.update({
                "ml_features": features.get("ml_features"),
                "mt_features": features.get("mt_features"),
                "mr_features": features.get("mr_features")
            })
        
        return result
    
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
        batch_size, num_engines, input_dim = x.shape
        
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
    
    def __init__(self, input_dim: int, hidden_dim: int, num_heads: int = 4, dropout: float = 0.2):
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
            batch_first=True
        )
        
        # Layer normalization
        self.layer_norm1 = nn.LayerNorm(input_dim)
        self.layer_norm2 = nn.LayerNorm(input_dim)
        
        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, input_dim)
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
        batch_size, num_engines, input_dim = x.shape
        
        # Apply attention
        query = self.layer_norm1(x)
        attn_output, attn_weights = self.attention(query, query, query)
        
        # Residual connection
        x = x + self.dropout(attn_output)
        
        # Apply feed-forward network with residual connection
        x = x + self.dropout(self.ffn(self.layer_norm2(x)))
        
        # Extract attention weights for each engine
        engine_weights = attn_weights.mean(dim=1)  # Average attention across batch
        
        # Apply output projec
        pass
