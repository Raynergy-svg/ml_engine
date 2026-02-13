"""
Abstract Base Trainer for Modular Training System.

This module defines the abstract base class that all specialized trainers
must inherit from. It establishes the core interface for training, prediction,
saving, and loading models in the ML Engine.

All concrete trainers (Transformer, XGBoost, RandomForest, Ridge, etc.) must
implement the abstract methods defined in BaseTrainer.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, List

import numpy as np

from src.training.trainers.config import TrainerConfig


class BaseTrainer(ABC):
    """Abstract base class for all modular trainers."""

    def __init__(self, config: Optional[TrainerConfig] = None):
        self.config = config or TrainerConfig()
        self.model = None
        self.is_trained = False
        self.metrics: Dict[str, float] = {}

    @abstractmethod
    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        x_val: np.ndarray,
        y_val: np.ndarray,
        feature_names: Optional[List[str]] = None,
        instrument: str = '',
        fold_id: Optional[int] = None,
        **kwargs: Any,
    ) -> Dict[str, float]:
        """
        Train the model and return metrics.
        
        Args:
            X_train: Training features
            y_train: Training labels
            x_val: Validation features
            y_val: Validation labels
            feature_names: Optional list of feature names for the model
            instrument: Instrument/pair name (e.g., 'EUR_USD') for context
            fold_id: Optional fold ID for walk-forward cross-validation
            **kwargs: Additional trainer-specific keyword arguments
            
        Returns:
            Dictionary of training metrics
        """

    @abstractmethod
    def predict(self, X: np.ndarray) -> Dict[str, Any]:
        """Make predictions."""

    @abstractmethod
    def save(self, path: str) -> None:
        """Save model to disk."""

    @abstractmethod
    def load(self, path: str) -> None:
        """Load model from disk."""
