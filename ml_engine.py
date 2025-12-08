"""
Optimized ML Engine with improved CPU and GPU performance.
Includes enhanced training loops, mixed precision, and memory optimization.
"""

import os
import gc
import time
import logging
import numpy as np
import json
from pathlib import Path
from typing import Dict, Any, Tuple, Optional, Union, List, Iterator, Callable
from functools import partial

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torch.cuda.amp import autocast, GradScaler
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingWarmRestarts
from sklearn.metrics import r2_score, mean_squared_error

# Import optimized modules
from memory_manager_enhanced import MemoryManager, memory_efficient, mixed_precision_context
from data_processing_optimized import StockDataset, create_optimized_dataloader
from models_enhanced import (
    StockPredictor,
    AttentiveLSTM,
    GRUPredictor,
    TransformerPredictor,
    TCNPredictor,
    EnsemblePredictor
)

# Configure logging
logger = logging.getLogger(__name__)


class EnhancedMLEngine:
    """
    Enhanced ML Engine for stock market prediction with optimized CPU/GPU performance.
    
    Features:
    - Automatic mixed precision training
    - Dynamic batch sizing
    - Memory-efficient operations
    - Gradient accumulation
    - Advanced learning rate scheduling
    - Model architecture selection
    - Ensemble methods
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the ML Engine with configuration.
        
        Args:
            config: Configuration dictionary with model and training parameters
        """
        self.config = config
        
        # Set device
        self.device = config.get("device", "cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")
        
        # Initialize memory manager
        self.memory_manager = MemoryManager(
            device=self.device,
            threshold_mb=config.get("memory", {}).get("threshold_mb", 1000),
            log_level=config.get("memory", {}).get("log_level", "INFO"),
            proactive_cleanup=config.get("memory", {}).get("proactive_cleanup", True)
        )
        
        # Set paths
        self.model_dir = Path(config.get("model_dir", "./models"))
        self.model_dir.mkdir(exist_ok=True, parents=True)
        
        # Initialize training state
        self.train_losses = []
        self.val_losses = []
        self.best_val_loss = float('inf')
        self.epochs_without_improvement = 0
        
        # Initialize model
        self.model = self._create_model()
        
        # Move model to device
        self.model = self.model.to(self.device)
        
        # Initialize optimizer
        self.optimizer = self._create_optimizer()
        
        # Initialize scheduler
        self.scheduler = self._create_scheduler()
        
        # Initialize gradient scaler for mixed precision
        self.scaler = GradScaler(enabled=config.get("mixed_precision", True) and self.device == "cuda")
        
        # Set training parameters
        self.batch_size = config.get("batch_size", 32)
        self.accumulation_steps = config.get("gradient_accumulation_steps", 1)
        self.mixed_precision = config.get("mixed_precision", True) and self.device == "cuda"
        self.clip_grad_norm = config.get("clip_grad_norm", 1.0)
        self.early_stopping_patience = config.get("early_stopping_patience", 10)
        
        logger.info(f"EnhancedMLEngine initialized with {self.model.__class__.__name__} model")
        
    def _create_model(self) -> nn.Module:
        """
        Create model based on configuration.
        
        Returns:
            PyTorch model
        """
        model_config = self.config.get("model", {})
        model_type = model_config.get("type", "lstm")
        input_size = model_config.get("input_size", 7)
        hidden_size = model_config.get("hidden_size", 128)
        num_layers = model_config.get("num_layers", 3)
        dropout = model_config.get("dropout", 0.2)
        bidirectional = model_config.get("bidirectional", False)
        
        # Create model based on type
        if model_type == "lstm":
            model = StockPredictor(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                dropout=dropout,
                bidirectional=bidirectional
            )
        elif model_type == "attention_lstm":
            model = AttentiveLSTM(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                num_heads=model_config.get("num_heads", 4),
                dropout=dropout,
                bidirectional=bidirectional,
                use_flash_attention=model_config.get("use_flash_attention", True)
            )
        elif model_type == "gru":
            model = GRUPredictor(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                dropout=dropout,
                bidirectional=bidirectional
            )
        elif model_type == "transformer":
            model = TransformerPredictor(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                dropout=dropout,
                num_heads=model_config.get("num_heads", 8),
                use_flash_attention=model_config.get("use_flash_attention", True),
                positional_encoding=model_config.get("positional_encoding", "learned")
            )
        elif model_type == "tcn":
            model = TCNPredictor(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                dropout=dropout,
                kernel_size=model_config.get("kernel_size", 3)
            )
        elif model_type == "ensemble":
            # Create base models
            base_models = []
            for base_config in model_config.get("base_models", []):
                base_type = base_config.get("type", "lstm")
                if base_type == "lstm":
                    base_model = StockPredictor(
                        input_size=input_size,
                        hidden_size=base_config.get("hidden_size", hidden_size),
                        num_layers=base_config.get("num_layers", num_layers),
                        dropout=base_config.get("dropout", dropout),
                        bidirectional=base_config.get("bidirectional", bidirectional)
                    )
                elif base_type == "attention_lstm":
                    base_model = AttentiveLSTM(
                        input_size=input_size,
                        hidden_size=base_config.get("hidden_size", hidden_size),
                        num_layers=base_config.get("num_layers", num_layers),
                        num_heads=base_config.get("num_heads", 4),
                        dropout=base_config.get("dropout", dropout),
                        bidirectional=base_config.get("bidirectional", bidirectional)
                    )
                elif base_type == "gru":
                    base_model = GRUPredictor(
                        input_size=input_size,
                        hidden_size=base_config.get("hidden_size", hidden_size),
                        num_layers=base_config.get("num_layers", num_layers),
                        dropout=base_config.get("dropout", dropout),
                        bidirectional=base_config.get("bidirectional", bidirectional)
                    )
                elif base_type == "transformer":
                    base_model = TransformerPredictor(
                        input_size=input_size,
                        hidden_size=base_config.get("hidden_size", hidden_size),
                        num_layers=base_config.get("num_layers", num_layers),
                        dropout=base_config.get("dropout", dropout),
                        num_heads=base_config.get("num_heads", 8)
                    )
                elif base_type == "tcn":
                    base_model = TCNPredictor(
                        input_size=input_size,
                        hidden_size=base_config.get("hidden_size", hidden_size),
                        num_layers=base_config.get("num_layers", num_layers),
                        dropout=base_config.get("dropout", dropout)
                    )
                base_models.append(base_model)
            
            # If no base models specified, create default set
            if not base_models:
                base_models = [
                    StockPredictor(input_size=input_size, hidden_size=hidden_size),
                    AttentiveLSTM(input_size=input_size, hidden_size=hidden_size),
                    GRUPredictor(input_size=input_size, hidden_size=hidden_size)
                ]
            
            # Create ensemble model
            model = EnsemblePredictor(
                models=base_models,
                input_size=input_size,
                hidden_size=hidden_size,
                dropout=dropout,
                ensemble_method=model_config.get("ensemble_method", "attention")
            )
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        # Load pre-trained weights if specified
        pretrained_path = model_config.get("pretrained_path")
        if pretrained_path:
            try:
                model.load_state_dict(torch.load(pretrained_path, map_location=self.device))
                logger.info(f"Loaded pre-trained weights from {pretrained_path}")
            except Exception as e:
                logger.warning(f"Failed to load pre-trained weights: {e}")
        
        return model
    
    def _create_optimizer(self) -> torch.optim.Optimizer:
        """
        Create optimizer based on configuration.
        
        Returns:
            PyTorch optimizer
        """
        optimizer_config = self.config.get("optimizer", {})
        optimizer_type = optimizer_config.get("type", "adam")
        lr = optimizer_config.get("learning_rate", 0.001)
        weight_decay = optimizer_config.get("weight_decay", 0.0)
        
        # Create optimizer based on type
        if optimizer_type.lower() == "adam":
            return optim.Adam(
                self.model.parameters(),
                lr=lr,
                weight_decay=weight_decay,
                betas=optimizer_config.get("betas", (0.9, 0.999))
            )
        elif optimizer_type.lower() == "adamw":
            return optim.AdamW(
                self.model.parameters(),
                lr=lr,
                weight_decay=weight_decay,
                betas=optimizer_config.get("betas", (0.9, 0.999))
            )
        elif optimizer_type.lower() == "sgd":
            return optim.SGD(
                self.model.parameters(),
                lr=lr,
                momentum=optimizer_config.get("momentum", 0.9),
                weight_decay=weight_decay
            )
        elif optimizer_type.lower() == "rmsprop":
            return optim.RMSprop(
                self.model.parameters(),
                lr=lr,
                weight_decay=weight_decay,
                momentum=optimizer_config.get("momentum", 0.0)
            )
        else:
            logger.warning(f"Unknown optimizer type: {optimizer_type}, using Adam")
            return optim.Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)
    
    def _create_scheduler(self) -> Optional[torch.optim.lr_scheduler._LRScheduler]:
        """
        Create learning rate scheduler based on configuration.
        
        Returns:
            PyTorch learning rate scheduler
        """
        scheduler_config = self.config.get("scheduler", {})
        scheduler_type = scheduler_config.get("type")
        
        if not scheduler_type:
            return None
        
        # Create scheduler based on type
        if scheduler_type.lower() == "plateau":
            return ReduceLROnPlateau(
                self.optimizer,
                mode="min",
                factor=scheduler_config.get("factor", 0.5),
                patience=scheduler_config.get("patience", 5),
                verbose=True
            )
        elif scheduler_type.lower() == "cosine":
            return CosineAnnealingWarmRestarts(
                self.optimizer,
                T_0=scheduler_config.get("t_0", 10),
                T_mult=scheduler_config.get("t_mult", 1),
                eta_min=scheduler_config.get("eta_min", 1e-6)
            )
        else:
            logger.warning(f"Unknown scheduler type: {scheduler_type}, not using scheduler")
            return None
    
    def _prepare_data(
        self, 
        features: Union[List, np.ndarray, torch.Tensor],
        targets: Union[List, np.ndarray, torch.Tensor],
        validation_split: float = 0.2,
        shuffle: bool = True
    ) -> Tuple[DataLoader, Optional[DataLoader]]:
        """
        Prepare data loaders for training and validation.
        
        Args:
            features: Input features
            targets: Target values
            validation_split: Fraction of data to use for validation
            shuffle: Whether to shuffle the data
            
        Returns:
            Tuple of (train_loader, val_loader)
        """
        # Convert to numpy arrays if needed
        if isinstance(features, list):
            features = np.array(features, dtype=np.float32)
        if isinstance(targets, list):
            targets = np.array(targets, dtype=np.float32)
        
        # Convert to tensors if needed
        if isinstance(features, np.ndarray):
            features = torch.tensor(features, dtype=torch.float32)
        if isinstance(targets, np.ndarray):
            targets = torch.tensor(targets, dtype=torch.float32)
        
        # Ensure targets are 2D
        if len(targets.shape) == 1:
            targets = targets.unsqueeze(1)
        
        # Split data into train and validation sets
        if validation_split > 0:
            # Calculate split index
            split_idx = int(len(features) * (1 - validation_split))
            
            if shuffle:
                # Generate random indices
                indices = torch.randperm(len(features))
                train_indices = indices[:split_idx]
                val_indices = indices[split_idx:]
                
                # Split data
                train_features = features[train_indices]
                train_targets = targets[train_indices]
                val_features = features[val_indices]
                val_targets = targets[val_indices]
            else:
                # Split data sequentially
                train_features = features[:split_idx]
                train_targets = targets[:split_idx]
                val_features = features[split_idx:]
                val_targets = targets[split_idx:]
            
            # Create datasets
            train_dataset = TensorDataset(train_features, train_targets)
            val_dataset = TensorDataset(val_features, val_targets)
            
            # Create data loaders with optimized settings
            train_loader = create_optimized_dataloader(
                train_dataset,
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=self.config.get("num_workers"),
                pin_memory=self.config.get("pin_memory"),
                device=self.device
            )
            
            val_loader = create_optimized_dataloader(
                val_dataset,
                batch_size=self.batch_size * 2,  # Larger batch size for validation
                shuffle=False,
                num_workers=self.config.get("num_workers"),
             <response clipped><NOTE>To save on context only part of this file has been shown to you. You should retry this tool after you have searched inside the file with `grep -n` in order to find the line numbers of what you are looking for.</NOTE>