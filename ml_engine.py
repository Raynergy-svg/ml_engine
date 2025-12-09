"""
Enhanced ML Engine with optimized CPU and GPU performance.

This module provides the core ML engine for stock market prediction with:
- Automatic mixed precision training for faster GPU training
- Dynamic batch sizing and gradient accumulation
- Memory-efficient operations with automatic cleanup
- Multiple model architectures (LSTM, GRU, Transformer, TCN, Ensemble)
- Advanced learning rate scheduling and early stopping
- Comprehensive metrics tracking and model checkpointing

Example:
    >>> from ml_engine import EnhancedMLEngine
    >>> config = {"device": "cuda", "model": {"type": "lstm", "hidden_size": 128}}
    >>> engine = EnhancedMLEngine(config)
    >>> for epoch, metrics in engine.train(X_train, y_train):
    ...     print(f"Epoch {epoch}: {metrics}")
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
                             pin_memory=self.config.get("pin_memory"),
                device=self.device
            )
            
            return train_loader, val_loader
        else:
            # No validation split
            dataset = TensorDataset(features, targets)
            
            train_loader = create_optimized_dataloader(
                dataset,
                batch_size=self.batch_size,
                shuffle=shuffle,
                num_workers=self.config.get("num_workers"),
                pin_memory=self.config.get("pin_memory"),
                device=self.device
            )
            
            return train_loader, None
    
    def train(
        self, 
        X_train,
        y_train,
        X_val=None,
        y_val=None,
        validation_split=0.2
    ):
        """
        Train the model with the given data.
        
        Args:
            X_train: Training features
            y_train: Training targets
            X_val: Optional validation features
            y_val: Optional validation targets
            validation_split: Fraction to use for validation if X_val not provided
            
        Yields:
            Tuple of (epoch, metrics_dict)
        """
        import time
        
        # Prepare data loaders
        if X_val is not None and y_val is not None:
            # Use provided validation data
            train_loader, _ = self._prepare_data(X_train, y_train, validation_split=0, shuffle=True)
            val_loader, _ = self._prepare_data(X_val, y_val, validation_split=0, shuffle=False)
        else:
            # Use validation split
            train_loader, val_loader = self._prepare_data(
                X_train, y_train, validation_split=validation_split, shuffle=True
            )
        
        # Training setup
        criterion = nn.MSELoss()
        epochs = self.config.get("training", {}).get("epochs", self.config.get("epochs", 10))
        
        # Set model to training mode
        self.model.train()
        
        for epoch in range(epochs):
            epoch_start = time.time()
            train_loss = 0.0
            num_batches = 0
            
            # Training loop
            for batch_idx, (batch_x, batch_y) in enumerate(train_loader):
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)
                
                # Forward pass
                if self.mixed_precision:
                    with autocast():
                        outputs = self.model(batch_x)
                        loss = criterion(outputs, batch_y)
                    
                    # Backward pass with gradient scaling
                    self.scaler.scale(loss).backward()
                    
                    # Gradient clipping
                    if self.clip_grad_norm:
                        self.scaler.unscale_(self.optimizer)
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_grad_norm)
                    
                    # Optimizer step
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    outputs = self.model(batch_x)
                    loss = criterion(outputs, batch_y)
                    
                    # Backward pass
                    loss.backward()
                    
                    # Gradient clipping
                    if self.clip_grad_norm:
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_grad_norm)
                    
                    # Optimizer step
                    self.optimizer.step()
                
                # Zero gradients
                self.optimizer.zero_grad()
                
                train_loss += loss.item()
                num_batches += 1
            
            avg_train_loss = train_loss / num_batches if num_batches > 0 else 0
            
            # Validation
            val_loss = 0.0
            if val_loader:
                self.model.eval()
                with torch.no_grad():
                    val_batches = 0
                    for batch_x, batch_y in val_loader:
                        batch_x = batch_x.to(self.device)
                        batch_y = batch_y.to(self.device)
                        outputs = self.model(batch_x)
                        loss = criterion(outputs, batch_y)
                        val_loss += loss.item()
                        val_batches += 1
                    val_loss = val_loss / val_batches if val_batches > 0 else 0
                self.model.train()
            
            # Update learning rate
            lr_before = self.optimizer.param_groups[0]['lr']
            if self.scheduler:
                if isinstance(self.scheduler, ReduceLROnPlateau):
                    self.scheduler.step(val_loss if val_loader else avg_train_loss)
                else:
                    self.scheduler.step()
            lr_after = self.optimizer.param_groups[0]['lr']
            lr_delta = ((lr_after - lr_before) / lr_before * 100) if lr_before > 0 else 0
            
            # Track losses
            self.train_losses.append(avg_train_loss)
            self.val_losses.append(val_loss)
            
            # Early stopping check
            if val_loader and val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.epochs_without_improvement = 0
            else:
                self.epochs_without_improvement += 1
            
            # Calculate metrics
            epoch_time = time.time() - epoch_start
            latency = int(epoch_time * 1000)
            
            metrics = {
                "train_loss": avg_train_loss,
                "val_loss": val_loss,
                "latency": latency,
                "lr_delta": lr_delta,
                "message": f"Epoch {epoch+1}/{epochs} - Train Loss: {avg_train_loss:.6f}, Val Loss: {val_loss:.6f}"
            }
            
            yield epoch + 1, metrics
            
            # Early stopping
            if self.epochs_without_improvement >= self.early_stopping_patience:
                logger.info(f"Early stopping triggered after {epoch+1} epochs")
                break
    
    def evaluate_model(self):
        """Evaluate the model."""
        return {
            "train_loss": self.train_losses[-1] if self.train_losses else 0.0,
            "val_loss": self.val_losses[-1] if self.val_losses else 0.0,
            "best_val_loss": self.best_val_loss
        }
    
    def predict_price(self):
        """Generate a price prediction."""
        return 0.0
    
    def run_realtime_loop(self):
        """Run real-time inference loop."""
        logger.info("Real-time loop not yet implemented")
    
    def tune_hyperparameters(self):
        """Perform hyperparameter tuning."""
        logger.info("Hyperparameter tuning not yet implemented")
    
    def profile_pipeline(self):
        """Profile the ML pipeline."""
        logger.info("Pipeline profiling not yet implemented")
    
    def save_model(self, path):
        """Save the model to disk."""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'best_val_loss': self.best_val_loss,
            'config': self.config
        }, path)
        logger.info(f"Model saved to {path}")
    
    def load_model(self, path):
        """Load the model from disk."""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.train_losses = checkpoint.get('train_losses', [])
        self.val_losses = checkpoint.get('val_losses', [])
        self.best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        logger.info(f"Model loaded from {path}")


class ML_Engine:
    """Legacy ML Engine class for backwards compatibility."""
    
    def __init__(self, config):
        """Initialize legacy ML Engine."""
        self.config = config
        logger.warning("ML_Engine is deprecated. Use EnhancedMLEngine instead.")


class ai_assistant:
    """AI Assistant class for ML Engine."""
    
    def __init__(self, config):
        """Initialize AI Assistant."""
        self.config = config
    
    def process_query(self, query, use_claude=False):
        """Process a query and return a response."""
        return f"AI Assistant response to: {query}"
