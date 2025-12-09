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
    TransformerPredictor
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
        # elif model_type == "tcn":
        #     # TCNPredictor not yet implemented
        #     model = TCNPredictor(
        #         input_size=input_size,
        #         hidden_size=hidden_size,
        #         num_layers=num_layers,
        #         dropout=dropout,
        #         kernel_size=model_config.get("kernel_size", 3)
        #     )
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
                # elif base_type == "tcn":
                #     # TCNPredictor not yet implemented
                #     base_model = TCNPredictor(
                #         input_size=input_size,
                #         hidden_size=base_config.get("hidden_size", hidden_size),
                #         num_layers=base_config.get("num_layers", num_layers),
                #         dropout=base_config.get("dropout", dropout)
                #     )
                base_models.append(base_model)
            
            # If no base models specified, create default set
            if not base_models:
                base_models = [
                    StockPredictor(input_size=input_size, hidden_size=hidden_size),
                    AttentiveLSTM(input_size=input_size, hidden_size=hidden_size),
                    GRUPredictor(input_size=input_size, hidden_size=hidden_size)
                ]
            
            # Create ensemble model - EnsemblePredictor not yet implemented
            # For now, just use the first base model as a fallback
            # model = EnsemblePredictor(
            #     models=base_models,
            #     input_size=input_size,
            #     hidden_size=hidden_size,
            #     dropout=dropout,
            #     ensemble_method=model_config.get("ensemble_method", "attention")
            # )
            # Fallback to using first base model
            if base_models:
                model = base_models[0]
                logger.warning("EnsemblePredictor not yet implemented, using first base model instead")
            else:
                model = StockPredictor(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, dropout=dropout)
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
            # No validation split - return only training loader
            train_dataset = TensorDataset(features, targets)
            train_loader = create_optimized_dataloader(
                train_dataset,
                batch_size=self.batch_size,
                shuffle=shuffle,
                num_workers=self.config.get("num_workers"),
                pin_memory=self.config.get("pin_memory"),
                device=self.device
            )
            
            return train_loader, None
    
    def _init_weights(self, module: nn.Module):
        """
        Initialize model weights using Xavier/He initialization (PyTorch best practices).
        
        Args:
            module: PyTorch module to initialize
        """
        if isinstance(module, nn.Linear):
            # Use Xavier uniform for linear layers
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LSTM) or isinstance(module, nn.GRU):
            # Use orthogonal initialization for recurrent layers (PyTorch recommendation)
            for name, param in module.named_parameters():
                if 'weight_ih' in name:
                    nn.init.xavier_uniform_(param.data)
                elif 'weight_hh' in name:
                    nn.init.orthogonal_(param.data)
                elif 'bias' in name:
                    nn.init.zeros_(param.data)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)
    
    def _get_loss_function(self) -> nn.Module:
        """
        Get loss function based on configuration.
        
        Returns:
            Loss function module
        """
        loss_type = self.config.get("loss_type", "mse")
        
        if loss_type == "mse":
            return nn.MSELoss()
        elif loss_type == "mae":
            return nn.L1Loss()
        elif loss_type == "huber":
            # Huber loss is more robust to outliers (PyTorch best practice)
            delta = self.config.get("huber_delta", 1.0)
            return nn.HuberLoss(delta=delta)
        elif loss_type == "smooth_l1":
            return nn.SmoothL1Loss()
        else:
            logger.warning(f"Unknown loss type {loss_type}, defaulting to MSE")
            return nn.MSELoss()
    
    def train(
        self,
        train_features: Union[np.ndarray, torch.Tensor],
        train_targets: Union[np.ndarray, torch.Tensor],
        val_features: Optional[Union[np.ndarray, torch.Tensor]] = None,
        val_targets: Optional[Union[np.ndarray, torch.Tensor]] = None,
        epochs: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Train the model with enhanced training loop using PyTorch best practices.
        
        Features:
        - Gradient clipping
        - Learning rate warmup
        - Mixed precision training
        - Early stopping
        - Model checkpointing
        
        Args:
            train_features: Training features
            train_targets: Training targets
            val_features: Validation features (optional)
            val_targets: Validation targets (optional)
            epochs: Number of epochs (overrides config)
            
        Returns:
            Dictionary with training history
        """
        # Initialize weights using proper initialization
        self.model.apply(self._init_weights)
        
        # Get number of epochs
        num_epochs = epochs if epochs is not None else self.config.get("epochs", 100)
        
        # Prepare data loaders
        if val_features is not None and val_targets is not None:
            # Convert validation data to tensors
            if isinstance(val_features, np.ndarray):
                val_features = torch.tensor(val_features, dtype=torch.float32)
            if isinstance(val_targets, np.ndarray):
                val_targets = torch.tensor(val_targets, dtype=torch.float32)
            
            train_loader = self._prepare_data(train_features, train_targets, validation_split=0.0)[0]
            val_dataset = TensorDataset(val_features, val_targets)
            val_loader = create_optimized_dataloader(
                val_dataset,
                batch_size=self.batch_size * 2,
                shuffle=False,
                num_workers=self.config.get("num_workers"),
                pin_memory=self.config.get("pin_memory"),
                device=self.device
            )
        else:
            train_loader, val_loader = self._prepare_data(
                train_features, 
                train_targets,
                validation_split=self.config.get("validation_split", 0.2)
            )
        
        # Get loss function
        criterion = self._get_loss_function()
        
        # Learning rate warmup setup (PyTorch best practice for transformers)
        warmup_steps = self.config.get("warmup_steps", 1000)
        
        # Training loop
        best_val_loss = float('inf')
        epochs_without_improvement = 0
        train_losses = []
        val_losses = []
        
        logger.info(f"Starting training for {num_epochs} epochs")
        logger.info(f"Using {criterion.__class__.__name__} loss function")
        logger.info(f"Mixed precision: {self.mixed_precision}, Gradient clipping: {self.clip_grad_norm}")
        
        for epoch in range(num_epochs):
            # Training phase
            self.model.train()
            epoch_train_loss = 0.0
            train_steps = 0
            
            for batch_idx, (batch_features, batch_targets) in enumerate(train_loader):
                # Move to device
                batch_features = batch_features.to(self.device)
                batch_targets = batch_targets.to(self.device)
                
                # Mixed precision training context
                with mixed_precision_context(self.mixed_precision):
                    # Forward pass
                    outputs = self.model(batch_features)
                    
                    # Ensure shapes match
                    if outputs.shape != batch_targets.shape:
                        if len(batch_targets.shape) == 1:
                            batch_targets = batch_targets.unsqueeze(1)
                    
                    loss = criterion(outputs, batch_targets)
                    
                    # Scale loss for gradient accumulation
                    loss = loss / self.accumulation_steps
                
                # Backward pass with gradient scaling for mixed precision
                if self.mixed_precision:
                    self.scaler.scale(loss).backward()
                else:
                    loss.backward()
                
                # Gradient accumulation
                if (batch_idx + 1) % self.accumulation_steps == 0:
                    # Gradient clipping (PyTorch best practice)
                    if self.mixed_precision:
                        self.scaler.unscale_(self.optimizer)
                    
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), 
                        self.clip_grad_norm
                    )
                    
                    # Optimizer step
                    if self.mixed_precision:
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                    else:
                        self.optimizer.step()
                    
                    self.optimizer.zero_grad()
                    
                    # Learning rate warmup (PyTorch best practice)
                    global_step = epoch * len(train_loader) + batch_idx
                    if global_step < warmup_steps:
                        lr_scale = min(1.0, float(global_step + 1) / warmup_steps)
                        for param_group in self.optimizer.param_groups:
                            param_group['lr'] = self.config.get("learning_rate", 0.001) * lr_scale
                
                epoch_train_loss += loss.item() * self.accumulation_steps
                train_steps += 1
            
            avg_train_loss = epoch_train_loss / train_steps
            train_losses.append(avg_train_loss)
            
            # Validation phase
            if val_loader is not None:
                self.model.eval()
                epoch_val_loss = 0.0
                val_steps = 0
                
                with torch.no_grad():
                    for batch_features, batch_targets in val_loader:
                        batch_features = batch_features.to(self.device)
                        batch_targets = batch_targets.to(self.device)
                        
                        outputs = self.model(batch_features)
                        
                        # Ensure shapes match
                        if outputs.shape != batch_targets.shape:
                            if len(batch_targets.shape) == 1:
                                batch_targets = batch_targets.unsqueeze(1)
                        
                        loss = criterion(outputs, batch_targets)
                        epoch_val_loss += loss.item()
                        val_steps += 1
                
                avg_val_loss = epoch_val_loss / val_steps
                val_losses.append(avg_val_loss)
                
                # Learning rate scheduling
                if self.scheduler is not None:
                    if isinstance(self.scheduler, ReduceLROnPlateau):
                        self.scheduler.step(avg_val_loss)
                    else:
                        self.scheduler.step()
                
                # Early stopping
                if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                    epochs_without_improvement = 0
                    
                    # Save best model (PyTorch best practice)
                    self.save_model("best_model.pth")
                else:
                    epochs_without_improvement += 1
                
                logger.info(
                    f"Epoch {epoch+1}/{num_epochs} - "
                    f"Train Loss: {avg_train_loss:.6f}, "
                    f"Val Loss: {avg_val_loss:.6f}, "
                    f"LR: {self.optimizer.param_groups[0]['lr']:.6f}"
                )
                
                # Early stopping check
                if epochs_without_improvement >= self.early_stopping_patience:
                    logger.info(f"Early stopping triggered after {epoch+1} epochs")
                    break
            else:
                logger.info(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {avg_train_loss:.6f}")
        
        # Store losses for analysis
        self.train_losses = train_losses
        self.val_losses = val_losses
        self.best_val_loss = best_val_loss
        
        return {
            "train_losses": train_losses,
            "val_losses": val_losses,
            "best_val_loss": best_val_loss,
            "epochs_trained": len(train_losses)
        }
    
    def evaluate(
        self, 
        features: Union[np.ndarray, torch.Tensor],
        targets: Union[np.ndarray, torch.Tensor]
    ) -> Dict[str, float]:
        """
        Evaluate the model on test data.
        
        Args:
            features: Test features
            targets: Test targets
            
        Returns:
            Dictionary with evaluation metrics
        """
        self.model.eval()
        
        # Prepare data
        test_loader, _ = self._prepare_data(features, targets, validation_split=0.0)
        
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for batch_features, batch_targets in test_loader:
                batch_features = batch_features.to(self.device)
                outputs = self.model(batch_features)
                
                all_predictions.append(outputs.cpu().numpy())
                all_targets.append(batch_targets.numpy())
        
        # Concatenate all batches
        predictions = np.concatenate(all_predictions, axis=0)
        targets = np.concatenate(all_targets, axis=0)
        
        # Calculate metrics (PyTorch best practice for evaluation)
        mse = mean_squared_error(targets, predictions)
        rmse = np.sqrt(mse)
        mae = np.mean(np.abs(targets - predictions))
        r2 = r2_score(targets, predictions)
        
        metrics = {
            "mse": float(mse),
            "rmse": float(rmse),
            "mae": float(mae),
            "r2_score": float(r2)
        }
        
        logger.info(f"Evaluation metrics: {metrics}")
        return metrics
    
    def predict(
        self,
        features: Union[np.ndarray, torch.Tensor]
    ) -> np.ndarray:
        """
        Make predictions on new data.
        
        Args:
            features: Input features
            
        Returns:
            Predictions as numpy array
        """
        self.model.eval()
        
        # Convert to tensor if needed
        if isinstance(features, np.ndarray):
            features = torch.tensor(features, dtype=torch.float32)
        
        # Move to device
        features = features.to(self.device)
        
        with torch.no_grad():
            predictions = self.model(features)
        
        return predictions.cpu().numpy()
    
    def save_model(self, filepath: str):
        """
        Save model state.
        
        Args:
            filepath: Path to save model
        """
        save_path = self.model_dir / filepath
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'config': self.config,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'best_val_loss': self.best_val_loss
        }, save_path)
        
        logger.info(f"Model saved to {save_path}")
    
    def load_model(self, filepath: str):
        """
        Load model state.
        
        Args:
            filepath: Path to load model from
        """
        load_path = self.model_dir / filepath
        
        checkpoint = torch.load(load_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if checkpoint['scheduler_state_dict'] and self.scheduler:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        self.train_losses = checkpoint.get('train_losses', [])
        self.val_losses = checkpoint.get('val_losses', [])
        self.best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        
        logger.info(f"Model loaded from {load_path}")